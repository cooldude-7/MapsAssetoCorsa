"""The offline region database ("Download Ontario roads"): building it from an OSM file, the
queries giving what the Overpass path gives, every fetch using it when it covers the area and
falling back to Overpass when it doesn't, the download (resume, cancel) and the API.

All offline: the region is `offline_fixture.write_region` (the committed Overpass fixtures as an
OSM file, < 100 KB) and downloads come from a local HTTP server.
"""

from __future__ import annotations

import gzip
import hashlib
import http.client
import http.server
import json
import threading
import time
from pathlib import Path

import numpy as np
import osmium
import pytest
from offline_fixture import (
    BARRIERS,
    BUILDINGS,
    LAKE_REL,
    MILL_ST,
    OSM_BASE,
    POLY,
    ROADS,
    write_region,
)

from driftmaps.app import offline as app_offline
from driftmaps.app.offline import OfflineManager, Source, download_file
from driftmaps.app.server import App, AppServer
from driftmaps.app.settings import Settings
from driftmaps.app.tiles import TileStore, tile_key, tile_xy, tiles_for_box
from driftmaps.geo import barriers as gbar
from driftmaps.geo import buildings as gb
from driftmaps.geo import offline
from driftmaps.geo import overpass as gop
from driftmaps.geo.barriers import fetch_barriers, fetch_roadside, parse_barriers
from driftmaps.geo.buildings import fetch_osm_buildings, parse_buildings
from driftmaps.geo.overpass import OverpassError, fetch_corridor, parse_overpass
from driftmaps.geo.router import route_from_pins

FORKS_PINS = [(43.80410, -79.99231), (43.81060, -79.98185), (43.83180, -79.95940)]
TORONTO_PINS = [(43.6532, -79.3832), (43.6600, -79.3900)]  # outside the fixture region
MILL_ST_WAYS = (34632407, 24486613, 24486612)
MILL_ST_RAILS = (427465490, 427465491, 427465492)


def write_pbf(osm: Path, pbf: Path) -> Path:
    """The OSM XML as a PBF with a replication timestamp in its header (like Geofabrik's)."""
    header = osmium.io.Header()
    header.set("osmosis_replication_timestamp", OSM_BASE)
    writer = osmium.SimpleWriter(str(pbf), 4096, header)
    try:
        for obj in osmium.FileProcessor(str(osm)):
            writer.add(obj)
    finally:
        writer.close()
    return pbf


@pytest.fixture(scope="module")
def region(tmp_path_factory) -> dict:
    """The fixture region as .osm, .pbf, .poly and a built database in its own folder."""
    folder = tmp_path_factory.mktemp("region")
    osm, poly = write_region(folder)
    pbf = write_pbf(osm, folder / "region.osm.pbf")
    dbdir = folder / "db"
    dbdir.mkdir()
    stats = offline.build_db(pbf, dbdir / "region.sqlite", region="region", poly=POLY)
    return {"osm": osm, "pbf": pbf, "poly": poly, "dir": dbdir, "stats": stats}


@pytest.fixture
def use_db(region, monkeypatch):
    """Point the database lookup at the fixture database."""
    monkeypatch.setenv(offline.ENV_OFFLINE_DIR, str(region["dir"]))
    offline.configure(None)
    return region


def no_network(*a, **k):
    raise AssertionError("Overpass must not be asked")


def forks() -> gop.OsmData:
    return parse_overpass(ROADS.read_bytes())


def mill_street_line() -> np.ndarray:
    ways = {w.id: w for w in parse_overpass(BARRIERS.read_bytes()).ways}
    return np.concatenate([ways[i].lonlat[(1 if k else 0) :] for k, i in enumerate(MILL_ST_WAYS)])


# ---------------------------------------------------------------------------------------------
# Building the database
# ---------------------------------------------------------------------------------------------


def test_build_keeps_what_the_app_uses(region):
    stats = region["stats"]
    assert region["pbf"].stat().st_size < 200_000 and region["osm"].stat().st_size < 200_000
    # 37 Forks roads + 3 Snake Rd ways + 3 Mill St ways + 1 driveway; 3 houses; 3 rails;
    # pond + river + the lake's member way; 1 cliff.
    assert stats.counts == {"roads": 44, "buildings": 3, "barriers": 3, "water": 3, "rock": 1}
    assert stats.osm_base == OSM_BASE
    db = offline.OfflineDB(stats.path)
    try:
        assert db.region == "region" and db.osm_base == OSM_BASE
        assert db.counts == stats.counts
        assert db.cache_key == f"offline:region:{OSM_BASE}"
    finally:
        db.close()


def test_xml_and_pbf_build_the_same_rows(region, tmp_path):
    out = offline.build_db(region["osm"], tmp_path / "x.sqlite", region="x", poly=POLY)
    a, b = offline.OfflineDB(out.path), offline.OfflineDB(region["stats"].path)
    try:
        x, y = tile_xy(43.81, -79.98)
        assert a.tile_ways(x, y) == b.tile_ways(x, y)
        assert out.counts == region["stats"].counts
    finally:
        a.close()
        b.close()


def test_build_without_poly_uses_the_data_bbox_and_cancel_cleans_up(region, tmp_path):
    stats = offline.build_db(region["osm"], tmp_path / "bb.sqlite", region="bb")
    db = offline.OfflineDB(stats.path)
    try:
        w, s_, e, n = db.poly.bounds
        assert w < -79.99 < -79.88 < e and s_ < 43.30 < 43.83 < n
    finally:
        db.close()
    stop = threading.Event()
    stop.set()
    with pytest.raises(offline.BuildCancelled):
        offline.build_db(region["osm"], tmp_path / "c.sqlite", region="c", cancel=stop)
    assert not list(tmp_path.glob("c.sqlite*"))
    with pytest.raises(ValueError):
        offline.build_db(region["osm"], tmp_path / "x.sqlite", region="../evil")


def test_parse_poly_reads_outer_rings_and_holes():
    text = "r\n1\n 0 0\n 10 0\n 10 10\n 0 10\n 0 0\nEND\n!2\n 4 4\n 6 4\n 6 6\n 4 6\nEND\nEND\n"
    geom = offline.parse_poly(text)
    assert geom.area == pytest.approx(100 - 4)
    with pytest.raises(ValueError):
        offline.parse_poly("nothing\nEND\n")


def test_is_drivable_mirrors_the_overpass_filter():
    assert offline.is_drivable({"highway": "tertiary"})
    assert offline.is_drivable({"highway": "service"})
    assert not offline.is_drivable({"highway": "service", "service": "driveway"})
    assert not offline.is_drivable({"highway": "residential", "area": "yes"})
    assert not offline.is_drivable({"highway": "footway"})
    assert offline.is_drivable({"highway": "track", "surface": "asphalt"})
    assert not offline.is_drivable({"highway": "track", "surface": "gravel"})
    assert offline.is_drivable({"highway": "track"}, "all")
    assert not offline.is_drivable({"highway": "track", "surface": "asphalt"}, "none")


# ---------------------------------------------------------------------------------------------
# Queries == the Overpass path
# ---------------------------------------------------------------------------------------------


def test_tiles_equal_the_overpass_path(use_db, tmp_path):
    ll = np.concatenate([w.lonlat for w in forks().ways])  # every tile the fixture touches
    tiles = tiles_for_box(ll[:, 1].min(), ll[:, 0].min(), ll[:, 1].max(), ll[:, 0].max())
    over = TileStore(tmp_path / "a", post=lambda q: ROADS.read_bytes(), use_region_db=False)
    found_o, err_o = over.get(tiles)
    store = TileStore(tmp_path / "b", post=no_network)
    t0 = time.perf_counter()
    found_d, err_d = store.get(tiles)
    took = time.perf_counter() - t0
    assert err_o is None and err_d is None and set(found_o) == set(found_d) == set(tiles)
    payload_o = TileStore.api_payload(found_o[t] for t in tiles)
    payload_d = TileStore.api_payload(found_d[t] for t in tiles)
    assert payload_d == payload_o
    assert len(payload_d[1]) == 37
    for t in tiles:  # tile by tile too, and geometry exactly as Overpass gave it
        assert [w["id"] for w in found_d[t].ways] == sorted(w["id"] for w in found_o[t].ways)
        by_id = {w["id"]: w for w in found_o[t].ways}
        for w in found_d[t].ways:
            assert w["geom"] == [tuple(p) for p in by_id[w["id"]]["geom"]]
    assert took < 1.0
    assert not (tmp_path / "b" / "tiles").exists()  # nothing written to the tile cache


def test_corridor_equals_the_overpass_fixture(use_db, monkeypatch, tmp_path):
    monkeypatch.setattr(gop, "post_query", no_network)
    got = fetch_corridor(FORKS_PINS, tmp_path, buffer_m=400.0)
    want = forks()
    assert [w.id for w in got.ways] == [w.id for w in want.ways]
    for a, b in zip(got.ways, want.ways, strict=True):
        assert a.nodes == b.nodes
        np.testing.assert_array_equal(a.lonlat, b.lonlat)
        assert a.tags == {k: v for k, v in b.tags.items() if k != "source"}
    assert got.osm_base == OSM_BASE and got.cache_key.startswith("offline:region:")
    assert not (tmp_path / "overpass").exists()


def test_router_uses_the_database_and_falls_back_to_overpass(use_db, monkeypatch, tmp_path):
    monkeypatch.setattr(gop, "post_query", no_network)
    route = route_from_pins(FORKS_PINS, tmp_path, buffer_m=400.0)
    assert route.length_m > 3000 and route.cache_key.startswith("offline:")
    asked = []

    def fake_post(query, mirrors=None, **kw):
        asked.append(query)
        return ROADS.read_bytes()

    monkeypatch.setattr(gop, "post_query", fake_post)
    with pytest.raises(Exception):  # noqa: B017 - Toronto pins on Forks roads: no route
        route_from_pins(TORONTO_PINS, tmp_path, buffer_m=400.0)
    assert asked, "outside the region the corridor comes from Overpass"


def test_without_a_database_everything_uses_overpass(monkeypatch, tmp_path):
    asked = []

    def fake_post(query, mirrors=None, **kw):
        asked.append(query)
        return ROADS.read_bytes()

    monkeypatch.setattr(gop, "post_query", fake_post)
    osm = fetch_corridor(FORKS_PINS, tmp_path, buffer_m=400.0)
    assert len(osm.ways) == 37 and len(asked) == 1 and osm.cache_key != "offline"
    assert offline.corridor(FORKS_PINS, 400.0) is None


def test_refresh_skips_the_database(use_db, monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(gop, "post_query", lambda q, m=None, **k: asked.append(q) or b"x")
    with pytest.raises(OverpassError):
        fetch_corridor(FORKS_PINS, tmp_path, buffer_m=400.0, refresh=True)
    assert asked


def test_buildings_come_from_the_database(use_db, monkeypatch, tmp_path):
    monkeypatch.setattr(gb, "post_query", no_network)
    ways = {w.id: w for w in parse_overpass(BUILDINGS.read_bytes()).ways}
    route = np.concatenate([ways[121184920].lonlat, ways[24233890].lonlat[1:]])
    fps, base = fetch_osm_buildings(route, cache_dir=tmp_path)
    want = parse_buildings(BUILDINGS.read_bytes())
    assert [f.id for f in fps] == [f.id for f in want] and base == OSM_BASE
    for a, b in zip(fps, want, strict=True):
        np.testing.assert_array_equal(a.lonlat, b.lonlat)
        assert a.tags == b.tags
    # Outside the region: Overpass (here a fake answering with the fixture).
    monkeypatch.setattr(gb, "post_query", lambda q, m=None, **k: BUILDINGS.read_bytes())
    toronto = np.array([[-79.3832, 43.6532], [-79.3900, 43.6600]])
    got, _ = fetch_osm_buildings(toronto, cache_dir=tmp_path)
    assert len(got) == 3


def test_building_relations_come_whole(use_db, region, tmp_path):
    db = offline.databases()[0]
    lat, lon = 43.3102, -79.8898
    box = (lat - 0.01, lon - 0.01, lat + 0.01, lon + 0.01)
    payload = db.buildings_payload([box])
    ids = sorted(e["id"] for e in payload["elements"])
    assert ids == [832043557, 832043679, 832043713]
    assert db.buildings_payload([(43.65, -79.39, 43.66, -79.38)]) is None  # not covered


def test_barriers_come_from_the_database(use_db, monkeypatch, tmp_path):
    monkeypatch.setattr(gbar, "post_query", no_network)
    line = mill_street_line()
    got = fetch_barriers(line, cache_dir=tmp_path)
    want = parse_barriers(parse_overpass(BARRIERS.read_bytes()))
    assert [(b.id, b.kind) for b in got] == [(b.id, b.kind) for b in want]
    assert [b.id for b in got] == list(MILL_ST_RAILS)
    assert fetch_barriers(line, cache_dir=tmp_path, kinds=("wall",)) == []


def test_roadside_features_come_from_the_database(use_db, monkeypatch, tmp_path):
    monkeypatch.setattr(gbar, "post_query", no_network)
    feats = fetch_roadside(mill_street_line(), cache_dir=tmp_path)
    water = {(f.id, f.kind, f.area) for f in feats.water}
    assert (LAKE_REL, "water", True) in water  # the relation, its member as an area edge
    assert (9_200_000_002, "water", True) in water  # the closed pond
    assert (9_200_000_004, "river", False) in water
    assert [(f.id, f.area) for f in feats.rock] == [(9_200_000_003, True)]
    access = {f.id: f for f in feats.access}
    assert set(MILL_ST_WAYS) <= access.keys() and 9_200_000_005 in access  # + the driveway
    road = parse_overpass(BARRIERS.read_bytes()).ways
    mill = next(w for w in road if w.id == MILL_ST_WAYS[0])
    assert access[mill.id].nodes == mill.nodes
    assert feats.osm_base == OSM_BASE


def test_roadside_outside_the_region_asks_overpass(use_db, monkeypatch, tmp_path):
    asked = []

    def fake_post(query, mirrors=None, **kw):
        asked.append(query)
        return json.dumps({"elements": []}).encode()

    monkeypatch.setattr(gbar, "post_query", fake_post)
    toronto = np.array([[-79.3832, 43.6532], [-79.3900, 43.6600]])
    assert fetch_roadside(toronto, cache_dir=tmp_path).water == ()
    assert fetch_barriers(toronto, cache_dir=tmp_path) == []
    assert len(asked) == 2


def test_tile_outside_the_region_is_not_answered(use_db):
    x, y = tile_xy(43.6532, -79.3832)
    assert offline.tile_ways(x, y) is None
    x, y = tile_xy(43.81, -79.98)
    ways, base = offline.tile_ways(x, y)
    assert ways and base == OSM_BASE


def test_a_replaced_database_is_reopened_and_a_broken_one_ignored(region, tmp_path, monkeypatch):
    folder = tmp_path / "dbs"
    folder.mkdir()
    (folder / "broken.sqlite").write_bytes(b"not a database")
    (folder / "x.txt").write_text("ignored")
    monkeypatch.setenv(offline.ENV_OFFLINE_DIR, str(folder))
    offline.configure(None)
    assert offline.databases() == []
    offline.build_db(region["osm"], folder / "fresh.sqlite", region="fresh", poly=POLY)
    first = offline.databases()
    assert [d.region for d in first] == ["fresh"]
    offline.build_db(region["osm"], folder / "fresh.sqlite", region="fresh", poly=POLY)
    again = offline.databases()
    assert again[0] is not first[0] and again[0].region == "fresh"


# ---------------------------------------------------------------------------------------------
# Download (local HTTP server with Range support)
# ---------------------------------------------------------------------------------------------


class _Files(http.server.BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}
    log: list[tuple[str, str | None]] = []
    delay_s = 0.0

    def do_GET(self) -> None:  # noqa: N802
        type(self).log.append((self.path, self.headers.get("Range")))
        body = type(self).files.get(self.path)
        if body is None:
            self.send_error(404)
            return
        etag = '"' + hashlib.md5(body).hexdigest() + '"'  # noqa: S324 - a test ETag
        rng = self.headers.get("Range")
        start = 0
        if rng and self.headers.get("If-Range") == etag:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
        else:
            self.send_response(200)
        self.send_header("ETag", etag)
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        step = 4096
        for i in range(start, len(body), step):
            try:
                self.wfile.write(body[i : i + step])
            except OSError:
                return
            if type(self).delay_s:
                time.sleep(type(self).delay_s)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def files_server():
    _Files.files, _Files.log, _Files.delay_s = {}, [], 0.0
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Files)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield _Files, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_download_resumes_with_range(files_server, tmp_path):
    files, base = files_server
    body = bytes(range(256)) * 400
    files.files["/x.pbf"] = body
    dest = tmp_path / "x.pbf"
    download_file(base + "/x.pbf", dest)
    assert dest.read_bytes() == body
    # A part file from an earlier, broken try of the same URL: only the rest is fetched.
    dest.unlink()
    etag = '"' + hashlib.md5(body).hexdigest() + '"'  # noqa: S324
    (tmp_path / "x.pbf.part").write_bytes(body[:30_000])
    (tmp_path / "x.pbf.part.json").write_text(
        json.dumps({"url": base + "/x.pbf", "validator": etag, "total": len(body)})
    )
    seen = []
    download_file(base + "/x.pbf", dest, progress=lambda d, t: seen.append((d, t)))
    assert dest.read_bytes() == body
    assert files.log[-1] == ("/x.pbf", "bytes=30000-")
    assert seen[0][0] > 30_000 and seen[-1] == (len(body), len(body))
    assert not (tmp_path / "x.pbf.part").exists() and not (tmp_path / "x.pbf.part.json").exists()
    # A part file of another URL is thrown away.
    dest.unlink()
    (tmp_path / "x.pbf.part").write_bytes(b"junk")
    (tmp_path / "x.pbf.part.json").write_text(json.dumps({"url": "http://other/"}))
    download_file(base + "/x.pbf", dest)
    assert dest.read_bytes() == body


def test_download_cancel_keeps_the_part_file(files_server, tmp_path):
    files, base = files_server
    files.files["/big.pbf"] = b"z" * 400_000
    files.delay_s = 0.002
    stop = threading.Event()

    def progress(done, total):
        if done > 50_000:
            stop.set()

    with pytest.raises(app_offline.Cancelled):
        download_file(base + "/big.pbf", tmp_path / "big.pbf", progress=progress, cancel=stop)
    part = tmp_path / "big.pbf.part"
    assert part.is_file() and 50_000 < part.stat().st_size < 400_000
    files.delay_s = 0.0
    download_file(base + "/big.pbf", tmp_path / "big.pbf")
    assert (tmp_path / "big.pbf").read_bytes() == b"z" * 400_000
    assert files.log[-1][1] is not None  # resumed


def test_download_errors_are_plain(files_server, tmp_path):
    _, base = files_server
    with pytest.raises(app_offline.DownloadError, match="HTTP 404"):
        download_file(base + "/missing.pbf", tmp_path / "m.pbf")


def _sources(base: str) -> dict:
    return {
        "ontario": (
            Source("broken", base + "/nothing.pbf", base + "/nothing.poly"),
            Source("test", base + "/ontario.osm.pbf", base + "/ontario.poly"),
        )
    }


def test_manager_downloads_processes_and_deletes(files_server, region, tmp_path):
    files, base = files_server
    pbf = region["pbf"].read_bytes()
    files.files["/ontario.osm.pbf"] = pbf
    files.files["/ontario.osm.pbf.md5"] = (hashlib.md5(pbf).hexdigest() + "  x\n").encode()  # noqa: S324
    files.files["/ontario.poly"] = POLY.encode()
    mgr = OfflineManager(tmp_path / "off", sources=_sources(base))
    assert mgr.status()["state"] == "none"
    steps = []
    stats = mgr.run("ontario", progress=lambda s, f, t: steps.append((s, f, t)))
    assert stats.counts["roads"] == 44
    assert {s for s, _, _ in steps} == {"downloading", "processing"}
    st = mgr.status()
    assert st["state"] == "ready" and st["updated"] == OSM_BASE[:10] and st["size_mb"] > 0
    assert not (tmp_path / "off" / "ontario.osm.pbf").exists()  # deleted once processed
    assert (tmp_path / "off" / "ontario.poly").is_file()
    assert mgr.hint(*MILL_ST) is None  # ready: no tip
    mgr.delete("ontario")
    assert mgr.status()["state"] == "none" and not list((tmp_path / "off").iterdir())
    assert mgr.hint(*MILL_ST) == "ontario"
    assert mgr.hint(48.85, 2.35) is None  # Paris


def test_manager_rejects_a_damaged_download(files_server, region, tmp_path):
    files, base = files_server
    files.files["/ontario.osm.pbf"] = region["pbf"].read_bytes()
    files.files["/ontario.osm.pbf.md5"] = b"0" * 32 + b"\n"
    files.files["/ontario.poly"] = POLY.encode()
    mgr = OfflineManager(tmp_path / "off", sources=_sources(base))
    with pytest.raises(app_offline.DownloadError, match="MD5"):
        mgr.run("ontario")


# ---------------------------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------------------------


class Client:
    def __init__(self, port: int) -> None:
        self.port = port

    def call(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers={"Content-Type": "application/json",
                                                       **(headers or {})})  # fmt: skip
        resp = conn.getresponse()
        out = json.loads(resp.read() or b"null")
        conn.close()
        return resp.status, out


@pytest.fixture
def api(files_server, region, tmp_path, monkeypatch):
    files, base = files_server
    files.files["/ontario.osm.pbf"] = region["pbf"].read_bytes()
    files.files["/ontario.poly"] = POLY.encode()
    app = App(Settings(tmp_path / "appdata", discover=lambda: None))
    app.offline.sources = _sources(base)
    srv = AppServer(app, port=0).start()
    yield Client(srv.port), app, files
    srv.stop()


def wait_state(client: Client, states, timeout=20.0) -> dict:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        status, st = client.call("GET", "/api/offline")
        assert status == 200
        if st["state"] in states:
            return st
        time.sleep(0.02)
    raise AssertionError(f"offline state still {st['state']}")


def test_api_download_ready_roads_and_delete(api, monkeypatch, tmp_path):
    client, app, _ = api
    status, st = client.call("GET", "/api/offline")
    assert status == 200 and st["state"] == "none" and st["region"] == "ontario"
    assert st["region_name"] == "Ontario" and st["download_mb"] == 1075
    status, st = client.call("POST", "/api/offline/download", {"region": "ontario"})
    assert status == 200 and st["state"] in ("downloading", "processing", "ready")
    st = wait_state(client, ("ready", "error"))
    assert st["state"] == "ready", st
    assert st["updated"] == OSM_BASE[:10] and st["progress"] == 0.0
    # Roads now come from the database: Overpass is never asked.
    monkeypatch.setattr("driftmaps.app.tiles.post_query", no_network)
    x, y = tile_xy(43.81, -79.98)
    status, roads = client.call("POST", "/api/roads", {"tiles": [f"14/{x}/{y}"]})
    assert status == 200 and roads["tiles"] == {tile_key((14, x, y)): "ok"}
    assert roads["ways"] and roads["offline_hint"] is None
    status, st = client.call("POST", "/api/offline/delete", {"region": "ontario"})
    assert status == 200 and st["state"] == "none"
    assert offline.databases() == []


def test_api_checks_region_origin_and_busy(api, tmp_path):
    client, app, files = api
    for bad in ("../../etc", "ontario.sqlite", 5, None, "quebec"):
        status, out = client.call("POST", "/api/offline/download", {"region": bad})
        assert status == 400 and "region" in out["error"].lower()
    status, _ = client.call(
        "POST", "/api/offline/download", {}, headers={"Origin": "http://evil.example"}
    )
    assert status == 403
    status, _ = client.call("GET", "/api/offline", headers={"Host": "evil.example"})
    assert status == 403
    status, _ = client.call("POST", "/api/offline/explode", {})
    assert status == 405
    assert not (tmp_path / "appdata" / "offline").exists() or not any(
        (tmp_path / "appdata" / "offline").iterdir()
    )
    files.delay_s = 0.01  # slow enough to be busy for a while
    status, _ = client.call("POST", "/api/offline/download", {})
    assert status == 200
    status, out = client.call("POST", "/api/offline/download", {})
    assert status == 409
    status, out = client.call("POST", "/api/offline/delete", {})
    assert status == 409
    status, st = client.call("POST", "/api/offline/cancel", {})
    assert status == 200
    st = wait_state(client, ("none", "error"))
    assert st["state"] == "none" and "Cancelled" in st["message"]


def test_api_roads_hint_after_a_slow_overpass_load(api, monkeypatch):
    client, app, _ = api
    monkeypatch.setattr("driftmaps.app.server.SLOW_ROADS_S", 0.0)
    app.tiles.post = lambda q: ROADS.read_bytes()
    x, y = tile_xy(*MILL_ST)
    status, roads = client.call("POST", "/api/roads", {"tiles": [f"14/{x}/{y}"]})
    assert status == 200 and roads["offline_hint"] == "ontario"
    x, y = tile_xy(48.85, 2.35)  # Paris: no region to offer
    status, roads = client.call("POST", "/api/roads", {"tiles": [f"14/{x}/{y}"]})
    assert status == 200 and roads["offline_hint"] is None


def test_the_cli_reports_and_deletes(region, tmp_path):
    from typer.testing import CliRunner

    from driftmaps.cli import app as cli

    runner = CliRunner()
    data = tmp_path / "data"
    res = runner.invoke(cli, ["offline", "status", "--data-dir", str(data)])
    assert res.exit_code == 0 and "not downloaded" in res.output
    res = runner.invoke(
        cli,
        ["offline", "download", "--data-dir", str(data), "--source-file", str(region["pbf"]),
         "--poly-file", str(region["poly"])],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    assert "ready" in res.output and "44 roads" in res.output
    res = runner.invoke(cli, ["offline", "status", "--data-dir", str(data)])
    assert res.exit_code == 0 and "ready" in res.output and OSM_BASE[:10] in res.output
    res = runner.invoke(cli, ["offline", "delete", "--data-dir", str(data)])
    assert res.exit_code == 0 and not (data / "offline" / "ontario.sqlite").exists()
    res = runner.invoke(cli, ["offline", "status", "--region", "quebec"])
    assert res.exit_code != 0


def test_gzip_tile_cache_is_not_touched_by_the_database(use_db, tmp_path):
    store = TileStore(tmp_path)
    x, y = tile_xy(43.81, -79.98)
    found, err = store.get([(14, x, y)])
    assert err is None and found
    assert not list(tmp_path.rglob("*.json.gz"))
    assert gzip  # (imported for symmetry with the tile tests)
