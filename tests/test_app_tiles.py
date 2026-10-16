"""Road tiles for the turn-by-turn builder (`driftmaps.app.tiles`), offline.

The Overpass call is a fake that answers every query with the Forks of the Credit fixture and
counts the calls; `TileStore` must split it into z14 tiles, cache them (memory and disk) and
never ask again for a cached tile.
"""

from __future__ import annotations

import gzip
import json
import math
import threading
import time
from pathlib import Path

import pytest

from driftmaps.app import tiles as tiles_mod
from driftmaps.app.tiles import (
    MAX_TILES,
    TILE_ZOOM,
    TileStore,
    parse_tile,
    tile_bounds,
    tile_key,
    tile_xy,
    tiles_for_box,
)
from driftmaps.geo import router
from driftmaps.geo.overpass import OverpassError, parse_overpass
from driftmaps.geo.router import route_from_path

FIXTURES = Path(__file__).parent / "fixtures"
FORKS = FIXTURES / "osm" / "forks_of_the_credit.json"
ROUTE_TILE = (14, 4551, 5969)  # holds the fixture route's pins (43.804..43.811, -79.99..-79.98)


def _fmt(value: float) -> str:  # as the Overpass query writes coordinates
    return f"{value:.7f}".rstrip("0").rstrip(".")


class FakePost:
    def __init__(self, body: bytes | None = None, error: Exception | None = None) -> None:
        self.body = body if body is not None else FORKS.read_bytes()
        self.error = error
        self.queries: list[str] = []

    def __call__(self, query: str) -> bytes:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.body


@pytest.fixture(scope="module")
def forks():
    return parse_overpass(FORKS.read_bytes())


# -- tile maths ---------------------------------------------------------------------------------


def test_tile_maths():
    assert tile_xy(43.80410, -79.99231) == (4551, 5969)
    assert tile_xy(43.31, -79.885) == (4556, 6000)
    s, w, n, e = tile_bounds(*ROUTE_TILE)
    assert s < 43.80410 < n and w < -79.99231 < e
    assert e - w == pytest.approx(360 / 2**14)
    # Neighbours share edges; a point just inside each corner maps back to the tile.
    assert tile_bounds(14, 4552, 5969)[1] == pytest.approx(e)
    assert tile_bounds(14, 4551, 5970)[2] == pytest.approx(s)
    for lat, lon in ((s + 1e-7, w + 1e-7), (n - 1e-7, e - 1e-7)):
        assert tile_xy(lat, lon) == ROUTE_TILE[1:]
    assert tiles_for_box(s + 0.001, w + 0.001, n - 0.001, e - 0.001) == [ROUTE_TILE]
    box = tiles_for_box(43.80, -80.0, 43.83, -79.95)
    assert len(box) == len(set(box)) == 9 and ROUTE_TILE in box


def test_parse_tile():
    assert parse_tile("14/4551/5969") == ROUTE_TILE
    assert tile_key(ROUTE_TILE) == "14/4551/5969"
    for bad in ("13/4551/5969", "14/16384/1", "14/1", "a/b/c", 5, None, "14/1/2/3", " 14/1/2"):
        with pytest.raises(ValueError):
            parse_tile(bad)


# -- the store ----------------------------------------------------------------------------------


def test_fetch_once_then_cached(tmp_path, forks):
    post = FakePost()
    store = TileStore(tmp_path, post=post)
    want = [ROUTE_TILE, (14, 4552, 5969), (14, 4551, 5968)]
    found, error = store.get(want)
    assert error is None and set(found) == set(want)
    assert len(post.queries) == 1  # one Overpass query for all the missing tiles
    query = post.queries[0]
    assert "out body geom" in query and query.count('way["highway"~') == 1  # one box round them
    s, w, _, e = tile_bounds(14, 4551, 5969)  # the box: 4551..4552 x 5968..5969
    n = tile_bounds(14, 4551, 5968)[2]
    e = tile_bounds(14, 4552, 5969)[3]
    assert f"({_fmt(s)},{_fmt(w)},{_fmt(n)},{_fmt(e)})" in query

    # Each tile holds the ways crossing it, whole; a way crossing two tiles is in both.
    ids = {t: {w["id"] for w in found[t].ways} for t in want}
    assert ids[ROUTE_TILE]
    for t, way_ids in ids.items():
        tb = tile_bounds(*t)
        for way in forks.ways:
            lons, lats = way.lonlat[:, 0], way.lonlat[:, 1]
            inside = ((lats > tb[0]) & (lats < tb[2]) & (lons > tb[1]) & (lons < tb[3])).any()
            if inside:
                assert way.id in way_ids, (t, way.id)
    whole = next(w for w in found[ROUTE_TILE].ways)
    assert len(whole["nodes"]) == len(next(w for w in forks.ways if w.id == whole["id"]).nodes)

    # Again: memory, no query. A new store on the same folder: disk, no query.
    assert store.get(want)[0].keys() == found.keys() and len(post.queries) == 1
    path = tmp_path / "tiles" / "14" / "4551" / "5969.json.gz"
    assert path.is_file() and json.loads(gzip.decompress(path.read_bytes()))["format"] == 1
    other_post = FakePost()
    again, error = TileStore(tmp_path, post=other_post).get(want)
    assert error is None and not other_post.queries
    assert {t: {w["id"] for w in d.ways} for t, d in again.items()} == ids

    # Only the missing tile is asked for.
    store.get([ROUTE_TILE, (14, 4553, 5969)])
    assert len(post.queries) == 2
    assert f"({','.join(_fmt(v) for v in tile_bounds(14, 4553, 5969))})" in post.queries[1]


def test_scattered_tiles_one_query(tmp_path):
    post = FakePost()
    TileStore(tmp_path, post=post).get([(14, 4551, 5969), (14, 4560, 5980)])
    assert len(post.queries) == 1
    assert post.queries[0].count('way["highway"~') == 2  # a box each, not one huge box


def test_empty_tile_is_ok(tmp_path):
    post = FakePost(json.dumps({"elements": [], "osm3s": {}}).encode())
    store = TileStore(tmp_path, post=post)
    found, error = store.get([(14, 100, 100)])
    assert error is None and found[(14, 100, 100)].ways == []
    store.get([(14, 100, 100)])
    assert len(post.queries) == 1


def test_download_failure(tmp_path):
    post = FakePost(error=OverpassError("all Overpass mirrors failed"))
    store = TileStore(tmp_path, post=post)
    found, error = store.get([ROUTE_TILE])
    assert found == {} and "mirrors failed" in error
    assert not (tmp_path / "tiles" / "14" / "4551" / "5969.json.gz").exists()
    post.error = None  # the next try downloads it
    found, error = store.get([ROUTE_TILE])
    assert error is None and ROUTE_TILE in found and len(post.queries) == 2


def test_broken_cache_file_is_refetched(tmp_path):
    path = tmp_path / "tiles" / "14" / "4551" / "5969.json.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not gzip")
    post = FakePost()
    found, error = TileStore(tmp_path, post=post).get([ROUTE_TILE])
    assert error is None and found[ROUTE_TILE].ways and len(post.queries) == 1


def test_concurrent_requests_download_once(tmp_path):
    post = FakePost()
    gate = threading.Event()

    def slow(query: str) -> bytes:
        gate.wait(5)
        return post(query)

    store = TileStore(tmp_path, post=slow)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(store.get([ROUTE_TILE]))) for _ in range(3)
    ]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join(10)
    assert len(results) == 3 and all(ROUTE_TILE in r[0] for r in results)
    assert len(post.queries) == 1


def test_different_tiles_download_side_by_side(tmp_path):
    """A request for other tiles doesn't wait for a slow download; one for the same tile does
    (and doesn't ask again); a failed download is an error for both."""
    post = FakePost()
    gate = threading.Event()
    started = threading.Semaphore(0)

    def slow(query: str) -> bytes:
        started.release()
        assert gate.wait(5)
        return post(query)

    store = TileStore(tmp_path, post=slow)
    first = []
    t = threading.Thread(target=lambda: first.append(store.get([ROUTE_TILE])))
    t.start()
    assert started.acquire(timeout=5)  # the first download is under way
    gate_other = threading.Thread(target=lambda: (started.acquire(timeout=5), gate.set()))
    gate_other.start()
    found, error = store.get([(14, 4560, 5980)])  # its own download, side by side
    assert error is None and (14, 4560, 5980) in found
    t.join(10)
    gate_other.join(10)
    assert first and first[0][1] is None and ROUTE_TILE in first[0][0]
    assert len(post.queries) == 2

    post.error = OverpassError("all Overpass mirrors failed")
    gate.clear()
    results = []
    tile = (14, 4570, 5990)
    threads = [threading.Thread(target=lambda: results.append(store.get([tile])))]
    threads[0].start()
    assert started.acquire(timeout=5)
    threads.append(threading.Thread(target=lambda: results.append(store.get([tile]))))
    threads[1].start()
    time.sleep(0.3)  # the second request is waiting for the first's download by now
    gate.set()
    for th in threads:
        th.join(10)
    assert len(post.queries) == 3  # the waiting request didn't ask again
    assert all(found == {} and error for found, error in results)


def test_api_payload(tmp_path, forks):
    store = TileStore(tmp_path, post=FakePost())
    found, _ = store.get([ROUTE_TILE, (14, 4552, 5969)])
    nodes, ways = TileStore.api_payload(found.values())
    assert [w["id"] for w in ways] == sorted({w["id"] for d in found.values() for w in d.ways})
    by_id = {w.id: w for w in forks.ways}
    for w in ways:
        assert set(w) == {"id", "name", "ref", "highway", "oneway", "nodes"}
        src = by_id[w["id"]]
        assert w["nodes"] == list(src.nodes) and w["highway"] == src.tags["highway"]
        assert w["name"] == src.tags.get("name") and w["oneway"] in (-1, 0, 1)
        for nid, (lon, lat) in zip(src.nodes, src.lonlat.tolist(), strict=True):
            assert nodes[str(nid)] == pytest.approx([lon, lat], abs=1e-7)
    json.dumps({"nodes": nodes, "ways": ways})  # plain JSON


def test_oneway_in_payload(tmp_path):
    body = {
        "osm3s": {"timestamp_osm_base": "2026-10-15T00:00:00Z"},
        "elements": [
            {"type": "way", "id": i, "nodes": [10 * i, 10 * i + 1],
             "geometry": [{"lat": 43.8041, "lon": -79.99}, {"lat": 43.8045, "lon": -79.989}],
             "tags": {"highway": "residential", **tags}}
            for i, tags in enumerate(
                ({}, {"oneway": "yes"}, {"oneway": "-1"}, {"junction": "roundabout"}), start=1
            )
        ],
    }  # fmt: skip
    store = TileStore(tmp_path, post=FakePost(json.dumps(body).encode()))
    found, _ = store.get([ROUTE_TILE])
    _, ways = TileStore.api_payload(found.values())
    assert [w["oneway"] for w in ways] == [0, 1, -1, 1]


def test_for_nodes_and_route_path(tmp_path, forks):
    store = TileStore(tmp_path, post=FakePost())
    store.get([ROUTE_TILE])

    pinned = router.route_from_pins([(43.80410, -79.99231), (43.81060, -79.98185)], osm=forks)
    path = pinned.node_path()
    osm, missing = store.for_nodes(path)
    assert missing == set()
    assert {w.id for w in osm.ways} == {w.id for w in forks.ways if set(w.nodes) & set(path)}
    assert osm.osm_base == forks.osm_base
    route = route_from_path(osm, path, pinned.start_t, pinned.end_t)
    assert route.edges == pinned.edges and route.stubs == pinned.stubs
    _, missing = store.for_nodes([*path, 123])
    assert missing == {123}
    # A new store (app restarted): nothing in memory yet.
    assert TileStore(tmp_path, post=FakePost()).for_nodes(path)[1] == set(path)


def test_for_nodes_adds_the_tiles_of_the_path(tmp_path):
    """A junction reached through a way stored whole in another tile still gets every road
    meeting there (its own tile, from the disk cache), so the route's side-road stubs are
    complete."""
    t1, t2 = ROUTE_TILE, (14, 4552, 5969)  # t2 is east of t1
    s, w, n, e = tile_bounds(*t1)
    lat, dx = (s + n) / 2, (e - w) * 0.3

    def geom(*pts):
        return [{"lat": la, "lon": lo} for la, lo in pts]

    body = {
        "elements": [  # Main Road from t1 into t2, where Side Road leaves it at node 2
            {"type": "way", "id": 1, "nodes": [1, 2, 3], "tags": {"highway": "tertiary"},
             "geometry": geom((lat, w + dx), (lat, e + dx), (lat, e + 2 * dx))},
            {"type": "way", "id": 2, "nodes": [2, 4], "tags": {"highway": "residential"},
             "geometry": geom((lat, e + dx), (lat + 0.003, e + dx))},
        ]
    }  # fmt: skip
    TileStore(tmp_path, post=FakePost(json.dumps(body).encode())).get([t1, t2])  # both on disk
    store = TileStore(tmp_path, post=FakePost())
    found, _ = store.get([t1])  # the page only loaded t1
    assert [w["id"] for w in found[t1].ways] == [1]
    osm, missing = store.for_nodes([1, 2, 3])
    assert not missing and [w.id for w in osm.ways] == [1, 2]
    route = route_from_path(osm, [1, 2, 3])
    assert route.stubs == ((2, 2, 4),)


def test_covering(tmp_path):
    store = TileStore(tmp_path, post=FakePost())
    s, w, n, e = tile_bounds(*ROUTE_TILE)
    inner = (s + 0.002, w + 0.002, n - 0.002, e - 0.002)
    assert store.covering(*inner) is None  # nothing cached
    store.get([ROUTE_TILE])
    osm = store.covering(*inner)
    assert osm is not None and osm.ways and osm.cache_key == "tiles-z14"
    assert store.covering(s - 0.01, w, n, e) is None  # a tile isn't cached
    assert store.covering(40.0, -82.0, 45.0, -78.0) is None  # far too many tiles


def test_constants():
    assert TILE_ZOOM == 14 and MAX_TILES == 16
    assert tiles_mod.QUERY_TIMEOUT_S < tiles_mod.POST_KWARGS["timeout_s"]
    # z14 tiles in southern Ontario are about 1.8 km wide.
    s, w, n, e = tile_bounds(14, 4556, 6000)
    width_m = (e - w) * 111_320 * math.cos(math.radians((s + n) / 2))
    assert 1500 < width_m < 2200
