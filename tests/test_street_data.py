"""Street furniture data: OSM nodes and power lines (`geo.street_features`) and Mapillary map
features (`geo.mapillary`). All offline: Overpass and Mapillary are faked."""

from __future__ import annotations

import gzip
import json
import urllib.error

import numpy as np
import pytest

from driftmaps.geo import mapillary as gm
from driftmaps.geo import street_features as gsf
from driftmaps.geo.overpass import OverpassError

LINE = np.array([[-79.8200, 43.4410], [-79.8150, 43.4400], [-79.8100, 43.4390]])
TOKEN = "MLY|1234567890|abcdefsecret"

OVERPASS = {
    "osm3s": {"timestamp_osm_base": "2026-10-15T00:00:00Z"},
    "elements": [
        {"type": "node", "id": 1, "lat": 43.441, "lon": -79.82,
         "tags": {"highway": "stop", "direction": "forward"}},
        {"type": "node", "id": 2, "lat": 43.440, "lon": -79.815, "tags": {"power": "pole"}},
        {"type": "node", "id": 3, "lat": 43.440, "lon": -79.815, "tags": {"amenity": "bench"}},
        {"type": "node", "id": 4, "lat": 43.440, "lon": -79.814,
         "tags": {"emergency": "fire_hydrant"}},
        {"type": "way", "id": 10, "nodes": [2, 5],
         "geometry": [{"lat": 43.440, "lon": -79.815}, {"lat": 43.4395, "lon": -79.812}],
         "tags": {"power": "minor_line"}},
        {"type": "way", "id": 11, "nodes": [6, 7],
         "geometry": [{"lat": 43.440, "lon": -79.815}, {"lat": 43.4395, "lon": -79.812}],
         "tags": {"highway": "residential"}},
    ],
}  # fmt: skip


# --------------------------------------------------------------------------------------------
# OSM
# --------------------------------------------------------------------------------------------


def test_street_query_asks_for_the_furniture_nodes_and_power_lines():
    q = gsf.build_street_query(LINE[:, ::-1], 40.0)
    assert q.startswith("[out:json]") and "out body geom;" in q
    for part in ('node["highway"~"^(crossing|give_way|stop|street_lamp|traffic_signals)$"]',
                 'node["power"~"^(pole|tower)$"]', 'node["emergency"="fire_hydrant"]',
                 'node["traffic_sign"]', 'way["power"~"^(line|minor_line)$"]'):  # fmt: skip
        assert part in q
    assert "around:40," in q
    with pytest.raises(ValueError):
        gsf.build_street_query(np.zeros((0, 2)))


def test_parse_keeps_furniture_only():
    feats = gsf.parse_street_features(json.dumps(OVERPASS))
    assert [n.id for n in feats.nodes] == [1, 2, 4]
    assert [(w.id, w.nodes) for w in feats.power_lines] == [(10, (2, 5))]
    assert feats.osm_base == "2026-10-15T00:00:00Z" and len(feats) == 4
    assert gsf.is_street_node({"traffic_sign": "CA:rb-1"})
    assert not gsf.is_street_node({"highway": "bus_stop"})
    with pytest.raises(OverpassError):
        gsf.parse_street_features(b"not json")
    with pytest.raises(OverpassError):
        gsf.parse_street_features({"remark": "runtime error: out of memory", "elements": []})


def test_fetch_caches_and_respects_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("DRIFTMAPS_OFFLINE_DIR", str(tmp_path / "no_db"))
    from driftmaps.geo import offline

    offline.configure(None)
    asked = []

    def fake_post(query, mirrors=None, **kw):
        asked.append(query)
        return json.dumps(OVERPASS).encode()

    monkeypatch.setattr(gsf, "post_query", fake_post)
    with pytest.raises(OverpassError, match="no cached"):
        gsf.fetch_street_features(LINE, cache_dir=tmp_path, offline=True)
    first = gsf.fetch_street_features(LINE, cache_dir=tmp_path)
    again = gsf.fetch_street_features(LINE, cache_dir=tmp_path, offline=True)
    assert len(asked) == 1 and [n.id for n in again.nodes] == [n.id for n in first.nodes]
    cached = list((tmp_path / "overpass").glob("*.json.gz"))
    assert len(cached) == 1 and json.loads(gzip.decompress(cached[0].read_bytes()))["elements"]


# --------------------------------------------------------------------------------------------
# Mapillary
# --------------------------------------------------------------------------------------------


def _row(fid, value, lon, lat, ad=90.0, last="2019-06-01T12:00:00+0000"):
    return {"id": fid, "object_value": value, "geometry": {"type": "Point",
            "coordinates": [lon, lat]}, "aligned_direction": ad, "first_seen_at": last,
            "last_seen_at": last}  # fmt: skip


class FakeApi:
    """A fake graph.mapillary.com: answers from `rows` by bbox; records the requests."""

    def __init__(self, rows, limit=None, status=None):
        self.rows = rows
        self.limit = limit
        self.status = status
        self.requests = []

    def __call__(self, req, timeout):
        self.requests.append(req)
        if self.status:
            raise urllib.error.HTTPError(req.full_url, self.status, "Unauthorized", {}, None)
        from urllib.parse import parse_qs, urlparse

        q = parse_qs(urlparse(req.full_url).query)
        w, s, e, n = (float(v) for v in q["bbox"][0].split(","))
        key = "computed_geometry" if "images" in req.full_url else "geometry"
        out = []
        for r in self.rows:
            lon, lat = r["geometry"]["coordinates"]
            if w <= lon < e and s <= lat < n:
                out.append({**r, key: r["geometry"]} if key != "geometry" else r)
        if self.limit:
            out = out[: self.limit]
        return json.dumps({"data": out}).encode()


def test_tiles_are_on_a_fixed_grid_and_cover_the_line():
    tiles = gm.tiles_along(LINE, 60.0)
    assert tiles == sorted(tiles) and len(tiles) >= 2
    for w, s, e, n in tiles:
        assert round(e - w, 6) == 0.01 and round(n - s, 6) == 0.01
        assert abs(w * 100 - round(w * 100)) < 1e-6  # aligned: neighbours share cached tiles
    for lon, lat in LINE:
        assert any(w <= lon < e and s <= lat < n for w, s, e, n in tiles)


def test_parse_and_stale_filter():
    rows = [
        _row("1", "regulatory--stop--g1", -79.82, 43.441, 45.0),
        _row("2", "object--support--utility-pole", -79.8201, 43.4411, None),
        {"id": "3", "object_value": "x"},  # no geometry: skipped
        _row("4", "object--support--utility-pole", -79.8202, 43.4410, last="2014-05-01T00:00:00Z"),
        _row("5", "object--support--utility-pole", -79.80, 43.45, last="2014-05-01T00:00:00Z"),
        _row("6", "object--support--utility-pole", -79.79, 43.45, last="2012-01-01T00:00:00Z"),
    ]
    feats = gm.parse_map_features(rows)
    assert [f.id for f in feats] == ["1", "2", "4", "5", "6"]
    assert feats[0].aligned_direction == 45.0 and feats[1].aligned_direction is None
    fresh = {f.id for f in gm.drop_stale(feats)}
    # 4: a 2019 photo pass nearby did not see it again; 5: nothing newer near it; 6: < 2013
    assert fresh == {"1", "2", "5"}


def test_fetch_sends_the_token_in_a_header_only_and_caches_gzip(tmp_path):
    rows = [_row("1", "regulatory--stop--g1", -79.8195, 43.4405),
            _row("2", "object--street-light", -79.8105, 43.4391)]  # fmt: skip
    api = FakeApi(rows)
    feats = gm.fetch_map_features(LINE, token=TOKEN, cache_dir=tmp_path, opener=api)
    assert [f.id for f in feats] == ["1", "2"]
    assert api.requests
    for req in api.requests:
        assert TOKEN not in req.full_url and req.get_header("Authorization") == f"OAuth {TOKEN}"
    files = list((tmp_path / "mapillary").glob("*.json.gz"))
    assert len(files) == len(api.requests)
    for f in files:
        assert TOKEN not in f.name and TOKEN.encode() not in gzip.decompress(f.read_bytes())
    # Cached: no token, no network.
    again = gm.fetch_map_features(
        LINE, token=None, cache_dir=tmp_path, opener=FakeApi([], status=500)
    )
    assert [f.id for f in again] == ["1", "2"]


def test_a_full_tile_is_split(tmp_path):
    lons = (-79.8195, -79.819, -79.8185, -79.8145, -79.814, -79.8135)  # 3 per half tile
    rows = [_row(str(k), "object--support--utility-pole", lon, 43.4405)
            for k, lon in enumerate(lons)]  # fmt: skip
    api = FakeApi(rows, limit=4)
    old = gm.LIMIT
    gm.LIMIT = 4
    try:
        feats = gm.fetch_map_features(LINE[:1], token=TOKEN, cache_dir=tmp_path, opener=api,
                                      pad_m=1.0)  # fmt: skip
    finally:
        gm.LIMIT = old
    assert len(feats) == 6 and len(api.requests) > 1


def test_errors_never_show_the_token(tmp_path, monkeypatch):
    monkeypatch.setattr(gm.time, "sleep", lambda s: None)
    with pytest.raises(gm.MapillaryError) as exc:
        gm.fetch_map_features(LINE, token=TOKEN, cache_dir=tmp_path, opener=FakeApi([], status=401))
    assert "token" in str(exc.value) and TOKEN not in str(exc.value)

    def broken(req, timeout):
        raise urllib.error.URLError(f"failed {req.full_url} {TOKEN}")

    with pytest.raises(gm.MapillaryError) as exc:
        gm.fetch_map_features(LINE, token=TOKEN, cache_dir=tmp_path / "b", opener=broken)
    assert TOKEN not in str(exc.value)
    with pytest.raises(gm.MapillaryError, match="no cached"):
        gm.fetch_map_features(LINE, token=None, cache_dir=tmp_path / "c", opener=broken)


def test_image_points(tmp_path):
    rows = [_row("i1", "", -79.8195, 43.4405), _row("i2", "", -79.8105, 43.4391)]
    pts = gm.fetch_image_points(LINE, token=TOKEN, cache_dir=tmp_path, opener=FakeApi(rows))
    assert pts.shape == (2, 2) and np.allclose(pts[0], [-79.8195, 43.4405])
    assert gm.parse_image_points([{"id": "x"}]).shape == (0, 2)
