"""Buildings: footprint fetch/parse/merge, height and roof choice, extrusion, meshes, atlas.

Offline except `test_live_fetch_buildings_beside_snake_road` (marker `network`; skipped unless
run with `-m network` or DRIFTMAPS_NETWORK=1). Real-data fixtures (tests/fixtures/README.md):
`osm/buildings_snake_rd.json` (3 OSM houses + Snake Road ways), `osm/buildings_snake_rd_ms.
geojsonl.gz` (6 Microsoft footprints over the same spot) and the HRDEM DSM/DTM windows
`dem/hrdem_snake_{dsm,dtm}_1m.tif` centred on OSM building 832043557.
"""

from __future__ import annotations

import gzip
import io
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import shapely

from driftmaps.ac import kn5 as kn5mod
from driftmaps.ac.building_textures import (
    ATLAS_U_M,
    BAND_COUNT,
    STYLES,
    band_v,
    building_atlas,
    building_atlas_texture,
)
from driftmaps.ac.scene import Scene
from driftmaps.ac.textures import read_dds_info
from driftmaps.elevation.sources import CogSource, canopy_height_grid, default_sources
from driftmaps.geo import buildings as gb
from driftmaps.geo.buildings import (
    MS_ATTRIBUTION,
    Footprint,
    build_buildings_query,
    fetch_buildings,
    fetch_ms_buildings,
    fetch_osm_buildings,
    merge_footprints,
    ms_tile_urls,
    parse_buildings,
    parse_ms_features,
    quadkey,
    quadkeys_for_bbox,
)
from driftmaps.geo.overpass import OverpassError, parse_overpass
from driftmaps.geo.project import LocalProjection
from driftmaps.geometry.buildings import (
    ROOF_FLAT,
    ROOF_GABLED,
    ROOF_HIPPED,
    Building,
    BuildingOptions,
    Roof,
    _min_rect,
    building_meshes,
    building_scene,
    buildings_polygon,
    choose_height,
    extrude_building,
    place_buildings,
)
from driftmaps.geometry.primitives import face_normals
from driftmaps.geometry.road import RoadModel
from driftmaps.geometry.walls import corridor_polygon

FIX = Path(__file__).parent / "fixtures"
OSM_FIXTURE = FIX / "osm" / "buildings_snake_rd.json"
MS_FIXTURE = FIX / "osm" / "buildings_snake_rd_ms.geojsonl.gz"
DSM_TIF = FIX / "dem" / "hrdem_snake_dsm_1m.tif"
DTM_TIF = FIX / "dem" / "hrdem_snake_dtm_1m.tif"
FIXTURE_HOUSE = "osm:w832043557"  # the house in the middle of the DSM/DTM fixture

PROJ = LocalProjection(43.31024, -79.88979)


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _fp(fid: str, ring_xz, tags=None, ms_height=None, source="osm") -> Footprint:
    """A footprint from an XZ ring (AC: x = east, z = south) on PROJ."""
    r = np.asarray(ring_xz, np.float64)
    r = np.vstack([r, r[:1]])
    lon, lat = PROJ.to_lonlat(r[:, 0], -r[:, 1])
    return Footprint(
        fid, source, np.stack([lon, lat], 1), dict(tags or {"building": "house"}), ms_height
    )


def _rect(x0, z0, w, d):
    return [(x0, z0), (x0 + w, z0), (x0 + w, z0 + d), (x0, z0 + d)]


def _straight_road(length=600.0, half_width=3.0, shoulder=1.0) -> RoadModel:
    """A road along +X through the origin (Y = 0)."""
    x = np.arange(-length / 2, length / 2 + 1e-9, 2.0)
    centre = np.stack([x, np.zeros_like(x), np.zeros_like(x)], 1)
    return RoadModel(x - x[0], centre, half_width, shoulder, True, False)


def _flat(y=100.0):
    return lambda xz: np.full(len(xz), y)


def _building(
    ring,
    shape=ROOF_GABLED,
    rise=3.0,
    ground=10.0,
    eave=5.0,
    sink=0.5,
    wall="siding_white",
    roof="shingle_charcoal",
) -> Building:
    poly = shapely.orient_polygons(shapely.Polygon(ring), exterior_cw=True)
    c, L, S, hl, hw = _min_rect(poly)
    r = Roof(shape, c, L, S, hl, hw, rise if shape != ROOF_FLAT else 0.0)
    return Building(
        "t",
        "osm",
        "house",
        np.asarray(poly.exterior.coords)[:-1],
        r,
        eave,
        ground,
        ground - sink,
        "test",
        wall,
        roof,
        u_offset=7.0,
    )


def _open_edges(mesh, tol=1e-5) -> int:
    """Directed edges without exactly one opposite partner (0 = closed, consistently wound)."""
    key = np.round(mesh.positions / tol).astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    tri = vid.reshape(-1)[mesh.triangles]
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    count = Counter(map(tuple, edges))
    return sum(1 for (a, b), n in count.items() if n != 1 or count.get((b, a), 0) != 1)


def _volume(mesh) -> float:
    p = mesh.positions[mesh.triangles]
    return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0)


_L_SHAPE = [(0, 0), (14, 0), (14, 6), (6, 6), (6, 12), (0, 12)]
_ROTATED_L = [
    (np.cos(0.4) * x - np.sin(0.4) * z + 100.0, np.sin(0.4) * x + np.cos(0.4) * z - 50.0)
    for x, z in _L_SHAPE
]
SHAPES = {
    "rect": _rect(0, 0, 12, 8),
    "square": _rect(0, 0, 9, 9),
    "L": _L_SHAPE,
    "rotated_L": _ROTATED_L,
}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Offline tests must never reach Overpass or Microsoft."""

    def boom(*a, **k):
        raise AssertionError("network access in an offline test")

    monkeypatch.setattr(gb, "post_query", boom)
    monkeypatch.setattr(gb, "_open_url", boom)


# --------------------------------------------------------------------------------------------
# Fixture parse, Microsoft tiles, merge
# --------------------------------------------------------------------------------------------


def test_parse_osm_fixture_keeps_only_closed_building_ways():
    fps = parse_buildings(OSM_FIXTURE.read_bytes())
    assert [f.id for f in fps] == ["osm:w832043557", "osm:w832043679", "osm:w832043713"]
    for f in fps:
        assert f.source == "osm" and f.kind == "yes"
        assert len(f.lonlat) >= 5 and np.allclose(f.lonlat[0], f.lonlat[-1])
        assert f.height_m is None and f.levels is None and f.roof_shape is None
    house = fps[0]
    assert len(house.lonlat) == 11  # the 10-corner house of the DSM fixture
    assert house.lonlat[:, 0].min() == pytest.approx(-79.889884, abs=1e-6)


def test_parse_tags_and_relations():
    payload = {
        "elements": [
            {
                "type": "way",
                "id": 5,
                "tags": {
                    "building": "house",
                    "height": "7.5 m",
                    "building:levels": "2",
                    "roof:shape": "hipped",
                },
                "geometry": [
                    {"lon": 0, "lat": 0},
                    {"lon": 1e-4, "lat": 0},
                    {"lon": 1e-4, "lat": 1e-4},
                    {"lon": 0, "lat": 0},
                ],
            },
            {
                "type": "way",
                "id": 6,
                "tags": {"building": "yes"},  # not closed
                "geometry": [
                    {"lon": 0, "lat": 0},
                    {"lon": 1e-4, "lat": 0},
                    {"lon": 1e-4, "lat": 1e-4},
                    {"lon": 0, "lat": 1e-4},
                ],
            },
            {
                "type": "way",
                "id": 7,
                "tags": {"building": "no"},
                "geometry": [
                    {"lon": 0, "lat": 0},
                    {"lon": 1e-4, "lat": 0},
                    {"lon": 1e-4, "lat": 1e-4},
                    {"lon": 0, "lat": 0},
                ],
            },
            {
                "type": "relation",
                "id": 9,
                "tags": {"building": "barn", "type": "multipolygon"},
                "members": [
                    {
                        "type": "way",
                        "role": "outer",
                        "geometry": [
                            {"lon": 0, "lat": 0},
                            {"lon": 2e-4, "lat": 0},
                            {"lon": 2e-4, "lat": 2e-4},
                        ],
                    },
                    {
                        "type": "way",
                        "role": "outer",
                        "geometry": [
                            {"lon": 2e-4, "lat": 2e-4},
                            {"lon": 0, "lat": 2e-4},
                            {"lon": 0, "lat": 0},
                        ],
                    },
                    {
                        "type": "way",
                        "role": "inner",
                        "geometry": [
                            {"lon": 5e-5, "lat": 5e-5},
                            {"lon": 1e-4, "lat": 5e-5},
                            {"lon": 1e-4, "lat": 1e-4},
                            {"lon": 5e-5, "lat": 5e-5},
                        ],
                    },
                ],
            },
        ]
    }
    fps = parse_buildings(json.dumps(payload))
    assert [f.id for f in fps] == ["osm:w5", "osm:r9.0"]
    w, r = fps
    assert w.height_m == 7.5 and w.levels == 2.0 and w.roof_shape == "hipped"
    assert r.kind == "barn" and len(r.lonlat) == 5  # outer ways joined into one ring
    with pytest.raises(OverpassError):
        parse_buildings(b"<html>busy</html>")


def test_query_uses_small_boxes_along_the_route():
    q = build_buildings_query([(43.3, -79.9, 43.31, -79.89)])
    assert 'way["building"](43.3,-79.9,43.31,-79.89);' in q
    assert 'relation["building"]["type"="multipolygon"](43.3,-79.9,43.31,-79.89);' in q
    assert "out body geom" in q
    # An L-shaped 3.4 km route: one box per km, each covering its piece + 150 m, not the
    # whole 2 x 1.4 km bounding box.
    xy = np.array([[0.0, 0.0], [2000.0, 0.0], [2000.0, 1400.0]])
    lon, lat = PROJ.to_lonlat(xy[:, 0], xy[:, 1])
    dense = np.stack(
        [
            np.interp(np.linspace(0, 2, 341), [0, 1, 2], lon),
            np.interp(np.linspace(0, 2, 341), [0, 1, 2], lat),
        ],
        1,
    )
    boxes = gb.route_boxes(dense, 150.0)
    assert len(boxes) == 4
    for s_, w, n, e in boxes:
        x0, y0 = PROJ.to_enu(w, s_)
        x1, y1 = PROJ.to_enu(e, n)
        assert (x1 - x0) * (y1 - y0) < 1400 * 1400
    lo, la = dense[:, 0], dense[:, 1]
    covered = np.zeros(len(dense), bool)
    for s_, w, n, e in boxes:
        covered |= (lo >= w) & (lo <= e) & (la >= s_) & (la <= n)
    assert covered.all()


def test_parse_ms_fixture_heights_and_bbox():
    lines = gzip.decompress(MS_FIXTURE.read_bytes()).splitlines()
    fps = parse_ms_features(lines, id_prefix="ms:030223132")
    assert len(fps) == 6 and all(f.source == "microsoft" and f.kind == "yes" for f in fps)
    heights = [f.ms_height_m for f in fps]
    assert heights[0] == pytest.approx(5.0678, abs=1e-3) and heights[3] is None  # -1 = unknown
    assert fps[4].id == "ms:030223132:4"
    near = parse_ms_features(lines, bbox=(-79.8900, 43.3100, -79.8895, 43.3105))
    assert [f.ms_height_m for f in near] == [pytest.approx(3.974, abs=1e-3)]
    assert parse_ms_features([b"", b"not json", b'{"geometry": {"type": "Point"}}']) == []


def test_quadkeys():
    assert quadkey(3, 5, 3) == "213"  # Bing Maps Tile System documentation example
    assert quadkeys_for_bbox(-79.8905, 43.3095, -79.8890, 43.3110) == ["030223132"]  # Snake Rd
    assert quadkeys_for_bbox(-79.995, 43.810, -79.985, 43.820) == ["030223130"]  # Forks
    two = quadkeys_for_bbox(-80.17, 43.3, -80.15, 43.31)  # straddles a tile edge (-80.15625)
    assert len(two) == 2


def test_merge_adds_only_microsoft_footprints_osm_does_not_cover():
    osm = parse_buildings(OSM_FIXTURE.read_bytes())
    ms = parse_ms_features(gzip.decompress(MS_FIXTURE.read_bytes()).splitlines())
    merged = merge_footprints(osm, ms)
    # ms:0, ms:1 and ms:4 trace the three OSM houses (verified by overlap); the rest are new.
    assert [f.id for f in merged] == [f.id for f in osm] + ["ms:2", "ms:3", "ms:5"]
    assert merge_footprints(osm, ms, max_overlap=1.0)[3:] == ms
    assert merge_footprints([], ms) == ms and merge_footprints(osm, []) == osm


def _links_csv(url: str) -> bytes:
    return (
        "Location,QuadKey,Url,Size,UploadDate\n"
        "Elsewhere,111111111,https://x/other.csv.gz,1KB,2026-08-13\n"
        f"Canada,030223132,{url},670B,2026-08-13\n"
    ).encode()


def test_ms_tile_urls_pick_the_wanted_quadkeys():
    rows = ms_tile_urls(_links_csv("https://x/tile.csv.gz"), ["030223132", "999"])
    assert rows == [("030223132", "https://x/tile.csv.gz")]


def test_fetch_ms_streams_filters_and_caches(tmp_path):
    calls = []

    def opener(url):
        calls.append(url)
        if url == gb.MS_LINKS_URL:
            return io.BytesIO(_links_csv("https://x/tile.csv.gz"))
        assert url == "https://x/tile.csv.gz"
        return io.BytesIO(MS_FIXTURE.read_bytes())

    bbox = (-79.8905, 43.3095, -79.8890, 43.3110)
    fps = fetch_ms_buildings(bbox, cache_dir=tmp_path, opener=opener)
    assert len(calls) == 2 and 1 <= len(fps) <= 6
    assert all(f.id.startswith("ms:030223132:") for f in fps)
    again = fetch_ms_buildings(bbox, cache_dir=tmp_path, offline=True, opener=opener)
    assert len(calls) == 2  # served from the cache
    assert [f.id for f in again] == [f.id for f in fps]
    assert all(np.allclose(a.lonlat, b.lonlat, atol=1e-8) for a, b in zip(again, fps, strict=True))
    assert fetch_ms_buildings((0.0, 0.0, 0.1, 0.1), cache_dir=tmp_path, offline=True) == []
    with pytest.raises(OSError):  # a listed tile, but this bbox was never cached
        fetch_ms_buildings((-79.80, 43.30, -79.79, 43.31), cache_dir=tmp_path, offline=True)


def _snake_route() -> np.ndarray:
    ways = {w.id: w for w in parse_overpass(OSM_FIXTURE.read_bytes()).ways}
    return np.concatenate([ways[121184920].lonlat, ways[24233890].lonlat[1:]])


def test_fetch_osm_caches_and_offline(tmp_path, monkeypatch):
    route = _snake_route()
    with pytest.raises(OverpassError):
        fetch_osm_buildings(route, cache_dir=tmp_path, offline=True)
    sent = []

    def fake_post(query, mirrors=None, **kw):
        sent.append(query)
        return OSM_FIXTURE.read_bytes()

    monkeypatch.setattr(gb, "post_query", fake_post)
    fps, base = fetch_osm_buildings(route, cache_dir=tmp_path)
    assert len(fps) == 3 and base == "2026-10-15T17:04:49Z" and len(sent) == 1
    assert sent[0].count('way["building"](') == len(gb.route_boxes(route, 155.0)) >= 2
    fps2, _ = fetch_osm_buildings(route, cache_dir=tmp_path, offline=True)
    assert [f.id for f in fps2] == [f.id for f in fps] and len(sent) == 1


def test_fetch_buildings_merges_and_survives_a_microsoft_outage(tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "post_query", lambda q, m=None, **k: OSM_FIXTURE.read_bytes())
    route = _snake_route()

    def broken(url):
        raise OSError("blob store down")

    res = fetch_buildings(route, cache_dir=tmp_path, ms_opener=broken)
    assert res.osm_count == 3 and res.ms_count == 0 and len(res.footprints) == 3
    assert res.notes and "OSM only" in res.notes[0]
    assert res.attributions == [gb.OSM_BUILDINGS_ATTRIBUTION]

    def opener(url):
        if url == gb.MS_LINKS_URL:
            return io.BytesIO(_links_csv("https://x/tile.csv.gz"))
        return io.BytesIO(MS_FIXTURE.read_bytes())

    res = fetch_buildings(route, cache_dir=tmp_path, ms_opener=opener)
    assert res.osm_count == 3 and res.ms_count == 3
    assert {f.id.split(":")[0] for f in res.footprints} == {"osm", "ms"}
    assert MS_ATTRIBUTION in res.attributions and "CDLA-Permissive-2.0" in MS_ATTRIBUTION
    only_osm = fetch_buildings(route, cache_dir=tmp_path, microsoft=False, offline=True)
    assert only_osm.ms_count == 0 and len(only_osm.footprints) == 3


# --------------------------------------------------------------------------------------------
# Height and roof selection
# --------------------------------------------------------------------------------------------


def test_choose_height_order_and_clamps():
    o = BuildingOptions()
    lidar = np.array([6.0, 7.0, 8.0, 9.0, np.nan, 7.5])
    src, h, med, p90 = choose_height(lidar, tag_height=20.0, levels=5, ms_height=4.0)
    assert src == "lidar" and h == pytest.approx(7.5) and med == pytest.approx(7.5)
    assert p90 == pytest.approx(np.percentile([6, 7, 7.5, 8, 9], 90))
    assert choose_height(np.full(20, 55.0))[1] == 30.0  # clamp 3-30 m
    assert choose_height(np.full(20, 2.5))[:2] == ("lidar", 3.0)
    # Lidar sees (almost) nothing: the building is newer than the survey -> fall back.
    assert choose_height(np.full(20, 0.4), levels=2)[:2] == ("levels", 6.0)
    assert choose_height(np.array([9.0, np.nan, np.nan]), levels=2)[0] == "levels"  # too few
    assert choose_height(None, tag_height=11.0, levels=2)[:2] == ("height", 11.0)
    assert choose_height(None, levels=3)[:2] == ("levels", 9.0)
    assert choose_height(None, ms_height=4.7)[:2] == ("microsoft", 4.7)
    assert choose_height(None)[:2] == ("default", o.default_height_m) == ("default", 6.0)


def _place_one(ring, tags=None, canopy=None, ms_height=None, **opts):
    fp = _fp("osm:w1", ring, tags, ms_height)
    out = place_buildings(
        [fp],
        PROJ,
        _straight_road(),
        _flat(),
        canopy=canopy,
        seed="t",
        options=BuildingOptions(**opts),
    )
    assert len(out) == 1
    return out[0]


def test_roof_types_from_tags_kind_size_and_lidar():
    house = _rect(-6, 30, 12, 8)
    b = _place_one(house)
    assert b.roof.shape in (ROOF_GABLED, ROOF_HIPPED) and b.height_source == "default"
    assert b.ridge_h == pytest.approx(6.0) and b.eave_h == pytest.approx(6.0 - b.roof.rise)
    assert b.roof.half_length == pytest.approx(6.0) and b.roof.half_width == pytest.approx(4.0)
    assert abs(b.roof.long_axis[0]) == pytest.approx(1.0)  # ridge along the long (X) side
    assert _place_one(house, {"building": "house", "roof:shape": "hipped"}).roof.shape == "hipped"
    assert _place_one(house, {"building": "house", "roof:shape": "flat"}).roof.shape == "flat"
    assert _place_one(house, {"building": "retail"}).roof.shape == ROOF_FLAT
    assert _place_one(_rect(-20, 30, 40, 20), {"building": "yes"}).roof.shape == ROOF_FLAT
    assert _place_one(_rect(-20, 30, 40, 20), {"building": "barn"}).roof.shape == ROOF_GABLED
    assert _place_one(_rect(-6, 30, 12, 3), {"building": "shed"}).roof.shape == ROOF_FLAT
    flat_lidar = _place_one(house, canopy=lambda xz: np.full(len(xz), 7.0))
    assert flat_lidar.roof.shape == ROOF_FLAT and flat_lidar.eave_h == pytest.approx(7.0)
    assert flat_lidar.height_source == "lidar"
    # A gable seen by lidar: height rises linearly to the ridge (eave 4 m, rise 3 m).
    gable = lambda xz: 4.0 + 3.0 * (1.0 - np.abs(xz[:, 1] - 34.0) / 4.0)  # noqa: E731
    g = _place_one(house, canopy=gable)
    assert g.height_source == "lidar" and g.roof.shape in (ROOF_GABLED, ROOF_HIPPED)
    if g.roof.shape == ROOF_GABLED:  # the 1 m erosion and the lattice are accounted for
        assert g.roof.rise == pytest.approx(3.0, abs=0.05)
        assert g.ridge_h == pytest.approx(7.0, abs=0.3) and g.eave_h == pytest.approx(4.0, abs=0.3)
    else:
        assert g.ridge_h == pytest.approx(7.0, abs=0.6) and g.eave_h == pytest.approx(4.0, abs=1.0)
    lv = _place_one(house, {"building": "house", "building:levels": "2", "roof:shape": "gabled"})
    assert lv.height_source == "levels" and lv.eave_h == pytest.approx(6.0)
    assert lv.ridge_h > lv.eave_h + 1.0


def test_lidar_height_of_the_real_fixture_house(tmp_path):
    real = default_sources()
    srcs = {
        "hrdem-1m-dsm": CogSource(real["hrdem-1m-dsm"].info, str(DSM_TIF)),
        "hrdem-1m": CogSource(real["hrdem-1m"].info, str(DTM_TIF)),
    }

    def canopy(xz):
        lon, lat = PROJ.to_lonlat(xz[:, 0], -xz[:, 1])
        return canopy_height_grid(
            lon, lat, cache_dir=tmp_path, sources=srcs, chain=["hrdem-1m-dsm"]
        ).heights

    osm = parse_buildings(OSM_FIXTURE.read_bytes())
    route = _snake_route()
    e, n = PROJ.to_enu(route[:, 0], route[:, 1])
    xz = np.stack([e, -n], 1)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))])
    road = RoadModel(s, np.stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]], 1), 3.3, 1.0, True, False)
    bs = {
        b.id: b
        for b in place_buildings(
            osm,
            PROJ,
            road,
            _flat(150.0),
            canopy=canopy,
            corridor=corridor_polygon(road, road.outer + 3.0),
        )
    }
    assert set(bs) == {f.id for f in osm}
    house = bs[FIXTURE_HOUSE]
    # DSM - DTM over the roof: 8.49 m at the reference roof pixel (test_elevation_dsm.REF).
    assert house.height_source == "lidar"
    assert 7.5 <= house.ridge_h <= 10.0 and 3.0 <= house.eave_h <= house.ridge_h
    assert house.roof.shape in (ROOF_GABLED, ROOF_HIPPED)
    assert 15.0 < house.road_distance_m < 25.0  # ~28 m from the centreline (README)
    # The other two lie outside the 64 m lidar window: tag/default heights.
    assert {bs[k].height_source for k in bs if k != FIXTURE_HOUSE} == {"default"}


# --------------------------------------------------------------------------------------------
# Corridor exclusion, distance, cap, collision
# --------------------------------------------------------------------------------------------


def test_corridor_exclusion_distance_and_cap():
    road = _straight_road()
    fps = [
        _fp("osm:w1", _rect(0, 30, 10, 8)),  # 30 m off: kept
        _fp("osm:w2", _rect(20, 3, 10, 8)),  # overlaps the asphalt: dropped
        _fp("osm:w3", _rect(40, 5.5, 10, 8)),  # 1.5 m from the drivable edge (4 m): dropped
        _fp("osm:w4", _rect(60, 10, 10, 8)),  # in the corridor (outer + 8 m verge): dropped
        _fp("osm:w5", _rect(80, 310, 10, 8)),  # > 300 m from the centreline: dropped
        _fp("osm:w6", _rect(100, -40, 10, 8)),  # 40 m off on the other side: kept
        _fp("osm:w7", _rect(120, 20, 2, 2)),  # 4 m^2, too small: dropped
    ]
    corridor = corridor_polygon(road, road.outer + 8.0)
    bs = place_buildings(fps, PROJ, road, _flat(), corridor=corridor)
    assert [b.id for b in bs] == ["osm:w1", "osm:w6"]
    assert all(not b.polygon.intersects(corridor) for b in bs)
    no_corridor = place_buildings(fps, PROJ, road, _flat())
    assert [b.id for b in no_corridor] == ["osm:w1", "osm:w4", "osm:w6"]
    capped = place_buildings(fps, PROJ, road, _flat(), options=BuildingOptions(max_buildings=1))
    assert [b.id for b in capped] == ["osm:w4"]  # nearest the road first
    assert buildings_polygon(bs, 2.0).area == pytest.approx(
        2 * (14 * 12 - (4 - np.pi) * 4), rel=1e-2
    )


def test_tree_canopy_over_a_small_house_is_not_its_height():
    house = _rect(-6, 30, 12, 8)
    trees = lambda xz: np.full(len(xz), 18.0)  # noqa: E731
    assert _place_one(house, canopy=trees).height_source == "default"
    big = _place_one(_rect(-20, 30, 30, 20), {"building": "yes"}, canopy=trees)
    assert big.height_source == "lidar" and big.eave_h == pytest.approx(18.0)  # 600 m2
    silo = _place_one(_rect(-3, 30, 6, 6), {"building": "silo"}, canopy=trees)
    assert silo.height_source == "lidar"
    # Overhanging branches inflate p90: the pitch stays <= 45 degrees.
    spiky = lambda xz: np.where(xz[:, 0] > 3.0, 16.0, 6.0)  # noqa: E731
    s = _place_one(house, {"building": "house", "roof:shape": "gabled"}, canopy=spiky)
    assert s.height_source == "lidar" and s.roof.rise <= s.roof.half_width + 1e-9


def test_lidar_failure_falls_back_to_tags():
    def broken(xz):
        raise RuntimeError("HRDEM unreachable")

    b = _place_one(
        _rect(-6, 30, 12, 8), {"building": "house", "building:levels": "1"}, canopy=broken
    )
    assert b.height_source == "levels" and b.eave_h == pytest.approx(3.0)


def test_ground_base_and_no_ground():
    road = _straight_road()
    slope = lambda xz: 100.0 + 0.2 * xz[:, 0]  # noqa: E731
    b = place_buildings([_fp("osm:w1", _rect(0, 30, 10, 8))], PROJ, road, slope)[0]
    assert b.base_y == pytest.approx(100.0 - 0.5, abs=0.05)  # lowest corner - sink
    assert b.ground_y == pytest.approx(101.0, abs=0.3)
    nothing = place_buildings(
        [_fp("osm:w1", _rect(0, 30, 10, 8))], PROJ, road, lambda xz: np.full(len(xz), np.nan)
    )
    assert nothing == []


def test_collision_walls_only_near_the_road():
    road = _straight_road()
    fps = [_fp("osm:w1", _rect(0, 12, 10, 8)), _fp("osm:w2", _rect(0, -60, 10, 8))]
    bs = place_buildings(fps, PROJ, road, _flat())
    assert [b.collide for b in bs] == [True, False]  # 8 m vs 56 m from the drivable edge
    meshes = building_meshes(bs)
    names = [m.name for m in meshes]
    # Cells sort by (x, z) cell index: w2 (z < 0) is cell 000, w1 cell 001.
    assert names == ["buildings_000", "buildings_001", "1WALL_bld_001"]
    _, vis, wall = meshes
    assert not vis.collision and vis.flags == {"lod_out": 1200.0, "cast_shadows": True}
    assert wall.collision and wall.flags == {"is_renderable": False, "cast_shadows": False}
    top = wall.data.positions[:, 1].max()
    assert top == pytest.approx(bs[0].eave_y + 0.5)
    assert wall.data.positions[:, 1].min() == pytest.approx(bs[0].base_y)


# --------------------------------------------------------------------------------------------
# Extrusion validity
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [ROOF_FLAT, ROOF_GABLED, ROOF_HIPPED])
@pytest.mark.parametrize("name", list(SHAPES))
def test_extrusion_is_closed_with_outward_normals(name, shape):
    b = _building(SHAPES[name], shape)
    m = extrude_building(b, floor=True)
    assert _open_edges(m) == 0
    area = shapely.Polygon(SHAPES[name]).area
    prism = area * (b.eave_y - b.base_y)
    vol = _volume(m)
    assert vol > 0  # outward winding
    if shape == ROOF_FLAT:
        assert vol == pytest.approx(prism)
    else:
        assert prism < vol < prism + area * b.roof.rise
    fn = face_normals(m)
    assert np.allclose(m.normals[m.triangles], fn[:, None, :], atol=1e-9)  # flat, matching
    # Walls are vertical; roofs face up; the floor faces down.
    y = fn[:, 1]
    assert np.all((np.abs(y) < 1e-9) | (y > 0.3) | np.isclose(y, -1.0))
    # Without the floor the only open edges are the bottom ring at base_y.
    m2 = extrude_building(b)
    key = np.round(m2.positions / 1e-5).astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    tri = vid.reshape(-1)[m2.triangles]
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    count = Counter(map(tuple, edges))
    open_ = [(a, c) for (a, c) in count if (c, a) not in count]
    ys = {round(float(m2.positions[np.flatnonzero(vid == v)[0], 1]), 6) for e in open_ for v in e}
    assert ys == {round(b.base_y, 6)}


def test_convex_box_faces_point_away_from_the_centre():
    b = _building(_rect(0, 0, 12, 8), ROOF_FLAT, eave=6.0)
    m = extrude_building(b, floor=True)
    fn = face_normals(m)
    centre = np.array([6.0, (b.base_y + b.eave_y) / 2, 4.0])
    tri_centre = m.positions[m.triangles].mean(axis=1)
    assert np.all(np.einsum("ij,ij->i", fn, tri_centre - centre) > 0)


def test_roof_shapes_put_the_ridge_where_expected():
    ring = _rect(0, 0, 16, 8)  # long axis X, half length 8, half width 4
    for shape, ridge_len in ((ROOF_GABLED, 16.0), (ROOF_HIPPED, 8.0)):
        b = _building(ring, shape, rise=3.0)
        m = extrude_building(b)
        top = m.positions[np.isclose(m.positions[:, 1], b.ridge_y)]
        assert len(top) and np.allclose(top[:, 2], 4.0)  # on the centre line along X
        assert np.ptp(top[:, 0]) == pytest.approx(ridge_len)
        assert m.positions[:, 1].max() == pytest.approx(b.ridge_y)
    flat = extrude_building(_building(ring, ROOF_FLAT))
    roof = face_normals(flat)[:, 1] > 0.5
    assert np.allclose(flat.positions[flat.triangles[roof]][..., 1], 15.0)  # eave_y = 10 + 5


def test_uvs_unmirrored_and_inside_one_band():
    for name, ring in SHAPES.items():
        for shape in (ROOF_FLAT, ROOF_GABLED, ROOF_HIPPED):
            b = _building(ring, shape, wall="brick_red", roof="metal_green")
            m = extrude_building(b)
            uv = m.uvs[m.triangles]
            signed = (uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1]) - (
                uv[:, 2, 0] - uv[:, 0, 0]
            ) * (uv[:, 1, 1] - uv[:, 0, 1])
            assert np.all(signed <= 1e-12), (name, shape)
            fn = face_normals(m)
            walls, roofs = np.abs(fn[:, 1]) < 1e-9, fn[:, 1] > 0.3
            for sel, style in ((walls, "brick_red"), (roofs, "metal_green")):
                v0, v1 = band_v(style)
                v = uv[sel][..., 1]
                assert v.min() >= v0 - 1e-9 and v.max() <= v1 + 1e-9, (name, shape, style)
            # Walls upright: v falls as Y rises along every wall triangle edge.
            p = m.positions[m.triangles][walls]
            dy = p[:, 1:, 1] - p[:, :1, 1]
            dv = uv[walls][:, 1:, 1] - uv[walls][:, :1, 1]
            assert np.all(dy * dv <= 1e-12)
            # Wall u is metres along the wall / ATLAS_U_M.
            assert np.isclose(np.ptp(m.uvs[:, 0]) * ATLAS_U_M, np.ptp(m.uvs[:, 0] * ATLAS_U_M))


def test_windows_never_cut_by_the_eave():
    # Whole storeys between ground and eave: a 4.5 m flat building gets two 2.25 m storeys.
    b = _building(_rect(0, 0, 20, 15), ROOF_FLAT, eave=4.5)
    m = extrude_building(b)
    fn = face_normals(m)
    walls = np.abs(fn[:, 1]) < 1e-9
    v0, _ = band_v("siding_white")
    top_verts = m.positions[m.triangles[walls]][..., 1] > b.eave_y - 1e-6
    assert np.allclose(m.uvs[m.triangles[walls]][..., 1][top_verts], v0)  # band top at eave


# --------------------------------------------------------------------------------------------
# Meshes, scene, KN5
# --------------------------------------------------------------------------------------------


def test_scene_kn5_is_valid_and_cells_are_batched():
    road = _straight_road(1200.0)
    fps = [_fp(f"osm:w{i}", _rect(-500 + 37 * i, 30 + (i % 3) * 25, 11, 8)) for i in range(28)]
    bs = place_buildings(fps, PROJ, road, _flat(), seed="track")
    assert len(bs) == 28
    scene = Scene("t", lambda name: pytest.fail("no texture factory expected"))
    stats = building_scene(scene, bs)
    assert stats.buildings == 28 and stats.meshes == len(scene.nodes) >= 4
    names = [n.name for n in scene.nodes]
    assert all(n.startswith("buildings_") for n in names)
    assert all(n.lod_out == 1200.0 for n in scene.nodes)
    k = scene.kn5()
    assert kn5mod.validate(k) == []
    data = kn5mod.kn5_to_bytes(k)
    back = kn5mod.read_kn5(data)
    assert {n.name for n in back.iter_nodes()} >= set(names)
    mat = k.materials[0]
    assert mat.shader == "ksPerPixel" and mat.textures[0].texture_name == "dm_buildings.dds"
    info = read_dds_info(k.textures[0].data)
    assert info.format == "BC1" and (info.width, info.height) == (1024, 4096)
    assert info.mip_count == 13
    assert stats.vertices < 28 * 200  # lean: ~50-120 vertices per house
    # Deterministic: same seed, same looks; meshes identical.
    again = place_buildings(fps, PROJ, road, _flat(), seed="track")
    assert [(b.wall_style, b.roof_style, b.roof.shape) for b in again] == [
        (b.wall_style, b.roof_style, b.roof.shape) for b in bs
    ]
    other = place_buildings(fps, PROJ, road, _flat(), seed="other")
    assert [b.wall_style for b in other] != [b.wall_style for b in bs]
    assert building_scene(Scene("e", lambda n: None), []).meshes == 0


def test_cells_split_past_the_vertex_limit():
    b = _building(_rect(0, 0, 12, 8), ROOF_GABLED)
    many = []
    for i in range(1200):
        bb = _building(_rect((i % 40) * 6.0, (i // 40) * 6.0, 5, 4), ROOF_GABLED)
        bb.id = f"b{i:04d}"
        many.append(bb)
    meshes = building_meshes(many)
    assert len(meshes) >= 2 and all(m.data.vertex_count <= 65535 for m in meshes)
    assert meshes[0].name == "buildings_s000_0"  # 20 m2 each: the short-range batches
    assert meshes[0].flags["lod_out"] == BuildingOptions().detail_lod_out_m
    assert b.area_m2 == pytest.approx(96.0)


# --------------------------------------------------------------------------------------------
# Atlas
# --------------------------------------------------------------------------------------------


def test_atlas_layout_tiling_and_determinism():
    assert len(STYLES) <= BAND_COUNT and sorted(s.band for s in STYLES) == list(range(len(STYLES)))
    img = np.asarray(building_atlas(256, seed=3)).astype(float)
    assert img.shape == (256 * BAND_COUNT // 16, 256, 3)
    assert np.array_equal(img, np.asarray(building_atlas(256, seed=3)))
    # Bands tile left-right: the seam step is like any other column step.
    seam = np.abs(img[:, 0] - img[:, -1]).mean()
    typical = np.abs(np.diff(img, axis=1)).mean()
    assert seam < 2.5 * typical
    for s in STYLES:
        v0, v1 = band_v(s, 256)
        assert s.band / BAND_COUNT < v0 < v1 < (s.band + 1) / BAND_COUNT
    with pytest.raises(ValueError):
        building_atlas(100)
    tex = building_atlas_texture("x.png", 64)
    assert tex.name == "x.png" and tex.data[:4] == b"\x89PNG"


# --------------------------------------------------------------------------------------------
# Live
# --------------------------------------------------------------------------------------------


def _network_enabled(config) -> bool:
    return os.environ.get("DRIFTMAPS_NETWORK") == "1" or "network" in (config.option.markexpr or "")


@pytest.mark.network
def test_live_fetch_buildings_beside_snake_road(request, tmp_path, monkeypatch):
    if not _network_enabled(request.config):
        pytest.skip("live network test: run with -m network or DRIFTMAPS_NETWORK=1")
    monkeypatch.undo()  # re-enable the network for this test
    route = _snake_route()
    res = fetch_buildings(route, cache_dir=tmp_path)
    assert res.osm_count >= 3, res.notes
    assert not res.notes and res.ms_count >= 1
    assert all(f.source in ("osm", "microsoft") for f in res.footprints)
