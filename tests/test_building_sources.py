"""NRCan / municipal building sources, outline regularisation and conflation (offline).

The GeoPackage reader is checked against SQLite itself on a synthetic GeoPackage (R*-tree,
interior B-tree pages, an overflowing record); the municipal fetcher against canned ArcGIS
JSON pages.
"""

from __future__ import annotations

import io
import json
import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest
import shapely

from driftmaps.geo import building_sources as bs
from driftmaps.geo.buildings import BuildingFootprints, Footprint, conflate, regularize
from driftmaps.geo.project import LocalProjection

PROJ = LocalProjection(43.43, -79.85)


def _gpkg_blob(poly: shapely.Geometry) -> bytes:
    x0, y0, x1, y1 = poly.bounds
    return b"GP\x00\x03" + struct.pack("<i4d", 4617, x0, x1, y0, y1) + shapely.to_wkb(poly)


def _make_gpkg(path: Path, n: int = 3000, seed: int = 1) -> list[tuple[int, shapely.Polygon]]:
    """A small GeoPackage like NRCan's: feature table + R*-tree, random small squares."""
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(path)
    con.execute(
        'CREATE TABLE "blds" ("fid" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, "geom" POLYGON,'
        ' "heightmin" REAL, "heightmax" REAL, "datemin" TEXT, "comment" TEXT)'
    )
    con.execute('CREATE VIRTUAL TABLE "rtree_blds_geom" USING rtree(id, minx, maxx, miny, maxy)')
    feats = []
    for fid in range(1, n + 1):
        lon, lat = -79.9 + rng.uniform(0, 0.1), 43.4 + rng.uniform(0, 0.06)
        d = rng.uniform(5e-5, 2e-4)
        if fid == 7:  # a big ring: its record overflows onto overflow pages
            ang = np.linspace(0, 2 * np.pi, 400, endpoint=False)
            poly = shapely.Polygon(np.c_[lon + d * np.cos(ang), lat + d * np.sin(ang)])
        else:
            poly = shapely.box(lon, lat, lon + d, lat + d)
        con.execute(
            'INSERT INTO "blds" VALUES (?, ?, ?, ?, ?, ?)',
            (fid, _gpkg_blob(poly), 2.0, 3.0 + fid % 7, "20230409", "x" * (fid % 50)),
        )
        x0, y0, x1, y1 = poly.bounds
        con.execute('INSERT INTO "rtree_blds_geom" VALUES (?, ?, ?, ?, ?)', (fid, x0, x1, y0, y1))
        feats.append((fid, poly))
    con.commit()
    con.close()
    return feats


@pytest.fixture(scope="module")
def gpkg(tmp_path_factory):
    path = tmp_path_factory.mktemp("gpkg") / "blds.gpkg"
    return path, _make_gpkg(path)


def test_geopackage_reader_matches_sqlite(gpkg):
    path, feats = gpkg
    box = (-79.87, 43.41, -79.84, 43.43)
    gp = bs.GeoPackage(path)
    assert gp.table == "blds" and gp.columns[:3] == ["fid", "geom", "heightmin"]
    con = sqlite3.connect(path)
    want = sorted(
        r[0]
        for r in con.execute(
            'SELECT id FROM "rtree_blds_geom" WHERE maxx >= ? AND minx <= ? AND maxy >= ? '
            "AND miny <= ?",
            (box[0], box[2], box[1], box[3]),
        )
    )
    assert 50 < len(want) < len(feats) and gp.rtree_ids(box) == want
    rows = gp.query(box, ["heightmax", "datemin"])
    assert [r["fid"] for r in rows] == want
    byid = dict(feats)
    for r in rows:
        assert r["geometry"].equals_exact(byid[r["fid"]], 1e-12)
        assert r["heightmax"] == 3.0 + r["fid"] % 7 and r["datemin"] == "20230409"
    # The overflowing record (a 400-vertex ring) reads back whole.
    big = gp.db.lookup(gp.root, [7])[7]
    assert len(big) > 4096
    assert shapely.from_wkb(bs._gpkg_wkb(bs._record(big)[1])).equals_exact(byid[7], 1e-12)


def test_http_pages_merge_runs_and_serve_ranges(gpkg):
    path, _ = gpkg
    blob = path.read_bytes()
    calls = []

    class Resp(io.BytesIO):
        status = 206

    def opener(req):
        a, b = map(int, req.get_header("Range").removeprefix("bytes=").split("-"))
        calls.append((a, b))
        assert "Mozilla" in req.get_header("User-agent")
        return Resp(blob[a : b + 1])

    pages = bs.HttpPages("https://example.invalid/x.gpkg", opener=opener, workers=1)
    gp = bs.GeoPackage(pages)
    box = (-79.87, 43.41, -79.84, 43.43)
    assert gp.rtree_ids(box) == bs.GeoPackage(path).rtree_ids(box)
    pages = bs.HttpPages("https://example.invalid/x.gpkg", opener=opener, workers=1)
    n = len(calls)
    pages.get([5, 6, 7, 9])  # consecutive pages come in one request
    assert len(calls) == n + 2 and calls[-2][1] - calls[-2][0] + 1 == 3 * pages.page_size
    pages.get([5, 9])  # cached
    assert len(calls) == n + 2


def test_fetch_nrcan_caches_per_box(gpkg, tmp_path):
    path, _ = gpkg
    boxes = [(-79.87, 43.41, -79.86, 43.42), (-79.865, 43.415, -79.85, 43.43)]
    fps = bs.fetch_nrcan(boxes, cache_dir=tmp_path, source=path)
    assert fps and len({f.id for f in fps}) == len(fps)
    f = fps[0]
    assert f.source == "nrcan" and f.lidar and f.year == 2023 and f.low_m == 2.0
    again = bs.fetch_nrcan(boxes, cache_dir=tmp_path, source=path, offline=True)
    assert [g.id for g in again] == [g.id for g in fps]
    assert np.allclose(again[0].lonlat, fps[0].lonlat, atol=1e-7)
    with pytest.raises(bs.SourceError):
        bs.fetch_nrcan([(-79.0, 43.0, -78.99, 43.01)], cache_dir=tmp_path, offline=True)


def _esri(oid, ring, **attrs):
    return {"attributes": {"OBJECTID": oid, **attrs}, "geometry": {"rings": ring}}


def test_municipal_features_heights_kinds_and_paging(tmp_path):
    cw = [[-79.85, 43.43], [-79.85, 43.4301], [-79.8499, 43.4301], [-79.8499, 43.43],
          [-79.85, 43.43]]  # fmt: skip
    hole = cw[::-1]
    feat = _esri(1, [cw, hole], PEAKHGT=6.5, EAVEHGT=3.2, BUILDINGTYPE="RESIDENTIAL",
                 BUILDINGSUBTYPE="Single Family Dwelling")  # fmt: skip
    (f,) = bs.municipal_feature(bs.BURLINGTON, feat)  # the hole is dropped
    assert f.id == "burlington:1" and f.peak_m == 6.5 and f.eave_m == 3.2 and f.year == 2017
    assert f.tags == {"building": "house"} and not f.lidar
    bad = bs.municipal_feature(bs.BURLINGTON, _esri(2, [cw], PEAKHGT=0.0, EAVEHGT=9.0))[0]
    assert bad.peak_m is None and bad.eave_m == 9.0
    pages = [
        {"features": [_esri(i, [cw]) for i in range(3)], "exceededTransferLimit": True},
        {"features": [_esri(i, [cw]) for i in range(3, 5)]},
    ]
    seen = []

    def get_json(url, params):
        seen.append(params["resultOffset"])
        assert url.endswith("/query") and params["outSR"] == 4326
        return pages[len(seen) - 1]

    box = (-79.86, 43.42, -79.84, 43.44)
    got = bs.fetch_municipal(bs.MILTON, [box], cache_dir=tmp_path, get_json=get_json)
    assert seen == [0, 3] and [g.id for g in got] == [f"milton:{i}" for i in range(5)]
    again = bs.fetch_municipal(bs.MILTON, [box], cache_dir=tmp_path, offline=True)
    assert [g.id for g in again] == [g.id for g in got]
    far = bs.fetch_municipal(bs.MILTON, [(-75.0, 45.0, -74.9, 45.1)], cache_dir=tmp_path,
                             offline=True)  # fmt: skip
    assert far == []  # outside the layer: not even looked up


# --------------------------------------------------------------------------------------------
# Regularisation and conflation (metres on PROJ; AC XZ: z = -north)
# --------------------------------------------------------------------------------------------


def _fp(fid, xz, source="osm", **kw) -> Footprint:
    r = np.asarray(xz, np.float64)
    r = np.vstack([r, r[:1]])
    lon, lat = PROJ.to_lonlat(r[:, 0], -r[:, 1])
    tags = kw.pop("tags", {"building": "yes"})
    return Footprint(fid, source, np.stack([lon, lat], 1), tags, **kw)


def _rect(x, z, w, d):
    return [(x, z), (x + w, z), (x + w, z + d), (x, z + d)]


def _stairs(x, z, w, d, step=0.5):
    """A w x d rectangle with lidar-like 0.5 m stair steps along two sides."""
    pts = [(x, z)]
    k = 0
    while (k + 1) * step < w:
        pts += [(x + (k + 1) * step, z + (0.3 if k % 2 == 0 else 0.0))]
        k += 1
    pts += [(x + w, z), (x + w, z + d), (x, z + d)]
    return pts


def test_regularize_rectangles_and_l_shapes():
    rough = shapely.Polygon(_stairs(0, 0, 12, 9))
    r = regularize(rough)
    assert len(r.exterior.coords) == 5 and r.area == pytest.approx(rough.area, rel=1e-6)
    ell = shapely.Polygon([(0, 0), (14, 0), (14, 6), (6.2, 6), (6, 6.3), (6, 12), (0, 12)])
    out = regularize(ell.simplify(0))
    coords = np.asarray(out.exterior.coords)[:-1]
    assert len(coords) == 6  # the 0.3 m jog is gone, the L stays an L
    d = np.diff(np.vstack([coords, coords[:1]]), axis=0)
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 90.0
    assert np.all((ang < 1e-6) | (ang > 90 - 1e-6))
    tri = shapely.Polygon([(0, 0), (10, 0), (5, 9)])
    assert regularize(tri).symmetric_difference(tri).area < 0.05 * tri.area  # left alone


def test_conflate_priority_iou_overlap_cut_and_confirmation():
    muni = [
        _fp("burlington:1", _rect(0, 0, 12, 10), "burlington", peak_m=6.0, eave_m=3.0,
            tags={"building": "yes"}),
        _fp("burlington:2", _rect(40, 0, 6, 6), "burlington", peak_m=5.0, eave_m=2.5),
        _fp("burlington:3", _rect(46.5, 0, 6, 6), "burlington", peak_m=4.0, eave_m=2.5),
    ]  # fmt: skip
    osm = [
        _fp("osm:w1", _rect(0.5, 0.3, 12, 10), tags={"building": "house", "name": "Farm"}),
        _fp("osm:w2", _rect(200, 0, 10, 10), tags={"building": "house"}),  # new: lidar says no
        _fp("osm:w3", _rect(300, 0, 10, 10), tags={"building": "house"}),  # new: NRCan has it
    ]
    nrcan = [
        _fp("nrcan:1", _stairs(0, 0, 12, 10), "nrcan", lidar=True, peak_m=6.4, year=2023),
        _fp("nrcan:2", _rect(39.5, -0.5, 13.5, 7), "nrcan", lidar=True, peak_m=5.2, year=2023),
        _fp("nrcan:3", _rect(100, 0, 20, 30), "nrcan", lidar=True, peak_m=8.0, year=2023),
        _fp("nrcan:4", _rect(300, 0, 10.2, 10), "nrcan", lidar=True, peak_m=6.0, year=2023),
        _fp("nrcan:5", _rect(11, 9, 6, 8), "nrcan", lidar=True, peak_m=4.0, year=2023),  # cut
        _fp("nrcan:6", _rect(60, 30, 4, 4), "nrcan", lidar=True, peak_m=3.0, year=2023),  # RV?
        _fp("nrcan:7", _rect(400, 0, 4, 4), "nrcan", lidar=True, peak_m=1.2, year=2023),  # low
    ]
    ms = [_fp("ms:a:1", _rect(100.5, 0.5, 20, 30), "microsoft", ms_height_m=7.0)]

    def canopy(xz):  # 7 m over the NRCan barn and the OSM-only house at 300 m, else ground
        x = xz[:, 0]
        return np.where(((x > 100) & (x < 120)) | ((x > 300) & (x < 310)), 7.0, 0.2)

    out, rep = conflate([muni, osm, nrcan, ms], PROJ, canopy)
    ids = {f.id: f for f in out}
    assert set(ids) == {"burlington:1", "burlington:2", "burlington:3", "nrcan:3", "osm:w3",
                        "nrcan:5"}  # fmt: skip
    b1 = ids["burlington:1"]
    assert b1.tags["building"] == "house" and b1.tags["name"] == "Farm"  # OSM tags merged
    assert b1.peak_m == 6.0 and set(b1.also) == {"osm:w1", "nrcan:1"}
    assert ids["osm:w3"].peak_m == 6.0  # NRCan height merged into the OSM outline
    assert "ms:a:1" in ids["nrcan:3"].also and ids["nrcan:3"].ms_height_m == 7.0
    cut = shapely.Polygon(PROJ.to_xy(ids["nrcan:5"].lonlat))
    assert cut.area < 47.5 and not cut.buffer(-0.05).intersects(
        shapely.Polygon(PROJ.to_xy(b1.lonlat)))  # fmt: skip
    assert rep.overlapping >= 1  # nrcan:2 over two municipal pieces
    assert rep.unconfirmed == 3  # osm:w2 (lidar 0.2 m), nrcan:6 (small, survey), nrcan:7 (low)
    assert rep.kept == {"burlington": 3, "nrcan": 2, "osm": 1}


def test_conflate_flags_demolished_and_needs_lidar_to_do_so():
    muni = [_fp("burlington:9", _rect(0, 0, 10, 10), "burlington", peak_m=6.0, year=2017)]
    nrcan = [_fp("nrcan:1", _rect(200, 0, 10, 10), "nrcan", lidar=True, peak_m=6.0, year=2023)]
    empty_lot = lambda xz: np.full(len(xz), 0.3)  # noqa: E731
    out, rep = conflate([muni, [], nrcan, []], PROJ, empty_lot)
    assert rep.demolished == ["burlington:9"] and [f.id for f in out] == ["nrcan:1"]
    out, rep = conflate([muni, [], nrcan, []], PROJ, None)  # no DSM: keep it
    assert rep.demolished == [] and len(out) == 2
    house = lambda xz: np.full(len(xz), 5.0)  # noqa: E731
    assert conflate([muni, [], nrcan, []], PROJ, house)[1].demolished == []


def test_osm_is_the_base_without_a_survey():
    osm = [_fp("osm:w1", _rect(0, 0, 10, 10))]
    out, rep = conflate([[], osm, [], []], PROJ, lambda xz: np.zeros(len(xz)))
    assert [f.id for f in out] == ["osm:w1"] and rep.unconfirmed == 0


def test_attributions_follow_geometry_and_merged_sources():
    a = _fp("burlington:1", _rect(0, 0, 10, 10), "burlington", also=("nrcan:4", "osm:w1"))
    b = _fp("ms:q:3", _rect(50, 0, 10, 10), "microsoft")
    res = BuildingFootprints([a, b])
    lines = res.attributions_for({"burlington:1"})
    assert len(lines) == 3
    assert "OpenStreetMap" in lines[0] and "Burlington" in lines[1] and "NRCan" not in lines[1]
    assert bs.BURLINGTON_TERMS_URL in lines[1] and "Natural Resources Canada" in lines[2]
    assert any("Microsoft" in x for x in res.attributions)
    assert json.dumps(lines)  # plain strings
