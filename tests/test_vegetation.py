"""Tests for trees: placement (`driftmaps.geometry.vegetation`), card meshes and the procedural
atlas (`driftmaps.ac.tree_textures`).

Offline: a synthetic canopy height model (paraboloid crowns, a pole, a wire, a bridge deck, a flat
canopy plateau) along a straight 600 m road; the real 64 x 64 m HRDEM DSM/DTM pair beside Snake
Road (tests/fixtures/dem/hrdem_snake_{dsm,dtm}_1m.tif); and a 48 x 48 px cut of NRCan Land Cover
2020 around the same spot (tests/fixtures/landcover/landcover2020_snake_30m.tif). See
tests/fixtures/README.md. The `network` test reads the live land cover COG.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
import rasterio
import shapely
from rasterio.warp import transform as warp_transform
from scipy.spatial import cKDTree

from driftmaps.ac.kn5 import kn5_to_bytes, read_kn5
from driftmaps.ac.model import MAX_MESH_VERTICES, BlendMode, DepthMode, Mesh
from driftmaps.ac.scene import Scene
from driftmaps.ac.textures import decode_dds_level, mip_chain, read_dds_info
from driftmaps.ac.tree_textures import (
    BUSH,
    CONIFER,
    DECIDUOUS,
    TREE_TILES,
    alpha_coverage,
    alpha_tested_mips,
    encode_alpha_tested_dds,
    tile_pixel_box,
    tile_uv,
    tiles_of_kind,
    tree_atlas,
    tree_atlas_texture,
)
from driftmaps.elevation.sources import CogSource, ElevationSource, default_sources
from driftmaps.geo.project import LocalProjection
from driftmaps.geometry.road import RoadModel
from driftmaps.geometry.vegetation import (
    LANDCOVER_ATTRIBUTION,
    MIXED,
    OSM_WOODS_ATTRIBUTION,
    CanopySampler,
    LandcoverSampler,
    TreeMeshOptions,
    TreeSet,
    VegetationOptions,
    WoodArea,
    detect_tree_tops,
    place_trees,
    sample_landcover,
    tree_material,
    tree_meshes,
    tree_scene,
    woods_from_overpass,
    woods_query,
)

FIXTURES = Path(__file__).parent / "fixtures"
DSM_TIF = FIXTURES / "dem" / "hrdem_snake_dsm_1m.tif"
DTM_TIF = FIXTURES / "dem" / "hrdem_snake_dtm_1m.tif"
LANDCOVER_TIF = FIXTURES / "landcover" / "landcover2020_snake_30m.tif"
SNAKE_HOUSE = (-79.88979, 43.31024)  # OSM building at the centre of both fixtures
SNAKE_TALLEST = (-79.8900168277, 43.3104533266, 19.648)  # lon, lat, CHM (test_elevation_dsm)
SNAKE_ROOF = (-79.8897903099, 43.3102417704)


@pytest.fixture(autouse=True)
def _network_opt_in(request):
    if request.node.get_closest_marker("network") and "network" not in (
        request.config.option.markexpr or ""
    ):
        pytest.skip("live network test: run with `pytest -m network`")


# ---------------------------------------------------------------------------------------------
# Synthetic world: a straight road along X (x = -300..300, z = 0), asphalt 3 m + shoulder 1 m
# each side, so the footprint edge is at |z| = 4 and the clearance edge at |z| = 5.5.
# ---------------------------------------------------------------------------------------------

BRIDGE_X = (-20.0, 20.0)


def _road(bridge: bool = True) -> RoadModel:
    s = np.arange(0.0, 601.0)
    x = s - 300.0
    centre = np.stack([x, np.zeros_like(s), np.zeros_like(s)], axis=1)
    on_bridge = (x >= BRIDGE_X[0]) & (x <= BRIDGE_X[1]) if bridge else np.zeros(len(s), bool)
    return RoadModel(
        s=s, centre=centre, half_width=3.0, shoulder=1.0, centre_line=False, bridge=on_bridge
    )


# Planted crowns (x, z, height, radius) that must be found, all well clear of the road.
PLANTED = [
    (-250.0, 15.0, 18.0, 4.0),
    (-230.0, -22.0, 9.0, 3.0),
    (-200.0, 40.0, 24.0, 5.0),
    (-180.0, -60.0, 6.0, 2.5),
    (-150.0, 12.0, 14.0, 3.5),
    (-120.0, -14.0, 20.0, 4.5),
    (-60.0, 100.0, 16.0, 4.0),
    (60.0, -30.0, 12.0, 3.0),
    (80.0, 25.0, 22.0, 5.0),
    (150.0, -80.0, 10.0, 3.0),
    (200.0, 130.0, 15.0, 4.0),
    (240.0, -11.0, 17.0, 4.0),
]
ON_ROAD = (-80.0, 0.5, 15.0, 4.0)  # trunk on the asphalt: no tree
OVERHANG = (-20.0 - 90.0, 4.6, 13.0, 4.0)  # top 0.6 m past the footprint edge: pushed out
NEAR_BRIDGE = (5.0, 9.0, 12.0, 3.0)  # 5 m beyond the bridge footprint: excluded
POLE = (30.0, 60.0)  # 1 x 1 m, 10 m tall
WIRE = (-100.0, -40.0, 30.0)  # x from, x to, z: 1 m wide, 9 m up
PLATEAU = (100.0, 130.0, 30.0, 60.0)  # x0, x1, z0, z1: perfectly flat canopy, 12 m
FAR_GRID = [(x, z) for x in np.arange(-240.0, 241.0, 12.0) for z in (190.0, 202.0, 214.0, 226.0)]


def _crowns(extra=()):
    crowns = list(PLANTED) + [ON_ROAD, OVERHANG, NEAR_BRIDGE] + list(extra)
    return np.array(crowns, np.float64)


def _synthetic_chm(far_forest: bool = False):
    crowns = _crowns([(x, z, 14.0, 4.0) for x, z in FAR_GRID] if far_forest else [])

    def chm(xz: np.ndarray) -> np.ndarray:
        x, z = xz[:, 0], xz[:, 1]
        out = np.zeros(len(xz))
        for cx, cz, h, r in crowns:
            near = (np.abs(x - cx) < r) & (np.abs(z - cz) < r)
            if near.any():
                r2 = ((x[near] - cx) ** 2 + (z[near] - cz) ** 2) / r**2
                out[near] = np.maximum(out[near], np.where(r2 < 1, h * (1 - r2), 0.0))
        out[(np.abs(x - POLE[0]) < 0.5) & (np.abs(z - POLE[1]) < 0.5)] = 10.0
        wire = (x >= WIRE[0]) & (x <= WIRE[1]) & (np.abs(z - WIRE[2]) < 0.5)
        out[wire] = np.maximum(out[wire], 9.0)
        x0, x1, z0, z1 = PLATEAU
        out[(x >= x0) & (x <= x1) & (z >= z0) & (z <= z1)] = 12.0
        deck = (x >= BRIDGE_X[0] - 5) & (x <= BRIDGE_X[1] + 5) & (np.abs(z) <= 4.0)
        out[deck] = 10.0
        return out

    return chm


FAST = VegetationOptions(far_m=150.0)  # near band only


@pytest.fixture(scope="module")
def synthetic_trees() -> TreeSet:
    return place_trees(_road(), _synthetic_chm(), seed="dm_test_road", options=FAST)


def _nearest(trees: TreeSet, x: float, z: float) -> tuple[float, int]:
    d = np.hypot(trees.xz[:, 0] - x, trees.xz[:, 1] - z)
    i = int(np.argmin(d))
    return float(d[i]), i


# ---------------------------------------------------------------------------------------------
# Placement on the synthetic CHM
# ---------------------------------------------------------------------------------------------


def test_finds_every_planted_tree_with_its_height(synthetic_trees):
    t = synthetic_trees
    for x, z, h, _ in PLANTED:
        dist, i = _nearest(t, x, z)
        assert dist <= 1.5, (x, z, dist)
        assert t.height[i] == pytest.approx(h, rel=0.08), (x, z)
        assert t.source[i] == 0 and t.band[i] == 0
        assert 1.5 <= t.radius[i] <= 6.0


def test_keeps_trees_out_of_the_road_corridor_and_off_bridges(synthetic_trees):
    t = synthetic_trees
    assert len(t) > len(PLANTED)
    edge = np.abs(t.xz[:, 1]) - 4.0  # distance beyond the footprint (straight road)
    assert edge.min() >= 1.5 - 1e-6
    assert np.allclose(t.distance, edge, atol=0.6)
    assert _nearest(t, ON_ROAD[0], ON_ROAD[1])[0] > 4.0  # the tree on the asphalt is gone
    # Nothing on or beside the bridge (deck + 6 m margin, padded 10 m along the road).
    on_bridge = (np.abs(t.xz[:, 0]) <= 30.0) & (np.abs(t.xz[:, 1]) < 4.0 + 6.0)
    assert not on_bridge.any()
    assert _nearest(t, NEAR_BRIDGE[0], NEAR_BRIDGE[1])[0] > 2.0


def test_pushes_overhanging_roadside_tops_out_to_the_clearance_edge(synthetic_trees):
    dist, i = _nearest(synthetic_trees, OVERHANG[0], 5.75)
    assert dist <= 1.0
    assert synthetic_trees.xz[i, 1] >= 5.5
    assert synthetic_trees.height[i] == pytest.approx(OVERHANG[2], rel=0.08)


def test_poles_and_wires_are_not_trees(synthetic_trees):
    t = synthetic_trees
    assert _nearest(t, *POLE)[0] > 3.0
    x, z = t.xz.T
    on_wire = (x >= WIRE[0] - 2) & (x <= WIRE[1] + 2) & (np.abs(z - WIRE[2]) < 3)
    assert not on_wire.any()


def test_flat_canopy_gets_blue_noise_gap_fill(synthetic_trees):
    t = synthetic_trees
    x0, x1, z0, z1 = PLATEAU
    inside = (t.xz[:, 0] >= x0) & (t.xz[:, 0] <= x1) & (t.xz[:, 1] >= z0) & (t.xz[:, 1] <= z1)
    pts = t.xz[inside]
    assert 20 <= len(pts) <= 80  # ~900 m2 at one tree per ~20 m2
    d = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 0.7 * 4.5 - 1e-6
    assert np.allclose(t.height[inside], 12.0)
    assert (t.source[inside] == 2).all()  # gap fill, not lidar tops


def test_placement_is_deterministic_and_seeded(synthetic_trees):
    again = place_trees(_road(), _synthetic_chm(), seed="dm_test_road", options=FAST)
    for col in TreeSet._COLUMNS:
        assert np.array_equal(getattr(again, col), getattr(synthetic_trees, col)), col
    other = place_trees(_road(), _synthetic_chm(), seed="another_route", options=FAST)
    assert not np.array_equal(other.angle[: len(PLANTED)], synthetic_trees.angle[: len(PLANTED)])
    for x, z, _, _ in PLANTED:  # lidar tops don't move with the seed
        assert _nearest(other, x, z)[0] <= 1.5


def test_detect_tree_tops_on_a_grid():
    chm = np.zeros((40, 40))
    yy, xx = np.mgrid[0:40, 0:40]
    for r, c, h in ((10, 10, 15.0), (25, 30, 8.0)):
        chm = np.maximum(chm, h * np.clip(1 - ((yy - r) ** 2 + (xx - c) ** 2) / 16.0, 0, None))
    chm[0:3, 30:33] = np.nan
    rows, cols, h = detect_tree_tops(chm, 1.0)
    assert sorted(zip(rows.tolist(), cols.tolist(), strict=True)) == [(10, 10), (25, 30)]
    assert np.allclose(sorted(h), [8.0, 15.0])


def test_far_band_is_thinned_into_bigger_clumps():
    opts = VegetationOptions(far_m=260.0)
    t = place_trees(_road(), _synthetic_chm(far_forest=True), seed=3, options=opts)
    far = t.band == 1
    assert far.any() and (t.distance[far] > 150.0).all()
    share = far.sum() / len(FAR_GRID)
    assert 0.15 <= share <= 0.55
    assert np.median(t.radius[far]) > 1.3 * np.median(t.radius[~far])


def test_budget_keeps_a_far_share_and_drops_distant_small_trees():
    chm = _synthetic_chm(far_forest=True)
    full = place_trees(_road(), chm, seed=3, options=VegetationOptions(far_m=260.0))
    n_far = int((full.band == 1).sum())
    assert n_far >= 20
    t = place_trees(_road(), chm, seed=3, options=VegetationOptions(far_m=260.0, max_trees=40))
    assert len(t) == 40
    assert int((t.band == 1).sum()) == 12  # 30 % for the far band
    near = full.subset(full.band == 0)
    kept = t.subset(t.band == 0)
    assert np.median(kept.distance) <= np.median(near.distance)
    assert (kept.source == 2).mean() <= (near.source == 2).mean()  # gap fill goes first
    # Unused far share flows to the near band.
    no_far = place_trees(_road(), _synthetic_chm(), seed=3,
                         options=VegetationOptions(far_m=260.0, max_trees=40))
    assert len(no_far) == 40 and (no_far.band == 0).all()
    per_km = VegetationOptions(far_m=150.0, max_trees_per_km=20.0)  # 600 m road -> 12 trees
    assert len(place_trees(_road(), _synthetic_chm(), options=per_km)) == 12


def test_exclude_geometry_removes_trees(synthetic_trees):
    x, z = PLANTED[4][:2]
    zone = shapely.Point(x, z).buffer(6.0)
    t = place_trees(_road(), _synthetic_chm(), seed="dm_test_road", options=FAST, exclude=zone)
    assert _nearest(t, x, z)[0] > 3.0
    assert len(t) < len(synthetic_trees)


def test_land_cover_decides_the_kind():
    chm = _synthetic_chm(far_forest=True)
    opts = VegetationOptions(far_m=260.0, far_keep=1.0)

    def share(cls: int | None, kind: int) -> float:
        lc = None if cls is None else (lambda xz: np.full(len(xz), cls, np.uint8))
        t = place_trees(_road(), chm, landcover=lc, seed=5, options=opts)
        return float((t.kind == kind).mean())

    assert share(1, CONIFER) >= 0.75  # needleleaf forest
    assert share(5, DECIDUOUS) >= 0.75  # broadleaf forest
    assert share(15, DECIDUOUS) >= 0.6  # cropland hedgerows
    assert 0.3 <= share(None, CONIFER) <= 0.7  # no land cover: mixed
    water = place_trees(
        _road(), chm, landcover=lambda xz: np.full(len(xz), 18, np.uint8), options=opts
    )
    assert len(water) == 0


def test_shrubland_keeps_low_bushes():
    crowns = [(x, 20.0, 2.5, 2.0) for x in np.arange(-200.0, 201.0, 10.0)]

    def chm(xz):
        out = np.zeros(len(xz))
        for cx, cz, h, r in crowns:
            r2 = ((xz[:, 0] - cx) ** 2 + (xz[:, 1] - cz) ** 2) / r**2
            out = np.maximum(out, np.where(r2 < 1, h * (1 - r2), 0.0))
        return out

    none = place_trees(_road(False), chm, options=FAST)
    assert len(none) == 0  # below 3 m outside shrubland
    lc8 = place_trees(
        _road(False), chm, options=FAST, landcover=lambda xz: np.full(len(xz), 8, np.uint8)
    )
    assert len(lc8) >= 0.8 * len(crowns)
    assert (lc8.kind == BUSH).mean() >= 0.6
    assert set(lc8.tile[lc8.kind == BUSH].tolist()) <= set(tiles_of_kind(BUSH))


def test_tiles_match_kinds(synthetic_trees):
    t = synthetic_trees
    for k in (DECIDUOUS, CONIFER, BUSH):
        assert set(t.tile[t.kind == k].tolist()) <= set(tiles_of_kind(k))


# ---------------------------------------------------------------------------------------------
# OSM fallback
# ---------------------------------------------------------------------------------------------


def _no_lidar(xz):
    return np.full(len(xz), np.nan)


def test_osm_woods_fill_only_where_there_is_no_lidar():
    wood = WoodArea(shapely.box(-100.0, -60.0, 0.0, 60.0), MIXED)
    t = place_trees(_road(False), _no_lidar, woods=[wood], seed=9, options=FAST)
    assert len(t) > 100
    assert (t.source == 1).all()
    assert shapely.contains_xy(wood.polygon.buffer(1e-6), t.xz[:, 0], t.xz[:, 1]).all()
    assert (np.abs(t.xz[:, 1]) - 4.0 >= 1.5 - 1e-6).all()
    assert ((t.height >= 8.0) & (t.height <= 22.0)).all()
    assert 0.25 <= (t.kind == CONIFER).mean() <= 0.75
    assert OSM_WOODS_ATTRIBUTION in t.attributions
    again = place_trees(_road(False), _no_lidar, woods=[wood], seed=9, options=FAST)
    assert np.array_equal(again.xz, t.xz)
    # Lidar present (an empty field): the CHM wins, no OSM trees.
    bare = place_trees(_road(False), lambda xz: np.zeros(len(xz)), woods=[wood], options=FAST)
    assert len(bare) == 0
    # No canopy function at all (outside Canada): OSM everywhere.
    assert len(place_trees(_road(False), None, woods=[wood], seed=9, options=FAST)) == len(t)


def test_woods_from_overpass_builds_polygons_with_holes():
    proj = LocalProjection(43.3, -79.9)

    def ring(e0, n0, e1, n1):
        pts = [(e0, n0), (e1, n0), (e1, n1), (e0, n1), (e0, n0)]
        lon, lat = proj.to_lonlat([p[0] for p in pts], [p[1] for p in pts])
        return [{"lat": float(a), "lon": float(o)} for o, a in zip(lon, lat, strict=True)]

    outer = ring(0, 0, 100, 100)
    doc = {
        "elements": [
            {"type": "way", "id": 1, "tags": {"landuse": "forest", "leaf_type": "needleleaved"},
             "geometry": ring(200, 0, 250, 50)},
            {"type": "way", "id": 2, "tags": {"natural": "scrub"},
             "geometry": ring(300, 0, 310, 10)},
            {"type": "relation", "id": 3, "tags": {"natural": "wood", "type": "multipolygon"},
             "members": [
                 {"type": "way", "role": "outer", "geometry": outer[:3]},
                 {"type": "way", "role": "outer", "geometry": outer[2:]},
                 {"type": "way", "role": "inner", "geometry": ring(40, 40, 60, 60)},
             ]},
        ]
    }
    areas = woods_from_overpass(doc, proj)
    assert [a.kind for a in areas] == [CONIFER, BUSH, MIXED]
    assert areas[0].polygon.area == pytest.approx(2500.0, rel=1e-3)
    assert areas[2].polygon.area == pytest.approx(100 * 100 - 20 * 20, rel=1e-3)
    # AC Z = south: the relation spans north 0..100 -> Z -100..0.
    assert areas[2].polygon.bounds[1] == pytest.approx(-100.0, abs=0.01)
    q = woods_query(43.3, -79.9, 43.4, -79.8)
    assert "out geom" in q and '"landuse"="forest"' in q and "43.300000,-79.900000" in q


# ---------------------------------------------------------------------------------------------
# Real lidar (Snake Road fixture) and land cover fixture
# ---------------------------------------------------------------------------------------------


class _NoData(ElevationSource):
    """A source with no data anywhere (points outside the fixture fall through to NaN)."""

    def __init__(self, name: str) -> None:
        real = default_sources()[name]
        super().__init__(real.info)

    def sample(self, lon, lat, cache_dir):
        return np.full(np.shape(lon), np.nan)


def _fixture_sources():
    real = default_sources()
    srcs = {name: _NoData(name) for name in real}
    srcs["hrdem-1m-dsm"] = CogSource(real["hrdem-1m-dsm"].info, str(DSM_TIF))
    srcs["hrdem-1m"] = CogSource(real["hrdem-1m"].info, str(DTM_TIF))
    return srcs


def test_real_lidar_finds_the_tallest_tree_and_skips_the_masked_roof(tmp_path):
    proj = LocalProjection(SNAKE_HOUSE[1], SNAKE_HOUSE[0])
    # Snake Road runs N-S about 28 m west of the house: a straight stand-in at X = -27.
    z = np.arange(-60.0, 61.0)
    centre = np.stack([np.full_like(z, -27.0), np.zeros_like(z), z], axis=1)
    road = RoadModel(s=z + 60.0, centre=centre, half_width=3.0, shoulder=1.0,
                     centre_line=False, bridge=False)
    canopy = CanopySampler(proj, cache_dir=tmp_path, sources=_fixture_sources())
    roof = proj.lonlat_to_ac(*SNAKE_ROOF)[[0, 2]]
    trees = place_trees(
        road, canopy, seed="dm_snake_road", options=VegetationOptions(far_m=0.0),
        exclude=shapely.Point(*roof).buffer(7.0),
    )
    tall = proj.lonlat_to_ac(SNAKE_TALLEST[0], SNAKE_TALLEST[1])[[0, 2]]
    dist, i = _nearest(trees, *tall)
    assert dist <= 2.5
    assert trees.height[i] == pytest.approx(SNAKE_TALLEST[2], abs=1.0)
    assert np.hypot(*(trees.xz - roof).T).min() > 7.0
    # Only where the 64 m lidar window has data (EPSG:3979 is rotated ~11 deg from local north
    # here), except roadside tops pushed out of the clearance (straight road: d = 1.5 + 0.25).
    inside = np.isfinite(canopy(trees.xz))
    assert (inside | np.isclose(trees.distance, 1.75)).all()
    assert inside.mean() > 0.8
    assert 8 <= len(trees) <= 120
    assert any("Open Government Licence - Canada" in a for a in trees.attributions)


def _fixture_class(lon: float, lat: float) -> int:
    with rasterio.open(LANDCOVER_TIF) as ds:
        xs, ys = warp_transform("EPSG:4326", ds.crs, [lon], [lat])
        r, c = ds.index(xs[0], ys[0])
        return int(ds.read(1)[r, c])


def test_sample_landcover_reads_nearest_pixel(tmp_path):
    rng = np.random.default_rng(0)
    lon = SNAKE_HOUSE[0] + rng.uniform(-0.004, 0.004, 40)  # +-325 m (the cut is +-720 m)
    lat = SNAKE_HOUSE[1] + rng.uniform(-0.003, 0.003, 40)
    got = sample_landcover(lon, lat, cache_dir=tmp_path, url=str(LANDCOVER_TIF))
    want = [_fixture_class(a, b) for a, b in zip(lon, lat, strict=True)]
    assert got.tolist() == want
    assert set(got.tolist()) <= {1, 5, 6, 8, 15, 16, 17, 18}
    assert sample_landcover([-79.5], [43.31], cache_dir=tmp_path, url=str(LANDCOVER_TIF))[0] == 0


def test_landcover_sampler_in_and_outside_canada(tmp_path):
    proj = LocalProjection(SNAKE_HOUSE[1], SNAKE_HOUSE[0])
    lc = LandcoverSampler(proj, cache_dir=tmp_path, url=str(LANDCOVER_TIF))
    got = lc(np.array([[0.0, 0.0], [200.0, -150.0]]))
    assert got.tolist() == [
        _fixture_class(*SNAKE_HOUSE),
        _fixture_class(*proj.to_lonlat(200.0, 150.0)),
    ]
    assert lc.attributions == [LANDCOVER_ATTRIBUTION]
    # Aurora, IL: never opens the (bogus) URL.
    far = LandcoverSampler(LocalProjection(41.72, -88.29), url="/nonexistent/landcover.tif")
    assert far(np.zeros((3, 2))).tolist() == [0, 0, 0]
    assert far.attributions == []


@pytest.mark.network
def test_live_landcover_matches_the_fixture(tmp_path):
    got = sample_landcover([SNAKE_HOUSE[0]], [SNAKE_HOUSE[1]], cache_dir=tmp_path)
    assert got[0] == _fixture_class(*SNAKE_HOUSE)


# ---------------------------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------------------------


def _ground(xz: np.ndarray) -> np.ndarray:
    return 0.05 * xz[:, 0] + 0.1 * xz[:, 1]  # a tilted plane


def _mesh_trees() -> TreeSet:
    t = place_trees(
        _road(), _synthetic_chm(far_forest=True), seed=1, options=VegetationOptions(far_m=260.0)
    )
    assert (t.band == 0).any() and (t.band == 1).any()
    return t


def test_tree_meshes_are_valid_chunked_cards():
    trees = _mesh_trees()
    opts = TreeMeshOptions(atlas_size=256)
    meshes = tree_meshes(trees, _ground, options=opts)
    assert sum(m.tree_count for m in meshes) == len(trees)
    names = [m.name for m in meshes]
    assert len(set(names)) == len(names)
    assert all(re.fullmatch(r"trees_(near|far)_\d{3}(_\d+)?", n) for n in names)
    near_cells = {n.split("_")[2] for n in names if n.startswith("trees_near")}
    assert len(near_cells) >= 3
    for m in meshes:
        d = m.data
        assert d.vertex_count <= MAX_MESH_VERTICES
        assert m.tree_count <= opts.max_trees_per_mesh
        assert d.vertex_count in range(8 * m.tree_count, 12 * m.tree_count + 1, 4)
        assert d.indices.size == d.vertex_count // 4 * 12  # 2 front + 2 back triangles per card
        assert np.isfinite(d.positions).all()
        assert np.allclose(np.linalg.norm(d.normals, axis=1), 1.0, atol=0.01)  # 1/256 grid
        assert (d.normals[:, 1] > 0).all()  # blended towards +Y
        assert np.array_equal(d.tangents, np.tile([1.0, 0.0, 0.0], (d.vertex_count, 1)))
        assert np.array_equal(d.positions, np.round(d.positions * 64) / 64)  # 1/64 m grid
        if m.band == "near":
            assert m.flags == {"lod_out": 700.0, "cast_shadows": True}
        else:
            assert m.flags == {"lod_out": 1500.0, "cast_shadows": False}
        # Every card's UVs stay inside one atlas tile.
        uv = d.uvs.reshape(-1, 4, 2)
        rects = np.array([tile_uv(i, 256) for i in range(len(TREE_TILES))])
        lo = uv.min(axis=1)
        hi = uv.max(axis=1)
        card = np.concatenate([lo, hi], axis=1)  # (cards, 4) = u0, v0, u1, v1
        match = np.isclose(card[:, None, :], rects[None, :, :]).all(axis=2)
        assert match.any(axis=1).all()


def _one_cell(trees: TreeSet, n: int) -> TreeSet:
    """The first `n` near trees with X > 0 and Z > 0 (one batching cell when cells are big)."""
    sel = np.flatnonzero((trees.band == 0) & (trees.xz[:, 0] > 0) & (trees.xz[:, 1] > 0))
    assert len(sel) >= n
    return trees.subset(sel[:n])


def test_cards_stand_on_the_terrain_with_the_tree_height():
    trees = _one_cell(_mesh_trees(), 40)
    opts = TreeMeshOptions(cell_m=10_000.0)
    (mesh,) = tree_meshes(trees, _ground, options=opts)
    pos = mesh.data.positions.reshape(-1, 4, 3)
    base = _ground(trees.xz)
    k = np.where((trees.kind == DECIDUOUS) & (trees.band == 0) & (trees.height >= 12.0), 3, 2)
    assert len(pos) == k.sum()
    # Cards cross at the trunk: each card's bottom midpoint is a tree's trunk.
    mid = 0.5 * (pos[:, 0, [0, 2]] + pos[:, 1, [0, 2]])
    d, owner = cKDTree(trees.xz).query(mid)
    assert d.max() < 0.02  # positions sit on a 1/64 m grid
    assert np.array_equal(np.bincount(owner, minlength=len(trees)), k)
    assert np.allclose(pos[:, 2, 1], base[owner] + trees.height[owner], atol=0.01)  # top
    assert np.allclose(pos[:, 3, 1], base[owner] + trees.height[owner], atol=0.01)
    corner_ground = _ground(pos[:, :2][:, :, [0, 2]].reshape(-1, 2)).reshape(-1, 2)
    assert (pos[:, :2, 1] <= base[owner, None] - 0.3 + 1e-9).all()  # never floats at the trunk
    assert (pos[:, :2, 1] <= corner_ground - 0.3 + 0.01).all()  # nor at its corners


def test_tree_meshes_split_big_cells_and_drop_trees_without_ground():
    trees = _one_cell(_mesh_trees(), 25)
    opts = TreeMeshOptions(cell_m=10_000.0, max_trees_per_mesh=10)

    def ground(xz):
        g = _ground(xz)
        g[xz[:, 0] < np.sort(trees.xz[:, 0])[2]] = np.nan  # some trunks outside the terrain
        return g

    meshes = tree_meshes(trees, ground, options=opts)
    kept = int(np.isfinite(ground(trees.xz)).sum())
    assert sum(m.tree_count for m in meshes) == kept
    assert all(m.tree_count <= 10 for m in meshes)
    names = [m.name for m in meshes]
    assert names == ["trees_near_000_0", "trees_near_000_1", "trees_near_000_2"]
    assert tree_meshes(TreeSet.empty(), _ground) == []


def test_tree_scene_adds_one_alpha_tested_material_and_round_trips(tmp_path):
    trees = _mesh_trees()

    def no_texture(name):
        raise AssertionError(f"tree_scene must not use the scene's ksPerPixel factory: {name}")

    scene = Scene("dm_test", no_texture)
    stats = tree_scene(scene, trees, _ground, options=TreeMeshOptions(atlas_size=256))
    assert stats.trees == len(trees) and stats.near + stats.far == stats.trees
    assert stats.meshes == len(scene.nodes)
    (mat,) = scene.materials
    assert mat.shader == "ksTree" and mat.alpha_tested
    assert mat.blend_mode == BlendMode.OPAQUE and mat.depth_mode == DepthMode.NORMAL
    props = {p.name: p.a for p in mat.properties}
    assert props == {
        "ksAmbient": 0.4, "ksDiffuse": 0.25, "ksSpecular": 0.0, "ksSpecularEXP": 1.0,
        "ksEmissive": 0.0, "ksAlphaRef": 0.5,
    }
    assert mat.textures[0].slot_name == "txDiffuse"
    tex = scene.textures[mat.textures[0].texture_name]
    info = read_dds_info(tex.data)
    assert info.format == "BC3" and info.width == 256 and info.mip_count == 9
    assert stats.texture_bytes == len(tex.data)
    # A second call reuses the texture and material.
    tree_scene(scene, trees.subset(slice(0, 5)), _ground, options=TreeMeshOptions(atlas_size=256))
    assert len(scene.materials) == 1 and len(scene.textures) == 1
    back = read_kn5(kn5_to_bytes(scene.kn5()))
    meshes = [n for n in back.iter_nodes() if isinstance(n, Mesh)]
    assert len(meshes) == len(scene.nodes)
    assert all(not m.name[0].isdigit() for m in meshes)  # visual only, no collision
    assert back.materials[0].alpha_tested and back.materials[0].shader == "ksTree"
    assert tree_scene(Scene("x", no_texture), TreeSet.empty(), _ground).meshes == 0


def test_tree_material_defaults():
    m = tree_material()
    assert m.name == "dm_trees" and m.textures[0].texture_name == "dm_trees.dds"


# ---------------------------------------------------------------------------------------------
# Atlas
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def atlas256():
    return tree_atlas(256, seed=7)


def test_atlas_is_deterministic(atlas256):
    again = tree_atlas(256, seed=7)
    assert again.tobytes() == atlas256.tobytes()
    assert tree_atlas(256, seed=8).tobytes() != atlas256.tobytes()
    with pytest.raises(ValueError):
        tree_atlas(300)


def test_atlas_tiles_have_real_silhouettes(atlas256):
    a = np.asarray(atlas256)
    assert atlas256.mode == "RGBA" and atlas256.size == (256, 256)
    for i, tile in enumerate(TREE_TILES):
        x0, y0, x1, y1 = tile_pixel_box(i, 256)
        t = a[y0:y1, x0:x1]
        alpha = t[..., 3]
        cov = alpha_coverage(alpha)
        assert 0.15 <= cov <= 0.6, (tile.name, cov)
        # Transparent margins on the left, right and top (no bleeding between tiles) ...
        assert alpha[:, 0].max() == 0 and alpha[:, -1].max() == 0 and alpha[0].max() == 0
        # ... and the tree stands on the bottom edge (trunk or foliage reaches the ground).
        assert alpha[-1].max() >= 128, tile.name
        cols = np.flatnonzero((alpha >= 128).any(axis=0))
        span = (cols[-1] - cols[0] + 1) / alpha.shape[1]
        assert span == pytest.approx(tile.fill, abs=0.12), tile.name
        # Foliage is green: G is the largest channel on average among opaque pixels.
        rgb = t[alpha >= 128][:, :3].astype(float)
        assert rgb[:, 1].mean() > rgb[:, 0].mean() and rgb[:, 1].mean() > rgb[:, 2].mean()
    # Transparent texels carry foliage colour, not black (no dark halos when filtered).
    clear = a[..., 3] == 0
    assert a[clear][:, 1].mean() > 40


def test_alpha_tested_mips_keep_coverage(atlas256):
    rects = tuple(t.rect for t in TREE_TILES)
    levels = alpha_tested_mips(atlas256, 0.5, rects)
    plain = mip_chain(atlas256)
    assert [lv.shape for lv in levels] == [lv.shape for lv in plain]
    assert np.array_equal(levels[0], plain[0])
    ours, box = [], []
    for i in range(len(TREE_TILES)):
        x0, y0, x1, y1 = rects[i]

        def cov(lv, x0=x0, y0=y0, x1=x1, y1=y1):
            h, w = lv.shape[:2]
            sub = lv[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w), 3]
            return alpha_coverage(sub)

        top = cov(levels[0])
        for k in range(1, 5):
            assert cov(levels[k]) == pytest.approx(top, abs=0.05), (TREE_TILES[i].name, k)
        ours.append(abs(cov(levels[4]) - top))
        box.append(abs(cov(plain[4]) - top))
    assert sum(ours) < 0.5 * sum(box)  # plain box-filtered mips thin the foliage out


def test_alpha_tested_dds_and_texture_names(atlas256):
    data = encode_alpha_tested_dds(atlas256, 0.5, tuple(t.rect for t in TREE_TILES))
    info = read_dds_info(data)
    assert info.format == "BC3" and info.mip_count == 9 and len(info.levels) == 9
    top = np.asarray(decode_dds_level(data, 0))
    src = np.asarray(atlas256)
    assert np.mean((top[..., 3] >= 128) == (src[..., 3] >= 128)) > 0.99
    png = tree_atlas_texture("trees.png", size=64, seed=1)
    assert png.data[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError):
        tree_atlas_texture("trees.tga", size=64)


def test_tile_layout():
    for i, tile in enumerate(TREE_TILES):
        u0, v0, u1, v1 = tile_uv(i, 1024)
        x0, y0, x1, y1 = tile.rect
        assert x0 < u0 < u1 < x1 and y0 < v0 < v1 < y1
        assert u0 - x0 == pytest.approx(0.5 / 1024)
        assert 0.5 < tile.fill <= 1.0 and 0.0 <= tile.crown_base < 0.5
    boxes = [tile_pixel_box(i, 1024) for i in range(len(TREE_TILES))]
    area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
    assert area == 1024 * 1024  # the tiles fill the atlas without overlap
    assert {t.kind for t in TREE_TILES} == {DECIDUOUS, CONIFER, BUSH}
    assert math.isclose(TREE_TILES[0].aspect, 0.5)
