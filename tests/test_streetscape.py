"""The urban cross-section (`geometry.streetscape`): urban or rural, sidewalk sides, the
section's profile on the route ribbon and round the network's curbed side roads, curb cuts,
and the municipal sidewalk fetcher (`geo.sidewalks`)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely

from driftmaps.geo.building_sources import SourceError
from driftmaps.geo.sidewalks import OAKVILLE_SIDEWALKS, fetch_sidewalks
from driftmaps.geometry.network import NetWay, NetworkOptions, build_network
from driftmaps.geometry.road import RoadModel, build_road_meshes
from driftmaps.geometry.streetscape import (
    CURB_FACE_RUN_M,
    CURB_REVEAL_M,
    DEPRESSED_CURB_M,
    GUTTER_M,
    RouteSection,
    SidewalkIndex,
    UrbanContext,
    section_lift,
    section_offsets,
    section_width,
    sidewalk_sides,
    smooth_runs,
    tag_context,
    walk_sides,
)


def test_urban_or_rural_from_tags():
    assert tag_context({"highway": "residential", "sidewalk": "both"}) == "urban"
    assert tag_context({"highway": "secondary", "lit": "yes", "maxspeed": "60"}) == "urban"
    assert tag_context({"highway": "primary", "lanes": "4", "maxspeed": "70"}) == "urban"
    assert tag_context({"highway": "secondary", "oneway": "yes", "lanes": "2",
                        "maxspeed": "60"}) == "urban"  # fmt: skip
    # Rural Ontario: Appleby Line (70 km/h, unlit, no sidewalk), a 2-lane concession road.
    assert tag_context({"highway": "primary", "lanes": "2", "maxspeed": "70", "lit": "no",
                        "sidewalk": "no"}) == "rural"  # fmt: skip
    assert tag_context({"highway": "tertiary", "maxspeed": "80", "lanes": "4"}) == ""
    assert tag_context({"highway": "tertiary", "maxspeed": "50"}) == ""  # the context decides
    assert sidewalk_sides({"sidewalk": "left"}) == (True, False)
    assert sidewalk_sides({"sidewalk:both": "separate"}) == (True, True)
    assert sidewalk_sides({"sidewalk": "both", "sidewalk:right": "no"}) == (True, False)
    assert sidewalk_sides({}) == (None, None)


def _grid_streets(n: int = 8, step: float = 80.0) -> list:
    out = []
    for k in range(n):
        x = (k - n / 2) * step
        out.append((np.array([[x, -300.0], [x, 300.0]]), {"highway": "residential"}))
    return out


def test_subdivision_context():
    ctx = UrbanContext(_grid_streets())
    xz = np.zeros((3, 2))
    tags = [{"highway": "tertiary", "maxspeed": "50"}] * 3
    assert ctx.urban(xz, tags).all()  # a street grid around: a subdivision
    far = xz + 5000.0
    assert not ctx.urban(far, tags).any()
    fast = [{"highway": "tertiary", "maxspeed": "80"}] * 3
    assert not ctx.urban(xz, fast).any()  # 80 km/h through it: still rural
    # A municipal sidewalk line beside the road makes it urban too.
    walk = UrbanContext((), [np.array([[-50.0, 6.0], [50.0, 6.0]])])
    assert walk.urban(far * 0, tags).all()


def test_section_profile():
    offs, lifts = section_offsets(1.5)
    assert offs[1] == GUTTER_M and lifts[1] == 0.0  # the gutter is at road level
    assert offs[2] - offs[1] == pytest.approx(CURB_FACE_RUN_M)  # a sloped curb face ...
    assert lifts[2] == pytest.approx(CURB_REVEAL_M)  # ... 150 mm high
    assert 1.5 < offs[5] - offs[4] + 0.01 and lifts[5] > lifts[4] > CURB_REVEAL_M
    assert float(section_width(1.5)) == pytest.approx(offs[-1])
    assert section_lift(np.array([0.1, 0.39, 1.0]), 1.5) == pytest.approx(
        [0.0, CURB_REVEAL_M / 2, CURB_REVEAL_M + (1.0 - offs[3]) * 0.02], abs=0.01
    )


def test_walk_sides_precedence():
    n = 30
    s = np.arange(n, dtype=float)
    urban = np.ones(n, bool)
    collector = [{"highway": "tertiary"}] * n
    left, right = walk_sides(collector, np.ones(n, bool), urban, None, s)
    assert left.all() and right.all()  # rule: urban collectors on both sides
    local = [{"highway": "residential"}] * n
    left, right = walk_sides(local, np.ones(n, bool), urban, None, s)
    assert not left.any() and not right.any()  # local streets: per OSM / town data only
    municipal = (np.ones(n, bool), np.zeros(n, bool))
    left, right = walk_sides(local, np.ones(n, bool), urban, municipal, s)
    assert left.all() and not right.any()
    tagged = [{"highway": "residential", "sidewalk": "right"}] * n
    left, right = walk_sides(tagged, np.ones(n, bool), urban, municipal, s)
    assert not left.any() and right.all()  # OSM wins
    left, right = walk_sides(tagged, np.zeros(n, bool), urban, None, s)
    assert left.all() and not right.any()  # driven against the way: its right is our left
    assert not walk_sides(collector, np.ones(n, bool), ~urban, None, s)[0].any()


def test_sidewalk_index_sides():
    idx = SidewalkIndex([np.array([[0.0, 7.0], [100.0, 7.0]])])  # 7 m from the centreline
    xz = np.stack([np.arange(10.0, 90.0, 10.0), np.zeros(8)], axis=1)
    left = np.tile([0.0, 1.0], (8, 1))  # left = +z here
    lt, rt = idx.sides(xz, left, np.full(8, 3.5))
    assert lt.all() and not rt.any()


def _straight(n: int = 120, hw: float = 4.0, shoulder: float = 1.0) -> RoadModel:
    s = np.arange(n, dtype=float)
    centre = np.stack([s, np.zeros(n), np.zeros(n)], axis=1)
    return RoadModel(s=s, centre=centre, half_width=hw, shoulder=shoulder, centre_line=False,
                     bridge=False)  # fmt: skip


def test_route_urban_section_meshes():
    n = 120
    urban = np.zeros(n, bool)
    urban[40:] = True
    walk_w = np.full(n, 1.5)
    model = _straight(n)
    model.shoulder = np.where(urban, section_width(walk_w), model.shoulder)
    sec = RouteSection(urban, urban, np.zeros(n, bool), walk_w)

    class Keep:  # the network's masks: a driveway curb cut at x = 80..86 on the left
        cut_left = np.zeros(n - 1, bool)
        cut_left[80:86] = True

    meshes = build_road_meshes(model, chunk_m=50.0, section=sec, keep=Keep())
    for key in ("curb", "boulevard", "sidewalk", "apron"):
        assert any(m is not None for m in meshes.urban[key]), key
    curb = np.concatenate([m.positions for m in meshes.urban["curb"] if m is not None])
    # The curb rises 150 mm over its sloped face, and only on the urban stretch.
    assert curb[:, 0].min() >= 39.0 and curb[:, 1].max() == pytest.approx(CURB_REVEAL_M)
    in_cut = (curb[:, 0] > 81.5) & (curb[:, 0] < 84.5) & (curb[:, 2] < 0)  # left = -z here
    assert in_cut.any() and curb[in_cut, 1].max() == pytest.approx(DEPRESSED_CURB_M)
    walk = np.concatenate([m.positions for m in meshes.urban["sidewalk"] if m is not None])
    assert (walk[:, 2] < 0).all()  # a sidewalk on the left (-z) only
    # No gravel on the urban stretch; the terrain meets the lifted outer edge.
    gravel = np.concatenate([m.positions for m in meshes.gravel if m is not None])
    assert gravel[:, 0].max() <= 41.0
    assert meshes.left_outer[60, 1] == pytest.approx(section_offsets(1.5)[1][-1])
    assert meshes.left_outer[10, 1] == pytest.approx(0.0)


def test_network_side_road_gets_curbs_and_sidewalks():
    """An urban residential street (sidewalks both sides) leaves an urban route at x = 100;
    the curb wraps round both corners (curb returns) and the sidewalk follows."""
    model = _straight(201, hw=4.0)
    urban = np.ones(len(model), bool)
    model.shoulder = np.full(len(model), float(section_width(1.5)))
    ways = [
        NetWay(
            1,
            tuple(range(1, 22)),
            np.stack([np.linspace(0, 200, 21), np.zeros(21)], 1),
            {"highway": "tertiary"},
        ),  # fmt: skip
        NetWay(
            2,
            (11, 100, 101),
            np.array([[100.0, 0.0], [100.0, -60.0], [100.0, -120.0]]),
            {"highway": "residential", "sidewalk": "both"},
        ),  # fmt: skip
    ]
    segs = {(a, a + 1) for a in range(1, 21)}

    class Ctx:
        def urban(self, xz, tags, hw=4.0, s=None):
            return np.ones(len(np.asarray(xz).reshape(-1, 2)), bool)

    net = build_network(
        model, ways, route_nodes=list(range(1, 22)), route_segments=segs,
        route_highway=["tertiary"] * len(model), dtm=lambda xz: np.zeros(len(xz)),
        options=NetworkOptions(radius_m=120.0), urban=Ctx(), route_urban=urban,
    )  # fmt: skip
    side = net.side_roads[0]
    assert side.shoulder == 0.0 and side.urban.all()
    assert {"gutter", "curb", "boulevard", "walk"} <= set(net.urban_bands)
    walk = net.urban_bands["walk"]
    assert walk.area > 2 * 1.4 * 80.0  # both sides of the street
    # Curb returns: the curb top reaches round the corner towards the route's edge.
    top = net.urban_bands["curb"]
    corner = shapely.Point(100.0 + 4.0 + 9.0, -4.0 - 9.0).buffer(9.0)
    assert top.intersects(corner)
    # The section is part of the paved (drivable, walled) area; its meshes come separately.
    assert net.paved.contains(walk.representative_point())
    names = {p.name for p in net.urban_pieces}
    assert any(n.startswith("1KERB_net") for n in names)
    assert any(n.startswith("1ROAD_sw_net") for n in names)
    ys = np.concatenate([p.mesh.positions[:, 1] for p in net.urban_pieces])
    assert ys.max() == pytest.approx(section_offsets(1.5)[1][-1], abs=0.01)
    assert not any(math.isnan(v) for v in ys)


def test_oakville_sidewalks_fetch_filters_and_caches(tmp_path):
    calls = []

    def fake(url, params):
        calls.append(params)
        return {"features": [
            {"attributes": {"OBJECTID": 1, "ASSET_TYPE": "Sidewalk"},
             "geometry": {"paths": [[[-79.762, 43.4317], [-79.7623, 43.4319]]]}},
            {"attributes": {"OBJECTID": 2, "ASSET_TYPE": "Road Crossing"},
             "geometry": {"paths": [[[-79.762, 43.4317], [-79.7621, 43.4318]]]}},
        ]}  # fmt: skip

    box = (-79.7625, 43.4310, -79.7600, 43.4330)
    lines = fetch_sidewalks(OAKVILLE_SIDEWALKS, [box], cache_dir=tmp_path, get_json=fake)
    assert len(lines) == 1 and lines[0].shape == (2, 2)  # the road crossing is left out
    again = fetch_sidewalks(OAKVILLE_SIDEWALKS, [box], cache_dir=tmp_path, offline=True)
    assert len(calls) == 1 and np.allclose(again[0], lines[0])
    outside = (-80.5, 43.0, -80.4, 43.1)  # outside Oakville: never asked
    assert fetch_sidewalks(OAKVILLE_SIDEWALKS, [outside], cache_dir=tmp_path, get_json=fake) == []
    with pytest.raises(SourceError):
        fetch_sidewalks(OAKVILLE_SIDEWALKS, [(-79.70, 43.45, -79.69, 43.46)],
                        cache_dir=tmp_path, offline=True)  # fmt: skip
    assert "Open Government Licence - Town of Oakville" in OAKVILLE_SIDEWALKS.attribution


def test_sidewalks_and_urban_stretches_are_continuous():
    # Stations every 10 m over 1 km: gaps under 150 m between sidewalk runs are filled (the
    # user's Oakville check: "some sidewalks breaking/stopping"), unless OSM says there is
    # none there; stretches under 150 m that reach neither end are dropped; the ends stay.
    s = np.arange(0.0, 2000.0, 10.0)
    m = np.zeros(len(s), bool)
    m[0:30] = True  # 0-290 m, from the start
    m[40:70] = True  # a 100 m gap (300-390 m) before it: filled
    m[100:105] = True  # 50 m alone between long gaps: dropped
    m[197:] = True  # the last 30 m: reaches the end, kept
    out = smooth_runs(m, s, 150.0)
    assert out[:70].all() and not out[70:197].any() and out[197:].all()
    no = np.zeros(len(s), bool)
    no[33] = True  # sidewalk=no at 330 m: the gap stays
    held = smooth_runs(m, s, 150.0, fixed=no)
    assert not held[30:40].any() and held[40:70].all()
    # walk_sides: municipal lines found in pieces along an urban arterial -> one sidewalk.
    n = len(s)
    tags = [{"highway": "secondary"}] * n
    right = np.zeros(n, bool)
    for a in range(0, n, 12):
        right[a : a + 8] = True  # 80 m found, 40 m missed (a wide boulevard, a driveway)
    left, rr = walk_sides(tags, np.ones(n, bool), np.ones(n, bool), (np.zeros(n, bool), right), s)
    assert rr.all() and not left.any()
    # A short rural gap in a town road is filled; a short urban blip on a rural road dropped.
    urban = np.ones(n, bool)
    urban[50:58] = False
    assert smooth_runs(urban, s).all()
    assert not smooth_runs(~urban, s).any()


def test_far_probe_finds_a_sidewalk_behind_a_wide_boulevard_but_not_across_a_median():
    # A road along +X (half width 5 m), a municipal sidewalk 11 m beyond each edge.
    xs = np.arange(0.0, 200.0, 5.0)
    idx = SidewalkIndex([np.array([[0.0, 16.0], [200.0, 16.0]]),
                         np.array([[0.0, -16.0], [200.0, -16.0]])])  # fmt: skip
    xz = np.stack([xs, 0 * xs], 1)
    left_dir = np.tile([0.0, -1.0], (len(xs), 1))  # +X travel: left is -Z (north)
    hw = np.full(len(xs), 5.0)
    left, right = idx.sides(xz, left_dir, hw)
    assert left.all() and right.all()
    left, right = idx.sides(xz, left_dir, hw, far=(False, True))  # left is the median
    assert not left.any() and right.all()
