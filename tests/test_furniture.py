"""Street furniture (`geometry.furniture`, `ac.furniture_textures`): which items a route gets
from Mapillary, OSM and the Ontario rules, de-duplication, pole lines and wires, the meshes and
the markings, and keeping items off the drivable surface (`geometry.clearance`: side roads,
roundabout aprons, signs in fields). Synthetic roads in AC coordinates (X east, Z south)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import shapely

from driftmaps.ac.furniture_textures import (
    ATLAS_SIZE,
    furniture_atlas,
    furniture_atlas_texture,
    sign_tile,
)
from driftmaps.geo.mapillary import MapFeature
from driftmaps.geo.project import LocalProjection
from driftmaps.geo.street_features import OsmLine, OsmNode, StreetFeatures
from driftmaps.geometry import furniture as fu
from driftmaps.geometry.clearance import CLEARANCE_M, DrivableSurface, clear_items
from driftmaps.geometry.primitives import merge
from driftmaps.geometry.road import RoadModel
from driftmaps.geometry.roundabouts import Roundabout
from driftmaps.geometry.street_items import (
    HYDRANT,
    MAILBOX,
    POLE_UTILITY,
    SIGN_ALL_WAY,
    SIGN_CHEVRON_LEFT,
    SIGN_CHEVRON_RIGHT,
    SIGN_SPEED_LIMIT,
    SIGN_STOP,
    SIGN_YIELD,
    StreetItem,
)

PROJ = LocalProjection(43.44, -79.82)


@dataclass
class Edge:
    way: int
    u: int
    v: int
    dir: int = 1


class FakeRoute:
    """The parts of `geo.router.Route` the furniture layer reads."""

    def __init__(self, nodes_xz, edges, ways, stubs=()):
        self.nodes = {k: tuple(_lonlat(xz)) for k, xz in nodes_xz.items()}
        self.edges = tuple(edges)
        self.ways = ways
        self.stubs = tuple(stubs)

    def node_path(self):
        return [self.edges[0].u] + [e.v for e in self.edges]


def _lonlat(xz) -> np.ndarray:
    x, z = xz
    return PROJ.from_xy(np.array([x, -z]))


def _model(centre_xz: np.ndarray) -> RoadModel:
    c = np.column_stack([centre_xz[:, 0], np.zeros(len(centre_xz)), centre_xz[:, 1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    return RoadModel(s, c, 3.5, 1.0, True, False)


def ground(xz):
    return np.zeros(len(np.asarray(xz).reshape(-1, 2)))


STRAIGHT = _model(np.column_stack([np.arange(0.0, 1001.0), np.zeros(1001)]))
# A tertiary road along +X crossing a primary at x = 500; the limit goes 60 -> 80 there.
CROSSING = FakeRoute(
    {1: (0, 0), 2: (500, 0), 3: (1000, 0), 4: (500, -100), 5: (500, 100)},
    [Edge(11, 1, 2), Edge(12, 2, 3)],
    {11: {"highway": "tertiary", "maxspeed": "60"}, 12: {"highway": "tertiary", "maxspeed": "80"},
     13: {"highway": "primary"}},
    [(2, 13, 4), (2, 13, 5)],
)  # fmt: skip
WAY_ID = np.where(STRAIGHT.s < 500.0, 11, 12)


def _plan(**kw):
    args = dict(s_range=(0.0, 1000.0), way_id=WAY_ID)
    return fu.plan_furniture(STRAIGHT, PROJ, CROSSING, ground, **(args | kw))


def _of(plan, kind, source=None):
    return [it for it in plan.items if it.kind == kind and (source is None or it.source == source)]


def _feature(fid, value, x, z, heading=None, last="2019-06-01T00:00:00Z"):
    lon, lat = _lonlat((x, z))
    return MapFeature(fid, value, float(lon), float(lat), heading, last, last)


# --------------------------------------------------------------------------------------------
# Junction control on the route
# --------------------------------------------------------------------------------------------


def test_the_route_stops_where_it_meets_a_bigger_road():
    plan = _plan()
    stops = sorted(_of(plan, SIGN_STOP, "rule"), key=lambda it: it.position[0])
    assert len(stops) == 2
    ours, theirs = stops
    # Ours: before the crossing, right of the road (+Z), facing back at us (-X).
    assert ours.position[0] == pytest.approx(495.0, abs=0.5)  # 3.5 m half road + 1.5 m
    assert ours.position[2] == pytest.approx(3.5 + 2.0 + 0.375, abs=0.05)
    assert ours.facing == pytest.approx((-1.0, 0.0))
    # Theirs (driving B -> A): after the crossing as we see it, on our left, facing +X.
    assert theirs.position[0] == pytest.approx(505.0, abs=0.5) and theirs.position[2] < -5.0
    assert theirs.facing == pytest.approx((1.0, 0.0))
    bars = [mk for mk in plan.markings if mk.kind == "stop_bar"]
    assert sorted((round(b.s), b.l0 < 0) for b in bars) == [(495, True), (505, False)]
    ctl = {(c["forward"], c["control"]) for c in plan.controls}
    assert ctl == {(True, "stop"), (False, "stop")}


def test_no_stop_on_the_bigger_road_and_no_invented_all_way():
    route = FakeRoute(
        {1: (0, 0), 2: (500, 0), 3: (1000, 0), 4: (500, -100)},
        [Edge(11, 1, 2), Edge(12, 2, 3)],
        {11: {"highway": "primary"}, 12: {"highway": "primary"}, 13: {"highway": "tertiary"}},
        [(2, 13, 4)],
    )
    plan = fu.plan_furniture(STRAIGHT, PROJ, route, ground, s_range=(0.0, 1000.0))
    assert _of(plan, SIGN_STOP) == [] and not plan.markings
    # Equal classes: no control is inferred (all-way stops come only from data).
    same = FakeRoute(route.nodes and {1: (0, 0), 2: (500, 0), 3: (1000, 0), 4: (500, -100)},
                     route.edges, {11: {"highway": "tertiary"}, 12: {"highway": "tertiary"},
                                   13: {"highway": "tertiary"}}, route.stubs)  # fmt: skip
    plan = fu.plan_furniture(STRAIGHT, PROJ, same, ground, s_range=(0.0, 1000.0))
    assert _of(plan, SIGN_STOP) == [] and _of(plan, SIGN_ALL_WAY) == []


def test_osm_all_way_stop_adds_the_tab():
    lon, lat = _lonlat((500, 0))
    osm = StreetFeatures((OsmNode(2, float(lon), float(lat), {"highway": "stop", "stop": "all"}),))
    plan = _plan(osm=osm)
    assert len(_of(plan, SIGN_ALL_WAY)) == 2 and len(_of(plan, SIGN_STOP)) == 2
    tabs = {(round(t.position[0]), round(t.position[2], 1)) for t in _of(plan, SIGN_ALL_WAY)}
    stops = {(round(t.position[0]), round(t.position[2], 1)) for t in _of(plan, SIGN_STOP)}
    assert tabs == stops  # on the stop signs' posts


def test_a_photographed_stop_sign_replaces_the_rule_one():
    # Mapillary put it on the road with its facing flipped: it stands before the junction, so
    # it is our approach's sign, moved to our stop line.
    sign = _feature("9", "regulatory--stop--g1", 470.0, 1.0, heading=90.0)
    plan = _plan(mapillary=[sign])
    mly = _of(plan, SIGN_STOP, "mapillary")
    assert len(mly) == 1 and mly[0].ref == "mly:9"
    assert mly[0].position[0] == pytest.approx(495.0, abs=0.5) and mly[0].position[2] > 5.0
    assert mly[0].facing == pytest.approx((-1.0, 0.0))
    assert len(_of(plan, SIGN_STOP, "rule")) == 1  # only the other approach's
    assert plan.mapillary_used == 1


# --------------------------------------------------------------------------------------------
# Speed limits, chevrons, coverage
# --------------------------------------------------------------------------------------------


def test_speed_signs_where_the_limit_changes_and_after_the_junction():
    plan = _plan()
    signs = {(it.kind, round(it.position[0]), it.position[2] > 0)
             for it in plan.items if it.kind.startswith(SIGN_SPEED_LIMIT)}  # fmt: skip
    assert (f"{SIGN_SPEED_LIMIT}:80", 500, True) in signs  # ours, right
    assert (f"{SIGN_SPEED_LIMIT}:60", 500, False) in signs  # theirs, left
    assert (f"{SIGN_SPEED_LIMIT}:80", 600, True) in signs  # confirmation after the junction
    assert (f"{SIGN_SPEED_LIMIT}:60", 400, False) in signs


def test_rules_stay_out_of_stretches_mapillary_photographed():
    photos = np.array([_lonlat((x, 0.0)) for x in np.arange(0.0, 1001.0, 20.0)])
    plan = _plan(coverage=photos)
    assert plan.covered_fraction > 0.95
    assert not [it for it in plan.items if it.kind.startswith(SIGN_SPEED_LIMIT)]
    assert len(_of(plan, SIGN_STOP)) == 2  # junction control is always applied


def _curve_model(radius=60.0, turn_deg=90.0, left=True):
    straight = np.column_stack([np.arange(-200.0, 0.0, 1.0), np.zeros(200)])
    a = np.radians(np.arange(0.0, turn_deg, 180.0 / (math.pi * radius)))
    sgn = -1.0 if left else 1.0  # left of +X is -Z
    arc = np.column_stack([radius * np.sin(a), sgn * radius * (1 - np.cos(a))])
    end, heading = arc[-1], a[-1]
    d = np.array([math.cos(heading), sgn * math.sin(heading)])
    out = end + np.arange(1.0, 200.0)[:, None] * d
    return _model(np.concatenate([straight, arc, out]))


def test_chevrons_on_the_outside_of_a_sharp_curve_back_to_back():
    m = _curve_model(left=True)
    route = FakeRoute({1: (-200, 0), 2: (0, -400)}, [Edge(11, 1, 2)],
                      {11: {"highway": "secondary", "maxspeed": "80"}})  # fmt: skip
    plan = fu.plan_furniture(m, PROJ, route, ground, way_id=np.full(len(m), 11))
    ours = _of(plan, SIGN_CHEVRON_LEFT)
    theirs = _of(plan, SIGN_CHEVRON_RIGHT)
    assert len(ours) >= 3 and len(ours) == len(theirs)
    frame = fu._Frame(m, (m.s[0], m.s[-1]))
    for it in ours + theirs:
        st, off, _ = frame.project(np.array([[it.position[0], it.position[2]]]))
        assert off[0] < -5.0  # outside of a left turn: the right side
    gaps = np.diff(
        sorted(frame.project(np.array([[i.position[0], i.position[2]] for i in ours]))[0])
    )
    assert np.allclose(gaps, 24.0, atol=0.5)  # advisory ~41 km/h: 24 m spacing
    # A gentle curve gets none.
    gentle = _curve_model(radius=400.0, turn_deg=40.0)
    plan = fu.plan_furniture(gentle, PROJ, route, ground, way_id=np.full(len(gentle), 11))
    assert not _of(plan, SIGN_CHEVRON_LEFT)


# --------------------------------------------------------------------------------------------
# Poles, de-duplication
# --------------------------------------------------------------------------------------------


def test_pole_line_gaps_filled_and_wired():
    xs = [0, 45, 90, 135, 180, 225, 270, 405, 450, 495, 540]  # a 135 m gap after 270
    poles = [_feature(f"p{k}", "object--support--utility-pole", x, 9.0) for k, x in enumerate(xs)]
    poles.append(_feature("dup", "object--support--utility-pole", 92.0, 9.5))  # seen twice
    poles.append(_feature("road", "object--support--utility-pole", 600.0, 1.0))  # on the road
    photos = np.array([_lonlat((x, 0.0)) for x in np.arange(0.0, 1001.0, 20.0)])
    plan = _plan(mapillary=poles, coverage=photos)
    got = sorted(_of(plan, POLE_UTILITY), key=lambda it: it.position[0])
    rule = [it for it in got if it.source == "rule"]
    assert [round(it.position[0]) for it in rule] == [315, 360]
    assert all(it.position[2] == pytest.approx(9.0, abs=0.1) for it in got)  # one line
    assert len(got) == len(xs) + 2 + 1  # the duplicate merged; the road one moved into line
    assert len(plan.wires) == len(got) - 1


def test_dedupe_prefers_mapillary_and_keeps_sides_and_facings_apart():
    a = StreetItem(SIGN_STOP, (0.0, 0.0, 6.0), (-1.0, 0.0), "rule")
    b = StreetItem(SIGN_STOP, (5.0, 0.0, 6.0), (-1.0, 0.0), "mapillary")
    c = StreetItem(SIGN_STOP, (2.0, 0.0, 6.0), (1.0, 0.0), "rule")  # back to back: kept
    d = StreetItem(f"{SIGN_SPEED_LIMIT}:70", (40.0, 0.0, 6.0), (-1.0, 0.0), "rule")
    e = StreetItem(f"{SIGN_SPEED_LIMIT}:60", (45.0, 0.0, 6.0), (-1.0, 0.0), "osm")
    p = StreetItem(POLE_UTILITY, (100.0, 0.0, 5.0), (0.0, 1.0), "rule")
    q = StreetItem(POLE_UTILITY, (100.0, 0.0, -2.0), (0.0, 1.0), "mapillary")
    kept = fu.dedupe([a, b, c, d, e, p, q], sides=[1, 1, 1, 1, 1, 1, -1])
    assert kept == [b, c, e, p, q]


def test_parse_speed_and_mapillary_classes():
    assert fu.parse_speed("70") == 70 and fu.parse_speed("40 mph") == 65
    assert fu.parse_speed("signals") is None and fu.parse_speed(None) is None
    assert fu.mapillary_kind("regulatory--maximum-speed-limit-70--g3") == "sign:speed_limit:70"
    assert fu.mapillary_kind("regulatory--stop--g1") == SIGN_STOP
    assert fu.mapillary_kind("complementary--maximum-speed-limit-40--g1") is None
    assert fu.mapillary_kind("object--support--utility-pole") == POLE_UTILITY


# --------------------------------------------------------------------------------------------
# Meshes, markings, atlas
# --------------------------------------------------------------------------------------------


def test_meshes_per_cell_with_faces_toward_the_drivers():
    poles = [_feature(f"p{k}", "object--support--utility-pole", x, 7.5)
             for k, x in enumerate(range(0, 500, 45))]  # fmt: skip
    plan = _plan(mapillary=poles)
    meshes = fu.furniture_meshes(plan, STRAIGHT, verge_m=4.0)
    names = [m.name for m in meshes]
    assert any(n.startswith("furn_") for n in names) and any(n.startswith("signs_") for n in names)
    assert any(n.startswith("wires_") for n in names)
    hidden = [m for m in meshes if m.name.startswith("1WALL_furn")]
    assert hidden and all(m.material == "hidden" and not m.flags["is_renderable"] for m in hidden)
    for m in meshes:
        assert m.mesh.vertex_count <= 65535 and np.isfinite(m.mesh.positions).all()
        assert (m.mesh.uvs >= 0).all() and (m.mesh.uvs <= 1).all()
        assert m.flags.get("lod_out", 1e9) >= 300.0
    # The stop sign face nearest to (495, 5.9) faces -X and is ~1.5 m up.
    signs = merge([m.mesh for m in meshes if m.material == "signs"])
    near = np.linalg.norm(signs.positions[:, [0, 2]] - [495.0, 5.9], axis=1) < 0.6
    assert near.any()
    assert np.allclose(signs.normals[near], [-1.0, 0.0, 0.0], atol=1e-6)
    assert signs.positions[near, 1].min() == pytest.approx(1.5, abs=0.05)
    # Its UVs lie in the stop tile.
    u0, v0, u1, v1 = sign_tile("stop").uv
    uv = signs.uvs[near]
    assert (uv[:, 0] >= u0).all() and (uv[:, 0] <= u1).all()
    assert (uv[:, 1] >= v0).all() and (uv[:, 1] <= v1).all()
    # Poles are low-poly: under 150 vertices each with their crossarm and insulators.
    furn = sum(m.mesh.vertex_count for m in meshes if m.name.startswith("furn_"))
    assert furn < 150 * len(_of(plan, POLE_UTILITY)) + 200 * len(plan.items)


def test_markings_lie_on_the_road():
    plan = _plan()
    out = fu.marking_meshes(plan.markings, STRAIGHT)
    assert [n for n, _ in out] == ["marks_001", "marks_002"]  # 495 m and 505 m: two cells
    mesh = merge([m for _, m in out])
    assert np.allclose(mesh.positions[:, 1], fu.MARK_LIFT_M)
    assert np.allclose(mesh.normals, [0.0, 1.0, 0.0])
    z = mesh.positions[:, 2]
    assert z.min() == pytest.approx(-3.3, abs=0.01) and z.max() == pytest.approx(3.3, abs=0.01)
    zebra = fu.marking_meshes([fu.Marking("zebra", 100.0, -3.5, 3.5)], STRAIGHT)[0][1]
    assert zebra.triangle_count == 2 * 6  # 0.5 m bars, 0.6 m gaps across 7 m


def test_atlas_draws_the_faces():
    img = furniture_atlas(["Appleby Line"])
    assert img.size == (ATLAS_SIZE, ATLAS_SIZE)
    t = sign_tile("stop")
    r, g, b = img.getpixel((t.x + t.w // 2, t.y + t.h // 5))  # red above the legend
    assert r > 150 and g < 60 and b < 60
    tex = furniture_atlas_texture("dm_furniture.dds", ["Appleby Line"])
    assert tex.data[:4] == b"DDS " and len(tex.data) < 800_000
    with pytest.raises(ValueError):
        furniture_atlas_texture("x.jpg")


# --------------------------------------------------------------------------------------------
# Clear of the drivable surface
# --------------------------------------------------------------------------------------------

# A side road south from the route (along +Z) at x = 360: 6 m wide, 0.5 m shoulders.
SIDE_X = 360.0
SIDE_XZ = np.array([[SIDE_X, 0.0], [SIDE_X, 150.0]])
# A roundabout off the route's north side (no arms: only its ring, apron and island).
RB = Roundabout((7,), frozenset(), np.array([700.0, -60.0]), radius=15.0)


def _net(side: bool = True, roundabout: bool = False):
    line = shapely.LineString(SIDE_XZ)
    paved, foot, islands, rbs, sides = [], [], [], [], []
    if side:
        paved.append(line.buffer(3.0, cap_style="flat"))
        foot.append(line.buffer(3.5, cap_style="flat"))
        sides.append(SimpleNamespace(xz=SIDE_XZ, oneway=0))
    if roundabout:
        c = shapely.Point(RB.centre)
        ring = c.buffer(RB.outer_r, quad_segs=32).difference(c.buffer(RB.inner_r, quad_segs=32))
        disc = c.buffer(RB.inner_r, quad_segs=32)
        paved += [ring, disc]
        foot += [ring, disc]
        islands.append(disc)
        rbs.append(RB)
    return SimpleNamespace(
        side_roads=sides,
        roundabouts=rbs,
        islands=islands,
        driveways=[],
        paved=shapely.union_all(paved),
        footprint=shapely.union_all(foot),
    )


def _surface(**kw) -> DrivableSurface:
    return DrivableSurface.build(STRAIGHT, net=_net(**kw))


def _plan_on(surface, **kw):
    return _plan(surface=surface, **kw)


def _xz(it: StreetItem) -> np.ndarray:
    return np.array([it.position[0], it.position[2]])


def _assert_clear(plan, surface: DrivableSurface) -> None:
    for it in plan.items:
        d = surface.distance(_xz(it))[0]
        assert d > (0.0 if it.kind == MAILBOX else CLEARANCE_M - 0.05), (it, d)
    assert plan.clearance["on_surface"] == 0 and plan.clearance["near_surface"] == 0


def test_items_on_a_side_road_move_off_it():
    surface = _surface()
    pole = _feature("p", "object--support--utility-pole", SIDE_X + 1.0, 30.0)
    stop = _feature("s", "regulatory--stop--g1", SIDE_X - 0.5, 25.0, heading=180.0)  # wrong way
    hydrant = StreetItem(HYDRANT, (SIDE_X - 2.0, 0.0, 90.0), (1.0, 0.0), "osm")
    plan = _plan_on(surface, mapillary=[pole, stop], items=[hydrant])
    (p,) = [it for it in plan.items if it.ref == "mly:p"]
    # beyond the side road's shoulder (3.5 m) by the pole offset, on its own side
    assert p.position[0] == pytest.approx(SIDE_X + 3.5 + fu.POLE_EDGE_M, abs=0.3)
    assert p.position[2] == pytest.approx(30.0, abs=0.01)
    (s,) = [it for it in plan.items if it.ref == "mly:s"]
    # west of the side road, plate edge 2 m past the pavement: on the right of the drivers
    # coming south from the route, facing them (north)
    w = fu._plate_size(SIGN_STOP, None)[0]
    assert s.position[0] == pytest.approx(SIDE_X - 3.0 - 2.0 - w / 2, abs=0.3)
    assert s.facing == pytest.approx((0.0, -1.0), abs=1e-6)
    (h,) = [it for it in plan.items if it.kind == HYDRANT]
    assert h.position[0] < SIDE_X - 3.5 - CLEARANCE_M
    assert plan.clearance["moved:pole"] == 1 and plan.clearance["moved:sign"] == 1
    _assert_clear(plan, surface)


def test_signs_in_fields_and_stray_poles_are_dropped():
    surface = _surface(side=False)
    field_sign = _feature("f", "regulatory--yield--g1", 200.0, 40.0, heading=90.0)
    road_sign = _feature("r", "regulatory--yield--g1", 250.0, 7.0, heading=90.0)
    stray = _feature("x", "object--support--utility-pole", 100.0, 35.0)
    on_line = _feature("l", "object--support--utility-pole", 150.0, 35.0)
    line = OsmLine(1, np.array([_lonlat((120.0, 35.0)), _lonlat((1000.0, 35.0))]))
    plan = _plan_on(surface, mapillary=[field_sign, road_sign, stray, on_line],
                 osm=StreetFeatures(power_lines=(line,)))  # fmt: skip
    refs = {it.ref for it in plan.items}
    assert "mly:f" not in refs and "mly:x" not in refs
    assert {"mly:r", "mly:l"} <= refs
    (r,) = [it for it in plan.items if it.ref == "mly:r"]
    # right of the eastbound drivers (+Z), facing them; Mapillary said the other way
    assert r.kind == SIGN_YIELD and r.facing == pytest.approx((-1.0, 0.0), abs=1e-6)
    assert plan.clearance["dropped_far:sign"] == 1 and plan.clearance["dropped_far:pole"] == 1
    assert plan.clearance["flipped:sign"] == 1


def test_a_pole_on_a_roundabout_apron_moves_out_or_goes():
    surface = _surface(roundabout=True)
    c = RB.centre
    apron_r = (RB.inner_r + RB.island_r) / 2
    pole = StreetItem(POLE_UTILITY, (c[0], 0.0, c[1] + apron_r), (0.0, 1.0), "mapillary")
    # a rule sign on the island's very edge: into the island; one on a tiny island: dropped
    edge = StreetItem(SIGN_YIELD, (c[0] + RB.island_r - 0.1, 0.0, c[1]), (1.0, 0.0), "rule")
    res = clear_items([pole, edge], surface, ground, plate_width=lambda k: 0.75,
                      sign_gap_m=2.0, pole_edge_m=fu.POLE_EDGE_M)  # fmt: skip
    for it in res.items:
        q = _xz(it)
        assert surface.distance(q)[0] >= CLEARANCE_M - 0.05
    poles = [it for it in res.items if it.kind == POLE_UTILITY]
    assert len(poles) + res.stats["dropped_no_room:pole"] + res.stats["dropped_island:pole"] == 1
    for p in poles:  # out of the ring, not on the island
        assert np.linalg.norm(_xz(p) - c) > RB.outer_r + CLEARANCE_M
    (s,) = [it for it in res.items if it.kind == SIGN_YIELD]
    assert np.linalg.norm(_xz(s) - c) <= RB.island_r - CLEARANCE_M + 0.01
    tiny = DrivableSurface(
        shapely.box(-50, -50, 50, 50).difference(shapely.Point(0, 0).buffer(0.4)),
        shapely.Polygon(),
        [shapely.Point(0, 0).buffer(0.4)],
        [],
    )
    lone = StreetItem(SIGN_YIELD, (0.2, 0.0, 0.0), (1.0, 0.0), "rule")
    res = clear_items([lone], tiny, ground, plate_width=lambda k: 0.75, sign_gap_m=2.0,
                      pole_edge_m=fu.POLE_EDGE_M)  # fmt: skip
    assert res.items == [] and res.stats["dropped_island:sign"] == 1


def test_wires_follow_poles_moved_off_a_side_road():
    # The pole line's gap is filled with rule poles at x = 315 and 360: the one at 360 stands
    # on the side road and moves off it; the wires stay on the poles.
    surface = _surface()
    xs = [0, 45, 90, 135, 180, 225, 270, 405, 450, 495, 540]
    poles = [_feature(f"p{k}", "object--support--utility-pole", x, 9.0) for k, x in enumerate(xs)]
    photos = np.array([_lonlat((x, 0.0)) for x in np.arange(0.0, 1001.0, 20.0)])
    plan = _plan_on(surface, mapillary=poles, coverage=photos)
    got = [it for it in plan.items if it.kind == POLE_UTILITY]
    assert len(got) == len(xs) + 2
    assert not any(abs(it.position[0] - SIDE_X) < 3.5 + CLEARANCE_M for it in got)
    at = {(round(it.position[0], 2), round(it.position[2], 2)) for it in got}
    assert len(plan.wires) == len(got) - 1
    for a, b in plan.wires:
        assert (round(a[0], 2), round(a[2], 2)) in at and (round(b[0], 2), round(b[2], 2)) in at
    _assert_clear(plan, surface)


def test_nothing_ends_up_on_the_surface():
    surface = _surface(roundabout=True)
    rng = np.random.default_rng(3)
    values = ["object--support--utility-pole", "regulatory--stop--g1", "object--street-light",
              "regulatory--maximum-speed-limit-60--g1", "object--fire-hydrant"]  # fmt: skip
    feats = [
        _feature(f"r{k}", values[k % len(values)], float(x), float(z), heading=float(h))
        for k, (x, z, h) in enumerate(
            zip(
                rng.uniform(0, 1000, 150),
                rng.uniform(-80, 40, 150),
                rng.uniform(0, 360, 150),
                strict=True,
            )  # fmt: skip
        )
    ]
    rule = [
        StreetItem(SIGN_STOP, (SIDE_X + 1.0, 0.0, 20.0), (0.0, 1.0), "rule"),
        StreetItem(MAILBOX, (SIDE_X - 3.2, 0.0, 70.0), (1.0, 0.0), "rule"),
    ]
    plan = _plan_on(surface, mapillary=feats, items=rule)
    assert len(plan.items) > 20
    _assert_clear(plan, surface)
