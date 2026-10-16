"""Ontario traffic signals: every approach of a signalised junction, heads per lane, no stop sign
under a signal, poles off the road with their arms over it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from driftmaps.geo.mapillary import MapFeature
from driftmaps.geo.project import LocalProjection
from driftmaps.geo.street_features import OsmNode, StreetFeatures
from driftmaps.geometry import furniture as fu
from driftmaps.geometry.clearance import CLEARANCE_M, DrivableSurface
from driftmaps.geometry.road import RoadModel
from driftmaps.geometry.signals import (
    ARM_Y_M,
    HEAD_CLEAR_M,
    SignalApproach,
    approach_lanes,
    arm_reach,
    plan_junction,
    signalised,
)
from driftmaps.geometry.street_items import (
    LIGHT_STREET,
    SIGN_STOP,
    SIGN_YIELD,
    SIGNAL_TRAFFIC,
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


def ground(xz):
    return np.zeros(len(np.asarray(xz).reshape(-1, 2)))


def _model() -> RoadModel:
    x = np.arange(0.0, 1001.0)
    c = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    return RoadModel(x.copy(), c, 7.0, 0.0, True, False)  # a 4-lane road, 14 m wide


ARTERIAL = {"highway": "secondary", "name": "Brant Street", "lanes": "4",
            "turn:lanes:forward": "left|through|through;right",
            "turn:lanes:backward": "left|through|through;right"}  # fmt: skip
SIDE = {"highway": "tertiary", "name": "Caroline Street", "lanes": "2"}
# The route (an arterial along +X) crosses a tertiary at x = 500, signalised.
ROUTE = FakeRoute(
    {1: (0, 0), 2: (500, 0), 3: (1000, 0), 4: (500, -100), 5: (500, 100)},
    [Edge(11, 1, 2), Edge(12, 2, 3)],
    {11: ARTERIAL, 12: ARTERIAL, 13: SIDE},
    [(2, 13, 4), (2, 13, 5)],
)
SIGNAL = OsmNode(99, *_lonlat((500.0, 0.0)), {"highway": "traffic_signals"})


def _plan(route=ROUTE, nodes=(SIGNAL,), **kw):
    m = _model()
    feats = StreetFeatures(tuple(nodes), ())
    args = dict(s_range=(0.0, 1000.0), way_id=np.where(m.s < 500.0, 11, 12), osm=feats)
    return fu.plan_furniture(m, PROJ, route, ground, **(args | kw)), m


def test_approach_lanes_from_osm_tags():
    assert approach_lanes(ARTERIAL, True) == (2, 1)  # left | through | through;right
    assert approach_lanes({"lanes": "4"}, None) == (2, 0)
    assert approach_lanes({"lanes": "2"}, False) == (1, 0)
    assert approach_lanes({"lanes:forward": "3", "lanes:backward": "1"}, True) == (3, 0)
    assert approach_lanes({"lanes:forward": "3", "lanes:backward": "1"}, False) == (1, 0)
    assert approach_lanes({"oneway": "yes", "turn:lanes": "left|left|through|right"}, True) == (
        1,
        2,
    )
    assert approach_lanes({}, None) == (1, 0)


def test_signals_on_every_approach_with_a_head_per_lane():
    plan, m = _plan()
    sig = plan.signals
    assert len(sig.junctions) == 1 and sig.approaches == 4
    arms = [p for p in sig.poles if not p.post]
    assert len(arms) == 4  # one corner pole per approach, side streets included
    by_axis = {tuple(np.round(p.axis).astype(int)): p for p in arms}
    for axis in ((1, 0), (-1, 0)):  # the route's approaches: 2 through + 1 left arrow, green
        p = by_axis[axis]
        assert sorted(k for _, k in p.heads) == ["left", "through", "through"]
        assert p.go and p.name == "Caroline Street"
    for axis in ((0, 1), (0, -1)):  # the side street's: 1 through lane each way, red
        p = by_axis[axis]
        assert [k for _, k in p.heads] == ["through"] and not p.go
        assert p.name == "Brant Street"
    stats = sig.stats()
    assert stats["heads"] == 2 * 3 + 2 * 1 + 4  # overhead + one pole-mounted per approach
    assert stats["arrow_heads"] == 2 and stats["ped_heads"] == 8 and stats["push_buttons"] == 8
    items = [it for it in plan.items if it.kind == SIGNAL_TRAFFIC]
    assert len(items) == len(sig.poles)
    assert sum(1 for it in plan.items if it.kind == LIGHT_STREET and it.text == "on_pole") == 4


def test_no_stop_sign_on_a_signalised_approach():
    # Without the signal the tertiary-class side street would stop; the network's (wrongly
    # facing) stop sign on the side street's approach is dropped too.
    wrong = StreetItem(SIGN_STOP, (494.0, 0.0, 12.0), (0.0, -1.0), "rule", ref="junction:2")
    yield_ = StreetItem(SIGN_YIELD, (507.0, 0.0, -14.0), (0.0, -1.0), "osm")
    far = StreetItem(SIGN_STOP, (800.0, 0.0, 12.0), (-1.0, 0.0), "osm")
    plan, _ = _plan(items=[wrong, yield_, far])
    stops = [it for it in plan.items if it.kind in (SIGN_STOP, SIGN_YIELD)]
    assert stops == [far] and plan.stops_under_signals == 2
    # The same junction without signals keeps its stop signs.
    plan, _ = _plan(nodes=(), items=[wrong])
    assert any(it.kind == SIGN_STOP for it in plan.items) and not plan.signals.junctions


def test_signal_poles_stay_off_the_road_and_arms_reach_over_it():
    plan, m = _plan()
    surf = DrivableSurface.build(m)
    poles = {p.ref: p for p in plan.signals.poles}
    for it in plan.items:
        if it.kind != SIGNAL_TRAFFIC:
            continue
        xz = np.array([[it.position[0], it.position[2]]])
        assert surf.distance(xz)[0] >= CLEARANCE_M - 0.02
        p = poles[it.ref]
        if p.post or abs(p.axis[1]) > 0.5:
            continue  # posts; side-street arms reach over the side street (not built here)
        # A route approach's arm spans from its pole to beyond the leftmost head.
        lat = abs(it.position[2])
        assert arm_reach(p, xz[0]) > lat - min(off for off, _ in p.heads)
        assert arm_reach(p, xz[0]) < 15.5


def test_signal_meshes_have_heads_at_ontario_heights():
    plan, m = _plan()
    meshes = fu.furniture_meshes(plan, m, names=fu.street_names(plan.items, plan.signals))
    signs = np.concatenate([fm.mesh.positions for fm in meshes if fm.name.startswith("signs_")])
    near = np.hypot(signs[:, 0] - 500.0, signs[:, 2]) < 30.0
    ys = signs[near, 1]
    over = ys[(ys > 4.5) & (ys < 8.0)]  # overhead heads and name signs (not the luminaire)
    assert len(over) and over.min() >= HEAD_CLEAR_M - 0.01  # lenses of the overhead heads
    assert over.max() <= ARM_Y_M  # under the arm
    names = fu.street_names(plan.items, plan.signals)
    assert {"Brant Street", "Caroline Street"} <= set(names)


def test_signalised_by_osm_node_or_mapillary_light():
    j = np.array([0.0, 0.0])
    assert signalised(j, np.array([[25.0, 3.0]]), np.zeros((0, 2)), np.zeros((0, 2)))
    assert not signalised(j, np.array([[60.0, 0.0]]), np.zeros((0, 2)), np.zeros((0, 2)))
    # a signal node nearer another junction belongs to it
    assert not signalised(j, np.array([[30.0, 0.0]]), np.zeros((0, 2)), np.array([[45.0, 0.0]]))
    assert signalised(j, np.zeros((0, 2)), np.array([[10.0, 12.0]]), np.zeros((0, 2)))
    lon, lat = _lonlat((500.0, 8.0))
    light = MapFeature("t1", "object--traffic-light--general-upright", float(lon), float(lat),
                       90.0, "2023-06-01T00:00:00Z", "2023-06-01T00:00:00Z")  # fmt: skip
    plan, _ = _plan(nodes=(), mapillary=[light])
    assert len(plan.signals.junctions) == 1


def test_t_junction_gets_poles_for_all_three_approaches():
    legs = [
        SignalApproach(np.array([-1.0, 0.0]), 7.0, ARTERIAL, True, on_route=True),
        SignalApproach(np.array([1.0, 0.0]), 7.0, ARTERIAL, False, on_route=True),
        SignalApproach(np.array([0.0, 1.0]), 3.5, SIDE),
    ]
    poles, reach = plan_junction(7, np.array([0.0, 0.0]), legs)
    arms = [p for p in poles if not p.post]
    assert len(arms) == 3 and reach > 0
    side = next(p for p in arms if p.axis[1] < -0.5)  # the side street's approach (going -Z)
    # its far-right pole stands across the arterial (z < -7) on the driver's right (x > 0)
    assert side.xz[1] < -7.0 and side.xz[0] > 0.0
    assert all(p.ped for p in poles)
    with pytest.raises(StopIteration):
        next(p for p in arms if p.axis[1] > 0.5)  # nobody arrives from the missing leg


def test_a_divided_road_crossing_is_one_intersection():
    # The route (+X) crosses a divided road: its southbound carriageway at x = 492, the
    # northbound one at x = 508, one OSM signal node between them.
    sb = {"highway": "primary", "oneway": "yes", "lanes": "2", "name": "Dundas Street"}
    route = FakeRoute(
        {1: (0, 0), 2: (492, 0), 3: (508, 0), 4: (1000, 0), 5: (492, -100), 6: (492, 100),
         7: (508, -100), 8: (508, 100)},
        [Edge(11, 1, 2), Edge(11, 2, 3), Edge(11, 3, 4)],
        {11: SIDE | {"name": "Caroline Street"}, 21: sb, 22: sb},
        [(2, 21, 5), (2, 21, 6), (3, 22, 7), (3, 22, 8)],
    )  # fmt: skip
    m = _model()
    feats = StreetFeatures((SIGNAL,), ())
    plan = fu.plan_furniture(m, PROJ, route, ground, s_range=(0.0, 1000.0),
                             way_id=np.full(len(m), 11), osm=feats)  # fmt: skip
    sig = plan.signals
    assert len(sig.junctions) == 1 and sig.nodes == {2, 3}
    arms = [p for p in sig.poles if not p.post]
    assert len(arms) == 4  # EB, WB on the route; NB, SB on the divided road
    for p in arms:
        if abs(p.axis[1]) < 0.5:
            continue
        r = np.array([-p.axis[1], p.axis[0]])
        xs = [float((p.origin + r * off)[0]) for off, _ in p.heads]
        # Southbound (+Z) traffic keeps right, on the west carriageway; northbound on the east.
        want = 492.0 if p.axis[1] > 0 else 508.0
        assert len(xs) == 2 and all(abs(x - want) <= 3.6 for x in xs)
        assert p.name == "Caroline Street"
