"""Divided roads (`geometry.divided`): pairing a dual road's two carriageways and its median,
and the network building it (a raised curbed median, junctions across both carriageways)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely

from driftmaps.geometry.divided import (
    carriageway,
    carriageway_key,
    median_kind,
    median_polygons,
    median_width,
)
from driftmaps.geometry.network import NetWay, NetworkOptions, build_network
from driftmaps.geometry.road import RoadModel


def _line(z: float, x0: float = 0.0, x1: float = 200.0) -> np.ndarray:
    x = np.arange(x0, x1 + 0.5, 1.0)
    return np.stack([x, np.full(len(x), z)], axis=1)


def test_pairing_needs_a_shared_name_and_opposite_directions():
    tags = {"name": "Dundas Street West", "ref": "5", "oneway": "yes"}
    a = carriageway(_line(0.0), 5.4, +1, tags)
    b = carriageway(_line(15.0), 5.4, -1, tags)  # 15 m apart: a 4.2 m median
    strips = median_polygons([a, b])
    assert len(strips) == 1
    assert median_width(strips[0]) == pytest.approx(4.2, abs=0.3)
    assert median_kind(strips[0]) == "raised_grass"
    same_way = carriageway(_line(15.0), 5.4, +1, tags)
    assert median_polygons([a, same_way]) == []
    other = carriageway(_line(15.0), 5.4, -1, {"name": "Other Road", "oneway": "yes"})
    assert median_polygons([a, other]) == []
    far = carriageway(_line(60.0), 5.4, -1, tags)
    assert median_polygons([a, far]) == []
    assert carriageway(_line(0.0), 5.4, 0, tags) is None  # two-way: not a carriageway
    assert carriageway_key({"ref": "5;25"}) == {"ref:5", "ref:25"}


def test_median_kinds_by_width():
    narrow = shapely.box(0, 0, 100, 1.5)
    mid = shapely.box(0, 0, 100, 5.0)
    wide = shapely.box(0, 0, 100, 14.0)
    assert [median_kind(p) for p in (narrow, mid, wide)] == [
        "raised_concrete",
        "raised_grass",
        "grass",
    ]


def _model(xz: np.ndarray, hw: float) -> RoadModel:
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))])
    centre = np.stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]], axis=1)
    return RoadModel(s=s, centre=centre, half_width=hw, shoulder=1.0, centre_line=False,
                     bridge=False)  # fmt: skip


def test_network_builds_the_median_and_a_junction_across_both_carriageways():
    """The route drives the eastbound carriageway of a divided road (z = 0, x -> +); the
    westbound one runs 14 m north (z = -14); a crossing road at x = 100 joins both through a
    short connector between them."""
    east = {"highway": "primary", "name": "Dundas Street West", "oneway": "yes", "lanes": "2"}
    west = dict(east)
    route_xz = _line(0.0)
    model = _model(route_xz, 3.6)
    n_nodes = 21  # every 10 m
    xs = np.linspace(0.0, 200.0, n_nodes)
    r_nodes = list(range(1, n_nodes + 1))
    w_nodes = list(range(101, 101 + n_nodes))
    ways = [
        NetWay(1, tuple(r_nodes), np.stack([xs, np.zeros(n_nodes)], 1), east),
        NetWay(
            2, tuple(reversed(w_nodes)), np.stack([xs[::-1], np.full(n_nodes, -14.0)], 1), west
        ),  # fmt: skip
        # The crossing road: south arm, the connector between the carriageways, north arm.
        NetWay(
            3,
            (201, r_nodes[10]),
            np.array([[100.0, 60.0], [100.0, 0.0]]),
            {"highway": "secondary", "name": "Neyagawa"},
        ),  # fmt: skip
        NetWay(
            4,
            (r_nodes[10], w_nodes[10]),
            np.array([[100.0, 0.0], [100.0, -14.0]]),
            {"highway": "secondary", "name": "Neyagawa"},
        ),  # fmt: skip
        NetWay(
            5,
            (w_nodes[10], 202),
            np.array([[100.0, -14.0], [100.0, -80.0]]),
            {"highway": "secondary", "name": "Neyagawa"},
        ),  # fmt: skip
    ]
    segs = {(min(a, b), max(a, b)) for a, b in zip(r_nodes[:-1], r_nodes[1:], strict=True)}
    net = build_network(
        model, ways, route_nodes=r_nodes, route_segments=segs,
        route_highway=["primary"] * len(model), route_oneway=np.ones(len(model), np.int64),
        dtm=lambda xz: np.zeros(len(np.asarray(xz).reshape(-1, 2))),
        options=NetworkOptions(radius_m=120.0),
        route_way_id=np.ones(len(model), np.int64), route_way_tags={1: east},
    )  # fmt: skip
    assert net.stats["medians"] >= 2  # west and east of the crossing road
    assert net.stats["medians_raised_m2"] > 500.0
    # The medians stop at the junction: the crossing road (the connector too) is paved.
    connector = [sr for sr in net.side_roads if set(sr.way_ids.tolist()) == {4}]
    assert connector and connector[0].length < 15.0
    gap = shapely.box(95.0, -12.0, 105.0, -2.0)
    for isl in net.islands:
        assert isl.intersection(gap).area < 1.0
    assert net.paved.contains(shapely.Point(100.0, -7.0))
    # The median sits between the carriageways, clear of the route's asphalt.
    median = [isl for isl in net.islands if isl.area > 100.0]
    assert median and all(isl.bounds[3] < -3.5 and isl.bounds[1] > -10.5 for isl in median)
    keys = {key for key, *_ in [(p.name,) for p in net.pieces]}
    assert any(k.startswith(("1ROAD_curb", "1KERB_curb")) for k in keys)  # the median's curbs
    assert any(k.startswith("1GRASS_isl") for k in keys)  # ... and its grass top
    # No control inside the junction: the connector between the carriageways has no stop.
    assert not any(it.ref == f"junction:{w_nodes[10]}" and it.kind.startswith("sign:stop")
                   and math.hypot(it.xz[0] - 100.0, it.xz[1] + 7.0) < 4.0
                   for it in net.planned_items)  # fmt: skip
