"""`route_from_path`: a Route along an explicit OSM node path (the app's turn-by-turn builder).

A small synthetic block (two-way streets round a square, plus two one-ways) and the Forks of the
Credit fixture (the path of a pin route must give the same route back). Offline.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from driftmaps.geo import router
from driftmaps.geo.overpass import OsmData, Way, parse_overpass
from driftmaps.geo.router import Route, RouteError, route_from_path

FIXTURES = Path(__file__).parent / "fixtures"
LAT0, LON0 = 43.80, -79.90
D_LAT, D_LON = 0.0015, 0.002  # ~167 m north, ~160 m east

# Node id -> (column, row) on the grid.
GRID = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0), 5: (2, 1), 6: (2, 2), 7: (0, 2), 9: (0, 1),
        10: (3, 1), 11: (1, -1)}  # fmt: skip


def _way(way_id: int, nodes: list[int], **tags: str) -> Way:
    lonlat = np.array([[LON0 + GRID[n][0] * D_LON, LAT0 + GRID[n][1] * D_LAT] for n in nodes])
    return Way(way_id, tuple(nodes), lonlat, {"highway": "residential", **tags})


@pytest.fixture(scope="module")
def block() -> OsmData:
    """A square of two-way streets 1-2-3 / 3-5-6 / 6-7 / 7-9-1, Main St going on to 4, a
    one-way north from 4 to 10 and a one-way (oneway=-1) north from 11 to 2."""
    ways = (
        _way(101, [1, 2, 3, 4], name="Main Street"),
        _way(102, [3, 5, 6], name="North Road"),
        _way(103, [6, 7], name="Top Lane"),
        _way(104, [7, 9, 1], name="West Road"),
        _way(105, [4, 10], name="One Way Street", oneway="yes"),
        _way(106, [2, 11], oneway="-1"),
    )
    return OsmData(ways, "2026-10-15T00:00:00Z", "test")


def _dist(a: int, b: int) -> float:
    (x0, y0), (x1, y1) = GRID[a], GRID[b]
    dy = (y1 - y0) * D_LAT * 111_195.0
    dx = (x1 - x0) * D_LON * 111_195.0 * math.cos(math.radians(LAT0))
    return math.hypot(dx, dy)


def test_path_route_along_the_nodes(block):
    route = route_from_path(block, [1, 2, 3, 5, 6], 0.5, 0.25)
    assert [(e.way, e.u, e.v, e.dir) for e in route.edges] == [
        (101, 1, 2, 1),
        (101, 2, 3, 1),
        (102, 3, 5, 1),
        (102, 5, 6, 1),
    ]
    want = 0.5 * _dist(1, 2) + _dist(2, 3) + _dist(3, 5) + 0.25 * _dist(5, 6)
    assert route.length_m == pytest.approx(want, rel=2e-3)
    assert route.start_t == 0.5 and route.end_t == 0.25 and not route.closed
    assert route.node_path() == [1, 2, 3, 5, 6]
    # The two pins are where it starts and ends, on the road.
    a, b = route.pins
    assert (a.u, a.v, a.t, a.node) == (1, 2, 0.5, None)
    assert a.snap_lon == pytest.approx(LON0 + 0.5 * D_LON, abs=1e-7)
    assert a.snap_lat == pytest.approx(LAT0, abs=1e-7)
    assert (b.u, b.v, b.t) == (5, 6, 0.25)
    assert b.snap_lat == pytest.approx(LAT0 + 1.25 * D_LAT, abs=1e-7)
    # Side roads at the junctions it passes become stubs (junction stubs / walls).
    assert (2, 106, 11) in route.stubs and (3, 101, 4) in route.stubs
    # It is a normal route: the JSON contract round-trips.
    assert Route.from_json(route.to_json()).to_json() == route.to_json()


def test_path_against_the_way_order(block):
    route = route_from_path(block, [6, 5, 3, 2], 0.0, 1.0)
    assert [(e.way, e.dir) for e in route.edges] == [(102, -1), (102, -1), (101, -1)]
    assert route.pins[0].node == 6 and route.pins[-1].node == 2


def test_path_matches_the_pin_route():
    """The node path of a route_from_pins route gives the same route back."""
    osm = parse_overpass((FIXTURES / "osm" / "forks_of_the_credit.json").read_bytes())
    pinned = router.route_from_pins([(43.80410, -79.99231), (43.81060, -79.98185)], osm=osm)
    path = route_from_path(osm, pinned.node_path(), pinned.start_t, pinned.end_t)
    assert path.edges == pinned.edges
    assert path.length_m == pytest.approx(pinned.length_m, abs=0.01)
    assert path.stubs == pinned.stubs
    assert path.nodes.keys() >= {n for e in pinned.edges for n in (e.u, e.v)}


def test_one_ways(block):
    route_from_path(block, [4, 10])  # with the way
    route_from_path(block, [11, 2])  # oneway=-1: against the node order
    with pytest.raises(RouteError, match="one-way") as info:
        route_from_path(block, [10, 4])
    assert info.value.road == "One Way Street"
    with pytest.raises(RouteError, match="one-way") as info:
        route_from_path(block, [3, 2, 11])
    assert info.value.road is None
    # The A-to-B router's choice: one-ways can be ignored.
    assert route_from_path(block, [10, 4], respect_oneway=False).edges[0].dir == -1


@pytest.mark.parametrize(
    ("path", "start_t", "end_t", "match"),
    [
        ([1], 0.0, 1.0, "at least two"),
        ([1, 3], 0.0, 1.0, "not next to each other"),
        ([1, 2, 1], 0.0, 1.0, "turns back"),
        ([1, 1, 2], 0.0, 1.0, "repeated"),
        ([1, 999], 0.0, 1.0, "not in the loaded road data"),
        ([1, 2], 0.6, 0.4, "ends before it starts"),
        ([1, 2], 0.0, 1.5, "between 0 and 1"),
    ],
)
def test_bad_paths(block, path, start_t, end_t, match):
    with pytest.raises(RouteError, match=match):
        route_from_path(block, path, start_t, end_t)


def test_loops(block):
    lap = [1, 2, 3, 5, 6, 7, 9, 1]
    perimeter = sum(_dist(a, b) for a, b in zip(lap, lap[1:], strict=False))
    route = route_from_path(block, lap, 0.0, loop=True)
    assert route.closed and route.length_m == pytest.approx(perimeter, rel=2e-3)
    assert route.node_path() == lap
    # Starting mid-segment, the lap ends where it started (the first segment again).
    mid = route_from_path(block, lap, 0.4, loop=True)
    assert mid.closed and mid.length_m == pytest.approx(perimeter, rel=2e-3)
    assert mid.edges[0] == mid.edges[-1] and mid.start_t == mid.end_t == 0.4
    assert mid.node_path() == [*lap, 2]
    # Every junction on a lap is interior (stubs at the start node too).
    assert (1, 101, 2) not in mid.stubs and (2, 106, 11) in mid.stubs

    # A path that drives on over the start segment (to its second node) closes at A too.
    over = route_from_path(block, [*lap, 2], 0.4, loop=True)
    assert over.edges == mid.edges and over.length_m == mid.length_m

    with pytest.raises(RouteError, match="first node"):
        route_from_path(block, [1, 2, 3, 5], loop=True)
    with pytest.raises(RouteError, match="no loop"):  # round twice: drives every road twice
        route_from_path(block, lap + lap[1:], loop=True)
