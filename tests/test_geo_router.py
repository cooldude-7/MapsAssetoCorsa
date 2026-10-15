"""Tests for driftmaps.geo.graph and driftmaps.geo.router (all offline).

Synthetic networks are laid out in metres on a local projection; the real-road tests use the
committed Forks of the Credit Road extract (tests/fixtures/osm, ODbL).
"""

from __future__ import annotations

import gzip
import random
import warnings
from pathlib import Path

import numpy as np
import pytest

from driftmaps.geo.graph import build_graph
from driftmaps.geo.overpass import (
    OsmData,
    Way,
    build_query,
    parse_overpass,
    query_cache_key,
)
from driftmaps.geo.project import LocalProjection
from driftmaps.geo.router import (
    Route,
    RouteError,
    RouterOptions,
    route_from_pins,
    route_on_graph,
    snap_pin,
)

FIXTURE = Path(__file__).parent / "fixtures" / "osm" / "forks_of_the_credit.json"
FORKS_PINS = [(43.80410, -79.99231), (43.81060, -79.98185), (43.83180, -79.95940)]
FORKS_BUFFER_M = 400.0

ORIGIN = (45.0, -75.0)
PROJ = LocalProjection(*ORIGIN)


def make_osm(nodes, ways) -> OsmData:
    """nodes: id -> (east, north) metres; ways: [(way id, [node ids], tags)]."""
    out = []
    for wid, nids, tags in ways:
        xy = np.array([nodes[n] for n in nids], dtype=float)
        lon, lat = PROJ.to_lonlat(xy[:, 0], xy[:, 1])
        ll = np.stack([lon, lat], axis=-1)
        out.append(Way(wid, tuple(nids), ll, {"highway": "residential", **tags}))
    return OsmData(tuple(out), "2026-01-01T00:00:00Z", "testkey")


def pin(east: float, north: float) -> tuple[float, float]:
    lon, lat = PROJ.to_lonlat(east, north)
    return float(lat), float(lon)


def assert_no_u_turns(route: Route) -> None:
    for a, b in zip(route.edges, route.edges[1:], strict=False):
        assert a.v == b.u, "edges must be contiguous"
        assert not (a.u == b.v and a.v == b.u), f"U-turn at node {a.v}"


def lollipop(with_loop: bool = True) -> OsmData:
    """A W-E dead-end road 0..300 m with (optionally) a loop at its west end."""
    nodes = {1: (0, 0), 2: (100, 0), 3: (200, 0), 4: (300, 0)}
    ways = [(100, [1, 2, 3, 4], {"name": "Main"})]
    if with_loop:
        nodes.update({5: (-50, 50), 6: (-100, 0), 7: (-50, -50)})
        ways.append((200, [1, 5, 6, 7, 1], {"name": "Loop"}))
    return make_osm(nodes, ways)


def grid() -> OsmData:
    """3 x 3 junctions, 100 m apart; node id = 10*row + col + 11 (row = north index)."""
    nodes = {10 * r + c + 11: (100.0 * c, 100.0 * r) for r in range(3) for c in range(3)}
    ways = []
    for r in range(3):
        ways.append((100 + r, [10 * r + c + 11 for c in range(3)], {"name": f"Row {r}"}))
    for c in range(3):
        ways.append((200 + c, [10 * r + c + 11 for r in range(3)], {"name": f"Col {c}"}))
    return make_osm(nodes, ways)


# ---------------------------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------------------------


def test_graph_edges_and_lengths():
    g = build_graph(lollipop(), PROJ)
    assert g.n_edges == 2 * (3 + 4)  # every segment in both directions
    assert sorted(set(g.seg_way.tolist())) == [100, 200]
    # Main road segments are 100 m; loop segments 70.711 m.
    assert sorted(set(g.seg_len_mm.tolist())) == pytest.approx([70711, 100000], abs=2)
    assert g.node_degree[g.node_index(1)] == 3
    assert g.node_degree[g.node_index(4)] == 1
    for x in range(g.n_nodes):
        out = g.outgoing(x)
        assert np.all(g.edge_u[out] == x)
        assert np.all(np.diff(out) > 0)


def test_graph_oneway_rules():
    osm = make_osm({1: (0, 0), 2: (100, 0)}, [(1, [1, 2], {"oneway": "yes"})])
    assert build_graph(osm, PROJ).n_edges == 2  # ignored by default (race track)
    g = build_graph(osm, PROJ, respect_oneway=True)
    assert g.n_edges == 1 and g.edge_dir[0] == 1
    osm = make_osm({1: (0, 0), 2: (100, 0)}, [(1, [1, 2], {"oneway": "-1"})])
    g = build_graph(osm, PROJ, respect_oneway=True)
    assert g.n_edges == 1 and g.edge_dir[0] == -1


# ---------------------------------------------------------------------------------------------
# Snapping
# ---------------------------------------------------------------------------------------------


def test_snap_to_nearest_edge():
    g = build_graph(grid(), PROJ)
    s = snap_pin(g, *pin(30.0, 10.0), RouterOptions())
    assert s.node == -1
    assert g.seg_way[s.seg] == 100
    assert s.t == pytest.approx(0.3, abs=1e-6)
    assert s.dist == pytest.approx(10.0, abs=1e-3)
    assert s.xy == pytest.approx((30.0, 0.0), abs=1e-3)


def test_snap_to_node_near_segment_end():
    g = build_graph(grid(), PROJ)
    s = snap_pin(g, *pin(100.4, 0.3), RouterOptions())
    assert s.node == g.node_index(12)
    assert s.xy == pytest.approx((100.0, 0.0), abs=1e-3)


def test_snap_prefers_higher_class_within_tolerance():
    osm = make_osm(
        {1: (0, 0), 2: (100, 0), 3: (0, 2.5), 4: (100, 2.5)},
        [(1, [1, 2], {"highway": "residential"}), (2, [3, 4], {"highway": "primary"})],
    )
    g = build_graph(osm, PROJ)
    s = snap_pin(g, *pin(50.0, -0.5), RouterOptions())
    assert g.seg_way[s.seg] == 2
    s = snap_pin(g, *pin(50.0, -0.5), RouterOptions(snap_class_tolerance_m=0.0))
    assert g.seg_way[s.seg] == 1


def test_snap_too_far():
    g = build_graph(grid(), PROJ)
    with pytest.raises(RouteError, match="from the nearest road"):
        snap_pin(g, *pin(50.0, 500.0), RouterOptions())


# ---------------------------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------------------------


def test_no_u_turn_takes_the_loop():
    pins = [pin(150, 0), pin(50, 0), pin(250, 0)]
    r = route_on_graph(build_graph(lollipop(), PROJ), pins)
    assert_no_u_turns(r)
    assert {e.way for e in r.edges} == {100, 200}
    # 150 m west + the 4 x 70.71 m loop + 250 m east.
    assert r.length_m == pytest.approx(150 + 4 * 50 * np.sqrt(2) + 250, abs=0.01)
    assert set(r.repeated_nodes()) >= {1, 2}
    with pytest.warns(UserWarning, match="same node twice"):
        route_from_pins(pins, osm=lollipop())


def test_no_route_without_u_turn():
    with pytest.raises(RouteError, match="without U-turns"):
        route_on_graph(build_graph(lollipop(False), PROJ), [pin(150, 0), pin(50, 0), pin(250, 0)])


def test_simple_route_partial_edges():
    r = route_on_graph(build_graph(lollipop(), PROJ), [pin(130, 5), pin(270, -5)])
    assert [(e.u, e.v, e.dir) for e in r.edges] == [(2, 3, 1), (3, 4, 1)]
    assert r.start_t == pytest.approx(0.3) and r.end_t == pytest.approx(0.7)
    assert r.length_m == pytest.approx(140.0, abs=0.002)


def test_waypoint_order_respected():
    g = build_graph(grid(), PROJ)
    a, b, c = pin(50, 0), pin(200, 150), pin(50, 200)
    for pins in ([a, b, c], [a, c, b], [c, a, b]):
        r = route_on_graph(g, pins)
        assert_no_u_turns(r)
        pl = r.polyline(1.0)
        xy = PROJ.to_xy(pl.lonlat)
        idx = []
        for p in r.pins:
            sxy = PROJ.to_xy(np.array([p.snap_lon, p.snap_lat]))
            d = np.linalg.norm(xy - sxy, axis=1)
            # The polyline is sampled every 1 m, so a waypoint can sit up to half a step from
            # the nearest sample (exactly how far depends on the platform's float rounding).
            assert d.min() <= 0.5 + 1e-6
            idx.append(int(np.argmin(d)))
        assert idx == sorted(idx) and idx[0] == 0 and idx[-1] == len(pl) - 1


def test_turn_penalty_prefers_fewer_turns():
    # From the SW corner going to the NE corner: every monotone path is 400 m; the router must
    # pick one with a single turn (straight up a column, then along a row), not a staircase.
    r = route_on_graph(build_graph(grid(), PROJ), [pin(0.2, 0.3), pin(199.8, 199.7)])
    assert r.pins[0].node == 11 and r.pins[1].node == 33
    ways = [e.way for e in r.edges]
    changes = sum(1 for x, y in zip(ways, ways[1:], strict=False) if x != y)
    assert changes == 1
    assert r.length_m == pytest.approx(400.0, abs=0.01)


def test_respect_oneway_option():
    # Short direct road is one-way eastbound; a 3-segment detour is two-way.
    nodes = {1: (0, 0), 2: (300, 0), 3: (0, 100), 4: (300, 100)}
    ways = [
        (1, [1, 2], {"oneway": "yes"}),
        (2, [1, 3], {}),
        (3, [3, 4], {}),
        (4, [4, 2], {}),
    ]
    osm = make_osm(nodes, ways)
    west = [pin(250, 0), pin(50, 0)]
    assert {e.way for e in route_from_pins(west, osm=osm).edges} == {1}
    r = route_from_pins(west, osm=osm, options=RouterOptions(respect_oneway=True))
    assert {e.way for e in r.edges} == {1, 2, 3, 4}
    assert r.repeated_nodes() == []  # both ends lie on segment 1-2, but don't overlap
    assert r.options["respect_oneway"] is True


def test_determinism_and_json_round_trip():
    pins = [pin(50, 0), pin(200, 150), pin(50, 200)]
    text = route_from_pins(pins, osm=grid()).to_json()
    assert route_from_pins(pins, osm=grid()).to_json() == text
    shuffled = list(grid().ways)
    random.Random(4).shuffle(shuffled)
    osm = OsmData(tuple(shuffled), "2026-01-01T00:00:00Z", "testkey")
    assert route_from_pins(pins, osm=osm).to_json() == text
    assert Route.from_json(text).to_json() == text


def test_reverse():
    pins = [pin(30, 0), pin(200, 150), pin(70, 200)]
    r = route_from_pins(pins, osm=grid())
    rev = r.reverse()
    assert rev.length_m == r.length_m
    assert rev.pins == tuple(reversed(r.pins))
    assert rev.edges[0].u == r.edges[-1].v and rev.edges[-1].v == r.edges[0].u
    assert rev.start_t == pytest.approx(1 - r.end_t) and rev.end_t == pytest.approx(1 - r.start_t)
    assert_no_u_turns(rev)
    assert rev.reverse().to_json() == r.to_json()
    a, b = r.polyline(2.0), rev.polyline(2.0)
    np.testing.assert_allclose(b.lonlat, a.lonlat[::-1], atol=1e-9)
    assert b.s[-1] == pytest.approx(a.s[-1])


def test_polyline_attributes():
    osm = make_osm(
        {1: (0, 0), 2: (100, 0), 3: (200, 0)},
        [
            (1, [1, 2], {"highway": "secondary", "lanes": "2", "name": "A", "oneway": "yes"}),
            (2, [2, 3], {"highway": "tertiary", "bridge": "yes", "layer": "1", "width": "9"}),
        ],
    )
    r = route_from_pins([pin(10, 0), pin(190, 0)], osm=osm)
    pl = r.polyline(0.5)
    assert len(pl) == 361
    assert np.all(np.diff(pl.s) <= 0.5 + 1e-9)
    assert pl.s[-1] == pytest.approx(r.length_m, abs=0.002) == pytest.approx(180.0, abs=0.002)
    first, last = pl.s < 89.9, pl.s > 90.1
    assert set(pl.way_id[first]) == {1} and set(pl.way_id[last]) == {2}
    np.testing.assert_allclose(pl.width_m[first], 6.8)  # 2 lanes x 3.4 m (North America)
    np.testing.assert_allclose(pl.width_m[last], 9.0)
    assert set(pl.highway[first]) == {"secondary"} and set(pl.name[first]) == {"A"}
    assert not pl.bridge[first].any() and pl.bridge[last].all()
    assert set(pl.layer[last]) == {1} and not pl.tunnel.any()
    assert set(pl.oneway[first]) == {1} and set(pl.oneway[last]) == {0}
    assert set(r.reverse().polyline(0.5).oneway[:10]) == {0}
    assert set(r.reverse().polyline(0.5).oneway[-10:]) == {-1}  # now against the one-way
    assert set(pl.surface) == {"asphalt"} and set(pl.lanes) == {2}


# ---------------------------------------------------------------------------------------------
# Real road: Forks of the Credit Road, Caledon, Ontario
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def forks_osm() -> OsmData:
    key = query_cache_key(build_query(FORKS_PINS, FORKS_BUFFER_M))
    return parse_overpass(FIXTURE.read_bytes(), key)


@pytest.fixture(scope="module")
def forks_route(forks_osm) -> Route:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no self-crossing / length warnings expected
        return route_from_pins(FORKS_PINS, osm=forks_osm)


def test_forks_route_follows_the_road(forks_route):
    r = forks_route
    assert 4300.0 < r.length_m < 4400.0
    assert {r.ways[e.way].get("name") for e in r.edges} == {"Forks of the Credit Road"}
    assert_no_u_turns(r)
    assert r.repeated_nodes() == []
    assert r.driving_side == "right"
    assert r.osm_base == "2026-10-15T13:51:13Z"
    assert all(p.dist_m < 15.0 for p in r.pins)
    # Junctions with side roads (McLaren Road, Dominion Street, ...) are recorded as stubs.
    stub_names = {r.ways[w].get("name") for _, w, _ in r.stubs}
    assert "McLaren Road" in stub_names
    for at, way, to in r.stubs:
        assert at in r.nodes and to in r.nodes and way in r.ways


def test_forks_polyline(forks_route):
    pl = forks_route.polyline(1.0)
    assert np.all(np.diff(pl.s) <= 1.0 + 1e-9)
    assert pl.s[-1] == pytest.approx(forks_route.length_m, abs=0.01)
    first, last = forks_route.pins[0], forks_route.pins[-1]
    np.testing.assert_allclose(pl.lonlat[0], [first.snap_lon, first.snap_lat], atol=1e-7)
    np.testing.assert_allclose(pl.lonlat[-1], [last.snap_lon, last.snap_lat], atol=1e-7)
    assert set(pl.highway) == {"secondary"}
    assert set(pl.surface) == {"asphalt"}
    # A twisty road (with a hairpin): over 360 degrees of total heading change in 4.3 km.
    xy = forks_route.projection().to_xy(pl.lonlat[::5])
    h = np.unwrap(np.arctan2(*np.diff(xy, axis=0).T[::-1]))
    assert np.degrees(np.sum(np.abs(np.diff(h)))) > 360.0


def test_forks_deterministic_and_cached_offline(forks_osm, forks_route, tmp_path):
    text = forks_route.to_json()
    assert route_from_pins(FORKS_PINS, osm=forks_osm).to_json() == text
    # Same result through the on-disk cache, fully offline.
    key = forks_osm.cache_key
    cached = tmp_path / "overpass" / f"{key}.json.gz"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(gzip.compress(FIXTURE.read_bytes()))
    offline = route_from_pins(FORKS_PINS, tmp_path, buffer_m=FORKS_BUFFER_M, offline=True)
    assert offline.to_json() == text
    path = forks_route.save(tmp_path / "route.json")
    assert path.read_text(encoding="utf-8") == text
    assert Route.load(path).to_json() == text


def test_forks_reverse_matches_rerouting(forks_osm, forks_route):
    rerouted = route_from_pins(FORKS_PINS[::-1], osm=forks_osm)
    assert rerouted.edges == forks_route.reverse().edges
    assert rerouted.length_m == pytest.approx(forks_route.length_m, abs=0.002)
