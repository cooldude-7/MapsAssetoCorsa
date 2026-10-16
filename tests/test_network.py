"""The road network around the route (geometry.network, geometry.roundabouts): meshing, the
height field, Ontario widths and corner radii, junction control, roundabouts, driveways,
the real run-up / run-off and the terrain and wall hooks."""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely

from driftmaps.geo.project import LocalProjection
from driftmaps.geo.router import Route, RouteEdge, SnappedPin, route_continuations
from driftmaps.geometry.network import (
    CORNER_RADIUS_M,
    CentreLine,
    HeightField,
    NetNode,
    NetWay,
    NetworkOptions,
    PlaneZone,
    build_driveways,
    build_network,
    corner_radius,
    mesh_polygon,
    side_road_width,
    triangulate,
    way_kind,
)
from driftmaps.geometry.road import RoadModel, build_road_meshes
from driftmaps.geometry.street_items import (
    MAILBOX,
    SIGN_CHEVRON_RIGHT,
    SIGN_KEEP_RIGHT,
    SIGN_ROUNDABOUT_AHEAD,
    SIGN_STOP,
    SIGN_YIELD,
)
from driftmaps.geometry.terrain import TerrainExtra, TerrainOptions, build_terrain
from driftmaps.geometry.walls import wall_lines

OPTS = NetworkOptions(radius_m=120.0)


def _model(xz: np.ndarray, hw: float = 3.5, y=None) -> RoadModel:
    seg = np.linalg.norm(np.diff(xz, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = int(math.ceil(s[-1]))
    t = np.linspace(0.0, s[-1], n + 1)
    pts = np.stack([np.interp(t, s, xz[:, 0]), np.interp(t, s, xz[:, 1])], axis=1)
    yy = np.zeros(len(t)) if y is None else y(pts)
    centre = np.stack([pts[:, 0], yy, pts[:, 1]], axis=1)
    return RoadModel(
        s=t, centre=centre, half_width=hw, shoulder=1.0, centre_line=True, bridge=False
    )


def flat(xz: np.ndarray) -> np.ndarray:
    return np.zeros(len(np.asarray(xz).reshape(-1, 2)))


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def test_way_kinds_and_widths():
    assert way_kind({"highway": "residential"}) == "road"
    assert way_kind({"highway": "service", "service": "driveway"}) == "driveway"
    assert way_kind({"highway": "service", "access": "private"}) == "driveway"
    assert way_kind({"highway": "service", "service": "parking_aisle"}) == ""
    assert way_kind({"highway": "track", "surface": "gravel"}) == ""
    assert way_kind({"highway": "secondary", "junction": "roundabout"}) == "ring"
    assert way_kind({"highway": "footway"}) == ""
    # OSM width wins, then lanes x the North American lane width, then the Ontario default.
    assert side_road_width({"highway": "tertiary", "width": "8"}) == 8.0
    assert side_road_width({"highway": "secondary", "lanes": "2"}) == pytest.approx(6.8)
    assert side_road_width({"highway": "tertiary"}) == pytest.approx(6.7)
    assert side_road_width({"highway": "tertiary", "oneway": "yes"}) == pytest.approx(4.8)
    # Corner radius: that of the lower class at the corner.
    assert corner_radius("primary", "residential") == CORNER_RADIUS_M["residential"]
    assert corner_radius("service", "primary") == CORNER_RADIUS_M["service"]


def test_triangulate_covers_polygons_and_shares_edges():
    a = shapely.Polygon([(0, 0), (10.3, 0), (10.3, 7.1), (0, 7.1)])
    b = shapely.Point(10.3, 3.5).buffer(4.0).difference(a)  # shares a's right edge
    for poly in (a, b):
        v, t = triangulate(poly, 3.0)
        e1, e2 = v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]
        area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()
        assert area == pytest.approx(poly.area, rel=1e-6)
    # Vertices on the shared edge are the same on both sides (no T-junction cracks).
    va, _ = triangulate(a, 3.0)
    vb, _ = triangulate(b, 3.0)
    on_a = {tuple(np.round(p, 6)) for p in va if abs(p[0] - 10.3) < 1e-9 and 0.0 < p[1] < 7.1}
    on_b = {tuple(np.round(p, 6)) for p in vb if abs(p[0] - 10.3) < 1e-9 and 0.0 < p[1] < 7.1}
    assert on_a == on_b and len(on_a) >= 4


def test_mesh_polygon_faces_up_on_the_height_field():
    poly = shapely.Polygon([(0, 0), (12, 0), (12, 6), (0, 6)])
    mesh = mesh_polygon(poly, lambda xz: 0.1 * np.asarray(xz)[:, 0], cell_m=3.0, lift=0.02)
    p = mesh.positions
    assert np.allclose(p[:, 1], 0.1 * p[:, 0] + 0.02)
    tri = mesh.triangles
    n = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    assert (n[:, 1] > 0).all()


def test_height_field_meets_the_route_and_blends_out():
    route = CentreLine(np.array([[-50.0, 0.0], [50.0, 0.0]]), np.array([2.0, 2.0]), 4.5)
    side = CentreLine(np.array([[0.0, 0.0], [0.0, 60.0]]), np.array([2.0, 5.0]), 3.0)
    h = HeightField([route], [side])
    # On the route's footprint (asphalt + shoulder): exactly the route's height.
    assert h(np.array([[10.0, 4.4], [-30.0, -4.0]])) == pytest.approx([2.0, 2.0])
    # Far out on the side road: its own profile.
    assert h(np.array([[0.0, 60.0]]))[0] == pytest.approx(5.0, abs=1e-6)
    # Continuous in between (no step at the blend's edge).
    zs = np.linspace(4.5, 12.0, 200)
    y = h(np.stack([np.zeros_like(zs), zs], axis=1))
    assert np.abs(np.diff(y)).max() < 0.05
    # A roundabout plane rules inside its disc.
    z = PlaneZone(np.array([200.0, 0.0]), 25.0, 7.0, np.array([0.02, 0.0]))
    hp = HeightField([], [], [z])
    assert hp(np.array([[210.0, 5.0]]))[0] == pytest.approx(7.2)


# --------------------------------------------------------------------------------------------
# A T-junction: a residential street off a tertiary route
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def t_junction():
    model = _model(np.array([[-200.0, 0.0], [200.0, 0.0]]))
    ways = [
        NetWay(
            1,
            (1, 2, 3, 4, 5),
            np.array([[-200, 0], [-100, 0], [0, 0], [100, 0], [200, 0.0]]),
            {"highway": "tertiary", "lanes": "2"},
        ),  # fmt: skip
        NetWay(
            2,
            (3, 10, 11),
            np.array([[0, 0], [0, 60], [0, 110.0]]),
            {"highway": "residential", "lanes": "2"},
        ),  # fmt: skip
    ]
    net = build_network(
        model,
        ways,
        [],
        route_nodes={1, 2, 3, 4, 5},
        route_segments={(1, 2), (2, 3), (3, 4), (4, 5)},
        route_highway=["tertiary"] * len(model),
        dtm=lambda xz: 0.03 * np.asarray(xz).reshape(-1, 2)[:, 1],
        options=OPTS,
    )
    return model, net


def test_side_road_is_built_with_its_own_profile(t_junction):
    model, net = t_junction
    assert len(net.side_roads) == 1
    sr = net.side_roads[0]
    assert sr.start == ("route", 3) and sr.end[0] == "end"
    assert sr.length == pytest.approx(110.0, abs=1.0)
    # Pinned to the route at the junction, then climbing the 3 % DTM slope.
    assert sr.model.centre[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert sr.model.centre[-1, 1] == pytest.approx(3.3, abs=0.2)
    assert sr.model.half_width[50] == pytest.approx(3.0)  # 2 lanes x 3.0 m (residential)
    assert net.stats["side_road_km"] == pytest.approx(0.11, abs=0.002)


def test_junction_mouth_fillets_and_route_masks(t_junction):
    model, net = t_junction
    (j,) = net.junctions
    assert j.radius == CORNER_RADIUS_M["residential"]
    # Two corner fillets of radius R: area 2 (R^2 - pi R^2 / 4).
    r = j.radius
    assert j.fillets.area == pytest.approx(2 * (r * r - math.pi * r * r / 4), rel=0.08)
    # The route's right shoulder (z > 0 is its right: it runs east) is gone across the mouth,
    # the left one stays; its right edge line too. Nothing is hidden.
    i = model.index_at(200.0)
    assert not net.masks.shoulder_right[i] and net.masks.shoulder_left[i]
    assert not net.masks.edge_right[i] and net.masks.edge_left[i]
    assert net.masks.shoulder_right[model.index_at(150.0)]
    assert not net.masks.hidden.any()
    # The mouth is paved: the fillets and the removed shoulder are in the network's area.
    assert net.paved.contains(shapely.Point(4.0, 5.0))


def test_stop_bar_and_sign_on_the_minor_approach(t_junction):
    _, net = t_junction
    stops = [it for it in net.planned_items if it.kind == SIGN_STOP]
    assert len(stops) == 1  # the route (tertiary) has the right of way
    (stop,) = stops
    x, z = stop.xz
    # Drivers come up the street northwards (-z): the sign is on their right (east, +x),
    # beyond the asphalt edge, near the mouth, and faces them (south, +z).
    assert x > 3.0 + 2.0 and 5.0 < z < 20.0
    assert stop.facing == pytest.approx((0.0, 1.0), abs=1e-3)
    paint = [p for p in net.pieces if p.material == "paint_white"]
    assert paint, "a stop bar"


def test_network_footprint_and_terrain_hole(t_junction):
    model, net = t_junction
    roads = build_road_meshes(model, keep=net.masks)
    st, outer = net.terrain_stations()
    extra = TerrainExtra(net.footprint, net.edge_points, st, outer, net.masks.hidden)
    opts = TerrainOptions(rings=((25.0, 2.0), (150.0, 6.0)), near_m=150.0)
    ter = build_terrain(
        model, roads.left_outer, roads.right_outer, lambda p, far: flat(p), opts, extra
    )
    cent = ter.points[ter.triangles].mean(axis=1)
    inside = shapely.contains_xy(net.footprint.buffer(-0.05), cent[:, 0], cent[:, 1])
    assert not inside.any()
    # The network's edge points are in the terrain (it joins the side road exactly).
    pts = {tuple(np.round(p, 4)) for p in ter.points[ter.is_edge]}
    assert sum(tuple(np.round(p[[0, 2]], 4)) in pts for p in net.edge_points) > 0.9 * len(
        net.edge_points
    )


def test_wall_corridor_includes_the_side_road(t_junction):
    _, net = t_junction
    corridor = net.corridor(4.0)
    assert corridor.contains(shapely.Point(0.0, 100.0))
    assert not corridor.contains(shapely.Point(0.0, 120.0))  # a wall closes its far end


# --------------------------------------------------------------------------------------------
# A roundabout with four two-way legs
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def roundabout():
    r = 18.0
    ang = np.radians(np.arange(0, 360, 22.5))  # 16 ring nodes, CCW seen from above (north up)
    ring = np.stack([r * np.cos(ang), -r * np.sin(ang)], axis=1)  # AC: z = -north
    ring_ids = list(range(100, 116))
    west, south, east, north = 108, 112, 100, 104  # 180, 270, 0, 90 degrees
    path = [108, 109, 110, 111, 112, 113, 114, 115, 100]  # west -> south -> east (CCW)
    ways = [
        NetWay(1, (1, west), np.array([[-160.0, 0.0], ring[8]]), {"highway": "secondary"}),
        NetWay(2, (east, 2), np.array([ring[0], [160.0, 0.0]]), {"highway": "secondary"}),
        NetWay(3, (north, 3), np.array([ring[4], [0.0, -140.0]]), {"highway": "tertiary"}),
        NetWay(4, (south, 4), np.array([ring[12], [0.0, 140.0]]), {"highway": "tertiary"}),
        NetWay(
            9,
            (*ring_ids, ring_ids[0]),
            np.vstack([ring, ring[:1]]),
            {"highway": "secondary", "junction": "roundabout"},
        ),  # fmt: skip
    ]
    arc = np.array([ring[k - 100] for k in path])
    xz = np.vstack([[[-160.0, 0.0]], arc, [[160.0, 0.0]]])
    model = _model(xz, hw=3.5, y=lambda p: 0.01 * p[:, 0])
    route_nodes = {1, 2, *path}
    segs = {
        tuple(sorted(p)) for p in [(1, west), (east, 2), *zip(path[:-1], path[1:], strict=True)]
    }
    net = build_network(
        model,
        ways,
        [],
        route_nodes=route_nodes,
        route_segments=segs,
        route_highway=["secondary"] * len(model),
        dtm=lambda p: 0.01 * np.asarray(p).reshape(-1, 2)[:, 0],
        options=OPTS,
    )
    return model, net


def test_roundabout_is_found_and_the_route_settles_on_its_plane(roundabout):
    model, net = roundabout
    (rb,) = net.roundabouts
    assert rb.radius == pytest.approx(18.0, abs=0.1)
    assert np.allclose(rb.centre, 0.0, atol=0.1)
    assert rb.island_r < rb.inner_r < rb.radius < rb.outer_r < rb.zone_r
    # The route's ribbon is replaced inside the zone, and sits on the plane there.
    m = net.model
    assert net.masks.hidden.any()
    inside = np.linalg.norm(m.centre[:, [0, 2]] - rb.centre, axis=1) < rb.zone_r - 1.0
    assert np.allclose(m.centre[inside, 1], rb.zone.height(m.centre[inside][:, [0, 2]]), atol=1e-6)


def test_roundabout_islands_signs_and_markings(roundabout):
    _, net = roundabout
    # Central island (with its apron) + one teardrop splitter per two-way leg.
    assert len(net.islands) == 5
    kinds = [it.kind for it in net.planned_items]
    assert kinds.count(SIGN_YIELD) >= 4
    assert kinds.count(SIGN_CHEVRON_RIGHT) == 4
    assert kinds.count(SIGN_KEEP_RIGHT) == 4
    assert kinds.count(SIGN_ROUNDABOUT_AHEAD) >= 2
    # Chevron boards stand on the central island and face out towards their entry.
    (rb,) = net.roundabouts
    for it in net.planned_items:
        if it.kind == SIGN_CHEVRON_RIGHT:
            v = np.asarray(it.xz) - rb.centre
            assert np.linalg.norm(v) < rb.island_r
            assert float(np.dot(v / np.linalg.norm(v), it.facing)) > 0.99
    names = {p.name[:9] for p in net.pieces}
    assert {"1ROAD_apr", "1GRASS_is", "1ROAD_cur", "paint_whi"} <= names
    # Hidden walls inside every island.
    assert len(net.island_walls) == 5


# --------------------------------------------------------------------------------------------
# Driveways
# --------------------------------------------------------------------------------------------


def test_inferred_rural_driveway_with_mailbox(t_junction):
    model, net = t_junction
    house = shapely.Polygon([(60, -40), (75, -40), (75, -30), (60, -30)])  # north of the road
    res = build_driveways(net, [(house, "house")], flat, OPTS)
    assert res.stats["driveways_inferred"] == 1
    (d,) = res.driveways
    assert d.rural and d.road == -1
    assert d.normal == pytest.approx((0.0, -1.0), abs=1e-6)  # away from the road, north
    assert d.polygon.distance(house) < 0.1 and not d.polygon.intersects(house.buffer(-0.1))
    (mb,) = res.items
    assert mb.kind == MAILBOX
    # Right of the mouth seen from the road looking up the driveway (north): east.
    assert mb.xz[0] > d.mouth[0] + 2.0
    assert mb.facing == pytest.approx((0.0, 1.0), abs=1e-6)  # facing the road
    i = model.index_at(200.0 + float(d.mouth[0]))
    assert not net.masks.edge_left[i]  # the edge line is cut across the mouth
    assert res.masks_changed
    # A house that already has a driveway gets no second one.
    again = build_driveways(net, [(house, "house"), (house.buffer(-2), "house")], flat, OPTS)
    assert again.stats["driveways_inferred"] == 1


# --------------------------------------------------------------------------------------------
# Run-up and run-off on the real roads
# --------------------------------------------------------------------------------------------


def test_route_continuations_follow_the_road_and_turn_at_a_t():
    proj = LocalProjection(43.0, -80.0)
    xy = {
        1: (0.0, 0.0), 2: (100.0, 0.0), 3: (200.0, 0.0),  # the route, eastwards
        4: (300.0, 5.0), 5: (400.0, 10.0), 6: (300.0, 100.0),  # beyond B: on, or a side road
        7: (0.0, 80.0), 8: (0.0, -80.0),  # behind A: a T, bigger road to the north
    }  # fmt: skip

    def ll(*ids):
        return proj.from_xy(np.array([xy[i] for i in ids]))

    class W:
        def __init__(self, wid, nodes, tags):
            self.id, self.nodes, self.tags = wid, nodes, tags
            self.lonlat = ll(*nodes)

    ways = [
        W(1, (1, 2, 3, 4, 5), {"highway": "tertiary", "name": "Main"}),
        W(2, (4, 6), {"highway": "secondary", "name": "Side"}),
        W(3, (7, 1), {"highway": "secondary", "name": "North"}),
        W(4, (1, 8), {"highway": "residential", "name": "South"}),
    ]
    nodes = {i: tuple(ll(i)[0]) for i in xy}
    pin = SnappedPin(43.0, -80.0, 43.0, -80.0, 0.0, 1, 1, 2, 0.0)
    route = Route(
        origin=(43.0, -80.0),
        pins=(pin, pin),
        edges=(RouteEdge(1, 1, 2, 1), RouteEdge(1, 2, 3, 1)),
        start_t=0.0,
        end_t=1.0,
        length_m=200.0,
        nodes=nodes,
        ways={1: {"highway": "tertiary", "name": "Main"}},
    )
    before, after = route_continuations(route, ways, 70.0, 150.0)
    assert after.nodes[:3] == (3, 4, 5)  # straight on along the same road, not the side road
    assert after.length_m == pytest.approx(150.0, abs=0.5)
    assert before.nodes[:2] == (1, 7)  # the T: onto the bigger road
    assert before.length_m == pytest.approx(70.0, abs=0.5)


# --------------------------------------------------------------------------------------------
# Walls
# --------------------------------------------------------------------------------------------


def test_wall_lines_thin_out_away_from_the_route():
    corridor = shapely.box(0, 0, 200, 20)
    fine = shapely.box(-1, -1, 50, 21)
    plain = wall_lines(corridor)
    thin = wall_lines(corridor, fine_zone=fine)
    assert len(thin.rings[0]) < 0.7 * len(plain.rings[0])
    assert thin.length == pytest.approx(plain.length, rel=1e-9)
    ring = thin.rings[0]
    assert np.allclose(ring[0], ring[-1])
    gaps = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    near = (ring[:-1, 0] < 48.0) & (ring[1:, 0] < 48.0)
    assert gaps[near].max() <= 2.0 + 1e-6 and gaps.max() <= 4.0 + 1e-6


def test_turning_circle_bulb_and_stop_from_osm_nodes():
    model = _model(np.array([[-150.0, 0.0], [150.0, 0.0]]))
    ways = [
        NetWay(1, (1, 2, 3), np.array([[-150, 0], [0, 0], [150, 0.0]]), {"highway": "tertiary"}),
        NetWay(2, (2, 10), np.array([[0, 0], [0, -80.0]]), {"highway": "residential"}),
    ]
    nodes = [
        NetNode(10, np.array([0.0, -80.0]), {"highway": "turning_circle"}),
        NetNode(11, np.array([0.0, 12.0]), {"highway": "stop"}),  # wrong side: no effect
    ]
    net = build_network(
        model, ways, nodes, route_nodes={1, 2, 3}, route_segments={(1, 2), (2, 3)},
        route_highway=["tertiary"] * len(model), dtm=flat, options=OPTS,
    )  # fmt: skip
    assert net.stats["turning_circles"] == 1
    # The bulb: a 12 m disc at the dead end is paved and drivable.
    assert net.paved.contains(shapely.Point(10.0, -80.0))
    assert net.corridor(4.0).contains(shapely.Point(0.0, -91.0))
