"""Street lights along urban roads: spacing by class, one or both sides, none in the country,
off junction corners and driveways, measured lights taking the place of rule-made ones."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely

from driftmaps.geo.mapillary import MapFeature
from driftmaps.geo.project import LocalProjection
from driftmaps.geometry import furniture as fu
from driftmaps.geometry.clearance import DrivableSurface, _Road
from driftmaps.geometry.road import RoadModel
from driftmaps.geometry.street_items import LIGHT_STREET

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
    return PROJ.from_xy(np.array([xz[0], -xz[1]]))


def ground(xz):
    return np.zeros(len(np.asarray(xz).reshape(-1, 2)))


def _model(half: float) -> RoadModel:
    x = np.arange(0.0, 1001.0)
    c = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    return RoadModel(x.copy(), c, half, 0.5, True, False)


def _lights(tags, half=4.0, stubs=(), nodes=None, **kw):
    m = _model(half)
    nodes = nodes or {1: (0, 0), 2: (1000, 0)}
    edges = [Edge(11, 1, 2)] if len(nodes) == 2 else [Edge(11, 1, 2), Edge(11, 2, 3)]
    route = FakeRoute(nodes, edges, {11: tags, 13: {"highway": "residential"}}, stubs)
    plan = fu.plan_furniture(m, PROJ, route, ground, way_id=np.full(len(m), 11), **kw)
    lights = [it for it in plan.items if it.kind == LIGHT_STREET]
    return lights, plan


def _stations(lights, side=None):
    xz = np.array([(it.position[0], it.position[2]) for it in lights]).reshape(-1, 2)
    keep = np.ones(len(xz), bool) if side is None else np.sign(-xz[:, 1]) == side
    return np.sort(xz[keep, 0])


def test_lit_residential_street_one_side_every_50_m():
    lights, _ = _lights({"highway": "residential", "lit": "yes"})
    s = _stations(lights)
    assert len(s) >= 18
    assert np.allclose(np.diff(s), 50.0, atol=5.1)
    z = {np.sign(it.position[2]) for it in lights}
    assert len(z) == 1  # one side
    assert all(abs(it.position[2]) >= 4.5 + 1.0 - 1e-6 for it in lights)  # 1 m behind the edge


def test_wide_arterial_lights_on_both_sides_every_40_m():
    lights, _ = _lights({"highway": "primary", "lit": "yes"}, half=7.5)
    for side in (-1, 1):
        s = _stations(lights, side)
        assert len(s) >= 23 and np.allclose(np.diff(s), 40.0, atol=5.1)


def test_no_rule_lights_in_the_country_or_where_lit_is_no():
    assert not _lights({"highway": "secondary", "maxspeed": "80"})[0]
    assert not _lights({"highway": "residential", "lit": "no"}, town_at=lambda xz: True)[0]
    # In a town without tags: lit.
    assert _lights({"highway": "residential"}, town_at=lambda xz: True)[0]
    assert _lights({"highway": "residential", "sidewalk": "both"})[0]


def test_lights_keep_off_junction_corners_and_driveways():
    stubs = [(2, 13, 4), (2, 13, 5)]
    nodes = {1: (0, 0), 2: (500, 0), 3: (1000, 0), 4: (500, -100), 5: (500, 100)}
    lights, _ = _lights({"highway": "residential", "lit": "yes"}, stubs=stubs, nodes=nodes)
    s = _stations(lights)
    assert not np.any(np.abs(s - 500.0) < 3.5 + 10.0)  # the side street's half + 10 m
    # A driveway where a light would stand: it moves along the road, off the driveway.
    m = _model(4.0)
    drive = shapely.box(98.0, 4.0, 102.0, 20.0)
    paved = shapely.union(shapely.box(0, -4.5, 1000, 4.5), drive)
    surf = DrivableSurface(paved, paved, [], [_Road(m.centre[:, [0, 2]], np.zeros(len(m)))])
    route = FakeRoute({1: (0, 0), 2: (1000, 0)}, [Edge(11, 1, 2)],
                      {11: {"highway": "residential", "lit": "yes"}})  # fmt: skip
    plan = fu.plan_furniture(m, PROJ, route, ground, way_id=np.full(len(m), 11), surface=surf)
    for it in plan.items:
        if it.kind == LIGHT_STREET:
            assert surf.distance(np.array([[it.position[0], it.position[2]]]))[0] >= 0.5


def test_a_photographed_light_takes_the_place_of_a_rule_one():
    lon, lat = _lonlat((322.0, 6.0))  # right of the road (+Z is the right of +X travel)
    mly = [MapFeature("l1", "object--street-light--g1", float(lon), float(lat), None,
                      "2023-06-01T00:00:00Z", "2023-06-01T00:00:00Z")]  # fmt: skip
    lights, _ = _lights({"highway": "residential", "lit": "yes"}, mapillary=mly)
    s = _stations(lights)
    assert np.sum(np.abs(s - 322.0) < 30.0) == 1  # the measured one, no rule light beside it
    assert any(it.source == "mapillary" for it in lights)
