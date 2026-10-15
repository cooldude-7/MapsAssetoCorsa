"""Offline build of a LOOP (a circuit) from a synthetic street block (drift-maps loops).

The block sits inside the Forks of the Credit DEM test window (tests/fixtures/dem), so the
heights are real lidar ground; the roads are made up: a "D" shaped block (a straight south
side, a curved east side, a straight north side and a west side) with a cross street. The
route goes A (south side, heading east) -> B (east curve) -> C (north side) and back to A,
anticlockwise. The written folder is re-read and checked for what makes it a circuit: no
seam where the lap starts, walls outside and inside the ring and never across the road, the
start/finish and sector gates across the road, grid/pits/hotlap on the road behind the line,
a closed AI line on the right-hand lane, and the "circuit" tag with the right direction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import shapely
from scipy.spatial import cKDTree

from driftmaps.ac.ai_line import read_ai
from driftmaps.ac.coords import left_of
from driftmaps.ac.kn5 import read_kn5, validate
from driftmaps.ac.model import Dummy, Kn5, Mesh
from driftmaps.build import (
    HOTLAP_BEFORE_M,
    BuildOptions,
    BuildResult,
    build_track,
    lane_offset,
    loop_direction,
)
from driftmaps.elevation.sources import CogSource, SourceInfo
from driftmaps.geo.overpass import OsmData, Way
from driftmaps.geo.project import LocalProjection
from driftmaps.geo.router import Route, route_from_pins
from driftmaps.geometry.terrain import TerrainOptions

FIXTURES = Path(__file__).parent / "fixtures"
TRACK_ID = "dm_loop_test"
TEST_TERRAIN = TerrainOptions(rings=((15.0, 3.0), (60.0, 8.0), (150.0, 20.0)), near_m=60.0)
ORIGIN = (43.8075, -79.9870)  # inside the hrdem_forks_5m.tif window
PROJ = LocalProjection(*ORIGIN)


def block_osm() -> OsmData:
    """A 'D' block: south side y = -130 from x = -250 to 150, an arc of radius 130 round
    (150, 0) on the east, the north side back to x = -250 and the west side; a cross street
    at x = -50 joins the south and north sides."""
    nodes: dict[int, tuple[float, float]] = {}
    south = [1, 2, 3, 4, 5]
    for i, x in zip(south, (-250, -150, -50, 50, 150), strict=True):
        nodes[i] = (x, -130)
    arc = list(range(10, 27))
    for i, a in zip(arc, np.linspace(-90, 90, 19)[1:-1], strict=True):
        nodes[i] = (150 + 130 * np.cos(np.radians(a)), 130 * np.sin(np.radians(a)))
    north = [30, 31, 32, 33, 34]
    for i, x in zip(north, (150, 50, -50, -150, -250), strict=True):
        nodes[i] = (x, 130)
    nodes[40] = (-250, 0)
    nodes[41] = (-50, 0)
    ways = [
        (1001, [*south, *arc, 30], {"name": "Loop Road"}),
        (1002, north, {"name": "Top Street"}),
        (1003, [34, 40, 1], {"name": "West Street"}),
        (1004, [3, 41, 32], {"name": "Cross Street"}),
    ]
    out = []
    for wid, nids, tags in ways:
        xy = np.array([nodes[n] for n in nids], dtype=float)
        lon, lat = PROJ.to_lonlat(xy[:, 0], xy[:, 1])
        tags = {"highway": "residential", "surface": "asphalt", **tags}
        out.append(Way(wid, tuple(nids), np.stack([lon, lat], axis=-1), tags))
    return OsmData(tuple(out), "2026-01-01T00:00:00Z", "loopkey")


def pin(east: float, north: float) -> tuple[float, float]:
    lon, lat = PROJ.to_lonlat(east, north)
    return float(lat), float(lon)


LOOP_PINS = [pin(-150, -130), pin(280, 0), pin(-150, 130)]


def fixture_options(cache_dir: Path, **kw) -> BuildOptions:
    info = SourceInfo(
        "hrdem-fixture", "NRCan HRDEM 1 m mosaic (5 m test window)", 5.0, "DTM",
        "OGL-Canada-2.0", "Contains information licensed under the Open Government Licence - "
        "Canada. Source: Natural Resources Canada.",
    )  # fmt: skip
    dem = {"hrdem-fixture": CogSource(info, str(FIXTURES / "dem" / "hrdem_forks_5m.tif"))}
    args = dict(
        name="Loop Test",
        track_id=TRACK_ID,
        cache_dir=cache_dir,
        elevation_sources=dem,
        elevation_chain=["hrdem-fixture"],
        terrain=TEST_TERRAIN,
        debug_png=True,
        trees=False,
        buildings=False,
        osm_barriers=(),
        offline=True,
    )
    return BuildOptions(**(args | kw))


@pytest.fixture(scope="module")
def loop_route() -> Route:
    return route_from_pins(LOOP_PINS, osm=block_osm(), loop=True)


@pytest.fixture(scope="module")
def built(tmp_path_factory, loop_route) -> BuildResult:
    root = tmp_path_factory.mktemp("loop")
    return build_track(loop_route, root / "out", fixture_options(root / "cache"))


@pytest.fixture(scope="module")
def kn5(built) -> Kn5:
    return read_kn5(built.track_dir / f"{TRACK_ID}.kn5")


def meshes(kn5: Kn5, pattern: str) -> list[Mesh]:
    return [n for n in kn5.root.children if isinstance(n, Mesh) and re.match(pattern, n.name)]


def dummies(kn5: Kn5) -> dict[str, Dummy]:
    return {n.name: n for n in kn5.root.children if isinstance(n, Dummy)}


def test_route_is_the_block(loop_route):
    assert loop_route.closed
    perimeter = 400 + np.pi * 130 + 400 + 260
    assert loop_route.length_m == pytest.approx(perimeter, rel=0.01)
    assert {e.way for e in loop_route.edges} == {1001, 1002, 1003}


def test_folder_and_kn5(built, kn5):
    validate(kn5)
    for rel in ("ai/fast_lane.ai", "data/ideal_line.ai", "ui/ui_track.json", "map.png",
                "data/map.ini", "route.json", "LICENSE-DATA.txt"):  # fmt: skip
        assert (built.track_dir / rel).is_file(), rel
    assert built.stats["closed"] is True
    lap = built.stats["length_m"]
    assert lap == pytest.approx(built.geometry.route.length_m, rel=0.03)
    # No run-up or run-off on a loop: the road is exactly one lap.
    assert built.stats["road_length_m"] == pytest.approx(lap)
    assert "lap" in built.summary()
    assert Route.load(built.track_dir / "route.json").closed
    assert (built.debug_images[0]).is_file()


def test_centreline_has_no_seam(built):
    m = built.geometry.model
    assert m.closed
    np.testing.assert_array_equal(m.centre[0], m.centre[-1])
    np.testing.assert_array_equal(m.forward[0], m.forward[-1])
    step = np.linalg.norm(np.diff(m.centre[:, [0, 2]], axis=0), axis=1)
    assert step.max() < 1.2 and step.min() > 0.8  # even 1 m stations, also across the join
    # Heading changes smoothly across the join like anywhere else.
    heading = np.unwrap(np.arctan2(m.forward[:, 2], m.forward[:, 0]))
    turn = np.abs(np.diff(heading))
    join = max(turn[0], turn[-1])
    assert join < 0.05 and join <= 3 * np.percentile(turn, 99) + 1e-3
    # Height and grade carry on across the join.
    y = m.centre[:, 1]
    grade = np.diff(y) / np.diff(m.s)
    assert abs(grade[0] - grade[-1]) < 0.01
    # One full turn anticlockwise (seen from above), as driven.
    assert heading[-1] - heading[0] == pytest.approx(-2 * np.pi, abs=0.05)  # Z points south


def test_road_meshes_join_without_a_gap(kn5, built):
    m = built.geometry.model
    road = meshes(kn5, r"1ROAD_c")
    pts = np.concatenate([r.positions.astype(np.float64) for r in road])
    # Asphalt vertices exist exactly on the cross-section at the join, from both chunks.
    first, last = road[0].positions.astype(np.float64), road[-1].positions.astype(np.float64)
    hw = m.half_width[0]
    for side in (1.0, -1.0, 0.0):
        p = m.centre[0] + side * hw * m.left[0]
        assert np.linalg.norm(first - p, axis=1).min() < 1e-3
        assert np.linalg.norm(last - p, axis=1).min() < 1e-3
    # And the asphalt covers the road all round: no hole around the start.
    tri = np.concatenate(
        [r.positions.astype(np.float64)[r.indices.astype(np.int64).reshape(-1, 3)] for r in road]
    )
    asphalt = shapely.union_all(shapely.polygons(tri[:, :, [0, 2]]))
    probe = m.centre[np.r_[0:20, -20:0]]
    assert shapely.contains_xy(asphalt, probe[:, 0], probe[:, 2]).all()
    assert len(pts)


def test_walls_ring_outside_and_inside_never_across_the_road(kn5, built):
    m = built.geometry.model
    walls = meshes(kn5, r"1WALL_c")
    assert walls and all(not w.is_renderable for w in walls)
    pts = np.concatenate([w.positions.astype(np.float64) for w in walls])[:, [0, 2]]
    d, i = cKDTree(m.centre[:, [0, 2]]).query(pts)
    assert (d >= m.outer[i] + 0.5).all(), "a wall stands on the road"
    loop = shapely.Polygon(m.centre[:, [0, 2]])
    inside = shapely.contains_xy(loop, pts[:, 0], pts[:, 1])
    assert inside.any() and (~inside).any()  # an inner ring and an outer ring
    assert built.stats["wall_length_m"] > 2 * built.stats["length_m"] - 200


def test_lap_gates(kn5, built):
    m = built.geometry.model
    dm = dummies(kn5)
    assert not any(n.startswith("AC_AB_") for n in dm)
    lap = built.stats["length_m"]
    for k, s in enumerate((0.0, lap / 3, 2 * lap / 3)):
        left = dm[f"AC_TIME_{k}_L"].transform[3, :3].astype(np.float64)
        right = dm[f"AC_TIME_{k}_R"].transform[3, :3].astype(np.float64)
        pos, fwd, lft = m.at(s)
        mid = 0.5 * (left + right)
        assert np.linalg.norm((mid - pos)[[0, 2]]) < 0.2
        assert np.dot(left - right, lft) > 2 * m.interp(m.half_width, s)  # _L on the left
    assert "AC_TIME_3_L" not in dm


def test_grid_pits_and_hotlap_on_the_road_behind_the_line(kn5, built):
    m = built.geometry.model
    lap = built.stats["length_m"]
    dm = dummies(kn5)
    tree = cKDTree(m.centre[:, [0, 2]])
    names = [n for n in dm if n.startswith(("AC_START_", "AC_PIT_", "AC_HOTLAP_START_"))]
    assert len([n for n in names if n.startswith("AC_PIT_")]) == 8
    for name in names:
        pos = dm[name].transform[3, :3].astype(np.float64)
        d, i = tree.query(pos[[0, 2]])
        assert d < m.half_width[i], f"{name} is off the asphalt"
        s = m.s[i]
        behind = lap - s
        if name.startswith("AC_START_"):
            assert 3.0 < behind < 40.0, name
        elif name.startswith("AC_PIT_"):
            assert 30.0 < behind < 70.0, name
        else:
            assert behind == pytest.approx(HOTLAP_BEFORE_M, abs=2.0)
        fwd = dm[name].transform[2, :3].astype(np.float64)
        assert np.dot(fwd, m.forward[i]) > 0.95, f"{name} faces the wrong way"
    # The first car of each row is on the driving (right-hand) lane.
    pos0 = dm["AC_START_0"].transform[3, :3].astype(np.float64)
    pos1 = dm["AC_START_1"].transform[3, :3].astype(np.float64)
    i0 = tree.query(pos0[[0, 2]])[1]
    assert np.dot(pos0 - m.centre[i0], m.left[i0]) < 0 < np.dot(pos1 - m.centre[i0], m.left[i0])


def test_ai_line_closed_on_the_driving_lane(built):
    ai = read_ai(built.track_dir / "ai" / "fast_lane.ai")
    assert ai.is_closed(max_gap=2.0)
    m = built.geometry.model
    lap = built.stats["length_m"]
    assert ai.lengths[-1] == pytest.approx(lap, rel=0.02)
    pos = ai.positions.astype(np.float64)
    assert np.linalg.norm((pos[0] - m.centre[0])[[0, 2]]) < m.half_width[0]  # starts at A
    d, i = cKDTree(m.centre[:, [0, 2]]).query(pos[:, [0, 2]])
    lateral = np.einsum("ij,ij->i", pos - m.centre[i], m.left[i])
    target = lane_offset(built.geometry)[i]
    assert np.all(lateral < 0.0)  # right-hand traffic: always right of the centre line
    assert np.abs(lateral - target).max() < 0.5
    assert np.isfinite(ai.speed).all() and ai.speed.min() > 1.0  # never stops on a loop
    # Driven forward: each point's forward follows the road.
    assert np.all(np.einsum("ij,ij->i", ai.forward.astype(np.float64), m.forward[i]) > 0.9)


def test_ui_track_is_a_circuit_with_its_direction(built):
    ui = json.loads((built.track_dir / "ui" / "ui_track.json").read_text())
    assert "circuit" in ui["tags"] and "A2B" not in ui["tags"]
    assert ui["run"] == "anticlockwise" == built.stats["run"]
    assert "loop" in ui["description"]
    assert ui["length"] == str(round(built.stats["length_m"]))


def test_reverse_loop_runs_clockwise(loop_route, tmp_path):
    rev = loop_route.reverse()
    assert rev.closed and rev.pins[0] == loop_route.pins[0]
    res = build_track(rev, tmp_path / "out", fixture_options(tmp_path / "cache", debug_png=False))
    assert res.stats["run"] == "clockwise"
    left_of_road = lane_offset(res.geometry)
    assert left_of_road.max() < 0.0  # still the right-hand lane


def test_loop_direction_helper():
    square = np.array([[0, 0, 0], [10, 0, 0], [10, 0, -10], [0, 0, -10], [0, 0, 0]], float)
    assert loop_direction(square) == "anticlockwise"  # east, then north (-Z): anticlockwise
    assert loop_direction(square[::-1]) == "clockwise"


def test_too_short_loop_is_refused(tmp_path):
    nodes = [(0, 0), (40, 0), (40, 30), (0, 30), (0, 0)]
    xy = np.array(nodes, float)
    lon, lat = PROJ.to_lonlat(xy[:, 0], xy[:, 1])
    way = Way(1, (1, 2, 3, 4, 1), np.stack([lon, lat], -1), {"highway": "residential"})
    osm = OsmData((way,), "2026-01-01T00:00:00Z", "tiny")
    route = route_from_pins([pin(20, 0), pin(40, 15)], osm=osm, loop=True)
    with pytest.raises(ValueError, match="loop is only"):
        build_track(route, tmp_path / "out", fixture_options(tmp_path / "cache"))


def test_left_of_matches_the_model(built):
    m = built.geometry.model
    np.testing.assert_allclose(left_of(m.forward[5]), m.left[5], atol=1e-6)


def test_cli_route_loop(tmp_path):
    from typer.testing import CliRunner

    from driftmaps import cli

    osm = block_osm()
    payload = {
        "elements": [
            {
                "type": "way",
                "id": w.id,
                "nodes": list(w.nodes),
                "geometry": [{"lat": float(lat), "lon": float(lon)} for lon, lat in w.lonlat],
                "tags": dict(w.tags),
            }
            for w in osm.ways
        ]
    }
    osm_json = tmp_path / "block.json"
    osm_json.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "route.json"
    (a_lat, a_lon), (b_lat, b_lon), (c_lat, c_lon) = LOOP_PINS
    res = CliRunner().invoke(
        cli.app,
        ["route", "--start", f"{a_lat},{a_lon}", "--via", f"{b_lat},{b_lon}",
         "--end", f"{c_lat},{c_lon}", "--loop", "--osm-json", str(osm_json), "--out", str(out)],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    route = Route.load(out)
    assert route.closed and route.length_m == pytest.approx(400 + np.pi * 130 + 660, rel=0.01)
