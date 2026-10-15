"""Offline end-to-end build of a real road (drift-maps M1).

Route: the Forks of the Credit Road OSM fixture (Caledon, Ontario; tests/fixtures/osm), first
leg (1.36 km, with a bridge and tight bends), routed offline. Elevation: a 5 m window of
NRCan HRDEM around it (tests/fixtures/dem/hrdem_forks_5m.tif) as the only DEM source. The
terrain rings are reduced so the build stays fast. The written folder is re-read with our own
KN5 and AI readers and checked for the M1 invariants: no fold-over, road above terrain, walls
never crossing the road, gates the right way round, the AI line on the driving lane, and
byte-identical rebuilds.
"""

from __future__ import annotations

import configparser
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import shapely
from scipy.spatial import cKDTree

from driftmaps.ac.ai_line import read_ai
from driftmaps.ac.coords import left_of
from driftmaps.ac.kn5 import read_kn5, validate
from driftmaps.ac.model import MAX_MESH_VERTICES, Dummy, Kn5, Mesh
from driftmaps.ac.track_files import read_map_ini
from driftmaps.build import SPAWN_HEIGHT_M, BuildOptions, BuildResult, build_track, lane_offset
from driftmaps.elevation.sources import CogSource, ElevationSource, SourceInfo
from driftmaps.geo.overpass import parse_overpass
from driftmaps.geo.router import Route, route_from_pins
from driftmaps.geometry.terrain import TerrainOptions

FIXTURES = Path(__file__).parent / "fixtures"
FORKS_PINS = [(43.80410, -79.99231), (43.81060, -79.98185)]
TRACK_ID = "dm_forks_test"
TEST_TERRAIN = TerrainOptions(rings=((15.0, 3.0), (60.0, 8.0), (150.0, 20.0)), near_m=60.0)

REQUIRED_FILES = (
    f"{TRACK_ID}.kn5",
    "ai/fast_lane.ai",
    "data/ideal_line.ai",
    "data/surfaces.ini",
    "data/map.ini",
    "map.png",
    "data/cameras.ini",
    "data/lighting.ini",
    "data/groove.ini",
    "data/crew.ini",
    "ui/ui_track.json",
    "ui/preview.png",
    "ui/outline.png",
    "LICENSE-DATA.txt",
    "route_osm.geojson",
    "route.json",
)


def fixture_dem() -> dict[str, CogSource]:
    info = SourceInfo(
        "hrdem-fixture",
        "NRCan HRDEM 1 m mosaic (5 m test window)",
        5.0,
        "DTM",
        "OGL-Canada-2.0",
        "Contains information licensed under the Open Government Licence - Canada. "
        "Source: Natural Resources Canada.",
    )
    return {"hrdem-fixture": CogSource(info, str(FIXTURES / "dem" / "hrdem_forks_5m.tif"))}


def fixture_options(cache_dir: Path, **kw) -> BuildOptions:
    args = dict(
        name="Forks Test",
        track_id=TRACK_ID,
        cache_dir=cache_dir,
        elevation_sources=fixture_dem(),
        elevation_chain=["hrdem-fixture"],
        terrain=TEST_TERRAIN,
        debug_png=False,
    )
    return BuildOptions(**(args | kw))


@pytest.fixture(scope="module")
def forks_route() -> Route:
    osm = parse_overpass((FIXTURES / "osm" / "forks_of_the_credit.json").read_bytes())
    return route_from_pins(FORKS_PINS, osm=osm)


@pytest.fixture(scope="module")
def built(tmp_path_factory, forks_route) -> BuildResult:
    root = tmp_path_factory.mktemp("build")
    return build_track(forks_route, root / "out", fixture_options(root / "cache"))


@pytest.fixture(scope="module")
def kn5(built) -> Kn5:
    return read_kn5(built.track_dir / f"{TRACK_ID}.kn5")


def meshes(kn5: Kn5, pattern: str) -> list[Mesh]:
    return [n for n in kn5.root.children if isinstance(n, Mesh) and re.match(pattern, n.name)]


def triangles(ms: list[Mesh]) -> np.ndarray:
    """(T, 3, 3) float64 world-space triangles of the meshes."""
    out = [m.positions.astype(np.float64)[m.indices.astype(np.int64).reshape(-1, 3)] for m in ms]
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def dummies(kn5: Kn5) -> dict[str, Dummy]:
    return {n.name: n for n in kn5.root.children if isinstance(n, Dummy)}


# --------------------------------------------------------------------------------------------
# Folder and KN5
# --------------------------------------------------------------------------------------------


def test_folder_complete(built):
    for rel in REQUIRED_FILES:
        assert (built.track_dir / rel).is_file(), rel
    assert (built.track_dir / "ai" / "fast_lane.ai").read_bytes() == (
        built.track_dir / "data" / "ideal_line.ai"
    ).read_bytes()
    assert not built.warnings or all("crosses itself" not in w for w in built.warnings)
    assert 1300 < built.stats["length_m"] < 1450
    assert built.stats["elevation_source"] == "hrdem-fixture"


def test_kn5_valid_and_named(kn5):
    assert validate(kn5) == []
    names = [n.name for n in kn5.root.children]
    for prefix in ("1ROAD_c", "1GRAVEL_c", "1GRASS_t", "1WALL_c", "paint_white_c", "far_grass_t"):
        assert any(n.startswith(prefix) for n in names), prefix
    assert any(n.startswith("paint_yellow_c") for n in names)  # Ontario: double yellow
    for m in kn5.root.children:
        if isinstance(m, Mesh):
            assert m.vertex_count <= MAX_MESH_VERTICES
            assert kn5.materials[m.material_id].shader == "ksPerPixel"
    for w in meshes(kn5, r"1WALL_"):
        assert not w.is_renderable and not w.cast_shadows
    for m in meshes(kn5, r"(paint|far)_"):
        assert not m.cast_shadows
        assert not re.match(r"\d", m.name)  # visual only: no surface prefix


def test_surfaces_cover_physical_meshes(built, kn5):
    ini = configparser.ConfigParser()
    ini.read(built.track_dir / "data" / "surfaces.ini")
    keys = {ini[s]["KEY"] for s in ini.sections()}
    assert {"ROAD", "GRAVEL", "GRASS", "DIRT"} <= keys
    for m in kn5.root.children:
        match = re.match(r"\d+([A-Z]+)", m.name)
        if isinstance(m, Mesh) and match and match.group(1) != "WALL":
            assert match.group(1) in keys, m.name


def test_track_files(built, forks_route):
    ui = json.loads((built.track_dir / "ui" / "ui_track.json").read_text())
    assert ui["tags"] == ["A2B", "touge", "drift-maps"]
    assert ui["country"] == "Canada"
    assert ui["run"] == "A2B"
    lat, lon = map(float, ui["geotags"])
    assert abs(lat - forks_route.pins[0].snap_lat) < 1e-5
    assert abs(lon - forks_route.pins[0].snap_lon) < 1e-5
    assert abs(int(ui["length"]) - built.stats["length_m"]) <= 1
    assert "OpenStreetMap" in ui["description"]
    lic = (built.track_dir / "LICENSE-DATA.txt").read_text(encoding="utf-8")
    assert "OpenStreetMap" in lic and "Open Government Licence - Canada" in lic
    geo = json.loads((built.track_dir / "route_osm.geojson").read_text(encoding="utf-8"))
    way_ids = {f["properties"]["osm_way_id"] for f in geo["features"]}
    assert way_ids == {e.way for e in forks_route.edges}
    assert Route.load(built.track_dir / "route.json").to_json() == forks_route.to_json()
    info = read_map_ini(built.track_dir / "data" / "map.ini")
    assert info.width > 100 and info.height > 100


# --------------------------------------------------------------------------------------------
# Geometry invariants
# --------------------------------------------------------------------------------------------


def test_no_foldover(kn5):
    """Every road, shoulder and paint triangle faces up and has area: nothing folds over."""
    tris = triangles(meshes(kn5, r"(1ROAD|1GRAVEL|paint)_"))
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    assert (n[:, 1] > 0).all()
    assert (0.5 * np.linalg.norm(n, axis=1) > 1e-7).all()
    # And in plan view (XZ) the triangles are counter-clockwise seen from above.
    a, b, c = tris[:, 0][:, [0, 2]], tris[:, 1][:, [0, 2]], tris[:, 2][:, [0, 2]]
    cross_y = (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]) - (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
    assert (cross_y > -1e-9).all()


def _heights_under(tris: np.ndarray, points: np.ndarray) -> list[tuple[int, float]]:
    """(point index, height of a triangle under it) for every triangle containing a point."""
    a, b, c = (tris[:, k][:, [0, 2]] for k in range(3))
    centre = (a + b + c) / 3.0
    radius = np.max([np.linalg.norm(p - centre, axis=1) for p in (a, b, c)], axis=0)
    tree = cKDTree(points[:, [0, 2]])
    hits = tree.query_ball_point(centre, radius + 1e-6)
    ti = np.repeat(np.arange(len(tris)), [len(h) for h in hits])
    pi = np.concatenate([np.asarray(h, int) for h in hits]) if len(ti) else np.zeros(0, int)
    p = points[pi][:, [0, 2]]
    v0, v1, v2 = b[ti] - a[ti], c[ti] - a[ti], p - a[ti]
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    ok = np.abs(den) > 1e-12
    u = np.where(ok, (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / np.where(ok, den, 1), -1)
    v = np.where(ok, (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / np.where(ok, den, 1), -1)
    inside = (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6)
    t = tris[ti]
    h = t[:, 0, 1] + u * (t[:, 1, 1] - t[:, 0, 1]) + v * (t[:, 2, 1] - t[:, 0, 1])
    return list(zip(pi[inside].tolist(), h[inside].tolist(), strict=True))


def test_road_above_terrain(kn5, built):
    """No terrain pokes through the road or the shoulders anywhere; under the bridge deck,
    away from the abutments (> 6 m), the terrain stays at least 1 m below."""
    road = triangles(meshes(kn5, r"(1ROAD|1GRAVEL)_"))
    terrain = triangles(meshes(kn5, r"(1GRASS|1DIRT|far_)"))
    pts = np.concatenate([road.reshape(-1, 3), road.mean(axis=1)])
    pts = np.unique(np.round(pts, 4), axis=0)
    model = built.geometry.model
    _, nearest = cKDTree(model.centre[:, [0, 2]]).query(pts[:, [0, 2]])
    assert model.bridge.any(), "the fixture route crosses the Credit River on a bridge"
    ground_s = model.s[~model.bridge]
    to_ground = np.abs(ground_s[None, :] - model.s[model.bridge][:, None]).min(axis=1)
    mid_span = np.zeros(len(model), bool)
    mid_span[np.flatnonzero(model.bridge)[to_ground > 6.0]] = True
    assert mid_span.any()
    hits = _heights_under(terrain, pts)
    assert hits, "terrain meets the road edges"
    worst = max(h - (pts[i, 1] - (1.0 if mid_span[nearest[i]] else 0.0)) for i, h in hits)
    assert worst <= 2e-3, f"terrain {worst:.3f} m above the road"


def test_terrain_hole_matches_road(kn5, built):
    """Terrain triangle centroids never lie on the asphalt (the hole is cut out), except
    under the bridge deck, where the terrain carries on below."""
    road = triangles(meshes(kn5, r"1ROAD_"))
    foot = shapely.union_all(shapely.polygons(road[:, :, [0, 2]]))
    terrain = triangles(meshes(kn5, r"(1GRASS|1DIRT)"))
    cent = terrain.mean(axis=1)
    inside = shapely.contains_xy(foot.buffer(-0.05), cent[:, 0], cent[:, 2])
    model = built.geometry.model
    _, i = cKDTree(model.centre[:, [0, 2]]).query(cent[inside][:, [0, 2]])
    assert model.bridge[i].all()


def test_walls_do_not_cross_road(kn5, built):
    walls = meshes(kn5, r"1WALL_")
    road = triangles(meshes(kn5, r"(1ROAD|1GRAVEL)_"))
    foot = shapely.union_all(shapely.polygons(road[:, :, [0, 2]]))
    verge = 4.0
    for w in walls:
        half = w.vertex_count // 2  # front face: rows of (top, bottom) along the wall
        base = w.positions[:half:2][:, [0, 2]].astype(np.float64)
        line = shapely.LineString(base)
        assert not line.intersects(foot), w.name
        assert shapely.distance(line, foot) > verge - 0.6, w.name


def test_walls_enclose_road_and_ends(kn5, built):
    """Walls stand around the whole drivable corridor including both ends (end walls)."""
    walls = meshes(kn5, r"1WALL_")
    pts = np.concatenate([w.positions[:, [0, 2]].astype(np.float64) for w in walls])
    model = built.geometry.model
    for i in (0, len(model) - 1):
        d = np.linalg.norm(pts - model.centre[i, [0, 2]], axis=1)
        assert d.min() < model.outer[i] + 4.0 + 1.0  # a wall just beyond each road end


# --------------------------------------------------------------------------------------------
# Dummies and AI
# --------------------------------------------------------------------------------------------


def test_spawns_and_gates(kn5, built):
    d = dummies(kn5)
    model = built.geometry.model
    geo = built.geometry
    expected = {"AC_START_0", "AC_START_1", "AC_HOTLAP_START_0"} | {f"AC_PIT_{n}" for n in range(8)}
    expected |= {f"AC_AB_{g}_{s}" for g in ("START", "FINISH") for s in ("L", "R")}
    assert expected <= set(d)
    tree = cKDTree(model.centre[:, [0, 2]])
    for name in expected:
        if not name.startswith(("AC_START", "AC_PIT", "AC_HOTLAP")):
            continue
        pos = d[name].transform[3, :3].astype(np.float64)
        _, i = tree.query(pos[[0, 2]])
        assert model.s[i] < geo.s_a  # every car starts behind the start gate
        assert abs(pos[1] - model.centre[i, 1] - SPAWN_HEIGHT_M) < 0.15
        fwd = d[name].transform[2, :3].astype(np.float64)
        assert fwd @ model.forward[i] > 0.95  # facing down the road
    s0 = d["AC_START_0"].transform[3, :3]
    s1 = d["AC_START_1"].transform[3, :3]
    assert 2.0 < np.linalg.norm(s0 - s1) < 6.0  # side by side
    for gate in ("AC_AB_START", "AC_AB_FINISH"):
        left_pos = d[f"{gate}_L"].transform[3, :3].astype(np.float64)
        right_pos = d[f"{gate}_R"].transform[3, :3].astype(np.float64)
        _, i = tree.query(((left_pos + right_pos) / 2)[[0, 2]])
        assert (left_pos - right_pos) @ left_of(model.forward[i]) > 0, gate
    start = d["AC_AB_START_L"].transform[3, :3]
    finish = d["AC_AB_FINISH_L"].transform[3, :3]
    assert np.linalg.norm(start - finish) > 500.0


def test_ai_line_on_driving_lane(built):
    ai = read_ai(built.track_dir / "ai" / "fast_lane.ai")
    geo = built.geometry
    model = geo.model
    pts = ai.positions.astype(np.float64)
    tree = cKDTree(model.centre[:, [0, 2]])
    _, i = tree.query(pts[:, [0, 2]])
    lateral = np.einsum("ij,ij->i", pts - model.centre[i], model.left[i])
    two_way = model.centre_line[i]
    # Ontario drives on the right: the AI follows the right lane on two-way stretches.
    assert (lateral[two_way] < -0.5).mean() > 0.97
    assert (np.abs(lateral) < model.half_width[i] - 0.4).all()  # inside the asphalt
    assert np.allclose(lateral, lane_offset(geo)[i], atol=0.35)
    # It starts behind every spawn and runs ~220 m past B, ending before the road does.
    s_ai = model.s[i]
    assert s_ai[0] < geo.s_a - 40.0
    assert geo.s_b + 200.0 < s_ai[-1] < model.s[-1]
    assert (ai.side_left > 0).all() and (ai.side_right > 0).all()
    assert (ai.radius > 0).all()
    assert ai.grid is not None


# --------------------------------------------------------------------------------------------
# Determinism, reverse, CLI
# --------------------------------------------------------------------------------------------


def _files(track_dir: Path) -> dict[str, bytes]:
    return {
        p.relative_to(track_dir).as_posix(): p.read_bytes()
        for p in sorted(track_dir.rglob("*"))
        if p.is_file()
    }


def test_deterministic(built, forks_route, tmp_path):
    again = build_track(forks_route, tmp_path / "out", fixture_options(tmp_path / "cache"))
    a, b = _files(built.track_dir), _files(again.track_dir)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k], k


def test_reverse_build(forks_route, tmp_path):
    rev = forks_route.reverse()
    result = build_track(
        rev, tmp_path / "out", fixture_options(tmp_path / "cache", track_id="dm_forks_rev")
    )
    ui = json.loads((result.track_dir / "ui" / "ui_track.json").read_text())
    assert abs(float(ui["geotags"][0]) - forks_route.pins[-1].snap_lat) < 1e-5
    ai = read_ai(result.track_dir / "ai" / "fast_lane.ai")
    model = result.geometry.model
    _, i = cKDTree(model.centre[:, [0, 2]]).query(ai.positions[:, [0, 2]].astype(np.float64))
    lateral = np.einsum("ij,ij->i", ai.positions - model.centre[i], model.left[i])
    assert (lateral[model.centre_line[i]] < -0.5).mean() > 0.97  # still the right lane


def test_cli_route_and_build(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from driftmaps import cli

    runner = CliRunner()
    route_json = tmp_path / "route.json"
    osm = FIXTURES / "osm" / "forks_of_the_credit.json"
    pins = [f"{lat},{lon}" for lat, lon in FORKS_PINS]
    res = runner.invoke(
        cli.app,
        [
            "route",
            "--start",
            pins[0],
            "--end",
            pins[1],
            "--osm-json",
            str(osm),
            "--out",
            str(route_json),
        ],
    )
    assert res.exit_code == 0, res.output
    assert Route.load(route_json).edges

    # Build from route.json with the fixture DEM file (small terrain via monkeypatch).
    import driftmaps.build as build_mod

    monkeypatch.setattr(
        build_mod, "BuildOptions", lambda **kw: BuildOptions(terrain=TEST_TERRAIN, **kw)
    )
    monkeypatch.setenv("DRIFTMAPS_CACHE_DIR", str(tmp_path / "cache"))
    res = runner.invoke(
        cli.app,
        [
            "build",
            "--route",
            str(route_json),
            "--name",
            "Forks CLI",
            "--id",
            "dm_forks_cli",
            "--out",
            str(tmp_path / "tracks"),
            "--dem-file",
            str(FIXTURES / "dem" / "hrdem_forks_5m.tif"),
            "--no-debug-png",
            "--zip",
            str(tmp_path / "t.zip"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert (tmp_path / "tracks" / "dm_forks_cli" / "dm_forks_cli.kn5").is_file()
    assert (tmp_path / "t.zip").is_file()


# --------------------------------------------------------------------------------------------
# Review fixes: spawn grid, wall height, folder safety, datum offset, splits
# --------------------------------------------------------------------------------------------


def test_spawn_grid_covers_every_pitbox(kn5, built):
    """A race with as many cars as pit boxes has a grid slot for each: AC_START_n side by side
    in rows, then the pits, then the hotlap start, all on the asphalt behind A, none
    overlapping."""
    d = dummies(kn5)
    ui = json.loads((built.track_dir / "ui" / "ui_track.json").read_text())
    pits = int(ui["pitboxes"])
    starts = [n for n in d if re.fullmatch(r"AC_START_\d+", n)]
    assert len(starts) >= pits
    assert {f"AC_START_{n}" for n in range(pits)} <= set(starts)
    names = sorted(n for n in d if n.startswith(("AC_START_", "AC_PIT_", "AC_HOTLAP_START_")))
    geo = built.geometry
    model = geo.model
    pos = np.array([d[n].transform[3, :3] for n in names], np.float64)
    _, i = cKDTree(model.centre[:, [0, 2]]).query(pos[:, [0, 2]])
    lateral = np.einsum("ij,ij->i", pos - model.centre[i], model.left[i])
    assert (np.abs(lateral) < model.half_width[i] - 0.8).all()  # on the asphalt
    s = model.s[i]
    assert (s < geo.s_a - 3.0).all() and (s > 3.0).all()  # behind A, before the road's end
    # Footprints ~4.6 x 2 m: two cars are clear if 5 m apart along the road or 2.4 m across.
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            clear = abs(s[a] - s[b]) >= 5.0 or abs(lateral[a] - lateral[b]) >= 2.4
            assert clear, (names[a], names[b])


def test_wall_tops_stand_above_the_road(kn5, built):
    """Hidden walls reach at least 2 m above the nearest road everywhere, also where the
    road runs on a bridge or a fill high above the ground the wall stands on."""
    model = built.geometry.model
    tree = cKDTree(model.centre[:, [0, 2]])
    for w in meshes(kn5, r"1WALL_"):
        pairs = w.positions.astype(np.float64).reshape(-1, 2, 3)  # (top, bottom) per row
        top = pairs[np.arange(len(pairs)), np.argmax(pairs[:, :, 1], axis=1)]
        _, i = tree.query(top[:, [0, 2]])
        clearance = top[:, 1] - model.centre[i, 1]
        assert clearance.min() >= 2.0, (w.name, float(clearance.min()))


def test_splits_are_clamped_to_one(built):
    from driftmaps.build import _dummies

    with pytest.warns(UserWarning, match="at most 1"):
        gates = _dummies(built.geometry, BuildOptions(splits=3)).gates
    assert sorted(gates) == ["AC_AB_FINISH", "AC_AB_START", "AC_TIME_0"]


def test_build_refuses_to_replace_a_folder_it_did_not_make(forks_route, tmp_path):
    from driftmaps.ac.scene import TrackFolderError

    foreign = tmp_path / "out" / TRACK_ID
    foreign.mkdir(parents=True)
    (foreign / "kunos.kn5").write_bytes(b"not ours")
    with pytest.raises(TrackFolderError, match="not made by drift-maps"):
        build_track(forks_route, tmp_path / "out", fixture_options(tmp_path / "cache"))
    assert (foreign / "kunos.kn5").read_bytes() == b"not ours"


def test_install_replaces_only_drift_maps_folders(built, tmp_path):
    from driftmaps.ac.scene import TrackFolderError, install_track, is_driftmaps_track

    assert is_driftmaps_track(built.track_dir)
    tracks = tmp_path / "content" / "tracks"
    dest = install_track(built.track_dir, tracks)
    assert (dest / f"{TRACK_ID}.kn5").is_file()
    (dest / "stale.txt").write_text("old build")
    dest = install_track(built.track_dir, tracks)  # ours: replaced
    assert not (dest / "stale.txt").exists()

    foreign = tmp_path / "other" / "tracks"
    (foreign / TRACK_ID / "ui").mkdir(parents=True)
    ui = {"name": "Someone else's track", "tags": ["circuit"]}
    (foreign / TRACK_ID / "ui" / "ui_track.json").write_text(json.dumps(ui))
    with pytest.raises(TrackFolderError):
        install_track(built.track_dir, foreign)
    assert (foreign / TRACK_ID / "ui" / "ui_track.json").is_file()
    install_track(built.track_dir, foreign, force=True)
    assert (foreign / TRACK_ID / f"{TRACK_ID}.kn5").is_file()


def test_cli_build_refuses_foreign_folder(forks_route, tmp_path):
    from typer.testing import CliRunner

    from driftmaps import cli

    route_json = tmp_path / "route.json"
    forks_route.save(route_json)
    foreign = tmp_path / "tracks" / "ks_some_track"
    foreign.mkdir(parents=True)
    (foreign / "ks_some_track.kn5").write_bytes(b"kunos")
    args = ["build", "--route", str(route_json), "--name", "X", "--id", "ks_some_track"]
    res = CliRunner().invoke(cli.app, [*args, "--out", str(tmp_path / "tracks")])
    assert res.exit_code != 0
    assert "--force" in res.output
    assert (foreign / "ks_some_track.kn5").read_bytes() == b"kunos"


class _FnSource(ElevationSource):
    def __init__(self, name: str, fn, resolution_m: float) -> None:
        super().__init__(SourceInfo(name, name, resolution_m, "DTM", "test", name))
        self.fn = fn

    def sample(self, lon, lat, cache_dir):
        return self.fn(np.asarray(lon, np.float64), np.asarray(lat, np.float64))


@pytest.mark.parametrize("datum_m, expected_m", [(1.0, 1.0), (5.0, 0.0)])
def test_datum_offset_measured_on_gentle_terrain_off_the_road(tmp_path, datum_m, expected_m):
    """A fine DTM covers 200 m either side of a straight road; a coarse source covers
    everything but sits `datum_m` lower and reads 8 m high along the road itself (a cut it
    smooths over). The offset applied to the coarse heights comes from gentle terrain near the
    road, not from the road, and is dropped when it is implausibly large (> 2 m)."""
    from driftmaps.build import _Elevation

    lat0, lon0 = 43.5, -80.0
    m_lat = 111_320.0
    m_lon = m_lat * np.cos(np.radians(lat0))

    def north_m(lat):
        return (lat - lat0) * m_lat

    def true(lon, lat):
        return 200.0 + 0.03 * (lon - lon0) * m_lon + 0.02 * north_m(lat)

    def fine(lon, lat):
        return np.where(np.abs(north_m(lat)) < 200.0, true(lon, lat), np.nan)

    def coarse(lon, lat):
        return true(lon, lat) - datum_m + 8.0 * np.exp(-((north_m(lat) / 15.0) ** 2))

    options = BuildOptions(
        cache_dir=tmp_path,
        elevation_sources={
            "fine": _FnSource("fine", fine, 1.0),
            "coarse": _FnSource("coarse", coarse, 30.0),
        },
        elevation_chain=["fine", "coarse"],
    )
    elev = _Elevation(options, lon0, lat0)
    x = np.arange(0.0, 800.0, 2.0)
    across = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    road_lon = lon0 + x[:, None] / m_lon + 0.0 * across[None, :]
    road_lat = lat0 + across[None, :] / m_lat + 0.0 * x[:, None]
    z = elev.road(road_lon, road_lat)
    assert np.allclose(z, true(road_lon, road_lat))
    far_lon = lon0 + np.linspace(0.0, 800.0, 50) / m_lon
    far_lat = np.full(50, lat0 + 600.0 / m_lat)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        far = elev.terrain(far_lon, far_lat, far=True)
    assert any("not corrected" in str(w.message) for w in caught) == (expected_m == 0.0)
    assert elev.offsets["coarse"] == pytest.approx(expected_m, abs=0.05)
    assert np.allclose(far, true(far_lon, far_lat) - datum_m + expected_m, atol=0.05)
