"""Offline end-to-end build WITH scenery (milestone M2): bridges, guardrails, buildings, trees.

Same route and DEM as tests/test_build.py (the first 1.36 km of Forks of the Credit Road, with
its Credit River bridge, and a 5 m NRCan HRDEM window). The live scenery sources are replaced
through `BuildOptions` hooks, so nothing touches the network:
- canopy: a synthetic canopy height model (8 m lattice of paraboloid crowns, 15 m tall, in
  patches of forest, open ground elsewhere);
- land cover: class 6 (mixed forest) everywhere;
- OSM barriers: one guard rail traced 4.5 m right of the centreline over 120 m of the road;
- guardrails.json: one left-side span (route metres from A);
- buildings: one house 30 m left of the road, one shed inside the wall corridor and one barn
  in the forest reaching into the corridor (both must go, and no tree may grow out of them).

The written KN5 is re-read and checked: every scenery mesh is named, sized and flagged right,
trees stand outside the hidden walls and off the buildings, rails exist only where a source
says so, the bridge has a structure and no terrain inside its deck, and the flags turn each
part off.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import shapely
from scipy.spatial import cKDTree

from driftmaps.ac.kn5 import read_kn5, validate
from driftmaps.ac.model import MAX_MESH_VERTICES, Kn5, Mesh
from driftmaps.build import BuildOptions, BuildResult, _merge_notices, build_track
from driftmaps.elevation.sources import CogSource, SourceInfo
from driftmaps.geo.barriers import OsmBarrier
from driftmaps.geo.buildings import Footprint
from driftmaps.geo.overpass import parse_overpass
from driftmaps.geo.project import LocalProjection
from driftmaps.geo.router import Route, route_from_pins
from driftmaps.geometry.bridges import terrain_through_deck
from driftmaps.geometry.terrain import TerrainOptions, build_terrain, clamp_heights
from driftmaps.geometry.walls import corridor_polygon

FIXTURES = Path(__file__).parent / "fixtures"
FORKS_PINS = [(43.80410, -79.99231), (43.81060, -79.98185)]
TEST_TERRAIN = TerrainOptions(rings=((15.0, 3.0), (60.0, 8.0), (150.0, 20.0)), near_m=60.0)
TRACK_ID = "dm_forks_scenery"
RAIL_FROM_A_M = (500.0, 620.0)  # the synthetic OSM guard rail, route metres from A (right)
USER_SPAN = {"side": "left", "from_m": 200.0, "to_m": 240.0}  # guardrails.json


def BARN(route: Route) -> np.ndarray:  # noqa: N802
    """A barn in the synthetic forest 8-24 m left of the road at 400 m: it reaches into the
    wall corridor, so it is not drawn, but the canopy over it is its roof, not trees."""
    return _box(route, 400.0, 16.0, (20.0, 16.0))


def _projection(route: Route) -> LocalProjection:
    pin = route.pins[0]
    return LocalProjection(pin.snap_lat, pin.snap_lon)


def synthetic_canopy(projection: LocalProjection):
    """Canopy height (m) at AC XZ points: forest patches of 15 m crowns on an 8 m lattice."""

    def fn(xz: np.ndarray) -> np.ndarray:
        xz = np.asarray(xz, np.float64).reshape(-1, 2)
        x, z = xz[:, 0], xz[:, 1]
        forest = (np.sin(x / 70.0) + np.cos(z / 55.0)) > 0.3
        r2 = ((x + 4.0) % 8.0 - 4.0) ** 2 + ((z + 4.0) % 8.0 - 4.0) ** 2
        return np.where(forest, np.clip(15.0 - 0.6 * r2, 0.0, None), 0.0)

    return fn


def mixed_forest(projection: LocalProjection):
    return lambda xz: np.full(len(np.asarray(xz).reshape(-1, 2)), 6, np.uint8)


def _offset_line(route: Route, a_m: float, b_m: float, right_m: float) -> np.ndarray:
    """(N, 2) lon/lat of the route between a_m and b_m from A, `right_m` to its right."""
    proj = _projection(route)
    poly = route.polyline(step_m=2.0)
    sel = (poly.s >= a_m) & (poly.s <= b_m)
    en = proj.to_xy(poly.lonlat[sel])
    d = np.gradient(en, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None]
    right = np.stack([d[:, 1], -d[:, 0]], axis=1)  # (east, north) to the right of travel
    pts = en + right_m * right
    return proj.from_xy(pts)


def _box(route: Route, at_m: float, left_m: float, size_m: tuple[float, float]) -> np.ndarray:
    """A closed lon/lat rectangle centred `left_m` left of the route at `at_m` from A."""
    proj = _projection(route)
    poly = route.polyline(step_m=1.0)
    i = int(np.searchsorted(poly.s, at_m))
    en = proj.to_xy(poly.lonlat[i - 1 : i + 2])
    d = en[2] - en[0]
    d /= np.linalg.norm(d)
    left = np.array([-d[1], d[0]])
    c = en[1] + left_m * left
    hx, hy = size_m[0] / 2, size_m[1] / 2
    ring = [c + sx * hx * d + sy * hy * left for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    ring.append(ring[0])
    return proj.from_xy(np.array(ring))


@pytest.fixture(scope="module")
def forks_route() -> Route:
    osm = parse_overpass((FIXTURES / "osm" / "forks_of_the_credit.json").read_bytes())
    return route_from_pins(FORKS_PINS, osm=osm)


def scenery_options(route: Route, root: Path, **kw) -> BuildOptions:
    spans = root / "guardrails.json"
    spans.parent.mkdir(parents=True, exist_ok=True)
    spans.write_text(json.dumps([USER_SPAN]), encoding="utf-8")
    rail = OsmBarrier(1, "guard_rail", _offset_line(route, *RAIL_FROM_A_M, 4.5), {})
    house = Footprint("osm:w1", "osm", _box(route, 700.0, 30.0, (12.0, 9.0)), {"building": "house"})
    shed = Footprint("osm:w2", "osm", _box(route, 900.0, 6.0, (4.0, 3.0)), {"building": "shed"})
    barn = Footprint("osm:w3", "osm", BARN(route), {"building": "barn"})
    info = SourceInfo(
        "hrdem-fixture",
        "NRCan HRDEM 1 m mosaic (5 m test window)",
        5.0,
        "DTM",
        "OGL-Canada-2.0",
        "Contains information licensed under the Open Government Licence - Canada.",
    )
    dem = CogSource(info, str(FIXTURES / "dem" / "hrdem_forks_5m.tif"))
    args = dict(
        name="Forks Scenery",
        track_id=TRACK_ID,
        cache_dir=root / "cache",
        elevation_sources={"hrdem-fixture": dem},
        elevation_chain=["hrdem-fixture"],
        terrain=TEST_TERRAIN,
        offline=True,
        trees=True,
        buildings=True,
        guardrails=True,
        canopy_source=synthetic_canopy,
        landcover_source=mixed_forest,
        osm_barriers=[rail],
        building_footprints=[house, shed, barn],
        guardrails_file=spans,
        debug_png=True,
    )
    return BuildOptions(**(args | kw))


@pytest.fixture(scope="module")
def built(tmp_path_factory, forks_route) -> BuildResult:
    root = tmp_path_factory.mktemp("scenery")
    return build_track(forks_route, root / "out", scenery_options(forks_route, root))


@pytest.fixture(scope="module")
def kn5(built) -> Kn5:
    return read_kn5(built.track_dir / f"{TRACK_ID}.kn5")


def meshes(kn5: Kn5, pattern: str) -> list[Mesh]:
    return [n for n in kn5.root.children if isinstance(n, Mesh) and re.match(pattern, n.name)]


# --------------------------------------------------------------------------------------------
# KN5 contents
# --------------------------------------------------------------------------------------------


def test_scenery_meshes_are_named_sized_and_flagged(kn5, built):
    assert validate(kn5) == []
    for prefix in (
        r"trees_near_",
        r"trees_far_",
        r"buildings_",
        r"guardrail_",
        r"1WALL_GR_",
        r"bridge00_concrete$",
        r"1WALL_BRIDGE00_L",
        r"1WALL_BRIDGE00_R",
    ):
        assert meshes(kn5, prefix), prefix
    for m in kn5.root.children:
        if not isinstance(m, Mesh):
            continue
        assert m.vertex_count <= MAX_MESH_VERTICES, m.name
        if re.match(r"(trees|buildings|guardrail|bridge|parapet)", m.name):
            assert m.is_renderable and m.lod_out > 0, m.name  # visual only, culled with distance
        if m.name.startswith("1WALL"):
            assert not m.is_renderable and not m.cast_shadows, m.name
    shaders = {kn5.materials[m.material_id].shader for m in meshes(kn5, r"trees_")}
    assert shaders == {"ksTree"}
    tree_mat = kn5.materials[meshes(kn5, r"trees_")[0].material_id]
    assert tree_mat.alpha_tested
    names = {t.name for t in kn5.textures}
    assert {"dm_trees.dds", "dm_buildings.dds", "concrete.png", "steel.png"} <= names
    s = built.stats
    assert s["vertices"]["trees"] > 0 and s["vertices"]["buildings"] > 0
    assert s["draw_meshes"] + s["collision_meshes"] == s["mesh_count"]


def test_summary_and_credits(built):
    text = built.summary()
    for word in ("trees", "houses", "rails", "bridge"):
        assert re.search(rf"^{word}\s", text, re.M), word
    sc = built.stats["scenery"]
    assert sc["trees"]["total"] > 500
    assert sc["buildings"]["buildings"] == 1  # the shed and the barn (corridor) are dropped
    assert sc["user_spans"] == 1 and sc["osm_barriers"] == 1
    lic = (built.track_dir / "LICENSE-DATA.txt").read_text(encoding="utf-8")
    assert "building data" in lic  # OSM building footprints are credited
    assert "synthetic" not in lic and "route_osm.geojson" in lic
    # One credit per source and licence text: OSM once (road + buildings + barriers).
    bodies = [p.split(": ", 1)[1] for p in lic.split("\n\n") if ": " in p]
    assert len(bodies) == len(set(bodies))
    assert lic.count("OpenStreetMap contributors") == 1
    ogl = "Contains information licensed under the OGL-Canada. Source: NRCan, HRDEM."
    merged = _merge_notices(
        [f"Elevation (HRDEM DTM): {ogl}", f"Tree heights (HRDEM DSM): {ogl}", "Land cover: x"]
    )
    assert merged == [f"Elevation (HRDEM DTM); Tree heights (HRDEM DSM): {ogl}", "Land cover: x"]
    # ODbL: the OSM ways used are listed with the route's, buildings and barriers too.
    geo = json.loads((built.track_dir / "route_osm.geojson").read_text(encoding="utf-8"))
    uses = {
        (f["properties"]["use"], f["properties"]["osm_way_id"])
        for f in geo["features"]
        if "use" in f["properties"]
    }
    assert uses == {("building", 1), ("tree mask", 2), ("tree mask", 3), ("barrier", 1)}
    ui = json.loads((built.track_dir / "ui" / "ui_track.json").read_text())
    assert "Trees" in ui["description"] and "Buildings" in ui["description"]
    assert json.loads((built.track_dir / "guardrails.json").read_text()) == [USER_SPAN]
    names = {p.name for p in built.debug_images}
    assert {f"{TRACK_ID}.png", f"{TRACK_ID}_detail.png", f"{TRACK_ID}_bridge0.png"} <= names


# --------------------------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------------------------


def test_trees_stand_outside_the_hidden_walls_and_off_buildings(built):
    m = built.geometry.model
    trees = built.scenery.trees
    corridor = corridor_polygon(m, m.outer + 4.0)
    assert not shapely.contains_xy(corridor, trees.xz[:, 0], trees.xz[:, 1]).any()
    for b in built.scenery.buildings:
        inside = shapely.contains_xy(b.polygon.buffer(1.9), trees.xz[:, 0], trees.xz[:, 1])
        assert not inside.any(), b.id
    # Not drawn, but a real building: no clump of trees where the barn stands.
    proj = built.geometry.projection
    e, n = proj.to_enu(*BARN(built.geometry.route).T)
    barn = shapely.Polygon(np.stack([e, -n], axis=1)).buffer(1.9)
    assert not shapely.contains_xy(barn, trees.xz[:, 0], trees.xz[:, 1]).any()
    # The synthetic forest is where the trees are: open ground gets none (a tree may sit
    # between crowns, a gap fill, or be pushed up to 3 m out of the road's clearance).
    x, z = trees.xz[:, 0], trees.xz[:, 1]
    assert ((np.sin(x / 70.0) + np.cos(z / 55.0)) > 0.25).mean() > 0.97


def test_rails_only_where_a_source_says_so(built, kn5):
    geo = built.geometry
    m = geo.model
    spans = built.scenery.spans
    by_source = {}
    for sp in spans:
        by_source.setdefault(sp.source, []).append(sp)
    assert set(by_source) == {"osm", "user"}  # no invented bridge-approach rails
    (osm,) = by_source["osm"]
    assert osm.side == "right" and osm.kind == "guard_rail"
    assert osm.s0 - geo.s_a == pytest.approx(RAIL_FROM_A_M[0], abs=4.0)
    assert osm.s1 - geo.s_a == pytest.approx(RAIL_FROM_A_M[1], abs=4.0)
    (user,) = by_source["user"]
    assert user.side == "left"
    assert (user.s0 - geo.s_a, user.s1 - geo.s_a) == pytest.approx((200.0, 240.0))
    # Every guardrail vertex stands at the shoulder edge, within one of those spans.
    pts = np.concatenate([g.positions.astype(np.float64) for g in meshes(kn5, r"guardrail_")])
    _, i = cKDTree(m.centre[:, [0, 2]]).query(pts[:, [0, 2]])
    lateral = np.abs(np.einsum("ij,ij->i", pts - m.centre[i], m.left[i]))
    assert np.all(lateral > m.outer[i] - 0.1) and np.all(lateral < m.outer[i] + 1.0)
    s = m.s[i]
    covered = np.zeros(len(s), bool)
    for sp in spans:
        covered |= (s >= sp.s0 - 5.0) & (s <= sp.s1 + 5.0)  # + turned-down terminal ends
    assert covered.all()


def test_bridge_has_structure_and_no_terrain_in_the_deck(built, kn5):
    plans = built.scenery.plans
    assert plans, "the fixture route crosses the Credit River on a bridge"
    sc = built.stats["scenery"]
    assert sc["deck_intrusion_m"] == 0.0
    m = built.geometry.model
    for p in plans:
        check = terrain_through_deck(m, p, built.scenery.ground)
        assert check.ok, check.max_intrusion
    concrete = meshes(kn5, r"bridge00_concrete$")[0]
    pos = concrete.positions.astype(np.float64)
    deck_y = np.interp(0.5 * (plans[0].s0 + plans[0].s1), m.s, m.centre[:, 1])
    assert pos[:, 1].min() < deck_y - 2.0  # abutments/piers reach down into the ground


def test_terrain_under_the_deck_is_the_real_ground(built):
    """Away from the abutments the terrain under the deck is the DEM itself (the river
    valley), not a pit dug a fixed depth under the road."""
    geo = built.geometry
    m = geo.model
    p = built.scenery.plans[0]
    if p.solid:
        pytest.skip("solid bridge: nothing open under the deck")
    s = np.linspace(p.face0 + 2.0, p.face1 - 2.0, 9)
    xz = np.stack([np.interp(s, m.s, m.centre[:, 0]), np.interp(s, m.s, m.centre[:, 2])], 1)
    terrain = built.scenery.ground(xz)
    lon, lat = geo.projection.to_lonlat(xz[:, 0], -xz[:, 1])
    dem = geo.elevation.terrain(np.asarray(lon), np.asarray(lat), far=False) - geo.z_ref
    deck = np.interp(s, m.s, m.centre[:, 1])
    low = dem < deck - TerrainOptions().bridge_clearance_m
    assert low.any()
    # The terrain mesh interpolates a 3 m test grid: within 0.5 m of the DEM. (The old 3 m
    # clearance dug the ground here 0.8 m below the river bed.)
    assert np.abs(terrain[low] - dem[low]).max() < 0.5


# --------------------------------------------------------------------------------------------
# Flags and soft failures
# --------------------------------------------------------------------------------------------


def test_flags_turn_scenery_off_but_bridges_stay(forks_route, tmp_path):
    opts = scenery_options(
        forks_route, tmp_path, trees=False, buildings=False, guardrails=False, debug_png=False
    )
    res = build_track(forks_route, tmp_path / "out", opts)
    kn5 = read_kn5(res.track_dir / f"{TRACK_ID}.kn5")
    names = [n.name for n in kn5.root.children]
    for prefix in ("trees_", "buildings_", "guardrail_", "1WALL_GR_", "1WALL_bld"):
        assert not any(n.startswith(prefix) for n in names), prefix
    assert any(n.startswith("bridge00_concrete") for n in names)
    assert any(n.startswith("1WALL_BRIDGE00_") for n in names)
    assert res.scenery.spans == []
    assert not (res.track_dir / "guardrails.json").exists()


def test_density_zero_means_no_trees_and_missing_data_only_warns(forks_route, tmp_path):
    opts = scenery_options(
        forks_route,
        tmp_path,
        scenery_density=0.0,
        osm_barriers=None,  # fetch: offline with an empty cache -> a warning, no OSM rails
        building_footprints=None,
        debug_png=False,
    )
    res = build_track(forks_route, tmp_path / "out", opts)
    assert res.stats["scenery"]["trees"]["total"] == 0
    assert any("OSM guardrails unavailable" in w for w in res.warnings)
    assert any("building footprints unavailable" in w for w in res.warnings)
    assert {sp.source for sp in res.scenery.spans} == {"user"}


def test_clamp_heights_lowers_only_grid_points(built):
    m = built.geometry.model
    left = m.centre + m.outer[:, None] * m.left
    right = m.centre - m.outer[:, None] * m.left

    def flat(xz: np.ndarray, far: bool) -> np.ndarray:
        return np.full(len(xz), 50.0)

    opts = TerrainOptions(rings=((15.0, 3.0), (40.0, 10.0)), near_m=40.0)
    terrain = build_terrain(m, left, right, flat, opts)
    ceiling = np.where(terrain.points[:, 0] > np.median(terrain.points[:, 0]), -100.0, np.inf)
    out, moved = clamp_heights(terrain, ceiling, opts)
    lowered = ~terrain.is_edge & (terrain.heights > ceiling)
    assert moved == lowered.sum() > 0
    assert np.all(out.heights[lowered] == -100.0)
    assert np.array_equal(out.heights[~lowered], terrain.heights[~lowered])
    assert np.allclose(np.linalg.norm(out.normals, axis=1), 1.0)
    assert clamp_heights(terrain, np.full(len(terrain.heights), np.inf))[1] == 0


# --------------------------------------------------------------------------------------------
# guardrails.json: rebuilds in place, reversed builds; the CLI
# --------------------------------------------------------------------------------------------

FAST = dict(trees=False, buildings=False, debug_png=False)


def test_rebuild_in_place_keeps_the_users_guardrails(forks_route, tmp_path):
    """guardrails.json kept next to route.json INSIDE the track folder survives a rebuild over
    that folder (it used to be deleted with the old folder, and then the copy crashed)."""
    out = tmp_path / "out"
    first = build_track(forks_route, out, scenery_options(forks_route, tmp_path, **FAST))
    user_file = first.track_dir / "guardrails.json"
    raw = b"\xef\xbb\xbf" + json.dumps({"guardrails": [USER_SPAN]}, indent=2).encode()  # + BOM
    user_file.write_bytes(raw)
    opts = scenery_options(
        forks_route, tmp_path, guardrails_file=user_file, bridge_approach_rails=True, **FAST
    )
    res = build_track(Route.load(first.track_dir / "route.json"), out, opts)
    assert user_file.read_bytes() == raw
    # Asked for: invented approach rails, up to 20 m before/after each railed deck.
    by_source = {}
    for sp in res.scenery.spans:
        by_source.setdefault(sp.source, []).append(sp)
    assert set(by_source) == {"osm", "user", "bridge"}
    for sp in by_source["bridge"]:
        assert any(abs(sp.s1 - p.s0) < 0.5 or abs(sp.s0 - p.s1) < 0.5 for p in res.scenery.plans)
        assert sp.length <= 20.0 + 1e-6


def _reverse_via_cli(forks_route, tmp_path, monkeypatch) -> BuildResult:
    import dataclasses

    from typer.testing import CliRunner

    import driftmaps.build as build_mod
    from driftmaps import cli

    route_json = tmp_path / "routes" / "route.json"
    forks_route.save(route_json)
    (route_json.parent / "guardrails.json").write_text(json.dumps([USER_SPAN]), encoding="utf-8")
    base = scenery_options(forks_route, tmp_path, **FAST)
    monkeypatch.setattr(build_mod, "BuildOptions", lambda **kw: dataclasses.replace(base, **kw))
    results = []
    real = build_mod.build_track
    monkeypatch.setattr(build_mod, "build_track", lambda *a: results.append(real(*a)) or results[0])
    args = ["build", "--route", str(route_json), "--name", "Rev", "--id", "dm_forks_rev",
            "--reverse", "--offline", "--no-trees", "--no-buildings", "--no-debug-png",
            "--cache-dir", str(tmp_path / "cache"), "--out", str(tmp_path / "out")]  # fmt: skip
    res = CliRunner().invoke(cli.app, args)
    assert res.exit_code == 0, res.output
    return results[0]


def test_cli_reverse_build_mirrors_the_users_guardrails(forks_route, tmp_path, monkeypatch):
    """guardrails.json is measured as driven along route.json; `--reverse` drives from B, so
    the left rail 200-240 m from A is on the right, 200-240 m before the new B."""
    res = _reverse_via_cli(forks_route, tmp_path, monkeypatch)
    geo = res.geometry
    (user,) = [sp for sp in res.scenery.spans if sp.source == "user"]
    length = geo.length_m
    assert user.side == "right"
    assert (user.s0 - geo.s_a, user.s1 - geo.s_a) == pytest.approx((length - 240, length - 200))
    # The copy next to the reversed route.json says it is measured the other way.
    copy = json.loads((res.track_dir / "guardrails.json").read_text(encoding="utf-8"))
    assert copy == {"reversed": True, "guardrails": [USER_SPAN]}
    again = build_track(
        Route.load(res.track_dir / "route.json"),
        tmp_path / "again",
        scenery_options(
            forks_route, tmp_path, guardrails_file=res.track_dir / "guardrails.json", **FAST
        ),
    )
    (user2,) = [sp for sp in again.scenery.spans if sp.source == "user"]
    assert (user2.side, user2.s0, user2.s1) == pytest.approx(("right", user.s0, user.s1))


def test_cli_prints_the_zip_size_and_stops_big_zips(tmp_path, capsys):
    import typer

    from driftmaps.cli import ZIP_ERROR_MB, ZIP_WARN_MB, check_zip_size

    def zip_of(mb: float):
        path = tmp_path / f"{mb}.zip"
        with path.open("wb") as f:
            f.truncate(int(mb * 1e6))
        return path

    assert check_zip_size(zip_of(3.2)) == pytest.approx(3.2)
    out = capsys.readouterr()
    assert "(3.2 MB)" in out.out and not out.err
    check_zip_size(zip_of(ZIP_WARN_MB))
    assert "warning" in capsys.readouterr().err
    with pytest.raises(typer.Exit):
        check_zip_size(zip_of(ZIP_ERROR_MB))
    assert "error" in capsys.readouterr().err
