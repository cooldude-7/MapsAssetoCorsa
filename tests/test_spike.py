"""Tests for driftmaps.spike: the M0 in-game test tracks (`drift-maps spike`).

Every track is built once into a temporary folder, then each file is parsed back with our own
readers (KN5, AI line, map.ini, INI files, JSON) and compared with the in-memory description
(`make_spike_track`). When the dev-only GPL oracle parser is present (fetch it with
`uv run python tools/fetch_oracles.py`), every KN5 is also parsed by it; otherwise that test
skips. No game files are needed.
"""

from __future__ import annotations

import configparser
import importlib.util
import io
import json
import re
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from driftmaps.ac.ai_line import DEFAULT_MAX_SPEED_MPS, read_ai
from driftmaps.ac.coords import left_of
from driftmaps.ac.kn5 import kn5_to_bytes, read_kn5, texture_info, validate
from driftmaps.ac.model import MAX_MESH_VERTICES, Dummy, Kn5, Mesh
from driftmaps.ac.textures import read_dds_info
from driftmaps.ac.track_files import read_map_ini
from driftmaps.geometry.centreline import total_turn
from driftmaps.spike import (
    GATE_EXTRA_M,
    GATE_HEIGHT_M,
    GRASS_WIDTH_M,
    SPIKE_TAG,
    SPIKE_TRACK_IDS,
    SpikeTrack,
    build_spike_tracks,
    make_spike_track,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ORACLE_PATH = REPO_ROOT / "tools" / "oracles" / "nazariykul_kn5.py"

REQUIRED_FILES = (
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
)
AI_TRACKS = ("dm_t11_ai_loop", "dm_t12_ai_ab")
GATE_TRACKS = ("dm_t4_ab_timing", "dm_t5_swapped_gates", "dm_t11_ai_loop", "dm_t12_ai_ab")
AB_TRACKS = ("dm_t4_ab_timing", "dm_t5_swapped_gates", "dm_t12_ai_ab")

PHYSICAL_RE = re.compile(r"^\d+(ROAD|GRASS|GRAVEL)")
# Anything that could make AC (or CM's `\d+(KEY|...)` matcher) treat a visual mesh as physical.
SURFACE_ANYWHERE_RE = re.compile(r"\d+(ROAD|GRASS|GRAVEL|KERB|SAND|WALL)")
HIDDEN_WALLS = {"1WALL_L", "1WALL_R"}
VISIBLE_WALLS = {"1WALL_VIS", "1WALL_END"}


# --------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def build(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[Path], Path]:
    root = tmp_path_factory.mktemp("spike")
    zip_path = root / "drift-maps-m0-tests.zip"
    dirs = build_spike_tracks(root / "tracks", zip_path=zip_path)
    return root / "tracks", dirs, zip_path


@pytest.fixture(scope="module")
def out_dir(build: tuple[Path, list[Path], Path]) -> Path:
    return build[0]


@pytest.fixture(scope="module")
def specs() -> dict[str, SpikeTrack]:
    return {t: make_spike_track(t) for t in SPIKE_TRACK_IDS}


@pytest.fixture(scope="module")
def kn5s(out_dir: Path) -> dict[str, Kn5]:
    return {t: read_kn5(out_dir / t / f"{t}.kn5") for t in SPIKE_TRACK_IDS}


def root_dummies(kn5: Kn5) -> dict[str, Dummy]:
    return {n.name: n for n in kn5.root.children if isinstance(n, Dummy)}


def meshes(kn5: Kn5) -> dict[str, Mesh]:
    found = [n for n in kn5.iter_nodes() if isinstance(n, Mesh)]
    names = [m.name for m in found]
    assert len(names) == len(set(names)), "mesh names should be unique"
    return {m.name: m for m in found}


def read_ini(path: Path) -> configparser.ConfigParser:
    cp = configparser.ConfigParser(interpolation=None, strict=True)
    cp.optionxform = str  # keep KEY case
    cp.read_string(path.read_text(encoding="ascii"))
    return cp


def position(d: Dummy) -> np.ndarray:
    return d.transform[3, :3].astype(np.float64)


# --------------------------------------------------------------------------------------------
# Folder contents
# --------------------------------------------------------------------------------------------


def test_builds_every_track_in_canonical_order(build) -> None:
    out, dirs, _ = build
    assert [d.name for d in dirs] == list(SPIKE_TRACK_IDS)
    assert all(d.parent == out for d in dirs)
    assert len(SPIKE_TRACK_IDS) == 12


@pytest.mark.parametrize("track_id", SPIKE_TRACK_IDS)
def test_required_files(out_dir: Path, track_id: str) -> None:
    track = out_dir / track_id
    for rel in (f"{track_id}.kn5", *REQUIRED_FILES):
        assert (track / rel).is_file(), rel
    has_ai = track_id in AI_TRACKS
    assert (track / "ai" / "fast_lane.ai").is_file() == has_ai
    assert (track / "data" / "ideal_line.ai").is_file() == has_ai
    assert (track / "models.ini").is_file() == (track_id == "dm_t8_models_ini")
    assert not (track / "ai").exists() or has_ai  # never a placeholder AI folder


@pytest.mark.parametrize("track_id", SPIKE_TRACK_IDS)
def test_ui_track_json(out_dir: Path, specs: dict[str, SpikeTrack], track_id: str) -> None:
    spec = specs[track_id]
    raw = (out_dir / track_id / "ui" / "ui_track.json").read_bytes()
    assert raw.isascii()
    ui = json.loads(raw)
    number = SPIKE_TRACK_IDS.index(track_id) + 1
    assert ui["name"] == spec.name and ui["name"].startswith(f"DM T{number} ")
    assert SPIKE_TAG in ui["tags"]
    assert len(ui["description"]) > 40 and "PASS" in ui["description"]
    assert int(ui["pitboxes"]) == spec.pit_count >= 4
    assert int(ui["length"]) == round(spec.road.length)
    if track_id == "dm_t11_ai_loop":
        assert "circuit" in ui["tags"]
    if track_id in ("dm_t2_handedness", "dm_t12_ai_ab"):
        lat, lon = (float(v) for v in ui["geotags"])
        assert -90 <= lat <= 90 and -180 <= lon <= 180


@pytest.mark.parametrize("track_id", SPIKE_TRACK_IDS)
def test_small_data_files_parse(out_dir: Path, track_id: str) -> None:
    data = out_dir / track_id / "data"
    surfaces = read_ini(data / "surfaces.ini")
    keys = [surfaces[s]["KEY"] for s in surfaces.sections()]
    assert keys == ["ROAD", "GRASS", "GRAVEL"]
    cams = read_ini(data / "cameras.ini")
    count = int(cams["HEADER"]["CAMERA_COUNT"])
    assert count >= 1 and len(cams.sections()) == count + 1
    assert float(cams[f"CAMERA_{count - 1}"]["OUT_POINT"]) == 1.0
    assert read_ini(data / "groove.ini")["HEADER"]["GROOVES_NUMBER"] == "0"
    assert read_ini(data / "crew.ini")["HEADER"]["SIDE"] == "-1"
    assert "SUN_PITCH_ANGLE" in read_ini(data / "lighting.ini")["LIGHTING"]
    licence = (out_dir / track_id / "LICENSE-DATA.txt").read_text(encoding="utf-8")
    assert "Kunos" in licence and "synthetic test track" in licence


@pytest.mark.parametrize("track_id", SPIKE_TRACK_IDS)
def test_map_matches_the_road(out_dir: Path, specs: dict[str, SpikeTrack], track_id: str) -> None:
    info = read_map_ini(out_dir / track_id / "data" / "map.ini")
    img = Image.open(out_dir / track_id / "map.png")
    assert img.mode == "RGBA" and img.size == info.size_px
    alpha = np.asarray(img)[..., 3]
    road = specs[track_id].road
    for s in np.linspace(0.0, road.length, 25):
        px, py = info.world_to_pixel(road.at(float(s))[0])
        assert alpha[int(py), int(px)] == 255, f"road centre at s={s:.0f} m is not drawn"
    assert alpha[0, 0] == 0 and alpha[-1, -1] == 0  # transparent background


def test_t2_minimap_goes_up_then_left(out_dir: Path, specs: dict[str, SpikeTrack]) -> None:
    road = specs["dm_t2_handedness"].road
    info = read_map_ini(out_dir / "dm_t2_handedness" / "data" / "map.ini")
    start, corner, end = (info.world_to_pixel(road.at(s)[0]) for s in (0.0, 165.0, road.length))
    assert corner[1] < start[1] - 200  # up the image = north
    assert end[0] < corner[0] - 200  # then left = west
    assert abs(end[1] - corner[1]) < 40


# --------------------------------------------------------------------------------------------
# KN5 structure
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("track_id", SPIKE_TRACK_IDS)
def test_kn5_files_are_valid_and_round_trip(out_dir: Path, specs, track_id: str) -> None:
    for name in specs[track_id].kn5_files:
        data = (out_dir / track_id / name).read_bytes()
        kn5 = read_kn5(data)
        assert validate(kn5) == []
        assert kn5_to_bytes(kn5) == data
        assert kn5.version == (6 if track_id == "dm_t10_kn5v6" else 5)
        assert kn5.extra == 0
        assert kn5.root.name == name.removesuffix(".kn5")
        assert np.array_equal(kn5.root.transform, np.eye(4, dtype=np.float32))
        for mesh in meshes(kn5).values():
            assert 0 < mesh.vertex_count <= MAX_MESH_VERTICES
            assert kn5.materials[mesh.material_id].shader == "ksPerPixel"
        assert all(t.name.isascii() and t.data for t in kn5.textures)


@pytest.mark.parametrize("track_id", SPIKE_TRACK_IDS)
def test_ac_dummies_are_root_children(kn5s: dict[str, Kn5], specs, track_id: str) -> None:
    kn5 = kn5s[track_id]
    deep = [
        n.name
        for child in kn5.root.children
        if isinstance(child, Dummy)
        for n in Kn5(root=child).iter_nodes()
        if n is not child and n.name.startswith("AC_")
    ]
    assert deep == []
    dummies = root_dummies(kn5)
    pits = sorted(n for n in dummies if n.startswith("AC_PIT_"))
    starts = sorted(n for n in dummies if n.startswith("AC_START_"))
    assert len(pits) >= 4 and len(starts) >= 4
    assert set(pits) == {f"AC_PIT_{i}" for i in range(len(pits))}
    assert set(starts) == {f"AC_START_{i}" for i in range(len(starts))}
    assert "AC_HOTLAP_START_0" in dummies
    spawn_names = {n for n in dummies if n.startswith(("AC_PIT_", "AC_START_", "AC_HOTLAP_"))}
    assert spawn_names == set(specs[track_id].spawn_heights)
    for d in dummies.values():
        assert not d.children or not d.name.startswith("AC_")


@pytest.mark.parametrize("track_id", SPIKE_TRACK_IDS)
def test_spawns_face_along_the_road(kn5s: dict[str, Kn5], specs, track_id: str) -> None:
    spec = specs[track_id]
    road = spec.road
    for name, height in spec.spawn_heights.items():
        d = root_dummies(kn5s[track_id])[name]
        m = d.transform.astype(np.float64)
        p = position(d)
        s, off = road.nearest(p)
        on_road, fwd, left = road.at(s)
        assert off <= road.half_width - 1.0, f"{name} is not on the road"
        assert np.dot(m[2, :3], fwd) > 0.9999, f"{name} does not face down the road"
        assert np.allclose(m[0, :3], left_of(m[2, :3]), atol=1e-6)  # +X = driver's left
        assert np.allclose(m[1, :3], (0, 1, 0), atol=1e-6)
        assert p[1] - on_road[1] == pytest.approx(height, abs=1e-5)
        if track_id in AB_TRACKS:
            assert s + 2.5 < spec.gates["AC_AB_START"], f"{name} is not behind the start gate"
        if track_id == "dm_t11_ai_loop":
            assert road.length - 150.0 < s < road.length - 2.5  # behind start/finish


def test_t6_spawn_heights_and_arrows(kn5s: dict[str, Kn5], specs) -> None:
    spec, kn5 = specs["dm_t6_spawns"], kn5s["dm_t6_spawns"]
    dummies, mesh_by_name = root_dummies(kn5), meshes(kn5)
    expected = {"AC_PIT_0": 0.0, "AC_START_0": 1.0, "AC_HOTLAP_START_0": 2.0}
    for (name, height), paint in zip(
        expected.items(),
        ("paint_pit_arrow", "paint_start_arrow", "paint_hotlap_arrow"),
        strict=True,
    ):
        p = position(dummies[name])
        assert p[1] == pytest.approx(height, abs=1e-6)
        decal = mesh_by_name[paint].positions.astype(np.float64)
        s_spawn, _ = spec.road.nearest(p)
        s_decal, _ = spec.road.nearest(decal.mean(axis=0))
        assert s_decal == pytest.approx(s_spawn, abs=0.1)  # beside it
        _, fwd, left = spec.road.at(s_spawn)
        assert np.dot(p - decal.mean(axis=0), left) > 3.0  # car on the left, arrow on the right
        top = decal[mesh_by_name[paint].uvs[:, 1] == 0.0].mean(axis=0)  # image top edge
        assert np.dot(top - decal.mean(axis=0), fwd) > 1.0  # the arrow points forward


@pytest.mark.parametrize("track_id", GATE_TRACKS)
def test_gates_put_l_on_the_drivers_left(kn5s: dict[str, Kn5], specs, track_id: str) -> None:
    spec = specs[track_id]
    dummies = root_dummies(kn5s[track_id])
    reach = spec.road.half_width + GATE_EXTRA_M
    for prefix, s in spec.gates.items():
        lpos, rpos = position(dummies[f"{prefix}_L"]), position(dummies[f"{prefix}_R"])
        centre, _, left = spec.road.at(s)
        assert np.allclose((lpos + rpos) / 2, centre + (0, GATE_HEIGHT_M, 0), atol=1e-3)
        assert np.linalg.norm(lpos - rpos) == pytest.approx(2 * reach, abs=1e-3)
        side = np.dot(lpos - rpos, left) / (2 * reach)
        expected = -1.0 if spec.swapped_gates else 1.0
        assert side == pytest.approx(expected, abs=1e-4), prefix


def test_gate_sets(specs) -> None:
    ab = {"AC_AB_START": 50.0, "AC_TIME_0": 250.0, "AC_AB_FINISH": 450.0}
    assert specs["dm_t4_ab_timing"].gates == ab and not specs["dm_t4_ab_timing"].swapped_gates
    assert specs["dm_t5_swapped_gates"].gates == ab and specs["dm_t5_swapped_gates"].swapped_gates
    assert specs["dm_t4_ab_timing"].road.length - 450.0 == pytest.approx(50.0, abs=0.5)
    assert specs["dm_t11_ai_loop"].gates["AC_TIME_0"] == 0.0
    assert set(specs["dm_t12_ai_ab"].gates) == {"AC_AB_START", "AC_AB_FINISH"}
    for t in SPIKE_TRACK_IDS:
        if t not in GATE_TRACKS:
            assert specs[t].gates == {}


def test_t5_is_t4_with_swapped_gates(kn5s: dict[str, Kn5]) -> None:
    a, b = root_dummies(kn5s["dm_t4_ab_timing"]), root_dummies(kn5s["dm_t5_swapped_gates"])
    assert set(a) == set(b)
    for name in a:
        if name.endswith(("_L", "_R")):
            twin = name[:-1] + ("R" if name.endswith("L") else "L")
            assert np.allclose(position(a[name]), position(b[twin]))
        else:
            assert np.array_equal(a[name].transform, b[name].transform)
    assert meshes(kn5s["dm_t4_ab_timing"]).keys() == meshes(kn5s["dm_t5_swapped_gates"]).keys()


# --------------------------------------------------------------------------------------------
# Meshes: names, flags, geometry
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("track_id", SPIKE_TRACK_IDS)
def test_mesh_names_and_flags(out_dir: Path, specs, track_id: str) -> None:
    surface_keys = {"ROAD", "GRASS", "GRAVEL"}
    all_names: list[str] = []
    for file_name in specs[track_id].kn5_files:
        for name, mesh in meshes(read_kn5(out_dir / track_id / file_name)).items():
            all_names.append(name)
            physical = PHYSICAL_RE.match(name)
            if physical:
                assert physical.group(1) in surface_keys
                assert mesh.is_renderable and mesh.is_visible
            elif "WALL" in name:
                assert name.startswith("1WALL_")
                hidden = name in HIDDEN_WALLS
                assert name in HIDDEN_WALLS | VISIBLE_WALLS
                assert mesh.is_renderable != hidden and mesh.cast_shadows != hidden
                assert mesh.is_visible
            else:  # visual-only marker: no surface prefix anywhere in the name
                assert not SURFACE_ANYWHERE_RE.search(name), name
                assert mesh.is_renderable and mesh.is_visible
            if name.startswith("paint_"):
                assert not mesh.cast_shadows
    assert any(n.startswith("1ROAD") for n in all_names)
    if track_id != "dm_t8_models_ini":
        assert {"1GRASS_L", "1GRASS_R"} <= set(all_names)


@pytest.mark.parametrize("track_id", ["dm_t1_minimal", "dm_t11_ai_loop", "dm_t12_ai_ab"])
def test_road_and_grass_share_edges(kn5s: dict[str, Kn5], specs, track_id: str) -> None:
    road = specs[track_id].road
    by_name = meshes(kn5s[track_id])

    def rounded(points: np.ndarray) -> set[tuple[float, ...]]:
        return {tuple(p) for p in np.round(points.astype(np.float64), 4)}

    road_pts = rounded(by_name["1ROAD_main"].positions)
    for side in ("L", "R"):
        grass = by_name[f"1GRASS_{side}"].positions
        offsets = np.array([road.nearest(p)[1] for p in grass[::7]])
        assert offsets.max() == pytest.approx(road.half_width + GRASS_WIDTH_M, abs=0.05)
        inner = grass[np.isclose([road.nearest(p)[1] for p in grass], road.half_width, atol=1e-3)]
        assert len(inner) > 50 and rounded(inner) <= road_pts  # no crack between surfaces


def test_t3_walls(kn5s: dict[str, Kn5]) -> None:
    by_name = meshes(kn5s["dm_t3_walls"])
    for name, x in (("1WALL_L", -5.0), ("1WALL_R", 5.0)):
        w = by_name[name]
        assert not w.is_renderable and not w.cast_shadows
        p = w.positions
        assert np.allclose(p[:, 0], x, atol=1e-4)  # 1 m outside the 8 m road (north: left = -X)
        assert p[:, 1].min() == 0.0 and p[:, 1].max() == pytest.approx(2.5)
        assert p[:, 2].max() == pytest.approx(0.0) and p[:, 2].min() == pytest.approx(-250.0)
        assert {round(float(v), 3) for v in w.normals[:, 0]} == {-1.0, 1.0}  # double-sided
    vis = by_name["1WALL_VIS"]
    assert vis.is_renderable and vis.cast_shadows
    assert vis.positions[:, 2].max() == pytest.approx(-250.0)
    assert vis.positions[:, 2].min() == pytest.approx(-300.0)
    assert set(np.round(np.abs(vis.positions[:, 0]), 4)) == {5.0}


def test_t9_limits(kn5s: dict[str, Kn5]) -> None:
    kn5 = kn5s["dm_t9_limits"]
    by_name = meshes(kn5)
    roads = [m for n, m in by_name.items() if n.startswith("1ROAD")]
    assert len(roads) == 1 and roads[0].vertex_count == 65_535 == MAX_MESH_VERTICES
    assert int(roads[0].indices.max()) == 65_534
    order = [n.name for n in kn5.root.children]
    group = next(n for n in kn5.root.children if n.name == "GROUP_scenery")
    assert isinstance(group, Dummy) and len(group.children) >= 5
    assert all(isinstance(c, Mesh) and c.is_renderable for c in group.children)
    empty = next(n for n in kn5.root.children if n.name == "DM_EMPTY_MARKER")
    assert isinstance(empty, Dummy) and empty.children == []
    after = order.index("AFTER_EMPTY_post")
    assert after > order.index("DM_EMPTY_MARKER") > order.index("GROUP_scenery")
    assert after == len(order) - 1
    # Everything the car needs comes before the unusual nodes.
    first_unusual = order.index("GROUP_scenery")
    assert all(order.index(n) < first_unusual for n in order if n.startswith(("AC_", "1")))
    red = by_name["AFTER_EMPTY_post"]
    assert red.is_renderable and red.is_visible
    texture = kn5.materials[red.material_id].textures[0].texture_name
    assert texture == "post_red.png"
    lod = [m for n, m in by_name.items() if n.startswith("LOD60_post_")]
    assert len(lod) >= 10
    assert all(m.lod_out == 60.0 and m.lod_in == 0.0 and not m.cast_shadows for m in lod)
    assert all(m.lod_out == 0.0 for n, m in by_name.items() if not n.startswith("LOD60_"))


def test_t10_is_t1_as_kn5_v6(out_dir: Path, kn5s: dict[str, Kn5]) -> None:
    raw = (out_dir / "dm_t10_kn5v6" / "dm_t10_kn5v6.kn5").read_bytes()
    assert raw[:6] == b"sc6969" and raw[6:14] == (6).to_bytes(4, "little") + bytes(4)
    v6, v5 = kn5s["dm_t10_kn5v6"], kn5s["dm_t1_minimal"]
    assert (v6.version, v6.extra, v5.version) == (6, 0, 5)
    assert [t.data for t in v6.textures] == [t.data for t in v5.textures]
    assert [n.name for n in v6.iter_nodes()][1:] == [n.name for n in v5.iter_nodes()][1:]
    for a, b in zip(list(v6.iter_nodes())[1:], list(v5.iter_nodes())[1:], strict=True):
        if isinstance(a, Mesh):
            assert np.array_equal(a.positions, b.positions)
        else:
            assert np.array_equal(a.transform, b.transform)


def test_t7_lane_textures(kn5s: dict[str, Kn5]) -> None:
    kn5 = kn5s["dm_t7_textures"]
    by_name = meshes(kn5)
    textures = {t.name: t.data for t in kn5.textures}

    def lane_texture(lane: str) -> bytes:
        mat = kn5.materials[by_name[f"1ROAD_{lane}"].material_id]
        return textures[mat.textures[0].texture_name]

    assert texture_info(lane_texture("PNG")).startswith("PNG 1024x1024")
    dds = read_dds_info(lane_texture("DDS"))
    assert (dds.width, dds.height, dds.format, dds.mip_count) == (1024, 1024, "BC1", 11)
    assert texture_info(lane_texture("NPOT")).startswith("PNG 1000x600")
    # Lanes left to right (heading north, left = west = -X): PNG, DDS, NPOT; 4 m each.
    xs = {lane: by_name[f"1ROAD_{lane}"].positions[:, 0] for lane in ("PNG", "DDS", "NPOT")}
    assert (xs["PNG"].min(), xs["PNG"].max()) == (-6.0, -2.0)
    assert (xs["DDS"].min(), xs["DDS"].max()) == (-2.0, 2.0)
    assert (xs["NPOT"].min(), xs["NPOT"].max()) == (2.0, 6.0)
    assert by_name["1ROAD_PNG"].positions[:, 2].min() == pytest.approx(-600.0)
    for label in ("png", "dds", "npot"):
        assert f"paint_label_{label}" in by_name


def test_t8_models_ini(out_dir: Path, specs) -> None:
    track = out_dir / "dm_t8_models_ini"
    ini = read_ini(track / "models.ini")
    files = [ini[s]["FILE"] for s in ini.sections()]
    assert ini.sections() == [f"MODEL_{i}" for i in range(4)]
    assert files == specs["dm_t8_models_ini"].kn5_files
    assert files[0] == "dm_t8_models_ini.kn5"
    for s in ini.sections():
        assert ini[s]["POSITION"] == "0,0,0" and ini[s]["ROTATION"] == "0,0,0"
    main = read_kn5(track / files[0])
    assert set(meshes(main)) == {"1ROAD_pad", "paint_label_main"}
    assert all(n.name.startswith("AC_") for n in main.root.children if isinstance(n, Dummy))
    z_ranges = [(0.0, 60.0)]  # (north end, south end) as -z ranges: the pad spans z 60..0
    for k, name in enumerate(files[1:], start=1):
        chunk = read_kn5(track / name)
        assert not any(isinstance(n, Dummy) for n in chunk.root.children)
        by_name = meshes(chunk)
        assert {f"1ROAD_c{k}", f"1GRASS_L_c{k}", f"1GRASS_R_c{k}"} <= set(by_name)
        z = by_name[f"1ROAD_c{k}"].positions[:, 2]
        z_ranges.append((float(z.min()), float(z.max())))
        assert z.max() - z.min() == pytest.approx(100.0)
    pad_z = meshes(main)["1ROAD_pad"].positions[:, 2]
    assert (pad_z.min(), pad_z.max()) == (0.0, 60.0)
    # Consecutive pieces meet exactly (no gap, no overlap): chunk k starts where k-1 ended.
    assert [r[1] for r in z_ranges[1:]] == [0.0, -100.0, -200.0]


def test_t2_handedness_markers(kn5s: dict[str, Kn5], specs) -> None:
    spec = specs["dm_t2_handedness"]
    assert total_turn(spec.road.centre) == pytest.approx(90.0, abs=0.5)  # a LEFT turn
    by_name = meshes(kn5s["dm_t2_handedness"])
    arrow = by_name["paint_arrow_left"]
    p, uv = arrow.positions.astype(np.float64), arrow.uvs
    centre = p.mean(axis=0)
    s, _ = spec.road.nearest(centre)
    assert s < 150.0  # before the corner
    _, fwd, left = spec.road.at(s)
    top_left = p[(uv[:, 0] == 0) & (uv[:, 1] == 0)][0] - centre  # image top-left corner
    assert np.dot(top_left, fwd) > 0 and np.dot(top_left, left) > 0  # not mirrored
    sign = by_name["sign_N"]
    assert np.allclose(sign.normals, (0, 0, 1))  # faces south, towards the driver
    right_edge = sign.positions[sign.uvs[:, 0] == 1.0][:, 0]
    left_edge = sign.positions[sign.uvs[:, 0] == 0.0][:, 0]
    assert right_edge.min() > left_edge.max()  # image right = east = viewer's right
    assert sign.positions[:, 2].max() < spec.road.centre[:, 2].min() - 20  # beyond the corner


def test_ab_tracks_end_with_a_visible_wall(kn5s: dict[str, Kn5], specs) -> None:
    for track_id in AB_TRACKS:
        spec = specs[track_id]
        end_wall = meshes(kn5s[track_id])["1WALL_END"]
        assert end_wall.is_renderable
        s, _ = spec.road.nearest(end_wall.positions.astype(np.float64).mean(axis=0))
        assert s == pytest.approx(spec.road.length - 1.0, abs=0.01)
        assert s > spec.gates["AC_AB_FINISH"] + 40.0
        colours = {"post_ac_ab_start": "post_green", "post_ac_ab_finish": "post_red"}
        if "AC_TIME_0" in spec.gates:
            colours["post_ac_time_0"] = "post_yellow"
        kn5 = kn5s[track_id]
        for post_name, material in colours.items():
            assert kn5.materials[meshes(kn5)[post_name].material_id].name == material


# --------------------------------------------------------------------------------------------
# AI lines
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("track_id", AI_TRACKS)
def test_ai_files(out_dir: Path, specs, track_id: str) -> None:
    spec = specs[track_id]
    raw = (out_dir / track_id / "ai" / "fast_lane.ai").read_bytes()
    assert (out_dir / track_id / "data" / "ideal_line.ai").read_bytes() == raw
    line = read_ai(raw)
    assert line.version == 7 and line.n_points > 400
    assert np.array_equal(line.ids, np.arange(line.n_points))
    assert np.allclose(line.segment_lengths(), 1.0, atol=0.01)
    assert np.all(np.isfinite(line.speed)) and line.speed.max() <= DEFAULT_MAX_SPEED_MPS + 1e-3
    assert np.allclose(line.side_left, spec.road.half_width - 0.5)
    assert np.allclose(line.side_right, spec.road.half_width - 0.5)
    assert np.all(line.radius > 0) and np.all(line.direction == 1.0)  # only left turns
    assert np.all((line.gas == 0) | (line.brake == 0))
    offsets = [spec.road.nearest(p)[1] for p in line.positions]
    assert max(offsets) < 0.01  # on the road centreline everywhere
    if track_id == "dm_t11_ai_loop":
        assert line.is_closed() and line.grid is None
        assert line.speed.min() > 15.0  # never stops on a loop
        assert line.lengths[-1] + line.closing_gap == pytest.approx(spec.road.length, rel=1e-3)
        s0, _ = spec.road.nearest(line.positions[0])
        assert s0 == pytest.approx(0.0, abs=1e-3)  # spline starts at start/finish
    else:
        assert not line.is_closed() and line.grid is not None
        assert line.speed[0] == 0.0 and line.speed[-1] == 0.0
        s_end, _ = spec.road.nearest(line.positions[-1])
        assert s_end == pytest.approx(spec.gates["AC_AB_FINISH"] + 100.0, abs=0.01)
        assert spec.road.length - s_end >= 4.0  # still on the road at its end
        assert spec.road.nearest(line.positions[0])[0] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------------------------
# Determinism, zip, options, CLI
# --------------------------------------------------------------------------------------------


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_build_is_deterministic(build, tmp_path: Path) -> None:
    out, _, zip_path = build
    again = tmp_path / "again"
    build_spike_tracks(again / "tracks", zip_path=again / "m0.zip")
    first, second = tree_bytes(out), tree_bytes(again / "tracks")
    assert first.keys() == second.keys()
    assert [k for k in first if first[k] != second[k]] == []
    assert (again / "m0.zip").read_bytes() == zip_path.read_bytes()


def test_zip_contents(build) -> None:
    out, _, zip_path = build
    files = tree_bytes(out)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert names == sorted(files)
        assert {n.split("/")[0] for n in names} == set(SPIKE_TRACK_IDS)
        for info in zf.infolist():
            assert info.date_time == (2026, 1, 1, 0, 0, 0)
            assert zf.read(info) == files[info.filename]
        assert zf.testzip() is None


def test_only_builds_the_requested_tracks(tmp_path: Path) -> None:
    built = build_spike_tracks(
        tmp_path, only=["dm_t10_kn5v6", "dm_t1_minimal"], zip_path=tmp_path / "z.zip"
    )
    assert [p.name for p in built] == ["dm_t1_minimal", "dm_t10_kn5v6"]
    with zipfile.ZipFile(tmp_path / "z.zip") as zf:
        assert {n.split("/")[0] for n in zf.namelist()} == {"dm_t1_minimal", "dm_t10_kn5v6"}
    with pytest.raises(ValueError, match="dm_t99"):
        build_spike_tracks(tmp_path, only=["dm_t99_nope"])
    with pytest.raises(ValueError):
        make_spike_track("nope")


def test_rebuild_replaces_only_its_own_folders(tmp_path: Path) -> None:
    (tmp_path / "dm_t1_minimal").mkdir()
    stale = tmp_path / "dm_t1_minimal" / "stale.kn5"
    stale.write_bytes(b"old")
    other = tmp_path / "some_other_track" / "keep.txt"
    other.parent.mkdir()
    other.write_text("keep")
    build_spike_tracks(tmp_path, only=["dm_t1_minimal"])
    assert not stale.exists() and other.read_text() == "keep"


def test_cli_spike(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from driftmaps.cli import app

    out, zip_path = tmp_path / "out", tmp_path / "m0.zip"
    args = ["spike", "--out", str(out), "--only", "dm_t1_minimal", "--zip", str(zip_path)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "built" in result.output and zip_path.is_file()
    assert (out / "dm_t1_minimal" / "dm_t1_minimal.kn5").is_file()


def test_previews_are_cm_sized(out_dir: Path) -> None:
    for track_id in SPIKE_TRACK_IDS:
        ui = out_dir / track_id / "ui"
        assert Image.open(ui / "preview.png").size == (355, 200)
        assert Image.open(ui / "outline.png").size == (365, 192)


# --------------------------------------------------------------------------------------------
# Oracle cross-check (independent GPL parser, dev-only; skipped when not downloaded)
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oracle() -> types.ModuleType:
    if not ORACLE_PATH.exists():
        pytest.skip("oracle missing: run `uv run python tools/fetch_oracles.py`")
    spec = importlib.util.spec_from_file_location("nazariykul_kn5_oracle_spike", ORACLE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def oracle_names(node) -> list[str]:
    names = [node.name]
    for child in node.children:
        names += oracle_names(child)
    return names


@pytest.mark.parametrize("track_id", SPIKE_TRACK_IDS)
def test_oracle_parses_every_kn5(oracle, out_dir: Path, specs, track_id: str) -> None:
    for name in specs[track_id].kn5_files:
        path = out_dir / track_id / name
        theirs = oracle.parse(path)
        ours = read_kn5(path)
        assert theirs.warnings == []
        assert theirs.version == ours.version
        assert oracle_names(theirs.root) == [n.name for n in ours.iter_nodes()]
        assert theirs.textures == {t.name: bytes(t.data) for t in ours.textures}
        # Spot-check textures decode as images (PNG through Pillow; DDS header via our reader).
        for tex_name, data in theirs.textures.items():
            if tex_name.endswith(".png"):
                Image.open(io.BytesIO(data)).verify()
