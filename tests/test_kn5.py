"""Tests for driftmaps.ac.kn5: writer, reader, validation, dump and cross-checks.

Two optional extras, skipped when unavailable:
- Oracle cross-check: an independent GPL parser (nazariyKul kn5.py). Fetch it with
  `uv run python tools/fetch_oracles.py`; it lands in tools/oracles/ (gitignored).
- Real files: a few small KN5s made with ksEditor / Content Manager by Custom Shaders Patch
  config authors (versions 5 and 6, DDS/PNG/JPEG textures, nested dummies). They are downloaded
  at test time from a pinned commit into .cache/kn5_samples/ (gitignored) and never committed.
  Set DRIFTMAPS_KN5_SAMPLES to a folder (e.g. your assettocorsa/content) to also round-trip
  every .kn5 under it, including Kunos skinned meshes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import struct
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from driftmaps.ac.coords import frame_transform, translation_transform
from driftmaps.ac.kn5 import (
    SKINNED_VERTEX_DTYPE,
    VERTEX_DTYPE,
    Kn5Error,
    Kn5ValidationError,
    dump,
    kn5_to_bytes,
    loose_bounding_sphere,
    read_kn5,
    texture_info,
    validate,
    write_kn5,
)
from driftmaps.ac.model import (
    MAX_MESH_VERTICES,
    BlendMode,
    Bone,
    DepthMode,
    Dummy,
    Kn5,
    Material,
    MaterialProperty,
    Mesh,
    Node,
    SkinnedMesh,
    Texture,
    TextureMapping,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ORACLE_PATH = REPO_ROOT / "tools" / "oracles" / "nazariykul_kn5.py"
CACHE_DIR = REPO_ROOT / ".cache" / "kn5_samples"

# Real KN5 files from ac-custom-shaders-patch/acc-extension-config, pinned by commit + SHA-256.
_SAMPLES_COMMIT = "963a683cf2366eebeb4a5fc0d0796cf01f07d638"
_SAMPLES_BASE = (
    "https://raw.githubusercontent.com/ac-custom-shaders-patch/acc-extension-config/"
    f"{_SAMPLES_COMMIT}/"
)
REAL_SAMPLES = {
    # name: (path in repo, sha256, version)
    "silverstoneDisplay": (
        "config/tracks/silverstoneDisplay.kn5",
        "435262f5e264ccb7baa2886c384fad6afa7524f9a8e5b612196678fad4301acd",
        6,
    ),
    "gallardo_dashboard": (
        "config/cars/kunos/ks_lamborghini_gallardo_sl__dashboard.kn5",
        "021d313e20518e825f4be3f3da951ee7bb3d51733a70a29ed2924f341ad39605",
        5,
    ),
    "monaco_1966_lits": (
        "config/tracks/monaco_1966__lits.kn5",
        "0fdee5733fa7cfcccdc71762521709f424379cd131adddb5b52ffc595a0826ee",
        5,
    ),
    "nordschleife_ferriswheel": (
        "config/tracks/ks_nordschleife__lights_ferriswheel.kn5",
        "082541fbf495f7485dd9d977bd1e2e8adb856e9041292d8fcb6de577acfaca05",
        6,
    ),
    "brno_lights": (
        "config/tracks/brno_circuit__lights.kn5",
        "d04ccc140632d3604d61f529eeb014ea6fb2f0ec9bd075091bac2f242d06430c",
        6,
    ),
}


# --------------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------------


def png_bytes(width: int, height: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (width, height), (90, 90, 90, 255)).save(buf, "PNG")
    return buf.getvalue()


def dds_bytes(width: int, height: int, mips: int, fourcc: bytes = b"DXT1") -> bytes:
    """A DDS file with a valid header and zeroed BC1 blocks (content is irrelevant here)."""
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000  # CAPS|HEIGHT|WIDTH|PIXELFORMAT|MIPMAPCOUNT
    header = struct.pack("<4s7I44x", b"DDS ", 124, flags, height, width, 0, 0, mips)
    pixel_format = struct.pack("<2I4s5I", 32, 0x4, fourcc, 0, 0, 0, 0, 0)
    caps = struct.pack("<5I", 0x1000 | 0x400000 | 0x8, 0, 0, 0, 0)
    blocks = sum(max(1, (width >> i) // 4) * max(1, (height >> i) // 4) for i in range(mips))
    return header + pixel_format + caps + bytes(8 * blocks)


def grid_mesh(name: str, nx: int, nz: int, material_id: int = 0, spacing: float = 1.0) -> Mesh:
    """A flat nx-by-nz vertex grid at y = 0 with CCW (seen from above) triangles."""
    xs, zs = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(nz, dtype=np.float32))
    n = nx * nz
    positions = np.zeros((n, 3), np.float32)
    positions[:, 0] = xs.ravel() * spacing
    positions[:, 2] = zs.ravel() * spacing
    normals = np.tile(np.array([0, 1, 0], np.float32), (n, 1))
    tangents = np.tile(np.array([1, 0, 0], np.float32), (n, 1))
    uvs = positions[:, [0, 2]] * np.float32(0.125)
    i = np.arange(nz - 1)[:, None] * nx + np.arange(nx - 1)[None, :]
    a, b, c, d = i, i + 1, i + nx, i + nx + 1
    # Viewed from +Y: cross(p1 - p0, p2 - p0) must point up, so go a -> c -> b.
    tris = np.stack([a, c, b, b, c, d], axis=-1).reshape(-1)
    return Mesh(
        name=name,
        positions=positions,
        normals=normals,
        uvs=uvs,
        tangents=tangents,
        indices=tris.astype(np.uint16),
        material_id=material_id,
    )


def skinned_mesh() -> SkinnedMesh:
    base = grid_mesh("tmp", 3, 2)
    n = base.vertex_count
    weights = np.zeros((n, 4), np.float32)
    weights[:, 0] = 0.75
    weights[:, 1] = 0.25
    bone_indices = np.full((n, 4), -1.0, np.float32)
    bone_indices[:, 0] = 0
    bone_indices[:, 1] = 1
    return SkinnedMesh(
        name="FLAG_skin",
        positions=base.positions,
        normals=base.normals,
        uvs=base.uvs,
        tangents=base.tangents,
        bone_weights=weights,
        bone_indices=bone_indices,
        indices=base.indices,
        material_id=1,
        bones=[
            Bone("flag_bone_0", translation_transform(np.array([0.0, 2.0, 0.0]))),
            Bone("flag_bone_1", translation_transform(np.array([1.0, 2.0, 0.0]))),
        ],
        cast_shadows=False,
        is_transparent=True,
        layer=1,
        lod_in=0.0,
        lod_out=250.0,
    )


def make_kn5(version: int = 5, extra: int = 0) -> Kn5:
    """A small KN5 exercising every field: textures, materials with vector props, nested
    dummies with transforms, meshes with every flag, an explicit sphere and a skinned mesh."""
    road = grid_mesh("1ROAD", 4, 6, material_id=0)
    wall = grid_mesh("1WALL_L", 2, 6, material_id=1)
    wall.positions[:, 1] = wall.positions[:, 0]  # stand it up: y = x
    wall.positions[:, 0] = -4.5
    wall.cast_shadows = False
    wall.is_renderable = False
    wall.lod_in = 0.0
    wall.lod_out = 500.0
    sign = grid_mesh("sign", 2, 2, material_id=1)
    sign.is_visible = False
    sign.is_transparent = True
    sign.layer = 3
    sign.lod_in = 10.0
    sign.active = False
    sign.bounding_sphere = ((0.5, 0.0, 0.5), 2.0)
    group = Dummy(
        "GROUP_signs",
        transform=frame_transform(np.array([5.0, 0.5, -20.0]), np.array([1.0, 0.0, -1.0])),
        children=[sign, Dummy("empty_inactive", active=False)],
    )
    root = Dummy(
        "track_root",
        children=[
            road,
            wall,
            Dummy(
                "AC_START_0", frame_transform(np.array([0.0, 1.0, -10.0]), np.array([0, 0, 1.0]))
            ),
            Dummy("AC_PIT_0", translation_transform(np.array([0.0, 0.5, -5.0]))),
            group,
            skinned_mesh(),
        ],
    )
    textures = [
        Texture("asphalt.dds", dds_bytes(8, 8, 4)),
        Texture("marks.png", png_bytes(4, 4), active=False),
    ]
    materials = [
        Material(
            "road",
            shader="ksPerPixel",
            properties=[
                MaterialProperty("ksAmbient", 0.5),
                MaterialProperty("ksDiffuse", 0.25),
                MaterialProperty("ksSpecularEXP", 10.0),
                MaterialProperty("ksEmissive", 0.0, c=(1.0, 0.5, 0.25)),
            ],
            textures=[TextureMapping("txDiffuse", 0, "asphalt.dds")],
        ),
        Material(
            "marks",
            shader="ksPerPixelAT",
            blend_mode=BlendMode.ALPHA_BLEND,
            alpha_tested=True,
            depth_mode=DepthMode.READ_ONLY,
            properties=[
                MaterialProperty("vec", 1.5, b=(2.0, 3.0), c=(4.0, 5.0, 6.0), d=(7, 8, 9, 10))
            ],
            textures=[
                TextureMapping("txDiffuse", 0, "marks.png"),
                TextureMapping("txNormal", 1, "asphalt.dds"),
            ],
        ),
    ]
    return Kn5(root=root, textures=textures, materials=materials, version=version, extra=extra)


def walk(node: Node, path: str = "") -> list[tuple[str, Node]]:
    path = f"{path}/{node.name}"
    out = [(path, node)]
    if isinstance(node, Dummy):
        for child in node.children:
            out += walk(child, path)
    return out


def find(kn5: Kn5, name: str) -> Node:
    return next(n for n in kn5.iter_nodes() if n.name == name)


# --------------------------------------------------------------------------------------------
# Equality helpers
# --------------------------------------------------------------------------------------------


def _f32(value: float) -> float:
    return float(np.float32(value))


def assert_arrays(a: np.ndarray, b: np.ndarray, dtype: type) -> None:
    assert b.dtype == dtype
    assert b.flags.writeable and b.flags.c_contiguous
    assert np.array_equal(np.asarray(a, dtype=dtype), b)


def assert_kn5_equal(expected: Kn5, actual: Kn5) -> None:
    """Field-by-field equality. Meshes whose expected sphere is None must come back with the
    writer's loose sphere."""
    assert actual.version == expected.version
    assert actual.extra == (expected.extra if expected.version > 5 else 0)
    assert [(t.name, bytes(t.data), t.active) for t in actual.textures] == [
        (t.name, bytes(t.data), t.active) for t in expected.textures
    ]
    assert len(actual.materials) == len(expected.materials)
    for exp, act in zip(expected.materials, actual.materials, strict=True):
        assert (act.name, act.shader, act.alpha_tested) == (exp.name, exp.shader, exp.alpha_tested)
        assert isinstance(act.blend_mode, BlendMode) and act.blend_mode == exp.blend_mode
        assert isinstance(act.depth_mode, DepthMode) and act.depth_mode == exp.depth_mode
        assert [(p.name, p.a, tuple(p.b), tuple(p.c), tuple(p.d)) for p in act.properties] == [
            (
                p.name,
                _f32(p.a),
                tuple(map(_f32, p.b)),
                tuple(map(_f32, p.c)),
                tuple(map(_f32, p.d)),
            )
            for p in exp.properties
        ]
        assert act.textures == exp.textures
    exp_nodes, act_nodes = walk(expected.root), walk(actual.root)
    assert [p for p, _ in act_nodes] == [p for p, _ in exp_nodes]
    for (path, exp), (_, act) in zip(exp_nodes, act_nodes, strict=True):
        assert type(act) is type(exp), path
        assert act.name == exp.name and act.active == exp.active, path
        if isinstance(exp, Dummy):
            assert_arrays(exp.transform, act.transform, np.float32)
            assert len(act.children) == len(exp.children)
            continue
        for field in ("positions", "normals", "uvs", "tangents"):
            assert_arrays(getattr(exp, field), getattr(act, field), np.float32)
        assert_arrays(exp.indices, act.indices, np.uint16)
        for field in ("material_id", "cast_shadows", "is_visible", "is_transparent", "layer"):
            assert getattr(act, field) == getattr(exp, field), (path, field)
        assert act.lod_in == _f32(exp.lod_in) and act.lod_out == _f32(exp.lod_out), path
        if isinstance(exp, Mesh):
            assert act.is_renderable == exp.is_renderable, path
            want = exp.bounding_sphere or loose_bounding_sphere(exp.positions)
            (cx, cy, cz), r = want
            assert act.bounding_sphere == ((_f32(cx), _f32(cy), _f32(cz)), _f32(r)), path
        else:
            assert_arrays(exp.bone_weights, act.bone_weights, np.float32)
            assert_arrays(exp.bone_indices, act.bone_indices, np.float32)
            assert [b.name for b in act.bones] == [b.name for b in exp.bones]
            for eb, ab in zip(exp.bones, act.bones, strict=True):
                assert_arrays(eb.transform, ab.transform, np.float32)


# --------------------------------------------------------------------------------------------
# Round trips and layout
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("version,extra", [(5, 0), (6, 0), (6, 1234)])
def test_round_trip_all_fields(version: int, extra: int) -> None:
    kn5 = make_kn5(version, extra)
    assert validate(kn5) == []
    assert_kn5_equal(kn5, read_kn5(kn5_to_bytes(kn5)))


@pytest.mark.parametrize("version", [5, 6])
def test_write_read_write_is_byte_identical(version: int) -> None:
    first = kn5_to_bytes(make_kn5(version))
    second = kn5_to_bytes(read_kn5(first))  # validated again on the way out
    assert second == first


def test_v6_header_has_extra_and_v5_does_not() -> None:
    v5 = kn5_to_bytes(make_kn5(5))
    v6 = kn5_to_bytes(make_kn5(6, extra=-7))
    assert v5[:6] == v6[:6] == b"sc6969"
    assert struct.unpack_from("<i", v5, 6)[0] == 5
    assert struct.unpack_from("<i", v5, 10)[0] == 2  # texture count follows the version
    assert struct.unpack_from("<ii", v6, 6) == (6, -7)
    assert struct.unpack_from("<i", v6, 14)[0] == 2
    assert len(v6) == len(v5) + 4
    back = read_kn5(v6)
    assert (back.version, back.extra) == (6, -7)


def test_golden_bytes_for_minimal_file() -> None:
    """Field order restated by hand from AcTools Kn5Writer.cs, independently of kn5.py."""
    positions = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]], np.float32)
    mesh = Mesh(
        name="1ROAD",
        positions=positions,
        normals=np.tile(np.float32([0, 1, 0]), (3, 1)),
        uvs=np.zeros((3, 2), np.float32),
        tangents=np.tile(np.float32([1, 0, 0]), (3, 1)),
        indices=np.array([0, 1, 2], np.uint16),
        material_id=0,
        lod_out=100.0,
    )
    root = Dummy(
        "root", transform=translation_transform(np.array([1.0, 2.0, 3.0])), children=[mesh]
    )
    tex = b"\x89PNG fake"
    kn5 = Kn5(
        root=root,
        textures=[Texture("t.png", tex)],
        materials=[
            Material(
                "m",
                properties=[MaterialProperty("ksDiffuse", 0.5)],
                textures=[TextureMapping("txDiffuse", 0, "t.png")],
            )
        ],
    )

    def s(text: str) -> bytes:
        return struct.pack("<i", len(text)) + text.encode()

    expected = b"sc6969" + struct.pack("<i", 5)
    expected += struct.pack("<i", 1) + struct.pack("<i", 1) + s("t.png")
    expected += struct.pack("<I", len(tex)) + tex
    expected += struct.pack("<i", 1) + s("m") + s("ksPerPixel") + struct.pack("<B?i", 0, False, 0)
    expected += struct.pack("<i", 1) + s("ksDiffuse") + struct.pack("<10f", 0.5, *[0.0] * 9)
    expected += struct.pack("<i", 1) + s("txDiffuse") + struct.pack("<i", 0) + s("t.png")
    expected += struct.pack("<i", 1) + s("root") + struct.pack("<i?", 1, True)
    expected += struct.pack("<16f", 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1)
    expected += struct.pack("<i", 2) + s("1ROAD") + struct.pack("<i?", 0, True)
    expected += struct.pack("<???", True, True, False) + struct.pack("<I", 3)
    for p in positions:
        expected += struct.pack("<3f3f2f3f", *p, 0, 1, 0, 0, 0, 1, 0, 0)
    expected += struct.pack("<I", 3) + struct.pack("<3H", 0, 1, 2)
    centre, radius = (0.5, 0.0, 0.5), float(np.float32(np.sqrt(0.5)))
    if radius < np.sqrt(0.5):
        radius = float(np.nextafter(np.float32(radius), np.float32(1)))
    expected += struct.pack("<IIff3ff?", 0, 0, 0.0, 100.0, *centre, radius, True)
    assert kn5_to_bytes(kn5) == expected


def test_write_kn5_to_disk(tmp_path: Path) -> None:
    target = tmp_path / "content" / "tracks" / "dm_test" / "dm_test.kn5"
    kn5 = make_kn5()
    assert write_kn5(kn5, target) == target
    assert target.read_bytes() == kn5_to_bytes(kn5)
    assert sorted(p.name for p in target.parent.iterdir()) == ["dm_test.kn5"]  # no .tmp left
    assert_kn5_equal(kn5, read_kn5(target))
    assert_kn5_equal(kn5, read_kn5(str(target)))


def test_vertex_dtypes_match_on_disk_sizes() -> None:
    assert VERTEX_DTYPE.itemsize == 44
    assert SKINNED_VERTEX_DTYPE.itemsize == 76


def test_float64_input_is_written_as_float32() -> None:
    kn5 = make_kn5()
    road = find(kn5, "1ROAD")
    road.positions = road.positions.astype(np.float64) + 0.1
    road.indices = road.indices.astype(np.int64)
    assert validate(kn5) == []
    back = find(read_kn5(kn5_to_bytes(kn5)), "1ROAD")
    assert back.positions.dtype == np.float32 and back.indices.dtype == np.uint16
    assert np.array_equal(back.positions, road.positions.astype(np.float32))


def test_loose_bounding_sphere_contains_all_vertices() -> None:
    rng = np.random.default_rng(1)
    pts = (rng.normal(size=(500, 3)) * [300.0, 20.0, 900.0] + [1234.5, 80.0, -4321.0]).astype(
        np.float32
    )
    centre, radius = loose_bounding_sphere(pts)
    aabb_centre = ((pts.min(0).astype(float) + pts.max(0)) / 2).astype(np.float32)
    assert np.array_equal(np.float32(centre), aabb_centre)
    assert np.float32(radius) == radius  # representable exactly as stored
    dist = np.sqrt(((pts.astype(np.float64) - np.array(centre)) ** 2).sum(1))
    assert dist.max() <= radius
    assert radius - dist.max() < 1e-3
    assert loose_bounding_sphere(np.zeros((0, 3), np.float32)) == ((0.0, 0.0, 0.0), 0.0)


def test_iterative_reader_and_writer_handle_deep_trees() -> None:
    root = Dummy("root")
    node = root
    for i in range(3000):  # deeper than Python's default recursion limit
        child = Dummy(f"d{i}")
        node.children.append(child)
        node = child
    node.children.append(grid_mesh("leaf", 2, 2))
    kn5 = Kn5(root=root, materials=[Material("m")])
    back = read_kn5(kn5_to_bytes(kn5))
    assert sum(1 for _ in back.iter_nodes()) == 3002


# --------------------------------------------------------------------------------------------
# Big meshes and speed
# --------------------------------------------------------------------------------------------


def test_65535_vertex_mesh_round_trips_fast() -> None:
    mesh = grid_mesh("1ROAD_big", 255, 257)  # 255 * 257 = 65,535
    assert mesh.vertex_count == MAX_MESH_VERTICES == 65535
    assert int(mesh.indices.max()) == 65534
    kn5 = Kn5(root=Dummy("root", children=[mesh]), materials=[Material("road")])
    start = time.perf_counter()
    data = kn5_to_bytes(kn5)
    back = read_kn5(data)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"write+read took {elapsed:.3f} s"
    assert_kn5_equal(kn5, back)


def test_65536_vertex_mesh_is_rejected() -> None:
    mesh = grid_mesh("too_big", 256, 256)
    mesh.indices = np.array([0, 1, 2], np.uint16)  # indices can't even address the last vertex
    kn5 = Kn5(root=Dummy("root", children=[mesh]), materials=[Material("road")])
    with pytest.raises(Kn5ValidationError, match="65,536 vertices exceeds"):
        kn5_to_bytes(kn5)


# --------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------


def _set(name: str, field: str, value):
    def mutate(kn5: Kn5) -> None:
        setattr(find(kn5, name), field, value)

    return mutate


def _road(fn):
    def mutate(kn5: Kn5) -> None:
        fn(find(kn5, "1ROAD"))

    return mutate


def _nan_at(array_field: str, index: tuple[int, ...], value: float = np.nan):
    def fn(mesh: Mesh) -> None:
        getattr(mesh, array_field)[index] = value

    return _road(fn)


def _append_child(node_factory):
    def mutate(kn5: Kn5) -> None:
        kn5.root.children.append(node_factory(kn5))

    return mutate


VALIDATION_CASES = {
    "version_unwritable": (lambda k: setattr(k, "version", 7), "version 7 is not writable"),
    "extra_lost_in_v5": (lambda k: setattr(k, "extra", 3), "only stored when version > 5"),
    "extra_not_int32": (
        lambda k: (setattr(k, "version", 6), setattr(k, "extra", 2**31)),
        "extra must be an int32",
    ),
    "texture_name_non_ascii": (
        lambda k: setattr(k.textures[0], "name", "asfalté.dds"),
        "is not ASCII",
    ),
    "texture_name_duplicate": (
        lambda k: setattr(k.textures[1], "name", "asphalt.dds"),
        "duplicate texture name",
    ),
    "texture_empty": (lambda k: setattr(k.textures[0], "data", b""), "texture has no data"),
    "texture_data_not_bytes": (lambda k: setattr(k.textures[0], "data", "x"), "must be bytes"),
    "material_name_non_ascii": (
        lambda k: setattr(k.materials[0], "name", "röad"),
        "material[0] 'röad': name 'röad' is not ASCII",
    ),
    "material_name_duplicate": (
        lambda k: setattr(k.materials[1], "name", "road"),
        "duplicate material name",
    ),
    "material_name_empty": (lambda k: setattr(k.materials[0], "name", ""), "name is empty"),
    "shader_non_ascii": (
        lambda k: setattr(k.materials[0], "shader", "ksPerPixél"),
        "shader 'ksPerPixél' is not ASCII",
    ),
    "blend_mode_invalid": (lambda k: setattr(k.materials[0], "blend_mode", 5), "blend_mode 5"),
    "depth_mode_invalid": (lambda k: setattr(k.materials[0], "depth_mode", -1), "depth_mode -1"),
    "property_name_non_ascii": (
        lambda k: setattr(k.materials[0].properties[0], "name", "ksÄmbient"),
        "is not ASCII",
    ),
    "property_nan": (
        lambda k: setattr(k.materials[0].properties[0], "a", float("nan")),
        "a must be 1 finite number(s)",
    ),
    "property_vector_wrong_length": (
        lambda k: setattr(k.materials[0].properties[0], "c", (1.0, 2.0)),
        "c must be 3 finite number(s)",
    ),
    "mapping_slot_non_ascii": (
        lambda k: setattr(k.materials[0].textures[0], "slot_name", "txÐiffuse"),
        "is not ASCII",
    ),
    "mapping_missing_texture": (
        lambda k: setattr(k.materials[0].textures[0], "texture_name", "nope.dds"),
        "refers to texture 'nope.dds', which is not embedded",
    ),
    "node_name_non_ascii": (_set("AC_PIT_0", "name", "AC_PÏT_0"), "is not ASCII"),
    "dummy_transform_nan": (
        lambda k: find(k, "AC_PIT_0").transform.__setitem__((3, 0), np.nan),
        "transform has NaN/inf",
    ),
    "dummy_transform_shape": (
        _set("AC_PIT_0", "transform", np.eye(3, dtype=np.float32)),
        "must be a numeric 4x4 matrix",
    ),
    "material_id_out_of_range": (_set("1ROAD", "material_id", 2), "material_id 2 is not a valid"),
    "material_id_negative": (_set("1ROAD", "material_id", -1), "material_id -1 is not a valid"),
    "material_id_not_int": (_set("1ROAD", "material_id", 0.0), "material_id must be an int"),
    "index_out_of_range": (
        _road(lambda m: m.indices.__setitem__(0, 24)),
        "1 index(es) out of range [0, 24)",
    ),
    "index_count_not_multiple_of_3": (
        _road(lambda m: setattr(m, "indices", m.indices[:-1])),
        "is not a multiple of 3",
    ),
    "indices_float_dtype": (
        _road(lambda m: setattr(m, "indices", m.indices.astype(np.float32))),
        "indices dtype float32 is not an integer type",
    ),
    "indices_2d": (
        _road(lambda m: setattr(m, "indices", m.indices.reshape(-1, 3))),
        "indices has shape",
    ),
    "positions_wrong_shape": (
        _road(lambda m: setattr(m, "positions", m.positions[:, :2].copy())),
        "positions has shape (24, 2), expected (N, 3)",
    ),
    "positions_not_numeric": (
        _road(lambda m: setattr(m, "positions", m.positions.astype(str))),
        "is not numeric",
    ),
    "normals_row_mismatch": (
        _road(lambda m: setattr(m, "normals", m.normals[:-1])),
        "normals has 23 rows but positions has 24",
    ),
    "uvs_wrong_columns": (
        _road(lambda m: setattr(m, "uvs", np.zeros((24, 3), np.float32))),
        "uvs has shape (24, 3), expected (N, 2)",
    ),
    "tangents_missing_rows": (
        _road(lambda m: setattr(m, "tangents", np.zeros((0, 3), np.float32))),
        "tangents has 0 rows",
    ),
    "positions_nan": (_nan_at("positions", (3, 1)), "positions has 1 row(s) with NaN/inf"),
    "normals_inf": (_nan_at("normals", (0, 0), np.inf), "normals has 1 row(s) with NaN/inf"),
    "uvs_nan": (_nan_at("uvs", (5, 0)), "uvs has 1 row(s) with NaN/inf"),
    "tangents_nan": (_nan_at("tangents", (2, 2)), "tangents has 1 row(s) with NaN/inf"),
    "positions_beyond_float32": (
        _road(lambda m: setattr(m, "positions", m.positions.astype(np.float64) * 1e38)),
        "beyond float32 range",
    ),
    "lod_nan": (_set("1ROAD", "lod_out", float("nan")), "lod_out must be a finite number"),
    "lod_negative": (_set("1ROAD", "lod_in", -5.0), "lod_in -5.0 is negative"),
    "layer_negative": (_set("1ROAD", "layer", -1), "layer must be an int in uint32 range"),
    "sphere_too_small": (
        _set("1ROAD", "bounding_sphere", ((1.5, 0.0, 2.5), 1.0)),
        "does not contain every vertex",
    ),
    "sphere_malformed": (_set("1ROAD", "bounding_sphere", (1.0, 2.0)), "bounding_sphere must be"),
    "sphere_nan": (
        _set("1ROAD", "bounding_sphere", ((0.0, np.nan, 0.0), 5.0)),
        "bounding_sphere has non-finite values",
    ),
    "empty_mesh": (
        _road(
            lambda m: (
                setattr(m, "positions", np.zeros((0, 3), np.float32)),
                setattr(m, "normals", np.zeros((0, 3), np.float32)),
                setattr(m, "uvs", np.zeros((0, 2), np.float32)),
                setattr(m, "tangents", np.zeros((0, 3), np.float32)),
                setattr(m, "indices", np.zeros(0, np.uint16)),
            )
        ),
        "mesh has no vertices",
    ),
    "skinned_bone_index_out_of_range": (
        lambda k: find(k, "FLAG_skin").bone_indices.__setitem__((0, 0), 2.0),
        "1 bone index(es) outside [-1, 2)",
    ),
    "skinned_weights_shape": (
        _set("FLAG_skin", "bone_weights", np.zeros((6, 3), np.float32)),
        "bone_weights has shape (6, 3), expected (N, 4)",
    ),
    "skinned_bone_name_non_ascii": (
        lambda k: setattr(find(k, "FLAG_skin").bones[0], "name", "böne"),
        "is not ASCII",
    ),
    "root_not_dummy": (lambda k: setattr(k, "root", find(k, "1ROAD")), "root must be a Dummy"),
    "cycle": (lambda k: find(k, "GROUP_signs").children.append(k.root), "appears twice"),
    "unsupported_node_type": (
        _append_child(lambda k: "not a node"),
        "unsupported node type str",
    ),
}


@pytest.mark.parametrize("case", sorted(VALIDATION_CASES))
def test_validation_error(case: str) -> None:
    mutate, expected = VALIDATION_CASES[case]
    kn5 = make_kn5()
    mutate(kn5)
    problems = validate(kn5)
    assert any(expected in p for p in problems), problems
    with pytest.raises(Kn5ValidationError) as info:
        kn5_to_bytes(kn5)
    assert isinstance(info.value, ValueError)
    assert info.value.problems == problems
    assert expected in str(info.value)


def test_validation_lists_every_problem_with_location() -> None:
    kn5 = make_kn5()
    find(kn5, "1ROAD").material_id = 9
    find(kn5, "sign").indices[0] = 99
    kn5.materials[1].textures[0].texture_name = "gone.png"
    problems = validate(kn5)
    assert len(problems) == 3
    assert any(p.startswith("mesh 'track_root/1ROAD'") for p in problems)
    assert any(p.startswith("mesh 'track_root/GROUP_signs/sign'") for p in problems)
    assert any(p.startswith("material[1] 'marks' mapping[0] 'txDiffuse'") for p in problems)


def test_check_false_skips_validation() -> None:
    kn5 = make_kn5()
    kn5.materials[0].textures[0].texture_name = "external.dds"
    with pytest.raises(Kn5ValidationError):
        kn5_to_bytes(kn5)
    back = read_kn5(kn5_to_bytes(kn5, check=False))
    assert back.materials[0].textures[0].texture_name == "external.dds"


# --------------------------------------------------------------------------------------------
# Reader robustness
# --------------------------------------------------------------------------------------------


def test_reader_rejects_bad_magic() -> None:
    with pytest.raises(Kn5Error, match="not a KN5 file"):
        read_kn5(b"version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(Kn5Error, match="not a KN5 file"):
        read_kn5(b"")


@pytest.mark.parametrize("version", [0, 7, -1])
def test_reader_rejects_unknown_versions(version: int) -> None:
    data = bytearray(kn5_to_bytes(make_kn5()))
    struct.pack_into("<i", data, 6, version)
    with pytest.raises(Kn5Error, match=f"unsupported KN5 version {version}"):
        read_kn5(bytes(data))


def test_reader_reports_truncation_at_every_offset() -> None:
    data = kn5_to_bytes(make_kn5(6))
    for cut in range(len(data)):
        with pytest.raises(Kn5Error):
            read_kn5(data[:cut])
    small = Kn5(root=Dummy("root", children=[grid_mesh("1ROAD", 4, 6)]), materials=[Material("m")])
    data = kn5_to_bytes(small)
    at = data.index(b"1ROAD") + 5 + 5 + 3 + 4 + 100  # name, head, flags, count, some vertices
    pattern = r"implausible vertex count 24.*truncated or corrupt.*node 'root/1ROAD'"
    with pytest.raises(Kn5Error, match=pattern):
        read_kn5(data[:at])
    with pytest.raises(Kn5Error, match=r"file is truncated: mesh tail.*node 'root/1ROAD'"):
        read_kn5(data[:-10])


def test_reader_rejects_trailing_bytes() -> None:
    with pytest.raises(Kn5Error, match="3 unexpected trailing bytes"):
        read_kn5(kn5_to_bytes(make_kn5()) + b"\0\0\0")


def test_reader_rejects_unknown_node_class() -> None:
    data = bytearray(kn5_to_bytes(make_kn5()))
    at = data.index(b"AC_PIT_0") - 8  # class int32, then name length int32
    assert struct.unpack_from("<i", data, at)[0] == 1
    struct.pack_into("<i", data, at, 9)
    with pytest.raises(Kn5Error, match="unknown node class 9 for node 'AC_PIT_0'"):
        read_kn5(bytes(data))


def test_reader_rejects_unknown_blend_mode() -> None:
    data = bytearray(kn5_to_bytes(make_kn5()))
    at = data.index(b"ksPerPixel") + len(b"ksPerPixel")
    data[at] = 7
    with pytest.raises(Kn5Error, match=r"unknown blend mode 7.*material\[0\] 'road'"):
        read_kn5(bytes(data))


def test_reader_rejects_mesh_with_children() -> None:
    data = bytearray(kn5_to_bytes(make_kn5()))
    at = data.index(b"1ROAD") + 5  # child count follows the name
    struct.pack_into("<i", data, at, 1)
    with pytest.raises(Kn5Error, match="mesh node '1ROAD' declares 1 children"):
        read_kn5(bytes(data))


def test_reader_rejects_implausible_counts() -> None:
    data = bytearray(kn5_to_bytes(make_kn5()))
    struct.pack_into("<i", data, 10, 2_000_000_000)  # texture count
    with pytest.raises(Kn5Error, match="implausible texture count"):
        read_kn5(bytes(data))


def test_reader_never_raises_anything_but_kn5error_on_corruption() -> None:
    data = kn5_to_bytes(make_kn5(6))
    rng = np.random.default_rng(1234)
    for _ in range(400):
        corrupt = bytearray(data)
        for pos in rng.integers(0, len(data), size=int(rng.integers(1, 4))):
            corrupt[pos] = int(rng.integers(0, 256))
        try:
            read_kn5(bytes(corrupt))
        except Kn5Error:
            pass


def test_reader_keeps_non_utf8_names_byte_exact() -> None:
    kn5 = make_kn5()
    data = kn5_to_bytes(kn5)
    patched = data.replace(b"\x08\x00\x00\x00AC_PIT_0", b"\x08\x00\x00\x00AC_PIT_\xff")
    back = read_kn5(patched)
    assert any("is not ASCII" in p for p in validate(back))
    assert kn5_to_bytes(back, check=False) == patched


# --------------------------------------------------------------------------------------------
# Dump
# --------------------------------------------------------------------------------------------


def test_texture_info() -> None:
    assert texture_info(dds_bytes(8, 8, 4)) == "DDS 8x8 DXT1 mips=4"
    assert texture_info(dds_bytes(16, 4, 1, b"DXT5")) == "DDS 16x4 DXT5 mips=1"
    assert texture_info(png_bytes(3, 5)) == "PNG 3x5 RGBA 8-bit"
    assert texture_info(b"\xff\xd8\xff\xe0rest") == "JPEG"
    assert texture_info(b"DDS short") == "DDS (truncated header)"
    assert texture_info(b"") == "empty"
    assert texture_info(b"\x01\x02").startswith("unknown format")


def test_dump_contains_key_information(tmp_path: Path) -> None:
    path = write_kn5(make_kn5(6, extra=0), tmp_path / "t.kn5")
    text = dump(path)
    for needle in [
        "version 6  extra 0",
        "textures: 2",
        "asphalt.dds",
        "DDS 8x8 DXT1 mips=4",
        "marks.png",
        "PNG 4x4 RGBA 8-bit  INACTIVE",
        "materials: 2",
        "road  shader=ksPerPixel  blend=OPAQUE  alpha_tested=no  depth=NORMAL",
        "marks  shader=ksPerPixelAT  blend=ALPHA_BLEND  alpha_tested=yes  depth=READ_ONLY",
        "ksAmbient=0.5; ksDiffuse=0.25",
        "ksEmissive=0 c=(1, 0.5, 0.25)",
        "txDiffuse (slot 0) -> asphalt.dds",
        "txNormal (slot 1) -> asphalt.dds",
        "+ track_root  [dummy]  identity  (6 children)",
        "+ AC_START_0  [dummy]  pos=(0.000, 1.000, -10.000)  fwd=(0.000, 0.000, 1.000)",
        "- 1ROAD  [mesh]  24 verts  30 tris  material=road(0)",
        "shadows=1 visible=1 transparent=0 renderable=1  layer=0  lod=0..0",
        "- 1WALL_L  [mesh]  12 verts  10 tris  material=marks(1)",
        "shadows=0 visible=1 transparent=0 renderable=0  layer=0  lod=0..500",
        "- sign  [mesh]",
        "INACTIVE",
        "- FLAG_skin  [skinned mesh]  6 verts  4 tris  material=marks(1)",
        "bones=2",
        "totals: 5 dummies, 3 meshes, 1 skinned meshes, 46 vertices, 46 triangles",
        "validation: ok",
    ]:
        assert needle in text, f"{needle!r} missing from dump:\n{text}"
    group_line = next(line for line in text.splitlines() if "GROUP_signs" in line)
    assert "pos=(5.000, 0.500, -20.000)  fwd=(0.707, 0.000, -0.707)  (2 children)" in group_line
    # A dummy under a moved parent also shows where it ends up in the world.
    assert (
        "+ empty_inactive  [dummy]  identity  world_pos=(5.000, 0.500, -20.000)  "
        "world_fwd=(0.707, 0.000, -0.707)  (0 children)  INACTIVE"
    ) in text
    # Indentation follows depth.
    assert "\n      - sign  [mesh]" in text


def test_dump_without_meshes_and_with_problems(tmp_path: Path) -> None:
    kn5 = make_kn5()
    kn5.materials[0].textures[0].texture_name = "external.dds"
    path = write_kn5(kn5, tmp_path / "t.kn5", check=False)
    text = dump(path, include_meshes=False)
    assert "[mesh]" not in text and "[skinned mesh]" not in text
    assert "+ AC_PIT_0  [dummy]" in text
    assert "external.dds  [not embedded]" in text
    assert "validation: 1 problem(s)" in text
    assert dump(path.read_bytes()).startswith("KN5 <bytes>")


def test_cli_kn5_dump(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from driftmaps.cli import app

    path = write_kn5(make_kn5(), tmp_path / "t.kn5")
    result = CliRunner().invoke(app, ["kn5-dump", str(path), "--no-meshes"])
    assert result.exit_code == 0, result.output
    assert "version 5" in result.output and "[mesh]" not in result.output


# --------------------------------------------------------------------------------------------
# Oracle cross-check (independent GPL parser, dev-only)
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oracle() -> types.ModuleType:
    if not ORACLE_PATH.exists():
        pytest.skip("oracle missing: run `uv run python tools/fetch_oracles.py`")
    spec = importlib.util.spec_from_file_location("nazariykul_kn5_oracle", ORACLE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_matches_oracle(ours: Kn5, theirs) -> None:
    """Compare our parse with the oracle's. The oracle flips V (u, 1 - v) and keys materials'
    properties and mappings by name."""
    assert theirs.warnings == []
    assert theirs.version == ours.version
    assert theirs.textures == {t.name: bytes(t.data) for t in ours.textures}
    for t, o in zip(theirs.materials, ours.materials, strict=True):
        assert (t.name, t.shader, t.blend_mode, t.alpha_test, t.depth_mode) == (
            o.name,
            o.shader,
            int(o.blend_mode),
            o.alpha_tested,
            int(o.depth_mode),
        )
        assert t.properties == {p.name: p.a for p in o.properties}
        assert t.vectors == {
            p.name: {"float2": tuple(p.b), "float3": tuple(p.c), "float4": tuple(p.d)}
            for p in o.properties
        }
        assert t.textures == {m.slot_name: m.texture_name for m in o.textures}
    stack = [(theirs.root, ours.root)]
    while stack:
        t, o = stack.pop()
        assert (t.name, t.active) == (o.name, o.active)
        if isinstance(o, Dummy):
            assert t.kind == 1
            assert np.array_equal(np.array(t.matrix, np.float32), o.transform)
            assert len(t.children) == len(o.children)
            stack.extend(zip(t.children, o.children, strict=True))
            continue
        assert t.kind == (3 if isinstance(o, SkinnedMesh) else 2) and t.skinned == (t.kind == 3)
        m = t.mesh
        assert np.array_equal(np.array(m.positions, np.float32).reshape(-1, 3), o.positions)
        assert np.array_equal(np.array(m.normals, np.float32).reshape(-1, 3), o.normals)
        flipped = o.uvs.astype(np.float64)
        flipped[:, 1] = 1.0 - flipped[:, 1]
        assert np.array_equal(np.array(m.uvs, np.float64).reshape(-1, 2), flipped)
        assert list(m.indices) == o.indices.tolist()
        assert m.material_id == o.material_id
        assert m.visible == o.is_visible
        assert m.renderable == (o.is_renderable if isinstance(o, Mesh) else True)


@pytest.mark.parametrize("version", [5, 6])
def test_oracle_parses_our_output(oracle, tmp_path: Path, version: int) -> None:
    kn5 = make_kn5(version)
    path = write_kn5(kn5, tmp_path / "ours.kn5")
    assert_matches_oracle(read_kn5(path), oracle.parse(path))


def test_oracle_parses_our_65535_vertex_mesh(oracle, tmp_path: Path) -> None:
    kn5 = Kn5(root=Dummy("root", children=[grid_mesh("big", 255, 257)]), materials=[Material("m")])
    path = write_kn5(kn5, tmp_path / "big.kn5")
    assert_matches_oracle(read_kn5(path), oracle.parse(path))


# --------------------------------------------------------------------------------------------
# Real files (downloaded at test time, never committed)
# --------------------------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def real_sample(name: str) -> Path:
    rel, sha, _ = REAL_SAMPLES[name]
    target = CACHE_DIR / Path(rel).name
    if target.exists() and _sha256(target.read_bytes()) == sha:
        return target
    url = _SAMPLES_BASE + rel
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        pytest.skip(f"offline? could not download {url}: {exc}")
    assert _sha256(data) == sha, f"{url} changed upstream (pinned commit?)"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


@pytest.mark.parametrize("name", sorted(REAL_SAMPLES))
def test_real_file_reads_and_rewrites_byte_identical(name: str) -> None:
    path = real_sample(name)
    data = path.read_bytes()
    kn5 = read_kn5(path)
    assert kn5.version == REAL_SAMPLES[name][2]
    assert isinstance(kn5.root, Dummy)
    assert kn5_to_bytes(kn5, check=False) == data
    text = dump(path)
    assert f"version {kn5.version}" in text and "totals:" in text


@pytest.mark.parametrize("name", sorted(REAL_SAMPLES))
def test_real_file_matches_oracle(oracle, name: str) -> None:
    path = real_sample(name)
    assert_matches_oracle(read_kn5(path), oracle.parse(path))


def test_real_file_details() -> None:
    """Spot checks that show we decode real data, not just round-trip bytes."""
    monaco = read_kn5(real_sample("monaco_1966_lits"))
    assert any("DX10/BC7_UNORM" in texture_info(t.data) for t in monaco.textures)
    assert any(isinstance(n, Mesh) and n.lod_out > 0 for n in monaco.iter_nodes())
    wheel = read_kn5(real_sample("nordschleife_ferriswheel"))
    dummies = [n for n in wheel.iter_nodes() if isinstance(n, Dummy)]
    assert len(dummies) > 50
    assert any(not np.array_equal(d.transform, np.eye(4)) for d in dummies)
    for d in dummies:  # rotation rows of a rigid transform stay orthonormal-ish
        assert abs(d.transform[3, 3] - 1.0) < 1e-6
    brno = read_kn5(real_sample("brno_lights"))
    kinds = {texture_info(t.data).split()[0] for t in brno.textures if t.data}
    assert {"PNG", "JPEG"} <= kinds


def _user_samples() -> list[Path]:
    root = os.environ.get("DRIFTMAPS_KN5_SAMPLES")
    if not root:
        return []
    return sorted(Path(root).rglob("*.kn5"))


@pytest.mark.skipif(
    not os.environ.get("DRIFTMAPS_KN5_SAMPLES"), reason="DRIFTMAPS_KN5_SAMPLES unset"
)
def test_user_kn5_files_round_trip() -> None:
    """Every .kn5 under $DRIFTMAPS_KN5_SAMPLES (e.g. assettocorsa/content) must parse and
    rewrite byte for byte. Files with a non-KN5 magic (e.g. git-lfs pointers) are skipped."""
    failures = []
    paths = _user_samples()
    for path in paths:
        data = path.read_bytes()
        if not data.startswith(b"sc6969"):
            continue
        try:
            if kn5_to_bytes(read_kn5(data), check=False) != data:
                failures.append(f"{path}: rewrite differs")
        except Kn5Error as exc:
            failures.append(str(exc))
    assert not failures, "\n".join(failures[:20])
