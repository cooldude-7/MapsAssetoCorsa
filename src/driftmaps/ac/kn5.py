"""KN5 writer, reader, validator and text dump (Assetto Corsa's 3D model container).

This is an independent implementation of the layout used by gro-ove/actools (Content Manager,
MS-PL) at commit e5e5c94; no AcTools code is copied. Field order was checked against:

- `AcTools/Kn5File/Kn5Writer.cs`: header, texture, material and node records.
- `AcTools/Kn5File/Kn5Basic.cs` (`Factory.Kn5Reader`): `ReadHeader`, `ReadTexture`,
  `ReadMaterial`, `ReadNode`, and the byte counts in `SkipMaterial` / `SkipNode`.
- `AcTools/ExtendedBinaryWriter.cs`: a string is an int32 byte length plus UTF-8 bytes (no
  terminator); a `Mat4x4` is 16 float32 in M11, M12, ... M44 order.

See docs/research/ac_format.md §4 for the annotated layout. Everything is little-endian:

    "sc6969", int32 version, [int32 extra, only if version > 5]
    int32 textureCount; per texture:
        int32 active (1), string name, uint32 size, byte[size] (a raw DDS or PNG file)
    int32 materialCount; per material:
        string name, string shader, uint8 blendMode, bool alphaTested,
        int32 depthMode (absent if version <= 4: ASSUMED, see below),
        int32 propertyCount; per property: string name, float a, float2 b, float3 c, float4 d
        int32 mappingCount; per mapping: string slotName, int32 slot, string textureName
    the root node, then its children, recursively (depth-first pre-order). Per node:
        int32 class (1 dummy, 2 mesh, 3 skinned mesh), string name, int32 childCount, bool active
        class 1: float32[16] transform (row-major; translation in floats 12-14)
        class 2: bool castShadows, bool isVisible, bool isTransparent,
                 uint32 vertexCount, vertexCount x 44 bytes (pos3, normal3, uv2, tangent3),
                 uint32 indexCount, uint16[indexCount],
                 uint32 materialId, uint32 layer, float lodIn, float lodOut,
                 float3 boundingSphereCentre, float boundingSphereRadius, bool isRenderable
        class 3: bool castShadows, bool isVisible, bool isTransparent,
                 uint32 boneCount; per bone: string name, float32[16] transform
                 uint32 vertexCount, vertexCount x 76 bytes (44 as above + float4 weights
                 + float4 bone indices stored as floats), uint32 indexCount, uint16[indexCount],
                 uint32 materialId, uint32 layer, 8 bytes (AcTools reads lodIn, lodOut floats)

Public API: `kn5_to_bytes`, `write_kn5`, `read_kn5`, `validate`, `loose_bounding_sphere`,
`texture_info`, `dump`.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import NoReturn

import numpy as np

from driftmaps.ac.model import (
    KN5_MAGIC,
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

__all__ = [
    "NODE_CLASS_DUMMY",
    "NODE_CLASS_MESH",
    "NODE_CLASS_SKINNED_MESH",
    "SKINNED_VERTEX_DTYPE",
    "VERTEX_DTYPE",
    "WRITABLE_VERSIONS",
    "Kn5Error",
    "Kn5ValidationError",
    "dump",
    "kn5_to_bytes",
    "loose_bounding_sphere",
    "read_kn5",
    "texture_info",
    "validate",
    "write_kn5",
]

NODE_CLASS_DUMMY = 1
NODE_CLASS_MESH = 2
NODE_CLASS_SKINNED_MESH = 3

# Versions the writer accepts. CM writes 5 (`CommonAcConsts.Kn5ActualVersion`); 6 adds the
# `extra` int32 after the version (KinG-InFeT writes 6 with extra = 0).
WRITABLE_VERSIONS = (5, 6)
# Versions the reader accepts. Kunos files are 5 or 6. Versions <= 4 are read without the
# material depthMode int32: that comes from the nazariyKul parser (GPL, facts only). AcTools
# always reads depthMode. ASSUMED; we have never seen a v4 file.
MIN_READ_VERSION = 1
MAX_READ_VERSION = 6

# One vertex of a class-2 mesh: exactly the 44-byte on-disk layout (no padding).
VERTEX_DTYPE = np.dtype(
    [
        ("position", "<f4", (3,)),
        ("normal", "<f4", (3,)),
        ("uv", "<f4", (2,)),
        ("tangent", "<f4", (3,)),
    ]
)
# One vertex of a class-3 skinned mesh: 44 bytes as above + weights + bone indices = 76 bytes.
SKINNED_VERTEX_DTYPE = np.dtype(
    [
        ("position", "<f4", (3,)),
        ("normal", "<f4", (3,)),
        ("uv", "<f4", (2,)),
        ("tangent", "<f4", (3,)),
        ("weights", "<f4", (4,)),
        ("bone_indices", "<f4", (4,)),
    ]
)
assert VERTEX_DTYPE.itemsize == 44 and SKINNED_VERTEX_DTYPE.itemsize == 76

_I32 = struct.Struct("<i")
_U32 = struct.Struct("<I")
_MATRIX = struct.Struct("<16f")
_PROPERTY_VALUES = struct.Struct("<10f")  # a, b[2], c[3], d[4]
_MATERIAL_FLAGS = struct.Struct("<BB")  # blendMode, alphaTested
_NODE_HEAD = struct.Struct("<iB")  # childCount, active (follows class + name)
_MESH_FLAGS = struct.Struct("<BBB")  # castShadows, isVisible, isTransparent
# materialId, layer, lodIn, lodOut, sphere centre xyz, sphere radius, isRenderable = 33 bytes,
# matching AcTools SkipNode's "+ 33" after the indices.
_MESH_TAIL = struct.Struct("<IIff3ffB")
# materialId, layer, then 8 bytes AcTools reads as lodIn, lodOut = 16 bytes ("+ 16" in SkipNode).
_SKINNED_TAIL = struct.Struct("<IIff")
assert _MESH_TAIL.size == 33 and _SKINNED_TAIL.size == 16

# Smallest possible records, used to reject absurd counts before allocating anything.
_MIN_TEXTURE_RECORD = 12  # active + name length + size
_MIN_PROPERTY_RECORD = 4 + _PROPERTY_VALUES.size
_MIN_MAPPING_RECORD = 12  # slot name length + slot + texture name length
_MIN_NODE_RECORD = 4 + 4 + _NODE_HEAD.size + _MESH_FLAGS.size + 4 + 4 + 4 + _SKINNED_TAIL.size
_MIN_BONE_RECORD = 4 + _MATRIX.size

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


class Kn5Error(ValueError):
    """The bytes are not a KN5 file this reader understands (bad magic, unsupported version,
    unknown node class, truncated data, trailing bytes...)."""


class Kn5ValidationError(ValueError):
    """`validate()` found problems, so the writer refused to write. `.problems` lists them."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        listing = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"KN5 failed validation ({len(self.problems)} problem(s)):\n{listing}")


# --------------------------------------------------------------------------------------------
# Bounding sphere
# --------------------------------------------------------------------------------------------


def loose_bounding_sphere(positions: np.ndarray) -> tuple[tuple[float, float, float], float]:
    """Loose culling sphere: centre = AABB centre, radius = max distance to a vertex.

    The values are rounded the way they are stored (float32) and the radius is rounded up, so
    the stored sphere still contains every vertex. An empty mesh gets ((0, 0, 0), 0).
    """
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if p.shape[0] == 0:
        return (0.0, 0.0, 0.0), 0.0
    centre = ((p.min(axis=0) + p.max(axis=0)) * 0.5).astype(np.float32).astype(np.float64)
    radius = float(np.sqrt(((p - centre) ** 2).sum(axis=1).max()))
    radius32 = np.float32(radius)
    if float(radius32) < radius:
        radius32 = np.nextafter(radius32, np.float32(np.inf))
    return (float(centre[0]), float(centre[1]), float(centre[2])), float(radius32)


# --------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------


def _is_int(value: object) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool | np.bool_)


def _is_finite_number(value: object) -> bool:
    """True for a real number that stays finite when stored as float32."""
    if not isinstance(value, int | float | np.integer | np.floating) or isinstance(value, bool):
        return False
    try:
        with np.errstate(over="ignore"):
            return bool(np.isfinite(np.float32(value)))
    except OverflowError:
        return False


def _enum_name(enum_type: type[BlendMode] | type[DepthMode], value: object) -> str | None:
    """The member name for a valid enum value (int or member), else None."""
    if not _is_int(value):
        return None
    try:
        return enum_type(int(value)).name
    except ValueError:
        return None


def _check_name(
    problems: list[str], where: str, what: str, value: object, *, required: bool
) -> None:
    if not isinstance(value, str):
        problems.append(f"{where}: {what} must be a str, got {type(value).__name__}")
        return
    if not value.isascii():
        problems.append(f"{where}: {what} {value!r} is not ASCII")
    if required and not value:
        problems.append(f"{where}: {what} is empty")


def _check_float_array(
    problems: list[str], where: str, label: str, value: object, cols: int, rows: int | None
) -> np.ndarray | None:
    """Check an (N, cols) float array. Returns it (as an ndarray) when its shape is usable."""
    arr = np.asarray(value)
    if arr.dtype.kind not in "fiu":
        problems.append(f"{where}: {label} dtype {arr.dtype} is not numeric (expected float32)")
        return None
    if arr.ndim != 2 or arr.shape[1] != cols:
        problems.append(f"{where}: {label} has shape {arr.shape}, expected (N, {cols})")
        return None
    if rows is not None and arr.shape[0] != rows:
        problems.append(f"{where}: {label} has {arr.shape[0]} rows but positions has {rows}")
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        bad = ~np.isfinite(arr.astype(np.float32))
    if bad.any():
        problems.append(
            f"{where}: {label} has {int(bad.any(axis=1).sum())} row(s) with NaN/inf "
            "(or values beyond float32 range)"
        )
    return arr


def _check_matrix(problems: list[str], where: str, label: str, value: object) -> None:
    arr = np.asarray(value)
    if arr.dtype.kind not in "fiu" or arr.shape != (4, 4):
        problems.append(
            f"{where}: {label} must be a numeric 4x4 matrix, got {arr.dtype} {arr.shape}"
        )
        return
    with np.errstate(over="ignore", invalid="ignore"):
        if not np.isfinite(arr.astype(np.float32)).all():
            problems.append(f"{where}: {label} has NaN/inf values")


def _check_geometry(
    problems: list[str], where: str, node: Mesh | SkinnedMesh, n_materials: int
) -> np.ndarray | None:
    """Checks shared by Mesh and SkinnedMesh. Returns the positions array if usable."""
    positions = _check_float_array(problems, where, "positions", node.positions, 3, None)
    n = None if positions is None else int(positions.shape[0])
    _check_float_array(problems, where, "normals", node.normals, 3, n)
    _check_float_array(problems, where, "uvs", node.uvs, 2, n)
    _check_float_array(problems, where, "tangents", node.tangents, 3, n)

    if n is not None:
        if n > MAX_MESH_VERTICES:
            problems.append(
                f"{where}: {n:,} vertices exceeds the uint16 index limit of "
                f"{MAX_MESH_VERTICES:,}; split the mesh"
            )
        if n == 0:
            # ASSUMED: D3D11 cannot create a zero-sized vertex buffer, so AC would likely fail.
            problems.append(f"{where}: mesh has no vertices")

    indices = np.asarray(node.indices)
    if indices.dtype.kind not in "iu":
        problems.append(f"{where}: indices dtype {indices.dtype} is not an integer type (uint16)")
    elif indices.ndim != 1:
        problems.append(f"{where}: indices has shape {indices.shape}, expected (M,)")
    else:
        if indices.size % 3 != 0:
            problems.append(
                f"{where}: index count {indices.size} is not a multiple of 3 (triangle list)"
            )
        if indices.size == 0:
            problems.append(f"{where}: mesh has no triangles")
        elif n is not None:
            lo, hi = int(indices.min()), int(indices.max())
            if lo < 0 or hi >= n:
                bad = int(((indices < 0) | (indices >= n)).sum())
                problems.append(
                    f"{where}: {bad} index(es) out of range [0, {n}) (min {lo}, max {hi})"
                )

    if not _is_int(node.material_id):
        problems.append(f"{where}: material_id must be an int, got {node.material_id!r}")
    elif not 0 <= node.material_id < n_materials:
        problems.append(
            f"{where}: material_id {node.material_id} is not a valid index "
            f"(the KN5 has {n_materials} material(s))"
        )
    if not _is_int(node.layer) or not 0 <= node.layer <= _UINT32_MAX:
        problems.append(f"{where}: layer must be an int in uint32 range, got {node.layer!r}")
    for label in ("lod_in", "lod_out"):
        value = getattr(node, label)
        if not _is_finite_number(value):
            problems.append(f"{where}: {label} must be a finite number, got {value!r}")
        elif value < 0:
            problems.append(f"{where}: {label} {value} is negative (0 means no limit)")
    return positions


def _check_mesh(problems: list[str], where: str, mesh: Mesh, n_materials: int) -> None:
    positions = _check_geometry(problems, where, mesh, n_materials)
    sphere = mesh.bounding_sphere
    if sphere is None:
        return
    try:
        (cx, cy, cz), radius = sphere
        values = (cx, cy, cz, radius)
    except (TypeError, ValueError):
        problems.append(f"{where}: bounding_sphere must be ((x, y, z), radius), got {sphere!r}")
        return
    if not all(_is_finite_number(v) for v in values):
        problems.append(f"{where}: bounding_sphere has non-finite values: {sphere!r}")
        return
    if radius < 0:
        problems.append(f"{where}: bounding_sphere radius {radius} is negative")
        return
    if positions is not None and positions.shape[0] > 0:
        p = positions.astype(np.float64)
        far = float(np.sqrt(((p - np.array([cx, cy, cz], dtype=np.float64)) ** 2).sum(1).max()))
        if far > radius * (1 + 1e-5) + 1e-3:
            problems.append(
                f"{where}: bounding_sphere radius {radius:.4f} does not contain every vertex "
                f"(farthest is {far:.4f} from the centre); geometry would be culled too early"
            )


def _check_skinned_mesh(
    problems: list[str], where: str, mesh: SkinnedMesh, n_materials: int
) -> None:
    positions = _check_geometry(problems, where, mesh, n_materials)
    n = None if positions is None else int(positions.shape[0])
    _check_float_array(problems, where, "bone_weights", mesh.bone_weights, 4, n)
    bone_indices = _check_float_array(problems, where, "bone_indices", mesh.bone_indices, 4, n)
    for i, bone in enumerate(mesh.bones):
        _check_name(problems, f"{where} bone[{i}]", "name", bone.name, required=False)
        _check_matrix(problems, f"{where} bone[{i}] {bone.name!r}", "transform", bone.transform)
    if bone_indices is not None and bone_indices.size:
        with np.errstate(invalid="ignore"):
            bad = (bone_indices < -1) | (bone_indices >= len(mesh.bones))
        if bad.any():
            problems.append(
                f"{where}: {int(bad.sum())} bone index(es) outside [-1, {len(mesh.bones)})"
            )


def validate(kn5: Kn5) -> list[str]:
    """Return every problem that would make this KN5 unsafe to write (empty list = OK).

    Checks: writable version, ASCII names, unique texture and material names, texture mappings
    that point at embedded textures, valid enums and material ids, array shapes and dtypes,
    finite values, <= 65,535 vertices per mesh, indices in range with a count divisible by 3,
    and that a given bounding sphere contains its mesh.
    """
    problems: list[str] = []
    add = problems.append

    if not _is_int(kn5.version) or kn5.version not in WRITABLE_VERSIONS:
        add(f"version {kn5.version!r} is not writable; use one of {WRITABLE_VERSIONS}")
    if not _is_int(kn5.extra) or not -(2**31) <= kn5.extra <= _INT32_MAX:
        add(f"extra must be an int32, got {kn5.extra!r}")
    elif _is_int(kn5.version) and kn5.version <= 5 and kn5.extra != 0:
        add(f"extra={kn5.extra} is only stored when version > 5 and would be lost")

    texture_names: dict[str, int] = {}
    for i, tex in enumerate(kn5.textures):
        where = f"texture[{i}] {tex.name!r}"
        _check_name(problems, where, "name", tex.name, required=True)
        if tex.name in texture_names:
            add(f"{where}: duplicate texture name (also texture[{texture_names[tex.name]}])")
        else:
            texture_names[tex.name] = i
        if not isinstance(tex.data, bytes | bytearray | memoryview):
            add(f"{where}: data must be bytes, got {type(tex.data).__name__}")
        elif len(tex.data) == 0:
            add(f"{where}: texture has no data")
        elif len(tex.data) > _INT32_MAX:
            add(f"{where}: texture is larger than 2 GiB")

    material_names: dict[str, int] = {}
    for i, mat in enumerate(kn5.materials):
        where = f"material[{i}] {mat.name!r}"
        _check_name(problems, where, "name", mat.name, required=True)
        if mat.name in material_names:
            # Content Manager keys materials by name, so duplicates break its showroom.
            add(f"{where}: duplicate material name (also material[{material_names[mat.name]}])")
        else:
            material_names[mat.name] = i
        _check_name(problems, where, "shader", mat.shader, required=True)
        if _enum_name(BlendMode, mat.blend_mode) is None:
            known = [int(b) for b in BlendMode]
            add(f"{where}: blend_mode {mat.blend_mode!r} is not one of {known}")
        if _enum_name(DepthMode, mat.depth_mode) is None:
            known = [int(d) for d in DepthMode]
            add(f"{where}: depth_mode {mat.depth_mode!r} is not one of {known}")
        for j, prop in enumerate(mat.properties):
            pwhere = f"{where} property[{j}] {prop.name!r}"
            _check_name(problems, pwhere, "name", prop.name, required=True)
            groups = (("a", (prop.a,), 1), ("b", prop.b, 2), ("c", prop.c, 3), ("d", prop.d, 4))
            for label, values, size in groups:
                try:
                    ok = len(values) == size and all(_is_finite_number(v) for v in values)
                except TypeError:
                    ok = False
                if not ok:
                    add(f"{pwhere}: {label} must be {size} finite number(s), got {values!r}")
        for j, mapping in enumerate(mat.textures):
            mwhere = f"{where} mapping[{j}] {mapping.slot_name!r}"
            _check_name(problems, mwhere, "slot_name", mapping.slot_name, required=True)
            _check_name(problems, mwhere, "texture_name", mapping.texture_name, required=True)
            if not _is_int(mapping.slot) or not -(2**31) <= mapping.slot <= _INT32_MAX:
                add(f"{mwhere}: slot must be an int32, got {mapping.slot!r}")
            if isinstance(mapping.texture_name, str) and mapping.texture_name not in texture_names:
                add(f"{mwhere}: refers to texture {mapping.texture_name!r}, which is not embedded")

    n_materials = len(kn5.materials)
    if not isinstance(kn5.root, Dummy):
        add(f"root must be a Dummy, got {type(kn5.root).__name__}")
        return problems
    seen_dummies: set[int] = set()
    stack: list[tuple[object, str]] = [(kn5.root, kn5.root.name)]
    while stack:
        node, path = stack.pop()
        if isinstance(node, Dummy):
            where = f"dummy {path!r}"
            if id(node) in seen_dummies:
                add(f"{where}: the same Dummy object appears twice in the tree (cycle?)")
                continue
            seen_dummies.add(id(node))
            _check_name(problems, where, "name", node.name, required=False)
            _check_matrix(problems, where, "transform", node.transform)
            for child in reversed(node.children):
                child_name = getattr(child, "name", "?")
                stack.append((child, f"{path}/{child_name}"))
        elif isinstance(node, Mesh):
            where = f"mesh {path!r}"
            _check_name(problems, where, "name", node.name, required=False)
            _check_mesh(problems, where, node, n_materials)
        elif isinstance(node, SkinnedMesh):
            where = f"skinned mesh {path!r}"
            _check_name(problems, where, "name", node.name, required=False)
            _check_skinned_mesh(problems, where, node, n_materials)
        else:
            add(f"node {path!r}: unsupported node type {type(node).__name__}")
    return problems


# --------------------------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------------------------


def _string(value: str) -> bytes:
    # surrogateescape round-trips names read from files that are not valid UTF-8.
    data = value.encode("utf-8", "surrogateescape")
    return _I32.pack(len(data)) + data


def _matrix_bytes(matrix: np.ndarray) -> bytes:
    return np.ascontiguousarray(matrix, dtype="<f4").reshape(16).tobytes()


def _vertex_bytes(dtype: np.dtype, **columns: np.ndarray) -> tuple[int, bytes]:
    count = int(np.shape(columns["position"])[0])
    block = np.empty(count, dtype=dtype)
    for name, values in columns.items():
        block[name] = values
    return count, block.tobytes()


def _index_bytes(indices: np.ndarray) -> tuple[int, bytes]:
    idx = np.ascontiguousarray(indices, dtype="<u2").reshape(-1)
    return int(idx.size), idx.tobytes()


def _write_material(out: list[bytes], mat: Material, version: int) -> None:
    out.append(_string(mat.name))
    out.append(_string(mat.shader))
    out.append(_MATERIAL_FLAGS.pack(int(mat.blend_mode), 1 if mat.alpha_tested else 0))
    if version > 4:
        out.append(_I32.pack(int(mat.depth_mode)))
    out.append(_I32.pack(len(mat.properties)))
    for prop in mat.properties:
        out.append(_string(prop.name))
        out.append(_PROPERTY_VALUES.pack(prop.a, *prop.b, *prop.c, *prop.d))
    out.append(_I32.pack(len(mat.textures)))
    for mapping in mat.textures:
        out.append(_string(mapping.slot_name))
        out.append(_I32.pack(mapping.slot))
        out.append(_string(mapping.texture_name))


def _write_mesh(out: list[bytes], mesh: Mesh) -> None:
    n_vertices, vertices = _vertex_bytes(
        VERTEX_DTYPE,
        position=mesh.positions,
        normal=mesh.normals,
        uv=mesh.uvs,
        tangent=mesh.tangents,
    )
    n_indices, indices = _index_bytes(mesh.indices)
    sphere = mesh.bounding_sphere
    if sphere is None:
        sphere = loose_bounding_sphere(mesh.positions)
    (cx, cy, cz), radius = sphere
    out.append(_I32.pack(NODE_CLASS_MESH))
    out.append(_string(mesh.name))
    out.append(_NODE_HEAD.pack(0, 1 if mesh.active else 0))
    out.append(
        _MESH_FLAGS.pack(
            1 if mesh.cast_shadows else 0,
            1 if mesh.is_visible else 0,
            1 if mesh.is_transparent else 0,
        )
    )
    out += [_U32.pack(n_vertices), vertices, _U32.pack(n_indices), indices]
    out.append(
        _MESH_TAIL.pack(
            int(mesh.material_id),
            int(mesh.layer),
            mesh.lod_in,
            mesh.lod_out,
            cx,
            cy,
            cz,
            radius,
            1 if mesh.is_renderable else 0,
        )
    )


def _write_skinned_mesh(out: list[bytes], mesh: SkinnedMesh) -> None:
    out.append(_I32.pack(NODE_CLASS_SKINNED_MESH))
    out.append(_string(mesh.name))
    out.append(_NODE_HEAD.pack(0, 1 if mesh.active else 0))
    out.append(
        _MESH_FLAGS.pack(
            1 if mesh.cast_shadows else 0,
            1 if mesh.is_visible else 0,
            1 if mesh.is_transparent else 0,
        )
    )
    out.append(_U32.pack(len(mesh.bones)))
    for bone in mesh.bones:
        out.append(_string(bone.name))
        out.append(_matrix_bytes(bone.transform))
    n_vertices, vertices = _vertex_bytes(
        SKINNED_VERTEX_DTYPE,
        position=mesh.positions,
        normal=mesh.normals,
        uv=mesh.uvs,
        tangent=mesh.tangents,
        weights=mesh.bone_weights,
        bone_indices=mesh.bone_indices,
    )
    n_indices, indices = _index_bytes(mesh.indices)
    out += [_U32.pack(n_vertices), vertices, _U32.pack(n_indices), indices]
    out.append(
        _SKINNED_TAIL.pack(int(mesh.material_id), int(mesh.layer), mesh.lod_in, mesh.lod_out)
    )


def _write_nodes(out: list[bytes], root: Dummy) -> None:
    """Depth-first pre-order: each node record is followed by its children's records."""
    seen_dummies: set[int] = set()
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, Dummy):
            if id(node) in seen_dummies:
                raise ValueError(f"dummy {node.name!r} appears twice in the node tree (cycle?)")
            seen_dummies.add(id(node))
            out.append(_I32.pack(NODE_CLASS_DUMMY))
            out.append(_string(node.name))
            out.append(_NODE_HEAD.pack(len(node.children), 1 if node.active else 0))
            out.append(_matrix_bytes(node.transform))
            stack.extend(reversed(node.children))
        elif isinstance(node, Mesh):
            _write_mesh(out, node)
        elif isinstance(node, SkinnedMesh):
            _write_skinned_mesh(out, node)
        else:
            raise TypeError(f"unsupported KN5 node type {type(node).__name__}")


def kn5_to_bytes(kn5: Kn5, *, check: bool = True) -> bytes:
    """Serialise a KN5 model to bytes.

    With `check=True` (the default) the model is validated first and `Kn5ValidationError`
    (a ValueError) lists every problem. `check=False` skips validation; use it only to rewrite
    files that came from `read_kn5` (Kunos files often reference textures stored elsewhere).
    Meshes whose `bounding_sphere` is None get `loose_bounding_sphere(positions)`.
    """
    if check:
        problems = validate(kn5)
        if problems:
            raise Kn5ValidationError(problems)
    out: list[bytes] = [KN5_MAGIC, _I32.pack(kn5.version)]
    if kn5.version > 5:
        out.append(_I32.pack(kn5.extra))
    out.append(_I32.pack(len(kn5.textures)))
    for tex in kn5.textures:
        data = bytes(tex.data)
        out += [_I32.pack(1 if tex.active else 0), _string(tex.name), _U32.pack(len(data)), data]
    out.append(_I32.pack(len(kn5.materials)))
    for mat in kn5.materials:
        _write_material(out, mat, kn5.version)
    _write_nodes(out, kn5.root)
    return b"".join(out)


def write_kn5(kn5: Kn5, path: str | os.PathLike[str], *, check: bool = True) -> Path:
    """Validate and write `kn5` to `path` (parent folders are created). Returns the path.

    The file is written to a temporary name first and then renamed, so a failed build never
    leaves a half-written KN5 where AC would load it.
    """
    data = kn5_to_bytes(kn5, check=check)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------------------------


class _Reader:
    """Bounds-checked cursor over the whole file. Every failure raises `Kn5Error` with the
    source label, what was being read and the byte offset."""

    def __init__(self, data: bytes, label: str) -> None:
        self.data = data
        self.size = len(data)
        self.pos = 0
        self.label = label
        self.context = "header"

    def fail(self, message: str, offset: int | None = None) -> NoReturn:
        offset = self.pos if offset is None else offset
        raise Kn5Error(
            f"{self.label}: {message} (in {self.context}, at byte offset {offset:,} "
            f"of {self.size:,})"
        )

    def take(self, n: int, what: str) -> int:
        """Consume `n` bytes and return the offset where they start."""
        start = self.pos
        if n > self.size - start:
            self.fail(f"file is truncated: {what} needs {n:,} bytes, {self.size - start:,} left")
        self.pos = start + n
        return start

    def unpack(self, st: struct.Struct, what: str) -> tuple:
        return st.unpack_from(self.data, self.take(st.size, what))

    def i32(self, what: str) -> int:
        return self.unpack(_I32, what)[0]

    def flag(self, what: str) -> bool:
        # Like .NET ReadBoolean: any non-zero byte is true.
        return self.data[self.take(1, what)] != 0

    def count(self, st: struct.Struct, what: str, item_size: int) -> int:
        """Read a count and reject it if that many items could not fit in the rest of the file."""
        start = self.pos
        (n,) = self.unpack(st, what)
        left = self.size - self.pos
        if n < 0 or n * item_size > left:
            self.fail(
                f"implausible {what} {n:,}: needs at least {max(n, 0) * item_size:,} bytes but "
                f"only {left:,} are left (truncated or corrupt file)",
                start,
            )
        return n

    def string(self, what: str) -> str:
        n = self.count(_I32, f"length of {what}", 1)
        start = self.take(n, what)
        return self.data[start : start + n].decode("utf-8", "surrogateescape")

    def matrix(self, what: str) -> np.ndarray:
        start = self.take(64, what)
        return np.frombuffer(self.data, "<f4", 16, start).reshape(4, 4).astype(np.float32)

    def array(self, dtype: np.dtype, n: int, what: str) -> np.ndarray:
        start = self.take(n * dtype.itemsize, what)
        return np.frombuffer(self.data, dtype, n, start)

    def enum(self, enum_type: type[BlendMode] | type[DepthMode], value: int, what: str, at: int):
        try:
            return enum_type(value)
        except ValueError:
            known = ", ".join(f"{e.value} {e.name}" for e in enum_type)
            self.fail(f"unknown {what} {value} (known: {known})", at)

    # -- sections ---------------------------------------------------------------------------

    def read(self) -> Kn5:
        magic = self.data[: len(KN5_MAGIC)]
        if magic != KN5_MAGIC:
            self.fail(f"not a KN5 file: starts with {magic!r}, expected {KN5_MAGIC!r}", 0)
        self.take(len(KN5_MAGIC), "magic")
        at = self.pos
        version = self.i32("version")
        if not MIN_READ_VERSION <= version <= MAX_READ_VERSION:
            self.fail(
                f"unsupported KN5 version {version} (this reader handles "
                f"{MIN_READ_VERSION}-{MAX_READ_VERSION})",
                at,
            )
        extra = self.i32("extra header int") if version > 5 else 0
        textures = self.read_textures()
        materials = self.read_materials(version)
        root = self.read_nodes()
        if self.pos != self.size:
            self.context = "end of file"
            self.fail(f"{self.size - self.pos:,} unexpected trailing bytes after the node tree")
        return Kn5(root=root, textures=textures, materials=materials, version=version, extra=extra)

    def read_textures(self) -> list[Texture]:
        self.context = "texture list"
        textures = []
        for i in range(self.count(_I32, "texture count", _MIN_TEXTURE_RECORD)):
            self.context = f"texture[{i}]"
            # AcTools: Active = (flag == 1). VERIFIED (code) Kn5Basic.ReadTexture.
            active = self.i32("texture active flag") == 1
            name = self.string("texture name")
            self.context = f"texture[{i}] {name!r}"
            size = self.count(_U32, "texture size", 1)
            start = self.take(size, "texture data")
            textures.append(Texture(name=name, data=self.data[start : start + size], active=active))
        return textures

    def read_materials(self, version: int) -> list[Material]:
        self.context = "material list"
        min_record = 4 + 4 + 2 + (4 if version > 4 else 0) + 4 + 4
        materials = []
        for i in range(self.count(_I32, "material count", min_record)):
            self.context = f"material[{i}]"
            name = self.string("material name")
            self.context = f"material[{i}] {name!r}"
            shader = self.string("shader name")
            at = self.pos
            blend_raw, alpha_tested = self.unpack(_MATERIAL_FLAGS, "blend mode and alpha test")
            blend = self.enum(BlendMode, blend_raw, "blend mode", at)
            depth = DepthMode.NORMAL
            if version > 4:
                at = self.pos
                depth = self.enum(DepthMode, self.i32("depth mode"), "depth mode", at)
            properties = []
            for _ in range(self.count(_I32, "property count", _MIN_PROPERTY_RECORD)):
                prop_name = self.string("property name")
                v = self.unpack(_PROPERTY_VALUES, f"values of property {prop_name!r}")
                properties.append(
                    MaterialProperty(name=prop_name, a=v[0], b=v[1:3], c=v[3:6], d=v[6:10])
                )
            mappings = []
            for _ in range(self.count(_I32, "texture mapping count", _MIN_MAPPING_RECORD)):
                slot_name = self.string("texture slot name")
                slot = self.i32("texture slot")
                texture_name = self.string("mapped texture name")
                mappings.append(TextureMapping(slot_name, slot, texture_name))
            materials.append(
                Material(
                    name=name,
                    shader=shader,
                    blend_mode=blend,
                    alpha_tested=alpha_tested != 0,
                    depth_mode=depth,
                    properties=properties,
                    textures=mappings,
                )
            )
        return materials

    def read_nodes(self) -> Dummy:
        self.context = "root node"
        root, n_children = self.read_node("")
        if not isinstance(root, Dummy):
            self.fail(f"root node {root.name!r} is a {type(root).__name__}, expected a dummy")
        # Iterative walk: (parent, children still to read, path) so deep trees can't overflow.
        stack: list[tuple[Dummy, int, str]] = [(root, n_children, root.name)]
        while stack:
            parent, remaining, path = stack[-1]
            if remaining == 0:
                stack.pop()
                continue
            stack[-1] = (parent, remaining - 1, path)
            node, n_children = self.read_node(path)
            parent.children.append(node)
            if isinstance(node, Dummy):
                stack.append((node, n_children, f"{path}/{node.name}"))
        return root

    def read_node(self, parent_path: str) -> tuple[Node, int]:
        start = self.pos
        self.context = f"node under {parent_path!r}" if parent_path else "root node"
        node_class = self.i32("node class")
        name = self.string("node name")
        path = f"{parent_path}/{name}" if parent_path else name
        self.context = f"node {path!r}"
        n_children = self.count(_I32, "child count", _MIN_NODE_RECORD)
        active = self.flag("active flag")

        if node_class == NODE_CLASS_DUMMY:
            transform = self.matrix("transform")
            return Dummy(name=name, transform=transform, children=[], active=active), n_children
        if node_class not in (NODE_CLASS_MESH, NODE_CLASS_SKINNED_MESH):
            self.fail(
                f"unknown node class {node_class} for node {name!r} "
                "(expected 1 dummy, 2 mesh or 3 skinned mesh)",
                start,
            )
        if n_children:
            # AcTools would read children under a mesh; we have never seen one and our model
            # cannot hold them, so refuse loudly rather than misplace nodes.
            self.fail(f"mesh node {name!r} declares {n_children} children (unsupported)", start)

        cast_shadows, is_visible, is_transparent = (
            b != 0 for b in self.unpack(_MESH_FLAGS, "mesh flags")
        )
        if node_class == NODE_CLASS_MESH:
            n = self.count(_U32, "vertex count", VERTEX_DTYPE.itemsize)
            vertices = self.array(VERTEX_DTYPE, n, "vertices")
            indices = self.read_indices()
            tail = self.unpack(_MESH_TAIL, "mesh tail (material, layer, lod, sphere, renderable)")
            material_id, layer, lod_in, lod_out, cx, cy, cz, radius, renderable = tail
            mesh = Mesh(
                name=name,
                positions=vertices["position"].astype(np.float32),
                normals=vertices["normal"].astype(np.float32),
                uvs=vertices["uv"].astype(np.float32),
                tangents=vertices["tangent"].astype(np.float32),
                indices=indices,
                material_id=material_id,
                cast_shadows=cast_shadows,
                is_visible=is_visible,
                is_transparent=is_transparent,
                is_renderable=renderable != 0,
                layer=layer,
                lod_in=lod_in,
                lod_out=lod_out,
                active=active,
                bounding_sphere=((cx, cy, cz), radius),
            )
            return mesh, 0

        bones = []
        for i in range(self.count(_U32, "bone count", _MIN_BONE_RECORD)):
            bone_name = self.string(f"bone[{i}] name")
            bones.append(Bone(name=bone_name, transform=self.matrix(f"bone {bone_name!r}")))
        n = self.count(_U32, "vertex count", SKINNED_VERTEX_DTYPE.itemsize)
        vertices = self.array(SKINNED_VERTEX_DTYPE, n, "skinned vertices")
        indices = self.read_indices()
        material_id, layer, lod_in, lod_out = self.unpack(_SKINNED_TAIL, "skinned mesh tail")
        skinned = SkinnedMesh(
            name=name,
            positions=vertices["position"].astype(np.float32),
            normals=vertices["normal"].astype(np.float32),
            uvs=vertices["uv"].astype(np.float32),
            tangents=vertices["tangent"].astype(np.float32),
            bone_weights=vertices["weights"].astype(np.float32),
            bone_indices=vertices["bone_indices"].astype(np.float32),
            indices=indices,
            material_id=material_id,
            bones=bones,
            cast_shadows=cast_shadows,
            is_visible=is_visible,
            is_transparent=is_transparent,
            layer=layer,
            lod_in=lod_in,
            lod_out=lod_out,
            active=active,
        )
        return skinned, 0

    def read_indices(self) -> np.ndarray:
        n = self.count(_U32, "index count", 2)
        return self.array(np.dtype("<u2"), n, "indices").astype(np.uint16)


def _load(source: str | os.PathLike[str] | bytes | bytearray | memoryview) -> tuple[bytes, str]:
    if isinstance(source, bytes | bytearray | memoryview):
        return bytes(source), "<bytes>"
    path = Path(source)
    return path.read_bytes(), str(path)


def read_kn5(source: str | os.PathLike[str] | bytes | bytearray | memoryview) -> Kn5:
    """Parse a KN5 file (path) or its bytes into a `Kn5` model.

    Reads versions 1-6 (Kunos files are 5 or 6), dummies, meshes and skinned meshes. Stored
    bounding spheres are kept on `Mesh.bounding_sphere`, so `kn5_to_bytes(read_kn5(b),
    check=False) == b` for well-formed files. Anything unexpected raises `Kn5Error` (a
    ValueError) naming the record and byte offset; the reader never guesses. It does not
    validate: call `validate()` for that.
    """
    data, label = _load(source)
    return _Reader(data, label).read()


# --------------------------------------------------------------------------------------------
# Texture sniffing and dump
# --------------------------------------------------------------------------------------------

# DDS header offsets (from the file start, after the 4-byte "DDS " magic): Microsoft
# DDS_HEADER / DDS_PIXELFORMAT docs, cross-checked with Pillow's DdsImagePlugin.
_DDSD_MIPMAPCOUNT = 0x20000
_DDPF_ALPHAPIXELS = 0x1
_DDPF_FOURCC = 0x4
_DXGI_NAMES = {
    28: "R8G8B8A8_UNORM",
    29: "R8G8B8A8_UNORM_SRGB",
    71: "BC1_UNORM",
    72: "BC1_UNORM_SRGB",
    74: "BC2_UNORM",
    75: "BC2_UNORM_SRGB",
    77: "BC3_UNORM",
    78: "BC3_UNORM_SRGB",
    80: "BC4_UNORM",
    83: "BC5_UNORM",
    87: "B8G8R8A8_UNORM",
    91: "B8G8R8A8_UNORM_SRGB",
    95: "BC6H_UF16",
    98: "BC7_UNORM",
    99: "BC7_UNORM_SRGB",
}
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_COLOUR_TYPES = {0: "grey", 2: "RGB", 3: "palette", 4: "grey+alpha", 6: "RGBA"}


def texture_info(data: bytes) -> str:
    """Describe an embedded texture from its bytes, e.g. "DDS 512x512 DXT1 mips=10"."""
    data = bytes(data[:256])
    if data[:4] == b"DDS ":
        if len(data) < 128:
            return "DDS (truncated header)"
        flags, height, width = struct.unpack_from("<3I", data, 8)
        (mip_count,) = struct.unpack_from("<I", data, 28)
        pf_flags, fourcc_raw, bit_count = struct.unpack_from("<3I", data, 80)
        if pf_flags & _DDPF_FOURCC:
            fourcc = data[84:88]
            if fourcc == b"DX10" and len(data) >= 148:
                (dxgi,) = struct.unpack_from("<I", data, 128)
                fmt = f"DX10/{_DXGI_NAMES.get(dxgi, f'DXGI {dxgi}')}"
            elif all(0x20 <= c < 0x7F for c in fourcc):
                fmt = fourcc.decode("ascii")
            else:
                fmt = f"D3DFMT {fourcc_raw}"
        else:
            alpha = "+alpha" if pf_flags & _DDPF_ALPHAPIXELS else ""
            fmt = f"uncompressed {bit_count}-bit{alpha}"
        mips = mip_count if mip_count > 0 else 1
        note = "" if flags & _DDSD_MIPMAPCOUNT or mips == 1 else " (MIPMAPCOUNT flag not set)"
        return f"DDS {width}x{height} {fmt} mips={mips}{note}"
    if data[:8] == _PNG_SIGNATURE:
        if len(data) < 26 or data[12:16] != b"IHDR":
            return "PNG (truncated header)"
        width, height = struct.unpack_from(">II", data, 16)
        bit_depth, colour_type = data[24], data[25]
        colour = _PNG_COLOUR_TYPES.get(colour_type, f"colour type {colour_type}")
        return f"PNG {width}x{height} {colour} {bit_depth}-bit"
    if data[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if not data:
        return "empty"
    return f"unknown format (starts {data[:8].hex(' ')})"


def _fmt_vec(values) -> str:
    # Values that round to zero print as 0.000, never -0.000.
    return "(" + ", ".join(f"{float(v) if abs(v) >= 5e-4 else 0.0:.3f}" for v in values) + ")"


def _fmt_num(value: float) -> str:
    return f"{float(value):g}"


def _material_label(kn5: Kn5, material_id: int) -> str:
    if 0 <= material_id < len(kn5.materials):
        return f"{kn5.materials[material_id].name}({material_id})"
    return f"INVALID({material_id})"


def _dump_material(lines: list[str], i: int, mat: Material, texture_names: set[str]) -> None:
    blend = _enum_name(BlendMode, mat.blend_mode) or repr(mat.blend_mode)
    depth = _enum_name(DepthMode, mat.depth_mode) or repr(mat.depth_mode)
    lines.append(
        f"  [{i}] {mat.name}  shader={mat.shader}  blend={blend}  "
        f"alpha_tested={'yes' if mat.alpha_tested else 'no'}  depth={depth}"
    )
    if mat.properties:
        parts = []
        for prop in mat.properties:
            text = f"{prop.name}={_fmt_num(prop.a)}"
            for label, values in (("b", prop.b), ("c", prop.c), ("d", prop.d)):
                if any(float(v) != 0.0 for v in values):
                    text += f" {label}=(" + ", ".join(_fmt_num(v) for v in values) + ")"
            parts.append(text)
        lines.append("      props: " + "; ".join(parts))
    for mapping in mat.textures:
        missing = "" if mapping.texture_name in texture_names else "  [not embedded]"
        lines.append(
            f"      {mapping.slot_name} (slot {mapping.slot}) -> {mapping.texture_name}{missing}"
        )


def _dump_mesh_line(kn5: Kn5, node: Mesh | SkinnedMesh) -> str:
    n_vertices = node.vertex_count
    n_triangles = int(np.asarray(node.indices).size) // 3
    flags = (
        f"shadows={int(node.cast_shadows)} visible={int(node.is_visible)} "
        f"transparent={int(node.is_transparent)}"
    )
    if isinstance(node, Mesh):
        kind = "mesh"
        flags += f" renderable={int(node.is_renderable)}"
        extra = ""
        if node.bounding_sphere is not None:
            centre, radius = node.bounding_sphere
            extra = f"  sphere={_fmt_vec(centre)} r={float(radius):.3f}"
    else:
        kind = "skinned mesh"
        extra = f"  bones={len(node.bones)}"
    inactive = "  INACTIVE" if not node.active else ""
    return (
        f"- {node.name}  [{kind}]  {n_vertices:,} verts  {n_triangles:,} tris  "
        f"material={_material_label(kn5, int(node.material_id))}  {flags}  layer={node.layer}  "
        f"lod={_fmt_num(node.lod_in)}..{_fmt_num(node.lod_out)}{extra}{inactive}"
    )


def _dump_dummy_line(node: Dummy, world: np.ndarray) -> str:
    t = np.asarray(node.transform, dtype=np.float64)
    inactive = "  INACTIVE" if not node.active else ""
    if np.array_equal(t, np.eye(4)):
        text = f"+ {node.name}  [dummy]  identity"
    else:
        text = f"+ {node.name}  [dummy]  pos={_fmt_vec(t[3, :3])}  fwd={_fmt_vec(t[2, :3])}"
    if not np.allclose(world, t, atol=1e-6):
        text += f"  world_pos={_fmt_vec(world[3, :3])}  world_fwd={_fmt_vec(world[2, :3])}"
    return text + f"  ({len(node.children)} children){inactive}"


def dump(
    source: str | os.PathLike[str] | bytes | bytearray | memoryview, include_meshes: bool = True
) -> str:
    """Human-readable summary of a KN5 file (used by `drift-maps kn5-dump`).

    Lists the version, textures (size, sniffed format, dimensions, DDS FourCC and mip count),
    materials (shader, blend, properties, texture mappings), the node tree (dummies with local
    position and forward axis = transform row 2; meshes with vertex and triangle counts,
    material, flags and LOD range), totals, and the result of `validate()`.
    """
    data, label = _load(source)
    kn5 = _Reader(data, label).read()
    lines = [f"KN5 {label}  ({len(data):,} bytes)"]
    header = f"version {kn5.version}"
    if kn5.version > 5:
        header += f"  extra {kn5.extra}"
    lines.append(header)

    lines.append(f"textures: {len(kn5.textures)}")
    for i, tex in enumerate(kn5.textures):
        inactive = "  INACTIVE" if not tex.active else ""
        lines.append(f"  [{i}] {tex.name}  {len(tex.data):,} B  {texture_info(tex.data)}{inactive}")

    lines.append(f"materials: {len(kn5.materials)}")
    texture_names = {t.name for t in kn5.textures}
    for i, mat in enumerate(kn5.materials):
        _dump_material(lines, i, mat, texture_names)

    lines.append("nodes:")
    counts = {"dummies": 0, "meshes": 0, "skinned": 0, "vertices": 0, "triangles": 0}
    # (node, depth, parent world transform); row-vector convention: world = local @ parent.
    stack: list[tuple[Node, int, np.ndarray]] = [(kn5.root, 0, np.eye(4))]
    while stack:
        node, depth, parent_world = stack.pop()
        indent = "  " * (depth + 1)
        if isinstance(node, Dummy):
            counts["dummies"] += 1
            world = np.asarray(node.transform, dtype=np.float64) @ parent_world
            lines.append(indent + _dump_dummy_line(node, world))
            stack.extend((child, depth + 1, world) for child in reversed(node.children))
            continue
        counts["skinned" if isinstance(node, SkinnedMesh) else "meshes"] += 1
        counts["vertices"] += node.vertex_count
        counts["triangles"] += int(np.asarray(node.indices).size) // 3
        if include_meshes:
            lines.append(indent + _dump_mesh_line(kn5, node))

    lines.append(
        f"totals: {counts['dummies']} dummies, {counts['meshes']} meshes, "
        f"{counts['skinned']} skinned meshes, {counts['vertices']:,} vertices, "
        f"{counts['triangles']:,} triangles"
    )
    problems = validate(kn5)
    if problems:
        lines.append(f"validation: {len(problems)} problem(s) (our writer would refuse this file)")
        lines.extend(f"  - {p}" for p in problems[:50])
        if len(problems) > 50:
            lines.append(f"  ... and {len(problems) - 50} more")
    else:
        lines.append("validation: ok")
    return "\n".join(lines) + "\n"
