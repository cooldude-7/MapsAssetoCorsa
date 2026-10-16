"""KN5 -> GLB (binary glTF 2.0) for the kn5view renderer.

The KN5 is read with the repo's own reader (`driftmaps.ac.kn5.read_kn5`) and written out as a
self-contained GLB that three.js (or Blender) can open:

- AC world space (X east, Y up, Z south, right-handed) is glTF's frame too, and the repo's meshes
  wind counter-clockwise seen from the visible side (`driftmaps.ac.model.Mesh`), as glTF wants.
  So positions, normals and indices are copied as they are.
- UVs: KN5 (DirectX) and glTF both put (0, 0) at the top-left of the image: copied as they are.
- Dummy transforms are DirectX row-vector matrices (translation in row 3). Flattened row-major
  they are exactly glTF's column-major `matrix`.
- Each material's `txDiffuse` becomes the base colour texture. DDS (BC1/BC3) is decoded with
  Pillow; textures used by alpha materials are stored as PNG (alpha kept), the rest as JPEG.
  PNG sources small enough are embedded untouched.
- Alpha: `alpha_tested` (or ALPHA_TO_COVERAGE, or an "AT"/tree/grass shader) -> alphaMode MASK
  with the material's ksAlphaRef as cutoff, double-sided. ALPHA_BLEND -> BLEND, double-sided.
- Hidden meshes (not visible / not renderable, or physics walls named like "1WALL_...") are kept
  but flagged `extras.hidden = true`; the viewer hides them unless asked to show collision.

`cached_glb()` keeps the result next to the KN5 as a dotfile (".<stem>.kn5view.glb", ignored by
drift-maps' CSP install copy) plus a ".json" sidecar keyed on the KN5's size and mtime.
"""

from __future__ import annotations

import io
import json
import os
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from driftmaps.ac.kn5 import read_kn5
from driftmaps.ac.model import BlendMode, Dummy, Kn5, Material, Mesh, SkinnedMesh

# Bump when the GLB layout or the sidecar changes, so old caches are rebuilt.
CONVERTER_VERSION = 2

COLLISION_RE = re.compile(r"^\d+WALL", re.IGNORECASE)
_AT_SHADER_RE = re.compile(r"(_AT(_|$)|AT$|tree|grass)", re.IGNORECASE)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_IDENTITY = np.eye(4, dtype=np.float32)

_GL_FLOAT = 5126
_GL_UNSIGNED_SHORT = 5123
_GL_UNSIGNED_INT = 5125
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_REPEAT = 10497
_LINEAR = 9729
_LINEAR_MIPMAP_LINEAR = 9987


@dataclass
class ConvertStats:
    meshes: int = 0
    hidden_meshes: int = 0
    vertices: int = 0
    triangles: int = 0
    materials: int = 0
    textures: int = 0
    texture_warnings: list[str] = field(default_factory=list)
    glb_bytes: int = 0


def is_hidden_mesh(node: Mesh | SkinnedMesh) -> bool:
    """True for meshes AC never draws: physics walls ("1WALL_..."), invisible or unrenderable."""
    if COLLISION_RE.match(node.name):
        return True
    if not node.is_visible:
        return True
    return isinstance(node, Mesh) and not node.is_renderable


def _prop(mat: Material, name: str) -> float | None:
    for p in mat.properties:
        if p.name == name:
            return float(p.a)
    return None


def _prop_c(mat: Material, name: str) -> tuple[float, float, float] | None:
    for p in mat.properties:
        if p.name == name:
            return tuple(float(v) for v in p.c)
    return None


def alpha_mode(mat: Material) -> str:
    """glTF alphaMode for a KN5 material: OPAQUE, MASK or BLEND."""
    if mat.blend_mode == BlendMode.ALPHA_BLEND:
        return "BLEND"
    if (
        mat.alpha_tested
        or mat.blend_mode == BlendMode.ALPHA_TO_COVERAGE
        or _AT_SHADER_RE.search(mat.shader)
    ):
        return "MASK"
    return "OPAQUE"


def _diffuse_texture_name(mat: Material) -> str | None:
    for t in mat.textures:
        if t.slot_name == "txDiffuse":
            return t.texture_name
    return None


def encode_texture(data: bytes, *, keep_alpha: bool, max_size: int) -> tuple[bytes, str]:
    """Decode a KN5 texture (DDS or PNG bytes) into PNG or JPEG bytes for the browser.

    Returns (bytes, mime type). Raises on undecodable data.
    """
    if data[:8] == _PNG_SIGNATURE:
        width, height = struct.unpack(">II", data[16:24])
        if max(width, height) <= max_size:
            return data, "image/png"
    img = Image.open(io.BytesIO(data))
    img.load()
    factor = 1
    while max(img.size) // factor > max_size:
        factor *= 2
    if factor > 1:
        img = img.reduce(factor)
    out = io.BytesIO()
    has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
    if keep_alpha and has_alpha:
        img.convert("RGBA").save(out, format="PNG", compress_level=1)
        return out.getvalue(), "image/png"
    img.convert("RGB").save(out, format="JPEG", quality=90)
    return out.getvalue(), "image/jpeg"


def _placeholder_png() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 255)).save(out, format="PNG")
    return out.getvalue()


class _Bin:
    """Accumulates the GLB binary chunk, bufferViews and accessors."""

    def __init__(self) -> None:
        self.parts: list[bytes] = []
        self.size = 0
        self.buffer_views: list[dict] = []
        self.accessors: list[dict] = []

    def view(self, data: bytes, target: int | None = None) -> int:
        pad = (-self.size) % 4
        if pad:
            self.parts.append(b"\0" * pad)
            self.size += pad
        bv: dict = {"buffer": 0, "byteOffset": self.size, "byteLength": len(data)}
        if target is not None:
            bv["target"] = target
        self.parts.append(data)
        self.size += len(data)
        self.buffer_views.append(bv)
        return len(self.buffer_views) - 1

    def accessor(self, arr: np.ndarray, gltf_type: str, target: int, minmax: bool = False) -> int:
        if arr.dtype == np.float32:
            ctype = _GL_FLOAT
        elif arr.dtype == np.uint16:
            ctype = _GL_UNSIGNED_SHORT
        elif arr.dtype == np.uint32:
            ctype = _GL_UNSIGNED_INT
        else:  # pragma: no cover - callers convert first
            raise TypeError(arr.dtype)
        arr = np.ascontiguousarray(arr)
        acc: dict = {
            "bufferView": self.view(arr.tobytes(), target),
            "componentType": ctype,
            "count": int(arr.shape[0]),
            "type": gltf_type,
        }
        if minmax:
            acc["min"] = [float(v) for v in arr.min(axis=0)]
            acc["max"] = [float(v) for v in arr.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def blob(self) -> bytes:
        pad = (-self.size) % 4
        return b"".join(self.parts) + b"\0" * pad


def _material_json(mat: Material, texture_index: int | None) -> dict:
    mode = alpha_mode(mat)
    pbr: dict = {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0}
    pbr["roughnessFactor"] = 1.0
    if texture_index is not None:
        pbr["baseColorTexture"] = {"index": texture_index}
    out: dict = {"name": mat.name, "pbrMetallicRoughness": pbr}
    if mode != "OPAQUE":
        out["alphaMode"] = mode
        out["doubleSided"] = True
    if mode == "MASK":
        ref = _prop(mat, "ksAlphaRef")
        out["alphaCutoff"] = float(ref) if ref and 0.0 < ref < 1.0 else 0.5
    emissive = _prop_c(mat, "ksEmissive")
    if emissive and max(emissive) > 0.0:
        out["emissiveFactor"] = [min(1.0, max(0.0, v)) for v in emissive]
        if texture_index is not None:
            out["emissiveTexture"] = {"index": texture_index}
    out["extras"] = {
        "shader": mat.shader,
        "blendMode": mat.blend_mode.name,
        "alphaTested": bool(mat.alpha_tested),
        "ksAmbient": _prop(mat, "ksAmbient"),
        "ksDiffuse": _prop(mat, "ksDiffuse"),
        "ksAlphaRef": _prop(mat, "ksAlphaRef"),
    }
    return out


def kn5_to_glb(
    kn5: Kn5, *, max_texture: int = 2048, workers: int | None = None
) -> tuple[bytes, ConvertStats, dict[str, list[float]]]:
    """Convert a parsed KN5 to GLB bytes.

    Returns (glb bytes, stats, dummies) where dummies maps each dummy name to its world position
    (AC frame), e.g. AC_START_0 or AC_AB_START_L.
    """
    stats = ConvertStats(materials=len(kn5.materials))
    out = _Bin()

    # --- textures: only those bound to txDiffuse are needed --------------------------------
    by_name = {t.name: t for t in kn5.textures}
    alpha_textures: set[str] = set()
    wanted: list[str] = []
    for mat in kn5.materials:
        name = _diffuse_texture_name(mat)
        if name is None or name not in by_name:
            continue
        if name not in wanted:
            wanted.append(name)
        if alpha_mode(mat) != "OPAQUE":
            alpha_textures.add(name)

    def _encode(name: str) -> tuple[bytes, str]:
        try:
            return encode_texture(
                by_name[name].data, keep_alpha=name in alpha_textures, max_size=max_texture
            )
        except Exception as exc:  # noqa: BLE001 - one bad texture must not stop the render
            stats.texture_warnings.append(f"{name}: {exc}")
            return _placeholder_png(), "image/png"

    with ThreadPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 2)) as pool:
        encoded = list(pool.map(_encode, wanted))

    images: list[dict] = []
    textures: list[dict] = []
    texture_index: dict[str, int] = {}
    for name, (data, mime) in zip(wanted, encoded, strict=True):
        images.append({"name": name, "mimeType": mime, "bufferView": out.view(data)})
        textures.append({"sampler": 0, "source": len(images) - 1, "name": name})
        texture_index[name] = len(textures) - 1
    stats.textures = len(textures)

    materials = [
        _material_json(m, texture_index.get(_diffuse_texture_name(m) or "")) for m in kn5.materials
    ]
    if not materials:
        materials = [{"name": "default", "pbrMetallicRoughness": {"metallicFactor": 0.0}}]

    # --- nodes -------------------------------------------------------------------------------
    nodes: list[dict] = []
    meshes: list[dict] = []
    dummies: dict[str, list[float]] = {}

    def add_mesh(node: Mesh | SkinnedMesh) -> int | None:
        idx = np.asarray(node.indices)
        if idx.size < 3 or node.vertex_count == 0:
            return None
        pos = np.asarray(node.positions, dtype=np.float32)
        attrs = {"POSITION": out.accessor(pos, "VEC3", _ARRAY_BUFFER, minmax=True)}
        nrm = np.asarray(node.normals, dtype=np.float32)
        if nrm.shape == pos.shape:
            attrs["NORMAL"] = out.accessor(nrm, "VEC3", _ARRAY_BUFFER)
        uv = np.asarray(node.uvs, dtype=np.float32)
        if uv.shape == (pos.shape[0], 2):
            attrs["TEXCOORD_0"] = out.accessor(uv, "VEC2", _ARRAY_BUFFER)
        itype = np.uint16 if node.vertex_count <= 65535 else np.uint32
        n_idx = idx.size - idx.size % 3
        ind = out.accessor(idx[:n_idx].astype(itype), "SCALAR", _ELEMENT_ARRAY_BUFFER)
        mat_id = int(node.material_id)
        prim: dict = {"attributes": attrs, "indices": ind, "mode": 4}
        if 0 <= mat_id < len(materials):
            prim["material"] = mat_id
        meshes.append({"name": node.name, "primitives": [prim]})
        stats.meshes += 1
        stats.vertices += node.vertex_count
        stats.triangles += n_idx // 3
        return len(meshes) - 1

    def visit(node, world: np.ndarray) -> int | None:
        entry: dict = {"name": node.name}
        if isinstance(node, Dummy):
            local = np.asarray(node.transform, dtype=np.float32)
            world = local.astype(np.float64) @ world
            dummies[node.name] = [float(v) for v in world[3, :3]]
            if not np.array_equal(local, _IDENTITY):
                entry["matrix"] = [float(v) for v in local.reshape(-1)]
            kids = [k for k in (visit(c, world) for c in node.children) if k is not None]
            if kids:
                entry["children"] = kids
            if not node.active:
                entry["extras"] = {"hidden": True, "inactive": True}
        else:
            mesh_index = add_mesh(node)
            if mesh_index is None:
                return None
            entry["mesh"] = mesh_index
            hidden = is_hidden_mesh(node) or not node.active
            if hidden:
                stats.hidden_meshes += 1
            entry["extras"] = {"hidden": hidden, "castShadows": bool(node.cast_shadows)}
        nodes.append(entry)
        return len(nodes) - 1

    root = visit(kn5.root, np.eye(4))
    gltf: dict = {
        "asset": {"version": "2.0", "generator": f"driftmaps kn5view v{CONVERTER_VERSION}"},
        "scene": 0,
        "scenes": [{"name": kn5.root.name, "nodes": [root] if root is not None else []}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "samplers": [
            {
                "magFilter": _LINEAR,
                "minFilter": _LINEAR_MIPMAP_LINEAR,
                "wrapS": _REPEAT,
                "wrapT": _REPEAT,
            }
        ],
    }
    if textures:
        gltf["images"] = images
        gltf["textures"] = textures
    blob = out.blob()
    if blob:
        gltf["buffers"] = [{"byteLength": len(blob)}]
        gltf["bufferViews"] = out.buffer_views
        gltf["accessors"] = out.accessors
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + (8 + len(blob) if blob else 0)
    parts = [struct.pack("<4sII", b"glTF", 2, total), struct.pack("<I4s", len(js), b"JSON"), js]
    if blob:
        parts += [struct.pack("<I4s", len(blob), b"BIN\0"), blob]
    glb = b"".join(parts)
    stats.glb_bytes = len(glb)
    return glb, stats, dummies


def read_glb(data: bytes) -> tuple[dict, bytes]:
    """Split GLB bytes into (glTF JSON dict, binary chunk). Used by tests and debugging."""
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or total != len(data):
        raise ValueError("not a GLB 2.0 file")
    jlen, jtype = struct.unpack_from("<I4s", data, 12)
    if jtype != b"JSON":
        raise ValueError("first chunk is not JSON")
    gltf = json.loads(data[20 : 20 + jlen])
    off = 20 + jlen
    blob = b""
    if off < len(data):
        blen, btype = struct.unpack_from("<I4s", data, off)
        if btype != b"BIN\0":
            raise ValueError("second chunk is not BIN")
        blob = data[off + 8 : off + 8 + blen]
    return gltf, blob


def cache_paths(kn5_path: Path, cache_dir: Path | None = None) -> tuple[Path, Path]:
    folder = cache_dir or kn5_path.parent
    stem = f".{kn5_path.stem}.kn5view"
    return folder / f"{stem}.glb", folder / f"{stem}.json"


def cached_glb(
    kn5_path: str | os.PathLike[str],
    *,
    cache_dir: Path | None = None,
    force: bool = False,
    max_texture: int = 2048,
    log=print,
) -> tuple[Path, dict]:
    """Return (GLB path, sidecar dict), converting only when the KN5 changed since last time."""
    kn5_path = Path(kn5_path).resolve()
    glb_path, meta_path = cache_paths(kn5_path, cache_dir)
    st = kn5_path.stat()
    key = {
        "kn5": kn5_path.name,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "converter": CONVERTER_VERSION,
        "max_texture": max_texture,
    }
    if not force and glb_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        if meta.get("key") == key:
            return glb_path, meta
    import time

    t0 = time.perf_counter()
    kn5 = read_kn5(kn5_path)
    t1 = time.perf_counter()
    glb, stats, dummies = kn5_to_glb(kn5, max_texture=max_texture)
    t2 = time.perf_counter()
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = glb_path.with_name(glb_path.name + ".tmp")
    tmp.write_bytes(glb)
    os.replace(tmp, glb_path)
    meta = {
        "key": key,
        "stats": stats.__dict__,
        "dummies": dummies,
        "timings_s": {"read": round(t1 - t0, 3), "convert": round(t2 - t1, 3)},
    }
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(
        f"converted {kn5_path.name}: {stats.meshes} meshes ({stats.hidden_meshes} hidden), "
        f"{stats.triangles:,} triangles, {stats.textures} textures, "
        f"{stats.glb_bytes / 1e6:.1f} MB GLB in {t2 - t0:.1f} s "
        f"(read {t1 - t0:.1f} s) -> {glb_path}"
    )
    for w in stats.texture_warnings:
        log(f"  texture warning: {w}")
    return glb_path, meta
