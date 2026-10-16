"""KN5 -> GLB conversion used by the kn5view renderer (tools/kn5view/kn5glb.py)."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from driftmaps.ac.coords import translation_transform
from driftmaps.ac.kn5 import write_kn5
from driftmaps.ac.model import (
    BlendMode,
    Dummy,
    Kn5,
    Material,
    MaterialProperty,
    Mesh,
    Texture,
    TextureMapping,
)
from driftmaps.ac.textures import encode_dds

KN5GLB_PATH = Path(__file__).resolve().parents[1] / "tools" / "kn5view" / "kn5glb.py"


def _load_kn5glb():
    spec = importlib.util.spec_from_file_location("kn5view_kn5glb", KN5GLB_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while being defined
    spec.loader.exec_module(module)
    return module


kn5glb = _load_kn5glb()


def _quad(name: str, material_id: int, x0: float, **flags) -> Mesh:
    pos = np.array([[x0, 0, 0], [x0 + 2, 0, 0], [x0 + 2, 0, -2], [x0, 0, -2]], np.float32)
    return Mesh(
        name=name,
        positions=pos,
        normals=np.tile(np.array([0, 1, 0], np.float32), (4, 1)),
        uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32),
        tangents=np.tile(np.array([1, 0, 0], np.float32), (4, 1)),
        indices=np.array([0, 2, 1, 0, 3, 2], np.uint16),
        material_id=material_id,
        **flags,
    )


def _png(colour: tuple[int, int, int]) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(out, format="PNG")
    return out.getvalue()


def _tiny_kn5() -> Kn5:
    leaf = Image.new("RGBA", (8, 8), (40, 120, 30, 255))
    leaf.paste((0, 0, 0, 0), (0, 0, 8, 4))  # top half transparent
    road = Material(
        "road",
        properties=[MaterialProperty("ksAmbient", a=0.45), MaterialProperty("ksDiffuse", a=0.6)],
        textures=[TextureMapping("txDiffuse", 0, "asphalt.png")],
    )
    trees = Material(
        "trees",
        shader="ksTree",
        alpha_tested=True,
        properties=[MaterialProperty("ksAlphaRef", a=0.4)],
        textures=[TextureMapping("txDiffuse", 0, "trees.dds")],
    )
    glass = Material("glass", blend_mode=BlendMode.ALPHA_BLEND)
    group = Dummy(
        "GROUP",
        transform=translation_transform(np.array([10.0, 0.0, 0.0])),
        children=[_quad("trees_near_0", 1, 0.0, lod_out=700.0)],
    )
    root = Dummy(
        "dm_test",
        children=[
            _quad("1ROAD_0", 0, 0.0),
            _quad("1WALL_0", 0, 4.0, is_renderable=False),
            _quad("glass_0", 2, 8.0),
            group,
            Dummy("AC_START_0", transform=translation_transform(np.array([1.0, 2.0, 3.0]))),
        ],
    )
    return Kn5(
        root=root,
        textures=[
            Texture("asphalt.png", _png((90, 90, 90))),
            Texture("trees.dds", encode_dds(leaf, "BC3")),
            Texture("unused.png", _png((255, 0, 0))),
        ],
        materials=[road, trees, glass],
    )


def _accessor(gltf: dict, blob: bytes, index: int) -> np.ndarray:
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    dtype = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}[acc["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[acc["type"]]
    data = np.frombuffer(blob, dtype, acc["count"] * width, view["byteOffset"])
    return data.reshape(acc["count"], width) if width > 1 else data


def _image(gltf: dict, blob: bytes, texture_name: str) -> tuple[str, bytes]:
    img = next(i for i in gltf["images"] if i["name"] == texture_name)
    view = gltf["bufferViews"][img["bufferView"]]
    return img["mimeType"], blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]


def test_kn5_to_glb_geometry_materials_and_textures(tmp_path: Path) -> None:
    kn5 = _tiny_kn5()
    kn5_path = write_kn5(kn5, tmp_path / "dm_test.kn5")

    glb_path, meta = kn5glb.cached_glb(kn5_path, log=lambda *_: None)
    assert glb_path == tmp_path / ".dm_test.kn5view.glb"  # dotfile: skipped by CSP install
    gltf, blob = kn5glb.read_glb(glb_path.read_bytes())
    assert len(blob) % 4 == 0

    nodes = {n["name"]: n for n in gltf["nodes"]}
    meshes = {m["name"]: m for m in gltf["meshes"]}
    assert set(meshes) == {"1ROAD_0", "1WALL_0", "glass_0", "trees_near_0"}

    # Geometry is copied as is (AC and glTF share frame, winding and UV origin).
    src = kn5.root.children[0]
    prim = meshes["1ROAD_0"]["primitives"][0]
    np.testing.assert_array_equal(
        _accessor(gltf, blob, prim["attributes"]["POSITION"]), src.positions
    )
    np.testing.assert_array_equal(_accessor(gltf, blob, prim["attributes"]["NORMAL"]), src.normals)
    np.testing.assert_array_equal(_accessor(gltf, blob, prim["attributes"]["TEXCOORD_0"]), src.uvs)
    np.testing.assert_array_equal(_accessor(gltf, blob, prim["indices"]), src.indices)
    pos_acc = gltf["accessors"][prim["attributes"]["POSITION"]]
    assert pos_acc["min"] == [0.0, 0.0, -2.0] and pos_acc["max"] == [2.0, 0.0, 0.0]

    # Dummy transforms: DirectX row-major = glTF column-major; world positions in the sidecar.
    assert nodes["GROUP"]["matrix"][12:15] == [10.0, 0.0, 0.0]
    assert "matrix" not in nodes["dm_test"]
    assert meta["dummies"]["AC_START_0"] == [1.0, 2.0, 3.0]

    # Hidden collision walls are flagged, drawn meshes are not; LOD distances are kept.
    assert nodes["1WALL_0"]["extras"]["hidden"] is True
    assert nodes["1ROAD_0"]["extras"]["hidden"] is False
    assert nodes["trees_near_0"]["extras"]["lodOut"] == 700.0
    assert meta["stats"]["hidden_meshes"] == 1

    # Materials: opaque road, alpha-tested trees (cutoff = ksAlphaRef), blended glass. AC has no
    # double-sided materials, so none are double-sided.
    mats = {m["name"]: m for m in gltf["materials"]}
    assert "alphaMode" not in mats["road"]
    assert mats["trees"]["alphaMode"] == "MASK"
    assert abs(mats["trees"]["alphaCutoff"] - 0.4) < 1e-6
    assert mats["glass"]["alphaMode"] == "BLEND"
    assert not any(m.get("doubleSided") for m in gltf["materials"])
    assert mats["road"]["extras"]["ksDiffuse"] == np.float32(0.6)

    # Textures: only txDiffuse ones; small PNGs pass through, BC3 DDS decodes with its alpha.
    assert {i["name"] for i in gltf["images"]} == {"asphalt.png", "trees.dds"}
    mime, data = _image(gltf, blob, "asphalt.png")
    assert mime == "image/png" and data == kn5.textures[0].data
    mime, data = _image(gltf, blob, "trees.dds")
    assert mime == "image/png"
    alpha = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))[..., 3]
    assert alpha[:4].max() == 0 and alpha[4:].min() == 255
    tex = gltf["textures"][mats["trees"]["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
    assert gltf["images"][tex["source"]]["name"] == "trees.dds"


def test_glb_cache_is_keyed_on_kn5_mtime(tmp_path: Path) -> None:
    kn5_path = write_kn5(_tiny_kn5(), tmp_path / "dm_test.kn5")
    glb_path, _ = kn5glb.cached_glb(kn5_path, log=lambda *_: None)
    # A re-conversion would give it a fresh mtime. Whole seconds: Windows keeps 100 ns ticks.
    mark = 1_000_000_000
    os.utime(glb_path, ns=(mark, mark))

    kn5glb.cached_glb(kn5_path, log=lambda *_: None)
    assert glb_path.stat().st_mtime_ns == mark  # cache hit

    st = kn5_path.stat()
    os.utime(kn5_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    _, meta = kn5glb.cached_glb(kn5_path, log=lambda *_: None)
    assert glb_path.stat().st_mtime_ns != mark  # KN5 changed: converted again
    assert meta["key"]["mtime_ns"] == kn5_path.stat().st_mtime_ns
    assert json.loads(glb_path.with_suffix(".json").read_text())["key"] == meta["key"]
