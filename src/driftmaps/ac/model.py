"""In-memory model of a KN5 file (Assetto Corsa's 3D model container).

This module is the shared contract between the KN5 writer/reader (`driftmaps.ac.kn5`) and the
geometry code that produces meshes. Field order and meaning follow AcTools' `Kn5Writer.cs`; see
`docs/research/ac_format.md` §4.

Conventions:
- Vertices are in AC world space (X = east, Y = up, Z = south; see `driftmaps.ac.coords`).
- Mesh nodes have no transform of their own; they sit under identity dummies.
- `Dummy.transform` is a 4x4 matrix in DirectX row-vector convention: rows 0-2 are the node's
  local X, Y, Z axes in parent space, row 3 is the translation (floats 12-14 when flattened).
- Index buffers are uint16, so a mesh holds at most 65,535 vertices (`MAX_MESH_VERTICES`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

KN5_MAGIC = b"sc6969"
KN5_VERSION = 5
MAX_MESH_VERTICES = 65535


class BlendMode(IntEnum):
    OPAQUE = 0
    ALPHA_BLEND = 1
    ALPHA_TO_COVERAGE = 2


class DepthMode(IntEnum):
    NORMAL = 0
    READ_ONLY = 1
    OFF = 2


@dataclass
class Texture:
    """A texture embedded in the KN5: the raw bytes of a DDS or PNG file."""

    name: str
    data: bytes
    active: bool = True


@dataclass
class MaterialProperty:
    """A shader constant. Most shaders only read `a` (e.g. ksDiffuse = 0.5)."""

    name: str
    a: float = 0.0
    b: tuple[float, float] = (0.0, 0.0)
    c: tuple[float, float, float] = (0.0, 0.0, 0.0)
    d: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass
class TextureMapping:
    """Binds a shader texture slot (e.g. "txDiffuse") to an embedded texture by name."""

    slot_name: str
    slot: int
    texture_name: str


@dataclass
class Material:
    name: str
    shader: str = "ksPerPixel"
    blend_mode: BlendMode = BlendMode.OPAQUE
    alpha_tested: bool = False
    depth_mode: DepthMode = DepthMode.NORMAL
    properties: list[MaterialProperty] = field(default_factory=list)
    textures: list[TextureMapping] = field(default_factory=list)


def identity_transform() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


@dataclass
class Dummy:
    """A transform node (KN5 node class 1). Spawn points, timing gates and groups are dummies."""

    name: str
    transform: np.ndarray = field(default_factory=identity_transform)
    children: list[Node] = field(default_factory=list)
    active: bool = True


@dataclass
class Mesh:
    """A static mesh (KN5 node class 2). Arrays are float32 except indices (uint16).

    positions (N, 3), normals (N, 3), uvs (N, 2), tangents (N, 3), indices (M,) with M % 3 == 0.
    Triangles wind counter-clockwise when seen from the visible side (right-handed, Y up).
    """

    name: str
    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    tangents: np.ndarray
    indices: np.ndarray
    material_id: int
    cast_shadows: bool = True
    is_visible: bool = True
    is_transparent: bool = False
    is_renderable: bool = True
    layer: int = 0
    lod_in: float = 0.0
    lod_out: float = 0.0
    active: bool = True
    # Optional override; when None the writer computes a loose sphere
    # (centre = AABB centre, radius = max distance to a vertex).
    bounding_sphere: tuple[tuple[float, float, float], float] | None = None

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])


Node = Dummy | Mesh


@dataclass
class Kn5:
    root: Dummy
    textures: list[Texture] = field(default_factory=list)
    materials: list[Material] = field(default_factory=list)
    version: int = KN5_VERSION
    # Only written/read when version > 5 (meaning undocumented; KinG-InFeT writes 0).
    extra: int = 0

    def material_index(self, name: str) -> int:
        for i, m in enumerate(self.materials):
            if m.name == name:
                return i
        raise KeyError(f"no material named {name!r}")

    def iter_nodes(self):
        """Depth-first pre-order walk over all nodes, starting at the root."""
        stack: list[Node] = [self.root]
        while stack:
            node = stack.pop()
            yield node
            if isinstance(node, Dummy):
                stack.extend(reversed(node.children))
