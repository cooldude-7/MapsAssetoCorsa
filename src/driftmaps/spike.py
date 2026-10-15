"""The M0 in-game test tracks (`drift-maps spike`; player checklist: docs/checklists/M0.md).

Each track is a complete `content/tracks/<id>/` folder that isolates ONE open question from
docs/research/ac_format.md §0.1 (spikes S1-S13) or docs/research/ai_line.md, so one in-game
result settles one thing:

    dm_t1_minimal        S1   minimal folder: flat 200 m strip, PNG asphalt, grass, spawns only
    dm_t2_handedness     S4   L-shaped road turning LEFT, "N" sign, painted arrow; + S11 minimap
    dm_t3_walls          S3   invisible 1WALL_L/R (isRenderable = false) + a visible control wall
    dm_t4_ab_timing      S6   A->B gates AC_AB_START, AC_TIME_0 split, AC_AB_FINISH on an S-bend
    dm_t5_swapped_gates  S7   t4 with every gate's L and R swapped (timing should NOT fire)
    dm_t6_spawns         S5   AC_PIT_0 / AC_START_0 / AC_HOTLAP_START_0 at 0 / 1 / 2 m height
    dm_t7_textures       S2   three lanes: power-of-two PNG, DDS BC1 + mips, 1000x600 PNG
    dm_t8_models_ini     S8   models.ini: main KN5 (dummies + start pad) + three road chunk KN5s
    dm_t9_limits      S9+S10  65,535-vertex mesh, unknown dummies (one empty), lodOut posts
    dm_t10_kn5v6         S13  t1 written as KN5 version 6 (extra = 0)
    dm_t11_ai_loop       AI   closed loop with a generated fast_lane.ai (has_grid = 0)
    dm_t12_ai_ab         AI   t2's L road, 10 m wide, A->B gates, open fast_lane.ai with grid

Conventions shared by every track (see `driftmaps.ac.coords` and ac_format.md §2-§3):
- AC frame X = east, Y = up, Z = south. Roads start at the origin. All roads are flat (y = 0).
- Roads are 8 m wide unless stated, with 15 m physical grass shoulders (`1GRASS_*`) each side.
  Road meshes are `1ROAD_*`; collision walls are `1WALL_*`. Visual-only markers (posts, signs,
  painted arrows and labels) never carry a surface prefix and never contain "WALL".
- Every `AC_*` dummy is a direct child of the root dummy. Spawns use `coords.frame_transform`
  (local +Z along the road in the driving direction) 1.0 m above the road unless the test says
  otherwise. Gates put `_L` on the driver's left (`coords.left_of`), road half-width + 5 m out.
- KN5 version 5 and PNG textures everywhere, except where a track tests exactly that (t7 DDS
  lane, t10 v6), so a failure points at one cause. t1 must pass before anything else counts.
- Output is deterministic: building twice gives identical bytes (the zip too).

Format details not verified from source are marked ASSUMED where they are chosen.
"""

from __future__ import annotations

import functools
import shutil
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Unpack

import numpy as np
from PIL import Image

from driftmaps.ac.ai_line import AiLine, build_ai_line, write_ai
from driftmaps.ac.coords import WORLD_UP, frame_transform, left_of, translation_transform
from driftmaps.ac.kn5 import write_kn5
from driftmaps.ac.model import (
    Dummy,
    Kn5,
    Material,
    MaterialProperty,
    Mesh,
    Node,
    Texture,
    TextureMapping,
)
from driftmaps.ac.textures import (
    Color,
    asphalt_texture,
    debug_grid_texture,
    grass_texture,
    kn5_texture,
    label_texture,
    solid_texture,
)
from driftmaps.ac.track_files import (
    UiTrack,
    cameras_along,
    render_map,
    validate_track_id,
    write_cameras_ini,
    write_crew_ini,
    write_data_license,
    write_groove_ini,
    write_lighting_ini,
    write_models_ini,
    write_outline_png,
    write_preview_png,
    write_surfaces_ini,
    write_ui_track_json,
)
from driftmaps.geometry.centreline import (
    Arc,
    Straight,
    cumulative_length,
    frames,
    path,
    rounded_rectangle_loop,
    straight,
)
from driftmaps.geometry.primitives import (
    MeshData,
    MeshFlags,
    billboard,
    merge,
    offset_polyline,
    post,
    quad,
    ribbon,
    split,
    to_kn5_mesh,
    wall,
)

__all__ = [
    "GATE_EXTRA_M",
    "GATE_HEIGHT_M",
    "GRASS_WIDTH_M",
    "SPAWN_HEIGHT_M",
    "SPIKE_TAG",
    "SPIKE_TRACK_IDS",
    "Road",
    "SpikeTrack",
    "build_spike_tracks",
    "make_spike_track",
    "write_spike_track",
    "zip_tracks",
]

SPIKE_TAG = "drift-maps-test"

ROAD_WIDTH_M = 8.0
GRASS_WIDTH_M = 15.0
# Spawn height above the road (ac_format.md §2.2: guidance conflicts, 0-2 m; t6 tests it).
SPAWN_HEIGHT_M = 1.0
# Gate dummies sit this far beyond the road edge on each side (AutoX used >= 14 m half-width so
# wide cars always cross the segment; ac_format.md §2.3).
GATE_EXTRA_M = 5.0
# ASSUMED: AC tests gate crossings in the horizontal plane, so the gate height only needs to be
# near the car. Timing dummies "don't need orientation, only position" (basic guide).
GATE_HEIGHT_M = 1.0
# Spawn grid: alternate left/right of the centreline, each car `SPAWN_GAP_M` behind the last
# (so 2 x gap between cars on the same side; a road car is ~4.5 m long).
SPAWN_LATERAL_M = 2.0
SPAWN_GAP_M = 4.0
# Painted arrows and labels float this far above the road (visual only; no collision).
DECAL_LIFT_M = 0.03
# Map resolution for these short test roads.
MAP_METERS_PER_PX = 0.5

# A real place for t2/t12's ui_track.json geotags: the M1 Aurora, IL test route start
# (docs/PLAN.md §6). The geometry is synthetic; only CM's map pin uses these.
AURORA_GEOTAGS = (41.71712, -88.29689)


# --------------------------------------------------------------------------------------------
# Roads
# --------------------------------------------------------------------------------------------


@dataclass
class Road:
    """A road centreline (N, 3) in AC coordinates, in driving order, with a constant half-width.

    `closed` roads are loops given without a repeated end point; distances along them wrap.
    `at(s)` gives the position, forward and left unit vectors at arc length `s`.
    """

    centre: np.ndarray
    half_width: float
    closed: bool = False

    def __post_init__(self) -> None:
        pts = np.asarray(self.centre, dtype=np.float64)
        self.centre = pts
        tangents, _, _ = frames(pts, self.closed)
        if self.closed:
            pts = np.vstack([pts, pts[:1]])
            tangents = np.vstack([tangents, tangents[:1]])
        self._pts = pts
        self._tangents = tangents
        self._s = cumulative_length(pts)

    @property
    def length(self) -> float:
        return float(self._s[-1])

    @property
    def polyline(self) -> np.ndarray:
        """The centreline as drawn: loops get their first point repeated at the end."""
        return self._pts

    @property
    def width(self) -> float:
        return 2.0 * self.half_width

    def _wrap(self, s: float) -> float:
        if self.closed:
            return float(s) % self.length
        if not -1e-9 <= s <= self.length + 1e-9:
            raise ValueError(f"s={s} is outside the road (0..{self.length:.3f} m)")
        return min(max(float(s), 0.0), self.length)

    def at(self, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(position, forward, left) at distance `s` from the start (linear interpolation)."""
        s = self._wrap(s)
        i = int(np.clip(np.searchsorted(self._s, s, side="right") - 1, 0, len(self._s) - 2))
        seg = self._s[i + 1] - self._s[i]
        t = 0.0 if seg <= 0.0 else (s - self._s[i]) / seg
        if t == 0.0:
            pos = self._pts[i].copy()
        elif t == 1.0:
            pos = self._pts[i + 1].copy()
        else:
            pos = self._pts[i] + t * (self._pts[i + 1] - self._pts[i])
        fwd = (1.0 - t) * self._tangents[i] + t * self._tangents[i + 1]
        fwd = fwd / np.linalg.norm(fwd)
        return pos, fwd, left_of(fwd)

    def slice(self, s0: float, s1: float) -> np.ndarray:
        """The open centreline between distances s0 < s1 (end points interpolated)."""
        if self.closed:
            raise ValueError("slice() is for open roads")
        a, b = self._wrap(s0), self._wrap(s1)
        if not b > a:
            raise ValueError(f"need s0 < s1, got {s0}, {s1}")
        inner = (self._s > a + 1e-6) & (self._s < b - 1e-6)
        return np.vstack([self.at(a)[0], self._pts[inner], self.at(b)[0]])

    def nearest(self, point: np.ndarray) -> tuple[float, float]:
        """(s, horizontal distance) of the centreline point nearest to `point` (XZ plane)."""
        p = np.asarray(point, dtype=np.float64)[[0, 2]]
        a = self._pts[:-1, [0, 2]]
        ab = self._pts[1:, [0, 2]] - a
        denom = np.maximum((ab * ab).sum(axis=1), 1e-18)
        t = np.clip(((p - a) * ab).sum(axis=1) / denom, 0.0, 1.0)
        d = np.linalg.norm(a + t[:, None] * ab - p, axis=1)
        i = int(np.argmin(d))
        return float(self._s[i] + t[i] * (self._s[i + 1] - self._s[i])), float(d[i])


# --------------------------------------------------------------------------------------------
# Textures and materials
# --------------------------------------------------------------------------------------------

_WHITE: Color = (255, 255, 255)
_PAINT_BG: Color = (25, 70, 170)  # painted road labels: white on blue, unlike any surface
_POST_COLOURS: dict[str, Color] = {
    "green": (40, 180, 70),
    "yellow": (245, 200, 30),
    "red": (215, 40, 35),
    "blue": (40, 90, 220),
    "orange": (255, 130, 0),
    "grey": (150, 150, 150),
}


def _label(text: str, arrow: str | None = None) -> Callable[[], Image.Image]:
    return lambda: label_texture(text, 512, fg=_WHITE, bg=_PAINT_BG, arrow=arrow, border_px=12)


# Texture file name (inside the KN5) -> image factory. The extension picks the encoding
# (`textures.kn5_texture`): .png = PNG bytes, .dds = BC1 DDS with a full mip chain.
_TEXTURES: dict[str, Callable[[], Image.Image]] = {
    "asphalt.png": lambda: asphalt_texture(256),
    "grass.png": lambda: grass_texture(256),
    # Magenta, so an "invisible" wall that does get drawn is impossible to miss.
    "wall_hidden.png": lambda: solid_texture((255, 0, 255), 16),
    "wall_visible.png": lambda: label_texture(
        "WALL", (512, 256), fg=_WHITE, bg=(200, 30, 30), border_px=16
    ),
    **{
        f"post_{c}.png": (lambda rgb=rgb: solid_texture(rgb, 16))
        for c, rgb in _POST_COLOURS.items()
    },
    "sign_n.png": lambda: label_texture("N", 512, fg=_WHITE, bg=_PAINT_BG, border_px=20),
    "paint_left.png": _label("LEFT", "left"),
    "paint_pit.png": _label("PIT 0.0 m", "up"),
    "paint_start.png": _label("START 1.0 m", "up"),
    "paint_hotlap.png": _label("HOTLAP 2.0 m", "up"),
    "paint_png.png": _label("PNG"),
    "paint_dds.png": _label("DDS"),
    "paint_npot.png": _label("NPOT"),
    "paint_main.png": _label("MAIN KN5"),
    **{f"paint_chunk{k}.png": _label(f"CHUNK {k}") for k in (1, 2, 3)},
    # t7 lanes: a high-contrast metre grid shimmers badly at a distance without mipmaps.
    # 1024 px = one 4 m x 4 m tile (8 cells of 0.5 m) for the two power-of-two lanes.
    "lane_pow2.png": lambda: debug_grid_texture(1024, cells=8, metres_per_cell=0.5),
    "lane_pow2.dds": lambda: debug_grid_texture(1024, cells=8, metres_per_cell=0.5),
    # 1000 x 600 px = 4 m x 2.4 m (cells of 0.4 m), the same 250 px per metre: tiles exactly.
    "lane_npot.png": lambda: debug_grid_texture(1000, cells=10, metres_per_cell=0.4).crop(
        (0, 0, 1000, 600)
    ),
}


@functools.cache
def _texture_bytes(name: str) -> bytes:
    return kn5_texture(name, _TEXTURES[name](), "BC1", mipmaps=True).data


def _texture(name: str) -> Texture:
    return Texture(name=name, data=_texture_bytes(name))


def _ks_per_pixel(name: str, texture_name: str, *, bright: bool = False) -> Material:
    """A `ksPerPixel` material with one diffuse texture (ac_format.md §5 "v1" values).

    `bright` lifts ambient so signs and markers read well in shadow.
    """
    return Material(
        name=name,
        shader="ksPerPixel",
        properties=[
            MaterialProperty("ksAmbient", a=0.6 if bright else 0.45),
            MaterialProperty("ksDiffuse", a=0.6),
            MaterialProperty("ksSpecular", a=0.08),
            MaterialProperty("ksSpecularEXP", a=12.0),
            MaterialProperty("ksEmissive", c=(0.0, 0.0, 0.0)),
            MaterialProperty("ksAlphaRef", a=0.0),
        ],
        textures=[TextureMapping("txDiffuse", 0, texture_name)],
    )


# --------------------------------------------------------------------------------------------
# Scene builder (one KN5)
# --------------------------------------------------------------------------------------------


class _Scene:
    """Collects the textures, materials and root children of one KN5 file."""

    def __init__(self, root_name: str) -> None:
        self.root_name = root_name
        self.textures: dict[str, Texture] = {}
        self.materials: list[Material] = []
        self.nodes: list[Node] = []

    def material(self, name: str, texture_name: str, *, bright: bool = False) -> int:
        """Index of material `name` (one diffuse texture), adding it on first use."""
        for i, mat in enumerate(self.materials):
            if mat.name == name:
                return i
        if texture_name not in self.textures:
            self.textures[texture_name] = _texture(texture_name)
        self.materials.append(_ks_per_pixel(name, texture_name, bright=bright))
        return len(self.materials) - 1

    def mesh(self, data: MeshData, name: str, material: int, **flags: Unpack[MeshFlags]) -> Mesh:
        mesh = to_kn5_mesh(data, name, material, **flags)
        self.nodes.append(mesh)
        return mesh

    def add(self, *nodes: Node) -> None:
        self.nodes.extend(nodes)

    def kn5(self, version: int = 5) -> Kn5:
        return Kn5(
            root=Dummy(self.root_name, children=list(self.nodes)),
            textures=list(self.textures.values()),
            materials=list(self.materials),
            version=version,
            extra=0,
        )


_HIDDEN_WALL_FLAGS: MeshFlags = {"is_renderable": False, "cast_shadows": False}
_DECAL_FLAGS: MeshFlags = {"cast_shadows": False}


def _add_ground(
    scene: _Scene,
    centre: np.ndarray,
    half_width: float,
    *,
    closed: bool = False,
    road_name: str = "1ROAD_main",
    grass_suffix: str = "",
    s_start: float = 0.0,
) -> None:
    """Road ribbon (`road_name`) plus 15 m grass shoulders `1GRASS_L/R<suffix>`.

    Road and grass share their edge vertices exactly, so there is no crack or step between them.
    """
    road_mat = scene.material("road", "asphalt.png")
    grass_mat = scene.material("grass", "grass.png")
    road = ribbon(centre, half_width, half_width, 4, 4.0, 4.0, closed, s_start=s_start)
    pieces = split(road)
    for i, piece in enumerate(pieces):
        scene.mesh(piece, road_name if len(pieces) == 1 else f"{road_name}_{i}", road_mat)
    outer = half_width + GRASS_WIDTH_M
    for side, lw, rw in (("L", outer, -half_width), ("R", -half_width, outer)):
        grass = ribbon(centre, lw, rw, 3, 8.0, 8.0, closed, s_start=s_start)
        scene.mesh(grass, f"1GRASS_{side}{grass_suffix}", grass_mat)


def _decal(road: Road, s: float, lateral: float, width: float, length: float) -> MeshData:
    """A flat painted marking on the road: image top edge points forward, not mirrored."""
    pos, fwd, left = road.at(s)
    centre = pos + lateral * left + DECAL_LIFT_M * WORLD_UP
    return quad(centre, -left, fwd, width, length)


def _add_paint(
    scene: _Scene, name: str, texture: str, road: Road, s: float, lateral: float, size: float
) -> None:
    mat = scene.material(texture.removesuffix(".png"), texture, bright=True)
    scene.mesh(_decal(road, s, lateral, size, size), name, mat, **_DECAL_FLAGS)


def _post_at(base: np.ndarray, height: float, width: float) -> MeshData:
    return post(np.asarray(base, dtype=np.float64), height, width)


# --------------------------------------------------------------------------------------------
# Dummies
# --------------------------------------------------------------------------------------------


def _spawn(name: str, road: Road, s: float, lateral: float, height: float) -> Dummy:
    """A spawn dummy facing down the road (+Z forward, +X = driver's left), `height` up."""
    pos, fwd, left = road.at(s)
    return Dummy(name, frame_transform(pos + lateral * left + height * WORLD_UP, fwd))


@dataclass
class _Spawns:
    dummies: list[Dummy] = field(default_factory=list)
    heights: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, road: Road, s: float, lateral: float, height: float) -> None:
        self.dummies.append(_spawn(name, road, s, lateral, height))
        self.heights[name] = height


def _spawn_grid(
    road: Road,
    first_start_s: float,
    hotlap_s: float,
    *,
    starts: int = 4,
    pits: int = 4,
    lateral: float = SPAWN_LATERAL_M,
    height: float = SPAWN_HEIGHT_M,
) -> _Spawns:
    """A staggered two-column grid: AC_START_n from `first_start_s` backwards, then the pits
    AC_PIT_n behind them, plus AC_HOTLAP_START_0 on the centreline at `hotlap_s`.

    Car n sits n * 4 m behind the first, alternating left/right of the centreline.
    """
    spawns = _Spawns()
    first_pit_s = first_start_s - (starts + 1) * SPAWN_GAP_M
    for n in range(pits):
        side = lateral if n % 2 == 0 else -lateral
        spawns.add(f"AC_PIT_{n}", road, first_pit_s - n * SPAWN_GAP_M, side, height)
    for n in range(starts):
        side = lateral if n % 2 == 0 else -lateral
        spawns.add(f"AC_START_{n}", road, first_start_s - n * SPAWN_GAP_M, side, height)
    spawns.add("AC_HOTLAP_START_0", road, hotlap_s, 0.0, height)
    return spawns


def _gate(prefix: str, road: Road, s: float, *, swapped: bool = False) -> list[Dummy]:
    """`<prefix>_L` on the driver's left and `<prefix>_R` on the right (swapped: reversed),
    each road half-width + 5 m from the centreline, oriented along the road."""
    pos, fwd, left = road.at(s)
    reach = road.half_width + GATE_EXTRA_M
    up = GATE_HEIGHT_M * WORLD_UP
    sides = {"L": pos + reach * left + up, "R": pos - reach * left + up}
    if swapped:
        sides = {"L": sides["R"], "R": sides["L"]}
    return [Dummy(f"{prefix}_{k}", frame_transform(sides[k], fwd)) for k in ("L", "R")]


def _gate_posts(scene: _Scene, road: Road, s: float, colour: str, name: str) -> None:
    """Visual-only coloured posts where the gate ends stand (always at their true L/R spots)."""
    pos, _, left = road.at(s)
    reach = road.half_width + GATE_EXTRA_M
    posts = [_post_at(pos + k * reach * left, 3.0, 0.4) for k in (1.0, -1.0)]
    mat = scene.material(f"post_{colour}", f"post_{colour}.png", bright=True)
    scene.mesh(merge(posts), name, mat)


def _end_wall(scene: _Scene, road: Road) -> None:
    """A visible, physical barrier across road + grass 1 m before the end of an open road."""
    pos, _, left = road.at(road.length - 1.0)
    reach = road.half_width + GRASS_WIDTH_M
    line = np.vstack([pos + reach * left, pos - reach * left])
    mat = scene.material("wall_visible", "wall_visible.png", bright=True)
    scene.mesh(wall(line, 1.5, double_sided=True, facing="right", u_tile_m=3.0), "1WALL_END", mat)


# --------------------------------------------------------------------------------------------
# Track description
# --------------------------------------------------------------------------------------------


@dataclass
class SpikeTrack:
    """Everything needed to write one test track folder, kept for tests to check against.

    `spawn_heights` maps each spawn dummy name to its height above the road; `gates` maps each
    gate prefix (e.g. "AC_AB_START") to its distance along `road`; `swapped_gates` marks t5.
    `extra_kn5` holds additional KN5 files (file name -> model) listed in models.ini after
    `<id>.kn5`. `ai_line` is written to ai/fast_lane.ai and data/ideal_line.ai.
    """

    track_id: str
    number: int
    title: str
    description: str
    kn5: Kn5
    road: Road
    spawn_heights: dict[str, float]
    gates: dict[str, float] = field(default_factory=dict)
    swapped_gates: bool = False
    tags: list[str] = field(default_factory=list)
    run: str = "A2B"
    extra_kn5: dict[str, Kn5] = field(default_factory=dict)
    ai_line: AiLine | None = None
    geotags: tuple[float, float] | None = None
    country: str = ""
    city: str = ""

    @property
    def name(self) -> str:
        return f"DM T{self.number} {self.title}"

    @property
    def kn5_files(self) -> list[str]:
        return [f"{self.track_id}.kn5", *self.extra_kn5]

    @property
    def pit_count(self) -> int:
        return sum(1 for n in self.spawn_heights if n.startswith("AC_PIT_"))


def _tags(*extra: str) -> list[str]:
    return [SPIKE_TAG, "generated", *extra]


def _finish(scene: _Scene, spawns: _Spawns, gates: Sequence[Dummy] = (), version: int = 5) -> Kn5:
    scene.add(*spawns.dummies, *gates)
    return scene.kn5(version)


# --------------------------------------------------------------------------------------------
# The tracks
# --------------------------------------------------------------------------------------------


def _minimal(track_id: str, number: int, title: str, version: int, description: str) -> SpikeTrack:
    road = Road(straight((0.0, 0.0, 0.0), 0.0, 200.0), ROAD_WIDTH_M / 2)
    scene = _Scene(track_id)
    _add_ground(scene, road.centre, road.half_width)
    spawns = _spawn_grid(road, first_start_s=60.0, hotlap_s=10.0)
    return SpikeTrack(
        track_id=track_id,
        number=number,
        title=title,
        description=description,
        kn5=_finish(scene, spawns, version=version),
        road=road,
        spawn_heights=spawns.heights,
        tags=_tags(),
        run="straight",
    )


def _t1_minimal() -> SpikeTrack:
    return _minimal(
        "dm_t1_minimal",
        1,
        "Minimal",
        5,
        "drift-maps M0 test S1 (do this one first). A flat 200 m straight heading north: KN5 "
        "v5, PNG asphalt, grass, spawn points only. PASS: it loads in Practice, the car sits "
        "on the road facing up the straight, and the road and grass are drawn.",
    )


def _t10_kn5v6() -> SpikeTrack:
    return _minimal(
        "dm_t10_kn5v6",
        10,
        "KN5 v6",
        6,
        "drift-maps M0 test S13. Exactly the T1 track, but the model file uses KN5 version 6. "
        "PASS: it loads in Practice and looks like T1.",
    )


def _l_road(half_width: float, west_m: float) -> Road:
    """150 m north, a 90 degree LEFT turn of radius 30 m, then `west_m` metres west."""
    centre = path((0.0, 0.0, 0.0), 0.0, [Straight(150.0), Arc(30.0, 90.0), Straight(west_m)])
    return Road(centre, half_width)


def _t2_handedness() -> SpikeTrack:
    track_id = "dm_t2_handedness"
    road = _l_road(ROAD_WIDTH_M / 2, 150.0)
    scene = _Scene(track_id)
    _add_ground(scene, road.centre, road.half_width)
    # Painted arrow 35 m before the corner (which starts at s = 150 m), pointing left.
    _add_paint(scene, "paint_arrow_left", "paint_left.png", road, 115.0, 0.0, 5.0)
    # A tall single-sided "N" sign beyond the corner, straight ahead of the driver on the first
    # straight, facing south (+Z) towards them. Seen mirrored ("И") = the world is mirrored.
    sign_centre = np.array([0.0, 6.0, -205.0])
    sign_mat = scene.material("sign_n", "sign_n.png", bright=True)
    scene.mesh(billboard(sign_centre, (0.0, 0.0, 1.0), 8.0, 8.0), "sign_N", sign_mat)
    legs = [_post_at((x, 0.0, -205.15), 2.05, 0.25) for x in (-3.5, 3.5)]
    grey = scene.material("post_grey", "post_grey.png", bright=True)
    scene.mesh(merge(legs), "sign_N_legs", grey)
    spawns = _spawn_grid(road, first_start_s=60.0, hotlap_s=10.0)
    return SpikeTrack(
        track_id=track_id,
        number=2,
        title="Handedness",
        description=(
            "drift-maps M0 test S4 + S11. The road goes 150 m north, turns LEFT (90 degrees), "
            "then runs 150 m west. PASS: a painted arrow before the corner points left, the road "
            "turns left, the blue N sign ahead reads N (not mirrored), and the minimap shows the "
            "road going up then left with your dot on it."
        ),
        kn5=_finish(scene, spawns),
        road=road,
        spawn_heights=spawns.heights,
        tags=_tags(),
        run="left turn",
        geotags=AURORA_GEOTAGS,
        country="United States",
        city="Aurora",
    )


def _t3_walls() -> SpikeTrack:
    track_id = "dm_t3_walls"
    road = Road(straight((0.0, 0.0, 0.0), 0.0, 300.0), ROAD_WIDTH_M / 2)
    scene = _Scene(track_id)
    _add_ground(scene, road.centre, road.half_width)
    offset = road.half_width + 1.0  # ~1 m outside the road edge
    hidden = scene.material("wall_hidden", "wall_hidden.png")
    visible = scene.material("wall_visible", "wall_visible.png", bright=True)
    first, last = road.slice(0.0, 250.0), road.slice(250.0, 300.0)
    for side, sign, facing in (("L", 1.0, "right"), ("R", -1.0, "left")):
        base = offset_polyline(first, sign * offset)
        data = wall(base, 2.5, double_sided=True, facing=facing)  # front faces the road
        scene.mesh(data, f"1WALL_{side}", hidden, **_HIDDEN_WALL_FLAGS)
    control = [
        wall(offset_polyline(last, sign * offset), 2.5, double_sided=True, facing=facing)
        for sign, facing in ((1.0, "right"), (-1.0, "left"))
    ]
    scene.mesh(merge(control), "1WALL_VIS", visible)
    # Orange visual-only posts just outside each invisible wall every 25 m, so the player can
    # see where to steer to touch it.
    orange = scene.material("post_orange", "post_orange.png", bright=True)
    marks = []
    for s in np.arange(0.0, 250.0 + 1e-9, 25.0):
        pos, _, left = road.at(float(s))
        marks += [_post_at(pos + k * (offset + 0.5) * left, 1.2, 0.2) for k in (1.0, -1.0)]
    scene.mesh(merge(marks), "post_wall_markers", orange)
    spawns = _spawn_grid(road, first_start_s=60.0, hotlap_s=10.0)
    return SpikeTrack(
        track_id=track_id,
        number=3,
        title="Walls",
        description=(
            "drift-maps M0 test S3. For the first 250 m invisible walls stand 1 m outside both "
            "road edges (orange posts mark them); the last 50 m has a visible red wall as a "
            "control. PASS: you bump into an invisible wall at walking pace, and nothing pink "
            "or magenta is ever drawn."
        ),
        kn5=_finish(scene, spawns),
        road=road,
        spawn_heights=spawns.heights,
        tags=_tags(),
        run="straight",
    )


_AB_GATES = (("AC_AB_START", 50.0, "green"), ("AC_TIME_0", 250.0, "yellow"))
_AB_FINISH = ("AC_AB_FINISH", 450.0, "red")


def _ab_timing(
    track_id: str, number: int, title: str, swapped: bool, description: str
) -> SpikeTrack:
    # ~500 m gentle S-bend heading north: 80 m, 35 degrees left (R 180 m), 40 m, 35 degrees
    # right, 160 m. Gates at 50 / 250 / 450 m, then ~50 m of run-off and a visible end wall.
    centre = path(
        (0.0, 0.0, 0.0),
        0.0,
        [Straight(80.0), Arc(180.0, 35.0), Straight(40.0), Arc(180.0, -35.0), Straight(160.0)],
    )
    road = Road(centre, ROAD_WIDTH_M / 2)
    scene = _Scene(track_id)
    _add_ground(scene, road.centre, road.half_width)
    gates: list[Dummy] = []
    gate_s: dict[str, float] = {}
    for prefix, s, colour in (*_AB_GATES, _AB_FINISH):
        _gate_posts(scene, road, s, colour, f"post_{prefix.lower()}")
        gates += _gate(prefix, road, s, swapped=swapped)
        gate_s[prefix] = s
    _end_wall(scene, road)
    # All spawns behind the start gate (else the timer may start at once; ai_line.md §6).
    spawns = _spawn_grid(road, first_start_s=42.0, hotlap_s=5.0)
    return SpikeTrack(
        track_id=track_id,
        number=number,
        title=title,
        description=description,
        kn5=_finish(scene, spawns, gates),
        road=road,
        spawn_heights=spawns.heights,
        gates=gate_s,
        swapped_gates=swapped,
        # "circuit" only so Content Manager's Time Attack doesn't ask for the tag (CM tags are
        # UI hints, QuickDriveModeViewModel.TagRequired); real A->B tracks won't carry it.
        tags=_tags("A2B", "point-to-point", "circuit"),
    )


def _t4_ab_timing() -> SpikeTrack:
    return _ab_timing(
        "dm_t4_ab_timing",
        4,
        "AB Timing",
        False,
        "drift-maps M0 test S6. A 500 m S-bend with point-to-point timing: start at the green "
        "posts (50 m), split at the yellow posts (250 m), finish at the red posts (450 m), "
        "then 50 m to a wall. PASS: the timer starts at green, shows a split at yellow and "
        "stops at red (try Practice, Hotlap and Time Attack).",
    )


def _t5_swapped_gates() -> SpikeTrack:
    return _ab_timing(
        "dm_t5_swapped_gates",
        5,
        "Swapped Gates",
        True,
        "drift-maps M0 test S7. The same road as T4, but every timing gate has its left and "
        "right ends swapped on purpose. PASS: the timer does NOT start, split or stop.",
    )


def _t6_spawns() -> SpikeTrack:
    track_id = "dm_t6_spawns"
    road = Road(straight((0.0, 0.0, 0.0), 0.0, 250.0), ROAD_WIDTH_M / 2)
    scene = _Scene(track_id)
    _add_ground(scene, road.centre, road.half_width)
    spawns = _Spawns()
    # The three tested spawns stand in the left half of the road; a painted arrow on the right
    # half beside each one shows the intended heading and height.
    tests = (
        ("AC_PIT_0", 60.0, 0.0, "paint_pit"),
        ("AC_START_0", 110.0, 1.0, "paint_start"),
        ("AC_HOTLAP_START_0", 160.0, 2.0, "paint_hotlap"),
    )
    for name, s, height, paint in tests:
        _add_paint(scene, f"{paint}_arrow", f"{paint}.png", road, s, -SPAWN_LATERAL_M, 3.5)
        spawns.add(name, road, s, SPAWN_LATERAL_M, height)
    # Extra pits and starts (same heights as their _0) queue behind the tested ones.
    for n in (1, 2, 3):
        spawns.add(f"AC_PIT_{n}", road, 60.0 - 8.0 * n, SPAWN_LATERAL_M, 0.0)
        spawns.add(f"AC_START_{n}", road, 110.0 - 8.0 * n, SPAWN_LATERAL_M, 1.0)
    order = sorted(spawns.dummies, key=lambda d: _dummy_sort_key(d.name))
    spawns.dummies = order
    return SpikeTrack(
        track_id=track_id,
        number=6,
        title="Spawns",
        description=(
            "drift-maps M0 test S5. Spawn points at three heights above the road: pit (Practice) "
            "0.0 m, start (Time Attack / Race) 1.0 m, hotlap (Hotlap) 2.0 m. Each has a painted "
            "arrow beside it. PASS: the car appears next to the matching arrow, faces the way "
            "it points, and does not drop, bounce or sink."
        ),
        kn5=_finish(scene, spawns),
        road=road,
        spawn_heights=spawns.heights,
        tags=_tags("circuit"),  # "circuit": see _ab_timing (Time Attack in CM)
        run="straight",
    )


def _dummy_sort_key(name: str) -> tuple[int, int]:
    groups = ("AC_PIT_", "AC_START_", "AC_HOTLAP_START_")
    for g, prefix in enumerate(groups):
        if name.startswith(prefix):
            return g, int(name[len(prefix) :])
    return len(groups), 0


def _t7_textures() -> SpikeTrack:
    track_id = "dm_t7_textures"
    lane = 4.0
    road = Road(straight((0.0, 0.0, 0.0), 0.0, 600.0), 1.5 * lane)
    scene = _Scene(track_id)
    # Three 4 m lanes, left to right: PNG (power of two), DDS (BC1 + mips), PNG 1000 x 600.
    # Each lane is its own 1ROAD mesh and material. u spans 0..1 across the lane.
    lanes = (
        ("PNG", "lane_pow2.png", 1.5 * lane, -0.5 * lane, 4.0, "paint_png.png"),
        ("DDS", "lane_pow2.dds", 0.5 * lane, 0.5 * lane, 4.0, "paint_dds.png"),
        ("NPOT", "lane_npot.png", -0.5 * lane, 1.5 * lane, 2.4, "paint_npot.png"),
    )
    for label, tex, lw, rw, v_tile, _ in lanes:
        mat = scene.material(f"lane_{label.lower()}", tex)
        data = ribbon(road.centre, lw, rw, 2, None, v_tile)
        scene.mesh(data, f"1ROAD_{label}", mat)
    grass = scene.material("grass", "grass.png")
    outer = road.half_width + GRASS_WIDTH_M
    for side, lw, rw in (("L", outer, -road.half_width), ("R", -road.half_width, outer)):
        scene.mesh(ribbon(road.centre, lw, rw, 3, 8.0, 8.0), f"1GRASS_{side}", grass)
    for label, _, lw, rw, _, paint in lanes:
        centre_offset = (lw - rw) / 2.0
        _add_paint(scene, f"paint_label_{label.lower()}", paint, road, 50.0, centre_offset, 3.5)
    # Two columns in the outer lanes (Practice starts in the PNG lane); hotlap in the middle.
    spawns = _spawn_grid(road, first_start_s=40.0, hotlap_s=4.0, lateral=lane)
    return SpikeTrack(
        track_id=track_id,
        number=7,
        title="Textures",
        description=(
            "drift-maps M0 test S2. Three 600 m lanes with the same grid pattern, labelled at "
            "the start: PNG (left), DDS with mipmaps (middle), NPOT = a 1000x600 PNG (right). "
            "PASS: all three lanes are drawn (no black or white lane). Note which lanes shimmer "
            "or flicker in the distance."
        ),
        kn5=_finish(scene, spawns),
        road=road,
        spawn_heights=spawns.heights,
        tags=_tags(),
        run="straight",
    )


T8_PAD_M = 60.0


def _t8_models_ini() -> SpikeTrack:
    track_id = "dm_t8_models_ini"
    # One road heading north: a 60 m start pad (main KN5), then three 100 m chunks. The pad is
    # road only (no grass) and holds every spawn, so the car has ground even if chunks fail.
    road = Road(straight((0.0, 0.0, T8_PAD_M), 0.0, T8_PAD_M + 300.0), ROAD_WIDTH_M / 2)
    main = _Scene(track_id)
    pad = ribbon(road.slice(0.0, T8_PAD_M), road.half_width, road.half_width, 4, 4.0, 4.0)
    main.mesh(pad, "1ROAD_pad", main.material("road", "asphalt.png"))
    _add_paint(main, "paint_label_main", "paint_main.png", road, T8_PAD_M - 6.0, 0.0, 4.0)
    extra: dict[str, Kn5] = {}
    for k in (1, 2, 3):
        s0 = T8_PAD_M + 100.0 * (k - 1)
        chunk = _Scene(f"{track_id}_chunk{k}")
        _add_ground(
            chunk,
            road.slice(s0, s0 + 100.0),
            road.half_width,
            road_name=f"1ROAD_c{k}",
            grass_suffix=f"_c{k}",
            s_start=s0,
        )
        _add_paint(chunk, f"paint_label_chunk{k}", f"paint_chunk{k}.png", road, s0 + 6, 0.0, 4.0)
        extra[f"{track_id}_chunk{k}.kn5"] = chunk.kn5()
    spawns = _spawn_grid(road, first_start_s=44.0, hotlap_s=4.0)
    return SpikeTrack(
        track_id=track_id,
        number=8,
        title="Models Ini",
        description=(
            "drift-maps M0 test S8. models.ini loads four KN5 files: the main one (spawn points "
            "and a 60 m start pad marked MAIN KN5) and three 100 m road pieces marked CHUNK 1, "
            "2 and 3. PASS: it loads and you can drive the whole 360 m on road and grass."
        ),
        kn5=_finish(main, spawns),
        road=road,
        spawn_heights=spawns.heights,
        tags=_tags(),
        run="straight",
        extra_kn5=extra,
    )


T9_ROAD_VERTICES = 65_535
T9_LOD_OUT_M = 60.0


def _t9_limits() -> SpikeTrack:
    track_id = "dm_t9_limits"
    # 65,535 = 1285 stations x 51 vertices across (50 lateral segments, 16 cm apart), with
    # stations 0.25 m apart: a 321 m road in ONE mesh at exactly the uint16 limit.
    stations, across = 1285, 51
    centre = np.zeros((stations, 3))
    centre[:, 2] = -np.linspace(0.0, (stations - 1) * 0.25, stations)
    road = Road(centre, ROAD_WIDTH_M / 2)
    scene = _Scene(track_id)
    big = ribbon(road.centre, road.half_width, road.half_width, across - 1, 4.0, 4.0)
    assert big.vertex_count == T9_ROAD_VERTICES
    scene.mesh(big, "1ROAD_65535", scene.material("road", "asphalt.png"))
    grass = scene.material("grass", "grass.png")
    outer = road.half_width + GRASS_WIDTH_M
    for side, lw, rw in (("L", outer, -road.half_width), ("R", -road.half_width, outer)):
        scene.mesh(ribbon(road.centre, lw, rw, 3, 8.0, 8.0), f"1GRASS_{side}", grass)
    spawns = _spawn_grid(road, first_start_s=40.0, hotlap_s=5.0)
    scene.add(*spawns.dummies)  # before anything unusual, so the car can always spawn
    # S10: yellow posts every 20 m on the right, each its own mesh with lodOut = 60 m and no
    # shadows: only the ones within ~60 m of the camera should be drawn.
    yellow = scene.material("post_yellow", "post_yellow.png", bright=True)
    for i, s in enumerate(np.arange(20.0, 301.0, 20.0)):
        pos, _, left = road.at(float(s))
        scene.mesh(
            _post_at(pos - 7.0 * left, 2.5, 0.4),
            f"LOD60_post_{i:02d}",
            yellow,
            lod_out=T9_LOD_OUT_M,
            cast_shadows=False,
        )
    # S9: an unknown-name dummy holding visible blue posts (left side, every 40 m)...
    blue = scene.material("post_blue", "post_blue.png", bright=True)
    group_posts: list[Node] = []
    for i, s in enumerate(np.arange(20.0, 301.0, 40.0)):
        pos, _, left = road.at(float(s))
        group_posts.append(
            to_kn5_mesh(_post_at(pos + 7.0 * left, 3.0, 0.4), f"scenery_post_{i:02d}", blue)
        )
    scene.add(Dummy("GROUP_scenery", children=group_posts))
    # ...then an EMPTY unknown-name dummy, and a red post AFTER it in node order (the last
    # node of the file). If loading stops at unknown empties, the red post is missing.
    marker_pos, _, marker_left = road.at(160.0)
    scene.add(Dummy("DM_EMPTY_MARKER", translation_transform(marker_pos + 10.0 * marker_left)))
    pos, _, left = road.at(30.0)
    red = scene.material("post_red", "post_red.png", bright=True)
    scene.mesh(_post_at(pos - 11.0 * left, 5.0, 0.8), "AFTER_EMPTY_post", red)
    return SpikeTrack(
        track_id=track_id,
        number=9,
        title="Limits",
        description=(
            "drift-maps M0 test S9 + S10. The road is one mesh with exactly 65,535 vertices. "
            "Blue posts (left) sit in an unknown group, a big red post (right) comes after an "
            "empty unknown object, and yellow posts (right) should only appear within about "
            "60 m. PASS: it loads, blue and red posts are there, yellow posts pop in and out."
        ),
        kn5=scene.kn5(),
        road=road,
        spawn_heights=spawns.heights,
        tags=_tags(),
        run="straight",
    )


def _t11_ai_loop() -> SpikeTrack:
    track_id = "dm_t11_ai_loop"
    # ~400 x 150 m loop, 10 m wide, corner radius 40 m, counter-clockwise from above (all
    # left turns). s = 0 is the middle of the south straight (heading east): start/finish.
    road = Road(rounded_rectangle_loop(400.0, 150.0, 40.0), 5.0, closed=True)
    scene = _Scene(track_id)
    _add_ground(scene, road.centre, road.half_width, closed=True)
    # AC_TIME_0 = start/finish. ASSUMED: like Kunos circuits, AC_TIME_1/2 split the lap into
    # three sectors (Hagnhofer docs); a lap needs only AC_TIME_0 if they are ignored.
    gate_s = {"AC_TIME_0": 0.0, "AC_TIME_1": road.length / 3, "AC_TIME_2": 2 * road.length / 3}
    gates = [d for prefix, s in gate_s.items() for d in _gate(prefix, road, s)]
    spawns = _spawn_grid(road, first_start_s=-8.0, hotlap_s=-110.0, starts=8, pits=8)
    # The AI line follows the centreline (no racing-line optimisation in M0). has_grid = 0:
    # the production default (CSP's ALT_GRID builds its own grid; ai_line.md §1.4).
    ai = build_ai_line(road.centre, road.half_width, road.half_width, closed=True)
    return SpikeTrack(
        track_id=track_id,
        number=11,
        title="AI Loop",
        description=(
            "drift-maps M0 AI test. A 1 km loop with a generated AI line (ai/fast_lane.ai, no "
            "lookup grid) and ideal line. PASS: the ideal line is drawn along the road, and in "
            "a 3-lap Quick Race the AI cars drive the loop and finish the race."
        ),
        kn5=_finish(scene, spawns, gates),
        road=road,
        spawn_heights=spawns.heights,
        gates=gate_s,
        tags=_tags("circuit"),
        run="anticlockwise",
        ai_line=ai,
    )


T12_START_S = 90.0
T12_RUNOFF_M = 100.0


def _t12_ai_ab() -> SpikeTrack:
    track_id = "dm_t12_ai_ab"
    half = 5.0  # 10 m wide
    # t2's L road with the west leg 255 m instead of 150 m: the finish stands where t2 ends,
    # the AI line runs 100 m past it, and the road 5 m further to a visible end wall.
    road = _l_road(half, 255.0)
    finish_s = 150.0 + 30.0 * np.pi / 2 + 150.0
    scene = _Scene(track_id)
    _add_ground(scene, road.centre, road.half_width)
    gate_s = {"AC_AB_START": T12_START_S, "AC_AB_FINISH": finish_s}
    gates: list[Dummy] = []
    for (prefix, s), colour in zip(gate_s.items(), ("green", "red"), strict=True):
        _gate_posts(scene, road, s, colour, f"post_{prefix.lower()}")
        gates += _gate(prefix, road, s)
    _end_wall(scene, road)
    spawns = _spawn_grid(road, first_start_s=T12_START_S - 8.0, hotlap_s=6.0, starts=8, pits=8)
    # Open AI line from the road start (behind every spawn) to 100 m past the finish, starting
    # from standstill (cars spawn on it) and braking to a stop at its end. With the lookup grid
    # (unlike t11), so vanilla AC without CSP gets its best chance here too (ai_line.md §1.4,
    # spike S3; CSP's default ALT_GRID=1 ignores it and builds its own).
    ai = build_ai_line(
        road.slice(0.0, finish_s + T12_RUNOFF_M),
        half,
        half,
        closed=False,
        start_speed_mps=0.0,
        end_speed_mps=0.0,
        with_grid=True,
    )
    return SpikeTrack(
        track_id=track_id,
        number=12,
        title="AI AB",
        description=(
            "drift-maps M0 AI test, point to point. T2's L road, 10 m wide, start at the green "
            "posts, finish at the red posts, then 100 m of run-off. Generated AI line (with "
            "lookup grid) and ideal line. PASS: the ideal line follows the road to past the "
            "finish. Note what the AI cars do at and after the finish."
        ),
        kn5=_finish(scene, spawns, gates),
        road=road,
        spawn_heights=spawns.heights,
        gates=gate_s,
        tags=_tags("A2B", "point-to-point", "touge"),
        ai_line=ai,
        geotags=AURORA_GEOTAGS,
        country="United States",
        city="Aurora",
    )


_BUILDERS: dict[str, Callable[[], SpikeTrack]] = {
    "dm_t1_minimal": _t1_minimal,
    "dm_t2_handedness": _t2_handedness,
    "dm_t3_walls": _t3_walls,
    "dm_t4_ab_timing": _t4_ab_timing,
    "dm_t5_swapped_gates": _t5_swapped_gates,
    "dm_t6_spawns": _t6_spawns,
    "dm_t7_textures": _t7_textures,
    "dm_t8_models_ini": _t8_models_ini,
    "dm_t9_limits": _t9_limits,
    "dm_t10_kn5v6": _t10_kn5v6,
    "dm_t11_ai_loop": _t11_ai_loop,
    "dm_t12_ai_ab": _t12_ai_ab,
}

SPIKE_TRACK_IDS: tuple[str, ...] = tuple(_BUILDERS)


def make_spike_track(track_id: str) -> SpikeTrack:
    """Build one test track in memory (no files written)."""
    if track_id not in _BUILDERS:
        raise ValueError(f"unknown spike track {track_id!r}; known: {', '.join(SPIKE_TRACK_IDS)}")
    track = _BUILDERS[track_id]()
    validate_track_id(track.track_id)
    return track


# --------------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------------


def write_spike_track(track: SpikeTrack, out_dir: Path) -> Path:
    """Write `track` as `<out_dir>/<id>/` (an existing folder of that id is replaced)."""
    track_dir = Path(out_dir) / validate_track_id(track.track_id)
    if track_dir.exists():
        shutil.rmtree(track_dir)
    data_dir, ui_dir = track_dir / "data", track_dir / "ui"
    track_dir.mkdir(parents=True)

    write_kn5(track.kn5, track_dir / f"{track.track_id}.kn5")
    for file_name, kn5 in track.extra_kn5.items():
        write_kn5(kn5, track_dir / file_name)
    if track.extra_kn5:
        write_models_ini(track_dir / "models.ini", track.kn5_files)

    road = track.road
    line = road.polyline
    write_surfaces_ini(data_dir / "surfaces.ini")
    render_map([line], road.width, data_dir, track_dir, meters_per_px=MAP_METERS_PER_PX)
    spline_length = float(track.ai_line.lengths[-1]) if track.ai_line is not None else None
    cameras = cameras_along(
        line,
        150.0,
        road_half_width_m=road.half_width,
        spline_length_m=spline_length if not road.closed else None,
    )
    write_cameras_ini(data_dir / "cameras.ini", cameras)
    write_lighting_ini(data_dir / "lighting.ini")
    write_groove_ini(data_dir / "groove.ini")
    write_crew_ini(data_dir / "crew.ini")

    ui = UiTrack(
        name=track.name,
        description=track.description,
        tags=list(track.tags),
        geotags=track.geotags,
        country=track.country,
        city=track.city,
        length_m=road.length,
        width=road.width,
        pitboxes=track.pit_count,
        run=track.run,
        year=2026,
    )
    write_ui_track_json(ui_dir / "ui_track.json", ui)
    write_preview_png(ui_dir / "preview.png", [line], title=f"T{track.number} {track.title}")
    write_outline_png(ui_dir / "outline.png", [line])
    write_data_license(track_dir, [], track_name=track.name)

    if track.ai_line is not None:
        ai_path = write_ai(track.ai_line, track_dir / "ai" / "fast_lane.ai")
        # COMMUNITY: data/ideal_line.ai is "the preferred ideal line shown in-game" (ai_line.md
        # §2.2). Same bytes as fast_lane.ai.
        (data_dir / "ideal_line.ai").write_bytes(ai_path.read_bytes())
    return track_dir


# Fixed metadata so the zip is byte-identical on every build and platform.
_ZIP_DATE = (2026, 1, 1, 0, 0, 0)
_ZIP_FILE_MODE = 0o100644


def zip_tracks(track_dirs: Sequence[Path], zip_path: Path) -> Path:
    """Zip track folders with each folder at the zip root, so extracting the zip into
    `content/tracks/` installs them. Deterministic: sorted entries, fixed dates and modes."""
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[str, Path]] = []
    for track_dir in track_dirs:
        track_dir = Path(track_dir)
        for file in track_dir.rglob("*"):
            if file.is_file():
                rel = file.relative_to(track_dir.parent).as_posix()
                entries.append((rel, file))
    entries.sort(key=lambda e: e[0])
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        for rel, file in entries:
            info = zipfile.ZipInfo(rel, date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3  # Unix, so the mode below means the same everywhere
            info.external_attr = _ZIP_FILE_MODE << 16
            zf.writestr(info, file.read_bytes(), compresslevel=6)
    tmp.replace(zip_path)
    return zip_path


def build_spike_tracks(
    out_dir: Path, only: list[str] | None = None, zip_path: Path | None = None
) -> list[Path]:
    """Build the M0 test tracks into `<out_dir>/<id>/` and return their folders.

    `only` picks track ids (any order; they are built in the canonical order). With `zip_path`,
    the built folders are also packed into that zip (folders at the zip root). Existing folders
    with the same ids are replaced; nothing else in `out_dir` is touched.
    """
    wanted = SPIKE_TRACK_IDS
    if only:
        unknown = sorted(set(only) - set(SPIKE_TRACK_IDS))
        if unknown:
            raise ValueError(
                f"unknown spike track(s) {unknown}; known: {', '.join(SPIKE_TRACK_IDS)}"
            )
        wanted = tuple(t for t in SPIKE_TRACK_IDS if t in set(only))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    built = [write_spike_track(make_spike_track(t), out_dir) for t in wanted]
    if zip_path is not None:
        zip_tracks(built, zip_path)
    return built
