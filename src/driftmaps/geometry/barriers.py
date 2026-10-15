"""Visible roadside barriers with hidden collision: W-beam guardrails, jersey barriers, walls,
and the bridge railings (concrete parapet, steel rail) that `geometry.bridges` puts on decks.

Only REAL barriers are built (docs/research/scenery.md §5, §7): `plan_barriers` combines
- OSM barrier ways matched to the road (`geo.barriers.match_barriers`),
- the user's `guardrails.json` spans (`geo.barriers.load_user_spans`), and
- a guard rail on both sides of every bridge approach (`approach_m`, default 20 m; real
  bridges here have parapets, and their approaches carry rails, ASSUMED),
in that priority, never over a bridge deck (the deck has its own railing). The steep drop-off
heuristic `drop_off_spans` exists but is OFF by default: nothing calls it unless the build
asks for it, because the user wants only real barriers.

Geometry (AC frame: X east, Y up, Z south; left = up x forward):
- A barrier runs along a **face line**: the road's shoulder edge (`RoadModel.outer`) plus
  `gap_m`, or the measured OSM offset if that is further out, built with `road.road_columns`
  so it never folds over on a hairpin (OSM rails mapped within `rail_snap_m` of the edge are
  snapped onto it). Its reference height is the road (or the ground under the face line
  when it stands more than `near_edge_m` beyond the shoulder edge).
- Cross-sections are given as (d, h) profiles, d = metres away from the road, h = metres above
  the reference, ordered counter-clockwise for the LEFT side (d plotted to the right): then
  `sweep` makes every face's front point outward (verified by tests). Right-side profiles are
  mirrored by reversing the point order.
- **W-beam** (`guard_rail`): a 310 mm x 83 mm W rail (front and back faces) with its top at
  0.75 m, on 150 mm posts every 2 m (real Ontario spacing is 1.905 m, ASSUMED close enough)
  with 150 mm blockouts; free ends are turned down into the ground over `terminal_m`.
- **Jersey** (`jersey_barrier`): the New Jersey shape (0.81 m tall, 0.61 m base, 0.15 m top).
- **Wall** (`wall`, `retaining_wall`): a 0.3 m concrete wall, `height` tag or 1.0 m above the
  higher ground beside it.
- **Parapet** / **bridge rail**: see `geometry.bridges`.
- **Collision**: every barrier gets a hidden, double-sided `1WALL_<TAG>_###` strip on its face
  line (ac_format.md §3.2), from `embed_m` below the ground to `collision_height_m` above the
  reference (lower over turned-down terminals), continuous over the whole span.
- Visual meshes have no digit prefix (no physics), use `lod_out`, and are chunked every
  ~`chunk_m` so each stays far below 65,535 vertices.

Textures (`TEXTURES`): procedural concrete and galvanised steel, generated with the same
tileable-noise helpers as `driftmaps.ac.textures`, so we own them outright.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from PIL import Image

from driftmaps.ac.coords import WORLD_UP
from driftmaps.ac.scene import HIDDEN_WALL_FLAGS
from driftmaps.ac.textures import _rng, _spectral_noise, _to_image, _worley
from driftmaps.geo.barriers import BarrierSpan, Side, merge_spans
from driftmaps.geometry.primitives import (
    MeshData,
    MeshFlags,
    _lattice_indices,
    box,
    compute_normals,
    compute_tangents,
    merge,
    wall,
)
from driftmaps.geometry.road import RoadModel, road_columns

__all__ = [
    "BARRIER_STYLES",
    "TEXTURES",
    "BarrierOptions",
    "BarrierPath",
    "GroundFn",
    "ScenePiece",
    "barrier_path",
    "barrier_pieces",
    "bridge_approach_spans",
    "cap",
    "concrete_texture",
    "drop_off_spans",
    "plan_barriers",
    "span_meshes",
    "steel_texture",
    "sweep",
]

GroundFn = Callable[[np.ndarray], np.ndarray]
"""Ground height under XZ points: (M, 2) -> (M,) AC Y; NaN where unknown."""

UP = np.asarray(WORLD_UP, np.float64)

# kind -> (visual mesh name stem, collision tag)
BARRIER_STYLES: dict[str, tuple[str, str]] = {
    "guard_rail": ("guardrail", "GR"),
    "jersey_barrier": ("jersey", "JB"),
    "wall": ("barrier_wall", "BW"),
    "retaining_wall": ("barrier_wall", "BW"),
    "parapet": ("parapet", "PA"),
    "bridge_rail": ("bridge_rail", "BR"),
}

# W-beam front face (d, h above the rail bottom), top -> bottom, peaks towards the road.
# AASHTO M180 W-beam: 310 mm (12.25 in) tall, 83 mm (3.25 in) deep (ASSUMED from the standard
# section; Ontario OPSD 912 uses the same steel beam).
W_BEAM_HEIGHT_M = 0.31
W_BEAM_DEPTH_M = 0.083
_W_FRONT = np.array(
    [
        [0.083, 0.310],
        [0.0, 0.258],
        [0.0, 0.212],
        [0.083, 0.155],
        [0.0, 0.098],
        [0.0, 0.052],
        [0.083, 0.0],
    ]
)
_W_THICKNESS_M = 0.004
_POST_M = 0.15  # post and blockout section
_BLOCKOUT_M = 0.15  # blockout depth between rail and post

# New Jersey barrier, 32 in tall, 24 in base, 6 in top, 3 in vertical toe, break at 13 in
# (the standard NJ shape, ASSUMED); (d, h) from the back bottom, CCW for the left side.
_NJ = np.array(
    [
        [0.610, 0.0],
        [0.610, 0.075],
        [0.4314, 0.330],
        [0.381, 0.813],
        [0.229, 0.813],
        [0.1786, 0.330],
        [0.0, 0.075],
        [0.0, 0.0],
    ]
)


@dataclass(frozen=True)
class BarrierOptions:
    gap_m: float = 0.0  # face line this far beyond the shoulder edge (model.outer)
    near_edge_m: float = 1.0  # face lines further out than this stand on the ground
    rail_top_m: float = 0.75  # W-beam top above the road
    post_spacing_m: float = 2.0
    terminal_m: float = 4.0  # turned-down W-beam ends (free ends only)
    wall_height_m: float = 1.0
    wall_thickness_m: float = 0.3
    parapet_height_m: float = 0.85
    parapet_width_m: float = 0.35
    collision_height_m: float = 1.2
    embed_m: float = 0.5  # posts, skirts and collision reach this far below the ground
    step_m: float = 1.0  # path resolution along the road
    chunk_m: float = 150.0
    lod_out_m: float = 800.0
    join_m: float = 1.0  # span ends this close to another span/bridge end count as joined
    # OSM rails/jersey barriers mapped within this of the shoulder edge are snapped onto it:
    # they are installed at the road edge and OSM tracing is rarely better than 1-2 m
    # (ASSUMED); walls keep their mapped offset.
    rail_snap_m: float = 2.0


@dataclass
class ScenePiece:
    """One mesh node to add to the KN5: `material` is the material name and `texture` the
    texture stem (`TEXTURES` key, or "wall_hidden"); `flags` go to `to_kn5_mesh`. Names with
    a digit prefix (`1WALL_*`) are physical; the rest are visual only."""

    name: str
    material: str
    mesh: MeshData
    flags: MeshFlags = field(default_factory=dict)
    texture: str = ""

    def __post_init__(self) -> None:
        if not self.texture:
            self.texture = self.material

    @property
    def physical(self) -> bool:
        return self.name[:1].isdigit()


# --------------------------------------------------------------------------------------------
# Textures
# --------------------------------------------------------------------------------------------


def _streaks(noise: np.ndarray, length: int, axis: int) -> np.ndarray:
    """Tileable smear of `noise` along `axis` (wrapping box filter)."""
    acc = np.zeros_like(noise)
    for j in range(length):
        acc += np.roll(noise, j, axis=axis)
    acc /= length
    std = acc.std()
    return acc / std if std > 0 else acc


def concrete_texture(size: int = 256, seed: int = 31) -> Image.Image:
    """Tileable light-grey cast concrete: soft patches, fine grain, faint vertical rain
    streaks and a formwork joint across the middle and the top (v grows downwards)."""
    if size < 8:
        raise ValueError(f"texture size must be at least 8 px, got {size}")
    rng = _rng(seed)
    shape = (size, size)
    patches = _spectral_noise(shape, rng, beta=3.0)
    mid = _spectral_noise(shape, rng, beta=1.6)
    grain = _spectral_noise(shape, rng, beta=0.2)
    streak = _streaks(_spectral_noise(shape, rng, beta=1.2), max(2, size // 6), axis=0)
    lum = 166.0 + 7.0 * patches + 4.0 * mid + 5.0 * grain - 5.0 * np.clip(streak, 0.0, None)
    joint = (np.arange(size) % (size // 2)) == 0
    lum[joint] -= 14.0
    tint = np.array([1.0, 0.985, 0.95], dtype=np.float32)
    return _to_image(lum[..., None] * tint)


def steel_texture(size: int = 128, seed: int = 32) -> Image.Image:
    """Tileable galvanised steel: zinc spangle (cellular tones), soft patches and faint
    streaks along u (along the rail)."""
    if size < 8:
        raise ValueError(f"texture size must be at least 8 px, got {size}")
    rng = _rng(seed)
    cells = max(2, size // 12)
    _, _, cell_id = _worley(size, cells, rng)
    tone = rng.uniform(-1.0, 1.0, cells * cells).astype(np.float32)[cell_id]
    shape = (size, size)
    patches = _spectral_noise(shape, rng, beta=2.5)
    grain = _spectral_noise(shape, rng, beta=0.3)
    streak = _streaks(_spectral_noise(shape, rng, beta=1.0), max(2, size // 4), axis=1)
    lum = 152.0 + 7.0 * tone + 6.0 * patches + 3.0 * grain + 4.0 * streak
    tint = np.array([0.97, 1.0, 1.04], dtype=np.float32)
    return _to_image(lum[..., None] * tint)


TEXTURES: dict[str, Callable[[], Image.Image]] = {
    "concrete": lambda: concrete_texture(256),
    "steel": lambda: steel_texture(128),
}


# --------------------------------------------------------------------------------------------
# Sweeps and caps
# --------------------------------------------------------------------------------------------


def _drop_degenerate(positions: np.ndarray, tris: np.ndarray) -> np.ndarray:
    p = positions
    area = np.linalg.norm(
        np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]]), axis=1
    )
    return tris[area > 1e-9]


def _lattice_mesh(pos: np.ndarray, uv: np.ndarray) -> MeshData | None:
    """A (R, C) lattice of positions/UVs -> MeshData, triangles facing cross(col dir, row dir)."""
    r, c = pos.shape[:2]
    p = pos.reshape(-1, 3)
    tris = _drop_degenerate(p, _lattice_indices(r, c).reshape(-1, 3))
    if len(tris) == 0:
        return None
    used = np.unique(tris)
    data = MeshData(
        p[used],
        np.tile(UP, (len(used), 1)),
        uv.reshape(-1, 2)[used],
        np.tile([1.0, 0.0, 0.0], (len(used), 1)),
        np.searchsorted(used, tris),
    )
    return compute_tangents(compute_normals(data))


def sweep(
    columns: Sequence[np.ndarray],
    s: npt.ArrayLike,
    *,
    smooth: bool = False,
    closed: bool = False,
    u_tile_m: float = 4.0,
    v_tile_m: float = 1.0,
) -> MeshData | None:
    """Surface through K `columns` (each (R, 3), one point per row along the path).

    Columns must be ordered counter-clockwise in the cross-section plane, with the
    horizontal driver's left (of the path direction) as the x axis and up as the y axis: then
    every face points outward (its normal is the edge direction turned clockwise). `smooth`
    shares columns between neighbouring edges (smooth shading across the profile, e.g. the
    W-beam); otherwise each edge is its own strip (crisp edges). `closed` adds the edge from
    the last column back to the first. UVs in metres: u = s / `u_tile_m` along the path, v =
    distance around the profile / `v_tile_m`, which is not mirrored on any outward face.
    """
    cols = [np.asarray(c, np.float64) for c in columns]
    if closed:
        cols = [*cols, cols[0]]
    if len(cols) < 2:
        raise ValueError("need at least two columns")
    s = np.asarray(s, np.float64)
    rows = len(s)
    if any(c.shape != (rows, 3) for c in cols):
        raise ValueError(f"every column must have shape ({rows}, 3)")
    if rows < 2:
        return None
    s_base = math.floor(s[0] / u_tile_m) * u_tile_m
    u = (s - s_base) / u_tile_m
    pos = np.stack(cols, axis=1)  # (R, K, 3)
    edge = np.linalg.norm(np.diff(pos, axis=1), axis=2).mean(axis=0)  # (K - 1,)
    v = np.concatenate([[0.0], np.cumsum(edge)]) / v_tile_m
    if smooth:
        uv = np.stack(np.broadcast_arrays(u[:, None], v[None, :]), axis=-1)
        return _lattice_mesh(pos, uv)
    parts = []
    for k in range(len(cols) - 1):
        uv = np.stack(np.broadcast_arrays(u[:, None], v[None, k : k + 2]), axis=-1)
        m = _lattice_mesh(pos[:, k : k + 2], uv)
        if m is not None:
            parts.append(m)
    return merge(parts) if parts else None


def cap(polygon: np.ndarray, facing: npt.ArrayLike, tile_m: float = 1.0) -> MeshData | None:
    """A flat polygon (K, 3), star-shaped about its vertex centroid (true of every barrier
    profile here, including the concave New Jersey shape), as a fan around that centroid,
    facing `facing` (flat normals), with planar metric UVs not mirrored seen from the front."""
    poly = np.asarray(polygon, np.float64)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(poly, axis=0), axis=1) > 1e-9])
    poly = poly[keep]
    if len(poly) > 3 and np.linalg.norm(poly[0] - poly[-1]) < 1e-9:
        poly = poly[:-1]
    if len(poly) < 3:
        return None
    n = np.asarray(facing, np.float64)
    n = n / np.linalg.norm(n)
    pts = np.concatenate([poly, poly.mean(axis=0, keepdims=True)])
    k = np.arange(len(poly))
    tris = np.stack([np.full_like(k, len(poly)), k, (k + 1) % len(poly)], axis=1)
    tris = _drop_degenerate(pts, tris)
    if len(tris) == 0:
        return None
    p = pts[tris]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    flip = fn @ n < 0.0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    right = np.cross(UP, n)
    if np.linalg.norm(right) < 1e-6:  # horizontal cap: image up = +X
        right = np.cross(np.array([1.0, 0.0, 0.0]), n)
    right /= np.linalg.norm(right)
    img_up = np.cross(n, right)
    rel = pts - pts[0]
    uvs = np.stack([rel @ right, -(rel @ img_up)], axis=1) / tile_m
    data = MeshData(pts, np.tile(n, (len(pts), 1)), uvs, np.tile(right, (len(pts), 1)), tris)
    return compute_tangents(data)


# --------------------------------------------------------------------------------------------
# Paths along the road
# --------------------------------------------------------------------------------------------


@dataclass
class BarrierPath:
    """A barrier's face line: stations `s` (R,), points `base` (R, 3) at the reference height,
    horizontal unit `fwd` (R, 3) and `out` (R, 3) (away from the road), on `side`."""

    side: Side
    s: np.ndarray
    base: np.ndarray
    fwd: np.ndarray
    out: np.ndarray

    def __len__(self) -> int:
        return len(self.s)

    def at(self, d: npt.ArrayLike, h: npt.ArrayLike) -> np.ndarray:
        """Points (R, 3) `d` metres out and `h` metres up from the face line (scalars or (R,))."""
        d = np.broadcast_to(np.asarray(d, np.float64), (len(self),))
        h = np.broadcast_to(np.asarray(h, np.float64), (len(self),))
        return self.base + d[:, None] * self.out + h[:, None] * UP

    def columns(self, profile: npt.ArrayLike) -> list[np.ndarray]:
        """Columns for a (d, h) profile, (K, 2) or per row (R, K, 2), given CCW for the left
        side; reversed on the right side so faces still point outward."""
        prof = np.asarray(profile, np.float64)
        if prof.ndim == 2:
            prof = np.broadcast_to(prof, (len(self), *prof.shape))
        if self.side == "right":
            prof = prof[:, ::-1]
        return [self.at(prof[:, k, 0], prof[:, k, 1]) for k in range(prof.shape[1])]

    def ground(self, ground_fn: GroundFn | None, d: float = 0.0) -> np.ndarray:
        """Ground height (R,) `d` metres out from the face line (reference height if unknown)."""
        pts = self.at(d, 0.0)
        if ground_fn is None:
            return pts[:, 1].copy()
        g = np.asarray(ground_fn(pts[:, [0, 2]]), np.float64).reshape(-1)
        return np.where(np.isfinite(g), g, pts[:, 1])

    def slice(self, i0: int, i1: int) -> BarrierPath:
        """Rows i0..i1 inclusive."""
        r = slice(i0, i1 + 1)
        return BarrierPath(self.side, self.s[r], self.base[r], self.fwd[r], self.out[r])

    def interp(self, s: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(base, fwd, out) at stations `s` (linear; directions re-normalised)."""
        s = np.asarray(s, np.float64)

        def lerp(a: np.ndarray) -> np.ndarray:
            return np.stack([np.interp(s, self.s, a[:, k]) for k in range(3)], axis=-1)

        fwd, out = lerp(self.fwd), lerp(self.out)
        fwd /= np.linalg.norm(fwd, axis=-1, keepdims=True)
        out /= np.linalg.norm(out, axis=-1, keepdims=True)
        return lerp(self.base), fwd, out


def barrier_path(
    model: RoadModel,
    side: Side,
    s0: float,
    s1: float,
    *,
    offset: float | None = None,
    ground_fn: GroundFn | None = None,
    options: BarrierOptions | None = None,
) -> BarrierPath | None:
    """The face line of a barrier on `side` between stations s0..s1 (clipped to the road).

    It stands at the shoulder edge + `gap_m`, or at `offset` metres from the centreline if
    that is further out. Points every `step_m`; fold-over merges (hairpins) are removed. None
    if less than 0.5 m of it is on the road.
    """
    opt = options or BarrierOptions()
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    lo, hi = max(float(s0), float(model.s[0])), min(float(s1), float(model.s[-1]))
    if hi - lo < 0.5:
        return None
    sign = 1.0 if side == "left" else -1.0
    edge = model.outer + opt.gap_m
    off = edge if offset is None else np.maximum(edge, float(offset))
    col = road_columns(model, sign * off)
    n = max(1, math.ceil((hi - lo) / opt.step_m - 1e-9))
    s = np.linspace(lo, hi, n + 1)
    base = np.stack([np.interp(s, model.s, col[:, k]) for k in range(3)], axis=1)
    keep = np.zeros(len(s), bool)
    last = 0
    keep[0] = True
    for i in range(1, len(s)):
        if np.linalg.norm(base[i, [0, 2]] - base[last, [0, 2]]) > 0.05:
            keep[i] = True
            last = i
    if not keep[-1]:  # always end exactly at s1: replace the last kept point
        keep[last] = last == 0
        keep[-1] = True
    s, base = s[keep], base[keep]
    if len(s) < 2:
        return None
    fwd = np.gradient(base, axis=0)
    fwd[:, 1] = 0.0
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    left = np.stack([fwd[:, 2], np.zeros(len(s)), -fwd[:, 0]], axis=1)  # up x forward
    path = BarrierPath(side, s, base, fwd, sign * left)
    if ground_fn is not None:
        beyond = np.interp(s, model.s, off - model.outer) > opt.near_edge_m
        if beyond.any():
            path.base[beyond, 1] = path.ground(ground_fn)[beyond]
    return path


# --------------------------------------------------------------------------------------------
# Planning: which spans get a barrier
# --------------------------------------------------------------------------------------------


def _subtract(a: float, b: float, covered: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pieces = [(a, b)]
    for c0, c1 in covered:
        nxt = []
        for p0, p1 in pieces:
            if c1 <= p0 or c0 >= p1:
                nxt.append((p0, p1))
                continue
            if c0 > p0:
                nxt.append((p0, c0))
            if c1 < p1:
                nxt.append((c1, p1))
        pieces = nxt
    return pieces


def bridge_approach_spans(
    model: RoadModel, bridges: Sequence[tuple[float, float]], approach_m: float = 20.0
) -> list[BarrierSpan]:
    """A guard rail on both sides of each bridge approach: `approach_m` before the deck
    and after it (clipped to the road)."""
    out = []
    lo, hi = float(model.s[0]), float(model.s[-1])
    for b0, b1 in bridges:
        for a, b in ((max(lo, b0 - approach_m), b0), (b1, min(hi, b1 + approach_m))):
            if b - a > 0.5:
                for side in ("left", "right"):
                    out.append(BarrierSpan(side, a, b, "guard_rail", "bridge"))
    return out


def plan_barriers(
    model: RoadModel,
    *,
    osm: Sequence[BarrierSpan] = (),
    user: Sequence[BarrierSpan] = (),
    bridges: Sequence[tuple[float, float]] = (),
    approach_m: float = 20.0,
    heuristic: Sequence[BarrierSpan] = (),
    min_piece_m: float = 2.0,
) -> list[BarrierSpan]:
    """The barriers to build: user spans, then OSM spans, then bridge approaches, then (only
    if given) heuristic spans, each clipped to the road and to what a higher-priority source
    or a bridge deck (station ranges in `bridges`) already covers on that side. With no OSM
    barriers, no user file and no bridges, the result is empty: nothing is invented."""
    lo, hi = float(model.s[0]), float(model.s[-1])
    covered: dict[str, list[tuple[float, float]]] = {
        "left": [(float(a), float(b)) for a, b in bridges],
        "right": [(float(a), float(b)) for a, b in bridges],
    }
    out: list[BarrierSpan] = []
    groups = (user, osm, bridge_approach_spans(model, bridges, approach_m), heuristic)
    for group in groups:
        added: list[BarrierSpan] = []
        for sp in merge_spans(list(group)):
            a, b = max(sp.s0, lo), min(sp.s1, hi)
            if b - a < min_piece_m:
                continue
            for p0, p1 in _subtract(a, b, covered[sp.side]):
                if p1 - p0 >= min_piece_m:
                    added.append(dataclasses.replace(sp, s0=p0, s1=p1))
                    covered[sp.side].append((p0, p1))
        out.extend(added)
    return sorted(out, key=lambda x: (x.side, x.s0, x.s1))


def drop_off_spans(
    model: RoadModel,
    ground_fn: GroundFn,
    *,
    drop_m: float = 3.0,
    probe_m: float = 4.0,
    min_length_m: float = 15.0,
    join_m: float = 10.0,
    bridges: Sequence[tuple[float, float]] = (),
) -> list[BarrierSpan]:
    """OPT-IN heuristic (off by default; the user wants only real barriers): spans where the
    ground `probe_m` beyond the shoulder edge lies more than `drop_m` below the road.
    Bridge decks are skipped. Gaps shorter than `join_m` are bridged; spans shorter than
    `min_length_m` are dropped. Source "heuristic"."""
    on_bridge = np.zeros(len(model), bool)
    for a, b in bridges:
        on_bridge |= (model.s >= a) & (model.s <= b)
    out = []
    for side, sign in (("left", 1.0), ("right", -1.0)):
        pts = road_columns(model, sign * (model.outer + probe_m))
        g = np.asarray(ground_fn(pts[:, [0, 2]]), np.float64)
        steep = np.isfinite(g) & (model.centre[:, 1] - g > drop_m) & ~on_bridge
        idx = np.flatnonzero(steep)
        if len(idx) == 0:
            continue
        breaks = np.flatnonzero(np.diff(model.s[idx]) > join_m)
        for run in np.split(idx, breaks + 1):
            a, b = float(model.s[run[0]]), float(model.s[run[-1]])
            if b - a >= min_length_m:
                out.append(BarrierSpan(side, a, b, "guard_rail", "heuristic"))
    return out


# --------------------------------------------------------------------------------------------
# Barrier builders (one path chunk -> visual meshes per material + collision bounds)
# --------------------------------------------------------------------------------------------


@dataclass
class _Built:
    visual: dict[str, list[MeshData]]
    bottom: np.ndarray  # (R,) collision strip bottom (absolute Y)
    top: np.ndarray  # (R,) collision strip top


def _heading_deg(fwd: np.ndarray) -> float:
    return math.degrees(math.atan2(float(fwd[0]), float(-fwd[2])))


def _end_caps(path: BarrierPath, cols: list[np.ndarray]) -> list[MeshData]:
    out = []
    for row, facing in ((0, -path.fwd[0]), (-1, path.fwd[-1])):
        m = cap(np.stack([c[row] for c in cols]), facing)
        if m is not None:
            out.append(m)
    return out


def _posts(
    path: BarrierPath,
    stations: np.ndarray,
    d_centre: float,
    top_h: np.ndarray,
    ground_fn: GroundFn | None,
    size: tuple[float, float],
    embed_m: float,
    min_above_ground: float = 0.15,
) -> list[MeshData]:
    """Upright boxes (size = (across, along)) centred `d_centre` out from the face line at
    `stations`, from `embed_m` below the ground up to `top_h` above the reference."""
    if len(stations) == 0:
        return []
    base, fwd, out = path.interp(stations)
    centre = base + d_centre * out
    g = centre[:, 1].copy()
    if ground_fn is not None:
        gg = np.asarray(ground_fn(centre[:, [0, 2]]), np.float64).reshape(-1)
        g = np.where(np.isfinite(gg), gg, g)
    meshes = []
    for k in range(len(stations)):
        top = base[k, 1] + top_h[k]
        if top - g[k] < min_above_ground:
            continue
        bottom = min(g[k], base[k, 1]) - embed_m
        c = np.array([centre[k, 0], 0.5 * (top + bottom), centre[k, 2]])
        meshes.append(box(c, (size[0], top - bottom, size[1]), _heading_deg(fwd[k])))
    return meshes


def _taper(s: np.ndarray, s0: float, s1: float, free: tuple[bool, bool], length: float):
    t = np.ones_like(s)
    length = min(length, (s1 - s0) / 3.0)
    if length <= 0:
        return t
    if free[0]:
        t = np.minimum(t, np.clip((s - s0) / length, 0.0, 1.0))
    if free[1]:
        t = np.minimum(t, np.clip((s1 - s) / length, 0.0, 1.0))
    return t


def _w_beam(path, span, ground_fn, opt, free) -> _Built:
    ref = path.base[:, 1]
    taper = _taper(path.s, span.s0, span.s1, free, opt.terminal_m)
    drop = (1.0 - taper) * (opt.rail_top_m + 0.1)
    bottom_h = opt.rail_top_m - W_BEAM_HEIGHT_M - drop  # (R,)
    front = np.stack(
        [
            np.broadcast_to(_W_FRONT[:, 0], (len(path), len(_W_FRONT))),
            bottom_h[:, None] + _W_FRONT[None, :, 1],
        ],
        axis=-1,
    )
    back = front[:, ::-1].copy()
    back[..., 0] += _W_THICKNESS_M
    parts = [
        sweep(path.columns(front), path.s, smooth=True, u_tile_m=4.0, v_tile_m=0.31),
        sweep(path.columns(back), path.s, smooth=True, u_tile_m=4.0, v_tile_m=0.31),
    ]
    n = max(1, round((span.s1 - span.s0) / opt.post_spacing_m))
    posts_s = np.linspace(span.s0, span.s1, n + 1)
    posts_s = posts_s[(posts_s >= path.s[0] - 1e-6) & (posts_s <= path.s[-1] + 1e-6)]
    post_taper = _taper(posts_s, span.s0, span.s1, free, opt.terminal_m)
    post_drop = (1.0 - post_taper) * (opt.rail_top_m + 0.1)
    d_post = W_BEAM_DEPTH_M + _BLOCKOUT_M + _POST_M / 2
    parts += _posts(
        path,
        posts_s,
        d_post,
        opt.rail_top_m - 0.03 - post_drop,
        ground_fn,
        (_POST_M, _POST_M),
        opt.embed_m,
    )
    # Blockouts between rail and post, where the rail is at full height.
    full = post_taper >= 1.0
    if full.any():
        base, fwd, out = path.interp(posts_s[full])
        h0 = opt.rail_top_m - W_BEAM_HEIGHT_M + 0.03
        h1 = opt.rail_top_m - 0.04
        d_c = W_BEAM_DEPTH_M + _BLOCKOUT_M / 2
        for k in range(len(base)):
            c = base[k] + d_c * out[k] + 0.5 * (h0 + h1) * UP
            parts.append(box(c, (_BLOCKOUT_M, h1 - h0, 0.12), _heading_deg(fwd[k])))
    g = path.ground(ground_fn)
    top = ref + np.maximum(0.3, opt.collision_height_m - drop)
    return _Built(
        {"steel": [p for p in parts if p is not None]}, np.minimum(g, ref) - opt.embed_m, top
    )


def _ground_rel(path, ground_fn, d: float) -> np.ndarray:
    return path.ground(ground_fn, d) - path.base[:, 1]


def _jersey(path, span, ground_fn, opt, free) -> _Built:
    gf = _ground_rel(path, ground_fn, 0.0)
    gb = _ground_rel(path, ground_fn, float(_NJ[0, 0]))
    prof = np.broadcast_to(_NJ, (len(path), *_NJ.shape)).copy()
    prof[:, 0, 1] = np.minimum(-0.05, gb - 0.1)
    prof[:, -1, 1] = np.minimum(-0.05, gf - 0.1)
    cols = path.columns(prof)
    parts = [sweep(cols, path.s, u_tile_m=4.0, v_tile_m=1.0), *_end_caps(path, cols)]
    ref = path.base[:, 1]
    bottom = ref + np.minimum(gf, 0.0) - opt.embed_m
    top = ref + max(opt.collision_height_m, float(_NJ[:, 1].max()))
    return _Built(
        {"concrete": [p for p in parts if p is not None]}, bottom, np.full(len(path), top)
    )


def _wall(path, span, ground_fn, opt, free) -> _Built:
    w = opt.wall_thickness_m
    gf = _ground_rel(path, ground_fn, 0.0)
    gb = _ground_rel(path, ground_fn, w)
    height = span.height_m if span.height_m and span.height_m > 0 else opt.wall_height_m
    top = np.maximum(gf, gb) + height
    prof = np.stack(
        [
            np.stack([np.full(len(path), w), gb - opt.embed_m], axis=1),
            np.stack([np.full(len(path), w), top], axis=1),
            np.stack([np.zeros(len(path)), top], axis=1),
            np.stack([np.zeros(len(path)), gf - opt.embed_m], axis=1),
        ],
        axis=1,
    )
    cols = path.columns(prof)
    parts = [sweep(cols, path.s, u_tile_m=4.0, v_tile_m=4.0), *_end_caps(path, cols)]
    ref = path.base[:, 1]
    return _Built(
        {"concrete": [p for p in parts if p is not None]},
        ref + np.minimum(gf, 0.0) - opt.embed_m,
        ref + np.maximum(opt.collision_height_m, top),
    )


def _parapet(path, span, ground_fn, opt, free) -> _Built:
    """Concrete bridge parapet on the deck (h = 0 is the deck top), small chamfer on top."""
    w, h = opt.parapet_width_m, opt.parapet_height_m
    prof = np.array([[w, 0.0], [w, h], [0.05, h], [0.0, h - 0.05], [0.0, 0.0]])
    cols = path.columns(prof)
    parts = [sweep(cols, path.s, u_tile_m=4.0, v_tile_m=4.0), *_end_caps(path, cols)]
    ref = path.base[:, 1]
    return _Built(
        {"concrete": [p for p in parts if p is not None]},
        ref - opt.embed_m,
        ref + max(opt.collision_height_m, h + 0.2),
    )


def _bridge_rail(path, span, ground_fn, opt, free) -> _Built:
    """Steel bridge railing: a low concrete curb, posts every `post_spacing_m`, two rails."""
    w, curb = opt.parapet_width_m, 0.25
    prof = np.array([[w, 0.0], [w, curb], [0.0, curb], [0.0, 0.0]])
    cols = path.columns(prof)
    concrete = [sweep(cols, path.s, u_tile_m=4.0, v_tile_m=4.0), *_end_caps(path, cols)]
    n = max(1, round((span.s1 - span.s0) / opt.post_spacing_m))
    posts_s = np.linspace(span.s0, span.s1, n + 1)
    posts_s = posts_s[(posts_s >= path.s[0] - 1e-6) & (posts_s <= path.s[-1] + 1e-6)]
    rail_top = 1.05
    steel: list[MeshData | None] = []
    if len(posts_s):
        base, fwd, out = path.interp(posts_s)
        for k in range(len(base)):
            c = base[k] + (w / 2) * out[k] + 0.5 * (curb + rail_top - 0.02) * UP
            steel.append(box(c, (0.1, rail_top - 0.02 - curb, 0.1), _heading_deg(fwd[k])))
    d0 = w / 2 - 0.05 - 0.08
    for hc in (0.6, rail_top - 0.04):
        tube = np.array(
            [[d0, hc - 0.04], [d0 + 0.08, hc - 0.04], [d0 + 0.08, hc + 0.04], [d0, hc + 0.04]]
        )
        steel.append(sweep(path.columns(tube), path.s, closed=True, u_tile_m=4.0, v_tile_m=0.32))
    ref = path.base[:, 1]
    return _Built(
        {
            "concrete": [p for p in concrete if p is not None],
            "steel": [p for p in steel if p is not None],
        },
        ref - opt.embed_m,
        ref + max(opt.collision_height_m, rail_top + 0.15),
    )


_BUILDERS = {
    "guard_rail": _w_beam,
    "jersey_barrier": _jersey,
    "wall": _wall,
    "retaining_wall": _wall,
    "parapet": _parapet,
    "bridge_rail": _bridge_rail,
}


def _collision(path: BarrierPath, bottom: np.ndarray, top: np.ndarray) -> MeshData:
    base = path.base.copy()
    base[:, 1] = bottom
    return wall(base, np.maximum(top - bottom, 0.05), double_sided=True, u_tile_m=4.0)


def _chunks(path: BarrierPath, chunk_m: float) -> list[tuple[int, int]]:
    length = path.s[-1] - path.s[0]
    count = max(1, math.ceil(length / chunk_m - 1e-9))
    cuts = np.unique(np.searchsorted(path.s, np.linspace(path.s[0], path.s[-1], count + 1)))
    cuts[0], cuts[-1] = 0, len(path) - 1
    cuts = np.unique(cuts)
    return [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:], strict=True) if b > a]


def span_meshes(
    model: RoadModel,
    span: BarrierSpan,
    ground_fn: GroundFn | None,
    options: BarrierOptions | None = None,
    *,
    free_ends: tuple[bool, bool] = (True, True),
    chunk_m: float | None = None,
) -> list[tuple[dict[str, MeshData], MeshData]]:
    """One barrier span as chunks of (visual meshes by material, collision strip). The
    chunks share their end rows, so the collision strips are continuous over the span."""
    opt = options or BarrierOptions()
    if span.kind not in _BUILDERS:
        raise ValueError(f"unknown barrier kind {span.kind!r}; known: {sorted(_BUILDERS)}")
    offset = span.offset_m
    if offset is not None and span.kind in ("guard_rail", "jersey_barrier"):
        on = (model.s >= span.s0) & (model.s <= span.s1)
        edge = float(np.median(model.outer[on])) if on.any() else float(model.outer.max())
        if offset <= edge + opt.gap_m + opt.rail_snap_m:
            offset = None
    path = barrier_path(
        model, span.side, span.s0, span.s1, offset=offset, ground_fn=ground_fn, options=opt
    )
    if path is None:
        return []
    clipped = dataclasses.replace(span, s0=float(path.s[0]), s1=float(path.s[-1]))
    out = []
    for i0, i1 in _chunks(path, chunk_m or opt.chunk_m):
        sub = path.slice(i0, i1)
        built = _BUILDERS[span.kind](sub, clipped, ground_fn, opt, free_ends)
        visual = {mat: merge(ms) for mat, ms in built.visual.items() if ms}
        out.append((visual, _collision(sub, built.bottom, built.top)))
    return out


def _free_ends(
    span: BarrierSpan, others: Sequence[BarrierSpan], bridges: Sequence[tuple[float, float]], join_m
) -> tuple[bool, bool]:
    ends = [e for a, b in bridges for e in (a, b)]
    ends += [e for o in others if o is not span and o.side == span.side for e in (o.s0, o.s1)]
    ends_arr = np.asarray(ends, np.float64)
    if len(ends_arr) == 0:
        return True, True
    return (
        bool(np.all(np.abs(ends_arr - span.s0) > join_m)),
        bool(np.all(np.abs(ends_arr - span.s1) > join_m)),
    )


def barrier_pieces(
    model: RoadModel,
    spans: Sequence[BarrierSpan],
    ground_fn: GroundFn | None,
    options: BarrierOptions | None = None,
    *,
    bridges: Sequence[tuple[float, float]] = (),
) -> list[ScenePiece]:
    """KN5 mesh pieces for `spans` (from `plan_barriers`): per chunk a visual mesh per
    material (`guardrail_###`, `jersey_###`, `barrier_wall_###`, ...; materials "steel" /
    "concrete", `lod_out`) and a hidden collision strip (`1WALL_GR_###`, ...). A W-beam end
    that does not meet another span or a bridge end (station ranges in `bridges`) is turned
    down into the ground."""
    opt = options or BarrierOptions()
    counters: dict[str, int] = {}
    pieces: list[ScenePiece] = []
    for span in spans:
        stem, tag = BARRIER_STYLES[span.kind]
        free = _free_ends(span, spans, bridges, opt.join_m)
        for visual, coll in span_meshes(model, span, ground_fn, opt, free_ends=free):
            k = counters.get(stem, 0)
            counters[stem] = k + 1
            for mat, mesh in visual.items():
                name = f"{stem}_{k:03d}" if len(visual) == 1 else f"{stem}_{mat}_{k:03d}"
                pieces.append(ScenePiece(name, mat, mesh, {"lod_out": opt.lod_out_m}))
            pieces.append(
                ScenePiece(f"1WALL_{tag}_{k:03d}", "wall_hidden", coll, dict(HIDDEN_WALL_FLAGS))
            )
    return pieces
