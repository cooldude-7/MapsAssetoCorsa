"""Mesh-building primitives: road ribbons, walls, boxes, signs, grids, merge/split.

Every builder returns a `MeshData`: plain numpy arrays with the same meaning as
`driftmaps.ac.model.Mesh` (world-space positions, unit normals, UVs, unit tangents, a flat
triangle-list index buffer), but in float64/int64 and with no vertex limit, so pieces can be
merged and post-processed before `split` + `to_kn5_mesh` turn them into KN5 mesh nodes.

Conventions (docs/research/ac_format.md §4.2, §4.4):
- AC world frame: X = east, Y = up, Z = south (`driftmaps.ac.coords`). Driver's left =
  up x forward.
- Winding: triangles are counter-clockwise seen from the visible side, i.e.
  `cross(p1 - p0, p2 - p0)` points along the visible normal (VERIFIED in game by a third
  party, see §4.2). Ribbons and grids face up, boxes face out, signs face their viewer.
- UVs: DirectX, origin top-left, so u grows to the image's right and v grows DOWN. Every
  builder lays UVs so the image is NOT mirrored seen from the visible side. Equivalently, the
  signed area of each front-facing triangle in (u, v) is negative. Tiling textures rely on
  wrap addressing for UVs outside 0..1 (ASSUMED to be AC's default sampler state for tracks;
  Kunos tracks tile road textures the same way).
- Tangents: the unit direction of increasing u (dP/du), made perpendicular to the normal, the
  same quantity AcTools' `Kn5MeshUtils.RecalculateTangents` computes (VERIFIED (code),
  gro-ove/actools `AcTools.ExtraKn5Utils/Kn5Utils/Kn5MeshUtils.cs`; independent implementation,
  nothing copied). KN5 stores no bitangent sign; the shader rebuilds it, so mirrored UVs would
  light wrongly (ASSUMED), which is one more reason we never mirror.

Road ribbon texture orientation: the image's top edge points FORWARD (u grows to the driver's
right, v falls along the road), so text or arrows painted in a road texture read correctly
for the driver, and the texture is not mirrored seen from above.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict, Unpack

import numpy as np
import numpy.typing as npt

from driftmaps.ac.coords import WORLD_UP
from driftmaps.ac.model import MAX_MESH_VERTICES, Mesh
from driftmaps.geometry.centreline import (
    _as_point,
    _as_points,
    _segments,
    _strip_closing_point,
    cumulative_length,
    frames,
    heading_vector,
)

__all__ = [
    "MeshData",
    "MeshFlags",
    "billboard",
    "box",
    "compute_normals",
    "compute_tangents",
    "face_normals",
    "flat_grid",
    "flipped",
    "grid_with_vertex_count",
    "merge",
    "miter_scale",
    "offset_polyline",
    "post",
    "quad",
    "ribbon",
    "split",
    "to_kn5_mesh",
    "wall",
]


# --------------------------------------------------------------------------------------------
# MeshData
# --------------------------------------------------------------------------------------------


@dataclass
class MeshData:
    """Mesh arrays before they become a KN5 node.

    positions (N, 3), normals (N, 3), uvs (N, 2), tangents (N, 3) as float64; indices (M,)
    int64 with M % 3 == 0, every index in [0, N). Same meaning as `model.Mesh`, but with no
    65,535-vertex limit (use `split`). Arrays are coerced and checked on construction.
    """

    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    tangents: np.ndarray
    indices: np.ndarray

    def __post_init__(self) -> None:
        self.positions = np.ascontiguousarray(self.positions, dtype=np.float64)
        self.normals = np.ascontiguousarray(self.normals, dtype=np.float64)
        self.uvs = np.ascontiguousarray(self.uvs, dtype=np.float64)
        self.tangents = np.ascontiguousarray(self.tangents, dtype=np.float64)
        self.indices = np.ascontiguousarray(self.indices, dtype=np.int64).reshape(-1)
        n = self.positions.shape[0]
        for label, arr, width in (
            ("positions", self.positions, 3),
            ("normals", self.normals, 3),
            ("uvs", self.uvs, 2),
            ("tangents", self.tangents, 3),
        ):
            if arr.shape != (n, width):
                raise ValueError(f"{label} has shape {arr.shape}, expected ({n}, {width})")
        if self.indices.size % 3:
            raise ValueError(f"index count {self.indices.size} is not a multiple of 3")
        if self.indices.size and (self.indices.min() < 0 or self.indices.max() >= n):
            raise ValueError(f"indices out of range for {n} vertices")

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.indices.size // 3)

    @property
    def triangles(self) -> np.ndarray:
        """Indices as (T, 3), a view of `indices`."""
        return self.indices.reshape(-1, 3)

    def copy(self) -> MeshData:
        return MeshData(
            self.positions.copy(),
            self.normals.copy(),
            self.uvs.copy(),
            self.tangents.copy(),
            self.indices.copy(),
        )

    def translated(self, offset: npt.ArrayLike) -> MeshData:
        """A copy moved by `offset` (3,)."""
        return dataclasses.replace(self, positions=self.positions + _as_point(offset))


def _empty_attrs(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Placeholder (normals, tangents) to be filled by compute_normals / compute_tangents."""
    normals = np.tile(WORLD_UP, (n, 1))
    tangents = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    return normals, tangents


def _unit_rows(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _unit(v: npt.ArrayLike, what: str) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if a.shape != (3,) or not n > 1e-12:
        raise ValueError(f"{what} must be a non-zero 3-vector, got {v!r}")
    return a / n


# --------------------------------------------------------------------------------------------
# Normals, tangents
# --------------------------------------------------------------------------------------------


def face_normals(data: MeshData, normalize: bool = True) -> np.ndarray:
    """Geometric normal of each triangle, `cross(p1 - p0, p2 - p0)`, shape (T, 3).

    Unnormalised, its length is twice the triangle's area. Degenerate triangles give zero.
    """
    p = data.positions[data.triangles]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    if normalize:
        length = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.divide(n, length, out=np.zeros_like(n), where=length > 0.0)
    return n


def _accumulate(per_triangle: np.ndarray, tris: np.ndarray, n_vertices: int) -> np.ndarray:
    """Sum a per-triangle (T, 3) quantity onto its three vertices -> (N, 3)."""
    flat_idx = tris.reshape(-1)
    rep = np.repeat(per_triangle, 3, axis=0)
    return np.stack(
        [np.bincount(flat_idx, weights=rep[:, k], minlength=n_vertices) for k in range(3)],
        axis=1,
    )


def _weld(
    acc: np.ndarray, positions: np.ndarray, tolerance: float, max_angle_deg: float
) -> np.ndarray:
    """Sum accumulated normals (N, 3) over coincident vertices whose own normals are within
    `max_angle_deg` of each other (so creases and opposite faces stay apart)."""
    n = positions.shape[0]
    keys = np.round(positions / tolerance).astype(np.int64)
    _, group, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    group = group.reshape(-1)
    if counts.max() == 1:
        return acc
    # Every ordered pair (i, j) of vertices in the same group, self-pairs included.
    order = np.argsort(group, kind="stable")
    group_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    size = counts[group]
    pair_i = np.repeat(np.arange(n), size)
    block_start = np.repeat(np.cumsum(size) - size, size)
    pair_j = order[group_start[group[pair_i]] + np.arange(pair_i.size) - block_start]
    length = np.linalg.norm(acc, axis=1, keepdims=True)
    unit = np.divide(acc, length, out=np.zeros_like(acc), where=length > 0.0)
    cos_limit = math.cos(math.radians(max_angle_deg)) - 1e-12
    keep = np.sum(unit[pair_i] * unit[pair_j], axis=1) >= cos_limit
    weights = acc[pair_j] * keep[:, None]
    return np.stack(
        [np.bincount(pair_i, weights=weights[:, k], minlength=n) for k in range(3)], axis=1
    )


def compute_normals(
    data: MeshData,
    weld_tolerance: float | None = None,
    weld_max_angle_deg: float = 60.0,
) -> MeshData:
    """Smooth, area-weighted vertex normals from the triangles. Returns a new MeshData.

    `weld_tolerance` (metres): also average across vertices at the same position (rounded to
    this grid), e.g. the duplicated seam of a closed ribbon or the joint between chunks, so
    shading is continuous there. Only vertices whose own normals are within
    `weld_max_angle_deg` of each other are averaged, so hard creases (box edges) and the two
    faces of a double-sided wall stay separate. Unreferenced vertices get world up. Call
    `compute_tangents` afterwards.
    """
    acc = _accumulate(face_normals(data, normalize=False), data.triangles, data.vertex_count)
    if weld_tolerance is not None and data.vertex_count:
        if not weld_tolerance > 0.0:
            raise ValueError(f"weld_tolerance must be > 0, got {weld_tolerance}")
        acc = _weld(acc, data.positions, weld_tolerance, weld_max_angle_deg)
    length = np.linalg.norm(acc, axis=1, keepdims=True)
    normals = np.where(length > 1e-20, acc / np.where(length > 1e-20, length, 1.0), WORLD_UP)
    return dataclasses.replace(data, normals=normals)


def _any_perpendicular(n: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to each row of `n` (N, 3)."""
    axis = np.where(np.abs(n[:, 0:1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]])
    t = axis - n * np.sum(axis * n, axis=1, keepdims=True)
    return _unit_rows(t)


def compute_tangents(data: MeshData) -> MeshData:
    """Per-vertex tangents = unit dP/du, orthogonalised against the normals. Returns a new
    MeshData.

    Per triangle: sdir = (e1 * dv2 - e2 * dv1) / (du1 * dv2 - du2 * dv1) (Lengyel's method),
    summed onto its vertices, then Gram-Schmidt against the vertex normal. This is the same
    method as AcTools `Kn5MeshUtils.RecalculateTangents` (cited in the module docstring), plus
    a guard for triangles with degenerate UVs (skipped) and a perpendicular fallback for
    vertices with no usable triangle.
    """
    tris = data.triangles
    p = data.positions[tris]
    t = data.uvs[tris]
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    du1 = t[:, 1, 0] - t[:, 0, 0]
    dv1 = t[:, 1, 1] - t[:, 0, 1]
    du2 = t[:, 2, 0] - t[:, 0, 0]
    dv2 = t[:, 2, 1] - t[:, 0, 1]
    det = du1 * dv2 - du2 * dv1
    good = np.abs(det) > 1e-12
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=good)
    sdir = (e1 * dv2[:, None] - e2 * dv1[:, None]) * inv[:, None]
    acc = _accumulate(sdir, tris, data.vertex_count)

    n = data.normals
    acc -= n * np.sum(acc * n, axis=1, keepdims=True)
    length = np.linalg.norm(acc, axis=1, keepdims=True)
    ok = length[:, 0] > 1e-9
    tangents = _any_perpendicular(n) if not ok.all() else np.empty_like(acc)
    tangents[ok] = acc[ok] / length[ok]
    return dataclasses.replace(data, tangents=tangents)


def _finish(
    positions: np.ndarray,
    uvs: np.ndarray,
    indices: np.ndarray,
    normals: np.ndarray | None = None,
    weld_tolerance: float | None = None,
) -> MeshData:
    """Build a MeshData; compute normals unless given, then tangents."""
    n0, t0 = _empty_attrs(positions.shape[0])
    data = MeshData(positions, n0 if normals is None else normals, uvs, t0, indices)
    if normals is None:
        data = compute_normals(data, weld_tolerance)
    return compute_tangents(data)


# --------------------------------------------------------------------------------------------
# Lattices (structured grids of vertices)
# --------------------------------------------------------------------------------------------


def _lattice_indices(rows: int, cols: int, flip: bool = False) -> np.ndarray:
    """Triangles for a rows x cols vertex lattice, vertex (r, c) at index r * cols + c.

    Each cell (a = (r, c), b = (r, c+1), c' = (r+1, c+1), d = (r+1, c)) becomes (a, b, c') and
    (a, c', d), whose normal is along cross(column direction, row direction); `flip` reverses
    both triangles.
    """
    r, c = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    a = (r * cols + c).reshape(-1)
    b = a + 1
    cc = a + cols + 1
    d = a + cols
    if flip:
        tris = np.stack([a, cc, b, a, d, cc], axis=1)
    else:
        tris = np.stack([a, b, cc, a, cc, d], axis=1)
    return tris.reshape(-1)


def _per_station(value: npt.ArrayLike, n: int, what: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{what} must be a scalar or shape ({n},), got {arr.shape}")
    return arr.copy()


def _closed_tile(length: float, tile: float) -> float:
    """Adjust a tile length so a closed loop of `length` holds a whole number of tiles."""
    return length / max(1, round(length / tile))


# --------------------------------------------------------------------------------------------
# Offsets, ribbons, walls
# --------------------------------------------------------------------------------------------


def miter_scale(
    points: npt.ArrayLike, closed: bool = False, miter_limit: float = 2.0
) -> np.ndarray:
    """Per-point factor (N,) >= 1 that keeps a lateral offset parallel to both adjacent
    segments at a polyline corner: 1 / cos(half the turn), capped at `miter_limit`.

    On a densely sampled curve it is ~1; at a hard 90 degree corner it is sqrt(2). Open ends
    get 1. `miter_limit` = 1 disables mitring.
    """
    pts = _as_points(points, min_points=2)
    if closed:
        pts = _strip_closing_point(pts)
    if not miter_limit >= 1.0:
        raise ValueError(f"miter_limit must be >= 1, got {miter_limit}")
    _, lefts, _ = frames(pts, closed)
    seg = _segments(pts, closed)
    seg_left = np.cross(WORLD_UP, seg)
    seg_left = _unit_rows(seg_left)
    if closed:
        cos_in = np.sum(lefts * np.roll(seg_left, 1, axis=0), axis=1)
        cos_out = np.sum(lefts * seg_left, axis=1)
    else:
        cos_in = np.concatenate([[1.0], np.sum(lefts[1:] * seg_left, axis=1)])
        cos_out = np.concatenate([np.sum(lefts[:-1] * seg_left, axis=1), [1.0]])
    cos = np.minimum(cos_in, cos_out)
    return 1.0 / np.maximum(cos, 1.0 / miter_limit)


def offset_polyline(
    centre: npt.ArrayLike,
    offset: npt.ArrayLike,
    closed: bool = False,
    miter_limit: float = 2.0,
) -> np.ndarray:
    """Shift a polyline sideways, horizontally: `offset` > 0 moves it to the driver's LEFT.

    `offset` is a scalar or per point (N,). Corners are mitred (see `miter_scale`). Heights
    are unchanged. Offsets larger than the inner radius of a curve fold over; the caller
    (road.py) must limit curvature or collapse folded points (geo.md §3.5).
    """
    pts = _as_points(centre, min_points=2)
    if closed:
        pts = _strip_closing_point(pts)
    off = _per_station(offset, pts.shape[0], "offset")
    _, lefts, _ = frames(pts, closed)
    scale = miter_scale(pts, closed, miter_limit)
    return pts + (off * scale)[:, None] * lefts


def ribbon(
    centre: npt.ArrayLike,
    left_width: npt.ArrayLike,
    right_width: npt.ArrayLike,
    lateral_segments: int = 4,
    u_tile_m: float | None = 4.0,
    v_tile_m: float = 8.0,
    closed: bool = False,
    *,
    s_start: float = 0.0,
    lefts: npt.ArrayLike | None = None,
    normals: Literal["geometry", "world_up"] = "geometry",
    miter_limit: float = 2.0,
) -> MeshData:
    """A road strip along `centre` (N, 3), facing up, from `left_width` metres left of the
    centreline to `right_width` metres right of it.

    - Widths are scalars or per station (N,). Negative values are allowed so a strip can sit
      wholly on one side (a left shoulder from 3.5 m to 5 m: left_width=5, right_width=-3.5),
      but the total width must stay > 0.
    - `lateral_segments` quads across; vertices are evenly spaced across the width.
    - Lateral direction: horizontal driver's left from `centreline.frames`, mitred at corners.
      Pass `lefts` (N, 3) to override it, e.g. with banked/cambered directions.
    - UVs in metres: u = distance from the LEFT edge / `u_tile_m` (None: u spans 0..1 across
      the strip); v = -(s_start + s) / `v_tile_m`, where s is the distance along the
      centreline. The image's top edge points forward (see module docstring). Use `s_start`
      to keep v continuous across chunks of one road; keep it small (e.g. modulo the tile)
      for float32 precision.
    - `closed`: the centreline is a loop (no repeated end point needed). The first station is
      repeated at the end with identical positions (no seam gap), v_tile_m is nudged so the
      loop holds a whole number of tiles (no texture seam), and normals are welded across the
      seam.
    - `normals`: "geometry" (area-weighted from the triangles) or "world_up" (all +Y).
    """
    pts = _as_points(centre, min_points=2)
    if closed:
        pts = _strip_closing_point(pts)
    n = pts.shape[0]
    if lateral_segments < 1:
        raise ValueError(f"lateral_segments must be >= 1, got {lateral_segments}")
    if not v_tile_m > 0.0 or (u_tile_m is not None and not u_tile_m > 0.0):
        raise ValueError("u_tile_m and v_tile_m must be > 0")
    lw = _per_station(left_width, n, "left_width")
    rw = _per_station(right_width, n, "right_width")
    width = lw + rw
    if (width <= 0.0).any():
        raise ValueError("left_width + right_width must be > 0 at every station")

    _, frame_lefts, _ = frames(pts, closed)
    if lefts is None:
        lat = frame_lefts
    else:
        lat = np.asarray(lefts, dtype=np.float64)
        if lat.shape != (n, 3):
            raise ValueError(f"lefts must have shape ({n}, 3), got {lat.shape}")
        lat = _unit_rows(lat)
    lat = lat * miter_scale(pts, closed, miter_limit)[:, None]

    k = lateral_segments + 1
    frac = np.linspace(0.0, 1.0, k)  # 0 = left edge, 1 = right edge
    across = frac[None, :] * width[:, None]  # metres from the left edge, (n, k)
    offsets = lw[:, None] - across  # signed lateral offset, + = left
    positions = pts[:, None, :] + offsets[..., None] * lat[:, None, :]
    s = cumulative_length(pts)

    v_tile = v_tile_m
    if closed:
        loop_length = float(s[-1] + np.linalg.norm(pts[0] - pts[-1]))
        positions = np.concatenate([positions, positions[:1]], axis=0)
        across = np.concatenate([across, across[:1]], axis=0)
        s = np.append(s, loop_length)
        v_tile = _closed_tile(loop_length, v_tile_m)
    rows = positions.shape[0]

    u = across / u_tile_m if u_tile_m is not None else np.broadcast_to(frac, across.shape)
    v = np.broadcast_to((0.0 - (s_start + s) / v_tile)[:, None], (rows, k))  # no -0.0
    uvs = np.stack([u, v], axis=-1).reshape(-1, 2)
    positions = positions.reshape(-1, 3)
    # Rows run forward, columns run left -> right: cross(right, forward) = up.
    indices = _lattice_indices(rows, k)

    if normals == "world_up":
        return _finish(positions, uvs, indices, normals=np.tile(WORLD_UP, (rows * k, 1)))
    if normals != "geometry":
        raise ValueError(f"normals must be 'geometry' or 'world_up', got {normals!r}")
    return _finish(positions, uvs, indices, weld_tolerance=1e-6 if closed else None)


def wall(
    polyline: npt.ArrayLike,
    height: npt.ArrayLike,
    double_sided: bool = True,
    closed: bool = False,
    facing: Literal["left", "right"] = "left",
    u_tile_m: float = 4.0,
    v_tile_m: float | None = None,
) -> MeshData:
    """A vertical strip standing on `polyline` (N, 3), `height` metres tall (scalar or (N,)).

    The front faces the polyline's `facing` side (as seen travelling along it). With
    `double_sided` the back is added as separate vertices with opposite normals and winding,
    so it is visible (and, ASSUMED, collidable) from both sides (ac_format.md §3.2).
    UVs: u = distance along the base / `u_tile_m`, running so the image is not mirrored on
    either face; v = 0 at the top, growing downwards, (top - y) / `v_tile_m` (None: 0..1 over
    the height). `closed` treats the polyline as a loop (seam repeated, whole tiles along u).
    """
    pts = _as_points(polyline, min_points=2)
    if closed:
        pts = _strip_closing_point(pts)
    n = pts.shape[0]
    h = _per_station(height, n, "height")
    if (h <= 0.0).any():
        raise ValueError("height must be > 0")
    if facing not in ("left", "right"):
        raise ValueError(f"facing must be 'left' or 'right', got {facing!r}")
    if not u_tile_m > 0.0 or (v_tile_m is not None and not v_tile_m > 0.0):
        raise ValueError("u_tile_m and v_tile_m must be > 0")
    frames(pts, closed)  # validates: no repeated points, no vertical segments

    s = cumulative_length(pts)
    u_tile = u_tile_m
    if closed:
        loop_length = float(s[-1] + np.linalg.norm(pts[0] - pts[-1]))
        pts = np.concatenate([pts, pts[:1]], axis=0)
        h = np.append(h, h[0])
        s = np.append(s, loop_length)
        u_tile = _closed_tile(loop_length, u_tile_m)
    rows = pts.shape[0]
    top = pts + h[:, None] * WORLD_UP
    positions = np.stack([top, pts], axis=1).reshape(-1, 3)  # columns: top, bottom
    v_col = np.array([0.0, 1.0])
    v = h[:, None] * v_col[None, :] / v_tile_m if v_tile_m is not None else v_col[None, :]
    v = np.broadcast_to(v, (rows, 2))

    def side(face: str) -> MeshData:
        # Columns run top -> bottom and rows forward: cross(down, forward) = -left, so a
        # left-facing side flips. Seen from the left, forward runs to the viewer's LEFT, so
        # u must fall along the polyline there (and rise on the right) to avoid mirroring.
        sign = -1.0 if face == "left" else 1.0
        u = np.broadcast_to((0.0 + sign * s / u_tile)[:, None], (rows, 2))  # no -0.0
        uvs = np.stack([u, v], axis=-1).reshape(-1, 2)
        indices = _lattice_indices(rows, 2, flip=face == "left")
        return _finish(positions, uvs, indices, weld_tolerance=1e-6 if closed else None)

    front = side(facing)
    if not double_sided:
        return front
    return merge([front, side("right" if facing == "left" else "left")])


# --------------------------------------------------------------------------------------------
# Quads, signs, boxes
# --------------------------------------------------------------------------------------------


def quad(
    centre: npt.ArrayLike,
    right: npt.ArrayLike,
    up: npt.ArrayLike,
    width: float,
    height: float,
) -> MeshData:
    """A `width` x `height` rectangle with the full texture (UV 0..1) on it.

    `right` and `up` are the image's right and up directions as seen by the viewer (`up` is
    re-orthogonalised against `right`); the visible side faces `cross(right, up)`. 4 vertices,
    2 triangles, flat normals, tangent = `right`.
    """
    c = _as_point(centre)
    r = _unit(right, "right")
    a = np.asarray(up, dtype=np.float64) - r * float(np.dot(r, up))
    a = _unit(a, "up (perpendicular to right)")
    if not (width > 0.0 and height > 0.0):
        raise ValueError(f"width and height must be > 0, got {width}, {height}")
    hw, hh = width / 2.0, height / 2.0
    positions = np.array(
        [c - r * hw - a * hh, c + r * hw - a * hh, c + r * hw + a * hh, c - r * hw + a * hh]
    )
    uvs = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    normal = np.cross(r, a)
    return MeshData(
        positions,
        np.tile(normal, (4, 1)),
        uvs,
        np.tile(r, (4, 1)),
        np.array([0, 1, 2, 0, 2, 3]),
    )


def billboard(
    centre: npt.ArrayLike,
    facing: npt.ArrayLike | float,
    width: float,
    height: float,
    up: npt.ArrayLike = WORLD_UP,
    double_sided: bool = False,
) -> MeshData:
    """An upright sign: a quad centred on `centre` whose front faces `facing`.

    `facing` is the direction the front looks towards (its normal): a 3-vector, or a compass
    heading in degrees (0 = the front faces north, readable by someone standing north of it).
    The full texture (UV 0..1) reads correctly, not mirrored, from the front. With
    `double_sided`, a back quad is added that also reads correctly from behind.
    """
    if np.ndim(facing) == 0:
        n = heading_vector(float(np.asarray(facing)))
    else:
        n = _unit(facing, "facing")
    up_v = _unit(up, "up")
    # The viewer looks along -n; their right is (-n) x up = up x n.
    r = np.cross(up_v, n)
    r = _unit(r, "facing (must not be parallel to up)")
    a = np.cross(n, r)
    front = quad(centre, r, a, width, height)
    if not double_sided:
        return front
    return merge([front, quad(centre, -r, a, width, height)])


def box(
    centre: npt.ArrayLike,
    size: npt.ArrayLike,
    heading_deg: float = 0.0,
) -> MeshData:
    """A closed box with outward faces: 6 faces x 4 vertices (flat normals), 12 triangles.

    `size` = (across, height, along): `along` lies on the compass heading `heading_deg`,
    `across` on the horizontal perpendicular, so at heading 0 the size is simply (X, Y, Z).
    Each face carries the full texture (UV 0..1), upright on the sides, not mirrored from
    outside; on top the image's top edge points along the heading.
    """
    c = _as_point(centre)
    sx, sy, sz = (float(x) for x in np.asarray(size, dtype=np.float64).reshape(3))
    if not (sx > 0.0 and sy > 0.0 and sz > 0.0):
        raise ValueError(f"size must be positive, got {size!r}")
    fwd = heading_vector(heading_deg)
    up = WORLD_UP.astype(np.float64)
    right = np.cross(fwd, up)  # the box's "across" axis; +X at heading 0
    faces: list[MeshData] = []
    # (outward normal, image up, half-extent along the normal, face width, face height)
    for normal, img_up, half, w, h in (
        (right, up, sx / 2, sz, sy),
        (-right, up, sx / 2, sz, sy),
        (fwd, up, sz / 2, sx, sy),
        (-fwd, up, sz / 2, sx, sy),
        (up, fwd, sy / 2, sx, sz),
        (-up, fwd, sy / 2, sx, sz),
    ):
        # Image right seen from outside: img_up x normal (so cross(right, img_up) = normal).
        img_right = np.cross(img_up, normal)
        faces.append(quad(c + normal * half, img_right, img_up, w, h))
    return merge(faces)


def post(
    base: npt.ArrayLike,
    height: float,
    width: float = 0.2,
    heading_deg: float = 0.0,
) -> MeshData:
    """A square post standing on `base` (the centre of its bottom face): a `box` of
    `width` x `height` x `width`."""
    b = _as_point(base)
    return box(b + np.array([0.0, height / 2.0, 0.0]), (width, height, width), heading_deg)


# --------------------------------------------------------------------------------------------
# Flat grids
# --------------------------------------------------------------------------------------------


def _xz_grid(
    xs: np.ndarray,
    zs: np.ndarray,
    y: float,
    u_tile_m: float,
    v_tile_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices of a grid over xs (columns) x zs (rows), row-major, with planar UVs."""
    gx, gz = np.meshgrid(xs, zs, indexing="xy")
    positions = np.stack([gx, np.full_like(gx, y), gz], axis=-1).reshape(-1, 3)
    # Seen from above with north up the screen, east is right (u) and south is down (v).
    uvs = np.stack([positions[:, 0] / u_tile_m, positions[:, 2] / v_tile_m], axis=1)
    return positions, uvs


def _check_range(rng: tuple[float, float], what: str) -> tuple[float, float]:
    lo, hi = float(rng[0]), float(rng[1])
    if not hi > lo:
        raise ValueError(f"{what} must be (min, max) with max > min, got {rng!r}")
    return lo, hi


def flat_grid(
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    step: float,
    y: float = 0.0,
    u_tile_m: float = 4.0,
    v_tile_m: float = 4.0,
) -> MeshData:
    """A flat, up-facing grid over [x0, x1] x [z0, z1] at height `y`.

    Cell count per axis is `ceil(extent / step)` (so cells are at most `step`); the grid
    covers the ranges exactly. UVs are planar in metres (u = x / u_tile_m, v = z / v_tile_m),
    not mirrored seen from above.
    """
    x0, x1 = _check_range(x_range, "x_range")
    z0, z1 = _check_range(z_range, "z_range")
    if not step > 0.0:
        raise ValueError(f"step must be > 0, got {step}")
    nx = max(1, math.ceil((x1 - x0) / step - 1e-9)) + 1
    nz = max(1, math.ceil((z1 - z0) / step - 1e-9)) + 1
    positions, uvs = _xz_grid(
        np.linspace(x0, x1, nx), np.linspace(z0, z1, nz), y, u_tile_m, v_tile_m
    )
    # Rows run south (+Z), columns east (+X): cross(X, Z) = -Y, so flip to face up.
    indices = _lattice_indices(nz, nx, flip=True)
    n = positions.shape[0]
    return _finish(positions, uvs, indices, normals=np.tile(WORLD_UP, (n, 1)))


def _grid_shape(n_vertices: int, aspect: float) -> tuple[int, int, int]:
    """(columns, full rows, vertices in a partial last row) for exactly n_vertices.

    Prefers an exact factorisation cols x rows whose cell aspect is within 2x of the requested
    one (65,535 = 255 x 257); otherwise uses full rows plus one partial row.
    """
    best: tuple[float, int] | None = None
    for d in range(2, math.isqrt(n_vertices) + 1):
        if n_vertices % d:
            continue
        for cols in (d, n_vertices // d):
            rows = n_vertices // cols
            err = abs(math.log(((cols - 1) / (rows - 1)) / aspect))
            if best is None or err < best[0]:
                best = (err, cols)
    if best is not None and best[0] <= math.log(2.0):
        cols = best[1]
        return cols, n_vertices // cols, 0
    # At least two full rows (n >= 4), so every vertex of the last full row is used.
    cols = max(2, min(max(2, n_vertices // 2), round(math.sqrt(n_vertices * aspect))))
    return cols, n_vertices // cols, n_vertices % cols


def grid_with_vertex_count(
    n_vertices: int,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    y: float = 0.0,
    u_tile_m: float = 4.0,
    v_tile_m: float = 4.0,
) -> MeshData:
    """A flat, up-facing grid with EXACTLY `n_vertices` vertices, all used by triangles.

    Made for the 65,535-vertex spike (S9): `grid_with_vertex_count(65535, ...)` is a
    255 x 257 grid. Counts without a usable factorisation get a partial last row, so any
    n >= 3 works. The grid spans x_range and z_range (the partial row sits on the max-Z edge).
    """
    if n_vertices < 3:
        raise ValueError(f"need at least 3 vertices for a triangle, got {n_vertices}")
    x0, x1 = _check_range(x_range, "x_range")
    z0, z1 = _check_range(z_range, "z_range")
    cols, full_rows, rem = _grid_shape(n_vertices, (x1 - x0) / (z1 - z0))
    total_rows = full_rows + (1 if rem else 0)
    xs = np.linspace(x0, x1, cols)
    zs = np.linspace(z0, z1, total_rows)
    positions, uvs = _xz_grid(xs, zs, y, u_tile_m, v_tile_m)
    positions, uvs = positions[:n_vertices], uvs[:n_vertices]
    parts = [_lattice_indices(full_rows, cols, flip=True)]
    if rem:
        # Join the partial row (rem vertices) to the last full row, facing up like the rest.
        # Same cell layout as the flipped lattice: (a, c, b), (a, d, c); plus one triangle
        # (a, d, b) to reach the first unpaired vertex of the full row.
        prev = (full_rows - 1) * cols + np.arange(cols)
        new = full_rows * cols + np.arange(rem)
        i = np.arange(rem - 1)
        cells = np.stack([prev[i], new[i + 1], prev[i + 1], prev[i], new[i], new[i + 1]], axis=1)
        parts.append(cells.reshape(-1))
        if rem < cols:
            parts.append(np.array([prev[rem - 1], new[rem - 1], prev[rem]]))
    indices = np.concatenate(parts)
    return _finish(positions, uvs, indices, normals=np.tile(WORLD_UP, (n_vertices, 1)))


# --------------------------------------------------------------------------------------------
# Merge, split, flip, convert
# --------------------------------------------------------------------------------------------


def merge(meshes: Sequence[MeshData]) -> MeshData:
    """Concatenate meshes into one (vertices appended in order, indices re-based)."""
    if not meshes:
        raise ValueError("merge needs at least one mesh")
    offsets = np.cumsum([0] + [m.vertex_count for m in meshes[:-1]])
    return MeshData(
        np.concatenate([m.positions for m in meshes]),
        np.concatenate([m.normals for m in meshes]),
        np.concatenate([m.uvs for m in meshes]),
        np.concatenate([m.tangents for m in meshes]),
        np.concatenate([m.indices + off for m, off in zip(meshes, offsets, strict=True)]),
    )


def flipped(data: MeshData) -> MeshData:
    """The same surface seen from the other side: winding reversed, normals negated.

    UVs are kept, so the texture appears mirrored from the new visible side; build a proper
    back face (as `wall` and `billboard` do) when that matters.
    """
    tris = data.triangles[:, [0, 2, 1]]
    return dataclasses.replace(data, normals=-data.normals, indices=tris.reshape(-1))


def _subset(data: MeshData, tris: np.ndarray) -> MeshData:
    """A compact mesh holding only `tris` (T, 3) and the vertices they use (in order)."""
    used = np.unique(tris)
    return MeshData(
        data.positions[used],
        data.normals[used],
        data.uvs[used],
        data.tangents[used],
        np.searchsorted(used, tris).reshape(-1),
    )


def split(data: MeshData, max_vertices: int = MAX_MESH_VERTICES) -> list[MeshData]:
    """Cut a mesh into pieces of at most `max_vertices` vertices, keeping whole triangles.

    Triangles are taken greedily in index-buffer order, so builders that emit triangles in
    order along the road give spatially compact chunks. Vertex attributes are copied as they
    are, so compute normals BEFORE splitting to keep shading smooth across the cuts. A mesh
    that already fits is returned as is (one element). Every piece has >= 1 triangle.
    """
    if max_vertices < 3:
        raise ValueError(f"max_vertices must be >= 3, got {max_vertices}")
    if data.vertex_count <= max_vertices:
        return [data]
    tris = data.triangles
    total = tris.shape[0]
    pieces: list[MeshData] = []
    t0 = 0
    while t0 < total:
        window = max_vertices  # triangles to look at; doubled while they all fit
        while True:
            t1 = min(total, t0 + window)
            flat = tris[t0:t1].reshape(-1)
            _, first = np.unique(flat, return_index=True)
            is_new = np.zeros(flat.size, dtype=np.int64)
            is_new[first] = 1
            used_so_far = np.cumsum(is_new.reshape(-1, 3).sum(axis=1))
            fit = int(np.searchsorted(used_so_far, max_vertices, side="right"))
            if fit < t1 - t0 or t1 == total:
                break
            window *= 2
        pieces.append(_subset(data, tris[t0 : t0 + fit]))
        t0 += fit
    return pieces


class MeshFlags(TypedDict, total=False):
    """Optional `model.Mesh` fields accepted by `to_kn5_mesh`."""

    cast_shadows: bool
    is_visible: bool
    is_transparent: bool
    is_renderable: bool
    layer: int
    lod_in: float
    lod_out: float
    active: bool
    bounding_sphere: tuple[tuple[float, float, float], float] | None


def to_kn5_mesh(data: MeshData, name: str, material_id: int, **flags: Unpack[MeshFlags]) -> Mesh:
    """Convert to a KN5 mesh node: float32 arrays, uint16 indices.

    Raises ValueError if the mesh has more than 65,535 vertices (use `split` first), no
    triangles, or non-finite values. Example of an invisible collision wall
    (ac_format.md §3.2): `to_kn5_mesh(w, "1WALL_L_c000", mat, is_renderable=False,
    cast_shadows=False)`.
    """
    if data.vertex_count > MAX_MESH_VERTICES:
        raise ValueError(
            f"{name}: {data.vertex_count:,} vertices exceeds the KN5 limit of "
            f"{MAX_MESH_VERTICES:,}; split() it first"
        )
    if data.triangle_count == 0:
        raise ValueError(f"{name}: mesh has no triangles")
    arrays = (data.positions, data.normals, data.uvs, data.tangents)
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError(f"{name}: mesh contains NaN or infinite values")
    return Mesh(
        name=name,
        positions=data.positions.astype(np.float32),
        normals=data.normals.astype(np.float32),
        uvs=data.uvs.astype(np.float32),
        tangents=data.tangents.astype(np.float32),
        indices=data.indices.astype(np.uint16),
        material_id=material_id,
        **flags,
    )
