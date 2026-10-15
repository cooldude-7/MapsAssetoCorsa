"""Assetto Corsa AI line files (`ai/fast_lane.ai`, also `pit_lane.ai` and `data/ideal_line.ai`).

Reader, writer, a unit-calibration dump (`drift-maps ai-dump`) and a simple generator that turns a
centreline into a complete AI line. See `docs/research/ai_line.md` §1, §2, §4 and §5.

File layout (format v7, little-endian; `f` = float32, `i` = int32). VERIFIED against
AcTools `AcTools/AiFile/AiSpline.cs`, `AiPoint.cs`, `AiPointExtra.cs`, `AiSplineGrid.cs`
(https://github.com/gro-ove/actools @ e5e5c94, MS-PL) and AssettoServer
`FastLaneUtils/FastLane.cs` + `AssettoServer/Server/Ai/Splines/FastLaneParser.cs`
(https://github.com/compujuckel/AssettoServer @ 1365f3e; AGPL, read for the layout only, no code
taken). Both readers agree field for field:

    i version (7)  i N  i lap_time  i sample_count          AcTools: "usually zeroes"
    N x { f x, f y, f z, f length, i id }                    20 bytes; length = cumulative metres
    i extra_count (must equal N)                             AssettoServer throws otherwise
    N x 18 f payload:                                        72 bytes, order below
        speed gas brake obsolete_lat_g radius side_left side_right camber direction
        normal.xyz length forward.xyz tag grade
    i has_grid (0 or not)                                    AcTools reads it; FastLaneUtils
                                                             writes no flag at all
    grid (only if has_grid != 0), AcTools AiSplineGrid:
        f max_extreme.xyz  f min_extreme.xyz  i neighbors_considered  f sampling_density
        i item_count, each item: i sub_count, each sub: i count, count x i point index

Payload semantics. Field NAMES are VERIFIED (AcTools). The meanings below are OBSERVED: they were
fitted to one real game-baked file (the `sepang` mod track's `fast_lane.ai` in
github.com/619motorsports/newksEditor, downloaded by the tests, never committed). They are not
yet checked on Kunos files; spike S0 (`drift-maps ai-dump` on Kunos tracks) settles them. Every
generator choice that depends on them is marked ASSUMED below.

- speed: m/s. OBSERVED: |obsolete_lat_g| = speed^2 / (radius * g) to within 0.1% with speed in
  m/s (13x off with km/h).
- gas, brake: 0..1, never both non-zero at a point. OBSERVED.
- obsolete_lat_g: lateral acceleration in g, signed like `direction`. OBSERVED.
- radius: horizontal turn radius in metres, unsigned, huge (up to ~2e6) on straights. OBSERVED:
  |P[i+5] - P[i]|_xz / (2 sin(theta / 2)), theta = turn angle between the chords
  P[i-5] -> P[i] and P[i] -> P[i+5] (median error 0.03%).
- side_left, side_right: metres from the line to the left/right usable edge. VERIFIED (CM's
  AiLaneObject draws `p - s * SideLeft` and `p + s * SideRight` with `s = cross(forward, Y)`,
  the driver's right).
- direction: +1 where the line turns left, -1 where it turns right. OBSERVED (100% of corners).
- camber: direction * dot(normal, left) = sine of the bank angle, positive when the road is
  banked into the turn. OBSERVED (ratio 1.001). AssettoServer uses camber * direction as roll.
- normal: road surface normal under the point. OBSERVED (|n| = 1, n.y ~ 0.9995).
- length: horizontal distance from point i-5 to point i. OBSERVED (median error 1e-7 m). Not the
  segment length the research notes guessed.
- forward: unit tangent (ASSUMED: all zero in the observed file; FastLaneUtils' Reverse() flips it).
- tag: 0 in every file seen.
- grade: (y[i] - y[i-5]) / length, a rise-over-run ratio. OBSERVED (median error 2e-10).

Grid semantics. OBSERVED on the same file: cell (ix, iz) lists the `neighbors_considered` (10)
spline points nearest (XZ distance) to the cell centre `min_extreme + (i + 0.5) * density`,
nearest first. Extremes are the points' XZ bounds padded by 350 m with y = 0. Cell counts are
int(extent / density) along X (items) and Z (subs). CSP's default ALT_GRID=1 replaces this grid,
so we write `has_grid = 0` by default (spike S3 checks vanilla AC); `build_grid` is the fallback.

Coordinates are AC world space (X = east, Y = up, Z = south), see `driftmaps.ac.coords`.
"""

from __future__ import annotations

import csv
import math
import os
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from driftmaps.ac.coords import WORLD_UP

AI_VERSION = 7
GRAVITY = 9.81

POINT_DTYPE = np.dtype([("position", "<f4", (3,)), ("length", "<f4"), ("id", "<i4")])
PAYLOAD_DTYPE = np.dtype(
    [
        ("speed", "<f4"),
        ("gas", "<f4"),
        ("brake", "<f4"),
        ("obsolete_lat_g", "<f4"),
        ("radius", "<f4"),
        ("side_left", "<f4"),
        ("side_right", "<f4"),
        ("camber", "<f4"),
        ("direction", "<f4"),
        ("normal", "<f4", (3,)),
        ("length", "<f4"),
        ("forward", "<f4", (3,)),
        ("tag", "<f4"),
        ("grade", "<f4"),
    ]
)
POINT_SIZE = POINT_DTYPE.itemsize  # 20
PAYLOAD_SIZE = PAYLOAD_DTYPE.itemsize  # 72
PAYLOAD_FIELDS: tuple[str, ...] = tuple(PAYLOAD_DTYPE.names or ())
_VECTOR_FIELDS = ("normal", "forward")

_HEADER = struct.Struct("<4i")
_INT = struct.Struct("<i")
_GRID_HEADER = struct.Struct("<3f3fifi")

# Point window of the game's own payload maths (length, grade, radius). OBSERVED, see above.
PAYLOAD_WINDOW = 5
# Radius written on straights (curvature ~ 0). The observed file peaks near 2.1e6. ASSUMED.
RADIUS_MAX = 1.0e6
# Smallest side distance the generator writes, so sides stay positive on narrow roads. ASSUMED.
MIN_SIDE_M = 0.1
# Lowest target speed the generator writes, unless an explicit start/end speed asks for less.
MIN_TARGET_SPEED_MPS = 1.0
# Aerodynamic drag deceleration per (m/s)^2 for a generic street car (0.5 rho CdA / m with
# CdA ~ 0.65 m^2, m ~ 1300 kg). Only used to derive cruise throttle. ASSUMED.
DRAG_DECEL_PER_V2 = 3.0e-4
DEFAULT_MAX_SPEED_MPS = 220.0 / 3.6  # PLAN.md §5.5 / ai_line.md §5.4 top-speed cap.
# Height of the spline above the road surface. OBSERVED: 0.47 m (median; 0.46-0.49) in the
# sepang fast_lane.ai and pit_lane.ai, measured against that track's own road meshes. CSP's Traffic
# Planner exports splines with a default vertical offset of 0.5 m (EditorTabLanes.lua); CSP's
# AI Spline Editor snaps points 0.3 m above the surface by default. A line exactly on the road
# mesh is coplanar with it, and CSP raycasts down from spline points while preparing the AI
# spline ("no physical surface below an AI spline" is a known crash, SAFE_NORMALS_RAYCASTING).
AI_HEIGHT_OFFSET_M = 0.5

# Original AC lookup grid defaults. OBSERVED in one file.
GRID_SAMPLING_DENSITY = 10.0
GRID_NEIGHBORS = 10
GRID_PADDING_M = 350.0

# AcTools joins the last point to the first if they are < 20 m apart (AiSpline.Stats.cs);
# AssettoServer treats the spline as a loop below 50 m (FastLaneParser.cs). VERIFIED.
ACTOOLS_CLOSED_GAP_M = 20.0
ASSETTOSERVER_CLOSED_GAP_M = 50.0


def _empty_f32() -> np.ndarray:
    return np.zeros(0, dtype=np.float32)


def _empty_i32() -> np.ndarray:
    return np.zeros(0, dtype=np.int32)


# --------------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------------


@dataclass(eq=False)
class AiGrid:
    """The original AC spatial lookup grid stored after the payload (AcTools `AiSplineGrid`).

    `cells[ix][iz]` is an int32 array of spline point indices for the cell at column `ix`
    (along X) and row `iz` (along Z). Rows may be ragged and cells may be empty; the reader keeps
    whatever the file holds so it round-trips byte for byte.
    """

    max_extreme: np.ndarray
    min_extreme: np.ndarray
    neighbors_considered: int
    sampling_density: float
    cells: list[list[np.ndarray]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.max_extreme = _as_f32(self.max_extreme, (3,), "max_extreme")
        self.min_extreme = _as_f32(self.min_extreme, (3,), "min_extreme")
        self.neighbors_considered = int(self.neighbors_considered)
        self.sampling_density = float(self.sampling_density)
        self.cells = [
            [np.asarray(c, dtype=np.int32).reshape(-1) for c in row] for row in self.cells
        ]

    @property
    def shape(self) -> tuple[int, int]:
        """(cells along X, longest row along Z)."""
        return len(self.cells), max((len(row) for row in self.cells), default=0)

    @property
    def cell_count(self) -> int:
        return sum(len(row) for row in self.cells)

    @property
    def index_count(self) -> int:
        return sum(int(c.size) for row in self.cells for c in row)

    def cell_centre(self, ix: int, iz: int) -> tuple[float, float]:
        """World (x, z) of a cell centre (OBSERVED convention, see the module docstring)."""
        d = self.sampling_density
        return (
            float(self.min_extreme[0]) + (ix + 0.5) * d,
            float(self.min_extreme[2]) + (iz + 0.5) * d,
        )

    def to_bytes(self) -> bytes:
        """The grid block exactly as stored after a non-zero `has_grid` flag."""
        parts = [
            _GRID_HEADER.pack(
                *(float(v) for v in self.max_extreme),
                *(float(v) for v in self.min_extreme),
                self.neighbors_considered,
                self.sampling_density,
                len(self.cells),
            )
        ]
        for row in self.cells:
            parts.append(_INT.pack(len(row)))
            sizes = {cell.size for cell in row}
            if len(sizes) == 1 and len(row) > 1:  # fast path: every cell in the row same size
                block = np.empty((len(row), sizes.pop() + 1), dtype="<i4")
                block[:, 0] = block.shape[1] - 1
                block[:, 1:] = np.stack(row)
                parts.append(block.tobytes())
                continue
            for cell in row:
                arr = np.ascontiguousarray(cell, dtype="<i4")
                parts.append(_INT.pack(arr.size))
                parts.append(arr.tobytes())
        return b"".join(parts)


@dataclass(eq=False)
class AiLine:
    """One AI spline: N points plus the 18-float payload per point.

    Arrays are stored as float32 (ids as int32) so a file read and written again is
    byte-identical. Every array except `positions` may be left out (empty) when constructing:
    `lengths` then becomes the cumulative 3D distance, `ids` becomes 0..N-1, `direction` becomes
    1 (the value the in-game recorder writes before baking, OBSERVED in an unbaked pit_lane.ai)
    and every other payload field becomes 0.

    `grid` is the parsed lookup grid, or the raw bytes after the flag if they could not be
    parsed, or None for `has_grid = 0`. `trailing` holds any bytes found after the grid (none in
    files seen) so they survive a rewrite.
    """

    positions: np.ndarray
    lengths: np.ndarray = field(default_factory=_empty_f32)
    ids: np.ndarray = field(default_factory=_empty_i32)
    speed: np.ndarray = field(default_factory=_empty_f32)
    gas: np.ndarray = field(default_factory=_empty_f32)
    brake: np.ndarray = field(default_factory=_empty_f32)
    obsolete_lat_g: np.ndarray = field(default_factory=_empty_f32)
    radius: np.ndarray = field(default_factory=_empty_f32)
    side_left: np.ndarray = field(default_factory=_empty_f32)
    side_right: np.ndarray = field(default_factory=_empty_f32)
    camber: np.ndarray = field(default_factory=_empty_f32)
    direction: np.ndarray = field(default_factory=_empty_f32)
    normal: np.ndarray = field(default_factory=_empty_f32)
    length: np.ndarray = field(default_factory=_empty_f32)
    forward: np.ndarray = field(default_factory=_empty_f32)
    tag: np.ndarray = field(default_factory=_empty_f32)
    grade: np.ndarray = field(default_factory=_empty_f32)
    version: int = AI_VERSION
    lap_time: int = 0
    sample_count: int = 0
    grid: AiGrid | bytes | None = None
    trailing: bytes = b""

    def __post_init__(self) -> None:
        pos = np.asarray(self.positions, dtype=np.float32)
        if pos.size == 0:
            pos = pos.reshape(0, 3)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"positions must be (N, 3), got {pos.shape}")
        self.positions = np.ascontiguousarray(pos)
        n = pos.shape[0]

        if np.asarray(self.lengths).size == 0 and n:
            seg = np.linalg.norm(np.diff(pos.astype(np.float64), axis=0), axis=1)
            self.lengths = np.concatenate([[0.0], np.cumsum(seg)])
        self.lengths = _as_f32(self.lengths, (n,), "lengths")

        if np.asarray(self.ids).size == 0 and n:
            self.ids = np.arange(n, dtype=np.int32)
        ids = np.asarray(self.ids)
        if ids.size == 0:
            ids = ids.reshape(0)
        if ids.shape != (n,):
            raise ValueError(f"ids must have shape ({n},), got {ids.shape}")
        self.ids = np.ascontiguousarray(ids, dtype=np.int32)

        for name in PAYLOAD_FIELDS:
            shape = (n, 3) if name in _VECTOR_FIELDS else (n,)
            value = getattr(self, name)
            if np.asarray(value).size == 0 and n:
                value = np.ones(shape) if name == "direction" else np.zeros(shape)
            setattr(self, name, _as_f32(value, shape, name))

        self.version = int(self.version)
        self.lap_time = int(self.lap_time)
        self.sample_count = int(self.sample_count)
        self.trailing = bytes(self.trailing)

    @property
    def n_points(self) -> int:
        return int(self.positions.shape[0])

    @property
    def width(self) -> np.ndarray:
        """Usable width at each point: side_left + side_right (AcTools `Width`)."""
        return self.side_left + self.side_right

    @property
    def closing_gap(self) -> float:
        """Distance in metres from the last point back to the first (0 for N < 2)."""
        if self.n_points < 2:
            return 0.0
        return float(np.linalg.norm(self.positions[-1].astype(float) - self.positions[0]))

    def is_closed(self, max_gap: float = ACTOOLS_CLOSED_GAP_M) -> bool:
        """True when the ends are close enough for tools to treat the line as a loop."""
        return self.n_points >= 3 and self.closing_gap < max_gap

    def segment_lengths(self) -> np.ndarray:
        """3D distance from each point to the next, (N-1,) float64."""
        return np.linalg.norm(np.diff(self.positions.astype(np.float64), axis=0), axis=1)


def _as_f32(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.size == 0 and math.prod(shape) == 0:
        arr = arr.reshape(shape)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


# --------------------------------------------------------------------------------------------
# Reading and writing
# --------------------------------------------------------------------------------------------


def read_ai(source: str | os.PathLike[str] | bytes | bytearray | memoryview) -> AiLine:
    """Parse a v7 AI line from a file path or from the file's bytes."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
    else:
        data = Path(source).read_bytes()
    return _parse(data)


def ai_to_bytes(line: AiLine, *, include_grid: bool = True) -> bytes:
    """Serialise `line` in format v7.

    With `include_grid=True` (default) the grid and trailing bytes are written as held, so a
    file read with `read_ai` is rewritten byte for byte (with one exception: a file that had no
    `has_grid` flag at all, as FastLaneUtils writes, gains `has_grid = 0`). With
    `include_grid=False` the file ends with `has_grid = 0` (what we ship, see the module notes).
    """
    if line.version != AI_VERSION:
        raise ValueError(f"only AI format v{AI_VERSION} can be written, line has v{line.version}")
    n = line.n_points
    points = np.empty(n, dtype=POINT_DTYPE)
    points["position"] = line.positions
    points["length"] = line.lengths
    points["id"] = line.ids
    payload = np.empty(n, dtype=PAYLOAD_DTYPE)
    for name in PAYLOAD_FIELDS:
        payload[name] = getattr(line, name)

    parts = [
        _HEADER.pack(line.version, n, line.lap_time, line.sample_count),
        points.tobytes(),
        _INT.pack(n),
        payload.tobytes(),
    ]
    if include_grid and line.grid is not None:
        parts.append(_INT.pack(1))
        parts.append(line.grid if isinstance(line.grid, bytes) else line.grid.to_bytes())
        parts.append(line.trailing)
    else:
        parts.append(_INT.pack(0))
        if include_grid:
            parts.append(line.trailing)
    return b"".join(parts)


def write_ai(line: AiLine, path: str | os.PathLike[str], *, include_grid: bool = True) -> Path:
    """Write `line` to `path` (parent folders are created). Returns the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(ai_to_bytes(line, include_grid=include_grid))
    return out


def _parse(data: bytes) -> AiLine:
    size = len(data)
    if size < _HEADER.size:
        raise ValueError(f"AI file too short ({size} bytes) for its 16-byte header")
    version, n, lap_time, sample_count = _HEADER.unpack_from(data, 0)
    if version == -1:
        raise ValueError("AI file is AssettoServer's own v-1 format, which AC cannot read")
    if version != AI_VERSION:
        raise ValueError(f"AI format v{version} not supported (AcTools reads v7 only)")
    if n < 0:
        raise ValueError(f"negative point count {n}")
    offset = _HEADER.size
    needed = offset + n * POINT_SIZE + _INT.size + n * PAYLOAD_SIZE
    if size < needed:
        raise ValueError(f"AI file truncated: {n} points need {needed} bytes, file has {size}")

    points = np.frombuffer(data, dtype=POINT_DTYPE, count=n, offset=offset)
    offset += n * POINT_SIZE
    (extra_count,) = _INT.unpack_from(data, offset)
    offset += _INT.size
    if extra_count != n:
        raise ValueError(f"payload count {extra_count} does not match point count {n}")
    payload = np.frombuffer(data, dtype=PAYLOAD_DTYPE, count=n, offset=offset)
    offset += n * PAYLOAD_SIZE

    grid: AiGrid | bytes | None = None
    trailing = b""
    if offset == size:
        pass  # no has_grid flag at all (FastLaneUtils output); AcTools would fail here.
    elif size - offset < _INT.size:
        raise ValueError(f"AI file has {size - offset} stray bytes where has_grid should be")
    else:
        (has_grid,) = _INT.unpack_from(data, offset)
        offset += _INT.size
        if has_grid == 0:
            trailing = data[offset:]
        else:
            if has_grid != 1:
                warnings.warn(
                    f"has_grid flag is {has_grid}; it will be rewritten as 1", stacklevel=3
                )
            try:
                grid, end = _parse_grid(data, offset)
                trailing = data[end:]
            except ValueError as exc:
                warnings.warn(f"AI grid not parsed, kept as raw bytes: {exc}", stacklevel=3)
                grid = data[offset:]

    return AiLine(
        positions=points["position"].copy(),
        lengths=points["length"].copy(),
        ids=points["id"].copy(),
        **{name: payload[name].copy() for name in PAYLOAD_FIELDS},
        version=version,
        lap_time=lap_time,
        sample_count=sample_count,
        grid=grid,
        trailing=trailing,
    )


def _parse_grid(data: bytes, offset: int) -> tuple[AiGrid, int]:
    size = len(data)
    if size - offset < _GRID_HEADER.size:
        raise ValueError("grid header truncated")
    values = _GRID_HEADER.unpack_from(data, offset)
    offset += _GRID_HEADER.size
    max_extreme, min_extreme = values[0:3], values[3:6]
    neighbors, density, n_items = values[6], values[7], values[8]
    if n_items < 0 or n_items * _INT.size > size - offset:
        raise ValueError(f"implausible grid item count {n_items}")
    cells: list[list[np.ndarray]] = []
    for _ in range(n_items):
        if size - offset < _INT.size:
            raise ValueError("grid truncated")
        (n_sub,) = _INT.unpack_from(data, offset)
        offset += _INT.size
        if n_sub < 0 or n_sub * _INT.size > size - offset:
            raise ValueError(f"implausible grid sub-item count {n_sub}")
        row: list[np.ndarray] = []
        if n_sub > 1:  # fast path: the row is n_sub cells of one size (the case in game files)
            (first,) = _INT.unpack_from(data, offset)
            span = n_sub * (first + 1)
            if first >= 0 and span * _INT.size <= size - offset:
                block = np.frombuffer(data, dtype="<i4", count=span, offset=offset)
                block = block.reshape(n_sub, first + 1)
                if (block[:, 0] == first).all():
                    cells.append(list(block[:, 1:].copy()))
                    offset += span * _INT.size
                    continue
        for _ in range(n_sub):
            if size - offset < _INT.size:
                raise ValueError("grid truncated")
            (count,) = _INT.unpack_from(data, offset)
            offset += _INT.size
            if count < 0 or count * _INT.size > size - offset:
                raise ValueError(f"implausible grid cell size {count}")
            row.append(np.frombuffer(data, dtype="<i4", count=count, offset=offset).copy())
            offset += count * _INT.size
        cells.append(row)
    grid = AiGrid(
        max_extreme=np.array(max_extreme, dtype=np.float32),
        min_extreme=np.array(min_extreme, dtype=np.float32),
        neighbors_considered=neighbors,
        sampling_density=density,
        cells=cells,
    )
    return grid, offset


# --------------------------------------------------------------------------------------------
# Lookup grid generation (fallback for vanilla AC, spike S3)
# --------------------------------------------------------------------------------------------


def build_grid(
    positions: np.ndarray,
    *,
    sampling_density: float = GRID_SAMPLING_DENSITY,
    neighbors: int = GRID_NEIGHBORS,
    padding: float = GRID_PADDING_M,
) -> AiGrid:
    """Build the original AC lookup grid for spline points `positions` (N, 3).

    Follows the OBSERVED convention (module docstring): each cell lists the `neighbors` points
    nearest its centre in XZ, nearest first (ties by lower index). On the observed file this
    reproduces the extremes and shape exactly and 30154 of 30155 cells' point sets; the rest
    differ only where distances tie to within float precision. Exact k-nearest search with
    quadtree pruning (no KD-tree dependency): ~0.3 s for a 5 km loop, ~10 s for a 19 km route
    whose padded bounding box holds ~600k cells (the file then grows by ~25 MB, as in game files).
    """
    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise ValueError("positions must be a non-empty (N, 3) array")
    if sampling_density <= 0:
        raise ValueError("sampling_density must be positive")
    xz = np.ascontiguousarray(pts[:, [0, 2]])
    # The observed file stores the exact point bounds +- padding (not rounded), as float32.
    lo = (xz.min(axis=0) - padding).astype(np.float32).astype(np.float64)
    hi = (xz.max(axis=0) + padding).astype(np.float32).astype(np.float64)
    d = float(np.float32(sampling_density))
    nx = max(1, int((hi[0] - lo[0]) / d))
    nz = max(1, int((hi[1] - lo[1]) / d))
    k = min(int(neighbors), xz.shape[0])
    centres_x = lo[0] + (np.arange(nx) + 0.5) * d
    centres_z = lo[1] + (np.arange(nz) + 0.5) * d
    out = np.empty((nx, nz, k), dtype=np.int32)

    def prune(cand: np.ndarray, x0: float, x1: float, z0: float, z1: float) -> np.ndarray:
        """Candidates that can be among the k nearest of some centre in [x0,x1] x [z0,z1]."""
        px, pz = xz[cand, 0], xz[cand, 1]
        lower = np.hypot(
            np.maximum(np.maximum(x0 - px, px - x1), 0.0),
            np.maximum(np.maximum(z0 - pz, pz - z1), 0.0),
        )
        upper = np.hypot(
            np.maximum(np.abs(px - x0), np.abs(px - x1)),
            np.maximum(np.abs(pz - z0), np.abs(pz - z1)),
        )
        # Every centre in the box has k points within the k-th smallest upper bound.
        return cand[lower <= np.partition(upper, k - 1)[k - 1]]

    # Quadtree over the cells: prune candidates per block, brute-force small blocks.
    stack = [(np.arange(xz.shape[0]), 0, nx, 0, nz)]
    while stack:
        cand, x0, x1, z0, z1 = stack.pop()
        cx, cz = centres_x[x0:x1], centres_z[z0:z1]
        cand = prune(cand, cx[0], cx[-1], cz[0], cz[-1])
        if cx.size * cz.size > 64 and cand.size > 4 * k:
            if x1 - x0 >= z1 - z0:
                mid = (x0 + x1) // 2
                stack += [(cand, x0, mid, z0, z1), (cand, mid, x1, z0, z1)]
            else:
                mid = (z0 + z1) // 2
                stack += [(cand, x0, x1, z0, mid), (cand, x0, x1, mid, z1)]
            continue
        gx, gz = np.meshgrid(cx, cz, indexing="ij")
        d2 = (gx.reshape(-1, 1) - xz[cand, 0]) ** 2 + (gz.reshape(-1, 1) - xz[cand, 1]) ** 2
        idx = np.broadcast_to(cand, d2.shape)
        keep = min(cand.size, k + 8)  # spares so exact ties at the k-th place survive
        if keep < cand.size:
            part = np.argpartition(d2, keep - 1, axis=1)[:, :keep]
            d2 = np.take_along_axis(d2, part, axis=1)
            idx = np.take_along_axis(idx, part, axis=1)
        order = np.lexsort((idx, d2), axis=1)[:, :k]  # nearest first, ties by lower index
        chosen = np.take_along_axis(idx, order, axis=1)
        out[x0:x1, z0:z1] = chosen.reshape(cx.size, cz.size, k)

    cells = [list(out[ix]) for ix in range(nx)]
    return AiGrid(
        max_extreme=np.array([hi[0], 0.0, hi[1]], dtype=np.float32),
        min_extreme=np.array([lo[0], 0.0, lo[1]], dtype=np.float32),
        neighbors_considered=int(neighbors),
        sampling_density=d,
        cells=cells,
    )


# --------------------------------------------------------------------------------------------
# Geometry helpers (shared by the generator and the dump)
# --------------------------------------------------------------------------------------------


def _shifted(n: int, k: int, closed: bool) -> np.ndarray:
    """Indices i + k, wrapped on loops and clamped to the ends on open lines."""
    idx = np.arange(n) + k
    return idx % n if closed else np.clip(idx, 0, n - 1)


def _fill_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace invalid entries with the nearest valid one (edge extension); 0 if none valid."""
    if valid.all():
        return values
    if not valid.any():
        return np.zeros_like(values)
    idx = np.arange(values.size)
    good = idx[valid]
    nearest = good[np.clip(np.searchsorted(good, idx), 0, good.size - 1)]
    prev = good[np.clip(np.searchsorted(good, idx) - 1, 0, good.size - 1)]
    pick = np.where(np.abs(prev - idx) <= np.abs(nearest - idx), prev, nearest)
    return np.where(valid, values, values[pick])


def signed_curvature(
    positions: np.ndarray, *, closed: bool, window: int = PAYLOAD_WINDOW
) -> np.ndarray:
    """Horizontal curvature (1/m) at each point, positive for a left turn.

    Uses the formula OBSERVED for the game's `radius` field: kappa = 2 sin(theta / 2) / chord,
    with chord = |P[i+w] - P[i]|_xz and theta the signed turn angle between P[i-w] -> P[i] and
    P[i] -> P[i+w]. Left = up x forward (`driftmaps.ac.coords`), which is a positive Y
    component of back x fwd. Near the ends of an open line the window is clamped and therefore
    lopsided; there the circle through P[i-a], P[i], P[i+b] is used instead. Both are exact on
    circles.
    """
    p = np.asarray(positions, dtype=np.float64)
    n = p.shape[0]
    if n < 3:
        return np.zeros(n)
    xz = p[:, [0, 2]]
    back = xz - xz[_shifted(n, -window, closed)]
    fwd = xz[_shifted(n, window, closed)] - xz
    cross_y = back[:, 1] * fwd[:, 0] - back[:, 0] * fwd[:, 1]
    theta = np.arctan2(cross_y, np.einsum("ij,ij->i", back, fwd))
    len_back = np.linalg.norm(back, axis=1)
    chord = np.linalg.norm(fwd, axis=1)
    span = np.linalg.norm(back + fwd, axis=1)
    valid = (chord > 1e-9) & (len_back > 1e-9) & (span > 1e-9)
    safe = np.where(valid, 1.0, 0.0)
    kappa = 2.0 * np.sin(theta / 2.0) / np.where(valid, chord, 1.0) * safe
    if not closed:
        idx = np.arange(n)
        lopsided = (idx < window) | (idx > n - 1 - window)
        circum = 2.0 * cross_y / np.where(valid, len_back * chord * span, 1.0) * safe
        kappa = np.where(lopsided, circum, kappa)
    return _fill_invalid(kappa, valid)


def _lookback_chord(positions: np.ndarray, closed: bool, window: int = PAYLOAD_WINDOW):
    """Horizontal chord from point i-w to i, plus the rise over it (OBSERVED payload maths).

    On open lines the first points have no look-back; they use the chord to i+w instead
    (ASSUMED: only closed lines have been observed).
    """
    p = np.asarray(positions, dtype=np.float64)
    n = p.shape[0]
    j = _shifted(n, -window, closed)
    delta = p - p[j]
    if not closed:
        ahead = _shifted(n, window, closed)
        early = np.arange(n) < window
        delta[early] = p[ahead][early] - p[early]
    chord = np.linalg.norm(delta[:, [0, 2]], axis=1)
    return chord, delta[:, 1]


def _moving_average(values: np.ndarray, window: int, closed: bool) -> np.ndarray:
    if window <= 1 or values.size < 3:
        return values
    window = min(window | 1, values.size if values.size % 2 else values.size - 1)
    half = window // 2
    mode = "wrap" if closed else "edge"
    padded = np.pad(values, half, mode=mode)
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _unit_rows(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(norm > 1e-12, v / np.where(norm > 1e-12, norm, 1.0), 0.0)


# --------------------------------------------------------------------------------------------
# Speed profile and generator
# --------------------------------------------------------------------------------------------


def lateral_speed_limit(
    curvature: np.ndarray,
    *,
    mu: float = 1.0,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
    camber: np.ndarray | None = None,
) -> np.ndarray:
    """Cornering speed limit per point: v^2 = g R (mu + tan phi) / (1 - mu tan phi).

    phi is the bank angle into the turn (asin of the `camber` payload convention); with no
    camber this is sqrt(mu g R). Capped at `max_speed_mps`, floored at MIN_TARGET_SPEED_MPS.
    """
    kappa = np.abs(np.asarray(curvature, dtype=np.float64))
    if camber is None:
        tan_phi = np.zeros_like(kappa)
    else:
        tan_phi = np.tan(np.arcsin(np.clip(np.asarray(camber, dtype=np.float64), -0.95, 0.95)))
    num = mu + tan_phi
    den = 1.0 - mu * tan_phi
    with np.errstate(divide="ignore", invalid="ignore"):
        v2 = np.where((kappa > 1e-12) & (den > 1e-9), GRAVITY * num / (kappa * den), np.inf)
    v2 = np.where(num <= 0.0, 0.0, v2)
    return np.clip(np.sqrt(v2), MIN_TARGET_SPEED_MPS, max_speed_mps)


def speed_profile(
    curvature: np.ndarray,
    segment_lengths: np.ndarray,
    *,
    closed: bool,
    mu: float = 1.0,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
    max_accel_mps2: float = 4.0,
    max_brake_mps2: float = 8.0,
    segment_grade: np.ndarray | None = None,
    camber: np.ndarray | None = None,
    start_speed_mps: float | None = None,
    end_speed_mps: float | None = None,
) -> np.ndarray:
    """Forward-backward speed profile on a friction circle (ai_line.md §5.3).

    `curvature` is per point (1/m, signed or not). `segment_lengths[i]` is the distance from
    point i to i+1: N values on a loop (the last one closes it), N-1 on an open line.
    `segment_grade[i]` is that segment's rise over run (positive uphill), same length.
    Longitudinal grip left after cornering is sqrt((mu g)^2 - (v^2 kappa)^2), further capped
    by the engine (`max_accel_mps2`) and brakes (`max_brake_mps2`); gravity along the grade is
    added. Loops run each pass twice around so limits wrap past the start.
    """
    kappa = np.abs(np.asarray(curvature, dtype=np.float64))
    n = kappa.size
    ds = np.asarray(segment_lengths, dtype=np.float64)
    n_seg = n if closed else max(n - 1, 0)
    if ds.shape != (n_seg,):
        raise ValueError(f"segment_lengths must have {n_seg} entries, got {ds.shape}")
    grade = (
        np.zeros(n_seg) if segment_grade is None else np.asarray(segment_grade, dtype=np.float64)
    )
    if grade.shape != (n_seg,):
        raise ValueError(f"segment_grade must have {n_seg} entries, got {grade.shape}")
    v = lateral_speed_limit(kappa, mu=mu, max_speed_mps=max_speed_mps, camber=camber)
    if n == 0:
        return v
    if not closed:
        if start_speed_mps is not None:
            v[0] = min(v[0], max(0.0, start_speed_mps))
        if end_speed_mps is not None:
            v[-1] = min(v[-1], max(0.0, end_speed_mps))
    if n_seg == 0:
        return v

    grip2 = (mu * GRAVITY) ** 2
    vs = v.tolist()
    ks = kappa.tolist()
    dl = ds.tolist()
    gr = grade.tolist()
    laps = 2 if closed else 1

    for step in range(laps * n_seg):  # forward pass: acceleration limit
        i = step % n_seg
        j = (i + 1) % n
        lat = vs[i] * vs[i] * ks[i]
        a = min(max_accel_mps2, math.sqrt(max(0.0, grip2 - lat * lat))) - GRAVITY * gr[i]
        reach = vs[i] * vs[i] + 2.0 * max(a, 0.0) * dl[i]
        if reach < vs[j] * vs[j]:
            vs[j] = math.sqrt(reach)

    for step in range(laps * n_seg - 1, -1, -1):  # backward pass: braking limit
        i = step % n_seg
        j = (i + 1) % n
        lat = vs[j] * vs[j] * ks[j]
        b = min(max_brake_mps2, math.sqrt(max(0.0, grip2 - lat * lat))) + GRAVITY * gr[i]
        reach = vs[j] * vs[j] + 2.0 * max(b, 0.0) * dl[i]
        if reach < vs[i] * vs[i]:
            vs[i] = math.sqrt(reach)

    return np.asarray(vs)


def _resample(
    points: np.ndarray, attrs: np.ndarray, spacing: float, closed: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Linear resample of a polyline (and per-point attributes) at uniform arc length."""
    keep = np.ones(points.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9
    points, attrs = points[keep], attrs[keep]
    if closed and points.shape[0] >= 2 and np.linalg.norm(points[-1] - points[0]) <= 1e-6:
        points, attrs = points[:-1], attrs[:-1]  # input repeated the first point
    if points.shape[0] < 2:
        raise ValueError("centreline needs at least two distinct points")
    if closed:
        points = np.vstack([points, points[:1]])
        attrs = np.vstack([attrs, attrs[:1]])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    total = arc[-1]
    if closed:
        count = max(3, int(round(total / spacing)))
        t = np.arange(count) * (total / count)
    else:
        t = np.linspace(0.0, total, max(1, int(round(total / spacing))) + 1)
    new_points = np.column_stack([np.interp(t, arc, points[:, c]) for c in range(3)])
    new_attrs = np.column_stack([np.interp(t, arc, attrs[:, c]) for c in range(attrs.shape[1])])
    return new_points, new_attrs


def _per_point(value: float | np.ndarray, m: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(m, float(arr))
    if arr.shape != (m,):
        raise ValueError(f"{name} must be a scalar or have shape ({m},), got {arr.shape}")
    return arr


def build_ai_line(
    centreline: np.ndarray,
    half_width_left: float | np.ndarray,
    half_width_right: float | np.ndarray,
    *,
    closed: bool,
    mu: float = 1.0,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
    max_accel_mps2: float = 4.0,
    max_brake_mps2: float = 8.0,
    margin: float = 0.5,
    spacing: float = 1.0,
    surface_normals: np.ndarray | None = None,
    start_speed_mps: float | None = None,
    end_speed_mps: float | None = None,
    curvature_smoothing: int = 5,
    with_grid: bool = False,
    height_offset_m: float = AI_HEIGHT_OFFSET_M,
) -> AiLine:
    """Turn a driving line into a complete AI line with every payload field filled.

    `centreline` (M, 3) is the line the AI should follow, in AC world coordinates and in driving
    order (it is not optimised here; pass a racing line to get one). Half-widths are the
    distances from that line to the left/right road edges, scalar or per input point;
    `surface_normals` (M, 3) is the road normal under each point (default straight up). The
    line is resampled linearly at about `spacing` metres, so it should already be smooth.
    `centreline` lies on the road surface; the written positions are lifted `height_offset_m`
    straight up from it (AI_HEIGHT_OFFSET_M, like game-recorded lines; flat roads only at M0).

    `closed=True` makes a loop: the last point is not repeated and every windowed quantity and
    the speed profile wrap around. Open (A->B) lines treat their ends as ends;
    `start_speed_mps` / `end_speed_mps` optionally pin the speed there.

    Field choices (see the module docstring for what was OBSERVED in a game-baked file):
    - speed in m/s from `speed_profile` (lateral limit sqrt(mu g R), accel/brake limits).
      ASSUMED m/s until S0.
    - gas / brake from the profile's longitudinal acceleration plus grade and a small drag term,
      normalised by the accel / brake limits; never both non-zero. ASSUMED scaling.
    - obsolete_lat_g = v^2 kappa / g (signed like direction). ASSUMED to match the recorder.
    - radius = 1 / |kappa| from `signed_curvature` (5-point window, then a `curvature_smoothing`
      point moving average), capped at RADIUS_MAX on straights, never 0. ASSUMED.
    - direction = +1 left turn, -1 right turn (+1 on dead-straight points); camber =
      direction * dot(normal, left). ASSUMED (OBSERVED convention).
    - side_left / side_right = half-width - margin, at least MIN_SIDE_M.
    - length = horizontal distance to point i-5; grade = rise over that distance. ASSUMED.
    - forward = unit 3D tangent; tag = 0; lap_time = sample_count = 0.
    - grid only when `with_grid` (CSP builds its own; spike S3 decides for vanilla).
    """
    pts = np.asarray(centreline, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"centreline must be (M, 3), got {pts.shape}")
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    m = pts.shape[0]
    normals_in = (
        np.tile([0.0, 1.0, 0.0], (m, 1))
        if surface_normals is None
        else np.asarray(surface_normals, dtype=np.float64)
    )
    if normals_in.shape != (m, 3):
        raise ValueError(f"surface_normals must be ({m}, 3), got {normals_in.shape}")
    attrs = np.column_stack(
        [
            _per_point(half_width_left, m, "half_width_left"),
            _per_point(half_width_right, m, "half_width_right"),
            normals_in,
        ]
    )
    p, a = _resample(pts, attrs, spacing, closed)
    # Constant lift above the road: every quantity below is a difference of positions (or a
    # horizontal one), so curvature, grade, length, sides and the grid cells are unchanged.
    p = p + height_offset_m * WORLD_UP
    n = p.shape[0]
    hw_left, hw_right = a[:, 0], a[:, 1]
    normal = _unit_rows(a[:, 2:5])
    normal[np.linalg.norm(normal, axis=1) == 0] = (0.0, 1.0, 0.0)

    # Segments i -> i+1 (N on a loop, N-1 open) and cumulative length from point 0.
    nxt = p[_shifted(n, 1, True)] if closed else p[1:]
    seg_vec = nxt - (p if closed else p[:-1])
    seg_len = np.linalg.norm(seg_vec, axis=1)
    seg_run = np.linalg.norm(seg_vec[:, [0, 2]], axis=1)
    seg_grade = np.where(seg_run > 1e-9, seg_vec[:, 1] / np.where(seg_run > 1e-9, seg_run, 1), 0)
    lengths = np.concatenate([[0.0], np.cumsum(seg_len[: n - 1])])

    forward = _unit_rows(p[_shifted(n, 1, closed)] - p[_shifted(n, -1, closed)])
    up = np.array([0.0, 1.0, 0.0])
    left = _unit_rows(np.cross(up, forward))  # coords.left_of, vectorised

    kappa = _moving_average(signed_curvature(p, closed=closed), curvature_smoothing, closed)
    abs_k = np.abs(kappa)
    radius = np.where(abs_k > 1.0 / RADIUS_MAX, 1.0 / np.maximum(abs_k, 1e-30), RADIUS_MAX)
    # Straight points (radius capped at RADIUS_MAX) get +1 as documented; comparing with 0 let
    # float noise in the curvature (~1e-14) flip them to -1 at random.
    direction = np.where(kappa < -1.0 / RADIUS_MAX, -1.0, 1.0)
    camber = direction * np.einsum("ij,ij->i", normal, left)

    speed = speed_profile(
        kappa,
        seg_len,
        closed=closed,
        mu=mu,
        max_speed_mps=max_speed_mps,
        max_accel_mps2=max_accel_mps2,
        max_brake_mps2=max_brake_mps2,
        segment_grade=seg_grade,
        camber=camber,
        start_speed_mps=start_speed_mps,
        end_speed_mps=end_speed_mps,
    )

    # Longitudinal acceleration over each point's outgoing segment (last open point: incoming).
    v_next = speed[_shifted(n, 1, closed)]
    if closed:
        accel = (v_next**2 - speed**2) / (2.0 * seg_len)
        grade_out = seg_grade
    else:
        accel = np.zeros(n)
        grade_out = np.zeros(n)
        if n > 1:
            accel[:-1] = (v_next[:-1] ** 2 - speed[:-1] ** 2) / (2.0 * seg_len)
            accel[-1] = accel[-2]
            grade_out[:-1] = seg_grade
            grade_out[-1] = seg_grade[-1]
    drive = accel + GRAVITY * grade_out + DRAG_DECEL_PER_V2 * speed**2
    gas = np.clip(drive / max_accel_mps2, 0.0, 1.0)
    brake = np.clip(-drive / min(max_brake_mps2, mu * GRAVITY), 0.0, 1.0)

    look_len, look_rise = _lookback_chord(p, closed)
    grade = np.where(look_len > 1e-9, look_rise / np.where(look_len > 1e-9, look_len, 1.0), 0.0)

    line = AiLine(
        positions=p,
        lengths=lengths,
        ids=np.arange(n, dtype=np.int32),
        speed=speed,
        gas=gas,
        brake=brake,
        obsolete_lat_g=speed**2 * kappa / GRAVITY,
        radius=radius,
        side_left=np.maximum(hw_left - margin, MIN_SIDE_M),
        side_right=np.maximum(hw_right - margin, MIN_SIDE_M),
        camber=camber,
        direction=direction,
        normal=normal,
        length=look_len,
        forward=forward,
        tag=np.zeros(n),
        grade=grade,
    )
    if with_grid:
        line.grid = build_grid(line.positions)
    return line


# --------------------------------------------------------------------------------------------
# Dump / calibration (drift-maps ai-dump, spike S0)
# --------------------------------------------------------------------------------------------

CSV_COLUMNS: tuple[str, ...] = (
    "index",
    "id",
    "x",
    "y",
    "z",
    "cum_length",
    "speed",
    "gas",
    "brake",
    "obsolete_lat_g",
    "radius",
    "side_left",
    "side_right",
    "camber",
    "direction",
    "normal_x",
    "normal_y",
    "normal_z",
    "length",
    "forward_x",
    "forward_y",
    "forward_z",
    "tag",
    "grade",
    "segment_to_next",
)


def write_csv(line: AiLine, csv_path: str | os.PathLike[str]) -> Path:
    """Write every point and payload field (plus the distance to the next point) to a CSV.

    Floats use 9 significant digits, enough to round-trip float32 exactly.
    """
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = line.n_points
    seg = np.zeros(n)
    if n > 1:
        seg[:-1] = line.segment_lengths()
        seg[-1] = line.closing_gap
    columns: list[np.ndarray] = [
        np.arange(n),
        line.ids,
        *line.positions.T,
        line.lengths,
        line.speed,
        line.gas,
        line.brake,
        line.obsolete_lat_g,
        line.radius,
        line.side_left,
        line.side_right,
        line.camber,
        line.direction,
        *line.normal.T,
        line.length,
        *line.forward.T,
        line.tag,
        line.grade,
        seg,
    ]
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for i in range(n):
            writer.writerow(
                [
                    str(int(col[i])) if col.dtype.kind in "iu" else f"{float(col[i]):.9g}"
                    for col in columns
                ]
            )
    return out


def dump(
    path: str | os.PathLike[str] | bytes,
    csv_path: str | os.PathLike[str] | None = None,
) -> str:
    """Human-readable summary of an AI line, built to calibrate payload units (spike S0).

    Prints the header, point spacing, total length, open/closed, every payload field's range,
    unit hypotheses tested against the geometry (speed m/s vs km/h, radius vs circumradius,
    camber vs normal, length/grade windows, direction vs turn side), value counts of direction
    and tag, and the grid. Optionally writes every field to `csv_path`.
    """
    data = bytes(path) if isinstance(path, (bytes, bytearray)) else Path(path).read_bytes()
    line = read_ai(data)
    label = "<bytes>" if isinstance(path, (bytes, bytearray)) else str(path)
    out: list[str] = [f"AI line: {label}"]
    out += _dump_header(line, len(data))
    out += _dump_geometry(line)
    out += _dump_fields(line)
    out += _dump_hypotheses(line)
    out += _dump_grid(line)
    if csv_path is not None:
        write_csv(line, csv_path)
        out.append(f"\nCSV: wrote {line.n_points} rows x {len(CSV_COLUMNS)} columns to {csv_path}")
    return "\n".join(out)


def _f(x: float, digits: int = 4) -> str:
    """Compact number: fixed point without trailing zeros, or %g when very large or small."""
    x = float(x)
    if not math.isfinite(x):
        return str(x)
    if x != 0 and (abs(x) >= 1e6 or abs(x) < 1e-3):
        return f"{x:.{digits}g}"
    text = f"{x:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _stats(values: np.ndarray) -> str:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "n/a"
    return (
        f"min {_f(v.min())}  mean {_f(v.mean())}  median {_f(float(np.median(v)))}  "
        f"max {_f(v.max())}"
    )


def _counts(values: np.ndarray, limit: int = 12) -> str:
    uniq, counts = np.unique(np.asarray(values, dtype=np.float64), return_counts=True)
    order = np.argsort(-counts, kind="stable")[:limit]
    items = ", ".join(f"{_f(float(uniq[i]))}: {int(counts[i])}" for i in order)
    more = f", ... ({uniq.size} distinct)" if uniq.size > limit else ""
    return "{" + items + more + "}"


def _ratio(num: np.ndarray, den: np.ndarray, mask: np.ndarray) -> float:
    """Median of num/den over mask (NaN if the mask is empty)."""
    m = mask & np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-12)
    return float(np.median(num[m] / den[m])) if m.any() else float("nan")


def _dump_header(line: AiLine, size: int) -> list[str]:
    n = line.n_points
    ids_ok = bool(np.array_equal(line.ids, np.arange(n)))
    return [
        f"  file size       {size:,} bytes",
        f"  version         {line.version}",
        f"  points (N)      {n}   ids are 0..N-1: {'yes' if ids_ok else 'NO'}",
        f"  lap_time        {line.lap_time}",
        f"  sample_count    {line.sample_count}",
    ]


def _dump_geometry(line: AiLine) -> list[str]:
    n = line.n_points
    out = ["", "Geometry (from positions)"]
    if n < 2:
        out.append("  fewer than 2 points")
        return out
    seg = line.segment_lengths()
    gap = line.closing_gap
    total = float(seg.sum())
    if gap < ACTOOLS_CLOSED_GAP_M:
        verdict = f"CLOSED loop (AcTools joins < {ACTOOLS_CLOSED_GAP_M:g} m)"
    elif gap < ASSETTOSERVER_CLOSED_GAP_M:
        verdict = (
            f"open for AcTools (>= {ACTOOLS_CLOSED_GAP_M:g} m) but CLOSED for AssettoServer "
            f"(< {ASSETTOSERVER_CLOSED_GAP_M:g} m)"
        )
    else:
        verdict = "OPEN (point-to-point)"
    cum = line.lengths.astype(np.float64)
    cum_err = float(np.abs(np.diff(cum) - seg).max())
    pos = line.positions.astype(np.float64)
    out += [
        f"  spacing (m)     {_stats(seg)}",
        f"  spacing p5/p95  {_f(float(np.percentile(seg, 5)))} / "
        f"{_f(float(np.percentile(seg, 95)))}",
        f"  total length    {_f(total, 2)} m along the points"
        f" (+ {_f(gap, 2)} m end->start = {_f(total + gap, 2)} m as a loop)",
        f"  end -> start    {_f(gap, 3)} m: {verdict}",
        f"  cum. length     first {_f(float(cum[0]))}, last {_f(float(cum[-1]), 2)}; "
        f"max |diff - segment| = {_f(cum_err)} m"
        + ("  -> cumulative distance from point 0" if cum_err < 0.05 else "  -> NOT cumulative"),
        f"  x range         {_f(pos[:, 0].min(), 2)} .. {_f(pos[:, 0].max(), 2)}",
        f"  y range         {_f(pos[:, 1].min(), 2)} .. {_f(pos[:, 1].max(), 2)} (elevation)",
        f"  z range         {_f(pos[:, 2].min(), 2)} .. {_f(pos[:, 2].max(), 2)}",
    ]
    return out


def _dump_fields(line: AiLine) -> list[str]:
    out = ["", "Payload fields"]
    out.append(
        f"  {'field':16s}{'min':>12s}{'mean':>12s}{'median':>12s}{'max':>12s}"
        f"{'zero %':>9s}{'NaN':>6s}"
    )
    rows: list[tuple[str, np.ndarray]] = []
    for name in PAYLOAD_FIELDS:
        value = getattr(line, name)
        if value.ndim == 2:
            rows += [(f"{name}.{axis}", value[:, i]) for i, axis in enumerate("xyz")]
        else:
            rows.append((name, value))
    for name, value in rows:
        v = value.astype(np.float64)
        finite = v[np.isfinite(v)]
        if finite.size:
            cells = [finite.min(), finite.mean(), float(np.median(finite)), finite.max()]
        else:
            cells = [float("nan")] * 4
        zero = 100.0 * float(np.mean(v == 0)) if v.size else 0.0
        nan = int(np.isnan(v).sum())
        out.append(
            f"  {name:16s}" + "".join(f"{_f(c):>12s}" for c in cells) + f"{zero:9.1f}{nan:6d}"
        )
    return out


def _dump_hypotheses(line: AiLine) -> list[str]:
    n = line.n_points
    out = ["", "Unit hypotheses (spike S0: compare with known car / track facts)"]
    if n < 2 * PAYLOAD_WINDOW + 1:
        out.append(f"  needs at least {2 * PAYLOAD_WINDOW + 1} points")
        return out
    closed = line.is_closed()
    pos = line.positions.astype(np.float64)
    seg = line.segment_lengths()
    spd = line.speed.astype(np.float64)
    radius = line.radius.astype(np.float64)
    lat = line.obsolete_lat_g.astype(np.float64)
    direction = line.direction.astype(np.float64)
    w = PAYLOAD_WINDOW
    kappa_geo = signed_curvature(pos, closed=closed)
    r_geo = 1.0 / np.maximum(np.abs(kappa_geo), 1e-30)

    # speed
    vmax = float(np.nanmax(spd))
    moving = spd[:-1] > 0.1
    lap_ms = float(np.sum(seg[moving] / spd[:-1][moving]))
    out.append(
        f"  speed           max {_f(vmax, 3)}: as m/s = {_f(vmax * 3.6, 1)} km/h; "
        f"as km/h = {_f(vmax / 3.6, 2)} m/s"
    )
    out.append(
        f"                  time along the points: {_f(lap_ms, 1)} s if m/s, "
        f"{_f(lap_ms * 3.6, 1)} s if km/h"
    )
    for source, r in (("radius field", radius), ("geometric radius", r_geo)):
        pred = spd**2 / (np.maximum(r, 1e-6) * GRAVITY)
        mask = (r > 1) & (r < 500) & (np.abs(lat) > 0.05) & (spd > 1)
        r_ms = _ratio(np.abs(lat), pred, mask)
        r_kmh = r_ms * 3.6**2
        if not np.isfinite(r_ms):
            out.append(f"                  lat_g check ({source}): n/a (no usable corners)")
            continue
        guess = "m/s" if abs(r_ms - 1) < 0.2 else "km/h" if abs(r_kmh - 1) < 0.2 else "unclear"
        out.append(
            f"                  |obsolete_lat_g| / (v^2 / (R g)), R = {source}: {_f(r_ms, 4)} "
            f"if m/s, {_f(r_kmh, 4)} if km/h  -> speed looks like {guess} (if lat_g is in g)"
        )

    # radius
    straight = float(np.mean(radius > 1000)) * 100
    out.append(
        f"  radius          {_stats(radius)}; > 1000 m at {straight:.1f}% of points; "
        f"<= 0 at {int(np.sum(radius <= 0))} points"
    )
    r_ratio = _ratio(radius, r_geo, (radius > 1) & (radius < 2000) & (r_geo < 1e5))
    out.append(
        f"                  radius / horizontal circumradius over +-{w} points: median "
        f"{_f(r_ratio)}  (~1 -> metres, same window as the game's maths)"
    )

    # sides
    width = line.width.astype(np.float64)
    out.append(f"  side_left       {_stats(line.side_left)}")
    out.append(f"  side_right      {_stats(line.side_right)}")
    out.append(f"  width (l+r)     {_stats(width)}  (metres; CM draws max(side, 1))")

    # direction / turn side
    turning = np.abs(kappa_geo) > 1.0 / 300.0
    agree = (
        float(np.mean(direction[turning] == np.sign(kappa_geo[turning]))) if turning.any() else 0
    )
    out.append(f"  direction       counts {_counts(direction)}")
    if turning.any():
        out.append(
            f"                  == +1 on left turns / -1 on right turns at {agree * 100:.1f}% of "
            f"{int(turning.sum())} cornering points (R < 300 m); left = up x forward"
        )
    lat_mask = np.abs(lat) > 0.05
    if lat_mask.any():
        same = float(np.mean(np.sign(lat[lat_mask]) == direction[lat_mask])) * 100
        out.append(
            f"  obsolete_lat_g  max |.| {_f(float(np.abs(lat).max()))}; sign == direction "
            f"at {same:.1f}% of points with |lat_g| > 0.05"
        )

    # camber vs normal
    normal = line.normal.astype(np.float64)
    fwd_geo = _unit_rows(pos[_shifted(n, 1, closed)] - pos[_shifted(n, -1, closed)])
    left = _unit_rows(np.cross([0.0, 1.0, 0.0], fwd_geo))
    n_left = np.einsum("ij,ij->i", normal, left)
    cam = line.camber.astype(np.float64)
    cam_ratio = _ratio(cam * direction, n_left, np.abs(n_left) > 0.005)
    out.append(
        f"  camber          range {_f(float(cam.min()))} .. {_f(float(cam.max()))}; as rad = "
        f"{_f(math.degrees(float(cam.min())), 2)} .. {_f(math.degrees(float(cam.max())), 2)} deg"
    )
    out.append(
        f"                  camber * direction / dot(normal, left): median {_f(cam_ratio)}  "
        "(~1 -> sin(bank), positive = banked into the turn)"
    )

    # normal
    n_norm = np.linalg.norm(normal, axis=1)
    tilt = np.degrees(np.arccos(np.clip(normal[:, 1] / np.maximum(n_norm, 1e-12), -1, 1)))
    out.append(
        f"  normal          |n| {_stats(n_norm)}; tilt from vertical max "
        f"{_f(float(tilt[n_norm > 0].max()) if (n_norm > 0).any() else 0.0, 2)} deg"
    )

    # length
    plen = line.length.astype(np.float64)
    look_len, look_rise = _lookback_chord(pos, closed)
    seg_next = np.append(seg, line.closing_gap if closed else seg[-1])
    nz = plen != 0
    out.append(f"  length          {_stats(plen)}")
    if nz.any():
        err_look = float(np.median(np.abs(plen - look_len)[nz]))
        out.append(
            f"                  / segment to next: median {_f(_ratio(plen, seg_next, nz))}; "
            f"vs horizontal chord to point i-{w}: median |error| {_f(err_look)} m  "
            f"(~0 -> {w}-point look-back distance)"
        )

    # forward
    fwd = line.forward.astype(np.float64)
    f_norm = np.linalg.norm(fwd, axis=1)
    f_nz = f_norm > 0
    msg = f"  forward         |f| {_stats(f_norm)}; zero rows {100 * float(np.mean(~f_nz)):.1f}%"
    if f_nz.any():
        dots = np.einsum("ij,ij->i", fwd[f_nz] / f_norm[f_nz, None], fwd_geo[f_nz])
        msg += f"; mean dot with geometric tangent {_f(float(dots.mean()))}"
    out.append(msg)

    # tag, grade
    out.append(f"  tag             counts {_counts(line.tag)}")
    grade = line.grade.astype(np.float64)
    look_grade = np.where(look_len > 1e-9, look_rise / np.maximum(look_len, 1e-9), 0.0)
    g_nz = grade != 0
    g_lo, g_hi = float(grade.min()), float(grade.max())
    out.append(
        f"  grade           range {_f(g_lo)} .. {_f(g_hi)}; as ratio = "
        f"{_f(g_lo * 100, 2)} .. {_f(g_hi * 100, 2)} %; as rad = "
        f"{_f(math.degrees(g_lo), 2)} .. {_f(math.degrees(g_hi), 2)} deg"
    )
    if g_nz.any():
        err_g = float(np.median(np.abs(grade - look_grade)[g_nz]))
        out.append(
            f"                  vs rise / run over the {w}-point look-back: median |error| "
            f"{_f(err_g)}  (~0 -> rise-over-run ratio)"
        )

    # gas / brake
    gas, brake = line.gas.astype(np.float64), line.brake.astype(np.float64)
    out.append(
        f"  gas / brake     both > 0 at {100 * float(np.mean((gas > 0) & (brake > 0))):.1f}%; "
        f"gas >= 0.99 at {100 * float(np.mean(gas >= 0.99)):.1f}%; brake > 0 at "
        f"{100 * float(np.mean(brake > 0)):.1f}% of points"
    )
    return out


def _dump_grid(line: AiLine) -> list[str]:
    out = ["", "Grid"]
    grid = line.grid
    if grid is None:
        out.append("  present         no (has_grid = 0 or no flag)")
    elif isinstance(grid, bytes):
        out.append(f"  present         yes, NOT parsed: {len(grid):,} raw bytes kept")
    else:
        nx, nz = grid.shape
        sizes = [c.size for row in grid.cells for c in row]
        flat = [c for row in grid.cells for c in row if c.size]
        lo = np.concatenate(flat).min() if flat else -1
        hi = np.concatenate(flat).max() if flat else -1
        extent = grid.max_extreme.astype(np.float64) - grid.min_extreme
        d = grid.sampling_density
        pos = line.positions.astype(np.float64)
        pad_lo = pos.min(axis=0) - grid.min_extreme
        pad_hi = grid.max_extreme - pos.max(axis=0)
        out += [
            "  present         yes",
            f"  min_extreme     {np.round(grid.min_extreme, 3).tolist()}",
            f"  max_extreme     {np.round(grid.max_extreme, 3).tolist()}",
            f"  padding x / z   {_f(float(pad_lo[0]), 2)}, {_f(float(pad_hi[0]), 2)} / "
            f"{_f(float(pad_lo[2]), 2)}, {_f(float(pad_hi[2]), 2)} m beyond the points",
            f"  density         {_f(d)} m; neighbors_considered {grid.neighbors_considered}",
            f"  cells           {nx} x {nz} = {grid.cell_count} "
            f"(extent / density = {_f(extent[0] / d, 2)} x {_f(extent[2] / d, 2)})",
            f"  per cell        min {min(sizes, default=0)} max {max(sizes, default=0)}; "
            f"{grid.index_count} indices, range {lo}..{hi}",
        ]
        if line.n_points and grid.cell_count:
            out.append(f"  nearest check   {_grid_nearest_agreement(line, grid)}")
    out.append(f"  trailing bytes  {len(line.trailing)}")
    return out


def _grid_nearest_agreement(line: AiLine, grid: AiGrid, samples: int = 200) -> str:
    """How often a cell lists exactly the nearest points to its centre (OBSERVED convention)."""
    rng = np.random.default_rng(0)
    xz = line.positions[:, [0, 2]].astype(np.float64)
    cells = [(ix, iz) for ix, row in enumerate(grid.cells) for iz in range(len(row))]
    picks = rng.choice(len(cells), size=min(samples, len(cells)), replace=False)
    same = 0
    for p in picks:
        ix, iz = cells[int(p)]
        listed = grid.cells[ix][iz]
        if listed.size == 0:
            continue
        cx, cz = grid.cell_centre(ix, iz)
        dist = np.hypot(xz[:, 0] - cx, xz[:, 1] - cz)
        nearest = np.argsort(dist, kind="stable")[: listed.size]
        same += int(set(nearest.tolist()) == set(listed.tolist()))
    return f"{same}/{picks.size} sampled cells list exactly the nearest points to their centre"


__all__ = [
    "AI_VERSION",
    "AiGrid",
    "AiLine",
    "PAYLOAD_FIELDS",
    "ai_to_bytes",
    "build_ai_line",
    "build_grid",
    "dump",
    "lateral_speed_limit",
    "read_ai",
    "signed_curvature",
    "speed_profile",
    "write_ai",
    "write_csv",
]
