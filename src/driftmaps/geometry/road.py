"""The real road: OSM polyline -> smooth centreline -> road ribbon, shoulders and painted lines.

Pipeline pieces (PLAN §5.3, docs/research/geo.md §3.2-§3.5, ac_format.md §3, §6):

- `fillet_corners`: circular fillets at sharp OSM corners (> 30 deg), radius limited to 45% of
  the shorter adjacent segment, so the smoother does not round a junction turn into a huge arc.
- `smooth_centreline`: densify to 2 m, Whittaker smoother on x(s), y(s) with a 3rd-difference
  (curvature-continuity) penalty, re-weighted until the curve stays within `max_deviation_m` of
  the OSM polyline, then resampled at 1 m. Each output point carries `param`, the distance along
  the original polyline it came from, so OSM attributes can be looked up per station.
- `closed_centreline`: the same for a loop (a circuit): the corners at the join are filleted
  like any other and the smoother is periodic, so there is no seam or kink where the lap
  starts; the stations start at the loop's first point and end on it again.
- `RoadModel`: per-station frame, height and cross-section in AC coordinates
  (X = east, Y = up, Z = south; left = up x forward). A `closed` model's last station is its
  first one again; frames, curvature and offset columns wrap around the join.
- `road_columns`: lateral offset polylines (road edges, shoulder edges, paint edges) with
  fold-over removed: on a tight inside curve an offset point that would move backwards is
  merged with its neighbours (geo.md §3.5), so no triangle ever flips.
- `build_road_meshes`: `1ROAD_c###` carriageway, `1GRAVEL_c###` shoulders (both sides in one
  node per chunk), and visual-only painted lines (`paint_white_c###`, `paint_yellow_c###`),
  chunked every ~250 m. Normals come from the road frame, so shading is continuous across chunks.

The cross-section is flat (no camber/superelevation in M1): every column of a station has the
centreline height. Painted lines float `PAINT_LIFT_M` above the asphalt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import shapely
from scipy.linalg import solveh_banded
from scipy.spatial import cKDTree

from driftmaps.ac.coords import WORLD_UP
from driftmaps.ac.model import MAX_MESH_VERTICES
from driftmaps.geometry.primitives import MeshData, split

__all__ = [
    "CENTRE_LINE_DASH_M",
    "CENTRE_LINE_GAP_M",
    "PAINT_LIFT_M",
    "RoadMeshes",
    "RoadModel",
    "build_road_meshes",
    "closed_centreline",
    "extend_straight",
    "fillet_corners",
    "fix_foldover",
    "footprint_polygon",
    "road_columns",
    "smooth_centreline",
    "whittaker",
    "whittaker_periodic",
]

# Painted lines (visual only) sit this far above the asphalt. ASSUMED enough to avoid
# z-fighting at AC's depth precision near the camera; M0 decals used 0.03 m.
PAINT_LIFT_M = 0.02
EDGE_LINE_WIDTH_M = 0.15
EDGE_LINE_INSET_M = 0.20  # edge line centre this far inside the asphalt edge
CENTRE_LINE_WIDTH_M = 0.12
DOUBLE_YELLOW_GAP_M = 0.12  # between the two yellow lines (MUTCD 4-6 in; ASSUMED)
# White dashed centre line outside North America: 3 m dash, 9 m gap (ASSUMED typical rural
# values; France uses 3 m / 10 m "T1", the UK 4 m / 5 m on rural single carriageways).
CENTRE_LINE_DASH_M = 3.0
CENTRE_LINE_GAP_M = 9.0

ROAD_U_TILE_M = 4.0
ROAD_V_TILE_M = 4.0
GRAVEL_TILE_M = 2.0


# --------------------------------------------------------------------------------------------
# Smoothing
# --------------------------------------------------------------------------------------------


def _difference_band(n: int, order: int) -> np.ndarray:
    """D^T D for the `order`-th difference matrix of size n, in upper banded storage
    (order + 1 rows) as `scipy.linalg.solveh_banded` wants it."""
    coeffs = np.array([math.comb(order, k) * (-1) ** (order - k) for k in range(order + 1)], float)
    m = n - order
    band = np.zeros((order + 1, n))
    # (D^T D)[i, i + d] = sum over rows r of D[r, i] * D[r, i + d].
    for d in range(order + 1):
        for k in range(order + 1 - d):
            # rows r with column i = r + k and i + d = r + k + d.
            vals = np.zeros(n)
            vals[k : k + m] += coeffs[k] * coeffs[k + d]
            band[order - d, d:] += vals[: n - d]
    return band


def whittaker(y: npt.ArrayLike, weights: npt.ArrayLike, lam: float, order: int = 3) -> np.ndarray:
    """Whittaker-Eilers smoother: argmin sum w (y - z)^2 + lam * sum (diff^order z)^2.

    `y` is (n,) or (n, k) (columns smoothed independently with the same weights). Banded
    Cholesky, O(n). Needs at least `order + 1` points with positive weight.
    """
    y = np.asarray(y, np.float64)
    w = np.asarray(weights, np.float64)
    n = y.shape[0]
    if n <= order:
        return y.copy()
    band = lam * _difference_band(n, order)
    band[order] += w
    return solveh_banded(band, (w[:, None] * y.reshape(n, -1)), check_finite=False).reshape(y.shape)


def whittaker_periodic(
    y: npt.ArrayLike, weights: npt.ArrayLike, lam: float, order: int = 3
) -> np.ndarray:
    """`whittaker` on a closed curve: the differences wrap around (point n-1 is followed by
    point 0), so the result has no ends. `y` (n,) or (n, k), without a repeated first point."""
    from scipy import sparse
    from scipy.sparse.linalg import splu

    y = np.asarray(y, np.float64)
    w = np.asarray(weights, np.float64)
    n = y.shape[0]
    if n <= order + 1:
        return y.copy()
    coeffs = np.array([math.comb(order, k) * (-1) ** (order - k) for k in range(order + 1)], float)
    rows = np.repeat(np.arange(n), order + 1)
    cols = ((np.arange(n)[:, None] + np.arange(order + 1)[None, :]) % n).ravel()
    d = sparse.csr_array((np.tile(coeffs, n), (rows, cols)), shape=(n, n))
    a = sparse.diags_array(w) + lam * (d.T @ d)
    solved = splu(sparse.csc_array(a)).solve(w[:, None] * y.reshape(n, -1))
    return solved.reshape(y.shape)


def _cumlen(xy: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])


def _dedupe(xy: np.ndarray, param: np.ndarray, tol: float = 1e-6):
    keep = np.concatenate([[True], np.linalg.norm(np.diff(xy, axis=0), axis=1) > tol])
    return xy[keep], param[keep]


def _resample(xy: np.ndarray, param: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Uniform arc-length resampling (spacing <= step), carrying `param` along linearly."""
    s = _cumlen(xy)
    n = max(1, math.ceil(s[-1] / step - 1e-9))
    t = np.linspace(0.0, s[-1], n + 1)
    out = np.stack([np.interp(t, s, xy[:, k]) for k in range(2)], axis=1)
    return out, np.interp(t, s, param)


def simplify(
    xy: npt.ArrayLike, param: npt.ArrayLike, tolerance_m: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Ramer-Douglas-Peucker: drop vertices within `tolerance_m` of the simplified line.

    OSM junctions often split a straight road into short pieces; merging them first lets
    `fillet_corners` use the real leg lengths for the corner radius. Kept vertices keep their
    `param`; the end points are always kept.
    """
    xy = np.asarray(xy, np.float64)
    param = np.asarray(param, np.float64)
    keep = np.zeros(len(xy), bool)
    keep[[0, -1]] = True
    stack = [(0, len(xy) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        ab = xy[b] - xy[a]
        n = np.linalg.norm(ab)
        rel = xy[a + 1 : b] - xy[a]
        if n < 1e-9:
            d = np.linalg.norm(rel, axis=1)
        else:
            d = np.abs(rel[:, 0] * ab[1] - rel[:, 1] * ab[0]) / n
        k = int(np.argmax(d))
        if d[k] > tolerance_m:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return xy[keep], param[keep]


def fillet_corners(
    xy: npt.ArrayLike,
    param: npt.ArrayLike,
    *,
    min_turn_deg: float = 30.0,
    radius_m: float = 12.0,
    max_fraction: float = 0.45,
    step_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace every corner sharper than `min_turn_deg` with a circular arc.

    Radius `radius_m`, reduced so the tangent length stays within `max_fraction` of each
    adjacent segment (geo.md §3.2). Returns the new polyline and its `param` (the original
    distance along the input, interpolated across each fillet).
    """
    xy, param = _dedupe(np.asarray(xy, np.float64), np.asarray(param, np.float64))
    if len(xy) < 3:
        return xy, param
    seg = np.diff(xy, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    unit = seg / seg_len[:, None]
    out_xy: list[np.ndarray] = [xy[:1]]
    out_p: list[np.ndarray] = [param[:1]]
    # Tangent lengths already used at the start of each segment (by the previous fillet).
    used = np.zeros(len(seg))
    for i in range(1, len(xy) - 1):
        a, b = unit[i - 1], unit[i]
        cos = float(np.clip(a @ b, -1.0, 1.0))
        theta = math.acos(cos)
        if math.degrees(theta) < min_turn_deg or theta > math.radians(175.0):
            out_xy.append(xy[i : i + 1])
            out_p.append(param[i : i + 1])
            continue
        half = math.tan(theta / 2.0)
        t_max = max_fraction * min(seg_len[i - 1] - used[i - 1], seg_len[i])
        t = min(radius_m * half, max(t_max, 0.0))
        if t < 0.05:
            out_xy.append(xy[i : i + 1])
            out_p.append(param[i : i + 1])
            continue
        used[i] = t
        r = t / half
        p0 = xy[i] - a * t
        p1 = xy[i] + b * t
        cross = a[0] * b[1] - a[1] * b[0]
        normal = np.array([-a[1], a[0]]) * (1.0 if cross > 0 else -1.0)  # towards the centre
        centre = p0 + normal * r
        a0 = math.atan2(*(p0 - centre)[::-1])
        a1 = math.atan2(*(p1 - centre)[::-1])
        sweep = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        n = max(2, math.ceil(abs(sweep) * r / step_m))
        ang = a0 + sweep * np.linspace(0.0, 1.0, n + 1)
        arc = centre + r * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        pa = param[i] - t * (param[i] - param[i - 1]) / seg_len[i - 1]
        pb = param[i] + t * (param[i + 1] - param[i]) / seg_len[i]
        out_xy.append(arc)
        out_p.append(np.linspace(pa, pb, n + 1))
    out_xy.append(xy[-1:])
    out_p.append(param[-1:])
    return _dedupe(np.concatenate(out_xy), np.concatenate(out_p))


def extend_straight(
    xy: npt.ArrayLike, param: npt.ArrayLike, before_m: float, after_m: float, step_m: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Extend a polyline straight along its end directions; extension points get `param`
    beyond the ends (negative before the start, > end after it)."""
    xy = np.asarray(xy, np.float64)
    param = np.asarray(param, np.float64)
    parts_xy, parts_p = [], []
    if before_m > 0:
        d = xy[0] - xy[1]
        d = d / np.linalg.norm(d)
        k = np.arange(math.ceil(before_m / step_m), 0, -1) * (
            before_m / math.ceil(before_m / step_m)
        )
        parts_xy.append(xy[0] + k[:, None] * d)
        parts_p.append(param[0] - k)
    parts_xy.append(xy)
    parts_p.append(param)
    if after_m > 0:
        d = xy[-1] - xy[-2]
        d = d / np.linalg.norm(d)
        n = math.ceil(after_m / step_m)
        k = np.arange(1, n + 1) * (after_m / n)
        parts_xy.append(xy[-1] + k[:, None] * d)
        parts_p.append(param[-1] + k)
    return np.concatenate(parts_xy), np.concatenate(parts_p)


def smooth_centreline(
    xy: npt.ArrayLike,
    param: npt.ArrayLike,
    *,
    max_deviation_m: float = 1.5,
    cutoff_m: float = 40.0,
    spacing_m: float = 2.0,
    step_m: float = 1.0,
    iterations: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Curvature-regularised smoothing of a road polyline (geo.md §3.2 step 3).

    The polyline is resampled at `spacing_m`, smoothed with a 3rd-difference Whittaker
    smoother whose cut-off wavelength is ~`cutoff_m`, and points that end up more than 80% of
    `max_deviation_m` from the input get their weight raised (and their neighbours'), until
    every point is within `max_deviation_m`. The end points are pinned. Returns the curve
    resampled at `step_m` and its `param` (see module docstring).
    """
    xy, param = _dedupe(np.asarray(xy, np.float64), np.asarray(param, np.float64))
    dense, dparam = _resample(xy, param, spacing_m)
    fine, _ = _resample(xy, param, min(0.25, spacing_m / 4))
    tree = cKDTree(fine)
    n = len(dense)
    h = _cumlen(dense)[-1] / max(n - 1, 1)
    # Frequency response 1 / (1 + lam * (2 sin(w/2))^6): cut-off where lam * w^6 = 1.
    omega = 2.0 * math.pi * h / cutoff_m
    lam = omega**-6
    w = np.ones(n)
    w[[0, -1]] = 1e6
    smooth = dense
    for _ in range(iterations):
        smooth = whittaker(dense, w, lam)
        dev, _ = tree.query(smooth)
        over = dev > 0.8 * max_deviation_m
        if dev.max() <= max_deviation_m:
            break
        # Raise weights where the curve strays, spread over +-5 points to avoid kinks.
        bump = np.convolve(over.astype(float), np.ones(11), mode="same") > 0
        w = np.where(bump, w * 4.0, w)
    out, oparam = _resample(smooth, dparam, step_m)
    return out, oparam


def closed_centreline(
    xy: npt.ArrayLike,
    param: npt.ArrayLike,
    *,
    max_deviation_m: float = 1.5,
    cutoff_m: float = 40.0,
    simplify_m: float = 0.3,
    spacing_m: float = 2.0,
    step_m: float = 1.0,
    iterations: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """`simplify` + `fillet_corners` + `smooth_centreline` for a loop.

    `xy` (M, 2) goes round the loop from its start and back to it (last point == first);
    `param` is the distance along it (0 .. L). The loop is opened in the middle of its longest
    segment (no corner there), simplified and filleted, then smoothed with the periodic
    Whittaker smoother (same cut-off and deviation rule as `smooth_centreline`). Returns
    stations every ~`step_m` from the start round to the start again (the last point equals
    the first exactly) and their `param`, rising from 0 to L.
    """
    xy, param = _dedupe(np.asarray(xy, np.float64), np.asarray(param, np.float64))
    if np.linalg.norm(xy[-1] - xy[0]) > 1e-6:
        raise ValueError("a closed centreline must end where it starts")
    total = float(param[-1])
    # Open the loop in the middle of its longest segment: [mid, ..., start, ..., mid].
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    k = int(np.argmax(seg))
    mid, pm = 0.5 * (xy[k] + xy[k + 1]), 0.5 * (param[k] + param[k + 1])
    ring, ring_p = xy[:-1], param[:-1]  # without the repeated start
    order = np.r_[np.arange(k + 1, len(ring)), np.arange(0, k + 1)]
    unwrap = np.where(order <= k, ring_p[order] + total, ring_p[order])
    open_xy = np.vstack([mid, ring[order], mid])
    open_p = np.concatenate([[pm], unwrap, [pm + total]])
    f_xy, f_p = fillet_corners(*simplify(open_xy, open_p, simplify_m))

    # Periodic smoothing of the (closed) filleted line, resampled at `spacing_m`.
    dense, dparam = _resample(f_xy, f_p, spacing_m)  # first == last (both `mid`)
    fine, _ = _resample(f_xy, f_p, min(0.25, spacing_m / 4))
    tree = cKDTree(fine)
    pts, pts_p = dense[:-1], dparam[:-1]
    n = len(pts)
    h = _cumlen(dense)[-1] / max(n, 1)
    lam = (2.0 * math.pi * h / cutoff_m) ** -6
    w = np.ones(n)
    smooth = pts
    for _ in range(iterations):
        smooth = whittaker_periodic(pts, w, lam)
        dev, _ = tree.query(smooth)
        if dev.max() <= max_deviation_m:
            break
        over = (dev > 0.8 * max_deviation_m).astype(float)
        bump = np.convolve(np.r_[over[-5:], over, over[:5]], np.ones(11), mode="valid") > 0
        w = np.where(bump, w * 4.0, w)

    # Resample from the start (param = total, where the loop was opened) once round.
    curve = np.vstack([smooth, smooth[:1]])
    cp = np.concatenate([pts_p, [pts_p[0] + total]])
    sc = _cumlen(curve)
    lap = sc[-1]
    s2 = np.concatenate([sc, sc[1:] + lap])
    xy2 = np.vstack([curve, curve[1:]])
    p2 = np.concatenate([cp, cp[1:] + total])
    s_start = float(np.interp(total, cp, sc))
    count = max(3, math.ceil(lap / step_m - 1e-9))
    t = s_start + np.linspace(0.0, lap, count + 1)
    out = np.stack([np.interp(t, s2, xy2[:, c]) for c in range(2)], axis=1)
    out[-1] = out[0]
    oparam = np.interp(t, s2, p2) - total
    oparam[0], oparam[-1] = 0.0, total
    return out, np.clip(np.maximum.accumulate(oparam), 0.0, total)


# --------------------------------------------------------------------------------------------
# Road model
# --------------------------------------------------------------------------------------------


@dataclass
class RoadModel:
    """Per-station road description in AC world coordinates, in driving order.

    `centre` (N, 3); `forward` (N, 3) unit 3D tangents (with grade); `left` (N, 3) horizontal
    unit driver's-left vectors; `up` (N, 3) road normals. `half_width` is the asphalt half
    width, `shoulder` the gravel width beyond it on each side. `centre_line` says whether a
    centre line is painted at the station; `double_yellow` picks the North American style.
    `bridge` marks stations on a bridge deck. `s` is the arc length from the first station.
    `closed`: a loop whose last station is the first one again (same position and frame);
    `at` and `interp` then take any `s` modulo the lap length.
    """

    s: np.ndarray
    centre: np.ndarray
    half_width: np.ndarray
    shoulder: np.ndarray
    centre_line: np.ndarray
    bridge: np.ndarray
    double_yellow: bool = True
    closed: bool = False
    forward: np.ndarray = field(init=False)
    left: np.ndarray = field(init=False)
    up: np.ndarray = field(init=False)
    curvature: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.s = np.asarray(self.s, np.float64)
        self.centre = np.asarray(self.centre, np.float64)
        n = len(self.s)
        self.half_width = np.broadcast_to(np.asarray(self.half_width, float), (n,)).copy()
        self.shoulder = np.broadcast_to(np.asarray(self.shoulder, float), (n,)).copy()
        self.centre_line = np.broadcast_to(np.asarray(self.centre_line, bool), (n,)).copy()
        self.bridge = np.broadcast_to(np.asarray(self.bridge, bool), (n,)).copy()
        seg = np.diff(self.centre, axis=0)
        seg /= np.linalg.norm(seg, axis=1)[:, None]
        if self.closed:
            if n < 4 or np.linalg.norm(self.centre[-1] - self.centre[0]) > 1e-6:
                raise ValueError("a closed road model must end on its first station")
            self.centre[-1] = self.centre[0]
            for a in (self.half_width, self.shoulder, self.centre_line, self.bridge):
                a[-1] = a[0]
            ends = seg[-1:] + seg[:1]
            fwd = np.concatenate([ends, seg[:-1] + seg[1:], ends])
        else:
            fwd = np.concatenate([seg[:1], seg[:-1] + seg[1:], seg[-1:]])
        fwd /= np.linalg.norm(fwd, axis=1)[:, None]
        left = np.cross(WORLD_UP, fwd)
        left /= np.linalg.norm(left, axis=1)[:, None]
        up = np.cross(fwd, left)
        self.forward = fwd
        self.left = left
        self.up = up / np.linalg.norm(up, axis=1)[:, None]
        # Signed horizontal curvature (1/m, + = turning left): d(forward)/ds along left.
        if self.closed:  # central differences round the join
            lap = self.s[-1] - self.s[0]
            s_ring, f_ring = self.s[:-1], self.forward[:-1]
            ds = (np.roll(s_ring, -1) - np.roll(s_ring, 1)) % lap
            dfwd = (np.roll(f_ring, -1, axis=0) - np.roll(f_ring, 1, axis=0)) / ds[:, None]
            dfwd = np.vstack([dfwd, dfwd[:1]])
        else:
            ds = np.gradient(self.s) if n > 1 else np.ones(n)
            dfwd = np.gradient(self.forward, axis=0) / ds[:, None]
        self.curvature = np.einsum("ij,ij->i", dfwd, self.left)

    def __len__(self) -> int:
        return len(self.s)

    @property
    def outer(self) -> np.ndarray:
        """Asphalt half width + shoulder: the edge of the drivable footprint."""
        return self.half_width + self.shoulder

    @property
    def length(self) -> float:
        return float(self.s[-1] - self.s[0])

    def wrap(self, s: float) -> float:
        """`s` itself, or on a closed model `s` brought into [s0, s0 + lap)."""
        if not self.closed:
            return s
        return float(self.s[0] + (s - self.s[0]) % self.length)

    def index_at(self, s: float) -> int:
        return int(np.clip(np.searchsorted(self.s, self.wrap(s)), 0, len(self.s) - 1))

    def at(self, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(position, forward, left) at arc length `s` (linear between stations)."""
        s = self.wrap(s)
        i = int(np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.s) - 2))
        t = float(np.clip((s - self.s[i]) / (self.s[i + 1] - self.s[i]), 0.0, 1.0))
        pos = self.centre[i] + t * (self.centre[i + 1] - self.centre[i])
        fwd = (1 - t) * self.forward[i] + t * self.forward[i + 1]
        fwd /= np.linalg.norm(fwd)
        left = np.cross(WORLD_UP, fwd)
        return pos, fwd, left / np.linalg.norm(left)

    def interp(self, values: np.ndarray, s: float) -> float:
        return float(np.interp(self.wrap(s), self.s, values))


def fix_foldover(points: np.ndarray, tangents: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Merge offset points that would run backwards along the road (geo.md §3.5).

    `points` (N, 3) is one lateral offset of the centreline, `tangents` (N, 3) the centreline
    directions. Wherever a step between consecutive points goes against the local direction of
    travel (in the horizontal plane), the points on both sides are merged into their mean, and
    this repeats until every step goes forward or is zero. Runs of merged points share one
    position, so the ribbon fans around it instead of folding over.
    """
    pts = np.asarray(points, np.float64)
    n = len(pts)
    labels = np.arange(n)
    tan = tangents[:, [0, 2]]
    for _ in range(n):
        starts = np.flatnonzero(np.concatenate([[True], labels[1:] != labels[:-1]]))
        counts = np.diff(np.append(starts, n))
        means = np.add.reduceat(pts, starts, axis=0) / counts[:, None]
        if len(starts) < 2:
            break
        step = means[1:, [0, 2]] - means[:-1, [0, 2]]
        # Direction at the boundary between group g and g + 1.
        tb = tan[starts[1:] - 1] + tan[starts[1:]]
        bad = np.einsum("ij,ij->i", step, tb) < -eps
        if not bad.any():
            return means[np.repeat(np.arange(len(starts)), counts)]
        group_id = np.concatenate([[0], np.cumsum(~bad)])
        labels = group_id[np.repeat(np.arange(len(starts)), counts)]
    return pts


# Offsets towards the inside of a curve are clamped to this fraction of the local radius.
INSIDE_RADIUS_FRACTION = 0.9


def _running_max(values: np.ndarray, half_window: int, closed: bool = False) -> np.ndarray:
    if closed:  # the last value repeats the first
        out = _running_max_ring(values[:-1], half_window)
        return np.append(out, out[:1])
    padded = np.pad(values, half_window, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half_window + 1)
    return windows.max(axis=1)


def _running_max_ring(values: np.ndarray, half_window: int) -> np.ndarray:
    padded = np.pad(values, half_window, mode="wrap")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half_window + 1)
    return windows.max(axis=1)


# Stations of context on each side of the join when fold-over is fixed on a closed model.
_CLOSED_PAD = 64


def road_columns(model: RoadModel, offsets: npt.ArrayLike) -> np.ndarray:
    """Positions (N, 3) at lateral `offsets` (scalar or (N,), + = left), never folding over.

    Offsets towards the inside of a curve are first clamped to 90% of the local radius (the
    tightest within +-5 stations), so no column crosses the centre of curvature; any step that
    still runs backwards is then merged away by `fix_foldover`. Columns clamp to the same
    curve, so neighbouring columns can coincide but never cross.
    """
    off = np.broadcast_to(np.asarray(offsets, np.float64), (len(model),)).copy()
    if not np.any(off):
        return model.centre.copy()
    closed = model.closed
    k_left = _running_max(np.maximum(model.curvature, 0.0), 5, closed)
    k_right = _running_max(np.maximum(-model.curvature, 0.0), 5, closed)
    with np.errstate(divide="ignore"):
        lim_left = np.where(k_left > 0, INSIDE_RADIUS_FRACTION / k_left, np.inf)
        lim_right = np.where(k_right > 0, INSIDE_RADIUS_FRACTION / k_right, np.inf)
    off = np.clip(off, -lim_right, lim_left)
    raw = model.centre + off[:, None] * model.left
    if not closed:
        return fix_foldover(raw, model.forward)
    # A loop: fix fold-over on the ring with stations from across the join on both sides, so
    # the start and the end see the same neighbours, then close it exactly.
    n = len(model) - 1
    pad = min(n, _CLOSED_PAD)
    idx = np.r_[np.arange(n - pad, n), np.arange(n), np.arange(pad)]
    ring = fix_foldover(raw[idx], model.forward[idx])[pad : pad + n]
    return np.vstack([ring, ring[:1]])


def footprint_polygon(
    left_edge: np.ndarray, right_edge: np.ndarray, mask: np.ndarray | None = None
) -> shapely.Geometry:
    """Union of the quads between two edge polylines (N, 3), in the XZ plane.

    `mask` (N - 1,) selects which station intervals to include (default all).
    """
    quads = np.stack(
        [left_edge[:-1], left_edge[1:], right_edge[1:], right_edge[:-1], left_edge[:-1]], axis=1
    )[..., [0, 2]]
    if mask is not None:
        quads = quads[np.asarray(mask, bool)]
    if len(quads) == 0:
        return shapely.Polygon()
    polys = shapely.polygons(quads)
    bad = ~shapely.is_valid(polys)
    if bad.any():
        polys[bad] = shapely.make_valid(polys[bad])
    polys = polys[shapely.area(polys) > 1e-9]
    return shapely.union_all(polys, grid_size=None)


# --------------------------------------------------------------------------------------------
# Meshes
# --------------------------------------------------------------------------------------------


@dataclass
class RoadMeshes:
    """Road mesh pieces per chunk (lists of the same length; None where a chunk has none)."""

    road: list[MeshData]
    gravel: list[MeshData]
    paint_white: list[MeshData | None]
    paint_yellow: list[MeshData | None]
    chunk_bounds: list[tuple[int, int]]
    left_outer: np.ndarray  # (N, 3) outer shoulder edge, fold-over fixed
    right_outer: np.ndarray

    @property
    def vertex_counts(self) -> dict[str, int]:
        def total(items: list) -> int:
            return sum(m.vertex_count for m in items if m is not None)

        return {
            "road": total(self.road),
            "gravel": total(self.gravel),
            "paint": total(self.paint_white) + total(self.paint_yellow),
        }


def chunk_ranges(s: np.ndarray, chunk_m: float) -> list[tuple[int, int]]:
    """Station index ranges [i0, i1] (inclusive, sharing end stations) of ~`chunk_m` metres."""
    n = len(s)
    count = max(1, round((s[-1] - s[0]) / chunk_m))
    cuts = np.searchsorted(s, np.linspace(s[0], s[-1], count + 1))
    cuts[0], cuts[-1] = 0, n - 1
    cuts = np.unique(cuts)
    return [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:], strict=True) if b > a]


def _strip(
    columns: list[np.ndarray],
    model: RoadModel,
    i0: int,
    i1: int,
    u: list[np.ndarray],
    v_tile: float,
    lift: float = 0.0,
    keep_quads: np.ndarray | None = None,
) -> MeshData | None:
    """A lattice between lateral columns (ordered left -> right) for stations i0..i1."""
    rows = slice(i0, i1 + 1)
    pos = np.stack([c[rows] for c in columns], axis=1)  # (R, K, 3)
    if lift:
        pos = pos + lift * WORLD_UP
    r, k = pos.shape[:2]
    s = model.s[rows]
    s0 = math.floor(s[0] / v_tile) * v_tile
    vv = np.broadcast_to((0.0 - (s - s0) / v_tile)[:, None], (r, k))
    uu = np.stack([x[rows] for x in u], axis=1)
    uvs = np.stack([uu, vv], axis=-1).reshape(-1, 2)
    normals = np.broadcast_to(model.up[rows][:, None, :], (r, k, 3)).reshape(-1, 3)
    tangents = np.broadcast_to(-model.left[rows][:, None, :], (r, k, 3)).reshape(-1, 3)
    ri, ci = np.meshgrid(np.arange(r - 1), np.arange(k - 1), indexing="ij")
    a = (ri * k + ci).reshape(-1)
    b, cc, d = a + 1, a + k + 1, a + k
    tris = np.stack([np.stack([a, b, cc], 1), np.stack([a, cc, d], 1)], axis=1)  # (Q, 2, 3)
    if keep_quads is not None:
        quad_ok = np.repeat(np.asarray(keep_quads[i0:i1], bool), k - 1)
        tris = tris[quad_ok]
    tris = tris.reshape(-1, 3)
    p = pos.reshape(-1, 3)
    area = 0.5 * np.linalg.norm(
        np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]]), axis=1
    )
    tris = tris[area > 1e-6]
    if len(tris) == 0:
        return None
    used = np.unique(tris)
    return MeshData(p[used], normals[used], uvs[used], tangents[used], np.searchsorted(used, tris))


def _lines(
    model: RoadModel, centres: list[np.ndarray], width: float, i0: int, i1: int, keep: np.ndarray
) -> list[MeshData]:
    out = []
    for c in centres:
        cols = [road_columns(model, c + width / 2), road_columns(model, c - width / 2)]
        u = [np.zeros(len(model)), np.ones(len(model))]
        m = _strip(cols, model, i0, i1, u, 1.0, PAINT_LIFT_M, keep)
        if m is not None:
            out.append(m)
    return out


def _merge(parts: list[MeshData]) -> MeshData | None:
    if not parts:
        return None
    offsets = np.cumsum([0] + [m.vertex_count for m in parts[:-1]])
    return MeshData(
        np.concatenate([m.positions for m in parts]),
        np.concatenate([m.normals for m in parts]),
        np.concatenate([m.uvs for m in parts]),
        np.concatenate([m.tangents for m in parts]),
        np.concatenate([m.indices + o for m, o in zip(parts, offsets, strict=True)]),
    )


def build_road_meshes(
    model: RoadModel,
    *,
    chunk_m: float = 250.0,
    lateral_segments: int = 4,
    edge_lines: bool = True,
    keep: object | None = None,
) -> RoadMeshes:
    """Asphalt, gravel shoulders and painted lines, chunked every ~`chunk_m` metres.

    `keep` (e.g. `geometry.network.RouteMasks`): per segment (N - 1,) boolean arrays
    `hidden` (no road, shoulders or lines: a roundabout replaces them), `shoulder_left`,
    `shoulder_right`, `edge_left`, `edge_right` and `centre` (True = draw), where junction
    mouths and islands take the shoulder's and the lines' place."""
    hw, outer = model.half_width, model.outer
    n_seg = len(model) - 1
    ones = np.ones(n_seg, bool)

    def mask(name: str) -> np.ndarray:
        value = getattr(keep, name, None) if keep is not None else None
        return ones if value is None else np.asarray(value, bool)

    hidden = getattr(keep, "hidden", None) if keep is not None else None
    shown = ones if hidden is None else ~np.asarray(hidden, bool)
    keep_sl, keep_sr = mask("shoulder_left") & shown, mask("shoulder_right") & shown
    keep_el, keep_er = mask("edge_left") & shown, mask("edge_right") & shown
    keep_c = mask("centre") & shown
    fracs = np.linspace(1.0, -1.0, lateral_segments + 1)
    road_cols = [road_columns(model, f * hw) for f in fracs]
    left_outer = road_columns(model, outer)
    right_outer = road_columns(model, -outer)
    road_u = [(hw - f * hw) / ROAD_U_TILE_M for f in fracs]
    gravel_u_l = [np.zeros(len(model)), model.shoulder / GRAVEL_TILE_M]
    gravel_u_r = gravel_u_l

    # Painted lines: solid white edge lines; centre line where `centre_line` (dashed white, or
    # double yellow in North America).
    edge_off = hw - EDGE_LINE_INSET_M
    centre_ok = model.centre_line
    seg_centre = centre_ok[:-1] & centre_ok[1:] & keep_c
    if model.double_yellow:
        g = DOUBLE_YELLOW_GAP_M / 2 + CENTRE_LINE_WIDTH_M / 2
        yellow_centres = [np.full(len(model), g), np.full(len(model), -g)]
        white_centre_keep = None
    else:
        yellow_centres = []
        period = CENTRE_LINE_DASH_M + CENTRE_LINE_GAP_M
        s_mid = 0.5 * (model.s[:-1] + model.s[1:])
        white_centre_keep = seg_centre & ((s_mid % period) < CENTRE_LINE_DASH_M)

    ranges = chunk_ranges(model.s, chunk_m)
    road, gravel, white, yellow = [], [], [], []
    for i0, i1 in ranges:
        rm = _strip(road_cols, model, i0, i1, road_u, ROAD_V_TILE_M, keep_quads=shown)
        gl = _strip([left_outer, road_cols[0]], model, i0, i1, gravel_u_l, GRAVEL_TILE_M,
                    keep_quads=keep_sl)  # fmt: skip
        gr = _strip([road_cols[-1], right_outer], model, i0, i1, gravel_u_r, GRAVEL_TILE_M,
                    keep_quads=keep_sr)  # fmt: skip
        road.append(rm)
        gravel.append(_merge([m for m in (gl, gr) if m is not None]))
        w_parts: list[MeshData] = []
        if edge_lines:
            w_parts += _lines(model, [edge_off], EDGE_LINE_WIDTH_M, i0, i1, keep_el)
            w_parts += _lines(model, [-edge_off], EDGE_LINE_WIDTH_M, i0, i1, keep_er)
        if white_centre_keep is not None:
            w_parts += _lines(
                model, [np.zeros(len(model))], CENTRE_LINE_WIDTH_M, i0, i1, white_centre_keep
            )
        white.append(_merge(w_parts))
        y_parts = (
            _lines(model, yellow_centres, CENTRE_LINE_WIDTH_M, i0, i1, seg_centre)
            if yellow_centres
            else []
        )
        yellow.append(_merge(y_parts))
    for name, items in (("road", road), ("gravel", gravel)):
        for m in items:
            # With masks a chunk may be all hidden or have no shoulder left (None).
            if (m is None and keep is None) or (
                m is not None and m.vertex_count > MAX_MESH_VERTICES
            ):
                raise ValueError(f"{name} chunk is empty or too large; use a smaller chunk_m")
    return RoadMeshes(road, gravel, white, yellow, ranges, left_outer, right_outer)


def split_all(items: list[MeshData | None]) -> list[list[MeshData]]:
    """`primitives.split` each piece (None -> [])."""
    return [split(m) if m is not None else [] for m in items]
