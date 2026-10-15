"""Centreline helpers: build, join, resample and frame polylines in AC world space.

Everything here works on polylines of shape (N, 3) in the AC frame of `driftmaps.ac.coords`:
X = east, Y = up, Z = south (so north = -Z).

Conventions:
- Headings are compass degrees, seen from above: 0 = north (-Z), 90 = east (+X), 180 = south,
  270 = west. `heading_vector(h)` = (sin h, 0, -cos h).
- Turns are signed as the driver sees them: positive = LEFT, negative = right. A left turn
  makes the compass heading go DOWN (north -> west), i.e. counter-clockwise seen from above,
  and bends the road towards the driver's left = up x forward (`coords.left_of`).
- A closed loop is given WITHOUT a repeated end point; `closed=True` adds the segment from the
  last point back to the first. Functions taking `closed=True` drop a repeated end point if the
  caller left one in.

These are the building blocks for the M0 test tracks (straights, arcs, a rounded-rectangle
loop) and for the real generator later (resampling and frames along a smoothed route).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from driftmaps.ac.coords import WORLD_UP

__all__ = [
    "Arc",
    "Straight",
    "angle_diff",
    "arc",
    "concat",
    "cumulative_length",
    "end_heading",
    "frames",
    "heading_of",
    "heading_vector",
    "headings",
    "left_vector",
    "path",
    "polyline_length",
    "resample",
    "rounded_rectangle_loop",
    "straight",
    "total_turn",
    "turn_angles",
]

# Two points closer than this (metres) count as the same point (duplicate joints, closing points).
JOINT_TOLERANCE = 1e-6


# --------------------------------------------------------------------------------------------
# Headings
# --------------------------------------------------------------------------------------------


def heading_vector(heading_deg: npt.ArrayLike) -> np.ndarray:
    """Unit horizontal direction (..., 3) for a compass heading in degrees (0 = north = -Z)."""
    h = np.radians(np.asarray(heading_deg, dtype=np.float64))
    return np.stack([np.sin(h), np.zeros_like(h), -np.cos(h)], axis=-1)


def left_vector(heading_deg: npt.ArrayLike) -> np.ndarray:
    """Unit horizontal vector (..., 3) to the driver's left when heading `heading_deg`.

    Equals `coords.left_of(heading_vector(h))` = up x forward, written out so it vectorises:
    up x (sin h, 0, -cos h) = (-cos h, 0, -sin h). Heading north, left is west (-X).
    """
    h = np.radians(np.asarray(heading_deg, dtype=np.float64))
    return np.stack([-np.cos(h), np.zeros_like(h), -np.sin(h)], axis=-1)


def heading_of(vector: npt.ArrayLike) -> np.ndarray | float:
    """Compass heading in degrees [0, 360) of a direction (..., 3); the Y part is ignored."""
    v = np.asarray(vector, dtype=np.float64)
    deg = np.degrees(np.arctan2(v[..., 0], -v[..., 2])) % 360.0
    # `% 360` can return 360.0 for tiny negative inputs; fold it back to 0.
    deg = np.where(deg >= 360.0, 0.0, deg)
    return float(deg) if deg.ndim == 0 else deg


def angle_diff(from_deg: npt.ArrayLike, to_deg: npt.ArrayLike) -> np.ndarray | float:
    """Signed smallest rotation from one compass heading to another, in [-180, 180).

    Positive means clockwise seen from above (a RIGHT turn); see `turn_angles` for the
    driver's sign convention.
    """
    d = np.asarray(to_deg, dtype=np.float64) - np.asarray(from_deg, dtype=np.float64) + 180.0
    d = d % 360.0 - 180.0
    return float(d) if np.ndim(d) == 0 else d


# --------------------------------------------------------------------------------------------
# Basic polyline measures
# --------------------------------------------------------------------------------------------


def _as_points(points: npt.ArrayLike, min_points: int = 1) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected points of shape (N, 3), got {pts.shape}")
    if pts.shape[0] < min_points:
        raise ValueError(f"need at least {min_points} points, got {pts.shape[0]}")
    if not np.isfinite(pts).all():
        raise ValueError("points contain NaN or infinite values")
    return pts


def _as_point(point: npt.ArrayLike) -> np.ndarray:
    p = np.asarray(point, dtype=np.float64)
    if p.shape != (3,):
        raise ValueError(f"expected a point of shape (3,), got {p.shape}")
    return p.copy()


def _strip_closing_point(pts: np.ndarray) -> np.ndarray:
    """Drop a repeated end point from a closed loop (first == last)."""
    if pts.shape[0] > 1 and np.linalg.norm(pts[-1] - pts[0]) <= JOINT_TOLERANCE:
        return pts[:-1]
    return pts


def _segments(pts: np.ndarray, closed: bool) -> np.ndarray:
    """Segment vectors (N-1, 3), or (N, 3) including last -> first when closed."""
    if closed:
        return np.roll(pts, -1, axis=0) - pts
    return np.diff(pts, axis=0)


def _segment_count(length: float, step: float) -> int:
    """Segments needed so each is at most `step` long (with a little float slack)."""
    if not step > 0.0:
        raise ValueError(f"step must be > 0, got {step}")
    return max(1, math.ceil(length / step - 1e-9))


def cumulative_length(points: npt.ArrayLike) -> np.ndarray:
    """Distance along the polyline from the first point to each point, shape (N,)."""
    pts = _as_points(points)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def polyline_length(points: npt.ArrayLike, closed: bool = False) -> float:
    """Total length in metres; with `closed=True` the closing segment is included."""
    pts = _as_points(points)
    if closed:
        pts = _strip_closing_point(pts)
    return float(np.linalg.norm(_segments(pts, closed), axis=1).sum())


# --------------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------------


def straight(
    start: npt.ArrayLike,
    heading_deg: float,
    length: float,
    step: float = 1.0,
    rise: float = 0.0,
) -> np.ndarray:
    """Points along a straight line, start included, horizontal spacing <= `step`.

    `length` and `step` are measured horizontally (on the map). `rise` is the height gained
    over the segment (linear grade); the end point is exactly
    `start + length * heading_vector(heading_deg) + (0, rise, 0)`.
    """
    p0 = _as_point(start)
    if not length > 0.0:
        raise ValueError(f"length must be > 0, got {length}")
    n = _segment_count(length, step)
    t = np.linspace(0.0, 1.0, n + 1)
    pts = p0 + np.outer(t * length, heading_vector(heading_deg))
    pts[:, 1] = p0[1] + rise * t
    pts[0] = p0
    return pts


def arc(
    start: npt.ArrayLike,
    heading_deg: float,
    radius: float,
    turn_deg: float,
    step: float = 1.0,
    rise: float = 0.0,
) -> np.ndarray:
    """Points along a circular arc, start included, horizontal spacing (along the arc) <= `step`.

    `turn_deg` > 0 turns LEFT (towards up x forward), < 0 turns right. The heading at the end
    is `heading_deg - turn_deg` (compass headings go down on a left turn). Seen from above the
    points lie exactly on the circle (chords <= `step`); `rise` is the height gained (a helix).
    Note the last chord's heading (`end_heading`) lags the exact end heading by half a chord
    angle; `path` carries the exact heading.
    """
    p0 = _as_point(start)
    if not radius > 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if turn_deg == 0.0 or not math.isfinite(turn_deg):
        raise ValueError(f"turn_deg must be finite and non-zero, got {turn_deg}")
    side = 1.0 if turn_deg > 0.0 else -1.0  # +1: centre on the driver's left
    length = radius * math.radians(abs(turn_deg))
    n = _segment_count(length, step)
    t = np.linspace(0.0, 1.0, n + 1)
    h = heading_deg - turn_deg * t
    centre = p0 + side * radius * left_vector(heading_deg)
    pts = centre - side * radius * left_vector(h)
    pts[:, 1] = p0[1] + rise * t
    pts[0] = p0
    return pts


def concat(*segments: npt.ArrayLike, tolerance: float = JOINT_TOLERANCE) -> np.ndarray:
    """Join polylines end to end, dropping a segment's first point if it repeats the joint.

    Segments are not required to touch; a gap is kept as a straight segment.
    """
    if not segments:
        raise ValueError("concat needs at least one segment")
    parts: list[np.ndarray] = []
    last: np.ndarray | None = None
    for seg in segments:
        pts = _as_points(seg)
        if last is not None and np.linalg.norm(pts[0] - last) <= tolerance:
            pts = pts[1:]
        if pts.shape[0]:
            parts.append(pts)
            last = pts[-1]
    return np.concatenate(parts, axis=0)


@dataclass(frozen=True)
class Straight:
    """A straight piece for `path`: `length` metres, gaining `rise` metres of height."""

    length: float
    rise: float = 0.0


@dataclass(frozen=True)
class Arc:
    """A circular piece for `path`. `turn_deg` > 0 = left. `radius` 0 = a sharp corner."""

    radius: float
    turn_deg: float
    rise: float = 0.0


def path(
    start: npt.ArrayLike,
    heading_deg: float,
    parts: Sequence[Straight | Arc],
    step: float = 1.0,
) -> np.ndarray:
    """Chain straights and arcs "turtle style" from `start`, heading `heading_deg`.

    Headings are carried analytically from piece to piece, so long paths don't drift.
    Zero-length straights are skipped; an `Arc` with radius 0 turns on the spot (a kink).

    Example, an L-shaped road that turns left: heading north, then west.
        path((0, 0, 0), 0, [Straight(100), Arc(25, 90), Straight(100)])
    """
    pos = _as_point(start)
    heading = float(heading_deg)
    pieces: list[np.ndarray] = [pos[None, :]]
    for part in parts:
        if isinstance(part, Straight):
            if part.length == 0.0:
                continue
            seg = straight(pos, heading, part.length, step, part.rise)
        elif isinstance(part, Arc):
            if part.radius == 0.0:
                heading -= part.turn_deg
                continue
            seg = arc(pos, heading, part.radius, part.turn_deg, step, part.rise)
            heading -= part.turn_deg
        else:
            raise TypeError(f"unknown path part {part!r}")
        pieces.append(seg)
        pos = seg[-1].copy()
    return concat(*pieces)


def rounded_rectangle_loop(
    width: float,
    height: float,
    corner_radius: float,
    step: float = 1.0,
    centre: npt.ArrayLike = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """A closed loop (no repeated end point) round a `width` (east-west, X) by `height`
    (north-south, Z) rectangle with rounded corners, at constant Y.

    Driving direction is counter-clockwise seen from above, so every corner is a LEFT turn.
    The loop starts in the middle of the south edge heading east. `corner_radius` 0 gives
    sharp corners.
    """
    c = _as_point(centre)
    r = float(corner_radius)
    if not (width > 0.0 and height > 0.0):
        raise ValueError(f"width and height must be > 0, got {width}, {height}")
    if not 0.0 <= r <= min(width, height) / 2.0:
        raise ValueError(f"corner_radius must be in [0, {min(width, height) / 2}], got {r}")
    start = c + np.array([0.0, 0.0, height / 2.0])  # middle of the south edge (+Z = south)
    corner = Arc(r, 90.0)
    parts = [
        Straight(width / 2.0 - r),
        corner,
        Straight(height - 2.0 * r),
        corner,
        Straight(width - 2.0 * r),
        corner,
        Straight(height - 2.0 * r),
        corner,
        Straight(width / 2.0 - r),
    ]
    pts = path(start, 90.0, parts, step)
    return _strip_closing_point(pts)


def resample(points: npt.ArrayLike, step: float, closed: bool = False) -> np.ndarray:
    """Resample at uniform arc-length spacing (linear interpolation between input points).

    Spacing is `L / ceil(L / step)` (so never more than `step`). Open polylines keep both end
    points exactly; closed loops start at the first point and return no repeated end point.
    """
    pts = _as_points(points, min_points=2)
    if closed:
        pts = _strip_closing_point(pts)
        pts = np.concatenate([pts, pts[:1]], axis=0)
    s = cumulative_length(pts)
    total = float(s[-1])
    if total <= 0.0:
        raise ValueError("polyline has zero length")
    n = _segment_count(total, step)
    targets = np.linspace(0.0, total, n + 1)
    out = np.stack([np.interp(targets, s, pts[:, k]) for k in range(3)], axis=1)
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out[:-1] if closed else out


# --------------------------------------------------------------------------------------------
# Frames, headings and turns along a polyline
# --------------------------------------------------------------------------------------------


def frames(
    points: npt.ArrayLike, closed: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-point (tangents, lefts, ups), each (N, 3) of unit vectors.

    - tangent: the bisector of the incoming and outgoing segment directions (the segment
      direction at open ends);
    - left: the driver's left, `up x tangent` with up = world up (the same rule as
      `coords.left_of`, vectorised). It is always horizontal;
    - up: `tangent x left`, the road's up for an unbanked road (tilted on a grade).

    Raises ValueError on repeated consecutive points or a vertical tangent.
    """
    pts = _as_points(points, min_points=2)
    if closed:
        pts = _strip_closing_point(pts)
        if pts.shape[0] < 3:
            raise ValueError("a closed loop needs at least 3 distinct points")
    seg = _segments(pts, closed)
    seg_len = np.linalg.norm(seg, axis=1)
    if (seg_len <= JOINT_TOLERANCE).any():
        i = int(np.argmax(seg_len <= JOINT_TOLERANCE))
        raise ValueError(f"repeated consecutive points at index {i}; use concat/resample")
    unit = seg / seg_len[:, None]

    if closed:
        incoming = np.roll(unit, 1, axis=0)
        outgoing = unit
    else:
        incoming = np.concatenate([unit[:1], unit], axis=0)
        outgoing = np.concatenate([unit, unit[-1:]], axis=0)
    tangents = incoming + outgoing
    norm = np.linalg.norm(tangents, axis=1)
    reversal = norm < 1e-9  # the path doubles back on itself: use the outgoing direction
    tangents[reversal] = outgoing[reversal]
    norm[reversal] = 1.0
    tangents /= norm[:, None]

    lefts = np.cross(WORLD_UP, tangents)
    left_norm = np.linalg.norm(lefts, axis=1)
    if (left_norm < 1e-9).any():
        i = int(np.argmax(left_norm < 1e-9))
        raise ValueError(f"vertical tangent at index {i}: no left direction")
    lefts /= left_norm[:, None]

    ups = np.cross(tangents, lefts)
    ups /= np.linalg.norm(ups, axis=1)[:, None]
    return tangents, lefts, ups


def headings(points: npt.ArrayLike, closed: bool = False) -> np.ndarray:
    """Compass heading (degrees, [0, 360)) of the tangent at each point, shape (N,)."""
    tangents, _, _ = frames(points, closed)
    return np.asarray(heading_of(tangents))


def end_heading(points: npt.ArrayLike) -> float:
    """Compass heading of the last segment (where a following piece should continue)."""
    pts = _as_points(points, min_points=2)
    return float(heading_of(pts[-1] - pts[-2]))


def turn_angles(points: npt.ArrayLike, closed: bool = False) -> np.ndarray:
    """Signed turn at each point in degrees, seen by the driver: positive = LEFT.

    Measured in the horizontal plane between the incoming and outgoing segments. Open ends
    get 0. A full counter-clockwise loop (seen from above) sums to +360.
    """
    pts = _as_points(points, min_points=2)
    if closed:
        pts = _strip_closing_point(pts)
    seg = _segments(pts, closed)
    h = np.asarray(heading_of(seg))
    if closed:
        h_in = np.roll(h, 1)
        h_out = h
    else:
        h_in = np.concatenate([h[:1], h])
        h_out = np.concatenate([h, h[-1:]])
    # A left turn lowers the compass heading, so negate the clockwise difference.
    return -np.asarray(angle_diff(h_in, h_out))


def total_turn(points: npt.ArrayLike, closed: bool = False) -> float:
    """Sum of `turn_angles` in degrees (positive = net left)."""
    return float(turn_angles(points, closed).sum())
