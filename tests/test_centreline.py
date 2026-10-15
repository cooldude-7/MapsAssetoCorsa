"""Tests for driftmaps.geometry.centreline: headings, builders, resampling and frames.

The handedness checks matter most: in the AC frame (X = east, Y = up, Z = south) a car heading
north (-Z) that turns LEFT must end up heading west (-X), on the side `coords.left_of` gives.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from driftmaps.ac.coords import enu_to_ac, left_of
from driftmaps.geometry.centreline import (
    Arc,
    Straight,
    angle_diff,
    arc,
    concat,
    cumulative_length,
    end_heading,
    frames,
    heading_of,
    heading_vector,
    headings,
    left_vector,
    path,
    polyline_length,
    resample,
    rounded_rectangle_loop,
    straight,
    total_turn,
    turn_angles,
)

NORTH = np.array([0.0, 0.0, -1.0])
EAST = np.array([1.0, 0.0, 0.0])
SOUTH = np.array([0.0, 0.0, 1.0])
WEST = np.array([-1.0, 0.0, 0.0])


def shoelace_en(points: np.ndarray) -> float:
    """Signed area in the (east, north) plane: > 0 = counter-clockwise seen from above."""
    e = points[:, 0]
    n = -points[:, 2]
    return 0.5 * float(np.sum(e * np.roll(n, -1) - np.roll(e, -1) * n))


# --------------------------------------------------------------------------------------------
# Headings
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("heading", "expected"),
    [(0, NORTH), (90, EAST), (180, SOUTH), (270, WEST), (-90, WEST), (360, NORTH)],
)
def test_heading_vector_compass(heading: float, expected: np.ndarray) -> None:
    np.testing.assert_allclose(heading_vector(heading), expected, atol=1e-12)


def test_heading_vector_matches_enu() -> None:
    # Compass headings are defined on the map: north in ENU is (0, 1, 0).
    np.testing.assert_allclose(heading_vector(0), enu_to_ac([0.0, 1.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(heading_vector(90), enu_to_ac([1.0, 0.0, 0.0]), atol=1e-12)


@pytest.mark.parametrize("heading", [0.0, 10.0, 90.0, 135.0, 225.5, 359.0])
def test_heading_of_round_trip_and_left_vector(heading: float) -> None:
    v = heading_vector(heading)
    assert heading_of(v) == pytest.approx(heading, abs=1e-9)
    np.testing.assert_allclose(left_vector(heading), left_of(v), atol=1e-12)


def test_heading_helpers_vectorise() -> None:
    hs = np.array([0.0, 90.0, 180.0, 270.0])
    vs = heading_vector(hs)
    assert vs.shape == (4, 3)
    np.testing.assert_allclose(heading_of(vs), hs, atol=1e-9)
    assert left_vector(hs).shape == (4, 3)


def test_angle_diff() -> None:
    assert angle_diff(350, 10) == pytest.approx(20)
    assert angle_diff(10, 350) == pytest.approx(-20)
    assert angle_diff(0, 270) == pytest.approx(-90)


# --------------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------------


def test_straight_north() -> None:
    pts = straight((1.0, 2.0, 3.0), 0.0, 10.0, step=1.0)
    assert pts.shape == (11, 3)
    np.testing.assert_allclose(pts[-1], [1.0, 2.0, -7.0], atol=1e-12)
    np.testing.assert_allclose(np.diff(cumulative_length(pts)), 1.0)


def test_straight_spacing_never_exceeds_step_and_rise() -> None:
    pts = straight((0, 0, 0), 90.0, 10.5, step=2.0, rise=3.0)
    seg = np.linalg.norm(np.diff(pts[:, [0, 2]], axis=0), axis=1)
    assert seg.max() <= 2.0 + 1e-12
    np.testing.assert_allclose(pts[-1], [10.5, 3.0, 0.0], atol=1e-12)


def test_left_turn_from_north_heads_west() -> None:
    """The key handedness check: north + left turn = west, curving to the driver's left."""
    r = 20.0
    pts = arc((0.0, 0.0, 0.0), 0.0, r, 90.0, step=0.5)
    np.testing.assert_allclose(pts[-1], [-r, 0.0, -r], atol=1e-9)
    tangents, _, _ = frames(pts)
    np.testing.assert_allclose(tangents[-1], WEST, atol=0.02)
    assert headings(pts)[-1] == pytest.approx(270.0, abs=1.0)
    # Every point is on the driver's left (west) of the start, i.e. along up x forward.
    assert (pts[1:] @ left_of(NORTH) > 0.0).all()
    assert (pts[1:, 0] < 0.0).all()


def test_right_turn_from_north_heads_east() -> None:
    pts = arc((0.0, 0.0, 0.0), 0.0, 20.0, -90.0, step=0.5)
    np.testing.assert_allclose(pts[-1], [20.0, 0.0, -20.0], atol=1e-9)
    assert headings(pts)[-1] == pytest.approx(90.0, abs=1.0)
    assert (pts[1:, 0] > 0.0).all()


@pytest.mark.parametrize("heading", [0.0, 37.0, 90.0, 200.0])
@pytest.mark.parametrize("turn", [45.0, 90.0, -120.0, 270.0])
def test_arc_geometry(heading: float, turn: float) -> None:
    r = 15.0
    pts = arc((5.0, 1.0, -3.0), heading, r, turn, step=1.0)
    side = 1.0 if turn > 0 else -1.0
    centre = np.array([5.0, 1.0, -3.0]) + side * r * left_vector(heading)
    np.testing.assert_allclose(np.linalg.norm(pts - centre, axis=1), r, atol=1e-9)
    chords = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert chords.max() <= 1.0 + 1e-9
    # The turn is measured by the driver: positive = left.
    # Interior chord turns add up to the arc's turn minus one chord angle (half at each end).
    chord_deg = math.degrees(chords.max() / r)
    assert total_turn(pts) == pytest.approx(turn - math.copysign(chord_deg, turn), abs=0.1)
    exact_end = (heading - turn) % 360.0
    assert abs(angle_diff(headings(pts)[-1], exact_end)) < 2.0


def test_arc_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        arc((0, 0, 0), 0, 0.0, 90)
    with pytest.raises(ValueError):
        arc((0, 0, 0), 0, 10.0, 0.0)
    with pytest.raises(ValueError):
        straight((0, 0, 0), 0, -1.0)


def test_concat_drops_duplicate_joints() -> None:
    a = straight((0, 0, 0), 0, 10)
    b = arc(a[-1], 0, 10, 90)
    c = straight(b[-1], 270, 5)
    joined = concat(a, b, c)
    assert joined.shape[0] == len(a) + len(b) + len(c) - 2
    assert (np.linalg.norm(np.diff(joined, axis=0), axis=1) > 1e-6).all()
    # A real gap is kept.
    gap = concat(a, straight((5, 0, -20), 0, 3))
    assert gap.shape[0] == len(a) + 4


def test_path_l_road_turns_left_to_west() -> None:
    pts = path((0, 0, 0), 0.0, [Straight(100), Arc(25, 90), Straight(100)], step=1.0)
    np.testing.assert_allclose(pts[-1], [-125.0, 0.0, -125.0], atol=1e-9)
    assert end_heading(pts) == pytest.approx(270.0, abs=1e-9)
    assert total_turn(pts) == pytest.approx(90.0, abs=1e-6)
    assert (turn_angles(pts) >= -1e-9).all()


def test_path_rise_and_sharp_corner() -> None:
    pts = path((0, 0, 0), 90.0, [Straight(10, rise=1.0), Arc(0, 90), Straight(10)], step=1.0)
    np.testing.assert_allclose(pts[10], [10.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(pts[-1], [10.0, 1.0, -10.0], atol=1e-12)
    assert turn_angles(pts)[10] == pytest.approx(90.0)


def test_rounded_rectangle_loop_is_counter_clockwise() -> None:
    w, h, r = 120.0, 80.0, 20.0
    loop = rounded_rectangle_loop(w, h, r, step=1.0, centre=(10.0, 2.0, -5.0))
    # Closed, no repeated end point, every corner a left turn adding up to one full turn.
    assert np.linalg.norm(loop[0] - loop[-1]) > 0.5
    turns = turn_angles(loop, closed=True)
    assert (turns >= -1e-9).all()
    assert turns.sum() == pytest.approx(360.0, abs=1e-6)
    assert shoelace_en(loop) > 0.0
    assert shoelace_en(loop) == pytest.approx(w * h - (4 - math.pi) * r * r, rel=2e-3)
    assert polyline_length(loop, closed=True) == pytest.approx(
        2 * (w + h) - 8 * r + 2 * math.pi * r, rel=1e-3
    )
    np.testing.assert_allclose(loop[:, 1], 2.0)
    np.testing.assert_allclose(loop.min(axis=0)[[0, 2]], [10 - w / 2, -5 - h / 2], atol=1e-9)
    np.testing.assert_allclose(loop.max(axis=0)[[0, 2]], [10 + w / 2, -5 + h / 2], atol=1e-9)
    # Starts mid south edge heading east.
    np.testing.assert_allclose(loop[0], [10.0, 2.0, -5.0 + h / 2], atol=1e-12)
    assert headings(loop, closed=True)[0] == pytest.approx(90.0, abs=1e-9)


def test_rounded_rectangle_sharp_and_full_round() -> None:
    sharp = rounded_rectangle_loop(10.0, 6.0, 0.0, step=1.0)
    assert total_turn(sharp, closed=True) == pytest.approx(360.0)
    circle = rounded_rectangle_loop(20.0, 20.0, 10.0, step=0.5)
    np.testing.assert_allclose(np.linalg.norm(circle[:, [0, 2]], axis=1), 10.0, atol=1e-9)
    with pytest.raises(ValueError):
        rounded_rectangle_loop(10.0, 6.0, 4.0)


# --------------------------------------------------------------------------------------------
# Measures and resampling
# --------------------------------------------------------------------------------------------


def test_cumulative_and_polyline_length() -> None:
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=float)
    np.testing.assert_allclose(cumulative_length(square), [0, 1, 2, 3])
    assert polyline_length(square) == pytest.approx(3.0)
    assert polyline_length(square, closed=True) == pytest.approx(4.0)
    # A repeated closing point is ignored for closed loops.
    assert polyline_length(np.vstack([square, square[:1]]), closed=True) == pytest.approx(4.0)


def test_resample_open_uniform_and_keeps_ends() -> None:
    pts = path((0, 0, 0), 0, [Straight(10), Arc(10, 90), Straight(7.3)], step=3.0)
    out = resample(pts, 0.5)
    seg = np.linalg.norm(np.diff(out, axis=0), axis=1)
    np.testing.assert_allclose(out[0], pts[0])
    np.testing.assert_allclose(out[-1], pts[-1])
    assert seg.max() <= 0.5 + 1e-9
    # Uniform in arc length of the input polyline.
    s = cumulative_length(out)
    np.testing.assert_allclose(np.diff(s), s[-1] / (len(out) - 1), rtol=0.02)
    assert s[-1] == pytest.approx(polyline_length(pts), rel=1e-3)


def test_resample_closed_loop() -> None:
    loop = rounded_rectangle_loop(40, 30, 8, step=3.0)
    out = resample(loop, 1.0, closed=True)
    total = polyline_length(loop, closed=True)
    assert len(out) == math.ceil(total / 1.0 - 1e-9)
    seg = np.linalg.norm(np.diff(np.vstack([out, out[:1]]), axis=0), axis=1)
    assert seg.max() <= 1.0 + 1e-9
    np.testing.assert_allclose(out[0], loop[0])
    assert total_turn(out, closed=True) == pytest.approx(360.0, abs=1e-6)


# --------------------------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------------------------


def test_frames_north_left_is_west() -> None:
    pts = straight((0, 0, 0), 0, 5)
    t, left, up = frames(pts)
    np.testing.assert_allclose(t, np.tile(NORTH, (6, 1)), atol=1e-12)
    np.testing.assert_allclose(left, np.tile(WEST, (6, 1)), atol=1e-12)
    np.testing.assert_allclose(up, np.tile([0.0, 1.0, 0.0], (6, 1)), atol=1e-12)


def test_frames_match_coords_left_of_on_a_curvy_road() -> None:
    pts = path((3, 1, 2), 33.0, [Straight(20, rise=2), Arc(12, 120), Arc(30, -200, rise=-3)])
    t, left, up = frames(pts)
    for i in range(len(pts)):
        np.testing.assert_allclose(left[i], left_of(t[i]), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(up, axis=1), 1.0)
    np.testing.assert_allclose(left[:, 1], 0.0, atol=1e-12)  # left is horizontal
    np.testing.assert_allclose(np.sum(up * t, axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(up * left, axis=1), 0.0, atol=1e-12)
    assert (up[:, 1] > 0.9).all()
    # Right-handed (left, up, forward) like a dummy's (X, Y, Z) rows.
    np.testing.assert_allclose(np.cross(up, t), left, atol=1e-12)


def test_frames_closed_wraps() -> None:
    loop = rounded_rectangle_loop(20, 20, 0.0, step=10.0)  # a square
    t, left, _ = frames(loop, closed=True)
    # The loop starts mid south edge heading east; its first corner is the south-east one.
    corner = int(np.argmax(loop[:, 0]))
    np.testing.assert_allclose(t[corner], _unit([1.0, 0.0, -1.0]), atol=1e-12)
    np.testing.assert_allclose(t[0], EAST, atol=1e-12)
    # The left of a counter-clockwise loop points inside it.
    assert (np.sum(left * (0.0 - loop), axis=1) > 0).all()


def test_frames_rejects_repeated_points() -> None:
    with pytest.raises(ValueError, match="repeated"):
        frames(np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float))


def _unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    return a / np.linalg.norm(a)


# --------------------------------------------------------------------------------------------
# Loops
# --------------------------------------------------------------------------------------------


def _square_loop(start_corner: bool) -> tuple[np.ndarray, np.ndarray]:
    """A 200 m square round the origin, anticlockwise, starting on its south-west corner or in
    the middle of its south side; returns (xy, param) with the start repeated at the end."""
    corners = np.array([[-100.0, -100.0], [100.0, -100.0], [100.0, 100.0], [-100.0, 100.0]])
    if not start_corner:
        corners = np.vstack([[0.0, -100.0], corners[1:], corners[:1]])
    xy = np.vstack([corners, corners[:1]])
    return xy, np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])


@pytest.mark.parametrize("start_corner", [False, True])
def test_closed_centreline_has_no_seam(start_corner):
    from driftmaps.geometry.road import closed_centreline

    xy, param = _square_loop(start_corner)
    out, p = closed_centreline(xy, param, max_deviation_m=1.5, cutoff_m=40.0)
    np.testing.assert_array_equal(out[0], out[-1])
    assert p[0] == 0.0 and p[-1] == pytest.approx(800.0) and np.all(np.diff(p) >= 0)
    step = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert step.max() - step.min() < 0.01  # even stations right round, also over the join
    # The start is where the loop started (on a corner: the filleted corner's middle).
    start = xy[0]
    assert np.linalg.norm(out[0] - start) < (12.0 if start_corner else 0.5)
    # Heading turns smoothly everywhere, including across the join.
    tan = np.diff(np.vstack([out, out[1:2]]), axis=0)
    heading = np.unwrap(np.arctan2(tan[:, 1], tan[:, 0]))
    turn = np.abs(np.diff(heading))
    assert turn.max() < 0.2  # ~10 m corner radius, 1 m stations
    join = np.r_[turn[:5], turn[-5:]].max()
    assert join <= turn.max() + 1e-9 and (start_corner or join < 1e-3)
    assert heading[-1] - heading[0] == pytest.approx(2 * np.pi, abs=0.1)  # one full turn
    # Every corner is rounded the same: the join corner is no different.
    from scipy.spatial import cKDTree

    d, _ = cKDTree(out).query(np.array([[-100, -100], [100, -100], [100, 100], [-100, 100.0]]))
    assert d.max() - d.min() < 0.3


def test_closed_road_model_wraps():
    from driftmaps.geometry.road import RoadModel, closed_centreline, road_columns

    xy, param = _square_loop(True)
    out, _ = closed_centreline(xy, param)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(out, axis=0), axis=1))])
    centre = np.stack([out[:, 0], np.zeros(len(out)), out[:, 1]], axis=1)
    m = RoadModel(s, centre, 3.0, 1.0, True, False, closed=True)
    np.testing.assert_array_equal(m.forward[0], m.forward[-1])
    assert m.curvature[0] == pytest.approx(m.curvature[-1])
    for off in (4.0, -4.0, 9.0):
        col = road_columns(m, off)
        np.testing.assert_array_equal(col[0], col[-1])
    lap = m.length
    for a, b in ((m.at(10.0), m.at(10.0 + lap)), (m.at(-5.0), m.at(lap - 5.0))):
        for u, v in zip(a, b, strict=True):
            np.testing.assert_allclose(u, v, atol=1e-9)
    with pytest.raises(ValueError, match="first station"):
        RoadModel(s[:-5], centre[:-5], 3.0, 1.0, True, False, closed=True)
