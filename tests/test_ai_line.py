"""Tests for `driftmaps.ac.ai_line`: fast_lane.ai v7 reader/writer, grid, dump and generator.

The real-file tests download a game-baked `fast_lane.ai` / `pit_lane.ai` (the `sepang` track in
github.com/619motorsports/newksEditor, pinned by commit and sha256) into the gitignored `.cache/`
at test time, and skip when offline. Nothing from a game install is committed.
"""

from __future__ import annotations

import csv
import hashlib
import http.client
import math
import struct
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from driftmaps.ac import ai_line as ai
from driftmaps.ac.ai_line import (
    AiGrid,
    AiLine,
    ai_to_bytes,
    build_ai_line,
    build_grid,
    dump,
    read_ai,
    signed_curvature,
    speed_profile,
    write_ai,
)
from driftmaps.ac.coords import WORLD_UP, left_of

G = ai.GRAVITY
REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / ".cache" / "ai_line"

# Field order of AcTools AiPointExtra.cs (and AssettoServer FastLaneUtils), checked by hand.
ACTOOLS_PAYLOAD_ORDER = (
    "speed",
    "gas",
    "brake",
    "obsolete_lat_g",
    "radius",
    "side_left",
    "side_right",
    "camber",
    "direction",
    "normal",
    "length",
    "forward",
    "tag",
    "grade",
)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def random_line(n: int, seed: int = 0, grid: AiGrid | bytes | None = None) -> AiLine:
    rng = np.random.default_rng(seed)

    def f(*shape: int) -> np.ndarray:
        return (rng.normal(size=shape) * 100).astype(np.float32)

    payload = {
        name: f(n, 3) if name in ("normal", "forward") else f(n) for name in ai.PAYLOAD_FIELDS
    }
    return AiLine(
        positions=f(n, 3),
        lengths=f(n),
        ids=rng.integers(-5, 10_000, n).astype(np.int32),
        **payload,
        lap_time=123,
        sample_count=-7,
        grid=grid,
    )


def raw(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a).tobytes()


def assert_identical(a: AiLine, b: AiLine) -> None:
    """Bit-for-bit comparison (NaN payloads and -0.0 included)."""
    for name in ("positions", "lengths", "ids", *ai.PAYLOAD_FIELDS):
        assert getattr(a, name).dtype == getattr(b, name).dtype, name
        assert raw(getattr(a, name)) == raw(getattr(b, name)), name
    assert (a.version, a.lap_time, a.sample_count) == (b.version, b.lap_time, b.sample_count)
    if isinstance(a.grid, AiGrid):
        assert isinstance(b.grid, AiGrid)
        assert a.grid.to_bytes() == b.grid.to_bytes()
    else:
        assert a.grid == b.grid
    assert a.trailing == b.trailing


def hand_pack(line: AiLine, grid_block: bytes | None = None) -> bytes:
    """Independent struct-based writer of the AcTools layout (the test oracle)."""
    n = line.n_points
    out = struct.pack("<4i", 7, n, line.lap_time, line.sample_count)
    for i in range(n):
        out += struct.pack("<4fi", *line.positions[i], line.lengths[i], line.ids[i])
    out += struct.pack("<i", n)
    for i in range(n):
        out += struct.pack(
            "<18f",
            line.speed[i],
            line.gas[i],
            line.brake[i],
            line.obsolete_lat_g[i],
            line.radius[i],
            line.side_left[i],
            line.side_right[i],
            line.camber[i],
            line.direction[i],
            *line.normal[i],
            line.length[i],
            *line.forward[i],
            line.tag[i],
            line.grade[i],
        )
    if grid_block is None:
        return out + struct.pack("<i", 0)
    return out + struct.pack("<i", 1) + grid_block


def hand_pack_grid(
    max_e: tuple[float, float, float],
    min_e: tuple[float, float, float],
    neighbors: int,
    density: float,
    cells: list[list[list[int]]],
) -> bytes:
    """AcTools AiSplineGrid.cs layout, written by hand."""
    out = struct.pack("<3f3fif", *max_e, *min_e, neighbors, density)
    out += struct.pack("<i", len(cells))
    for row in cells:
        out += struct.pack("<i", len(row))
        for cell in row:
            out += struct.pack(f"<i{len(cell)}i", len(cell), *cell)
    return out


def circle(radius: float, n: int = 720, *, ccw_on_map: bool = True) -> np.ndarray:
    """Circle about the origin. Counter-clockwise on a north-up map = continuous left turn."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    sign = 1.0 if ccw_on_map else -1.0
    # AC: x = east, z = south, so north = -z.
    return np.column_stack([radius * np.cos(t), np.zeros(n), -sign * radius * np.sin(t)])


def hairpin(radius: float = 15.0) -> np.ndarray:
    """300 m north, a left 180 degree hairpin of `radius`, then 200 m south."""
    approach = np.column_stack([np.zeros(301), np.zeros(301), -np.arange(301.0)])
    th = np.linspace(0.0, np.pi, 120)[1:]
    arc = np.column_stack(
        [-radius + radius * np.cos(th), np.zeros(th.size), -300.0 - radius * np.sin(th)]
    )
    exit_ = np.column_stack(
        [np.full(200, -2.0 * radius), np.zeros(200), -300.0 + np.arange(1.0, 201.0)]
    )
    return np.vstack([approach, arc, exit_])


# --------------------------------------------------------------------------------------------
# layout, reader and writer
# --------------------------------------------------------------------------------------------


def test_record_layout_matches_actools() -> None:
    assert ai.POINT_SIZE == 20
    assert ai.PAYLOAD_SIZE == 72
    assert ai.PAYLOAD_FIELDS == ACTOOLS_PAYLOAD_ORDER
    widths = [ai.PAYLOAD_DTYPE[name].itemsize // 4 for name in ai.PAYLOAD_FIELDS]
    assert sum(widths) == 18


def test_writer_matches_hand_packed_layout() -> None:
    line = random_line(5, seed=1)
    assert ai_to_bytes(line) == hand_pack(line)


def test_round_trip_is_bit_identical() -> None:
    line = random_line(64, seed=2)
    nan_with_payload = np.array([0x7FC12345], dtype=np.uint32).view(np.float32)[0]
    line.speed[3] = nan_with_payload
    line.radius[4] = -0.0
    line.camber[5] = np.inf
    line.grade[6] = np.float32(1e-45)  # denormal
    line.normal[7] = (np.nan, -np.inf, -0.0)
    data = ai_to_bytes(line)
    assert len(data) == 16 + 64 * 20 + 4 + 64 * 72 + 4
    back = read_ai(data)
    assert_identical(line, back)
    assert ai_to_bytes(back) == data


def test_write_and_read_file(tmp_path: Path) -> None:
    line = random_line(10, seed=3)
    path = write_ai(line, tmp_path / "track" / "ai" / "fast_lane.ai")
    assert path.read_bytes() == ai_to_bytes(line)
    assert_identical(read_ai(path), line)
    assert_identical(read_ai(str(path)), line)
    assert_identical(read_ai(bytearray(path.read_bytes())), line)
    assert_identical(read_ai(memoryview(path.read_bytes())), line)


def test_empty_line_round_trips() -> None:
    line = AiLine(positions=np.zeros((0, 3)))
    data = ai_to_bytes(line)
    assert data == struct.pack("<4i", 7, 0, 0, 0) + struct.pack("<ii", 0, 0)
    assert read_ai(data).n_points == 0


def test_grid_round_trip_matches_actools_layout() -> None:
    cells = [
        [[5, 1, 2], [], [9]],  # ragged row with an empty cell
        [],  # empty row
        [[0, 1], [2, 3], [4, 5], [6, 7]],  # uniform row (fast path)
    ]
    grid = AiGrid(
        max_extreme=(10.5, 0.0, 20.25),
        min_extreme=(-10.5, 0.0, -20.25),
        neighbors_considered=10,
        sampling_density=10.0,
        cells=[[np.array(c, dtype=np.int32) for c in row] for row in cells],
    )
    line = random_line(12, seed=4, grid=grid)
    block = hand_pack_grid((10.5, 0.0, 20.25), (-10.5, 0.0, -20.25), 10, 10.0, cells)
    assert grid.to_bytes() == block
    data = ai_to_bytes(line)
    assert data == hand_pack(line, block)

    back = read_ai(data)
    assert isinstance(back.grid, AiGrid)
    assert back.grid.shape == (3, 4)
    assert back.grid.cell_count == 7
    assert back.grid.index_count == 12
    assert [[c.tolist() for c in row] for row in back.grid.cells] == cells
    assert_identical(back, line)
    assert ai_to_bytes(back) == data


def test_include_grid_false_writes_flag_zero() -> None:
    grid = AiGrid((1, 0, 1), (0, 0, 0), 1, 1.0, [[np.array([0])]])
    line = random_line(8, seed=5, grid=grid)
    line.trailing = b"junk"
    data = ai_to_bytes(line, include_grid=False)
    assert len(data) == 16 + 8 * 20 + 4 + 8 * 72 + 4
    assert data[-4:] == struct.pack("<i", 0)
    assert read_ai(data).grid is None


def test_unparseable_grid_is_kept_as_raw_bytes() -> None:
    base = ai_to_bytes(random_line(6, seed=6))
    data = base[:-4] + struct.pack("<i", 1) + b"\x01\x02\x03"
    with pytest.warns(UserWarning, match="raw bytes"):
        line = read_ai(data)
    assert line.grid == b"\x01\x02\x03"
    assert ai_to_bytes(line) == data


def test_trailing_bytes_survive_a_rewrite() -> None:
    grid = AiGrid((1, 0, 1), (0, 0, 0), 2, 5.0, [[np.array([0, 1])]])
    data = ai_to_bytes(random_line(6, seed=7, grid=grid)) + b"tail"
    line = read_ai(data)
    assert line.trailing == b"tail"
    assert ai_to_bytes(line) == data

    no_grid = ai_to_bytes(random_line(6, seed=8)) + b"more"
    assert ai_to_bytes(read_ai(no_grid)) == no_grid


def test_missing_grid_flag_is_tolerated() -> None:
    """FastLaneUtils writes no has_grid flag; we read it and add has_grid = 0 on rewrite."""
    full = ai_to_bytes(random_line(6, seed=9))
    line = read_ai(full[:-4])
    assert line.grid is None
    assert ai_to_bytes(line) == full


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: struct.pack("<i", 6) + d[4:], "v6 not supported"),
        (lambda d: struct.pack("<i", -1) + d[4:], "v-1"),
        (lambda d: d[:8] + d[8:16], "truncated"),
        (lambda d: d[:4] + struct.pack("<i", -3) + d[8:], "negative"),
        (lambda d: d[:10], "too short"),
        (lambda d: d[: 16 + 3 * 20] + struct.pack("<i", 2) + d[16 + 3 * 20 + 4 :], "payload count"),
        (lambda d: d[:-4] + b"\x00\x00", "stray bytes"),
    ],
)
def test_rejects_malformed_files(mutate, message: str) -> None:
    data = ai_to_bytes(random_line(3, seed=10))
    with pytest.raises(ValueError, match=message):
        read_ai(mutate(data))


def test_defaults_fill_missing_arrays() -> None:
    line = AiLine(positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 4.0], [3.0, 10.0, 4.0]])
    assert line.lengths.tolist() == [0.0, 5.0, 15.0]
    assert line.ids.tolist() == [0, 1, 2]
    assert line.direction.tolist() == [1.0, 1.0, 1.0]
    assert line.speed.tolist() == [0.0, 0.0, 0.0]
    assert line.normal.shape == (3, 3) and not line.normal.any()
    assert line.positions.dtype == np.float32 and line.ids.dtype == np.int32
    assert line.closing_gap == pytest.approx(math.hypot(3, 10, 4))
    assert line.is_closed()  # 11.2 m gap < AcTools' 20 m
    assert not line.is_closed(max_gap=10.0)
    assert line.width.tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="speed"):
        AiLine(positions=np.zeros((3, 3)), speed=np.zeros(4))
    with pytest.raises(ValueError, match="positions"):
        AiLine(positions=np.zeros((3, 2)))


# --------------------------------------------------------------------------------------------
# lookup grid generation
# --------------------------------------------------------------------------------------------


def test_build_grid_matches_brute_force() -> None:
    s = np.linspace(0.0, 120.0, 61)
    pos = np.column_stack([s, np.zeros_like(s), 15.0 * np.sin(s / 17.0)])
    grid = build_grid(pos, sampling_density=5.0, neighbors=4, padding=25.0)
    lo = (pos[:, [0, 2]].min(axis=0) - 25.0).astype(np.float32)
    hi = (pos[:, [0, 2]].max(axis=0) + 25.0).astype(np.float32)
    assert grid.min_extreme.tolist() == [lo[0], 0.0, lo[1]]
    assert grid.max_extreme.tolist() == [hi[0], 0.0, hi[1]]
    nx, nz = int((float(hi[0]) - float(lo[0])) / 5.0), int((float(hi[1]) - float(lo[1])) / 5.0)
    assert grid.shape == (nx, nz)
    assert grid.neighbors_considered == 4 and grid.sampling_density == 5.0
    for ix in range(nx):
        for iz in range(nz):
            cx, cz = grid.cell_centre(ix, iz)
            d2 = (pos[:, 0] - cx) ** 2 + (pos[:, 2] - cz) ** 2
            expected = np.lexsort((np.arange(len(pos)), d2))[:4]
            assert grid.cells[ix][iz].tolist() == expected.tolist(), (ix, iz)


def test_build_grid_with_fewer_points_than_neighbors() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, 2.0, 5.0]])
    grid = build_grid(pos, padding=30.0)
    assert all(c.size == 3 for row in grid.cells for c in row)
    assert grid.neighbors_considered == ai.GRID_NEIGHBORS
    with pytest.raises(ValueError):
        build_grid(np.zeros((0, 3)))


# --------------------------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------------------------


def test_signed_curvature_is_positive_towards_coords_left() -> None:
    forward = np.array([1.0, 0.0, 0.0])  # heading east
    left = left_of(forward)  # north, from the shared coords convention
    radius = 40.0
    t = np.linspace(0.0, 1.5, 200)
    centre = radius * left
    pts = centre - radius * np.outer(np.cos(t), left) + radius * np.outer(np.sin(t), forward)
    kappa = signed_curvature(pts, closed=False)
    assert np.allclose(kappa, 1.0 / radius, rtol=1e-3)
    assert np.allclose(signed_curvature(pts[::-1], closed=False), -1.0 / radius, rtol=1e-3)


def check_common_invariants(line: AiLine) -> None:
    n = line.n_points
    assert np.array_equal(line.ids, np.arange(n))
    assert np.all(np.diff(line.lengths.astype(np.float64)) > 0)
    assert line.lengths[0] == 0.0
    assert np.allclose(np.linalg.norm(line.forward, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(line.normal, axis=1), 1.0, atol=1e-5)
    assert np.all(line.side_left > 0) and np.all(line.side_right > 0)
    assert np.all(line.radius > 0) and np.all(np.isfinite(line.radius))
    assert set(np.unique(line.direction).tolist()) <= {-1.0, 1.0}
    assert np.all((line.gas >= 0) & (line.gas <= 1))
    assert np.all((line.brake >= 0) & (line.brake <= 1))
    assert not np.any((line.gas > 0) & (line.brake > 0))
    assert np.all(line.tag == 0)
    for name in ("positions", "lengths", *ai.PAYLOAD_FIELDS):
        assert np.all(np.isfinite(getattr(line, name))), name
    seg = line.segment_lengths()
    assert np.allclose(np.diff(line.lengths), seg, atol=1e-3)


def test_straight_line() -> None:
    line = build_ai_line(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -400.0]]), 4.0, 3.0, closed=False)
    check_common_invariants(line)
    assert line.n_points == 401
    assert line.lengths[-1] == pytest.approx(400.0)
    assert np.all(line.radius == np.float32(ai.RADIUS_MAX))
    assert np.allclose(line.speed, ai.DEFAULT_MAX_SPEED_MPS)
    assert np.all(line.brake == 0)
    assert np.allclose(line.forward, [0.0, 0.0, -1.0])
    assert np.allclose(line.normal, WORLD_UP)
    assert np.all(line.camber == 0) and np.all(line.grade == 0)
    assert np.allclose(line.side_left, 3.5) and np.allclose(line.side_right, 2.5)
    assert np.allclose(line.length, 5.0)  # 5-point look-back at 1 m spacing
    assert np.all(line.obsolete_lat_g == 0)
    assert not line.is_closed() and line.closing_gap == pytest.approx(400.0)
    assert line.grid is None and line.lap_time == 0 and line.sample_count == 0


@pytest.mark.parametrize(("radius", "mu"), [(30.0, 1.0), (120.0, 1.0), (60.0, 0.8)])
def test_circle_radius_and_speed(radius: float, mu: float) -> None:
    line = build_ai_line(circle(radius), 5.0, 5.0, closed=True, mu=mu)
    check_common_invariants(line)
    assert np.allclose(line.radius, radius, rtol=0.01)
    assert np.allclose(line.speed, math.sqrt(mu * G * radius), rtol=0.01)
    assert np.all(line.direction == 1.0)  # counter-clockwise on the map = left turns
    assert np.allclose(line.obsolete_lat_g, mu, rtol=0.02)
    assert line.brake.max() < 0.1  # only the input polygon's facets cause speed ripples
    assert line.is_closed() and line.closing_gap == pytest.approx(1.0, rel=0.05)
    assert line.lengths[-1] == pytest.approx(2 * math.pi * radius - line.closing_gap, rel=1e-3)


def test_clockwise_circle_turns_right() -> None:
    line = build_ai_line(circle(50.0, ccw_on_map=False), 5.0, 5.0, closed=True)
    assert np.all(line.direction == -1.0)
    assert np.allclose(line.obsolete_lat_g, -1.0, rtol=0.02)
    assert np.allclose(line.radius, 50.0, rtol=0.01)


def test_closed_input_may_repeat_the_first_point() -> None:
    pts = circle(40.0, 400)
    a = build_ai_line(pts, 4.0, 4.0, closed=True)
    b = build_ai_line(np.vstack([pts, pts[:1]]), 4.0, 4.0, closed=True)
    assert a.n_points == b.n_points
    assert np.allclose(a.positions, b.positions, atol=1e-4)


def test_brakes_before_a_tight_bend() -> None:
    line = build_ai_line(hairpin(15.0), 4.0, 4.0, closed=False)
    check_common_invariants(line)
    entry = 300  # index of the arc start at 1 m spacing
    apex_speed = math.sqrt(G * 15.0)
    arc = slice(entry + 10, entry + 35)
    assert np.allclose(line.radius[arc], 15.0, rtol=0.02)
    assert np.allclose(line.speed[arc], apex_speed, rtol=0.02)
    assert np.all(line.direction[arc] == 1.0)
    assert line.speed.min() == pytest.approx(apex_speed, rel=0.02)
    # Full speed at the start, a braking zone that ends at the bend, nothing earlier.
    assert line.speed[10] == pytest.approx(ai.DEFAULT_MAX_SPEED_MPS)
    assert line.brake[10] == 0
    zone = slice(entry - 60, entry - 10)
    assert line.speed[entry - 100] > 2.5 * apex_speed
    assert np.all(np.diff(line.speed[zone]) < 0)
    assert np.all(line.brake[zone] > 0.5) and np.all(line.gas[zone] == 0)
    # Accelerates out of the hairpin.
    assert np.all(line.gas[entry + 60 : entry + 120] > 0.5)
    assert line.speed[-1] > 3 * apex_speed


def test_open_line_start_and_end_speeds() -> None:
    line = build_ai_line(
        np.array([[0.0, 0.0, 0.0], [300.0, 0.0, 0.0]]),
        4.0,
        4.0,
        closed=False,
        start_speed_mps=0.0,
        end_speed_mps=0.0,
        max_accel_mps2=4.0,
        max_brake_mps2=8.0,
    )
    assert line.speed[0] == 0.0 and line.speed[-1] == 0.0
    assert line.speed[50] == pytest.approx(math.sqrt(2 * 4.0 * 50), rel=1e-3)
    assert line.speed[-26] == pytest.approx(math.sqrt(2 * 8.0 * 25), rel=1e-3)
    assert line.gas[10] > 0.9 and line.brake[-10] > 0.9


def test_sides_follow_widths_and_stay_positive() -> None:
    pts = np.column_stack([np.linspace(0, 100, 11), np.zeros(11), np.zeros(11)])
    left = np.linspace(0.2, 6.0, 11)  # starts narrower than the margin
    line = build_ai_line(pts, left, 2.0, closed=False, margin=0.5)
    check_common_invariants(line)
    assert line.side_left.min() == pytest.approx(ai.MIN_SIDE_M)
    assert line.side_left[-1] == pytest.approx(5.5)
    assert line.side_left[50] == pytest.approx(np.interp(50, np.linspace(0, 100, 11), left) - 0.5)
    assert np.allclose(line.side_right, 1.5)
    with pytest.raises(ValueError, match="half_width_left"):
        build_ai_line(pts, np.ones(5), 2.0, closed=False)


def test_grade_on_a_ramp() -> None:
    run = np.linspace(0.0, 200.0, 3)
    pts = np.column_stack([run, 0.05 * run, np.zeros(3)])  # 5% uphill heading east
    line = build_ai_line(pts, 4.0, 4.0, closed=False)
    check_common_invariants(line)
    assert np.allclose(line.grade, 0.05, rtol=1e-4)
    step = 200.0 / (line.n_points - 1)  # horizontal run per point
    assert np.allclose(line.length, 5 * step, rtol=1e-5)  # horizontal chord, not 3D
    assert np.allclose(line.forward[:, 1], 0.05 / math.hypot(1, 0.05), rtol=1e-4)


@pytest.mark.parametrize("closed", [False, True])
def test_line_is_lifted_above_the_road(closed: bool) -> None:
    """Game-recorded lines sit ~0.47 m above the road (sepang); ours sit 0.5 m above it."""
    pts = circle(40.0) if closed else hairpin(15.0)  # both on the y = 0 road surface
    lifted = build_ai_line(pts, 4.0, 4.0, closed=closed, with_grid=True)
    flat = build_ai_line(pts, 4.0, 4.0, closed=closed, with_grid=True, height_offset_m=0.0)
    assert ai.AI_HEIGHT_OFFSET_M == 0.5
    assert np.all(lifted.positions[:, 1] == 0.5)
    assert np.all(flat.positions[:, 1] == 0.0)
    # A constant vertical lift changes nothing but y: same x/z, payload, lengths and grid.
    assert np.array_equal(lifted.positions[:, [0, 2]], flat.positions[:, [0, 2]])
    for name in ("lengths", "ids", *ai.PAYLOAD_FIELDS):
        assert np.array_equal(getattr(lifted, name), getattr(flat, name)), name
    assert isinstance(lifted.grid, AiGrid) and isinstance(flat.grid, AiGrid)
    assert lifted.grid.to_bytes() == flat.grid.to_bytes()


def test_banked_circle_is_faster() -> None:
    radius, bank = 50.0, math.radians(10.0)
    pts = circle(radius)
    inward = -pts / radius  # left of a counter-clockwise circle points at the centre
    normals = math.cos(bank) * np.array(WORLD_UP) + math.sin(bank) * inward
    line = build_ai_line(pts, 5.0, 5.0, closed=True, surface_normals=normals)
    assert np.allclose(line.camber, math.sin(bank), atol=2e-3)
    tan_phi = math.tan(bank)
    expected = math.sqrt(G * radius * (1 + tan_phi) / (1 - tan_phi))
    assert np.allclose(line.speed, expected, rtol=0.015)
    assert line.speed.mean() > math.sqrt(G * radius) * 1.15


def test_speed_profile_checks_lengths() -> None:
    with pytest.raises(ValueError, match="segment_lengths"):
        speed_profile(np.zeros(5), np.ones(5), closed=False)
    v = speed_profile(np.zeros(5), np.ones(4), closed=False, max_speed_mps=30.0)
    assert np.allclose(v, 30.0)


def test_generated_line_round_trips_with_grid() -> None:
    line = build_ai_line(circle(40.0, 300), 4.0, 4.0, closed=True, with_grid=True)
    assert isinstance(line.grid, AiGrid)
    assert all(c.size == ai.GRID_NEIGHBORS for row in line.grid.cells for c in row)
    data = ai_to_bytes(line)
    back = read_ai(data)
    assert_identical(back, line)
    assert ai_to_bytes(back) == data
    shipped = ai_to_bytes(line, include_grid=False)
    assert len(shipped) == 16 + line.n_points * (20 + 72) + 8


# --------------------------------------------------------------------------------------------
# dump
# --------------------------------------------------------------------------------------------


def test_dump_reports_units_and_writes_csv(tmp_path: Path) -> None:
    line = build_ai_line(circle(60.0), 5.0, 4.0, closed=True, with_grid=True)
    path = write_ai(line, tmp_path / "fast_lane.ai")
    csv_path = tmp_path / "out" / "fast_lane.csv"
    text = dump(path, csv_path=csv_path)

    for expected in (
        f"points (N)      {line.n_points}",
        "ids are 0..N-1: yes",
        "CLOSED loop",
        "spacing (m)",
        "km/h",
        "speed looks like m/s",
        "radius / horizontal circumradius",
        "direction       counts {1: ",
        "== +1 on left turns / -1 on right turns at 100.0%",
        "5-point look-back distance",
        "tag             counts {0: ",
        "present         yes",
        f"neighbors_considered {ai.GRID_NEIGHBORS}",
        "200/200 sampled cells",
        "trailing bytes  0",
        f"wrote {line.n_points} rows",
    ):
        assert expected in text, expected
    for name in ("speed", "gas", "obsolete_lat_g", "normal.y", "forward.z", "grade"):
        assert f"\n  {name} " in text, name

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == list(ai.CSV_COLUMNS)
    assert len(rows) == line.n_points
    for i in (0, 17, line.n_points - 1):
        assert int(rows[i]["id"]) == i
        assert np.float32(float(rows[i]["speed"])) == line.speed[i]
        assert np.float32(float(rows[i]["normal_y"])) == line.normal[i, 1]
        assert np.float32(float(rows[i]["z"])) == line.positions[i, 2]


def test_dump_open_line_without_grid() -> None:
    line = build_ai_line(hairpin(), 4.0, 4.0, closed=False)
    text = dump(ai_to_bytes(line, include_grid=False))
    assert "AI line: <bytes>" in text
    assert "OPEN (point-to-point)" in text
    assert "present         no" in text
    assert "speed looks like m/s" in text


def test_dump_tiny_file() -> None:
    text = dump(ai_to_bytes(AiLine(positions=np.zeros((1, 3)))))
    assert "fewer than 2 points" in text
    assert "needs at least" in text


def test_cli_ai_dump(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from driftmaps.cli import app

    path = write_ai(build_ai_line(circle(30.0), 4.0, 4.0, closed=True), tmp_path / "a.ai")
    result = CliRunner().invoke(app, ["ai-dump", str(path), "--csv", str(tmp_path / "a.csv")])
    assert result.exit_code == 0, result.output
    assert "Unit hypotheses" in result.output
    assert (tmp_path / "a.csv").exists()


# --------------------------------------------------------------------------------------------
# real game-baked files (downloaded at test time, skipped offline)
# --------------------------------------------------------------------------------------------

SAMPLE_REPO = "619motorsports/newksEditor"
SAMPLE_COMMIT = "93d21a4999fe37ee7b1c2cf7ad34cc81b2fcd771"
SAMPLES = {
    "fast_lane": (
        "test/content/tracks/sepang/ai/fast_lane.ai",
        "33f109e6d8faaa78dd0e8dbd5b614a5958649d1cd621d4bcd54d935b1e6e2364",
    ),
    "pit_lane": (
        "test/content/tracks/sepang/ai/pit_lane.ai",
        "81b483205bd3853ed147846f3279221996c0b347fe9fd83ec06d54aa32b77c4f",
    ),
}


def fetch_sample(name: str) -> bytes:
    rel, sha = SAMPLES[name]
    cached = CACHE_DIR / f"sepang_{name}.ai"
    if cached.exists():
        data = cached.read_bytes()
        if hashlib.sha256(data).hexdigest() == sha:
            return data
    # The files are Git LFS objects, served from media.githubusercontent.com.
    url = f"https://media.githubusercontent.com/media/{SAMPLE_REPO}/{SAMPLE_COMMIT}/{rel}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
        pytest.skip(f"cannot download sample AI file (offline?): {exc}")
    if hashlib.sha256(data).hexdigest() != sha:
        pytest.skip(f"downloaded {url} does not match the pinned sha256")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(cached)
    return data


@pytest.fixture(scope="module")
def sepang_fast_lane() -> bytes:
    return fetch_sample("fast_lane")


@pytest.fixture(scope="module")
def sepang_pit_lane() -> bytes:
    return fetch_sample("pit_lane")


def test_real_fast_lane_round_trips_byte_identical(sepang_fast_lane: bytes) -> None:
    line = read_ai(sepang_fast_lane)
    assert (line.version, line.n_points, line.lap_time, line.sample_count) == (7, 3536, 0, 0)
    assert isinstance(line.grid, AiGrid)
    assert line.grid.shape == (185, 163)
    assert line.grid.neighbors_considered == 10 and line.grid.sampling_density == 10.0
    assert all(c.size == 10 for row in line.grid.cells for c in row)
    assert line.trailing == b""
    assert ai_to_bytes(line) == sepang_fast_lane


def test_real_pit_lane_round_trips_byte_identical(sepang_pit_lane: bytes) -> None:
    line = read_ai(sepang_pit_lane)
    assert line.grid is None and line.n_points == 4329
    assert ai_to_bytes(line) == sepang_pit_lane


def test_real_file_matches_observed_conventions(sepang_fast_lane: bytes) -> None:
    """The payload conventions the generator copies (module docstring), checked independently."""
    line = read_ai(sepang_fast_lane)
    p = line.positions.astype(np.float64)
    back5 = p - np.roll(p, 5, axis=0)
    chord = np.linalg.norm(back5[:, [0, 2]], axis=1)
    assert np.median(np.abs(line.length - chord)) < 1e-3  # length = XZ distance to i-5
    assert np.median(np.abs(line.grade - back5[:, 1] / chord)) < 1e-6  # grade = rise / run

    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    assert np.abs(np.diff(line.lengths) - seg).max() < 0.01  # cumulative distance

    kappa = signed_curvature(p, closed=True)
    radius = line.radius.astype(np.float64)
    corners = radius < 300
    assert np.median(np.abs(radius[corners] * np.abs(kappa[corners]) - 1)) < 0.005
    assert np.mean(line.direction[corners] == np.sign(kappa[corners])) > 0.99  # +1 = left

    lat = line.obsolete_lat_g.astype(np.float64)
    v = line.speed.astype(np.float64)
    m = corners & (np.abs(lat) > 0.05)
    assert np.median(np.abs(lat[m]) * radius[m] * G / v[m] ** 2) == pytest.approx(1.0, abs=0.05)
    assert np.mean(np.sign(lat[m]) == line.direction[m]) > 0.99

    fwd = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    left = np.cross(WORLD_UP, fwd)
    left /= np.linalg.norm(left, axis=1, keepdims=True)
    n_left = np.einsum("ij,ij->i", line.normal, left)
    tilted = np.abs(n_left) > 0.005
    ratio = line.camber[tilted] * line.direction[tilted] / n_left[tilted]
    assert np.median(ratio) == pytest.approx(1.0, abs=0.05)


def test_build_grid_reproduces_real_grid(sepang_fast_lane: bytes) -> None:
    line = read_ai(sepang_fast_lane)
    assert isinstance(line.grid, AiGrid)
    ours = build_grid(line.positions)
    assert raw(ours.min_extreme) == raw(line.grid.min_extreme)
    assert raw(ours.max_extreme) == raw(line.grid.max_extreme)
    assert ours.shape == line.grid.shape
    pairs = [
        (a, b)
        for ra, rb in zip(ours.cells, line.grid.cells, strict=True)
        for a, b in zip(ra, rb, strict=True)
    ]
    same_set = np.mean([set(a.tolist()) == set(b.tolist()) for a, b in pairs])
    same_order = np.mean([np.array_equal(a, b) for a, b in pairs])
    # The rest differ only in the order of near-equal distances (float precision).
    assert same_set > 0.999 and same_order > 0.99


def test_dump_real_file(sepang_fast_lane: bytes, tmp_path: Path) -> None:
    path = tmp_path / "fast_lane.ai"
    path.write_bytes(sepang_fast_lane)
    text = dump(path, csv_path=tmp_path / "fast_lane.csv")
    assert "file size       1,652,932 bytes" in text
    assert "CLOSED loop" in text
    assert "speed looks like m/s" in text
    assert "200/200 sampled cells" in text
    assert "padding x / z   350, 350 / 350, 350" in text
    assert len((tmp_path / "fast_lane.csv").read_text().splitlines()) == 3536 + 1
