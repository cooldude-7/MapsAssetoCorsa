"""Centreline smoothing, fold-over removal and road meshes (geometry/road.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from driftmaps.ac.model import MAX_MESH_VERTICES
from driftmaps.geometry.road import (
    RoadModel,
    build_road_meshes,
    chunk_ranges,
    extend_straight,
    fillet_corners,
    fix_foldover,
    footprint_polygon,
    road_columns,
    simplify,
    smooth_centreline,
    whittaker,
)


def hairpin_xz(radius: float, straight: float = 80.0, step: float = 1.0) -> np.ndarray:
    """North for `straight` m, a 180 degree right turn of `radius`, then south: (N, 2) XZ."""
    up = np.stack([np.zeros(int(straight / step)), -np.arange(int(straight / step)) * step], 1)
    n = max(8, math.ceil(math.pi * radius / step))
    a = np.linspace(math.pi, 0.0, n + 1)  # centre at (radius, -straight)
    arc = np.stack([radius + radius * np.cos(a), -straight - radius * np.sin(a)], axis=1)
    down = np.stack([np.full(len(up), 2 * radius), up[::-1, 1]], axis=1)
    return np.concatenate([up, arc, down[1:]])


def make_model(xz: np.ndarray, *, half_width=3.5, shoulder=1.0, grade=0.05, **kw) -> RoadModel:
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))])
    centre = np.stack([xz[:, 0], grade * s, xz[:, 1]], axis=1)
    kw.setdefault("centre_line", True)
    kw.setdefault("bridge", False)
    return RoadModel(s=s, centre=centre, half_width=half_width, shoulder=shoulder, **kw)


def all_triangles(meshes) -> np.ndarray:
    out = [m.positions[m.triangles] for m in meshes if m is not None]
    return np.concatenate(out)


def plan_area2(tris: np.ndarray) -> np.ndarray:
    """Twice the signed XZ area, positive for counter-clockwise seen from above (+Y)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return (b[:, 2] - a[:, 2]) * (c[:, 0] - a[:, 0]) - (b[:, 0] - a[:, 0]) * (c[:, 2] - a[:, 2])


def test_whittaker_keeps_quadratics_and_pins_weights():
    x = np.linspace(0, 10, 50)
    y = 0.3 * x**2 - 2 * x + 1
    assert np.allclose(whittaker(y, np.ones(50), 1e4), y, atol=1e-6)
    noisy = y + np.random.default_rng(0).normal(0, 0.5, 50)
    smooth = whittaker(noisy, np.ones(50), 10.0)
    assert np.abs(np.diff(smooth, 3)).max() < np.abs(np.diff(noisy, 3)).max()


def test_fillet_rounds_a_right_angle():
    xy = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]])
    out, param = fillet_corners(xy, np.array([0.0, 100.0, 200.0]), radius_m=12.0)
    seg = np.diff(out, axis=0)
    ang = np.degrees(np.arctan2(seg[:, 1], seg[:, 0]))
    assert np.abs(np.diff(ang)).max() < 10.0  # no sharp corner left
    length = np.linalg.norm(seg, axis=1).sum()
    assert length == pytest.approx(200.0 - 24.0 + 12.0 * math.pi / 2, abs=0.05)
    assert (np.diff(param) >= 0).all() and param[0] == 0.0 and param[-1] == 200.0
    # Every arc point is 12 m from the fillet centre (88, 12).
    arc = out[(out[:, 0] > 88.01) & (out[:, 1] < 11.99)]
    assert np.allclose(np.linalg.norm(arc - [88.0, 12.0], axis=1), 12.0, atol=1e-6)


def test_fillet_radius_limited_by_short_segments():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    out, _ = fillet_corners(xy, np.array([0.0, 10.0, 20.0]), radius_m=50.0)
    # Tangent length is at most 45% of the 10 m segments.
    assert out[1][0] == pytest.approx(10.0 - 4.5, abs=1e-6)


def test_smooth_centreline_stays_close_and_is_smooth():
    rng = np.random.default_rng(1)
    x = np.arange(0.0, 600.0, 20.0)
    y = 30 * np.sin(x / 90.0) + rng.normal(0, 2.0, len(x))  # kinky OSM-like nodes
    xy = np.stack([x, y], axis=1)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    out, param = smooth_centreline(xy, s, max_deviation_m=1.5)
    seg = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert seg.max() <= 1.0 + 1e-9 and seg.min() > 0.9
    assert np.allclose(out[[0, -1]], xy[[0, -1]], atol=0.01)
    dense = np.concatenate(
        [np.linspace(a, b, 200, endpoint=False) for a, b in zip(xy[:-1], xy[1:], strict=True)]
    )
    dev = np.min(np.linalg.norm(out[:, None, :] - dense[None, ::4, :], axis=2), axis=1)
    assert dev.max() <= 1.5 + 0.1
    assert (np.diff(param) >= -1e-9).all()
    # Heading changes smoothly: no kink anywhere near the raw 20 m node kinks.
    head = np.unwrap(np.arctan2(np.diff(out[:, 1]), np.diff(out[:, 0])))
    raw_head = np.unwrap(np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0])))
    assert np.abs(np.diff(head)).max() < 0.5 * np.abs(np.diff(raw_head)).max()


def test_extend_straight():
    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    out, param = extend_straight(xy, np.array([0.0, 10.0]), 6.0, 4.0, step_m=2.0)
    assert out[0] == pytest.approx([-6.0, 0.0]) and out[-1] == pytest.approx([14.0, 0.0])
    assert param[0] == pytest.approx(-6.0) and param[-1] == pytest.approx(14.0)
    assert (np.diff(out[:, 0]) > 0).all()


def test_fix_foldover_on_too_tight_inside_curve():
    xz = hairpin_xz(radius=3.0)
    model = make_model(xz)
    raw = model.centre + 5.0 * model.left * -1.0  # right side = inside of a right turn
    step = np.einsum("ij,ij->i", np.diff(raw, axis=0)[:, [0, 2]], model.forward[1:, [0, 2]])
    assert (step < 0).any()  # the raw offset folds over
    fixed = fix_foldover(raw, model.forward)
    tb = model.forward[:-1, [0, 2]] + model.forward[1:, [0, 2]]
    assert (np.einsum("ij,ij->i", np.diff(fixed, axis=0)[:, [0, 2]], tb) >= -1e-9).all()
    # Away from the hairpin nothing moves.
    assert np.allclose(fixed[:40], raw[:40]) and np.allclose(fixed[-40:], raw[-40:])


@pytest.mark.parametrize("radius", [2.0, 4.0, 8.0, 30.0])
def test_road_meshes_never_fold(radius):
    model = make_model(hairpin_xz(radius), half_width=3.6, shoulder=1.0)
    rm = build_road_meshes(model, chunk_m=60.0)
    tris = all_triangles(rm.road + rm.gravel + rm.paint_white + rm.paint_yellow)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    assert (n[:, 1] > 0).all()
    assert (plan_area2(tris) > -1e-9).all()
    for m in rm.road + rm.gravel:
        assert m.vertex_count <= MAX_MESH_VERTICES
        assert np.allclose(np.linalg.norm(m.normals, axis=1), 1.0)
    # Asphalt and shoulders share their edge vertices: no crack between them.
    road_edge = road_columns(model, model.half_width)
    gravel_pts = np.concatenate([m.positions for m in rm.gravel])
    d = np.min(np.linalg.norm(road_edge[::7, None, :] - gravel_pts[None, :, :], axis=2), axis=1)
    assert d.max() < 1e-9


def test_chunks_cover_the_road_and_share_stations():
    model = make_model(np.stack([np.arange(1001.0), np.zeros(1001)], axis=1))
    ranges = chunk_ranges(model.s, 250.0)
    assert ranges == [(0, 250), (250, 500), (500, 750), (750, 1000)]
    rm = build_road_meshes(model)
    assert len(rm.road) == len(rm.gravel) == 4
    # v restarts per chunk on a tile multiple, so it stays small (float32 precision).
    assert all(np.abs(m.uvs[:, 1]).max() <= (250.0 + 4.0) / 4.0 for m in rm.road)


def test_paint_styles():
    xz = np.stack([np.arange(300.0), np.zeros(300)], axis=1)
    na = build_road_meshes(make_model(xz, double_yellow=True))
    intl = build_road_meshes(make_model(xz, double_yellow=False))
    assert all(m is not None for m in na.paint_yellow)
    assert all(m is None for m in intl.paint_yellow)
    # Dashed white centre (3 m on, 9 m off) adds about a quarter of the line's length.
    w_na = sum(m.triangle_count for m in na.paint_white)
    w_intl = sum(m.triangle_count for m in intl.paint_white)
    assert w_intl - w_na == pytest.approx(w_na / 2 * 0.25, rel=0.1)
    one_way = build_road_meshes(make_model(xz, double_yellow=True, centre_line=False))
    assert all(m is None for m in one_way.paint_yellow)
    # Painted lines float above the asphalt and stay on it.
    for m in na.paint_white + na.paint_yellow:
        assert (m.positions[:, 1] > 0.0).all()
        assert np.abs(m.positions[:, 2]).max() < 3.5


def test_footprint_polygon_area():
    model = make_model(np.stack([np.arange(101.0), np.zeros(101)], axis=1))
    poly = footprint_polygon(road_columns(model, 4.5), road_columns(model, -4.5))
    assert poly.area == pytest.approx(100.0 * 9.0, rel=1e-6)


def test_simplify_merges_collinear_junction_pieces():
    # A straight leg split by junction nodes 3 m apart, then a right angle.
    xy = np.array([[0.0, 0.0], [40.0, 0.05], [43.0, 0.0], [46.0, -0.04], [80.0, 0.0], [80.0, 50.0]])
    out, param = simplify(xy, np.arange(6.0), 0.3)
    assert out.tolist() == [[0.0, 0.0], [80.0, 0.0], [80.0, 50.0]]
    assert param.tolist() == [0.0, 4.0, 5.0]
    # The fillet now gets the full 12 m radius instead of 45% of a 3 m piece.
    f, _ = fillet_corners(out, param, radius_m=12.0)
    assert f[1][0] == pytest.approx(68.0)
