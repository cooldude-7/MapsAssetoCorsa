"""Tests for driftmaps.geometry.primitives.

Invariants checked on every builder:
- winding: each triangle's geometric normal cross(p1 - p0, p2 - p0) points to the visible side
  (up for ribbons and grids, outward for boxes, towards the viewer for signs);
- vertex normals agree with the faces, tangents are unit, perpendicular to the normal and run
  along increasing u;
- UVs are not mirrored from the visible side (DirectX top-left origin: u right, v down, so the
  signed (u, v) area of a front-facing triangle is negative).
"""

from __future__ import annotations

import numpy as np
import pytest

from driftmaps.ac.coords import WORLD_UP, left_of
from driftmaps.ac.model import MAX_MESH_VERTICES, Dummy, Kn5, Material
from driftmaps.geometry.centreline import (
    Arc,
    Straight,
    cumulative_length,
    frames,
    heading_vector,
    path,
    rounded_rectangle_loop,
    straight,
)
from driftmaps.geometry.primitives import (
    MeshData,
    billboard,
    box,
    compute_normals,
    compute_tangents,
    face_normals,
    flat_grid,
    flipped,
    grid_with_vertex_count,
    merge,
    miter_scale,
    offset_polyline,
    post,
    quad,
    ribbon,
    split,
    to_kn5_mesh,
    wall,
)

UP = np.array([0.0, 1.0, 0.0])


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def uv_areas(data: MeshData) -> np.ndarray:
    """Signed area of each triangle in (u, v) space."""
    t = data.uvs[data.triangles]
    d1 = t[:, 1] - t[:, 0]
    d2 = t[:, 2] - t[:, 0]
    return 0.5 * (d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1])


def triangle_uv_axes(data: MeshData) -> tuple[np.ndarray, np.ndarray]:
    """Per-triangle dP/du and dP/dv (T, 3), solved from the edge and UV deltas."""
    p = data.positions[data.triangles]
    t = data.uvs[data.triangles]
    e = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]], axis=1)  # (T, 2, 3)
    duv = np.stack([t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]], axis=1)  # (T, 2, 2)
    # e = duv @ [dP/du; dP/dv]
    axes = np.linalg.solve(duv, e)
    return axes[:, 0], axes[:, 1]


def centroids(data: MeshData) -> np.ndarray:
    return data.positions[data.triangles].mean(axis=1)


def assert_well_formed(data: MeshData) -> None:
    """Shared invariants: finite, unit normals/tangents, tangent along +u, not mirrored."""
    for arr in (data.positions, data.normals, data.uvs, data.tangents):
        assert np.isfinite(arr).all()
    np.testing.assert_allclose(np.linalg.norm(data.normals, axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(data.tangents, axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(np.sum(data.normals * data.tangents, axis=1), 0.0, atol=1e-9)
    fn = face_normals(data)
    assert (np.linalg.norm(fn, axis=1) > 0.5).all(), "degenerate triangles"
    # Each corner's vertex normal agrees with its face.
    for k in range(3):
        assert (np.sum(data.normals[data.triangles[:, k]] * fn, axis=1) > 0.5).all()
    # UVs are not mirrored from the visible side.
    assert (uv_areas(data) < 0.0).all()
    # Vertex tangents follow increasing u.
    dpdu, _ = triangle_uv_axes(data)
    for k in range(3):
        assert (np.sum(data.tangents[data.triangles[:, k]] * dpdu, axis=1) > 0.0).all()


def triangle_set(data: MeshData) -> list[tuple[float, ...]]:
    """Triangles as sorted, rounded position tuples (independent of vertex numbering)."""
    p = np.round(data.positions[data.triangles], 9).reshape(-1, 9)
    return sorted(map(tuple, p.tolist()))


L_ROAD = path((0.0, 0.0, 0.0), 0.0, [Straight(40), Arc(15, 90), Straight(40)], step=1.0)
S_ROAD = path(
    (5.0, 2.0, -3.0), 30.0, [Straight(20, rise=2.0), Arc(12, -110), Arc(25, 160, rise=-4.0)]
)


# --------------------------------------------------------------------------------------------
# Ribbon
# --------------------------------------------------------------------------------------------


def test_ribbon_straight_north_layout() -> None:
    centre = straight((0.0, 0.0, 0.0), 0.0, 20.0, step=2.0)  # 11 stations heading north
    r = ribbon(centre, 3.0, 4.0, lateral_segments=7, u_tile_m=1.0, v_tile_m=2.0)
    assert r.vertex_count == 11 * 8
    assert r.triangle_count == 10 * 7 * 2
    assert_well_formed(r)
    np.testing.assert_allclose(face_normals(r), np.tile(UP, (r.triangle_count, 1)), atol=1e-12)
    np.testing.assert_allclose(r.normals, np.tile(UP, (r.vertex_count, 1)), atol=1e-12)
    grid = r.positions.reshape(11, 8, 3)
    # Heading north the driver's left is west (-X): the left edge (column 0) is at x = -3.
    np.testing.assert_allclose(grid[:, 0, 0], -3.0)
    np.testing.assert_allclose(grid[:, -1, 0], 4.0)
    np.testing.assert_allclose(grid[:, :, 0], np.tile(np.linspace(-3, 4, 8), (11, 1)), atol=1e-12)
    # u = metres from the left edge, v falls along the road (image top points forward).
    uv = r.uvs.reshape(11, 8, 2)
    np.testing.assert_allclose(uv[0, :, 0], np.linspace(0, 7, 8), atol=1e-12)
    np.testing.assert_allclose(uv[:, 0, 1], -np.arange(11) * 2.0 / 2.0, atol=1e-12)
    # Tangent (dP/du) points to the driver's right (east).
    np.testing.assert_allclose(r.tangents, np.tile([1.0, 0.0, 0.0], (r.vertex_count, 1)))


@pytest.mark.parametrize("centre", [L_ROAD, S_ROAD], ids=["L", "S"])
def test_ribbon_faces_up_and_left_is_drivers_left(centre: np.ndarray) -> None:
    k = 5
    r = ribbon(centre, 3.5, 2.5, lateral_segments=k - 1)
    assert_well_formed(r)
    assert (face_normals(r)[:, 1] > 0.9).all()
    grid = r.positions.reshape(len(centre), k, 3)
    tangents, _, _ = frames(centre)
    for i in range(len(centre)):
        left = left_of(tangents[i])
        assert (grid[i, 0] - centre[i]) @ left == pytest.approx(3.5, rel=0.01)
        assert (grid[i, -1] - centre[i]) @ left == pytest.approx(-2.5, rel=0.01)


def test_ribbon_left_turn_road_edges() -> None:
    """On the L road (north, then left to west) the left edge is the inside of the bend."""
    r = ribbon(L_ROAD, 4.0, 4.0, lateral_segments=2)
    grid = r.positions.reshape(len(L_ROAD), 3, 3)
    left_len = np.linalg.norm(np.diff(grid[:, 0], axis=0), axis=1).sum()
    right_len = np.linalg.norm(np.diff(grid[:, -1], axis=0), axis=1).sum()
    assert left_len < right_len
    # End of the road heads west, so its left edge is to the south (+Z).
    assert grid[-1, 0, 2] > grid[-1, -1, 2]


def test_ribbon_uv_continuity_along() -> None:
    v_tile, u_tile = 5.0, 2.0
    r = ribbon(S_ROAD, 3.0, 3.0, lateral_segments=3, u_tile_m=u_tile, v_tile_m=v_tile)
    uv = r.uvs.reshape(len(S_ROAD), 4, 2)
    s = cumulative_length(S_ROAD)
    for col in range(4):
        np.testing.assert_allclose(uv[:, col, 1], -s / v_tile, atol=1e-12)
        np.testing.assert_allclose(uv[:, col, 0], col * 6.0 / 3 / u_tile, atol=1e-12)
    assert (np.diff(uv[:, 0, 1]) < 0).all()
    # No jumps: neighbouring stations differ by exactly their spacing / v_tile.
    spacing = np.linalg.norm(np.diff(S_ROAD, axis=0), axis=1)
    np.testing.assert_allclose(np.diff(uv[:, :, 1], axis=0), -np.tile(spacing, (4, 1)).T / v_tile)
    assert np.abs(np.diff(uv[:, :, 1], axis=0)).max() <= spacing.max() / v_tile + 1e-12


def test_ribbon_chunks_continue_uvs_with_s_start() -> None:
    a = L_ROAD[:30]
    b = L_ROAD[29:]
    ra = ribbon(a, 3.0, 3.0, lateral_segments=2)
    rb = ribbon(b, 3.0, 3.0, lateral_segments=2, s_start=float(cumulative_length(a)[-1]))
    np.testing.assert_allclose(ra.uvs.reshape(-1, 3, 2)[-1], rb.uvs.reshape(-1, 3, 2)[0])
    np.testing.assert_allclose(
        ra.positions.reshape(-1, 3, 3)[-1], rb.positions.reshape(-1, 3, 3)[0], atol=1e-9
    )


def test_ribbon_u_spans_strip_when_untiled_and_variable_widths() -> None:
    n = len(S_ROAD)
    lw = np.linspace(2.0, 5.0, n)
    r = ribbon(S_ROAD, lw, 1.0, lateral_segments=4, u_tile_m=None)
    uv = r.uvs.reshape(n, 5, 2)
    np.testing.assert_allclose(uv[:, :, 0], np.tile(np.linspace(0, 1, 5), (n, 1)))
    grid = r.positions.reshape(n, 5, 3)
    widths = np.linalg.norm(grid[:, 0] - grid[:, -1], axis=1)
    np.testing.assert_allclose(widths, lw + 1.0, rtol=0.01)
    assert_well_formed(r)


def test_ribbon_one_sided_shoulder() -> None:
    centre = straight((0, 0, 0), 90.0, 10.0)  # heading east: left is north (-Z)
    r = ribbon(centre, 5.0, -3.5, lateral_segments=1)
    np.testing.assert_allclose(np.unique(np.round(r.positions[:, 2], 9)), [-5.0, -3.5])
    assert_well_formed(r)
    with pytest.raises(ValueError):
        ribbon(centre, 3.0, -3.0)


def test_ribbon_banked_lefts_override() -> None:
    centre = straight((0, 0, 0), 0.0, 10.0)
    bank = np.radians(5.0)
    # Bank the road: the left edge goes up (tilted left vector), normals tilt to the right.
    lefts = np.tile([-np.cos(bank), np.sin(bank), 0.0], (len(centre), 1))
    r = ribbon(centre, 4.0, 4.0, lateral_segments=2, lefts=lefts)
    grid = r.positions.reshape(len(centre), 3, 3)
    np.testing.assert_allclose(grid[:, 0, 1], 4.0 * np.sin(bank), atol=1e-12)
    np.testing.assert_allclose(grid[:, -1, 1], -4.0 * np.sin(bank), atol=1e-12)
    np.testing.assert_allclose(r.normals[:, 0], np.sin(bank), atol=1e-9)
    assert_well_formed(r)


def test_ribbon_world_up_normals() -> None:
    r = ribbon(S_ROAD, 3.0, 3.0, normals="world_up")
    np.testing.assert_allclose(r.normals, np.tile(UP, (r.vertex_count, 1)))
    assert (face_normals(r)[:, 1] > 0.9).all()
    with pytest.raises(ValueError):
        ribbon(S_ROAD, 3.0, 3.0, normals="sideways")  # type: ignore[arg-type]


def test_closed_ribbon_has_no_seam_gap() -> None:
    loop = rounded_rectangle_loop(80.0, 50.0, 12.0, step=1.0)
    n, k = len(loop), 5
    v_tile = 7.0
    r = ribbon(loop, 4.0, 4.0, lateral_segments=k - 1, v_tile_m=v_tile, closed=True)
    assert r.vertex_count == (n + 1) * k
    assert r.triangle_count == n * (k - 1) * 2  # one strip all the way round
    grid = r.positions.reshape(n + 1, k, 3)
    np.testing.assert_array_equal(grid[-1], grid[0])  # the seam row is an exact copy
    assert_well_formed(r)
    assert (face_normals(r)[:, 1] > 0.99).all()
    normals = r.normals.reshape(n + 1, k, 3)
    np.testing.assert_allclose(normals[-1], normals[0], atol=1e-12)
    # v wraps by a whole number of tiles, so the texture has no seam either.
    uv = r.uvs.reshape(n + 1, k, 2)
    tiles = uv[0, 0, 1] - uv[-1, 0, 1]
    assert tiles == pytest.approx(round(tiles), abs=1e-9)
    assert round(tiles) == round(
        float(np.sum(np.linalg.norm(np.diff(np.vstack([loop, loop[:1]]), axis=0), axis=1))) / v_tile
    )
    # A repeated closing point in the input changes nothing.
    r2 = ribbon(
        np.vstack([loop, loop[:1]]), 4.0, 4.0, lateral_segments=k - 1, v_tile_m=v_tile, closed=True
    )
    np.testing.assert_array_equal(r2.positions, r.positions)


def test_closed_ribbon_left_edge_inside_ccw_loop() -> None:
    loop = rounded_rectangle_loop(60.0, 40.0, 10.0, step=1.0, centre=(5.0, 0.0, 5.0))
    r = ribbon(loop, 3.0, 3.0, lateral_segments=1, closed=True)
    grid = r.positions.reshape(len(loop) + 1, 2, 3)
    centre = np.array([5.0, 0.0, 5.0])
    inner = np.linalg.norm(grid[:, 0] - centre, axis=1)
    outer = np.linalg.norm(grid[:, 1] - centre, axis=1)
    # Counter-clockwise loop = left turns: the driver's left is the inside.
    assert (inner < outer).all()


# --------------------------------------------------------------------------------------------
# Offsets and walls
# --------------------------------------------------------------------------------------------


def test_offset_polyline_positive_is_left() -> None:
    north = straight((0, 0, 0), 0.0, 10.0)
    np.testing.assert_allclose(offset_polyline(north, 2.0)[:, 0], -2.0)  # west
    np.testing.assert_allclose(offset_polyline(north, -2.0)[:, 0], 2.0)
    east = straight((0, 1, 0), 90.0, 10.0)
    off = offset_polyline(east, 1.5)
    np.testing.assert_allclose(off[:, 2], -1.5)  # north of an eastbound road
    np.testing.assert_allclose(off[:, 1], 1.0)


def test_offset_polyline_mitres_hard_corner() -> None:
    corner = path((0, 0, 0), 0.0, [Straight(10), Arc(0, 90), Straight(10)], step=5.0)
    out = offset_polyline(corner, -2.0)  # outside of the left turn
    # Both offset legs stay exactly 2 m from their centre legs: x = 2 going north, z = -12
    # (2 m north of the westbound leg at z = -10) going west, meeting at the mitred corner.
    np.testing.assert_allclose(out[:3, 0], 2.0, atol=1e-12)
    np.testing.assert_allclose(out[2:, 2], -12.0, atol=1e-12)
    np.testing.assert_allclose(out[2], [2.0, 0.0, -12.0], atol=1e-12)
    assert miter_scale(corner)[2] == pytest.approx(np.sqrt(2.0))
    assert miter_scale(corner, miter_limit=1.0)[2] == pytest.approx(1.0)


def test_wall_single_sided_faces_left() -> None:
    w = wall(L_ROAD, 2.5, double_sided=False, facing="left", u_tile_m=3.0)
    assert w.vertex_count == 2 * len(L_ROAD)
    assert w.triangle_count == 2 * (len(L_ROAD) - 1)
    assert_well_formed(w)
    tangents, lefts, _ = frames(L_ROAD)
    fn = face_normals(w)
    np.testing.assert_allclose(fn[:, 1], 0.0, atol=1e-12)  # vertical
    # Each quad faces the polyline's left side.
    quad_left = (lefts[:-1] + lefts[1:]).repeat(2, axis=0)
    assert (np.sum(fn * quad_left, axis=1) > 0.9).all()
    ys = w.positions[:, 1].reshape(-1, 2)
    np.testing.assert_allclose(ys[:, 0], 2.5)
    np.testing.assert_allclose(ys[:, 1], 0.0)
    uv = w.uvs.reshape(-1, 2, 2)
    np.testing.assert_allclose(uv[:, 0, 1], 0.0)  # v = 0 at the top
    np.testing.assert_allclose(uv[:, 1, 1], 1.0)
    np.testing.assert_allclose(np.abs(uv[:, 0, 0]), cumulative_length(L_ROAD) / 3.0)


def test_wall_double_sided_has_both_faces() -> None:
    centre = straight((0, 0, 0), 0.0, 10.0)  # north: left = west (-X)
    w = wall(centre, np.linspace(1.0, 2.0, len(centre)), double_sided=True, v_tile_m=1.0)
    assert w.vertex_count == 4 * len(centre)
    assert w.triangle_count == 4 * (len(centre) - 1)
    assert_well_formed(w)  # includes "not mirrored" on both faces
    fn = face_normals(w)
    half = w.triangle_count // 2
    np.testing.assert_allclose(fn[:half], np.tile([-1.0, 0, 0], (half, 1)), atol=1e-12)
    np.testing.assert_allclose(fn[half:], np.tile([1.0, 0, 0], (half, 1)), atol=1e-12)
    # v in metres from the top: the bottom row reads the height.
    np.testing.assert_allclose(w.uvs[1 : 2 * len(centre) : 2, 1], np.linspace(1, 2, len(centre)))


def test_wall_facing_right() -> None:
    centre = straight((0, 0, 0), 0.0, 10.0)
    w = wall(centre, 2.0, double_sided=False, facing="right")
    np.testing.assert_allclose(face_normals(w), np.tile([1.0, 0, 0], (w.triangle_count, 1)))
    assert_well_formed(w)


def test_closed_wall_round_a_loop() -> None:
    loop = rounded_rectangle_loop(40.0, 30.0, 8.0, step=1.0)
    w = wall(loop, 2.0, double_sided=False, closed=True, facing="left")
    n = len(loop)
    assert w.triangle_count == 2 * n
    rows = w.positions.reshape(n + 1, 2, 3)
    np.testing.assert_array_equal(rows[-1], rows[0])
    assert_well_formed(w)
    # Left of a counter-clockwise loop = inwards.
    assert (np.sum(face_normals(w) * (0.0 - centroids(w)), axis=1) > 0).all()


# --------------------------------------------------------------------------------------------
# Quads, signs, boxes, posts
# --------------------------------------------------------------------------------------------


def test_quad_basic() -> None:
    q = quad((0, 1, 0), (1, 0, 0), (0, 1, 0), 2.0, 1.0)
    assert q.vertex_count == 4 and q.triangle_count == 2
    np.testing.assert_allclose(face_normals(q), [[0, 0, 1], [0, 0, 1]])
    assert_well_formed(q)
    np.testing.assert_allclose(q.positions.min(axis=0), [-1.0, 0.5, 0.0])
    np.testing.assert_allclose(q.positions.max(axis=0), [1.0, 1.5, 0.0])


@pytest.mark.parametrize("facing", [0.0, 90.0, 180.0, 250.0])
def test_billboard_faces_viewer_and_reads_correctly(facing: float) -> None:
    centre = np.array([3.0, 2.0, -4.0])
    sign = billboard(centre, facing, 4.0, 1.0)
    n = heading_vector(facing)
    viewer = centre + 10.0 * n
    fn = face_normals(sign)
    assert (np.sum(fn * (viewer - centroids(sign)), axis=1) > 0).all()
    np.testing.assert_allclose(sign.normals, np.tile(n, (4, 1)), atol=1e-12)
    assert_well_formed(sign)
    # Readable: u grows to the viewer's right, v grows downwards.
    looking = -n
    viewer_right = np.cross(looking, WORLD_UP)
    dpdu, dpdv = triangle_uv_axes(sign)
    assert (dpdu @ viewer_right > 0).all()
    assert (dpdv @ UP < 0).all()
    np.testing.assert_allclose(np.sort(np.unique(sign.uvs)), [0.0, 1.0])
    # Full texture: width along u, height along v.
    np.testing.assert_allclose(np.linalg.norm(dpdu, axis=1), 4.0)
    np.testing.assert_allclose(np.linalg.norm(dpdv, axis=1), 1.0)


def test_billboard_south_facing_concrete() -> None:
    sign = billboard((0, 0, 0), (0.0, 0.0, 1.0), 2.0, 2.0)  # front faces south (+Z)
    # Seen from the south (looking north), east (+X) is on the viewer's right.
    by_uv = {tuple(uv): p for uv, p in zip(sign.uvs.tolist(), sign.positions.tolist(), strict=True)}
    np.testing.assert_allclose(by_uv[(0.0, 0.0)], [-1.0, 1.0, 0.0])  # top-left = west, up
    np.testing.assert_allclose(by_uv[(1.0, 1.0)], [1.0, -1.0, 0.0])  # bottom-right = east, down


def test_billboard_double_sided_reads_from_both_sides() -> None:
    sign = billboard((0, 0, 0), 0.0, 3.0, 1.0, double_sided=True)
    assert sign.vertex_count == 8 and sign.triangle_count == 4
    assert_well_formed(sign)
    fn = face_normals(sign)
    np.testing.assert_allclose(fn[:2], np.tile(heading_vector(0.0), (2, 1)), atol=1e-12)
    np.testing.assert_allclose(fn[2:], np.tile(-heading_vector(0.0), (2, 1)), atol=1e-12)
    dpdu, _ = triangle_uv_axes(sign)
    # Front seen from the north (looking south): right = west. Back from the south: right = east.
    assert (dpdu[:2] @ [-1.0, 0, 0] > 0).all()
    assert (dpdu[2:] @ [1.0, 0, 0] > 0).all()


@pytest.mark.parametrize("heading", [0.0, 30.0, 90.0, 211.0])
def test_box_faces_outward(heading: float) -> None:
    centre = np.array([1.0, 2.0, 3.0])
    b = box(centre, (2.0, 1.0, 4.0), heading)
    assert b.vertex_count == 24 and b.triangle_count == 12
    assert_well_formed(b)
    fn = face_normals(b)
    assert (np.sum(fn * (centroids(b) - centre), axis=1) > 0).all()
    # Six distinct axis directions, two triangles each.
    assert len({tuple(np.round(n, 9)) for n in fn}) == 6
    # Corners are at the right distance along the box axes.
    fwd = heading_vector(heading)
    right = np.cross(fwd, WORLD_UP)
    rel = b.positions - centre
    np.testing.assert_allclose(np.abs(rel @ right), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(rel[:, 1]), 0.5, atol=1e-12)
    np.testing.assert_allclose(np.abs(rel @ fwd), 2.0, atol=1e-12)


def test_box_heading_zero_is_axis_aligned() -> None:
    b = box((0, 0, 0), (2.0, 4.0, 6.0))
    np.testing.assert_allclose(b.positions.min(axis=0), [-1, -2, -3])
    np.testing.assert_allclose(b.positions.max(axis=0), [1, 2, 3])


def test_post_stands_on_base() -> None:
    p = post((4.0, 1.0, -2.0), 3.0, width=0.3, heading_deg=45.0)
    assert p.positions[:, 1].min() == pytest.approx(1.0)
    assert p.positions[:, 1].max() == pytest.approx(4.0)
    np.testing.assert_allclose(p.positions.mean(axis=0), [4.0, 2.5, -2.0], atol=1e-12)
    assert_well_formed(p)


# --------------------------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------------------------


def test_flat_grid() -> None:
    g = flat_grid((-10.0, 10.0), (0.0, 5.0), 1.0, y=0.5)
    assert g.vertex_count == 21 * 6
    assert g.triangle_count == 20 * 5 * 2
    np.testing.assert_allclose(face_normals(g), np.tile(UP, (g.triangle_count, 1)))
    np.testing.assert_allclose(g.positions.min(axis=0), [-10.0, 0.5, 0.0])
    np.testing.assert_allclose(g.positions.max(axis=0), [10.0, 0.5, 5.0])
    assert_well_formed(g)
    # Seen from above with north up, east is u+ and south (+Z) is v+.
    np.testing.assert_allclose(g.tangents, np.tile([1.0, 0, 0], (g.vertex_count, 1)))
    np.testing.assert_allclose(g.uvs[:, 1], g.positions[:, 2] / 4.0)


def test_flat_grid_step_is_upper_bound() -> None:
    g = flat_grid((0.0, 10.0), (0.0, 1.0), 3.0)
    xs = np.unique(g.positions[:, 0])
    assert len(xs) == 5 and np.diff(xs).max() <= 3.0


@pytest.mark.parametrize("n", [3, 4, 5, 7, 12, 97, 1000, 65521, MAX_MESH_VERTICES])
def test_grid_with_exact_vertex_count(n: int) -> None:
    g = grid_with_vertex_count(n, (-100.0, 100.0), (-50.0, 50.0), y=1.0)
    assert g.vertex_count == n
    assert len(np.unique(g.indices)) == n  # every vertex is used
    np.testing.assert_allclose(face_normals(g), np.tile(UP, (g.triangle_count, 1)), atol=1e-12)
    assert (np.linalg.norm(face_normals(g, normalize=False), axis=1) > 0).all()
    assert g.positions[:, 0].min() >= -100.0 and g.positions[:, 0].max() <= 100.0
    assert g.positions[:, 2].min() >= -50.0 and g.positions[:, 2].max() <= 50.0
    np.testing.assert_allclose(g.positions[:, 1], 1.0)
    assert (uv_areas(g) < 0).all()


def test_65535_vertex_grid_fits_one_kn5_mesh_and_65536_does_not() -> None:
    g = grid_with_vertex_count(MAX_MESH_VERTICES, (-60.0, 60.0), (-60.0, 60.0))
    mesh = to_kn5_mesh(g, "1ROAD_fine", 0)
    assert mesh.vertex_count == 65535
    assert mesh.indices.dtype == np.uint16
    assert int(mesh.indices.max()) == 65534
    too_big = grid_with_vertex_count(MAX_MESH_VERTICES + 1, (-60.0, 60.0), (-60.0, 60.0))
    with pytest.raises(ValueError, match="split"):
        to_kn5_mesh(too_big, "1ROAD_big", 0)
    pieces = split(too_big)
    assert len(pieces) == 2
    assert all(p.vertex_count <= MAX_MESH_VERTICES for p in pieces)


# --------------------------------------------------------------------------------------------
# Merge, split, normals, tangents, conversion
# --------------------------------------------------------------------------------------------


def test_merge_rebases_indices() -> None:
    a = quad((0, 0, 0), (1, 0, 0), (0, 1, 0), 1.0, 1.0)
    b = box((5, 0, 0), (1, 1, 1))
    m = merge([a, b])
    assert m.vertex_count == 28 and m.triangle_count == 14
    np.testing.assert_array_equal(m.indices[:6], a.indices)
    np.testing.assert_array_equal(m.indices[6:], b.indices + 4)
    assert triangle_set(m) == sorted(triangle_set(a) + triangle_set(b))
    with pytest.raises(ValueError):
        merge([])


@pytest.mark.parametrize("limit", [MAX_MESH_VERTICES, 10_000, 500, 3])
def test_split_respects_limit_and_keeps_all_triangles(limit: int) -> None:
    big = flat_grid((0.0, 280.0), (0.0, 280.0), 1.0)  # 281^2 = 78,961 vertices
    if limit == 3:
        big = flat_grid((0.0, 4.0), (0.0, 3.0), 1.0)
    pieces = split(big, max_vertices=limit)
    assert len(pieces) >= 2
    assert all(p.vertex_count <= limit for p in pieces)
    assert all(p.triangle_count >= 1 for p in pieces)
    assert sum(p.triangle_count for p in pieces) == big.triangle_count
    if limit >= 500:
        assert triangle_set(merge(pieces)) == triangle_set(big)
    for p in pieces:
        # Every vertex in a piece is used, and attributes travel with their vertex.
        assert len(np.unique(p.indices)) == p.vertex_count
        np.testing.assert_allclose(p.uvs[:, 0], p.positions[:, 0] / 4.0)
        np.testing.assert_allclose(face_normals(p)[:, 1], 1.0)


def test_split_greedy_fills_chunks() -> None:
    centre = path((0, 0, 0), 0, [Straight(3000), Arc(200, 120)])
    r = ribbon(centre, 5.0, 5.0, lateral_segments=24)
    pieces = split(r, max_vertices=MAX_MESH_VERTICES)
    assert r.vertex_count > MAX_MESH_VERTICES
    # Greedy: all but the last chunk are (nearly) full.
    assert all(p.vertex_count > MAX_MESH_VERTICES - 10 for p in pieces[:-1])
    assert triangle_set(merge(pieces)) == triangle_set(r)


def test_split_small_mesh_and_bad_limit() -> None:
    q = quad((0, 0, 0), (1, 0, 0), (0, 1, 0), 1.0, 1.0)
    assert split(q) == [q]
    with pytest.raises(ValueError):
        split(q, max_vertices=2)


def test_compute_normals_weld_across_duplicated_vertices() -> None:
    # A shallow roof: two quads meeting at a ridge along X (y = 1, z = 0), built with separate
    # (duplicated) vertices. The north slope faces up-north, the south slope up-south.
    sn, cs = np.sin(np.radians(20.0)), np.cos(np.radians(20.0))
    north = quad((0, 1 - sn / 2, -cs / 2), (-1, 0, 0), (0, sn, cs), 1.0, 1.0)
    south = quad((0, 1 - sn / 2, cs / 2), (1, 0, 0), (0, sn, -cs), 1.0, 1.0)
    np.testing.assert_allclose(north.normals[0], [0, cs, -sn], atol=1e-12)
    np.testing.assert_allclose(south.normals[0], [0, cs, sn], atol=1e-12)
    m = merge([north, south])
    plain = compute_normals(m)
    welded = compute_normals(m, weld_tolerance=1e-6)
    ridge = np.flatnonzero(np.isclose(m.positions[:, 1], 1.0))
    assert len(ridge) == 4
    for i in ridge:
        twin = [j for j in ridge if j != i and np.allclose(m.positions[j], m.positions[i])]
        assert len(twin) == 1
        # Welded: coincident vertices share one normal, tilted less than either slope.
        np.testing.assert_allclose(welded.normals[i], welded.normals[twin[0]], atol=1e-12)
        assert welded.normals[i, 1] > plain.normals[i, 1] + 0.01
        assert not np.allclose(plain.normals[i], plain.normals[twin[0]])
    off_ridge = np.setdiff1d(np.arange(m.vertex_count), ridge)
    np.testing.assert_allclose(welded.normals[off_ridge], plain.normals[off_ridge])
    # A crease sharper than the weld angle is left alone.
    steep = compute_normals(m, weld_tolerance=1e-6, weld_max_angle_deg=30.0)
    np.testing.assert_allclose(steep.normals, plain.normals)


def test_weld_keeps_hard_edges_and_double_sided_faces_apart() -> None:
    b = box((0, 0, 0), (1, 2, 3), 20.0)
    np.testing.assert_allclose(compute_normals(b, weld_tolerance=1e-6).normals, b.normals)
    two_faced = billboard((0, 0, 0), 0.0, 1.0, 1.0, double_sided=True)
    welded = compute_normals(two_faced, weld_tolerance=1e-6)
    np.testing.assert_allclose(welded.normals, two_faced.normals, atol=1e-12)
    w = wall(rounded_rectangle_loop(20, 20, 5, step=1.0), 2.0, closed=True)
    assert_well_formed(compute_normals(w, weld_tolerance=1e-6))


def test_compute_tangents_handles_degenerate_uvs() -> None:
    q = quad((0, 0, 0), (1, 0, 0), (0, 0, -1), 1.0, 1.0)  # faces up
    q.uvs[:] = 0.0  # no usable UV gradient anywhere
    t = compute_tangents(q)
    np.testing.assert_allclose(np.linalg.norm(t.tangents, axis=1), 1.0)
    np.testing.assert_allclose(t.tangents @ UP, 0.0, atol=1e-12)


def test_flipped() -> None:
    q = quad((0, 0, 0), (1, 0, 0), (0, 1, 0), 1.0, 1.0)
    f = flipped(q)
    np.testing.assert_allclose(face_normals(f), -face_normals(q))
    np.testing.assert_allclose(f.normals, -q.normals)


def test_mesh_data_validation() -> None:
    q = quad((0, 0, 0), (1, 0, 0), (0, 1, 0), 1.0, 1.0)
    with pytest.raises(ValueError):
        MeshData(q.positions, q.normals, q.uvs, q.tangents, np.array([0, 1]))
    with pytest.raises(ValueError):
        MeshData(q.positions, q.normals, q.uvs, q.tangents, np.array([0, 1, 4]))
    with pytest.raises(ValueError):
        MeshData(q.positions, q.normals[:3], q.uvs, q.tangents, q.indices)
    moved = q.translated((1.0, 2.0, 3.0))
    np.testing.assert_allclose(moved.positions - q.positions, np.tile([1.0, 2.0, 3.0], (4, 1)))


def test_to_kn5_mesh_types_and_flags() -> None:
    w = wall(straight((0, 0, 0), 0, 25.0), 2.0)
    mesh = to_kn5_mesh(w, "1WALL_L_c000", 3, is_renderable=False, cast_shadows=False)
    assert mesh.name == "1WALL_L_c000" and mesh.material_id == 3
    assert not mesh.is_renderable and not mesh.cast_shadows and mesh.is_visible
    for arr in (mesh.positions, mesh.normals, mesh.uvs, mesh.tangents):
        assert arr.dtype == np.float32
    assert mesh.indices.dtype == np.uint16
    np.testing.assert_array_equal(mesh.indices, w.indices)
    bad = w.copy()
    bad.positions[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        to_kn5_mesh(bad, "x", 0)


def test_meshes_pass_kn5_validation() -> None:
    kn5_module = pytest.importorskip("driftmaps.ac.kn5")
    validate = getattr(kn5_module, "validate", None)
    if validate is None:
        pytest.skip("driftmaps.ac.kn5.validate not available")
    loop = rounded_rectangle_loop(100.0, 60.0, 15.0, step=1.0)
    nodes = [
        to_kn5_mesh(ribbon(L_ROAD, 4.0, 4.0), "1ROAD_c000", 0),
        to_kn5_mesh(ribbon(loop, 3.0, 3.0, closed=True), "1ROAD_c001", 0),
        to_kn5_mesh(
            wall(offset_polyline(L_ROAD, 5.0), 2.0),
            "1WALL_L_c000",
            0,
            is_renderable=False,
            cast_shadows=False,
        ),
        to_kn5_mesh(merge([box((0, 1, 0), (1, 2, 1)), post((3, 0, 0), 2.0)]), "props", 0),
        to_kn5_mesh(billboard((0, 3, -10), 180.0, 4.0, 1.0), "sign", 0),
        to_kn5_mesh(grid_with_vertex_count(MAX_MESH_VERTICES, (-50, 50), (-50, 50)), "1GRASS", 0),
    ]
    kn5 = Kn5(root=Dummy("root", children=list(nodes)), materials=[Material("m")])
    assert validate(kn5) == []
