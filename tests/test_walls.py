"""Hidden walls along the merged road corridor (geometry/walls.py)."""

from __future__ import annotations

import math

import numpy as np
import shapely

from driftmaps.geometry.road import RoadModel
from driftmaps.geometry.walls import corridor_polygon, wall_lines, wall_meshes


def model_from(xz: np.ndarray) -> RoadModel:
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))])
    centre = np.stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]], axis=1)
    return RoadModel(
        s=s, centre=centre, half_width=3.5, shoulder=1.0, centre_line=True, bridge=False
    )


def hairpin(radius: float, straight: float = 100.0) -> np.ndarray:
    up = np.stack([np.zeros(int(straight)), -np.arange(int(straight), dtype=float)], 1)
    n = math.ceil(math.pi * radius)
    a = np.linspace(math.pi, 0.0, n + 1)
    arc = np.stack([radius + radius * np.cos(a), -straight - radius * np.sin(a)], axis=1)
    down = np.stack([np.full(len(up), 2 * radius), up[::-1, 1]], axis=1)
    return np.concatenate([up, arc[1:-1], down])


def test_hairpin_walls_never_cut_across_the_other_leg():
    model = model_from(hairpin(radius=7.0))
    r = model.outer + 4.0  # 8.5 m: the two legs' corridors overlap (legs 14 m apart)
    corridor = corridor_polygon(model, r)
    lines = wall_lines(corridor)
    assert len(lines.rings) == 1  # one merged outline, nothing between the legs
    pts = np.concatenate(lines.rings)
    d = np.min(np.linalg.norm(pts[:, None, :] - model.centre[None, :, [0, 2]], axis=2), axis=1)
    assert d.min() > 8.5 - 0.05  # every wall point is a full corridor radius from the road
    # The gap between the legs is inside the corridor, i.e. drivable run-off, not a wall.
    assert corridor.contains(shapely.Point(7.0, -50.0))


def test_loop_gets_an_inner_wall():
    t = np.linspace(0.0, 1.8 * math.pi, 400)
    xz = np.stack([60 * np.cos(t), 60 * np.sin(t)], axis=1)
    model = model_from(xz)
    lines = wall_lines(corridor_polygon(model, model.outer + 4.0))
    assert len(lines.rings) == 1  # open loop: one outline around the C shape
    t = np.linspace(0.0, 2.2 * math.pi, 500)  # overlapping ends close the loop
    xz = np.stack([60 * np.cos(t), 60 * np.sin(t) + t], axis=1)
    model = model_from(xz)
    lines = wall_lines(corridor_polygon(model, model.outer + 4.0))
    assert len(lines.rings) == 2  # exterior + the hole inside the loop


def test_wall_meshes_stand_on_the_ground_and_are_double_sided():
    model = model_from(np.stack([np.arange(0.0, 301.0), np.zeros(301)], axis=1))
    lines = wall_lines(corridor_polygon(model, 8.5))
    meshes = wall_meshes(lines, lambda xz: 0.1 * xz[:, 0], height_m=3.0, depth_m=1.0)
    assert len(meshes) >= 7  # ~700 m of wall in 100 m pieces
    for m in meshes:
        y = m.positions[:, 1]
        ground = 0.1 * m.positions[:, 0]
        assert np.allclose(np.sort(np.unique(np.round(y - ground, 6))), [-1.0, 3.0])
        half = m.vertex_count // 2
        assert np.allclose(m.normals[:half], -m.normals[half:])
        assert np.allclose(m.normals[:, 1], 0.0, atol=1e-9)  # vertical
