"""Corridor terrain: hole under the road, shared edges, cut/fill batters, hairpins
(geometry/terrain.py), on synthetic roads and analytic DEMs."""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely

from driftmaps.geometry.road import RoadModel, build_road_meshes, footprint_polygon
from driftmaps.geometry.terrain import TerrainOptions, build_terrain

OPTS = TerrainOptions(rings=((15.0, 2.0), (60.0, 6.0), (200.0, 20.0)), near_m=60.0)


def model_from(xz: np.ndarray, grade: float = 0.0, **kw) -> RoadModel:
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))])
    centre = np.stack([xz[:, 0], grade * s, xz[:, 1]], axis=1)
    kw.setdefault("bridge", False)
    return RoadModel(s=s, centre=centre, half_width=3.5, shoulder=1.0, centre_line=True, **kw)


def hairpin(radius: float = 12.0, straight: float = 120.0) -> np.ndarray:
    up = np.stack([np.zeros(int(straight)), -np.arange(int(straight), dtype=float)], 1)
    n = math.ceil(math.pi * radius)
    a = np.linspace(math.pi, 0.0, n + 1)
    arc = np.stack([radius + radius * np.cos(a), -straight - radius * np.sin(a)], axis=1)
    down = np.stack([np.full(len(up), 2 * radius), up[::-1, 1]], axis=1)
    return np.concatenate([up, arc[1:-1], down])


def build(model: RoadModel, dem, options: TerrainOptions = OPTS):
    rm = build_road_meshes(model)
    terrain = build_terrain(model, rm.left_outer, rm.right_outer, lambda xz, far: dem(xz), options)
    return rm, terrain


def tri_xyz(terrain, select=None):
    tris = terrain.triangles if select is None else terrain.triangles[select]
    return terrain.positions[tris]


def test_straight_road_on_a_side_slope():
    """30% cross slope: the uphill side is cut back, the downhill side filled up, both within
    the batter planes; the terrain meets the shoulder edges exactly and faces up."""
    xz = np.stack([np.arange(0.0, 301.0), np.zeros(301)], axis=1)
    model = model_from(xz)
    rm, terrain = build(model, lambda p: 0.3 * p[:, 1] + 1.0)  # rises towards the south (+Z)
    pos = terrain.positions
    tris = tri_xyz(terrain)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    assert (n[:, 1] > 0).all()
    # Edge vertices are exactly the shoulder edge vertices (no crack, no step).
    edges = np.concatenate([rm.left_outer, rm.right_outer])
    tp = pos[terrain.is_edge]
    d = np.min(np.linalg.norm(tp[:, None, :] - edges[None, ::3, :], axis=2), axis=1)
    ends = (tp[:, 0] < 0.5) | (tp[:, 0] > 299.5)
    assert d[~ends].max() < 1.0 + 1e-6  # on the edge polylines (decimated by 3 above)
    assert np.allclose(tp[:, 1], 0.0, atol=1e-9)
    # Batters: |h - road| <= slope * distance beyond the shoulder edge (+ the widening).
    g = ~terrain.is_edge & (np.abs(pos[:, 2]) < 60.0) & (pos[:, 0] > 20) & (pos[:, 0] < 280)
    dist = np.abs(pos[g, 2]) - 4.5
    widen = OPTS.batter_widen * np.maximum(dist - OPTS.batter_free_m, 0.0) ** 2
    h = pos[g, 1]
    assert (h <= OPTS.cut_slope * dist + widen + 1e-6).all()
    assert (h >= -OPTS.fill_slope * dist - widen - 1e-6).all()
    uphill = g & (pos[:, 2] > 10)
    assert (pos[uphill, 1] < 0.3 * pos[uphill, 2] + 1.0 + 1e-6).all()  # only ever cut down
    # Steep cut batters are dirt, the gentle far slope is grass.
    assert terrain.dirt.any() and not terrain.dirt.all()


def test_hole_under_road_and_height_at():
    xz = np.stack([np.arange(0.0, 201.0), np.zeros(201)], axis=1)
    model = model_from(xz)
    rm, terrain = build(model, lambda p: np.full(len(p), -2.0))
    foot = footprint_polygon(rm.left_outer, rm.right_outer)
    cent = tri_xyz(terrain).mean(axis=1)
    assert not shapely.contains_xy(foot.buffer(-0.01), cent[:, 0], cent[:, 2]).any()
    h = terrain.height_at(np.array([[100.0, 0.0], [100.0, 3.0], [100.0, 30.0]]))
    assert np.isnan(h[0]) and np.isnan(h[1])  # on the road: no terrain
    assert -2.0 - 1e-6 <= h[2] < 0.0  # filled towards the road, but still below it
    # Near (physical) terrain is within near_m of the footprint, far terrain beyond.
    assert terrain.near.any() and (~terrain.near).any()


def test_hairpin_legs_share_one_terrain():
    """Two legs 24 m apart at different heights: one planar triangulation between them (no
    overlapping duplicate terrain), heights within both legs' batters or their midpoint."""
    model = model_from(hairpin(radius=12.0), grade=0.08)
    rm, terrain = build(model, lambda p: np.full(len(p), 5.0))
    tris = tri_xyz(terrain)
    polys = shapely.polygons(tris[:, :, [0, 2]])
    total = shapely.area(polys).sum()
    union = shapely.union_all(polys).area
    assert total == pytest.approx(union, rel=1e-6)  # no two terrain triangles overlap
    # The strip between the legs is covered, below both roads' shoulders where it touches.
    mid = np.array([[12.0, -60.0]])
    h = terrain.height_at(mid)[0]
    assert np.isfinite(h)
    z_up = np.interp(60.0, model.s, model.centre[:, 1])
    z_down = model.centre[
        np.argmin(np.linalg.norm(model.centre[:, [0, 2]] - [24.0, -60.0], axis=1)), 1
    ]
    lo, hi = sorted((z_up, z_down))
    assert lo - 12.0 * OPTS.fill_slope - 1e-6 <= h <= hi + 12.0 * OPTS.cut_slope + 1e-6


def test_bridge_keeps_terrain_below_the_deck():
    xz = np.stack([np.arange(0.0, 301.0), np.zeros(301)], axis=1)
    bridge = (xz[:, 0] > 120) & (xz[:, 0] < 180)
    model = model_from(xz, bridge=bridge)
    rm, terrain = build(model, lambda p: np.full(len(p), 2.0))  # DEM above the deck
    h = terrain.height_at(np.array([[150.0, 0.0], [150.0, 2.0]]))
    assert np.isfinite(h).all()  # terrain carries on under the deck
    assert (h <= -OPTS.bridge_clearance_m + 1e-6).all()
