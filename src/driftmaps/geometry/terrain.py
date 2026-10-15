"""Corridor terrain around the road: multi-resolution points, cut/fill, one triangulation.

Design (PLAN §5.2 "Terrain", §5.3; docs/research/geo.md §2.5, §3.4, §3.5):

- **Points.** Regular grids in rings by distance from the road footprint (default 2 m cells to
  25 m, 5 m to 150 m, 12 m to 500 m, 30 m to 1.5 km), plus the road's outer shoulder edges
  (exactly the shoulder mesh's edge vertices) and the two end caps. Grid points within
  `edge_clearance_m` of the footprint are dropped, so the first terrain triangles join the
  shoulder edge exactly: no crack and no step.
- **One Delaunay triangulation** of all points (scipy/Qhull), so rings of different density
  are stitched without T-junction cracks. Triangles whose centroid or any edge midpoint lies
  inside the road footprint are removed (the road hole), as are triangles with edges longer
  than `max_edge_m` (concave hull spans).
- **Cut/fill** (geo.md §2.5): each grid point's DEM height is clamped between batter planes
  from every nearby road station: `z_edge - fill_slope * d` <= h <= `z_edge + cut_slope * d`,
  with d the distance beyond the footprint edge, widened quadratically beyond `batter_free_m`
  so the clamp fades out with distance. Taking the max/min over all nearby stations handles
  hairpins: legs share one terrain, and where two legs disagree (stacked switchbacks) the
  midpoint is used. Bridge stations put no hole in the terrain, never pull terrain up (no fill
  under a deck) and push it at least `bridge_clearance_m` below the deck, so under a bridge
  the real (lidar) ground shows wherever it lies lower: the creek valley, not a dug pit.
- **Deck safety net:** `clamp_heights` lowers terrain points to a ceiling (e.g.
  `geometry.bridges.deck_ceiling`) and recomputes normals and materials, so no terrain can
  rise into a bridge deck whatever the DEM says.
- **Materials by slope:** near triangles steeper than `dirt_slope_deg` are dirt/rock, the rest
  grass. Near terrain (within `near_m` of the road) is physical (`1GRASS_t###`, `1DIRT_t###`);
  far terrain is visual only.

Works in the AC horizontal plane (X = east, Z = south); heights are AC Y.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.spatial import Delaunay, cKDTree

from driftmaps.geometry.primitives import MeshData, compute_tangents
from driftmaps.geometry.road import RoadModel, footprint_polygon

__all__ = ["Terrain", "TerrainOptions", "build_terrain", "clamp_heights", "terrain_points"]

HeightFn = Callable[[np.ndarray, bool], np.ndarray]


@dataclass(frozen=True)
class TerrainOptions:
    # (outer distance from the footprint edge, grid cell) per ring, inner to outer. ASSUMED
    # (geo.md §3.4 suggests 1-2 m to 30 m, 4 m to 100 m, 10 m to 300 m, 30 m to 2 km).
    rings: tuple[tuple[float, float], ...] = (
        (25.0, 2.0),
        (150.0, 5.0),
        (500.0, 12.0),
        (1500.0, 30.0),
    )
    near_m: float = 150.0  # physical terrain within this distance of the footprint edge
    edge_clearance_m: float = 1.2
    cut_slope: float = 1.2  # rise/run of cut batters above the road (50 deg; ASSUMED)
    fill_slope: float = 0.6  # rise/run of fill batters below the road (31 deg; ASSUMED)
    batter_free_m: float = 20.0
    batter_widen: float = 0.05  # 1/m: quadratic widening of the batter bounds beyond free_m
    influence_m: float = 120.0
    # Terrain under a bridge deck stays this far below the road surface: the deck
    # (geometry.bridges, 1.0 m) + 0.3 m air + the cut batter over the 0.35 m railing overhang
    # (1.2 x 0.35), so the slab and its fascias never touch the ground. The real ground shows
    # wherever it lies lower (it was 3.0 m, which dug a pit under every bridge end).
    bridge_clearance_m: float = 1.8
    dirt_slope_deg: float = 35.0
    max_edge_m: float = 100.0
    station_step: int = 2  # use every n-th station for the cut/fill bounds

    @property
    def far_m(self) -> float:
        return self.rings[-1][0]


@dataclass
class Terrain:
    """The triangulated terrain. `points` (M, 2) XZ, `heights` (M,) Y, `triangles` (T, 3)
    kept triangles, `near` (T,) physical, `dirt` (T,) steep."""

    points: np.ndarray
    heights: np.ndarray
    triangles: np.ndarray
    near: np.ndarray
    dirt: np.ndarray
    normals: np.ndarray
    is_edge: np.ndarray
    delaunay: Delaunay = field(repr=False)
    kept: np.ndarray = field(repr=False)  # (all Delaunay simplices,) bool

    @property
    def positions(self) -> np.ndarray:
        return np.stack([self.points[:, 0], self.heights, self.points[:, 1]], axis=1)

    def height_at(self, xz: np.ndarray) -> np.ndarray:
        """Terrain height under XZ points (NaN outside the kept triangles)."""
        xz = np.asarray(xz, np.float64).reshape(-1, 2)
        simplex = self.delaunay.find_simplex(xz)
        out = np.full(len(xz), np.nan)
        ok = simplex >= 0
        ok[ok] = self.kept[simplex[ok]]
        if not ok.any():
            return out
        t = self.delaunay.transform[simplex[ok]]
        b = np.einsum("ijk,ik->ij", t[:, :2], xz[ok] - t[:, 2])
        bary = np.concatenate([b, 1 - b.sum(axis=1, keepdims=True)], axis=1)
        verts = self.delaunay.simplices[simplex[ok]]
        out[ok] = (bary * self.heights[verts]).sum(axis=1)
        return out

    def mesh(self, select: np.ndarray, uv_tile_m: float = 8.0) -> MeshData | None:
        """The selected triangles as one mesh (planar UVs: u = X / tile, v = Z / tile, which
        is not mirrored seen from above with north up)."""
        tris = self.triangles[select]
        if len(tris) == 0:
            return None
        used = np.unique(tris)
        pos = self.positions[used]
        uvs = np.stack([pos[:, 0], pos[:, 2]], axis=1) / uv_tile_m
        data = MeshData(
            pos, self.normals[used], uvs, np.zeros_like(pos), np.searchsorted(used, tris)
        )
        return compute_tangents(data)


def _grid_in_band(
    tree: cKDTree,
    outer: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    r_min: float,
    r_max: float,
    cell: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid points (aligned to multiples of `cell`) whose footprint distance is in
    [r_min, r_max). Blocks of 64 cells far from the road are skipped before the fine query."""
    block = 64 * cell
    bx = np.arange(math.floor((lo[0] - r_max) / block), math.ceil((hi[0] + r_max) / block))
    bz = np.arange(math.floor((lo[1] - r_max) / block), math.ceil((hi[1] + r_max) / block))
    gx, gz = np.meshgrid(bx, bz, indexing="ij")
    centres = (np.stack([gx.ravel(), gz.ravel()], axis=1) + 0.5) * block
    d, i = tree.query(centres)
    fd = d - outer[i]
    half_diag = block * math.sqrt(0.5)
    wanted = (fd < r_max + half_diag) & (fd > r_min - half_diag - outer.max())
    pts = []
    k = np.arange(64)
    for c in centres[wanted]:
        x0, z0 = c - block / 2
        xx, zz = np.meshgrid(x0 + k * cell, z0 + k * cell, indexing="ij")
        pts.append(np.stack([xx.ravel(), zz.ravel()], axis=1))
    if not pts:
        return np.zeros((0, 2)), np.zeros(0)
    p = np.concatenate(pts)
    p = np.round(p / cell) * cell
    d, i = tree.query(p)
    fd = d - outer[i]
    keep = (fd >= r_min) & (fd < r_max)
    return p[keep], fd[keep]


def terrain_points(
    model: RoadModel, left_outer: np.ndarray, right_outer: np.ndarray, options: TerrainOptions
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(points (M, 2) XZ, footprint distance (M,), is_edge (M,), edge heights (M,) NaN for
    grid points). Edge points are the outer shoulder edges (minus points inside another part
    of the road) and the two end caps; bridge stations get no edge points."""
    xz = model.centre[:, [0, 2]]
    ground = ~model.bridge
    lo, hi = xz.min(axis=0), xz.max(axis=0)

    # Rings by distance to the whole road (so the fine ring also runs under bridge decks);
    # the clearance around the road hole only applies to ground stations.
    tree = cKDTree(xz)
    grids, dists = [], []
    r_min = -np.inf
    for r_max, cell in options.rings:
        p, fd = _grid_in_band(tree, model.outer, lo, hi, r_min, r_max, cell)
        grids.append(p)
        dists.append(fd)
        r_min = r_max
    if ground.any():
        ground_tree = cKDTree(xz[ground])
        for k, p in enumerate(grids):
            d, i = ground_tree.query(p)
            ok = d - model.outer[ground][i] >= options.edge_clearance_m
            grids[k], dists[k] = p[ok], np.maximum(dists[k][ok], 0.0)

    # Edge points: outer shoulder edges of ground stations, plus the end caps.
    seg_ground = ground[:-1] & ground[1:]
    foot = footprint_polygon(left_outer, right_outer, seg_ground)
    inner = foot.buffer(-0.05) if not foot.is_empty else foot
    edges = []
    for edge in (left_outer, right_outer):
        e = edge[ground]
        edges.append(e)
    for i in (0, len(model) - 1):
        a, b = left_outer[i], right_outer[i]
        n = max(2, math.ceil(np.linalg.norm(b - a) / 1.0))
        t = np.linspace(0.0, 1.0, n + 1)[1:-1, None]
        edges.append(a + t * (b - a))
    e3 = np.concatenate(edges)
    # Drop duplicates (merged fold-over points) and points inside another part of the road.
    _, first = np.unique(np.round(e3[:, [0, 2]], 4), axis=0, return_index=True)
    e3 = e3[np.sort(first)]
    if not inner.is_empty:
        e3 = e3[~shapely.contains_xy(inner, e3[:, 0], e3[:, 2])]
    grid = np.concatenate(grids)
    fd = np.concatenate(dists)
    pts = np.concatenate([e3[:, [0, 2]], grid])
    is_edge = np.concatenate([np.ones(len(e3), bool), np.zeros(len(grid), bool)])
    heights = np.concatenate([e3[:, 1], np.full(len(grid), np.nan)])
    fdist = np.concatenate([np.zeros(len(e3)), fd])
    return pts, fdist, is_edge, heights


def _batter_bounds(
    model: RoadModel, pts: np.ndarray, options: TerrainOptions
) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper height bounds (M,) from the batter planes of nearby road stations."""
    step = max(1, options.station_step)
    idx = np.arange(0, len(model), step)
    if idx[-1] != len(model) - 1:
        idx = np.append(idx, len(model) - 1)
    xz = model.centre[idx][:, [0, 2]]
    z = model.centre[idx, 1]
    outer = model.outer[idx]
    bridge = model.bridge[idx]
    tree = cKDTree(xz)
    k = min(48, len(idx))
    lo = np.full(len(pts), -np.inf)
    hi = np.full(len(pts), np.inf)
    for c0 in range(0, len(pts), 100_000):
        p = pts[c0 : c0 + 100_000]
        d, j = tree.query(p, k=k, distance_upper_bound=options.influence_m + outer.max())
        if k == 1:
            d, j = d[:, None], j[:, None]
        valid = np.isfinite(d)
        jj = np.where(valid, j, 0)
        dd = np.maximum(np.where(valid, d, 0.0) - outer[jj], 0.0)
        extra = options.batter_widen * np.maximum(dd - options.batter_free_m, 0.0) ** 2
        zz = z[jj]
        br = bridge[jj]
        lower = np.where(br, -np.inf, zz - options.fill_slope * dd - extra)
        upper = np.where(
            br,
            zz - options.bridge_clearance_m + options.cut_slope * dd + extra,
            zz + options.cut_slope * dd + extra,
        )
        lower = np.where(valid, lower, -np.inf)
        upper = np.where(valid, upper, np.inf)
        lo[c0 : c0 + len(p)] = lower.max(axis=1)
        hi[c0 : c0 + len(p)] = upper.min(axis=1)
    return lo, hi


def build_terrain(
    model: RoadModel,
    left_outer: np.ndarray,
    right_outer: np.ndarray,
    height_fn: HeightFn,
    options: TerrainOptions | None = None,
) -> Terrain:
    """Triangulated corridor terrain. `height_fn(xz (M, 2), far)` returns DEM heights in AC Y
    (relative to the same reference as the road); `far` is True for points beyond `near_m`,
    which may come from a coarser source. NaN heights are filled from the nearest valid point."""
    options = options or TerrainOptions()
    pts, fdist, is_edge, heights = terrain_points(model, left_outer, right_outer, options)
    grid = ~is_edge
    gi = np.flatnonzero(grid)
    far = fdist[gi] >= options.near_m
    dem = np.full(len(gi), np.nan)
    if (~far).any():
        dem[~far] = height_fn(pts[gi[~far]], False)
    if far.any():
        dem[far] = height_fn(pts[gi[far]], True)
    bad = ~np.isfinite(dem)
    if bad.all():
        raise ValueError("no terrain heights: the DEM has no data around the road")
    if bad.any():
        good = np.flatnonzero(~bad)
        _, nn = cKDTree(pts[gi[good]]).query(pts[gi[bad]])
        dem[bad] = dem[good[nn]]
    lo, hi = _batter_bounds(model, pts[gi], options)
    clamped = np.clip(dem, lo, hi)
    conflict = lo > hi
    clamped[conflict] = 0.5 * (lo[conflict] + hi[conflict])
    heights = heights.copy()
    heights[gi] = clamped

    tri = Delaunay(pts)
    simplices = tri.simplices
    a, b, c = (pts[simplices[:, k]] for k in range(3))
    centroid = (a + b + c) / 3.0
    seg_ground = (~model.bridge[:-1]) & (~model.bridge[1:])
    foot = footprint_polygon(left_outer, right_outer, seg_ground)
    keep = np.ones(len(simplices), bool)
    if not foot.is_empty:
        inner = foot.buffer(-0.02)
        keep &= ~shapely.contains_xy(foot, centroid[:, 0], centroid[:, 1])
        for p, q in ((a, b), (b, c), (c, a)):
            mid = 0.5 * (p + q)
            keep &= ~shapely.contains_xy(inner, mid[:, 0], mid[:, 1])
    edge_len = np.max(
        [
            np.linalg.norm(a - b, axis=1),
            np.linalg.norm(b - c, axis=1),
            np.linalg.norm(c - a, axis=1),
        ],
        axis=0,
    )
    keep &= edge_len <= options.max_edge_m
    tris = simplices[keep]

    # Delaunay simplices may be clockwise; make every kept triangle face up (+Y) in AC space.
    pos = np.stack([pts[:, 0], heights, pts[:, 1]], axis=1)
    n = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    flip = n[:, 1] < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    vnorm, dirt = _shading(pts, heights, tris, options.dirt_slope_deg)
    tcent = (pts[tris[:, 0]] + pts[tris[:, 1]] + pts[tris[:, 2]]) / 3.0
    xz = model.centre[:, [0, 2]]
    d, i = cKDTree(xz).query(tcent)
    cfd = d - model.outer[i]
    near = cfd < options.near_m
    return Terrain(
        points=pts,
        heights=heights,
        triangles=tris,
        near=near,
        dirt=dirt,
        normals=vnorm,
        is_edge=is_edge,
        delaunay=tri,
        kept=keep,
    )


def _shading(
    pts: np.ndarray, heights: np.ndarray, tris: np.ndarray, dirt_slope_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """(vertex normals (M, 3), dirt flag per triangle (T,)) of up-facing triangles."""
    pos = np.stack([pts[:, 0], heights, pts[:, 1]], axis=1)
    n = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    vnorm = np.zeros_like(pos)
    for k in range(3):
        np.add.at(vnorm, tris[:, k], n)
    length = np.linalg.norm(vnorm, axis=1)
    vnorm = np.where(
        length[:, None] > 1e-12, vnorm / np.maximum(length, 1e-12)[:, None], [0.0, 1.0, 0.0]
    )
    tri_n = n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    dirt = tri_n[:, 1] < math.cos(math.radians(dirt_slope_deg))
    return vnorm, dirt


def clamp_heights(
    terrain: Terrain, ceiling: np.ndarray, options: TerrainOptions | None = None
) -> tuple[Terrain, int]:
    """Lower every grid point above `ceiling` (M,) (one value per `terrain.points`; +inf = no
    limit, e.g. `geometry.bridges.deck_ceiling(model, plans, terrain.points)`) to it.

    The road's shoulder-edge points are never moved (they must meet the road mesh). Normals
    and the grass/dirt split are recomputed; triangle winding stays valid because lowering
    a point does not change a triangle's orientation seen from above. Returns the new terrain
    and how many points moved."""
    options = options or TerrainOptions()
    ceiling = np.asarray(ceiling, np.float64).reshape(-1)
    if ceiling.shape != terrain.heights.shape:
        raise ValueError(f"need one ceiling per terrain point, got {ceiling.shape}")
    move = (terrain.heights > ceiling) & ~terrain.is_edge
    if not move.any():
        return terrain, 0
    heights = np.where(move, ceiling, terrain.heights)
    normals, dirt = _shading(terrain.points, heights, terrain.triangles, options.dirt_slope_deg)
    out = dataclasses.replace(terrain, heights=heights, normals=normals, dirt=dirt)
    return out, int(move.sum())


def tile_keys(terrain: Terrain, select: np.ndarray, tile_m: float) -> np.ndarray:
    """(T, 2) integer tile index of each triangle's centroid."""
    tris = terrain.triangles
    cent = terrain.points[tris].mean(axis=1)
    return np.floor(cent / tile_m).astype(np.int64)
