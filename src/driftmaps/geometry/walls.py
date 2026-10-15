"""Hidden collision walls along the outer boundary of the merged road corridor.

geo.md §3.6 / PLAN §5.3: the corridor is the union of every road station's disc of radius
(asphalt half width + shoulder + verge), built from short buffered pieces of the centreline.
Its boundary (exterior ring plus any holes, e.g. inside a hairpin loop) becomes the wall line,
so walls never cut across another leg of the road: every wall point is at least the corridor
radius from the nearest centreline point. The round caps of the union close both ends of the
road, which gives the end walls behind the run-up and after the run-off.

Walls are double-sided vertical strips (`primitives.wall`) from `depth_m` below the terrain
to `height_m` above it, cut into ~`chunk_m` pieces named `1WALL_c###`, and are never rendered
(ac_format.md §3.2). Multi-level corridors (a bridge over its own road) are not split into
layers yet: such a crossing would get one merged footprint (rare on touge routes).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import shapely

from driftmaps.geometry.primitives import MeshData, wall
from driftmaps.geometry.road import RoadModel

__all__ = ["WallLines", "corridor_polygon", "wall_lines", "wall_meshes"]


def corridor_polygon(
    model: RoadModel, radius: np.ndarray | float, piece_m: float = 10.0, quad_segs: int = 8
) -> shapely.Geometry:
    """Union of the centreline buffered by `radius` (scalar or per station), in XZ.

    The centreline is cut into pieces of ~`piece_m`; each piece is buffered by the largest
    radius on it (round joins and caps), so the union contains every station's disc.
    """
    xz = model.centre[:, [0, 2]]
    r = np.broadcast_to(np.asarray(radius, np.float64), (len(model),))
    n = len(model)
    step = max(1, int(round(piece_m / max(np.median(np.diff(model.s)), 1e-6))))
    pieces = []
    for i0 in range(0, n - 1, step):
        i1 = min(n - 1, i0 + step)
        line = shapely.LineString(xz[i0 : i1 + 1])
        pieces.append(line.buffer(float(r[i0 : i1 + 1].max()), quad_segs=quad_segs))
    return shapely.union_all(pieces)


@dataclass
class WallLines:
    """Closed wall rings in XZ (each (K, 2), last point == first)."""

    rings: list[np.ndarray]

    @property
    def length(self) -> float:
        return float(sum(np.linalg.norm(np.diff(r, axis=0), axis=1).sum() for r in self.rings))


def wall_lines(corridor: shapely.Geometry, spacing_m: float = 2.0) -> WallLines:
    """Exterior and interior rings of the corridor, densified to `spacing_m`."""
    polys = list(corridor.geoms) if isinstance(corridor, shapely.MultiPolygon) else [corridor]
    rings = []
    for poly in polys:
        if poly.is_empty:
            continue
        for ring in [poly.exterior, *poly.interiors]:
            if ring.length < 5.0:  # slivers
                continue
            dense = shapely.segmentize(ring, spacing_m)
            rings.append(np.asarray(dense.coords)[:, :2])
    return WallLines(rings)


def wall_meshes(
    lines: WallLines,
    ground_fn: Callable[[np.ndarray], np.ndarray],
    *,
    height_m: float = 3.0,
    depth_m: float = 1.0,
    chunk_m: float = 100.0,
) -> list[MeshData]:
    """Double-sided wall strips along `lines`, standing on `ground_fn(xz)` heights."""
    out: list[MeshData] = []
    for ring in lines.rings:
        ground = np.asarray(ground_fn(ring), np.float64)
        seg = np.linalg.norm(np.diff(ring, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        count = max(1, math.ceil(s[-1] / chunk_m))
        cuts = np.unique(np.searchsorted(s, np.linspace(0.0, s[-1], count + 1)))
        cuts[-1] = len(ring) - 1
        for a, b in zip(cuts[:-1], cuts[1:], strict=True):
            if b <= a:
                continue
            base = np.stack(
                [ring[a : b + 1, 0], ground[a : b + 1] - depth_m, ring[a : b + 1, 1]], 1
            )
            keep = np.concatenate(
                [[True], np.linalg.norm(np.diff(base[:, [0, 2]], axis=0), axis=1) > 1e-6]
            )
            base = base[keep]
            if len(base) < 2:
                continue
            out.append(wall(base, height_m + depth_m, double_sided=True, u_tile_m=4.0))
    return out
