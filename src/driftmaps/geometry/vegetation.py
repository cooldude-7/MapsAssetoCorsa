"""Trees beside the road: where they really stand, how tall, what kind, and their KN5 cards.

Decisions: docs/research/scenery.md §1, §2, §4.1 and §7 ("Tree positions", "Tree species",
"Tree mesh", "Budgets").

**Placement** (`place_trees`), in AC coordinates (X = east, Z = south):

1. The canopy height model (CHM = lidar DSM - DTM, metres above bare ground; NRCan HRDEM,
   OGL-Canada) is sampled on a 1 m lattice out to `near_m` from the road footprint and a 2 m
   lattice out to `far_m`, in 256 m tiles with a halo, only where the band needs it.
   `CanopySampler` wraps `elevation.sources.canopy_height_grid`; tests pass any XZ -> CHM
   function.
2. Tree tops are local maxima of the CHM smoothed with a 1 m Gaussian, at least `min_height_m`
   (3 m) tall. The window grows with height (3 m below 8 m, 5 m to 15 m, 7 m above). A top must
   rise `min_prominence_m` above the lowest point of its window (a flat roof or a flat canopy
   plateau is not a crown top), and on the 1 m lattice most of its 3 x 3 neighbourhood must be at
   least half its height (rejects poles and wires, which are one pixel wide). Ties are broken by
   a hash of the global cell, so tiles agree in their overlap.
3. Gaps: canopy >= 3 m with no top within `gap_distance_m` (4 m) gets blue-noise points about
   `gap_radius_m` (4.5 m) apart, so a closed canopy with no distinct crowns still gets trees.
   Both distances grow to the height class's window (5 m, 7 m) in taller canopy: tops there
   stand about a window apart, and the 4 m rule would fill every space between them (measured on
   Snake Rd: 50 tops/ha became 324 trees/ha).
4. Exclusions: the road corridor (road + shoulder + `clearance_m`); tops less than `snap_m`
   inside it are pushed out to its edge instead (roadside trees whose crowns lean over the road
   are the most visible ones). Bridges (`RoadModel.bridge`, padded): the DSM sees the deck as a
   10 m "canopy". Anything in `exclude` (building footprints buffered 2 m, water, or the wall
   corridor for the stricter rule in scenery.md §4.1).
5. Crown radius = `crown_factor` x distance to the nearest other tree, clamped to 1.5-6 m (and
   to the height). Low-crowned kinds (conifers, shrubs) never reach over the road footprint.
6. Kind from NRCan Land Cover of Canada 2020 (30 m, OGL-Canada) at the tree, scenery.md §7:
   needleleaf 1/2 -> 90 % conifer; broadleaf 5 -> 90 % deciduous; mixed 6 -> 50/50;
   shrub/wetland 8/14 -> 80 % shrub + 20 % small deciduous below 5 m (taller ones as mixed);
   anything else -> 80 % deciduous. No land cover -> the mixed rule. Then an atlas tile of that
   kind (narrow crowns prefer the poplar tile).
7. Far band: one tree in `1 / far_keep` (3) kept, crowns x `far_scale` (1.5), as "clumps".
8. Budget: at most `max_trees` (40,000) and `max_trees_per_km` (6,000), of which `far_share`
   (30 %) is kept for the far band if it needs it, so hillsides beyond the near band don't go
   bare. Within a band the farthest, smallest and gap-fill trees go first.

Every random choice comes from a hash of (seed, quantised position, purpose), so the output is
deterministic, independent of tile order, and stable when the budget changes. The seed should
come from the route id (`place_trees(..., seed=track_id)`).

**Fallback without lidar** (outside Canada, or a coverage hole): OSM `landuse=forest`,
`natural=wood` and `natural=scrub` polygons (`woods_query` + `woods_from_overpass`) are filled
with jittered-lattice trees `osm_spacing_m` apart, heights drawn from a seeded range, only where
the CHM is NaN.

**Meshes** (`tree_meshes`, `tree_scene`): 2 crossed vertical cards per tree (3 for big near
deciduous trees), each card's 4 vertices shared by the front and the mirrored back triangles
(AC has no double-sided materials), baked "volumetric" normals (from the crown centre, blended
towards +Y, with a per-tree blend for brightness variety), a constant tangent. Card bottoms
follow the terrain at each corner and are sunk `sink_m`; the top is the trunk's terrain height +
the CHM height. Trees are batched per 250 m cell and band into visual-only meshes named
`trees_near_<n>` / `trees_far_<n>` (no digit prefix, no collision), at most 2,000 trees per mesh;
near meshes `lod_out` 700 m with shadows, far meshes 1,500 m without. One `ksTree` material
(Kunos' own NJMP settings: alpha-tested, blend opaque, ksAlphaRef 0.5, ksAmbient 0.4,
ksDiffuse 0.25, ksSpecular 0) on the procedural atlas from `driftmaps.ac.tree_textures`.

There is no CSP output here: CSP `[TREES]` needs `.bin` models that only Content Manager's tree
converter makes (scenery.md §4.2), so v1 is KN5 cards only, which CSP also renders.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import shapely
from rasterio.warp import transform as warp_transform
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from driftmaps.ac.model import (
    MAX_MESH_VERTICES,
    BlendMode,
    DepthMode,
    Material,
    MaterialProperty,
    TextureMapping,
)
from driftmaps.ac.tree_textures import (
    ATLAS_SEED,
    ATLAS_SIZE,
    BUSH,
    CONIFER,
    DECIDUOUS,
    TREE_TILES,
    tile_uv,
    tree_atlas_texture,
)
from driftmaps.elevation.sources import (
    ElevationSource,
    SourceInfo,
    SourceUnavailable,
    _CogRaster,
    canopy_height_grid,
    default_cache_dir,
    region_of,
)
from driftmaps.geometry.primitives import MeshData, MeshFlags
from driftmaps.geometry.road import RoadModel

if TYPE_CHECKING:
    from driftmaps.ac.scene import Scene
    from driftmaps.geo.project import LocalProjection

__all__ = [
    "BUSH",
    "CONIFER",
    "DECIDUOUS",
    "LANDCOVER_ATTRIBUTION",
    "LANDCOVER_URL",
    "MIXED",
    "OSM_WOODS_ATTRIBUTION",
    "CanopyFn",
    "CanopySampler",
    "GroundFn",
    "LandcoverFn",
    "LandcoverSampler",
    "TreeMesh",
    "TreeMeshOptions",
    "TreeSet",
    "TreeStats",
    "VegetationOptions",
    "WoodArea",
    "detect_tree_tops",
    "place_trees",
    "sample_landcover",
    "tree_material",
    "tree_meshes",
    "tree_scene",
    "woods_from_overpass",
    "woods_query",
]

log = logging.getLogger(__name__)

CanopyFn = Callable[[np.ndarray], np.ndarray]
"""(N, 2) AC XZ -> (N,) canopy height above bare ground in metres; NaN = no lidar there."""
LandcoverFn = Callable[[np.ndarray], np.ndarray]
"""(N, 2) AC XZ -> (N,) land cover class (NRCan 2020 codes); 0 = unknown."""
GroundFn = Callable[[np.ndarray], np.ndarray]
"""(N, 2) AC XZ -> (N,) terrain Y (the built terrain mesh); NaN = no ground (tree dropped)."""

MIXED = 3  # a kind hint meaning "50/50 deciduous/conifer" (OSM leaf_type=mixed, no land cover)

SOURCE_LIDAR = 0  # a lidar tree top
SOURCE_OSM = 1  # OSM woodland fill (no lidar there)
SOURCE_GAP = 2  # lidar canopy with no distinct top (gap fill)
BAND_NEAR = 0
BAND_FAR = 1
BAND_NAMES = ("near", "far")

# NRCan Land Cover of Canada 2020, 30 m COG (VERIFIED live 2026-10-15: EPSG:3979, uint8,
# NoData 0, 512 px tiles; docs/research/scenery.md §2).
LANDCOVER_URL = (
    "https://datacube-prod-data-public.s3.ca-central-1.amazonaws.com/store/land/landcover/"
    "landcover-2020-classification.tif"
)
LANDCOVER_ATTRIBUTION = (
    "Land cover: Natural Resources Canada, Land Cover of Canada 2020 (30 m). Contains "
    "information licensed under the Open Government Licence - Canada "
    "(https://open.canada.ca/en/open-government-licence-canada)."
)
OSM_WOODS_ATTRIBUTION = (
    "Woodland outlines (c) OpenStreetMap contributors, ODbL 1.0 (openstreetmap.org/copyright)"
)
_LC_NEEDLE = (1, 2)
_LC_BROAD = (5,)
_LC_MIXED = (6,)
_LC_SHRUB = (8, 14)
_LC_WATER = (18,)


# ---------------------------------------------------------------------------------------------
# Options and results
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VegetationOptions:
    near_m: float = 150.0  # near band: 1 m CHM lattice, individual trees
    far_m: float = 400.0  # far band: 2 m lattice, thinned clumps (<= near_m: no far band)
    near_cell_m: float = 1.0
    far_cell_m: float = 2.0
    tile_m: float = 256.0  # processing tile (plus a halo)
    clearance_m: float = 1.5  # beyond road + shoulder: no trunks in here
    snap_m: float = 3.0  # tops up to this far inside the clearance are pushed out to its edge
    bridge_pad_m: float = 10.0  # bridge stations padded this far along the road...
    bridge_margin_m: float = 6.0  # ...and no trees within this distance beyond their footprint
    min_height_m: float = 3.0
    shrub_min_height_m: float = 1.5  # in land cover 8 (shrubland) / 14 (wetland)
    max_height_m: float = 45.0  # CHM clipped here (towers, cranes)
    smooth_sigma_m: float = 1.0
    # (height below which, window in metres) for the local-maximum search, ascending.
    windows: tuple[tuple[float, float], ...] = ((8.0, 3.0), (15.0, 5.0), (math.inf, 7.0))
    min_prominence_m: float = 0.3
    min_fill: float = 0.55  # share of the 3 x 3 cells >= half the top's height (1 m lattice)
    gap_distance_m: float = 4.0
    gap_radius_m: float = 4.5
    crown_factor: float = 0.6
    crown_min_m: float = 1.5
    crown_max_m: float = 6.0
    far_keep: float = 1.0 / 3.0
    far_scale: float = 1.5
    max_trees: int = 40_000
    max_trees_per_km: float = 6_000.0
    far_share: float = 0.3  # of the budget kept for the far band (unused share goes to near)
    osm_spacing_m: float = 5.5
    osm_height_m: tuple[float, float] = (8.0, 22.0)
    osm_shrub_height_m: tuple[float, float] = (1.5, 3.5)


@dataclass
class TreeSet:
    """Placed trees, one row per tree (arrays of equal length).

    `xz` trunk position (AC X, Z); `height` card height above the trunk's ground (m); `radius`
    crown radius (m, card half width before the tile's fill factor); `kind` DECIDUOUS / CONIFER /
    BUSH; `tile` index into `tree_textures.TREE_TILES`; `distance` from the road footprint edge
    (m); `band` 0 near / 1 far; `source` 0 lidar top / 1 OSM / 2 lidar gap fill; `angle` card
    rotation (rad); `flip` mirror the texture; `shade` 0..1 normal-blend for brightness
    variety. `attributions` are LICENSE-DATA.txt lines for the data used.
    """

    xz: np.ndarray
    height: np.ndarray
    radius: np.ndarray
    kind: np.ndarray
    tile: np.ndarray
    distance: np.ndarray
    band: np.ndarray
    source: np.ndarray
    angle: np.ndarray
    flip: np.ndarray
    shade: np.ndarray
    attributions: list[str] = field(default_factory=list)

    _COLUMNS = (
        "xz", "height", "radius", "kind", "tile", "distance", "band", "source", "angle",
        "flip", "shade",
    )

    def __len__(self) -> int:
        return int(self.xz.shape[0])

    def subset(self, mask: np.ndarray) -> TreeSet:
        """The trees selected by a boolean mask or an index array."""
        cols = {c: getattr(self, c)[mask] for c in self._COLUMNS}
        return TreeSet(**cols, attributions=list(self.attributions))

    @classmethod
    def empty(cls, attributions: Sequence[str] = ()) -> TreeSet:
        z = np.zeros(0)
        i = np.zeros(0, np.int16)
        return cls(
            np.zeros((0, 2)), z, z.copy(), i, i.copy(), z.copy(), i.copy(), i.copy(), z.copy(),
            np.zeros(0, bool), z.copy(), list(attributions),
        )

    def counts(self) -> dict[str, int]:
        """Trees per band and kind, for build summaries."""
        out = {"total": len(self)}
        for b, name in enumerate(BAND_NAMES):
            out[name] = int((self.band == b).sum())
        for k, name in enumerate(("deciduous", "conifer", "bush")):
            out[name] = int((self.kind == k).sum())
        out["osm"] = int((self.source == SOURCE_OSM).sum())
        out["gap"] = int((self.source == SOURCE_GAP).sum())
        return out


# ---------------------------------------------------------------------------------------------
# Deterministic hashing
# ---------------------------------------------------------------------------------------------

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def _seed_int(seed: int | str) -> int:
    """A 32-bit seed from an int or a string (e.g. the track id)."""
    if isinstance(seed, str):
        return int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:4], "little")
    return int(seed) & 0xFFFFFFFF


def _mix(*keys: np.ndarray | int) -> np.ndarray:
    """SplitMix64-style hash of integer keys (arrays broadcast), as uint64."""
    with np.errstate(over="ignore"):
        h = np.atleast_1d(np.uint64(0x6A09E667F3BCC909))
        for k in keys:
            k = np.atleast_1d(np.asarray(k)).astype(np.int64).astype(np.uint64)
            h = h ^ (k + _GOLDEN + (h << np.uint64(6)) + (h >> np.uint64(2)))
            h = (h ^ (h >> np.uint64(30))) * _M1
            h = (h ^ (h >> np.uint64(27))) * _M2
            h = h ^ (h >> np.uint64(31))
    return h


def _u01(*keys: np.ndarray | int) -> np.ndarray:
    """Uniform [0, 1) floats from a hash of integer keys."""
    return (_mix(*keys) >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def _pos_keys(xz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions quantised to 1 cm, as hash keys."""
    q = np.rint(np.asarray(xz) * 100.0).astype(np.int64)
    return q[:, 0], q[:, 1]


# Hash "purposes" (salts), so different random choices are independent.
_S_JITTER, _S_GAP, _S_KIND, _S_TILE, _S_FAR, _S_ANGLE, _S_FLIP, _S_SHADE, _S_OSM = range(1, 10)


# ---------------------------------------------------------------------------------------------
# The road footprint
# ---------------------------------------------------------------------------------------------


def _pad_stations(mask: np.ndarray, s: np.ndarray, pad_m: float) -> np.ndarray:
    """Grow a boolean station mask by `pad_m` metres of arc length on both sides."""
    if not mask.any():
        return mask.copy()
    on = s[mask]
    idx = np.searchsorted(on, s)
    before = np.abs(s - on[np.clip(idx - 1, 0, len(on) - 1)])
    after = np.abs(on[np.clip(idx, 0, len(on) - 1)] - s)
    return np.minimum(before, after) <= pad_m


class _Footprint:
    """Distances to the road footprint (asphalt + shoulder) and to bridge decks, in XZ."""

    def __init__(self, model: RoadModel, bridge_pad_m: float) -> None:
        self.xz = model.centre[:, [0, 2]].astype(np.float64)
        self.outer = np.asarray(model.outer, np.float64)
        self.tree = cKDTree(self.xz)
        bridge = _pad_stations(np.asarray(model.bridge, bool), model.s, bridge_pad_m)
        self.bridge_tree = cKDTree(self.xz[bridge]) if bridge.any() else None
        self.bridge_outer = self.outer[bridge]
        self.length_m = float(model.s[-1] - model.s[0]) if len(model.s) > 1 else 0.0

    def edge_distance(self, xz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(distance beyond the footprint edge, nearest station) per point; < 0 = on it."""
        if len(xz) == 0:
            return np.zeros(0), np.zeros(0, np.int64)
        d, i = self.tree.query(xz)
        return d - self.outer[i], i

    def bridge_distance(self, xz: np.ndarray) -> np.ndarray:
        """Distance beyond the footprint edge of the nearest bridge station (inf: no bridge)."""
        if self.bridge_tree is None or len(xz) == 0:
            return np.full(len(xz), np.inf)
        d, i = self.bridge_tree.query(xz)
        return d - self.bridge_outer[i]

    def coarse_edge_distance(self, x: np.ndarray, z: np.ndarray, step: int = 4) -> np.ndarray:
        """`edge_distance` on a lattice (x along columns, z along rows), computed exactly every
        `step` cells and interpolated linearly in between (distance fields are 1-Lipschitz)."""
        ny, nx = len(z), len(x)
        ri = np.unique(np.r_[np.arange(0, ny, step), ny - 1])
        ci = np.unique(np.r_[np.arange(0, nx, step), nx - 1])
        gz, gx = np.meshgrid(z[ri], x[ci], indexing="ij")
        d, _ = self.edge_distance(np.stack([gx.ravel(), gz.ravel()], axis=1))
        d = d.reshape(len(ri), len(ci))
        fr = np.interp(np.arange(ny), ri, np.arange(len(ri)))
        fc = np.interp(np.arange(nx), ci, np.arange(len(ci)))
        rr, cc = np.meshgrid(fr, fc, indexing="ij")
        return ndi.map_coordinates(d, [rr, cc], order=1, mode="nearest")


# ---------------------------------------------------------------------------------------------
# Tree-top detection on one CHM tile
# ---------------------------------------------------------------------------------------------


def _odd_cells(size_m: float, cell_m: float) -> int:
    return max(3, 2 * int(round((size_m / cell_m - 1.0) / 2.0)) + 1)


def detect_tree_tops(
    chm: np.ndarray,
    cell_m: float,
    *,
    options: VegetationOptions | None = None,
    origin: tuple[int, int] = (0, 0),
    seed: int = 0,
    min_height_m: float | None = None,
    check_fill: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tree tops in a CHM grid: (rows, cols, height) of each top, row-major order. The height
    is the highest raw CHM cell in the top's 3 x 3 neighbourhood (smoothing lowers apexes).

    `chm` (ny, nx) metres above ground, NaN = no data; `cell_m` its spacing. `origin` = the
    grid's (row, col) on the global lattice (tie-breaking hash), `min_height_m` defaults to
    `options.min_height_m`, `check_fill` (the pole/wire test) defaults to cells <= 1 m.
    See the module docstring, step 2.
    """
    o = options or VegetationOptions()
    min_h = o.min_height_m if min_height_m is None else min_height_m
    check_fill = cell_m <= 1.0 if check_fill is None else check_fill
    chm = np.asarray(chm, np.float64)
    valid = np.isfinite(chm)
    h0 = np.where(valid, np.minimum(chm, o.max_height_m), 0.0)
    hs = _smoothed(chm, cell_m, o)
    ny, nx = hs.shape
    gi = (np.arange(ny) + origin[0])[:, None]
    gj = (np.arange(nx) + origin[1])[None, :]
    hj = hs + 1e-4 * _u01(gi, gj, seed, _S_JITTER).reshape(ny, nx)

    # Height estimate per cell: the highest raw CHM in its 3 x 3 neighbourhood (smoothing would
    # lower small crowns by half). It picks the window and passes the height thresholds.
    peak = ndi.maximum_filter(h0, size=3, mode="constant", cval=0.0)
    is_top = np.zeros(hs.shape, bool)
    lower = 0.0
    for upper, size_m in o.windows:
        n = _odd_cells(size_m, cell_m)
        band = (peak >= max(lower, min_h)) & (peak < upper)
        lower = upper
        if not band.any():
            continue
        mx = ndi.maximum_filter(hj, size=n, mode="constant", cval=0.0)
        mn = ndi.minimum_filter(h0, size=n, mode="nearest")
        is_top |= band & (hj >= mx) & (peak - mn >= o.min_prominence_m)
    # A top next to no-data (a masked bridge, the edge of the lidar) is a cut crown.
    is_top &= _inner(valid)
    rows, cols = np.nonzero(is_top)
    if check_fill and len(rows):
        # Share of the 3 x 3 raw cells at least half the top's height.
        pad = np.pad(h0, 1)
        half = 0.5 * peak[rows, cols]
        count = np.zeros(len(rows))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                count += pad[rows + 1 + dr, cols + 1 + dc] >= half
        good = count / 9.0 >= o.min_fill
        rows, cols = rows[good], cols[good]
    return rows, cols, peak[rows, cols]


def _inner(valid: np.ndarray) -> np.ndarray:
    """Valid cells whose 3 x 3 neighbourhood is all valid (the grid border counts as valid)."""
    return ndi.binary_erosion(valid, structure=np.ones((3, 3), bool), border_value=1)


def _raw_peak(h0: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Highest raw value in the 3 x 3 neighbourhood of each (row, col)."""
    pad = np.pad(h0, 1)
    peak = np.zeros(len(rows))
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            peak = np.maximum(peak, pad[rows + 1 + dr, cols + 1 + dc])
    return peak


def _gap_points(
    hs: np.ndarray,
    valid: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    cell_m: float,
    origin: tuple[int, int],
    o: VegetationOptions,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Blue-noise fill points (rows, cols) for canopy >= min height with no top nearby.

    Per height class (tallest first), "nearby" and the fill spacing are the larger of
    `gap_distance_m` / `gap_radius_m` and that class's search window, so the space between
    detected crowns (about one window apart) is not filled; only real holes are, such as a flat
    canopy with no distinct tops.
    """
    # Opening with 3 x 3 cells drops canopy narrower than ~3 cells (wires, poles, fences).
    canopy = ndi.binary_opening(valid & (hs >= o.min_height_m), structure=np.ones((3, 3)))
    canopy &= _inner(valid)
    out_r: list[np.ndarray] = []
    out_c: list[np.ndarray] = []
    have = np.stack([rows, cols], 1).astype(np.float64) * cell_m
    bounds = [0.0] + [w[0] for w in o.windows]
    classes = [(bounds[k], bounds[k + 1], o.windows[k][1]) for k in range(len(o.windows))]
    for lo, hi, window in reversed(classes):
        sel = canopy & (hs >= lo) & (hs < hi)
        if not sel.any():
            continue
        cr, cc = np.nonzero(sel)
        scale = max(o.gap_distance_m, window)
        radius = max(o.gap_radius_m, window)
        if len(have):
            d, _ = cKDTree(have).query(np.stack([cr, cc], 1) * cell_m)
            far = d > scale
            cr, cc = cr[far], cc[far]
        if len(cr) == 0:
            continue
        gi, gj = cr + origin[0], cc + origin[1]
        # One candidate per global block of `radius`: the cell with the lowest hash priority.
        block = max(1, int(round(radius / cell_m)))
        bi, bj = np.floor_divide(gi, block), np.floor_divide(gj, block)
        prio = _u01(gi, gj, seed, _S_GAP)
        order = np.lexsort((prio, bj, bi))
        bi_s, bj_s = bi[order], bj[order]
        first = np.ones(len(order), bool)
        first[1:] = (bi_s[1:] != bi_s[:-1]) | (bj_s[1:] != bj_s[:-1])
        pick = order[first]
        # Drop neighbours closer than 0.7 radius (greedy, by priority).
        pick = pick[np.argsort(prio[pick], kind="stable")]
        pts = np.stack([cr[pick], cc[pick]], 1) * cell_m
        keep = np.ones(len(pick), bool)
        if len(pick) > 1:
            for a, b in sorted(cKDTree(pts).query_pairs(0.7 * radius)):
                if keep[a] and keep[b]:
                    keep[max(a, b)] = False
        pick = pick[keep]
        out_r.append(cr[pick])
        out_c.append(cc[pick])
        have = np.concatenate([have, pts[keep]])
    if not out_r:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(out_r), np.concatenate(out_c)


# ---------------------------------------------------------------------------------------------
# Lidar candidates over the corridor
# ---------------------------------------------------------------------------------------------


@dataclass
class _Candidates:
    xz: np.ndarray
    height: np.ndarray
    band: np.ndarray
    source: np.ndarray
    hint: np.ndarray  # -1 = land cover decides; else DECIDUOUS / CONIFER / BUSH / MIXED

    @classmethod
    def concat(cls, parts: Sequence[_Candidates]) -> _Candidates:
        if not parts:
            z = np.zeros(0)
            i = np.zeros(0, np.int16)
            return cls(np.zeros((0, 2)), z, i, i.copy(), i.copy())
        return cls(
            np.concatenate([p.xz for p in parts]),
            np.concatenate([p.height for p in parts]),
            np.concatenate([p.band for p in parts]),
            np.concatenate([p.source for p in parts]),
            np.concatenate([p.hint for p in parts]),
        )


def _tiles_for_band(
    fp: _Footprint, lo: float, hi: float, cell: float, o: VegetationOptions
) -> list[tuple[int, int, int]]:
    """(tile row, tile col, cells per tile) of every tile that can hold cells of the band."""
    t_cells = max(16, int(round(o.tile_m / cell)))
    t_m = t_cells * cell
    reach = hi + float(fp.outer.max()) + t_m
    xmin, zmin = fp.xz.min(axis=0) - reach
    xmax, zmax = fp.xz.max(axis=0) + reach
    ti = np.arange(math.floor(zmin / t_m), math.ceil(zmax / t_m))
    tj = np.arange(math.floor(xmin / t_m), math.ceil(xmax / t_m))
    TI, TJ = np.meshgrid(ti, tj, indexing="ij")
    centre = np.stack([(TJ.ravel() + 0.5) * t_m, (TI.ravel() + 0.5) * t_m], axis=1)
    d, i = fp.tree.query(centre)
    half_diag = t_m * math.sqrt(0.5)
    d_min = d - half_diag - fp.outer[i]
    d_max = d + half_diag - float(fp.outer.min())
    ok = (d_min <= hi) & (d_max >= lo)
    return [(int(a), int(b), t_cells) for a, b in zip(TI.ravel()[ok], TJ.ravel()[ok], strict=True)]


def _lidar_candidates(
    fp: _Footprint,
    canopy: CanopyFn,
    band: int,
    lo: float,
    hi: float,
    cell: float,
    o: VegetationOptions,
    seed: int,
) -> _Candidates:
    """Tree tops + gap fill for one band (edge distance in (lo, hi]) on a `cell` lattice."""
    halo_cells = int(math.ceil((max(w for _, w in o.windows) + 4 * o.smooth_sigma_m) / cell))
    halo_m = halo_cells * cell
    tiles = _tiles_for_band(fp, lo, hi, cell, o)
    grids = []
    points = []
    for ti, tj, n in tiles:
        r0, c0 = ti * n - halo_cells, tj * n - halo_cells
        size = n + 2 * halo_cells
        x = (c0 + np.arange(size)) * cell
        z = (r0 + np.arange(size)) * cell
        d = fp.coarse_edge_distance(x, z)
        need = (d >= max(lo, -o.snap_m) - halo_m - 2 * cell) & (d <= hi + halo_m + 2 * cell)
        if not need.any():
            continue
        rr, cc = np.nonzero(need)
        xz = np.stack([x[cc], z[rr]], axis=1)
        need[rr, cc] = fp.bridge_distance(xz) >= o.bridge_margin_m
        rr, cc = np.nonzero(need)
        if len(rr) == 0:
            continue
        grids.append((n, r0, c0, size, rr, cc))
        points.append(np.stack([x[cc], z[rr]], axis=1))
    if not grids:
        return _Candidates.concat([])
    heights = np.asarray(canopy(np.concatenate(points)), np.float64)
    out = []
    k = 0
    for n, r0, c0, size, rr, cc in grids:
        chm = np.full((size, size), np.nan)
        chm[rr, cc] = heights[k : k + len(rr)]
        k += len(rr)
        if not np.isfinite(chm).any():
            continue
        rows, cols, h = detect_tree_tops(
            chm, cell, options=o, origin=(r0, c0), seed=seed,
            min_height_m=min(o.min_height_m, o.shrub_min_height_m),
        )
        gap_r = np.zeros(0, np.int64)
        gap_c = np.zeros(0, np.int64)
        gap_h = np.zeros(0)
        if band == BAND_NEAR:
            hs = _smoothed(chm, cell, o)
            valid = np.isfinite(chm)
            gap_r, gap_c = _gap_points(hs, valid, rows, cols, cell, (r0, c0), o, seed)
            raw = np.where(valid, np.minimum(chm, o.max_height_m), 0.0)
            gap_h = _raw_peak(raw, gap_r, gap_c)
        src = np.r_[np.full(len(rows), SOURCE_LIDAR), np.full(len(gap_r), SOURCE_GAP)]
        rows = np.concatenate([rows, gap_r])
        cols = np.concatenate([cols, gap_c])
        h = np.concatenate([h, gap_h])
        core = (
            (rows >= halo_cells) & (rows < halo_cells + n)
            & (cols >= halo_cells) & (cols < halo_cells + n)
        )
        rows, cols, h, src = rows[core], cols[core], h[core], src[core]
        xz = np.stack([(c0 + cols) * cell, (r0 + rows) * cell], axis=1)
        d, _ = fp.edge_distance(xz)
        keep = (d <= hi) & ((d > lo) if band == BAND_FAR else np.ones(len(d), bool))
        if keep.any():
            m = int(keep.sum())
            out.append(
                _Candidates(
                    xz[keep], h[keep], np.full(m, band, np.int16),
                    src[keep].astype(np.int16), np.full(m, -1, np.int16),
                )
            )
    return _Candidates.concat(out)


def _smoothed(chm: np.ndarray, cell_m: float, o: VegetationOptions) -> np.ndarray:
    valid = np.isfinite(chm)
    h0 = np.where(valid, np.minimum(chm, o.max_height_m), 0.0)
    sigma = o.smooth_sigma_m / cell_m
    if sigma < 0.3:
        return h0
    den = ndi.gaussian_filter(valid.astype(np.float64), sigma, mode="constant")
    num = ndi.gaussian_filter(h0, sigma, mode="constant")
    return np.where(valid & (den > 0.2), num / np.maximum(den, 1e-9), 0.0)


# ---------------------------------------------------------------------------------------------
# OSM fallback
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WoodArea:
    """An OSM woodland in AC XZ: `polygon` (shapely, X/Z) and its kind (DECIDUOUS, CONIFER,
    BUSH or MIXED)."""

    polygon: shapely.Geometry
    kind: int = MIXED


def woods_query(south: float, west: float, north: float, east: float, timeout_s: int = 90) -> str:
    """Overpass QL for forest, wood and scrub areas in a bbox (`out geom`, for
    `woods_from_overpass`). Post it with `driftmaps.geo.overpass.post_query`."""
    bbox = f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}"
    sel = ['["landuse"="forest"]', '["natural"~"^(wood|scrub)$"]']
    parts = "".join(f"{t}{s}({bbox});" for t in ("way", "relation") for s in sel)
    return f"[out:json][timeout:{int(timeout_s)}];({parts});out geom;"


def _kind_from_tags(tags: Mapping[str, str]) -> int:
    if tags.get("natural") == "scrub":
        return BUSH
    leaf = tags.get("leaf_type", "")
    if leaf == "broadleaved":
        return DECIDUOUS
    if leaf == "needleleaved":
        return CONIFER
    return MIXED


def woods_from_overpass(
    payload: bytes | str | Mapping, projection: LocalProjection
) -> list[WoodArea]:
    """`WoodArea`s from an Overpass JSON answer (`out geom`) to `woods_query`: closed ways and
    multipolygon relations (outer rings minus inner rings), projected to AC XZ."""
    doc = json.loads(payload) if isinstance(payload, (bytes, str)) else payload

    def line(geom: Sequence[Mapping]) -> np.ndarray | None:
        if not geom or len(geom) < 2:
            return None
        lon = np.array([g["lon"] for g in geom], np.float64)
        lat = np.array([g["lat"] for g in geom], np.float64)
        e, n = projection.to_enu(lon, lat)
        return np.stack([e, -n], axis=1)

    areas = []
    for el in doc.get("elements", []):
        tags = el.get("tags", {})
        kind = _kind_from_tags(tags)
        poly = None
        if el.get("type") == "way":
            pts = line(el.get("geometry"))
            if pts is not None and len(pts) >= 4 and np.allclose(pts[0], pts[-1]):
                poly = shapely.make_valid(shapely.Polygon(pts))
        elif el.get("type") == "relation":
            rings: dict[str, list] = {"outer": [], "inner": []}
            for m in el.get("members", []):
                pts = line(m.get("geometry"))
                if m.get("type") == "way" and pts is not None:
                    rings["inner" if m.get("role") == "inner" else "outer"].append(
                        shapely.LineString(pts)
                    )
            outer = shapely.union_all(list(shapely.polygonize(rings["outer"]).geoms))
            inner = shapely.union_all(list(shapely.polygonize(rings["inner"]).geoms))
            poly = shapely.make_valid(shapely.difference(outer, inner))
        if poly is not None and not poly.is_empty and poly.area > 1.0:
            areas.append(WoodArea(poly, kind))
    return areas


def _osm_candidates(
    fp: _Footprint,
    woods: Sequence[WoodArea],
    canopy: CanopyFn | None,
    o: VegetationOptions,
    seed: int,
) -> _Candidates:
    """Jittered-lattice trees inside OSM woods within the band, where the CHM has no data."""
    reach = max(o.far_m, o.near_m)
    out = []
    s = o.osm_spacing_m
    for k, area in enumerate(woods):
        xmin, zmin, xmax, zmax = area.polygon.bounds
        gi = np.arange(math.floor(zmin / s), math.ceil(zmax / s) + 1)
        gj = np.arange(math.floor(xmin / s), math.ceil(xmax / s) + 1)
        GI, GJ = np.meshgrid(gi, gj, indexing="ij")
        GI, GJ = GI.ravel(), GJ.ravel()
        jx = _u01(GI, GJ, seed, _S_OSM, 1) - 0.5
        jz = _u01(GI, GJ, seed, _S_OSM, 2) - 0.5
        xz = np.stack([(GJ + 0.5 + 0.8 * jx) * s, (GI + 0.5 + 0.8 * jz) * s], axis=1)
        inside = shapely.contains_xy(area.polygon, xz[:, 0], xz[:, 1])
        xz, GI, GJ = xz[inside], GI[inside], GJ[inside]
        if len(xz) == 0:
            continue
        d, _ = fp.edge_distance(xz)
        near = d <= reach
        xz, GI, GJ, d = xz[near], GI[near], GJ[near], d[near]
        if len(xz) and canopy is not None:
            no_lidar = ~np.isfinite(np.asarray(canopy(xz), np.float64))
            xz, GI, GJ, d = xz[no_lidar], GI[no_lidar], GJ[no_lidar], d[no_lidar]
        if len(xz) == 0:
            continue
        lo_h, hi_h = o.osm_shrub_height_m if area.kind == BUSH else o.osm_height_m
        h = lo_h + (hi_h - lo_h) * _u01(GI, GJ, seed, _S_OSM, 3 + k)
        m = len(xz)
        out.append(
            _Candidates(
                xz, h, np.where(d <= o.near_m, BAND_NEAR, BAND_FAR).astype(np.int16),
                np.full(m, SOURCE_OSM, np.int16), np.full(m, area.kind, np.int16),
            )
        )
    return _Candidates.concat(out)


# ---------------------------------------------------------------------------------------------
# Land cover
# ---------------------------------------------------------------------------------------------


def sample_landcover(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    cache_dir: Path | str | None = None,
    url: str = LANDCOVER_URL,
) -> np.ndarray:
    """NRCan Land Cover 2020 class (nearest pixel) at lon/lat; 0 outside / NoData.

    Reads the COG block by block through the elevation module's disk cache
    (`<cache>/landcover/`); one 512 px block covers 15 km, so a track costs 1-4 range reads.
    `url` may be a local GeoTIFF (tests).
    """
    lon = np.asarray(lon, np.float64).ravel()
    lat = np.asarray(lat, np.float64).ravel()
    out = np.zeros(lon.size, np.uint8)
    if lon.size == 0:
        return out
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    raster = _CogRaster(url, root / "landcover")
    if raster.missing:
        return out
    m = raster.meta()
    xs, ys = warp_transform("EPSG:4326", m["crs"], lon.tolist(), lat.tolist())
    a, b, c, d, e, f = m["transform"]
    det = a * e - b * d
    x = np.asarray(xs) - c
    y = np.asarray(ys) - f
    col = np.floor((e * x - b * y) / det).astype(np.int64)
    row = np.floor((-d * x + a * y) / det).astype(np.int64)
    vals = raster._gather(row, col)
    ok = np.isfinite(vals)
    out[ok] = np.clip(vals[ok], 0, 255).astype(np.uint8)
    return out


class LandcoverSampler:
    """`sample_landcover` on AC XZ points (a `LandcoverFn` for `place_trees`).

    Returns 0 everywhere, without touching the network, when no point is in or near Canada,
    and 0 with a warning if the COG can't be read (the trees then fall back to the mixed rule).
    """

    def __init__(
        self,
        projection: LocalProjection,
        *,
        cache_dir: Path | str | None = None,
        url: str = LANDCOVER_URL,
    ) -> None:
        self.projection = projection
        self.cache_dir = cache_dir
        self.url = url
        self.used = False

    def __call__(self, xz: np.ndarray) -> np.ndarray:
        xz = np.asarray(xz, np.float64).reshape(-1, 2)
        if len(xz) == 0:
            return np.zeros(0, np.uint8)
        lon, lat = self.projection.to_lonlat(xz[:, 0], -xz[:, 1])
        regions = region_of(lon, lat)
        if not np.isin(regions, ("CA", "CA-border", "US-border")).any():
            return np.zeros(len(xz), np.uint8)
        try:
            out = sample_landcover(lon, lat, cache_dir=self.cache_dir, url=self.url)
        except SourceUnavailable as exc:
            log.warning("land cover unavailable, trees use the mixed rule: %s", exc)
            return np.zeros(len(xz), np.uint8)
        self.used = self.used or bool(out.any())
        return out

    @property
    def attributions(self) -> list[str]:
        return [LANDCOVER_ATTRIBUTION] if self.used else []


class CanopySampler:
    """`elevation.sources.canopy_height_grid` on AC XZ points (a `CanopyFn` for `place_trees`).

    Remembers the DSM/DTM sources it used; `attributions` are their LICENSE-DATA.txt lines.
    """

    def __init__(
        self,
        projection: LocalProjection,
        *,
        cache_dir: Path | str | None = None,
        sources: Mapping[str, ElevationSource] | None = None,
        chunk: int = 2_000_000,
    ) -> None:
        self.projection = projection
        self.cache_dir = cache_dir
        self.sources = sources
        self.chunk = chunk
        self.info: dict[str, SourceInfo] = {}

    def __call__(self, xz: np.ndarray) -> np.ndarray:
        xz = np.asarray(xz, np.float64).reshape(-1, 2)
        out = np.full(len(xz), np.nan)
        for s in range(0, len(xz), self.chunk):
            part = xz[s : s + self.chunk]
            lon, lat = self.projection.to_lonlat(part[:, 0], -part[:, 1])
            smp = canopy_height_grid(lon, lat, cache_dir=self.cache_dir, sources=self.sources)
            out[s : s + len(part)] = smp.heights
            self.info.update(smp.info)
        return out

    @property
    def attributions(self) -> list[str]:
        lines: list[str] = []
        for info in self.info.values():
            line = f"Tree heights ({info.title}): {info.attribution}"
            if line not in lines:
                lines.append(line)
        return lines


# ---------------------------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------------------------


def _choose_kinds(
    hint: np.ndarray, lc: np.ndarray, height: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """Kind per tree from the land cover class (or an OSM hint), scenery.md §7."""
    kind = np.full(len(hint), DECIDUOUS, np.int16)
    rule = np.where(hint >= 0, -1, lc.astype(np.int16))  # -1: use the hint
    p_conifer = np.full(len(hint), 0.2)  # cropland, grass, urban, other: 80 % deciduous
    p_conifer[np.isin(rule, _LC_NEEDLE)] = 0.9
    p_conifer[np.isin(rule, _LC_BROAD)] = 0.1
    p_conifer[np.isin(rule, _LC_MIXED) | (rule == 0)] = 0.5
    shrub = np.isin(rule, _LC_SHRUB)
    p_conifer[shrub] = 0.5  # tall trees in shrubland/wetland: mixed
    kind[u < p_conifer] = CONIFER
    small = shrub & (height < 5.0)
    kind[small] = np.where(u[small] < 0.8, BUSH, DECIDUOUS)
    # OSM hints.
    kind[hint == DECIDUOUS] = DECIDUOUS
    kind[hint == CONIFER] = CONIFER
    kind[hint == BUSH] = BUSH
    mixed = hint == MIXED
    kind[mixed] = np.where(u[mixed] < 0.5, CONIFER, DECIDUOUS)
    return kind


# Tile weights within a kind: (tile index, weight).
_TILE_WEIGHTS = {
    DECIDUOUS: ((0, 0.35), (1, 0.30), (2, 0.15), (5, 0.20)),
    CONIFER: ((3, 0.40), (4, 0.30), (6, 0.30)),
    BUSH: ((7, 0.5), (8, 0.5)),
}
_NARROW_TILE = 2  # poplar: preferred for narrow, tall deciduous crowns


def _choose_tiles(kind: np.ndarray, radius: np.ndarray, height: np.ndarray, u: np.ndarray):
    tile = np.zeros(len(kind), np.int16)
    for k, entries in _TILE_WEIGHTS.items():
        sel = kind == k
        if not sel.any():
            continue
        idx = np.array([e[0] for e in entries])
        w = np.cumsum([e[1] for e in entries])
        tile[sel] = idx[np.searchsorted(w / w[-1], u[sel], side="right").clip(0, len(idx) - 1)]
    narrow = (kind == DECIDUOUS) & (radius < 0.2 * height) & (u < 0.6)
    tile[narrow] = _NARROW_TILE
    return tile


def place_trees(
    model: RoadModel,
    canopy: CanopyFn | None = None,
    *,
    landcover: LandcoverFn | None = None,
    woods: Sequence[WoodArea] = (),
    exclude: shapely.Geometry | None = None,
    seed: int | str = 0,
    options: VegetationOptions | None = None,
    attributions: Sequence[str] = (),
) -> TreeSet:
    """Place the trees along a road (see the module docstring for every step).

    `canopy`: XZ -> CHM metres (`CanopySampler`); None = no lidar. `landcover`: XZ -> NRCan
    class (`LandcoverSampler`); None = the mixed rule. `woods`: OSM fallback areas, used only
    where the CHM is NaN. `exclude`: XZ geometry with no trees (building footprints buffered
    2 m, water, or the wall corridor). `seed`: int or string, e.g. the track id.
    `attributions` are added to the result's own (taken from the samplers' `attributions`).
    """
    o = options or VegetationOptions()
    seed_i = _seed_int(seed)
    fp = _Footprint(model, o.bridge_pad_m)
    parts = []
    if canopy is not None:
        parts.append(_lidar_candidates(fp, canopy, BAND_NEAR, -math.inf, o.near_m,
                                       o.near_cell_m, o, seed_i))
        if o.far_m > o.near_m:
            parts.append(_lidar_candidates(fp, canopy, BAND_FAR, o.near_m, o.far_m,
                                           o.far_cell_m, o, seed_i))
    if woods:
        parts.append(_osm_candidates(fp, woods, canopy, o, seed_i))
    c = _Candidates.concat(parts)

    def notes() -> list[str]:
        out = list(attributions)
        for src in (canopy, landcover):
            out += [x for x in getattr(src, "attributions", None) or [] if x not in out]
        if (c.source == SOURCE_OSM).any() and OSM_WOODS_ATTRIBUTION not in out:
            out.append(OSM_WOODS_ATTRIBUTION)
        return out

    if len(c.xz) == 0:
        return TreeSet.empty(notes())

    xz = c.xz.copy()
    # 4. Road corridor: push shallow intruders out to the clearance edge, drop the rest.
    d, idx = fp.edge_distance(xz)
    push = (d < o.clearance_m) & (d >= o.clearance_m - o.snap_m)
    if push.any():
        v = xz[push] - fp.xz[idx[push]]
        n = np.linalg.norm(v, axis=1)
        ok = n > 1e-6
        step = (o.clearance_m + 0.25 - d[push])[:, None] * v / np.maximum(n, 1e-6)[:, None]
        moved = xz[push] + np.where(ok[:, None], step, 0.0)
        xz[push] = moved
        d[push], idx[push] = fp.edge_distance(moved)
    keep = (d >= o.clearance_m) & (d <= max(o.far_m, o.near_m))
    keep &= fp.bridge_distance(xz) >= o.bridge_margin_m
    if exclude is not None and not exclude.is_empty:
        keep &= ~shapely.contains_xy(exclude, xz[:, 0], xz[:, 1])
    lc = np.zeros(len(xz), np.uint8)
    if landcover is not None and keep.any():
        lc[keep] = np.asarray(landcover(xz[keep]), np.uint8)
    keep &= ~np.isin(lc, _LC_WATER)
    lidar = c.source != SOURCE_OSM
    tall = c.height >= o.min_height_m
    keep &= ~lidar | tall | np.isin(lc, _LC_SHRUB)
    keep &= c.height >= min(o.min_height_m, o.shrub_min_height_m)
    xz, d, lc = xz[keep], d[keep], lc[keep]
    height, source, hint = c.height[keep], c.source[keep], c.hint[keep]
    band = np.where(d <= o.near_m, BAND_NEAR, BAND_FAR).astype(np.int16)  # after snapping
    if len(xz) == 0:
        return TreeSet.empty(notes())

    # Merge duplicates (an OSM tree on top of a lidar one cannot happen, but snapping can stack).
    qx, qz = _pos_keys(xz)
    _, first = np.unique(np.stack([qx // 50, qz // 50], 1), axis=0, return_index=True)
    first.sort()
    xz, d, lc, height, band, source, hint = (
        a[first] for a in (xz, d, lc, height, band, source, hint)
    )
    qx, qz = _pos_keys(xz)

    # 5. Crown radius from the nearest neighbour (all bands together, before thinning).
    if len(xz) > 1:
        dnn = cKDTree(xz).query(xz, k=2)[0][:, 1]
    else:
        dnn = np.full(1, o.crown_max_m / o.crown_factor)
    radius = np.clip(o.crown_factor * dnn, o.crown_min_m, o.crown_max_m)
    radius = np.minimum(radius, np.maximum(0.25 * height + 1.0, o.crown_min_m))

    # 6. Kind and atlas tile.
    kind = _choose_kinds(hint, lc, height, _u01(qx, qz, seed_i, _S_KIND))
    shrub = kind == BUSH
    radius[shrub] = np.clip(radius[shrub], 0.8, np.maximum(1.0, 0.9 * height[shrub]))
    low = (kind == CONIFER) | shrub
    radius[low] = np.minimum(radius[low], np.maximum(d[low], 1.0))
    tile = _choose_tiles(kind, radius, height, _u01(qx, qz, seed_i, _S_TILE))

    # 7. Far band: keep a share, as bigger clumps.
    far = band == BAND_FAR
    keep = ~far | (_u01(qx, qz, seed_i, _S_FAR) < o.far_keep)
    radius = np.where(far, radius * o.far_scale, radius)

    trees = TreeSet(
        xz=xz,
        height=height,
        radius=radius,
        kind=kind,
        tile=tile,
        distance=d,
        band=band,
        source=source,
        angle=_u01(qx, qz, seed_i, _S_ANGLE) * (0.5 * math.pi),
        flip=_u01(qx, qz, seed_i, _S_FLIP) < 0.5,
        shade=_u01(qx, qz, seed_i, _S_SHADE),
        attributions=notes(),
    ).subset(keep)

    # 8. Budget: a share for each band; within a band the farthest, smallest and gap-fill
    # trees go first.
    cap = int(min(o.max_trees, math.ceil(o.max_trees_per_km * fp.length_m / 1000.0)))
    if len(trees) > cap:
        far = trees.band == BAND_FAR
        far_cap = min(int(far.sum()), int(round(o.far_share * cap)))
        near_cap = min(int((~far).sum()), cap - far_cap)
        far_cap = cap - near_cap
        score = (
            trees.distance / max(o.far_m, o.near_m, 1.0)
            + 0.5 * (1.0 - np.clip(trees.height, 0.0, 30.0) / 30.0)
            + 0.3 * (trees.source == SOURCE_GAP)
        )
        kept = []
        for sel, n in ((np.flatnonzero(~far), near_cap), (np.flatnonzero(far), far_cap)):
            order = np.lexsort((sel, score[sel]))
            kept.append(sel[order[:n]])
        log.info("tree budget: kept %d of %d trees", cap, len(trees))
        trees = trees.subset(np.sort(np.concatenate(kept)))

    # Stable output order: band, then X, then Z.
    order = np.lexsort((trees.xz[:, 1], trees.xz[:, 0], trees.band))
    return trees.subset(order)


# ---------------------------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TreeMeshOptions:
    cell_m: float = 250.0  # batching cell
    max_trees_per_mesh: int = 2000
    near_lod_out_m: float = 700.0
    far_lod_out_m: float = 1500.0
    near_shadows: bool = True
    far_shadows: bool = False
    three_cards_min_height_m: float = 12.0  # near deciduous trees this tall get a 3rd card
    sink_m: float = 0.3  # card bottoms this far below the terrain
    normal_blend: tuple[float, float] = (0.35, 0.6)  # weight of the radial normal (vs +Y)
    aspect_clamp: tuple[float, float] = (0.65, 1.5)  # card w/h relative to the tile's aspect
    atlas_size: int = ATLAS_SIZE


@dataclass
class TreeMesh:
    """One batch of tree cards, ready for `Scene.mesh(data, name, material, **flags)`."""

    name: str
    data: MeshData
    band: str
    tree_count: int
    flags: MeshFlags


@dataclass
class TreeStats:
    trees: int = 0
    near: int = 0
    far: int = 0
    meshes: int = 0
    vertices: int = 0
    triangles: int = 0
    dropped_no_ground: int = 0
    texture_bytes: int = 0


def tree_material(name: str = "dm_trees", texture_name: str = "dm_trees.dds") -> Material:
    """The `ksTree` card material with Kunos' NJMP settings (scenery.md §4.1): alpha-tested,
    blend opaque, depth normal, one diffuse atlas."""
    return Material(
        name=name,
        shader="ksTree",
        blend_mode=BlendMode.OPAQUE,
        alpha_tested=True,
        depth_mode=DepthMode.NORMAL,
        properties=[
            MaterialProperty("ksAmbient", a=0.4),
            MaterialProperty("ksDiffuse", a=0.25),
            MaterialProperty("ksSpecular", a=0.0),
            MaterialProperty("ksSpecularEXP", a=1.0),
            MaterialProperty("ksEmissive", c=(0.0, 0.0, 0.0)),
            MaterialProperty("ksAlphaRef", a=0.5),
        ],
        textures=[TextureMapping("txDiffuse", 0, texture_name)],
    )


_UP = np.array([0.0, 1.0, 0.0])
_FRONT = np.array([0, 1, 2, 0, 2, 3])
_BACK = np.array([0, 2, 1, 0, 3, 2])


def _cards(trees: TreeSet, base: np.ndarray, ground: GroundFn, o: TreeMeshOptions) -> MeshData:
    """Crossed cards for every tree in `trees` (base = terrain Y at each trunk)."""
    n = len(trees)
    big = (
        (trees.kind == DECIDUOUS) & (trees.band == BAND_NEAR)
        & (trees.height >= o.three_cards_min_height_m)
    )
    k_per = np.where(big, 3, 2)
    tq = np.repeat(np.arange(n), k_per)
    start = np.repeat(np.cumsum(k_per) - k_per, k_per)
    kq = np.arange(len(tq)) - start
    ang = trees.angle[tq] + kq * math.pi / k_per[tq]
    right = np.stack([np.cos(ang), np.zeros_like(ang), np.sin(ang)], axis=1)

    tiles = trees.tile.astype(np.int64)
    fill = np.array([t.fill for t in TREE_TILES])[tiles]
    aspect = np.array([t.aspect for t in TREE_TILES])[tiles]
    h = trees.height
    width = 2.0 * trees.radius / fill
    lo, hi = o.aspect_clamp
    width = np.clip(width, lo * aspect * h, hi * aspect * h)

    centre = np.stack([trees.xz[:, 0], base, trees.xz[:, 1]], axis=1)
    hw = 0.5 * width[tq][:, None]
    c = centre[tq]
    left_pt = c - right * hw
    right_pt = c + right * hw
    corners = np.concatenate([left_pt, right_pt])[:, [0, 2]]
    g = np.asarray(ground(corners), np.float64)
    trunk_y = base[tq] - o.sink_m
    g = np.where(np.isfinite(g), g - o.sink_m, np.concatenate([trunk_y, trunk_y]))
    nq = len(tq)
    y_left = np.minimum(g[:nq], trunk_y)
    y_right = np.minimum(g[nq:], trunk_y)
    y_top = base[tq] + h[tq]

    pos = np.empty((nq, 4, 3))
    pos[:, 0] = left_pt
    pos[:, 0, 1] = y_left
    pos[:, 1] = right_pt
    pos[:, 1, 1] = y_right
    pos[:, 2] = right_pt
    pos[:, 2, 1] = y_top
    pos[:, 3] = left_pt
    pos[:, 3, 1] = y_top

    uv_rect = np.array([tile_uv(t, o.atlas_size) for t in range(len(TREE_TILES))])[tiles[tq]]
    u0, v0, u1, v1 = uv_rect.T
    flip = trees.flip[tq]
    ua = np.where(flip, u1, u0)
    ub = np.where(flip, u0, u1)
    uvs = np.stack(
        [np.stack([ua, v1], 1), np.stack([ub, v1], 1), np.stack([ub, v0], 1),
         np.stack([ua, v0], 1)], axis=1,
    )

    crown = c.copy()
    crown[:, 1] = base[tq] + 0.55 * h[tq]
    radial = pos - crown[:, None, :]
    radial /= np.maximum(np.linalg.norm(radial, axis=2, keepdims=True), 1e-9)
    b0, b1 = o.normal_blend
    k = (b0 + (b1 - b0) * trees.shade[tq])[:, None, None]
    nrm = k * radial + (1.0 - k) * _UP
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    # Keep every normal at least a little upward ("tree normals point up", scenery.md §4.1).
    nrm[..., 1] = np.maximum(nrm[..., 1], 0.2)
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    # Smaller ZIPs (measured on Snake Rd, 23k trees: 5.15 -> 3.45 MB deflated): a constant
    # tangent (ksTree has no normal map; scenery.md §4.1), positions on a 1/64 m grid (card
    # bottoms rounded down, so they never float) and normals on a 1/256 grid (unit length
    # within 0.5 %).
    tan = np.broadcast_to(np.array([1.0, 0.0, 0.0]), pos.shape)
    q = 64.0
    pos[..., [0, 2]] = np.round(pos[..., [0, 2]] * q) / q
    pos[:, :2, 1] = np.floor(pos[:, :2, 1] * q) / q
    pos[:, 2:, 1] = np.round(pos[:, 2:, 1] * q) / q
    nrm = np.round(nrm * 256.0) / 256.0

    base_idx = (np.arange(nq) * 4)[:, None]
    idx = np.concatenate([base_idx + _FRONT, base_idx + _BACK], axis=1)
    return MeshData(
        pos.reshape(-1, 3), nrm.reshape(-1, 3), uvs.reshape(-1, 2), tan.reshape(-1, 3),
        idx.reshape(-1),
    )


def tree_meshes(
    trees: TreeSet, ground: GroundFn, *, options: TreeMeshOptions | None = None
) -> list[TreeMesh]:
    """Batch the trees into card meshes per band and `cell_m` cell (module docstring).

    `ground(xz)` gives the terrain Y (use the built terrain, not the raw DTM: cut/fill changed
    it); trees whose trunk has no ground are dropped. Mesh names: `trees_<band>_<nnn>` (cells in
    sorted order), with `_<p>` suffixes when a cell needs several meshes.
    """
    o = options or TreeMeshOptions()
    if len(trees) == 0:
        return []
    base = np.asarray(ground(trees.xz), np.float64)
    ok = np.isfinite(base)
    if not ok.all():
        log.info("dropped %d trees with no terrain under them", int((~ok).sum()))
        trees, base = trees.subset(ok), base[ok]
    per_mesh = max(1, min(o.max_trees_per_mesh, MAX_MESH_VERTICES // 12))
    out: list[TreeMesh] = []
    cell = np.floor(trees.xz / o.cell_m).astype(np.int64)
    for b, band_name in enumerate(BAND_NAMES):
        sel = np.flatnonzero(trees.band == b)
        if len(sel) == 0:
            continue
        keys = cell[sel]
        order = np.lexsort((trees.xz[sel, 0], trees.xz[sel, 1], keys[:, 0], keys[:, 1]))
        sel, keys = sel[order], keys[order]
        uniq, start = np.unique(keys, axis=0, return_index=True)
        bounds = np.r_[np.sort(start), len(sel)]
        flags: MeshFlags = {
            "lod_out": o.near_lod_out_m if b == BAND_NEAR else o.far_lod_out_m,
            "cast_shadows": o.near_shadows if b == BAND_NEAR else o.far_shadows,
        }
        for ci in range(len(uniq)):
            members = sel[bounds[ci] : bounds[ci + 1]]
            pieces = [members[i : i + per_mesh] for i in range(0, len(members), per_mesh)]
            for p, piece in enumerate(pieces):
                data = _cards(trees.subset(piece), base[piece], ground, o)
                name = f"trees_{band_name}_{ci:03d}"
                if len(pieces) > 1:
                    name = f"{name}_{p}"
                out.append(TreeMesh(name, data, band_name, len(piece), dict(flags)))
    return out


def tree_scene(
    scene: Scene,
    trees: TreeSet,
    ground: GroundFn,
    *,
    texture_format: str = "dds",
    atlas_seed: int = ATLAS_SEED,
    material_name: str = "dm_trees",
    options: TreeMeshOptions | None = None,
) -> TreeStats:
    """Add the tree cards to a KN5 `Scene`: the atlas texture, one `ksTree` material and the
    batched meshes. Returns counts for the build summary. Adds nothing if there are no trees."""
    o = options or TreeMeshOptions()
    meshes = tree_meshes(trees, ground, options=o)
    stats = TreeStats()
    if not meshes:
        return stats
    tex_name = f"dm_trees.{'png' if texture_format == 'png' else 'dds'}"
    if tex_name not in scene.textures:
        scene.textures[tex_name] = tree_atlas_texture(tex_name, o.atlas_size, atlas_seed)
    mat_id = next((i for i, m in enumerate(scene.materials) if m.name == material_name), None)
    if mat_id is None:
        scene.materials.append(tree_material(material_name, tex_name))
        mat_id = len(scene.materials) - 1
    for m in meshes:
        scene.mesh(m.data, m.name, mat_id, **m.flags)
        stats.meshes += 1
        stats.trees += m.tree_count
        stats.vertices += m.data.vertex_count
        stats.triangles += m.data.triangle_count
        if m.band == "near":
            stats.near += m.tree_count
        else:
            stats.far += m.tree_count
    stats.dropped_no_ground = len(trees) - stats.trees
    stats.texture_bytes = len(scene.textures[tex_name].data)
    return stats
