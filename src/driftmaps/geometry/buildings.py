"""Buildings beside the road: footprints -> heights, roofs, textured KN5 meshes.

Decisions: docs/research/scenery.md §3 and §7 ("Buildings", "Budgets"), geo.md §4.

**Placement** (`place_buildings`), in AC coordinates (X = east, Y = up, Z = south):

1. Footprints (`geo.buildings.Footprint`, lon/lat) are projected to XZ, repaired, holes dropped
   (drawn solid), and kept when `min_area_m2` <= area <= `max_area_m2`, within
   `max_distance_m` (150 m) of the centreline, and clear of the road: a footprint that touches
   the corridor polygon (road + verge, as the hidden walls use) grown by `clearance_m`, or that
   comes within `clearance_m` of the road's drivable edge, is dropped (never clipped). At most
   `max_buildings` (400) are kept, nearest the road first.
2. Ground: the built terrain (`ground(xz)`, not the raw DTM: cut/fill changed it) under the
   ring and a 3 m interior lattice. The walls start `sink_m` (0.5 m) below its LOWEST point;
   heights are measured from its median.
3. Height, in order (the first that applies):
   - **lidar** `canopy(xz)` = DSM - DTM (NRCan HRDEM, OGL-Canada) on a 1 m lattice inside the
     footprint eroded by 1 m: when at least `min_lidar_samples` are valid and their median is
     >= `min_lidar_height_m` (a lower median means the building is newer than the survey). The
     median, clamped to `height_clamp` (3-30 m), is the roof's mean height; the spread
     p90 - median gives the pitch (a gable sampled uniformly has median = eave + rise / 2 and
     p90 = eave + 0.9 rise, so rise = 2.5 (p90 - median), corrected for the eroded lattice
     seeing only the upper part of each slope; pitch capped at 45 degrees). On a small house or
     shed (< 400 m2) a median above 12 m is tree canopy over the roof, not the roof: ignored.
   - OSM `height` (roof top), then `building:levels` x `storey_m` (3 m, the eave), then the
     Microsoft ML height (roof top), then `default_height_m` (6 m, roof top).
4. Roof: OSM `roof:shape` when tagged (flat / gabled family / hipped family); else flat for
   commercial / industrial kinds, footprints >= `flat_roof_area_m2` that are not barns, narrow
   ones (< `min_pitched_width_m`), and lidar spreads < `flat_lidar_spread_m`; otherwise pitched:
   hipped for near-square houses (some, deterministic), gabled for the rest. The ridge runs
   along the long axis of the footprint's minimum rotated rectangle. Without lidar the pitch is
   22-35 degrees (deterministic per building), rise capped at `max_roof_rise_m`.
5. Look: wall and roof styles from `ac.building_textures` (siding / brick / stucco for houses,
   red or grey boards and metal roofs for barns and sheds, block and membrane for commercial),
   chosen by a hash of (seed, building id), so a rebuild gives the same street.

**Meshes** (`building_meshes`, `building_scene`): the roof is a piecewise-planar height field
over the minimum rotated rectangle (gable: 2 planes split by the ridge; hip: 2 trapezoids + 2
triangles; flat: 1 plane) cut to the REAL footprint, so L-shaped and irregular houses get one
roof whose gable-end walls follow it. The footprint ring is noded where it crosses a roof
crease and where its roof edge crosses a storey line; walls are cut at storey lines (every wall
quad stays inside one atlas band, see `ac.building_textures`), roof pieces are triangulated
with constrained Delaunay on the same ring vertices, so walls and roof meet exactly (the solid
is closed apart from its open bottom; `floor=True` closes it too, for tests). Normals are flat
and outward, winding CCW from outside, UVs never mirrored.

Batched per `cell_m` (250 m) cell into visual-only meshes `buildings_<nnn>` (no digit prefix,
no collision) with `lod_out_m`; one `ksPerPixel` material on one BC1 atlas, so a cell is one
draw call. Buildings whose footprint is within `collide_within_m` (15 m) of the road's drivable
edge also get hidden, double-sided collision walls `1WALL_bld_<nnn>` (ac_format.md §3.2), up to
the eave.

`buildings_polygon` gives the footprints (buffered) for the tree builder's exclusion mask.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import shapely
from scipy.spatial import cKDTree

from driftmaps.ac.building_textures import (
    ATLAS_SEED,
    ATLAS_SIZE,
    ATLAS_U_M,
    STOREY_M,
    band_v,
    building_atlas_texture,
)
from driftmaps.ac.model import Material, MaterialProperty, TextureMapping
from driftmaps.geometry.primitives import (
    MeshData,
    MeshFlags,
    compute_tangents,
    merge,
    split,
    wall,
)

if TYPE_CHECKING:
    from driftmaps.ac.scene import Scene
    from driftmaps.geo.buildings import Footprint
    from driftmaps.geo.project import LocalProjection
    from driftmaps.geometry.road import RoadModel

__all__ = [
    "ROOF_FLAT",
    "ROOF_GABLED",
    "ROOF_HIPPED",
    "Building",
    "BuildingMesh",
    "BuildingOptions",
    "BuildingStats",
    "CanopyFn",
    "GroundFn",
    "Roof",
    "building_material",
    "building_meshes",
    "building_scene",
    "buildings_polygon",
    "choose_height",
    "extrude_building",
    "place_buildings",
]

log = logging.getLogger(__name__)

GroundFn = Callable[[np.ndarray], np.ndarray]  # XZ (N, 2) -> terrain Y (N,), NaN where none
CanopyFn = Callable[[np.ndarray], np.ndarray]  # XZ (N, 2) -> DSM - DTM metres, NaN where none

ROOF_FLAT = "flat"
ROOF_GABLED = "gabled"
ROOF_HIPPED = "hipped"

# OSM roof:shape values -> our three shapes.
_ROOF_TAGS = {
    "flat": ROOF_FLAT,
    "skillion": ROOF_FLAT,
    "gabled": ROOF_GABLED,
    "gambrel": ROOF_GABLED,
    "saltbox": ROOF_GABLED,
    "round": ROOF_GABLED,
    "hipped": ROOF_HIPPED,
    "half-hipped": ROOF_HIPPED,
    "pyramidal": ROOF_HIPPED,
    "mansard": ROOF_HIPPED,
    "side_hipped": ROOF_HIPPED,
}
_FLAT_KINDS = frozenset({
    "commercial", "industrial", "retail", "warehouse", "office", "school", "supermarket",
    "hospital", "public", "civic", "government", "college", "university", "hotel",
    "apartments", "parking", "service", "transportation", "train_station", "fire_station",
    "sports_hall", "kindergarten", "manufacture", "storage_tank",
})
_FARM_KINDS = frozenset({
    "barn", "farm_auxiliary", "stable", "cowshed", "sty", "silo", "greenhouse", "shed",
    "hut", "garage", "garages", "carport", "boathouse", "outbuilding",
})
_SMALL_CAPPED_KINDS = frozenset({
    "yes", "house", "detached", "residential", "semidetached_house", "bungalow", "cabin",
    "garage", "garages", "carport", "shed", "hut", "outbuilding", "farm_auxiliary",
})

# (style, weight) tables; deterministic draws per building.
_HOUSE_WALLS = (
    ("siding_white", 18), ("siding_cream", 14), ("siding_grey", 14), ("siding_blue", 7),
    ("brick_red", 18), ("brick_brown", 12), ("brick_buff", 9), ("stucco", 8),
)
_HOUSE_ROOFS = (("shingle_charcoal", 60), ("shingle_brown", 30), ("metal_grey", 10))
_BARN_WALLS = (("barn_red", 55), ("shed_wood", 45))
_BARN_ROOFS = (("metal_grey", 60), ("metal_green", 25), ("shingle_charcoal", 15))
_SHED_WALLS = (("shed_wood", 60), ("siding_white", 20), ("siding_grey", 20))
_COMMERCIAL_WALLS = (("commercial", 60), ("brick_red", 20), ("brick_buff", 20))


@dataclass(frozen=True)
class BuildingOptions:
    max_distance_m: float = 150.0  # from the centreline (scenery.md §7: footprints within 150 m)
    clearance_m: float = 2.0  # kept clear of the corridor polygon / the drivable edge
    min_area_m2: float = 9.0
    max_area_m2: float = 20_000.0
    max_buildings: int = 400  # scenery.md §7 budget
    # Height (see module docstring).
    erode_m: float = 1.0
    sample_step_m: float = 1.0
    min_lidar_samples: int = 4
    min_lidar_height_m: float = 2.0
    height_clamp: tuple[float, float] = (3.0, 30.0)
    storey_m: float = 3.0
    default_height_m: float = 6.0
    # Roof.
    flat_roof_area_m2: float = 600.0
    flat_lidar_spread_m: float = 0.5  # p90 - median below this -> flat (ASSUMED threshold)
    min_pitched_width_m: float = 3.5  # narrower minimum-rectangle -> flat
    pitch_deg: tuple[float, float] = (22.0, 35.0)  # without lidar
    max_roof_rise_m: float = 6.0
    max_pitch_deg: float = 45.0
    # Small houses / sheds (area < small_area_m2) whose lidar median exceeds this are under
    # trees: the canopy is ignored and tags / defaults are used (ASSUMED; rural Ontario).
    small_area_m2: float = 400.0
    small_max_lidar_m: float = 12.0
    min_eave_m: float = 2.4
    hipped_share: float = 0.35  # share of near-square houses (aspect < 1.4) given a hip roof
    # Mesh.
    sink_m: float = 0.5
    collide_within_m: float = 15.0  # from the drivable edge; None-like: < 0 disables
    collide_extra_m: float = 0.5  # collision walls this much taller than the eave
    cell_m: float = 250.0
    lod_out_m: float = 1200.0
    cast_shadows: bool = True
    atlas_size: int = ATLAS_SIZE


# --------------------------------------------------------------------------------------------
# Deterministic choices
# --------------------------------------------------------------------------------------------


def _u01(*keys: object) -> float:
    """A uniform number in [0, 1) from a hash of the keys (stable across runs and platforms)."""
    h = hashlib.blake2b("|".join(map(str, keys)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") / 2.0**64


def _pick(table: Sequence[tuple[str, float]], u: float) -> str:
    total = sum(w for _, w in table)
    acc = 0.0
    for name, w in table:
        acc += w / total
        if u < acc:
            return name
    return table[-1][0]


# --------------------------------------------------------------------------------------------
# Roof model
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Roof:
    """A piecewise-planar roof over the footprint's minimum rotated rectangle, in XZ.

    `height(xz)` is metres above the eave: 0 everywhere for flat, up to `rise` at the ridge.
    `long_axis` / `short_axis` are unit XZ vectors; the ridge runs along `long_axis` through
    `centre`. For a hip roof all four slopes share the gradient rise / half_width.
    """

    shape: str
    centre: np.ndarray
    long_axis: np.ndarray
    short_axis: np.ndarray
    half_length: float
    half_width: float
    rise: float = 0.0

    def local(self, xz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = np.asarray(xz, np.float64).reshape(-1, 2) - self.centre
        return d @ self.long_axis, d @ self.short_axis

    def height(self, xz: np.ndarray) -> np.ndarray:
        dl, ds = self.local(xz)
        if self.shape == ROOF_FLAT or self.rise <= 0.0:
            return np.zeros(len(dl))
        k = self.rise / self.half_width
        if self.shape == ROOF_GABLED:
            h = k * (self.half_width - np.abs(ds))
        else:
            h = k * np.minimum(self.half_width - np.abs(ds), self.half_length - np.abs(dl))
        return np.clip(h, 0.0, self.rise)

    def _xz(self, pts_ls: Sequence[tuple[float, float]]) -> np.ndarray:
        p = np.asarray(pts_ls, np.float64)
        return self.centre + p[:, :1] * self.long_axis + p[:, 1:] * self.short_axis

    def regions(self, margin: float = 1.0) -> list[tuple[shapely.Polygon, np.ndarray, float]]:
        """(polygon, gradient (2,), height at `centre`) of each planar piece, covering the
        rectangle grown by `margin` (corners moved along the hip lines, so the creases stay
        straight)."""
        hl, hw, m = self.half_length, self.half_width, margin
        if self.shape == ROOF_FLAT or self.rise <= 0.0:
            rect = [(hl + m, hw + m), (-hl - m, hw + m), (-hl - m, -hw - m), (hl + m, -hw - m)]
            return [(shapely.Polygon(self._xz(rect)), np.zeros(2), 0.0)]
        k = self.rise / hw
        L, S = self.long_axis, self.short_axis
        if self.shape == ROOF_GABLED:
            a = hl + m
            plus = [(a, 0.0), (a, hw + m), (-a, hw + m), (-a, 0.0)]
            minus = [(-a, 0.0), (-a, -hw - m), (a, -hw - m), (a, 0.0)]
            return [
                (shapely.Polygon(self._xz(plus)), -k * S, self.rise),
                (shapely.Polygon(self._xz(minus)), k * S, self.rise),
            ]
        a = hl - hw  # half the ridge length (0 for a pyramid)
        e, w = a + hw + m, hw + m
        return [
            (shapely.Polygon(self._xz([(e, w), (-e, w), (-a, 0.0), (a, 0.0)])), -k * S, self.rise),
            (shapely.Polygon(self._xz([(-e, -w), (e, -w), (a, 0.0), (-a, 0.0)])), k * S, self.rise),
            (shapely.Polygon(self._xz([(e, -w), (e, w), (a, 0.0)])), -k * L, k * hl),
            (shapely.Polygon(self._xz([(-e, w), (-e, -w), (-a, 0.0)])), k * L, k * hl),
        ]

    def creases(self, margin: float = 1.0) -> list[np.ndarray]:
        """The ridge and hip lines as (2, 2) XZ segments (empty for a flat roof)."""
        hl, hw, m = self.half_length, self.half_width, margin
        if self.shape == ROOF_FLAT or self.rise <= 0.0:
            return []
        if self.shape == ROOF_GABLED:
            return [self._xz([(-hl - m, 0.0), (hl + m, 0.0)])]
        a = hl - hw
        e, w = a + hw + m, hw + m
        out = [self._xz([(a, 0.0), (e, w)]), self._xz([(a, 0.0), (e, -w)]),
               self._xz([(-a, 0.0), (-e, w)]), self._xz([(-a, 0.0), (-e, -w)])]
        if a > 1e-9:
            out.append(self._xz([(-a, 0.0), (a, 0.0)]))
        return out


def _min_rect(poly: shapely.Polygon) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """(centre, long axis, short axis, half length, half width) of the minimum rotated
    rectangle. The axes are canonicalised (long axis with +X, or +Z when vertical; short = the
    long axis turned so the frame is fixed) so the result does not depend on vertex order."""
    rect = shapely.oriented_envelope(poly)
    c = np.asarray(rect.exterior.coords)[:4]
    centre = c.mean(axis=0)
    e0, e1 = c[1] - c[0], c[2] - c[1]
    n0, n1 = float(np.linalg.norm(e0)), float(np.linalg.norm(e1))
    if n0 < n1:
        e0, e1, n0, n1 = e1, e0, n1, n0
    long_axis = e0 / max(n0, 1e-12)
    if long_axis[0] < -1e-9 or (abs(long_axis[0]) <= 1e-9 and long_axis[1] < 0):
        long_axis = -long_axis
    short_axis = np.array([-long_axis[1], long_axis[0]])
    return centre, long_axis, short_axis, n0 / 2.0, n1 / 2.0


# --------------------------------------------------------------------------------------------
# Buildings
# --------------------------------------------------------------------------------------------


@dataclass
class Building:
    """One building ready to mesh. `ring` (K, 2) is the open XZ outline, clockwise in the
    (x, z) plane (so `left of each edge` is outside). Heights are metres above `ground_y`
    (median terrain under the footprint); walls start at `base_y`."""

    id: str
    source: str
    kind: str
    ring: np.ndarray
    roof: Roof
    eave_h: float
    ground_y: float
    base_y: float
    height_source: str
    wall_style: str
    roof_style: str
    road_distance_m: float = math.inf
    collide: bool = False
    u_offset: float = 0.0

    @property
    def polygon(self) -> shapely.Polygon:
        return shapely.Polygon(self.ring)

    @property
    def area_m2(self) -> float:
        return float(self.polygon.area)

    @property
    def eave_y(self) -> float:
        return self.ground_y + self.eave_h

    @property
    def ridge_h(self) -> float:
        return self.eave_h + self.roof.rise

    @property
    def ridge_y(self) -> float:
        return self.ground_y + self.ridge_h

    @property
    def centroid(self) -> np.ndarray:
        c = self.polygon.centroid
        return np.array([c.x, c.y])


def choose_height(
    lidar: np.ndarray | None,
    *,
    tag_height: float | None = None,
    levels: float | None = None,
    ms_height: float | None = None,
    max_lidar_m: float | None = None,
    options: BuildingOptions | None = None,
) -> tuple[str, float, float | None, float | None]:
    """(source, height, lidar median, lidar p90) by the rule in the module docstring.

    `height` is the roof's MEAN height for "lidar", the eave for "levels", the roof TOP for
    "height", "microsoft" and "default". `lidar` holds DSM - DTM samples inside the footprint
    (NaN = no data). A lidar median above `max_lidar_m` is taken for tree canopy over the roof
    and ignored (`place_buildings` sets it for small houses and sheds)."""
    o = options or BuildingOptions()
    lo, hi = o.height_clamp
    if lidar is not None:
        v = np.asarray(lidar, np.float64)
        v = v[np.isfinite(v)]
        if v.size >= o.min_lidar_samples:
            med = float(np.median(v))
            if med >= o.min_lidar_height_m and (max_lidar_m is None or med <= max_lidar_m):
                p90 = float(np.percentile(v, 90))
                return "lidar", float(np.clip(med, lo, hi)), med, p90
    if tag_height is not None and tag_height > 0:
        return "height", float(np.clip(tag_height, lo, hi)), None, None
    if levels is not None and levels > 0:
        return "levels", float(np.clip(levels * o.storey_m, lo, hi)), None, None
    if ms_height is not None and ms_height > 0:
        return "microsoft", float(np.clip(ms_height, lo, hi)), None, None
    return "default", float(np.clip(o.default_height_m, lo, hi)), None, None


def _roof_shape(
    fp: Footprint, area: float, half_width: float, aspect: float, spread: float | None,
    o: BuildingOptions, seed: object,
) -> str:
    tagged = _ROOF_TAGS.get((fp.roof_shape or "").strip().lower())
    if half_width * 2.0 < o.min_pitched_width_m:
        return ROOF_FLAT
    if tagged is not None:
        return tagged
    kind = fp.kind
    if kind in _FLAT_KINDS:
        return ROOF_FLAT
    barn = kind in _FARM_KINDS
    if spread is not None and spread < o.flat_lidar_spread_m:
        return ROOF_FLAT
    if area >= o.flat_roof_area_m2 and not barn and spread is None:
        return ROOF_FLAT
    if aspect < 1.4 and not barn and _u01(seed, fp.id, "hip") < o.hipped_share:
        return ROOF_HIPPED
    return ROOF_GABLED


def _styles(fp: Footprint, area: float, roof: str, seed: object) -> tuple[str, str]:
    kind = fp.kind
    uw, ur = _u01(seed, fp.id, "wall"), _u01(seed, fp.id, "roofstyle")
    if kind in _FLAT_KINDS or (roof == ROOF_FLAT and area >= 400.0):
        walls = _COMMERCIAL_WALLS
    elif kind in ("barn", "farm_auxiliary", "stable", "cowshed", "sty", "silo") or (
        kind == "yes" and area >= 450.0
    ):
        walls = _BARN_WALLS
    elif kind in _FARM_KINDS or area < 40.0:
        walls = _SHED_WALLS
    else:
        walls = _HOUSE_WALLS
    wall_style = _pick(walls, uw)
    if roof == ROOF_FLAT:
        roof_style = "flat_membrane" if area >= 60.0 else "metal_grey"
    elif walls is _BARN_WALLS:
        roof_style = _pick(_BARN_ROOFS, ur)
    else:
        roof_style = _pick(_HOUSE_ROOFS, ur)
    return wall_style, roof_style


def _to_polygon(fp: Footprint, projection: LocalProjection) -> shapely.Polygon | None:
    e, n = projection.to_enu(fp.lonlat[:, 0], fp.lonlat[:, 1])
    poly = shapely.Polygon(np.stack([e, -n], axis=1))
    if not poly.is_valid:
        poly = shapely.make_valid(poly)
    parts = [p for p in shapely.get_parts(poly) if isinstance(p, shapely.Polygon)]
    if not parts:
        return None
    poly = max(parts, key=lambda p: p.area)
    poly = shapely.Polygon(poly.exterior).simplify(0.05)
    if poly.is_empty or not isinstance(poly, shapely.Polygon) or not poly.is_valid:
        return None
    return shapely.orient_polygons(poly, exterior_cw=True)  # clockwise in (x, z)


def _lattice(poly: shapely.Polygon, step: float) -> np.ndarray:
    """Points of a `step` lattice inside `poly` (cell centres), (N, 2)."""
    if poly.is_empty:
        return np.zeros((0, 2))
    x0, z0, x1, z1 = poly.bounds
    xs = np.arange(x0 + step / 2, x1, step)
    zs = np.arange(z0 + step / 2, z1, step)
    if len(xs) == 0 or len(zs) == 0:
        return np.zeros((0, 2))
    gx, gz = np.meshgrid(xs, zs)
    pts = np.stack([gx.ravel(), gz.ravel()], axis=1)
    return pts[shapely.contains_xy(poly, pts[:, 0], pts[:, 1])]


def _road_frame(model: RoadModel) -> tuple[shapely.LineString, cKDTree, np.ndarray]:
    xz = model.centre[:, [0, 2]]
    return shapely.LineString(xz), cKDTree(xz), model.outer


def place_buildings(
    footprints: Sequence[Footprint],
    projection: LocalProjection,
    model: RoadModel,
    ground: GroundFn,
    *,
    corridor: shapely.Geometry | None = None,
    canopy: CanopyFn | None = None,
    options: BuildingOptions | None = None,
    seed: int | str = 0,
) -> list[Building]:
    """Turn footprints into `Building`s (module docstring, steps 1-5), sorted by id.

    `corridor` is the road corridor polygon in XZ (e.g. `walls.corridor_polygon(model,
    model.outer + verge)`); footprints touching it (grown by `clearance_m`) are dropped.
    `canopy(xz)` returns lidar DSM - DTM heights (e.g. `vegetation.CanopySampler`); None or
    NaN means no lidar, and the tag / default heights are used. `seed` should be the track id.
    """
    o = options or BuildingOptions()
    if not footprints:
        return []
    line, tree, outer = _road_frame(model)
    keep_out = corridor.buffer(o.clearance_m) if corridor is not None else None
    cand: list[tuple[float, Footprint, shapely.Polygon]] = []
    for fp in footprints:
        poly = _to_polygon(fp, projection)
        if poly is None or not (o.min_area_m2 <= poly.area <= o.max_area_m2):
            continue
        d_centre = float(poly.distance(line))
        if d_centre > o.max_distance_m:
            continue
        # Distance from the drivable edge: from the footprint to the centreline, minus the
        # road's half width + shoulder at the station nearest that closest approach.
        p_near = np.asarray(shapely.shortest_line(poly, line).coords)[1]
        _, j = tree.query(p_near)
        d_edge = d_centre - float(outer[j])
        if d_edge < o.clearance_m:
            continue
        if keep_out is not None and poly.intersects(keep_out):
            continue
        cand.append((d_edge, fp, poly))
    cand.sort(key=lambda t: (t[0], t[1].id))
    if len(cand) > o.max_buildings:
        log.info("building cap: keeping the %d nearest of %d", o.max_buildings, len(cand))
        cand = cand[: o.max_buildings]
    if not cand:
        return []

    # Ground under every footprint, one batched call.
    g_pts, g_owner = [], []
    for k, (_, _, poly) in enumerate(cand):
        pts = np.concatenate([np.asarray(poly.exterior.coords)[:-1], _lattice(poly, 3.0)])
        g_pts.append(pts)
        g_owner.append(np.full(len(pts), k))
    g_all = np.asarray(ground(np.concatenate(g_pts)), np.float64)
    g_owner_all = np.concatenate(g_owner)

    # Lidar samples inside every eroded footprint, one batched call.
    lidar: list[np.ndarray | None] = [None] * len(cand)
    eroded = np.zeros(len(cand))  # how far inside the outline the lidar was sampled
    if canopy is not None:
        c_pts, c_owner = [], []
        for k, (_, _, poly) in enumerate(cand):
            inner = poly.buffer(-o.erode_m)
            pts = _lattice(inner, o.sample_step_m)
            eroded[k] = o.erode_m
            if len(pts) < o.min_lidar_samples:
                pts = _lattice(poly, o.sample_step_m / 2.0)
                eroded[k] = 0.0
            c_pts.append(pts)
            c_owner.append(np.full(len(pts), k))
        pts_all = np.concatenate(c_pts)
        chm = None
        if len(pts_all):
            try:
                chm = np.asarray(canopy(pts_all), np.float64).reshape(-1)
            except (OSError, RuntimeError, ValueError) as exc:  # SourceUnavailable: RuntimeError
                log.warning("lidar building heights unavailable, using tags/defaults: %s", exc)
        if chm is not None:
            owner = np.concatenate(c_owner)
            for k in range(len(cand)):
                lidar[k] = chm[owner == k]

    out: list[Building] = []
    for k, (d_edge, fp, poly) in enumerate(cand):
        g = g_all[g_owner_all == k]
        g = g[np.isfinite(g)]
        if g.size == 0:
            continue
        ground_y, base_y = float(np.median(g)), float(g.min()) - o.sink_m
        centre, L, S, hl, hw = _min_rect(poly)
        small = poly.area < o.small_area_m2 and fp.kind in _SMALL_CAPPED_KINDS
        src, h, med, p90 = choose_height(
            lidar[k], tag_height=fp.height_m, levels=fp.levels, ms_height=fp.ms_height_m,
            max_lidar_m=o.small_max_lidar_m if small else None, options=o,
        )
        spread = (p90 - med) if (med is not None and p90 is not None) else None
        shape = _roof_shape(fp, poly.area, hw, hl / max(hw, 1e-9), spread, o, seed)
        rise = 0.0
        ridge = None
        rise_cap = min(o.max_roof_rise_m, math.tan(math.radians(o.max_pitch_deg)) * hw)
        if shape != ROOF_FLAT:
            if spread is not None:
                # Uniform slope samples: p90 - median = 0.4 x the rise they span. The eroded
                # lattice (cell centres) only spans the upper (hw - erode - step / 2) / hw of
                # each slope, so scale up.
                # Pitch capped at `max_pitch_deg` (overhanging trees inflate p90).
                visible = 2.5 * spread
                span = max(0.3, (hw - eroded[k] - 0.5 * o.sample_step_m) / hw)
                rise = float(np.clip(visible / span, 0.8, max(0.8, rise_cap)))
                visible = min(visible, rise * span)
                ridge = min(med + 0.5 * visible, o.height_clamp[1])
            else:
                lo_p, hi_p = o.pitch_deg
                pitch = lo_p + (hi_p - lo_p) * _u01(seed, fp.id, "pitch")
                rise = math.tan(math.radians(pitch)) * hw
            rise = float(np.clip(rise, 0.8, max(0.8, rise_cap)))
        # Eave from the chosen height (see choose_height for what `h` means per source).
        if shape == ROOF_FLAT:
            eave = h
        elif src == "lidar":
            eave = (ridge if ridge is not None else h + 0.5 * rise) - rise
        elif src == "levels":
            eave = h
        else:
            eave = h - rise
        if eave < o.min_eave_m:
            rise = max(0.0, rise - (o.min_eave_m - eave))
            eave = o.min_eave_m
            if rise < 0.5:
                shape, rise = ROOF_FLAT, 0.0
        eave = min(eave, o.height_clamp[1])
        wall_style, roof_style = _styles(fp, poly.area, shape, seed)
        roof = Roof(shape, centre, L, S, hl, hw, rise if shape != ROOF_FLAT else 0.0)
        out.append(
            Building(
                id=fp.id,
                source=fp.source,
                kind=fp.kind,
                ring=np.asarray(poly.exterior.coords)[:-1].copy(),
                roof=roof,
                eave_h=float(eave),
                ground_y=ground_y,
                base_y=base_y,
                height_source=src,
                wall_style=wall_style,
                roof_style=roof_style,
                road_distance_m=float(d_edge),
                collide=o.collide_within_m >= 0 and d_edge <= o.collide_within_m,
                u_offset=round(_u01(seed, fp.id, "u") * ATLAS_U_M, 3),
            )
        )
    return sorted(out, key=lambda b: b.id)


def buildings_polygon(buildings: Sequence[Building], buffer_m: float = 2.0) -> shapely.Geometry:
    """Union of the footprints grown by `buffer_m` (XZ), e.g. to keep trees off the houses."""
    if not buildings:
        return shapely.Polygon()
    polys = np.array([shapely.Polygon(b.ring) for b in buildings])
    return shapely.union_all(shapely.buffer(polys, buffer_m))


# --------------------------------------------------------------------------------------------
# Extrusion
# --------------------------------------------------------------------------------------------


def _seg_cross(p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray) -> float | None:
    """Parameter t in (0, 1) where segment p-q crosses segment a-b, or None."""
    r, s = q - p, b - a
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-12:
        return None
    w = a - p
    t = (w[0] * s[1] - w[1] * s[0]) / den
    u = (w[0] * r[1] - w[1] * r[0]) / den
    if 1e-9 < t < 1 - 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return float(t)
    return None


def _noded_ring(b: Building, storey_lines: np.ndarray, margin: float) -> np.ndarray:
    """The ring with extra vertices where it crosses a roof crease, then where its roof edge
    (eave_y + roof height) crosses a storey line, so roof and wall tops are linear between
    consecutive vertices and no wall row is cut inside an edge."""
    ring = b.ring
    creases = b.roof.creases(margin)
    pts: list[np.ndarray] = []
    n = len(ring)
    for i in range(n):
        p, q = ring[i], ring[(i + 1) % n]
        ts = sorted(t for a_b in creases if (t := _seg_cross(p, q, a_b[0], a_b[1])) is not None)
        pts.append(p)
        pts.extend(p + t * (q - p) for t in ts)
    ring1 = np.asarray(pts)
    top = b.eave_y + b.roof.height(ring1)
    pts = []
    n = len(ring1)
    for i in range(n):
        p, q = ring1[i], ring1[(i + 1) % n]
        ya, yb = top[i], top[(i + 1) % n]
        pts.append(p)
        if abs(yb - ya) > 1e-9:
            lo, hi = min(ya, yb), max(ya, yb)
            cut = storey_lines[(storey_lines > lo + 1e-6) & (storey_lines < hi - 1e-6)]
            ts = np.sort((cut - ya) / (yb - ya))
            pts.extend(p + t * (q - p) for t in ts)
    ring2 = np.asarray(pts)
    step = np.linalg.norm(np.roll(ring2, -1, axis=0) - ring2, axis=1)
    return ring2[step > 1e-7]


# Gable ends above the eave use the window-free strip of a wall band between these heights
# above the storey floor (below every painted sill, building_textures._paint_windows).
_PLAIN_STRIP_M = (0.05, 0.8)
_FOUNDATION_MIN_M = 0.3  # extra drop below the sink before a separate foundation row is cut


def _wall_rows(b: Building, storey_m: float, sink_m: float = 0.5) -> np.ndarray:
    """Horizontal wall rows (R, 4): (y_lo, y_hi, band fraction at y_lo, at y_hi), bottom up.

    Band fraction 0 = the band's top (v_top), 1 = its bottom. Rows: a whole number of storeys
    between ground and eave (the storey height stretches by at most 1.5x, so the eave never
    cuts a window); the first storey reaches down to `base_y`, unless the ground under the
    building drops more than `sink_m` + 0.3 m below its median, where a plain foundation row
    (bottom 10 % of the band) is cut instead; then one row from the eave to the ridge (gable
    ends) on the plain strip under the sills, stretched (few vertices; reads as wide boards).
    """
    h = b.eave_y - b.ground_y
    n = max(1, round(h / storey_m))
    floors = b.ground_y + np.arange(n + 1) * (h / n)
    rows = [(floors[k], floors[k + 1], 1.0, 0.0) for k in range(n)]
    if b.ground_y - b.base_y > sink_m + _FOUNDATION_MIN_M:
        rows.insert(0, (b.base_y, b.ground_y, 1.0, 0.9))
    else:
        rows[0] = (b.base_y, floors[1], 1.0, 0.0)
    if b.ridge_y > b.eave_y + 1e-6:
        p_lo, p_hi = _PLAIN_STRIP_M
        rows.append((b.eave_y, b.ridge_y, 1.0 - p_lo / STOREY_M, 1.0 - p_hi / STOREY_M))
    return np.asarray(rows, np.float64)


def _dedupe_loop(poly: list[tuple[float, float]], tol: float = 1e-9) -> list[tuple[float, float]]:
    """Drop consecutive (and closing) duplicate points of a small polygon loop."""
    out: list[tuple[float, float]] = []
    for pt in poly:
        if not out or abs(pt[0] - out[-1][0]) + abs(pt[1] - out[-1][1]) > tol:
            out.append(pt)
    if len(out) > 1 and abs(out[-1][0] - out[0][0]) + abs(out[-1][1] - out[0][1]) <= tol:
        out.pop()
    return out


def _walls(b: Building, ring: np.ndarray, rows: np.ndarray, size: int) -> MeshData:
    v_top, v_bot = band_v(b.wall_style, size)
    top = b.eave_y + b.roof.height(ring)
    n = len(ring)
    seg = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)
    s0 = np.concatenate([[0.0], np.cumsum(seg)[:-1]]) + b.u_offset
    pos, nrm, uvs, idx = [], [], [], []
    count = 0
    for i in range(n):
        a, c = ring[i], ring[(i + 1) % n]
        length = seg[i]
        d = (c - a) / length
        normal = np.array([-d[1], 0.0, d[0]])  # left of the edge = outside (clockwise ring)
        ya, yb = top[i], top[(i + 1) % n]
        for lo, hi, f_lo, f_hi in rows:
            if lo >= max(ya, yb) - 1e-9:
                break
            poly = [(0.0, lo), (length, lo), (length, min(hi, yb)), (0.0, min(hi, ya))]
            poly = _dedupe_loop(poly)
            if len(poly) < 3:
                continue
            uv_ = np.asarray(poly)
            xyz = np.empty((len(poly), 3))
            xyz[:, 0] = a[0] + uv_[:, 0] * d[0]
            xyz[:, 2] = a[1] + uv_[:, 0] * d[1]
            xyz[:, 1] = uv_[:, 1]
            u = (s0[i] + uv_[:, 0]) / ATLAS_U_M
            t = np.clip((uv_[:, 1] - lo) / (hi - lo), 0.0, 1.0)
            v = v_top + (f_lo + t * (f_hi - f_lo)) * (v_bot - v_top)
            pos.append(xyz)
            nrm.append(np.tile(normal, (len(poly), 1)))
            uvs.append(np.stack([u, v], axis=1))
            fan = [(count, count + j, count + j + 1) for j in range(1, len(poly) - 1)]
            idx.extend(fan)
            count += len(poly)
    if not pos:
        return _empty()
    return MeshData(
        np.concatenate(pos), np.concatenate(nrm), np.concatenate(uvs),
        np.tile([1.0, 0.0, 0.0], (count, 1)), np.asarray(idx, np.int64).reshape(-1),
    )


def _empty() -> MeshData:
    z3 = np.zeros((0, 3))
    return MeshData(z3, z3, np.zeros((0, 2)), z3, np.zeros(0, np.int64))


def _triangles(poly: shapely.Geometry) -> np.ndarray:
    """Constrained Delaunay triangles of a polygon (no Steiner points), (T, 3, 2)."""
    tris = shapely.get_parts(shapely.constrained_delaunay_triangles(poly))
    if len(tris) == 0:
        return np.zeros((0, 3, 2))
    return np.stack([np.asarray(t.exterior.coords)[:3] for t in tris])


def _snap(xz: np.ndarray, ring: np.ndarray, tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Move points within `tol` of a ring vertex onto it; returns (points, snapped mask)."""
    d, j = cKDTree(ring).query(xz)
    hit = d < tol
    out = xz.copy()
    out[hit] = ring[j[hit]]
    return out, hit


def _planar_piece(
    tris: np.ndarray, y: np.ndarray, normal: np.ndarray, u: np.ndarray, v: np.ndarray, up: bool
) -> MeshData:
    """Triangles (T, 3, 2) with heights (T, 3) as one mesh, wound to face along `normal`
    (`up`: +Y side, else -Y), vertices shared, UVs flipped in u if they would be mirrored."""
    t = len(tris)
    p3 = np.empty((t, 3, 3))
    p3[..., 0] = tris[..., 0]
    p3[..., 1] = y
    p3[..., 2] = tris[..., 1]
    cr = np.cross(p3[:, 1] - p3[:, 0], p3[:, 2] - p3[:, 0])
    area = np.linalg.norm(cr, axis=1)
    keep = area > 1e-9
    p3, cr, u, v = p3[keep], cr[keep], u[keep], v[keep]
    if len(p3) == 0:
        return _empty()
    wrong = (cr[:, 1] < 0) if up else (cr[:, 1] > 0)
    order = np.where(wrong[:, None], [0, 2, 1], [0, 1, 2])
    rows = np.arange(len(p3))[:, None]
    p3, u, v = p3[rows, order], u[rows, order], v[rows, order]
    # Not mirrored: front-facing triangles have negative signed area in (u, v) (v down).
    du1, dv1 = u[:, 1] - u[:, 0], v[:, 1] - v[:, 0]
    du2, dv2 = u[:, 2] - u[:, 0], v[:, 2] - v[:, 0]
    signed = du1 * dv2 - du2 * dv1
    if np.sum(signed) > 0:
        u = -u
    flat_p = p3.reshape(-1, 3)
    flat_uv = np.stack([u.reshape(-1), v.reshape(-1)], axis=1)
    key = np.round(np.concatenate([flat_p, flat_uv], axis=1), 9)
    uniq, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    n = len(uniq)
    return MeshData(
        flat_p[first], np.tile(normal, (n, 1)), flat_uv[first], np.tile([1.0, 0.0, 0.0], (n, 1)),
        inv.reshape(-1),
    )


def _roof(b: Building, ring: np.ndarray, size: int, margin: float) -> list[MeshData]:
    v_top, v_bot = band_v(b.roof_style, size)
    foot = shapely.Polygon(ring)
    if not foot.is_valid:
        foot = shapely.make_valid(foot)
    ring_top = b.eave_y + b.roof.height(ring)
    out = []
    for region, grad, h_c in b.roof.regions(margin):
        piece = shapely.intersection(foot, region, grid_size=1e-7)
        polys = [p for p in shapely.get_parts(piece) if isinstance(p, shapely.Polygon)]
        polys = [p for p in polys if p.area > 1e-8]
        if not polys:
            continue
        tris = np.concatenate([_triangles(p) for p in polys])
        if len(tris) == 0:
            continue
        flat_xz, hit = _snap(tris.reshape(-1, 2), ring)
        tris = flat_xz.reshape(-1, 3, 2)
        h = h_c + (tris - b.roof.centre) @ grad
        y = b.eave_y + h
        # Exact wall-top heights on the ring (bitwise equal to the walls').
        _, j = cKDTree(ring).query(flat_xz)
        y = np.where(hit.reshape(-1, 3), ring_top[j].reshape(-1, 3), y)
        gnorm = float(np.linalg.norm(grad))
        if gnorm > 1e-12:
            right = np.array([-grad[1], grad[0]]) / gnorm
            u = (tris @ right) / ATLAS_U_M + b.u_offset / ATLAS_U_M
            frac = np.clip((b.roof.rise - h) / max(b.roof.rise, 1e-9), 0.0, 1.0)
        else:
            rel = tris - b.roof.centre
            u = (rel @ b.roof.long_axis + b.u_offset) / ATLAS_U_M
            frac = np.clip(0.5 + (rel @ b.roof.short_axis) / (2 * b.roof.half_width + 1e-9),
                           0.0, 1.0)
        v = v_top + frac * (v_bot - v_top)
        normal = np.array([-grad[0], 1.0, -grad[1]])
        normal /= np.linalg.norm(normal)
        out.append(_planar_piece(tris, y, normal, u, v, up=True))
    return out


def extrude_building(
    b: Building,
    *,
    storey_m: float = 3.0,
    sink_m: float = 0.5,
    floor: bool = False,
    atlas_size: int = ATLAS_SIZE,
) -> MeshData:
    """The building's visual mesh: walls (cut at storey lines) + roof (+ a floor at `base_y`
    with `floor=True`, which makes the solid closed). Flat outward normals, tangents along u."""
    margin = 1.0
    rows = _wall_rows(b, storey_m, sink_m)
    ring = _noded_ring(b, np.unique(rows[:, :2]), margin)
    parts = [_walls(b, ring, rows, atlas_size), *_roof(b, ring, atlas_size, margin)]
    if floor:
        tris = _triangles(shapely.Polygon(ring))
        if len(tris):
            tris, _ = _snap(tris.reshape(-1, 2), ring)
            tris = tris.reshape(-1, 3, 2)
            y = np.full(tris.shape[:2], b.base_y)
            u = tris[..., 0] / ATLAS_U_M
            v0, v1 = band_v(b.wall_style, atlas_size)
            v = np.full(tris.shape[:2], 0.5 * (v0 + v1))
            parts.append(_planar_piece(tris, y, np.array([0.0, -1.0, 0.0]), u, v, up=False))
    parts = [p for p in parts if p.vertex_count]
    if not parts:
        return _empty()
    return compute_tangents(merge(parts))


def _collision(b: Building, extra_m: float) -> MeshData | None:
    base = np.stack([b.ring[:, 0], np.full(len(b.ring), b.base_y), b.ring[:, 1]], axis=1)
    tall = (b.eave_y + extra_m) - b.base_y
    if tall <= 0.0:
        return None
    try:
        return wall(base, tall, double_sided=True, closed=True, u_tile_m=4.0)
    except ValueError:
        return None


# --------------------------------------------------------------------------------------------
# Meshes and scene
# --------------------------------------------------------------------------------------------


@dataclass
class BuildingMesh:
    """One batch, ready for `Scene.mesh(data, name, material, **flags)`. `collision` meshes
    are the hidden `1WALL_bld_*` walls."""

    name: str
    data: MeshData
    building_count: int
    collision: bool
    flags: MeshFlags


@dataclass
class BuildingStats:
    buildings: int = 0
    colliding: int = 0
    meshes: int = 0
    vertices: int = 0
    triangles: int = 0
    texture_bytes: int = 0
    by_height_source: dict[str, int] = field(default_factory=dict)
    by_roof: dict[str, int] = field(default_factory=dict)


def building_material(name: str = "dm_buildings", texture_name: str = "dm_buildings.dds"):
    """Plain `ksPerPixel` on the atlas (ac_format.md §5 v1 values, matte)."""
    return Material(
        name=name,
        shader="ksPerPixel",
        properties=[
            MaterialProperty("ksAmbient", a=0.45),
            MaterialProperty("ksDiffuse", a=0.6),
            MaterialProperty("ksSpecular", a=0.05),
            MaterialProperty("ksSpecularEXP", a=10.0),
            MaterialProperty("ksEmissive", c=(0.0, 0.0, 0.0)),
            MaterialProperty("ksAlphaRef", a=0.0),
        ],
        textures=[TextureMapping("txDiffuse", 0, texture_name)],
    )


def building_meshes(
    buildings: Sequence[Building], *, options: BuildingOptions | None = None
) -> list[BuildingMesh]:
    """Batch the buildings per `cell_m` cell: `buildings_<nnn>` (visual, `lod_out_m`) and, for
    buildings with `collide`, hidden `1WALL_bld_<nnn>` walls. Cells are numbered in sorted
    order; `_<p>` suffixes appear when a cell needs several meshes (> 65,535 vertices)."""
    o = options or BuildingOptions()
    if not buildings:
        return []
    cells: dict[tuple[int, int], list[Building]] = {}
    for b in buildings:
        key = tuple(int(v) for v in np.floor(b.centroid / o.cell_m))
        cells.setdefault(key, []).append(b)
    out: list[BuildingMesh] = []
    visual_flags: MeshFlags = {"lod_out": o.lod_out_m, "cast_shadows": o.cast_shadows}
    hidden: MeshFlags = {"is_renderable": False, "cast_shadows": False}
    for ci, key in enumerate(sorted(cells)):
        members = sorted(cells[key], key=lambda b: b.id)
        parts = [extrude_building(b, storey_m=o.storey_m, sink_m=o.sink_m,
                                  atlas_size=o.atlas_size)
                 for b in members]
        parts = [p for p in parts if p.vertex_count]
        if parts:
            pieces = split(merge(parts))
            for p, piece in enumerate(pieces):
                name = f"buildings_{ci:03d}" + (f"_{p}" if len(pieces) > 1 else "")
                out.append(BuildingMesh(name, piece, len(members), False, dict(visual_flags)))
        walls_ = [w for b in members if b.collide and (w := _collision(b, o.collide_extra_m))]
        if walls_:
            pieces = split(merge(walls_))
            for p, piece in enumerate(pieces):
                name = f"1WALL_bld_{ci:03d}" + (f"_{p}" if len(pieces) > 1 else "")
                out.append(BuildingMesh(name, piece, len(walls_), True, dict(hidden)))
    return out


def building_scene(
    scene: Scene,
    buildings: Sequence[Building],
    *,
    texture_format: str = "dds",
    atlas_seed: int = ATLAS_SEED,
    material_name: str = "dm_buildings",
    options: BuildingOptions | None = None,
) -> BuildingStats:
    """Add the buildings to a KN5 `Scene`: the atlas texture, one material, the batched
    meshes. Returns counts for the build summary. Adds nothing if there are no buildings."""
    o = options or BuildingOptions()
    meshes = building_meshes(buildings, options=o)
    stats = BuildingStats()
    if not meshes:
        return stats
    tex_name = f"dm_buildings.{'png' if texture_format == 'png' else 'dds'}"
    if tex_name not in scene.textures:
        scene.textures[tex_name] = building_atlas_texture(tex_name, o.atlas_size, atlas_seed)
    mat_id = next((i for i, m in enumerate(scene.materials) if m.name == material_name), None)
    if mat_id is None:
        scene.materials.append(building_material(material_name, tex_name))
        mat_id = len(scene.materials) - 1
    for m in meshes:
        scene.mesh(m.data, m.name, mat_id, **m.flags)
        stats.meshes += 1
        stats.vertices += m.data.vertex_count
        stats.triangles += m.data.triangle_count
    stats.buildings = len(buildings)
    stats.colliding = sum(b.collide for b in buildings)
    for b in buildings:
        stats.by_height_source[b.height_source] = stats.by_height_source.get(b.height_source, 0) + 1
        stats.by_roof[b.roof.shape] = stats.by_roof.get(b.roof.shape, 0) + 1
    stats.texture_bytes = len(scene.textures[tex_name].data)
    return stats
