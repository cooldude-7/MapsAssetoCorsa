"""Buildings beside the road: footprints -> heights, roofs, doors, textured KN5 meshes.

Decisions: docs/research/scenery.md §3 and §7 ("Buildings", "Budgets"), geo.md §4, and the
builders' buildings.md (2026-10-16: sources, conflation, heights, typical Ontario sizes).

**Placement** (`place_buildings`), in AC coordinates (X = east, Y = up, Z = south):

1. Footprints (`geo.buildings.Footprint`, lon/lat, already conflated by
   `geo.buildings.conflate`) are projected to XZ, repaired, holes dropped (drawn solid), and
   kept when `min_area_m2` (10 m2: garages, sheds and barns count) <= area <= `max_area_m2`,
   within `max_distance_m` (300 m) of the centreline, and clear of the road: in the country a
   footprint that touches the corridor polygon (road + verge, as the hidden walls use) grown
   by `clearance_m`, or that comes within `clearance_m` of the road's drivable edge, is
   dropped (never clipped); in a town (step 3), where buildings stand at the sidewalk inside
   the wall corridor, only one within `town_clearance_m` (1 m) of the drivable surface (the
   route, `paved` side roads and junctions) is.
2. Ground: the built terrain (`ground(xz)`, not the raw DTM: cut/fill changed it) under the
   ring and a 3 m interior lattice. The walls start `sink_m` (0.5 m) below its LOWEST point;
   heights are measured from its median.
3. Context and category. A footprint is in a **dense town** (`_town_context`) with at least
   20 other footprints of 60 m2 or more within 100 m (a subdivision has 40-80), in a **town**
   with at least 10 unless most are barn-sized (> 450 m2) or the NRCan land cover is known
   and under 30 % urban (a hamlet among fields), else in the **country**.
   Category: house, townhouse, garage, shed, barn or commercial, from the type (OSM
   `building`, the municipal type) or, untyped, the size (buildings.md §5: sheds < 25 m2,
   garages < 70 m2, barns >= 450 m2 unless flat). Barns only stand in the country (the
   user's Oakville check, 2026-10-16: town houses and rows drawn as barns): in a town a big
   footprint is a house, or commercial when its roof is flat, and in a dense town a long,
   house-deep one (>= 18 m by 7-16 m, >= 180 m2) a townhouse row; only an OSM-tagged barn
   stays a barn.
4. Heights, the first that applies (buildings.md §4):
   - **measured**: municipal eave + peak (Burlington EAVEHGT / PEAKHGT, Milton): the roof
     exactly; a municipal or NRCan lidar peak (`heightmax`, median +0.24 m against Burlington)
     alone: the peak, the pitch as below.
   - **lidar** DSM - DTM (NRCan HRDEM, OGL-Canada) on a 1 m lattice inside the footprint
     eroded by 1 m: when at least `min_lidar_samples` are valid and their median is >=
     `min_lidar_height_m`. The median and p90 give the ridge (a gable sampled uniformly has
     median = eave + rise / 2, p90 = eave + 0.9 rise, corrected for the eroded lattice). On a
     small house or shed (< 400 m2) a median above 12 m is tree canopy over the roof: ignored.
   - OSM `height` (roof top), then `building:levels` x `storey_m` (3 m, the eave), then the
     Microsoft ML height (the MEAN height over the footprint = mid-roof: median -0.15 m
     against Burlington's mid-roof, -1.30 m against the peak), then a typical peak per
     category (shed 3.2 m, garage 4 m, house 6 m, barn 8.5 m; rural Caledon / Burlington).
   - **Pitch**, without a measured eave: from the lidar spread only when it is clearly steep
     (>= `steep_pitch_deg`) or clearly low (<= `low_pitch_deg`); otherwise drawn (per
     building, deterministic) from the measured Ontario distribution (Burlington, n = 12,000:
     p10 14.8, median 22.5, p90 33.7 degrees); barns steeper, sheds lower.
5. Roof: OSM `roof:shape` when tagged; flat for commercial kinds, a measured peak - eave <
   0.6 m, big untyped footprints without lidar, narrow ones (< `min_pitched_width_m`) and
   lidar spreads < `flat_lidar_spread_m`; otherwise pitched: hipped for some near-square
   houses (deterministic), gabled for the rest. The ridge runs along the long axis of the
   footprint's minimum rotated rectangle.
6. Look: wall and roof styles from `ac.building_textures` by category and context, chosen
   by a hash of (seed, building id), so a rebuild gives the same street. Country houses:
   brick, vinyl siding and stone with asphalt shingles or some metal roofs; barns: red /
   grey boards or white steel with metal roofs. Town houses: mostly brick (grey-white, pale
   red, buff, brown, red), some stone and stucco, a little vinyl, shingle roofs; townhouse
   rows brick or stone. A garage within 35 m of a house wears its colours, a shed usually.
   The FRONT wall faces the nearest road (`roads`, the route by default): houses get a front
   door there (and, if wide, an attached garage's door at one end), each townhouse unit a
   front door and usually a single garage door, garages their garage doors, barns a sliding
   or steel door, sheds a door. House walls show whole windows, centred per wall; garage
   walls are blank.
7. Budget: every building within `frontage_m` (60 m) of the route's drivable edge (the
   street front) is always placed; then the rest in order of how far they are from the road,
   relative to their size (a big barn 200 m away comes before a shed at 100 m), until
   `budget_bytes` of mesh (44 B a vertex + 6 B a triangle in the KN5), `town_budget_bytes`
   (12 MB) when most candidates stand in a town, where the outlines of buildings beyond the
   street front are also simplified to 0.5 m (fewer wall quads).

**Meshes** (`building_meshes`, `building_scene`): the roof is a piecewise-planar height field
over the minimum rotated rectangle (gable: 2 planes split by the ridge; hip: 2 trapezoids + 2
triangles; flat: 1 plane) cut to the REAL footprint, so L-shaped and irregular houses get one
roof whose gable-end walls follow it. The footprint ring is noded where it crosses a roof
crease and where its roof edge crosses a storey line; walls are cut at storey lines (every wall
quad stays inside one atlas band, see `ac.building_textures`), roof pieces are triangulated
with constrained Delaunay on the same ring vertices, so walls and roof meet exactly (the solid
is closed apart from its open bottom; `floor=True` closes it too, for tests). Doors are quads
3 cm in front of their wall. Normals are flat and outward, winding CCW from outside, UVs never
mirrored.

Batched per `cell_m` (250 m) cell into visual-only meshes (no digit prefix, no collision):
`buildings_<nnn>` with `lod_out_m` (1200 m) and, for small buildings (< `detail_area_m2`,
sheds and small garages), `buildings_s<nnn>` with `detail_lod_out_m` (400 m: they are a few
pixels beyond). One `ksPerPixel` material on one BC1 atlas, so a batch is one draw call.
Buildings whose footprint is within `collide_within_m` (15 m) of the road's drivable edge also
get hidden, double-sided collision walls `1WALL_bld_<nnn>` (ac_format.md §3.2), up to the eave.

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
    DOORS,
    STOREY_M,
    WINDOW_ZONE_M,
    WINDOWED,
    band_v,
    building_atlas_texture,
    door_uv,
    wall_u_start,
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
    "CATEGORIES",
    "ROOF_FLAT",
    "ROOF_GABLED",
    "ROOF_HIPPED",
    "Building",
    "BuildingMesh",
    "BuildingOptions",
    "BuildingStats",
    "CanopyFn",
    "Door",
    "GroundFn",
    "Roof",
    "building_material",
    "building_meshes",
    "building_scene",
    "buildings_polygon",
    "choose_height",
    "extrude_building",
    "mesh_bytes",
    "ontario_pitch",
    "place_buildings",
]

log = logging.getLogger(__name__)

GroundFn = Callable[[np.ndarray], np.ndarray]  # XZ (N, 2) -> terrain Y (N,), NaN where none
CanopyFn = Callable[[np.ndarray], np.ndarray]  # XZ (N, 2) -> DSM - DTM metres, NaN where none
LandcoverFn = Callable[[np.ndarray], np.ndarray]  # XZ (N, 2) -> NRCan 2020 class, 0 = unknown

ROOF_FLAT = "flat"
ROOF_GABLED = "gabled"
ROOF_HIPPED = "hipped"
CATEGORIES = ("house", "townhouse", "garage", "shed", "barn", "commercial")

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
_FLAT_KINDS = frozenset(
    {
        "commercial",
        "industrial",
        "retail",
        "warehouse",
        "office",
        "school",
        "supermarket",
        "hospital",
        "public",
        "civic",
        "government",
        "college",
        "university",
        "hotel",
        "apartments",
        "parking",
        "service",
        "transportation",
        "train_station",
        "fire_station",
        "sports_hall",
        "kindergarten",
        "manufacture",
        "storage_tank",
    }
)
_BARN_KINDS = frozenset(
    {
        "barn",
        "farm_auxiliary",
        "stable",
        "cowshed",
        "sty",
        "silo",
        "greenhouse",
        "riding_hall",
    }
)
_GARAGE_KINDS = frozenset({"garage", "garages", "carport", "boathouse"})
_SHED_KINDS = frozenset({"shed", "hut", "outbuilding", "kiosk"})
_HOUSE_KINDS = frozenset(
    {
        "house",
        "detached",
        "residential",
        "semidetached_house",
        "bungalow",
        "cabin",
        "terrace",
        "farm",
        "church",
        "chapel",
        "static_caravan",
        "dormitory",
    }
)
_SMALL_CAPPED_KINDS = frozenset(
    {
        "yes",
        "house",
        "detached",
        "residential",
        "semidetached_house",
        "bungalow",
        "cabin",
        "garage",
        "garages",
        "carport",
        "shed",
        "hut",
        "outbuilding",
        "farm_auxiliary",
    }
)

# (style, weight) tables; deterministic draws per building. ASSUMED Ontario mixes.
_HOUSE_WALLS = (
    ("brick_red", 15),
    ("brick_brown", 11),
    ("brick_buff", 9),
    ("brick_orange", 4),
    ("brick_charcoal", 4),
    ("stone", 4),
    ("siding_white", 13),
    ("siding_cream", 10),
    ("siding_grey", 10),
    ("siding_tan", 6),
    ("siding_blue", 5),
    ("brick_grey", 3),
    ("siding_brown", 3),
    ("board_white", 2),
    ("stucco", 2),
)
_HOUSE_ROOFS = (
    ("shingle_charcoal", 38),
    ("shingle_grey", 22),
    ("shingle_brown", 18),
    ("shingle_black", 14),
    ("metal_charcoal", 4),
    ("metal_grey", 2),
    ("metal_green", 2),
)
_BARN_WALLS = (
    ("barn_red", 40),
    ("barn_grey", 25),
    ("steel_white", 25),
    ("shed_wood", 10),
)
_BARN_ROOFS = (
    ("metal_grey", 45),
    ("metal_charcoal", 16),
    ("metal_green", 12),
    ("metal_red", 12),
    ("metal_blue", 5),
    ("shingle_charcoal", 10),
)
_SHED_WALLS = (
    ("shed_wood", 30),
    ("siding_white", 15),
    ("siding_grey", 15),
    ("siding_cream", 10),
    ("barn_red", 15),
    ("steel_white", 15),
)
_SHED_ROOFS = (
    ("shingle_charcoal", 40),
    ("shingle_brown", 15),
    ("shingle_grey", 15),
    ("metal_grey", 20),
    ("metal_green", 10),
)
_COMMERCIAL_WALLS = (("commercial", 60), ("brick_red", 15), ("brick_buff", 15),
                     ("steel_white", 10))  # fmt: skip
# In towns (subdivisions, downtowns: `_town_context`) houses wear what GTA builders have used
# since the 1970s: mostly clay brick (grey-white and pale red blends above all in newer
# Oakville / Burlington subdivisions, buff, brown, red), some stone and stucco fronts, and
# vinyl siding mostly on older or rural houses (the user's Oakville check, 2026-10-16;
# weights ASSUMED, no cladding statistics exist). No board-and-batten, no metal roofs.
_TOWN_HOUSE_WALLS = (
    ("brick_grey", 16),
    ("brick_pale_red", 14),
    ("brick_buff", 12),
    ("brick_brown", 12),
    ("brick_red", 9),
    ("brick_charcoal", 4),
    ("stone", 9),
    ("stucco", 4),
    ("siding_white", 5),
    ("siding_grey", 5),
    ("siding_cream", 4),
    ("siding_tan", 3),
    ("siding_blue", 3),
)
_TOWNHOUSE_WALLS = (
    ("brick_grey", 30),
    ("brick_pale_red", 25),
    ("brick_buff", 15),
    ("brick_brown", 15),
    ("brick_red", 8),
    ("stone", 7),
)
_TOWN_HOUSE_ROOFS = (
    ("shingle_charcoal", 45),
    ("shingle_black", 25),
    ("shingle_grey", 15),
    ("shingle_brown", 15),
)
_TOWN_COMMERCIAL_WALLS = (("commercial", 45), ("brick_red", 15), ("brick_buff", 15),
                          ("brick_brown", 10), ("brick_grey", 10), ("stucco", 5))  # fmt: skip
_DEFAULT_PEAK_M = {"shed": 3.2, "garage": 4.0, "house": 6.0, "townhouse": 9.5, "barn": 8.5,
                   "commercial": 6.0}  # fmt: skip
_MIN_EAVE_M = {"shed": 2.0, "garage": 2.3, "house": 2.4, "townhouse": 2.4, "barn": 3.0,
               "commercial": 3.0}  # fmt: skip
_MAX_RISE_M = {"shed": 2.5, "garage": 3.5, "house": 6.0, "townhouse": 4.0, "barn": 9.0,
               "commercial": 4.0}  # fmt: skip
# A commercial building (shops, offices, apartments) with a measured eave and peak keeps its
# flat roof unless the peak is at least this far above the eave on a low building (a real
# pitched roof); on one this tall the peak is a rooftop penthouse (Burlington PEAKHGT,
# ASSUMED). The flat roof then stands a parapet above the eave.
_COMMERCIAL_PITCH_M = 2.5
_TALL_EAVE_M = 9.0
_PARAPET_M = 0.6
_WIDE_HALF_M = 8.0  # a house or row this wide (16 m) with a too-high peak: flat (see _heights)
# NRCan Land Cover 2020 classes (geometry.vegetation.sample_landcover).
_LC_FARM = (10, 15, 16)  # temperate grassland (pasture, hay), cropland, barren
_LC_URBAN = 17

# Ontario roof pitch (degrees): quantiles of the implied pitch of Burlington's houses
# (buildings.md §5: p10 14.8, median 22.5, p90 33.7), tails ASSUMED.
_PITCH_QUANTILES = ((0.0, 10.0), (0.1, 14.8), (0.5, 22.5), (0.9, 33.7), (1.0, 42.0))


def ontario_pitch(u: float, category: str = "house") -> float:
    """Roof pitch in degrees for a uniform `u` in [0, 1): the measured Ontario distribution
    for houses and garages; barns steeper (+6 degrees), sheds lower (x 0.8)."""
    qs, ds = zip(*_PITCH_QUANTILES, strict=True)
    p = float(np.interp(u, qs, ds))
    if category == "barn":
        return min(45.0, p + 6.0)
    if category == "shed":
        return p * 0.8
    return p


@dataclass(frozen=True)
class BuildingOptions:
    max_distance_m: float = 300.0  # from the centreline: what is seen from the road
    clearance_m: float = 2.0  # kept clear of the corridor polygon / the drivable edge
    min_area_m2: float = 10.0
    max_area_m2: float = 20_000.0
    max_buildings: int = 6000  # hard cap (nearest first); the byte budget usually binds first
    # Context (`_town_context`): footprints of `min_neighbour_m2` or more within
    # `context_radius_m`; a town from `town_neighbours` (mostly not barn-sized, with some urban
    # land cover), a dense town from `dense_neighbours`. Barns only stand in the country,
    # townhouse rows only in a dense town.
    context_radius_m: float = 100.0
    min_neighbour_m2: float = 60.0
    big_m2: float = 450.0
    town_neighbours: int = 10
    dense_neighbours: int = 20
    urban_town: float = 0.75
    urban_min_neighbours: int = 4
    # Built-up land (at least this urban share) keeps buildings clear of the drivable surface
    # only, like a town, even where they count as country for their looks.
    urban_clearance: float = 0.5
    # Townhouse rows (in towns; untyped, residential or terrace footprints): the minimum
    # rectangle at least this long and this deep, and this big.
    row_min_length_m: float = 18.0
    row_depth_m: tuple[float, float] = (7.0, 16.0)
    row_min_area_m2: float = 180.0
    row_unit_m: float = 6.5  # one unit (front door, garage door) per this much frontage
    budget_bytes: int = 5_000_000  # mesh bytes in the KN5 (44 B / vertex + 6 B / triangle)
    # Towns (most candidates in a town): a bigger budget, and buildings farther than
    # `frontage_m` from the drivable edge get outlines simplified to `far_simplify_m`.
    town_budget_bytes: int = 12_000_000
    far_simplify_m: float = 0.5
    # Every building whose footprint comes within this of the route's drivable edge is placed,
    # whatever the budget: the street front (the user's Burlington check, 2026-10-16).
    frontage_m: float = 60.0
    # In town buildings stand at the sidewalk: kept clear of the drivable surface (the route,
    # `paved` side roads and junctions) by this much instead of the wall corridor.
    town_clearance_m: float = 1.0
    # Height (see module docstring).
    erode_m: float = 1.0
    sample_step_m: float = 1.0
    min_lidar_samples: int = 4
    min_lidar_height_m: float = 2.0
    height_clamp: tuple[float, float] = (3.0, 30.0)
    storey_m: float = 3.0
    default_height_m: float = 6.0  # an untyped house's peak without any height source
    # Roof.
    flat_roof_area_m2: float = 600.0
    flat_lidar_spread_m: float = 0.5  # p90 - median below this -> flat (ASSUMED threshold)
    min_pitched_width_m: float = 3.5  # narrower minimum-rectangle -> flat
    steep_pitch_deg: float = 35.0  # lidar pitch at or above this is believed
    low_pitch_deg: float = 10.0  # ... and at or below this
    max_roof_rise_m: float = 6.0
    max_pitch_deg: float = 45.0
    # Small houses / sheds (area < small_area_m2) whose lidar median exceeds this are under
    # trees: the canopy is ignored and tags / defaults are used (ASSUMED; rural Ontario).
    small_area_m2: float = 400.0
    small_max_lidar_m: float = 12.0
    min_eave_m: float = 2.4
    hipped_share: float = 0.35  # share of near-square houses (aspect < 1.4) given a hip roof
    attached_garage_share: float = 0.55  # wide houses with an attached garage door in front
    doors: bool = True
    # Mesh.
    sink_m: float = 0.5
    collide_within_m: float = 15.0  # from the drivable edge; None-like: < 0 disables
    collide_extra_m: float = 0.5  # collision walls this much taller than the eave
    cell_m: float = 250.0
    lod_out_m: float = 1200.0
    detail_area_m2: float = 40.0  # smaller buildings go to the short-range batches
    detail_lod_out_m: float = 400.0
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
        out = [
            self._xz([(a, 0.0), (e, w)]),
            self._xz([(a, 0.0), (e, -w)]),
            self._xz([(-a, 0.0), (-e, w)]),
            self._xz([(-a, 0.0), (-e, -w)]),
        ]
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


@dataclass(frozen=True)
class Door:
    """A door quad on wall `edge` of the ring: from XZ `a` to `b` (along the wall, in the
    ring's direction), `y0` to `y0 + height`, textured with `DOORS[style]`."""

    style: str
    edge: int
    a: np.ndarray
    b: np.ndarray
    y0: float
    height: float

    @property
    def centre(self) -> np.ndarray:
        return 0.5 * (self.a + self.b)


@dataclass
class Building:
    """One building ready to mesh. `ring` (K, 2) is the open XZ outline, clockwise in the
    (x, z) plane (so `left of each edge` is outside). Heights are metres above `ground_y`
    (median terrain under the footprint); walls start at `base_y`.

    `front_edge` is the ring edge (i -> i + 1) facing the nearest road (None if unknown),
    `doors` the door quads, `wall_u0` the atlas u start (metres) of each ring edge's wall
    (None: continuous along the ring from `u_offset`)."""

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
    category: str = "house"
    town: bool = False  # in a town (subdivision, downtown), not the country: `_town_context`
    front_edge: int | None = None
    doors: tuple[Door, ...] = ()
    wall_u0: tuple[float, ...] | None = None
    mesh: MeshData | None = field(default=None, repr=False, compare=False)

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

    @property
    def front_normal(self) -> np.ndarray | None:
        """Outward unit XZ normal of the front wall."""
        if self.front_edge is None:
            return None
        a, b = self.ring[self.front_edge], self.ring[(self.front_edge + 1) % len(self.ring)]
        d = (b - a) / max(float(np.linalg.norm(b - a)), 1e-12)
        return np.array([-d[1], d[0]])


def choose_height(
    lidar: np.ndarray | None,
    *,
    tag_height: float | None = None,
    levels: float | None = None,
    ms_height: float | None = None,
    max_lidar_m: float | None = None,
    options: BuildingOptions | None = None,
) -> tuple[str, float, float | None, float | None]:
    """(source, height, lidar median, lidar p90) of the unmeasured sources, in order.

    `height` is the roof's MEAN height for "lidar" and "microsoft" (Microsoft's ML height is
    the mean over the footprint: about mid-roof), the eave for "levels", the roof TOP for
    "height" and "default". `lidar` holds DSM - DTM samples inside the footprint (NaN = no
    data). A lidar median above `max_lidar_m` is taken for tree canopy over the roof and
    ignored (`place_buildings` sets it for small houses and sheds)."""
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


def _category(
    fp: Footprint, area: float, town: bool = False, length_m: float = 0.0, depth_m: float = 0.0,
    o: BuildingOptions | None = None, dense: bool = False,
) -> str:  # fmt: skip
    """house / townhouse / garage / shed / barn / commercial (module docstring, step 3).

    In the country a big untyped footprint is a barn here and may become commercial once its
    roof is known to be flat. In a town (`town`) nothing untyped is a barn: in a dense town
    (`dense`) a long, house-deep footprint (`length_m` x `depth_m`, the minimum rectangle) is
    a townhouse row, a big one a house (it becomes commercial when its roof is flat), and only
    OSM-tagged barns stay barns.
    """
    o = o or BuildingOptions()
    kind = fp.kind
    if kind in _FLAT_KINDS:
        return "commercial"
    if kind in _BARN_KINDS:
        return "barn"
    if kind in _GARAGE_KINDS:
        return "garage" if area < 150.0 or town else "barn"
    if kind in _SHED_KINDS:
        return "shed" if area < 30.0 else "garage"
    lo, hi = o.row_depth_m
    row = (
        dense
        and kind in ("yes", "residential", "terrace", "house", "townhouse")
        and area >= o.row_min_area_m2
        and length_m >= o.row_min_length_m
        and lo <= depth_m <= hi
        and length_m >= 1.6 * depth_m
    )
    if kind in ("terrace", "townhouse") or row:
        return "townhouse"
    if kind in _HOUSE_KINDS:
        return "house"
    if area < 25.0:
        return "shed"
    if area < 70.0:
        return "garage"
    if area >= 450.0 and not town:
        return "barn"
    return "house"


def _town_context(
    polys: Sequence[shapely.Polygon], cand: Sequence[int], landcover: LandcoverFn | None,
    o: BuildingOptions,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Per candidate (indices into `polys`, every footprint found): 0 in the country, 1 in a
    town, 2 in a dense town (a subdivision or downtown: townhouse rows only stand there); and
    the urban share of the land cover around it (0 when unknown).

    Counted: the footprints of at least `min_neighbour_m2` (no sheds) within
    `context_radius_m`, other than the building itself (a subdivision has 40-80 houses in
    100 m, a downtown block as many shops; a farm a house or two among barns, and NRCan maps a
    big farmstead's yards as urban, so the land cover alone cannot tell). With at least
    `dense_neighbours` a building is in a dense town; with at least `town_neighbours` in a
    town, unless most of them are big (> `big_m2`: a farm's barns) or the land cover around
    it (its centre and 8 points 75 m out) is known and less than 30 % urban (class 17: a
    hamlet among fields and woods); with at least `urban_min_neighbours` where at least
    `urban_town` of the land cover is urban, also in a town (a district of big buildings);
    otherwise in the country.
    """
    if not len(cand):
        return np.zeros(0, np.int8), np.zeros(0)
    cents = np.array([[c.x, c.y] for c in (p.centroid for p in polys)])
    area = np.array([p.area for p in polys])
    idx = np.asarray(cand, np.int64)

    def neighbours(mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.zeros(len(idx), np.int64)
        got = cKDTree(cents[mask]).query_ball_point(cents[idx], o.context_radius_m,
                                                     return_length=True)  # fmt: skip
        return got - mask[idx].astype(np.int64)

    n = neighbours(area >= o.min_neighbour_m2)
    big = neighbours(area > o.big_m2)
    urban = np.zeros(len(cand))
    known = np.zeros(len(cand), bool)
    if landcover is not None:
        ang = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        ring = np.vstack([[0.0, 0.0], 75.0 * np.stack([np.cos(ang), np.sin(ang)], axis=1)])
        pts = (cents[cand][:, None, :] + ring[None, :, :]).reshape(-1, 2)
        try:
            lc = np.asarray(landcover(pts)).reshape(len(cand), len(ring))
        except (OSError, RuntimeError, ValueError) as exc:  # SourceUnavailable: RuntimeError
            log.warning("land cover unavailable for the building context: %s", exc)
            lc = np.zeros((len(cand), len(ring)))
        k = (lc > 0).sum(axis=1)
        known = k > 0
        urban = (lc == _LC_URBAN).sum(axis=1) / np.maximum(k, 1)
    dense = n >= o.dense_neighbours
    town = dense | ((n >= o.town_neighbours) & (2 * big < n) & (~known | (urban >= 0.3)))
    # A district of big buildings (shops, offices, towers by the lake) has few neighbours
    # within 100 m, mostly big: a town where the land cover is (almost) all urban.
    town |= known & (urban >= o.urban_town) & (n >= o.urban_min_neighbours)
    return town.astype(np.int8) + dense.astype(np.int8), np.where(known, urban, 0.0)


def _measured(fp: Footprint) -> tuple[float | None, float | None, str]:
    """(peak, eave, source label) measured for this footprint (municipal or NRCan lidar)."""
    peak = fp.peak_m if fp.peak_m is not None and 1.8 <= fp.peak_m <= 80.0 else None
    if peak is None:
        return None, None, ""
    eave = fp.eave_m if fp.eave_m is not None and 1.5 <= fp.eave_m <= peak else None
    if eave is not None or not (fp.lidar or any(a.startswith("nrcan:") for a in fp.also)):
        return peak, eave, fp.source
    return peak, None, "nrcan"


def _roof_shape(
    fp: Footprint, cat: str, area: float, half_width: float, aspect: float,
    spread: float | None, o: BuildingOptions, seed: object,
) -> str:  # fmt: skip
    tagged = _ROOF_TAGS.get((fp.roof_shape or "").strip().lower())
    if half_width * 2.0 < o.min_pitched_width_m:
        return ROOF_FLAT
    if tagged is not None:
        return tagged
    if cat == "commercial":
        return ROOF_FLAT
    if spread is not None and spread < o.flat_lidar_spread_m:
        return ROOF_FLAT
    if (area >= o.flat_roof_area_m2 and fp.kind not in _BARN_KINDS and spread is None
            and cat != "townhouse"):  # fmt: skip
        return ROOF_FLAT
    if aspect < 1.4 and cat == "house" and _u01(seed, fp.id, "hip") < o.hipped_share:
        return ROOF_HIPPED
    return ROOF_GABLED


def _styles(
    fp: Footprint, cat: str, area: float, roof: str, seed: object, town: bool = False
) -> tuple[str, str]:
    uw, ur = _u01(seed, fp.id, "wall"), _u01(seed, fp.id, "roofstyle")
    walls, roofs = {
        "commercial": (_COMMERCIAL_WALLS, None),
        "barn": (_BARN_WALLS, _BARN_ROOFS),
        "shed": (_SHED_WALLS, _SHED_ROOFS),
        "garage": (_HOUSE_WALLS, _HOUSE_ROOFS),
        "house": (_HOUSE_WALLS, _HOUSE_ROOFS),
        "townhouse": (_TOWNHOUSE_WALLS, _TOWN_HOUSE_ROOFS),
    }[cat]
    if town and cat in ("house", "garage"):
        walls, roofs = _TOWN_HOUSE_WALLS, _TOWN_HOUSE_ROOFS
    elif town and cat == "commercial":
        walls = _TOWN_COMMERCIAL_WALLS
    wall_style = _pick(walls, uw)
    if roof == ROOF_FLAT:
        roof_style = "flat_membrane" if area >= 60.0 else "metal_grey"
    elif cat == "barn" and wall_style.startswith("steel"):
        roof_style = _pick([r for r in _BARN_ROOFS if r[0].startswith("metal")], ur)
    else:
        roof_style = _pick(roofs or _HOUSE_ROOFS, ur)
    return wall_style, roof_style


def _heights(
    fp: Footprint, cat: str, area: float, perimeter: float, half_width: float, aspect: float,
    lid: tuple[float, float] | None, eroded: float, o: BuildingOptions, seed: object,
    town: bool = False,
) -> tuple[str, float, float, str, str]:  # fmt: skip
    """(height source, eave, rise, roof shape, category) by the module docstring's steps 4-5."""
    peak, eave_m, msrc = _measured(fp)
    spread = (lid[1] - lid[0]) if lid is not None else None
    if peak is not None and spread is not None and area < 400.0:
        # A house with a measured peak: only a clearly flat DSM makes its roof flat (a low
        # gable, eroded, already looks flat to p90 - median < 0.5 m).
        spread = spread if spread < 0.5 * o.flat_lidar_spread_m else None
    shape = _roof_shape(fp, cat, area, half_width, aspect, None if eave_m is not None else spread,
                        o, seed)  # fmt: skip
    flat_top = False  # a measured flat roof whose peak is a parapet or a rooftop penthouse
    if peak is not None and eave_m is not None:  # measured: flat exactly when it is
        if peak - eave_m < 0.6:
            shape = ROOF_FLAT
        elif (
            shape == ROOF_FLAT
            and cat == "commercial"
            and (peak - eave_m < _COMMERCIAL_PITCH_M or eave_m >= _TALL_EAVE_M)
        ):
            flat_top = True  # shops, offices, apartment towers: flat, the peak is equipment
        elif (
            town
            and cat not in ("barn", "commercial")
            and half_width >= _WIDE_HALF_M
            and peak - eave_m > _MAX_RISE_M[cat]
        ):
            # A wide building whose peak stands higher above its eave than any roof of its kind
            # rises (a parking garage's stair tower, a church spire): flat at the eave.
            shape, flat_top = ROOF_FLAT, True
        elif shape == ROOF_FLAT and half_width * 2.0 >= o.min_pitched_width_m:
            shape = ROOF_GABLED
    if cat == "barn" and shape == ROOF_FLAT and area >= 400.0 and fp.kind not in _BARN_KINDS:
        cat = "commercial"
    lo, hi = o.height_clamp
    hi_cat = min(o.max_roof_rise_m, _MAX_RISE_M[cat]) if cat != "barn" else _MAX_RISE_M[cat]
    cap = min(hi_cat, math.tan(math.radians(o.max_pitch_deg)) * half_width)
    if shape == ROOF_FLAT:
        if flat_top and eave_m is not None and peak is not None:
            top = min(peak, eave_m + _PARAPET_M)
            return msrc, float(np.clip(top, 2.0, hi)), 0.0, shape, cat
        if peak is not None:
            return msrc, float(np.clip(peak, 2.0, hi)), 0.0, shape, cat
        if lid is not None:
            return "lidar", float(np.clip(lid[0], lo, hi)), 0.0, shape, cat
    lidar_pitch = None
    ridge_lidar = None
    if lid is not None and shape != ROOF_FLAT:
        med, p90 = lid
        visible = 2.5 * (p90 - med)
        span = max(0.3, (half_width - eroded - 0.5 * o.sample_step_m) / half_width)
        rise_l = float(np.clip(visible / span, 0.3, max(0.8, cap)))
        lidar_pitch = math.degrees(math.atan(rise_l / half_width))
        ridge_lidar = (rise_l, med, span)
    if shape == ROOF_FLAT:
        rise = 0.0
    elif peak is not None and eave_m is not None:
        rise = min(peak - eave_m, math.tan(math.radians(o.max_pitch_deg)) * half_width)
    else:
        pitch = None
        # Lidar pitch from a 1 m DSM is believed only when clearly steep, or clearly low
        # without a measured peak (it underestimates the rise: buildings.md §4).
        if lidar_pitch is not None and (
            lidar_pitch >= o.steep_pitch_deg or (peak is None and lidar_pitch <= o.low_pitch_deg)
        ):
            pitch = min(lidar_pitch, o.max_pitch_deg)
        if pitch is None:
            pitch = ontario_pitch(_u01(seed, fp.id, "pitch"), cat)
        # Wings, not the bounding rectangle, set the rise of an L- or T-shaped house.
        wing = min(half_width, 2.0 * area / max(perimeter, 1e-9))
        rise = math.tan(math.radians(pitch)) * wing
        if peak is not None and pitch != lidar_pitch:
            # Burlington (n = 675): eave = peak - this rise, kept within 0.30-0.45 of the peak,
            # is off by |0.51| m (p50) / 1.61 m (p90); the bare pitch rule by 0.58 / 1.88 m.
            rise = float(np.clip(rise, 0.30 * peak, 0.45 * peak))
        rise = float(np.clip(rise, 0.8, max(0.8, cap)))
    # Eave from the height source.
    if peak is not None:
        src, eave = msrc, peak - rise
    elif ridge_lidar is not None:
        _, med, span = ridge_lidar
        ridge = min(med + 0.5 * rise * span, hi)  # median = eave + the seen half of the rise
        src, eave = "lidar", ridge - rise
    else:
        src, h, _, _ = choose_height(None, tag_height=fp.height_m, levels=fp.levels,
                                     ms_height=fp.ms_height_m, options=o)  # fmt: skip
        if src == "default":
            h = o.default_height_m if cat == "house" else _DEFAULT_PEAK_M[cat]
            if cat == "barn" and area >= 800.0:
                h = 9.7
        if shape == ROOF_FLAT:
            eave = h if src != "microsoft" else h
        elif src == "levels":
            eave = h
        elif src == "microsoft":
            eave = h - 0.5 * rise
        else:
            eave = h - rise
    min_eave = min(o.min_eave_m, _MIN_EAVE_M[cat]) if cat in ("shed", "garage") else max(
        o.min_eave_m, _MIN_EAVE_M[cat])  # fmt: skip
    if eave < min_eave:
        rise = max(0.0, rise - (min_eave - eave))
        eave = min_eave
        if rise < 0.5 and shape != ROOF_FLAT:
            shape, rise = ROOF_FLAT, 0.0
    eave = float(min(eave, hi))
    return src, eave, float(rise), shape, cat


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


# --------------------------------------------------------------------------------------------
# Front walls, doors, window layout
# --------------------------------------------------------------------------------------------


def _front_edge(ring: np.ndarray, towards: np.ndarray) -> int:
    """The ring edge whose outward normal best faces the unit XZ direction `towards`
    (edges weighted by length up to 10 m, so a short bay does not win)."""
    d = np.roll(ring, -1, axis=0) - ring
    length = np.linalg.norm(d, axis=1)
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1) / np.maximum(length, 1e-12)[:, None]
    score = (nrm @ towards) * np.minimum(length, 10.0) + 1e-3 * length
    score[length < 1.0] = -np.inf
    return int(np.argmax(score))


def _door_spans(
    b: Building, length: float, o: BuildingOptions, seed: object
) -> list[tuple[str, float, float, float]]:
    """(door style, centre along the front wall in metres, width, height) for a building."""
    u = _u01(seed, b.id, "doors")
    out: list[tuple[str, float, float, float]] = []
    cat = b.category
    if cat == "house":
        garage = None
        if length >= 10.0 and b.area_m2 >= 110.0 and u < o.attached_garage_share:
            double = length >= 13.0 and _u01(seed, b.id, "double") < 0.6
            gs = ("garage2_" if double else "garage1_") + (
                "white" if _u01(seed, b.id, "gcol") < 0.75 else "almond")  # fmt: skip
            d = DOORS[gs]
            left = _u01(seed, b.id, "gside") < 0.5
            c = 0.8 + 0.5 * d.width_m if left else length - 0.8 - 0.5 * d.width_m
            garage = (gs, c, d.width_m, d.height_m)
            out.append(garage)
        door = _pick((("door_white", 45), ("door_dark", 35), ("door_red", 20)),
                     _u01(seed, b.id, "dcol"))  # fmt: skip
        w = DOORS[door].width_m
        if garage is not None:
            g0, g1 = garage[1] - 0.5 * garage[2], garage[1] + 0.5 * garage[2]
            free = (g1 + 1.0, length - 0.6) if g0 < length - g1 else (0.6, g0 - 1.0)
        else:
            free = (0.6, length - 0.6)
        if free[1] - free[0] >= w:
            mid = 0.5 * (free[0] + free[1])
            jitter = (_u01(seed, b.id, "dpos") - 0.5) * 0.3 * (free[1] - free[0] - w)
            out.append((door, mid + jitter, w, DOORS[door].height_m))
    elif cat == "townhouse":
        # One unit per `row_unit_m` of frontage: a single garage door and a front door each,
        # mirrored in pairs as builders lay them out.
        n = max(2, int(round(length / o.row_unit_m)))
        unit = length / n
        gd = DOORS["garage1_white" if _u01(seed, b.id, "gcol") < 0.75 else "garage1_almond"]
        door = _pick((("door_white", 40), ("door_dark", 45), ("door_red", 15)),
                     _u01(seed, b.id, "dcol"))  # fmt: skip
        dw = DOORS[door].width_m
        garages = unit >= gd.width_m + dw + 1.4 and _u01(seed, b.id, "rowgarage") < 0.8
        for k in range(n):
            x0 = k * unit
            flip = k % 2 == 1
            if garages:
                g = x0 + (unit - 0.5 - 0.5 * gd.width_m if flip else 0.5 + 0.5 * gd.width_m)
                d = x0 + (0.4 + 0.5 * dw + 0.3 if flip else unit - 0.4 - 0.5 * dw - 0.3)
                out.append((gd.name, g, gd.width_m, gd.height_m))
            else:
                d = x0 + 0.5 * unit
            if unit >= dw + 0.8:
                out.append((door, d, dw, DOORS[door].height_m))
    elif cat == "garage":
        for name in ("garage2_white", "garage1_white"):
            d = DOORS[name]
            if length >= d.width_m + 0.8:
                if _u01(seed, b.id, "gcol") >= 0.75:
                    name = name.replace("white", "almond")
                out.append((name, 0.5 * length, d.width_m, d.height_m))
                break
    elif cat == "barn":
        name = "steel_door" if b.wall_style.startswith("steel") else "barn_door"
        d = DOORS[name]
        k = float(np.clip((b.eave_h - 0.4) / d.height_m, 0.8, 1.4))
        if length >= d.width_m * k + 1.0:
            out.append((name, 0.5 * length, d.width_m * k, d.height_m * k))
    elif cat == "shed":
        name = "door_white" if _u01(seed, b.id, "dcol") < 0.5 else "door_dark"
        d = DOORS[name]
        if length >= d.width_m + 0.4:
            out.append((name, 0.5 * length, d.width_m, min(d.height_m, 2.1)))
    return out


def _place_doors(
    buildings: Sequence[Building], ground: GroundFn, o: BuildingOptions, seed: object
) -> None:
    """Fill `doors` (and keep `front_edge`) for every building with a front wall."""
    plans = []
    for b in buildings:
        if b.front_edge is None:
            continue
        i = b.front_edge
        a, c = b.ring[i], b.ring[(i + 1) % len(b.ring)]
        length = float(np.linalg.norm(c - a))
        if length < 1.0:
            continue
        d = (c - a) / length
        for style_name, mid, w, h in _door_spans(b, length, o, seed):
            plans.append((b, i, a + d * (mid - 0.5 * w), a + d * (mid + 0.5 * w), style_name, h))
    if not plans:
        return
    centres = np.array([0.5 * (p[2] + p[3]) for p in plans])
    nrm = np.array([[-(p[3] - p[2])[1], (p[3] - p[2])[0]] for p in plans])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)[:, None]
    g = np.asarray(ground(centres + 0.8 * nrm), np.float64)  # just outside the wall
    doors: dict[str, list[Door]] = {}
    for (b, i, pa, pb, style_name, h), gy in zip(plans, g, strict=True):
        y0 = float(gy) if np.isfinite(gy) else b.ground_y
        y0 = max(y0, b.base_y + 0.2)
        if style_name.startswith("door_"):
            y0 += 0.15  # a step up to the front door
        h = min(h, b.eave_y - 0.15 - y0)
        if h < 1.8:
            continue
        doors.setdefault(b.id, []).append(Door(style_name, i, pa, pb, y0, h))
    for b in buildings:
        b.doors = tuple(doors.get(b.id, ()))


def _wall_starts(b: Building, seed: object) -> tuple[float, ...] | None:
    """Per-edge atlas u start (metres): whole windows centred on house walls, blank walls on
    garages and sheds in a windowed style; None (continuous) for the other bands."""
    if b.wall_style not in WINDOWED:
        return None
    windows = b.category in ("house", "townhouse", "commercial")
    ring = b.ring
    seg = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)
    return tuple(
        round(wall_u_start(float(L), _u01(seed, b.id, "win", i), windows), 4)
        for i, L in enumerate(seg)
    )


def _match_outbuildings(buildings: Sequence[Building], seed: object) -> None:
    """Garages within 35 m of a house wear the house's colours; sheds usually do."""
    houses = [b for b in buildings if b.category == "house"]
    if not houses:
        return
    tree = cKDTree(np.array([h.centroid for h in houses]))
    for b in buildings:
        if b.category not in ("garage", "shed"):
            continue
        d, j = tree.query(b.centroid)
        if d <= 35.0 and (b.category == "garage" or _u01(seed, b.id, "match") < 0.75):
            h = houses[int(j)]
            b.wall_style = h.wall_style
            if b.roof.shape != ROOF_FLAT:
                b.roof_style = h.roof_style


def mesh_bytes(mesh: MeshData) -> int:
    """What a mesh costs in the KN5: 44 B per vertex + 6 B per triangle (uint16 indices)."""
    return 44 * mesh.vertex_count + 6 * mesh.triangle_count


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
    roads: Sequence[np.ndarray] | None = None,
    landcover: LandcoverFn | None = None,
    paved: shapely.Geometry | None = None,
    report: dict[str, int] | None = None,
) -> list[Building]:
    """Turn footprints into `Building`s (module docstring, steps 1-7), sorted by id.

    `corridor` is the road corridor polygon in XZ (e.g. `walls.corridor_polygon(model,
    model.outer + verge)`); footprints touching it (grown by `clearance_m`) are dropped.
    `canopy(xz)` returns lidar DSM - DTM heights (e.g. `vegetation.CanopySampler`); None or
    NaN means no lidar, and the measured / tag / default heights are used. `roads` are extra
    XZ polylines (side roads) a front wall may face; the route always counts. `seed` should be
    the track id. Each building's mesh is made here (for the byte budget) and kept in `mesh`.
    `landcover(xz)` (NRCan 2020 classes, e.g. `vegetation.LandcoverSampler`) tells farmland
    from town where the building density alone does not (`_town_context`). `paved` is the
    drivable surface beyond the route (side roads, junctions: `RoadNetwork.paved`); in town
    buildings only keep clear of it and of the route by `town_clearance_m`, not of the wall
    corridor. `report`, if given, receives the counts of the steps (found, dropped and why,
    candidates, town, street front, placed, budget).
    """
    o = options or BuildingOptions()
    rep = report if report is not None else {}
    rep.update(dict.fromkeys(("found", "size", "far", "road", "corridor", "candidates", "town",
                              "street_front", "placed", "budget"), 0))  # fmt: skip
    rep["found"] = len(footprints)
    if not footprints:
        return []
    line, tree, outer = _road_frame(model)
    keep_out = corridor.buffer(o.clearance_m) if corridor is not None else None
    paved_out = (
        paved.buffer(o.town_clearance_m) if paved is not None and not paved.is_empty else None
    )
    pre: list[tuple[float, Footprint, shapely.Polygon, int]] = []
    every: list[shapely.Polygon] = []  # every footprint found, for the building density
    for fp in footprints:
        poly = _to_polygon(fp, projection)
        if poly is not None:
            every.append(poly)
        if poly is None or not (o.min_area_m2 <= poly.area <= o.max_area_m2):
            rep["size"] += 1
            continue
        d_centre = float(poly.distance(line))
        if d_centre > o.max_distance_m:
            rep["far"] += 1
            continue
        # Distance from the drivable edge: from the footprint to the centreline, minus the
        # road's half width + shoulder at the station nearest that closest approach.
        p_near = np.asarray(shapely.shortest_line(poly, line).coords)[1]
        _, j = tree.query(p_near)
        d_edge = d_centre - float(outer[j])
        if d_edge < min(o.clearance_m, o.town_clearance_m):
            rep["road"] += 1
            continue
        pre.append((d_edge, fp, poly, len(every) - 1))
    # Context first: in town buildings stand at the sidewalk, inside the wall corridor.
    ctx_pre, urban_pre = _town_context(every, [c[3] for c in pre], landcover, o)
    cand: list[tuple[float, Footprint, shapely.Polygon, int]] = []
    for (d_edge, fp, poly, _), ctx, urban in zip(pre, ctx_pre, urban_pre, strict=True):
        if ctx >= 1 or urban >= o.urban_clearance:
            if d_edge < o.town_clearance_m or (
                paved_out is not None and poly.intersects(paved_out)
            ):
                rep["road"] += 1
                continue
        elif d_edge < o.clearance_m:
            rep["road"] += 1
            continue
        elif keep_out is not None and poly.intersects(keep_out):
            rep["corridor"] += 1
            continue
        cand.append((d_edge, fp, poly, int(ctx)))
    cand.sort(key=lambda t: (t[0], t[1].id))
    rep["candidates"] = len(cand)
    rep["town"] = sum(1 for c in cand if c[3] >= 1)
    if len(cand) > o.max_buildings:
        log.info("building cap: keeping the %d nearest of %d", o.max_buildings, len(cand))
        cand = cand[: o.max_buildings]
    if not cand:
        return []

    # Ground under every footprint, one batched call.
    g_pts, g_owner = [], []
    for k, (_, _, poly, _) in enumerate(cand):
        pts = np.concatenate([np.asarray(poly.exterior.coords)[:-1], _lattice(poly, 3.0)])
        g_pts.append(pts)
        g_owner.append(np.full(len(pts), k))
    g_all = np.asarray(ground(np.concatenate(g_pts)), np.float64)
    g_owner_all = np.concatenate(g_owner)

    # Lidar samples inside every eroded footprint, one batched call (not needed when the
    # footprint has measured heights).
    lidar: list[np.ndarray | None] = [None] * len(cand)
    eroded = np.zeros(len(cand))  # how far inside the outline the lidar was sampled
    if canopy is not None:
        c_pts, c_owner = [], []
        for k, (_, fp, poly, _) in enumerate(cand):
            if _measured(fp)[1] is not None:  # eave and peak measured: nothing to learn
                continue
            inner = poly.buffer(-o.erode_m)
            pts = _lattice(inner, o.sample_step_m)
            eroded[k] = o.erode_m
            if len(pts) < o.min_lidar_samples:
                pts = _lattice(poly, o.sample_step_m / 2.0)
                eroded[k] = 0.0
            c_pts.append(pts)
            c_owner.append(np.full(len(pts), k))
        pts_all = np.concatenate(c_pts) if c_pts else np.zeros((0, 2))
        chm = None
        if len(pts_all):
            try:
                chm = np.asarray(canopy(pts_all), np.float64).reshape(-1)
            except (OSError, RuntimeError, ValueError) as exc:  # SourceUnavailable: RuntimeError
                log.warning("lidar building heights unavailable, using tags/defaults: %s", exc)
        if chm is not None:
            owner = np.concatenate(c_owner)
            order = np.argsort(owner, kind="stable")
            bounds = np.searchsorted(owner[order], np.arange(len(cand) + 1))
            for k in range(len(cand)):
                if bounds[k + 1] > bounds[k]:
                    lidar[k] = chm[order[bounds[k] : bounds[k + 1]]]

    road_lines = [line] + [shapely.LineString(r) for r in (roads or []) if len(r) >= 2]
    road_tree = shapely.STRtree(road_lines)

    out: list[Building] = []
    for k, (d_edge, fp, poly, ctx) in enumerate(cand):
        town, dense = ctx >= 1, ctx == 2
        g = g_all[g_owner_all == k]
        g = g[np.isfinite(g)]
        if g.size == 0:
            continue
        ground_y, base_y = float(np.median(g)), float(g.min()) - o.sink_m
        centre, L, S, hl, hw = _min_rect(poly)
        cat = _category(fp, poly.area, town, 2.0 * hl, 2.0 * hw, o, dense)
        small = poly.area < o.small_area_m2 and fp.kind in _SMALL_CAPPED_KINDS
        src_l, _, med, p90 = choose_height(
            lidar[k], max_lidar_m=o.small_max_lidar_m if small else None, options=o
        )
        lid = (med, p90) if src_l == "lidar" and med is not None and p90 is not None else None
        src, eave, rise, shape, cat = _heights(
            fp, cat, poly.area, poly.length, hw, hl / max(hw, 1e-9), lid, eroded[k], o, seed, town
        )
        if town and cat == "house" and shape == ROOF_FLAT and poly.area >= 400.0:
            cat = "commercial"  # a big flat-roofed building in town: shops, offices, a school
        wall_style, roof_style = _styles(fp, cat, poly.area, shape, seed, town)
        roof = Roof(shape, centre, L, S, hl, hw, rise if shape != ROOF_FLAT else 0.0)
        if town and d_edge > o.frontage_m and o.far_simplify_m > 0:
            poly = _simplified(poly, o.far_simplify_m)
        ring = np.asarray(poly.exterior.coords)[:-1].copy()
        cpt = poly.centroid
        nearest = road_lines[int(road_tree.nearest(cpt))]
        q = np.asarray(shapely.shortest_line(cpt, nearest).coords)
        towards = q[1] - q[0]
        nt = float(np.linalg.norm(towards))
        out.append(
            Building(
                id=fp.id,
                source=fp.source,
                kind=fp.kind,
                ring=ring,
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
                category=cat,
                town=town,
                front_edge=_front_edge(ring, towards / nt) if nt > 1e-6 else None,
            )
        )
    _match_outbuildings(out, seed)
    for b in out:
        b.wall_u0 = _wall_starts(b, seed)
    if o.doors:
        _place_doors(out, ground, o, seed)

    # Byte budget: the street front always (within `frontage_m` of the drivable edge), then
    # the rest nearest the road relative to size first, within the town or country budget.
    def priority(b: Building) -> tuple[bool, float, str]:
        size = math.sqrt(np.clip(b.area_m2, 15.0, 1500.0) / 150.0)
        return b.road_distance_m > o.frontage_m, b.road_distance_m / size, b.id

    budget = o.town_budget_bytes if 2 * rep["town"] > len(cand) else o.budget_bytes
    kept: list[Building] = []
    used = 0
    for b in sorted(out, key=priority):
        m = extrude_building(b, storey_m=o.storey_m, sink_m=o.sink_m, atlas_size=o.atlas_size)
        cost = mesh_bytes(m)
        front = b.road_distance_m <= o.frontage_m
        if not front and used + cost > budget:
            log.info("building budget: %d of %d buildings (%.1f MB)", len(kept), len(out),
                     used / 1e6)  # fmt: skip
            break
        b.mesh = m
        used += cost
        kept.append(b)
        rep["street_front"] += front
    rep["placed"] = len(kept)
    rep["budget"] = len(out) - len(kept)
    rep["mesh_bytes"] = used
    rep["budget_bytes"] = budget
    return sorted(kept, key=lambda b: b.id)


def _simplified(poly: shapely.Polygon, tol: float) -> shapely.Polygon:
    """`poly` with its outline simplified to `tol` (fewer wall quads far from the road),
    clockwise; the original when simplifying breaks it."""
    s = poly.simplify(tol, preserve_topology=True)
    if (
        not isinstance(s, shapely.Polygon)
        or s.is_empty
        or not s.is_valid
        or s.area < 0.8 * poly.area
    ):
        return poly
    return shapely.orient_polygons(s, exterior_cw=True)


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
# A gable end up to this wide on a windowed (or plain steel / stained board) band is mapped on
# the band's window-free last 12 m at full storey height instead: brick and siding keep their
# proportions (stretched at most rise / 3 m), where the 0.75 m plain strip stretched over a 3 m
# gable turned brick into wood grain (the "barn" look of the user's Oakville check).
_GABLE_BLANK_M = ATLAS_U_M - WINDOW_ZONE_M
_GABLE_FULL_STYLES = WINDOWED | {"steel_white", "shed_wood"}


def _wall_rows(b: Building, storey_m: float, sink_m: float = 0.5) -> np.ndarray:
    """Horizontal wall rows (R, 4): (y_lo, y_hi, band fraction at y_lo, at y_hi), bottom up.

    Band fraction 0 = the band's top (v_top), 1 = its bottom. Rows: a whole number of storeys
    between ground and eave (the storey height stretches by at most 1.5x, so the eave never
    cuts a window); the first storey reaches down to `base_y`, unless the ground under the
    building drops more than `sink_m` + 0.3 m below its median, where a plain foundation row
    (bottom 10 % of the band) is cut instead; then one row from the eave to the ridge (gable
    ends) on the plain strip under the sills, stretched (few vertices; `_walls` maps narrow
    gables on the window-free end of the band instead).
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


def _ring_edges(b: Building, ring: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each vertex of the noded `ring`: the original ring edge it lies on and its distance
    from that edge's start."""
    edge = np.empty(len(ring), np.int64)
    dist = np.empty(len(ring))
    j = -1
    n0 = len(b.ring)
    for k, p in enumerate(ring):
        nxt = b.ring[(j + 1) % n0]
        if j < n0 - 1 and abs(p[0] - nxt[0]) + abs(p[1] - nxt[1]) < 1e-9:
            j += 1
        edge[k] = max(j, 0)
        dist[k] = float(np.linalg.norm(p - b.ring[edge[k]]))
    return edge, dist


def _ring_u(b: Building, ring: np.ndarray) -> np.ndarray:
    """Atlas u (metres) at each vertex of the noded `ring`, for the wall that starts there:
    continuous along the ring from `u_offset`, or, with `wall_u0`, each original edge's own
    start plus the distance along that edge (so its windows stay centred)."""
    seg = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)
    if b.wall_u0 is None or len(b.wall_u0) != len(b.ring):
        return np.concatenate([[0.0], np.cumsum(seg)[:-1]]) + b.u_offset
    edge, dist = _ring_edges(b, ring)
    return np.asarray(b.wall_u0, np.float64)[edge] + dist


def _walls(b: Building, ring: np.ndarray, rows: np.ndarray, size: int) -> MeshData:
    v_top, v_bot = band_v(b.wall_style, size)
    top = b.eave_y + b.roof.height(ring)
    n = len(ring)
    seg = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)
    s0 = _ring_u(b, ring)
    # Gable ends on the band's window-free end (`_GABLE_BLANK_M`), at full storey height.
    edge, dist = _ring_edges(b, ring)
    e_len = np.linalg.norm(np.roll(b.ring, -1, axis=0) - b.ring, axis=1)[edge]
    full = b.wall_style in _GABLE_FULL_STYLES
    g0 = WINDOW_ZONE_M + 0.5 * (_GABLE_BLANK_M - e_len) + dist  # u (metres) of each vertex
    g_hi = max(0.0, 1.0 - (b.ridge_y - b.eave_y) / STOREY_M)
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
            if full and lo >= b.eave_y - 1e-6 and e_len[i] <= _GABLE_BLANK_M:
                u = (g0[i] + uv_[:, 0]) / ATLAS_U_M
                f_lo, f_hi = 1.0, g_hi
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
        np.concatenate(pos),
        np.concatenate(nrm),
        np.concatenate(uvs),
        np.tile([1.0, 0.0, 0.0], (count, 1)),
        np.asarray(idx, np.int64).reshape(-1),
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
        flat_p[first],
        np.tile(normal, (n, 1)),
        flat_uv[first],
        np.tile([1.0, 0.0, 0.0], (n, 1)),
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
            frac = np.clip(
                0.5 + (rel @ b.roof.short_axis) / (2 * b.roof.half_width + 1e-9), 0.0, 1.0
            )
        v = v_top + frac * (v_bot - v_top)
        normal = np.array([-grad[0], 1.0, -grad[1]])
        normal /= np.linalg.norm(normal)
        out.append(_planar_piece(tris, y, normal, u, v, up=True))
    return out


def _door_quads(b: Building, size: int, offset_m: float = 0.03) -> MeshData:
    """The doors: one quad each, `offset_m` in front of its wall, facing out."""
    if not b.doors:
        return _empty()
    pos, nrm, uvs, idx = [], [], [], []
    for d in b.doors:
        along = d.b - d.a
        length = float(np.linalg.norm(along))
        if length < 1e-6:
            continue
        t = along / length
        n2 = np.array([-t[1], t[0]])
        a = d.a + offset_m * n2
        c = d.b + offset_m * n2
        y0, y1 = d.y0, d.y0 + d.height
        pos.append([[a[0], y0, a[1]], [c[0], y0, c[1]], [c[0], y1, c[1]], [a[0], y1, a[1]]])
        nrm.append(np.tile([n2[0], 0.0, n2[1]], (4, 1)))
        u0, u1, v_top, v_bot = door_uv(d.style, size)
        uvs.append([[u0, v_bot], [u1, v_bot], [u1, v_top], [u0, v_top]])
        base = 4 * len(idx) // 6
        idx.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    if not pos:
        return _empty()
    count = 4 * len(pos)
    return MeshData(
        np.asarray(pos, np.float64).reshape(-1, 3),
        np.concatenate(nrm),
        np.asarray(uvs, np.float64).reshape(-1, 2),
        np.tile([1.0, 0.0, 0.0], (count, 1)),
        np.asarray(idx, np.int64),
    )


def extrude_building(
    b: Building,
    *,
    storey_m: float = 3.0,
    sink_m: float = 0.5,
    floor: bool = False,
    atlas_size: int = ATLAS_SIZE,
    doors: bool = True,
) -> MeshData:
    """The building's visual mesh: walls (cut at storey lines) + roof + door quads (+ a floor
    at `base_y` with `floor=True`, which makes the solid closed). Flat outward normals,
    tangents along u."""
    margin = 1.0
    rows = _wall_rows(b, storey_m, sink_m)
    ring = _noded_ring(b, np.unique(rows[:, :2]), margin)
    parts = [_walls(b, ring, rows, atlas_size), *_roof(b, ring, atlas_size, margin)]
    if doors:
        parts.append(_door_quads(b, atlas_size))
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
    mesh_bytes: int = 0
    doors: int = 0
    by_height_source: dict[str, int] = field(default_factory=dict)
    by_roof: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)


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
    """Batch the buildings per `cell_m` cell: `buildings_<nnn>` (visual, `lod_out_m`),
    `buildings_s<nnn>` for the small ones (< `detail_area_m2`, `detail_lod_out_m`) and, for
    buildings with `collide`, hidden `1WALL_bld_<nnn>` walls. Cells are numbered in sorted
    order; `_<p>` suffixes appear when a batch needs several meshes (> 65,535 vertices). A
    building's `mesh` (from `place_buildings`) is used when present."""
    o = options or BuildingOptions()
    if not buildings:
        return []
    cells: dict[tuple[int, int], list[Building]] = {}
    for b in buildings:
        key = tuple(int(v) for v in np.floor(b.centroid / o.cell_m))
        cells.setdefault(key, []).append(b)
    out: list[BuildingMesh] = []
    hidden: MeshFlags = {"is_renderable": False, "cast_shadows": False}
    for ci, key in enumerate(sorted(cells)):
        members = sorted(cells[key], key=lambda b: b.id)
        groups = (
            ("", [b for b in members if b.area_m2 >= o.detail_area_m2], o.lod_out_m),
            ("s", [b for b in members if b.area_m2 < o.detail_area_m2], o.detail_lod_out_m),
        )
        for tag, group, lod in groups:
            parts = [
                b.mesh if b.mesh is not None else extrude_building(
                    b, storey_m=o.storey_m, sink_m=o.sink_m, atlas_size=o.atlas_size)
                for b in group
            ]  # fmt: skip
            parts = [p for p in parts if p.vertex_count]
            if not parts:
                continue
            pieces = split(merge(parts))
            flags: MeshFlags = {"lod_out": lod, "cast_shadows": o.cast_shadows}
            for p, piece in enumerate(pieces):
                name = f"buildings_{tag}{ci:03d}" + (f"_{p}" if len(pieces) > 1 else "")
                out.append(BuildingMesh(name, piece, len(group), False, dict(flags)))
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
        if not m.collision:
            stats.mesh_bytes += mesh_bytes(m.data)
    stats.buildings = len(buildings)
    stats.colliding = sum(b.collide for b in buildings)
    stats.doors = sum(len(b.doors) for b in buildings)

    def count(d: dict[str, int], k: str) -> None:
        d[k] = d.get(k, 0) + 1

    for b in buildings:
        count(stats.by_height_source, b.height_source)
        count(stats.by_roof, b.roof.shape)
        count(stats.by_category, b.category)
        count(stats.by_source, b.source)
    stats.texture_bytes = len(scene.textures[tex_name].data)
    return stats
