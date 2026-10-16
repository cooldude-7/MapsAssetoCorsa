"""Build a real Assetto Corsa track from a route: `drift-maps build` (milestone M1).

Pipeline (PLAN §5.1-§5.5, docs/research/geo.md §1-§3, ac_format.md §2, §3, §6, §9):

1. **Route** (`driftmaps.geo`): pins -> OSM corridor -> `route.json`. The build reads only the
   `Route`; `--reverse` drives the same edges the other way.
2. **Centreline** (`geometry.road`): the OSM node polyline is projected on a transverse
   Mercator plane with its origin at the start pin A (AC X = east, Z = south), sharp corners
   are filleted, the line is extended straight by `run_up_m` before A and `run_off_m` after B,
   smoothed within `max_deviation_m` of the OSM line and resampled every 1 m.
3. **Elevation** (`driftmaps.elevation`): five DEM samples across the road at every station,
   weight 0 on bridges and tunnels (+10 m), then `road_profile` (smoothing plus hard grade and
   vertical-curvature limits). The source covering most of the road is the reference; points
   that fell through to another source are shifted by the median offset between the two, so a
   coverage gap cannot leave a step. That offset is measured on gentle terrain in the near ring
   beside the road (not on the road, where a coarse DSM is biased by cuts and trees) and
   dropped if it is over 2 m. Heights are relative to the road at A (A sits at Y = 0).
4. **Road, terrain, walls** (`geometry.road`, `.terrain`, `.walls`).
5. **Dummies and AI**: spawns 1 m above the road (M0 verified), one `AC_START_n` per pit box
   in side-by-side rows behind A, `AC_PIT_n` in pairs behind them, `AC_HOTLAP_START_0` further
   back (the run-up grows to fit them), A->B gates with `_L` on the driver's left, at most one
   `AC_TIME_0` split. Hidden walls reach `height_m` above the higher of the ground and the
   nearest road, so they hold on fills and bridges too. The AI line follows the centre of
   the driving lane (right in Canada/US/France, left in Japan/UK) from the start of the run-up
   to `ai_past_b_m` past B; the run-off road ends in the corridor's end wall.
6. **Scenery** (M2, docs/research/scenery.md §7), every part optional (`trees`, `buildings`,
   `guardrails`) and failing soft (a download error becomes a warning, never a failed build):
   - **Bridges** (`geometry.bridges`, always): deck slab, concrete parapets with hidden
     collision, abutments and piers under every OSM bridge span; an at-grade culvert (solid,
     or deck under 2 m above the ground) is just road, without parapets. The terrain under a
     deck is the real lidar ground, kept `TerrainOptions.bridge_clearance_m` below the road
     and, as a safety net, clamped under `bridges.deck_ceiling` (no terrain through a deck).
   - **Guardrails** (`geo.barriers`, `geometry.barriers`, `geometry.warrants`): the user's
     `guardrails.json` (`guardrails_file`), then OSM
     `barrier=guard_rail|jersey_barrier|wall|retaining_wall` beside the road, then (on by
     default, `auto_guardrails`) Type M steel-beam rails wherever Ontario road-design rules
     require one (docs/research/guardrails.md: high/steep fills and drop-offs measured on the
     lidar DTM, deep water and bridge ends inside the clear zone; cut at side roads and
     driveways). Water and driveways come from one more Overpass query. `data/
     guardrails_auto.json` says why each automatic span exists; guardrails.json can switch
     them off (`"auto": false`) or erase them (`"kind": "none"` spans). Rails on bridge
     approaches without a rule are invented, so only with `bridge_approach_rails`.
   - **Buildings** (`geo.buildings`, `geometry.buildings`): OSM + Microsoft footprints, lidar
     heights, outside the wall corridor.
   - **Street furniture** (`geometry.furniture`, `furniture`): signs, hydro poles and wires,
     street lights, signals, hydrants and mailboxes from Mapillary detections (with
     `mapillary_token`), OSM nodes and power lines, Ontario rules where Mapillary has no
     photos, and the route's junction control (`street_items.approach_stops`: stop signs and
     stop bars where the route stops), plus every `StreetItem` other stages added to
     `street_items`; de-duplicated and merged per 250 m cell on one procedural atlas. Nothing
     stands on the drivable surface (`geometry.clearance`: route, side roads, junctions,
     roundabouts, driveways); signs in fields and stray Mapillary poles are dropped.
   - **Trees** (`geometry.vegetation`): lidar tree tops (DSM - DTM) with NRCan land cover
     kinds, trunks kept outside the hidden walls (a car can never drive through a tree) and off
     building footprints and poles; `scenery_density` scales the tree budget.
   All of it goes into the one KN5 (tree and building atlases as DDS); the trees' and
   buildings' data sources are credited in LICENSE-DATA.txt.
7. **Files** (`ac.track_files`): surfaces (ROAD, GRASS, GRAVEL, DIRT), map, cameras, UI, credits
   (`LICENSE-DATA.txt`, `route_osm.geojson` with the OSM road, building and barrier ways used,
   `route.json`, and `guardrails.json` if one was used: its bytes are read before the old
   track folder is cleared, so a file kept inside the track folder survives a rebuild).
   A->B tracks are tagged "touge" and get `data/acmg_touge.json` (gates, grid and chase start
   for the Touge Battle CSP New Mode in `csp/new-modes/acmg-touge`, see `driftmaps.csp_modes`).

**Loops** (a closed `Route`, PLAN §5.5 circuits): the centreline is smoothed as a closed
curve (`geometry.road.closed_centreline`, no seam at the join), the height profile is fitted
with the lap before and after it so it meets itself, and the `RoadModel` is `closed`: road,
shoulders, paint, terrain and walls run on across the join (no run-up, no run-off, no end
walls; the corridor is a ring with walls outside and inside). Timing is a lap: AC_TIME_0 on
the start/finish line at A and AC_TIME_1/2 at a third and two thirds of the lap; the grid and
the pits are behind the line in pairs like on an A->B track, the hotlap start
`HOTLAP_BEFORE_M` before it, and the AI line is closed. ui_track.json gets the "circuit" tag
(multi-lap races in Content Manager) and `run` = "clockwise" or "anticlockwise".

`build_track(route, out_dir, options)` writes `<out_dir>/<track_id>/` (replacing an existing
folder only if drift-maps made it, or with `options.force`) and returns a
`BuildResult` with the statistics the CLI prints. Same route + same DEM cache -> identical
bytes (no clocks or random numbers in the output).
"""

from __future__ import annotations

import dataclasses
import functools
import json
import time
import urllib.error
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import shapely
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from driftmaps.ac.ai_line import AiLine, build_ai_line
from driftmaps.ac.building_textures import building_atlas_texture
from driftmaps.ac.coords import WORLD_UP
from driftmaps.ac.furniture_textures import furniture_atlas_texture
from driftmaps.ac.kn5 import write_kn5
from driftmaps.ac.model import Dummy, Kn5, Mesh, Texture
from driftmaps.ac.scene import (
    DECAL_FLAGS,
    HIDDEN_WALL_FLAGS,
    MARKER_FILE,
    MARKER_TEXT,
    Scene,
    check_track_folder,
    clear_track_folder,
    gate_dummies,
    ks_per_pixel,
    spawn_dummy,
    write_ai_files,
)
from driftmaps.ac.textures import (
    asphalt_texture,
    grass_texture,
    gravel_texture,
    kn5_texture,
    solid_texture,
)
from driftmaps.ac.track_files import (
    DATA_SOURCES,
    DEFAULT_SURFACES,
    GENERATOR_NAME,
    Surface,
    UiTrack,
    cameras_along,
    make_track_id,
    render_map,
    validate_track_id,
    write_cameras_ini,
    write_crew_ini,
    write_data_license,
    write_groove_ini,
    write_lighting_ini,
    write_outline_png,
    write_preview_png,
    write_surfaces_ini,
    write_ui_track_json,
)
from driftmaps.ac.tree_textures import tree_atlas_texture
from driftmaps.elevation import road_profile, sample_elevation
from driftmaps.elevation.sources import (
    ElevationSource,
    SourceInfo,
    SourceUnavailable,
    default_sources,
    region_of,
)
from driftmaps.geo.attributes import MAX_WIDTH_M, MIN_DRIVE_WIDTH_M, region_for
from driftmaps.geo.barriers import (
    BarrierSpan,
    OsmBarrier,
    RoadsideFeatures,
    fetch_barriers,
    fetch_roadside,
    load_user_file,
    match_barriers,
)
from driftmaps.geo.building_sources import (
    MUNICIPAL_SOURCES,
    SourceError,
    fetch_municipal,
    fetch_nrcan,
)
from driftmaps.geo.buildings import (
    OSM_BUILDINGS_ATTRIBUTION,
    BuildingFootprints,
    Footprint,
    conflate,
    fetch_ms_buildings,
    fetch_osm_buildings,
    route_boxes,
    within_route,
)
from driftmaps.geo.mapillary import (
    MAPILLARY_ATTRIBUTION,
    MapFeature,
    MapillaryError,
    drop_stale,
    fetch_image_points,
    fetch_map_features,
    token_from_env,
)
from driftmaps.geo.overpass import (
    OSM_ATTRIBUTION,
    NetworkData,
    OverpassError,
    default_cache_dir,
    fetch_network,
    load_cached,
)
from driftmaps.geo.project import LocalProjection
from driftmaps.geo.router import Continuation, Route, route_continuations
from driftmaps.geo.street_features import StreetFeatures, fetch_street_features
from driftmaps.geometry.barriers import (
    ScenePiece,
    barrier_pieces,
    concrete_texture,
    plan_barriers,
    steel_texture,
)
from driftmaps.geometry.bridges import (
    BridgeOptions,
    BridgePlan,
    bridge_pieces,
    bridge_spans,
    deck_ceiling,
    deck_half_width,
    plan_bridge,
    terrain_through_deck,
)
from driftmaps.geometry.buildings import (
    Building,
    BuildingStats,
    building_scene,
    place_buildings,
)
from driftmaps.geometry.clearance import DrivableSurface
from driftmaps.geometry.furniture import (
    FurnitureMesh,
    FurnitureOptions,
    FurniturePlan,
    furniture_meshes,
    marking_meshes,
    parse_speed,
    plan_furniture,
    street_names,
)
from driftmaps.geometry.network import (
    MATERIALS as NETWORK_MATERIALS,
)
from driftmaps.geometry.network import (
    NetNode,
    NetWay,
    NetworkOptions,
    RoadNetwork,
    build_driveways,
    build_network,
)
from driftmaps.geometry.primitives import MeshData, split
from driftmaps.geometry.road import (
    RoadMeshes,
    RoadModel,
    build_road_meshes,
    closed_centreline,
    extend_straight,
    fillet_corners,
    road_columns,
    simplify,
    smooth_centreline,
)
from driftmaps.geometry.roundabouts import pavers_texture
from driftmaps.geometry.street_items import POLE_UTILITY, StreetItem
from driftmaps.geometry.terrain import (
    Terrain,
    TerrainExtra,
    TerrainOptions,
    build_terrain,
    clamp_heights,
    tile_keys,
)
from driftmaps.geometry.vegetation import (
    BUSH,
    CONIFER,
    CanopyFn,
    CanopySampler,
    LandcoverFn,
    LandcoverSampler,
    TreeSet,
    TreeStats,
    VegetationOptions,
    place_trees,
    tree_scene,
)
from driftmaps.geometry.walls import WallLines, corridor_polygon, wall_lines, wall_meshes
from driftmaps.geometry.warrants import (
    Hazard,
    WarrantOptions,
    warrant_inputs,
    warrant_spans,
)

__all__ = [
    "BuildOptions",
    "BuildResult",
    "TrackGeometry",
    "build_track",
    "dtm_height_fn",
    "loop_direction",
    "prepare_geometry",
    "railed_decks",
    "render_debug_png",
]

SPAWN_HEIGHT_M = 1.0  # M0 verified (dm_t1, dm_t12): cars drop onto the road
GATE_EXTRA_M = 5.0  # gate dummies this far beyond the asphalt edge (ac_format.md §2.3)
GATE_HEIGHT_M = 1.0
# Spawn grid (distances behind A): first start row, row spacing, gap between the start rows and
# the pit rows and between the pit rows and the hotlap start, and the road kept behind the
# hotlap start (before the run-up's end wall).
GRID_FIRST_M = 6.0
GRID_ROW_M = 7.0
GRID_GAP_M = 8.0
GRID_TAIL_M = 8.0
# Touge Battle mode data (csp/new-modes/acmg-touge reads data/<TOUGE_FILE>): the "chase" start
# puts the chaser this far (centre to centre) behind the leader, one car length plus a gap.
TOUGE_FILE = "acmg_touge.json"
TOUGE_CHASE_GAP_M = 10.0
MAX_SPLITS = 1  # A->B tracks: one AC_TIME split is verified; more are not
# Loops (circuits): the hotlap start this far before the start/finish line (ASSUMED: enough to
# be at speed over the line on a city block, short enough to stay on the lap's last straight),
# and how much of a lap the height profile sees before and after the join.
HOTLAP_BEFORE_M = 120.0
LOOP_PROFILE_PAD_M = 600.0
MIN_LOOP_M = 200.0
# Datum offset between DEM sources (geo.md §2): measured on gentle near-ring terrain beside the
# road. ASSUMED values; real datum differences (CGVD28/2013, EGM2008) are well under 2 m.
OFFSET_RING_M = (30.0, 60.0, 100.0, 140.0)
OFFSET_ROAD_CLEAR_M = 20.0
OFFSET_SLOPE_STEP_M = 10.0
OFFSET_MAX_SLOPE = 0.15
OFFSET_MAX_M = 2.0
M_PER_DEG = 111_320.0
BRIDGE_WEIGHT_PAD_M = 10.0  # geo.md §2.4: zero elevation weight 10 m beyond bridges/tunnels
WIDTH_RAMP_M = 30.0  # geo.md §1.5: ramp width changes over 30 m
# ASSUMED: lodOut far enough that far terrain never pops out inside the corridor.
FAR_LOD_OUT_M = 5000.0

# DIRT: steep cut/fill batters. ASSUMED values between vanilla GRASS and our GRAVEL; no CSP
# surface type (none is documented for dirt).
DIRT_SURFACE = Surface(
    key="DIRT",
    friction=0.7,
    damping=0.02,
    wav="sand.wav",
    dirt_additive=1.0,
    is_valid_track=False,
    sin_height=0.03,
    sin_length=0.5,
    vibration_gain=0.25,
    vibration_length=0.4,
)
SURFACES: tuple[Surface, ...] = (*DEFAULT_SURFACES, DIRT_SURFACE)

# Elevation source name -> LICENSE-DATA.txt notice key (others use the source's own text).
_LICENCE_KEYS = {
    "glo30": "copernicus",
    "3dep-1m": "usgs_3dep",
    "3dep-13": "usgs_3dep",
    "ign-rge-alti": "ign",
}
# Coarse fallback chains for far terrain (visual only), by elevation region.
_COARSE_CHAINS = {
    "CA": ("mrdem-30", "glo30"),
    "CA-border": ("mrdem-30", "3dep-13", "glo30"),
    "US": ("3dep-13", "glo30"),
    "US-border": ("3dep-13", "mrdem-30", "glo30"),
    "FR": ("glo30",),
    "WORLD": ("glo30",),
}
# Sources whose 1 m grid is (at least in places) coarser data upsampled by nearest neighbour:
# terrain heights are averaged over this footprint so the grid does not alias the steps.
# OBSERVED 2026-10-15: IGN WMS-R heights near Ollioules are flat for 3-4 m and then jump, i.e.
# 5 m RGE ALTI resampled to 1 m. The road profile is smoothed much more anyway.
_TERRAIN_FILTER_M = {"ign-rge-alti": 5.0}
_COUNTRIES = {
    "CA": "Canada",
    "CA-border": "Canada",
    "US": "United States",
    "US-border": "United States",
    "FR": "France",
}


@dataclass
class BuildOptions:
    """Everything a build can be told. Defaults are the M1 values (ASSUMED until tuned)."""

    name: str = "Map Generator track"
    track_id: str | None = None
    description: str = ""
    city: str = ""
    country: str | None = None  # None: from the elevation region (Canada, US, France)
    cache_dir: Path | None = None
    run_up_m: float = 60.0
    run_off_m: float = 250.0
    ai_past_b_m: float = 220.0  # AI keeps going this far past B, so it never brakes at the gate
    shoulder_m: float = 1.0
    verge_m: float = 4.0  # walls stand this far beyond the gravel shoulder
    max_deviation_m: float = 1.5
    smoothing_cutoff_m: float = 40.0
    max_grade: float = 0.20
    max_vertical_curvature: float = 1.0 / 150.0
    chunk_m: float = 250.0
    pits: int = 8
    splits: int = 0
    terrain: TerrainOptions = field(default_factory=TerrainOptions)
    texture_format: Literal["png", "dds"] = "png"
    elevation_sources: Mapping[str, ElevationSource] | None = None
    elevation_chain: Sequence[str] | None = None
    elevation_prefer: str | Sequence[str] | None = None
    debug_png: bool = True
    force: bool = False  # replace an existing <id> folder even if drift-maps did not make it
    # Scenery (M2). Bridges always get their structure (parapets: all but at-grade culverts).
    trees: bool = True
    buildings: bool = True
    guardrails: bool = True  # OSM barriers and guardrails.json (and the automatic ones)
    # Also rails where Ontario design rules require one (geometry.warrants; needs guardrails).
    auto_guardrails: bool = True
    warrants: WarrantOptions = field(default_factory=WarrantOptions)
    bridge_approach_rails: bool = False  # also invent rails on bridge approaches (no source)
    scenery_density: float = 1.0  # scales the tree budget (0 = no trees, 2 = twice the cap)
    guardrails_file: Path | None = None  # user spans (route metres from A); see geo.barriers
    # The route is route.json driven the other way (--reverse): guardrails.json spans, measured
    # in route.json's direction, are mirrored (left <-> right, s -> length - s).
    route_reversed: bool = False
    microsoft_buildings: bool = True  # Microsoft ML footprints where no other source has one
    nrcan_buildings: bool = True  # NRCan lidar footprints and heights (geo.building_sources)
    # Municipal footprint layers (geo.building_sources.MUNICIPAL_SOURCES) to use where they
    # cover the route. Burlington's terms are revocable and must travel with the track
    # (LICENSE-DATA.txt carries their URL); drop "burlington" to build without it.
    municipal_buildings: tuple[str, ...] = ("burlington", "milton", "oakville")
    # Street furniture (geometry.furniture): signs, poles and wires, lights, signals, hydrants,
    # mailboxes and the route's stop bars. Mapillary detections need a client token
    # (None: $DRIFTMAPS_MAPILLARY_TOKEN; without one, OSM and the rules only).
    furniture: bool = True
    mapillary_token: str | None = field(default=None, repr=False)
    furniture_options: FurnitureOptions = field(default_factory=FurnitureOptions)
    offline: bool = False  # scenery downloads (Overpass, Microsoft) come from the cache only
    bridge: BridgeOptions = field(default_factory=BridgeOptions)
    vegetation: VegetationOptions = field(default_factory=VegetationOptions)
    # Data hooks for tests and power users; None = the live sources (NRCan lidar + land cover,
    # Overpass, Microsoft). Factories get the build's projection (AC XZ <-> lon/lat).
    canopy_source: Callable[[LocalProjection], CanopyFn] | None = None
    landcover_source: Callable[[LocalProjection], LandcoverFn] | None = None
    osm_barriers: Sequence[OsmBarrier] | None = None
    roadside_features: RoadsideFeatures | None = None  # water, rock, driveways (warrants)
    building_footprints: Sequence[Footprint] | None = None
    street_features: StreetFeatures | None = None  # OSM furniture nodes and power lines
    mapillary_features: Sequence[MapFeature] | None = None
    mapillary_coverage: np.ndarray | None = field(default=None, repr=False)  # photos (lon, lat)
    # The road network (geometry.network): side roads, junctions, roundabouts, driveways and
    # cul-de-sacs within `network_options.radius_m`; `real_run_ends`: the run-up and run-off
    # follow the real roads beyond A and B. `osm_network`: the OSM data (None: fetched).
    network: bool = True
    real_run_ends: bool = True
    network_options: NetworkOptions = field(default_factory=NetworkOptions)
    osm_network: NetworkData | None = None
    # Called at each major step with a short plain-English step name and the overall progress
    # (0..1), e.g. for the local app's progress bar. None: no reports.
    progress: Callable[[str, float], None] | None = field(default=None, repr=False)


@dataclass
class BuildResult:
    track_dir: Path
    track_id: str
    stats: dict[str, object]
    warnings: list[str]
    geometry: TrackGeometry | None = field(default=None, repr=False)
    debug_images: list[Path] = field(default_factory=list)
    scenery: _Scenery | None = field(default=None, repr=False)  # trees, buildings, rails, bridges

    def summary(self) -> str:
        s = self.stats
        if s.get("closed"):
            length = f"length   {s['length_m'] / 1000:.2f} km lap, {s.get('run', '')}"
        else:
            length = (
                f"length   {s['length_m'] / 1000:.2f} km A->B ({s['road_length_m'] / 1000:.2f} "
                "km of road incl. run-up/run-off)"
            )
        lines = [
            f"track    {self.track_dir}",
            length,
            f"climb    {s['elevation_min_m']:.1f} .. {s['elevation_max_m']:.1f} m, max grade "
            f"{100 * s['max_grade']:.1f}%",
            f"DEM      {s['elevation_source']} (road coverage {s['elevation_coverage']})",
            f"KN5      {s['kn5_bytes'] / 1e6:.1f} MB, {s['mesh_count']} meshes "
            f"({s['draw_meshes']} drawn, {s['collision_meshes']} hidden walls), vertices "
            + ", ".join(f"{k} {v:,}" for k, v in s["vertices"].items()),
        ]
        sc = s.get("scenery") or {}
        if sc:
            tr = sc["trees"]
            lines.append(
                f"trees    {tr['total']:,} (near {tr['near']:,}, far {tr['far']:,}; deciduous "
                f"{tr['deciduous']:,}, conifer {tr['conifer']:,}, bush {tr['bush']:,}) in "
                f"{tr['meshes']} meshes"
            )
            bd = sc["buildings"]
            found = bd.get("footprints_found") or {
                "osm": bd["footprints_osm"],
                "microsoft": bd["footprints_microsoft"],
            }
            cats = ", ".join(f"{k} {v}" for k, v in sorted((bd.get("categories") or {}).items()))
            lines.append(
                f"houses   {bd['buildings']} buildings"
                + (f" ({cats})" if cats else "")
                + ", footprints found: "
                + (", ".join(f"{k} {v}" for k, v in sorted(found.items())) or "none")
                + "; heights "
                + (", ".join(f"{k} {v}" for k, v in sorted(bd["height_source"].items())) or "-")
            )
            pl = bd.get("placement") or {}
            if pl.get("found"):
                lines.append(
                    f"         placed {pl['placed']} of {pl['candidates']} in reach "
                    f"({pl['town']} in town, {pl['street_front']} street front within 60 m); "
                    f"dropped: on the road {pl['road']}, wall corridor {pl['corridor']}, "
                    f"too far {pl['far']}, size {pl['size']}, budget {pl['budget']} "
                    f"({pl.get('mesh_bytes', 0) / 1e6:.1f} of "
                    f"{pl.get('budget_bytes', 0) / 1e6:.0f} MB)"
                )
            rails = ", ".join(f"{k} {v:.0f} m" for k, v in sorted(sc["barriers_m"].items()))
            lines.append(
                f"rails    {rails or 'none'} (OSM barriers found: {sc['osm_barriers']}, "
                f"guardrails.json spans: {sc['user_spans']}, automatic: "
                f"{sc.get('warrant_spans', 0)} spans)"
            )
            fu = sc.get("furniture") or {}
            if fu:
                by = fu["by_kind"]

                def total(prefix: str, by=by) -> int:
                    return sum(sum(v.values()) for k, v in by.items() if k.startswith(prefix))

                lines.append(
                    f"street   {total('sign:')} signs, {total('pole:')} poles "
                    f"({fu['wire_spans']} wire spans), {total('light:')} lights, "
                    f"{total('signal:')} signals, {total('marking:')} markings; Mapillary "
                    f"{fu['mapillary_used']} items, photos on "
                    f"{100 * fu['mapillary_photo_coverage']:.0f}% of the road"
                )
            for b in sc["bridges"]:
                lines.append(
                    f"bridge   {b['length_m']:.0f} m at {b['from_a_m']:.0f} m from A, deck up to "
                    f"{b['max_height_m']:.1f} m above ground, {b['piers']} piers"
                    + (" (solid: culvert/embankment)" if b["solid"] else "")
                    + ("" if b["railing"] else ", no parapets (at-grade culvert)")
                )
            net = s.get("network") or {}
            if net:
                lines.append(
                    f"network  {net.get('side_roads', 0)} side roads "
                    f"({net.get('side_road_km', 0.0):.2f} km), {net.get('junctions', 0)} "
                    f"junctions, {net.get('roundabouts', 0)} roundabouts, driveways "
                    f"{net.get('driveways_osm', 0)} OSM + {net.get('driveways_inferred', 0)} "
                    f"inferred, {s.get('street_items', 0)} street items"
                )
            if sc["deck_intrusion_m"] > DECK_TOLERANCE_M:
                lines.append(f"warning  terrain {sc['deck_intrusion_m']:.2f} m into a bridge deck")
        lines.append(f"time     {s['build_s']:.1f} s")
        lines += [f"warning  {w}" for w in self.warnings]
        return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Elevation
# --------------------------------------------------------------------------------------------


class _Elevation:
    """DEM access for one build: picks the reference source and corrects the others to it."""

    def __init__(self, options: BuildOptions, lon0: float, lat0: float) -> None:
        self.cache_dir = options.cache_dir
        self.custom = options.elevation_sources is not None or options.elevation_chain is not None
        self.sources = (
            dict(options.elevation_sources) if options.elevation_sources else default_sources()
        )
        self.chain = options.elevation_chain
        self.prefer = options.elevation_prefer
        self.region = str(region_of(np.array([lon0]), np.array([lat0]))[0])
        self.primary: str | None = None
        self.offsets: dict[str, float] = {}
        self.info: dict[str, SourceInfo] = {}
        self.coverage: dict[str, float] = {}
        self._road_lonlat: tuple[np.ndarray, np.ndarray] | None = None
        self._ref: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _sample(self, lon: np.ndarray, lat: np.ndarray, chain: Sequence[str] | None = None):
        return sample_elevation(
            lon,
            lat,
            cache_dir=self.cache_dir,
            sources=self.sources,
            chain=chain if chain is not None else self.chain,
            prefer=self.prefer if chain is None else None,
        )

    def _reference(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(lon, lat, reference height) at gentle-slope terrain points of the near ring beside
        the road, where datum offsets to other sources are measured. The road itself is left
        out: a coarse DSM is biased there (cuts, valleys, trees), and that bias must not shift
        the far terrain or leave a step at the near/far seam."""
        if self._ref is None:
            lon, lat = self._road_lonlat
            lon0, lat0 = float(lon.mean()), float(lat.mean())
            kx, ky = M_PER_DEG * np.cos(np.radians(lat0)), M_PER_DEG
            road = np.stack([(lon - lon0) * kx, (lat - lat0) * ky], axis=1)
            ang = np.arange(8) * (np.pi / 4.0)
            ring = np.asarray(OFFSET_RING_M)[:, None, None] * np.stack(
                [np.cos(ang), np.sin(ang)], axis=1
            )
            centres = road[:: max(1, len(road) // 300)]
            cand = (centres[:, None, :] + ring.reshape(1, -1, 2)).reshape(-1, 2)
            cand = cand[cKDTree(road).query(cand)[0] >= OFFSET_ROAD_CLEAR_M]
            h = OFFSET_SLOPE_STEP_M
            stencil = np.array([[0.0, 0.0], [h, 0.0], [-h, 0.0], [0.0, h], [0.0, -h]])
            pts = (cand[:, None, :] + stencil[None]).reshape(-1, 2)
            z = self._sample(
                pts[:, 0] / kx + lon0, pts[:, 1] / ky + lat0, chain=[self.primary]
            ).heights.reshape(-1, 5)
            with np.errstate(invalid="ignore"):
                slope = np.hypot(z[:, 1] - z[:, 2], z[:, 3] - z[:, 4]) / (2.0 * h)
                ok = np.isfinite(z).all(axis=1) & (slope <= OFFSET_MAX_SLOPE)
            self._ref = (cand[ok, 0] / kx + lon0, cand[ok, 1] / ky + lat0, z[ok, 0])
        return self._ref

    def _offset(self, name: str) -> float:
        if name == self.primary or not name:
            return 0.0
        if name not in self.offsets:
            lon, lat, z = self._reference()
            other = self._sample(lon, lat, chain=[name]).heights if len(lon) else z
            ok = np.isfinite(other)
            offset = float(np.median(z[ok] - other[ok])) if ok.any() else 0.0
            if abs(offset) > OFFSET_MAX_M:
                warnings.warn(
                    f"elevation source {name} differs from {self.primary} by {offset:+.1f} m on "
                    "flat ground near the road; that is not a datum shift, so it is not corrected",
                    stacklevel=2,
                )
                offset = 0.0
            self.offsets[name] = offset
        return self.offsets[name]

    def _corrected(self, smp) -> np.ndarray:
        z = smp.heights.astype(np.float64).copy()
        for name in np.unique(smp.source):
            if name and name != self.primary:
                z[smp.source == name] += self._offset(str(name))
        self.info.update(smp.info)
        return z

    def road(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Heights at road sample points; the first call picks the reference source."""
        smp = self._sample(lon.ravel(), lat.ravel())
        names, counts = np.unique(smp.source[smp.valid], return_counts=True)
        if len(names) == 0:
            raise RuntimeError("no elevation data along the road (check the network or cache)")
        order = np.lexsort((names, -counts))
        self.primary = str(names[order[0]])
        total = counts.sum()
        self.coverage = {str(n): float(c) / total for n, c in zip(names, counts, strict=True)}
        self._road_lonlat = (lon.ravel(), lat.ravel())
        return self._corrected(smp).reshape(lon.shape)

    def resolution_m(self) -> float:
        """Smoothing resolution: the coarsest source covering >= 10% of the road."""
        res = [self.info[n].resolution_m for n, f in self.coverage.items() if f >= 0.1]
        return max(res) if res else self.info[self.primary].resolution_m

    def terrain(self, lon: np.ndarray, lat: np.ndarray, far: bool) -> np.ndarray:
        chain = None
        if far and not self.custom:
            chain = _COARSE_CHAINS.get(self.region, ("glo30",))
        width = 0.0 if far else _TERRAIN_FILTER_M.get(self.primary or "", 0.0)
        if width <= 0.0:
            return self._corrected(self._sample(lon, lat, chain))
        # Box filter: mean of a 3 x 3 stencil spanning `width` metres around each point.
        k = np.array([-1.0, 0.0, 1.0]) * width / 3.0
        dx, dy = (a.ravel() for a in np.meshgrid(k, k))
        m_per_deg = 111_320.0
        lon_s = lon[:, None] + dx[None, :] / (m_per_deg * np.cos(np.radians(lat[:, None])))
        lat_s = lat[:, None] + dy[None, :] / m_per_deg
        z = self._corrected(self._sample(lon_s.ravel(), lat_s.ravel(), chain))
        return np.nanmean(z.reshape(len(lon), 9), axis=1)

    @property
    def any_dsm(self) -> bool:
        return any(i.surface == "DSM" for i in self.info.values())


# --------------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------------


@dataclass
class TrackGeometry:
    """Everything derived from the route before meshing (in AC coordinates)."""

    route: Route
    projection: LocalProjection
    model: RoadModel
    s_a: float
    s_b: float
    z_ref: float  # absolute DEM height of Y = 0
    raw_z: np.ndarray  # (N,) median raw DEM height per station (relative to z_ref)
    lanes: np.ndarray
    oneway: np.ndarray
    tunnel: np.ndarray
    driving_side: str
    region: str
    elevation: _Elevation
    osm_polyline: np.ndarray  # (M, 3) the OSM node polyline (Y = 0), A..B
    # (N,) OSM way id per station (its tags: route.ways), clamped at the run-up/run-off.
    way_id: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    @property
    def length_m(self) -> float:
        return self.s_b - self.s_a

    @property
    def closed(self) -> bool:
        """A loop (circuit): s_a = 0 is the start/finish line and s_b the lap length."""
        return self.model.closed


def _box_smooth(values: np.ndarray, window: int, closed: bool = False) -> np.ndarray:
    """Moving average over `window` stations; `closed`: round a loop (last value = first)."""
    if window <= 1 or len(values) < 3:
        return values.astype(float)
    pad = window // 2
    kernel = np.ones(2 * pad + 1) / (2 * pad + 1)
    if closed:
        ring = values[:-1].astype(float)
        idx = np.arange(-pad, len(ring) + pad) % len(ring)
        out = np.convolve(ring[idx], kernel, mode="valid")
        return np.append(out, out[:1])
    padded = np.pad(values.astype(float), pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _pad_mask(mask: np.ndarray, s: np.ndarray, pad_m: float) -> np.ndarray:
    """Grow a boolean station mask by `pad_m` metres of arc length on both sides."""
    if not mask.any():
        return mask.copy()
    on = s[mask]
    idx = np.searchsorted(on, s)
    before = np.abs(s - on[np.clip(idx - 1, 0, len(on) - 1)])
    after = np.abs(on[np.clip(idx, 0, len(on) - 1)] - s)
    return np.minimum(before, after) <= pad_m


def run_up_length_m(options: BuildOptions) -> float:
    """How far the road runs on behind A: `run_up_m`, or more to fit the grid and pits."""
    return max(options.run_up_m, grid_length_m(options.pits) + GRID_TAIL_M)


def _continued(
    xz: np.ndarray,
    param: np.ndarray,
    proj: LocalProjection,
    ends: tuple[Continuation | None, Continuation | None],
    before_m: float,
    after_m: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """The OSM polyline with the real roads behind A and beyond B joined on (params below 0
    before A, above the route length after B), and how much straight run-up and run-off is
    still missing where the real road ended early."""
    parts_xz, parts_p = [xz], [param]
    missing = [before_m, after_m]
    for k, cont in enumerate(ends):
        if cont is None or len(cont.lonlat) < 2:
            continue
        en = proj.to_xy(cont.lonlat)
        c = np.stack([en[:, 0], -en[:, 1]], axis=1)
        c[0] = xz[0] if k == 0 else xz[-1]
        d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
        keep = np.concatenate([[False], np.diff(d) > 1e-6])
        c, d = c[keep], d[keep]
        if not len(d):
            continue
        missing[k] = max(0.0, missing[k] - float(d[-1]))
        if k == 0:
            parts_xz.insert(0, c[::-1])
            parts_p.insert(0, -d[::-1])
        else:
            parts_xz.append(c)
            parts_p.append(param[-1] + d)
    return np.concatenate(parts_xz), np.concatenate(parts_p), missing[0], missing[1]


def prepare_geometry(
    route: Route,
    options: BuildOptions,
    continuations: tuple[Continuation | None, Continuation | None] = (None, None),
) -> TrackGeometry:
    """Steps 2-3 of the pipeline: centreline, attributes per station and the height profile.

    `continuations` (`geo.router.route_continuations`): the real roads behind A and beyond B;
    the run-up and run-off follow them, and go on straight only where they end early."""
    pin_a = route.pins[0]
    proj = LocalProjection(pin_a.snap_lat, pin_a.snap_lon)
    nodes = route.polyline(step_m=1e9)
    attrs = route.polyline(step_m=1.0)
    en = proj.to_xy(nodes.lonlat)
    xz = np.stack([en[:, 0], -en[:, 1]], axis=1)
    raw_length = float(nodes.s[-1])
    closed = route.closed

    if closed:  # a loop: no run-up or run-off, smoothed round the join
        xz_s, param = closed_centreline(
            xz,
            nodes.s,
            max_deviation_m=options.max_deviation_m,
            cutoff_m=options.smoothing_cutoff_m,
        )
    else:
        run_up = run_up_length_m(options)
        xz_c, p_c, up, off = _continued(xz, nodes.s, proj, continuations, run_up, options.run_off_m)
        xz_f, p_f = fillet_corners(*simplify(xz_c, p_c, 0.3))
        xz_e, p_e = extend_straight(xz_f, p_f, up, off)
        xz_s, param = smooth_centreline(
            xz_e,
            p_e,
            max_deviation_m=options.max_deviation_m,
            cutoff_m=options.smoothing_cutoff_m,
        )
    param = np.maximum.accumulate(param)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xz_s, axis=0), axis=1))])
    if closed:
        s_a, s_b = 0.0, float(s[-1])
        if s_b < MIN_LOOP_M:
            raise ValueError(
                f"the loop is only {s_b:.0f} m long; make it at least {MIN_LOOP_M:.0f} m"
            )
    else:
        s_a = float(np.interp(0.0, param, s))
        s_b = float(np.interp(raw_length, param, s))

    # Attributes per station: the OSM polyline point at that distance (clamped at the ends).
    j = np.clip(
        np.searchsorted(attrs.s, np.clip(param, 0.0, raw_length), side="right") - 1,
        0,
        len(attrs) - 1,
    )
    width = np.clip(attrs.width_m[j], MIN_DRIVE_WIDTH_M, MAX_WIDTH_M)
    width = _box_smooth(width, int(WIDTH_RAMP_M), closed)
    lanes = attrs.lanes[j]
    oneway = attrs.oneway[j]
    way_id = attrs.way_id[j]
    on_route = np.ones(len(s), bool) if closed else (param > 0) & (param < raw_length)
    bridge = attrs.bridge[j] & on_route
    tunnel = attrs.tunnel[j] & on_route
    if closed:
        for a in (lanes, oneway, bridge, tunnel):
            a[-1] = a[0]
    region = region_for(pin_a.snap_lat, pin_a.snap_lon)

    # Elevation: five samples across the asphalt at every station.
    if closed:
        ring = xz_s[:-1]
        tangent = np.roll(ring, -1, axis=0) - np.roll(ring, 1, axis=0)
        tangent = np.vstack([tangent, tangent[:1]])
    else:
        tangent = np.gradient(xz_s, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    left = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)  # left = up x forward, in XZ
    fr = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    lat_xz = xz_s[:, None, :] + (fr[None, :, None] * width[:, None, None]) * left[:, None, :]
    lon_s, lat_s = proj.to_lonlat(lat_xz[..., 0], -lat_xz[..., 1])
    elev = _Elevation(options, pin_a.snap_lon, pin_a.snap_lat)
    z_samples = elev.road(np.asarray(lon_s), np.asarray(lat_s))
    weights = np.where(_pad_mask(bridge | tunnel, s, BRIDGE_WEIGHT_PAD_M), 0.0, 1.0)
    profile = functools.partial(
        road_profile,
        source_resolution_m=elev.resolution_m(),
        max_grade=options.max_grade,
        max_vertical_curvature=options.max_vertical_curvature,
    )
    if closed:
        z = _closed_profile(s, z_samples, weights, profile)
    else:
        z = profile(s, z_samples, weights)
    z_ref = float(np.interp(s_a, s, z))
    raw_z = np.nanmedian(z_samples, axis=1) - z_ref

    centre = np.stack([xz_s[:, 0], z - z_ref, xz_s[:, 1]], axis=1)
    model = RoadModel(
        s=s,
        centre=centre,
        half_width=width / 2.0,
        shoulder=options.shoulder_m,
        centre_line=(lanes >= 2) & (oneway == 0),
        bridge=bridge,
        double_yellow=region == "na",
        closed=closed,
    )
    osm = np.stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]], axis=1)
    return TrackGeometry(
        route=route,
        projection=proj,
        model=model,
        s_a=s_a,
        s_b=s_b,
        z_ref=z_ref,
        raw_z=raw_z,
        lanes=lanes,
        oneway=oneway,
        tunnel=tunnel,
        driving_side=route.driving_side,
        region=region,
        elevation=elev,
        osm_polyline=osm,
        way_id=np.asarray(way_id, np.int64),
    )


def _closed_profile(
    s: np.ndarray,
    z_samples: np.ndarray,
    weights: np.ndarray,
    profile: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """`road_profile` round a loop (the last station is the first): the stations are padded
    with up to `LOOP_PROFILE_PAD_M` of the lap before and after, so the smoothing and the grade
    limits see the road go on across the join, then the tiny mismatch left between the end and
    the start is spread evenly over the lap, so the road meets itself exactly."""
    lap = float(s[-1])
    pad = min(LOOP_PROFILE_PAD_M, lap)
    before = np.flatnonzero(s[:-1] >= lap - pad)
    after = np.flatnonzero((s > 0.0) & (s <= pad))
    idx = np.r_[before, np.arange(len(s)), after]
    s_ext = np.r_[s[before] - lap, s, s[after] + lap]
    z = profile(s_ext, z_samples[idx], weights[idx])[len(before) : len(before) + len(s)]
    return z - (z[-1] - z[0]) * (s / lap)


def loop_direction(centre: np.ndarray) -> str:
    """Which way a closed centreline (N, 3) in AC coordinates (X = east, Z = south) goes
    round, seen from above with north up: "clockwise" or "anticlockwise" (sign of its area)."""
    x, north = centre[:, 0], -centre[:, 2]
    area = 0.5 * float(np.sum(x[:-1] * north[1:] - x[1:] * north[:-1]))
    area += 0.5 * float(x[-1] * north[0] - x[0] * north[-1])  # 0 when the last point is the first
    return "anticlockwise" if area > 0.0 else "clockwise"


def _terrain_height_fn(geo: TrackGeometry):
    def fn(xz: np.ndarray, far: bool) -> np.ndarray:
        lon, lat = geo.projection.to_lonlat(xz[:, 0], -xz[:, 1])
        return geo.elevation.terrain(np.asarray(lon), np.asarray(lat), far) - geo.z_ref

    return fn


def dtm_height_fn(geo: TrackGeometry) -> Callable[[np.ndarray], np.ndarray]:
    """The bare-earth DEM (the lidar DTM where there is one) under AC XZ points, as AC Y:
    the real ground, before the build cuts and fills it to meet the road."""
    fn = _terrain_height_fn(geo)
    return lambda xz: fn(np.asarray(xz, np.float64).reshape(-1, 2), False)


# --------------------------------------------------------------------------------------------
# Dummies and AI
# --------------------------------------------------------------------------------------------


def lane_offset(geo: TrackGeometry) -> np.ndarray:
    """Lateral offset (+ = left) of the driving-lane centre at every station: half the
    asphalt half-width towards the driving side on two-way roads, 0 on one-lane/one-way ones
    (smoothed over 30 m so a change of lane count never jumps)."""
    m = geo.model
    sign = -1.0 if geo.driving_side == "right" else 1.0
    raw = np.where(m.centre_line, sign * m.half_width / 2.0, 0.0)
    return _box_smooth(raw, int(WIDTH_RAMP_M), m.closed)


def _grid(pits: int) -> tuple[list[float], list[float], float]:
    """Distances behind A of the start rows (two cars each, one slot per pit box, at least
    one row), of the pit rows, and of the hotlap start."""
    start_rows = (max(2, pits) + 1) // 2
    starts = [GRID_FIRST_M + GRID_ROW_M * r for r in range(start_rows)]
    pit0 = starts[-1] + GRID_GAP_M
    pit_rows = [pit0 + GRID_ROW_M * r for r in range((pits + 1) // 2)]
    hot = (pit_rows[-1] if pit_rows else starts[-1]) + GRID_GAP_M
    return starts, pit_rows, hot


def grid_length_m(pits: int) -> float:
    """How far behind A the last spawn (the hotlap start) sits."""
    return _grid(pits)[2]


@dataclass
class _Dummies:
    nodes: list[Dummy]
    spawns: dict[str, tuple[float, float]]  # name -> (s, lateral)
    gates: dict[str, float]  # prefix -> s


def _start_lanes(geo: TrackGeometry, offs: np.ndarray, s: float) -> tuple[float, float]:
    """(driving lane, other lane) lateral offsets at `s` (`offs` = `lane_offset(geo)`); on a
    one-way/one-lane stretch two cars still sit side by side, a quarter of the asphalt width
    either side."""
    m = geo.model
    drive = m.interp(offs, s)
    if abs(drive) < 0.25:
        quarter = m.interp(m.half_width, s) / 2.0
        sign = -1.0 if geo.driving_side == "right" else 1.0
        return sign * quarter, -sign * quarter
    return drive, -drive


def _dummies(geo: TrackGeometry, options: BuildOptions) -> _Dummies:
    m = geo.model
    offs = lane_offset(geo)
    up = SPAWN_HEIGHT_M * WORLD_UP
    nodes: list[Dummy] = []
    spawns: dict[str, tuple[float, float]] = {}

    def spawn(name: str, s: float, lateral: float) -> None:
        pos, fwd, left = m.at(s)
        nodes.append(spawn_dummy(name, pos + lateral * left + up, fwd))
        spawns[name] = (s, lateral)

    def lanes_at(s: float) -> tuple[float, float]:
        return _start_lanes(geo, offs, s)

    # Grid: a start slot per pit box (at least 2), side by side in rows every 7 m from 6 m
    # behind A (the first car of a row on the driving lane), the pits in pairs behind them and
    # the hotlap start behind those (`grid_length_m`; prepare_geometry sized the run-up). On a
    # loop "behind A" is the end of the lap, and the hotlap start is HOTLAP_BEFORE_M back.
    starts, pit_rows, hot = _grid(options.pits)
    if geo.closed:
        lap = geo.length_m
        if hot + GRID_GAP_M > lap / 2:
            raise ValueError(
                f"the loop ({lap:.0f} m) is too short for {options.pits} pit boxes; make it "
                "longer or use fewer pits"
            )
        hot = max(hot, min(HOTLAP_BEFORE_M, lap / 2))
    hot_s = m.wrap(geo.s_a - hot)
    if hot_s < 3.0 and not geo.closed:
        raise ValueError("run_up_m is too short for the spawn grid; increase it or use fewer pits")
    n_starts = 2 * len(starts)
    for prefix, count, rows in (("AC_START", n_starts, starts), ("AC_PIT", options.pits, pit_rows)):
        for n in range(count):
            s = m.wrap(geo.s_a - rows[n // 2])
            d, o = lanes_at(s)
            spawn(f"{prefix}_{n}", s, d if n % 2 == 0 else o)
    spawn("AC_HOTLAP_START_0", hot_s, lanes_at(hot_s)[0])

    gates: dict[str, float]
    if geo.closed:
        # A lap: AC_TIME_0 is the start/finish line, AC_TIME_1/2 split it in thirds.
        gates = {f"AC_TIME_{k}": geo.s_a + k * geo.length_m / 3.0 for k in range(3)}
    else:
        splits = min(max(int(options.splits), 0), MAX_SPLITS)
        if splits != options.splits:
            warnings.warn(
                f"{options.splits} splits asked; A->B tracks get at most {MAX_SPLITS}",
                stacklevel=2,
            )
        gates = {"AC_AB_START": geo.s_a, "AC_AB_FINISH": geo.s_b}
        for k in range(splits):
            gates[f"AC_TIME_{k}"] = geo.s_a + (k + 1) * geo.length_m / (splits + 1)
    for prefix, s in gates.items():
        pos, fwd, left = m.at(s)
        reach = m.interp(m.half_width, s) + GATE_EXTRA_M
        nodes += gate_dummies(prefix, pos, fwd, left, reach, GATE_HEIGHT_M)
    return _Dummies(nodes, spawns, gates)


def _touge_data(geo: TrackGeometry, dummies: _Dummies, track_id: str) -> dict:
    """data/acmg_touge.json for the Touge Battle New Mode (A->B tracks only): the A and B
    gates, the side-by-side grid (`AC_START_0` on the driving lane, `AC_START_1` beside it)
    and the chase start (both on the driving lane, `TOUGE_CHASE_GAP_M` apart). Positions are
    AC world coordinates (x east, y up, z south) in metres, spawns `SPAWN_HEIGHT_M` above the
    road like the dummies."""
    m = geo.model

    def vec(v: np.ndarray, nd: int) -> list[float]:
        return [round(float(x), nd) + 0.0 for x in v]  # + 0.0: no "-0.0" in the file

    def pose(s: float, lateral: float) -> dict:
        pos, fwd, left = m.at(s)
        return {
            "position": vec(pos + lateral * left + SPAWN_HEIGHT_M * WORLD_UP, 3),
            "forward": vec(fwd, 5),
        }

    def gate(s: float) -> dict:
        pos, fwd, left = m.at(s)
        hw = m.interp(m.half_width, s)
        return {
            "centre": vec(pos, 3),
            "forward": vec(fwd, 5),
            "left": vec(left, 5),
            "half_width_m": round(hw, 2),
            "reach_m": round(hw + GATE_EXTRA_M, 2),
        }

    grid = []
    for name in ("AC_START_0", "AC_START_1"):
        s, lateral = dummies.spawns[name]
        grid.append({"name": name} | pose(s, lateral))
    s_lead = dummies.spawns["AC_START_0"][0]
    s_chase = s_lead - TOUGE_CHASE_GAP_M
    offs = lane_offset(geo)
    return {
        "format": 1,
        "about": "Touge Battle mode data (Assetto Corsa Map Generator, csp/new-modes/"
        "acmg-touge). AC world coordinates: x east, y up, z south, metres.",
        "track_id": track_id,
        "length_m": round(geo.length_m, 1),
        "driving_side": geo.driving_side,
        "start_gate": gate(geo.s_a),
        "finish_gate": gate(geo.s_b),
        "run_off_m": round(float(m.s[-1]) - geo.s_b, 1),
        "grid": grid,
        "chase": {
            "leader": pose(s_lead, _start_lanes(geo, offs, s_lead)[0]),
            "chaser": pose(s_chase, _start_lanes(geo, offs, s_chase)[0]),
        },
    }


def _ai_line(geo: TrackGeometry, options: BuildOptions) -> tuple[AiLine, slice]:
    m = geo.model
    if geo.closed:  # the whole lap, starting on the start/finish line
        offs = lane_offset(geo)
        hw = m.half_width
        ai = build_ai_line(
            m.centre + offs[:, None] * m.left,
            hw - offs,
            hw + offs,
            closed=True,
            surface_normals=m.up,
            with_grid=True,
        )
        return ai, slice(0, len(m))
    s_end = min(geo.s_b + options.ai_past_b_m, m.s[-1] - 5.0)
    i0 = m.index_at(2.0)
    i1 = m.index_at(s_end)
    sl = slice(i0, i1 + 1)
    offs = lane_offset(geo)[sl]
    hw = m.half_width[sl]
    pts = m.centre[sl] + offs[:, None] * m.left[sl]
    ai = build_ai_line(
        pts,
        hw - offs,
        hw + offs,
        closed=False,
        surface_normals=m.up[sl],
        start_speed_mps=0.0,
        end_speed_mps=0.0,
        with_grid=True,
    )
    return ai, sl


# --------------------------------------------------------------------------------------------
# Textures and the KN5
# --------------------------------------------------------------------------------------------


def _dirt_texture(size: int = 256):
    img = np.asarray(gravel_texture(size, seed=7, stones=size // 6), np.float32)
    tint = np.array([1.08, 0.92, 0.72], np.float32)
    return Image.fromarray(np.clip(img * tint * 0.85, 0, 255).astype(np.uint8))


_IMAGES = {
    "asphalt": lambda: asphalt_texture(512),
    "gravel": lambda: gravel_texture(256),
    "grass": lambda: grass_texture(256),
    "dirt": _dirt_texture,
    "paint_white": lambda: solid_texture((235, 235, 230), 16),
    "paint_yellow": lambda: solid_texture((235, 180, 20), 16),
    "wall_hidden": lambda: solid_texture((255, 0, 255), 16),
    "concrete": concrete_texture,  # bridges, parapets, jersey barriers, walls
    "steel": steel_texture,  # W-beam guardrails, steel bridge rails
    "pavers": pavers_texture,  # roundabout truck aprons
}


@functools.cache
def _texture_data(name: str) -> bytes:
    stem = name.rsplit(".", 1)[0]
    return kn5_texture(name, _IMAGES[stem](), "BC1", mipmaps=True).data


def _texture(name: str) -> Texture:
    return Texture(name=name, data=_texture_data(name))


# The scenery atlases take seconds to paint and never change: one copy per process. Always DDS:
# the tree atlas needs its coverage-preserving alpha mips (scenery.md §4.1, §6).
TREE_ATLAS = "dm_trees.dds"
BUILDING_ATLAS = "dm_buildings.dds"


@functools.cache
def _atlas(name: str) -> Texture:
    return tree_atlas_texture(name) if name == TREE_ATLAS else building_atlas_texture(name)


FURNITURE_ATLAS = "dm_furniture.dds"  # signs, poles, lights (BC1; street names drawn per track)


def _furniture_material(name: str, texture: str, *, bright: bool, scene: Scene) -> int:
    """Index of a furniture material on the atlas (sign faces `bright`: they are retro-
    reflective in real life, so they never sink into shade)."""
    for i, mat in enumerate(scene.materials):
        if mat.name == name:
            return i
    scene.materials.append(ks_per_pixel(name, texture, bright=bright))
    return len(scene.materials) - 1


def _tiles(terrain: Terrain, select: np.ndarray, tile_m: float) -> list[np.ndarray]:
    keys = tile_keys(terrain, select, tile_m)
    idx = np.flatnonzero(select)
    if len(idx) == 0:
        return []
    k = keys[idx]
    order = np.lexsort((k[:, 1], k[:, 0]))
    idx, k = idx[order], k[order]
    bounds = np.flatnonzero(np.any(np.diff(k, axis=0) != 0, axis=1)) + 1
    return np.split(idx, bounds)


def _add_split(
    scene: Scene,
    data: MeshData | None,
    name: str,
    material: int,
    counts: dict,
    key: str,
    **flags,
) -> None:
    if data is None:
        return
    pieces = split(data)
    for p, piece in enumerate(pieces):
        scene.mesh(piece, name if len(pieces) == 1 else f"{name}_{p}", material, **flags)
        counts[key] = counts.get(key, 0) + piece.vertex_count


_NETWORK_PREFIXES = tuple(sorted({prefix for prefix, *_ in NETWORK_MATERIALS.values()}))


def _assemble_kn5(
    track_id: str,
    roads: RoadMeshes,
    terrain: Terrain,
    walls: list[MeshData],
    dummies: list[Dummy],
    ext: str,
    scenery: _Scenery | None = None,
    ground: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[Kn5, dict[str, int], dict[str, object]]:
    """The track's one KN5: road, terrain, hidden walls, then the scenery (bridges, barriers,
    buildings, trees) and the dummies. Returns it with vertex counts per category and the
    tree/building statistics."""
    scene = Scene(track_id, _texture)
    mats = {
        name: scene.material(name, f"{tex}{ext}")
        for name, tex in (
            ("road", "asphalt"),
            ("gravel", "gravel"),
            ("grass", "grass"),
            ("dirt", "dirt"),
            ("paint_white", "paint_white"),
            ("paint_yellow", "paint_yellow"),
            ("wall_hidden", "wall_hidden"),
        )
    }
    counts: dict[str, int] = {}
    for k, mesh in enumerate(roads.road):
        _add_split(scene, mesh, f"1ROAD_c{k:03d}", mats["road"], counts, "road")
    for k, mesh in enumerate(roads.gravel):
        _add_split(scene, mesh, f"1GRAVEL_c{k:03d}", mats["gravel"], counts, "gravel")
    for k, (w, y) in enumerate(zip(roads.paint_white, roads.paint_yellow, strict=True)):
        _add_split(
            scene, w, f"paint_white_c{k:03d}", mats["paint_white"], counts, "paint", **DECAL_FLAGS
        )
        _add_split(
            scene, y, f"paint_yellow_c{k:03d}", mats["paint_yellow"], counts, "paint", **DECAL_FLAGS
        )
    groups = (
        ("1GRASS_t", terrain.near & ~terrain.dirt, "grass", 200.0, {}),
        ("1DIRT_t", terrain.near & terrain.dirt, "dirt", 200.0, {}),
        (
            "far_grass_t",
            ~terrain.near & ~terrain.dirt,
            "grass",
            600.0,
            {"cast_shadows": False, "lod_out": FAR_LOD_OUT_M},
        ),
        (
            "far_dirt_t",
            ~terrain.near & terrain.dirt,
            "dirt",
            600.0,
            {"cast_shadows": False, "lod_out": FAR_LOD_OUT_M},
        ),
    )
    for prefix, select, mat, tile_m, flags in groups:
        key = "terrain_near" if prefix[0] == "1" else "terrain_far"
        for k, tris in enumerate(_tiles(terrain, select, tile_m)):
            mask = np.zeros(len(terrain.triangles), bool)
            mask[tris] = True
            _add_split(
                scene, terrain.mesh(mask), f"{prefix}{k:03d}", mats[mat], counts, key, **flags
            )
    for k, mesh in enumerate(walls):
        _add_split(
            scene,
            mesh,
            f"1WALL_c{k:03d}",
            mats["wall_hidden"],
            counts,
            "walls",
            **HIDDEN_WALL_FLAGS,
        )
    extra: dict[str, object] = {"trees": TreeStats(), "buildings": BuildingStats()}
    if scenery is not None:
        for piece in scenery.pieces:
            if piece.name.startswith(_NETWORK_PREFIXES) or "_dw" in piece.name:
                key = "network"
            elif piece.physical:
                key = "scenery_collision"
            elif piece.name.startswith("bridge"):
                key = "bridges"
            else:
                key = "barriers"
            mat = scene.material(piece.material, f"{piece.texture}{ext}")
            _add_split(scene, piece.mesh, piece.name, mat, counts, key, **piece.flags)
        if scenery.buildings:
            scene.textures.setdefault(BUILDING_ATLAS, _atlas(BUILDING_ATLAS))
            stats = building_scene(scene, scenery.buildings, texture_format="dds")
            counts["buildings"] = stats.vertices
            extra["buildings"] = stats
        if scenery.furniture_meshes:
            tex = FURNITURE_ATLAS
            scene.textures.setdefault(tex, furniture_atlas_texture(tex, scenery.furniture_names))
            fmats = {
                "furniture": _furniture_material("dm_furniture", tex, bright=False, scene=scene),
                "signs": _furniture_material("dm_signs", tex, bright=True, scene=scene),
                "hidden": mats["wall_hidden"],
            }
            for fm in scenery.furniture_meshes:
                key = "scenery_collision" if fm.material == "hidden" else "furniture"
                _add_split(scene, fm.mesh, fm.name, fmats[fm.material], counts, key, **fm.flags)
        if scenery.trees is not None and len(scenery.trees) and ground is not None:
            scene.textures.setdefault(TREE_ATLAS, _atlas(TREE_ATLAS))
            stats = tree_scene(scene, scenery.trees, ground, texture_format="dds")
            counts["trees"] = stats.vertices
            extra["trees"] = stats
    scene.add(*dummies)
    return scene.kn5(), counts, extra


# --------------------------------------------------------------------------------------------
# Scenery: bridges, guardrails, buildings, trees (scenery.md §7)
# --------------------------------------------------------------------------------------------

# Tree trunks stay this far outside the hidden walls' corridor (road + shoulder + verge), so a
# car can never reach a tree card, which has no collision (scenery.md §4.1).
TREE_WALL_MARGIN_M = 0.5
DECK_TOLERANCE_M = 0.05  # terrain grazing a deck by less than this is not reported
APPROACH_RAIL_M = 20.0  # opt-in rails on both sides this far before and after each bridge
BUILDINGS_AROUND_M = 300.0  # buildings drawn this close to the road (BuildingOptions)
BUILDINGS_FETCH_M = 400.0  # footprints fetched this far: the far trees keep off them too
_DATA_ERRORS = (OverpassError, SourceUnavailable, OSError, urllib.error.URLError, EOFError)


@dataclass
class _Scenery:
    """What the scenery step made, for the KN5, the debug images, the stats and credits."""

    pieces: list[ScenePiece] = field(default_factory=list)  # bridges + barriers
    plans: list[BridgePlan] = field(default_factory=list)
    spans: list[BarrierSpan] = field(default_factory=list)  # roadside barriers built
    trees: TreeSet | None = None
    buildings: list[Building] = field(default_factory=list)
    footprints: BuildingFootprints | None = None
    building_report: dict[str, int] = field(default_factory=dict)  # place_buildings' counts
    attributions: list[str] = field(default_factory=list)
    osm_barriers: int = 0
    osm_barrier_ways: list[OsmBarrier] = field(default_factory=list)  # matched to the road
    footprints_masked: bool = False  # every fetched footprint kept trees off (tree mask)
    user_spans: int = 0
    warrant_spans: int = 0
    warrant_report: list[dict] = field(default_factory=list)  # data/guardrails_auto.json
    warrant_hazards: list[Hazard] = field(default_factory=list)
    user_file: Path | None = None
    user_bytes: bytes = b""  # the user file as read, rewritten after the folder is cleared
    deck_intrusion_m: float = 0.0
    terrain_clamped: int = 0
    ground: Callable[[np.ndarray], np.ndarray] | None = field(default=None, repr=False)
    furniture: FurniturePlan | None = None
    furniture_meshes: list[FurnitureMesh] = field(default_factory=list)
    furniture_names: tuple[str, ...] = ()  # street name blades on the atlas
    street_features: StreetFeatures | None = None  # OSM nodes / lines fetched (geojson)
    network: RoadNetwork | None = field(default=None, repr=False)  # side roads, junctions ...
    street_items: list[StreetItem] = field(default_factory=list)  # signs, mailboxes ...


def _ground_fn(terrain: Terrain, road_height: Callable[[np.ndarray], np.ndarray]):
    """The visible terrain height under XZ points; the road height where there is none."""

    def ground(xz: np.ndarray) -> np.ndarray:
        h = terrain.height_at(xz)
        bad = ~np.isfinite(h)
        if bad.any():
            h[bad] = road_height(np.asarray(xz, np.float64).reshape(-1, 2)[bad])
        return h

    return ground


def _route_lonlat(geo: TrackGeometry, step: int = 5) -> np.ndarray:
    """The built centreline (with run-up and run-off) as (N, 2) lon/lat, every `step` m."""
    c = geo.model.centre[::step]
    lon, lat = geo.projection.to_lonlat(c[:, 0], -c[:, 2])
    return np.stack([np.asarray(lon), np.asarray(lat)], axis=1)


# --------------------------------------------------------------------------------------------
# The road network: side roads, junctions, roundabouts (geometry.network)
# --------------------------------------------------------------------------------------------


def _fetch_network(route: Route, options: BuildOptions) -> NetworkData | None:
    """The OSM highways and tagged highway nodes around the route: `options.osm_network`, else
    the region database / cache / Overpass, else the route's own cached routing corridor."""
    if options.osm_network is not None:
        return options.osm_network
    try:
        return fetch_network(
            route.polyline(step_m=5.0).lonlat,
            around_m=options.network_options.radius_m,
            cache_dir=options.cache_dir,
            offline=options.offline,
        )
    except _DATA_ERRORS as e:
        reason = _short(e)
    if route.cache_key:
        cached = load_cached(Path(options.cache_dir or default_cache_dir()), route.cache_key)
        if cached is not None:
            return NetworkData(cached.ways, (), cached.osm_base, nodes_known=False)
    warnings.warn(f"road network unavailable ({reason}); no side roads", stacklevel=2)
    return None


def _xz(projection: LocalProjection, lonlat: np.ndarray) -> np.ndarray:
    en = projection.to_xy(np.asarray(lonlat, np.float64).reshape(-1, 2))
    return np.stack([en[:, 0], -en[:, 1]], axis=1)


def _network(
    geo: TrackGeometry,
    osm: NetworkData,
    ends: tuple[Continuation | None, Continuation | None],
    options: BuildOptions,
) -> RoadNetwork:
    """Side roads, junctions and roundabouts around the route (`geometry.network`)."""
    proj = geo.projection
    ways = [NetWay(w.id, w.nodes, _xz(proj, w.lonlat), w.tags) for w in osm.ways]
    nodes = [NetNode(n.id, _xz(proj, np.array([n.lonlat]))[0], n.tags) for n in osm.nodes]
    route = geo.route
    route_nodes = set(route.node_path())
    route_segments = {(min(e.u, e.v), max(e.u, e.v)) for e in route.edges}
    for cont in ends:
        if cont is not None:
            route_nodes |= set(cont.nodes)
            route_segments |= {
                (min(a, b), max(a, b)) for a, b in zip(cont.nodes[:-1], cont.nodes[1:], strict=True)
            }
    highway = [route.ways.get(int(w), {}).get("highway", "") for w in geo.way_id.tolist()]
    # The real run-up and run-off can be on other roads: their class from the nearest way.
    m = geo.model
    beyond = np.flatnonzero((m.s < geo.s_a) | (m.s > geo.s_b))
    if len(beyond) and ways:
        dense = [
            (
                np.asarray(shapely.segmentize(shapely.LineString(w.xz), 2.0).coords),
                w.tags["highway"],
            )
            for w in ways
            if w.tags.get("highway") and len(w.xz) >= 2
        ]
        pts = np.concatenate([d for d, _ in dense])
        cls = [c for d, c in dense for _ in d]
        _, k = cKDTree(pts).query(m.centre[beyond][:, [0, 2]])
        for i, kk in zip(beyond.tolist(), k.tolist(), strict=True):
            highway[i] = cls[kk]
    nopt = dataclasses.replace(
        options.network_options,
        source_resolution_m=geo.elevation.resolution_m(),
        max_grade=options.max_grade,
        max_vertical_curvature=options.max_vertical_curvature,
        shoulder_m=options.shoulder_m,
        verge_m=options.verge_m,
    )
    return build_network(
        geo.model,
        ways,
        nodes,
        route_nodes=route_nodes,
        route_segments=route_segments,
        route_highway=highway or None,
        route_oneway=geo.oneway,
        dtm=dtm_height_fn(geo),
        driving_side=geo.driving_side,
        options=nopt,
        nodes_known=osm.nodes_known,
    )


def _driveways(
    net: RoadNetwork,
    m: RoadModel,
    ground: Callable[[np.ndarray], np.ndarray],
    options: BuildOptions,
    sc: _Scenery,
    roads: RoadMeshes,
) -> RoadMeshes:
    """OSM and inferred driveways (they need the placed buildings) with their mailboxes; the
    route meshes are rebuilt when driveway mouths cut its edge lines or shoulders."""
    footprints = [(b.polygon, b.kind) for b in sc.buildings]
    res = build_driveways(net, footprints, ground, options.network_options)
    net.driveways = res.driveways
    net.planned_items += res.items
    net.stats.update(res.stats)
    sc.pieces += res.pieces
    if res.masks_changed:
        return build_road_meshes(m, chunk_m=options.chunk_m, keep=net.masks)
    return roads


def _terrain_extra(net: RoadNetwork | None) -> TerrainExtra | None:
    if net is None or net.footprint.is_empty:
        if net is not None and net.masks.hidden.any():
            return TerrainExtra(shapely.Polygon(), np.zeros((0, 3)), hidden=net.masks.hidden)
        return None
    stations, outer = net.terrain_stations()
    return TerrainExtra(net.footprint, net.edge_points, stations, outer, net.masks.hidden)


def _bridges(
    geo: TrackGeometry,
    terrain: Terrain,
    options: BuildOptions,
    road_height: Callable[[np.ndarray], np.ndarray],
    sc: _Scenery,
) -> Terrain:
    """Plan every OSM bridge, keep the terrain out of the decks (clamping it under
    `deck_ceiling` if it still pokes in) and add the bridge pieces. Returns the terrain."""
    m, bopt = geo.model, options.bridge
    spans = bridge_spans(m)
    if not spans:
        return terrain
    ground = _ground_fn(terrain, road_height)
    plans = [plan_bridge(m, s0, s1, ground, bopt, k) for k, (s0, s1) in enumerate(spans)]
    ceiling = deck_ceiling(m, plans, terrain.points, bopt)
    terrain, sc.terrain_clamped = clamp_heights(terrain, ceiling, options.terrain)
    ground = _ground_fn(terrain, road_height)
    pieces, sc.plans = bridge_pieces(m, spans, ground, bopt)
    sc.pieces += pieces
    checks = [terrain_through_deck(m, p, ground, bopt) for p in sc.plans]
    sc.deck_intrusion_m = max((c.max_intrusion for c in checks), default=0.0)
    if sc.deck_intrusion_m > DECK_TOLERANCE_M:
        warnings.warn(f"terrain rises {sc.deck_intrusion_m:.2f} m into a bridge deck", stacklevel=2)
    return terrain


def railed_decks(plans: Sequence[BridgePlan]) -> list[tuple[float, float]]:
    """Station ranges of the bridge decks that have a railing (not at-grade culverts)."""
    return [(p.s0, p.s1) for p in plans if p.railing]


def _auto_rails(
    geo: TrackGeometry,
    options: BuildOptions,
    decks: Sequence[tuple[float, float]],
    erase: Sequence[BarrierSpan],
    sc: _Scenery,
) -> list[BarrierSpan]:
    """The automatic (warrant) guardrails, minus the user's `"kind": "none"` stretches."""
    features = options.roadside_features
    if features is None:
        try:
            features = fetch_roadside(
                _route_lonlat(geo), cache_dir=options.cache_dir, offline=options.offline
            )
        except _DATA_ERRORS as e:
            warnings.warn(
                f"OSM water and driveways unavailable ({_short(e)}); automatic guardrails "
                "use the lidar, bridges and the route's side roads only",
                stacklevel=2,
            )
            features = None
    m = geo.model
    inputs = warrant_inputs(
        m,
        dtm_height_fn(geo),
        geo.projection,
        geo.route,
        geo.way_id if len(geo.way_id) == len(m) else np.zeros(len(m), np.int64),
        geo.oneway,
        decks=decks,
        features=features,
        s_range=(geo.s_a, geo.s_b),
        options=options.warrants,
    )
    inputs.driving_side = geo.driving_side
    result = warrant_spans(inputs, options.warrants)
    sc.warrant_hazards = result.hazards
    spans: list[BarrierSpan] = []
    report: list[dict] = []
    for span, entry in zip(result.spans, result.report, strict=True):
        pieces = [(span.s0, span.s1)]
        for e in erase:
            if e.side == span.side:
                pieces = [
                    (a, b)
                    for p0, p1 in pieces
                    for a, b in ((p0, min(p1, e.s0)), (max(p0, e.s1), p1))
                    if b - a > 0.5
                ]
        for a, b in pieces:
            spans.append(dataclasses.replace(span, s0=a, s1=b))
            report.append(
                {**entry, "from_m": round(a - geo.s_a, 1), "to_m": round(b - geo.s_a, 1),
                 "length_m": round(b - a, 1)}
            )  # fmt: skip
    sc.warrant_report = report
    return spans


def _barriers(
    geo: TrackGeometry,
    ground: Callable[[np.ndarray], np.ndarray],
    options: BuildOptions,
    sc: _Scenery,
) -> None:
    """Roadside barriers: the user's guardrails.json, OSM ways, then (unless turned off) the
    automatic rails the design rules require (bridge approach rails only if asked for).
    Decks with a railing are skipped; culverts are plain road."""
    m = geo.model
    decks = railed_decks(sc.plans)
    osm_spans: list[BarrierSpan] = []
    if options.osm_barriers is not None:
        osm = list(options.osm_barriers)
    else:
        try:
            osm = fetch_barriers(
                _route_lonlat(geo), cache_dir=options.cache_dir, offline=options.offline
            )
        except _DATA_ERRORS as e:
            warnings.warn(f"OSM guardrails unavailable ({_short(e)}); none built", stacklevel=2)
            osm = []
    sc.osm_barriers = len(osm)
    if osm:
        osm_spans = match_barriers(osm, geo.projection, m.centre[:, [0, 2]], m.s)
        used = {sp.osm_id for sp in osm_spans}
        sc.osm_barrier_ways = [b for b in osm if b.id in used]
    user: list[BarrierSpan] = []
    erase: list[BarrierSpan] = []
    auto = options.auto_guardrails
    if options.guardrails_file is not None and Path(options.guardrails_file).is_file():
        path = Path(options.guardrails_file)
        # Read now: the file may live in the track folder that is cleared before writing.
        sc.user_bytes = path.read_bytes()
        parsed = load_user_file(
            path, s_origin=geo.s_a, reverse=options.route_reversed, length_m=geo.length_m
        )
        user, erase, auto = list(parsed.spans), list(parsed.erase), auto and parsed.auto
        sc.user_file = path
        if options.route_reversed:  # the copy sits next to the reversed route.json: say so
            data = json.loads(sc.user_bytes.decode("utf-8-sig"))
            if not isinstance(data, Mapping):
                data = {"guardrails": data}
            data = {**data, "reversed": not data.get("reversed", False)}
            sc.user_bytes = (json.dumps(data, indent=1) + "\n").encode("utf-8")
    sc.user_spans = len(user)
    warranted = _auto_rails(geo, options, decks, erase, sc) if auto else []
    approach = APPROACH_RAIL_M if options.bridge_approach_rails else 0.0
    sc.spans = plan_barriers(
        m, osm=osm_spans, user=user, bridges=decks, approach_m=approach, heuristic=warranted
    )
    sc.warrant_spans = sum(1 for sp in sc.spans if sp.source == "warrant")
    sc.pieces += barrier_pieces(m, sc.spans, ground, options.bridge.barrier, bridges=decks)


def _building_tiers(fps: Sequence[Footprint]) -> list[list[Footprint]]:
    """Footprints grouped by geometry priority (municipal, OSM, NRCan lidar, Microsoft)."""
    tiers: list[list[Footprint]] = [[], [], [], []]
    for f in fps:
        if f.source == "osm":
            tiers[1].append(f)
        elif f.source == "nrcan" or f.lidar:
            tiers[2].append(f)
        elif f.source == "microsoft":
            tiers[3].append(f)
        else:
            tiers[0].append(f)
    return tiers


def _fetch_building_sources(
    geo: TrackGeometry, options: BuildOptions
) -> tuple[list[list[Footprint]], str | None] | None:
    """Every enabled footprint source near the route (BUILDINGS_FETCH_M), as conflation
    tiers, and the OSM timestamp. A failing source is a warning; None when all failed."""
    ll = _route_lonlat(geo)
    pad = BUILDINGS_FETCH_M
    boxes = [(w, s, e, n) for s, w, n, e in route_boxes(ll, pad + 5.0)]
    tiers: list[list[Footprint]] = [[], [], [], []]
    failed: list[str] = []
    osm_base = None
    for name in options.municipal_buildings:
        src = MUNICIPAL_SOURCES.get(name)
        if src is None:
            warnings.warn(f"unknown municipal building source {name!r}", stacklevel=3)
            continue
        try:
            tiers[0] += fetch_municipal(
                src, boxes, cache_dir=options.cache_dir, offline=options.offline
            )
        except (SourceError, *_DATA_ERRORS) as e:
            warnings.warn(f"{src.title} unavailable ({_short(e)})", stacklevel=3)
    try:
        osm, osm_base = fetch_osm_buildings(
            ll, cache_dir=options.cache_dir, around_m=pad, offline=options.offline
        )
        tiers[1] = within_route(osm, ll, pad)
    except _DATA_ERRORS as e:
        failed.append(f"OSM: {_short(e)}")
    if options.nrcan_buildings:
        try:
            tiers[2] = within_route(
                fetch_nrcan(boxes, cache_dir=options.cache_dir, offline=options.offline), ll, pad
            )
        except (SourceError, *_DATA_ERRORS) as e:
            failed.append(f"NRCan: {_short(e)}")
    if options.microsoft_buildings:
        pad_lat = pad / M_PER_DEG
        pad_lon = pad_lat / max(np.cos(np.radians(float(ll[:, 1].mean()))), 0.1)
        bbox = (
            float(ll[:, 0].min() - pad_lon),
            float(ll[:, 1].min() - pad_lat),
            float(ll[:, 0].max() + pad_lon),
            float(ll[:, 1].max() + pad_lat),
        )
        try:
            ms = fetch_ms_buildings(bbox, cache_dir=options.cache_dir, offline=options.offline)
            tiers[3] = within_route(ms, ll, pad)
        except (*_DATA_ERRORS, ValueError) as e:
            failed.append(f"Microsoft: {_short(e)}")
    if failed:
        got = any(tiers)
        warnings.warn(
            "building footprints unavailable ("
            + "; ".join(failed)
            + ")"
            + ("; using the other sources" if got else "; none built"),
            stacklevel=3,
        )
        if not got:
            return None
    return tiers, osm_base


def _buildings(
    geo: TrackGeometry,
    ground: Callable[[np.ndarray], np.ndarray],
    corridor: shapely.Geometry,
    canopy: CanopyFn | None,
    options: BuildOptions,
    track_id: str,
    sc: _Scenery,
) -> None:
    osm_base = None
    if options.building_footprints is not None:
        tiers = _building_tiers(options.building_footprints)
    else:
        got = _fetch_building_sources(geo, options)
        if got is None:
            return
        tiers, osm_base = got
    try:
        fps, report = conflate(tiers, geo.projection, canopy)
    except _DATA_ERRORS as e:
        warnings.warn(f"lidar unavailable to check building footprints ({_short(e)})",
                      stacklevel=2)  # fmt: skip
        fps, report = conflate(tiers, geo.projection, None)
    found = {}
    for tier in tiers:
        for f in tier:
            found[f.source] = found.get(f.source, 0) + 1
    res = BuildingFootprints(
        fps,
        osm_count=report.kept.get("osm", 0),
        ms_count=report.kept.get("microsoft", 0),
        osm_base=osm_base,
        found=found,
        report=report,
        osm=list(tiers[1]),
    )
    sc.footprints = res
    roads = _building_roads(geo, options)

    landcover = _landcover_fn(geo, options)
    paved = sc.network.paved if sc.network is not None else None

    def place(c: CanopyFn | None) -> list[Building]:
        return place_buildings(
            res.footprints, geo.projection, geo.model, ground,
            corridor=corridor, canopy=c, seed=track_id, roads=roads, landcover=landcover,
            paved=paved, report=sc.building_report,
        )  # fmt: skip

    try:
        sc.buildings = place(canopy)
    except _DATA_ERRORS as e:
        warnings.warn(f"lidar unavailable for building heights ({_short(e)})", stacklevel=2)
        sc.buildings = place(None)
    if sc.buildings:
        lines = res.attributions_for({b.id for b in sc.buildings})
        sc.attributions += [a for a in lines if a not in sc.attributions]


def _landcover_fn(geo: TrackGeometry, options: BuildOptions) -> LandcoverFn:
    """NRCan land cover on AC XZ points (trees: kinds; buildings: farm or town)."""
    if options.landcover_source is not None:
        return options.landcover_source(geo.projection)
    return LandcoverSampler(geo.projection, cache_dir=options.cache_dir)


def _building_roads(geo: TrackGeometry, options: BuildOptions) -> list[np.ndarray]:
    """Public roads near the route (OSM ways joining it, from the cached roadside query) as
    XZ polylines, so houses on a side road face it. Empty when unavailable."""
    features = options.roadside_features
    if features is None:
        try:
            features = fetch_roadside(_route_lonlat(geo), cache_dir=options.cache_dir,
                                      offline=True)  # fmt: skip
        except _DATA_ERRORS:
            return []
    out = []
    for f in features.access:
        if f.tags.get("highway") == "service" or len(f.lonlat) < 2:
            continue
        e, n = geo.projection.to_enu(f.lonlat[:, 0], f.lonlat[:, 1])
        out.append(np.stack([e, -n], axis=1))
    return out


def _short(e: Exception) -> str:
    text = str(e).strip().splitlines()
    return text[0][:160] if text else type(e).__name__


TOWN_RADIUS_M = 150.0  # a point with TOWN_BUILDINGS town buildings this close is in a town
TOWN_BUILDINGS = 5


def _town_at(sc: _Scenery) -> Callable[[np.ndarray], bool] | None:
    """XZ -> True in a town: TOWN_BUILDINGS of the placed buildings in a town context
    (`geometry.buildings._town_context`) within TOWN_RADIUS_M. None without buildings."""
    town = [b.centroid for b in sc.buildings if b.town]
    if not sc.buildings:
        return None
    if not town:
        return lambda xz: False
    tree = cKDTree(np.array(town))

    def at(xz: np.ndarray) -> bool:
        p = np.asarray(xz, np.float64).reshape(-1)[:2]
        return len(tree.query_ball_point(p, TOWN_RADIUS_M)) >= TOWN_BUILDINGS

    return at


def _furniture(
    geo: TrackGeometry,
    ground: Callable[[np.ndarray], np.ndarray],
    options: BuildOptions,
    sc: _Scenery,
    street_items: Sequence[StreetItem],
) -> None:
    """Street furniture (`geometry.furniture`): fetch OSM nodes and Mapillary detections
    (each failing soft), plan every item with the other stages' `street_items`, and build the
    meshes and the route's markings."""
    ll = _route_lonlat(geo)
    if options.street_features is not None:
        osm: StreetFeatures | None = options.street_features
    else:
        try:
            osm = fetch_street_features(ll, cache_dir=options.cache_dir, offline=options.offline)
        except _DATA_ERRORS as e:
            warnings.warn(f"OSM signs and poles unavailable ({_short(e)})", stacklevel=2)
            osm = None
    sc.street_features = osm
    mly: list[MapFeature] = []
    coverage = options.mapillary_coverage
    if options.mapillary_features is not None:
        mly = list(options.mapillary_features)
    else:
        token = options.mapillary_token or token_from_env()
        try:
            mly = drop_stale(
                fetch_map_features(ll, token=token, cache_dir=options.cache_dir,
                                   offline=options.offline or not token)
            )  # fmt: skip
            if coverage is None:
                coverage = fetch_image_points(
                    ll, token=token, cache_dir=options.cache_dir,
                    offline=options.offline or not token,
                )  # fmt: skip
        except (MapillaryError, *_DATA_ERRORS) as e:
            if token:  # without a token the layer is simply off
                warnings.warn(f"Mapillary signs and poles unavailable ({_short(e)})", stacklevel=2)
            mly, coverage = [], None
    m = geo.model
    surface = DrivableSurface.build(
        m, oneway=geo.oneway, net=sc.network, driving_side=geo.driving_side
    )
    plan = plan_furniture(
        m, geo.projection, geo.route, ground, s_range=(geo.s_a, geo.s_b), way_id=geo.way_id,
        items=street_items, osm=osm, mapillary=mly, coverage=coverage, surface=surface,
        options=options.furniture_options, town_at=_town_at(sc),
    )  # fmt: skip
    sc.furniture = plan
    names = street_names(plan.items)
    sc.furniture_names = tuple(sorted(names, key=names.get))
    way_id = geo.way_id if len(geo.way_id) == len(m) else None
    tree = cKDTree(m.centre[:, [0, 2]])

    def speed_at(xz: np.ndarray) -> int | None:
        if way_id is None:
            return None
        wid = int(way_id[tree.query(np.asarray(xz, float).reshape(2))[1]])
        return parse_speed((geo.route.ways.get(wid) or {}).get("maxspeed"))

    sc.furniture_meshes = furniture_meshes(
        plan, m, s_range=(geo.s_a, geo.s_b), verge_m=options.verge_m, names=names,
        speed_at=speed_at, options=options.furniture_options,
    )  # fmt: skip
    for name, mesh in marking_meshes(plan.markings, m, options.furniture_options.cell_m):
        sc.pieces.append(ScenePiece(name, "paint_white", mesh, dict(DECAL_FLAGS)))
    if plan.mapillary_used and MAPILLARY_ATTRIBUTION not in sc.attributions:
        sc.attributions.append(MAPILLARY_ATTRIBUTION)


def _trees(
    geo: TrackGeometry,
    canopy: CanopyFn | None,
    options: BuildOptions,
    track_id: str,
    sc: _Scenery,
) -> None:
    density = max(0.0, float(options.scenery_density))
    if canopy is None or density <= 0.0:
        return
    vopt = options.vegetation
    vopt = dataclasses.replace(
        vopt,
        clearance_m=max(vopt.clearance_m, options.verge_m + TREE_WALL_MARGIN_M),
        max_trees=int(round(vopt.max_trees * density)),
        max_trees_per_km=vopt.max_trees_per_km * density,
    )
    landcover = _landcover_fn(geo, options)
    exclude = _tree_exclusion(geo, sc)
    try:
        sc.trees = place_trees(
            geo.model, canopy, landcover=landcover, exclude=exclude, seed=track_id, options=vopt
        )
    except _DATA_ERRORS as e:
        warnings.warn(f"lidar canopy unavailable ({_short(e)}); no trees", stacklevel=2)
        return
    if sc.footprints is not None and sc.footprints.footprints:
        sc.footprints_masked = True
        sc.attributions += [a for a in sc.footprints.attributions if a not in sc.attributions]
    if len(sc.trees) == 0 and region_of(
        np.array([geo.route.pins[0].snap_lon]), np.array([geo.route.pins[0].snap_lat])
    )[0] not in ("CA", "CA-border"):
        warnings.warn("no trees: lidar canopy heights exist only in Canada", stacklevel=2)
    sc.attributions += [a for a in sc.trees.attributions if a not in sc.attributions]


TREE_BUILDING_BUFFER_M = 2.0  # no tree trunk this close to any building footprint


def _footprint_polygon(fp: Footprint, projection: LocalProjection) -> shapely.Geometry | None:
    if len(fp.lonlat) < 4:
        return None
    e, n = projection.to_enu(fp.lonlat[:, 0], fp.lonlat[:, 1])
    return shapely.make_valid(shapely.Polygon(np.stack([e, -n], axis=1)))


def _tree_exclusion(geo: TrackGeometry, sc: _Scenery) -> shapely.Geometry | None:
    """Where no tree may stand: EVERY fetched building footprint, also the ones not drawn
    (in the wall corridor, too small, ...), and the drawn buildings, buffered 2 m. The lidar
    canopy sees roofs as tall flat "trees", so without this a house that is not drawn becomes
    a clump of trees. `place_trees` also masks the canopy there before looking for tree tops."""
    polys = [b.polygon for b in sc.buildings]
    if sc.footprints is not None:
        polys += [_footprint_polygon(fp, geo.projection) for fp in sc.footprints.footprints]
    if sc.network is not None and not sc.network.paved.is_empty:
        # Side roads, junctions, islands: trees stay out of their wall corridor too.
        verge = sc.network.stats.get("verge_m", 4.0)
        polys.append(
            sc.network.paved.buffer(float(verge) + TREE_WALL_MARGIN_M - TREE_BUILDING_BUFFER_M)
        )
    if sc.network is not None:  # no tree on a driveway (2 m clear, like buildings)
        polys += [d.polygon for d in sc.network.driveways]
    polys = [p for p in polys if p is not None and not p.is_empty]
    if sc.furniture is not None:  # no tree through a hydro pole or its wires' first metres
        polys += [shapely.Point(it.position[0], it.position[2]) for it in sc.furniture.items
                  if it.kind == POLE_UTILITY]  # fmt: skip
    if not polys:
        return None
    return shapely.union_all(shapely.buffer(np.array(polys), TREE_BUILDING_BUFFER_M))


def _conflation_stats(fps: BuildingFootprints | None) -> dict:
    rep = fps.report if fps is not None else None
    if rep is None:
        return {}
    return {
        "kept": dict(rep.kept),
        "same_building": rep.matched,
        "overlapping": rep.overlapping,
        "lidar_cut": rep.cut,
        "unconfirmed": rep.unconfirmed,
        "too_small": rep.small,
        "demolished": list(rep.demolished),
    }


def _scenery_stats(sc: _Scenery, geo: TrackGeometry, extra: Mapping[str, object]) -> dict:
    m = geo.model
    tree_stats: TreeStats = extra["trees"]  # type: ignore[assignment]
    bld_stats: BuildingStats = extra["buildings"]  # type: ignore[assignment]
    trees = sc.trees.counts() if sc.trees is not None else TreeSet.empty().counts()
    trees.update(meshes=tree_stats.meshes, vertices=tree_stats.vertices, built=tree_stats.trees)
    barriers: dict[str, float] = {}
    for span in sc.spans:
        key = f"{span.kind} ({span.source})"
        barriers[key] = barriers.get(key, 0.0) + span.length
    bridges = []
    for p in sc.plans:
        s = np.linspace(p.s0, p.s1, 50)
        c = np.stack([np.interp(s, m.s, m.centre[:, k]) for k in range(3)], axis=1)
        bridges.append(
            {
                "from_a_m": p.s0 - geo.s_a,
                "length_m": p.length,
                "piers": len(p.piers),
                "solid": p.solid,
                "railing": p.railing,
                "max_height_m": float(np.nanmax(c[:, 1] - sc.ground(c[:, [0, 2]]), initial=0.0)),
            }
        )
    return {
        "trees": trees,
        "buildings": {
            "buildings": bld_stats.buildings,
            "meshes": bld_stats.meshes,
            "vertices": bld_stats.vertices,
            "colliding": bld_stats.colliding,
            "height_source": dict(bld_stats.by_height_source),
            "roofs": dict(bld_stats.by_roof),
            "footprints_osm": sc.footprints.osm_count if sc.footprints else 0,
            "footprints_microsoft": sc.footprints.ms_count if sc.footprints else 0,
            "footprints_found": dict(sc.footprints.found) if sc.footprints else {},
            "conflation": _conflation_stats(sc.footprints),
            "categories": dict(bld_stats.by_category),
            "placement": dict(sc.building_report),
            "sources": dict(bld_stats.by_source),
            "doors": bld_stats.doors,
            "mesh_bytes": bld_stats.mesh_bytes,
        },
        "barriers_m": barriers,
        "osm_barriers": sc.osm_barriers,
        "user_spans": sc.user_spans,
        "warrant_spans": sc.warrant_spans,
        "bridges": bridges,
        "deck_intrusion_m": sc.deck_intrusion_m,
        "terrain_points_clamped": sc.terrain_clamped,
        "furniture": _furniture_stats(sc),
    }


def _furniture_stats(sc: _Scenery) -> dict:
    plan = sc.furniture
    if plan is None:
        return {}
    return {
        "items": len(plan.items),
        "by_kind": {k: dict(sorted(v.items())) for k, v in sorted(plan.counts.items())},
        "wire_spans": len(plan.wires),
        "markings": len(plan.markings),
        "mapillary_used": plan.mapillary_used,
        "mapillary_photo_coverage": round(plan.covered_fraction, 3),
        "meshes": len(sc.furniture_meshes),
        "route_controls": [c for c in plan.controls if c["control"]],
        "mapillary_driveways": plan.driveways,
        "clearance": plan.clearance,
    }


# --------------------------------------------------------------------------------------------
# Credits
# --------------------------------------------------------------------------------------------


def _merge_notices(notices: Sequence[str]) -> list[str]:
    """One line per credit: "label: text" notices with the same text (e.g. the HRDEM licence
    for the elevation and for the tree heights) become one "label; label: text" line."""
    order: list[str] = []
    labels: dict[str, list[str]] = {}
    for notice in notices:
        label, sep, body = notice.partition(": ")
        key = body if sep else notice
        if key not in labels:
            order.append(key)
            labels[key] = []
        if sep and label not in labels[key]:
            labels[key].append(label)
    return [f"{'; '.join(labels[k])}: {k}" if labels[k] else k for k in order]


def _licence(geo: TrackGeometry, sc: _Scenery | None = None) -> tuple[list[str], list[str]]:
    """LICENSE-DATA.txt keys (`DATA_SOURCES`) and extra notices, duplicates merged."""
    keys = ["osm"]
    extra: list[str] = []
    for name, info in sorted(geo.elevation.info.items()):
        key = _LICENCE_KEYS.get(name)
        if key and key in DATA_SOURCES:
            if key not in keys:
                keys.append(key)
        else:
            extra.append(f"Elevation ({info.title}): {info.attribution}")
    if sc is not None:
        extra += sc.attributions
        osm_furniture = sc.furniture is not None and bool(
            sc.furniture.osm_nodes or sc.furniture.osm_lines
        )
        if OSM_BUILDINGS_ATTRIBUTION in extra or sc.osm_barrier_ways or osm_furniture:
            # One OSM credit for the road, the buildings, the barriers and the furniture.
            extra = [a for a in extra if a != OSM_BUILDINGS_ATTRIBUTION]
            keys[keys.index("osm")] = "osm_scenery"
    return keys, _merge_notices(extra)


_BUILDING_CREDITS = {
    "osm": "OpenStreetMap",
    "nrcan": "NRCan lidar",
    "burlington": "City of Burlington",
    "milton": "Town of Milton",
    "oakville": "Town of Oakville",
    "microsoft": "Microsoft footprints",
}


def _building_credit(sc: _Scenery) -> str:
    """The sources of the drawn buildings, as "OpenStreetMap, NRCan lidar and ..."."""
    used: set[str] = set()
    fps = {f.id: f for f in sc.footprints.footprints} if sc.footprints is not None else {}
    for b in sc.buildings:
        f = fps.get(b.id)
        for sid in (b.id, *(f.also if f is not None else ())):
            head = sid.split(":", 1)[0]
            used.add("microsoft" if head == "ms" else head)
    names = [v for k, v in _BUILDING_CREDITS.items() if k in used]
    if len(names) <= 1:
        return names[0] if names else "OpenStreetMap"
    return ", ".join(names[:-1]) + " and " + names[-1]


def _osm_ref(footprint_id: str) -> tuple[str, int] | None:
    """("way" | "relation", id) of an OSM footprint id ("osm:w1", "osm:r2.0")."""
    head, _, rest = footprint_id.partition(":")
    if head != "osm" or rest[:1] not in ("w", "r"):
        return None
    return ("way" if rest[0] == "w" else "relation"), int(rest[1:].split(".")[0])


def _lonlat_list(lonlat: np.ndarray) -> list[list[float]]:
    return [[round(float(lon), 7), round(float(lat), 7)] for lon, lat in lonlat]


def _scenery_features(sc: _Scenery) -> list[dict]:
    """route_osm.geojson features for the OSM buildings (drawn, or used to keep trees off)
    and barrier ways the build used (ODbL: the OSM data in the track is listed with it)."""
    features = []
    drawn = {b.id for b in sc.buildings}
    kept = sc.footprints.footprints if sc.footprints is not None else []
    for f in kept:  # an OSM way merged into a drawn building (its tags were used)
        if f.id in drawn:
            drawn.update(f.also)
    fps = kept
    if sc.footprints is not None and sc.footprints.osm is not None:
        fps = sc.footprints.osm
    for fp in fps:
        ref = _osm_ref(fp.id)
        if ref is None or not (fp.id in drawn or sc.footprints_masked):
            continue
        kind, oid = ref
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [_lonlat_list(fp.lonlat)]},
                "properties": {
                    f"osm_{kind}_id": oid,
                    "osm_url": f"https://www.openstreetmap.org/{kind}/{oid}",
                    "use": "building" if fp.id in drawn else "tree mask",
                    "tags": dict(sorted(fp.tags.items())),
                },
            }
        )
    plan, feats = sc.furniture, sc.street_features
    if plan is not None and feats is not None:
        used = set(plan.osm_nodes)
        for n in feats.nodes:
            if n.id in used:
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": _lonlat_list(
                            np.array([[n.lon, n.lat]]))[0]},
                        "properties": {
                            "osm_node_id": n.id,
                            "osm_url": f"https://www.openstreetmap.org/node/{n.id}",
                            "use": "street furniture",
                            "tags": dict(sorted(n.tags.items())),
                        },
                    }
                )  # fmt: skip
        lines = set(plan.osm_lines)
        for w in feats.power_lines:
            if w.id in lines:
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": _lonlat_list(w.lonlat)},
                        "properties": {
                            "osm_way_id": w.id,
                            "osm_url": f"https://www.openstreetmap.org/way/{w.id}",
                            "use": "power line (poles)",
                            "tags": dict(sorted(w.tags.items())),
                        },
                    }
                )
    for b in sc.osm_barrier_ways:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": _lonlat_list(b.lonlat)},
                "properties": {
                    "osm_way_id": b.id,
                    "osm_url": f"https://www.openstreetmap.org/way/{b.id}",
                    "use": "barrier",
                    "tags": dict(sorted(b.tags.items())),
                },
            }
        )
    return features


def _geojson(geo: TrackGeometry, sc: _Scenery | None = None) -> str:
    route = geo.route
    features = []
    edges = route.edges
    start = 0
    for i in range(1, len(edges) + 1):
        if i < len(edges) and edges[i].way == edges[start].way:
            continue
        ids = [edges[start].u] + [e.v for e in edges[start:i]]
        coords = [[round(c, 7) for c in route.nodes[n]] for n in ids]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "osm_way_id": edges[start].way,
                    "osm_url": f"https://www.openstreetmap.org/way/{edges[start].way}",
                    "tags": dict(sorted(route.ways[edges[start].way].items())),
                },
            }
        )
        start = i
    doc = {
        "type": "FeatureCollection",
        "attribution": OSM_ATTRIBUTION,
        "license": "ODbL-1.0 (https://opendatacommons.org/licenses/odbl/1-0/)",
        "osm_base": route.osm_base,
        "pins": [{"lat": p.lat, "lon": p.lon} for p in route.pins],
        "features": features + (_scenery_features(sc) if sc is not None else []),
    }
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


# --------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------


def build_track(route: Route, out_dir: Path, options: BuildOptions | None = None) -> BuildResult:
    """Build `<out_dir>/<track_id>/` from a route. An existing folder of that id is replaced
    if drift-maps made it (or `options.force`); otherwise `TrackFolderError` is raised before
    anything is built."""
    options = options or BuildOptions()
    t0 = time.perf_counter()
    track_id = validate_track_id(options.track_id or make_track_id(options.name))
    out_dir = Path(out_dir)
    track_dir = out_dir / track_id
    check_track_folder(track_dir, force=options.force)  # fail before the slow part
    notes: list[str] = []

    def progress(step: str, fraction: float) -> None:
        if options.progress is not None:
            options.progress(step, fraction)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        progress("Getting elevation and shaping the road", 0.0)
        osm_net = None
        if options.network or (options.real_run_ends and not route.closed):
            osm_net = _fetch_network(route, options)
        ends: tuple[Continuation | None, Continuation | None] = (None, None)
        if osm_net is not None and options.real_run_ends and not route.closed:
            ends = route_continuations(
                route, osm_net.ways, run_up_length_m(options), options.run_off_m
            )
        geo = prepare_geometry(route, options, ends)
        street_items: list[StreetItem] = []
        net: RoadNetwork | None = None
        if osm_net is not None and options.network:
            progress("Building side roads, junctions and roundabouts", 0.3)
            net = _network(geo, osm_net, ends, options)
            geo.model = net.model  # heights moved onto roundabout planes
        m = geo.model
        progress("Building the road and terrain", 0.35)
        roads = build_road_meshes(m, chunk_m=options.chunk_m, keep=net.masks if net else None)
        terrain = build_terrain(
            m,
            roads.left_outer,
            roads.right_outer,
            _terrain_height_fn(geo),
            options.terrain,
            _terrain_extra(net),
        )
        corridor = corridor_polygon(m, m.outer + options.verge_m)
        fine_zone = None
        if net is not None and not net.paved.is_empty:
            fine_zone = corridor.buffer(1.0)  # the route's walls keep their 2 m spacing
            corridor = shapely.union(corridor, net.corridor(options.verge_m))
        lines = wall_lines(corridor, fine_zone=fine_zone)
        if net is not None:
            lines.rings += net.island_walls

        centres = m.centre if net is None else np.concatenate([m.centre, net.centre_points()])
        centre_tree = cKDTree(centres[:, [0, 2]])

        def road_height(xz: np.ndarray) -> np.ndarray:
            return centres[centre_tree.query(xz)[1], 1]

        # Scenery (scenery.md §7): bridges first (they may lower the terrain under a deck).
        progress("Adding bridges, walls and guardrails", 0.45)
        sc = _Scenery()
        terrain = _bridges(geo, terrain, options, road_height, sc)
        ground = _ground_fn(terrain, road_height)
        sc.ground = ground
        walls = wall_meshes(lines, ground, road_fn=road_height)
        if net is not None:
            sc.network = net
            sc.pieces += net.pieces
        if options.guardrails:
            _barriers(geo, ground, options, sc)
        canopy: CanopyFn | None = None
        if options.trees or options.buildings:
            if options.canopy_source is not None:
                canopy = options.canopy_source(geo.projection)
            else:
                canopy = CanopySampler(geo.projection, cache_dir=options.cache_dir)
        if options.buildings:
            progress("Placing buildings", 0.55)
            _buildings(geo, ground, corridor, canopy, options, track_id, sc)
        if net is not None:  # driveways need the placed houses
            roads = _driveways(net, m, ground, options, sc, roads)
            street_items += net.street_items(ground)
        if options.furniture:
            progress("Adding signs, poles and lights", 0.62)
            _furniture(geo, ground, options, sc, street_items)
        if options.trees:
            progress("Planting trees", 0.7)
            _trees(geo, canopy, options, track_id, sc)
        for line in getattr(canopy, "attributions", None) or []:
            if line not in sc.attributions:  # lidar used for building heights only
                sc.attributions.append(line)
        sc.street_items = street_items
        progress("Adding the start, timing gates and AI line", 0.82)
        dummies = _dummies(geo, options)
        ai, ai_slice = _ai_line(geo, options)
    for w in caught:
        notes.append(str(w.message))

    progress("Writing the track files", 0.87)
    ext = ".dds" if options.texture_format == "dds" else ".png"
    kn5, counts, kn5_extra = _assemble_kn5(
        track_id, roads, terrain, walls, dummies.nodes, ext, sc, ground
    )

    clear_track_folder(track_dir, force=options.force)
    data_dir, ui_dir = track_dir / "data", track_dir / "ui"
    track_dir.mkdir(parents=True)
    (track_dir / MARKER_FILE).write_text(MARKER_TEXT, encoding="utf-8", newline="\n")
    kn5_path = write_kn5(kn5, track_dir / f"{track_id}.kn5")
    write_ai_files(ai, track_dir)

    line = m.centre
    mean_width = float(2 * m.half_width.mean())
    write_surfaces_ini(data_dir / "surfaces.ini", SURFACES)
    render_map([line], mean_width, data_dir, track_dir, meters_per_px=1.0)
    cams = cameras_along(m.centre[ai_slice], 150.0, road_half_width_m=mean_width / 2)
    write_cameras_ini(data_dir / "cameras.ini", cams)
    write_lighting_ini(data_dir / "lighting.ini")
    write_groove_ini(data_dir / "groove.ini")
    write_crew_ini(data_dir / "crew.ini")

    region = geo.elevation.region
    country = options.country if options.country is not None else _COUNTRIES.get(region, "")
    keys, extra = _licence(geo, sc)
    pin_a = route.pins[0]
    dem_titles = ", ".join(geo.elevation.info[n].title for n in sorted(geo.elevation.info))
    scenery_text = ""
    if sc.trees is not None and len(sc.trees):
        scenery_text += " Trees placed from NRCan lidar."
    if sc.buildings:
        scenery_text += " Buildings from " + _building_credit(sc) + "."
    if sc.furniture is not None and sc.furniture.mapillary_used:
        scenery_text += " Signs and poles from Mapillary (mapillary.com), CC BY-SA 4.0."
    run = loop_direction(m.centre) if geo.closed else "A2B"
    what = (
        f"a {geo.length_m / 1000:.1f} km loop on real roads"
        if geo.closed
        else f"{geo.length_m / 1000:.1f} km A to B on a real road"
    )
    description = options.description or (
        f"{options.name}: {what}, generated by "
        f"{GENERATOR_NAME}. Road data (c) OpenStreetMap contributors, ODbL. "
        f"Elevation: {dem_titles}."
        f"{scenery_text} See LICENSE-DATA.txt. Not affiliated with Kunos Simulazioni."
    )
    # "circuit" lets Content Manager run multi-lap races (A->B tracks deliberately lack it).
    tags = ["circuit", "street", "drift-maps"] if geo.closed else ["A2B", "touge", "drift-maps"]
    ui = UiTrack(
        name=options.name,
        description=description,
        tags=tags,
        geotags=(round(pin_a.snap_lat, 6), round(pin_a.snap_lon, 6)),
        country=country,
        city=options.city,
        length_m=geo.length_m,
        width=mean_width,
        pitboxes=options.pits,
        run=run,
    )
    write_ui_track_json(ui_dir / "ui_track.json", ui)
    if not geo.closed:  # the "touge" tag's New Mode reads its gates and starts from here
        (data_dir / TOUGE_FILE).write_text(
            json.dumps(_touge_data(geo, dummies, track_id), indent=1) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    write_preview_png(ui_dir / "preview.png", [line], title=options.name)
    write_outline_png(ui_dir / "outline.png", [line])
    write_data_license(track_dir, keys, track_name=options.name, extra_notices=extra)
    (track_dir / "route_osm.geojson").write_text(_geojson(geo, sc), encoding="utf-8", newline="\n")
    route.save(track_dir / "route.json")
    if sc.user_file is not None:  # bytes read before the old folder was cleared
        (track_dir / sc.user_file.name).write_bytes(sc.user_bytes)
    if sc.warrant_report or sc.warrant_hazards:  # why each automatic guardrail is there
        report = {
            "about": "Automatic guardrails (docs/research/guardrails.md), metres from A as "
            "driven. 'spans': what the rules warrant and why; 'built': what was built after "
            "guardrails.json, OSM rails and bridge decks took their share.",
            "spans": sc.warrant_report,
            "built": [
                {
                    "side": sp.side,
                    "from_m": round(sp.s0 - geo.s_a, 1),
                    "to_m": round(sp.s1 - geo.s_a, 1),
                }
                for sp in sc.spans
                if sp.source == "warrant"
            ],  # fmt: skip
        }
        (data_dir / "guardrails_auto.json").write_text(
            json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n"
        )

    images: list[Path] = []
    if options.debug_png:
        images = render_debug_png(
            out_dir / "debug" / f"{track_id}.png", geo, roads, terrain, lines, ai, dummies,
            scenery=sc,
        )  # fmt: skip
        if sc.street_items:  # what the furniture layer gets, for checking placements
            items_path = out_dir / "debug" / f"{track_id}_street_items.json"
            items_path.write_text(
                json.dumps([dataclasses.asdict(it) for it in sc.street_items], indent=0) + "\n",
                encoding="utf-8",
                newline="\n",
            )

    z = m.centre[:, 1]
    grade = np.abs(np.diff(z) / np.diff(m.s))
    kn5_meshes = [n for n in kn5.iter_nodes() if isinstance(n, Mesh)]
    stats: dict[str, object] = {
        "length_m": geo.length_m,
        "closed": geo.closed,
        "run": run,
        "road_length_m": float(m.s[-1]),
        "elevation_source": geo.elevation.primary,
        "elevation_coverage": ", ".join(
            f"{n} {100 * f:.0f}%" for n, f in sorted(geo.elevation.coverage.items())
        ),
        "elevation_offsets_m": dict(geo.elevation.offsets),
        "elevation_resolution_m": geo.elevation.resolution_m(),
        "elevation_dsm": geo.elevation.any_dsm,
        "elevation_min_m": float(z.min()),
        "elevation_max_m": float(z.max()),
        "max_grade": float(grade.max()),
        "max_profile_deviation_m": float(np.nanmax(np.abs(z - geo.raw_z))),
        "profile_rms_vs_dem_m": float(
            np.sqrt(np.nanmean((z - geo.raw_z)[~(m.bridge | geo.tunnel)] ** 2))
        ),
        "kn5_bytes": kn5_path.stat().st_size,
        "mesh_count": len(kn5_meshes),
        "draw_meshes": sum(1 for n in kn5_meshes if n.is_renderable),
        "collision_meshes": sum(1 for n in kn5_meshes if n.name.startswith("1WALL")),
        "vertices": counts,
        "vertices_total": int(sum(counts.values())),
        "scenery": _scenery_stats(sc, geo, kn5_extra),
        "triangles_terrain": int(len(terrain.triangles)),
        "wall_length_m": lines.length,
        "network": dict(sc.network.stats) if sc.network is not None else {},
        "street_items": len(sc.street_items),
        "ai_points": ai.n_points,
        "build_s": time.perf_counter() - t0,
    }
    if stats["max_profile_deviation_m"] > 15.0:
        notes.append(
            f"road profile deviates {stats['max_profile_deviation_m']:.1f} m from the DEM "
            "somewhere (bad DEM, or a bridge/tunnel without tags?)"
        )
    if route.repeated_nodes():
        notes.append("the route passes the same junction twice (it crosses itself)")
    progress("Track built", 1.0)
    return BuildResult(track_dir, track_id, stats, notes, geo, images, sc)


# --------------------------------------------------------------------------------------------
# Debug render
# --------------------------------------------------------------------------------------------

_COLOURS = {
    "grass": np.array([86, 140, 62]),
    "dirt": np.array([150, 112, 72]),
    "far_grass": np.array([120, 150, 100]),
    "far_dirt": np.array([160, 135, 105]),
}


def render_debug_png(
    path: Path,
    geo: TrackGeometry,
    roads: RoadMeshes,
    terrain: Terrain,
    lines: WallLines,
    ai: AiLine,
    dummies: _Dummies,
    *,
    scenery: _Scenery | None = None,
    max_px: int = 2000,
) -> list[Path]:
    """Top-down debug images: `<path>` (whole terrain, north up), `<stem>_detail.png` (300 m
    around the tightest curve) and `<stem>_bridge<k>.png` (200 m around each bridge). Terrain
    coloured by material and hill-shaded; road dark grey, gravel tan, paint white/yellow, walls
    red, AI line blue, gates green/red, spawns cyan, distance from A every 250 m (black ticks
    on the driver's left). Scenery: trees as crown discs (green
    deciduous, dark green conifer, olive bush), buildings brick red, guardrails orange, bridge
    decks light grey with white parapets, brown abutments and black piers.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = geo.model
    pts = terrain.points
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    k = np.abs(np.gradient(np.unwrap(np.arctan2(m.forward[:, 2], m.forward[:, 0]))))
    i_tight = int(np.argmax(_box_smooth(k, 9)))
    c = m.centre[i_tight, [0, 2]]
    # Colour of every kept terrain triangle (material, hill-shaded), and of every Delaunay
    # simplex (white where the triangle was removed: the road hole, hull slivers).
    p = terrain.positions
    tri = terrain.triangles
    n = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    n /= np.linalg.norm(n, axis=1)[:, None]
    light = np.array([-0.5, 0.75, -0.45])
    shade = np.clip(n @ (light / np.linalg.norm(light)), 0.25, 1.0)[:, None]
    base = np.where(
        terrain.near[:, None],
        np.where(terrain.dirt[:, None], _COLOURS["dirt"], _COLOURS["grass"]),
        np.where(terrain.dirt[:, None], _COLOURS["far_dirt"], _COLOURS["far_grass"]),
    )
    simplex_rgb = np.full((len(terrain.kept) + 1, 3), 255.0)  # last row: outside the hull
    rgb = base * (0.35 + 0.75 * shade)
    sc = scenery or _Scenery()
    if sc.buildings or (sc.trees is not None and len(sc.trees)):
        rgb = 0.55 * rgb + 0.45 * 235.0  # paler ground so trees and buildings stand out
    simplex_rgb[np.flatnonzero(terrain.kept)] = rgb
    out = []
    views = [
        (path, (lo, hi)),
        (path.with_name(path.stem + "_detail.png"), (c - 150.0, c + 150.0)),
    ]
    for k, plan in enumerate(sc.plans):
        bc = m.at(0.5 * (plan.s0 + plan.s1))[0][[0, 2]]
        views.append((path.with_name(f"{path.stem}_bridge{k}.png"), (bc - 100.0, bc + 100.0)))
    if sc.network is not None:
        for k, rb in enumerate(sc.network.roundabouts):
            rc = np.asarray(rb.centre)
            views.append((path.with_name(f"{path.stem}_roundabout{k}.png"), (rc - 75.0, rc + 75.0)))
        on_route = [j for j in sc.network.junctions if j.route_s is not None]
        for k, j in enumerate(on_route[:4]):
            views.append(
                (path.with_name(f"{path.stem}_junction{k}.png"), (j.xz - 45.0, j.xz + 45.0))
            )
    if sc.furniture is not None:  # the first junctions where the route stops (signs, bars)
        stops = sorted({c["s"] for c in sc.furniture.controls if c["control"]})[:4]
        for k, s in enumerate(stops):
            jc = m.at(float(s))[0][[0, 2]]
            views.append((path.with_name(f"{path.stem}_stop{k}.png"), (jc - 45.0, jc + 45.0)))
    for target, (a, b) in views:
        scale = max(float(np.max(b - a)) / max_px, 0.15)
        w, h = (np.ceil((b - a) / scale).astype(int) + 1).tolist()
        xs = a[0] + (np.arange(w) + 0.5) * scale
        zs = a[1] + (np.arange(h) + 0.5) * scale
        gx, gz = np.meshgrid(xs, zs)
        simplex = terrain.delaunay.find_simplex(np.stack([gx.ravel(), gz.ravel()], axis=1))
        img = simplex_rgb[simplex]  # -1 picks the last row
        im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8).reshape(h, w, 3))
        dr = ImageDraw.Draw(im)

        def px(xyz: np.ndarray, a=a, scale=scale) -> list[tuple[float, float]]:
            xyz = np.asarray(xyz).reshape(-1, xyz.shape[-1])
            xz = xyz[:, [0, 2]] if xyz.shape[1] == 3 else xyz
            return [tuple(v) for v in ((xz - a) / scale).tolist()]

        def strip(left: np.ndarray, right: np.ndarray, colour, dr=dr) -> None:
            for i in range(len(left) - 1):
                dr.polygon(
                    px(np.stack([left[i], left[i + 1], right[i + 1], right[i]])), fill=colour
                )

        _draw_scenery_under(dr, px, a, b, scale, m, sc)
        hw = m.half_width
        if sc.network is not None:
            _draw_network(dr, px, a, b, sc.network)
            shown = np.concatenate([~sc.network.masks.hidden, [True]])
        else:
            shown = np.ones(len(m), bool)
        for run in _true_runs(shown):
            strip(roads.left_outer[run], roads.right_outer[run], (196, 176, 140))
            strip((m.centre + hw[:, None] * m.left)[run], (m.centre - hw[:, None] * m.left)[run],
                  (70, 70, 74))  # fmt: skip
        edge = hw - 0.2
        for side in (1.0, -1.0):
            for run in _true_runs(shown):
                dr.line(px((m.centre + side * edge[:, None] * m.left)[run]), fill=(250, 250, 250))
        centre_colour = (250, 200, 30) if m.double_yellow else (250, 250, 250)
        on = np.flatnonzero(m.centre_line & shown)
        for run in np.split(on, np.flatnonzero(np.diff(on) > 1) + 1):
            if len(run) > 1:
                dr.line(px(m.centre[run]), fill=centre_colour, width=1)
        _draw_scenery_over(dr, px, m, sc)
        for ring in lines.rings:
            dr.line(px(ring), fill=(220, 30, 30), width=2)
        dr.line(px(ai.positions.astype(np.float64)), fill=(40, 90, 230), width=2)
        for name, node in ((d.name, d) for d in dummies.nodes):
            pos = node.transform[3, :3].astype(np.float64)
            (x, y), r = px(pos[None, :])[0], 3
            colour = (40, 200, 230)
            if name.startswith("AC_AB_FINISH") or (geo.closed and name.startswith("AC_TIME_0")):
                colour = (230, 40, 40)  # finish (a loop's start/finish line)
            elif name.startswith("AC_AB_START") or name.startswith("AC_TIME"):
                colour = (30, 200, 60)
            dr.ellipse((x - r, y - r, x + r, y + r), fill=colour)
        # Distance from A every 250 m (to place user feedback such as guardrails.json spans).
        for d in np.arange(0.0, geo.length_m + 1.0, 250.0):
            pos, _, left = m.at(geo.s_a + float(d))
            reach = float(np.interp(geo.s_a + d, m.s, m.outer)) + 6.0
            (x0, y0), (x1, y1) = px(np.stack([pos + reach * left, pos + (reach + 4) * left]))
            dr.line((x0, y0, x1, y1), fill=(0, 0, 0), width=2)
            dr.text((x1 + 2, y1 - 5), f"{d / 1000:.2f} km", fill=(0, 0, 0))
        _legend(dr, sc)
        im.save(target, format="PNG", optimize=False)
        out.append(target)
    return out


_TREE_RGB = {CONIFER: (18, 72, 46), BUSH: (122, 140, 48)}  # anything else: deciduous
_DECIDUOUS_RGB = (46, 122, 40)
_BUILDING_RGB = (165, 62, 48)
_RAIL_RGB = (255, 150, 0)
_AUTO_RAIL_RGB = (255, 235, 0)  # automatic (warrant) guardrails
_PARAPET_RGB = (250, 250, 250)
_ABUTMENT_RGB = (125, 95, 60)


def _true_runs(mask: np.ndarray) -> list[slice]:
    """Slices of the True runs of a station mask (with one station of overlap, for strips)."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1) + 1
    return [slice(int(r[0]), int(r[-1]) + 2) for r in np.split(idx, cuts) if len(r)]


_NETWORK_RGB = {
    "gravel": (196, 176, 140),
    "asphalt": (78, 78, 82),
    "islands": (120, 170, 90),
    "driveways": (170, 150, 120),
}
_ITEM_RGB = {
    "sign:stop": (230, 20, 20),
    "sign:yield": (255, 140, 0),
    "sign:keep_right": (255, 255, 255),
    "sign:roundabout_ahead": (255, 220, 0),
    "sign:chevron_right": (255, 220, 0),
    "sign:chevron_left": (255, 220, 0),
    "mailbox": (40, 40, 200),
}


def _union_all(geoms: Sequence[shapely.Geometry]) -> shapely.Geometry:
    return shapely.union_all(np.array(geoms, dtype=object)) if geoms else shapely.Polygon()


def _draw_network(dr, px, a: np.ndarray, b: np.ndarray, net: RoadNetwork) -> None:
    """Side roads, junction mouths, roundabouts (gravel tan, asphalt grey, islands green),
    their painted markings and the planned street items (coloured dots)."""
    view = shapely.box(a[0] - 5, a[1] - 5, b[0] + 5, b[1] + 5)
    polygons = dict(net.polygons, driveways=_union_all([d.polygon for d in net.driveways]))
    for key in ("gravel", "asphalt", "islands", "driveways"):
        geom = polygons.get(key)
        if geom is None or geom.is_empty:
            continue
        for poly in shapely.get_parts(geom.intersection(view)):
            if isinstance(poly, shapely.Polygon) and poly.area > 0.01:
                dr.polygon(px(np.asarray(poly.exterior.coords)), fill=_NETWORK_RGB[key])
                for hole in poly.interiors:
                    dr.polygon(px(np.asarray(hole.coords)), fill=(150, 150, 150))
    for piece in net.pieces:
        if piece.material not in ("paint_white", "paint_yellow"):
            continue
        colour = (250, 250, 250) if piece.material == "paint_white" else (250, 200, 30)
        pos = piece.mesh.positions
        inside = np.all((pos[:, [0, 2]] >= a - 5) & (pos[:, [0, 2]] <= b + 5), axis=1)
        if not inside.any():
            continue
        for t in piece.mesh.triangles:
            if inside[t].any():
                dr.polygon(px(pos[t]), fill=colour)
    for it in net.planned_items:
        x, y = px(np.array([it.xz]))[0]
        dr.ellipse((x - 3, y - 3, x + 3, y + 3), fill=_ITEM_RGB.get(it.kind, (255, 0, 255)),
                   outline=(0, 0, 0))  # fmt: skip


def _span_points(m: RoadModel, lateral: np.ndarray, s0: float, s1: float) -> np.ndarray:
    pts = road_columns(m, lateral)
    sel = (m.s >= s0) & (m.s <= s1)
    return pts[sel]


def _draw_scenery_under(dr, px, a, b, scale: float, m: RoadModel, sc: _Scenery) -> None:
    """Buildings, trees and bridge decks, drawn before the road."""
    for bld in sc.buildings:
        if np.any(bld.ring.max(axis=0) < a) or np.any(bld.ring.min(axis=0) > b):
            continue
        dr.polygon(px(bld.ring), fill=_BUILDING_RGB, outline=(60, 20, 15))
    trees = sc.trees
    if trees is not None and len(trees):
        inside = np.all((trees.xz > a - 10.0) & (trees.xz < b + 10.0), axis=1)
        for i in np.flatnonzero(inside):
            (x, y), r = px(trees.xz[i])[0], max(0.8, 0.8 * float(trees.radius[i]) / scale)
            colour = _TREE_RGB.get(int(trees.kind[i]), _DECIDUOUS_RGB)
            dr.ellipse((x - r, y - r, x + r, y + r), fill=colour)
    if sc.plans:
        half = deck_half_width(m)
        lc, rc = road_columns(m, half), road_columns(m, -half)
        for plan in sc.plans:
            sel = (m.s >= plan.s0 - 4.0) & (m.s <= plan.s1 + 4.0)
            ring = np.concatenate([lc[sel], rc[sel][::-1]])
            if len(ring) >= 3:
                dr.polygon(px(ring), fill=(205, 205, 205), outline=(90, 90, 90))


def _draw_scenery_over(dr, px, m: RoadModel, sc: _Scenery) -> None:
    """Guardrails, bridge parapets, abutments and piers, drawn over the road."""
    edge = m.outer
    for span in sc.spans:
        sign = 1.0 if span.side == "left" else -1.0
        pts = _span_points(m, sign * edge, span.s0, span.s1)
        if len(pts) >= 2:
            colour = _AUTO_RAIL_RGB if span.source == "warrant" else _RAIL_RGB
            dr.line(px(pts), fill=colour, width=3)
    half = deck_half_width(m)
    for plan in sc.plans:
        for sign in (1.0, -1.0) if plan.railing else ():
            pts = _span_points(m, sign * (edge + 0.15), plan.s0, plan.s1)
            if len(pts) >= 2:
                dr.line(px(pts), fill=_PARAPET_RGB, width=3)
        blocks = [(plan.s0 - 4.0, plan.s1 + 4.0)] if plan.solid else [
            (plan.s0 - 4.0, plan.face0), (plan.face1, plan.s1 + 4.0)
        ]  # fmt: skip
        for s0, s1 in blocks:
            lc = _span_points(m, half + 0.4, s0, s1)
            rc = _span_points(m, -(half + 0.4), s0, s1)
            if len(lc) >= 2:
                dr.polygon(px(np.concatenate([lc, rc[::-1]])), outline=_ABUTMENT_RGB, width=3)
        for sp in plan.piers:
            pos, fwd, left = m.at(sp)
            w = float(np.interp(sp, m.s, half))
            quad = np.array(
                [pos + w * left + 0.6 * fwd, pos + w * left - 0.6 * fwd,
                 pos - w * left - 0.6 * fwd, pos - w * left + 0.6 * fwd]
            )  # fmt: skip
            dr.polygon(px(quad), fill=(15, 15, 15))
    _draw_furniture(dr, px, sc)


_FURNITURE_RGB = {
    "sign:stop": (220, 20, 30),
    "sign:yield": (255, 120, 120),
    "sign:speed_limit": (255, 255, 255),
    "sign:": (250, 210, 0),  # other signs (warning yellow)
    "pole:": (120, 80, 40),
    "light:": (255, 255, 160),
    "signal:": (0, 220, 120),
    "": (60, 160, 255),  # hydrants, mailboxes, ...
}


def _furniture_rgb(kind: str) -> tuple[int, int, int]:
    for prefix, rgb in _FURNITURE_RGB.items():
        if kind.startswith(prefix):
            return rgb
    return _FURNITURE_RGB[""]


def _draw_furniture(dr, px, sc: _Scenery) -> None:
    """Wires (thin dark lines), then every street item: a dot in its colour with a tick
    towards where it faces, outlined black (rule items: grey outline)."""
    plan = sc.furniture
    if plan is None:
        return
    for a, b in plan.wires:
        dr.line(px(np.stack([a, b])), fill=(40, 40, 40), width=1)
    for mesh in (p.mesh for p in sc.pieces if p.name.startswith("marks_")):
        for tri in mesh.triangles:
            dr.polygon(px(mesh.positions[tri]), fill=(255, 255, 255))
    for it in sorted(plan.items, key=lambda it: it.kind != "sign:all_way"):  # tabs below
        p = np.array([[it.position[0], it.position[2]]])
        (x, y) = px(p)[0]
        (fx, fy) = px(p + 2.5 * np.array([it.facing]))[0]
        r = 3 if it.kind.startswith("pole:") else 4
        dr.line((x, y, fx, fy), fill=(0, 0, 0), width=1)
        outline = (130, 130, 130) if it.source == "rule" else (0, 0, 0)
        dr.ellipse((x - r, y - r, x + r, y + r), fill=_furniture_rgb(it.kind), outline=outline)


def _legend(dr, sc: _Scenery) -> None:
    items = [
        ("deciduous tree", _DECIDUOUS_RGB),
        ("conifer", _TREE_RGB[CONIFER]),
        ("bush", _TREE_RGB[BUSH]),
        ("building", _BUILDING_RGB),
        ("guardrail", _RAIL_RGB),
        ("guardrail (auto)", _AUTO_RAIL_RGB),
        ("bridge parapet", _PARAPET_RGB),
        ("abutment", _ABUTMENT_RGB),
        ("pier", (15, 15, 15)),
        ("hidden wall", (220, 30, 30)),
    ]
    if sc.furniture is not None and sc.furniture.items:
        items += [
            ("stop sign", _FURNITURE_RGB["sign:stop"]),
            ("speed limit", _FURNITURE_RGB["sign:speed_limit"]),
            ("other sign", _FURNITURE_RGB["sign:"]),
            ("hydro pole", _FURNITURE_RGB["pole:"]),
            ("street light", _FURNITURE_RGB["light:"]),
        ]
    dr.rectangle((4, 4, 140, 8 + 13 * len(items)), fill=(255, 255, 255), outline=(0, 0, 0))
    for k, (label, colour) in enumerate(items):
        y = 8 + 13 * k
        dr.rectangle((8, y + 1, 18, y + 10), fill=colour, outline=(0, 0, 0))
        dr.text((24, y), label, fill=(0, 0, 0))
