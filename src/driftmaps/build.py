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
   coverage gap cannot leave a step. Heights are relative to the road at A (A sits at Y = 0).
4. **Road, terrain, walls** (`geometry.road`, `.terrain`, `.walls`).
5. **Dummies and AI**: spawns 1 m above the road (M0 verified), `AC_START_0/1` side by side
   behind A, `AC_PIT_n` in pairs behind them, `AC_HOTLAP_START_0` further back, A->B gates with
   `_L` on the driver's left, optional `AC_TIME_n` splits. The AI line follows the centre of
   the driving lane (right in Canada/US/France, left in Japan/UK) from the start of the run-up
   to `ai_past_b_m` past B; the run-off road ends in the corridor's end wall.
6. **Files** (`ac.track_files`): surfaces (ROAD, GRASS, GRAVEL, DIRT), map, cameras, UI, credits
   (`LICENSE-DATA.txt`, `route_osm.geojson`, `route.json`).

`build_track(route, out_dir, options)` writes `<out_dir>/<track_id>/` and returns a
`BuildResult` with the statistics the CLI prints. Same route + same DEM cache -> identical
bytes (no clocks or random numbers in the output).
"""

from __future__ import annotations

import functools
import json
import shutil
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from driftmaps.ac.ai_line import AiLine, build_ai_line
from driftmaps.ac.coords import WORLD_UP
from driftmaps.ac.kn5 import write_kn5
from driftmaps.ac.model import Dummy, Kn5, Texture
from driftmaps.ac.scene import (
    DECAL_FLAGS,
    HIDDEN_WALL_FLAGS,
    Scene,
    gate_dummies,
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
from driftmaps.elevation import road_profile, sample_elevation
from driftmaps.elevation.sources import ElevationSource, SourceInfo, default_sources, region_of
from driftmaps.geo.attributes import MAX_WIDTH_M, MIN_DRIVE_WIDTH_M, region_for
from driftmaps.geo.overpass import OSM_ATTRIBUTION
from driftmaps.geo.project import LocalProjection
from driftmaps.geo.router import Route
from driftmaps.geometry.primitives import MeshData, split
from driftmaps.geometry.road import (
    RoadMeshes,
    RoadModel,
    build_road_meshes,
    extend_straight,
    fillet_corners,
    simplify,
    smooth_centreline,
)
from driftmaps.geometry.terrain import Terrain, TerrainOptions, build_terrain, tile_keys
from driftmaps.geometry.walls import WallLines, corridor_polygon, wall_lines, wall_meshes

__all__ = [
    "BuildOptions",
    "BuildResult",
    "TrackGeometry",
    "build_track",
    "prepare_geometry",
    "render_debug_png",
]

SPAWN_HEIGHT_M = 1.0  # M0 verified (dm_t1, dm_t12): cars drop onto the road
GATE_EXTRA_M = 5.0  # gate dummies this far beyond the asphalt edge (ac_format.md §2.3)
GATE_HEIGHT_M = 1.0
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

    name: str = "Drift Maps track"
    track_id: str | None = None
    description: str = ""
    city: str = ""
    country: str | None = None  # None: from the elevation region (Canada, US, France)
    cache_dir: Path | None = None
    run_up_m: float = 60.0
    run_off_m: float = 130.0
    ai_past_b_m: float = 100.0
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


@dataclass
class BuildResult:
    track_dir: Path
    track_id: str
    stats: dict[str, object]
    warnings: list[str]
    geometry: TrackGeometry | None = field(default=None, repr=False)
    debug_images: list[Path] = field(default_factory=list)

    def summary(self) -> str:
        s = self.stats
        lines = [
            f"track    {self.track_dir}",
            f"length   {s['length_m'] / 1000:.2f} km A->B ({s['road_length_m'] / 1000:.2f} km "
            "of road incl. run-up/run-off)",
            f"climb    {s['elevation_min_m']:.1f} .. {s['elevation_max_m']:.1f} m, max grade "
            f"{100 * s['max_grade']:.1f}%",
            f"DEM      {s['elevation_source']} (road coverage {s['elevation_coverage']})",
            f"KN5      {s['kn5_bytes'] / 1e6:.1f} MB, {s['mesh_count']} meshes, vertices "
            + ", ".join(f"{k} {v:,}" for k, v in s["vertices"].items()),
            f"time     {s['build_s']:.1f} s",
        ]
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

    def _offset(self, name: str) -> float:
        if name == self.primary or not name:
            return 0.0
        if name not in self.offsets:
            lon, lat, z = self._ref
            other = self._sample(lon, lat, chain=[name]).heights
            ok = np.isfinite(other)
            self.offsets[name] = float(np.median(z[ok] - other[ok])) if ok.any() else 0.0
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
        sel = np.flatnonzero(smp.source == self.primary)
        sel = sel[np.linspace(0, len(sel) - 1, min(len(sel), 2000)).astype(int)]
        self._ref = (lon.ravel()[sel], lat.ravel()[sel], smp.heights[sel])
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

    @property
    def length_m(self) -> float:
        return self.s_b - self.s_a


def _box_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values.astype(float)
    pad = window // 2
    padded = np.pad(values.astype(float), pad, mode="edge")
    kernel = np.ones(2 * pad + 1) / (2 * pad + 1)
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


def prepare_geometry(route: Route, options: BuildOptions) -> TrackGeometry:
    """Steps 2-3 of the pipeline: centreline, attributes per station and the height profile."""
    pin_a = route.pins[0]
    proj = LocalProjection(pin_a.snap_lat, pin_a.snap_lon)
    nodes = route.polyline(step_m=1e9)
    attrs = route.polyline(step_m=1.0)
    en = proj.to_xy(nodes.lonlat)
    xz = np.stack([en[:, 0], -en[:, 1]], axis=1)
    raw_length = float(nodes.s[-1])

    xz_f, p_f = fillet_corners(*simplify(xz, nodes.s, 0.3))
    xz_e, p_e = extend_straight(xz_f, p_f, options.run_up_m, options.run_off_m)
    xz_s, param = smooth_centreline(
        xz_e,
        p_e,
        max_deviation_m=options.max_deviation_m,
        cutoff_m=options.smoothing_cutoff_m,
    )
    param = np.maximum.accumulate(param)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xz_s, axis=0), axis=1))])
    s_a = float(np.interp(0.0, param, s))
    s_b = float(np.interp(raw_length, param, s))

    # Attributes per station: the OSM polyline point at that distance (clamped at the ends).
    j = np.clip(
        np.searchsorted(attrs.s, np.clip(param, 0.0, raw_length), side="right") - 1,
        0,
        len(attrs) - 1,
    )
    width = np.clip(attrs.width_m[j], MIN_DRIVE_WIDTH_M, MAX_WIDTH_M)
    width = _box_smooth(width, int(WIDTH_RAMP_M))
    lanes = attrs.lanes[j]
    oneway = attrs.oneway[j]
    bridge = attrs.bridge[j] & (param > 0) & (param < raw_length)
    tunnel = attrs.tunnel[j] & (param > 0) & (param < raw_length)
    region = region_for(pin_a.snap_lat, pin_a.snap_lon)

    # Elevation: five samples across the asphalt at every station.
    tangent = np.gradient(xz_s, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    left = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)  # left = up x forward, in XZ
    fr = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    lat_xz = xz_s[:, None, :] + (fr[None, :, None] * width[:, None, None]) * left[:, None, :]
    lon_s, lat_s = proj.to_lonlat(lat_xz[..., 0], -lat_xz[..., 1])
    elev = _Elevation(options, pin_a.snap_lon, pin_a.snap_lat)
    z_samples = elev.road(np.asarray(lon_s), np.asarray(lat_s))
    weights = np.where(_pad_mask(bridge | tunnel, s, BRIDGE_WEIGHT_PAD_M), 0.0, 1.0)
    z = road_profile(
        s,
        z_samples,
        weights,
        source_resolution_m=elev.resolution_m(),
        max_grade=options.max_grade,
        max_vertical_curvature=options.max_vertical_curvature,
    )
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
    )


def _terrain_height_fn(geo: TrackGeometry):
    def fn(xz: np.ndarray, far: bool) -> np.ndarray:
        lon, lat = geo.projection.to_lonlat(xz[:, 0], -xz[:, 1])
        return geo.elevation.terrain(np.asarray(lon), np.asarray(lat), far) - geo.z_ref

    return fn


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
    return _box_smooth(raw, int(WIDTH_RAMP_M))


@dataclass
class _Dummies:
    nodes: list[Dummy]
    spawns: dict[str, tuple[float, float]]  # name -> (s, lateral)
    gates: dict[str, float]  # prefix -> s


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
        """(driving lane, other lane) lateral offsets; on a one-way/one-lane stretch the two
        cars still sit side by side, a quarter of the asphalt width either side."""
        drive = m.interp(offs, s)
        if abs(drive) < 0.25:
            quarter = m.interp(m.half_width, s) / 2.0
            sign = -1.0 if geo.driving_side == "right" else 1.0
            return sign * quarter, -sign * quarter
        return drive, -drive

    # Grid: AC_START_0/1 side by side 6 m behind A, pits in pairs every 7 m behind them,
    # the hotlap start 4 m behind the last pit row.
    first = geo.s_a - 6.0
    rows = (options.pits + 1) // 2
    hot_s = first - 8.0 - 7.0 * rows - 4.0
    if hot_s < 3.0:
        raise ValueError("run_up_m is too short for the spawn grid; increase it or use fewer pits")
    d, o = lanes_at(first)
    spawn("AC_START_0", first, d)
    spawn("AC_START_1", first, o)
    for n in range(options.pits):
        s = first - 8.0 - 7.0 * (n // 2)
        d, o = lanes_at(s)
        spawn(f"AC_PIT_{n}", s, d if n % 2 == 0 else o)
    spawn("AC_HOTLAP_START_0", hot_s, lanes_at(hot_s)[0])

    gates: dict[str, float] = {"AC_AB_START": geo.s_a, "AC_AB_FINISH": geo.s_b}
    for k in range(options.splits):
        gates[f"AC_TIME_{k}"] = geo.s_a + (k + 1) * geo.length_m / (options.splits + 1)
    for prefix, s in gates.items():
        pos, fwd, left = m.at(s)
        reach = m.interp(m.half_width, s) + GATE_EXTRA_M
        nodes += gate_dummies(prefix, pos, fwd, left, reach, GATE_HEIGHT_M)
    return _Dummies(nodes, spawns, gates)


def _ai_line(geo: TrackGeometry, options: BuildOptions) -> tuple[AiLine, slice]:
    m = geo.model
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
}


@functools.cache
def _texture_data(name: str) -> bytes:
    stem = name.rsplit(".", 1)[0]
    return kn5_texture(name, _IMAGES[stem](), "BC1", mipmaps=True).data


def _texture(name: str) -> Texture:
    return Texture(name=name, data=_texture_data(name))


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


def _assemble_kn5(
    track_id: str,
    roads: RoadMeshes,
    terrain: Terrain,
    walls: list[MeshData],
    dummies: list[Dummy],
    ext: str,
) -> tuple[Kn5, dict[str, int]]:
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
    scene.add(*dummies)
    return scene.kn5(), counts


# --------------------------------------------------------------------------------------------
# Credits
# --------------------------------------------------------------------------------------------


def _licence(geo: TrackGeometry) -> tuple[list[str], list[str]]:
    keys = ["osm"]
    extra: list[str] = []
    for name, info in sorted(geo.elevation.info.items()):
        key = _LICENCE_KEYS.get(name)
        if key and key in DATA_SOURCES:
            if key not in keys:
                keys.append(key)
        else:
            notice = f"Elevation ({info.title}): {info.attribution}"
            if notice not in extra:
                extra.append(notice)
    return keys, extra


def _geojson(geo: TrackGeometry) -> str:
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
        "features": features,
    }
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


# --------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------


def build_track(route: Route, out_dir: Path, options: BuildOptions | None = None) -> BuildResult:
    """Build `<out_dir>/<track_id>/` from a route. An existing folder of that id is replaced."""
    options = options or BuildOptions()
    t0 = time.perf_counter()
    track_id = validate_track_id(options.track_id or make_track_id(options.name))
    notes: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        geo = prepare_geometry(route, options)
        m = geo.model
        roads = build_road_meshes(m, chunk_m=options.chunk_m)
        terrain = build_terrain(
            m, roads.left_outer, roads.right_outer, _terrain_height_fn(geo), options.terrain
        )
        corridor = corridor_polygon(m, m.outer + options.verge_m)
        lines = wall_lines(corridor)

        def ground(xz: np.ndarray) -> np.ndarray:
            h = terrain.height_at(xz)
            bad = ~np.isfinite(h)
            if bad.any():
                _, i = cKDTree(m.centre[:, [0, 2]]).query(xz[bad])
                h[bad] = m.centre[i, 1]
            return h

        walls = wall_meshes(lines, ground)
        dummies = _dummies(geo, options)
        ai, ai_slice = _ai_line(geo, options)
    for w in caught:
        notes.append(str(w.message))

    ext = ".dds" if options.texture_format == "dds" else ".png"
    kn5, counts = _assemble_kn5(track_id, roads, terrain, walls, dummies.nodes, ext)

    out_dir = Path(out_dir)
    track_dir = out_dir / track_id
    if track_dir.exists():
        shutil.rmtree(track_dir)
    data_dir, ui_dir = track_dir / "data", track_dir / "ui"
    track_dir.mkdir(parents=True)
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
    keys, extra = _licence(geo)
    pin_a = route.pins[0]
    dem_titles = ", ".join(geo.elevation.info[n].title for n in sorted(geo.elevation.info))
    description = options.description or (
        f"{options.name}: {geo.length_m / 1000:.1f} km A to B on a real road, generated by "
        f"drift-maps. Road data (c) OpenStreetMap contributors, ODbL. Elevation: {dem_titles}. "
        "See LICENSE-DATA.txt. Not affiliated with Kunos Simulazioni."
    )
    ui = UiTrack(
        name=options.name,
        description=description,
        tags=["A2B", "touge", "drift-maps"],
        geotags=(round(pin_a.snap_lat, 6), round(pin_a.snap_lon, 6)),
        country=country,
        city=options.city,
        length_m=geo.length_m,
        width=mean_width,
        pitboxes=options.pits,
        run="A2B",
    )
    write_ui_track_json(ui_dir / "ui_track.json", ui)
    write_preview_png(ui_dir / "preview.png", [line], title=options.name)
    write_outline_png(ui_dir / "outline.png", [line])
    write_data_license(track_dir, keys, track_name=options.name, extra_notices=extra)
    (track_dir / "route_osm.geojson").write_text(_geojson(geo), encoding="utf-8", newline="\n")
    route.save(track_dir / "route.json")

    images: list[Path] = []
    if options.debug_png:
        images = render_debug_png(
            out_dir / "debug" / f"{track_id}.png", geo, roads, terrain, lines, ai, dummies
        )

    z = m.centre[:, 1]
    grade = np.abs(np.diff(z) / np.diff(m.s))
    stats: dict[str, object] = {
        "length_m": geo.length_m,
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
        "mesh_count": sum(1 for n in kn5.iter_nodes() if not isinstance(n, Dummy)),
        "vertices": counts,
        "triangles_terrain": int(len(terrain.triangles)),
        "wall_length_m": lines.length,
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
    return BuildResult(track_dir, track_id, stats, notes, geo, images)


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
    max_px: int = 2000,
) -> list[Path]:
    """Top-down debug images: `<path>` (whole terrain, north up) and `<stem>_detail.png`
    (300 m around the tightest curve). Terrain coloured by material and hill-shaded; road dark
    grey, gravel tan, paint white/yellow, walls red, AI line blue, gates green/red, spawns cyan.
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
    simplex_rgb[np.flatnonzero(terrain.kept)] = base * (0.35 + 0.75 * shade)
    out = []
    for target, (a, b) in (
        (path, (lo, hi)),
        (path.with_name(path.stem + "_detail.png"), (c - 150.0, c + 150.0)),
    ):
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

        hw = m.half_width
        strip(roads.left_outer, roads.right_outer, (196, 176, 140))
        strip(m.centre + hw[:, None] * m.left, m.centre - hw[:, None] * m.left, (70, 70, 74))
        edge = hw - 0.2
        for side in (1.0, -1.0):
            dr.line(px(m.centre + side * edge[:, None] * m.left), fill=(250, 250, 250), width=1)
        centre_colour = (250, 200, 30) if m.double_yellow else (250, 250, 250)
        on = np.flatnonzero(m.centre_line)
        for run in np.split(on, np.flatnonzero(np.diff(on) > 1) + 1):
            if len(run) > 1:
                dr.line(px(m.centre[run]), fill=centre_colour, width=1)
        for ring in lines.rings:
            dr.line(px(ring), fill=(220, 30, 30), width=2)
        dr.line(px(ai.positions.astype(np.float64)), fill=(40, 90, 230), width=2)
        for name, node in ((d.name, d) for d in dummies.nodes):
            pos = node.transform[3, :3].astype(np.float64)
            (x, y), r = px(pos[None, :])[0], 3
            colour = (40, 200, 230)
            if name.startswith("AC_AB_START") or name.startswith("AC_TIME"):
                colour = (30, 200, 60)
            elif name.startswith("AC_AB_FINISH"):
                colour = (230, 40, 40)
            dr.ellipse((x - r, y - r, x + r, y + r), fill=colour)
        im.save(target, format="PNG", optimize=False)
        out.append(target)
    return out
