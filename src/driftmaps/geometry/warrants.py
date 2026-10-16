"""Automatic guardrails where Ontario road-design rules call for them ("warrants").

Implements docs/research/guardrails.md §7 (the rulebook: MTO Roadside Design Manual tables,
the AASHTO 2011 embankment heights via NJDOT, length of need, joining, minimum length, gaps at
side roads). `warrant_spans(inputs, options)` returns `BarrierSpan`s (kind "guard_rail",
source "warrant") plus a report that says why each span exists; the build passes the spans to
`geometry.barriers.plan_barriers(..., heuristic=...)`, so the user's guardrails.json, then OSM,
then bridge approaches win over them, and bridge decks are never railed twice.

Per road side, every `station_step_m` (2 m) over the A..B part of the road:

1. **Speed and traffic** (§2, §7.2): V_posted = OSM `maxspeed` (mph understood), else a class
   default; design speed V_d = V_posted + `v_margin_kmh` (+20, calibrated in §8); AADT from
   the road class. Roads posted
   <= 40 km/h get nothing; urban curbed streets (sidewalk, <= 50 km/h) only water, drops >= 2 m
   and bridge ends.
2. **Cross profile** from the lidar DTM (`dtm`, AC heights at XZ points): samples every 0.5 m
   from 1.5 m inside the lane edge (a misplaced OSM line can hide a slope's top there) out to
   the widest clear zone + 10 m, built with `road_columns` so profiles never cross on hairpins;
   1.5 m moving mean across, 3-station median along the road.
3. **Clear zone** (§3): MTO Table 2-2 by V_d, AADT and the ground's mean slope out to the flat
   clear zone (critical runs left out); x K_cz (Table 2-3) on the outside of curves with
   R <= 900 m, tapered 20:1 along the road (all of the taper on the tangent side, ASSUMED);
   a non-recoverable fill (3H:1V-4H:1V) that starts inside it pushes it to the toe + the
   recovery area (Table 2-2 Note 1).
4. **Hazards** (§4): a *fall* (run steeper than 3H:1V, gaps <= 1 m bridged) whose top is in the
   clear zone and whose height exceeds `h_scale` x h_max(slope) (0.6 m at 1:1 ... 6 m at 3:1);
   falls that climb back out within 6 m (V ditches up to 2 m deep) are not hazards; *deep water*
   (OSM water areas, rivers, canals) whose edge is inside the clear zone; optional *rock cuts*;
   *bridge ends* (every railed deck end, both sides). Trees, poles, signs and ditches are not
   triggers (§4.6).
5. **Rails** (§5): hazard runs shorter than `min_hazard_m` are dropped (water, drops >= 2 m and
   bridge ends exempt); each run gets its length of need La = E (1 - A/B) upstream of the near
   traffic and La' = E (1 - A'/B') for opposing traffic on two-way roads (E: MTO Table 2-16);
   same-side rails <= 50 m apart are joined unless a side road or driveway lies between;
   rails are cut at every joining way (its width / sin(angle) + 5 m, driveways 2 m) and over
   railed decks; pieces under 25 m grow to 25 m unless blocked (then dropped under 12 m).

Deterministic: same inputs, same spans (no randomness, stable sorting).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import shapely
from scipy.ndimage import median_filter, uniform_filter1d

from driftmaps.geo.attributes import parse_width
from driftmaps.geo.barriers import BarrierSpan, RoadsideFeatures, Side, project_points
from driftmaps.geo.project import LocalProjection
from driftmaps.geometry.road import RoadModel, road_columns

__all__ = [
    "CLASS_AADT",
    "CLASS_DEFAULT_SPEED",
    "Connection",
    "Hazard",
    "WarrantInputs",
    "WarrantOptions",
    "WarrantResult",
    "clear_zone",
    "connections_from_route",
    "curve_factor",
    "h_max",
    "parse_maxspeed",
    "rock_geoms",
    "runout_length",
    "warrant_inputs",
    "warrant_spans",
    "water_lines",
]

GroundFn = Callable[[np.ndarray], np.ndarray]

# --------------------------------------------------------------------------------------------
# Rule tables (guardrails.md §3, §4.1, §5.1, §7.2)
# --------------------------------------------------------------------------------------------

# Posted speed (km/h) when OSM has no maxspeed (ASSUMED, §7.2).
CLASS_DEFAULT_SPEED: dict[str, float] = {
    "motorway": 100.0,
    "trunk": 90.0,
    "primary": 80.0,
    "secondary": 80.0,
    "tertiary": 60.0,
    "unclassified": 60.0,
    "residential": 50.0,
    "living_street": 30.0,
    "service": 30.0,
    "track": 30.0,
}
# Representative AADT per class (ASSUMED, §7.2; no open counts checked for the test roads).
CLASS_AADT: dict[str, float] = {
    "motorway": 40_000.0,
    "trunk": 12_000.0,
    "primary": 8_000.0,
    "secondary": 3_000.0,
    "tertiary": 1_200.0,
    "unclassified": 400.0,
    "residential": 300.0,
    "living_street": 100.0,
    "service": 100.0,
    "track": 50.0,
}

# MTO RDM Table 2-2 (VERIFIED, §3.1): metres from the edge of the travelled way.
# Row: design speed key -> AADT lower bound -> columns
#   (fill 4H:1V-5H:1V, fill 6H:1V or flatter, 10H:1V or flatter, cut 6H:1V or flatter,
#    cut 5H:1V-4H:1V, cut 3H:1V). Fill 3H:1V is Note 1 (non-recoverable, see `_dcz`).
_DCZ_TABLE: dict[int, dict[int, tuple[float, ...]]] = {
    110: {
        6000: (14.0, 10.5, 9.5, 9.0, 9.0, 7.5),
        1500: (13.0, 10.0, 9.0, 8.5, 7.5, 6.0),
        750: (11.0, 8.0, 7.0, 6.5, 6.0, 5.0),
        0: (8.0, 6.0, 5.5, 5.0, 5.0, 3.5),
    },
    100: {
        6000: (13.5, 10.0, 9.0, 8.5, 8.0, 6.5),
        1500: (12.0, 9.0, 8.5, 8.0, 6.5, 5.5),
        750: (10.0, 7.5, 7.0, 6.5, 5.5, 4.5),
        0: (7.5, 5.5, 5.5, 5.0, 4.5, 3.5),
    },
    90: {
        6000: (10.0, 7.5, 7.5, 7.5, 6.5, 5.5),
        1500: (9.0, 6.5, 6.5, 6.5, 5.5, 5.0),
        750: (7.5, 5.5, 5.5, 5.5, 5.0, 3.5),
        0: (5.5, 4.5, 4.0, 3.5, 3.5, 3.0),
    },
    80: {  # 70-80 km/h
        6000: (8.5, 6.5, 6.5, 6.5, 6.0, 5.0),
        1500: (8.0, 5.5, 5.5, 5.5, 5.0, 4.5),
        750: (6.0, 5.0, 5.0, 5.0, 4.5, 3.5),
        0: (4.5, 3.5, 3.5, 3.5, 3.0, 3.0),
    },
    60: {  # <= 60 km/h
        6000: (5.5, 5.0, 5.0, 5.0, 5.0, 5.0),
        1500: (5.0, 4.5, 4.5, 4.5, 4.5, 4.5),
        750: (4.5, 3.5, 3.5, 3.5, 3.5, 3.5),
        0: (3.0, 3.0, 3.0, 3.0, 3.0, 3.0),
    },
}
COL_FILL_4_5, COL_FILL_6, COL_FLAT, COL_CUT_6, COL_CUT_5_4, COL_CUT_3 = range(6)
_COLUMN_NAMES = ("fill 4-5:1", "fill 6:1", "flat", "cut 6:1", "cut 5-4:1", "cut 3:1")

# MTO RDM Table 2-3 K_cz (VERIFIED, §3.2): radius -> factor per design speed column.
_K_SPEEDS = (60, 70, 80, 90, 100, 110)
_K_TABLE: tuple[tuple[float, tuple[float | None, ...]], ...] = (
    (900.0, (1.1, 1.1, 1.1, 1.2, 1.2, 1.2)),
    (700.0, (1.1, 1.1, 1.2, 1.2, 1.2, 1.3)),
    (600.0, (1.1, 1.2, 1.2, 1.2, 1.3, 1.4)),
    (500.0, (1.1, 1.2, 1.2, 1.3, 1.3, 1.4)),
    (450.0, (1.2, 1.2, 1.3, 1.3, 1.4, 1.5)),
    (400.0, (1.2, 1.2, 1.3, 1.3, 1.4, None)),
    (350.0, (1.2, 1.2, 1.3, 1.4, 1.5, None)),
    (300.0, (1.2, 1.3, 1.4, 1.5, 1.5, None)),
    (250.0, (1.3, 1.3, 1.4, 1.5, None, None)),
    (200.0, (1.3, 1.4, 1.5, None, None, None)),
    (150.0, (1.4, 1.5, None, None, None, None)),
    (100.0, (1.5, None, None, None, None, None)),
)
K_MAX = 1.5  # "—" cells and R < 100 m (ASSUMED: the table's ceiling)

# MTO RDM Table 2-16 runout length E (m) (VERIFIED, §5.1): V_d -> (> 10k, 5-10k, 1-5k, < 1k).
_RUNOUT: dict[int, tuple[float, float, float, float]] = {
    130: (143.0, 131.0, 116.0, 101.0),
    120: (127.0, 116.0, 102.0, 89.0),
    110: (110.0, 101.0, 88.0, 76.0),
    100: (91.0, 76.0, 64.0, 61.0),
    90: (81.0, 67.0, 57.0, 54.0),
    80: (70.0, 58.0, 49.0, 46.0),
    70: (60.0, 49.0, 42.0, 38.0),
    60: (49.0, 40.0, 34.0, 30.0),
    50: (34.0, 27.0, 24.0, 21.0),
}

# Critical-slope heights (§4.1): (H:V ratio, max height without a rail), linear between rows.
_H_MAX = ((1.0, 0.6), (1.5, 0.9), (2.0, 1.8), (2.5, 2.7), (3.0, 6.0))

CRITICAL_SLOPE = 1.0 / 3.0  # steeper than 3H:1V
NON_RECOVERABLE_SLOPE = 1.0 / 4.0  # 3H:1V..4H:1V: travel to the bottom (Note 1)

# Joining-way widths (m) without a `width` tag (ASSUMED, §5.4).
_ACCESS_WIDTH = {
    "driveway": 4.0,
    "service": 5.0,
    "residential": 7.0,
    "unclassified": 7.0,
    "track": 7.0,
    "living_street": 7.0,
}
_PUBLIC_WIDTH = 8.0  # tertiary and up


def h_max(ratio: npt.ArrayLike) -> np.ndarray:
    """Greatest fill height (m) without a rail at slope `ratio` (H:V): 0.6 m at 1:1 or
    steeper, 0.9 at 1.5:1, 1.8 at 2:1, 2.7 at 2.5:1, 6.0 at 3:1, infinite flatter than 3:1."""
    r = np.asarray(ratio, np.float64)
    xs, ys = zip(*_H_MAX, strict=True)
    out = np.interp(r, xs, ys)
    return np.where(r > xs[-1] + 0.05, np.inf, out)


def _speed_key(v_d: float) -> int:
    if v_d <= 60.0:
        return 60
    if v_d <= 80.0:
        return 80
    if v_d <= 90.0:
        return 90
    if v_d <= 100.0:
        return 100
    return 110


def _aadt_key(aadt: float) -> int:
    for lo in (6000, 1500, 750):
        if aadt >= lo:
            return lo
    return 0


def clear_zone(v_d: float, aadt: float, column: int) -> float:
    """MTO Table 2-2 desirable clear zone (m from the edge of the travelled way)."""
    return _DCZ_TABLE[_speed_key(v_d)][_aadt_key(aadt)][column]


def curve_factor(radius_m: float, v_d: float) -> float:
    """MTO Table 2-3 K_cz for the outside of a curve of `radius_m` at design speed `v_d`:
    the nearest table radius at or below R and speed column at or above V_d; 1.0 for
    R > 900 m, K_MAX for R < 100 m or a "—" cell."""
    if not radius_m <= 900.0:
        return 1.0
    col = next((k for k, v in enumerate(_K_SPEEDS) if v >= v_d - 1e-9), len(_K_SPEEDS) - 1)
    for r, row in _K_TABLE:
        if r <= radius_m + 1e-9:
            k = row[col]
            return K_MAX if k is None else float(k)
    return K_MAX


def runout_length(v_d: float, aadt: float) -> float:
    """MTO Table 2-16 runout length E (m): the nearest table speed at or above V_d."""
    keys = sorted(_RUNOUT)
    key = next((k for k in keys if k >= v_d - 1e-9), keys[-1])
    row = _RUNOUT[key]
    if aadt > 10_000:
        return row[0]
    if aadt > 5_000:
        return row[1]
    if aadt >= 1_000:
        return row[2]
    return row[3]


def parse_maxspeed(value: str | None) -> float | None:
    """OSM `maxspeed` in km/h ("50", "40 mph", "50;60" -> the first), else None."""
    if not value:
        return None
    text = value.split(";")[0].strip().lower()
    mph = text.endswith("mph")
    num = text.removesuffix("mph").removesuffix("km/h").strip()
    try:
        v = float(num)
    except ValueError:
        return None
    if not 5.0 <= v <= 200.0:
        return None
    return v * 1.609344 if mph else v


def _base_class(highway: str) -> str:
    return highway[:-5] if highway.endswith("_link") else highway


# --------------------------------------------------------------------------------------------
# Inputs and options
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Connection:
    """A way joining the road at station `s` on `side`: the rail is cut over
    s +- (width / sin(angle) / 2 + clearance)."""

    s: float
    side: Side
    width_m: float
    clearance_m: float
    sin_angle: float = 1.0
    kind: str = "road"  # highway class, or "driveway"
    way_id: int | None = None

    @property
    def half_cut(self) -> float:
        return 0.5 * self.width_m / max(self.sin_angle, 0.5) + self.clearance_m

    @property
    def zone(self) -> tuple[float, float]:
        return (self.s - self.half_cut, self.s + self.half_cut)


@dataclass(frozen=True)
class WarrantOptions:
    """Knobs of the warrant model (guardrails.md §7.2; calibration knobs first)."""

    h_scale: float = 1.0  # multiplier on the §4.1 heights (calibration knob 1)
    min_hazard_m: float = 10.0  # shorter hazard runs are ignored (water, big drops exempt)
    # Design speed = posted + this. guardrails.md §2 proposed +10 (ASSUMED); the validation
    # against OSM-mapped rails (§8) favours +20, MTO's historical Ontario margin (VERIFIED).
    v_margin_kmh: float = 20.0
    class_speed: Mapping[str, float] = field(default_factory=lambda: dict(CLASS_DEFAULT_SPEED))
    class_aadt: Mapping[str, float] = field(default_factory=lambda: dict(CLASS_AADT))
    aadt: float | None = None  # one AADT for the whole road (overrides the class table)
    exempt_drop_m: float = 2.0  # falls at least this high need no minimum hazard length
    join_m: float = 50.0
    min_run_m: float = 25.0
    min_cut_piece_m: float = 12.0
    public_clearance_m: float = 5.0
    driveway_clearance_m: float = 2.0
    station_step_m: float = 2.0
    profile_step_m: float = 0.5
    profile_inside_m: float = 1.5  # profiles start this far inside the lane edge
    profile_margin_m: float = 10.0  # ... and reach this far beyond the widest clear zone,
    profile_min_m: float = 30.0  # ... at least this far (the toe of a 6 m 3H:1V fill)
    profile_max_m: float = 40.0
    smooth_across_m: float = 1.5
    smooth_along_m: float = 5.0
    gap_bridge_m: float = 1.0  # flatter bits this short inside a critical run are bridged
    ditch_max_depth_m: float = 2.0  # falls this shallow that climb back out are ditches
    ditch_back_m: float = 6.0
    ditch_rise: float = 0.7  # ... climbing back at least this fraction of their depth
    rounding_m: float = 1.0  # slope rounding for the recovery area (1.5 m at >= 100 km/h)
    water: bool = True
    bridge_ends: bool = True
    rock_cuts: bool = False  # optional rule (§4.5), off until confirmed
    rock_min_height_m: float = 2.0
    rock_osm_m: float = 15.0
    min_posted_kmh: float = 40.0  # roads posted at or below this get nothing
    urban_posted_kmh: float = 50.0  # ... and curbed streets up to this only big hazards


@dataclass
class WarrantInputs:
    """What the warrant model reads. Per station (N = len(model)): `highway`, `maxspeed_kmh`
    (NaN = not tagged), `sidewalk`, `oneway` (+1 one-way in the driving direction, -1 against
    it, 0 two-way). `dtm` gives the bare-earth height (AC Y) under XZ points (NaN unknown).
    `decks` are the railed bridge decks (station ranges); `connections` the joining ways;
    `water` deep-water edges and `rock` rock features as shapely geometries in AC XZ.
    Rails are only placed within `s_range` (the real road, A..B)."""

    model: RoadModel
    dtm: GroundFn
    highway: np.ndarray
    maxspeed_kmh: np.ndarray
    sidewalk: np.ndarray
    oneway: np.ndarray
    decks: Sequence[tuple[float, float]] = ()
    connections: Sequence[Connection] = ()
    water: Sequence[shapely.Geometry] = ()
    rock: Sequence[shapely.Geometry] = ()
    s_range: tuple[float, float] | None = None
    driving_side: str = "right"


@dataclass(frozen=True)
class Hazard:
    """One hazard run on a side: stations [s0, s1] (s0 == s1 for a bridge end), back
    distance `b_m` (<= clear zone), reason `kind` ("fall", "water", "bridge", "rock") and a
    human-readable `why`."""

    side: Side
    s0: float
    s1: float
    kind: str
    why: str
    b_m: float
    dcz_m: float
    v_d: float
    aadt: float
    h_m: float = 0.0
    ratio: float = 0.0
    la_m: float = 0.0
    la2_m: float = 0.0
    lo: float = 0.0  # rail extent from this hazard after the length of need
    hi: float = 0.0


@dataclass
class WarrantResult:
    spans: list[BarrierSpan]
    hazards: list[Hazard]
    report: list[dict]  # one entry per span: why it exists
    stations: dict[str, np.ndarray] = field(default_factory=dict)  # per-station diagnostics


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[a, b) index runs where `mask` is True."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(), strict=True))


def _bridge_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill False gaps of at most `max_gap` samples between True runs."""
    out = mask.copy()
    runs = _runs(mask)
    for (_, b), (a2, _) in zip(runs[:-1], runs[1:], strict=True):
        if a2 - b <= max_gap:
            out[b:a2] = True
    return out


def _union(intervals: Sequence[tuple[float, float]], join: float = 0.0) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + join:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _subtract(pieces: list[list[float]], cuts: Sequence[tuple[float, float]]) -> list[list[float]]:
    for c0, c1 in cuts:
        nxt = []
        for p0, p1 in pieces:
            if c1 <= p0 or c0 >= p1:
                nxt.append([p0, p1])
                continue
            if c0 > p0:
                nxt.append([p0, c0])
            if c1 < p1:
                nxt.append([c1, p1])
        pieces = nxt
    return pieces


def _round_half(x: np.ndarray) -> np.ndarray:
    """Round up to the next 0.5 m (DCZ_c rounding, halves up, ASSUMED)."""
    return np.ceil(np.asarray(x) * 2.0 - 1e-9) / 2.0


# --------------------------------------------------------------------------------------------
# Inputs from a route
# --------------------------------------------------------------------------------------------


def _xz(projection: LocalProjection, lonlat: np.ndarray) -> np.ndarray:
    en = projection.to_xy(np.asarray(lonlat, np.float64).reshape(-1, 2))
    return np.stack([en[:, 0], -en[:, 1]], axis=1)


def water_lines(
    features: RoadsideFeatures | None, projection: LocalProjection
) -> list[shapely.Geometry]:
    """Deep-water edges in AC XZ: area boundaries as lines, rivers and canals as the outline
    of their centre line buffered by half their width."""
    out: list[shapely.Geometry] = []
    if features is None:
        return out
    for f in features.water:
        xz = _xz(projection, f.lonlat)
        if len(xz) < 2:
            continue
        line = shapely.LineString(xz)
        if f.kind in ("river", "canal"):
            out.append(line.buffer(f.half_width_m, cap_style="flat").boundary)
        else:
            out.append(line)
    return out


def rock_geoms(
    features: RoadsideFeatures | None, projection: LocalProjection
) -> list[shapely.Geometry]:
    if features is None:
        return []
    return [
        shapely.LineString(_xz(projection, f.lonlat)) for f in features.rock if len(f.lonlat) > 1
    ]


def _access_width(tags: Mapping[str, str]) -> tuple[float, str, bool]:
    """(width, kind, public) of a joining way."""
    hw = _base_class(tags.get("highway", ""))
    kind = "driveway" if tags.get("service") == "driveway" else hw
    w = parse_width(tags.get("width"))
    public = hw not in ("service", "track") and kind != "driveway"
    if w is None or not w > 0:
        w = _ACCESS_WIDTH.get(kind, _PUBLIC_WIDTH)
    return float(w), kind, public


def connections_from_route(
    model: RoadModel,
    projection: LocalProjection,
    route,
    features: RoadsideFeatures | None = None,
    options: WarrantOptions | None = None,
    *,
    s_origin: float = 0.0,
    route_s: np.ndarray | None = None,
) -> list[Connection]:
    """Joining ways: the route's side-road `stubs` and, if given, every OSM road, track or
    driveway in `features.access` that shares a node with the route or ends at its edge.
    `route` is a `geo.router.Route` (read-only)."""
    opt = options or WarrantOptions()
    centre = model.centre[:, [0, 2]]
    out: list[Connection] = []
    route_nodes = set(route.node_path())
    route_segs = {(e.u, e.v) for e in route.edges} | {(e.v, e.u) for e in route.edges}

    def add(p0: np.ndarray, p1: np.ndarray, tags: Mapping[str, str], way: int | None) -> None:
        st, off, inside = project_points(p0[None, :], centre, model.s)
        if not inside[0]:
            return
        s = float(st[0])
        i = model.index_at(s)
        d = p1 - p0
        norm = float(np.linalg.norm(d))
        if norm < 1e-6:
            return
        d = d / norm
        left = model.left[i, [0, 2]]
        fwd = model.forward[i, [0, 2]] / max(1e-9, float(np.linalg.norm(model.forward[i, [0, 2]])))
        side: Side = "left" if float(d @ left) > 0.0 else "right"
        sin_a = abs(float(fwd[0] * d[1] - fwd[1] * d[0]))
        w, kind, public = _access_width(tags)
        clear = opt.public_clearance_m if public else opt.driveway_clearance_m
        out.append(Connection(s, side, w, clear, sin_a, kind, way))

    def far_point(xz: np.ndarray, k: int, step: int) -> np.ndarray | None:
        """The first point of the way from index k in direction `step` at least 5 m away."""
        j = k + step
        while 0 <= j < len(xz):
            if np.linalg.norm(xz[j] - xz[k]) >= 5.0:
                return xz[j]
            j += step
        j -= step
        return xz[j] if j != k and 0 <= j < len(xz) else None

    for node, way, other in route.stubs:
        if node not in route.nodes or other not in route.nodes:
            continue
        p0, p1 = _xz(projection, np.array([route.nodes[node], route.nodes[other]]))
        add(p0, p1, route.ways.get(way, {}), way)
    if features is not None:
        on_route_ways = {e.way for e in route.edges}
        for f in features.access:
            xz = _xz(projection, f.lonlat)
            ids = f.nodes
            shared = [k for k, n in enumerate(ids) if n in route_nodes] if ids else []
            for k in shared:
                for step in (-1, 1):
                    j = k + step
                    if not 0 <= j < len(ids) or (ids[k], ids[j]) in route_segs:
                        continue
                    p1 = far_point(xz, k, step)
                    if p1 is not None:
                        add(xz[k], p1, f.tags, f.id)
            if shared or f.id in on_route_ways:
                continue
            # A way ending at the road edge without a shared node (common for driveways).
            for k, step in ((0, 1), (len(xz) - 1, -1)):
                st, off, inside = project_points(xz[k][None, :], centre, model.s)
                if not inside[0]:
                    continue
                edge = float(np.interp(st[0], model.s, model.outer))
                p1 = far_point(xz, k, step)
                if p1 is None or abs(off[0]) > edge + 3.0:
                    continue
                _, off1, _ = project_points(p1[None, :], centre, model.s)
                if abs(off1[0]) > abs(off[0]) + 1.0:  # it leaves the road
                    add(xz[k], p1, f.tags, f.id)
    # Dedupe: one connection per (side, station to 1 m, kind).
    seen: dict[tuple, Connection] = {}
    for c in out:
        seen.setdefault((c.side, round(c.s), c.kind), c)
    return sorted(seen.values(), key=lambda c: (c.side, c.s, c.kind))


# --------------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------------


@dataclass
class _Road:
    """Per-station road parameters at the sampled stations."""

    idx: np.ndarray  # model indices
    s: np.ndarray
    v_posted: np.ndarray
    v_d: np.ndarray
    aadt: np.ndarray
    urban: np.ndarray
    skip: np.ndarray
    oneway: np.ndarray


def _road_params(inp: WarrantInputs, opt: WarrantOptions) -> _Road:
    m = inp.model
    lo, hi = inp.s_range if inp.s_range is not None else (float(m.s[0]), float(m.s[-1]))
    targets = np.arange(lo, hi + 1e-9, opt.station_step_m)
    idx = np.unique(np.clip(np.searchsorted(m.s, targets), 0, len(m) - 1))
    hw = np.asarray(inp.highway)[idx]
    cls = np.array([_base_class(str(h)) for h in hw])
    default_v = np.array([opt.class_speed.get(c, 60.0) for c in cls])
    tagged = np.asarray(inp.maxspeed_kmh, np.float64)[idx]
    v_posted = np.where(np.isfinite(tagged), tagged, default_v)
    v_d = np.ceil((v_posted + opt.v_margin_kmh) / 10.0 - 1e-9) * 10.0
    if opt.aadt is not None:
        aadt = np.full(len(idx), float(opt.aadt))
    else:
        aadt = np.array([opt.class_aadt.get(c, 400.0) for c in cls])
    sidewalk = np.asarray(inp.sidewalk, bool)[idx]
    urban = sidewalk & (v_posted <= opt.urban_posted_kmh + 1e-9)
    on_deck = np.zeros(len(idx), bool)
    s = m.s[idx]
    for a, b in inp.decks:
        on_deck |= (s >= a - 1.0) & (s <= b + 1.0)
    in_junction = np.zeros(len(idx), bool)
    for c in inp.connections:
        a, b = c.zone
        in_junction |= (s >= a) & (s <= b)
    skip = on_deck | in_junction | (v_posted <= opt.min_posted_kmh + 1e-9)
    oneway = np.asarray(inp.oneway, np.int64)[idx]
    return _Road(idx, s, v_posted, v_d, aadt, urban, skip, oneway)


def _profiles(
    inp: WarrantInputs, road: _Road, sign: float, xs: np.ndarray, opt: WarrantOptions
) -> tuple[np.ndarray, np.ndarray]:
    """Ground heights (M, K) at lateral offsets `xs` beyond the lane edge, and the XZ points
    (M, K, 2), smoothed across (moving mean) and along the road (median)."""
    m = inp.model
    pts = np.empty((len(road.idx), len(xs), 2))
    for k, x in enumerate(xs):
        col = road_columns(m, sign * np.maximum(m.half_width + x, 0.0))
        pts[:, k] = col[road.idx][:, [0, 2]]
    z = np.asarray(inp.dtm(pts.reshape(-1, 2)), np.float64).reshape(len(road.idx), len(xs))
    # Fill NaNs along each profile (nearest valid), so smoothing does not spread them.
    bad = ~np.isfinite(z)
    if bad.any():
        for i in np.flatnonzero(bad.any(axis=1)):
            ok = ~bad[i]
            if ok.any():
                z[i, ~ok] = np.interp(xs[~ok], xs[ok], z[i, ok])
    across = max(1, int(round(opt.smooth_across_m / opt.profile_step_m)))
    if across > 1:
        z = uniform_filter1d(z, across, axis=1, mode="nearest")
    along = max(1, int(round(opt.smooth_along_m / opt.station_step_m)))
    along += 1 - along % 2  # odd window
    if along > 1:
        z = median_filter(z, size=(along, 1), mode="nearest")
    return z, pts


def _taper(s: np.ndarray, extra: np.ndarray, ratio: float = 20.0) -> np.ndarray:
    """max_j(extra_j - |s - s_j| / ratio): a 20:1 transition out of each widened zone."""
    out = extra.copy()
    order = np.argsort(-extra)
    for j in order:
        if extra[j] <= 0.0:
            break
        cand = extra[j] - np.abs(s - s[j]) / ratio
        np.maximum(out, cand, out=out)
    return out


def _side_hazards(
    inp: WarrantInputs, road: _Road, side: Side, opt: WarrantOptions
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Per-station hazards on one side: a list of dicts (station index j, kind, ...)."""
    m = inp.model
    sign = 1.0 if side == "left" else -1.0
    n = len(road.idx)
    # Widest possible clear zone on this road -> profile length.
    dcz_flat = np.array(
        [clear_zone(v, a, COL_FLAT) for v, a in zip(road.v_d, road.aadt, strict=True)]
    )
    dcz_wide = np.array(
        [clear_zone(v, a, COL_FILL_4_5) for v, a in zip(road.v_d, road.aadt, strict=True)]
    )
    x_max = min(
        opt.profile_max_m,
        max(opt.profile_min_m, float(dcz_wide.max(initial=5.0)) * K_MAX + opt.profile_margin_m),
    )
    dx = opt.profile_step_m
    xs = np.arange(-opt.profile_inside_m, x_max + 1e-9, dx)
    z, pts = _profiles(inp, road, sign, xs, opt)
    slope = np.diff(z, axis=1) / dx  # (M, K-1), segment k spans xs[k]..xs[k+1]
    x0 = xs[:-1]
    crit = slope < -CRITICAL_SLOPE + 0.002  # 3H:1V itself counts (its 6 m row)
    gap = max(0, int(round(opt.gap_bridge_m / dx)))

    # Column of Table 2-2: mean slope from the lane edge out to the flat DCZ, critical falls
    # left out.
    cols = np.full(n, COL_FLAT)
    for i in range(n):
        sel = (x0 >= 0.0) & (x0 < dcz_flat[i]) & ~crit[i]
        g = float(slope[i, sel].mean()) if sel.any() else 0.0
        if g < -1.0 / 6.0:
            cols[i] = COL_FILL_4_5
        elif g < -0.1:
            cols[i] = COL_FILL_6
        elif g <= 0.1:
            cols[i] = COL_FLAT
        elif g <= 1.0 / 6.0:
            cols[i] = COL_CUT_6
        elif g <= 0.25:
            cols[i] = COL_CUT_5_4
        else:
            cols[i] = COL_CUT_3
    dcz_t = np.array(
        [clear_zone(v, a, int(c)) for v, a, c in zip(road.v_d, road.aadt, cols, strict=True)]
    )
    # Outside of curves: K_cz, tapered 20:1.
    kappa = uniform_filter1d(m.curvature, 15, mode="nearest")[road.idx]
    outside = (kappa < 0.0) if side == "left" else (kappa > 0.0)
    radius = np.where(np.abs(kappa) > 1e-9, 1.0 / np.maximum(np.abs(kappa), 1e-9), np.inf)
    k_cz = np.array(
        [
            curve_factor(r, v) if o else 1.0
            for r, v, o in zip(radius, road.v_d, outside, strict=True)
        ]
    )
    extra = _taper(road.s, dcz_t * (k_cz - 1.0))
    dcz = _round_half(dcz_t + extra)
    # Non-recoverable fill (steeper than 4H:1V) starting inside the DCZ: toe + recovery.
    recovery = np.maximum(
        dcz_flat - (m.shoulder[road.idx] + 0.5 * np.where(road.v_d >= 100, 1.5, opt.rounding_m)),
        3.0,
    )
    # 3H:1V..4H:1V only: a critical fall is a hazard of its own, not a recoverable slope.
    nonrec = (slope < -NON_RECOVERABLE_SLOPE) & ~crit
    dcz_eff = dcz.copy()
    for i in range(n):
        for a, b in _runs(_bridge_gaps(nonrec[i], gap)):
            if x0[a] > dcz[i]:
                break
            if xs[b] - x0[a] >= 2.0 and xs[b] > 0.0:
                dcz_eff[i] = max(dcz_eff[i], xs[b] + recovery[i])
    hmax_scale = opt.h_scale

    hazards: list[dict] = []
    fall_h = np.zeros(n)
    for i in range(n):
        if road.skip[i]:
            continue
        zi = z[i]
        for a, b in _runs(_bridge_gaps(crit[i], gap)):
            x_t, x_b = float(x0[a]), float(xs[b])
            if x_t > dcz_eff[i]:
                break
            h = float(zi[a] - zi[b])
            if h <= 0.05 or x_b <= 0.0:
                continue
            ratio = (x_b - x_t) / h
            limit = float(h_max(ratio)) * hmax_scale
            if h <= limit:
                continue
            if road.urban[i] and h < opt.exempt_drop_m:
                continue
            # A V ditch: shallow, and the ground climbs back out right after the toe.
            if h < opt.ditch_max_depth_m:
                back = (xs > x_b) & (xs <= x_b + opt.ditch_back_m)
                if back.any() and float(zi[back].max()) >= zi[b] + opt.ditch_rise * h:
                    continue
            b_m = min(max(x_b, 0.5), float(dcz[i]))
            hazards.append(
                dict(j=i, kind="fall", h=h, ratio=ratio, x_t=x_t, x_b=x_b, limit=limit,
                     b=b_m, dcz=float(dcz_eff[i]))
            )  # fmt: skip
            fall_h[i] = max(fall_h[i], h)
            break
    # Deep water whose edge is inside the clear zone.
    if opt.water and inp.water:
        tree = shapely.STRtree(list(inp.water))
        k0 = int(np.searchsorted(xs, 0.0))
        live = np.flatnonzero(~road.skip)
        if len(live):
            k1 = np.clip(np.searchsorted(xs, dcz_eff[live]), k0 + 1, len(xs) - 1)
            segs = shapely.linestrings(
                np.stack([pts[live, k0], pts[live, k1]], axis=1)  # (L, 2, 2)
            )
            pairs = tree.query(segs, predicate="intersects")
            best: dict[int, float] = {}
            for li, wi in zip(pairs[0].tolist(), pairs[1].tolist(), strict=True):
                inter = shapely.intersection(segs[li], inp.water[wi])
                d = float(shapely.distance(shapely.Point(pts[live[li], k0]), inter))
                best[li] = min(best.get(li, np.inf), d)
            for li, d in sorted(best.items()):
                i = int(live[li])
                hazards.append(dict(j=i, kind="water", d=d, b=float(dcz[i]), dcz=float(dcz_eff[i])))
    # Optional rough rock cuts (§4.5).
    if opt.rock_cuts and inp.rock:
        rock = shapely.STRtree(list(inp.rock))
        rise = slope > 1.0
        for i in range(n):
            if road.skip[i] or road.urban[i]:
                continue
            for a, b in _runs(_bridge_gaps(rise[i], gap)):
                x_toe = float(x0[a])
                if x_toe > dcz[i]:
                    break
                h = float(z[i, b] - z[i, a])
                if h < opt.rock_min_height_m:
                    continue
                p = shapely.Point(pts[i, a])
                near = rock.query(p, predicate="dwithin", distance=opt.rock_osm_m)
                if len(near):
                    hazards.append(
                        dict(j=i, kind="rock", h=h, x_t=x_toe, b=max(x_toe, 0.5),
                             dcz=float(dcz[i]))
                    )  # fmt: skip
                    break
    diag = {
        "dcz": dcz,
        "dcz_eff": dcz_eff,
        "column": cols,
        "k_cz": k_cz,
        "fall_h": fall_h,
    }
    return hazards, diag


def _group(
    inp: WarrantInputs, road: _Road, side: Side, found: list[dict], opt: WarrantOptions
) -> list[Hazard]:
    """Station hazards -> hazard runs (consecutive stations of one kind), minimum length."""
    out: list[Hazard] = []
    step = opt.station_step_m
    for kind in ("fall", "water", "rock"):
        rows = sorted((h for h in found if h["kind"] == kind), key=lambda h: h["j"])
        if not rows:
            continue
        groups: list[list[dict]] = [[rows[0]]]
        for h in rows[1:]:
            prev = groups[-1][-1]
            if road.s[h["j"]] - road.s[prev["j"]] <= max(3.0, 1.5 * step):
                groups[-1].append(h)
            else:
                groups.append([h])
        for g in groups:
            j0, j1 = g[0]["j"], g[-1]["j"]
            s0, s1 = float(road.s[j0]), float(road.s[j1])
            hmax = max((h.get("h", 0.0) for h in g), default=0.0)
            exempt = kind == "water" or (kind == "fall" and hmax >= opt.exempt_drop_m)
            if s1 - s0 < opt.min_hazard_m and not (exempt and len(g) >= 2):
                continue
            top = max(g, key=lambda h: h.get("h", 0.0) - h.get("limit", 0.0))
            v_d = float(road.v_d[j0 : j1 + 1].max())
            aadt = float(road.aadt[j0 : j1 + 1].max())
            b = min(h["b"] for h in g)
            dcz = max(h["dcz"] for h in g)
            if kind == "fall":
                why = (
                    f"fill {top['ratio']:.1f}H:1V, {top['h']:.1f} m drop {top['x_t']:.1f} m "
                    f"from the lane edge (limit {top['limit']:.1f} m, "
                    f"clear zone {top['dcz']:.1f} m)"
                )
            elif kind == "water":
                d = min(h["d"] for h in g)
                why = f"deep water {d:.1f} m from the lane edge (clear zone {dcz:.1f} m)"
            else:
                why = f"rock cut {hmax:.1f} m high {top['x_t']:.1f} m from the lane edge"
            out.append(
                Hazard(side, s0, s1, kind, why, b, dcz, v_d, aadt, hmax, top.get("ratio", 0.0))
            )
    return out


def _bridge_hazards(inp: WarrantInputs, road: _Road, opt: WarrantOptions) -> list[Hazard]:
    out: list[Hazard] = []
    if not opt.bridge_ends:
        return out
    m = inp.model
    lo, hi = inp.s_range if inp.s_range is not None else (float(m.s[0]), float(m.s[-1]))
    for d0, d1 in inp.decks:
        for s_e in (d0, d1):
            if not lo - 1.0 <= s_e <= hi + 1.0:
                continue
            j = int(np.clip(np.searchsorted(road.s, s_e), 0, len(road.s) - 1))
            if road.v_posted[j] <= opt.min_posted_kmh + 1e-9:
                continue
            if road.urban[j] and road.v_d[j] <= 60.0:
                continue  # urban curbed bridge: its railing runs on, no approach rail (ASSUMED)
            v_d, aadt = float(road.v_d[j]), float(road.aadt[j])
            dcz = clear_zone(v_d, aadt, COL_FLAT)
            for side in ("left", "right"):
                why = f"bridge end at {s_e - lo:.0f} m from A (clear zone {dcz:.1f} m)"
                out.append(Hazard(side, s_e, s_e, "bridge", why, dcz, dcz, v_d, aadt))
    return out


def _length_of_need(inp: WarrantInputs, road: _Road, hz: Hazard) -> Hazard:
    m = inp.model
    i = m.index_at(0.5 * (hz.s0 + hz.s1))
    hw = float(m.half_width[i])
    a = float(m.shoulder[i])  # rail face at the shoulder edge (guardrails.md §5.5)
    e = runout_length(hz.v_d, hz.aadt)
    b = max(hz.b_m, 1e-6)
    la = max(0.0, e * (1.0 - a / b)) if b > a else 0.0
    j = int(np.clip(np.searchsorted(road.s, hz.s0), 0, len(road.s) - 1))
    oneway = int(road.oneway[j])
    la2 = 0.0
    if oneway == 0:
        a2, b2 = hw + a, min(hw + hz.b_m, hz.dcz_m)
        la2 = max(0.0, e * (1.0 - a2 / b2)) if b2 > a2 else 0.0
    # Which way is upstream for the near traffic on this side.
    plus_near = (hz.side == "right") == (inp.driving_side == "right")
    if oneway == 1:
        lo, hi = hz.s0 - la, hz.s1
    elif oneway == -1:
        lo, hi = hz.s0, hz.s1 + la
    elif plus_near:
        lo, hi = hz.s0 - la, hz.s1 + la2
    else:
        lo, hi = hz.s0 - la2, hz.s1 + la
    return dataclasses.replace(hz, la_m=la, la2_m=la2, lo=lo, hi=hi)


def _blocks(inp: WarrantInputs, side: Side) -> tuple[list[tuple[float, float]], list]:
    cuts = [c.zone for c in inp.connections if c.side == side]
    decks = [(float(a), float(b)) for a, b in inp.decks]
    return cuts, decks


def _side_rails(
    inp: WarrantInputs, side: Side, hazards: list[Hazard], opt: WarrantOptions
) -> list[tuple[float, float]]:
    m = inp.model
    lo, hi = inp.s_range if inp.s_range is not None else (float(m.s[0]), float(m.s[-1]))
    cuts, decks = _blocks(inp, side)
    spans = _union([(max(lo, h.lo), min(hi, h.hi)) for h in hazards if h.side == side])
    spans = [p for p in spans if p[1] > p[0]]
    # Join gaps <= join_m unless a joining way lies in the gap.
    joined: list[list[float]] = []
    for p in spans:
        if joined:
            g0, g1 = joined[-1][1], p[0]
            blocked = any(c0 < g1 and c1 > g0 for c0, c1 in cuts)
            if g1 - g0 <= opt.join_m and not blocked:
                joined[-1][1] = max(joined[-1][1], p[1])
                continue
        joined.append(list(p))
    pieces = _subtract(joined, cuts + decks)
    blockers = sorted(cuts + decks + [(-np.inf, lo), (hi, np.inf)])
    out: list[tuple[float, float]] = []
    for p0, p1 in pieces:
        if p1 - p0 < 0.5:
            continue
        left = max((b for _, b in blockers if b <= p0 + 1e-6), default=-np.inf)
        right = min((a for a, _ in blockers if a >= p1 - 1e-6), default=np.inf)
        need = opt.min_run_m - (p1 - p0)
        if need > 0:
            g0 = min(need / 2.0, p0 - left)
            g1 = min(need - g0, right - p1)
            g0 = min(need - g1, p0 - left)
            p0, p1 = p0 - g0, p1 + g1
            if p1 - p0 < opt.min_run_m - 1e-6 and p1 - p0 < opt.min_cut_piece_m:
                continue
        out.append((float(p0), float(p1)))
    return [(a, b) for a, b in _union(out)]


def warrant_spans(inp: WarrantInputs, options: WarrantOptions | None = None) -> WarrantResult:
    """Guardrail spans warranted by the rules (see the module docstring), with the hazards
    behind them and a per-span report."""
    opt = options or WarrantOptions()
    m = inp.model
    road = _road_params(inp, opt)
    if len(road.idx) < 2:
        return WarrantResult([], [], [])
    hazards: list[Hazard] = []
    diags: dict[str, np.ndarray] = {"s": road.s, "v_d": road.v_d, "aadt": road.aadt}
    for side in ("left", "right"):
        found, diag = _side_hazards(inp, road, side, opt)
        hazards += _group(inp, road, side, found, opt)
        diags.update({f"{k}_{side}": v for k, v in diag.items()})
    hazards += _bridge_hazards(inp, road, opt)
    hazards = [_length_of_need(inp, road, h) for h in hazards]
    hazards.sort(key=lambda h: (h.side, h.s0, h.s1, h.kind))
    spans: list[BarrierSpan] = []
    report: list[dict] = []
    s_a = inp.s_range[0] if inp.s_range is not None else float(m.s[0])
    for side in ("left", "right"):
        for a, b in _side_rails(inp, side, hazards, opt):
            spans.append(BarrierSpan(side, a, b, "guard_rail", "warrant"))
            why = [h for h in hazards if h.side == side and h.lo < b and h.hi > a]
            report.append(
                {
                    "side": side,
                    "from_m": round(a - s_a, 1),
                    "to_m": round(b - s_a, 1),
                    "length_m": round(b - a, 1),
                    "reasons": [h.why for h in why],
                    "hazards": [
                        {
                            "kind": h.kind,
                            "from_m": round(h.s0 - s_a, 1),
                            "to_m": round(h.s1 - s_a, 1),
                            "v_d": h.v_d,
                            "aadt": h.aadt,
                            "dcz_m": round(h.dcz_m, 2),
                            "h_m": round(h.h_m, 2),
                            "slope_h_per_v": round(h.ratio, 2),
                            "b_m": round(h.b_m, 2),
                            "la_m": round(h.la_m, 1),
                            "la2_m": round(h.la2_m, 1),
                        }
                        for h in why
                    ],
                }
            )
    return WarrantResult(spans, hazards, report, diags)


def warrant_inputs(
    model: RoadModel,
    dtm: GroundFn,
    projection: LocalProjection,
    route,
    way_id: np.ndarray,
    oneway: np.ndarray,
    *,
    decks: Sequence[tuple[float, float]] = (),
    features: RoadsideFeatures | None = None,
    s_range: tuple[float, float] | None = None,
    options: WarrantOptions | None = None,
) -> WarrantInputs:
    """`WarrantInputs` for a built road: per-station tags from the route's ways (`way_id` per
    station), joining ways from the route's stubs and `features`, deep water and rock from
    `features`. `route` is a `geo.router.Route` (read-only)."""
    ways = route.ways
    way_id = np.asarray(way_id, np.int64)
    tags = [ways.get(int(w), {}) for w in way_id]
    highway = np.array([t.get("highway", "") for t in tags])
    speed = np.array(
        [v if (v := parse_maxspeed(t.get("maxspeed"))) is not None else np.nan for t in tags],
        np.float64,
    )
    sidewalk = np.array(
        [
            t.get("sidewalk", "no") not in ("no", "none", "separate", "")
            or t.get("sidewalk:both", "no") not in ("no", "none", "separate", "")
            for t in tags
        ]
    )
    conns = connections_from_route(model, projection, route, features, options)
    return WarrantInputs(
        model=model,
        dtm=dtm,
        highway=highway,
        maxspeed_kmh=speed,
        sidewalk=sidewalk,
        oneway=np.asarray(oneway, np.int64),
        decks=list(decks),
        connections=conns,
        water=water_lines(features, projection),
        rock=rock_geoms(features, projection),
        s_range=s_range,
        driving_side=getattr(route, "driving_side", "right"),
    )
