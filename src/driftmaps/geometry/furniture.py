"""Street furniture: signs, hydro poles and wires, street lights, signals, hydrants, mailboxes,
and the road markings tied to them (stop bars, crosswalks, arrows).

**Where items come from** (`plan_furniture`), best first for this layer:

1. Mapillary detections (`geo.mapillary`): the user asked for Mapillary as the main source
   (2026-10-16): positions triangulated from street photos, the facing from `aligned_direction`
   (calibrated on the pilot route: 51 of 65 signs beside a road stand to the left of their
   `aligned_direction`, i.e. it is the direction the sign face points, towards the drivers who
   read it), and road paint (`marking--discrete--*`).
2. OSM nodes and power lines (`geo.street_features`).
3. Ontario rules, only where Mapillary has no photos (`coverage`): speed-limit signs where
   `maxspeed` changes and after major junctions, chevrons on sharp curves, hydro poles filling a
   measured pole line across unphotographed gaps. The junction control of the route's own
   approaches (`street_items.approach_stops`) is always applied: a stop sign where the route
   stops is never left out because a photo missed it, and the stop bar comes with it.
   Street lights along urban stretches (rural chevrons and urban lights: notes below).
4. Items other builders made (`items`, e.g. side-road stop signs, roundabout signs, mailboxes).
5. Signals (`geometry.signals`, after the pole lines): every approach of each signalised route
   junction gets its far-side corner pole, mast arm, heads per lane and pedestrian signals;
   stop and yield signs on signalised approaches, or by any OSM signal node, are dropped.

`dedupe` merges items of the same kind within `dedupe_m` on the same side of the road, keeping
the best source (`PRIORITY`). `geometry.clearance` keeps every item off the drivable surface
(the route and, with the network, side roads, junctions, roundabouts and driveways), drops
signs in fields and stray Mapillary poles, and faces measured signs to their road's traffic:
once before the pole lines and once after them (their gap-filling poles; wires follow).

**Placement numbers** (ASSUMED unless a source is given; docs/research/signs.md when it lands):
- Sign lateral offset: nearest plate edge 2.0 m beyond the edge of pavement (OTM Book 1b /
  TAC MUTCDC Div. A: 2.0 m minimum, 2.0-4.5 m typical on rural roads).
- Sign height: plate bottom 1.5 m above the edge of pavement on rural roads (OTM / TAC: 1.5 m
  minimum rural, 2.1 m minimum where pedestrians walk), stacked tabs below it.
- Sizes (OTM Book 5/6, "conventional road" columns): stop 750 mm (900 at 70 km/h and up),
  yield 750, warning diamonds 600 (750 at 70+), speed limit 600 x 750, chevron 450 x 600,
  keep right 600 x 750, all-way tab 600 x 200, street name blade 200 mm tall.
- Chevrons: on the outside of curves whose advisory speed (v = sqrt(127 R (e + f)),
  e = 0.06, f = 0.16) is at least 20 km/h below the limit and that turn 30 degrees or more;
  spacing by advisory speed from the MUTCD 2009 Table 2C-6 metric equivalents (12/24/36/48 m).
  Rule-made chevrons only on rural roads (`_Planner.rural_at`): a limit of 60 km/h or more
  (untagged: 50 on residential and service streets, 80 elsewhere, the Highway Traffic Act
  s.128 defaults), no sidewalk or `lit=yes` tag, and outside towns (`town_at`, from the
  buildings' context). Curvy suburban streets get none (the user's Oakville check,
  2026-10-16); photographed chevrons stay wherever they are.
- Hydro poles: Ontario wood distribution poles, 11 m above ground, 2.4 m crossarm 0.3 m below
  the top, three pin insulators; spans 30-70 m (gaps filled at the run's median span);
  wires sag 2 % of the span.
- Street lights (`_Planner.street_lights`), on urban stretches only (`lit=yes`, a sidewalk tag,
  or in a town; never `lit=no`): Ontario cobra heads on 8.2 m galvanized poles 1 m behind the
  road's edge, every 40 m on arterials (primary / secondary), 45 m on collectors (tertiary)
  and 50 m on local streets (TAC Guide for the Design of Roadway Lighting, 2006: spacing
  about 3.5-5 mounting heights for the usual staggered / one-sided layouts; ASSUMED within
  that range, typical of Halton municipal standards). One side on roads narrower than 14 m
  (the side with the hydro poles, whose poles then carry the lights, else the right), both
  sides opposite on wider ones. None within 10 m of a junction's crossing road (corners: the
  signal poles carry their own luminaires) or on a driveway (moved up to 6 m along the road,
  else left out); a photographed or OSM light on that side within 0.6 spacings takes the
  place of a rule-made one.

**Meshes** (`furniture_meshes`): low-poly, merged per `cell_m` cell into at most four nodes:
`furn_<cell>` (posts, poles, backs, lights; atlas, `lod_out_far_m`), `signs_<cell>` (sign faces,
bright material, `lod_out_m`), `wires_<cell>` and hidden `1WALL_furn_<cell>` octagonal
cylinders around poles inside the wall corridor. Markings are `marks_<cell>` paint decals.
Coordinates: AC world (X east, Y up, Z south).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from driftmaps.ac.furniture_textures import (
    MAX_NAMES,
    SPEEDS,
    WOOD,
    name_tile,
    sign_tile,
    swatch_uv,
)
from driftmaps.geo.barriers import _project
from driftmaps.geo.project import LocalProjection
from driftmaps.geometry.clearance import CLEARANCE_M, DrivableSurface, clear_items
from driftmaps.geometry.primitives import MeshData, merge, split
from driftmaps.geometry.road import RoadModel
from driftmaps.geometry.signals import SignalApproach, SignalPlan, draw_pole, plan_signals
from driftmaps.geometry.street_items import (
    HYDRANT,
    LIGHT_STREET,
    MAILBOX,
    POLE_UTILITY,
    SIGN_ALL_WAY,
    SIGN_CHEVRON_LEFT,
    SIGN_CHEVRON_RIGHT,
    SIGN_DEAD_END,
    SIGN_KEEP_RIGHT,
    SIGN_ROUNDABOUT_AHEAD,
    SIGN_SPEED_LIMIT,
    SIGN_STOP,
    SIGN_STOP_AHEAD,
    SIGN_STREET_NAME,
    SIGN_YIELD,
    SIGNAL_TRAFFIC,
    StreetItem,
    approach_stops,
    highway_rank,
)

__all__ = [
    "PRIORITY",
    "FurnitureMesh",
    "FurnitureOptions",
    "FurniturePlan",
    "Marking",
    "dedupe",
    "furniture_meshes",
    "mapillary_kind",
    "marking_meshes",
    "parse_speed",
    "plan_furniture",
]

GroundFn = Callable[[np.ndarray], np.ndarray]
# This layer's source order: Mapillary first, as the user asked (2026-10-16); the shared
# `street_items.SOURCE_PRIORITY` puts OSM first. OSM stop/give-way nodes sit on the road's
# centreline, so a photographed sign gives the better position anyway.
PRIORITY = ("municipal", "mapillary", "osm", "rule")
# Streets that are urban without a `maxspeed` tag (50 km/h, HTA s.128), and the sidewalk tags
# that make a road urban (chevrons: module notes).
_URBAN_HIGHWAYS = frozenset({"residential", "living_street", "service", "pedestrian"})
_SIDEWALK_KEYS = ("sidewalk", "sidewalk:both", "sidewalk:left", "sidewalk:right")
_SIDEWALK_YES = frozenset({"both", "left", "right", "yes", "separate"})
# Street lights (module notes): spacing by road class, both sides from this half width.
_LIGHT_SPACING_M = {"motorway": 40.0, "trunk": 40.0, "primary": 40.0, "secondary": 40.0,
                    "tertiary": 45.0}  # fmt: skip
_LIGHT_SPACING_LOCAL_M = 50.0
LIGHT_BOTH_SIDES_HALF_M = 7.0
LIGHT_BEHIND_EDGE_M = 1.0
LIGHT_CORNER_M = 10.0
# A pole with no reliable position across the road stands this far beyond the gravel shoulder
# (ASSUMED: rural Ontario pole lines run near the right-of-way limit, ~8 m from the centre of
# a 7 m road with 1 m shoulders).
POLE_EDGE_M = 3.5


@dataclass
class FurnitureOptions:
    dedupe_m: float = 12.0  # same kind, same side of the road, this close: one item
    pole_dedupe_m: float = 8.0  # poles stand >= 30 m apart; two detections of one pole are not
    sign_edge_gap_m: float = 2.0  # nearest plate edge beyond the edge of pavement
    sign_bottom_m: float = 1.5  # plate bottom above the edge of pavement (rural)
    max_offset_m: float = 40.0  # items farther from the centreline are ignored
    coverage_m: float = 40.0  # a photo this close to a station: Mapillary saw the stretch
    min_curve_deflection_deg: float = 30.0
    chevron_speed_drop_kmh: float = 20.0
    pole_height_m: float = 11.0
    pole_span_m: tuple[float, float] = (30.0, 70.0)
    pole_join_m: float = 600.0  # join two pole runs on one side across an unphotographed gap
    wire_sag: float = 0.02
    cell_m: float = 250.0
    lod_out_m: float = 450.0  # sign faces, hydrants, mailboxes
    lod_out_far_m: float = 1200.0  # poles, wires, lights
    collide: bool = True


@dataclass
class Marking:
    """A road marking on the route: `kind` "stop_bar", "crosswalk", "zebra" or "arrow:<dir>",
    at station `s`, between lateral offsets `l0` < `l1` (+ = left), from `source`."""

    kind: str
    s: float
    l0: float
    l1: float
    source: str = "rule"
    forward: bool = True  # arrows: pointing along the route's direction


@dataclass
class FurniturePlan:
    items: list[StreetItem] = field(default_factory=list)
    wires: list[np.ndarray] = field(default_factory=list)  # (2, 3) attachment pairs per span
    markings: list[Marking] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)  # kind -> source -> n
    osm_nodes: list[int] = field(default_factory=list)  # OSM node ids used (for the geojson)
    osm_lines: list[int] = field(default_factory=list)
    mapillary_used: int = 0
    covered_fraction: float = 0.0
    controls: list[dict] = field(default_factory=list)  # route approaches and their control
    driveways: list[tuple[float, float]] = field(default_factory=list)  # Mapillary (lon, lat)
    # `geometry.clearance`: items moved off / dropped, signs re-faced, and what still stands
    # on the drivable surface ("on_surface", "near_surface": within CLEARANCE_M; both 0)
    clearance: dict[str, int] = field(default_factory=dict)
    signals: SignalPlan = field(default_factory=SignalPlan)  # signalised junctions, poles
    stops_under_signals: int = 0  # stop / yield signs dropped from signalised approaches
    signal_stations: list[float] = field(default_factory=list)  # metres from the start (A)


@dataclass
class FurnitureMesh:
    name: str
    material: str  # "furniture" | "signs" | "hidden"
    mesh: MeshData
    flags: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------

_SPEED_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(mph|km/h|kmh|kph)?\s*$", re.I)


def parse_speed(text: str | None) -> int | None:
    """OSM `maxspeed` -> km/h (rounded to 5), or None ("signals", "none", ...)."""
    if not text:
        return None
    m = _SPEED_RE.match(str(text).split(";")[0])
    if not m:
        return None
    v = float(m.group(1)) * (1.609344 if (m.group(2) or "").lower() == "mph" else 1.0)
    return int(round(v / 5.0) * 5) if v > 0 else None


def _unit2(v) -> np.ndarray:
    v = np.asarray(v, np.float64)[:2]
    n = float(np.hypot(*v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _heading_xz(compass_deg: float) -> np.ndarray:
    """A compass heading (0 = north, 90 = east) as an AC (x, z) unit vector (z = south)."""
    a = math.radians(compass_deg)
    return np.array([math.sin(a), -math.cos(a)])


def _source_rank(source: str) -> int:
    return PRIORITY.index(source) if source in PRIORITY else len(PRIORITY)


def _dedupe_group(kind: str) -> str:
    """Kinds that are one thing for de-duplication (a speed sign is one sign, whatever the
    number reads)."""
    return SIGN_SPEED_LIMIT if kind.startswith(SIGN_SPEED_LIMIT) else kind


# Mapillary class -> item kind (None: not rendered). Speed limits: `regulatory` only (a
# `complementary` speed is an advisory tab under a curve sign).
_MLY_PREFIX: tuple[tuple[str, str], ...] = (
    ("regulatory--stop--", SIGN_STOP),
    ("regulatory--yield--", SIGN_YIELD),
    ("regulatory--all-way--", SIGN_ALL_WAY),
    ("warning--stop-ahead--", SIGN_STOP_AHEAD),
    ("complementary--chevron-left--", SIGN_CHEVRON_LEFT),
    ("complementary--chevron-right--", SIGN_CHEVRON_RIGHT),
    ("regulatory--keep-right--", SIGN_KEEP_RIGHT),
    ("warning--roundabout--", SIGN_ROUNDABOUT_AHEAD),
    ("warning--dead-end--", SIGN_DEAD_END),
    ("information--dead-end--", SIGN_DEAD_END),
    ("object--support--utility-pole", POLE_UTILITY),
    ("object--street-light", LIGHT_STREET),
    ("object--fire-hydrant", HYDRANT),
    ("object--mailbox", MAILBOX),
)
_MLY_SPEED = re.compile(r"^regulatory--maximum-speed-limit-(\d+)--")


def mapillary_kind(value: str) -> str | None:
    """The StreetItem kind of a Mapillary class, or None."""
    m = _MLY_SPEED.match(value)
    if m:
        return f"{SIGN_SPEED_LIMIT}:{int(m.group(1))}"
    for prefix, kind in _MLY_PREFIX:
        if value.startswith(prefix):
            return kind
    return None


def _is_sign(kind: str) -> bool:
    return kind.startswith("sign:")


# --------------------------------------------------------------------------------------------
# The road frame
# --------------------------------------------------------------------------------------------


class _Frame:
    """Stations and lateral offsets on the built centreline."""

    def __init__(self, model: RoadModel, s_range: tuple[float, float]) -> None:
        self.m = model
        self.xz = np.ascontiguousarray(model.centre[:, [0, 2]])
        self.s0, self.s1 = s_range
        self.tree = cKDTree(self.xz)

    def project(self, xz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(station, lateral (+ left), inside) of (M, 2) XZ points."""
        pts = np.asarray(xz, float).reshape(-1, 2)
        if len(pts) == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0, bool)
        return _project(pts, self.xz, self.m.s, self.tree)

    def side(self, item: StreetItem, reach: float = 30.0) -> int:
        """-1 right, +1 left of the route; 0 beyond it or farther than `reach`."""
        _, off, inside = self.project(np.array([[item.position[0], item.position[2]]]))
        return int(np.sign(off[0])) if inside[0] and abs(off[0]) < reach else 0

    def at(self, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(centre xyz, forward xz unit, left xz unit) at station s."""
        pos, fwd, left = self.m.at(float(np.clip(s, self.m.s[0], self.m.s[-1])))
        return pos, _unit2(fwd[[0, 2]]), _unit2(left[[0, 2]])

    def half(self, s: float) -> float:
        return self.m.interp(self.m.half_width, s)

    def edge(self, s: float) -> float:
        """Half width + gravel shoulder at s."""
        return self.m.interp(self.m.half_width + self.m.shoulder, s)

    def y(self, s: float) -> float:
        return float(self.m.at(float(np.clip(s, self.m.s[0], self.m.s[-1])))[0][1])

    def point(self, s: float, lateral: float) -> np.ndarray:
        pos, _, left = self.at(s)
        return pos[[0, 2]] + left * lateral


# --------------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------------


def _xz_of(projection: LocalProjection, lonlat: np.ndarray) -> np.ndarray:
    xy = projection.to_xy(np.asarray(lonlat, float).reshape(-1, 2))
    return np.stack([xy[:, 0], -xy[:, 1]], axis=1)


def _item(kind, xz, facing, ground: GroundFn, source, text="", ref="") -> StreetItem:
    xz = np.asarray(xz, float)
    y = float(ground(xz.reshape(1, 2))[0])
    f = _unit2(facing)
    return StreetItem(kind, (float(xz[0]), y, float(xz[1])), (float(f[0]), float(f[1])), source,
                      text, ref)  # fmt: skip


@dataclass
class _Junction:
    node: int
    s: float
    in_way: int
    out_way: int
    legs: list[tuple[int, np.ndarray]]  # (side way, direction xz unit away from the node)


def _junctions(route, frame: _Frame, projection: LocalProjection) -> list[_Junction]:
    """Interior route nodes with side roads (`route.stubs`), in driving order."""
    path = route.node_path()
    side: dict[int, list[tuple[int, int]]] = {}
    for node, way, nb in route.stubs:
        side.setdefault(int(node), []).append((int(way), int(nb)))
    out = []
    for k in range(1, len(path) - 1):
        node = path[k]
        if node not in side or node not in route.nodes:
            continue
        p = _xz_of(projection, np.array(route.nodes[node]))[0]
        st, _, inside = frame.project(p[None, :])
        if not inside[0] or not frame.s0 - 1.0 <= st[0] <= frame.s1 + 1.0:
            continue
        legs = []
        for way, nb in side[node]:
            if nb in route.nodes:
                q = _xz_of(projection, np.array(route.nodes[nb]))[0]
                legs.append((way, _unit2(q - p)))
        out.append(_Junction(node, float(st[0]), route.edges[k - 1].way, route.edges[k].way, legs))
    return out


def _highway(route, way: int) -> str:
    return str((route.ways.get(way) or {}).get("highway", ""))


def _road_half(tags: Mapping[str, str] | None) -> float:
    """Half the width of a road from its tags (lanes x 3.5 m), 3.5 m by default."""
    tags = tags or {}
    try:
        w = float(str(tags.get("width", "")).split()[0])
        if 3.0 <= w <= 30.0:
            return w / 2
    except (ValueError, IndexError):
        pass
    try:
        lanes = int(str(tags.get("lanes", "")).split(";")[0])
        if 1 <= lanes <= 8:
            return max(lanes * 3.5, 5.0) / 2
    except ValueError:
        pass
    return 3.5


class _Planner:
    def __init__(self, model, projection, route, s_range, ground, way_id, options, coverage,
                 town_at=None):  # fmt: skip
        self.town_at = town_at  # XZ (2,) -> True in a town (subdivision, downtown)
        self.m = model
        self.proj = projection
        self.route = route
        self.frame = _Frame(model, s_range)
        self.ground = ground
        self.way_id = way_id
        self.o = options
        self.items: list[StreetItem] = []
        self.markings: list[Marking] = []
        self.plan = FurniturePlan()
        self.mly_control: list[tuple[int, float]] = []  # (item index, station) of stop signs
        self.covered = self._coverage(coverage)

    # -- coverage -----------------------------------------------------------------------------

    def _coverage(self, pts_lonlat: np.ndarray | None) -> np.ndarray:
        """Per station: Mapillary has a photo within `coverage_m` (smoothed over 50 m)."""
        n = len(self.m.s)
        if pts_lonlat is None or len(pts_lonlat) == 0:
            return np.zeros(n, bool)
        from scipy.spatial import cKDTree

        xz = _xz_of(self.proj, pts_lonlat)
        d, _ = cKDTree(xz).query(self.frame.xz, distance_upper_bound=self.o.coverage_m)
        near = np.isfinite(d)
        k = 25  # ~50 m window at the 1-2 m station spacing
        smooth = np.convolve(near.astype(float), np.ones(2 * k + 1), mode="same") > 0
        return smooth

    def covered_at(self, s: float) -> bool:
        return bool(self.covered[self.m.index_at(s)])

    # -- helpers ------------------------------------------------------------------------------

    def side_sign(self, kind, s, side, facing_fwd: bool, source="rule", text="", post_of=None):
        """A sign beside the route at station s: side +1 = left, -1 = right of the route;
        `facing_fwd` True: it faces drivers going the other way (its face points along +fwd).
        `post_of`: stand on the post of that kind of sign (a tab under its main sign)."""
        _, fwd, _ = self.frame.at(s)
        w = _plate_size(post_of or kind, self._speed_at(s))[0]
        lat = side * (self.frame.half(s) + self.o.sign_edge_gap_m + w / 2)
        facing = fwd if facing_fwd else -fwd
        it = _item(kind, self.frame.point(s, lat), facing, self.ground, source, text)
        self.items.append(it)
        return it

    def _speed_at(self, s: float) -> int | None:
        return parse_speed(self._tags_at(s).get("maxspeed"))

    def _tags_at(self, s: float) -> Mapping[str, str]:
        if len(self.way_id) != len(self.m.s):
            return {}
        wid = int(self.way_id[self.m.index_at(s)])
        return self.route.ways.get(wid) or {}

    def rural_at(self, s: float) -> bool:
        """A rural road at station s (module notes, "Chevrons")."""
        tags = self._tags_at(s)
        v = parse_speed(tags.get("maxspeed"))
        if v is None:
            v = 50 if tags.get("highway") in _URBAN_HIGHWAYS else 80
        if v < 60 or tags.get("lit") == "yes":
            return False
        if any(tags.get(k) in _SIDEWALK_YES for k in _SIDEWALK_KEYS):
            return False
        return not (self.town_at is not None and self.town_at(self.frame.point(s, 0.0)))

    def urban_at(self, s: float) -> bool:
        """A lit (urban) stretch at station s (module notes, "Street lights")."""
        tags = self._tags_at(s)
        lit = str(tags.get("lit", ""))
        if lit == "no":
            return False
        if lit == "yes" or any(tags.get(k) in _SIDEWALK_YES for k in _SIDEWALK_KEYS):
            return True
        return self.town_at is not None and bool(self.town_at(self.frame.point(s, 0.0)))

    # -- street lights ----------------------------------------------------------------------------

    def street_lights(self, juncs: list[_Junction], surface: DrivableSurface | None) -> None:
        """Rule-made street lights along the urban stretches (module notes)."""
        f = self.frame
        meas: dict[int, list[float]] = {-1: [], 1: []}
        poles: dict[int, list[float]] = {-1: [], 1: []}
        its = [it for it in self.items if it.kind in (LIGHT_STREET, POLE_UTILITY)]
        if its:
            st, off, inside = f.project(np.array([(it.position[0], it.position[2]) for it in its]))
            for it, s_, o_, ins in zip(its, st, off, inside, strict=True):
                if ins and abs(o_) <= 25.0:
                    (meas if it.kind == LIGHT_STREET else poles)[1 if o_ > 0 else -1].append(
                        float(s_)
                    )
        meas = {k: np.sort(v) for k, v in meas.items()}
        poles = {k: np.sort(v) for k, v in poles.items()}
        corners = [(j.s, max([_road_half(self.route.ways.get(w)) for w, _ in j.legs], default=3.5))
                   for j in juncs]  # fmt: skip
        last = {-1: -1e9, 1: -1e9}
        s = f.s0
        while s <= f.s1:
            if not self.urban_at(s) or any(abs(s - js) < h + LIGHT_CORNER_M for js, h in corners):
                s += 5.0
                continue
            hw = (self._tags_at(s).get("highway") or "").removesuffix("_link")
            spacing = _LIGHT_SPACING_M.get(hw, _LIGHT_SPACING_LOCAL_M)
            if f.half(s) >= LIGHT_BOTH_SIDES_HALF_M:
                sides = (-1, 1)
            else:
                near = {k: int(np.sum(np.abs(v - s) < 100.0)) for k, v in poles.items()}
                sides = (1,) if near[1] > near[-1] else (-1,)
            for side in sides:
                if s - last[side] < spacing:
                    continue
                m_ = meas[side]
                if len(m_) and float(np.min(np.abs(m_ - s))) < 0.6 * spacing:
                    last[side] = s  # a measured light stands here
                    continue
                at = s
                p_ = poles[side]
                if len(p_) and float(np.min(np.abs(p_ - s))) < 0.4 * spacing:
                    at = float(p_[np.argmin(np.abs(p_ - s))])  # hung on the hydro pole
                p = self._light_spot(at, side, surface)
                if p is None:
                    continue
                facing = -side * f.at(at)[2]
                self.items.append(_item(LIGHT_STREET, p, facing, self.ground, "rule"))
                last[side] = at
            s += 5.0

    def _light_spot(self, s: float, side: int, surface: DrivableSurface | None):
        """Where a light stands at station s (shifted up to 6 m off a driveway), or None."""
        for ds in (0.0, 3.0, -3.0, 6.0, -6.0):
            q = s + ds
            if not self.frame.s0 <= q <= self.frame.s1:
                continue
            p = self.frame.point(q, side * (self.frame.edge(q) + LIGHT_BEHIND_EDGE_M))
            if surface is None or float(surface.distance(p[None, :])[0]) >= 0.9:
                return p
        return None

    # -- measured items -----------------------------------------------------------------------

    def add_mapillary(self, features) -> None:
        if not features:
            return
        ll = np.array([(f.lon, f.lat) for f in features])
        xz = _xz_of(self.proj, ll)
        st, off, inside = self.frame.project(xz)
        used = 0
        for f, p, s, lat, ins in zip(features, xz, st, off, inside, strict=True):
            if f.value == "construction--flat--driveway":
                self.plan.driveways.append((round(f.lon, 7), round(f.lat, 7)))
                continue
            if f.value.startswith("marking--discrete--"):
                if ins and abs(lat) < self.frame.half(s) + 0.5:
                    self._mly_marking(f.value, float(s), float(lat))
                continue
            kind = mapillary_kind(f.value)
            if kind is None or abs(lat) > self.o.max_offset_m:
                continue
            if f.aligned_direction is not None:
                facing = _heading_xz(f.aligned_direction)
            else:
                facing = -np.sign(lat or 1.0) * self.frame.at(s)[2]  # towards the road
            pos = p
            if ins and _is_sign(kind):
                pos, facing = self._tidy_sign(kind, float(s), float(lat), facing)
            elif ins and abs(lat) < self.frame.edge(s) + 0.6:  # never on the road itself
                pos = self.frame.point(s, np.sign(lat or 1.0) * (self.frame.edge(s) + 0.6))
            item = _item(kind, pos, facing, self.ground, "mapillary", ref=f"mly:{f.id}")
            if ins and kind in (SIGN_STOP, SIGN_YIELD, SIGN_ALL_WAY) and abs(lat) < 20.0:
                self.mly_control.append((len(self.items), float(s)))
            self.items.append(item)
            used += 1
        self.plan.mapillary_used = used

    def _tidy_sign(self, kind, s, lat, facing) -> tuple[np.ndarray, np.ndarray]:
        """A measured sign beside the route: facing straight along the road if it is within
        35 degrees of it, and out of the road (plate edge >= 0.5 m past the shoulder). A sign
        triangulated onto the road (Mapillary's positions are a few metres off) goes to the
        right of the drivers it faces (Ontario signs stand on the right)."""
        _, fwd, _ = self.frame.at(s)
        d = float(facing @ fwd)
        along = abs(d) > math.cos(math.radians(35.0))
        if along:
            facing = fwd * np.sign(d)
        w = _plate_size(kind, None)[0]
        lo = self.frame.edge(s) + 0.5 + w / 2
        if abs(lat) < lo:
            side = (1.0 if d > 0 else -1.0) if along else (1.0 if lat >= 0 else -1.0)
            lat = side * max(lo, self.frame.half(s) + self.o.sign_edge_gap_m + w / 2)
        return self.frame.point(s, lat), facing

    def _mly_marking(self, value: str, s: float, lat: float) -> None:
        half = self.frame.half(s)
        name = value[len("marking--discrete--") :].split("--g")[0]
        mine = lat < 0  # right of the centreline: the route's own direction
        l0, l1 = (-half + 0.2, -0.2) if mine else (0.2, half - 0.2)
        if name == "stop-line":
            self.markings.append(Marking("stop_bar", s, l0, l1, "mapillary"))
        elif name in ("crosswalk-zebra",):
            self.markings.append(Marking("zebra", s, -half, half, "mapillary"))
        elif name.startswith("arrow--"):
            what = name[len("arrow--") :]
            if what in ("straight", "left", "right"):
                c = (l0 + l1) / 2
                self.markings.append(
                    Marking(f"arrow:{what}", s, c - 0.5, c + 0.5, "mapillary", forward=mine)
                )

    def add_osm(self, feats) -> None:
        if feats is None:
            return
        nodes = list(feats.nodes)
        if nodes:
            xz = _xz_of(self.proj, np.array([(n.lon, n.lat) for n in nodes]))
            st, off, inside = self.frame.project(xz)
        for k, n in enumerate(nodes):
            t = n.tags
            s, lat, ins = float(st[k]), float(off[k]), bool(inside[k])
            ref = f"osm:n{n.id}"
            kind = None
            if t.get("highway") == "street_lamp":
                kind = LIGHT_STREET
            elif t.get("power") == "pole":
                kind = POLE_UTILITY
            elif t.get("emergency") == "fire_hydrant":
                kind = HYDRANT
            elif t.get("highway") == "crossing" and ins and abs(lat) < 2.0:
                if t.get("crossing") not in ("unmarked", "no") and t.get(
                    "crossing:markings"
                ) not in ("no",):
                    zebra = t.get("crossing") in ("zebra", "marked", "uncontrolled") or t.get(
                        "crossing:markings", ""
                    ).startswith(("zebra", "ladder"))
                    h = self.frame.half(s)
                    self.markings.append(
                        Marking("zebra" if zebra else "crosswalk", s, -h, h, "osm")
                    )
                    self.plan.osm_nodes.append(n.id)
                continue
            if kind is None:
                continue
            p = xz[k]
            if ins and abs(lat) < self.frame.edge(s) + 0.6:
                p = self.frame.point(s, math.copysign(self.frame.edge(s) + 0.6, lat or 1.0))
            facing = -np.sign(lat or 1.0) * self.frame.at(s)[2] if ins else np.array([1.0, 0.0])
            self.items.append(_item(kind, p, facing, self.ground, "osm", ref=ref))
            self.plan.osm_nodes.append(n.id)
        # Poles along OSM power lines: every vertex near the road.
        for line in feats.power_lines:
            xz_l = _xz_of(self.proj, line.lonlat)
            st_l, off_l, ins_l = self.frame.project(xz_l)
            keep = ins_l & (np.abs(off_l) <= self.o.max_offset_m)
            if not keep.any():
                continue
            self.plan.osm_lines.append(line.id)
            for p, s, lat in zip(xz_l[keep], st_l[keep], off_l[keep], strict=True):
                if abs(lat) < self.frame.edge(s) + 0.6:
                    continue  # where the line crosses the road
                facing = -np.sign(lat) * self.frame.at(s)[2]
                self.items.append(_item(POLE_UTILITY, p, facing, self.ground, "osm",
                                        ref=f"osm:w{line.id}"))  # fmt: skip

    # -- junction control on the route's own approaches -----------------------------------

    def junction_controls(self, feats, mly) -> None:
        juncs = _junctions(self.route, self.frame, self.proj)
        osm_ctl = (
            _osm_node_controls(feats, self.proj, self.frame, self.route)
            if feats is not None
            else []
        )
        claimed: set[int] = set()
        for j in juncs:
            in_hw = _highway(self.route, j.in_way)
            out_hw = _highway(self.route, j.out_way)
            sides = [_highway(self.route, w) for w, _ in j.legs]
            # Our approach (driving A -> B into the node) and the opposite one (B -> A, coming
            # back along the outgoing way).
            for forward, hw, others in ((True, in_hw, [out_hw, *sides]),
                                        (False, out_hw, [in_hw, *sides])):  # fmt: skip
                mapped = _mapped(j, forward, osm_ctl)
                # Photographed stop / yield / all-way signs standing on this approach (before
                # the node as its drivers see it) and facing along the road: they are the
                # approach's signs, whatever side Mapillary put them on.
                mine = []
                for idx, s in self.mly_control:
                    if idx in claimed:
                        continue
                    ok = j.s - 45.0 <= s < j.s if forward else j.s < s <= j.s + 45.0
                    it = self.items[idx]
                    if ok and abs(float(np.dot(it.facing, self.frame.at(s)[1]))) > 0.7:
                        mine.append(idx)
                kinds = {self.items[i].kind for i in mine}
                if SIGN_ALL_WAY in kinds or mapped == "all_way":
                    mapped = "all_way"
                elif SIGN_STOP in kinds:
                    mapped = mapped or "stop"
                elif SIGN_YIELD in kinds:
                    mapped = mapped or "yield"
                ctl = approach_stops(hw, others, mapped=mapped)
                self.plan.controls.append({"node": j.node, "s": round(j.s, 1), "forward": forward,
                                           "approach": hw, "others": others,
                                           "mapped": mapped, "control": ctl})  # fmt: skip
                if ctl is None:
                    continue
                cross_half = max(
                    (_road_half(self.route.ways.get(w)) for w in [j.out_way if forward else
                     j.in_way] + [w for w, _ in j.legs]),
                    default=3.5,
                )  # fmt: skip
                gap = cross_half + 1.5  # OTM Book 11: stop line 1-3 m short of the crossing road
                s_bar = j.s - gap if forward else j.s + gap
                side = -1 if forward else 1
                if ctl in ("stop", "yield"):
                    want = [SIGN_STOP if ctl == "stop" else SIGN_YIELD]
                    if mapped == "all_way":
                        want.append(SIGN_ALL_WAY)
                    for kind in want:
                        found = [i for i in mine if self.items[i].kind == kind]
                        src = "mapillary" if found else "rule"
                        placed = self.side_sign(
                            kind, s_bar, side, not forward, source=src, post_of=want[0]
                        )
                        if found:  # the photographed sign, moved to the stop line
                            self.items.pop()
                            old = self.items[found[0]]
                            self.items[found[0]] = StreetItem(
                                kind, placed.position, placed.facing, old.source, old.text, old.ref
                            )
                            claimed.update(found)
                    claimed.update(i for i in mine if self.items[i].kind in want)
                if ctl in ("stop", "signal"):
                    h = self.frame.half(s_bar)
                    l0, l1 = (-h + 0.2, -0.2) if forward else (0.2, h - 0.2)
                    self.markings.append(Marking("stop_bar", s_bar, l0, l1, "rule"))
        # OSM stop / give-way nodes on the route away from any routed junction (a side road
        # the router does not know, a rail crossing): the node is the stop line; its
        # `direction` says who stops.
        for _, s, lat, ctl, fwd in osm_ctl:
            if fwd is None or ctl not in ("stop", "yield") or abs(lat) > 3.0:
                continue
            if not self.frame.s0 <= s <= self.frame.s1:
                continue
            if any(abs(s - j.s) < 40.0 for j in juncs):
                continue
            side = -1 if fwd else 1
            self.side_sign(SIGN_STOP if ctl == "stop" else SIGN_YIELD, s, side, not fwd, "osm")
            if ctl == "stop":
                h = self.frame.half(s)
                l0, l1 = (-h + 0.2, -0.2) if fwd else (0.2, h - 0.2)
                self.markings.append(Marking("stop_bar", s, l0, l1, "osm"))

    def signal_junctions(self, juncs: list[_Junction]):
        """(node, centre XZ, legs) of every route junction for `signals.plan_signals`: the
        route's two legs (widths from the built road, lanes from their ways and the
        direction the route runs along them) and the side roads (widths from their tags)."""
        path = self.route.node_path()
        at = {node: k for k, node in enumerate(path)}
        out = []
        for j in juncs:
            k = at.get(j.node)
            if k is None or not 0 < k < len(path) - 1:
                continue
            e_in, e_out = self.route.edges[k - 1], self.route.edges[k]
            _, fwd, _ = self.frame.at(j.s)
            legs = []
            for edge, d, s_leg, arrive_dir in ((e_in, -fwd, j.s - 15.0, 1),
                                               (e_out, fwd, j.s + 15.0, -1)):  # fmt: skip
                tags = self.route.ways.get(edge.way) or {}
                toward = edge.dir * arrive_dir > 0  # arriving traffic moves along the way
                ow = str(tags.get("oneway", ""))
                oneway = 0
                if ow in ("yes", "1", "true", "-1"):
                    flows = toward if ow != "-1" else not toward
                    oneway = -1 if flows else 1
                legs.append(SignalApproach(np.asarray(d, float), self.frame.half(s_leg), tags,
                                           toward, oneway, on_route=True))  # fmt: skip
            for way, d in j.legs:
                tags = self.route.ways.get(way) or {}
                legs.append(SignalApproach(np.asarray(d, float), _road_half(tags), tags))
            out.append((j.node, self.frame.point(j.s, 0.0), legs))
        return out

    # -- speed limits ---------------------------------------------------------------------------

    def speed_signs(self, juncs: list[_Junction]) -> None:
        """Rb-1 where `maxspeed` changes (both directions) and 100 m after junctions with a road
        of the same or a higher class, or where the route turns (OTM Book 5: confirm the limit
        after major intersections; ASSUMED 100 m). Only where Mapillary has no photos."""
        if len(self.way_id) != len(self.m.s):
            return
        lo = self.m.index_at(self.frame.s0)
        hi = self.m.index_at(self.frame.s1)
        ways = self.way_id[lo : hi + 1]
        speeds = np.array([parse_speed((self.route.ways.get(int(w)) or {}).get("maxspeed")) or 0
                           for w in ways])  # fmt: skip
        change = np.flatnonzero(np.diff(speeds) != 0)
        for c in change.tolist():
            s = float(self.m.s[lo + c + 1])
            before, after = int(speeds[c]), int(speeds[c + 1])
            if not self.covered_at(s):
                if after in SPEEDS:
                    self.side_sign(f"{SIGN_SPEED_LIMIT}:{after}", s, -1, facing_fwd=False)
                if before in SPEEDS:
                    self.side_sign(f"{SIGN_SPEED_LIMIT}:{before}", s, 1, facing_fwd=True)
        for j in juncs:
            rank = highway_rank(_highway(self.route, j.in_way))
            major = j.in_way != j.out_way or any(
                highway_rank(_highway(self.route, w)) >= rank for w, _ in j.legs
            )
            if not major:
                continue
            for s, side, fwd in ((j.s + 100.0, -1, False), (j.s - 100.0, 1, True)):
                if not self.frame.s0 <= s <= self.frame.s1 or self.covered_at(s):
                    continue
                v = self._speed_at(s)
                if v in SPEEDS:
                    self.side_sign(f"{SIGN_SPEED_LIMIT}:{v}", s, side, facing_fwd=fwd)

    # -- chevrons -------------------------------------------------------------------------------

    def chevrons(self) -> None:
        """Chevron alignment signs on the outside of sharp curves (see the module notes),
        back to back for both directions. Only where Mapillary has no photos."""
        m = self.m
        k = np.abs(m.curvature)
        lo, hi = self.m.index_at(self.frame.s0), self.m.index_at(self.frame.s1)
        i = lo
        while i < hi:
            if k[i] < 1.0 / 400.0:
                i += 1
                continue
            j = i
            while (
                j < hi
                and k[j] >= 1.0 / 400.0
                and np.sign(m.curvature[j]) == np.sign(m.curvature[i])
            ):
                j += 1
            seg = slice(i, j)
            s_a, s_b = float(m.s[i]), float(m.s[j - 1])
            defl = math.degrees(float(np.abs(np.trapezoid(m.curvature[seg], m.s[seg]))))
            r_min = 1.0 / max(float(k[seg].max()), 1e-6)
            i = j
            v = self._speed_at((s_a + s_b) / 2) or 80
            v_adv = math.sqrt(127.0 * r_min * (0.06 + 0.16))
            if defl < self.o.min_curve_deflection_deg or v_adv > v - self.o.chevron_speed_drop_kmh:
                continue
            if not (self.rural_at(s_a) and self.rural_at(s_b) and self.rural_at((s_a + s_b) / 2)):
                continue
            spacing = (
                12.0 if v_adv <= 25 else 24.0 if v_adv <= 50 else 36.0 if v_adv <= 70 else 48.0
            )
            tight = np.flatnonzero(k[seg] >= 0.5 / r_min) + seg.start
            t0, t1 = float(m.s[tight[0]]), float(m.s[tight[-1]])
            n = max(3, int(round((t1 - t0) / spacing)) + 1)
            mid = (t0 + t1) / 2
            left_turn = m.curvature[seg].mean() > 0
            outside = -1 if left_turn else 1  # outside of a left turn: the right side
            for s in mid + (np.arange(n) - (n - 1) / 2) * spacing:
                if self.covered_at(float(s)):
                    continue
                # Ours: facing back at us, pointing the way we turn; theirs on the same post.
                ours = SIGN_CHEVRON_LEFT if left_turn else SIGN_CHEVRON_RIGHT
                theirs = SIGN_CHEVRON_RIGHT if left_turn else SIGN_CHEVRON_LEFT
                self.side_sign(ours, float(s), outside, facing_fwd=False)
                self.side_sign(theirs, float(s), outside, facing_fwd=True)

    # -- hydro poles ------------------------------------------------------------------------------

    def _settle_poles(self, poles, st, off, inside) -> None:
        """Poles placed on the road or its shoulder (`edge + 1 m`): Mapillary often
        triangulates a pole onto the road (on the pilot route 51 of the 78 poles within 12 m
        of the centreline), while its station and side are good (they form steady lines).
        Such a pole keeps its side and takes the offset of the reliable roadside poles on that
        side within 200 m (1-12 m beyond the shoulder), else `edge + POLE_EDGE_M`. In place:
        `poles`, `off`."""
        edge = np.array([self.frame.edge(float(s)) for s in st])
        reliable = inside & (np.abs(off) >= edge + 1.0) & (np.abs(off) <= edge + 12.0)
        for i in np.flatnonzero(inside & (np.abs(off) < edge + 1.0)).tolist():
            side = float(np.sign(off[i])) or -1.0
            mates = reliable & (np.abs(st - st[i]) < 200.0) & (np.sign(off) == side)
            lat = float(np.median(np.abs(off[mates]))) if mates.any() else edge[i] + POLE_EDGE_M
            off[i] = side * lat
            p = poles[i]
            q = self.frame.point(float(st[i]), off[i])
            poles[i] = _item(POLE_UTILITY, q, p.facing, self.ground, p.source, ref=p.ref)

    def pole_lines(self) -> None:
        """Group the poles into lines along the road (one side, a steady offset); fill gaps
        inside a line at its median span, join lines across unphotographed gaps, and string
        the wires."""
        poles = [it for it in self.items if it.kind == POLE_UTILITY]
        if len(poles) < 2:
            return
        xz = np.array([(p.position[0], p.position[2]) for p in poles])
        st, off, inside = self.frame.project(xz)
        self._settle_poles(poles, st, off, inside)
        lo_span, hi_span = self.o.pole_span_m
        runs: list[list[int]] = []
        for side in (-1, 1):
            idx = [i for i in np.argsort(st).tolist() if inside[i] and np.sign(off[i]) == side]
            run: list[int] = []
            for i in idx:
                if run and (st[i] - st[run[-1]] > 2.6 * hi_span
                            or abs(off[i] - np.median(off[run[-3:]])) > 5.0):  # fmt: skip
                    runs.append(run)
                    run = []
                run.append(i)
            if run:
                runs.append(run)
        runs = [r for r in runs if len(r) >= 3]
        # Join consecutive runs on one side across a gap Mapillary did not photograph.
        joined: list[list[int]] = []
        for r in sorted(runs, key=lambda r: (np.sign(off[r[0]]), st[r[0]])):
            if joined:
                prev = joined[-1]
                gap0, gap1 = st[prev[-1]], st[r[0]]
                same_side = np.sign(off[prev[0]]) == np.sign(off[r[0]])
                close = abs(np.median(off[prev[-3:]]) - np.median(off[r[:3]])) < 4.0
                unseen = not any(
                    self.covered_at(float(s)) for s in np.arange(gap0 + 40, gap1 - 40, 20.0)
                )
                if same_side and close and gap1 - gap0 <= self.o.pole_join_m and unseen:
                    prev.extend(r)
                    continue
            joined.append(list(r))
        new_items: list[StreetItem] = []
        for r in joined:
            ss = st[r]
            spans = np.diff(ss)
            span = float(np.clip(np.median(spans[spans > 15.0]) if (spans > 15.0).any() else 45.0,
                                 lo_span, hi_span))  # fmt: skip
            lat = np.array([float(np.median(off[r[max(0, k - 2) : k + 3]])) for k in range(len(r))])
            pts: list[tuple[float, float, str]] = []
            for k in range(len(r)):
                pts.append((float(ss[k]), float(lat[k]), "keep"))
                if k + 1 < len(r):
                    gap = ss[k + 1] - ss[k]
                    n_new = int(round(gap / span)) - 1
                    if gap > 1.6 * span and n_new > 0:
                        for t in range(1, n_new + 1):
                            f = t / (n_new + 1)
                            pts.append(
                                (
                                    float(ss[k] + f * gap),
                                    float(lat[k] + f * (lat[k + 1] - lat[k])),
                                    "new",
                                )
                            )
            chain = []
            for s, la, what in pts:
                if what == "new":
                    _, fwd, left = self.frame.at(s)
                    it = _item(POLE_UTILITY, self.frame.point(s, la), -np.sign(la) * left,
                               self.ground, "rule")  # fmt: skip
                    new_items.append(it)
                    chain.append(np.array([it.position[0], it.position[1], it.position[2]]))
                else:
                    chain.append(None)
            # measured poles keep their position (snapped onto the line's offset if within 3 m)
            ki = 0
            for c, (_, la, what) in enumerate(pts):
                if what != "keep":
                    continue
                i = r[ki]
                ki += 1
                p = poles[i]
                if abs(off[i] - la) < 3.0:
                    q = self.frame.point(float(st[i]), la)
                    p = _item(POLE_UTILITY, q, p.facing, self.ground, p.source, ref=p.ref)
                    poles[i] = p
                chain[c] = np.array(p.position)
            for a, b in zip(chain[:-1], chain[1:], strict=True):
                if float(np.hypot(a[0] - b[0], a[2] - b[2])) <= 1.8 * hi_span:
                    self.plan.wires.append(np.stack([a, b]))
        rest = [it for it in self.items if it.kind != POLE_UTILITY]
        self.items = rest + poles + new_items


def _mapped(j: _Junction, forward: bool, osm_ctl) -> str | None:
    """What OSM says about one route approach to junction j: a stop=all or signal node on
    the junction, or a stop / give-way node on the approach within 35 m of it."""
    found = None
    for node, s, lat, ctl, fwd in osm_ctl:
        if node == j.node and ctl in ("all_way", "signal"):
            return ctl
        on_approach = (j.s - 35.0 <= s <= j.s - 1.0) if forward else (j.s + 1.0 <= s <= j.s + 35.0)
        if on_approach and abs(lat) < 6.0 and (fwd is None or fwd == forward):
            found = ctl
    return found


def _osm_node_controls(feats, projection, frame: _Frame, route=None) -> list:
    """(node id, station, lateral, control, forward) of OSM stop / give-way / signal nodes on
    or beside the route. `forward` is True / False when the node lies on the route and its
    `direction=forward|backward` (relative to its way) says which way of travel it stops, else
    None."""
    out = []
    nodes = [n for n in feats.nodes if n.tags.get("highway") in ("stop", "give_way",
             "traffic_signals")]  # fmt: skip
    if not nodes:
        return out
    way_dir: dict[int, int] = {}  # route node -> +1 if the route runs along its way's order
    if route is not None:
        for e in route.edges:
            way_dir.setdefault(e.u, e.dir)
            way_dir.setdefault(e.v, e.dir)
    xz = _xz_of(projection, np.array([(n.lon, n.lat) for n in nodes]))
    st, off, inside = frame.project(xz)
    for n, s, lat, ins in zip(nodes, st, off, inside, strict=True):
        if not ins:
            continue
        hw = n.tags["highway"]
        ctl = {"stop": "stop", "give_way": "yield", "traffic_signals": "signal"}[hw]
        if hw == "stop" and n.tags.get("stop") == "all":
            ctl = "all_way"
        fwd = None
        d = n.tags.get("direction", "")
        if n.id in way_dir and d in ("forward", "backward"):
            fwd = (d == "forward") == (way_dir[n.id] > 0)
        out.append((n.id, float(s), float(lat), ctl, fwd))
    return out


def dedupe(
    items: Sequence[StreetItem],
    radius_m: float = 12.0,
    *,
    pole_radius_m: float = 8.0,
    sides: Sequence[int] | None = None,
) -> list[StreetItem]:
    """One item per kind (speed limits count as one kind) within `radius_m` (`pole_radius_m`
    for poles), keeping the best source (`PRIORITY`, then input order), in input order. Signs
    also need to face the same way (within 60 degrees); with `sides` (-1 / +1 / 0 per item:
    right / left of the road / unknown), items on opposite sides never merge."""
    order = sorted(range(len(items)), key=lambda i: (_source_rank(items[i].source), i))
    keep: list[int] = []
    by_group: dict[str, list[int]] = {}
    for i in order:
        it = items[i]
        g = _dedupe_group(it.kind)
        r = pole_radius_m if it.kind == POLE_UTILITY else radius_m
        dup = False
        for j in by_group.get(g, ()):
            k = items[j]
            if math.hypot(k.position[0] - it.position[0], k.position[2] - it.position[2]) > r:
                continue
            if _is_sign(it.kind) and k.facing[0] * it.facing[0] + k.facing[1] * it.facing[1] < 0.5:
                continue
            if sides is not None and sides[i] * sides[j] < 0:
                continue
            dup = True
            break
        if not dup:
            keep.append(i)
            by_group.setdefault(g, []).append(i)
    return [items[i] for i in sorted(keep)]


def plan_furniture(
    model: RoadModel,
    projection: LocalProjection,
    route,
    ground: GroundFn,
    *,
    s_range: tuple[float, float] | None = None,
    way_id: np.ndarray | None = None,
    items: Sequence[StreetItem] = (),
    osm=None,
    mapillary: Sequence = (),
    coverage: np.ndarray | None = None,
    surface: DrivableSurface | None = None,
    options: FurnitureOptions | None = None,
    town_at: Callable[[np.ndarray], bool] | None = None,
) -> FurniturePlan:
    """Every street item for the track (see the module notes), de-duplicated, with the pole
    wires and the route's markings. `coverage` is Mapillary's photo positions (N, 2) lon/lat
    (None: no photos, rules everywhere); `way_id` the OSM way per station (speed limits);
    `surface` the drivable surface items keep off (`geometry.clearance`; default: the route's
    asphalt and shoulders); `town_at(xz)` True in a town (no rule-made chevrons there)."""
    o = options or FurnitureOptions()
    s_range = s_range or (float(model.s[0]), float(model.s[-1]))
    way_id = np.asarray(way_id if way_id is not None else np.zeros(0), np.int64)
    pl = _Planner(model, projection, route, s_range, ground, way_id, o, coverage, town_at)
    pl.plan.covered_fraction = float(pl.covered.mean()) if len(pl.covered) else 0.0
    pl.add_mapillary(list(mapillary))
    pl.add_osm(osm)
    juncs = _junctions(route, pl.frame, projection)
    pl.junction_controls(osm, list(mapillary))
    pl.speed_signs(juncs)
    pl.chevrons()
    pl.items += list(items)
    frame = pl.frame
    surf = surface if surface is not None else DrivableSurface.build(model)
    lines = [_xz_of(projection, ln.lonlat) for ln in (osm.power_lines if osm is not None else ())]

    def clear(first: bool) -> Counter:
        res = clear_items(
            pl.items, surf, ground, plate_width=lambda k: _plate_size(k, None)[0],
            sign_gap_m=o.sign_edge_gap_m, pole_edge_m=POLE_EDGE_M, wires=pl.plan.wires,
            max_span_m=1.8 * o.pole_span_m[1], power_lines=lines, first=first,
        )  # fmt: skip
        pl.items, pl.plan.wires = res.items, res.wires
        return res.stats

    stats = clear(first=True)  # signs in fields, poles on side roads, facings

    def sides(its: Sequence[StreetItem]) -> np.ndarray:
        if not its:
            return np.zeros(0, int)
        xz = np.array([(it.position[0], it.position[2]) for it in its])
        _, off, inside = frame.project(xz)
        return np.where(inside & (np.abs(off) < 30.0), np.sign(off), 0).astype(int)

    pl.items = dedupe(pl.items, o.dedupe_m, pole_radius_m=o.pole_dedupe_m, sides=sides(pl.items))
    pl.pole_lines()
    pl.street_lights(juncs, surf)
    pl.items = dedupe(pl.items, o.dedupe_m, pole_radius_m=o.pole_dedupe_m, sides=sides(pl.items))
    _signals(pl, juncs, osm, list(mapillary), surf)
    stats += clear(first=False)  # what the pole lines put on a side road or junction
    pl.items = _attach_lights(pl.items)
    plan = pl.plan
    plan.items = pl.items
    plan.clearance = dict(sorted(stats.items())) | _on_surface(plan.items, surf)
    plan.markings = _dedupe_markings(pl.markings)
    for it in plan.items:
        kind = _dedupe_group(it.kind) if it.kind.startswith(SIGN_SPEED_LIMIT) else it.kind
        plan.counts.setdefault(kind, {})
        plan.counts[kind][it.source] = plan.counts[kind].get(it.source, 0) + 1
    for mk in plan.markings:
        key = f"marking:{mk.kind.split(':')[0]}"
        plan.counts.setdefault(key, {})
        plan.counts[key][mk.source] = plan.counts[key].get(mk.source, 0) + 1
    plan.osm_nodes = sorted(set(plan.osm_nodes))
    return plan


_TRAFFIC_LIGHT = "object--traffic-light--"
_UNDER_SIGNALS = (SIGN_STOP, SIGN_YIELD, SIGN_ALL_WAY)
UNDER_SIGNAL_M = 25.0  # stop / yield signs this far beyond a signalised junction's poles go
UNDER_SIGNAL_NODE_M = 35.0  # ... and rule / OSM ones this close to an OSM signal node
SIGNAL_POLE_EDGE_M = 1.2  # signal poles stand at least this far off the drivable surface


def _signals(
    pl: _Planner, juncs: list[_Junction], osm, mapillary: Sequence,
    surface: DrivableSurface | None = None,
) -> None:  # fmt: skip
    """Ontario signals on every approach of each signalised route junction
    (`geometry.signals`): corner poles (signal items, moved off the surface by the next
    clearance pass) with a luminaire each; stop and yield signs on those approaches, and any
    other signal item there, are dropped. So are rule-made and OSM stop and yield signs
    within `UNDER_SIGNAL_NODE_M` of any OSM traffic signal node, also at junctions this layer
    does not plan (beyond A and B, in the run-off: the Burlington check's wrongly facing stop
    sign on a signalised side street). Poles planned on the real drivable `surface` (a big
    junction's corners are wider than the roads' tags say) first move to the nearest spot
    `SIGNAL_POLE_EDGE_M` off it (`_off_surface`), so the clearance pass does not drop them."""
    nodes = [n for n in (osm.nodes if osm is not None else ()) if
             n.tags.get("highway") == "traffic_signals"]  # fmt: skip
    sig_xz = _xz_of(pl.proj, np.array([(n.lon, n.lat) for n in nodes])) if nodes else np.zeros(
        (0, 2))  # fmt: skip
    lights = [f for f in mapillary if f.value.startswith(_TRAFFIC_LIGHT)]
    light_xz = (_xz_of(pl.proj, np.array([(f.lon, f.lat) for f in lights])) if lights
                else np.zeros((0, 2)))  # fmt: skip
    plan = plan_signals(pl.signal_junctions(juncs), sig_xz, light_xz)
    pl.plan.signals = plan
    if len(sig_xz):
        tree = cKDTree(sig_xz)
        keep = []
        for it in pl.items:
            if it.kind in _UNDER_SIGNALS and it.source in ("rule", "osm"):
                d, _ = tree.query([it.position[0], it.position[2]])
                if d <= UNDER_SIGNAL_NODE_M:
                    pl.plan.stops_under_signals += 1
                    continue
            keep.append(it)
        pl.items = keep
    if not plan.junctions:
        return
    st, _, _ = pl.frame.project(np.array([c for _, c, _ in plan.junctions]))
    pl.plan.signal_stations = sorted(round(float(v) - pl.frame.s0, 1) for v in st)
    nodes_sig = {f"junction:{node}" for node in plan.nodes}
    keep = []
    for it in pl.items:
        xz = np.array([it.position[0], it.position[2]])
        if it.kind in _UNDER_SIGNALS and (it.ref in nodes_sig or plan.near(xz, UNDER_SIGNAL_M)):
            pl.plan.stops_under_signals += 1
            continue
        if it.kind == SIGNAL_TRAFFIC and plan.near(xz, 5.0):
            continue  # our corner poles replace it
        keep.append(it)
    for pole in plan.poles:
        if surface is not None:
            pole.xz = _off_surface(pole.xz, pole.origin, surface)
        facing = pole.axis if pole.axis is not None else np.array([1.0, 0.0])
        keep.append(_item(SIGNAL_TRAFFIC, pole.xz, -facing, pl.ground, "rule", ref=pole.ref))
        if not pole.post:  # Ontario signal poles carry the junction's luminaires
            keep.append(_item(LIGHT_STREET, pole.xz, -facing, pl.ground, "rule"))
    pl.items = keep


def _off_surface(xz: np.ndarray, centre: np.ndarray, surface: DrivableSurface) -> np.ndarray:
    """The nearest point to `xz` (within 20 m, not nearer the junction `centre`) that stands
    SIGNAL_POLE_EDGE_M off the drivable surface: the corner the plan meant, where a big
    junction's real pavement reaches farther than the roads' tags say. `xz` itself when it
    already stands clear or nothing does."""
    p = np.asarray(xz, np.float64)
    if float(surface.distance(p[None, :])[0]) >= SIGNAL_POLE_EDGE_M:
        return p
    ang = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    rad = np.arange(0.5, 20.01, 0.5)
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    cand = (p[None, None, :] + rad[:, None, None] * ring[None, :, :]).reshape(-1, 2)
    c = np.asarray(centre, np.float64)
    keep = np.hypot(*(cand - c).T) >= float(np.hypot(*(p - c))) - 1.0
    cand = cand[keep]
    ok = cand[surface.distance(cand) >= SIGNAL_POLE_EDGE_M]
    if not len(ok):
        return p
    return ok[int(np.argmin(np.hypot(*(ok - p).T)))]


def _on_surface(items: Sequence[StreetItem], surf: DrivableSurface) -> dict[str, int]:
    """How many standing items (signs, poles, lights, signals, hydrants) are on the drivable
    surface, and within `CLEARANCE_M` of it (mailboxes may stand at its edge)."""
    its = [it for it in items if it.kind != MAILBOX and (_is_sign(it.kind) or it.kind in (
        POLE_UTILITY, LIGHT_STREET, SIGNAL_TRAFFIC, HYDRANT))]  # fmt: skip
    if not its:
        return {"on_surface": 0, "near_surface": 0}
    xz = np.array([(it.position[0], it.position[2]) for it in its])
    d = surf.distance(xz)
    near = int((d < CLEARANCE_M - 0.02).sum())  # 0.02: the distance queries' tolerance
    return {"on_surface": int((d <= 0.0).sum()), "near_surface": near}


def _attach_lights(items: list[StreetItem]) -> list[StreetItem]:
    """A street light within 8 m of a utility or signal pole hangs on that pole (its position
    becomes the pole's; the renderer then draws only the arm and the head)."""
    poles = [it for it in items if it.kind in (POLE_UTILITY, SIGNAL_TRAFFIC)]
    if not poles:
        return items
    pxz = np.array([(p.position[0], p.position[2]) for p in poles])
    out = []
    for it in items:
        if it.kind == LIGHT_STREET:
            d = np.hypot(pxz[:, 0] - it.position[0], pxz[:, 1] - it.position[2])
            k = int(np.argmin(d))
            if d[k] < 8.0:
                p = poles[k]
                it = StreetItem(it.kind, p.position, it.facing, it.source, "on_pole", it.ref)
        out.append(it)
    return out


def _dedupe_markings(marks: list[Marking]) -> list[Marking]:
    order = sorted(range(len(marks)), key=lambda i: (_source_rank(marks[i].source), i))
    kept: list[Marking] = []
    for i in order:
        mk = marks[i]
        base = mk.kind.split(":")[0]
        if any(k.kind.split(":")[0] == base and abs(k.s - mk.s) < 15.0
               and (k.l0 + k.l1) * (mk.l0 + mk.l1) >= 0 for k in kept):  # fmt: skip
            continue
        kept.append(mk)
    return sorted(kept, key=lambda k: k.s)


# --------------------------------------------------------------------------------------------
# Meshes
# --------------------------------------------------------------------------------------------

# Plate shapes in their own frame (x right, y up, metres), unit size; scaled by (w, h).
_OCT = np.array([[math.cos(a), math.sin(a)] for a in np.radians(22.5 + 45.0 * np.arange(8))])
_OCT = _OCT / np.abs(_OCT).max(axis=0) * 0.5
_SHAPES = {
    "octagon": _OCT,
    "triangle_down": np.array([[-0.5, 0.5], [0.0, -0.5], [0.5, 0.5]])[::-1],
    "diamond": np.array([[0.0, -0.5], [0.5, 0.0], [0.0, 0.5], [-0.5, 0.0]]),
    "rect": np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]),
}


def _plate_size(kind: str, speed: int | None) -> tuple[float, float]:
    """(width, height) in metres (OTM Book 5/6 sizes; see the module notes)."""
    fast = (speed or 0) >= 70
    if kind == SIGN_STOP:
        return (0.9, 0.9) if fast else (0.75, 0.75)
    if kind == SIGN_YIELD:
        return 0.75, 0.65
    if kind in (SIGN_STOP_AHEAD, SIGN_ROUNDABOUT_AHEAD, SIGN_DEAD_END):
        return (0.75, 0.75) if fast else (0.6, 0.6)
    if kind.startswith(SIGN_SPEED_LIMIT):
        return 0.6, 0.75
    if kind in (SIGN_CHEVRON_LEFT, SIGN_CHEVRON_RIGHT):
        return 0.45, 0.6
    if kind == SIGN_KEEP_RIGHT:
        return 0.6, 0.75
    if kind == SIGN_ALL_WAY:
        return 0.6, 0.2
    if kind == SIGN_STREET_NAME:
        return 1.2, 0.2
    return 0.6, 0.6


def _face(kind: str) -> tuple[str, str, bool] | None:
    """(atlas tile, shape, mirrored) of a sign kind; None if it has no face."""
    if kind == SIGN_STOP:
        return "stop", "octagon", False
    if kind == SIGN_YIELD:
        return "yield", "triangle_down", False
    if kind == SIGN_STOP_AHEAD:
        return "stop_ahead", "diamond", False
    if kind == SIGN_ROUNDABOUT_AHEAD:
        return "roundabout_ahead", "diamond", False
    if kind == SIGN_DEAD_END:
        return "dead_end", "diamond", False
    if kind == SIGN_CHEVRON_LEFT:
        return "chevron", "rect", False
    if kind == SIGN_CHEVRON_RIGHT:
        return "chevron", "rect", True
    if kind == SIGN_KEEP_RIGHT:
        return "keep_right", "rect", False
    if kind == SIGN_ALL_WAY:
        return "all_way", "rect", False
    if kind.startswith(SIGN_SPEED_LIMIT + ":"):
        try:
            v = int(kind.rsplit(":", 1)[1])
        except ValueError:
            return None
        return (f"speed_limit:{v}", "rect", False) if v in SPEEDS else None
    return None


class _Builder:
    """Accumulates triangles (positions, normals, uvs) for one material."""

    def __init__(self) -> None:
        self.p: list[np.ndarray] = []
        self.n: list[np.ndarray] = []
        self.uv: list[np.ndarray] = []
        self.i: list[np.ndarray] = []
        self.count = 0

    def add(self, pos, nrm, uv, tris) -> None:
        pos = np.asarray(pos, float).reshape(-1, 3)
        k = len(pos)
        self.p.append(pos)
        self.n.append(np.broadcast_to(np.asarray(nrm, float), (k, 3)).copy())
        self.uv.append(np.broadcast_to(np.asarray(uv, float), (k, 2)).copy())
        self.i.append(np.asarray(tris, np.int64).reshape(-1) + self.count)
        self.count += k

    def quad(self, a, b, c, d, uv, out_hint=None) -> None:
        """Quad a-b-c-d (any winding); the visible side faces `out_hint` if given."""
        a, b, c, d = (np.asarray(v, float) for v in (a, b, c, d))
        nrm = np.cross(b - a, d - a)
        if out_hint is not None and float(nrm @ np.asarray(out_hint, float)) < 0:
            a, b, c, d = a, d, c, b
            nrm = -nrm
        ln = float(np.linalg.norm(nrm))
        nrm = nrm / ln if ln > 1e-12 else np.array([0.0, 1.0, 0.0])
        self.add([a, b, c, d], nrm, uv, [0, 1, 2, 0, 2, 3])

    def prism(self, base, height, r0, r1, sides, uv, axis_a=None, top=True, uv_v=None) -> None:
        """An upright `sides`-gon prism (radius r0 at the base, r1 at the top), smooth
        normals; `uv` a constant (u, v) or, with `uv_v` = (u0, u1, v0, v1), wrapped."""
        base = np.asarray(base, float)
        a = np.array([1.0, 0.0, 0.0]) if axis_a is None else np.asarray(axis_a, float)
        b = np.cross(a, [0.0, 1.0, 0.0])
        ang = 2 * math.pi * np.arange(sides + 1) / sides
        ring = np.cos(ang)[:, None] * a + np.sin(ang)[:, None] * b
        lo = base + ring * r0
        hi = base + ring * r1 + np.array([0.0, height, 0.0])
        slope = (r0 - r1) / max(height, 1e-6)
        nrm = ring + np.array([0.0, slope, 0.0])
        nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        pos = np.concatenate([lo, hi])
        nn = np.concatenate([nrm, nrm])
        if uv_v is not None:
            u0, u1, v0, v1 = uv_v
            us = u0 + (u1 - u0) * np.arange(sides + 1) / sides
            uvs = np.concatenate([np.stack([us, np.full_like(us, v1)], 1),
                                  np.stack([us, np.full_like(us, v0)], 1)])  # fmt: skip
        else:
            uvs = np.broadcast_to(np.asarray(uv, float), (len(pos), 2))
        tris = []
        n = sides + 1
        for k in range(sides):
            # outward: (lo_k, hi_k, hi_k+1) ... checked against the ring normal
            t = [k, n + k, n + k + 1, k, n + k + 1, k + 1]
            tris += t
        tri = np.array(tris).reshape(-1, 3)
        # Orientation check on the first triangle.
        p0, p1, p2 = pos[tri[0]]
        if float(np.cross(p1 - p0, p2 - p0) @ nn[tri[0, 0]]) < 0:
            tri = tri[:, [0, 2, 1]]
        self.add(pos, nn, uvs, tri)
        if top and r1 > 0:
            c = base + np.array([0.0, height, 0.0])
            cap = np.concatenate([[c], hi[:-1]])
            ct = np.array([[0, k + 1, (k + 1) % sides + 1] for k in range(sides)])
            q0, q1, q2 = cap[ct[0]]
            if float(np.cross(q1 - q0, q2 - q0)[1]) < 0:
                ct = ct[:, [0, 2, 1]]
            self.add(cap, [0.0, 1.0, 0.0], uv if uv_v is None else (uv_v[0], uv_v[2]), ct)

    def box(self, centre, size, axis_a, uv) -> None:
        """A box: size (along a, height, along b) with a = `axis_a` (horizontal), b = a x up."""
        c = np.asarray(centre, float)
        a = np.asarray(axis_a, float)
        a = a / np.linalg.norm(a)
        up = np.array([0.0, 1.0, 0.0])
        b = np.cross(a, up)
        ha, hh, hb = (s / 2 for s in size)
        for nrm, u, v, hn, hu, hv in ((a, b, up, ha, hb, hh), (-a, b, up, ha, hb, hh),
                                      (b, a, up, hb, ha, hh), (-b, a, up, hb, ha, hh),
                                      (up, a, b, hh, ha, hb), (-up, a, b, hh, ha, hb)):  # fmt: skip
            f = c + nrm * hn
            self.quad(f - u * hu - v * hv, f + u * hu - v * hv, f + u * hu + v * hv,
                      f - u * hu + v * hv, uv, out_hint=nrm)  # fmt: skip

    def polygon(self, pts3, nrm, uvs) -> None:
        """A convex polygon (fan), visible side `nrm`."""
        pts3 = np.asarray(pts3, float)
        k = len(pts3)
        tri = np.array([[0, j, j + 1] for j in range(1, k - 1)])
        p0, p1, p2 = pts3[tri[0]]
        if float(np.cross(p1 - p0, p2 - p0) @ nrm) < 0:
            tri = tri[:, [0, 2, 1]]
        self.add(pts3, nrm, uvs, tri)

    def ribbon(self, pts, width, uv) -> None:
        """A wire: two crossed strips (vertical and horizontal) along the (K, 3) points."""
        pts = np.asarray(pts, float)
        d = np.gradient(pts, axis=0)
        d /= np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]
        side = np.cross(d, [0.0, 1.0, 0.0])
        side /= np.maximum(np.linalg.norm(side, axis=1), 1e-9)[:, None]
        vert = np.cross(side, d)
        k = len(pts)
        for off, nrm in ((side * width / 2, vert), (vert * width / 2, side)):
            pos = np.concatenate([pts - off, pts + off])
            nn = np.concatenate([nrm, nrm])
            tris = []
            for j in range(k - 1):
                tris += [j, j + 1, k + j + 1, j, k + j + 1, k + j]
            # Wires are seen from every side: add both windings (no back-face culling issues).
            t = np.array(tris).reshape(-1, 3)
            self.add(pos, nn, uv, np.concatenate([t, t[:, [0, 2, 1]]]))

    def mesh(self) -> MeshData | None:
        if not self.count:
            return None
        n = np.concatenate(self.n)
        t = np.cross([0.0, 1.0, 0.0], n)
        bad = np.linalg.norm(t, axis=1) < 1e-6
        t[bad] = [1.0, 0.0, 0.0]
        t /= np.linalg.norm(t, axis=1)[:, None]
        return MeshData(np.concatenate(self.p), n, np.concatenate(self.uv), t,
                        np.concatenate(self.i))  # fmt: skip


@dataclass
class _Cell:
    furn: _Builder = field(default_factory=_Builder)
    signs: _Builder = field(default_factory=_Builder)
    wires: _Builder = field(default_factory=_Builder)
    hidden: _Builder = field(default_factory=_Builder)


_UP = np.array([0.0, 1.0, 0.0])


def _face_uv(tile_name: str, shape: str, mirrored: bool, name_index: int | None = None):
    tile = name_tile(name_index) if name_index is not None else sign_tile(tile_name)
    u0, v0, u1, v1 = tile.uv
    pts = _SHAPES[shape]
    # half-texel inset so mips never sample the neighbour tile
    du, dv = 0.5 / 1024, 0.5 / 1024
    x = pts[:, 0] + 0.5
    if mirrored:
        x = 1.0 - x
    u = u0 + du + (u1 - u0 - 2 * du) * x
    v = v1 - dv - (v1 - v0 - 2 * dv) * (pts[:, 1] + 0.5)
    return np.stack([u, v], axis=1)


class _Renderer:
    def __init__(self, road_y: Callable[[np.ndarray], np.ndarray], walls_lat, frame, o, names):
        self.road_y = road_y
        self.walls_lat = walls_lat
        self.frame = frame
        self.o = o
        self.cells: dict[tuple[int, int], _Cell] = {}
        self.names = names  # street name -> atlas index
        self.metal = swatch_uv("galvanized")
        self.back = swatch_uv("sign_back")

    def cell(self, x: float, z: float) -> _Cell:
        key = (int(math.floor(x / self.o.cell_m)), int(math.floor(z / self.o.cell_m)))
        return self.cells.setdefault(key, _Cell())

    # -- signs ----------------------------------------------------------------------------------

    def sign_post(self, base: np.ndarray, facing: np.ndarray, stack: list[StreetItem],
                  speed: int | None, post: bool = True) -> None:  # fmt: skip
        """One post carrying `stack` (top first): plates with grey backs. `post` False: the
        plates go on a post already drawn (a back-to-back pair)."""
        c = self.cell(base[0], base[2])
        f3 = np.array([facing[0], 0.0, facing[1]])
        right = np.cross(_UP, f3)  # viewer's right when facing the sign
        ry = float(self.road_y(base[[0, 2]][None, :])[0])
        sizes = [_plate_size(it.kind, speed) for it in stack]
        bottom = max(ry + self.o.sign_bottom_m, base[1] + 1.0)
        top = bottom + sum(h for _, h in sizes) + 0.03 * (len(sizes) - 1)
        post_r = 0.035
        if post:
            c.furn.prism(base + np.array([0.0, -0.4, 0.0]), top - base[1] + 0.35, post_r, post_r,
                         4, self.metal, axis_a=right, top=True)  # fmt: skip
        y = top
        for it, (w, h) in zip(stack, sizes, strict=True):
            y -= h
            face = _face(it.kind)
            centre = base + np.array([0.0, y + h / 2 - base[1], 0.0]) + f3 * (post_r + 0.012)
            shape = face[1] if face else "rect"
            local = _SHAPES[shape] * np.array([w, h])
            pts = centre + local[:, :1] * right + local[:, 1:] * _UP
            if face is not None:
                c.signs.polygon(pts, f3, _face_uv(face[0], face[1], face[2]))
            c.furn.polygon(pts - f3 * 0.006, -f3, self.back)
            y -= 0.03

    def street_name(self, it: StreetItem) -> None:
        idx = self.names.get(it.text)
        base = np.array(it.position)
        c = self.cell(base[0], base[2])
        f3 = np.array([it.facing[0], 0.0, it.facing[1]])
        right = np.cross(_UP, f3)
        c.furn.prism(base + np.array([0.0, -0.4, 0.0]), 3.25, 0.04, 0.04, 4, self.metal, top=True)
        if idx is None:
            return
        w, h = _plate_size(SIGN_STREET_NAME, None)
        w = min(1.5, max(0.8, 0.09 * len(it.text) + 0.2))
        centre = base + np.array([0.0, 2.75, 0.0])
        local = _SHAPES["rect"] * np.array([w, h])
        for sgn in (1.0, -1.0):  # readable from both sides
            pts = centre + sgn * f3 * 0.02 + sgn * local[:, :1] * right + local[:, 1:] * _UP
            uv = _face_uv("", "rect", False, idx)
            c.signs.polygon(pts, sgn * f3, uv)

    # -- poles, lights, signals -----------------------------------------------------------------

    def utility_pole(self, it: StreetItem, along: np.ndarray) -> list[np.ndarray]:
        """A wooden distribution pole; returns its wire attachment points (5, 3)."""
        base = np.array(it.position)
        h = self.o.pole_height_m
        c = self.cell(base[0], base[2])
        u0, v0, u1, v1 = WOOD.uv
        c.furn.prism(base + np.array([0.0, -0.5, 0.0]), h + 0.5, 0.15, 0.11, 7, None,
                     uv_v=(u0, u1, v0, v1))  # fmt: skip
        a3 = np.array([along[0], 0.0, along[1]])
        across = np.cross(a3, _UP)
        across /= np.linalg.norm(across)
        top = base[1] + h
        arm_y = top - 0.3
        arm = swatch_uv("crossarm")
        c.furn.box(np.array([base[0], arm_y, base[2]]) + a3 * 0.13, (0.09, 0.11, 2.4), a3, arm)
        ins = swatch_uv("insulator")
        att = []
        for off in (-1.05, 1.05):
            p = np.array([base[0], arm_y + 0.055, base[2]]) + across * off + a3 * 0.13
            c.furn.prism(p, 0.16, 0.05, 0.035, 5, ins)
            att.append(p + np.array([0.0, 0.15, 0.0]))
        p = np.array([base[0], top, base[2]])
        c.furn.prism(p, 0.18, 0.05, 0.035, 5, ins)
        att.append(p + np.array([0.0, 0.17, 0.0]))
        att.append(np.array([base[0], top - 1.6, base[2]]) + across * 0.17)  # neutral
        att.append(np.array([base[0], top - 4.8, base[2]]) - across * 0.17)  # telecom cable
        if self.o.collide:
            self.collider(base, 0.2)
        return att

    def collider(self, base: np.ndarray, r: float) -> None:
        """A hidden octagonal cylinder if the pole stands inside the wall corridor."""
        if self.walls_lat is None:
            return
        st, off, inside = self.frame.project(base[[0, 2]][None, :])
        if not inside[0] or abs(off[0]) > self.walls_lat(float(st[0])) + 0.5:
            return
        c = self.cell(base[0], base[2])
        c.hidden.prism(base + np.array([0.0, -1.0, 0.0]), 5.0, r, r, 8, (0.5, 0.5), top=False)

    def light(self, it: StreetItem, toward: np.ndarray) -> None:
        """A cobra-head luminaire: on its own galvanized pole, or on a utility/signal pole
        (`text` "on_pole"), its arm reaching over the road."""
        base = np.array(it.position)
        c = self.cell(base[0], base[2])
        t3 = np.array([toward[0], 0.0, toward[1]])
        t3 /= max(np.linalg.norm(t3), 1e-9)
        mount = 8.2
        if it.text != "on_pole":
            c.furn.prism(base + np.array([0.0, -0.4, 0.0]), mount + 0.9, 0.11, 0.065, 8,
                         self.metal)  # fmt: skip
            if self.o.collide:
                self.collider(base, 0.15)
        p0 = base + np.array([0.0, mount, 0.0]) + t3 * 0.12
        p1 = p0 + t3 * 2.3 + np.array([0.0, 0.35, 0.0])
        arm = np.stack([p0 + (p1 - p0) * t for t in np.linspace(0, 1, 4)])
        c.furn.ribbon(arm, 0.07, self.metal)
        head = p1 + t3 * 0.35 - np.array([0.0, 0.05, 0.0])
        c.furn.box(head, (0.75, 0.16, 0.32), t3, swatch_uv("lamp_head"))
        lens = head - np.array([0.0, 0.085, 0.0])
        b3 = np.cross(t3, _UP)
        c.signs.quad(lens - t3 * 0.3 - b3 * 0.12, lens + t3 * 0.3 - b3 * 0.12,
                     lens + t3 * 0.3 + b3 * 0.12, lens - t3 * 0.3 + b3 * 0.12,
                     swatch_uv("lamp_lens"), out_hint=-_UP)  # fmt: skip

    def signal(self, it: StreetItem, over: float) -> None:
        """A signal pole with a mast arm reaching `over` metres across the road, a 3-section
        head over the lanes and one on the pole, facing the approach."""
        base = np.array(it.position)
        c = self.cell(base[0], base[2])
        f3 = np.array([it.facing[0], 0.0, it.facing[1]])
        across = np.cross(f3, _UP)  # points to the approach's left (over the road)
        c.furn.prism(base + np.array([0.0, -0.4, 0.0]), 7.6, 0.16, 0.11, 8, self.metal)
        p0 = base + np.array([0.0, 6.4, 0.0])
        arm = np.stack([p0 + across * t for t in np.linspace(0, over, 5)])
        c.furn.ribbon(arm, 0.14, self.metal)
        for p in (p0 + across * (over - 0.4) - np.array([0.0, 0.7, 0.0]),
                  base + np.array([0.0, 3.2, 0.0]) + f3 * 0.25):  # fmt: skip
            c.furn.box(p, (0.36, 1.05, 0.26), across, swatch_uv("signal_yellow"))
            for k, lens in enumerate(("lens_red", "lens_amber", "lens_green")):
                q = p + f3 * 0.135 + np.array([0.0, 0.33 - 0.33 * k, 0.0])
                c.signs.quad(q - across * 0.11 - _UP * 0.11, q + across * 0.11 - _UP * 0.11,
                             q + across * 0.11 + _UP * 0.11, q - across * 0.11 + _UP * 0.11,
                             swatch_uv(lens), out_hint=f3)  # fmt: skip
        if self.o.collide:
            self.collider(base, 0.2)

    def hydrant(self, it: StreetItem) -> None:
        base = np.array(it.position)
        c = self.cell(base[0], base[2])
        red, cap = swatch_uv("hydrant_red"), swatch_uv("hydrant_cap")
        c.furn.prism(base + np.array([0.0, -0.1, 0.0]), 0.12, 0.17, 0.15, 6, red, top=False)
        c.furn.prism(base, 0.62, 0.12, 0.11, 6, red, top=False)
        c.furn.prism(base + np.array([0.0, 0.62, 0.0]), 0.14, 0.14, 0.06, 6, cap)
        f3 = np.array([it.facing[0], 0.0, it.facing[1]])
        c.furn.box(base + np.array([0.0, 0.45, 0.0]) + f3 * 0.14, (0.12, 0.12, 0.14), f3, cap)
        side = np.cross(f3, _UP)
        for sgn in (1, -1):
            c.furn.box(base + np.array([0.0, 0.45, 0.0]) + side * sgn * 0.13, (0.09, 0.09, 0.1),
                       side, cap)  # fmt: skip

    def mailbox(self, it: StreetItem) -> None:
        base = np.array(it.position)
        c = self.cell(base[0], base[2])
        f3 = np.array([it.facing[0], 0.0, it.facing[1]])
        c.furn.box(base + np.array([0.0, 0.45, 0.0]), (0.09, 1.3, 0.09), f3, swatch_uv("post_wood"))
        c.furn.box(base + np.array([0.0, 1.17, 0.0]), (0.5, 0.2, 0.22), f3, swatch_uv("mailbox"))

    def wires(self, spans: list[np.ndarray]) -> None:
        wire = swatch_uv("wire")
        for a, b in spans:
            mid = (a + b) / 2
            c = self.cell(mid[0], mid[2])
            span = float(np.linalg.norm((b - a)[[0, 2]]))
            sag = self.o.wire_sag * span
            t = np.linspace(0.0, 1.0, 9)[:, None]
            pts = a + (b - a) * t - np.array([0.0, 1.0, 0.0]) * (4 * sag * t * (1 - t))
            c.wires.ribbon(pts, 0.022, wire)


def furniture_meshes(
    plan: FurniturePlan,
    model: RoadModel,
    *,
    s_range: tuple[float, float] | None = None,
    verge_m: float | None = None,
    names: Mapping[str, int] | None = None,
    speed_at: Callable[[np.ndarray], int | None] | None = None,
    options: FurnitureOptions | None = None,
) -> list[FurnitureMesh]:
    """The meshes of every item and wire in `plan`, merged per cell (see the module notes).
    `verge_m`: hidden walls stand this far beyond the shoulder (poles inside get colliders);
    `names`: atlas index per street name (`street_names`); `speed_at(xz)`: the limit there
    (km/h; bigger signs at 70 and up)."""
    o = options or FurnitureOptions()
    frame = _Frame(model, s_range or (float(model.s[0]), float(model.s[-1])))
    from scipy.spatial import cKDTree

    tree = cKDTree(frame.xz)

    def road_y(xz: np.ndarray) -> np.ndarray:
        return model.centre[tree.query(xz)[1], 1]

    walls = None
    if verge_m is not None:

        def walls(s: float) -> float:
            return frame.edge(s) + verge_m

    r = _Renderer(road_y, walls, frame, o, dict(names or {}))
    items = plan.items
    sig_poles = {p.ref: p for p in plan.signals.poles}
    # Poles first: their wire attachment points depend on the line direction.
    poles = [it for it in items if it.kind == POLE_UTILITY]
    att: dict[int, list[np.ndarray]] = {}
    if poles:
        pxz = np.array([(p.position[0], p.position[2]) for p in poles])
        ptree = cKDTree(pxz)
        for k, p in enumerate(poles):
            d, nb = ptree.query(pxz[k], k=min(3, len(poles)))
            nb = [j for j in np.atleast_1d(nb).tolist() if j != k and j < len(poles)]
            if nb and np.atleast_1d(d)[1] < 90.0:
                along = _unit2(pxz[nb[0]] - pxz[k])
            else:
                st, _, _ = frame.project(pxz[k][None, :])
                along = frame.at(float(st[0]))[1]
            att[k] = r.utility_pole(p, along)
    # Wires connect attachment points of neighbouring poles (matched by position).
    if poles and plan.wires:
        key = {(round(p.position[0], 2), round(p.position[2], 2)): k for k, p in enumerate(poles)}
        spans = []
        for a, b in plan.wires:
            ka = key.get((round(a[0], 2), round(a[2], 2)))
            kb = key.get((round(b[0], 2), round(b[2], 2)))
            if ka is None or kb is None:
                continue
            pa, pb = att[ka], att[kb]
            # pair each attachment with the nearer one on the other pole (crossarms may face
            # either way)
            used = set()
            for q in pa:
                d = [
                    float(np.linalg.norm(q - w)) if j not in used else 1e9 for j, w in enumerate(pb)
                ]
                j = int(np.argmin(d))
                used.add(j)
                spans.append(np.stack([q, pb[j]]))
        r.wires(spans)
    # Signs: group plates sharing a post (same spot, same facing): main sign above its tabs.
    signs = [it for it in items if _is_sign(it.kind) and it.kind != SIGN_STREET_NAME]
    posts: list[list[StreetItem]] = []
    for it in signs:
        for grp in posts:
            g = grp[0]
            if (math.hypot(g.position[0] - it.position[0], g.position[2] - it.position[2]) < 0.8
                    and g.facing[0] * it.facing[0] + g.facing[1] * it.facing[1] > 0.9):  # fmt: skip
                grp.append(it)
                break
        else:
            posts.append([it])
    drawn: list[np.ndarray] = []  # posts already standing (back-to-back plates share one)
    for grp in posts:
        grp.sort(key=lambda it: it.kind == SIGN_ALL_WAY)  # tabs below the main sign
        g = grp[0]
        base = np.array(g.position, float)
        speed = speed_at(base[[0, 2]]) if speed_at is not None else None
        shared = any(float(np.hypot(*(d[[0, 2]] - base[[0, 2]]))) < 0.3 for d in drawn)
        r.sign_post(base, np.array(g.facing, float), grp, speed, post=not shared)
        if not shared:
            drawn.append(base)
    for it in items:
        if it.kind == SIGN_STREET_NAME:
            r.street_name(it)
        elif it.kind == LIGHT_STREET:
            st, off, inside = frame.project(np.array([[it.position[0], it.position[2]]]))
            toward = (
                -np.sign(off[0] or 1.0) * frame.at(float(st[0]))[2]
                if inside[0]
                else np.array(it.facing)
            )
            r.light(it, toward)
        elif it.kind == SIGNAL_TRAFFIC and it.ref in sig_poles:
            base = np.array(it.position, float)
            draw_pole(r.cell(base[0], base[2]), sig_poles[it.ref], base, road_y, r.names, r.metal)
            if o.collide:
                r.collider(base, 0.2)
        elif it.kind == SIGNAL_TRAFFIC:
            st, off, _ = frame.project(np.array([[it.position[0], it.position[2]]]))
            over = max(2.5, abs(float(off[0])) - 1.75)
            r.signal(it, over)
        elif it.kind == HYDRANT:
            r.hydrant(it)
        elif it.kind == MAILBOX:
            r.mailbox(it)
    out: list[FurnitureMesh] = []
    near = {"lod_out": o.lod_out_m}
    far = {"lod_out": o.lod_out_far_m}
    for ci, key in enumerate(sorted(r.cells)):
        cell = r.cells[key]
        for prefix, b, material, flags in (
            ("furn", cell.furn, "furniture", far),
            ("signs", cell.signs, "signs", near),
            ("wires", cell.wires, "furniture", {**far, "cast_shadows": False}),
            ("1WALL_furn", cell.hidden, "hidden", {"is_renderable": False, "cast_shadows": False}),
        ):
            data = b.mesh()
            if data is None:
                continue
            pieces = split(data)
            for p, piece in enumerate(pieces):
                name = f"{prefix}_{ci:03d}" + (f"_{p}" if len(pieces) > 1 else "")
                out.append(FurnitureMesh(name, material, piece, dict(flags)))
    return out


def street_names(items: Sequence[StreetItem], signals: SignalPlan | None = None) -> dict[str, int]:
    """Atlas index per street name on the blades and the signal mast arms (the first
    MAX_NAMES names, sorted)."""
    names = {it.text for it in items if it.kind == SIGN_STREET_NAME and it.text}
    names |= {p.name for p in (signals.poles if signals is not None else ()) if p.name}
    names = sorted(names)
    return {n: k for k, n in enumerate(names[:MAX_NAMES])}


# --------------------------------------------------------------------------------------------
# Markings
# --------------------------------------------------------------------------------------------

STOP_BAR_M = 0.45  # OTM Book 11: stop lines 0.30-0.60 m wide (ASSUMED 0.45)
CROSSWALK_LINE_M = 0.2  # two lines 0.1-0.2 m wide ...
CROSSWALK_GAP_M = 3.0  # ... about 2.5-3.0 m apart (ASSUMED)
ZEBRA_BAR_M, ZEBRA_GAP_M, ZEBRA_LEN_M = 0.5, 0.6, 3.0  # ladder blocks (ASSUMED)
MARK_LIFT_M = 0.025  # above the road (edge lines sit at 0.02)


def marking_meshes(
    marks: Sequence[Marking], model: RoadModel, cell_m: float = 250.0
) -> list[tuple[str, MeshData]]:
    """Paint decals for the markings (`marks_<cell>`), white, on the road surface."""
    cells: dict[int, _Builder] = {}

    def rect(s0: float, s1: float, l0: float, l1: float) -> None:
        pa, fa, la = model.at(s0)
        pb, fb, lb = model.at(s1)
        y0, y1 = pa[1] + MARK_LIFT_M, pb[1] + MARK_LIFT_M
        q = [
            np.array([pa[0], y0, pa[2]]) + la * l0,
            np.array([pb[0], y1, pb[2]]) + lb * l0,
            np.array([pb[0], y1, pb[2]]) + lb * l1,
            np.array([pa[0], y0, pa[2]]) + la * l1,
        ]
        b = cells.setdefault(int((s0 + s1) / 2 // cell_m), _Builder())
        b.quad(*q, (0.5, 0.5), out_hint=_UP)

    def arrow(mk: Marking) -> None:
        c = (mk.l0 + mk.l1) / 2
        d = 1.0 if mk.forward else -1.0
        s = mk.s
        # Shaft 0.15 x 3.0 m, head 0.6 wide x 1.2 m (OTM Book 11 arrows ~5 m; ASSUMED).
        rect(min(s, s + d * 3.0), max(s, s + d * 3.0), c - 0.08, c + 0.08)
        tip = s + d * 4.2
        if mk.kind == "arrow:straight":
            base_s = s + d * 3.0
            pa = model.at(base_s)
            pt = model.at(tip)
            y = pa[0][1] + MARK_LIFT_M
            p0 = np.array([pa[0][0], y, pa[0][2]]) + pa[2] * (c - 0.3)
            p1 = np.array([pa[0][0], y, pa[0][2]]) + pa[2] * (c + 0.3)
            p2 = np.array([pt[0][0], pt[0][1] + MARK_LIFT_M, pt[0][2]]) + pt[2] * c
            b = cells.setdefault(int(s // cell_m), _Builder())
            b.polygon([p0, p1, p2], _UP, (0.5, 0.5))
        else:
            turn = 1.0 if mk.kind == "arrow:left" else -1.0
            turn *= d  # left of the arrow's own direction
            rect(min(s + d * 2.4, s + d * 3.0), max(s + d * 2.4, s + d * 3.0), c, c + turn * 0.6)
            pa = model.at(s + d * 2.7)
            y = pa[0][1] + MARK_LIFT_M
            o = np.array([pa[0][0], y, pa[0][2]])
            p0 = o + pa[2] * (c + turn * 0.6) + pa[1] * 0.35
            p1 = o + pa[2] * (c + turn * 0.6) - pa[1] * 0.35
            p2 = o + pa[2] * (c + turn * 1.1)
            b = cells.setdefault(int(s // cell_m), _Builder())
            b.polygon([p0, p1, p2], _UP, (0.5, 0.5))

    for mk in marks:
        if mk.kind == "stop_bar":
            rect(mk.s - STOP_BAR_M / 2, mk.s + STOP_BAR_M / 2, mk.l0, mk.l1)
        elif mk.kind == "crosswalk":
            for off in (-CROSSWALK_GAP_M / 2, CROSSWALK_GAP_M / 2):
                rect(mk.s + off - CROSSWALK_LINE_M / 2, mk.s + off + CROSSWALK_LINE_M / 2, mk.l0,
                     mk.l1)  # fmt: skip
        elif mk.kind == "zebra":
            width = mk.l1 - mk.l0
            n = max(1, int((width + ZEBRA_GAP_M) // (ZEBRA_BAR_M + ZEBRA_GAP_M)))
            used = n * ZEBRA_BAR_M + (n - 1) * ZEBRA_GAP_M
            for k in range(n):
                l0 = mk.l0 + (width - used) / 2 + k * (ZEBRA_BAR_M + ZEBRA_GAP_M)
                rect(mk.s - ZEBRA_LEN_M / 2, mk.s + ZEBRA_LEN_M / 2, l0, l0 + ZEBRA_BAR_M)
        elif mk.kind.startswith("arrow:"):
            arrow(mk)
    out = []
    for k in sorted(cells):
        data = cells[k].mesh()
        if data is not None:
            out.append((f"marks_{k:03d}", merge([data])))
    return out
