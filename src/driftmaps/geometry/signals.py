"""Ontario traffic signals: corner poles, mast arms, a head per lane, pedestrian signals.

The furniture layer (`geometry.furniture`) asks `plan_signals` for every signalised junction on
the route and draws the result with `draw_pole`. Coordinates: AC world (X east, Y up, Z south).

**Which junctions.** A route junction (a route node with side roads, `route.stubs`) is
signalised when an OSM `highway=traffic_signals` node lies within `SIGNAL_NODE_M` of it (on
the junction or on any approach: Ontario mappers do both), or Mapillary saw a traffic light
(`object--traffic-light--*`) within `MLY_LIGHT_M`. Every approach of a signalised junction is
signalised (the side streets too: the user's Burlington check, 2026-10-16, found signals on the
route's two approaches only), and no stop or yield sign stands on any of them
(`plan_furniture` drops them).

**Layout.** Numbers from docs/research/intersections.md §4 (the network builder's research:
[TOR-SIG] City of Toronto TTR 801.320, Location of Traffic Signal Equipment, 2012, after the
Highway Traffic Act s.124 and OTM Book 12), the rest ASSUMED:
- At least two heads per approach (HTA s.124(20)). The primary heads are on the FAR right
  ([TOR-SIG] §3.6.1, 12-55 m beyond the near stop line): each approach gets a corner pole on
  its far-right corner with a mast arm reaching left over its lanes, one 3-section 300 mm
  head (§3.4) centred over each through lane and a 3-section left-arrow head over each
  left-turn lane (lane counts from `lanes`, `lanes:forward|backward`, `turn:lanes*`); the
  secondary head is on the FAR LEFT (§3.6.2), side-mounted on the corner pole there (the far
  right pole of the approach from the left), or on the far-right pole when there is none
  (the open side of a T). With four legs each corner carries one arm.
- Heads: black housing 0.36 x 1.10 m, black backplate 0.66 x 1.36 m with a 75 mm fluorescent
  yellow retroreflective border (as Ontario installs them). Overhead heads: housing bottom
  `HEAD_CLEAR_M` (4.9 m, the standard; 4.5 m minimum) above the road, the arm `ARM_Y_M`
  above it; side-mounted heads: bottom `SIDE_HEAD_M` (4.9 m standard; 2.75 m minimum).
- Poles: galvanized steel, 9.1 m tall, at the corner `POLE_SETBACK_M` behind both curb
  lines; the furniture clearance (`geometry.clearance`) then moves them off the drivable
  surface (1 m behind its edge, §3.9: at least 1.0 m from the curb). The arm keeps its
  direction (square to the approach) and its heads stay over their lanes wherever the pole
  ends up; it is at most `MAX_ARM_M` (7.6 m, the longest standard arm, Tables 1-2) long, and
  heads beyond its reach are left out.
- A street name sign (white on green, the furniture atlas's blades) on each mast arm, naming
  the street the approach crosses.
- Pedestrian signals: at both ends of a crosswalk across every leg (centred 2.5 m beyond the
  crossing road's edge: a 3 m crosswalk 1 m out, intersections.md §3), a countdown head
  (bottom `PED_HEAD_M`, 2.75 m, §3.6.4) facing across the street, on the corner pole when one
  stands within `PED_SNAP_M` (3 m: push buttons within 3.0 m of the crosswalk, §3.9), else on
  a 3.9 m pedestrian post; each end also gets a push button at `BUTTON_M` (1.05 m; AODA
  accessible pedestrian signals: about 1.0-1.1 m, ASSUMED).
- Signals cannot change in AC: heads on the route's approaches show green, the side streets'
  red, left arrows dark; pedestrian heads show the orange hand.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from driftmaps.ac.furniture_textures import name_tile, sign_tile, swatch_uv

__all__ = [
    "ARM_Y_M",
    "HEAD_CLEAR_M",
    "SignalApproach",
    "SignalPlan",
    "SignalPole",
    "approach_lanes",
    "arm_reach",
    "draw_pole",
    "plan_junction",
    "plan_signals",
    "signalised",
]

SIGNAL_NODE_M = 40.0  # an OSM traffic_signals node this close to a route junction signals it
MLY_LIGHT_M = 30.0  # a Mapillary traffic light this close does too
POLE_SETBACK_M = 1.5  # corner poles stand this far behind both curb lines (before clearance)
MAX_ARM_M = 7.6  # the longest standard mast arm ([TOR-SIG] Tables 1-2)
POLE_M = 9.1  # signal pole height above the ground (a 7.6 m arm's pole)
HEAD_CLEAR_M = 4.9  # bottom of an overhead head's housing above the road (>= 4.5 m)
ARM_Y_M = HEAD_CLEAR_M + 1.10 + 0.25  # the arm, above the road (head + hanger)
SIDE_HEAD_M = 4.9  # bottom of a pole-mounted head above the ground (>= 2.75 m)
PED_HEAD_M = 2.75  # bottom of a pedestrian head above the ground (>= 2.75 m)
PED_SNAP_M = 3.0  # a crosswalk end this close to a corner pole uses it
BUTTON_M = 1.05  # push button height
CROSSWALK_SETBACK_M = 2.5  # crosswalk centre this far beyond the crossing road's curb line
HEAD_W, HEAD_H, HEAD_D = 0.36, 1.10, 0.24  # 3-section 300 mm head housing
PLATE_W, PLATE_H, PLATE_BORDER = 0.66, 1.36, 0.075  # backplate and its yellow border
LANE_W = (2.8, 3.75)  # lane width clamp
_UP = np.array([0.0, 1.0, 0.0])


def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)[:2]
    n = float(np.hypot(*v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _right(t: np.ndarray) -> np.ndarray:
    """The driver's right of travel direction t (XZ, X east, Z south: left = (t_z, -t_x))."""
    return np.array([-t[1], t[0]])


# --------------------------------------------------------------------------------------------
# Lanes
# --------------------------------------------------------------------------------------------


def _count(tags: Mapping[str, str], key: str) -> int | None:
    try:
        v = int(str(tags.get(key, "")).split(";")[0])
    except ValueError:
        return None
    return v if 1 <= v <= 8 else None


def _turns(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [lane.strip() for lane in str(value).split("|")]


def approach_lanes(tags: Mapping[str, str], toward: bool | None) -> tuple[int, int]:
    """(through lanes, left-turn-only lanes) for traffic arriving at a junction on a way with
    these OSM `tags`. `toward` True: that traffic travels in the way's direction (OSM
    forward), False: against it, None: unknown (the busier direction is assumed). Through
    lanes are every lane that is not left-only or right-only (at least one); without
    `turn:lanes` all lanes are through lanes."""
    oneway = str(tags.get("oneway", "")) in ("yes", "1", "true", "-1")
    if oneway:
        n = _count(tags, "lanes") or 1
        turns = _turns(tags.get("turn:lanes"))
    else:
        fwd, bwd = _count(tags, "lanes:forward"), _count(tags, "lanes:backward")
        total = _count(tags, "lanes")
        tf, tb = _turns(tags.get("turn:lanes:forward")), _turns(tags.get("turn:lanes:backward"))
        if toward is None:
            n = max(fwd or 0, bwd or 0) or (max(1, total // 2) if total else 1)
            turns = max((tf or [], tb or []), key=lambda t: sum("left" in x for x in t)) or None
        else:
            n = (fwd if toward else bwd) or (max(1, total // 2) if total else 1)
            turns = tf if toward else tb
    if turns:
        n = len(turns)
        left = sum(1 for x in turns if "left" in x and "through" not in x and "right" not in x)
        right = sum(1 for x in turns if "right" in x and "through" not in x and "left" not in x)
        return max(1, n - left - right), left
    return n, 0


# --------------------------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------------------------


@dataclass
class SignalApproach:
    """One leg of a junction: `direction` points away from the junction along it; traffic
    arriving on it travels -direction. `half` is its half width (two-way: centreline to edge;
    one-way: half the carriageway); `oneway` +1 traffic leaves the junction only, -1 arrives
    only, 0 both; `on_route` True for the route's own two legs. A divided road's two
    carriageways merged into one leg (`_merge`) carry `extent`, the paved span across the leg
    (offsets right of the arriving traffic, from the junction centre's axis), and `span`, the
    arriving lanes' part of it."""

    direction: np.ndarray
    half: float
    tags: Mapping[str, str]
    toward: bool | None = None  # arriving traffic moves in the OSM way's direction
    oneway: int = 0
    on_route: bool = False
    extent: tuple[float, float] | None = None
    span: tuple[float, float] | None = None

    @property
    def name(self) -> str:
        return str(self.tags.get("name", "")).strip()

    @property
    def arrives(self) -> bool:
        return self.oneway <= 0

    @property
    def across(self) -> tuple[float, float]:
        """(lo, hi): the paved span right of the arriving traffic's axis."""
        return self.extent if self.extent is not None else (-self.half, self.half)

    @property
    def lanes_span(self) -> tuple[float, float]:
        if self.span is not None:
            return self.span
        return (0.0, self.half) if self.oneway == 0 else (-self.half, self.half)


@dataclass
class SignalPole:
    """A corner pole (or, with `post`, a pedestrian post). `axis` is the approach's travel
    direction (its arm is square to it, reaching left), `origin` the junction centre; `heads`
    are (lateral offset right of the axis line through `origin`, "through" | "left"); `side`
    the facings of vehicle heads mounted on the pole, `ped` those of pedestrian heads, and
    `buttons` those of push buttons. `go` True: the vehicle heads show green."""

    ref: str
    xz: np.ndarray
    origin: np.ndarray
    axis: np.ndarray | None = None
    heads: list[tuple[float, str]] = field(default_factory=list)
    side: list[np.ndarray] = field(default_factory=list)
    side_go: list[bool] = field(default_factory=list)  # per side head: its approach's green
    ped: list[np.ndarray] = field(default_factory=list)
    buttons: list[np.ndarray] = field(default_factory=list)
    name: str = ""
    go: bool = False
    post: bool = False


@dataclass
class SignalPlan:
    poles: list[SignalPole] = field(default_factory=list)
    junctions: list[tuple[int, np.ndarray, float]] = field(default_factory=list)  # node, xz, r
    approaches: int = 0
    nodes: set[int] = field(default_factory=set)  # every route node of a signalised junction

    def stats(self) -> dict[str, int]:
        arms = [p for p in self.poles if not p.post]
        return {
            "junctions": len(self.junctions),
            "approaches": self.approaches,
            "poles": len(arms),
            "ped_posts": len(self.poles) - len(arms),
            "heads": sum(len(p.heads) + len(p.side) for p in arms),
            "arrow_heads": sum(1 for p in arms for _, k in p.heads if k == "left"),
            "ped_heads": sum(len(p.ped) for p in self.poles),
            "push_buttons": sum(len(p.buttons) for p in self.poles),
        }

    def near(self, xz: np.ndarray, extra_m: float = 0.0) -> bool:
        """True within a signalised junction's reach (its radius + `extra_m`)."""
        p = np.asarray(xz, np.float64)[:2]
        return any(float(np.hypot(*(p - c))) <= r + extra_m for _, c, r in self.junctions)


def signalised(
    junction_xz: np.ndarray,
    signal_xz: np.ndarray,
    light_xz: np.ndarray,
    others_xz: np.ndarray,
) -> bool:
    """A junction at `junction_xz` is signalised by an OSM signal node within SIGNAL_NODE_M
    that is not nearer another route junction (`others_xz`), or a Mapillary traffic light
    within MLY_LIGHT_M."""
    j = np.asarray(junction_xz, np.float64)
    for p in np.asarray(signal_xz, np.float64).reshape(-1, 2):
        d = float(np.hypot(*(p - j)))
        if d > SIGNAL_NODE_M:
            continue
        o = np.asarray(others_xz, np.float64).reshape(-1, 2)
        if d <= 12.0 or not len(o) or d <= float(np.min(np.hypot(*(o - p).T))) + 1e-6:
            return True
    lights = np.asarray(light_xz, np.float64).reshape(-1, 2)
    return bool(len(lights)) and float(np.min(np.hypot(*(lights - j).T))) <= MLY_LIGHT_M


def _lane_offsets(leg: SignalApproach) -> list[tuple[float, str]]:
    """(lateral offset right of the leg's axis, kind) of each arriving lane that gets an
    overhead head: left-turn lanes nearest the centre line, then the through lanes."""
    through, left = approach_lanes(leg.tags, leg.toward)
    n = through + left
    start, end = leg.lanes_span
    w = float(np.clip((end - start) / n, *LANE_W))
    kinds = ["left"] * left + ["through"] * through
    return [(start + (i + 0.5) * w, k) for i, k in enumerate(kinds)]


def plan_junction(
    node: int, centre: np.ndarray, legs: Sequence[SignalApproach]
) -> tuple[list[SignalPole], float]:
    """Corner poles and pedestrian posts for one signalised junction (module notes), and the
    junction's reach (centre to the farthest pole). Needs at least three legs."""
    c = np.asarray(centre, np.float64)
    if len(legs) < 3:
        return [], 0.0
    poles: list[SignalPole] = []
    far_left: list[tuple[np.ndarray, np.ndarray]] = []  # per arm pole: far-left corner, t
    for k, leg in enumerate(legs):
        if not leg.arrives:
            continue
        t = -_unit(leg.direction)  # arriving traffic's direction
        r = _right(t)
        across = [m for m in legs if m is not leg and abs(float(_unit(m.direction) @ t)) < 0.6]
        if not across:
            continue
        # The far curb of the crossing roads (along t), and the far leg's edge on our right
        # (the far leg's arriving traffic runs the other way: its "left" is our right).
        a = max(_reach(m, t) for m in across) + POLE_SETBACK_M
        ahead = [m for m in legs if float(_unit(m.direction) @ t) > 0.7]
        far = max(ahead, key=lambda m: float(_unit(m.direction) @ t)) if ahead else None
        b = (-far.across[0] if far is not None else leg.across[1]) + POLE_SETBACK_M
        heads = _lane_offsets(leg)
        name = max(across, key=lambda m: abs(float(_right(t) @ _unit(m.direction)))).name
        poles.append(
            SignalPole(
                ref=f"sig:{node}:{k}",
                xz=c + t * a + r * b,
                origin=c,
                axis=t,
                heads=heads,
                name=name,
                go=leg.on_route,
            )
        )
        far_left.append((c + t * a - r * ((far.across[1] if far is not None else leg.half)
                                          + POLE_SETBACK_M), t))  # fmt: skip
    # Secondary heads on the far-left corner's pole (on the far-right one when there is none).
    for pole, (corner, t) in zip(poles, far_left, strict=True):
        dist = [float(np.hypot(*(q.xz - corner))) for q in poles]
        k = int(np.argmin(dist))
        host = poles[k] if dist[k] <= 6.0 and poles[k] is not pole else pole
        host.side.append(-t)
        host.side_go.append(pole.go)  # the approach's own colour, not the host pole's
    # Pedestrian signals and push buttons at both ends of a crosswalk across every leg.
    for k, leg in enumerate(legs):
        d = _unit(leg.direction)
        n = _right(d)
        others = [m for m in legs if m is not leg and abs(float(_unit(m.direction) @ d)) < 0.6]
        cdist = (max(_reach(m, d) for m in others) if others else 6.0) + CROSSWALK_SETBACK_M
        lo, hi = leg.across  # right of the arriving traffic = left of d
        ends = [c + d * cdist - n * o for o in (lo - 1.0, hi + 1.0)]
        for e, other in ((ends[0], ends[1]), (ends[1], ends[0])):
            facing = _unit(other - e)
            arms = [p for p in poles if not p.post]
            dist = [float(np.hypot(*(p.xz - e))) for p in arms]
            if arms and min(dist) <= PED_SNAP_M:
                host = arms[int(np.argmin(dist))]
            else:
                host = SignalPole(ref=f"sig:{node}:p{k}{len(poles)}", xz=e, origin=c, post=True)
                poles.append(host)
            host.ped.append(facing)
            host.buttons.append(facing)
    reach = max((float(np.hypot(*(p.xz - c))) for p in poles), default=0.0)
    return poles, reach


def _reach(m: SignalApproach, t: np.ndarray) -> float:
    """How far leg m's paved span reaches from the junction centre along direction t (m runs
    across t)."""
    r_m = _right(-_unit(m.direction))  # right of m's arriving traffic
    lo, hi = m.across
    return hi if float(r_m @ t) > 0 else -lo


CLUSTER_M = 30.0  # route junctions this close are one intersection (a divided road's two)
_PARALLEL = math.cos(math.radians(25.0))


def _merge(members: Sequence[tuple[int, np.ndarray, Sequence[SignalApproach]]]):
    """One intersection from route junctions a few metres apart (the two carriageways of a
    divided road): legs that lead to another member are dropped, parallel legs (the two
    carriageways leaving the same way) become one with the carriageways' joint `extent` and
    the arriving one's lanes as `span`. Returns (node, centre, legs)."""
    if len(members) == 1:
        return members[0]
    centres = np.array([np.asarray(c, np.float64) for _, c, _ in members])
    centre = centres.mean(axis=0)
    legs: list[tuple[SignalApproach, np.ndarray]] = []
    for i, (_, c, ls) in enumerate(members):
        for leg in ls:
            d = _unit(leg.direction)
            inner = False
            for j, oc in enumerate(centres):
                v = oc - np.asarray(c, np.float64)
                dist = float(np.hypot(*v))
                if j != i and 0.5 < dist <= CLUSTER_M and float(d @ v) > 0.85 * dist:
                    inner = True
            if not inner:
                legs.append((leg, np.asarray(c, np.float64)))
    groups: list[list[tuple[SignalApproach, np.ndarray]]] = []
    for leg, c in legs:
        for g in groups:
            if float(_unit(g[0][0].direction) @ _unit(leg.direction)) > _PARALLEL:
                g.append((leg, c))
                break
        else:
            groups.append([(leg, c)])
    out: list[SignalApproach] = []
    for g in groups:
        d = _unit(np.sum([_unit(leg.direction) for leg, _ in g], axis=0))
        r = _right(-d)  # right of the arriving traffic
        spans = []
        for leg, c in g:
            o = float((c - centre) @ r)
            lo, hi = leg.across
            spans.append((o + lo, o + hi, leg))
        lo = min(x[0] for x in spans)
        hi = max(x[1] for x in spans)
        if len(g) == 1:
            leg, c = g[0]
            o = float((c - centre) @ r)
            s0, s1 = leg.lanes_span
            out.append(SignalApproach(d, max(-lo, hi), leg.tags, leg.toward, leg.oneway,
                                      leg.on_route, (lo, hi), (o + s0, o + s1)))  # fmt: skip
            continue
        # The arriving carriageway keeps right: the one farthest right.
        a_lo, a_hi, arriving = max(spans, key=lambda x: x[0] + x[1])
        out.append(SignalApproach(d, max(-lo, hi), arriving.tags, None, 0,
                                  any(leg.on_route for leg, _ in g), (lo, hi),
                                  (a_lo, a_hi)))  # fmt: skip
    node = min(int(n) for n, _, _ in members)
    return node, centre, out


def _clusters(centres: np.ndarray) -> list[list[int]]:
    """Indices of junctions grouped by chains of centres within CLUSTER_M."""
    n = len(centres)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if float(np.hypot(*(centres[i] - centres[j]))) <= CLUSTER_M:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def plan_signals(
    junctions: Sequence[tuple[int, np.ndarray, Sequence[SignalApproach]]],
    signal_xz: np.ndarray,
    light_xz: np.ndarray,
    nodes: Collection[int] = (),
) -> SignalPlan:
    """Poles for every signalised junction among `junctions` ((node, centre XZ, legs)):
    `signal_xz` are the OSM traffic signal nodes, `light_xz` Mapillary traffic lights (XZ).
    Route junctions within CLUSTER_M of each other (a divided road's two carriageways) are
    one intersection, signalised when any of them is. `nodes` are route nodes known to be
    signalised otherwise (the network's junction control, `RoadNetwork.junctions[].controls`:
    mapped or inferred signals)."""
    plan = SignalPlan()
    centres = np.array([np.asarray(j[1], np.float64) for j in junctions]).reshape(-1, 2)
    for group in _clusters(centres):
        others = np.delete(centres, group, axis=0)
        known = any(int(junctions[i][0]) in nodes for i in group)
        if not known and not any(
            signalised(centres[i], signal_xz, light_xz, others) for i in group
        ):
            continue
        node, centre, legs = _merge([junctions[i] for i in group])
        poles, reach = plan_junction(node, centre, legs)
        if not poles:
            continue
        plan.nodes.update(int(junctions[i][0]) for i in group)
        plan.poles += poles
        plan.junctions.append((int(node), np.asarray(centre, np.float64), reach))
        plan.approaches += sum(1 for p in poles if not p.post)
    return plan


# --------------------------------------------------------------------------------------------
# Meshes
# --------------------------------------------------------------------------------------------


def _quad_uv(b, centre, right, up, w, h, nrm, uv) -> None:
    """A w x h quad facing `nrm`; `uv` a constant (u, v) or a (u0, v0, u1, v1) tile."""
    pts = [centre - right * w / 2 - up * h / 2, centre + right * w / 2 - up * h / 2,
           centre + right * w / 2 + up * h / 2, centre - right * w / 2 + up * h / 2]  # fmt: skip
    if len(uv) == 4:
        u0, v0, u1, v1 = uv
        du, dv = 0.5 / 1024, 0.5 / 1024
        uvs = np.array([[u0 + du, v1 - dv], [u1 - du, v1 - dv], [u1 - du, v0 + dv],
                        [u0 + du, v0 + dv]])  # fmt: skip
    else:
        uvs = np.tile(np.asarray(uv, float), (4, 1))
    b.polygon(np.array(pts), nrm, uvs)


def _vehicle_head(cell, centre: np.ndarray, facing: np.ndarray, kind: str, go: bool) -> None:
    """A 3-section 300 mm head (housing centre `centre`) with its yellow-bordered backplate."""
    f3 = np.array([facing[0], 0.0, facing[1]])
    right = np.cross(_UP, f3)  # viewer's right when facing the head
    black = swatch_uv("signal_black")
    cell.furn.box(centre, (HEAD_D, HEAD_H, HEAD_W), f3, black)
    plate = centre - f3 * (HEAD_D / 2 + 0.01)
    _quad_uv(cell.furn, plate, right, _UP, PLATE_W, PLATE_H, f3, swatch_uv("backplate_yellow"))
    inner = plate + f3 * 0.004
    _quad_uv(cell.furn, inner, right, _UP, PLATE_W - 2 * PLATE_BORDER, PLATE_H - 2 * PLATE_BORDER,
             f3, black)  # fmt: skip
    _quad_uv(cell.furn, plate - f3 * 0.004, right, _UP, PLATE_W, PLATE_H, -f3, black)  # back
    front = centre + f3 * (HEAD_D / 2 + 0.006)
    if kind == "left":
        lit = ("lens_off", "lens_off", "lens_arrow_left") if go else ("lens_red", "lens_off",
                                                                       "lens_off")  # fmt: skip
    else:
        lit = ("lens_off", "lens_off", "lens_green") if go else ("lens_red", "lens_off",
                                                                  "lens_off")  # fmt: skip
    for k, lens in enumerate(lit):
        q = front + _UP * (0.35 - 0.35 * k)
        if lens == "lens_off":
            _quad_uv(cell.furn, q, right, _UP, 0.28, 0.28, f3, swatch_uv(lens))
        elif lens == "lens_arrow_left":
            _quad_uv(cell.signs, q, right, _UP, 0.28, 0.28, f3, sign_tile(lens).uv)
        else:
            _quad_uv(cell.signs, q, right, _UP, 0.28, 0.28, f3, swatch_uv(lens))


def _ped_head(cell, centre: np.ndarray, facing: np.ndarray) -> None:
    f3 = np.array([facing[0], 0.0, facing[1]])
    right = np.cross(_UP, f3)
    cell.furn.box(centre, (0.22, 0.46, 0.46), f3, swatch_uv("signal_black"))
    front = centre + f3 * 0.116
    _quad_uv(cell.signs, front + _UP * 0.0, right, _UP, 0.36, 0.36, f3, sign_tile("ped_hand").uv)


def _button(cell, base: np.ndarray, facing: np.ndarray, y: float) -> None:
    f3 = np.array([facing[0], 0.0, facing[1]])
    right = np.cross(_UP, f3)
    c = np.array([base[0], y, base[2]]) + f3 * 0.2
    cell.furn.box(c, (0.08, 0.24, 0.12), f3, swatch_uv("signal_black"))
    _quad_uv(cell.furn, c + f3 * 0.042, right, _UP, 0.08, 0.12, f3, swatch_uv("paint_white"))


def draw_pole(
    cell,
    pole: SignalPole,
    base: np.ndarray,
    road_y: Callable[[np.ndarray], np.ndarray],
    names: Mapping[str, int],
    metal: tuple[float, float],
) -> None:
    """The meshes of one pole standing at `base` (its position after clearance, XYZ on the
    ground): `cell` has `furn` (atlas) and `signs` (bright faces) builders; `road_y(xz)` is
    the road height there; `names` the atlas index per street name."""
    base = np.asarray(base, np.float64)
    gy = float(base[1])
    pxz = base[[0, 2]]
    if pole.post:
        cell.furn.prism(base + np.array([0.0, -0.4, 0.0]), 4.3, 0.07, 0.06, 8, metal)
        for k, f in enumerate(pole.ped):
            f3 = np.array([f[0], 0.0, f[1]])
            _ped_head(cell, np.array([base[0], gy + PED_HEAD_M + 0.23 + 0.5 * (k % 2), base[2]])
                      + f3 * 0.3, f)  # fmt: skip
        for f in pole.buttons:
            _button(cell, base, f, gy + BUTTON_M)
        return
    t = _unit(pole.axis if pole.axis is not None else np.array([1.0, 0.0]))
    r = _right(t)
    lat = float((pxz - pole.origin) @ r)  # the pole's offset right of the approach's axis
    lanes = [(lat - off, kind) for off, kind in pole.heads if 0.6 <= lat - off <= MAX_ARM_M]
    hxz = [pxz - r * along for along, _ in lanes]
    ry = max([float(road_y(np.asarray(h)[None, :])[0]) for h in hxz] + [gy - 0.3])
    arm_y = ry + ARM_Y_M
    top = max(arm_y + 0.9, gy + POLE_M)
    cell.furn.prism(base + np.array([0.0, -0.4, 0.0]), top - gy + 0.4, 0.17, 0.12, 8, metal)
    f2 = -t  # heads face the arriving drivers
    f3 = np.array([f2[0], 0.0, f2[1]])
    a3 = -np.array([r[0], 0.0, r[1]])  # the arm reaches left over the lanes
    if lanes:
        sign_w = float(np.clip(0.11 * len(pole.name) + 0.5, 1.2, 2.4)) if pole.name else 0.0
        near = min(a for a, _ in lanes) - HEAD_W  # free arm between the pole and the 1st head
        far = max(a for a, _ in lanes) + 0.5
        sign_at = None
        if sign_w:
            sign_at = 0.4 + sign_w / 2 if near >= sign_w + 0.6 else far + 0.2 + sign_w / 2
        length = max(far, (sign_at + sign_w / 2 + 0.2) if sign_at else 0.0)
        p0 = np.array([base[0], arm_y, base[2]])
        cell.furn.box(p0 + a3 * length / 2, (length, 0.2, 0.2), a3, metal)
        for along, kind in lanes:
            hc = p0 + a3 * along - _UP * (0.25 + HEAD_H / 2)
            cell.furn.box(p0 + a3 * along - _UP * 0.12, (0.06, 0.24, 0.06), a3, metal)  # hanger
            _vehicle_head(cell, hc, f2, kind, pole.go)
        idx = names.get(pole.name) if pole.name else None
        if sign_at is not None and idx is not None:
            sc = p0 + a3 * sign_at - _UP * (0.1 + 0.25)
            right = np.cross(_UP, f3)
            for sgn in (1.0, -1.0):  # readable from both sides
                _quad_uv(cell.signs, sc + f3 * 0.02 * sgn, right * sgn, _UP, sign_w, 0.5,
                         f3 * sgn, name_tile(idx).uv)  # fmt: skip
    # Pole-mounted heads, pedestrian heads, push buttons.
    for k, f in enumerate(pole.side):
        s3 = np.array([f[0], 0.0, f[1]])
        c = np.array([base[0], gy + SIDE_HEAD_M + HEAD_H / 2, base[2]]) + s3 * 0.35
        go = pole.side_go[k] if k < len(pole.side_go) else pole.go
        _vehicle_head(cell, c + _UP * (1.3 * k), f, "through", go)
    for f in pole.ped:
        p3 = np.array([f[0], 0.0, f[1]])
        _ped_head(cell, np.array([base[0], gy + PED_HEAD_M + 0.23, base[2]]) + p3 * 0.32, f)
    for f in pole.buttons:
        _button(cell, base, f, gy + BUTTON_M)


def arm_reach(pole: SignalPole, xz: np.ndarray) -> float:
    """How far the arm of `pole`, standing at `xz`, reaches (0 for a post)."""
    if pole.post or pole.axis is None or not pole.heads:
        return 0.0
    r = _right(_unit(pole.axis))
    lat = float((np.asarray(xz, np.float64)[:2] - pole.origin) @ r)
    reach = [lat - off for off, _ in pole.heads if 0.6 <= lat - off <= MAX_ARM_M]
    return max(reach, default=0.0) + 0.5
