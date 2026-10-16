"""Street furniture kept off the drivable surface.

Mapillary positions are triangulated from street photos and are often metres off: poles land
on the road, signs in fields (they belong to roads we do not build, or the triangulation went
wrong), and about 22 % of the signs face the wrong way. Pole lines and rule-made items are laid
out along the route and can cross side roads, junction mouths or roundabout aprons.

`DrivableSurface` is everything a car can drive on, in the AC XZ plane (X east, Z south): the
route's asphalt and gravel shoulders, the network's side roads with their shoulders, junction
surfaces, roundabout rings and truck aprons, cul-de-sac bulbs and driveways. Islands (roundabout
central islands inside their aprons, splitter islands, turning-loop islands) are not drivable:
an item may stand on one if it fits.

`clear_items` then, per item:
- **Measured signs** (Mapillary) farther than `SIGN_FAR_M` from the nearest built pavement are
  dropped (signs in fields), and Mapillary poles, lights and hydrants farther than `MLY_FAR_M`
  (unless a pole lies on an OSM power line). Measured signs are re-faced towards the traffic
  of their road (the nearby road most aligned with the measured facing): square to it,
  facing the drivers who have them on the driving side (`SIGN_SIDE` for chevrons and
  keep-right signs; on a one-way road: facing its traffic, either side).
- **Poles, lights, signals, hydrants, sign posts and mailboxes** on the surface or closer than
  `CLEARANCE_M` to it move out past the nearest edge of the surface (within
  `NEAREST_EDGE_MOVE_M`; a sign stays at its junction corner), else square off the nearest
  road (on the item's side if there is room), at the kind's offset beyond the edge: sign plates
  `sign_gap_m` beyond the pavement edge, poles `pole_edge_m` beyond the shoulder, lights,
  signal poles and hydrants `LIGHT_EDGE_M`, mailboxes `MAILBOX_EDGE_M`. On an island and too
  close to its edge: into the island; an island too small, or no room within `MAX_MOVE_M`:
  dropped. Rule-made items keep their own placement otherwise.
- **Wires** follow their poles; a dropped pole's two spans become one where it stays short.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.spatial import cKDTree

from driftmaps.geometry.road import RoadModel, footprint_polygon
from driftmaps.geometry.street_items import (
    HYDRANT,
    LIGHT_STREET,
    MAILBOX,
    POLE_UTILITY,
    SIGN_CHEVRON_LEFT,
    SIGN_CHEVRON_RIGHT,
    SIGN_KEEP_RIGHT,
    SIGNAL_TRAFFIC,
    StreetItem,
)

__all__ = [
    "CLEARANCE_M",
    "MLY_FAR_M",
    "SIGN_FAR_M",
    "ClearResult",
    "DrivableSurface",
    "clear_items",
]

# Nothing stands closer than this to the drivable surface (the user's report: "a power pole
# in the middle of the road"); 0.5 m keeps a post's base and a sign plate's edge off the
# shoulder.
CLEARANCE_M = 0.5
# Rural Ontario mailboxes stand right at the shoulder's edge (ASSUMED).
MAILBOX_CLEARANCE_M = 0.2
# Where a moved item stands beyond the drivable edge (ASSUMED; OPSD / municipal practice puts
# street lights, signal poles and hydrants 0.5-2 m behind the curb or shoulder).
LIGHT_EDGE_M = 1.0
MAILBOX_EDGE_M = 0.6
# A sign is for drivers on a road we build: farther than this from the nearest built pavement
# it stands in a field (its road is not built, or Mapillary triangulated it badly).
SIGN_FAR_M = 15.0
# A Mapillary pole, light or hydrant this far from every built road is not part of the
# roadside (poles on an OSM power line excepted: they carry a line across the fields).
MLY_FAR_M = 25.0
POWER_LINE_M = 5.0  # a pole this close to an OSM power line stands on it
# An item that would have to move farther than this has no sensible place: dropped.
MAX_MOVE_M = 20.0
# Moved past the nearest edge of the surface when that is at most this far (else squared off
# its road, which also finds the way out of a wide junction or across a roundabout's apron).
NEAREST_EDGE_MOVE_M = 8.0
# Into an island: at most this far (else the island is too small or the item not on it).
ISLAND_MOVE_M = 4.0
# The search for a road's outside edge: from its centreline, in steps, this far out.
SCAN_STEP_M = 0.25
SCAN_MAX_M = 45.0
# Past the kind's offset, look this much farther for room (another road's shoulder, a fillet).
SCAN_SLACK_M = 3.0
# A sign faces the road it is most aligned with among those at most this much farther than
# the nearest one (a stop sign on a side road's corner stands near the route too).
REFACE_SLACK_M = 8.0
# Which side of their drivers signs stand on: "right" / "left", "keep" (keep the measured
# sense, only square it to the road), default: the driving side. A chevron stands on the
# outside of the curve and points the way the road turns, so a left-pointing one is on the
# drivers' right and a right-pointing one on their left (both ways of travel share the post);
# a keep-right sign stands on a median or island, which the centreline may run through.
SIGN_SIDE = {SIGN_CHEVRON_LEFT: "right", SIGN_CHEVRON_RIGHT: "left", SIGN_KEEP_RIGHT: "keep"}
EDGE_SPACING_M = 0.2  # boundary points for the distance queries (error < 0.01 m at 0.5 m)
ROAD_SPACING_M = 1.0  # centreline points for the nearest-road queries

_MOVABLE = (POLE_UTILITY, LIGHT_STREET, SIGNAL_TRAFFIC, HYDRANT, MAILBOX)


def _is_sign(kind: str) -> bool:
    return kind.startswith("sign:")


def _group(kind: str) -> str:
    return "sign" if _is_sign(kind) else kind.split(":")[0]


def _union(geoms: Sequence[shapely.Geometry]) -> shapely.Geometry:
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return shapely.Polygon()
    return shapely.union_all(np.array(geoms, dtype=object))


def _edge_tree(geom: shapely.Geometry) -> cKDTree | None:
    if geom.is_empty:
        return None
    pts = shapely.get_coordinates(shapely.segmentize(geom.boundary, EDGE_SPACING_M))
    return cKDTree(pts) if len(pts) else None


@dataclass
class _Road:
    xz: np.ndarray  # (K, 2) centreline
    oneway: np.ndarray  # (K,) +1 traffic along the polyline, -1 against, 0 both ways
    signed: bool = True  # signs may face its traffic (not a driveway's)


class DrivableSurface:
    """The drivable surface (`drivable`), the built pavement (`pavement`: asphalt and gravel
    roads without shoulders, aprons or driveways), the islands, and the road centrelines
    (for an item's road, its edge and its traffic)."""

    def __init__(
        self,
        drivable: shapely.Geometry,
        pavement: shapely.Geometry,
        islands: Sequence[shapely.Geometry],
        roads: Sequence[_Road],
        driving_side: str = "right",
    ) -> None:
        self.drivable = shapely.make_valid(drivable) if not drivable.is_valid else drivable
        self.pavement = pavement
        self.islands = [g for g in islands if not g.is_empty]
        self.island_all = _union(self.islands)
        for g in (self.drivable, self.pavement, self.island_all):
            shapely.prepare(g)
        self._drv_tree = _edge_tree(self.drivable)
        self._pav_tree = _edge_tree(self.pavement)
        self.right = driving_side != "left"
        pts, tan, ow, rid, signed = [], [], [], [], []
        for n, r in enumerate(roads):
            xz = np.asarray(r.xz, float)
            if len(xz) < 2:
                continue
            line = shapely.LineString(xz)
            dense = shapely.get_coordinates(shapely.segmentize(line, ROAD_SPACING_M))
            t = np.gradient(dense, axis=0)
            t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
            # oneway per dense point: from the nearest original vertex
            k = cKDTree(xz).query(dense)[1]
            pts.append(dense)
            tan.append(t)
            ow.append(np.asarray(r.oneway, np.int64)[k])
            rid.append(np.full(len(dense), n))
            signed.append(np.full(len(dense), r.signed))
        self.road_pts = np.concatenate(pts) if pts else np.zeros((0, 2))
        self.road_tan = np.concatenate(tan) if tan else np.zeros((0, 2))
        self.road_oneway = np.concatenate(ow) if ow else np.zeros(0, np.int64)
        self.road_id = np.concatenate(rid) if rid else np.zeros(0, np.int64)  # 0: the route
        self.road_signed = np.concatenate(signed) if signed else np.zeros(0, bool)
        self._road_tree = cKDTree(self.road_pts) if len(self.road_pts) else None
        self._signed_ids = np.flatnonzero(self.road_signed)
        self._signed_tree = (
            cKDTree(self.road_pts[self._signed_ids]) if len(self._signed_ids) else None
        )

    # -- construction -----------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        model: RoadModel,
        *,
        oneway: np.ndarray | None = None,
        net=None,
        driving_side: str = "right",
    ) -> DrivableSurface:
        """The route (`model`, its per-station `oneway`) and, with `net` (a
        `geometry.network.RoadNetwork`), its side roads, junctions, roundabouts, driveways."""
        c = model.centre
        left = model.left
        hw = np.asarray(model.half_width, float)[:, None]
        outer = hw + np.asarray(model.shoulder, float)[:, None]
        route_asphalt = footprint_polygon(c + left * hw, c - left * hw)
        route_foot = footprint_polygon(c + left * outer, c - left * outer)
        n = len(c)
        ow = np.zeros(n, np.int64)
        if oneway is not None and len(oneway) == n:
            ow = np.asarray(oneway, np.int64)
        roads = [_Road(c[:, [0, 2]], ow)]
        drivable = [route_foot]
        pavement = [route_asphalt]
        islands: list[shapely.Geometry] = []
        if net is not None:
            right = driving_side != "left"
            aprons, cores = [], []
            for rb in getattr(net, "roundabouts", ()):
                ctr = np.asarray(rb.centre, float)
                aprons.append(
                    shapely.Point(ctr)
                    .buffer(rb.inner_r, quad_segs=48)
                    .difference(shapely.Point(ctr).buffer(rb.island_r, quad_segs=48))
                )
                ang = np.linspace(0.0, 2 * math.pi, max(24, int(2 * math.pi * rb.radius)) + 1)
                ring = ctr + rb.radius * np.stack([np.cos(ang), -np.sin(ang)], axis=1)
                # increasing angle runs counter-clockwise on the map: right-hand traffic
                roads.append(_Road(ring, np.full(len(ring), 1 if right else -1)))
            apron_all = _union(aprons)
            island_full = _union(list(net.islands))
            for isl in net.islands:
                core = isl.difference(apron_all) if not apron_all.is_empty else isl
                if not core.is_empty:
                    cores.append(core)
            islands = cores
            for sr in net.side_roads:
                roads.append(_Road(sr.xz, np.full(len(sr.xz), int(sr.oneway))))
            dw_polys = []
            for d in getattr(net, "driveways", ()) or ():
                dw_polys.append(d.polygon)
                path = np.asarray(d.path, float)
                roads.append(_Road(path, np.zeros(len(path), np.int64), signed=False))
            drivable += [net.footprint, apron_all, *dw_polys]
            pav = net.paved.difference(island_full) if not island_full.is_empty else net.paved
            pavement.append(pav)
        drv = _union(drivable)
        core_all = _union(islands)
        if not core_all.is_empty:
            drv = drv.difference(core_all)
        return cls(drv, _union(pavement), islands, roads, driving_side)

    # -- queries ----------------------------------------------------------------------------

    def inside(self, xz: np.ndarray) -> np.ndarray:
        p = np.asarray(xz, float).reshape(-1, 2)
        return shapely.contains_xy(self.drivable, p[:, 0], p[:, 1])

    def distance(self, xz: np.ndarray) -> np.ndarray:
        """Distance to the drivable surface (0 on it)."""
        p = np.asarray(xz, float).reshape(-1, 2)
        if self._drv_tree is None:
            return np.full(len(p), np.inf)
        d = self._drv_tree.query(p)[0]
        d[self.inside(p)] = 0.0
        return d

    def pavement_distance(self, xz: np.ndarray) -> np.ndarray:
        """Distance to the built pavement (0 on it)."""
        p = np.asarray(xz, float).reshape(-1, 2)
        if self._pav_tree is None:
            return np.full(len(p), np.inf)
        d = self._pav_tree.query(p)[0]
        d[shapely.contains_xy(self.pavement, p[:, 0], p[:, 1])] = 0.0
        return d

    def nearest_edge(self, xz: np.ndarray) -> np.ndarray | None:
        """The nearest point of the drivable surface's outline (None: no surface)."""
        if self._drv_tree is None:
            return None
        return self._drv_tree.data[self._drv_tree.query(np.asarray(xz, float).reshape(2))[1]]

    def island_at(self, xz: np.ndarray) -> int:
        """Index of the island holding the point, -1 if none."""
        p = shapely.Point(np.asarray(xz, float).reshape(2))
        if self.island_all.is_empty or not shapely.contains(self.island_all, p):
            return -1
        for k, isl in enumerate(self.islands):
            if isl.contains(p):
                return k
        return -1

    def roads_near(self, xz: np.ndarray, slack: float = 0.0, signed: bool = False) -> list[int]:
        """Indices of road points: the nearest first, then (with `slack`) every point within
        `slack` m more, nearest first. `signed`: only roads whose traffic signs may face."""
        tree, ids = (self._signed_tree, self._signed_ids) if signed else (self._road_tree, None)
        if tree is None:
            return []
        p = np.asarray(xz, float).reshape(2)
        d0, i0 = tree.query(p)
        if not np.isfinite(d0):
            return []
        near = [int(i0)]
        if slack > 0.0:
            ball = np.asarray(tree.query_ball_point(p, d0 + slack), np.int64)
            order = np.argsort(np.linalg.norm(tree.data[ball] - p, axis=1), kind="stable")
            near = ball[order].tolist()
        return near if ids is None else [int(ids[i]) for i in near]


# --------------------------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------------------------


@dataclass
class ClearResult:
    items: list[StreetItem]
    wires: list[np.ndarray]
    stats: Counter = field(default_factory=Counter)
    dropped: list[tuple[StreetItem, str]] = field(default_factory=list)  # (item, why)


def _left(t: np.ndarray) -> np.ndarray:
    """The left of travel direction `t` in AC XZ (X east, Z south)."""
    return np.array([t[1], -t[0]])


def _facing(surface: DrivableSurface, xz: np.ndarray, facing: np.ndarray, kind: str) -> np.ndarray:
    """A sign's facing towards the traffic of its road (see the module notes, `SIGN_SIDE`)."""
    cand = surface.roads_near(xz, REFACE_SLACK_M, signed=True)
    if not cand:
        return facing
    f = facing / max(float(np.linalg.norm(facing)), 1e-9)
    best = max(cand, key=lambda i: abs(float(surface.road_tan[i] @ f)))
    t = surface.road_tan[best]
    ow = int(surface.road_oneway[best])
    if ow != 0:
        return -ow * t
    lat = float((xz - surface.road_pts[best]) @ _left(t))
    rule = SIGN_SIDE.get(kind, "right" if surface.right else "left")
    if rule == "keep" or abs(lat) < 1e-6:
        return t if float(f @ t) >= 0.0 else -t
    side = 1.0 if lat > 0 else -1.0  # +1: left of t, the right of drivers going -t
    return (side if rule == "right" else -side) * t


class _Placer:
    def __init__(self, surface: DrivableSurface, plate_width, sign_gap_m, pole_edge_m):
        self.s = surface
        self.plate_width = plate_width
        self.sign_gap = sign_gap_m
        self.pole_edge = pole_edge_m

    def clearance(self, kind: str) -> float:
        return MAILBOX_CLEARANCE_M if kind == MAILBOX else CLEARANCE_M

    def fits(self, kind: str, q: np.ndarray) -> bool:
        clear = self.clearance(kind)
        if self.s.distance(q)[0] < clear:
            return False
        if kind == POLE_UTILITY and self.s.island_at(q) >= 0:
            return False  # a hydro line does not cross a roundabout
        return True

    def _scan(self, kind: str, foot: np.ndarray, out: np.ndarray) -> np.ndarray | None:
        steps = np.arange(0.0, SCAN_MAX_M, SCAN_STEP_M)
        pts = foot + out * steps[:, None]
        on_drv = self.s.inside(pts)
        on_pav = shapely.contains_xy(self.s.pavement, pts[:, 0], pts[:, 1])
        off_drv = np.flatnonzero(~on_drv)
        if not len(off_drv):
            return None
        l_edge = steps[off_drv[0]]
        off_pav = np.flatnonzero(~on_pav)
        l_pave = steps[off_pav[0]] if len(off_pav) else l_edge
        clear = self.clearance(kind)
        if _is_sign(kind):
            w = self.plate_width(kind) / 2
            lat = max(l_pave + self.sign_gap + w, l_edge + clear + w)
        elif kind == POLE_UTILITY:
            lat = l_edge + self.pole_edge
        elif kind == MAILBOX:
            lat = l_edge + MAILBOX_EDGE_M
        else:
            lat = l_edge + LIGHT_EDGE_M
        for extra in np.arange(0.0, SCAN_SLACK_M + 1e-9, 0.5):
            q = foot + out * (lat + extra)
            if self.fits(kind, q):
                return q
        return None

    def place(self, it: StreetItem, xz: np.ndarray) -> tuple[np.ndarray | None, str]:
        """Where an item on (or too close to) the surface goes: (xz, "moved"), (None, why)."""
        k = self.s.island_at(xz)
        if k >= 0:  # on an island, too close to its edge: into it, if it fits
            if it.kind == POLE_UTILITY:
                return None, "island"
            inner = self.s.islands[k].buffer(-(self.clearance(it.kind) + 0.05))
            if inner.is_empty:
                return None, "island"
            q = shapely.get_coordinates(shapely.shortest_line(inner, shapely.Point(xz)))[0]
            if float(np.linalg.norm(q - xz)) > ISLAND_MOVE_M or not self.fits(it.kind, q):
                return None, "island"
            return q, "moved"
        # The nearest way out: past the nearest edge (keeps a sign at its junction corner).
        edge = self.s.nearest_edge(xz)
        if edge is not None:
            u = edge - xz if self.s.inside(xz)[0] else xz - edge
            if float(np.linalg.norm(u)) > 1e-3:
                q = self._scan(it.kind, xz, u / np.linalg.norm(u))
                if q is not None and float(np.linalg.norm(q - xz)) <= NEAREST_EDGE_MOVE_M:
                    return q, "moved"
        # Else square off the nearest road: the item's side first, else the other.
        cand = self.s.roads_near(xz)
        if not cand:
            return None, "no_road"
        i = cand[0]
        foot, t = self.s.road_pts[i], self.s.road_tan[i]
        n = _left(t)
        lat = float((xz - foot) @ n)
        own = 1.0 if lat > 0 else -1.0
        if abs(lat) < 0.3 and _is_sign(it.kind):  # on the centreline: the facing decides
            f = np.asarray(it.facing, float)
            along = float(f @ t)
            if abs(along) > 0.3:
                own = (1.0 if along > 0 else -1.0) * (1.0 if self.s.right else -1.0)
        best, best_cost = None, math.inf
        for side in (own, -own):
            q = self._scan(it.kind, foot, side * n)
            if q is None:
                continue
            cost = float(np.linalg.norm(q - xz)) + (0.0 if side == own else 2.0)
            if cost < best_cost:
                best, best_cost = q, cost
        if best is None:
            return None, "no_room"
        if float(np.linalg.norm(best - xz)) > MAX_MOVE_M:
            return None, "no_room"
        return best, "moved"


def _on_power_line(xz: np.ndarray, lines: Sequence[np.ndarray]) -> bool:
    p = shapely.Point(xz)
    return any(len(ln) >= 2 and shapely.LineString(ln).distance(p) <= POWER_LINE_M for ln in lines)


def clear_items(
    items: Sequence[StreetItem],
    surface: DrivableSurface,
    ground: Callable[[np.ndarray], np.ndarray],
    *,
    plate_width: Callable[[str], float],
    sign_gap_m: float,
    pole_edge_m: float,
    wires: Sequence[np.ndarray] = (),
    max_span_m: float = 126.0,
    power_lines: Sequence[np.ndarray] = (),
    first: bool = True,
) -> ClearResult:
    """Items off the drivable surface (see the module notes). `first`: the pass before the
    pole lines: also drop measured items far from every road and re-face measured signs, and
    leave poles beside the route to the pole lines (they take the offset of their line);
    `wires` (2, 3) pairs follow their poles; a dropped pole's spans are joined when the new
    span is at most `max_span_m`."""
    pl = _Placer(surface, plate_width, sign_gap_m, pole_edge_m)
    stats: Counter = Counter()
    dropped: list[tuple[StreetItem, str]] = []
    out: list[StreetItem] = []
    moved: dict[tuple[float, float], np.ndarray | None] = {}  # pole (x, z) -> new xyz / None
    pav_d = surface.pavement_distance(
        np.array([(it.position[0], it.position[2]) for it in items]).reshape(-1, 2)
    )
    for it, pd in zip(items, pav_d, strict=True):
        if not (_is_sign(it.kind) or it.kind in _MOVABLE):
            out.append(it)
            continue
        xz = np.array([it.position[0], it.position[2]], float)
        g = _group(it.kind)
        key = (round(it.position[0], 2), round(it.position[2], 2))
        mly = it.source == "mapillary"
        if first and mly:
            far = (_is_sign(it.kind) and pd > SIGN_FAR_M) or (
                it.kind in (POLE_UTILITY, LIGHT_STREET, HYDRANT)
                and pd > MLY_FAR_M
                and not (it.kind == POLE_UTILITY and _on_power_line(xz, power_lines))
            )
            if far:
                stats[f"dropped_far:{g}"] += 1
                dropped.append((it, "far"))
                if it.kind == POLE_UTILITY:
                    moved[key] = None
                continue
        new = it
        route_pole = (
            first
            and it.kind == POLE_UTILITY
            and bool(cand := surface.roads_near(xz))
            and surface.road_id[cand[0]] == 0
        )
        if not route_pole and surface.distance(xz)[0] < pl.clearance(it.kind):
            q, why = pl.place(it, xz)
            if q is None:
                stats[f"dropped_{why}:{g}"] += 1
                dropped.append((it, why))
                if it.kind == POLE_UTILITY:
                    moved[key] = None
                continue
            y = float(np.asarray(ground(q.reshape(1, 2)), float).reshape(-1)[0])
            new = StreetItem(it.kind, (float(q[0]), y, float(q[1])), it.facing, it.source,
                             it.text, it.ref)  # fmt: skip
            stats[f"moved:{g}"] += 1
            if it.kind == POLE_UTILITY:
                moved[key] = np.array(new.position)
        if first and mly and _is_sign(it.kind):
            pos = np.array([new.position[0], new.position[2]])
            f0 = np.asarray(new.facing, float)
            f = _facing(surface, pos, f0, it.kind)
            if float(f @ f0) < 0.0:
                stats["flipped:sign"] += 1
            if float(np.hypot(*(f - f0))) > 1e-3:
                stats["refaced:sign"] += 1
                new = StreetItem(new.kind, new.position, (round(float(f[0]), 6),
                                 round(float(f[1]), 6)), new.source, new.text, new.ref)  # fmt: skip
        out.append(new)
    return ClearResult(out, _rewire(wires, moved, max_span_m), stats, dropped)


def _rewire(
    wires: Sequence[np.ndarray], moved: dict[tuple[float, float], np.ndarray | None], max_span_m
) -> list[np.ndarray]:
    """Wires on moved poles follow them; around a dropped pole its two neighbours join up."""
    if not moved:
        return list(wires)

    def key(p) -> tuple[float, float]:
        return (round(float(p[0]), 2), round(float(p[2]), 2))

    out: list[np.ndarray] = []
    loose: dict[tuple[float, float], list[np.ndarray]] = {}
    for a, b in wires:
        ka, kb = key(a), key(b)
        na = moved.get(ka, a) if ka in moved else a
        nb = moved.get(kb, b) if kb in moved else b
        if na is None and nb is None:
            continue
        if na is None:
            loose.setdefault(ka, []).append(np.asarray(nb))
            continue
        if nb is None:
            loose.setdefault(kb, []).append(np.asarray(na))
            continue
        out.append(np.stack([np.asarray(na, float), np.asarray(nb, float)]))
    for ends in loose.values():
        if len(ends) == 2:
            a, b = ends
            if float(np.hypot(a[0] - b[0], a[2] - b[2])) <= max_span_m:
                out.append(np.stack([a, b]))
    return out
