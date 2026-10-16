"""Roundabouts, built whole the way Ontario builds them (TAC / Region of Halton designs).

An OSM `junction=roundabout` ring (its ways grouped by shared nodes) becomes:

- **Circulatory roadway**: a true circle fitted to the ring's nodes (OSM traces the middle of
  the circulatory road), `CIRCULATORY_WIDTH_M` wide.
- **Truck apron** inside it: `APRON_WIDTH_M` of pavers, raised `APRON_RAISE_M` behind a
  mountable (sloped) curb, so a car can roll onto it (it is physical road).
- **Central island**: a barrier curb (`ISLAND_CURB_M`, a real vertical face) and a grass mound
  (`MOUND_M` high) with a hidden wall inside the curb: it blocks.
- **Approach legs**: every road meeting the ring within the scenery radius. Entries and exits
  are rounded into the ring with `ENTRY_RADIUS_M` (closing of ring + legs). Where OSM maps a leg
  as a pair of one-way carriageways, the raised **splitter island** is the ground between them
  (from the ring to where they merge); a two-way leg gets a standard teardrop splitter
  (`SPLITTER_LENGTH_M`) and its carriageway flares round it.
- **Markings**: a yield line of shark's teeth across every entry.
- **Signs** (`StreetItem`s, rendered by the furniture layer): yield signs at every entry (right
  side, and on the splitter), keep-right signs on the splitter noses, chevron boards on the
  central island facing every entry, and roundabout-ahead signs `AHEAD_SIGN_M` up each approach.
- **Heights**: one plane (`settle_route`) fitted to the lidar ground and the route inside the
  roundabout's zone, its slope capped at `MAX_PLANE_GRADE`; the route is moved onto it (and
  blended back to its own profile over `ROUTE_BLEND_M`), its ribbon is hidden inside the zone
  and the roundabout's pavement takes over, so the AI line and the spawn heights follow it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import shapely

from driftmaps.geometry.network import (
    HeightField,
    NetGraph,
    PlaneZone,
    PlannedItem,
    SideRoad,
    _polygons,
    _smoothstep,
    _union,
    mesh_polygon,
)
from driftmaps.geometry.primitives import MeshData, compute_normals, compute_tangents
from driftmaps.geometry.road import RoadModel
from driftmaps.geometry.street_items import (
    SIGN_CHEVRON_LEFT,
    SIGN_CHEVRON_RIGHT,
    SIGN_KEEP_RIGHT,
    SIGN_ROUNDABOUT_AHEAD,
    SIGN_YIELD,
)

__all__ = [
    "Roundabout",
    "RoundaboutParts",
    "build_roundabout",
    "find_roundabouts",
    "island_meshes",
    "pavers_texture",
    "settle_route",
]

# Single-lane roundabout (TAC GDG Ch. 9 / FHWA NCHRP 672, as used by Halton): circulatory
# roadway 4.8-6.1 m, truck apron 3-4.5 m raised 50-75 mm, central island barrier curb ~150 mm,
# entry radius 15-30 m. ASSUMED mid-range values until checked against Halton drawings.
CIRCULATORY_WIDTH_M = {1: 5.5, 2: 9.0}
APRON_WIDTH_M = 3.5
APRON_RAISE_M = 0.07
MOUNTABLE_CURB_M = 0.30  # width of the sloped curb in front of the apron
ISLAND_CURB_M = 0.15
CURB_TOP_M = 0.25
# Curbs keep their real 150 mm reveal but rise over 180 mm (streetscape research §7: AC tests
# tyre contact at one point per wheel, so a vertical step makes the wheel jump harshly).
CURB_FACE_RUN_M = 0.18
MOUND_M = 1.2
ENTRY_RADIUS_M = 15.0
ZONE_MARGIN_M = 10.0  # the route ribbon is replaced this far beyond the circulatory roadway
MAX_PLANE_GRADE = 0.04
ROUTE_BLEND_M = 40.0
SPLITTER_GAP_M = 0.6  # splitter islands stop this short of the circulatory roadway
SPLITTER_CURB_M = 0.15
SPLITTER_LENGTH_M = 20.0
SPLITTER_WIDTH_M = (3.0, 0.8)  # at the ring, at the nose
ENTRY_LANE_M = 4.5
YIELD_TOOTH_M = (0.5, 0.7, 0.25)  # base, height, gap (OTM Book 11 / MUTCD 3B.16 range)
AHEAD_SIGN_M = 120.0  # roundabout-ahead sign before the yield line (rural 60-80 km/h)
SIGN_OFFSET_M = 2.0


@dataclass
class Roundabout:
    way_ids: tuple[int, ...]
    nodes: frozenset[int]
    centre: np.ndarray
    radius: float  # circulatory road centreline
    lanes: int = 1
    zone: PlaneZone | None = None
    name: str = ""

    @property
    def width(self) -> float:
        return CIRCULATORY_WIDTH_M.get(self.lanes, 4.5 * self.lanes)

    @property
    def outer_r(self) -> float:
        return self.radius + self.width / 2

    @property
    def inner_r(self) -> float:
        return self.radius - self.width / 2

    @property
    def apron(self) -> float:
        return float(np.clip(self.inner_r - 5.0, 0.0, APRON_WIDTH_M))

    @property
    def island_r(self) -> float:
        return self.inner_r - self.apron

    @property
    def zone_r(self) -> float:
        return self.outer_r + ZONE_MARGIN_M


@dataclass
class RoundaboutParts:
    pavement: shapely.Geometry = field(default_factory=shapely.Polygon)
    islands: list[tuple[shapely.Geometry, str]] = field(default_factory=list)
    meshes: list[tuple[str, MeshData]] = field(default_factory=list)
    paint: list[tuple[str, shapely.Geometry]] = field(default_factory=list)
    items: list[PlannedItem] = field(default_factory=list)
    walls: list[np.ndarray] = field(default_factory=list)
    arms: int = 0
    entries: int = 0


def _fit_circle(xz: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares (Kasa) circle through the points."""
    x, z = xz[:, 0], xz[:, 1]
    a = np.stack([x, z, np.ones(len(x))], axis=1)
    b = x * x + z * z
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    c = np.array([sol[0] / 2, sol[1] / 2])
    r = math.sqrt(max(sol[2] + c @ c, 1e-9))
    return c, r


def find_roundabouts(graph: NetGraph, driving_side: str = "right") -> list[Roundabout]:
    """Roundabouts from the graph's ring ways (grouped by shared nodes). Rings that are not
    round enough or too big for a roundabout (> 60 m radius) are left as ordinary roads."""
    groups: list[tuple[set[int], list]] = []
    for way in graph.rings:
        ns = {int(n) for n in way.nodes}
        hit = [g for g in groups if g[0] & ns]
        merged = (ns.union(*(g[0] for g in hit)), [way, *(w for g in hit for w in g[1])])
        groups = [g for g in groups if not any(g is h for h in hit)] + [merged]
    out = []
    for ns, ways in groups:
        xz = np.array([graph.coords[n] for n in sorted(ns)])
        if len(xz) < 5:
            continue
        c, r = _fit_circle(xz)
        resid = np.abs(np.linalg.norm(xz - c, axis=1) - r)
        if not (6.0 <= r <= 60.0) or resid.max() > max(3.0, 0.2 * r):
            for way in ways:  # not a roundabout we can build: an ordinary road
                for a, b in zip(way.nodes[:-1], way.nodes[1:], strict=True):
                    graph.adj[int(a)].append((int(b), way.id, True))
                    graph.adj[int(b)].append((int(a), way.id, False))
                graph.kind[way.id] = "road"
            graph.ring_nodes -= ns
            continue
        lanes_raw = str(ways[0].tags.get("lanes", "1")).split(";")[0]
        lanes = int(lanes_raw) if lanes_raw.isdigit() and int(lanes_raw) > 0 else 1
        name = next((w.tags.get("name", "") for w in ways if w.tags.get("name")), "")
        out.append(
            Roundabout(
                tuple(sorted(w.id for w in ways)), frozenset(ns), c, r, min(lanes, 2), name=name
            )
        )
    return sorted(out, key=lambda rb: rb.way_ids)


def settle_route(
    model: RoadModel, rbs: Sequence[Roundabout], dtm: Callable[[np.ndarray], np.ndarray]
) -> tuple[RoadModel, np.ndarray]:
    """Fit each roundabout's plane (lidar ground in its zone + the route there), move the route
    onto it inside the zone (blending back over `ROUTE_BLEND_M`) and return the new model and
    the route segments (N - 1,) the roundabout replaces."""
    n = len(model)
    hidden = np.zeros(n - 1, bool)
    if not rbs:
        return model, hidden
    xz = model.centre[:, [0, 2]]
    y = model.centre[:, 1].copy()
    s = model.s
    for rb in rbs:
        c, zr = rb.centre, rb.zone_r
        g = np.arange(-zr, zr + 1e-9, 3.0)
        gx, gz = np.meshgrid(g, g, indexing="ij")
        pts = np.stack([gx.ravel(), gz.ravel()], axis=1)
        pts = pts[np.linalg.norm(pts, axis=1) <= zr] + c
        h = np.asarray(dtm(pts), np.float64)
        ok = np.isfinite(h)
        inside = np.linalg.norm(xz - c, axis=1) <= zr
        p_all = np.concatenate([pts[ok], xz[inside]])
        h_all = np.concatenate([h[ok], y[inside]])
        w_all = np.concatenate([np.ones(ok.sum()), np.full(inside.sum(), 3.0)])
        if len(p_all) < 3:
            rb.zone = PlaneZone(
                c, zr, float(np.nanmean(y[inside])) if inside.any() else 0.0, np.zeros(2)
            )
        else:
            a = np.stack([np.ones(len(p_all)), *(p_all - c).T], axis=1) * np.sqrt(w_all)[:, None]
            sol, *_ = np.linalg.lstsq(a, h_all * np.sqrt(w_all), rcond=None)
            grad = sol[1:]
            norm = float(np.linalg.norm(grad))
            if norm > MAX_PLANE_GRADE:
                grad = grad * (MAX_PLANE_GRADE / norm)
                sol[0] = float(np.average(h_all - (p_all - c) @ grad, weights=w_all))
            rb.zone = PlaneZone(c, zr, float(sol[0]), grad)
        if not inside.any():
            continue
        corr = np.where(inside, rb.zone.height(xz) - y, 0.0)
        idx = np.flatnonzero(inside)
        pos = np.clip(np.searchsorted(idx, np.arange(n)), 0, len(idx) - 1)
        prev = np.clip(pos - 1, 0, len(idx) - 1)
        pick = np.where(np.abs(s[idx[pos]] - s) <= np.abs(s[idx[prev]] - s), idx[pos], idx[prev])
        fade = 1.0 - _smoothstep(np.abs(s - s[pick]) / ROUTE_BLEND_M)
        y = y + np.where(inside, corr, corr[pick] * fade)
        hidden |= inside[:-1] & inside[1:]
    centre = np.stack([xz[:, 0], y, xz[:, 1]], axis=1)
    new = RoadModel(
        s=model.s,
        centre=centre,
        half_width=model.half_width,
        shoulder=model.shoulder,
        centre_line=model.centre_line,
        bridge=model.bridge,
        double_yellow=model.double_yellow,
        closed=model.closed,
    )
    return new, hidden


# --------------------------------------------------------------------------------------------
# Arms (roads meeting the ring)
# --------------------------------------------------------------------------------------------


@dataclass
class _Arm:
    """A road meeting the ring: `xz` (K, 2) from the ring outward, `flow` "in", "out" or
    "both", `poly` its carriageway, `road` side road index (-1: the route)."""

    xz: np.ndarray
    hw: float
    flow: str
    poly: shapely.Geometry
    road: int = -1
    route_s: tuple[float, float] | None = None  # (ring end s, far end s) on the route
    far_node: int = -1


def _polyline_at(xz: np.ndarray, d: float) -> tuple[np.ndarray, np.ndarray]:
    """Point and unit direction at arc length `d` along a polyline."""
    seg = np.diff(xz, axis=0)
    ll = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(ll)])
    d = float(np.clip(d, 0.0, s[-1]))
    k = int(np.clip(np.searchsorted(s, d, side="right") - 1, 0, len(seg) - 1))
    t = (d - s[k]) / max(ll[k], 1e-9)
    return xz[k] + t * seg[k], seg[k] / max(ll[k], 1e-9)


def _radial_cross(xz: np.ndarray, c: np.ndarray, r: float) -> float | None:
    """Arc length along the polyline where it is first `r` from `c` (it starts inside)."""
    d = np.linalg.norm(xz - c, axis=1)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))])
    out = np.flatnonzero(d >= r)
    if not len(out):
        return None
    k = int(out[0])
    if k == 0:
        return 0.0
    t = (r - d[k - 1]) / max(d[k] - d[k - 1], 1e-9)
    return float(s[k - 1] + t * (s[k] - s[k - 1]))


def _arms(
    rb: Roundabout,
    model: RoadModel,
    hidden: np.ndarray,
    route_quads: tuple[np.ndarray, np.ndarray, np.ndarray],
    sides: Sequence[SideRoad],
    carriage: Sequence[shapely.Geometry],
    route_oneway: np.ndarray,
) -> list[_Arm]:
    arms: list[_Arm] = []
    c = rb.centre
    xz = model.centre[:, [0, 2]]
    r = np.linalg.norm(xz - c, axis=1)
    on = np.flatnonzero(r <= rb.outer_r + 0.5)
    if len(on):
        asphalt_q = route_quads[0]
        for i_ring, step in ((int(on[0]), -1), (int(on[-1]), 1)):
            i_far = int(np.clip(i_ring + step * 260, 0, len(model) - 1))
            lo, hi = sorted((i_ring, i_far))
            if hi - lo < 3:
                continue
            idx = np.arange(i_ring, i_far + step, step)
            pts = xz[idx]
            ow = int(route_oneway[min(max(i_ring + 5 * step, 0), len(model) - 1)])
            # Traffic on the route arm: along the route (+s) unless one-way against it.
            if ow == 0:
                flow = "both"
            else:
                legal_dir = 1 if ow > 0 else -1  # +1: traffic drives +s
                flow = "in" if legal_dir * step < 0 else "out"
            poly = _union(list(asphalt_q[lo:hi]))
            arms.append(
                _Arm(
                    pts,
                    float(model.half_width[i_ring]),
                    flow,
                    poly,
                    -1,
                    (float(model.s[i_ring]), float(model.s[i_far])),
                )
            )
    for sr in sides:
        if sr.start[0] != "ring" or sr.start[1] not in rb.nodes:
            continue
        pts = sr.xz
        flow = "both" if sr.oneway == 0 else ("in" if sr.oneway < 0 else "out")
        arms.append(
            _Arm(
                pts,
                float(sr.model.half_width[0]),
                flow,
                carriage[sr.index],
                sr.index,
                far_node=sr.end[1],
            )
        )
    return arms


def _pairs(rb: Roundabout, arms: Sequence[_Arm]) -> list[tuple[int, int]]:
    """(entry arm, exit arm) pairs of one-way carriageways of the same leg (they converge)."""
    pairs = []
    used: set[int] = set()
    cands = []
    for a, arm_a in enumerate(arms):
        if arm_a.flow != "in":
            continue
        for b, arm_b in enumerate(arms):
            if arm_b.flow != "out":
                continue
            va, vb = arm_a.xz[0] - rb.centre, arm_b.xz[0] - rb.centre
            ang = abs(math.atan2(va[0] * vb[1] - va[1] * vb[0], va @ vb))
            if ang > math.radians(120):  # OSM splits can meet the ring far apart
                continue
            la, lb = shapely.LineString(arm_a.xz), shapely.LineString(arm_b.xz)
            far = shapely.Point(arm_a.xz[-1]).distance(lb)
            near_far = min(far, shapely.Point(arm_b.xz[-1]).distance(la))
            if near_far > arm_a.hw + arm_b.hw + 3.0:
                continue
            cands.append((ang, a, b))
    for _, a, b in sorted(cands):
        if a in used or b in used:
            continue
        used |= {a, b}
        pairs.append((a, b))
    return pairs


def _cut(xz: np.ndarray, at: np.ndarray) -> np.ndarray:
    """The polyline from its start to its closest point to `at`."""
    line = shapely.LineString(xz)
    d = line.project(shapely.Point(at))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))])
    k = int(np.searchsorted(s, d))
    end = np.asarray(line.interpolate(d).coords[0])
    return np.vstack([xz[:k], end[None]])


def _splitter(rb: Roundabout, a: _Arm, b: _Arm) -> tuple[shapely.Geometry, shapely.Geometry]:
    """(raised island, paved filler) between a pair of one-way carriageways: the ground
    between their centrelines from the ring to where they merge (the far end of the shorter
    one), less the carriageways and a gap before the circulatory road."""
    la = np.linalg.norm(np.diff(a.xz, axis=0), axis=1).sum()
    lb = np.linalg.norm(np.diff(b.xz, axis=0), axis=1).sum()
    if la <= lb:
        pa, pb = a.xz, _cut(b.xz, a.xz[-1])
    else:
        pa, pb = _cut(a.xz, b.xz[-1]), b.xz
    ring = np.concatenate([pa, pb[::-1]])
    if len(ring) < 3:
        return shapely.Polygon(), shapely.Polygon()
    enclosed = shapely.make_valid(shapely.Polygon(ring))
    enclosed = _union(_polygons(enclosed))
    core = shapely.Point(rb.centre).buffer(rb.outer_r + SPLITTER_GAP_M, quad_segs=32)
    island = enclosed.difference(_union([a.poly, b.poly, core]))
    island = island.buffer(-0.4, quad_segs=4).buffer(0.4, quad_segs=4).intersection(enclosed)
    island = island.difference(_union([a.poly, b.poly, core]))
    island = _union([p for p in _polygons(island, 2.0)])
    filler = enclosed.difference(island)
    return island, filler


def _teardrop(rb: Roundabout, arm: _Arm) -> tuple[shapely.Geometry, shapely.Geometry]:
    """(standard splitter island, flared carriageway round it) for a two-way arm."""
    d0 = _radial_cross(arm.xz, rb.centre, rb.outer_r + SPLITTER_GAP_M + 0.4)
    if d0 is None:
        return shapely.Polygon(), shapely.Polygon()
    t = np.linspace(0.0, 1.0, 11)
    left, right = [], []
    for k in t:
        p, f = _polyline_at(arm.xz, d0 + k * SPLITTER_LENGTH_M)
        w = (1 - k) * SPLITTER_WIDTH_M[0] + k * SPLITTER_WIDTH_M[1]
        n = np.array([-f[1], f[0]])
        left.append(p + n * w / 2)
        right.append(p - n * w / 2)
    island = shapely.Polygon(np.concatenate([left, right[::-1]])).buffer(0.3).buffer(-0.3)
    flare = island.buffer(ENTRY_LANE_M, quad_segs=8).difference(island)
    return island, flare


# --------------------------------------------------------------------------------------------
# Meshes
# --------------------------------------------------------------------------------------------


def _ring_band(
    c: np.ndarray,
    radii: Sequence[float],
    raises: Sequence[float],
    height: Callable[[np.ndarray], np.ndarray],
    n_ang: int,
    *,
    vertical: bool = False,
    uv_tile: float = 2.0,
) -> MeshData:
    """A polar lattice between concentric rings (radius, raise above the height field)."""
    ang = np.linspace(0.0, 2 * np.pi, n_ang + 1)
    rows = []
    for r, dy in zip(radii, raises, strict=True):
        p = c + r * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        y = np.asarray(height(p)) + dy
        rows.append(np.stack([p[:, 0], y, p[:, 1]], axis=1))
    pos = np.concatenate(rows)
    k = n_ang + 1
    tri = []
    for j in range(len(radii) - 1):
        a = j * k + np.arange(n_ang)
        tri += [np.stack([a, a + 1, a + k + 1], 1), np.stack([a, a + k + 1, a + k], 1)]
    tri = np.concatenate(tri)
    n = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
    ok = np.linalg.norm(n, axis=1) > 1e-9  # a ring of radius 0 (the mound's top) collapses
    tri, n = tri[ok], n[ok]
    if vertical:  # face outward
        out = pos[tri].mean(axis=1)[:, [0, 2]] - c
        flip = np.einsum("ij,ij->i", n[:, [0, 2]], out) < 0
    else:
        flip = n[:, 1] < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    ring_r = np.repeat(np.asarray(radii, float), k)
    u = np.tile(ang, len(radii)) * (np.asarray(radii).mean() / uv_tile)
    v = (ring_r if not vertical else np.repeat(np.asarray(raises, float), k)) / uv_tile
    uv = np.stack([u, v], axis=1)
    data = MeshData(
        pos, np.tile([0.0, 1.0, 0.0], (len(pos), 1)), uv, np.zeros_like(pos), tri.reshape(-1)
    )
    return compute_tangents(compute_normals(data))


def _central_island_meshes(rb: Roundabout, height: HeightField) -> list[tuple[str, MeshData]]:
    c = rb.centre
    n_ang = max(48, math.ceil(2 * math.pi * rb.inner_r / 1.5))
    out: list[tuple[str, MeshData]] = []
    r_in, r_isl = rb.inner_r, rb.island_r
    top = APRON_RAISE_M + 0.02
    if rb.apron > MOUNTABLE_CURB_M + 0.2:
        out.append(
            (
                "curb",
                _ring_band(c, [r_in, r_in - MOUNTABLE_CURB_M], [0.0, APRON_RAISE_M], height, n_ang),
            )
        )
        out.append(
            (
                "pavers",
                _ring_band(
                    c,
                    [r_in - MOUNTABLE_CURB_M, r_isl],
                    [APRON_RAISE_M, top],
                    height,
                    n_ang,
                    uv_tile=1.0,
                ),
            )
        )
    else:
        top = 0.0
    curb_top = top + ISLAND_CURB_M
    r_face = r_isl - CURB_FACE_RUN_M  # sloped face: AC's one-point tyre contact
    out.append(("curb", _ring_band(c, [r_isl, r_face], [top, curb_top], height, n_ang)))
    out.append(
        ("curb", _ring_band(c, [r_face, r_face - CURB_TOP_M], [curb_top, curb_top], height, n_ang))
    )
    r0 = r_face - CURB_TOP_M
    rr = np.linspace(r0, 0.0, 9)
    mound = curb_top + MOUND_M * _smoothstep((r0 - rr) / max(0.6 * r0, 1e-3))
    out.append(("island_grass", _ring_band(c, list(rr), list(mound), height, n_ang, uv_tile=4.0)))
    return out


def island_meshes(
    island: shapely.Geometry, height: HeightField, cell: float
) -> list[tuple[str, MeshData]]:
    out: list[tuple[str, MeshData]] = []
    for poly in _polygons(island, 1.0):
        # Curb face sloped over CURB_FACE_RUN_M (a vertical step makes AC's single tyre
        # contact point jump), then the curb top, then the island's top surface.
        top_edge = poly.buffer(-CURB_FACE_RUN_M, join_style="mitre")
        ring = poly.exterior

        def face_height(xz: np.ndarray, ring=ring) -> np.ndarray:
            d = shapely.distance(ring, shapely.points(np.asarray(xz).reshape(-1, 2)))
            return height(xz) + SPLITTER_CURB_M * np.clip(d / CURB_FACE_RUN_M, 0.0, 1.0)

        m = mesh_polygon(poly.difference(top_edge), face_height, cell_m=cell, uv_tile_m=2.0)
        if m is not None:
            out.append(("curb", m))
        inner = top_edge.buffer(-CURB_TOP_M)
        band = top_edge.difference(inner)
        m = mesh_polygon(band, height, cell_m=cell, uv_tile_m=2.0, lift=SPLITTER_CURB_M)
        if m is not None:
            out.append(("curb", m))
        key = "island_grass" if poly.area > 60.0 and inner.area > 10.0 else "island_concrete"
        m = mesh_polygon(
            inner,
            height,
            cell_m=cell,
            uv_tile_m=4.0 if key == "island_grass" else 2.0,
            lift=SPLITTER_CURB_M,
        )
        if m is not None:
            out.append((key, m))
    return out


def _teeth(p: np.ndarray, t: np.ndarray, lo: float, hi: float) -> list[shapely.Geometry]:
    """Shark's teeth across a lane at `p` (offsets [lo, hi] to the drivers' right), apex
    pointing against the travel direction `t`, towards the approaching drivers."""
    base, h, gap = YIELD_TOOTH_M
    n = np.array([-t[1], t[0]])  # the drivers' right in AC XZ (X east, Z south)
    out = []
    x = lo + gap / 2
    while x + base <= hi - gap / 2 + 1e-9:
        a = p + n * x
        b = p + n * (x + base)
        apex = p + n * (x + base / 2) - t * h
        out.append(shapely.Polygon([a, b, apex]))
        x += base + gap
    return out


# --------------------------------------------------------------------------------------------
# Building one roundabout
# --------------------------------------------------------------------------------------------


def _upstream(arm: _Arm, sides: Sequence[SideRoad], need: float) -> tuple[np.ndarray, float] | None:
    """The polyline from the ring outward along the arm and on along the side roads it
    continues as (straightest), and its length."""
    pts = [arm.xz]
    length = float(np.linalg.norm(np.diff(arm.xz, axis=0), axis=1).sum())
    node = arm.far_node
    used = {arm.road}
    while length < need and node >= 0:
        here = pts[-1][-1]
        d_in = pts[-1][-1] - pts[-1][-2]
        d_in = d_in / max(np.linalg.norm(d_in), 1e-9)
        best = None
        for sr in sides:
            if sr.index in used:
                continue
            if sr.start == ("road", node) or sr.start == ("route", node):
                xz = sr.xz
            elif sr.end == ("road", node) or sr.end == ("route", node):
                xz = sr.xz[::-1]
            else:
                continue
            if np.linalg.norm(xz[0] - here) > 30.0 or len(xz) < 3:
                continue
            d = xz[min(5, len(xz) - 1)] - xz[0]
            cos = float(d_in @ d) / max(np.linalg.norm(d), 1e-9)
            if cos > 0.5 and (best is None or cos > best[0]):
                best = (cos, sr, xz)
        if best is None:
            break
        _, sr, xz = best
        used.add(sr.index)
        pts.append(xz)
        length += float(np.linalg.norm(np.diff(xz, axis=0), axis=1).sum())
        node = sr.end[1] if xz is sr.xz else sr.start[1]
    line = np.concatenate(pts)
    return line, length


def build_roundabout(
    rb: Roundabout,
    *,
    model: RoadModel,
    hidden: np.ndarray,
    route_quads: tuple[np.ndarray, np.ndarray, np.ndarray],
    sides: Sequence[SideRoad],
    carriage: Sequence[shapely.Geometry],
    graph: NetGraph,
    height: HeightField,
    driving_side: str = "right",
    route_oneway: np.ndarray | None = None,
    cell_m: float = 3.0,
) -> RoundaboutParts:
    """Pavement, islands, meshes, markings, signs and island walls of one roundabout."""
    parts = RoundaboutParts()
    c = rb.centre
    right = driving_side == "right"
    route_oneway = route_oneway if route_oneway is not None else np.zeros(len(model), np.int64)
    arms = _arms(rb, model, hidden, route_quads, sides, carriage, route_oneway)
    parts.arms = len(arms)
    zone = shapely.Point(c).buffer(rb.zone_r, quad_segs=32)
    asphalt_q, sh_l, sh_r = route_quads
    idx = np.flatnonzero(hidden)
    route_hidden = (
        _union(list(asphalt_q[idx]) + list(sh_l[idx]) + list(sh_r[idx]))
        if len(idx)
        else shapely.Polygon()
    )

    islands: list[shapely.Geometry] = []
    fillers: list[shapely.Geometry] = []
    flares: list[shapely.Geometry] = []
    paired: set[int] = set()
    for a, b in _pairs(rb, arms):
        isl, fill = _splitter(rb, arms[a], arms[b])
        if not isl.is_empty:
            islands.append(isl)
            fillers.append(fill)
            paired |= {a, b}
    for k, arm in enumerate(arms):
        if k in paired or arm.flow != "both":
            continue
        isl, flare = _teardrop(rb, arm)
        if not isl.is_empty:
            islands.append(isl)
            flares.append(flare)
            paired.add(k)

    legs = [arm.poly.intersection(zone) for arm in arms]
    base = _union([shapely.Point(c).buffer(rb.outer_r, quad_segs=48), *legs, *flares, route_hidden])
    closed = base.buffer(ENTRY_RADIUS_M, quad_segs=16).buffer(-ENTRY_RADIUS_M, quad_segs=16)
    pavement = _union([closed.intersection(zone), route_hidden, *flares, *fillers])
    centre_disc = shapely.Point(c).buffer(rb.inner_r, quad_segs=48)
    parts.pavement = pavement.difference(centre_disc)
    isl_r = rb.island_r
    parts.islands = [(centre_disc, "central")] + [(i, "splitter") for i in islands]

    parts.meshes += _central_island_meshes(rb, height)
    for isl in islands:
        parts.meshes += island_meshes(isl, height, cell_m)

    # Hidden walls: inside the central island's curb and inside every splitter.
    ang = np.linspace(0.0, 2 * np.pi, max(24, math.ceil(2 * math.pi * isl_r / 2.0)) + 1)
    ring = c + (isl_r - 0.5) * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    ring[-1] = ring[0]
    parts.walls.append(ring)
    for isl in islands:
        for poly in _polygons(isl.buffer(-0.5), 1.0):
            w = np.asarray(shapely.segmentize(poly.exterior, 2.0).coords)
            if len(w) >= 4:
                parts.walls.append(w)

    # Entries: yield line, yield signs, chevron board, roundabout-ahead sign.
    chevron = SIGN_CHEVRON_RIGHT if right else SIGN_CHEVRON_LEFT
    isl_union = _union(islands)
    for k, arm in enumerate(arms):
        if arm.flow not in ("in", "both"):
            continue
        d_y = _radial_cross(arm.xz, c, rb.outer_r + 0.4)
        if d_y is None:
            continue
        p, f = _polyline_at(arm.xz, d_y)
        t = -f  # travel direction: towards the ring
        nr = np.array([-t[1], t[0]])  # the drivers' right in AC XZ (X east, Z south)
        if arm.flow == "in":
            lo, hi = -arm.hw, arm.hw
        elif k in paired:  # two-way arm with a teardrop: the entry lane is right of it
            lo, hi = SPLITTER_WIDTH_M[0] / 2, SPLITTER_WIDTH_M[0] / 2 + ENTRY_LANE_M
            if not right:
                lo, hi = -hi, -lo
        else:
            lo, hi = (0.0, arm.hw) if right else (-arm.hw, 0.0)
        teeth = [g.difference(isl_union) for g in _teeth(p, t, lo, hi)]
        parts.paint += [("paint_white", g) for g in teeth if not g.is_empty]
        parts.entries += 1
        sp = p - t * 2.0
        side = hi if right else lo
        pos = sp + nr * (side + SIGN_OFFSET_M) if right else sp + nr * (side - SIGN_OFFSET_M)
        face = (-t[0], -t[1])
        parts.items.append(
            PlannedItem(
                SIGN_YIELD, (float(pos[0]), float(pos[1])), face, ref=f"roundabout:{rb.way_ids[0]}"
            )
        )
        if k in paired and arm.flow == "in":  # a second yield sign on the splitter
            other = lo if right else hi
            pos2 = sp + nr * (other - 1.0) if right else sp + nr * (other + 1.0)
            if isl_union.contains(shapely.Point(pos2)):
                parts.items.append(
                    PlannedItem(
                        SIGN_YIELD,
                        (float(pos2[0]), float(pos2[1])),
                        face,
                        ref=f"roundabout:{rb.way_ids[0]}",
                    )
                )
        u = p - c
        u = u / max(np.linalg.norm(u), 1e-9)
        cp = c + u * (isl_r - 0.8)
        parts.items.append(
            PlannedItem(
                chevron,
                (float(cp[0]), float(cp[1])),
                (float(u[0]), float(u[1])),
                ref=f"roundabout:{rb.way_ids[0]}",
            )
        )
        # Roundabout ahead, on the approach.
        if arm.road < 0 and arm.route_s is not None:
            s_ring, s_far = arm.route_s
            s_sign = (
                s_ring - (d_y + AHEAD_SIGN_M) if s_far < s_ring else s_ring + (d_y + AHEAD_SIGN_M)
            )
            if model.s[0] <= s_sign <= model.s[-1]:
                q, fw, left = model.at(s_sign)
                i = model.index_at(s_sign)
                go = fw if s_far < s_ring else -fw  # travel direction
                off = float(model.half_width[i] + model.shoulder[i] + SIGN_OFFSET_M)
                lat = -left if (s_far < s_ring) == right else left
                pos = q + lat * off
                parts.items.append(
                    PlannedItem(
                        SIGN_ROUNDABOUT_AHEAD,
                        (float(pos[0]), float(pos[2])),
                        (float(-go[0]), float(-go[2])),
                        ref=f"roundabout:{rb.way_ids[0]}",
                    )
                )
        else:
            up = _upstream(arm, sides, d_y + AHEAD_SIGN_M)
            if up is not None and up[1] >= d_y + 40.0:
                line, length = up
                q, fo = _polyline_at(line, min(d_y + AHEAD_SIGN_M, length - 1.0))
                go = -fo
                nrr = np.array([-go[1], go[0]])
                hw = arm.hw if arm.flow == "in" else arm.hw
                pos = q + nrr * (hw + 1.0 + SIGN_OFFSET_M) * (1.0 if right else -1.0)
                parts.items.append(
                    PlannedItem(
                        SIGN_ROUNDABOUT_AHEAD,
                        (float(pos[0]), float(pos[1])),
                        (float(fo[0]), float(fo[1])),
                        ref=f"roundabout:{rb.way_ids[0]}",
                    )
                )
    # Keep-right signs on the splitter noses, facing the entering traffic.
    for isl in islands:
        for poly in _polygons(isl, 2.0):
            ext = np.asarray(poly.exterior.coords)
            k = int(np.argmax(np.linalg.norm(ext - c, axis=1)))
            nose = ext[k]
            u = nose - c
            u = u / max(np.linalg.norm(u), 1e-9)
            pos = nose - u * 1.0
            if not poly.contains(shapely.Point(pos)):
                pos = np.asarray(poly.representative_point().coords[0])
            parts.items.append(
                PlannedItem(
                    SIGN_KEEP_RIGHT,
                    (float(pos[0]), float(pos[1])),
                    (float(u[0]), float(u[1])),
                    ref=f"roundabout:{rb.way_ids[0]}",
                )
            )
    return parts


# --------------------------------------------------------------------------------------------
# Texture
# --------------------------------------------------------------------------------------------


def pavers_texture(size: int = 256, seed: int = 41):
    """Tileable red-brown concrete pavers in a running bond (truck aprons): 8 x 4 bricks."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rows, cols = 8, 4
    bh, bw = size / rows, size / cols
    row = np.floor(yy / bh)
    shift = (row % 2) * (bw / 2)
    col = np.floor(((xx + shift) % size) / bw)
    tone = rng.normal(0.0, 1.0, (rows, cols)).astype(np.float32)
    base = 120.0 + 12.0 * tone[row.astype(int) % rows, col.astype(int) % cols]
    grain = rng.normal(0.0, 6.0, (size, size)).astype(np.float32)
    joint = (np.mod(yy, bh) < 2) | (np.mod(xx + shift, bw) < 2)
    lum = base + grain
    lum[joint] = 70.0
    tint = np.array([1.12, 0.78, 0.62], np.float32)
    rgb = np.clip(lum[..., None] * tint, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)
