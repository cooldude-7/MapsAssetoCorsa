"""Pins -> route over real roads, and the `route.json` contract the build reads.

Algorithm (docs/research/geo.md §1.3, PLAN §5.1):
- **Snap**: each pin goes to the nearest OSM segment on the local projection. Among segments
  within `snap_class_tolerance_m` of the nearest, the higher road class wins, then the nearer,
  then the lower (way id, index). A snap within `node_snap_m` of a segment end becomes a *node
  pin* (passed by any edge through that node); otherwise it is an *edge pin* that splits the
  segment virtually at parameter t.
- **Route**: Dijkstra over states (directed edge, pins passed so far). Entering edge e2 from e1
  costs len(e2) + turn cost. A U-turn (e2 = reverse of e1, or at a junction a turn sharper than
  `u_turn_angle_deg`) is forbidden, also at the pins themselves, so the route never doubles back.
  At junctions (nodes with 3+ neighbours) a turn costs `sharp_turn_penalty_m * (1 - cos a)^2`
  (5 m at 90 deg, 20 m at 180) plus `name_change_penalty_m` when the road's name/ref changes.
  Pins must be passed in order, in either direction; the result is the global optimum.
- **Loops** (`loop=True`): the route goes through the pins and back to the first one, which
  is routed as an extra last pin. It must arrive there the way it left (along the same
  directed edge for a pin on a segment; with an allowed turn for a pin on a node), so the
  closed path has no U-turn at the start either: one search per way of leaving pin A, the
  cheapest wins. A loop that drives any stretch of road twice (back along the same road, or
  round the same stretch again) is refused (`LOOP_MESSAGE`); crossing itself at a junction
  (a figure 8) is fine. The route is `closed` and its first and last pins are the same.
- **Determinism**: costs are integer millimetres, the heap orders ties by the state number
  (edge index, then pins passed), and edges are numbered in sorted way order. Same OSM snapshot
  + same pins + same options => identical `route.json` bytes. (ASSUMED across machines: the
  projection uses libm/SIMD transcendental functions whose last-bit results can differ between
  CPUs; the build never re-routes, it reads `route.json`, so this cannot split preview and build.)

`Route.to_json()` is the contract: snapshot time and cache key, pins (as given and as snapped),
the ordered directed edges `(way, u, v, dir)`, the coordinates of every node used, the tags of
every way used, side-road stubs at route junctions, the length, and `closed` for a loop (only
written when true). `Route.reverse()` drives the same edges the other way.
`Route.polyline(step)` densifies it with per-point attributes.

`route_from_path` builds a Route from an explicit node path (the app's turn-by-turn builder):
no search, just checks that each pair of consecutive nodes is a road segment.
"""

from __future__ import annotations

import heapq
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from driftmaps.geo.attributes import (
    driving_side,
    is_bridge,
    is_oneway,
    is_tunnel,
    lane_count,
    layer_of,
    region_for,
    road_width,
    route_tags,
    surface_class,
)
from driftmaps.geo.graph import EdgeGraph, build_graph
from driftmaps.geo.overpass import DEFAULT_BUFFER_M, OSM_ATTRIBUTION, OsmData, fetch_corridor
from driftmaps.geo.project import LocalProjection

__all__ = [
    "LOOP_MESSAGE",
    "ROUTE_VERSION",
    "Route",
    "RouteEdge",
    "RouteError",
    "RoutePolyline",
    "RouterOptions",
    "SnappedPin",
    "Continuation",
    "route_alternatives",
    "route_continuations",
    "route_from_path",
    "route_from_pins",
    "route_on_graph",
    "snap_pin",
]

ROUTE_VERSION = 1
DEFAULT_MAX_LENGTH_M = 25_000.0  # PLAN §5.1: v1 warns above this
# Route options (route_alternatives): cost multiplier for roads already found, the most of a
# new option that may run on an earlier one, and how much longer than the best it may be.
ALT_PENALTY = 2.0
ALT_MAX_OVERLAP = 0.6
ALT_MAX_STRETCH = 1.7
# Why a loop was refused (the app shows it as is).
LOOP_MESSAGE = "Can't make a loop here - add a point on a different road back to the start."

# Highway class rank for snapping preference (higher wins).
_CLASS_RANK = {
    "motorway": 20,
    "trunk": 18,
    "primary": 16,
    "secondary": 14,
    "tertiary": 12,
    "unclassified": 10,
    "residential": 8,
    "living_street": 6,
    "service": 4,
    "track": 2,
}


class RouteError(RuntimeError):
    """No route: a pin is too far from any road, or the pins can't be joined without U-turns."""


@dataclass(frozen=True)
class RouterOptions:
    respect_oneway: bool = False
    sharp_turn_penalty_m: float = 5.0
    name_change_penalty_m: float = 10.0
    u_turn_angle_deg: float = 170.0
    max_snap_m: float = 150.0
    node_snap_m: float = 1.0
    snap_class_tolerance_m: float = 3.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SnappedPin:
    """A pin as given and where it landed: on segment (way, u -> v in way order) at t in [0, 1].

    `node` is set when the pin snapped onto a node (then t is 0 or 1).
    """

    lat: float
    lon: float
    snap_lat: float
    snap_lon: float
    dist_m: float
    way: int
    u: int
    v: int
    t: float
    node: int | None = None


@dataclass(frozen=True)
class RouteEdge:
    """One directed OSM node pair: `dir` is +1 when u -> v follows the way's node order."""

    way: int
    u: int
    v: int
    dir: int


@dataclass(frozen=True)
class RoutePolyline:
    """The route densified on the local projection, with one attribute value per point.

    `edge_index` points into `Route.edges`; `oneway` is relative to the driving direction
    (+1 legal, -1 driving against a one-way, 0 two-way).
    """

    lonlat: np.ndarray  # (N, 2) degrees
    s: np.ndarray  # (N,) arc length, metres
    edge_index: np.ndarray
    way_id: np.ndarray
    width_m: np.ndarray
    lanes: np.ndarray
    surface: np.ndarray
    bridge: np.ndarray
    tunnel: np.ndarray
    layer: np.ndarray
    highway: np.ndarray
    oneway: np.ndarray
    name: np.ndarray

    def __len__(self) -> int:
        return len(self.lonlat)


def _ll(v: float) -> float:
    return round(float(v), 8)


@dataclass(frozen=True)
class Route:
    """A routed road. Everything the build needs, with no reference back to the OSM cache."""

    origin: tuple[float, float]  # (lat, lon) of the projection origin (the first pin)
    pins: tuple[SnappedPin, ...]
    edges: tuple[RouteEdge, ...]
    start_t: float  # where the route starts along edges[0] (0..1)
    end_t: float  # where it ends along edges[-1]
    length_m: float
    nodes: Mapping[int, tuple[float, float]]  # OSM node id -> (lon, lat)
    ways: Mapping[int, Mapping[str, str]]  # OSM way id -> kept tags
    stubs: tuple[tuple[int, int, int], ...] = ()  # (route node, side way, neighbour node)
    osm_base: str | None = None
    cache_key: str | None = None
    driving_side: str = "right"
    options: Mapping[str, object] = field(default_factory=dict)
    version: int = ROUTE_VERSION
    closed: bool = False  # a loop: the last pin is the first, the end meets the start
    # Other ways through the same pins (route_from_pins(alternatives=k)); never serialised.
    alternatives: tuple[Route, ...] = field(default=(), compare=False, repr=False)

    # -- serialisation -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "version": self.version,
            "attribution": OSM_ATTRIBUTION,
            "osm_base": self.osm_base,
            "overpass_cache_key": self.cache_key,
            "origin": {"lat": self.origin[0], "lon": self.origin[1]},
            "driving_side": self.driving_side,
            "options": dict(self.options),
            "pins": [asdict(p) for p in self.pins],
            "edges": [[e.way, e.u, e.v, e.dir] for e in self.edges],
            "start_t": self.start_t,
            "end_t": self.end_t,
            "length_m": self.length_m,
            "nodes": {str(k): list(self.nodes[k]) for k in sorted(self.nodes)},
            "ways": {str(k): dict(self.ways[k]) for k in sorted(self.ways)},
            "stubs": [list(s) for s in self.stubs],
        }
        if self.closed:  # only for loops, so A->B route.json bytes stay as they were
            d["closed"] = True
        return d

    def to_json(self) -> str:
        """Canonical JSON text: sorted keys, one edge/node/way per line, trailing newline."""
        return _dump(self.to_dict()) + "\n"

    @classmethod
    def from_dict(cls, d: Mapping) -> Route:
        if d.get("version") != ROUTE_VERSION:
            raise ValueError(f"unsupported route.json version {d.get('version')!r}")
        return cls(
            origin=(float(d["origin"]["lat"]), float(d["origin"]["lon"])),
            pins=tuple(SnappedPin(**p) for p in d["pins"]),
            edges=tuple(RouteEdge(*map(int, e)) for e in d["edges"]),
            start_t=float(d["start_t"]),
            end_t=float(d["end_t"]),
            length_m=float(d["length_m"]),
            nodes={int(k): (float(v[0]), float(v[1])) for k, v in d["nodes"].items()},
            ways={int(k): dict(v) for k, v in d["ways"].items()},
            stubs=tuple(tuple(int(x) for x in s) for s in d.get("stubs", [])),
            osm_base=d.get("osm_base"),
            cache_key=d.get("overpass_cache_key"),
            driving_side=d.get("driving_side", "right"),
            options=dict(d.get("options", {})),
            version=int(d["version"]),
            closed=bool(d.get("closed", False)),
        )

    @classmethod
    def from_json(cls, text: str | bytes) -> Route:
        return cls.from_dict(json.loads(text))

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8", newline="\n")
        return path

    @classmethod
    def load(cls, path: Path | str) -> Route:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # -- derived views -------------------------------------------------------------------------

    def reverse(self) -> Route:
        """The same road driven the other way (edge list reversed, pins reversed)."""
        return Route(
            origin=self.origin,
            pins=tuple(reversed(self.pins)),
            edges=tuple(RouteEdge(e.way, e.v, e.u, -e.dir) for e in reversed(self.edges)),
            start_t=_round_t(1.0 - self.end_t),
            end_t=_round_t(1.0 - self.start_t),
            length_m=self.length_m,
            nodes=self.nodes,
            ways=self.ways,
            stubs=self.stubs,
            osm_base=self.osm_base,
            cache_key=self.cache_key,
            driving_side=self.driving_side,
            options=self.options,
            version=self.version,
            closed=self.closed,
        )

    def projection(self) -> LocalProjection:
        return LocalProjection(*self.origin)

    def edge_tags(self, i: int) -> Mapping[str, str]:
        return self.ways[self.edges[i].way]

    def node_path(self) -> list[int]:
        """OSM node ids in driving order (tail of the first edge, then every head)."""
        return [self.edges[0].u] + [e.v for e in self.edges]

    def polyline(self, step_m: float = 1.0) -> RoutePolyline:
        """(N, 2) lon/lat from the first snapped pin to the last, no gap longer than `step_m`.

        Points on a node take the attributes of the edge that leaves it.
        """
        if step_m <= 0.0:
            raise ValueError("step_m must be positive")
        proj = self.projection()
        ids = self.node_path()
        ll = np.array([self.nodes[n] for n in ids], dtype=np.float64)
        xy = proj.to_xy(ll)
        m = len(self.edges)
        pts: list[np.ndarray] = []
        idx: list[np.ndarray] = []
        for i in range(m):
            a = self.start_t if i == 0 else 0.0
            b = self.end_t if i == m - 1 else 1.0
            p0 = xy[i] + (xy[i + 1] - xy[i]) * a
            p1 = xy[i] + (xy[i + 1] - xy[i]) * b
            length = float(np.sqrt(np.sum((p1 - p0) ** 2)))
            if length < 1e-9 and not (i == m - 1 and not pts):
                continue
            n = max(1, math.ceil(length / step_m - 1e-9))
            f = np.arange(n)[:, None] / n
            pts.append(p0 + (p1 - p0) * f)
            idx.append(np.full(n, i, dtype=np.int64))
            last = (p1, i)
        pts.append(last[0][None, :])
        idx.append(np.array([last[1]], dtype=np.int64))
        pxy = np.concatenate(pts)
        eidx = np.concatenate(idx)
        seglen = np.sqrt(np.sum(np.diff(pxy, axis=0) ** 2, axis=1))
        s = np.concatenate([[0.0], np.cumsum(seglen)])

        region = region_for(*self.origin)
        per_edge = [_edge_attributes(self.ways[e.way], e.dir, region) for e in self.edges]
        cols = list(zip(*per_edge, strict=True))

        def col(k: int, dtype) -> np.ndarray:
            return np.asarray(cols[k], dtype=dtype)[eidx]

        return RoutePolyline(
            lonlat=proj.from_xy(pxy),
            s=s,
            edge_index=eidx,
            way_id=np.array([e.way for e in self.edges], dtype=np.int64)[eidx],
            width_m=col(0, np.float64),
            lanes=col(1, np.int64),
            surface=col(2, str),
            bridge=col(3, bool),
            tunnel=col(4, bool),
            layer=col(5, np.int64),
            highway=col(6, str),
            oneway=col(7, np.int8),
            name=col(8, str),
        )

    def repeated_nodes(self) -> list[int]:
        """Node ids the route passes more than once (it overlaps or crosses itself at a node).

        The outer nodes of partly driven first/last edges don't count as visited, nor does the
        end of a loop (it is its start).
        """
        path = self.node_path()
        visited = path[1:-1]
        if self.start_t <= 0.0:
            visited.insert(0, path[0])
        if self.end_t >= 1.0 and not (self.closed and self.start_t <= 0.0):
            visited.append(path[-1])
        seen: set[int] = set()
        dup: list[int] = []
        for n in visited:
            if n in seen and n not in dup:
                dup.append(n)
            seen.add(n)
        return dup


def _compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _dump(d: Mapping) -> str:
    """Deterministic JSON: top-level keys sorted, list items / dict entries one per line."""
    lines = ["{"]
    keys = sorted(d)
    for i, key in enumerate(keys):
        value = d[key]
        tail = "," if i + 1 < len(keys) else ""
        if isinstance(value, list) and value:
            body = ",\n".join(f"  {_compact(v)}" for v in value)
            lines.append(f" {_compact(key)}: [\n{body}\n ]{tail}")
        elif isinstance(value, dict) and value and key in ("nodes", "ways"):
            body = ",\n".join(f"  {_compact(k)}: {_compact(value[k])}" for k in sorted(value))
            lines.append(f" {_compact(key)}: {{\n{body}\n }}{tail}")
        else:
            lines.append(f" {_compact(key)}: {_compact(value)}{tail}")
    lines.append("}")
    return "\n".join(lines)


def _edge_attributes(tags: Mapping[str, str], direction: int, region: str) -> tuple:
    return (
        road_width(tags, region),
        lane_count(tags),
        surface_class(tags),
        is_bridge(tags),
        is_tunnel(tags),
        layer_of(tags),
        tags.get("highway", ""),
        is_oneway(tags) * direction,
        tags.get("name") or tags.get("ref") or "",
    )


def _round_t(t: float) -> float:
    return round(float(t), 9)


# -------------------------------------------------------------------------------------------------
# Snapping
# -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Snap:
    seg: int
    t: float
    node: int  # graph node index, or -1 for an edge pin
    xy: tuple[float, float]
    dist: float


def _class_rank(highway: str) -> int:
    if highway.endswith("_link"):
        return _CLASS_RANK.get(highway[:-5], 0) - 1
    return _CLASS_RANK.get(highway, 0)


def snap_pin(graph: EdgeGraph, lat: float, lon: float, options: RouterOptions) -> _Snap:
    e, n = graph.proj.to_enu(lon, lat)
    p = np.array([float(e), float(n)])
    a = graph.node_xy[graph.seg_a]
    ab = graph.node_xy[graph.seg_b] - a
    ab2 = np.sum(ab * ab, axis=1)
    t = np.clip(np.sum((p - a) * ab, axis=1) / np.where(ab2 > 0.0, ab2, 1.0), 0.0, 1.0)
    closest = a + ab * t[:, None]
    dist = np.sqrt(np.sum((closest - p) ** 2, axis=1))
    if len(dist) == 0:
        raise RouteError("no road segments in the fetched area")
    dmin = float(dist.min())
    if dmin > options.max_snap_m:
        raise RouteError(
            f"pin ({lat:.6f}, {lon:.6f}) is {dmin:.0f} m from the nearest road "
            f"(max {options.max_snap_m:.0f} m)"
        )
    cand = np.flatnonzero(dist <= dmin + options.snap_class_tolerance_m)
    ranks = np.array(
        [-_class_rank(graph.way_tags[int(graph.seg_way[s])].get("highway", "")) for s in cand]
    )
    dist_mm = np.rint(dist[cand] * 1000.0).astype(np.int64)
    order = np.lexsort((graph.seg_idx[cand], graph.seg_way[cand], dist_mm, ranks))
    s = int(cand[order[0]])
    ts = float(t[s])
    seg_len = graph.seg_len_mm[s] / 1000.0
    node = -1
    if ts * seg_len <= options.node_snap_m:
        node, ts = int(graph.seg_a[s]), 0.0
    elif (1.0 - ts) * seg_len <= options.node_snap_m:
        node, ts = int(graph.seg_b[s]), 1.0
    if node >= 0:
        xy = graph.node_xy[node]
    else:
        xy = graph.node_xy[graph.seg_a[s]] + ab[s] * ts
    return _Snap(s, ts, node, (float(xy[0]), float(xy[1])), float(np.hypot(*(xy - p))))


# -------------------------------------------------------------------------------------------------
# Routing
# -------------------------------------------------------------------------------------------------


class _Search:
    """Layered edge-based Dijkstra; see the module docstring."""

    def __init__(self, graph: EdgeGraph, snaps: Sequence[_Snap], options: RouterOptions):
        self.g = graph
        self.options = options
        self.snaps = snaps
        self.n = len(snaps)
        self.len_mm = graph.edge_len_mm.tolist()
        self.w = list(self.len_mm)  # search cost of each edge (route_alternatives raises them)
        self.edge_seg = graph.edge_seg.tolist()
        self.edge_dir = graph.edge_dir.tolist()
        self.edge_u = graph.edge_u.tolist()
        self.edge_v = graph.edge_v.tolist()
        self.seg_fwd = graph.seg_fwd
        self.seg_bwd = graph.seg_bwd
        self.vec = graph.edge_vectors().tolist()
        self.degree = graph.node_degree.tolist()
        self.out = [graph.outgoing(x).tolist() for x in range(graph.n_nodes)]
        self.edge_way = graph.seg_way[graph.edge_seg].tolist()
        self.cos_u = math.cos(math.radians(options.u_turn_angle_deg))
        self.sharp_mm = options.sharp_turn_penalty_m * 1000.0
        self.name_mm = int(round(options.name_change_penalty_m * 1000.0))
        self.way_names = {
            w: frozenset(v for k in ("name", "ref") if (v := tags.get(k)))
            for w, tags in graph.way_tags.items()
        }

    def offset(self, e: int, pos: float) -> int:
        """Integer cost from the tail of edge e to parameter pos along it (millimetres unless
        the edge was penalized)."""
        return int(round(pos * self.w[e]))

    def penalize(self, segs: set[int], factor: float) -> None:
        """Multiply the cost of both directions of each segment in `segs` by `factor`."""
        for s in sorted(segs):
            for e in (int(self.seg_fwd[s]), int(self.seg_bwd[s])):
                if e >= 0:
                    self.w[e] = int(round(self.w[e] * factor))

    def advance(self, e: int, k: int, pos: float) -> tuple[int, float | None]:
        """Drive edge e from parameter `pos` with k pins passed; return (k', finish pos or None)."""
        seg = self.edge_seg[e]
        head = self.edge_v[e]
        while k + 1 < self.n:
            snap = self.snaps[k + 1]
            if snap.node < 0 and snap.seg == seg:
                t = snap.t if self.edge_dir[e] > 0 else 1.0 - snap.t
                if t < pos:
                    break
                pos = t
            elif snap.node >= 0 and snap.node == head:
                pos = 1.0
            else:
                break
            k += 1
            if k == self.n - 1:
                return k, pos
        return k, None

    def turn_cost(self, e1: int, e2: int) -> int | None:
        """Integer cost of turning from e1 into e2 at their shared node; None if forbidden."""
        if self.edge_seg[e1] == self.edge_seg[e2]:
            return None  # straight back along the same segment
        node = self.edge_v[e1]
        if self.degree[node] <= 2:
            return 0  # no choice here: just the road's own shape
        a, b = self.vec[e1], self.vec[e2]
        cos = a[0] * b[0] + a[1] * b[1]
        if cos < self.cos_u:
            return None
        cost = int(round(self.sharp_mm * (1.0 - cos) ** 2))
        w1, w2 = self.edge_way[e1], self.edge_way[e2]
        if w1 != w2 and self.way_names[w1].isdisjoint(self.way_names[w2]):
            if self.way_names[w1] or self.way_names[w2]:
                cost += self.name_mm
        return cost

    def seeds(self) -> list[tuple[int, float, int]]:
        """(edge, start position on it, pins passed) for every way of leaving pin 0."""
        n = self.n
        s0 = self.snaps[0]
        if s0.node >= 0:
            k0 = 0
            while k0 + 1 < n and self.snaps[k0 + 1].node == s0.node:
                k0 += 1
            if k0 == n - 1:
                raise RouteError("all pins snapped to the same point")
            return [(e, 0.0, k0) for e in self.out[s0.node]]
        seeds = []
        for e, pos in ((self.seg_fwd[s0.seg], s0.t), (self.seg_bwd[s0.seg], 1.0 - s0.t)):
            if e >= 0:
                seeds.append((int(e), pos, 0))
        return seeds

    def run(self) -> tuple[list[int], float, float]:
        """Return (edge indices, start_t on the first, end_t on the last)."""
        best = self.search(self.seeds())
        if best is None:
            raise RouteError(
                "no route joins the pins in order without U-turns inside the fetched area"
            )
        return best[1:]

    def run_loop(self) -> tuple[list[int], float, float]:
        """The cheapest closed route: back at pin 0 the way it left, every road stretch once."""
        found = []
        for seed in self.seeds():
            res = self.search([seed], closing=seed)
            if res is not None:
                found.append(res)
        found.sort(key=lambda r: (r[0], r[1]))
        for res in found:
            if not self.drives_a_segment_twice(res[1]):
                return res[1:]
        if found:
            raise RouteError(f"no loop: {LOOP_MESSAGE}")
        # No way back to the start at all: say whether the pins themselves can be joined.
        open_search = _Search(self.g, self.snaps[:-1], self.options)
        if open_search.n >= 2 and open_search.search(open_search.seeds()) is None:
            raise RouteError(
                "no route joins the pins in order without U-turns inside the fetched area"
            )
        raise RouteError(f"no loop: {LOOP_MESSAGE}")

    def drives_a_segment_twice(self, edges: list[int]) -> bool:
        """True if a closed route uses a segment twice (either way). The first and last edges
        of a loop that starts mid-segment are the same edge split at the start: one use."""
        inner = edges[:-1] if len(edges) > 1 and edges[0] == edges[-1] else edges
        segs = [self.edge_seg[e] for e in inner]
        return len(set(segs)) != len(segs)

    def search(
        self, seeds: list[tuple[int, float, int]], closing: tuple[int, float, int] | None = None
    ) -> tuple[int, list[int], float, float] | None:
        """Dijkstra from `seeds`; (cost, edges, start_t, end_t) or None. With `closing` (the one
        seed of a loop) the route must end back on that seed: on the same directed edge for a
        pin on a segment, or turning into it without a U-turn for a pin on a node."""
        n = self.n
        best: dict[int, int] = {}
        pred: dict[int, int] = {}
        start_pos: dict[int, float] = {}
        heap: list[tuple[int, int]] = []
        finish: tuple[int, int, float, int, float] | None = None  # cost, edge, pos, prev, start

        def offer_finish(cost: int, e: int, pos: float, prev: int, spos: float) -> None:
            nonlocal finish
            if closing is not None:
                if prev < 0:
                    return  # a loop must leave the start edge and come back to it
                if self.snaps[0].node >= 0:
                    tc = self.turn_cost(e, closing[0])
                    if tc is None:
                        return
                    cost += tc
                elif e != closing[0]:
                    return
            if finish is None or cost < finish[0]:
                finish = (cost, e, pos, prev, spos)

        for e, pos, k in seeds:
            k2, fin = self.advance(e, k, pos)
            if fin is not None:
                offer_finish(self.offset(e, fin) - self.offset(e, pos), e, fin, -1, pos)
                continue
            st = e * n + k2
            cost = self.w[e] - self.offset(e, pos)
            if cost < best.get(st, cost + 1):
                best[st] = cost
                start_pos[st] = pos
                heapq.heappush(heap, (cost, st))

        while heap:
            cost, st = heapq.heappop(heap)
            if cost > best[st]:
                continue
            if finish is not None and cost >= finish[0]:
                break
            e, k = divmod(st, n)
            for e2 in self.out[self.edge_v[e]]:
                tc = self.turn_cost(e, e2)
                if tc is None:
                    continue
                k2, fin = self.advance(e2, k, 0.0)
                if fin is not None:
                    offer_finish(cost + tc + self.offset(e2, fin), e2, fin, st, 0.0)
                    continue
                st2 = e2 * n + k2
                c2 = cost + tc + self.w[e2]
                if c2 < best.get(st2, c2 + 1):
                    best[st2] = c2
                    pred[st2] = st
                    start_pos.pop(st2, None)
                    heapq.heappush(heap, (c2, st2))

        if finish is None:
            return None
        total, last_e, end_pos, prev, spos = finish
        edges = [last_e]
        while prev >= 0:
            edges.append(prev // n)
            spos = start_pos.get(prev, spos)
            prev = pred.get(prev, -1)
        edges.reverse()
        return total, edges, spos, end_pos


def route_on_graph(
    graph: EdgeGraph,
    pins: Sequence[tuple[float, float]],
    options: RouterOptions | None = None,
    osm_base: str | None = None,
    cache_key: str | None = None,
    *,
    loop: bool = False,
) -> Route:
    """Snap (lat, lon) pins onto `graph` and route through them in order (`loop`: and back
    to the first pin; the returned route's pins end with the first one again)."""
    options = options or RouterOptions()
    pins, snaps = _snap_all(graph, pins, options, loop)
    search = _Search(graph, snaps, options)
    edges, start_t, end_t = search.run_loop() if loop else search.run()
    return _build_route(
        graph, pins, snaps, options, edges, start_t, end_t, osm_base, cache_key, loop
    )


def _snap_all(
    graph: EdgeGraph, pins: Sequence[tuple[float, float]], options: RouterOptions, loop: bool
) -> tuple[list[tuple[float, float]], list[_Snap]]:
    if len(pins) < 2:
        raise RouteError("need at least two pins")
    pins = list(pins)
    if loop:
        pins.append(pins[0])
    return pins, [snap_pin(graph, lat, lon, options) for lat, lon in pins]


def _driven_mm(graph: EdgeGraph, edges: list[int], start_t: float, end_t: float) -> dict[int, int]:
    """Millimetres driven on each segment (the first and last edges only in part)."""
    out: dict[int, int] = {}
    last = len(edges) - 1
    for i, e in enumerate(edges):
        full = int(graph.edge_len_mm[e])
        a = int(round(start_t * full)) if i == 0 else 0
        b = int(round(end_t * full)) if i == last else full
        seg = int(graph.edge_seg[e])
        out[seg] = out.get(seg, 0) + max(0, b - a)
    return out


def _build_route(
    graph: EdgeGraph,
    pins: list[tuple[float, float]],
    snaps: list[_Snap],
    options: RouterOptions,
    edges: list[int],
    start_t: float,
    end_t: float,
    osm_base: str | None,
    cache_key: str | None,
    loop: bool,
) -> Route:
    """The `Route` for a search result (edge indices on `graph`, start/end parameters)."""
    g = graph
    total_mm = sum(int(g.edge_len_mm[e]) for e in edges)
    total_mm -= int(round(start_t * g.edge_len_mm[edges[0]]))
    total_mm -= int(g.edge_len_mm[edges[-1]]) - int(round(end_t * g.edge_len_mm[edges[-1]]))

    used_nodes: set[int] = set()
    used_ways: set[int] = set()
    route_edges = []
    for e in edges:
        u, v = int(g.edge_u[e]), int(g.edge_v[e])
        way = int(g.seg_way[g.edge_seg[e]])
        route_edges.append(
            RouteEdge(way, int(g.node_ids[u]), int(g.node_ids[v]), int(g.edge_dir[e]))
        )
        used_nodes.update((u, v))
        used_ways.add(way)

    pins_out = []
    for (lat, lon), sn in zip(pins, snaps, strict=True):
        a, b = int(g.seg_a[sn.seg]), int(g.seg_b[sn.seg])
        used_nodes.update((a, b))
        used_ways.add(int(g.seg_way[sn.seg]))
        slon, slat = g.proj.to_lonlat(sn.xy[0], sn.xy[1])
        pins_out.append(
            SnappedPin(
                lat=float(lat),
                lon=float(lon),
                snap_lat=_ll(slat),
                snap_lon=_ll(slon),
                dist_m=round(sn.dist, 3),
                way=int(g.seg_way[sn.seg]),
                u=int(g.node_ids[a]),
                v=int(g.node_ids[b]),
                t=_round_t(sn.t),
                node=int(g.node_ids[sn.node]) if sn.node >= 0 else None,
            )
        )

    # Side roads leaving interior route nodes (for junction stubs / walls).
    on_route = {int(g.edge_seg[e]) for e in edges}
    path_nodes = [int(g.edge_u[edges[0]])] + [int(g.edge_v[e]) for e in edges]
    interior = sorted(set(path_nodes if loop else path_nodes[1:-1]))
    stubs: list[tuple[int, int, int]] = []
    for x in interior:
        if g.node_degree[x] <= 2:
            continue
        for s in np.flatnonzero((g.seg_a == x) | (g.seg_b == x)).tolist():
            if s in on_route:
                continue
            other = int(g.seg_b[s] if g.seg_a[s] == x else g.seg_a[s])
            stubs.append((int(g.node_ids[x]), int(g.seg_way[s]), int(g.node_ids[other])))
            used_nodes.add(other)
            used_ways.add(int(g.seg_way[s]))
    stubs = sorted(set(stubs))

    lat0, lon0 = graph.proj.lat0, graph.proj.lon0
    return Route(
        origin=(lat0, lon0),
        pins=tuple(pins_out),
        edges=tuple(route_edges),
        start_t=_round_t(start_t),
        end_t=_round_t(end_t),
        length_m=total_mm / 1000.0,
        nodes={
            int(g.node_ids[i]): (float(g.node_lonlat[i, 0]), float(g.node_lonlat[i, 1]))
            for i in sorted(used_nodes)
        },
        ways={w: route_tags(g.way_tags[w]) for w in sorted(used_ways)},
        stubs=tuple(stubs),
        osm_base=osm_base,
        cache_key=cache_key,
        driving_side=driving_side(lat0, lon0),
        options=options.to_dict(),
        closed=loop,
    )


def route_alternatives(
    graph: EdgeGraph,
    pins: Sequence[tuple[float, float]],
    k: int = 4,
    options: RouterOptions | None = None,
    osm_base: str | None = None,
    cache_key: str | None = None,
    *,
    loop: bool = False,
    penalty: float = ALT_PENALTY,
    max_overlap: float = ALT_MAX_OVERLAP,
    max_stretch: float = ALT_MAX_STRETCH,
    max_rounds: int | None = None,
) -> list[Route]:
    """Up to `k` different routes through the pins (the whole route, every pin in order; with
    `loop`, closed loops), best first.

    Penalty method: route_on_graph's best route comes first; then the search cost of every
    segment the last found route drove is multiplied by `penalty` and the search runs again,
    up to `max_rounds` times (default 4k). A new route is kept when no more than
    `max_overlap` of its length runs on any kept route and it is at most `max_stretch` times
    the best route's length. The same rules as route_on_graph apply to every route (no
    U-turns, pins in order, loops drive no stretch of road twice). The others follow the best
    in order of length (ties: edge order), so the result is deterministic.
    """
    options = options or RouterOptions()
    pins, snaps = _snap_all(graph, pins, options, loop)
    search = _Search(graph, snaps, options)
    run = search.run_loop if loop else search.run
    first = run()  # no route at all: RouteError, as route_on_graph
    found: list[tuple[int, tuple[int, ...], tuple[list[int], float, float]]] = []
    kept_driven: list[dict[int, int]] = []

    def keep(res: tuple[list[int], float, float]) -> dict[int, int]:
        driven = _driven_mm(graph, *res)
        found.append((sum(driven.values()), tuple(res[0]), res))
        kept_driven.append(driven)
        return driven

    best_mm = sum(keep(first).values())
    last = kept_driven[0]
    rounds = max_rounds if max_rounds is not None else 4 * k
    for _ in range(rounds):
        if len(found) >= k:
            break
        search.penalize(set(last), penalty)
        try:
            res = run()
        except RouteError:
            break
        last = driven = _driven_mm(graph, *res)
        length = sum(driven.values())
        if length > max_stretch * best_mm or any(tuple(res[0]) == f[1] for f in found):
            continue
        overlaps = (
            sum(min(mm, other.get(seg, 0)) for seg, mm in driven.items()) for other in kept_driven
        )
        if all(ov < max_overlap * length for ov in overlaps):
            keep(res)
    ordered = found[:1] + sorted(found[1:], key=lambda f: (f[0], f[1]))
    return [
        _build_route(graph, pins, snaps, options, *f[2], osm_base, cache_key, loop) for f in ordered
    ]


def route_from_pins(
    pins: Sequence[tuple[float, float]],
    cache_dir: Path | str | None = None,
    *,
    osm: OsmData | None = None,
    options: RouterOptions | None = None,
    buffer_m: float = DEFAULT_BUFFER_M,
    max_buffer_m: float | None = None,
    include_tracks: str = "paved",
    mirrors: Sequence[str] | None = None,
    offline: bool = False,
    refresh: bool = False,
    max_length_m: float = DEFAULT_MAX_LENGTH_M,
    loop: bool = False,
    alternatives: int = 1,
) -> Route:
    """Route through (lat, lon) pins on real roads. The CLI / web preview entry point.

    `loop=True` comes back to the first pin and returns a closed route (see the module
    docstring); a loop that can only come back along the same road raises `RouteError`.
    `alternatives=k` (k > 1) also finds up to k-1 other ways through the pins
    (`route_alternatives`); they are in the returned route's `alternatives`.

    Fetches (or reads from the cache) the drivable ways within `buffer_m` of the pin chain; if
    no route exists the corridor is doubled once (up to `max_buffer_m`, default 2 x buffer).
    Pass `osm` to route on already-loaded data (no fetch). Warns when the route is longer than
    `max_length_m` or passes a node twice (it crosses or overlaps itself).
    """
    pins = [(float(lat), float(lon)) for lat, lon in pins]
    if len(pins) < 2:
        raise RouteError("need at least two pins")
    options = options or RouterOptions()
    proj = LocalProjection(*pins[0])
    buffers = [buffer_m]
    if osm is None:
        limit = max_buffer_m if max_buffer_m is not None else 2.0 * buffer_m
        if limit > buffer_m:
            buffers.append(min(2.0 * buffer_m, limit))
    last_error: RouteError | None = None
    for buf in buffers:
        data = (
            osm
            if osm is not None
            else fetch_corridor(
                pins + [pins[0]] if loop else pins,
                cache_dir,
                buffer_m=buf,
                include_tracks=include_tracks,
                mirrors=mirrors,
                offline=offline,
                refresh=refresh,
            )
        )
        graph = build_graph(data, proj, respect_oneway=options.respect_oneway)
        try:
            if alternatives > 1:
                found = route_alternatives(
                    graph, pins, alternatives, options, data.osm_base, data.cache_key, loop=loop
                )
                route = replace(found[0], alternatives=tuple(found[1:]))
            else:
                route = route_on_graph(
                    graph, pins, options, data.osm_base, data.cache_key, loop=loop
                )
            break
        except RouteError as exc:
            last_error = exc
    else:
        assert last_error is not None
        raise last_error
    if route.length_m > max_length_m:
        warnings.warn(
            f"route is {route.length_m / 1000:.1f} km, over the {max_length_m / 1000:.0f} km limit",
            stacklevel=2,
        )
    if route.repeated_nodes():
        warnings.warn(
            "route passes the same node twice (it crosses or overlaps itself)", stacklevel=2
        )
    return route


def route_from_path(
    osm: OsmData,
    path: Sequence[int],
    start_t: float = 0.0,
    end_t: float = 1.0,
    *,
    loop: bool = False,
    respect_oneway: bool = True,
    options: RouterOptions | None = None,
) -> Route:
    """The Route along `path` (OSM node ids in driving order; each consecutive pair must be
    joined by a segment of one way), from `start_t` along the first segment to `end_t` along
    the last. No pathfinding.

    With `loop`, the path must end at its first node (or at its second, through the first);
    the route then closes where it started (mid-segment starts drive the first segment again
    up to `start_t`, as `route_from_pins` loops do). Going straight back along the segment
    just driven (a U-turn) is refused, and with `respect_oneway` (the default) so is driving
    the wrong way along a one-way road.
    """
    options = options or RouterOptions()
    path = [int(n) for n in path]
    if len(path) < 2:
        raise RouteError("need at least two nodes")
    for t in (start_t, end_t):
        if not (isinstance(t, int | float) and 0.0 <= float(t) <= 1.0):
            raise RouteError("start_t and end_t must be between 0 and 1")
    start_t, end_t = float(start_t), float(end_t)
    if loop:
        if len(path) >= 5 and path[-2:] == path[:2]:
            path.pop()  # it came round and drove on over the start segment: close it at A
        if path[-1] != path[0] or len(path) < 4:
            raise RouteError("a loop must come back to its first node")
        if start_t > 0.0:
            path.append(path[1])
            end_t = start_t
        else:
            end_t = 1.0
    for i in range(len(path) - 1):
        if path[i] == path[i + 1]:
            raise RouteError(f"node {path[i]} is repeated")
        if i + 2 < len(path) and path[i + 2] == path[i]:
            raise RouteError(f"the path turns back on itself at node {path[i + 1]}")
    if len(path) == 2 and end_t <= start_t:
        raise RouteError("the route ends before it starts")

    want = set(path)
    sub = OsmData(
        tuple(w for w in osm.ways if not want.isdisjoint(w.nodes)), osm.osm_base, osm.cache_key
    )
    lonlat = sub.node_lonlat()
    missing = [n for n in path if n not in lonlat]
    if missing:
        raise RouteError(f"node {missing[0]} is not in the loaded road data")
    a, b = np.array(lonlat[path[0]]), np.array(lonlat[path[1]])
    lon0, lat0 = (a + (b - a) * start_t).tolist()
    graph = build_graph(sub, LocalProjection(lat0, lon0), respect_oneway=False)

    def wrong_way(e: int) -> bool:
        tags = graph.way_tags[int(graph.seg_way[graph.edge_seg[e]])]
        return respect_oneway and is_oneway(tags) * int(graph.edge_dir[e]) < 0

    by_pair: dict[tuple[int, int], int] = {}
    for e in range(graph.n_edges):  # ascending: the lowest edge (way id) wins duplicates,
        pair = (int(graph.edge_u[e]), int(graph.edge_v[e]))  # a legal one before a wrong way
        if pair not in by_pair or (wrong_way(by_pair[pair]) and not wrong_way(e)):
            by_pair[pair] = e
    idx = [graph.node_index(n) for n in path]
    edges: list[int] = []
    for i in range(len(idx) - 1):
        e = by_pair.get((idx[i], idx[i + 1]))
        if e is None:
            raise RouteError(
                f"nodes {path[i]} and {path[i + 1]} are not next to each other on a road"
            )
        if wrong_way(e):
            tags = graph.way_tags[int(graph.seg_way[graph.edge_seg[e]])]
            err = RouteError(
                f"one-way: it goes the wrong way along way {int(graph.seg_way[graph.edge_seg[e]])}"
                f" (nodes {path[i]} to {path[i + 1]})"
            )
            err.road = tags.get("name") or tags.get("ref") or None  # type: ignore[attr-defined]
            raise err
        edges.append(e)
    if loop:
        inner = edges[:-1] if edges[0] == edges[-1] else edges
        segs = [int(graph.edge_seg[e]) for e in inner]
        if len(set(segs)) != len(segs):
            raise RouteError(f"no loop: {LOOP_MESSAGE}")

    def snap(e: int, t: float) -> _Snap:
        seg = int(graph.edge_seg[e])
        u, v = int(graph.edge_u[e]), int(graph.edge_v[e])
        xy = graph.node_xy[u] + (graph.node_xy[v] - graph.node_xy[u]) * t
        seg_t = t if graph.edge_dir[e] > 0 else 1.0 - t
        node = u if t <= 0.0 else v if t >= 1.0 else -1
        return _Snap(seg, seg_t, node, (float(xy[0]), float(xy[1])), 0.0)

    first, last = snap(edges[0], start_t), snap(edges[-1], end_t)
    pins_ll = []
    for sn in (first, last):
        slon, slat = graph.proj.to_lonlat(sn.xy[0], sn.xy[1])
        pins_ll.append((_ll(slat), _ll(slon)))
    route = _build_route(
        graph,
        pins_ll,
        [first, last],
        options,
        edges,
        start_t,
        end_t,
        osm.osm_base,
        osm.cache_key,
        loop,
    )
    if route.length_m <= 0.0:
        raise RouteError("the route has no length")
    return route


# --------------------------------------------------------------------------------------------
# Run-up and run-off on the real roads beyond A and B
# --------------------------------------------------------------------------------------------

# A continuation goes on as straight as it can, within this turn at a node; where nothing goes
# on that straight (a T-junction) it turns onto the bigger road, up to CONTINUE_T_TURN_DEG.
CONTINUE_MAX_TURN_DEG = 50.0
CONTINUE_T_TURN_DEG = 100.0
_NOT_THROUGH = frozenset({"driveway", "parking_aisle", "drive-through", "emergency_access"})


@dataclass(frozen=True)
class Continuation:
    """The real road beyond one end of a route: `lonlat` (K, 2) starts at the route's end
    point and runs away from it; `nodes` are the OSM nodes it passes, in order."""

    lonlat: np.ndarray
    nodes: tuple[int, ...]

    @property
    def length_m(self) -> float:
        proj = LocalProjection(float(self.lonlat[0, 1]), float(self.lonlat[0, 0]))
        xy = proj.to_xy(self.lonlat)
        return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def _through_road(tags: Mapping[str, str]) -> bool:
    hw = tags.get("highway", "")
    if hw not in _CLASS_RANK and not hw.endswith("_link"):
        return False
    return tags.get("area") != "yes" and tags.get("service", "") not in _NOT_THROUGH


def route_continuations(
    route: Route,
    ways: Sequence,
    before_m: float,
    after_m: float,
    *,
    max_turn_deg: float = CONTINUE_MAX_TURN_DEG,
) -> tuple[Continuation | None, Continuation | None]:
    """The real roads behind A and beyond B, as far as `before_m` / `after_m` metres.

    From each end the walk goes on along the OSM network (`ways`: anything with `id`,
    `nodes`, `lonlat` and `tags`, e.g. `overpass.Way`): at every node it takes the straightest
    way on, preferring the same way and then the same name, within `max_turn_deg`; at a
    T-junction (nothing that straight) it turns onto the bigger road, up to
    `CONTINUE_T_TURN_DEG`. Never onto the route itself or a driveway. It stops at a dead end
    or once long enough. None where there is no road at all, and for a loop."""
    if route.closed or not route.edges:
        return None, None
    proj = route.projection()
    coords: dict[int, np.ndarray] = {}
    nbrs: dict[int, list[tuple[int, int, str]]] = {}
    for way in ways:
        if not _through_road(way.tags):
            continue
        xy = proj.to_xy(np.asarray(way.lonlat, np.float64))
        name = way.tags.get("name", "") or way.tags.get("ref", "")
        for k, nid in enumerate(way.nodes):
            coords.setdefault(int(nid), xy[k])
        for a, b in zip(way.nodes[:-1], way.nodes[1:], strict=True):
            if a != b:
                nbrs.setdefault(int(a), []).append((int(b), int(way.id), name))
                nbrs.setdefault(int(b), []).append((int(a), int(way.id), name))
    ids = route.node_path()
    on_route = set(ids)
    names = {w: (t.get("name", "") or t.get("ref", "")) for w, t in route.ways.items()}
    rank = {int(w.id): _class_rank(w.tags.get("highway", "")) for w in ways}
    cos_max = math.cos(math.radians(max_turn_deg))
    cos_turn = math.cos(math.radians(CONTINUE_T_TURN_DEG))

    def walk(end: np.ndarray, prev: int, node: int, way: int, need: float) -> Continuation | None:
        if node not in coords:
            return None
        pts, path = [end, coords[node]], [node]
        length = float(np.linalg.norm(coords[node] - end))
        name = names.get(way, "")
        last = coords.get(prev, end)
        seen = {prev, node}
        while length < need:
            here = coords[node]
            d_in = here - last
            if np.linalg.norm(d_in) < 1e-6:
                d_in = here - end
            d_in = d_in / max(float(np.linalg.norm(d_in)), 1e-9)
            best = turn = None
            for q, w, nm in nbrs.get(node, ()):
                if q in seen or q in on_route or q not in coords:
                    continue
                d = coords[q] - here
                dl = float(np.linalg.norm(d))
                if dl < 1e-6:
                    continue
                cos = float(d_in @ d) / dl
                if cos >= cos_max:
                    score = (0 if w == way else 1 if (nm and nm == name) else 2, -cos)
                    if best is None or score < best[0]:
                        best = (score, q, w, nm)
                elif cos >= cos_turn:  # a T-junction: turn onto the bigger road
                    score = (-rank.get(w, 0), -cos, q)
                    if turn is None or score < turn[0]:
                        turn = (score, q, w, nm)
            best = best if best is not None else turn
            if best is None:
                break
            _, q, way, nm = best
            name = nm or name
            last, node = here, q
            seen.add(q)
            pts.append(coords[q])
            path.append(q)
            length += float(np.linalg.norm(coords[q] - here))
        xy = np.asarray(pts)
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        keep = np.concatenate([[0], np.flatnonzero(seg > 1e-6) + 1])
        xy = xy[keep]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        if len(xy) < 2 or s[-1] < 1.0:
            return None
        if s[-1] > need:  # cut at exactly `need` metres
            k = int(np.searchsorted(s, need))
            t = (need - s[k - 1]) / (s[k] - s[k - 1])
            xy = np.vstack([xy[:k], xy[k - 1] + t * (xy[k] - xy[k - 1])])
        return Continuation(proj.from_xy(xy), tuple(path))

    first, last = route.edges[0], route.edges[-1]
    xy = proj.to_xy(np.array([route.nodes[n] for n in ids], dtype=np.float64))
    start = xy[0] + (xy[1] - xy[0]) * route.start_t
    end = xy[-2] + (xy[-1] - xy[-2]) * route.end_t
    on_route.discard(first.u)
    on_route.discard(last.v)
    before = walk(start, first.v, first.u, first.way, before_m) if before_m > 0 else None
    after = walk(end, last.u, last.v, last.way, after_m) if after_m > 0 else None
    return before, after
