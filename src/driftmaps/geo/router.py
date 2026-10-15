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
- **Determinism**: costs are integer millimetres, the heap orders ties by the state number
  (edge index, then pins passed), and edges are numbered in sorted way order. Same OSM snapshot
  + same pins + same options => identical `route.json` bytes. (ASSUMED across machines: the
  projection uses libm/SIMD transcendental functions whose last-bit results can differ between
  CPUs; the build never re-routes, it reads `route.json`, so this cannot split preview and build.)

`Route.to_json()` is the contract: snapshot time and cache key, pins (as given and as snapped),
the ordered directed edges `(way, u, v, dir)`, the coordinates of every node used, the tags of
every way used, side-road stubs at route junctions, and the length. `Route.reverse()` drives
the same edges the other way. `Route.polyline(step)` densifies it with per-point attributes.
"""

from __future__ import annotations

import heapq
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
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
    "ROUTE_VERSION",
    "Route",
    "RouteEdge",
    "RouteError",
    "RoutePolyline",
    "RouterOptions",
    "SnappedPin",
    "route_from_pins",
    "route_on_graph",
    "snap_pin",
]

ROUTE_VERSION = 1
DEFAULT_MAX_LENGTH_M = 25_000.0  # PLAN §5.1: v1 warns above this

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

    # -- serialisation -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
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

        The outer nodes of partly driven first/last edges don't count as visited.
        """
        path = self.node_path()
        visited = path[1:-1]
        if self.start_t <= 0.0:
            visited.insert(0, path[0])
        if self.end_t >= 1.0:
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
        self.snaps = snaps
        self.n = len(snaps)
        self.len_mm = graph.edge_len_mm.tolist()
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
        """Integer millimetres from the tail of edge e to parameter pos along it."""
        return int(round(pos * self.len_mm[e]))

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

    def run(self) -> tuple[list[int], float, float]:
        """Return (edge indices, start_t on the first, end_t on the last)."""
        n = self.n
        best: dict[int, int] = {}
        pred: dict[int, int] = {}
        start_pos: dict[int, float] = {}
        heap: list[tuple[int, int]] = []
        finish: tuple[int, int, float, int, float] | None = None  # cost, edge, pos, prev, start

        def offer_finish(cost: int, e: int, pos: float, prev: int, spos: float) -> None:
            nonlocal finish
            if finish is None or cost < finish[0]:
                finish = (cost, e, pos, prev, spos)

        # Seeds: leave pin 0 along its segment in each direction (or every edge from its node).
        s0 = self.snaps[0]
        if s0.node >= 0:
            k0 = 0
            while k0 + 1 < n and self.snaps[k0 + 1].node == s0.node:
                k0 += 1
            if k0 == n - 1:
                raise RouteError("all pins snapped to the same point")
            seeds = [(e, 0.0, k0) for e in self.out[s0.node]]
        else:
            seeds = []
            for e, pos in ((self.seg_fwd[s0.seg], s0.t), (self.seg_bwd[s0.seg], 1.0 - s0.t)):
                if e >= 0:
                    seeds.append((int(e), pos, 0))
        for e, pos, k in seeds:
            k2, fin = self.advance(e, k, pos)
            if fin is not None:
                offer_finish(self.offset(e, fin) - self.offset(e, pos), e, fin, -1, pos)
                continue
            st = e * n + k2
            cost = self.len_mm[e] - self.offset(e, pos)
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
                c2 = cost + tc + self.len_mm[e2]
                if c2 < best.get(st2, c2 + 1):
                    best[st2] = c2
                    pred[st2] = st
                    start_pos.pop(st2, None)
                    heapq.heappush(heap, (c2, st2))

        if finish is None:
            raise RouteError(
                "no route joins the pins in order without U-turns inside the fetched area"
            )
        _, last_e, end_pos, prev, spos = finish
        edges = [last_e]
        while prev >= 0:
            edges.append(prev // n)
            spos = start_pos.get(prev, spos)
            prev = pred.get(prev, -1)
        edges.reverse()
        return edges, spos, end_pos


def route_on_graph(
    graph: EdgeGraph,
    pins: Sequence[tuple[float, float]],
    options: RouterOptions | None = None,
    osm_base: str | None = None,
    cache_key: str | None = None,
) -> Route:
    """Snap (lat, lon) pins onto `graph` and route through them in order."""
    options = options or RouterOptions()
    if len(pins) < 2:
        raise RouteError("need at least two pins")
    snaps = [snap_pin(graph, lat, lon, options) for lat, lon in pins]
    edges, start_t, end_t = _Search(graph, snaps, options).run()
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
    interior = sorted(set(path_nodes[1:-1]))
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
    )


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
) -> Route:
    """Route through (lat, lon) pins on real roads. The CLI / web preview entry point.

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
                pins,
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
            route = route_on_graph(graph, pins, options, data.osm_base, data.cache_key)
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
