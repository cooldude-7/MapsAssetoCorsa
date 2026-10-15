"""Real roadside barriers from OSM, and the per-track override file, as station ranges.

The user wants guardrails ONLY where they exist in real life (docs/research/scenery.md §5), so
nothing here guesses. Two sources say where a barrier stands:

1. **OSM ways** tagged `barrier=guard_rail | jersey_barrier | retaining_wall | wall` within
   `around_m` (default 15 m) of the route, fetched with one Overpass `around:` query along the
   (simplified) route line and cached like `geo.overpass` (same mirrors, retry policy and
   `<cache>/overpass/<sha256(query)>.json.gz` layout). On 2026-10-15 none of the three Ontario
   test roads had any (scenery.md §5a); they appear automatically once someone maps them.
2. **`guardrails.json`** next to `route.json`: a list of
   `{"side": "left" | "right", "from_m": s0, "to_m": s1, "kind": "guard_rail"}` in metres of
   route distance from A (the user has driven these roads and knows where the rails are).

`match_barriers` turns each OSM way into `BarrierSpan`s along the built centreline: every
point of the way (densified to 1 m) is projected onto the nearest centreline segment, which
gives its station and signed lateral offset (+ = driver's left, AC left = up x forward). Runs
of points on one side, within `max_offset_m`, not past either end of the road and without a
station jump (a jump means the way left this leg, e.g. at a hairpin) become one span. Runs that
do not run ALONG the road (station extent < `min_alignment` x their length: a wall at right
angles to the road, a fence around a yard) or that are shorter than `min_length_m` are dropped.

Stations are model stations (`RoadModel.s`, metres along the built centreline from its first
point, i.e. including the run-up before A); the override file is in route distance from A,
which `parse_user_spans(..., s_origin=s_a)` converts.

OSM data is (c) OpenStreetMap contributors, ODbL 1.0 (`overpass.OSM_ATTRIBUTION`).
"""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import shapely
from scipy.spatial import cKDTree

from driftmaps.geo.attributes import parse_width
from driftmaps.geo.overpass import (
    OsmData,
    OverpassError,
    _cache_path,
    default_cache_dir,
    load_cached,
    parse_overpass,
    post_query,
    query_cache_key,
)
from driftmaps.geo.project import LocalProjection

__all__ = [
    "BARRIER_KINDS",
    "DEFAULT_AROUND_M",
    "USER_SPANS_FILE",
    "BarrierSpan",
    "OsmBarrier",
    "Side",
    "build_barrier_query",
    "fetch_barriers",
    "load_user_spans",
    "match_barriers",
    "merge_spans",
    "parse_barriers",
    "parse_user_spans",
]

Side = Literal["left", "right"]

# OSM `barrier=*` values that stand beside a road and are worth modelling (scenery.md §5,
# geo.md §4). Fences and hedges are left to the scenery builder (no collision).
BARRIER_KINDS: tuple[str, ...] = ("guard_rail", "jersey_barrier", "retaining_wall", "wall")
DEFAULT_AROUND_M = 15.0  # scenery.md §7: barriers within 15 m of the route
USER_SPANS_FILE = "guardrails.json"
_USER_KINDS = ("guard_rail", "jersey_barrier", "wall", "retaining_wall")


@dataclass(frozen=True)
class OsmBarrier:
    """One OSM barrier way: its kind (`barrier` tag), (lon, lat) points (N, 2) and tags."""

    id: int
    kind: str
    lonlat: np.ndarray = field(repr=False)
    tags: Mapping[str, str] = field(default_factory=dict)

    @property
    def height_m(self) -> float | None:
        """The `height` tag in metres (ft/inch units understood), if any."""
        return parse_width(self.tags.get("height"))


@dataclass(frozen=True)
class BarrierSpan:
    """A barrier along one side of the road between model stations `s0` < `s1`.

    `side` is the driver's side in the model's driving direction. `kind` is an OSM barrier
    value (`guard_rail`, `jersey_barrier`, `wall`, `retaining_wall`) or a bridge railing
    (`parapet`, `bridge_rail`). `source` says where it came from: "osm", "user", "bridge"
    (bridge approach) or "heuristic". `offset_m` is the measured distance of the barrier from
    the centreline (None: at the road edge); `height_m` an OSM `height` tag.
    """

    side: Side
    s0: float
    s1: float
    kind: str = "guard_rail"
    source: str = "osm"
    osm_id: int | None = None
    offset_m: float | None = None
    height_m: float | None = None

    def __post_init__(self) -> None:
        if self.side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {self.side!r}")
        if not (math.isfinite(self.s0) and math.isfinite(self.s1) and self.s1 > self.s0):
            raise ValueError(f"need s0 < s1, got {self.s0}, {self.s1}")

    @property
    def length(self) -> float:
        return self.s1 - self.s0


# --------------------------------------------------------------------------------------------
# Overpass
# --------------------------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def build_barrier_query(
    route_latlon: Sequence[tuple[float, float]] | npt.ArrayLike,
    around_m: float = DEFAULT_AROUND_M,
    kinds: Sequence[str] = BARRIER_KINDS,
    timeout_s: int = 90,
) -> str:
    """Overpass QL for barrier ways within `around_m` of the (lat, lon) route line."""
    pts = np.asarray(route_latlon, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        raise ValueError("need at least one route point")
    if not kinds:
        raise ValueError("need at least one barrier kind")
    coords = ",".join(f"{_fmt(lat)},{_fmt(lon)}" for lat, lon in pts)
    regex = "^(" + "|".join(sorted(kinds)) + ")$"
    return (
        f"[out:json][timeout:{int(timeout_s)}];\n"
        f'way["barrier"~"{regex}"](around:{_fmt(around_m)},{coords});\n'
        "out body geom;\n"
    )


def _simplified_latlon(route_lonlat: np.ndarray, tolerance_m: float) -> np.ndarray:
    """The route line simplified to within `tolerance_m` (Douglas-Peucker), as (lat, lon)."""
    proj = LocalProjection(float(route_lonlat[0, 1]), float(route_lonlat[0, 0]))
    xy = proj.to_xy(route_lonlat)
    if len(xy) >= 2 and tolerance_m > 0:
        line = shapely.LineString(xy).simplify(tolerance_m, preserve_topology=False)
        xy = np.asarray(line.coords)
    ll = proj.from_xy(xy)
    return np.round(ll[:, ::-1], 6)


def parse_barriers(osm: OsmData | bytes | str | Mapping, kinds=BARRIER_KINDS) -> list[OsmBarrier]:
    """The barrier ways (of `kinds`) in an Overpass result, sorted by id."""
    data = osm if isinstance(osm, OsmData) else parse_overpass(osm)
    out = []
    for way in data.ways:
        kind = way.tags.get("barrier")
        if kind in kinds:
            out.append(OsmBarrier(way.id, kind, way.lonlat, dict(way.tags)))
    return out


def fetch_barriers(
    route_lonlat: npt.ArrayLike,
    *,
    cache_dir: Path | str | None = None,
    around_m: float = DEFAULT_AROUND_M,
    simplify_m: float = 5.0,
    kinds: Sequence[str] = BARRIER_KINDS,
    mirrors: Sequence[str] | None = None,
    refresh: bool = False,
    offline: bool = False,
    **post_kwargs,
) -> list[OsmBarrier]:
    """OSM barrier ways near the route line (N, 2) (lon, lat), from the cache or Overpass.

    The line is simplified by `simplify_m` to keep the query short, and the search radius
    grows by the same amount, so nothing within `around_m` of the real line is missed.
    `offline=True` never touches the network (an OverpassError on a cache miss).
    """
    ll = np.asarray(route_lonlat, dtype=np.float64).reshape(-1, 2)
    query = build_barrier_query(
        _simplified_latlon(ll, simplify_m), around_m + max(simplify_m, 0.0), kinds
    )
    key = query_cache_key(query)
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    if not refresh:
        cached = load_cached(cache, key)
        if cached is not None:
            return parse_barriers(cached, kinds)
    if offline:
        raise OverpassError(f"no cached Overpass barrier data for this route (key {key[:12]}...)")
    body = post_query(query, mirrors, **post_kwargs)
    osm = parse_overpass(body, key)
    path = _cache_path(cache, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(body, mtime=0))
    tmp.replace(path)
    return parse_barriers(osm, kinds)


# --------------------------------------------------------------------------------------------
# Matching to the centreline
# --------------------------------------------------------------------------------------------


def _densify(xz: np.ndarray, step_m: float) -> np.ndarray:
    parts = [xz[:1]]
    for a, b in zip(xz[:-1], xz[1:], strict=True):
        n = max(1, math.ceil(float(np.linalg.norm(b - a)) / step_m))
        t = np.linspace(0.0, 1.0, n + 1)[1:, None]
        parts.append(a + t * (b - a))
    return np.concatenate(parts)


def _project(
    pts: np.ndarray, centre: np.ndarray, s: np.ndarray, tree: cKDTree
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(station, signed lateral offset (+ left), inside) of each XZ point on the polyline."""
    n = len(centre)
    _, i = tree.query(pts)
    best_d = np.full(len(pts), np.inf)
    st = np.zeros(len(pts))
    off = np.zeros(len(pts))
    inside = np.ones(len(pts), bool)
    for j0 in (i - 1, i):
        ok = (j0 >= 0) & (j0 < n - 1)
        a = np.clip(j0, 0, n - 2)
        p, q = centre[a], centre[a + 1]
        seg = q - p
        length = np.linalg.norm(seg, axis=1)
        unit = seg / np.maximum(length, 1e-12)[:, None]
        t_raw = np.einsum("ij,ij->i", pts - p, unit)
        t = np.clip(t_raw, 0.0, length)
        foot = p + t[:, None] * unit
        d = np.linalg.norm(pts - foot, axis=1)
        better = ok & (d < best_d)
        left = np.stack([unit[:, 1], -unit[:, 0]], axis=1)  # AC: left = up x forward, in XZ
        best_d = np.where(better, d, best_d)
        st = np.where(better, s[a] + t, st)
        off = np.where(better, np.einsum("ij,ij->i", pts - foot, left), off)
        past = ((a == 0) & (t_raw < -1e-6)) | ((a == n - 2) & (t_raw > length + 1e-6))
        inside = np.where(better, ~past, inside)
    return st, off, inside


def match_barriers(
    barriers: Sequence[OsmBarrier],
    projection: LocalProjection,
    centre_xz: npt.ArrayLike,
    s: npt.ArrayLike,
    *,
    max_offset_m: float = DEFAULT_AROUND_M,
    min_offset_m: float = 1.0,
    min_length_m: float = 4.0,
    min_alignment: float = 0.5,
    step_m: float = 1.0,
    max_jump_m: float = 8.0,
) -> list[BarrierSpan]:
    """Match OSM barrier ways to the road: which side and which station range they run along.

    `centre_xz` (N, 2) is the centreline in AC X/Z (X = east, Z = south) on `projection`'s
    plane and `s` (N,) its stations. Points closer than `min_offset_m` to the centreline (on
    the carriageway: a mapping error or a crossing) are ignored. Returns spans sorted by
    (side, s0); `offset_m` is the median measured distance from the centreline.
    """
    centre = np.asarray(centre_xz, dtype=np.float64).reshape(-1, 2)
    st_all = np.asarray(s, dtype=np.float64).reshape(-1)
    if len(centre) < 2 or len(centre) != len(st_all):
        raise ValueError("centre_xz and s need the same length >= 2")
    tree = cKDTree(centre)
    spans: list[BarrierSpan] = []
    for b in barriers:
        if len(b.lonlat) < 2:
            continue
        e, n = projection.to_enu(b.lonlat[:, 0], b.lonlat[:, 1])
        pts = _densify(np.stack([e, -n], axis=1), step_m)
        st, off, inside = _project(pts, centre, st_all, tree)
        ok = inside & (np.abs(off) <= max_offset_m) & (np.abs(off) >= min_offset_m)
        side = np.sign(off)
        arc = np.concatenate([[0.0], np.linalg.norm(np.diff(pts, axis=0), axis=1)])
        start = 0
        for k in range(1, len(pts) + 1):
            if (
                k < len(pts)
                and ok[k]
                and ok[k - 1]
                and side[k] == side[k - 1]
                and abs(st[k] - st[k - 1]) <= max_jump_m
            ):
                continue
            run = slice(start, k)
            start = k
            if k - run.start < 2 or not ok[run.start]:
                continue
            s0, s1 = float(st[run].min()), float(st[run].max())
            length = float(arc[run.start + 1 : k].sum())
            if s1 - s0 < min_length_m or (s1 - s0) < min_alignment * length:
                continue
            spans.append(
                BarrierSpan(
                    side="left" if side[run.start] > 0 else "right",
                    s0=s0,
                    s1=s1,
                    kind=b.kind,
                    source="osm",
                    osm_id=b.id,
                    offset_m=float(np.median(np.abs(off[run]))),
                    height_m=b.height_m,
                )
            )
    return sorted(spans, key=lambda sp: (sp.side, sp.s0, sp.s1))


def merge_spans(spans: Sequence[BarrierSpan], join_m: float = 1.0) -> list[BarrierSpan]:
    """Join spans of the same side, kind and source that overlap or nearly touch (< `join_m`),
    e.g. one guard rail mapped as several OSM ways. The first span's metadata is kept."""
    out: list[BarrierSpan] = []
    for sp in sorted(spans, key=lambda x: (x.side, x.kind, x.source, x.s0)):
        last = out[-1] if out else None
        if (
            last is not None
            and (last.side, last.kind, last.source) == (sp.side, sp.kind, sp.source)
            and sp.s0 <= last.s1 + join_m
        ):
            out[-1] = BarrierSpan(
                last.side,
                last.s0,
                max(last.s1, sp.s1),
                last.kind,
                last.source,
                last.osm_id,
                last.offset_m,
                last.height_m,
            )
        else:
            out.append(sp)
    return sorted(out, key=lambda x: (x.side, x.s0, x.s1))


# --------------------------------------------------------------------------------------------
# User override file
# --------------------------------------------------------------------------------------------


def parse_user_spans(
    data: object, *, s_origin: float = 0.0, default_kind: str = "guard_rail"
) -> list[BarrierSpan]:
    """User barrier spans from parsed `guardrails.json`: a list of
    `{"side", "from_m", "to_m", "kind"?}` (or `{"guardrails": [...]}`), in metres of route
    distance from A; `s_origin` (the model station of A) converts them to model stations.
    `from_m` > `to_m` is accepted (swapped). Raises ValueError on anything malformed."""
    items = data.get("guardrails") if isinstance(data, Mapping) else data
    if not isinstance(items, list):
        raise ValueError("guardrails.json must be a list of spans (or {'guardrails': [...]})")
    out = []
    for k, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"span {k}: expected an object, got {item!r}")
        side = item.get("side")
        kind = item.get("kind", default_kind)
        if side not in ("left", "right"):
            raise ValueError(f"span {k}: side must be 'left' or 'right', got {side!r}")
        if kind not in _USER_KINDS:
            raise ValueError(f"span {k}: kind must be one of {_USER_KINDS}, got {kind!r}")
        try:
            a, b = float(item["from_m"]), float(item["to_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"span {k}: needs numeric from_m and to_m") from exc
        a, b = min(a, b), max(a, b)
        if not b > a:
            raise ValueError(f"span {k}: from_m and to_m must differ")
        out.append(BarrierSpan(side, s_origin + a, s_origin + b, kind, "user"))
    return out


def load_user_spans(path: Path | str, *, s_origin: float = 0.0) -> list[BarrierSpan]:
    """`parse_user_spans` of a JSON file; a missing file means no user spans ([])."""
    path = Path(path)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path}: not valid JSON: {exc}") from exc
    return parse_user_spans(data, s_origin=s_origin)
