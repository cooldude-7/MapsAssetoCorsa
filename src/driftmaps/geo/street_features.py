"""OSM street furniture near a route: traffic control nodes, lamps, poles, hydrants, signs and
power lines, for `geometry.furniture`.

One Overpass `around:` query along the (simplified) route line, cached like `geo.barriers`
(same mirrors and retry policy, `<cache>/overpass/<sha256(query)>.json.gz`), answered first by a
downloaded region database (`geo.offline`, FORMAT 3 and later) when it covers the route:

- nodes `highway=stop|give_way|traffic_signals|crossing|street_lamp` (with `direction=*`,
  `stop=all`, `crossing=*`), `power=pole|tower`, `emergency=fire_hydrant`, any `traffic_sign=*`;
- ways `power=line|minor_line` (the wires a pole line carries).

Speed limits come from the route's own ways (`maxspeed`, in route.json), not from here.

OSM data is (c) OpenStreetMap contributors, ODbL 1.0 (`overpass.OSM_ATTRIBUTION`).
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from driftmaps.geo.barriers import _fmt, _simplified_latlon
from driftmaps.geo.overpass import (
    OverpassError,
    _cache_path,
    default_cache_dir,
    post_query,
    query_cache_key,
)

__all__ = [
    "AROUND_M",
    "HIGHWAY_NODES",
    "POWER_LINES",
    "OsmLine",
    "OsmNode",
    "StreetFeatures",
    "build_street_query",
    "fetch_street_features",
    "is_street_node",
    "parse_street_features",
]

AROUND_M = 40.0  # furniture this close to the route line (poles stand up to ~20 m off)
HIGHWAY_NODES: tuple[str, ...] = ("crossing", "give_way", "stop", "street_lamp", "traffic_signals")
POWER_NODES: tuple[str, ...] = ("pole", "tower")
POWER_LINES: tuple[str, ...] = ("line", "minor_line")


def is_street_node(tags: Mapping[str, str]) -> bool:
    """An OSM node this layer keeps (the offline database stores exactly these)."""
    return (
        tags.get("highway") in HIGHWAY_NODES
        or tags.get("power") in POWER_NODES
        or tags.get("emergency") == "fire_hydrant"
        or "traffic_sign" in tags
    )


def is_power_line(tags: Mapping[str, str]) -> bool:
    return tags.get("power") in POWER_LINES


@dataclass(frozen=True)
class OsmNode:
    id: int
    lon: float
    lat: float
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OsmLine:
    """A power line way: node ids (may be empty), (N, 2) lon/lat points and tags."""

    id: int
    lonlat: np.ndarray = field(repr=False)
    tags: Mapping[str, str] = field(default_factory=dict)
    nodes: tuple[int, ...] = ()


@dataclass(frozen=True)
class StreetFeatures:
    nodes: tuple[OsmNode, ...] = ()
    power_lines: tuple[OsmLine, ...] = ()
    osm_base: str | None = None

    def __len__(self) -> int:
        return len(self.nodes) + len(self.power_lines)


def build_street_query(
    route_latlon: Sequence[tuple[float, float]] | npt.ArrayLike,
    around_m: float = AROUND_M,
    timeout_s: int = 90,
) -> str:
    """Overpass QL for the street furniture nodes and power lines within `around_m` of the
    (lat, lon) route line."""
    pts = np.asarray(route_latlon, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        raise ValueError("need at least one route point")
    coords = ",".join(f"{_fmt(lat)},{_fmt(lon)}" for lat, lon in pts)
    near = f"(around:{_fmt(around_m)},{coords})"
    hw = "^(" + "|".join(HIGHWAY_NODES) + ")$"
    pw = "^(" + "|".join(POWER_NODES) + ")$"
    pl = "^(" + "|".join(POWER_LINES) + ")$"
    return (
        f"[out:json][timeout:{int(timeout_s)}];\n"
        "(\n"
        f'  node["highway"~"{hw}"]{near};\n'
        f'  node["power"~"{pw}"]{near};\n'
        f'  node["emergency"="fire_hydrant"]{near};\n'
        f'  node["traffic_sign"]{near};\n'
        f'  way["power"~"{pl}"]{near};\n'
        ");\n"
        "out body geom;\n"
    )


def parse_street_features(payload: bytes | str | Mapping) -> StreetFeatures:
    """Overpass JSON (or the offline database's payload) -> StreetFeatures (sorted by id)."""
    if isinstance(payload, bytes | str):
        try:
            payload = json.loads(payload)
        except ValueError as exc:
            raise OverpassError(f"not an Overpass JSON response: {exc}") from exc
    if not isinstance(payload, Mapping) or "elements" not in payload:
        raise OverpassError("not an Overpass JSON response (no 'elements')")
    remark = payload.get("remark")
    if remark and "error" in str(remark).lower():
        raise OverpassError(f"Overpass reported: {remark}")
    nodes: dict[int, OsmNode] = {}
    lines: dict[int, OsmLine] = {}
    for el in payload["elements"]:
        tags = {str(k): str(v) for k, v in (el.get("tags") or {}).items()}
        if el.get("type") == "node" and "lat" in el and "lon" in el:
            if is_street_node(tags):
                nid = int(el["id"])
                nodes[nid] = OsmNode(nid, float(el["lon"]), float(el["lat"]), tags)
        elif el.get("type") == "way" and is_power_line(tags):
            geom = [g for g in (el.get("geometry") or []) if g is not None]
            if len(geom) < 2:
                continue
            ll = np.array([[g["lon"], g["lat"]] for g in geom], np.float64)
            ids = tuple(int(n) for n in (el.get("nodes") or []))
            wid = int(el["id"])
            lines[wid] = OsmLine(wid, ll, tags, ids if len(ids) == len(ll) else ())
    osm_base = (payload.get("osm3s") or {}).get("timestamp_osm_base")
    return StreetFeatures(
        tuple(nodes[k] for k in sorted(nodes)), tuple(lines[k] for k in sorted(lines)), osm_base
    )


def fetch_street_features(
    route_lonlat: npt.ArrayLike,
    *,
    cache_dir: Path | str | None = None,
    around_m: float = AROUND_M,
    simplify_m: float = 5.0,
    mirrors: Sequence[str] | None = None,
    refresh: bool = False,
    offline: bool = False,
    **post_kwargs,
) -> StreetFeatures:
    """Street furniture nodes and power lines near the (N, 2) lon/lat route line, from a
    region database, the cache or Overpass (like `barriers.fetch_barriers`)."""
    ll = np.asarray(route_lonlat, dtype=np.float64).reshape(-1, 2)
    line = _simplified_latlon(ll, simplify_m)
    radius = around_m + max(simplify_m, 0.0)
    if not refresh:
        from driftmaps.geo import offline as region_db

        found = region_db.street_payload(line, radius)
        if found is not None:
            return parse_street_features(found)
    query = build_street_query(line, radius)
    key = query_cache_key(query)
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    path = _cache_path(cache, key)
    if not refresh and path.is_file():
        try:
            return parse_street_features(gzip.decompress(path.read_bytes()))
        except (OSError, OverpassError):
            pass  # corrupt cache entry: refetch
    if offline:
        raise OverpassError(f"no cached Overpass street data for this route (key {key[:12]}...)")
    body = post_query(query, mirrors, **post_kwargs)
    feats = parse_street_features(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(body, mtime=0))
    tmp.replace(path)
    return feats
