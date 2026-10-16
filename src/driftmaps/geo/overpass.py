"""Fetch drivable OSM ways in a corridor around the pins from the Overpass API, with a disk cache.

Design (docs/research/geo.md §1.2, PLAN §5.1):
- One query per corridor: every `highway` way of a drivable class within `buffer_m` of the
  straight pin chain, using Overpass' linestring `around:` filter (since 0.7.55, VERIFIED in
  geo.md §1.2). `out body geom` returns each way's node ids (topology) and coordinates.
- Paved `highway=track` ways are included by default; service roads that are parking aisles,
  driveways or drive-throughs are left out so routes can't cut through car parks.
- Mirrors are raced (hedged requests): the first is asked, the next one too as soon as it fails
  or after `RACE_STAGGER_S` without an answer, and so on (threads); the first good answer wins
  and late ones are ignored. Exponential backoff between rounds. Overpass answers 429 when rate
  limited and 504 when overloaded (VERIFIED, geo.md §1.1); its fair-use policy asks for a 30 s
  pause after a 429 and an identifying User-Agent. On 2026-10-15 overpass-api.de reset
  connections from the dev sandbox while maps.mail.ru answered, so the mirror list matters.
- `build_bbox_query` asks for the same drivable ways inside one or more lat/lon boxes (the
  app's map tiles, `driftmaps.app.tiles`).
- The raw response is cached gzip'd under `<cache_dir>/overpass/<sha256(query)>.json.gz`. The
  hash is the cache key that `route.json` records, together with `osm3s.timestamp_osm_base`.

OSM data is (c) OpenStreetMap contributors, ODbL 1.0: anything built from it must say so.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = [
    "DEFAULT_BUFFER_M",
    "DEFAULT_MIRRORS",
    "ENV_CACHE_DIR",
    "ENV_OVERPASS_URL",
    "NETWORK_AROUND_M",
    "OSM_ATTRIBUTION",
    "NetworkData",
    "OsmData",
    "OsmNode",
    "OverpassError",
    "Way",
    "build_bbox_query",
    "build_network_query",
    "build_query",
    "default_cache_dir",
    "fetch_corridor",
    "fetch_network",
    "load_cached",
    "parse_network",
    "parse_overpass",
    "query_cache_key",
]

OSM_ATTRIBUTION = "Road data (c) OpenStreetMap contributors, ODbL 1.0 (openstreetmap.org/copyright)"
DEFAULT_BUFFER_M = 1500.0
ENV_CACHE_DIR = "DRIFTMAPS_CACHE_DIR"
ENV_OVERPASS_URL = "DRIFTMAPS_OVERPASS_URL"  # comma-separated; replaces DEFAULT_MIRRORS
DEFAULT_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)


def _version() -> str:
    try:
        return importlib.metadata.version("driftmaps")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


USER_AGENT = f"driftmaps/{_version()} (Assetto Corsa track builder; Python urllib)"

DRIVABLE_HIGHWAY_RE = (
    "^((motorway|trunk|primary|secondary|tertiary)(_link)?"
    "|unclassified|residential|living_street|service)$"
)
PAVED_SURFACE_RE = "^(asphalt|paved|concrete.*|chipseal)$"
EXCLUDED_SERVICE_RE = "^(parking_aisle|driveway|drive-through|emergency_access)$"
# Overpass retry policy: HTTP codes worth retrying, and the pause after a 429 (fair-use policy).
RETRY_HTTP_CODES = (429, 500, 502, 503, 504)
RATE_LIMIT_PAUSE_S = 30.0
RACE_MIRRORS = 3  # mirrors raced per pass; the first good answer wins
# ASSUMED: a healthy mirror answers a small query in a few seconds; only then is the next asked.
RACE_STAGGER_S = 3.0


class OverpassError(RuntimeError):
    """Every mirror failed, or the response was unusable."""


@dataclass(frozen=True)
class Way:
    """One OSM way: node ids in order, their (lon, lat) degrees as an (N, 2) array, all tags."""

    id: int
    nodes: tuple[int, ...]
    lonlat: np.ndarray = field(repr=False)
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OsmData:
    """Parsed Overpass result. `ways` are sorted by id; `osm_base` is the data timestamp."""

    ways: tuple[Way, ...]
    osm_base: str | None = None
    cache_key: str | None = None

    def node_lonlat(self) -> dict[int, tuple[float, float]]:
        """Node id -> (lon, lat) over all ways (shared nodes agree)."""
        out: dict[int, tuple[float, float]] = {}
        for way in self.ways:
            for nid, (lon, lat) in zip(way.nodes, way.lonlat.tolist(), strict=True):
                out.setdefault(nid, (lon, lat))
        return out


def default_cache_dir() -> Path:
    """`$DRIFTMAPS_CACHE_DIR`, else `.cache/` in a dev checkout, else the user cache dir."""
    env = os.environ.get(ENV_CACHE_DIR)
    if env:
        return Path(env).expanduser()
    repo = Path(__file__).resolve().parents[3]
    if (repo / "pyproject.toml").is_file() and (repo / "src" / "driftmaps").is_dir():
        return repo / ".cache"
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "driftmaps" / "Cache"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "driftmaps"


def _fmt(value: float) -> str:
    return f"{value:.7f}".rstrip("0").rstrip(".")


def _way_statements(spatial: str, include_tracks: str) -> list[str]:
    """The `way[...]` statements for drivable roads (and tracks) with a spatial filter."""
    if include_tracks not in ("none", "paved", "all"):
        raise ValueError(f"include_tracks must be none/paved/all, got {include_tracks!r}")
    common = f'["area"!="yes"]["service"!~"{EXCLUDED_SERVICE_RE}"]{spatial}'
    parts = [f'way["highway"~"{DRIVABLE_HIGHWAY_RE}"]{common};']
    if include_tracks == "paved":
        parts.append(f'way["highway"="track"]["surface"~"{PAVED_SURFACE_RE}"]{common};')
    elif include_tracks == "all":
        parts.append(f'way["highway"="track"]{common};')
    return parts


def _union_query(parts: Sequence[str], timeout_s: int) -> str:
    return (
        f"[out:json][timeout:{int(timeout_s)}];\n(\n  "
        + "\n  ".join(parts)
        + "\n);\nout body geom;\n"
    )


def build_query(
    pins: Sequence[tuple[float, float]],
    buffer_m: float = DEFAULT_BUFFER_M,
    include_tracks: str = "paved",
    timeout_s: int = 120,
) -> str:
    """Overpass QL for drivable ways within `buffer_m` of the (lat, lon) pin chain.

    include_tracks: "none", "paved" (tracks with a paved `surface` tag) or "all".
    """
    if not pins:
        raise ValueError("need at least one pin")
    coords = ",".join(f"{_fmt(lat)},{_fmt(lon)}" for lat, lon in pins)
    around = f"(around:{_fmt(buffer_m)},{coords})"
    return _union_query(_way_statements(around, include_tracks), timeout_s)


def build_bbox_query(
    boxes: Sequence[tuple[float, float, float, float]],
    include_tracks: str = "paved",
    timeout_s: int = 60,
) -> str:
    """Overpass QL for drivable ways crossing any of the (south, west, north, east) boxes.

    Ways come back whole (`out body geom`), also the parts outside the boxes.
    """
    if not boxes:
        raise ValueError("need at least one box")
    parts: list[str] = []
    for s_, w, n, e in boxes:
        parts += _way_statements(f"({_fmt(s_)},{_fmt(w)},{_fmt(n)},{_fmt(e)})", include_tracks)
    return _union_query(parts, timeout_s)


def query_cache_key(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / "overpass" / f"{key}.json.gz"


def parse_overpass(payload: bytes | str | Mapping, cache_key: str | None = None) -> OsmData:
    """Overpass JSON (`out body geom`) -> OsmData. Ways without usable geometry are skipped."""
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
    ways: dict[int, Way] = {}
    for el in payload["elements"]:
        if el.get("type") != "way":
            continue
        nodes = el.get("nodes") or []
        geom = el.get("geometry") or []
        if len(nodes) < 2 or len(nodes) != len(geom) or any(g is None for g in geom):
            continue
        lonlat = np.array([[g["lon"], g["lat"]] for g in geom], dtype=np.float64)
        tags = {str(k): str(v) for k, v in (el.get("tags") or {}).items()}
        ways[int(el["id"])] = Way(int(el["id"]), tuple(int(n) for n in nodes), lonlat, tags)
    osm_base = (payload.get("osm3s") or {}).get("timestamp_osm_base")
    return OsmData(tuple(ways[k] for k in sorted(ways)), osm_base, cache_key)


def load_cached(cache_dir: Path, key: str) -> OsmData | None:
    path = _cache_path(cache_dir, key)
    if not path.is_file():
        return None
    try:
        return parse_overpass(gzip.decompress(path.read_bytes()), key)
    except (OSError, ValueError, OverpassError):
        return None  # corrupt cache entry: refetch


Opener = Callable[[urllib.request.Request, float], bytes]


def _urlopen(request: urllib.request.Request, timeout: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        body = resp.read()
        if (resp.headers.get("Content-Encoding") or "").strip().lower() == "gzip":
            body = gzip.decompress(body)  # asked for with Accept-Encoding (JSON shrinks ~10x)
        return body


def _mirrors(mirrors: Sequence[str] | None) -> list[str]:
    if mirrors:
        return list(mirrors)
    env = os.environ.get(ENV_OVERPASS_URL)
    if env:
        return [u.strip() for u in env.split(",") if u.strip()]
    return list(DEFAULT_MIRRORS)


def _request(url: str, data: bytes) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        },
        method="POST",
    )


def _race(
    urls: Sequence[str], data: bytes, opener: Opener, timeout_s: float, stagger_s: float = 0.0
) -> tuple[bytes | None, list[str], list[int | None]]:
    """Ask the urls (one thread each) and return the first good body, or None and the errors
    (with HTTP codes) once they have all failed. Late answers are ignored.

    With `stagger_s` > 0 they are asked one after the other: the next one when the last
    failed, or when no answer came within `stagger_s` (a hedged request; a healthy first
    mirror answers alone). With 0 they are all asked at once.
    """
    results: queue.Queue = queue.Queue()

    def ask(url: str) -> None:
        try:
            body = opener(_request(url, data), timeout_s)
            parse_overpass(body)  # reject HTML error pages / runtime errors early
            results.put((body, None, None))
        except urllib.error.HTTPError as exc:
            results.put((None, f"{url}: HTTP {exc.code}", exc.code))
        except Exception as exc:  # noqa: BLE001 - every failure is reported, none may hang us
            results.put((None, f"{url}: {exc}", None))

    errors: list[str] = []
    codes: list[int | None] = []
    if len(urls) == 1:
        ask(urls[0])
        body, err, code = results.get()
        return (body, errors, codes) if body is not None else (None, [err], [code])

    started = pending = 0

    def launch() -> None:
        nonlocal started, pending
        threading.Thread(target=ask, args=(urls[started],), daemon=True, name="overpass").start()
        started += 1
        pending += 1

    for _ in range(len(urls) if stagger_s <= 0 else 1):
        launch()
    deadline = time.monotonic() + 2.0 * timeout_s + 5.0  # urlopen's timeout is per read
    while pending:
        left = deadline - time.monotonic()
        hedge = started < len(urls)
        wait = min(left, stagger_s) if hedge else left
        try:
            body, err, code = results.get(timeout=max(0.05, wait))
        except queue.Empty:
            if hedge and time.monotonic() < deadline:
                launch()  # no answer yet: ask the next mirror too
                continue
            errors.append(f"no answer in {timeout_s:.0f} s")
            break
        pending -= 1
        if body is not None:
            return body, errors, codes
        errors.append(err)
        codes.append(code)
        if code == 400:
            break
        if started < len(urls):
            launch()
    return None, errors, codes


def post_query(
    query: str,
    mirrors: Sequence[str] | None = None,
    rounds: int = 3,
    backoff_s: float = 5.0,
    timeout_s: float = 180.0,
    opener: Opener = _urlopen,
    sleep: Callable[[float], None] = time.sleep,
    parallel: int = RACE_MIRRORS,
    stagger_s: float = RACE_STAGGER_S,
) -> bytes:
    """POST `query` to the mirrors for up to `rounds` passes; return the first good body.

    Each pass races the mirrors `parallel` at a time (threads; the first good answer wins),
    starting the next one when one fails or `stagger_s` passes without an answer.
    Between passes it waits backoff_s * 2**pass (and at least RATE_LIMIT_PAUSE_S after a 429).
    A 400 (bad query) is not retried. `opener` and `sleep` are injectable for tests.
    """
    urls = _mirrors(mirrors)
    data = urllib.parse.urlencode({"data": query}).encode("ascii")
    step = max(1, int(parallel))
    errors: list[str] = []
    for attempt in range(rounds):
        rate_limited = False
        for i in range(0, len(urls), step):
            body, errs, codes = _race(urls[i : i + step], data, opener, timeout_s, stagger_s)
            if body is not None:
                return body
            errors += errs
            if 400 in codes:
                raise OverpassError(f"Overpass rejected the query: {errs[-1]}")
            rate_limited |= 429 in codes
        if attempt + 1 < rounds:
            wait = backoff_s * 2**attempt
            sleep(max(wait, RATE_LIMIT_PAUSE_S) if rate_limited else wait)
    raise OverpassError("all Overpass mirrors failed:\n  " + "\n  ".join(errors))


def fetch_corridor(
    pins: Sequence[tuple[float, float]],
    cache_dir: Path | str | None = None,
    buffer_m: float = DEFAULT_BUFFER_M,
    include_tracks: str = "paved",
    mirrors: Sequence[str] | None = None,
    refresh: bool = False,
    offline: bool = False,
    **post_kwargs,
) -> OsmData:
    """Drivable ways near the (lat, lon) pin chain, from the cache or Overpass.

    A downloaded region database (`driftmaps.geo.offline`) that covers the corridor answers
    first. offline=True never touches the network (raises OverpassError on a cache miss);
    refresh=True ignores the region database and an existing cache entry. Extra kwargs go to
    `post_query`.
    """
    if not refresh:  # the downloaded region database, when it covers the corridor
        from driftmaps.geo import offline as region_db

        found = region_db.corridor(pins, buffer_m, include_tracks)
        if found is not None:
            return found
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    query = build_query(pins, buffer_m, include_tracks)
    key = query_cache_key(query)
    if not refresh:
        cached = load_cached(cache, key)
        if cached is not None:
            return cached
    if offline:
        raise OverpassError(f"no cached Overpass data for this corridor (key {key[:12]}...)")
    body = post_query(query, mirrors, **post_kwargs)
    osm = parse_overpass(body, key)
    path = _cache_path(cache, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(body, mtime=0))
    tmp.replace(path)
    return osm


# --------------------------------------------------------------------------------------------
# The road network around a route (side roads, driveways, roundabouts, junction control)
# --------------------------------------------------------------------------------------------

# Every highway a car can use, also driveways, parking aisles and farm tracks: the network
# stage (`geometry.network`) sorts them into side roads, driveways and what it leaves out.
NETWORK_HIGHWAY_RE = (
    "^((motorway|trunk|primary|secondary|tertiary)(_link)?"
    "|unclassified|residential|living_street|service|track|road|busway)$"
)
# Tagged nodes that shape the network: cul-de-sac bulbs and junction control.
NETWORK_NODE_RE = "^(turning_circle|turning_loop|mini_roundabout|stop|give_way|traffic_signals)$"
NETWORK_AROUND_M = 150.0  # = the scenery radius (buildings, near terrain)


@dataclass(frozen=True)
class OsmNode:
    """One tagged OSM node: (lon, lat) degrees and its tags."""

    id: int
    lonlat: tuple[float, float]
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NetworkData:
    """Highway ways (whole, with node ids) and tagged highway nodes near a route.

    `nodes_known` is False when the tagged nodes could not be fetched (the region database
    answered the ways but Overpass was out of reach): cul-de-sacs then fall back to rules.
    """

    ways: tuple[Way, ...] = ()
    nodes: tuple[OsmNode, ...] = ()
    osm_base: str | None = None
    nodes_known: bool = True


def _line_latlon(route_lonlat: np.ndarray, tolerance_m: float) -> np.ndarray:
    """The (lon, lat) line simplified to within `tolerance_m`, as rounded (lat, lon)."""
    import shapely

    from driftmaps.geo.project import LocalProjection

    ll = np.asarray(route_lonlat, np.float64).reshape(-1, 2)
    proj = LocalProjection(float(ll[0, 1]), float(ll[0, 0]))
    xy = proj.to_xy(ll)
    if len(xy) >= 2 and tolerance_m > 0:
        xy = np.asarray(shapely.LineString(xy).simplify(tolerance_m).coords)
    return np.round(proj.from_xy(xy)[:, ::-1], 6)


def build_network_query(
    route_latlon: npt.ArrayLike, around_m: float, *, ways: bool = True, timeout_s: int = 120
) -> str:
    """Overpass QL for every highway way (`NETWORK_HIGHWAY_RE`) and the tagged highway nodes
    (`NETWORK_NODE_RE`) within `around_m` of the (lat, lon) line; `ways=False`: nodes only."""
    pts = np.asarray(route_latlon, np.float64).reshape(-1, 2)
    if len(pts) == 0:
        raise ValueError("need at least one route point")
    coords = ",".join(f"{_fmt(lat)},{_fmt(lon)}" for lat, lon in pts)
    near = f"(around:{_fmt(around_m)},{coords})"
    parts = [f'node["highway"~"{NETWORK_NODE_RE}"]{near};']
    if ways:
        parts.insert(0, f'way["highway"~"{NETWORK_HIGHWAY_RE}"]["area"!="yes"]{near};')
    return _union_query(parts, timeout_s)


def parse_network(payload: bytes | str | Mapping, nodes_known: bool = True) -> NetworkData:
    """Overpass JSON (`build_network_query`) -> NetworkData (ways and nodes sorted by id)."""
    if isinstance(payload, bytes | str):
        try:
            payload = json.loads(payload)
        except ValueError as exc:
            raise OverpassError(f"not an Overpass JSON response: {exc}") from exc
    osm = parse_overpass(payload)
    nodes = {}
    for el in payload.get("elements", ()):
        if el.get("type") == "node" and "lat" in el and "lon" in el:
            tags = {str(k): str(v) for k, v in (el.get("tags") or {}).items()}
            nid = int(el["id"])
            nodes[nid] = OsmNode(nid, (float(el["lon"]), float(el["lat"])), tags)
    ways = tuple(w for w in osm.ways if w.tags.get("highway") and w.tags.get("area") != "yes")
    return NetworkData(ways, tuple(nodes[k] for k in sorted(nodes)), osm.osm_base, nodes_known)


def _cached_query(
    query: str,
    cache_dir: Path | str | None,
    offline: bool,
    mirrors: Sequence[str] | None,
    **post_kwargs,
) -> bytes:
    """The answer to `query` from the gzip cache, else from Overpass (then cached)."""
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    key = query_cache_key(query)
    path = _cache_path(cache, key)
    if path.is_file():
        try:
            return gzip.decompress(path.read_bytes())
        except OSError:
            pass  # corrupt cache entry: refetch
    if offline:
        raise OverpassError(f"no cached Overpass data for this query (key {key[:12]}...)")
    body = post_query(query, mirrors, **post_kwargs)
    parse_overpass(body)  # never cache an error page
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(body, mtime=0))
    tmp.replace(path)
    return body


def fetch_network(
    route_lonlat: npt.ArrayLike,
    *,
    around_m: float = NETWORK_AROUND_M,
    cache_dir: Path | str | None = None,
    simplify_m: float = 5.0,
    mirrors: Sequence[str] | None = None,
    offline: bool = False,
    **post_kwargs,
) -> NetworkData:
    """Every highway way and the tagged highway nodes within `around_m` of the route line
    (N, 2) (lon, lat).

    The ways come from the downloaded region database when it covers the route. It stores
    no nodes: those then come from the cache or one small Overpass query, and when that
    fails the result says `nodes_known=False`. Without a region database everything comes
    from the cache or one Overpass query. `offline=True` never touches the network
    (OverpassError on a cache miss)."""
    line = _line_latlon(np.asarray(route_lonlat, np.float64), simplify_m)
    radius = around_m + max(simplify_m, 0.0)
    from driftmaps.geo import offline as region_db

    lat, lon = line[:, 0], line[:, 1]
    pad = (radius + 200.0) / 111_320.0
    clip = (lat.min() - pad, lon.min() - 1.5 * pad, lat.max() + pad, lon.max() + 1.5 * pad)
    found = region_db.roadside_payload(line, 0.0, radius, tuple(float(v) for v in clip))
    if found is not None:
        ways = parse_network(found).ways
        base = (found.get("osm3s") or {}).get("timestamp_osm_base")
        try:
            query = build_network_query(line, radius, ways=False)
            nodes = parse_network(_cached_query(query, cache_dir, offline, mirrors, **post_kwargs))
        except (OverpassError, OSError):
            return NetworkData(ways, (), base, nodes_known=False)
        return NetworkData(ways, nodes.nodes, base)
    query = build_network_query(line, radius)
    return parse_network(_cached_query(query, cache_dir, offline, mirrors, **post_kwargs))
