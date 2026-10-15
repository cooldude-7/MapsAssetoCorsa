"""Fetch drivable OSM ways in a corridor around the pins from the Overpass API, with a disk cache.

Design (docs/research/geo.md §1.2, PLAN §5.1):
- One query per corridor: every `highway` way of a drivable class within `buffer_m` of the
  straight pin chain, using Overpass' linestring `around:` filter (since 0.7.55, VERIFIED in
  geo.md §1.2). `out body geom` returns each way's node ids (topology) and coordinates.
- Paved `highway=track` ways are included by default; service roads that are parking aisles,
  driveways or drive-throughs are left out so routes can't cut through car parks.
- Mirrors are tried in order with exponential backoff. Overpass answers 429 when rate limited
  and 504 when overloaded (VERIFIED, geo.md §1.1); its fair-use policy asks for a 30 s pause
  after a 429 and an identifying User-Agent. On 2026-10-15 overpass-api.de reset connections
  from the dev sandbox while maps.mail.ru answered, so the mirror list matters.
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
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "DEFAULT_BUFFER_M",
    "DEFAULT_MIRRORS",
    "ENV_CACHE_DIR",
    "ENV_OVERPASS_URL",
    "OSM_ATTRIBUTION",
    "OsmData",
    "OverpassError",
    "Way",
    "build_query",
    "default_cache_dir",
    "fetch_corridor",
    "load_cached",
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
    if include_tracks not in ("none", "paved", "all"):
        raise ValueError(f"include_tracks must be none/paved/all, got {include_tracks!r}")
    coords = ",".join(f"{_fmt(lat)},{_fmt(lon)}" for lat, lon in pins)
    around = f"(around:{_fmt(buffer_m)},{coords})"
    common = f'["area"!="yes"]["service"!~"{EXCLUDED_SERVICE_RE}"]{around}'
    parts = [f'way["highway"~"{DRIVABLE_HIGHWAY_RE}"]{common};']
    if include_tracks == "paved":
        parts.append(f'way["highway"="track"]["surface"~"{PAVED_SURFACE_RE}"]{common};')
    elif include_tracks == "all":
        parts.append(f'way["highway"="track"]{common};')
    return (
        f"[out:json][timeout:{int(timeout_s)}];\n(\n  "
        + "\n  ".join(parts)
        + "\n);\nout body geom;\n"
    )


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
        return resp.read()


def _mirrors(mirrors: Sequence[str] | None) -> list[str]:
    if mirrors:
        return list(mirrors)
    env = os.environ.get(ENV_OVERPASS_URL)
    if env:
        return [u.strip() for u in env.split(",") if u.strip()]
    return list(DEFAULT_MIRRORS)


def post_query(
    query: str,
    mirrors: Sequence[str] | None = None,
    rounds: int = 3,
    backoff_s: float = 5.0,
    timeout_s: float = 180.0,
    opener: Opener = _urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """POST `query` to each mirror in turn for up to `rounds` passes; return the response body.

    Between passes it waits backoff_s * 2**pass (and at least RATE_LIMIT_PAUSE_S after a 429).
    A 400 (bad query) is not retried. `opener` and `sleep` are injectable for tests.
    """
    urls = _mirrors(mirrors)
    data = urllib.parse.urlencode({"data": query}).encode("ascii")
    errors: list[str] = []
    for attempt in range(rounds):
        rate_limited = False
        for url in urls:
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "User-Agent": USER_AGENT,
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                body = opener(req, timeout_s)
                parse_overpass(body)  # reject HTML error pages / runtime errors early
                return body
            except urllib.error.HTTPError as exc:
                errors.append(f"{url}: HTTP {exc.code}")
                if exc.code == 400:
                    raise OverpassError(f"Overpass rejected the query: {exc}") from exc
                rate_limited |= exc.code == 429
            except (urllib.error.URLError, OSError, ValueError, OverpassError) as exc:
                errors.append(f"{url}: {exc}")
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

    offline=True never touches the network (raises OverpassError on a cache miss);
    refresh=True ignores an existing cache entry. Extra kwargs go to `post_query`.
    """
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
