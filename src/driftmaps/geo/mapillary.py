"""Street furniture seen in Mapillary's street-level photos: signs, poles, lights, hydrants.

Mapillary's `map_features` API (https://www.mapillary.com/developer/api-documentation) lists
the objects its computer vision found and triangulated from several photos: a position, the
class (`object_value`, e.g. `regulatory--stop--g1`, `object--support--utility-pole`), the
compass direction it faces (`aligned_direction`) and when it was first and last seen.

- **Token.** The API needs a client token (`MLY|...`, free at
  https://mapillary.com/dashboard/developers): the user's own (the app's settings, or
  `$DRIFTMAPS_MAPILLARY_TOKEN`), else the app's built-in `DEFAULT_CLIENT_TOKEN`
  (`resolve_token`). Without one this layer is skipped. The token is sent in the
  `Authorization` header, never in a URL, a cache key, a log line or an error message.
- **Requests.** One request per 0.01 x 0.01 degree tile along the route (the API wants boxes
  under 0.01 deg^2 and returns at most 2000 features; 0.05 degree boxes gave HTTP 500 on
  2026-10-16), aligned to a 0.01 degree grid so neighbouring routes share cached tiles. A tile
  that hits the 2000 limit is split in four. Responses are kept gzipped under
  `<cache>/mapillary/<sha256(box, fields)>.json.gz`.
- **Stale features.** A feature not seen again by a newer photo pass nearby has probably been
  removed: features last seen more than `STALE_YEARS` before the newest feature within
  `STALE_RADIUS_M` are dropped, and so is anything last seen before `MIN_LAST_SEEN`.
- **Licence.** Mapillary's terms (https://www.mapillary.com/terms, effective 15 Feb 2024):
  images are CC BY-SA 4.0 (§3(b)); data extracted from them may be used in a product that
  materially supplements it, with the Mapillary logo and a link to mapillary.com (§11). The
  track credits `MAPILLARY_ATTRIBUTION` in LICENSE-DATA.txt, and the app shows the credit
  with a link. The positions derived here are shared under CC BY-SA 4.0.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import logging
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from driftmaps.geo.overpass import USER_AGENT, default_cache_dir
from driftmaps.geo.project import LocalProjection

log = logging.getLogger(__name__)

__all__ = [
    "ENV_TOKEN",
    "MAPILLARY_ATTRIBUTION",
    "MAPILLARY_URL",
    "MapFeature",
    "MapillaryError",
    "drop_stale",
    "fetch_image_points",
    "fetch_map_features",
    "parse_image_points",
    "parse_map_features",
    "tiles_along",
    "token_from_env",
    "default_token",
    "resolve_token",
    "MapillaryRevoked",
    "take_revoked_notice",
]

ENV_TOKEN = "DRIFTMAPS_MAPILLARY_TOKEN"
# Built-in Mapillary access, so the app's users get Mapillary's signs and poles with no setup
# (the repository owner chose to ship it). It is a public, read-only Mapillary *client* token
# (MLY|...): Mapillary designs client tokens to be embedded in apps, and this one can only read
# public map data. Its per-app rate limits are shared by everyone using the app. A user's own
# token (app settings, or $DRIFTMAPS_MAPILLARY_TOKEN) always wins, and the app's "Use Mapillary"
# switch turns the layer off. If Mapillary ever revokes it (HTTP 401/403), builds warn once and
# carry on with OpenStreetMap and Ontario's road rules. Read it only through `default_token()`
# (the tests blank it, so they never use it).
DEFAULT_CLIENT_TOKEN = "MLY|29514967621437915|046ac295251d9fcabb7983504e2c1308"
REVOKED_MESSAGE = (
    "The built-in Mapillary access was revoked; paste your own Mapillary token in the app's "
    "settings to get Mapillary's signs and poles again"
)
API_URL = "https://graph.mapillary.com/map_features"
IMAGES_URL = "https://graph.mapillary.com/images"
IMAGE_FIELDS = "id,computed_geometry,captured_at"
MAPILLARY_URL = "https://www.mapillary.com"
MAPILLARY_ATTRIBUTION = (
    "Street furniture positions (signs, poles, lights): Mapillary (mapillary.com), "
    "detected in street-level imagery by Mapillary contributors, CC BY-SA 4.0 "
    "(https://creativecommons.org/licenses/by-sa/4.0/); Mapillary terms: "
    "https://www.mapillary.com/terms"
)
FIELDS = "id,object_value,geometry,aligned_direction,first_seen_at,last_seen_at"
TILE_DEG = 0.01  # tiles of 0.01 x 0.01 degrees (the API wants < 0.01 deg^2)
MIN_TILE_DEG = 0.0025  # a crowded tile is split down to this size
LIMIT = 2000  # the API's per-request maximum
STALE_YEARS = 3.0
STALE_RADIUS_M = 150.0
MIN_LAST_SEEN = "2013-01-01"  # Mapillary started in 2013; anything older is bad data
WORKERS = 4
_SECONDS_PER_YEAR = 365.25 * 86400.0


class MapillaryError(RuntimeError):
    """The Mapillary API could not be used (no token, refused token, network)."""


class MapillaryRevoked(MapillaryError):
    """Mapillary refused the built-in token (HTTP 401/403): it has been revoked."""


@dataclass(frozen=True)
class MapFeature:
    """One Mapillary map feature: class, (lon, lat), facing (compass degrees, None if
    unknown) and the first/last time a photo showed it (ISO 8601)."""

    id: str
    value: str
    lon: float
    lat: float
    aligned_direction: float | None = None
    first_seen: str = ""
    last_seen: str = ""


def token_from_env() -> str | None:
    tok = os.environ.get(ENV_TOKEN, "").strip()
    return tok or None


_revoked = {"builtin": False, "noticed": False}
_revoked_lock = threading.Lock()


def default_token() -> str | None:
    """The built-in token, or None once Mapillary has refused it in this run."""
    with _revoked_lock:
        if _revoked["builtin"]:
            return None
    return DEFAULT_CLIENT_TOKEN.strip() or None


def resolve_token(own: str | None = None) -> tuple[str | None, str]:
    """`(token, source)`: the user's own token, else `$DRIFTMAPS_MAPILLARY_TOKEN`, else the
    built-in one; source is "own", "env", "built-in" or "none"."""
    own = (own or "").strip()
    if own:
        return own, "own"
    env = token_from_env()
    if env:
        return env, "env"
    builtin = default_token()
    return (builtin, "built-in") if builtin else (None, "none")


def _revoke_builtin() -> None:
    with _revoked_lock:
        if not _revoked["builtin"]:
            _revoked["builtin"] = True
            log.warning(REVOKED_MESSAGE)


def take_revoked_notice() -> bool:
    """True exactly once after the built-in token was refused (the build warns then)."""
    with _revoked_lock:
        if _revoked["builtin"] and not _revoked["noticed"]:
            _revoked["noticed"] = True
            return True
        return False


def reset_builtin_state() -> None:
    """Forget a refusal of the built-in token (tests)."""
    with _revoked_lock:
        _revoked.update(builtin=False, noticed=False)


def _safe(text: str, token: str) -> str:
    """`text` with the token masked (error messages may echo a request)."""
    return text.replace(token, "MLY|***") if token else text


def tiles_along(
    route_lonlat: np.ndarray, pad_m: float, tile_deg: float = TILE_DEG
) -> list[tuple[float, float, float, float]]:
    """The (west, south, east, north) grid tiles within ~`pad_m` of the (N, 2) lon/lat line,
    on a grid aligned to multiples of `tile_deg`, sorted."""
    ll = np.asarray(route_lonlat, np.float64).reshape(-1, 2)
    if len(ll) == 0:
        return []
    # Densify to ~25 m so no tile the line crosses is skipped.
    proj = LocalProjection(float(ll[0, 1]), float(ll[0, 0]))
    xy = proj.to_xy(ll)
    pts = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:], strict=True):
        n = max(1, math.ceil(float(np.hypot(*(b - a))) / 25.0))
        pts.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
    dense = proj.from_xy(np.concatenate(pts))
    dlat = pad_m / 111_320.0
    dlon = dlat / max(math.cos(math.radians(float(ll[:, 1].mean()))), 0.05)
    keys: set[tuple[int, int]] = set()
    for sx in (-dlon, 0.0, dlon):
        for sy in (-dlat, 0.0, dlat):
            ix = np.floor((dense[:, 0] + sx) / tile_deg).astype(np.int64)
            iy = np.floor((dense[:, 1] + sy) / tile_deg).astype(np.int64)
            keys.update(zip(ix.tolist(), iy.tolist(), strict=True))
    return [
        (round(x * tile_deg, 6), round(y * tile_deg, 6),
         round((x + 1) * tile_deg, 6), round((y + 1) * tile_deg, 6))
        for x, y in sorted(keys)
    ]  # fmt: skip


Endpoint = tuple[str, str]  # (URL, fields)
FEATURES: Endpoint = (API_URL, FIELDS)
IMAGES: Endpoint = (IMAGES_URL, IMAGE_FIELDS)


def _cache_path(cache_dir: Path, box: tuple[float, float, float, float], ep: Endpoint) -> Path:
    url, fields = ep
    key = hashlib.sha256(f"{url}|{fields}|{box}".encode()).hexdigest()
    return Path(cache_dir) / "mapillary" / f"{key}.json.gz"


Opener = Callable[[urllib.request.Request, float], bytes]


def _urlopen(request: urllib.request.Request, timeout: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        body = resp.read()
        if (resp.headers.get("Content-Encoding") or "").strip().lower() == "gzip":
            body = gzip.decompress(body)
        return body


def _get(
    box: tuple[float, float, float, float],
    token: str,
    opener: Opener,
    timeout_s: float,
    ep: Endpoint = FEATURES,
) -> bytes:
    url, fields = ep
    params = {"bbox": ",".join(f"{v:.6f}" for v in box), "fields": fields, "limit": str(LIMIT)}
    req = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}",
        headers={
            "Authorization": f"OAuth {token}",
            "Accept-Encoding": "gzip",
            "User-Agent": USER_AGENT,
        },
    )
    err: Exception | None = None
    for attempt in range(3):
        try:
            return opener(req, timeout_s)
        except urllib.error.HTTPError as e:
            builtin = DEFAULT_CLIENT_TOKEN.strip()
            if e.code in (401, 403) and builtin and token == builtin:
                _revoke_builtin()
                raise MapillaryRevoked(REVOKED_MESSAGE) from None
            if e.code in (400, 401, 403):
                hint = " (check the Mapillary token)" if e.code in (401, 403) else ""
                raise MapillaryError(
                    f"Mapillary refused the request: HTTP {e.code}{hint}"
                ) from None
            err = e
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            err = e
        time.sleep(1.5 * (attempt + 1))
    raise MapillaryError(f"Mapillary unreachable: {_safe(str(err), token)[:160]}")


def _tile(
    box: tuple[float, float, float, float],
    token: str | None,
    cache: Path,
    opener: Opener,
    offline: bool,
    refresh: bool,
    timeout_s: float,
    ep: Endpoint = FEATURES,
) -> list[dict]:
    """Raw feature (or image) dicts in `box` (from the cache or the API; crowded boxes
    split)."""
    path = _cache_path(cache, box, ep)
    data = None
    if not refresh and path.is_file():
        try:
            data = json.loads(gzip.decompress(path.read_bytes()))
        except (OSError, ValueError):
            data = None
    if data is None:
        if offline or not token:
            raise MapillaryError("no cached Mapillary data for this area")
        body = _get(box, token, opener, timeout_s, ep)
        try:
            data = json.loads(body)
        except ValueError as e:
            raise MapillaryError(f"Mapillary sent an unreadable answer: {e}") from None
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise MapillaryError("Mapillary sent an unexpected answer (no 'data')")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(gzip.compress(json.dumps(data).encode("utf-8"), mtime=0))
        tmp.replace(path)
    rows = list(data.get("data") or [])
    w, s, e, n = box
    if len(rows) >= LIMIT and (e - w) / 2 >= MIN_TILE_DEG - 1e-9:
        mx, my = round((w + e) / 2, 6), round((s + n) / 2, 6)
        rows = []
        for sub in ((w, s, mx, my), (mx, s, e, my), (w, my, mx, n), (mx, my, e, n)):
            rows += _tile(sub, token, cache, opener, offline, refresh, timeout_s, ep)
    return rows


def parse_map_features(rows: Sequence[dict]) -> list[MapFeature]:
    """MapFeatures from API rows (bad rows skipped, duplicates by id dropped), sorted by id."""
    out: dict[str, MapFeature] = {}
    for r in rows:
        try:
            lon, lat = r["geometry"]["coordinates"][:2]
            fid, value = str(r["id"]), str(r["object_value"])
        except (KeyError, TypeError, ValueError):
            continue
        ad = r.get("aligned_direction")
        try:
            ad = float(ad) % 360.0 if ad is not None else None
        except (TypeError, ValueError):
            ad = None
        if ad is not None and not math.isfinite(ad):
            ad = None
        out[fid] = MapFeature(
            fid, value, float(lon), float(lat), ad,
            str(r.get("first_seen_at") or ""), str(r.get("last_seen_at") or ""),
        )  # fmt: skip
    return [out[k] for k in sorted(out)]


def _epoch(stamp: str) -> float | None:
    if not stamp:
        return None
    try:
        dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).timestamp()


def drop_stale(
    features: Sequence[MapFeature],
    *,
    stale_years: float = STALE_YEARS,
    radius_m: float = STALE_RADIUS_M,
    min_last_seen: str = MIN_LAST_SEEN,
) -> list[MapFeature]:
    """Features still standing as far as the photos tell: drop those last seen before
    `min_last_seen`, or more than `stale_years` before the newest feature within `radius_m`
    (a newer photo pass went by and did not see them again)."""
    if not features:
        return []
    floor = _epoch(min_last_seen) or 0.0
    t = np.array([_epoch(f.last_seen) or np.nan for f in features])
    ll = np.array([(f.lon, f.lat) for f in features])
    proj = LocalProjection(float(ll[:, 1].mean()), float(ll[:, 0].mean()))
    xy = proj.to_xy(ll)
    from scipy.spatial import cKDTree

    tree = cKDTree(xy)
    keep = []
    gap = stale_years * _SECONDS_PER_YEAR
    for i, f in enumerate(features):
        if not np.isfinite(t[i]):
            keep.append(f)  # no date: keep
            continue
        if t[i] < floor:
            continue
        near = tree.query_ball_point(xy[i], radius_m)
        newest = np.nanmax(t[near]) if near else t[i]
        if t[i] >= newest - gap:
            keep.append(f)
    return keep


def fetch_map_features(
    route_lonlat: np.ndarray,
    *,
    token: str | None = None,
    pad_m: float = 60.0,
    cache_dir: Path | str | None = None,
    offline: bool = False,
    refresh: bool = False,
    opener: Opener = _urlopen,
    timeout_s: float = 30.0,
    workers: int = WORKERS,
) -> list[MapFeature]:
    """Mapillary map features in the tiles within `pad_m` of the (N, 2) lon/lat route line
    (the caller keeps what lies near the road). Cached tiles are used without a token;
    MapillaryError if a tile is neither cached nor fetchable."""
    rows = _fetch(route_lonlat, FEATURES, token, pad_m, cache_dir, offline, refresh, opener,
                  timeout_s, workers)  # fmt: skip
    return parse_map_features(rows)


def _fetch(route_lonlat, ep, token, pad_m, cache_dir, offline, refresh, opener, timeout_s, workers):
    token = (token or resolve_token()[0] or "").strip() or None
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    boxes = tiles_along(route_lonlat, pad_m)
    rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        futures = [
            pool.submit(_tile, b, token, cache, opener, offline, refresh, timeout_s, ep)
            for b in boxes
        ]
        for fut in futures:
            rows += fut.result()
    return rows


def parse_image_points(rows: Sequence[dict]) -> np.ndarray:
    """(N, 2) lon/lat of the photos (computed positions), duplicates by id dropped."""
    seen: dict[str, tuple[float, float]] = {}
    for r in rows:
        try:
            lon, lat = (r.get("computed_geometry") or r["geometry"])["coordinates"][:2]
            seen[str(r["id"])] = (float(lon), float(lat))
        except (KeyError, TypeError, ValueError):
            continue
    return np.array([seen[k] for k in sorted(seen)], np.float64).reshape(-1, 2)


def fetch_image_points(
    route_lonlat: np.ndarray,
    *,
    token: str | None = None,
    pad_m: float = 60.0,
    cache_dir: Path | str | None = None,
    offline: bool = False,
    refresh: bool = False,
    opener: Opener = _urlopen,
    timeout_s: float = 30.0,
    workers: int = WORKERS,
) -> np.ndarray:
    """Where Mapillary has photos near the route: (N, 2) lon/lat of every image in the same
    tiles as `fetch_map_features` (a crowded tile is split, so at most a few thousand per
    tile). Rule-based furniture is added only where there are none: a stretch with photos
    shows what is really there."""
    rows = _fetch(route_lonlat, IMAGES, token, pad_m, cache_dir, offline, refresh, opener,
                  timeout_s, workers)  # fmt: skip
    return parse_image_points(rows)
