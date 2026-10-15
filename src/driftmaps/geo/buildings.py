"""Building footprints near the route: OpenStreetMap first, Microsoft ML footprints to fill gaps.

docs/research/scenery.md §3 and §7 ("Buildings"):

- **OSM** `building=*` ways (and multipolygon relations) within `around_m` (150 m) of the route
  line: one Overpass query over a union of small bounding boxes along the line (one per km;
  `around:` along a long line timed out on the public mirrors), then filtered by distance,
  cached like `geo.overpass` (same mirrors and retry policy,
  `<cache>/overpass/<sha256(query)>.json.gz`). On 2026-10-15 OSM
  had ~100 buildings within 150 m of Snake Rd and of Sydenham Rd but only 6 at Forks of the
  Credit, and none carried `height` or `building:levels`.
- **Microsoft Global ML Building Footprints** (CDLA-Permissive-2.0, https://cdla.dev/permissive-2-0/;
  not ODbL) fill the gaps (55 at Forks, most with an ML height). Access, VERIFIED 2026-10-15:
  `dataset-links.csv` (columns `Location, QuadKey, Url, Size, UploadDate`) lists one gzipped
  GeoJSON-lines file per zoom-9 Bing quadkey; each line is a `Feature` with a `Polygon` and
  properties `height` (m, -1 = unknown) and `confidence`. Tiles are 30-40 MB, so they are
  streamed (never stored whole), filtered to the route bbox, and only that subset is cached
  (`<cache>/msbuildings/<sha256>.json.gz`; the links file is cached once per release). Streaming
  the 332k-feature Hamilton tile took ~4 s here.
- **Merge rule** (scenery.md §3, ASSUMED): every OSM footprint is kept; a Microsoft footprint is
  added only if no OSM footprint covers more than `max_overlap` (30 %) of its area.

Everything here is lon/lat; `geometry.buildings` projects, filters against the road corridor
and extrudes. Attribution lines for LICENSE-DATA.txt come back in `BuildingFootprints.attributions`.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import logging
import math
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import numpy as np
import numpy.typing as npt
import shapely

from driftmaps.geo.attributes import parse_width
from driftmaps.geo.overpass import (
    OSM_ATTRIBUTION,
    USER_AGENT,
    OverpassError,
    default_cache_dir,
    post_query,
    query_cache_key,
)
from driftmaps.geo.project import LocalProjection

__all__ = [
    "DEFAULT_AROUND_M",
    "MS_ATTRIBUTION",
    "MS_LINKS_URL",
    "MS_ZOOM",
    "BuildingFootprints",
    "Footprint",
    "build_buildings_query",
    "fetch_buildings",
    "fetch_ms_buildings",
    "fetch_osm_buildings",
    "merge_footprints",
    "ms_tile_urls",
    "parse_buildings",
    "parse_ms_features",
    "quadkey",
    "quadkeys_for_bbox",
    "route_boxes",
]

log = logging.getLogger(__name__)

DEFAULT_AROUND_M = 150.0
MS_RELEASE = "2026-08-13"
MS_LINKS_URL = f"https://bfppub.blob.core.windows.net/%24web/{MS_RELEASE}/dataset-links.csv"
MS_ZOOM = 9  # the dataset's quadkey level
MS_ATTRIBUTION = (
    "Building footprints: Microsoft Global ML Building Footprints "
    f"(release {MS_RELEASE}), licensed under the Community Data License Agreement - "
    "Permissive, Version 2.0 (CDLA-Permissive-2.0, https://cdla.dev/permissive-2-0/)"
)
OSM_BUILDINGS_ATTRIBUTION = OSM_ATTRIBUTION.replace("Road data", "Road and building data")

# Opens a URL for streaming: returns a binary file object (closed by the caller).
StreamOpener = Callable[[str], IO[bytes]]


@dataclass(frozen=True)
class Footprint:
    """One building outline: `lonlat` (N, 2) is the closed exterior ring (first == last).

    `id` is "osm:w<id>", "osm:r<id>.<k>" (k-th outer ring of a relation) or "ms:<quadkey>:<n>".
    `tags` are the OSM tags (empty for Microsoft); `ms_height_m` the Microsoft ML height
    (None when unknown). Holes (courtyards) are not kept: the building is drawn solid.
    """

    id: str
    source: str  # "osm" | "microsoft"
    lonlat: np.ndarray = field(repr=False)
    tags: Mapping[str, str] = field(default_factory=dict)
    ms_height_m: float | None = None

    @property
    def kind(self) -> str:
        """The `building` tag value ("yes" when untagged or from Microsoft)."""
        return self.tags.get("building", "yes")

    @property
    def height_m(self) -> float | None:
        """The OSM `height` tag in metres (top of the roof), if any."""
        h = parse_width(self.tags.get("height"))
        return h if h is not None and h > 0 else None

    @property
    def levels(self) -> float | None:
        """OSM `building:levels` (storeys above ground, roof storeys excluded), if any."""
        try:
            v = float(str(self.tags.get("building:levels", "")).split(";")[0])
        except ValueError:
            return None
        return v if math.isfinite(v) and v > 0 else None

    @property
    def roof_shape(self) -> str | None:
        return self.tags.get("roof:shape")


@dataclass
class BuildingFootprints:
    """What `fetch_buildings` found: the merged footprints and where they came from."""

    footprints: list[Footprint]
    osm_count: int = 0
    ms_count: int = 0  # Microsoft footprints added by the merge
    osm_base: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def attributions(self) -> list[str]:
        """LICENSE-DATA.txt lines for the sources that contributed footprints."""
        out = []
        if self.osm_count:
            out.append(OSM_BUILDINGS_ATTRIBUTION)
        if self.ms_count:
            out.append(MS_ATTRIBUTION)
        return out


# --------------------------------------------------------------------------------------------
# OSM (Overpass)
# --------------------------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def route_boxes(
    route_lonlat: npt.ArrayLike, pad_m: float, piece_m: float = 1000.0
) -> list[tuple[float, float, float, float]]:
    """(south, west, north, east) boxes covering the (N, 2) lon/lat route line grown by
    `pad_m`: one per ~`piece_m` of line, so a winding route does not pull in a whole town.
    Rounded outwards to 1e-5 degrees (stable cache keys)."""
    ll = np.asarray(route_lonlat, dtype=np.float64).reshape(-1, 2)
    if len(ll) == 0:
        raise ValueError("need at least one route point")
    proj = LocalProjection(float(ll[0, 1]), float(ll[0, 0]))
    xy = proj.to_xy(ll)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    count = max(1, math.ceil(s[-1] / piece_m))
    cuts = np.searchsorted(s, np.linspace(0.0, s[-1], count + 1))
    cuts[0], cuts[-1] = 0, len(xy) - 1
    boxes = []
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        part = xy[a : max(b, a) + 1]
        (x0, y0), (x1, y1) = part.min(axis=0) - pad_m, part.max(axis=0) + pad_m
        corners = proj.from_xy(np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]]))
        w, so = corners.min(axis=0)
        e, n = corners.max(axis=0)
        box = (math.floor(so * 1e5) / 1e5, math.floor(w * 1e5) / 1e5,
               math.ceil(n * 1e5) / 1e5, math.ceil(e * 1e5) / 1e5)
        if box not in boxes:
            boxes.append(box)
    return boxes


def build_buildings_query(
    boxes: Sequence[tuple[float, float, float, float]], timeout_s: int = 120
) -> str:
    """Overpass QL for building ways and multipolygon relations in the (south, west, north,
    east) boxes. Bounding boxes are index lookups, much cheaper for the server than an
    `around:` along a long line (which timed out on the public mirrors, 2026-10-15)."""
    if not boxes:
        raise ValueError("need at least one box")
    parts = []
    for s_, w, n, e in boxes:
        bb = f"({_fmt(s_)},{_fmt(w)},{_fmt(n)},{_fmt(e)})"
        parts.append(f'  way["building"]{bb};\n  relation["building"]["type"="multipolygon"]{bb};')
    head = f"[out:json][timeout:{int(timeout_s)}];\n(\n"
    return head + "\n".join(parts) + "\n);\nout body geom;\n"


def _ring(geom: Iterable[Mapping | None]) -> np.ndarray | None:
    pts = [(g["lon"], g["lat"]) for g in geom if g is not None]
    if len(pts) < 4:
        return None
    arr = np.asarray(pts, np.float64)
    if not np.allclose(arr[0], arr[-1]):
        return None
    return arr


def _relation_rings(members: Iterable[Mapping]) -> list[np.ndarray]:
    """Closed outer rings of a multipolygon relation (outer ways joined end to end)."""
    lines = []
    for m in members:
        if m.get("type") != "way" or m.get("role", "outer") not in ("outer", ""):
            continue
        pts = [(g["lon"], g["lat"]) for g in (m.get("geometry") or []) if g is not None]
        if len(pts) >= 2:
            lines.append(shapely.LineString(pts))
    if not lines:
        return []
    polys = shapely.get_parts(shapely.polygonize(lines))
    return [np.asarray(p.exterior.coords) for p in polys if p.area > 0]


def parse_buildings(payload: bytes | str | Mapping) -> list[Footprint]:
    """Building footprints in an Overpass JSON (`out body geom`) response, sorted by id.

    Closed `building=*` ways become one footprint each; multipolygon relations one per outer
    ring (inner rings are dropped). Elements without usable geometry are skipped. Other
    elements in the response (e.g. roads) are ignored.
    """
    if isinstance(payload, bytes | str):
        try:
            payload = json.loads(payload)
        except ValueError as exc:
            raise OverpassError(f"not an Overpass JSON response: {exc}") from exc
    if not isinstance(payload, Mapping) or "elements" not in payload:
        raise OverpassError("not an Overpass JSON response (no 'elements')")
    out: list[tuple[tuple[int, int, int], Footprint]] = []
    for el in payload["elements"]:
        tags = {str(k): str(v) for k, v in (el.get("tags") or {}).items()}
        if "building" not in tags or tags.get("building") == "no":
            continue
        oid = int(el["id"])
        if el.get("type") == "way":
            ring = _ring(el.get("geometry") or [])
            if ring is not None:
                out.append(((0, oid, 0), Footprint(f"osm:w{oid}", "osm", ring, tags)))
        elif el.get("type") == "relation":
            for k, ring in enumerate(_relation_rings(el.get("members") or [])):
                out.append(((1, oid, k), Footprint(f"osm:r{oid}.{k}", "osm", ring, tags)))
    return [f for _, f in sorted(out, key=lambda t: t[0])]


def _read_gz_json(path: Path) -> object | None:
    try:
        return json.loads(gzip.decompress(path.read_bytes()))
    except (OSError, ValueError, EOFError):
        return None  # missing or corrupt: refetch


def _write_gz(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(gzip.compress(data, mtime=0))
    tmp.replace(path)


def fetch_osm_buildings(
    route_lonlat: npt.ArrayLike,
    *,
    cache_dir: Path | str | None = None,
    around_m: float = DEFAULT_AROUND_M,
    piece_m: float = 1000.0,
    mirrors: Sequence[str] | None = None,
    refresh: bool = False,
    offline: bool = False,
    **post_kwargs,
) -> tuple[list[Footprint], str | None]:
    """(OSM building footprints near the (N, 2) lon/lat route line, osm_base timestamp).

    One query for the union of `route_boxes` (each ~`piece_m` of line grown by `around_m` +
    5 m); `fetch_buildings` then keeps footprints within `around_m` of the line. `offline=True`
    never touches the network (OverpassError on a cache miss). Extra kwargs go to
    `overpass.post_query`.
    """
    ll = np.asarray(route_lonlat, dtype=np.float64).reshape(-1, 2)
    query = build_buildings_query(route_boxes(ll, around_m + 5.0, piece_m))
    key = query_cache_key(query)
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    path = cache / "overpass" / f"{key}.json.gz"
    payload = None if refresh else _read_gz_json(path)
    if payload is None:
        if offline:
            raise OverpassError(f"no cached Overpass building data (key {key[:12]}...)")
        body = post_query(query, mirrors, **post_kwargs)
        payload = json.loads(body)
        _write_gz(path, body)
    osm_base = ((payload or {}).get("osm3s") or {}).get("timestamp_osm_base")
    return parse_buildings(payload), osm_base


# --------------------------------------------------------------------------------------------
# Microsoft Global ML Building Footprints
# --------------------------------------------------------------------------------------------


def _tile_xy(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """Web-Mercator (Bing) tile of a point."""
    lat = max(-85.05112878, min(85.05112878, lat))
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    s = math.sin(math.radians(lat))
    y = int((0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def quadkey(x: int, y: int, zoom: int) -> str:
    """Bing Maps quadkey of tile (x, y) at `zoom` (one base-4 digit per level)."""
    digits = []
    for i in range(zoom, 0, -1):
        mask = 1 << (i - 1)
        digits.append(str((1 if x & mask else 0) + (2 if y & mask else 0)))
    return "".join(digits)


def quadkeys_for_bbox(
    west: float, south: float, east: float, north: float, zoom: int = MS_ZOOM
) -> list[str]:
    """Quadkeys of every tile the lon/lat bbox touches, sorted."""
    x0, y1 = _tile_xy(west, south, zoom)
    x1, y0 = _tile_xy(east, north, zoom)
    return sorted(
        quadkey(x, y, zoom) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)
    )


def _open_url(url: str) -> IO[bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=300)


def _links_csv(cache: Path, opener: StreamOpener, refresh: bool, offline: bool) -> bytes:
    path = cache / "msbuildings" / f"dataset-links-{MS_RELEASE}.csv.gz"
    if not refresh and path.is_file():
        try:
            return gzip.decompress(path.read_bytes())
        except (OSError, EOFError):
            pass
    if offline:
        raise OSError("Microsoft dataset-links.csv is not cached (offline)")
    with opener(MS_LINKS_URL) as fh:
        data = fh.read()
    _write_gz(path, data)
    return data


def ms_tile_urls(links_csv: bytes | str, quadkeys: Iterable[str]) -> list[tuple[str, str]]:
    """(quadkey, url) rows of `dataset-links.csv` for the wanted quadkeys (every Location: a
    tile on a border is listed once per country), in file order."""
    text = links_csv.decode("utf-8") if isinstance(links_csv, bytes) else links_csv
    wanted = set(quadkeys)
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        qk = (row.get("QuadKey") or "").strip()
        if qk in wanted and row.get("Url"):
            out.append((qk, row["Url"].strip()))
    return out


def parse_ms_features(
    lines: Iterable[bytes | str],
    bbox: tuple[float, float, float, float] | None = None,
    id_prefix: str = "ms",
) -> list[Footprint]:
    """Footprints from Microsoft GeoJSON lines, keeping those whose outline touches `bbox`
    (west, south, east, north). Non-polygons, rings < 4 points and bad lines are skipped."""
    out = []
    for n, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            feat = json.loads(line)
            geom = feat["geometry"]
            if geom.get("type") != "Polygon":
                continue
            ring = np.asarray(geom["coordinates"][0], np.float64)[:, :2]
        except (ValueError, KeyError, TypeError, IndexError):
            continue
        if len(ring) < 4 or not np.isfinite(ring).all():
            continue
        if bbox is not None:
            w, s, e, nn = bbox
            if (ring[:, 0].max() < w or ring[:, 0].min() > e
                    or ring[:, 1].max() < s or ring[:, 1].min() > nn):
                continue
        if not np.allclose(ring[0], ring[-1]):
            ring = np.vstack([ring, ring[:1]])
        props = feat.get("properties") or {}
        try:
            h = float(props.get("height", -1.0))
        except (TypeError, ValueError):
            h = -1.0
        out.append(
            Footprint(f"{id_prefix}:{n}", "microsoft", ring, {},
                      h if math.isfinite(h) and h > 0 else None)
        )
    return out


def _footprint_json(f: Footprint) -> dict:
    return {"id": f.id, "ring": np.round(f.lonlat, 8).tolist(), "height": f.ms_height_m}


def _footprint_from_json(d: Mapping) -> Footprint:
    return Footprint(str(d["id"]), "microsoft", np.asarray(d["ring"], np.float64), {},
                     d.get("height"))


def fetch_ms_buildings(
    bbox: tuple[float, float, float, float],
    *,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
    offline: bool = False,
    opener: StreamOpener = _open_url,
) -> list[Footprint]:
    """Microsoft footprints touching the lon/lat `bbox` (west, south, east, north).

    Each zoom-9 tile the bbox touches is streamed and filtered; the subset is cached per
    (tile URL, bbox rounded outwards to 1e-3 degrees). `offline=True` uses only the cache
    (OSError on a miss). `opener(url)` returns a binary stream (injectable for tests).
    """
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    w, s, e, n = bbox
    box = (math.floor(w * 1e3) / 1e3, math.floor(s * 1e3) / 1e3,
           math.ceil(e * 1e3) / 1e3, math.ceil(n * 1e3) / 1e3)
    links = _links_csv(cache, opener, refresh, offline)
    out: list[Footprint] = []
    for qk, url in ms_tile_urls(links, quadkeys_for_bbox(*box)):
        key = hashlib.sha256(f"{url}|{box}".encode()).hexdigest()
        path = cache / "msbuildings" / f"{key}.json.gz"
        cached = None if refresh else _read_gz_json(path)
        if isinstance(cached, list):
            out.extend(_footprint_from_json(d) for d in cached)
            continue
        if offline:
            raise OSError(f"Microsoft footprints for tile {qk} are not cached (offline)")
        log.info("streaming Microsoft building tile %s", qk)
        with opener(url) as raw, gzip.GzipFile(fileobj=raw) as lines:
            feats = parse_ms_features(lines, box, id_prefix=f"ms:{qk}")
        _write_gz(path, json.dumps([_footprint_json(f) for f in feats]).encode())
        out.extend(feats)
    return out


# --------------------------------------------------------------------------------------------
# Merge and the one-call entry point
# --------------------------------------------------------------------------------------------


def _polygons(footprints: Sequence[Footprint], proj: LocalProjection) -> np.ndarray:
    polys = np.array([shapely.Polygon(proj.to_xy(f.lonlat)) for f in footprints])
    bad = ~shapely.is_valid(polys)
    if bad.any():
        polys[bad] = shapely.make_valid(polys[bad])
    return polys


def merge_footprints(
    osm: Sequence[Footprint], ms: Sequence[Footprint], max_overlap: float = 0.3
) -> list[Footprint]:
    """All OSM footprints, plus each Microsoft footprint that no OSM footprint covers by more
    than `max_overlap` of the Microsoft footprint's own area."""
    if not ms:
        return list(osm)
    if not osm:
        return list(ms)
    allpts = np.concatenate([f.lonlat for f in (*osm, *ms)])
    lon0, lat0 = allpts.mean(axis=0)
    proj = LocalProjection(float(lat0), float(lon0))
    p_osm = _polygons(osm, proj)
    p_ms = _polygons(ms, proj)
    tree = shapely.STRtree(p_osm)
    out = list(osm)
    for f, poly in zip(ms, p_ms, strict=True):
        area = poly.area
        if area <= 0:
            continue
        hits = tree.query(poly, predicate="intersects")
        covered = max(
            (shapely.intersection(poly, p_osm[i]).area for i in hits), default=0.0
        )
        if covered <= max_overlap * area:
            out.append(f)
    return out


def _within_route(
    footprints: Sequence[Footprint], route_lonlat: np.ndarray, around_m: float
) -> list[Footprint]:
    if not footprints:
        return []
    proj = LocalProjection(float(route_lonlat[0, 1]), float(route_lonlat[0, 0]))
    line = shapely.LineString(proj.to_xy(route_lonlat))
    polys = _polygons(footprints, proj)
    keep = shapely.distance(polys, line) <= around_m
    return [f for f, k in zip(footprints, keep, strict=True) if k]


def fetch_buildings(
    route_lonlat: npt.ArrayLike,
    *,
    cache_dir: Path | str | None = None,
    around_m: float = DEFAULT_AROUND_M,
    microsoft: bool = True,
    max_overlap: float = 0.3,
    mirrors: Sequence[str] | None = None,
    refresh: bool = False,
    offline: bool = False,
    ms_opener: StreamOpener = _open_url,
    **post_kwargs,
) -> BuildingFootprints:
    """Building footprints within `around_m` of the (N, 2) lon/lat route line: OSM, plus
    Microsoft footprints where OSM has none (`merge_footprints`).

    A failing Microsoft download is logged and noted, not raised (OSM alone still builds);
    a failing Overpass query raises OverpassError like the other OSM fetches.
    """
    ll = np.asarray(route_lonlat, dtype=np.float64).reshape(-1, 2)
    osm, osm_base = fetch_osm_buildings(
        ll, cache_dir=cache_dir, around_m=around_m, mirrors=mirrors, refresh=refresh,
        offline=offline, **post_kwargs,
    )
    osm = _within_route(osm, ll, around_m)
    result = BuildingFootprints(list(osm), osm_count=len(osm), osm_base=osm_base)
    if not microsoft:
        return result
    dlat = around_m / 111_000.0 + 1e-4
    dlon = dlat / max(math.cos(math.radians(float(ll[:, 1].mean()))), 0.1)
    bbox = (float(ll[:, 0].min() - dlon), float(ll[:, 1].min() - dlat),
            float(ll[:, 0].max() + dlon), float(ll[:, 1].max() + dlat))
    try:
        ms = fetch_ms_buildings(bbox, cache_dir=cache_dir, refresh=refresh, offline=offline,
                                opener=ms_opener)
    except (OSError, urllib.error.URLError, EOFError, ValueError) as exc:
        msg = f"Microsoft building footprints unavailable, using OSM only: {exc}"
        log.warning(msg)
        result.notes.append(msg)
        return result
    ms = _within_route(ms, ll, around_m)
    merged = merge_footprints(osm, ms, max_overlap)
    result.footprints = merged
    result.ms_count = len(merged) - len(osm)
    return result
