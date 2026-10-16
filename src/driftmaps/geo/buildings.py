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
import dataclasses
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
    "ConflationReport",
    "Footprint",
    "build_buildings_query",
    "build_use_areas_query",
    "conflate",
    "fetch_buildings",
    "fetch_ms_buildings",
    "fetch_use_areas",
    "fetch_osm_buildings",
    "merge_footprints",
    "ms_tile_urls",
    "parse_buildings",
    "parse_ms_features",
    "parse_use_areas",
    "quadkey",
    "quadkeys_for_bbox",
    "regularize",
    "route_boxes",
    "within_route",
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

    `id` is "osm:w<id>", "osm:r<id>.<k>" (k-th outer ring of a relation), "ms:<quadkey>:<n>",
    "nrcan:<fid>" or "<municipality>:<objectid>[.<k>]" (`geo.building_sources`).
    `tags` are the OSM tags (empty for Microsoft; a `building` value for municipal types);
    `ms_height_m` the Microsoft ML height (None when unknown; the MEAN height over the
    footprint, so about mid-roof). Holes (courtyards) are not kept: the building is drawn solid.

    Measured heights above ground, when a source has them: `peak_m` (roof peak: municipal
    PEAKHGT, NRCan `heightmax`), `eave_m` (municipal EAVEHGT), `low_m` (NRCan `heightmin`, the
    lowest lidar building point). `year` is when the outline was captured (lidar / photo
    year), `lidar` True for a lidar-derived outline (NRCan). `also` lists the ids of the other
    sources' footprints that `conflate` matched to this one (same building).
    """

    id: str
    source: str  # "osm" | "microsoft" | "nrcan" | a municipality ("burlington", ...)
    lonlat: np.ndarray = field(repr=False)
    tags: Mapping[str, str] = field(default_factory=dict)
    ms_height_m: float | None = None
    peak_m: float | None = None
    eave_m: float | None = None
    low_m: float | None = None
    year: int | None = None
    lidar: bool = False
    also: tuple[str, ...] = ()

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
    """What the footprint sources gave: the merged footprints and where they came from.

    `found` counts the fetched footprints per source, `report` is the conflation's
    (`ConflationReport`), `osm` every OSM footprint fetched (listed in route_osm.geojson)."""

    footprints: list[Footprint]
    osm_count: int = 0
    ms_count: int = 0  # Microsoft footprints added by the merge
    osm_base: str | None = None
    notes: list[str] = field(default_factory=list)
    found: dict[str, int] = field(default_factory=dict)
    report: ConflationReport | None = None
    osm: list[Footprint] | None = None

    @property
    def attributions(self) -> list[str]:
        """LICENSE-DATA.txt lines for the sources that contributed footprints."""
        return self.attributions_for(None)

    def attributions_for(self, ids: set[str] | None) -> list[str]:
        """LICENSE-DATA.txt lines for the sources of the footprints with these ids (all when
        None): their geometry's source and every source merged into them (`also`)."""
        used: list[str] = []
        for f in self.footprints:
            if ids is not None and f.id not in ids:
                continue
            for sid in (f.id, *f.also):
                src = _source_of(sid)
                if src not in used:
                    used.append(src)
        if ids is None and not self.footprints:
            used = (["osm"] if self.osm_count else []) + (["microsoft"] if self.ms_count else [])
        return [line for s in _ATTRIBUTION_ORDER if s in used and (line := _attribution(s))]


_ATTRIBUTION_ORDER = ("osm", "burlington", "milton", "oakville", "nrcan", "microsoft")


def _source_of(footprint_id: str) -> str:
    head = footprint_id.split(":", 1)[0]
    return "microsoft" if head == "ms" else head


def _attribution(source: str) -> str | None:
    if source == "osm":
        return OSM_BUILDINGS_ATTRIBUTION
    if source == "microsoft":
        return MS_ATTRIBUTION
    from driftmaps.geo import building_sources as bsrc  # no import cycle at module load

    if source == "nrcan":
        return bsrc.NRCAN_ATTRIBUTION
    muni = bsrc.MUNICIPAL_SOURCES.get(source)
    return muni.attribution if muni is not None else None


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
        box = (
            math.floor(so * 1e5) / 1e5,
            math.floor(w * 1e5) / 1e5,
            math.ceil(n * 1e5) / 1e5,
            math.ceil(e * 1e5) / 1e5,
        )
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
    5 m); `fetch_buildings` then keeps footprints within `around_m` of the line. A downloaded
    region database (`driftmaps.geo.offline`) covering the boxes answers first (not with
    `refresh`). `offline=True` never touches the network (OverpassError on a cache miss).
    Extra kwargs go to `overpass.post_query`.
    """
    ll = np.asarray(route_lonlat, dtype=np.float64).reshape(-1, 2)
    boxes = route_boxes(ll, around_m + 5.0, piece_m)
    if not refresh:  # the downloaded region database, when it covers the boxes
        from driftmaps.geo import offline as region_db

        found = region_db.buildings_payload(boxes)
        if found is not None:
            return parse_buildings(found), found["osm3s"]["timestamp_osm_base"]
    query = build_buildings_query(boxes)
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


# Land use and campuses around the buildings: what a commercial building is for
# (`geometry.buildings._use`: a plaza on `landuse=retail`, a plant on `landuse=industrial`, the
# school's buildings on its `amenity=school` grounds).
USE_LANDUSE = ("retail", "commercial", "industrial")
USE_AMENITIES = ("school", "hospital", "college", "university", "kindergarten")


def build_use_areas_query(
    boxes: Sequence[tuple[float, float, float, float]], timeout_s: int = 120
) -> str:
    """Overpass QL for `landuse=retail|commercial|industrial` and school / hospital / college
    grounds (ways and multipolygons) in the (south, west, north, east) boxes."""
    if not boxes:
        raise ValueError("need at least one box")
    lu = "|".join(USE_LANDUSE)
    am = "|".join(USE_AMENITIES)
    parts = []
    for s_, w, n, e in boxes:
        bb = f"({_fmt(s_)},{_fmt(w)},{_fmt(n)},{_fmt(e)})"
        parts.append(
            f'  way["landuse"~"^({lu})$"]{bb};\n'
            f'  relation["landuse"~"^({lu})$"]["type"="multipolygon"]{bb};\n'
            f'  way["amenity"~"^({am})$"]{bb};\n'
            f'  relation["amenity"~"^({am})$"]["type"="multipolygon"]{bb};'
        )
    head = f"[out:json][timeout:{int(timeout_s)}];\n(\n"
    return head + "\n".join(parts) + "\n);\nout body geom;\n"


def parse_use_areas(payload: bytes | str | Mapping) -> list[tuple[np.ndarray, dict[str, str]]]:
    """(lon/lat outer ring, tags) of each land use / campus area in an Overpass response."""
    if isinstance(payload, bytes | str):
        try:
            payload = json.loads(payload)
        except ValueError as exc:
            raise OverpassError(f"not an Overpass JSON response: {exc}") from exc
    if not isinstance(payload, Mapping) or "elements" not in payload:
        raise OverpassError("not an Overpass JSON response (no 'elements')")
    out: list[tuple[np.ndarray, dict[str, str]]] = []
    for el in payload["elements"]:
        tags = {str(k): str(v) for k, v in (el.get("tags") or {}).items()}
        if tags.get("landuse") not in USE_LANDUSE and tags.get("amenity") not in USE_AMENITIES:
            continue
        if el.get("type") == "way":
            ring = _ring(el.get("geometry") or [])
            if ring is not None:
                out.append((ring, tags))
        elif el.get("type") == "relation":
            out += [(ring, tags) for ring in _relation_rings(el.get("members") or [])]
    return out


def fetch_use_areas(
    route_lonlat: npt.ArrayLike,
    *,
    cache_dir: Path | str | None = None,
    around_m: float = DEFAULT_AROUND_M,
    piece_m: float = 1000.0,
    mirrors: Sequence[str] | None = None,
    refresh: bool = False,
    offline: bool = False,
    **post_kwargs,
) -> list[tuple[np.ndarray, dict[str, str]]]:
    """OSM land use and campus areas near the route (`build_use_areas_query`), cached like
    the buildings. `offline=True` never touches the network (OverpassError on a miss)."""
    ll = np.asarray(route_lonlat, dtype=np.float64).reshape(-1, 2)
    query = build_use_areas_query(route_boxes(ll, around_m + 5.0, piece_m))
    key = query_cache_key(query)
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    path = cache / "overpass" / f"{key}.json.gz"
    payload = None if refresh else _read_gz_json(path)
    if payload is None:
        if offline:
            raise OverpassError(f"no cached Overpass land use data (key {key[:12]}...)")
        body = post_query(query, mirrors, **post_kwargs)
        payload = json.loads(body)
        _write_gz(path, body)
    return parse_use_areas(payload)


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
    return sorted(quadkey(x, y, zoom) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))


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
            if (
                ring[:, 0].max() < w
                or ring[:, 0].min() > e
                or ring[:, 1].max() < s
                or ring[:, 1].min() > nn
            ):
                continue
        if not np.allclose(ring[0], ring[-1]):
            ring = np.vstack([ring, ring[:1]])
        props = feat.get("properties") or {}
        try:
            h = float(props.get("height", -1.0))
        except (TypeError, ValueError):
            h = -1.0
        out.append(
            Footprint(
                f"{id_prefix}:{n}", "microsoft", ring, {}, h if math.isfinite(h) and h > 0 else None
            )
        )
    return out


def _footprint_json(f: Footprint) -> dict:
    return {"id": f.id, "ring": np.round(f.lonlat, 8).tolist(), "height": f.ms_height_m}


def _footprint_from_json(d: Mapping) -> Footprint:
    return Footprint(
        str(d["id"]), "microsoft", np.asarray(d["ring"], np.float64), {}, d.get("height")
    )


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
    box = (
        math.floor(w * 1e3) / 1e3,
        math.floor(s * 1e3) / 1e3,
        math.ceil(e * 1e3) / 1e3,
        math.ceil(n * 1e3) / 1e3,
    )
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
        covered = max((shapely.intersection(poly, p_osm[i]).area for i in hits), default=0.0)
        if covered <= max_overlap * area:
            out.append(f)
    return out


def within_route(
    footprints: Sequence[Footprint], route_lonlat: np.ndarray, around_m: float
) -> list[Footprint]:
    """The footprints within `around_m` of the (N, 2) lon/lat route line."""
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
        ll,
        cache_dir=cache_dir,
        around_m=around_m,
        mirrors=mirrors,
        refresh=refresh,
        offline=offline,
        **post_kwargs,
    )
    osm = within_route(osm, ll, around_m)
    result = BuildingFootprints(list(osm), osm_count=len(osm), osm_base=osm_base)
    if not microsoft:
        return result
    dlat = around_m / 111_000.0 + 1e-4
    dlon = dlat / max(math.cos(math.radians(float(ll[:, 1].mean()))), 0.1)
    bbox = (
        float(ll[:, 0].min() - dlon),
        float(ll[:, 1].min() - dlat),
        float(ll[:, 0].max() + dlon),
        float(ll[:, 1].max() + dlat),
    )
    try:
        ms = fetch_ms_buildings(
            bbox, cache_dir=cache_dir, refresh=refresh, offline=offline, opener=ms_opener
        )
    except (OSError, urllib.error.URLError, EOFError, ValueError) as exc:
        msg = f"Microsoft building footprints unavailable, using OSM only: {exc}"
        log.warning(msg)
        result.notes.append(msg)
        return result
    ms = within_route(ms, ll, around_m)
    merged = merge_footprints(osm, ms, max_overlap)
    result.footprints = merged
    result.ms_count = len(merged) - len(osm)
    return result


# --------------------------------------------------------------------------------------------
# Outline regularisation (lidar footprints)
# --------------------------------------------------------------------------------------------


def _line_cross(e0: tuple, e1: tuple) -> np.ndarray | None:
    """Intersection of two edge lines ("h" y=v, "v" x=v, "o" through a and b), or None."""

    def pd(e: tuple) -> tuple[np.ndarray, np.ndarray]:
        kind, v, _, a, b = e
        if kind == "h":
            return np.array([0.0, v]), np.array([1.0, 0.0])
        if kind == "v":
            return np.array([v, 0.0]), np.array([0.0, 1.0])
        return a, b - a

    p0, d0 = pd(e0)
    p1, d1 = pd(e1)
    den = d0[0] * d1[1] - d0[1] * d1[0]
    if abs(den) < 1e-9 * (np.linalg.norm(d0) * np.linalg.norm(d1) + 1e-12):
        return None
    w = p1 - p0
    t = (w[0] * d1[1] - w[1] * d1[0]) / den
    return p0 + t * d0


def _orthogonal_ring(ring: np.ndarray, snap_deg: float, min_edge: float) -> np.ndarray | None:
    """Edges of an open ring (in the building's own frame, dominant axis = x) within
    `snap_deg` of x or y made exactly axis-parallel; consecutive parallel edges merged (length
    weighted), edges shorter than `min_edge` dropped; vertices = consecutive edge crossings."""
    edges: list[tuple] = []
    n = len(ring)
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        d = b - a
        length = float(np.hypot(*d))
        if length < 1e-9:
            continue
        ang = abs(math.degrees(math.atan2(d[1], d[0]))) % 180.0
        if ang <= snap_deg or ang >= 180.0 - snap_deg:
            edges.append(("h", 0.5 * (a[1] + b[1]), length, a, b))
        elif abs(ang - 90.0) <= snap_deg:
            edges.append(("v", 0.5 * (a[0] + b[0]), length, a, b))
        else:
            edges.append(("o", 0.0, length, a, b))
    changed = True
    while changed and len(edges) > 3:
        changed = False
        for i in range(len(edges)):
            j = (i + 1) % len(edges)
            ki, kj = edges[i][0], edges[j][0]
            if ki == kj and ki in "hv":
                li, lj = edges[i][2], edges[j][2]
                v = (edges[i][1] * li + edges[j][1] * lj) / (li + lj)
                merged = (ki, v, li + lj, edges[i][3], edges[j][4])
                if j == 0:
                    edges = [merged, *edges[1:i]]
                else:
                    edges[i] = merged
                    del edges[j]
                changed = True
                break
        if changed:
            continue
        k = min(range(len(edges)), key=lambda q: edges[q][2])
        if edges[k][2] < min_edge and len(edges) > 4:
            del edges[k]
            changed = True
    if len(edges) < 3:
        return None
    pts = []
    for i in range(len(edges)):
        p = _line_cross(edges[i - 1], edges[i])
        pts.append(edges[i][3] if p is None else p)
    return np.asarray(pts)


def regularize(
    poly: shapely.Polygon,
    *,
    tolerance_m: float = 0.6,
    snap_deg: float = 15.0,
    min_edge_m: float = 1.0,
    rect_min: float = 0.85,
) -> shapely.Polygon:
    """A cleaner outline for a lidar (stair-stepped) footprint, in metres (research §4):
    Douglas-Peucker at `tolerance_m`; if it then fills >= `rect_min` of its minimum rotated
    rectangle, that rectangle scaled to the same area; else edges within `snap_deg` of the
    rectangle's axes snapped to them, collinear runs merged, edges < `min_edge_m` dropped.
    The input comes back unchanged when a step would distort it (area off by > 20 % or
    IoU < 0.75)."""
    if poly.is_empty or not isinstance(poly, shapely.Polygon) or poly.area <= 0:
        return poly
    base = shapely.Polygon(poly.exterior)
    simple = base.simplify(tolerance_m, preserve_topology=True)
    if not isinstance(simple, shapely.Polygon) or simple.is_empty or not simple.is_valid:
        simple = base

    def ok(cand: shapely.Geometry) -> bool:
        if not isinstance(cand, shapely.Polygon) or cand.is_empty or not cand.is_valid:
            return False
        if abs(cand.area / base.area - 1.0) > 0.2:
            return False
        inter = cand.intersection(base).area
        return inter / (cand.area + base.area - inter) >= 0.75

    rect = shapely.oriented_envelope(simple)
    if not isinstance(rect, shapely.Polygon) or rect.area <= 0:
        return simple if ok(simple) else base
    c = np.asarray(rect.exterior.coords)[:4]
    if simple.area / rect.area >= rect_min:
        centre = c.mean(axis=0)
        k = math.sqrt(base.area / rect.area)
        cand = shapely.Polygon(centre + (c - centre) * k)
        if ok(cand):
            return cand
    e0 = c[1] - c[0] if np.hypot(*(c[1] - c[0])) >= np.hypot(*(c[2] - c[1])) else c[2] - c[1]
    theta = math.atan2(e0[1], e0[0])
    cs, sn = math.cos(theta), math.sin(theta)
    rot = np.array([[cs, -sn], [sn, cs]])  # local -> world
    ring = np.asarray(simple.exterior.coords)[:-1]
    origin = ring.mean(axis=0)
    local = (ring - origin) @ rot
    out = _orthogonal_ring(local, snap_deg, min_edge_m)
    if out is not None and len(out) >= 3:
        cand = shapely.Polygon(out @ rot.T + origin)
        if ok(cand):
            return cand
    return simple if ok(simple) else base


# --------------------------------------------------------------------------------------------
# Conflation: one building per real building, from several sources (research §4)
# --------------------------------------------------------------------------------------------


@dataclass
class ConflationReport:
    """What `conflate` did. `kept` counts the result's geometries per source."""

    kept: dict[str, int] = field(default_factory=dict)
    matched: int = 0  # lower-priority footprints merged into a kept one (IoU >= iou_same)
    overlapping: int = 0  # dropped: overlap >= overlap_drop of either footprint
    cut: int = 0  # lidar footprints added after cutting the kept ones out of them
    unconfirmed: int = 0  # new lower-priority footprints the lidar does not confirm
    small: int = 0  # dropped: < min_area_m2 (after cutting)
    demolished: list[str] = field(default_factory=list)  # dropped: lidar sees no building
    promoted: int = 0  # OSM outlines that replaced a split building's pieces


def _xz_polygon(fp: Footprint, projection: LocalProjection) -> shapely.Polygon | None:
    e, n = projection.to_enu(fp.lonlat[:, 0], fp.lonlat[:, 1])
    poly = shapely.Polygon(np.stack([e, -n], axis=1))
    if not poly.is_valid:
        poly = shapely.make_valid(poly)
    parts = [p for p in shapely.get_parts(poly) if isinstance(p, shapely.Polygon) and p.area > 0]
    if not parts:
        return None
    return shapely.Polygon(max(parts, key=lambda p: p.area).exterior)


def _xz_to_lonlat(poly: shapely.Polygon, projection: LocalProjection) -> np.ndarray:
    xz = np.asarray(poly.exterior.coords)
    lon, lat = projection.to_lonlat(xz[:, 0], -xz[:, 1])
    return np.stack([lon, lat], axis=1)


def _lidar_medians(
    polys: Sequence[shapely.Polygon], canopy: Callable[[np.ndarray], np.ndarray], erode_m: float
) -> np.ndarray:
    """Median DSM - DTM inside each polygon shrunk by `erode_m` (1 m lattice; half-metre on
    the full polygon when that leaves too few points). NaN = no lidar."""
    pts, owner = [], []
    for k, p in enumerate(polys):
        inner = p.buffer(-erode_m)
        q = _lattice_pts(inner, 1.0)
        if len(q) < 3:
            q = _lattice_pts(p, 0.5)
        pts.append(q)
        owner.append(np.full(len(q), k))
    out = np.full(len(polys), np.nan)
    if not pts or not sum(len(q) for q in pts):
        return out
    allp = np.concatenate(pts)
    own = np.concatenate(owner)
    try:
        h = np.asarray(canopy(allp), np.float64).reshape(-1)
    except (OSError, RuntimeError, ValueError) as exc:
        log.warning("lidar check of building footprints skipped: %s", exc)
        return out
    for k in range(len(polys)):
        v = h[own == k]
        v = v[np.isfinite(v)]
        if v.size >= 3:
            out[k] = float(np.median(v))
    return out


def _lattice_pts(poly: shapely.Geometry, step: float) -> np.ndarray:
    if poly.is_empty:
        return np.zeros((0, 2))
    x0, z0, x1, z1 = poly.bounds
    xs = np.arange(x0 + step / 2, x1, step)
    zs = np.arange(z0 + step / 2, z1, step)
    if len(xs) == 0 or len(zs) == 0:
        return np.zeros((0, 2))
    gx, gz = np.meshgrid(xs, zs)
    p = np.stack([gx.ravel(), gz.ravel()], axis=1)
    return p[shapely.contains_xy(poly, p[:, 0], p[:, 1])]


def _merge_attrs(keep: Footprint, other: Footprint) -> Footprint:
    """`keep` with what `other` (the same building, another source) adds: lidar heights,
    OSM tags (a more specific `building` value), the Microsoft height."""
    tags = dict(keep.tags)
    if other.tags:
        for k, v in other.tags.items():
            if k == "building":
                if tags.get("building", "yes") == "yes" and v not in ("yes", ""):
                    tags["building"] = v
            elif k not in tags:
                tags[k] = v
    return dataclasses.replace(
        keep,
        tags=tags,
        peak_m=keep.peak_m if keep.peak_m is not None else other.peak_m,
        eave_m=keep.eave_m if keep.eave_m is not None else other.eave_m,
        low_m=keep.low_m if keep.low_m is not None else other.low_m,
        ms_height_m=keep.ms_height_m if keep.ms_height_m is not None else other.ms_height_m,
        year=keep.year if keep.year is not None else other.year,
        also=(*keep.also, other.id, *other.also),
    )


def conflate(
    tiers: Sequence[Sequence[Footprint]],
    projection: LocalProjection,
    canopy: Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    iou_same: float = 0.5,
    overlap_drop: float = 0.3,
    new_max: float = 0.1,
    min_area_m2: float = 10.0,
    min_peak_m: float = 2.0,
    min_lidar_m: float = 2.0,
    demolished_max_m: float = 1.0,
    regularize_lidar: bool = True,
    survey_small_m2: float = 40.0,
    survey_near_m: float = 250.0,
    promote_min_m2: float = 1500.0,
) -> tuple[list[Footprint], ConflationReport]:
    """One footprint per building from `tiers` (highest priority first, e.g. municipal, OSM,
    NRCan lidar, Microsoft), by the research's rules (§4), in the build's metres
    (`projection`; `canopy(xz)` = lidar DSM - DTM on AC XZ points, as `place_buildings` uses):

    - The first non-empty tier is kept whole (near-duplicates inside it, IoU >= `iou_same`,
      dropped).
    - A lower-priority footprint with IoU >= `iou_same` against a kept one is the SAME
      building: its attributes are merged in (`also` lists it), its geometry dropped.
    - One overlapping a kept footprint by >= `overlap_drop` of either area is dropped.
    - Otherwise it is NEW when its overlap with all kept ones is < `new_max` of its own area,
      or (lidar footprints only) when the kept ones cut out of it leave >= `min_area_m2`.
      New buildings need lidar confirmation: a lidar outline with peak >= `min_peak_m`, an
      NRCan outline covering >= `overlap_drop` of it, or a DSM - DTM median >= `min_lidar_m`
      (no lidar at all: accepted). Where the first tier is a full survey (municipal: it maps
      sheds), a lidar-only outline < `survey_small_m2` within `survey_near_m` of it also
      needs OSM or Microsoft to have it (lidar takes RVs, trailers and hedges for sheds).
    - Lidar outlines are regularised (`regularize`) first.
    - SPLIT landmarks: an OSM outline of a notable building (tagged with a specific
      `building` type, an `amenity` or a `name`; at least `promote_min_m2`) that the first
      tier maps as two or more pieces (each mostly inside it, together covering at least half
      of it, none the same building) replaces those pieces: one hospital, not its wings
      (the user's Oakville check, 2026-10-16).
    - DEMOLISHED: a kept non-lidar footprint older than the lidar near it (`year`), with no
      lidar outline on it and a DSM - DTM median (shrunk by 0.7 m) < `demolished_max_m`, is
      dropped and listed in the report.

    Returns the footprints (lon/lat; regularised or cut outlines converted back) sorted by id.
    """
    rep = ConflationReport()
    tiers = [list(t) for t in tiers]
    polys: list[list[shapely.Polygon | None]] = []
    for tier in tiers:
        row = []
        for fp in tier:
            p = _xz_polygon(fp, projection)
            if p is not None and fp.lidar and regularize_lidar:
                p = regularize(p)
            row.append(p)
        polys.append(row)
    if len(tiers) > 1:
        rep.promoted = _promote_split(tiers, polys, promote_min_m2, iou_same)
    lidar_fps = [(fp, p) for t, row in zip(tiers, polys, strict=True)
                 for fp, p in zip(t, row, strict=True) if fp.lidar and p is not None]  # fmt: skip
    lidar_tree = shapely.STRtree([p for _, p in lidar_fps]) if lidar_fps else None
    # The first tier when it is a full survey (municipal); the non-lidar lower tiers.
    survey = [p for fp, p in zip(tiers[0], polys[0], strict=True) if p is not None and not fp.lidar]
    survey_tree = shapely.STRtree(survey) if survey else None
    other_fps = [
        (fp, p)
        for t, row in list(zip(tiers, polys, strict=True))[1:]
        for fp, p in zip(t, row, strict=True)
        if p is not None and not fp.lidar
    ]
    other_tree = shapely.STRtree([p for _, p in other_fps]) if other_fps else None

    kept: list[Footprint] = []
    kept_poly: list[shapely.Polygon] = []
    changed_geom: set[int] = set()
    needs_lidar: set[int] = set()  # new, unconfirmed: checked against the DSM at the end

    base_tier = next((k for k, row in enumerate(polys) if any(p is not None for p in row)), 0)
    for t_idx, (tier, row) in enumerate(zip(tiers, polys, strict=True)):
        tier_start = len(kept)
        tree = shapely.STRtree(kept_poly) if kept_poly else None
        for fp, p in zip(tier, row, strict=True):
            if p is None or p.area <= 0:
                continue
            area = p.area
            hits = []  # (kept index, polygon, intersection area) against earlier tiers
            if tree is not None:
                for j in tree.query(p, predicate="intersects"):
                    q = kept_poly[int(j)]
                    inter = shapely.intersection(p, q).area
                    if inter > 0:
                        hits.append((int(j), q, inter))
            # ... and against this tier's own earlier footprints (duplicates only)
            for k in range(tier_start, len(kept)):
                q = kept_poly[k]
                if q.intersects(p):
                    inter = shapely.intersection(p, q).area
                    if inter / (area + q.area - inter) >= iou_same:
                        hits.append((k, q, inter))
            best = max(hits, key=lambda h: h[2] / (area + h[1].area - h[2]), default=None)
            if best is not None and best[2] / (area + best[1].area - best[2]) >= iou_same:
                kept[best[0]] = _merge_attrs(kept[best[0]], fp)
                rep.matched += 1
                continue
            hits = [h for h in hits if h[0] < tier_start]
            if t_idx <= base_tier or not hits:
                geom = p
            else:
                covered = sum(h[2] for h in hits)
                big = any(h[2] >= overlap_drop * min(area, h[1].area) for h in hits)
                # A lidar blob over two kept pieces (a row, a house + garage) is dropped; one
                # that just touches a small kept shed is the rest of a bigger building.
                if big and not (fp.lidar and covered < overlap_drop * area):
                    rep.overlapping += 1
                    continue
                if not fp.lidar:
                    if covered >= new_max * area:
                        rep.overlapping += 1
                        continue
                    geom = p
                else:  # cut the kept outlines out (attached garages are not added twice)
                    kept_here = shapely.union_all([h[1] for h in hits])
                    cut = shapely.difference(p, kept_here.buffer(0.2, join_style="mitre"))
                    cut = shapely.simplify(cut, 0.05)
                    parts = [g for g in shapely.get_parts(cut) if isinstance(g, shapely.Polygon)]
                    if not parts:
                        rep.small += 1
                        continue
                    geom = shapely.Polygon(max(parts, key=lambda g: g.area).exterior)
                    if geom.area < min_area_m2 or 2.0 * geom.area / geom.length < 1.2:
                        rep.small += 1  # too small, or a sliver along a kept wall
                        continue
                    rep.cut += 1
            if geom.area < min_area_m2:
                rep.small += 1
                continue
            if t_idx > base_tier:
                if fp.lidar:
                    if fp.peak_m is not None and fp.peak_m < min_peak_m:
                        rep.unconfirmed += 1
                        continue
                    # A small lidar-only object where a full survey (the base tier, e.g. a
                    # municipal layer with sheds) found nothing: an RV, a trailer, a hedge, ...
                    # unless OSM or Microsoft has it too.
                    if (
                        geom.area < survey_small_m2
                        and survey_tree is not None
                        and len(survey_tree.query(geom.buffer(survey_near_m)))
                        and not _covered_by(geom, other_tree, other_fps, overlap_drop)
                    ):
                        rep.unconfirmed += 1
                        continue
                elif not _covered_by(geom, lidar_tree, lidar_fps, overlap_drop):
                    needs_lidar.add(len(kept))
            if geom is not p or (fp.lidar and regularize_lidar):
                changed_geom.add(len(kept))
            kept.append(fp)
            kept_poly.append(geom)

    # Demolished candidates: kept non-lidar outlines older than the lidar near them, with no
    # lidar outline on them.
    lidar_years = [(fp.year, p) for fp, p in lidar_fps if fp.year is not None]
    year_tree = shapely.STRtree([p for _, p in lidar_years]) if lidar_years else None
    demolish_chk: list[int] = []
    if canopy is not None and year_tree is not None:
        for k, (fp, p) in enumerate(zip(kept, kept_poly, strict=True)):
            if fp.lidar or fp.year is None or k in needs_lidar:
                continue
            if _covered_by(p, lidar_tree, lidar_fps, 1e-6):
                continue
            j = int(year_tree.nearest(p))
            near_year, near_p = lidar_years[j]
            if near_year > fp.year and shapely.distance(p, near_p) < 500.0:
                demolish_chk.append(k)
    drop: set[int] = set()
    new_chk = sorted(needs_lidar)
    if canopy is not None and (demolish_chk or new_chk):
        shrunk = []
        for k in demolish_chk:
            s = kept_poly[k].buffer(-0.7)
            shrunk.append(s if s.area > 4.0 else kept_poly[k])
        med = _lidar_medians([*shrunk, *[kept_poly[k] for k in new_chk]], canopy, 0.0)
        for k, m in zip(demolish_chk, med[: len(demolish_chk)], strict=True):
            if np.isfinite(m) and m < demolished_max_m:
                drop.add(k)
                rep.demolished.append(kept[k].id)
        for k, m in zip(new_chk, med[len(demolish_chk) :], strict=True):
            if np.isfinite(m) and m < min_lidar_m:
                drop.add(k)
                rep.unconfirmed += 1

    out = []
    for k, (fp, p) in enumerate(zip(kept, kept_poly, strict=True)):
        if k in drop:
            continue
        if k in changed_geom:
            fp = dataclasses.replace(fp, lonlat=_xz_to_lonlat(p, projection))
        rep.kept[fp.source] = rep.kept.get(fp.source, 0) + 1
        out.append(fp)
    return sorted(out, key=lambda f: f.id), rep


_PLAIN_KINDS = frozenset({"yes", "house", "detached", "residential", "semidetached_house",
                          "garage", "garages", "shed", "roof", "terrace", "carport"})  # fmt: skip


def _promote_split(
    tiers: list[list[Footprint]],
    polys: list[list[shapely.Polygon | None]],
    min_m2: float,
    iou_same: float,
) -> int:
    """Move notable OSM outlines that the first tier splits into pieces into the first tier,
    dropping the pieces (`conflate`, SPLIT landmarks). Returns how many were promoted."""
    base = [(k, p) for k, p in enumerate(polys[0]) if p is not None]
    if not base:
        return 0
    tree = shapely.STRtree([p for _, p in base])
    promoted = 0
    for t_idx in range(1, len(tiers)):
        for i, (fp, p) in enumerate(zip(tiers[t_idx], polys[t_idx], strict=True)):
            if p is None or fp.source != "osm" or p.area < min_m2:
                continue
            t = fp.tags
            if fp.kind in _PLAIN_KINDS and not t.get("amenity") and not t.get("name"):
                continue
            pieces, covered = [], 0.0
            same = False
            for j in tree.query(p, predicate="intersects"):
                k, q = base[int(j)]
                if polys[0][k] is None:
                    continue
                inter = shapely.intersection(p, q).area
                if inter / (p.area + q.area - inter) >= iou_same:
                    same = True
                    break
                if inter >= 0.6 * q.area:
                    pieces.append(k)
                    covered += inter
            if same or len(pieces) < 2 or covered < 0.5 * p.area:
                continue
            for k in pieces:
                polys[0][k] = None
            ids = tuple(tiers[0][k].id for k in pieces)
            tiers[0].append(dataclasses.replace(fp, also=(*fp.also, *ids)))
            polys[0].append(p)
            polys[t_idx][i] = None
            promoted += 1
    return promoted


def _covered_by(
    p: shapely.Polygon,
    tree: shapely.STRtree | None,
    items: Sequence[tuple[Footprint, shapely.Polygon]],
    share: float,
) -> bool:
    """Whether the lidar outlines in `tree` cover >= `share` of `p`'s area."""
    if tree is None:
        return False
    for j in tree.query(p, predicate="intersects"):
        if shapely.intersection(p, items[int(j)][1]).area >= share * p.area:
            return True
    return False
