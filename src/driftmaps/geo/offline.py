"""Offline OSM region database: a whole region's roads (and the builder's other OSM data) on
this PC, so map tiles, route corridors and the builder's queries skip Overpass.

Made once from a regional extract (Geofabrik `ontario-latest.osm.pbf`; downloaded by
`driftmaps.app.offline`) by `build_db`, with pyosmium (libosmium) reading the PBF in C++. Only
what the app asks Overpass for is kept:

- roads: `highway` ways of the classes `barriers.ACCESS_HIGHWAYS` (a superset of the router's
  drivable classes, `overpass.DRIVABLE_HIGHWAY_RE`; the query-time filter `is_drivable` mirrors
  the Overpass QL), with node ids (topology) and coordinates;
- buildings: `building=*` ways and multipolygon relations (only the tags `Footprint` reads);
- barriers: `barrier=` `barriers.BARRIER_KINDS` ways;
- roadside features for the warrant guardrails: deep-water candidates (`natural=water`,
  `waterway=riverbank|river|canal`, `landuse=reservoir` ways and relations) and rock
  (`natural=cliff|bare_rock|scree`).

Two passes with bounded memory: pass 1 reads only relations (which ways are members of a kept
multipolygon); pass 2 streams nodes and ways through libosmium's node-location index
(`flex_mem`: a sparse in-memory id -> location table, 16 bytes per node; ~2 GB for Ontario) and
hands Python only the ways it keeps. Rows are written in batches; nothing else is held.

SQLite file (stdlib sqlite3), FORMAT 1:
- `meta(key, value)`: format, region, osm_base (the extract's data timestamp), source, built,
  poly (the extract's boundary as WKT, lon/lat) and counts.
- `features(fid, kind, osm_id, rel_id, role, closed, tags, nodes, coords)`: one row per kept way
  and kind (KIND_*). A member way of a kept relation gets a row of its own with `rel_id`, its
  `role` and the relation's tags. `coords` are the way's points as zlib'd int32 deltas of
  lon/lat x 1e7 (OSM's own fixed point, so exactly what Overpass returns); `nodes` the node ids
  as zlib'd int64 deltas (not kept for buildings and relation members, nothing reads them).
  Coordinates live in the way rows rather than a node table: a tile query then reads its ways
  only (no join over millions of nodes) and the file is about half the size.
- `tiles(kind, tile, fid)` (WITHOUT ROWID): every z14 tile (`x * 16384 + y`) that a feature's
  segments' bounding boxes touch, the rule `driftmaps.app.tiles` uses, so a tile's roads are
  one index range.

Queries (`corridor`, `tile_ways`, `barrier_ways`, `buildings_payload`, `roadside_payload`) give
what the Overpass path gives: `OsmData`, or an Overpass-JSON-shaped payload for the existing
parsers. Each returns None unless a ready database covers every z14 tile the query touches
(the tile lies inside the extract's boundary polygon); the caller then asks Overpass exactly as
before. The database is found in `offline_dir()` (`$DRIFTMAPS_OFFLINE_DIR`, else the folder
`configure()` set, else `<app data>/offline`).

OSM data is (c) OpenStreetMap contributors, ODbL 1.0 (`overpass.OSM_ATTRIBUTION`).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import shapely

from driftmaps.geo.barriers import _ROCK, ACCESS_HIGHWAYS, BARRIER_KINDS
from driftmaps.geo.overpass import (
    DRIVABLE_HIGHWAY_RE,
    EXCLUDED_SERVICE_RE,
    PAVED_SURFACE_RE,
    OsmData,
    Way,
)
from driftmaps.geo.project import LocalProjection

__all__ = [
    "ENV_OFFLINE_DIR",
    "FORMAT",
    "BuildCancelled",
    "BuildStats",
    "OfflineDB",
    "barrier_ways",
    "build_db",
    "buildings_payload",
    "configure",
    "corridor",
    "databases",
    "db_path",
    "exclusive",
    "is_drivable",
    "offline_dir",
    "parse_poly",
    "release",
    "roadside_payload",
    "tile_ways",
]

log = logging.getLogger(__name__)

FORMAT = 1
ENV_OFFLINE_DIR = "DRIFTMAPS_OFFLINE_DIR"
DB_SUFFIX = ".sqlite"
TILE_Z = 14
_N = 2**TILE_Z
KIND_ROAD, KIND_BUILDING, KIND_BARRIER, KIND_WATER, KIND_ROCK = 1, 2, 3, 4, 5
KIND_NAMES = {
    KIND_ROAD: "roads",
    KIND_BUILDING: "buildings",
    KIND_BARRIER: "barriers",
    KIND_WATER: "water",
    KIND_ROCK: "rock",
}
_ROAD_HIGHWAYS = frozenset(ACCESS_HIGHWAYS)
_BARRIERS = frozenset(BARRIER_KINDS)
_ROCKS = frozenset(_ROCK)
_WATERWAYS = frozenset(("riverbank", "river", "canal"))
_RELATION_KEYS = ("natural", "waterway", "landuse", "building")
# Tags nothing in the app reads: import bookkeeping and translations (they are most of the
# text in Ontario's GeoBase/CanVec-imported roads).
_DROP_TAG_PREFIXES = (
    "source", "note", "fixme", "FIXME", "check_date", "created_by", "attribution", "tiger:",
    "geobase:", "canvec:", "nhn:", "nrn:", "statscan:", "lio:", "wikidata", "wikipedia",
    "name:", "old_name", "alt_name:", "survey:", "mapillary",
)  # fmt: skip
_BUILDING_TAGS = ("building", "height", "min_height", "name")
_BUILDING_TAG_PREFIXES = ("building:", "roof:")
_REGION_RE = re.compile(r"^[a-z][a-z0-9-]{0,40}$")
_SQL_CHUNK = 500  # ids per IN (...) (SQLite allows 999 variables in old builds)
BATCH_WAYS = 20_000  # ways encoded and written per batch
NEAR_STEP_M = 300.0  # sample spacing along a query line when collecting its tiles

_drivable_re = re.compile(DRIVABLE_HIGHWAY_RE)
_paved_re = re.compile(PAVED_SURFACE_RE)
_excluded_service_re = re.compile(EXCLUDED_SERVICE_RE)


class BuildCancelled(Exception):
    """`build_db` was cancelled (its `cancel` event was set)."""


# ---------------------------------------------------------------------------------------------
# Tags and tiles
# ---------------------------------------------------------------------------------------------


def is_drivable(tags: Mapping[str, str], include_tracks: str = "paved") -> bool:
    """The Overpass QL filter of `overpass._way_statements` in Python: a drivable road class
    (or a track per `include_tracks`), not an area, not a parking aisle / driveway."""
    if tags.get("area") == "yes":
        return False
    service = tags.get("service")
    if service is not None and _excluded_service_re.search(service):
        return False
    hw = tags.get("highway")
    if hw is None:
        return False
    if _drivable_re.search(hw):
        return True
    if hw != "track" or include_tracks == "none":
        return False
    return include_tracks == "all" or bool(_paved_re.search(tags.get("surface", "")))


def _way_kinds(tags: Mapping[str, str]) -> list[int]:
    kinds = []
    if tags.get("highway") in _ROAD_HIGHWAYS:
        kinds.append(KIND_ROAD)
    if tags.get("building", "no") != "no":
        kinds.append(KIND_BUILDING)
    if tags.get("barrier") in _BARRIERS:
        kinds.append(KIND_BARRIER)
    natural = tags.get("natural")
    if (
        natural == "water"
        or tags.get("waterway") in _WATERWAYS
        or tags.get("landuse") == "reservoir"
    ):
        kinds.append(KIND_WATER)
    if natural in _ROCKS:
        kinds.append(KIND_ROCK)
    return kinds


def _relation_kind(tags: Mapping[str, str]) -> int | None:
    if (
        tags.get("natural") == "water"
        or tags.get("waterway") == "riverbank"
        or tags.get("landuse") == "reservoir"
    ):
        return KIND_WATER
    if tags.get("building", "no") != "no" and tags.get("type") == "multipolygon":
        return KIND_BUILDING
    return None


def _trim_tags(tags: Mapping[str, str], kind: int) -> dict[str, str]:
    if kind == KIND_BUILDING:
        return {
            k: v
            for k, v in tags.items()
            if k in _BUILDING_TAGS or k.startswith(_BUILDING_TAG_PREFIXES)
        }
    return {k: v for k, v in tags.items() if not k.startswith(_DROP_TAG_PREFIXES)}


def _tile_float(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat_r = np.radians(np.clip(lat, -85.05112878, 85.05112878))
    x = (lon + 180.0) / 360.0 * _N
    y = (1.0 - np.log(np.tan(lat_r) + 1.0 / np.cos(lat_r)) / math.pi) / 2.0 * _N
    return x, y


def _tile_cells(lonlat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = _tile_float(lonlat[:, 0], lonlat[:, 1])
    return (
        np.clip(np.floor(x), 0, _N - 1).astype(np.int64),
        np.clip(np.floor(y), 0, _N - 1).astype(np.int64),
    )


def tile_box(x: int, y: int) -> tuple[float, float, float, float]:
    """(west, south, east, north) degrees of a z14 tile."""

    def lat(yy: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * yy / _N))))

    return x / _N * 360.0 - 180.0, lat(y + 1), (x + 1) / _N * 360.0 - 180.0, lat(y)


def _segment_tiles(cx: np.ndarray, cy: np.ndarray, owner: np.ndarray) -> np.ndarray:
    """Unique (owner, tile key) rows for points in cells (cx, cy) belonging to `owner` (points
    of one owner consecutive): every tile a segment's bounding box touches."""
    if len(cx) == 0:
        return np.zeros((0, 2), np.int64)
    same = owner[:-1] == owner[1:]
    lo_x = np.minimum(cx[:-1], cx[1:])[same]
    hi_x = np.maximum(cx[:-1], cx[1:])[same]
    lo_y = np.minimum(cy[:-1], cy[1:])[same]
    hi_y = np.maximum(cy[:-1], cy[1:])[same]
    seg_owner = owner[:-1][same]
    simple = (lo_x == hi_x) & (lo_y == hi_y)
    keys = [np.stack([owner, cx * _N + cy], axis=1)]  # every point's own tile
    if (~simple).any():
        extra = []
        for o, x0, x1, y0, y1 in zip(
            seg_owner[~simple].tolist(),
            lo_x[~simple].tolist(),
            hi_x[~simple].tolist(),
            lo_y[~simple].tolist(),
            hi_y[~simple].tolist(),
            strict=True,
        ):
            extra += [(o, x * _N + y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
        keys.append(np.asarray(extra, np.int64).reshape(-1, 2))
    return np.unique(np.concatenate(keys), axis=0)


def way_tiles(lonlat: np.ndarray) -> set[tuple[int, int]]:
    """The z14 (x, y) tiles a way's segments (their bounding boxes) touch."""
    cx, cy = _tile_cells(np.asarray(lonlat, np.float64).reshape(-1, 2))
    rows = _segment_tiles(cx, cy, np.zeros(len(cx), np.int64))
    return {(int(k) // _N, int(k) % _N) for k in rows[:, 1]}


def _tiles_for_box(west: float, south: float, east: float, north: float) -> set[int]:
    (x0, x1), (y1, y0) = _tile_cells(np.array([[west, south], [east, north]]))
    return {x * _N + y for x in range(int(x0), int(x1) + 1) for y in range(int(y0), int(y1) + 1)}


def _tiles_along(lonlat: np.ndarray, pad_m: float) -> set[int]:
    """Tile keys of every z14 tile within ~`pad_m` of the (N, 2) lon/lat line."""
    proj = LocalProjection(float(lonlat[0, 1]), float(lonlat[0, 0]))
    xy = proj.to_xy(lonlat)
    pts = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:], strict=True):
        n = max(1, math.ceil(float(np.hypot(*(b - a))) / NEAR_STEP_M))
        pts.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
    ll = proj.from_xy(np.concatenate(pts))
    pad = pad_m + NEAR_STEP_M
    dlat = pad / 111_320.0
    dlon = dlat / np.maximum(np.cos(np.radians(ll[:, 1])), 0.05)
    lo = np.stack([ll[:, 0] - dlon, ll[:, 1] - dlat], axis=1)
    hi = np.stack([ll[:, 0] + dlon, ll[:, 1] + dlat], axis=1)
    x0, y1 = _tile_cells(lo)
    x1, y0 = _tile_cells(hi)
    out: set[int] = set()
    for a, b, c, d in zip(x0.tolist(), x1.tolist(), y0.tolist(), y1.tolist(), strict=True):
        out.update(x * _N + y for x in range(a, b + 1) for y in range(c, d + 1))
    return out


# ---------------------------------------------------------------------------------------------
# Region boundary (.poly)
# ---------------------------------------------------------------------------------------------


def parse_poly(text: str) -> shapely.Geometry:
    """An Osmosis polygon file (Geofabrik's `<region>.poly`) -> shapely (Multi)Polygon in
    lon/lat. Sections whose name starts with "!" are holes."""
    lines = [ln.strip() for ln in text.splitlines()]
    if not lines:
        raise ValueError("empty polygon file")
    outers: list[list[tuple[float, float]]] = []
    holes: list[list[tuple[float, float]]] = []
    i = 1  # line 0 is the file's name
    while i < len(lines):
        name = lines[i]
        i += 1
        if not name:
            continue
        if name == "END":
            break
        ring: list[tuple[float, float]] = []
        while i < len(lines) and lines[i] != "END":
            parts = lines[i].split()
            if len(parts) >= 2:
                ring.append((float(parts[0]), float(parts[1])))
            i += 1
        i += 1
        if len(ring) >= 3:
            (holes if name.startswith("!") else outers).append(ring)
    if not outers:
        raise ValueError("no polygon in the .poly file")
    polys = [shapely.Polygon(r) for r in outers]
    geom = shapely.union_all([p.buffer(0) for p in polys])
    for h in holes:
        geom = geom.difference(shapely.Polygon(h).buffer(0))
    return geom


# ---------------------------------------------------------------------------------------------
# Building the database (pyosmium)
# ---------------------------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE features (
    fid INTEGER PRIMARY KEY,
    kind INTEGER NOT NULL,
    osm_id INTEGER NOT NULL,
    rel_id INTEGER,
    role TEXT,
    closed INTEGER NOT NULL,
    tags TEXT NOT NULL,
    nodes BLOB,
    coords BLOB NOT NULL
);
CREATE TABLE tiles (
    kind INTEGER NOT NULL,
    tile INTEGER NOT NULL,
    fid INTEGER NOT NULL,
    PRIMARY KEY (kind, tile, fid)
) WITHOUT ROWID;
"""


@dataclass
class BuildStats:
    """What `build_db` wrote."""

    path: Path
    counts: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    size_bytes: int = 0
    osm_base: str | None = None


def _enc_xy(xy: np.ndarray) -> bytes:
    d = xy.astype(np.int64)
    d[1:] -= xy[:-1]
    return zlib.compress(d.astype("<i4").tobytes(), 6)


def _enc_ids(ids: Sequence[int]) -> bytes:
    a = np.asarray(ids, np.int64)
    a[1:] -= a[:-1].copy()
    return zlib.compress(a.astype("<i8").tobytes(), 6)


def _dec_xy(blob: bytes) -> np.ndarray:
    d = np.frombuffer(zlib.decompress(blob), "<i4").reshape(-1, 2)
    return np.cumsum(d, axis=0, dtype=np.int64) / 1e7


def _dec_ids(blob: bytes | None) -> tuple[int, ...]:
    if not blob:
        return ()
    return tuple(np.cumsum(np.frombuffer(zlib.decompress(blob), "<i8")).tolist())


class _Writer:
    """Batches feature rows: encodes them, finds their tiles and inserts them."""

    def __init__(self, con: sqlite3.Connection) -> None:
        self.con = con
        self.batch: list[tuple] = []  # (kind, osm_id, rel_id, role, closed, tags, ids, xy)
        self.next_fid = 1
        self.tile_keys: list[np.ndarray] = []  # (kind << 60 | tile << 32 | fid) per row
        self.counts: dict[int, int] = {}

    def add(self, kind, osm_id, rel_id, role, closed, tags, ids, xy) -> None:
        self.batch.append((kind, osm_id, rel_id, role, closed, tags, ids, xy))
        if len(self.batch) >= BATCH_WAYS:
            self.flush()

    def flush(self) -> None:
        if not self.batch:
            return
        first = self.next_fid
        xy_all = np.concatenate([b[7] for b in self.batch])
        owner = np.repeat(np.arange(len(self.batch)), [len(b[7]) for b in self.batch])
        cx, cy = _tile_cells(xy_all / 1e7)
        pairs = _segment_tiles(cx, cy, owner)
        kinds = np.array([b[0] for b in self.batch], np.int64)
        fids = first + pairs[:, 0]
        self.tile_keys.append((kinds[pairs[:, 0]] << 60) | (pairs[:, 1] << 32) | fids)
        rows = []
        for i, (kind, osm_id, rel_id, role, closed, tags, ids, xy) in enumerate(self.batch):
            nodes = _enc_ids(ids) if ids is not None else None
            tag_text = json.dumps(tags, separators=(",", ":"), ensure_ascii=False)
            rows.append((first + i, kind, osm_id, rel_id, role, int(closed), tag_text, nodes,
                         _enc_xy(xy)))  # fmt: skip
            self.counts[kind] = self.counts.get(kind, 0) + 1
        self.con.executemany("INSERT INTO features VALUES (?,?,?,?,?,?,?,?,?)", rows)
        self.next_fid += len(self.batch)
        self.batch = []

    def write_tiles(self) -> None:
        self.flush()
        if not self.tile_keys:
            return
        keys = np.sort(np.concatenate(self.tile_keys))
        self.tile_keys = []
        mask32 = (1 << 32) - 1
        for i in range(0, len(keys), 200_000):
            part = keys[i : i + 200_000]
            self.con.executemany(
                "INSERT INTO tiles VALUES (?,?,?)",
                zip(
                    (part >> 60).tolist(),
                    ((part >> 32) & ((1 << 28) - 1)).tolist(),
                    (part & mask32).tolist(),
                    strict=True,
                ),
            )


ProgressFn = Callable[[float, str], None]


class _Ticker:
    """Progress of the second pass, which Python mostly doesn't see (libosmium reads the nodes):
    by time while nodes are read (expected duration from the file size), then by way id."""

    def __init__(self, progress: ProgressFn | None, size_mb: float, max_way_id: int) -> None:
        self.progress = progress
        self.t0 = time.monotonic()
        self.node_s = max(5.0, size_mb * NODE_SECONDS_PER_MB)
        self.max_way = max(1, max_way_id)
        self.last = 0.0

    def nodes(self) -> None:
        if self.progress is None:
            return
        t = time.monotonic() - self.t0
        f = 1.0 - math.exp(-t / self.node_s)  # approaches 1 as nodes take longer than expected
        self._send(PASS1_END + (NODES_END - PASS1_END) * f, "Processing roads")

    def way(self, way_id: int) -> None:
        if self.progress is None:
            return
        f = min(1.0, way_id / self.max_way)
        self._send(NODES_END + (WAYS_END - NODES_END) * f, "Processing roads")

    def _send(self, value: float, text: str) -> None:
        if value >= self.last + 0.002:
            self.last = value
            self.progress(value, text)


# Shares of the build in the progress fraction (measured on Ontario, 2026-10-16).
PASS1_END, NODES_END, WAYS_END = 0.08, 0.45, 0.93
NODE_SECONDS_PER_MB = 0.06  # ASSUMED from the sandbox run; only shapes the progress bar


def _header_timestamp(path: Path) -> str | None:
    import osmium

    try:
        header = osmium.io.Reader(str(path), osmium.osm.NOTHING).header()
        ts = header.get("osmosis_replication_timestamp") or header.get("timestamp")
    except Exception:  # noqa: BLE001 - only metadata
        return None
    return str(ts) if ts else None


def build_db(
    source: Path | str,
    out: Path | str,
    *,
    region: str,
    poly: shapely.Geometry | str | None = None,
    source_name: str = "",
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
    location_index: str = "flex_mem",
) -> BuildStats:
    """Read an OSM file (.osm.pbf, .osm, ...) and write the region database `out`.

    `poly` is the extract's boundary (shapely geometry or .poly text); without it the data's
    own bounding box is used (only safe for a box-shaped extract). `progress(fraction, text)`
    is called now and then; setting `cancel` stops at the next check with BuildCancelled (the
    half-written file is removed). The file is written next to `out` and moved there at the end.
    """
    import osmium

    t0 = time.monotonic()
    source, out = Path(source), Path(out)
    if not _REGION_RE.match(region):
        raise ValueError(f"bad region name {region!r}")
    if isinstance(poly, str):
        poly = parse_poly(poly)
    tmp = out.with_name(out.name + ".building")
    tmp.unlink(missing_ok=True)
    size_mb = source.stat().st_size / 1e6

    def check() -> None:
        if cancel is not None and cancel.is_set():
            raise BuildCancelled()

    def say(f: float, text: str) -> None:
        if progress is not None:
            progress(f, text)

    con = sqlite3.connect(tmp)
    try:
        con.executescript(
            "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-262144;"
            "PRAGMA temp_store=FILE;" + _SCHEMA
        )
        # Pass 1: relations -> which ways are members of a kept multipolygon.
        say(0.0, "Processing roads (relations)")
        rel_tags: dict[int, tuple[int, dict[str, str]]] = {}
        members: dict[int, list[tuple[int, str]]] = {}
        max_way_id = 0
        relations = osmium.FileProcessor(str(source), osmium.osm.RELATION).with_filter(
            osmium.filter.KeyFilter(*_RELATION_KEYS)
        )
        for k, rel in enumerate(relations):
            if k % 2048 == 0:
                check()
            tags = {t.k: t.v for t in rel.tags}
            kind = _relation_kind(tags)
            if kind is None:
                continue
            rel_tags[rel.id] = (kind, _trim_tags(tags, kind))
            for m in rel.members:
                if m.type == "w":
                    members.setdefault(m.ref, []).append((rel.id, m.role))
                    max_way_id = max(max_way_id, m.ref)
        check()
        say(PASS1_END, "Processing roads")

        # Pass 2: nodes (into libosmium's location index) and ways. Tagged highway nodes come
        # through as a heartbeat (progress and cancel) while the nodes are read.
        ticker = _Ticker(progress, size_mb, max_way_id)
        writer = _Writer(con)
        heartbeat = osmium.filter.KeyFilter("highway")
        heartbeat.enable_for(osmium.osm.NODE)
        stream = (
            osmium.FileProcessor(str(source), osmium.osm.NODE | osmium.osm.WAY)
            .with_locations(location_index)
            .with_filter(heartbeat)
        )
        west = south = math.inf
        east = north = -math.inf
        seen = 0
        for obj in stream:
            seen += 1
            if obj.is_node():
                if seen % 1024 == 0:
                    check()
                    ticker.nodes()
                continue
            if seen % 4096 == 0:
                check()
                ticker.way(obj.id)
            rels = members.get(obj.id)
            tags = None
            kinds: list[int] = []
            if len(obj.tags):
                tags = {t.k: t.v for t in obj.tags}
                kinds = _way_kinds(tags)
            if not kinds and not rels:
                continue
            ids: list[int] = []
            pts: list[tuple[int, int]] = []
            for n in obj.nodes:
                loc = n.location
                if loc.valid():  # nodes outside a clipped extract have no location
                    ids.append(n.ref)
                    pts.append((loc.x, loc.y))
            if len(pts) < 2:
                continue
            xy = np.asarray(pts, np.int64)
            closed = len(ids) > 3 and ids[0] == ids[-1]
            if poly is None:
                west, east = min(west, xy[:, 0].min() / 1e7), max(east, xy[:, 0].max() / 1e7)
                south, north = min(south, xy[:, 1].min() / 1e7), max(north, xy[:, 1].max() / 1e7)
            for kind in kinds:
                keep_ids = ids if kind != KIND_BUILDING else None
                writer.add(kind, obj.id, None, None, closed, _trim_tags(tags, kind), keep_ids, xy)
            for rel_id, role in rels or ():
                kind, rtags = rel_tags[rel_id]
                writer.add(kind, obj.id, rel_id, role, closed, rtags, None, xy)
        check()
        say(WAYS_END, "Indexing")
        writer.write_tiles()
        con.execute("CREATE INDEX features_rel ON features(rel_id) WHERE rel_id IS NOT NULL")
        if poly is None:
            if not math.isfinite(west):
                raise ValueError(f"no usable OSM data in {source.name}")
            poly = shapely.box(west, south, east, north)
        counts = {KIND_NAMES[k]: v for k, v in sorted(writer.counts.items())}
        osm_base = _header_timestamp(source)
        meta = {
            "format": str(FORMAT),
            "region": region,
            "osm_base": osm_base or "",
            "source": source_name or source.name,
            "built": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "poly": shapely.to_wkt(poly, rounding_precision=7),
            "counts": json.dumps(counts),
        }
        con.executemany("INSERT INTO meta VALUES (?,?)", meta.items())
        con.commit()
        check()
    except BaseException:
        con.close()
        tmp.unlink(missing_ok=True)
        raise
    con.close()
    with exclusive():
        release(out)
        os.replace(tmp, out)
    say(1.0, "Ready")
    return BuildStats(out, counts, time.monotonic() - t0, out.stat().st_size, osm_base)


# ---------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------


@dataclass
class _Row:
    fid: int
    kind: int
    osm_id: int
    rel_id: int | None
    role: str | None
    closed: bool
    tags: dict[str, str]
    nodes: tuple[int, ...]
    lonlat: np.ndarray


def _geom_json(lonlat: np.ndarray, keep: np.ndarray | None = None) -> list[dict | None]:
    out: list[dict | None] = []
    for k, (lon, lat) in enumerate(lonlat.tolist()):
        out.append({"lat": lat, "lon": lon} if keep is None or keep[k] else None)
    return out


class OfflineDB:
    """One region database, opened read-only. Thread-safe (one connection behind a lock)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        uri = self.path.resolve().as_uri() + "?mode=ro"
        self._con = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._lock = threading.Lock()
        try:
            meta = dict(self._con.execute("SELECT key, value FROM meta").fetchall())
            if meta.get("format") != str(FORMAT):
                raise ValueError(f"unsupported offline database format {meta.get('format')!r}")
            self.region = meta["region"]
            self.osm_base = meta.get("osm_base") or None
            self.source = meta.get("source", "")
            self.built = meta.get("built", "")
            self.counts = json.loads(meta.get("counts") or "{}")
            self.poly = shapely.from_wkt(meta["poly"])
        except Exception:
            self._con.close()
            raise
        shapely.prepare(self.poly)
        self._covered: dict[int, bool] = {}

    def close(self) -> None:
        with self._lock:
            self._con.close()

    @property
    def cache_key(self) -> str:
        return f"offline:{self.region}:{self.osm_base or self.built}"

    # -- low level -----------------------------------------------------------------------------

    def covers(self, tiles: Iterable[int]) -> bool:
        """Every tile (key x * 16384 + y) lies inside the extract's boundary."""
        todo = [t for t in set(tiles) if t not in self._covered]
        if todo:
            boxes = shapely.box(*np.array([tile_box(t // _N, t % _N) for t in todo]).T)
            inside = shapely.contains(self.poly, boxes)
            self._covered.update(zip(todo, inside.tolist(), strict=True))
        return all(self._covered[t] for t in set(tiles))

    def _fids(self, kind: int, tiles: Iterable[int]) -> list[int]:
        tiles = sorted(set(tiles))
        out: set[int] = set()
        with self._lock:
            for i in range(0, len(tiles), _SQL_CHUNK):
                part = tiles[i : i + _SQL_CHUNK]
                q = f"SELECT fid FROM tiles WHERE kind=? AND tile IN ({','.join('?' * len(part))})"
                out.update(r[0] for r in self._con.execute(q, (kind, *part)))
        return sorted(out)

    def _rows(self, fids: Sequence[int] = (), *, rel_kind: int = 0, rels=()) -> list[_Row]:
        cols = "fid, kind, osm_id, rel_id, role, closed, tags, nodes, coords"
        if rels:
            ids, where = sorted(set(rels)), f"kind={int(rel_kind)} AND rel_id IN"
        else:
            ids, where = list(fids), "fid IN"
        raw = []
        with self._lock:
            for i in range(0, len(ids), _SQL_CHUNK):
                part = ids[i : i + _SQL_CHUNK]
                q = f"SELECT {cols} FROM features WHERE {where} ({','.join('?' * len(part))})"
                raw += self._con.execute(q, part).fetchall()
        return [
            _Row(f, k, o, r, role, bool(c), json.loads(t), _dec_ids(n), _dec_xy(xy))
            for f, k, o, r, role, c, t, n, xy in raw
        ]

    @staticmethod
    def _near(rows: Sequence[_Row], line_ll: np.ndarray, dist_m: float) -> list[_Row]:
        """The rows within `dist_m` of the lon/lat line (a point when it has one vertex)."""
        if not rows:
            return []
        proj = LocalProjection(float(line_ll[0, 1]), float(line_ll[0, 0]))
        xy = proj.to_xy(line_ll)
        target = shapely.LineString(xy) if len(xy) >= 2 else shapely.Point(xy[0])
        coords = proj.to_xy(np.concatenate([r.lonlat for r in rows]))
        idx = np.repeat(np.arange(len(rows)), [len(r.lonlat) for r in rows])
        lines = shapely.linestrings(coords, indices=idx)
        keep = shapely.dwithin(lines, target, dist_m)
        return [r for r, k in zip(rows, keep.tolist(), strict=True) if k]

    def _osm(self, rows: Iterable[_Row]) -> OsmData:
        ways = {r.osm_id: Way(r.osm_id, r.nodes, r.lonlat, r.tags) for r in rows}
        return OsmData(tuple(ways[k] for k in sorted(ways)), self.osm_base, self.cache_key)

    # -- queries -------------------------------------------------------------------------------

    def tile_ways(self, x: int, y: int, include_tracks: str = "paved") -> list[dict] | None:
        """The drivable ways crossing z14 tile (x, y) as `app.tiles` way records ({id, nodes,
        geom [(lon, lat)], tags}), sorted by id; None if the tile isn't covered."""
        key = x * _N + y
        if not self.covers([key]):
            return None
        rows = [r for r in self._rows(self._fids(KIND_ROAD, [key])) if r.nodes]
        out = {}
        for r in rows:
            if is_drivable(r.tags, include_tracks):
                geom = [(lon, lat) for lon, lat in np.round(r.lonlat, 7).tolist()]
                out[r.osm_id] = {"id": r.osm_id, "nodes": list(r.nodes), "geom": geom,
                                 "tags": r.tags}  # fmt: skip
        return [out[k] for k in sorted(out)]

    def corridor(
        self, pins: Sequence[tuple[float, float]], buffer_m: float, include_tracks: str = "paved"
    ) -> OsmData | None:
        """Drivable ways within `buffer_m` of the (lat, lon) pin chain (Overpass `around:`)."""
        line = np.array([(lon, lat) for lat, lon in pins], np.float64)
        tiles = _tiles_along(line, buffer_m)
        if not self.covers(tiles):
            return None
        rows = [
            r
            for r in self._rows(self._fids(KIND_ROAD, tiles))
            if r.nodes and is_drivable(r.tags, include_tracks)
        ]
        return self._osm(self._near(rows, line, buffer_m))

    def barrier_ways(
        self, route_latlon: np.ndarray, around_m: float, kinds: Sequence[str]
    ) -> OsmData | None:
        """Barrier ways of `kinds` within `around_m` of the (lat, lon) line."""
        if not set(kinds) <= _BARRIERS:
            return None
        line = np.asarray(route_latlon, np.float64).reshape(-1, 2)[:, ::-1]
        tiles = _tiles_along(line, around_m)
        if not self.covers(tiles):
            return None
        rows = [r for r in self._rows(self._fids(KIND_BARRIER, tiles)) if r.rel_id is None]
        rows = [r for r in rows if r.tags.get("barrier") in kinds]
        return self._osm(self._near(rows, line, around_m))

    def buildings_payload(self, boxes: Sequence[tuple[float, float, float, float]]) -> dict | None:
        """Overpass-JSON-shaped building ways and multipolygon relations crossing the
        (south, west, north, east) boxes (`buildings.build_buildings_query`)."""
        tiles: set[int] = set()
        for s_, w, n, e in boxes:
            tiles |= _tiles_for_box(w, s_, e, n)
        if not tiles or not self.covers(tiles):
            return None
        rows = self._rows(self._fids(KIND_BUILDING, tiles))
        area = shapely.union_all([shapely.box(w, s_, e, n) for s_, w, n, e in boxes])
        if rows:
            idx = np.repeat(np.arange(len(rows)), [len(r.lonlat) for r in rows])
            lines = shapely.linestrings(np.concatenate([r.lonlat for r in rows]), indices=idx)
            hit = shapely.intersects(lines, area)
            rows = [r for r, h in zip(rows, hit.tolist(), strict=True) if h]
        elements = [
            {"type": "way", "id": r.osm_id, "tags": r.tags, "geometry": _geom_json(r.lonlat)}
            for r in sorted({r.osm_id: r for r in rows if r.rel_id is None}.values(),
                            key=lambda r: r.osm_id)
        ]  # fmt: skip
        rels = {r.rel_id for r in rows if r.rel_id is not None}
        elements += self._relations(self._rows(rel_kind=KIND_BUILDING, rels=rels))
        return {"elements": elements, "osm3s": {"timestamp_osm_base": self.osm_base}}

    @staticmethod
    def _relations(rows: Iterable[_Row], clip: tuple | None = None) -> list[dict]:
        """Relation elements (`out geom`, clipped to (south, west, north, east) if given)."""
        by_rel: dict[int, list[_Row]] = {}
        for r in rows:
            by_rel.setdefault(r.rel_id, []).append(r)
        out = []
        for rel_id in sorted(by_rel):
            members = []
            for r in sorted(by_rel[rel_id], key=lambda r: r.fid):
                keep = None
                if clip is not None:
                    s_, w, n, e = clip
                    ll = r.lonlat
                    inside = (ll[:, 0] >= w) & (ll[:, 0] <= e) & (ll[:, 1] >= s_) & (ll[:, 1] <= n)
                    keep = inside.copy()  # and the first point outside, so edges stay whole
                    keep[1:] |= inside[:-1]
                    keep[:-1] |= inside[1:]
                    if not keep.any():
                        continue
                member = {"type": "way", "ref": r.osm_id, "role": r.role or ""}
                member["geometry"] = _geom_json(r.lonlat, keep)
                members.append(member)
            if members:
                tags = by_rel[rel_id][0].tags
                out.append({"type": "relation", "id": rel_id, "tags": tags, "members": members})
        return out

    def roadside_payload(
        self,
        route_latlon: np.ndarray,
        around_m: float,
        access_m: float,
        clip: tuple[float, float, float, float],
    ) -> dict | None:
        """Overpass-JSON-shaped answer to `barriers.build_roadside_query`: water and rock
        within `around_m`, joining highways within `access_m` of the (lat, lon) line, and
        water relations near it with their members clipped to `clip` (s, w, n, e)."""
        line = np.asarray(route_latlon, np.float64).reshape(-1, 2)[:, ::-1]
        reach = max(around_m, access_m)
        tiles = _tiles_along(line, reach + 100.0)
        if not self.covers(tiles):
            return None
        near_tiles = _tiles_along(line, reach)
        ways: list[_Row] = []
        for kind, dist in ((KIND_WATER, around_m), (KIND_ROCK, around_m), (KIND_ROAD, access_m)):
            rows = [r for r in self._rows(self._fids(kind, near_tiles)) if r.rel_id is None]
            ways += self._near(rows, line, dist)
        members = [r for r in self._rows(self._fids(KIND_WATER, tiles)) if r.rel_id is not None]
        near_rels = {r.rel_id for r in self._near(members, line, around_m)}
        elements = []
        for r in sorted(ways, key=lambda r: (r.osm_id, r.kind)):
            nodes = list(r.nodes)
            if not nodes:  # parse_roadside reads "closed" from the node ids: stand-ins
                nodes = [-(k + 1) for k in range(len(r.lonlat))]
                if r.closed:
                    nodes[-1] = nodes[0]
            elements.append({"type": "way", "id": r.osm_id, "nodes": nodes, "tags": r.tags,
                             "geometry": _geom_json(r.lonlat)})  # fmt: skip
        elements += self._relations([r for r in members if r.rel_id in near_rels], clip)
        return {"elements": elements, "osm3s": {"timestamp_osm_base": self.osm_base}}


# ---------------------------------------------------------------------------------------------
# Finding the databases
# ---------------------------------------------------------------------------------------------

_registry_lock = threading.RLock()
_configured: Path | None = None
_open: dict[Path, tuple[tuple[int, int], OfflineDB]] = {}


def configure(directory: Path | str | None) -> None:
    """Use `directory` for the region databases (the app: `<data dir>/offline`); None resets
    to `$DRIFTMAPS_OFFLINE_DIR` / `<app data>/offline`."""
    global _configured
    with _registry_lock:
        _configured = Path(directory) if directory is not None else None
        for _, db in _open.values():
            db.close()
        _open.clear()


def offline_dir() -> Path:
    if _configured is not None:
        return _configured
    env = os.environ.get(ENV_OFFLINE_DIR, "").strip()
    if env:
        return Path(env).expanduser()
    from driftmaps.app.settings import app_data_dir

    return app_data_dir() / "offline"


def db_path(region: str, directory: Path | str | None = None) -> Path:
    if not _REGION_RE.match(region):
        raise ValueError(f"bad region name {region!r}")
    return Path(directory if directory is not None else offline_dir()) / f"{region}{DB_SUFFIX}"


def release(path: Path | str) -> None:
    """Close a database so its file can be replaced or deleted (Windows keeps open files)."""
    with _registry_lock:
        entry = _open.pop(Path(path), None)
        if entry is not None:
            entry[1].close()


@contextmanager
def exclusive() -> Iterator[None]:
    """Hold off every lookup while files are swapped (no one reopens one meanwhile)."""
    with _registry_lock:
        yield


def databases(directory: Path | str | None = None) -> list[OfflineDB]:
    """The ready region databases (opened once and kept; reopened when the file changes)."""
    folder = Path(directory) if directory is not None else offline_dir()
    try:
        names = sorted(p for p in os.listdir(folder) if p.endswith(DB_SUFFIX))
    except OSError:
        return []
    out = []
    with _registry_lock:
        for name in names:
            path = folder / name
            if not _REGION_RE.match(name[: -len(DB_SUFFIX)]):
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            sig = (st.st_size, st.st_mtime_ns)
            entry = _open.get(path)
            if entry is not None and entry[0] == sig:
                out.append(entry[1])
                continue
            release(path)
            try:
                db = OfflineDB(path)
            except (sqlite3.Error, ValueError, KeyError, OSError) as exc:
                log.warning("ignoring the offline map database %s: %s", path, exc)
                continue
            _open[path] = (sig, db)
            out.append(db)
    return out


def _first(fn: Callable[[OfflineDB], object]):
    for db in databases():
        try:
            found = fn(db)
        except (sqlite3.Error, OSError, ValueError, zlib.error) as exc:
            log.warning("offline map database %s failed, using Overpass: %s", db.path, exc)
            continue
        if found is not None:
            return found
    return None


def corridor(
    pins: Sequence[tuple[float, float]], buffer_m: float, include_tracks: str = "paved"
) -> OsmData | None:
    """`overpass.fetch_corridor` from a region database, or None (not covered)."""
    if not pins:
        return None
    return _first(lambda db: db.corridor(pins, buffer_m, include_tracks))


def tile_ways(
    x: int, y: int, include_tracks: str = "paved"
) -> tuple[list[dict], str | None] | None:
    """(way records, osm_base) of a z14 tile from a region database, or None."""

    def one(db: OfflineDB) -> tuple[list[dict], str | None] | None:
        ways = db.tile_ways(x, y, include_tracks)
        return None if ways is None else (ways, db.osm_base)

    return _first(one)


def barrier_ways(route_latlon: np.ndarray, around_m: float, kinds: Sequence[str]) -> OsmData | None:
    return _first(lambda db: db.barrier_ways(route_latlon, around_m, kinds))


def buildings_payload(boxes: Sequence[tuple[float, float, float, float]]) -> dict | None:
    return _first(lambda db: db.buildings_payload(boxes))


def roadside_payload(
    route_latlon: np.ndarray,
    around_m: float,
    access_m: float,
    clip: tuple[float, float, float, float],
) -> dict | None:
    return _first(lambda db: db.roadside_payload(route_latlon, around_m, access_m, clip))
