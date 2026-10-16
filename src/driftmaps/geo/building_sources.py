"""More building footprints: NRCan lidar-extracted buildings and municipal footprints.

Research: the builders' buildings.md (2026-10-16). OSM and Microsoft stay in `geo.buildings`;
this module adds the two sources that make the house count and the shapes match real life:

- **NRCan Automatically Extracted Buildings, Optimized Layer** (Natural Resources Canada,
  OGL-Canada): lidar footprints with `heightmin` / `heightmax` (lowest and highest building
  point above ground) and the lidar dates. 98-100 % of the municipal buildings >= 40 m2 were
  found in it, and its `heightmax` matched Burlington's measured roof peaks to a median of
  +0.24 m. The layer is one 6.8 GB GeoPackage (SQLite, 4 KiB pages, R*-tree spatial index) on
  NRCan's download server, which answers HTTP range requests. `GeoPackage` reads it like
  SQLite would, page by page, with nothing but the standard library: the R*-tree finds the
  features in a box, then the feature table's B-tree finds their rows. A route needs a few
  hundred pages (one or two MB), fetched level by level in parallel, merged into ranges.
  VERIFIED 2026-10-16: the server answers 206 with a browser User-Agent and 403 to
  "Python-urllib" or "driftmaps", so `NRCAN_USER_AGENT` is a browser string (with our name).
  A per-project GeoPackage downloaded by hand (`auto_building/gpkg/`) is read the same way
  from disk (`GeoPackage(Path(...))`).
- **Municipal footprints** (ArcGIS REST, queried by envelope, 1-2 k records per page):
  exact roof outlines; Burlington and Milton also measured eave and peak heights, Burlington a
  building type. Each source has its own licence line (LICENSE-DATA.txt):
  - Oakville: Open Government Licence - Town of Oakville (v2).
  - Milton: Open Government Licence - Town of Milton (v2).
  - Burlington: the City of Burlington Open Data Terms of Use. They are worldwide and
    royalty-free but REVOCABLE, and anyone the data is passed to must get the terms' URL
    (`BURLINGTON.terms_url`, written into LICENSE-DATA.txt). It can be switched off
    (`--no-burlington-buildings`).

Every result is cached per query box under `<cache>/buildings/` (gzip JSON), so a rebuild
(or `offline=True`) needs no network. Everything here is lon/lat (EPSG:4326 / NAD83(CSRS),
the same to within ~1 m here); `geo.buildings.conflate` merges the sources.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely

from driftmaps.geo.buildings import Footprint
from driftmaps.geo.overpass import USER_AGENT, default_cache_dir

__all__ = [
    "BURLINGTON",
    "MILTON",
    "MUNICIPAL_SOURCES",
    "NRCAN_ATTRIBUTION",
    "NRCAN_GPKG_URL",
    "OAKVILLE",
    "GeoPackage",
    "MunicipalSource",
    "SourceError",
    "fetch_municipal",
    "fetch_nrcan",
    "municipal_feature",
    "nrcan_footprint",
]

log = logging.getLogger(__name__)

NRCAN_GPKG_URL = (
    "https://download-telecharger.services.geo.ca/pub/nrcan_rncan/extraction/auto_building/"
    "auto_building_opti_2/auto_building_opti_2.gpkg"
)
# NRCan's download server refuses non-browser clients (403, VERIFIED 2026-10-16).
NRCAN_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
NRCAN_ATTRIBUTION = (
    "Buildings (footprints and heights): Natural Resources Canada, Automatically Extracted "
    "Buildings (optimized layer). Contains information licensed under the Open Government "
    "Licence - Canada (https://open.canada.ca/en/open-government-licence-canada)."
)
CACHE_VERSION = 1


class SourceError(RuntimeError):
    """A building source could not be read (network, format, or not cached when offline)."""


# --------------------------------------------------------------------------------------------
# SQLite / GeoPackage pages over HTTP range requests (or from a local file)
# --------------------------------------------------------------------------------------------


class _Pages:
    """Fixed-size pages of a file, fetched in batches; counts what was read."""

    page_size: int = 4096

    def __init__(self) -> None:
        self.cache: dict[int, bytes] = {}
        self.bytes_read = 0
        self.requests = 0
        self._lock = threading.Lock()

    def _read_range(self, offset: int, length: int) -> bytes:
        raise NotImplementedError

    def head(self, n: int = 100) -> bytes:
        return self._read_range(0, n)

    def get(self, pages: Iterable[int]) -> dict[int, bytes]:
        """{page number (1-based): bytes} for the pages (cached ones are not refetched)."""
        pages = [int(p) for p in pages]
        want = sorted(set(pages) - self.cache.keys())
        if want:
            runs: list[tuple[int, int]] = []  # (first page, count)
            for p in want:
                if runs and p == runs[-1][0] + runs[-1][1] and runs[-1][1] < self.max_run:
                    runs[-1] = (runs[-1][0], runs[-1][1] + 1)
                else:
                    runs.append((p, 1))
            ps = self.page_size

            def fetch(run: tuple[int, int]) -> tuple[int, bytes]:
                first, count = run
                return first, self._read_range((first - 1) * ps, count * ps)

            if len(runs) == 1 or self.workers <= 1:
                done = [fetch(r) for r in runs]
            else:
                with ThreadPoolExecutor(min(self.workers, len(runs))) as pool:
                    done = list(pool.map(fetch, runs))
            with self._lock:
                for (first, count), (_, data) in zip(runs, done, strict=True):
                    if len(data) < count * ps:
                        raise SourceError(f"short read at page {first}: {len(data)} bytes")
                    for k in range(count):
                        self.cache[first + k] = data[k * ps : (k + 1) * ps]
        return {p: self.cache[p] for p in pages}

    max_run = 32
    workers = 8


class FilePages(_Pages):
    """Pages of a local file."""

    def __init__(self, path: Path | str) -> None:
        super().__init__()
        self.path = Path(path)
        self.workers = 1

    def _read_range(self, offset: int, length: int) -> bytes:
        with self.path.open("rb") as fh:
            fh.seek(offset)
            data = fh.read(length)
        self.bytes_read += len(data)
        self.requests += 1
        return data


Opener = Callable[[urllib.request.Request], object]


class HttpPages(_Pages):
    """Pages of a remote file by HTTP range requests (parallel, merged runs, retried)."""

    def __init__(
        self,
        url: str,
        *,
        user_agent: str = NRCAN_USER_AGENT,
        timeout_s: float = 60.0,
        retries: int = 3,
        workers: int = 16,
        opener: Opener | None = None,
    ) -> None:
        super().__init__()
        self.url = url
        self.user_agent = user_agent
        self.timeout_s = timeout_s
        self.retries = retries
        self.workers = workers
        self.opener = opener or (lambda req: urllib.request.urlopen(req, timeout=timeout_s))

    def _read_range(self, offset: int, length: int) -> bytes:
        headers = {"User-Agent": self.user_agent, "Range": f"bytes={offset}-{offset + length - 1}"}
        err: Exception | None = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers=headers)
                with self.opener(req) as resp:  # type: ignore[attr-defined]
                    status = getattr(resp, "status", 206)
                    data = resp.read()
                if status != 206 and len(data) != length:
                    raise SourceError(f"server ignored the range request (HTTP {status})")
                with self._lock:
                    self.bytes_read += len(data)
                    self.requests += 1
                return data[:length]
            except urllib.error.HTTPError as e:
                err = e
                if e.code < 500 and e.code != 429:
                    break
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                err = e
            time.sleep(0.5 * 2**attempt)
        raise SourceError(f"range request failed ({self.url}): {err}")


def _varint(b: bytes, i: int) -> tuple[int, int]:
    """SQLite big-endian varint at b[i:]: (value, next index)."""
    v = 0
    for k in range(8):
        c = b[i + k]
        v = (v << 7) | (c & 0x7F)
        if c < 0x80:
            return v, i + k + 1
    return (v << 8) | b[i + 8], i + 9


def _record(payload: bytes) -> list[object]:
    """The column values of an SQLite record."""
    hsize, i = _varint(payload, 0)
    types = []
    while i < hsize:
        t, i = _varint(payload, i)
        types.append(t)
    pos = hsize
    out: list[object] = []
    for t in types:
        if t == 0:
            out.append(None)
        elif t <= 6:
            n = (1, 2, 3, 4, 6, 8)[t - 1]
            out.append(int.from_bytes(payload[pos : pos + n], "big", signed=True))
            pos += n
        elif t == 7:
            out.append(struct.unpack_from(">d", payload, pos)[0])
            pos += 8
        elif t in (8, 9):
            out.append(t - 8)
        elif t >= 12:
            n = (t - 12) // 2
            raw = bytes(payload[pos : pos + n])
            pos += n
            out.append(raw.decode("utf-8", "replace") if t & 1 else raw)
        else:
            raise SourceError(f"bad SQLite serial type {t}")
    return out


class _Btree:
    """Read-only SQLite table B-trees over `_Pages` (file format: sqlite.org/fileformat.html)."""

    def __init__(self, pages: _Pages) -> None:
        head = pages.head(100)
        if head[:16] != b"SQLite format 3\x00":
            raise SourceError("not an SQLite file")
        ps = int.from_bytes(head[16:18], "big")
        pages.page_size = 65536 if ps == 1 else ps
        self.pages = pages
        self.usable = pages.page_size - head[20]

    def _cells(self, pgno: int, page: bytes) -> tuple[int, list[int], int]:
        off = 100 if pgno == 1 else 0
        kind = page[off]
        if kind not in (5, 13):
            raise SourceError(f"page {pgno} is not a table B-tree page (type {kind})")
        n = int.from_bytes(page[off + 3 : off + 5], "big")
        hdr = 12 if kind == 5 else 8
        right = int.from_bytes(page[off + 8 : off + 12], "big") if kind == 5 else 0
        ptrs = [int.from_bytes(page[off + hdr + 2 * k : off + hdr + 2 * k + 2], "big")
                for k in range(n)]  # fmt: skip
        return kind, ptrs, right

    def _leaf_cell(self, page: bytes, ptr: int) -> tuple[int, bytes, int, int]:
        """(rowid, local payload, total payload size, first overflow page or 0)."""
        size, i = _varint(page, ptr)
        rowid, i = _varint(page, i)
        if rowid >= 1 << 63:
            rowid -= 1 << 64
        u = self.usable
        x = u - 35
        if size <= x:
            return rowid, page[i : i + size], size, 0
        m = ((u - 12) * 32 // 255) - 23
        k = m + (size - m) % (u - 4)
        local = k if k <= x else m
        ovf = int.from_bytes(page[i + local : i + local + 4], "big")
        return rowid, page[i : i + local], size, ovf

    def _overflow(self, local: bytes, size: int, first: int) -> bytes:
        parts = [local]
        got = len(local)
        pg = first
        while pg and got < size:
            page = self.pages.get([pg])[pg]
            chunk = page[4 : min(self.usable, 4 + size - got)]
            parts.append(chunk)
            got += len(chunk)
            pg = int.from_bytes(page[:4], "big")
        return b"".join(parts)

    def lookup(self, root: int, rowids: Iterable[int]) -> dict[int, bytes]:
        """{rowid: record payload} for the rowids found in the table rooted at `root`. All
        keys descend together, one batch of page reads per tree level."""
        want: dict[int, list[int]] = {root: sorted(set(rowids))}
        out: dict[int, bytes] = {}
        while want:
            got = self.pages.get(want)
            nxt: dict[int, list[int]] = {}
            for pgno, keys in want.items():
                page = got[pgno]
                kind, ptrs, right = self._cells(pgno, page)
                if kind == 5:
                    bounds, children = [], []
                    for p in ptrs:
                        children.append(int.from_bytes(page[p : p + 4], "big"))
                        bounds.append(_varint(page, p + 4)[0])
                    j = 0
                    for key in keys:  # sorted: walk the cells once
                        while j < len(bounds) and key > bounds[j]:
                            j += 1
                        child = children[j] if j < len(bounds) else right
                        nxt.setdefault(child, []).append(key)
                else:
                    wanted = set(keys)
                    for p in ptrs:
                        rowid, local, size, ovf = self._leaf_cell(page, p)
                        if rowid in wanted:
                            out[rowid] = self._overflow(local, size, ovf) if ovf else local
            want = nxt
        return out

    def scan(self, root: int) -> list[tuple[int, bytes]]:
        """Every (rowid, payload) of a (small) table, in rowid order."""
        level = [root]
        rows: list[tuple[int, bytes]] = []
        while level:
            got = self.pages.get(level)
            nxt = []
            for pgno in level:
                page = got[pgno]
                kind, ptrs, right = self._cells(pgno, page)
                if kind == 5:
                    nxt += [int.from_bytes(page[p : p + 4], "big") for p in ptrs] + [right]
                else:
                    for p in ptrs:
                        rowid, local, size, ovf = self._leaf_cell(page, p)
                        rows.append((rowid, self._overflow(local, size, ovf) if ovf else local))
            level = nxt
        return sorted(rows, key=lambda r: r[0])


def _columns(sql: str) -> list[str]:
    """Column names of a CREATE TABLE statement (constraints skipped)."""
    body = sql[sql.index("(") + 1 : sql.rindex(")")]
    parts, depth, cur = [], 0, []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    names = []
    for part in parts:
        tok = part.strip()
        if not tok:
            continue
        if tok.split()[0].upper() in ("CONSTRAINT", "PRIMARY", "UNIQUE", "CHECK", "FOREIGN"):
            continue
        if tok[0] in "\"'`[":
            close = {"[": "]"}.get(tok[0], tok[0])
            names.append(tok[1 : tok.index(close, 1)])
        else:
            names.append(tok.split()[0])
    return names


def _gpkg_wkb(blob: bytes) -> bytes:
    """The WKB inside a GeoPackage geometry blob (header and envelope skipped)."""
    if blob[:2] != b"GP":
        raise SourceError("not a GeoPackage geometry")
    env = (blob[3] >> 1) & 7
    return blob[8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env] :]


class GeoPackage:
    """A GeoPackage feature table read through its R*-tree: `query(bbox)` returns the
    features whose bounding boxes touch a lon/lat box. `source` is an http(s) URL (range
    requests) or a local path."""

    def __init__(self, source: str | Path | _Pages, table: str | None = None) -> None:
        if isinstance(source, _Pages):
            self.pages = source
        elif str(source).startswith(("http://", "https://")):
            self.pages = HttpPages(str(source))
        else:
            self.pages = FilePages(source)
        self.db = _Btree(self.pages)
        master = {}
        for _, payload in self.db.scan(1):
            typ, name, _tbl, root, sql = _record(payload)[:5]
            if typ == "table":
                master[str(name)] = (int(root or 0), str(sql or ""))
        if table is None:
            feats = [n for n in master if f"rtree_{n}_geom_node" in master]
            if not feats:
                raise SourceError("no feature table with an R*-tree index")
            table = feats[0]
        self.table = table
        self.root, sql = master[table]
        self.columns = _columns(sql)
        self.node_root = master[f"rtree_{table}_geom_node"][0]

    def rtree_ids(self, bbox: tuple[float, float, float, float]) -> list[int]:
        """Feature ids whose (float32, outward-rounded) R*-tree boxes touch `bbox`
        (west, south, east, north)."""
        w, s, e, n = bbox
        frontier, level, ids = [1], None, []
        while frontier:
            rows = self.db.lookup(self.node_root, frontier)
            nxt = []
            for nodeno in frontier:
                if nodeno not in rows:
                    raise SourceError(f"R*-tree node {nodeno} missing")
                data = _record(rows[nodeno])[1]
                assert isinstance(data, bytes)
                if level is None:  # the root holds the tree depth
                    level = int.from_bytes(data[0:2], "big")
                count = int.from_bytes(data[2:4], "big")
                for c in range(count):
                    o = 4 + 24 * c
                    rid = int.from_bytes(data[o : o + 8], "big", signed=True)
                    x0, x1, y0, y1 = struct.unpack_from(">4f", data, o + 8)
                    if x0 <= e and x1 >= w and y0 <= n and y1 >= s:
                        (ids if level == 0 else nxt).append(rid)
            frontier = nxt
            level = (level or 0) - 1
        return sorted(set(ids))

    def query(
        self, bbox: tuple[float, float, float, float], columns: Sequence[str] = ()
    ) -> list[dict[str, object]]:
        """Features touching `bbox`: dicts with "fid", "geometry" (shapely, lon/lat) and the
        requested attribute `columns`, in fid order."""
        ids = self.rtree_ids(bbox)
        rows = self.db.lookup(self.root, ids)
        names = self.columns
        gcol = names.index("geom") if "geom" in names else 1
        want = [(c, names.index(c)) for c in columns if c in names]
        out = []
        for fid in ids:
            payload = rows.get(fid)
            if payload is None:
                continue
            vals = _record(payload)
            blob = vals[gcol] if gcol < len(vals) else None
            if not isinstance(blob, bytes):
                continue
            try:
                geom = shapely.from_wkb(_gpkg_wkb(blob))
            except (shapely.errors.GEOSException, SourceError, ValueError):
                continue
            rec: dict[str, object] = {"fid": fid, "geometry": geom}
            for c, j in want:
                rec[c] = vals[j] if j < len(vals) else None
            out.append(rec)
        return out


# --------------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------------

Box = tuple[float, float, float, float]  # (west, south, east, north), degrees


def _round_box(box: Box) -> Box:
    w, s, e, n = box
    return (
        math.floor(w * 1e4) / 1e4,
        math.floor(s * 1e4) / 1e4,
        math.ceil(e * 1e4) / 1e4,
        math.ceil(n * 1e4) / 1e4,
    )


def _cache_path(cache_dir: Path | str | None, kind: str, key: str) -> Path:
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    digest = hashlib.sha256(f"v{CACHE_VERSION}|{key}".encode()).hexdigest()
    return cache / "buildings" / kind / f"{digest}.json.gz"


def _read_cache(path: Path) -> list | None:
    try:
        data = json.loads(gzip.decompress(path.read_bytes()))
    except (OSError, ValueError, EOFError):
        return None
    return data if isinstance(data, list) else None


def _write_cache(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(rows, separators=(",", ":")).encode(), mtime=0))
    tmp.replace(path)


def _num(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _footprint_json(f: Footprint) -> dict:
    d: dict[str, object] = {"id": f.id, "src": f.source, "ring": np.round(f.lonlat, 7).tolist()}
    for k in ("peak_m", "eave_m", "low_m", "year"):
        if getattr(f, k) is not None:
            d[k] = getattr(f, k)
    if f.tags:
        d["tags"] = dict(f.tags)
    if f.lidar:
        d["lidar"] = True
    return d


def _footprint_from_json(d: Mapping) -> Footprint:
    return Footprint(
        str(d["id"]),
        str(d["src"]),
        np.asarray(d["ring"], np.float64),
        dict(d.get("tags") or {}),
        peak_m=d.get("peak_m"),
        eave_m=d.get("eave_m"),
        low_m=d.get("low_m"),
        year=d.get("year"),
        lidar=bool(d.get("lidar", False)),
    )


def _outer_rings(geom: shapely.Geometry) -> list[np.ndarray]:
    """Exterior rings (lon/lat, closed) of a polygon or multipolygon, largest first."""
    parts = [p for p in shapely.get_parts(geom) if isinstance(p, shapely.Polygon) and p.area > 0]
    parts.sort(key=lambda p: -p.area)
    return [np.asarray(p.exterior.coords)[:, :2] for p in parts]


# --------------------------------------------------------------------------------------------
# NRCan Automatically Extracted Buildings
# --------------------------------------------------------------------------------------------

_NRCAN_COLUMNS = ("heightmin", "heightmax", "datemin")


def nrcan_footprint(rec: Mapping[str, object]) -> list[Footprint]:
    """Footprints of one NRCan record (`GeoPackage.query` row): one per polygon part."""
    year = None
    date = str(rec.get("datemin") or "")
    if len(date) >= 4 and date[:4].isdigit():
        year = int(date[:4])
    hmax, hmin = _num(rec.get("heightmax")), _num(rec.get("heightmin"))
    rings = _outer_rings(rec["geometry"])  # type: ignore[arg-type]
    out = []
    for k, ring in enumerate(rings):
        fid = f"nrcan:{rec['fid']}" + (f".{k}" if len(rings) > 1 else "")
        out.append(Footprint(fid, "nrcan", ring, {}, peak_m=hmax, low_m=hmin, year=year,
                             lidar=True))  # fmt: skip
    return out


def fetch_nrcan(
    boxes: Sequence[Box],
    *,
    cache_dir: Path | str | None = None,
    source: str | Path = NRCAN_GPKG_URL,
    refresh: bool = False,
    offline: bool = False,
) -> list[Footprint]:
    """NRCan lidar footprints touching any of the lon/lat boxes (west, south, east, north),
    without duplicates. Cached per box; `offline=True` reads only the cache (SourceError on a
    miss). The R*-tree of the remote GeoPackage is walked once per uncached box, then all the
    new features are read in one batched lookup."""
    found: dict[str, Footprint] = {}
    todo: list[tuple[Box, Path]] = []
    for box in boxes:
        rb = _round_box(box)
        path = _cache_path(cache_dir, "nrcan", f"{source}|{rb}")
        rows = None if refresh else _read_cache(path)
        if rows is None:
            todo.append((rb, path))
            continue
        for d in rows:
            f = _footprint_from_json(d)
            found.setdefault(f.id, f)
    if todo:
        if offline:
            raise SourceError("NRCan building footprints for this area are not cached (offline)")
        t0 = time.perf_counter()
        try:
            gp = GeoPackage(source)
            per_box = [(rb, path, gp.rtree_ids(rb)) for rb, path in todo]
            ids = sorted({i for _, _, b in per_box for i in b})
            rows = gp.db.lookup(gp.root, ids)
        except (OSError, urllib.error.URLError, struct.error, IndexError, ValueError) as e:
            raise SourceError(f"NRCan building footprints unavailable: {e}") from e
        recs: dict[int, list[Footprint]] = {}
        names = gp.columns
        gcol = names.index("geom") if "geom" in names else 1
        cols = [(c, names.index(c)) for c in _NRCAN_COLUMNS if c in names]
        for fid, payload in rows.items():
            vals = _record(payload)
            blob = vals[gcol] if gcol < len(vals) else None
            if not isinstance(blob, bytes):
                continue
            try:
                geom = shapely.from_wkb(_gpkg_wkb(blob))
            except (shapely.errors.GEOSException, SourceError, ValueError):
                continue
            rec = {"fid": fid, "geometry": geom, **{c: vals[j] for c, j in cols if j < len(vals)}}
            recs[fid] = nrcan_footprint(rec)
        for _rb, path, bids in per_box:
            fps = [f for i in bids for f in recs.get(i, [])]
            _write_cache(path, [_footprint_json(f) for f in fps])
            for f in fps:
                found.setdefault(f.id, f)
        log.info(
            "NRCan buildings: %d features, %.1f MB in %d requests, %.1f s",
            len(recs), gp.pages.bytes_read / 1e6, gp.pages.requests, time.perf_counter() - t0,
        )  # fmt: skip
    return sorted(found.values(), key=lambda f: f.id)


# --------------------------------------------------------------------------------------------
# Municipal footprints (ArcGIS REST)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MunicipalSource:
    """One municipality's building layer. `bbox` is the layer's lon/lat extent (queries
    outside it are skipped); `peak_field` / `eave_field` hold heights above ground in metres;
    `kind` maps a record's attributes to an OSM-style `building` value."""

    name: str
    title: str
    url: str  # the layer, ".../MapServer/0" or ".../FeatureServer/0"
    bbox: Box
    attribution: str
    year: int | None = None
    where: str = "1=1"
    out_fields: tuple[str, ...] = ()
    peak_field: str | None = None
    eave_field: str | None = None
    kind: Callable[[Mapping[str, object]], str] | None = field(default=None, compare=False)
    terms_url: str | None = None
    page_size: int = 1000


_BURLINGTON_KINDS = {
    "COMMERCIAL": "commercial",
    "INDUSTRIAL": "industrial",
    "ASSEMBLY": "public",
    "GOVERNMENT": "government",
    "UTILITY": "service",
    "MERCANTILE": "retail",
}
_BURLINGTON_SUBTYPES = {
    "Single Family Dwelling": "house",
    "Low Density Residential": "house",
    "Low-rise Residential": "apartments",
    "High-rise Residential": "apartments",
    "Garage": "garage",
    "Education": "school",
    "Place Of Worship": "church",
}


def _burlington_kind(attrs: Mapping[str, object]) -> str:
    sub = _BURLINGTON_SUBTYPES.get(str(attrs.get("BUILDINGSUBTYPE") or ""))
    if sub:
        return sub
    return _BURLINGTON_KINDS.get(str(attrs.get("BUILDINGTYPE") or "").upper(), "yes")


BURLINGTON_TERMS_URL = (
    "https://opendata.burlington.ca/opendata-terms-of-use/"
    "City%20of%20Burlington%20-%20Open%20Data%20Terms%20of%20Use.pdf"
)
BURLINGTON = MunicipalSource(
    name="burlington",
    title="City of Burlington Buildings",
    url="https://mapping.burlington.ca/arcgisweb/rest/services/COB/Buildings/MapServer/0",
    bbox=(-79.9584, 43.2857, -79.7229, 43.4742),
    # Wording from the City's terms (the research's attribution line); the terms travel
    # with the data, which they require of anyone passing it on.
    attribution=(
        "Buildings (outlines, eave and peak heights, types): contains City of Burlington "
        "data, used under the City of Burlington Open Data Terms of Use: "
        f"{BURLINGTON_TERMS_URL} . These terms apply to anyone receiving this data, and the "
        "City may change or revoke them."
    ),
    year=2017,  # 2017 photogrammetry (LAST_EDITED_DATE is a 2024-2025 bulk edit)
    where="STATEOFRECORD = 'Active'",
    out_fields=("OBJECTID", "EAVEHGT", "PEAKHGT", "BUILDINGTYPE", "BUILDINGSUBTYPE"),
    peak_field="PEAKHGT",
    eave_field="EAVEHGT",
    kind=_burlington_kind,
    terms_url=BURLINGTON_TERMS_URL,
)
MILTON = MunicipalSource(
    name="milton",
    title="Town of Milton Buildings",
    url="https://api.milton.ca/arcgis/rest/services/Datasets/Buildings/MapServer/0",
    bbox=(-80.1522, 43.4196, -79.7319, 43.6282),
    attribution=(
        "Buildings (outlines, eave and peak heights): contains information licensed under "
        "the Open Government Licence - Town of Milton "
        "(https://discover-milton.hub.arcgis.com/pages/disclaimer-and-terms-of-use)."
    ),
    # The fields are named *_ELEVEATION but hold heights above ground in metres (research).
    where=(
        "(Status IS NULL OR Status = 'ACTIVE') AND (OTHERINFO IS NULL OR "
        "(UPPER(OTHERINFO) NOT LIKE '%PROPOSED%' AND UPPER(OTHERINFO) NOT LIKE '%DEMOLISH%'))"
    ),
    out_fields=("OBJECTID", "EVE_ELEVEATION", "PEAK_ELEVATION"),
    peak_field="PEAK_ELEVATION",
    eave_field="EVE_ELEVEATION",
)
OAKVILLE = MunicipalSource(
    name="oakville",
    title="Town of Oakville Building Footprints",
    url=(
        "https://services5.arcgis.com/QJebCdoMf4PF8fJP/arcgis/rest/services/"
        "Building_Footprints/FeatureServer/0"
    ),
    bbox=(-79.8133, 43.3707, -79.6247, 43.5284),
    attribution=(
        "Buildings (outlines): contains information licensed under the Open Government "
        "Licence - Town of Oakville (https://www.oakville.ca/getmedia/"
        "3cb7902a-2220-433e-8aed-24f326b745b9/town-hall-oakville-open-data-licence.pdf)."
    ),
    year=2019,
    out_fields=("FID",),
)
MUNICIPAL_SOURCES: dict[str, MunicipalSource] = {s.name: s for s in (BURLINGTON, MILTON, OAKVILLE)}


def _signed_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def municipal_feature(src: MunicipalSource, feat: Mapping) -> list[Footprint]:
    """Footprints of one Esri JSON feature: one per outer ring (Esri outer rings run
    clockwise; holes, counter-clockwise, are dropped). Heights <= 0 count as unknown."""
    attrs = feat.get("attributes") or {}
    rings = (feat.get("geometry") or {}).get("rings") or []
    oid = attrs.get("OBJECTID", attrs.get("FID"))
    peak = _num(attrs.get(src.peak_field)) if src.peak_field else None
    eave = _num(attrs.get(src.eave_field)) if src.eave_field else None
    peak = peak if peak is not None and 0.5 < peak < 150.0 else None
    eave = eave if eave is not None and 0.5 < eave < 150.0 else None
    if peak is not None and eave is not None and eave > peak:
        eave = None
    tags = {"building": src.kind(attrs)} if src.kind is not None else {}
    outers = []
    for r in rings:
        ring = np.asarray(r, np.float64)[:, :2]
        if len(ring) < 4 or not np.isfinite(ring).all():
            continue
        if not np.allclose(ring[0], ring[-1]):
            ring = np.vstack([ring, ring[:1]])
        if _signed_area(ring) < 0:  # clockwise = outer
            outers.append(ring)
    out = []
    for k, ring in enumerate(outers):
        fid = f"{src.name}:{oid}" + (f".{k}" if len(outers) > 1 else "")
        out.append(Footprint(fid, src.name, ring, dict(tags), peak_m=peak, eave_m=eave,
                             year=src.year))  # fmt: skip
    return out


def _get_json(url: str, params: Mapping[str, object], timeout_s: float = 60.0) -> dict:
    full = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
    err: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read())
            if isinstance(data, dict) and "error" in data:
                raise SourceError(f"{url}: {data['error']}")
            return data
        except (urllib.error.URLError, OSError, ValueError) as e:
            err = e
            time.sleep(0.5 * 2**attempt)
    raise SourceError(f"{url}: {err}")


def _boxes_touch(a: Box, b: Box) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def fetch_municipal(
    src: MunicipalSource,
    boxes: Sequence[Box],
    *,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
    offline: bool = False,
    get_json: Callable[[str, Mapping[str, object]], dict] = _get_json,
) -> list[Footprint]:
    """`src`'s footprints touching any of the lon/lat boxes, without duplicates (boxes outside
    the layer's extent are skipped). Cached per box; `offline=True` reads only the cache."""
    found: dict[str, Footprint] = {}
    for box in boxes:
        if not _boxes_touch(box, src.bbox):
            continue
        rb = _round_box(box)
        path = _cache_path(cache_dir, src.name, f"{src.url}|{src.where}|{rb}")
        rows = None if refresh else _read_cache(path)
        if rows is None:
            if offline:
                raise SourceError(f"{src.title} for this area are not cached (offline)")
            fps: list[Footprint] = []
            offset = 0
            while True:
                data = get_json(src.url + "/query", {
                    "where": src.where,
                    "geometry": ",".join(f"{v:.6f}" for v in rb),
                    "geometryType": "esriGeometryEnvelope",
                    "inSR": 4326,
                    "spatialRel": "esriSpatialRelIntersects",
                    "outFields": ",".join(src.out_fields) or "*",
                    "returnGeometry": "true",
                    "outSR": 4326,
                    "geometryPrecision": 7,
                    "orderByFields": src.out_fields[0] if src.out_fields else "",
                    "resultOffset": offset,
                    "resultRecordCount": src.page_size,
                    "f": "json",
                })  # fmt: skip
                feats = data.get("features") or []
                for feat in feats:
                    fps.extend(municipal_feature(src, feat))
                offset += len(feats)
                if not feats or not data.get("exceededTransferLimit"):
                    break
            _write_cache(path, [_footprint_json(f) for f in fps])
            rows = [_footprint_json(f) for f in fps]
        for d in rows:
            f = _footprint_from_json(d)
            found.setdefault(f.id, f)
    return sorted(found.values(), key=lambda f: f.id)
