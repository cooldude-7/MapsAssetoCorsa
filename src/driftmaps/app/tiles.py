"""Road data in z14 map tiles for the turn-by-turn builder (`POST /api/roads`).

Tiles are standard slippy-map tiles ("14/x/y", Web Mercator). Each tile holds every drivable
way that crosses it, whole (the same road classes as the router's corridor query), cached
gzip'd on disk under `<cache_dir>/tiles/14/<x>/<y>.json.gz` and in memory. When the player
has downloaded a region's roads (`driftmaps.geo.offline`, e.g. Ontario), tiles inside it come
from that database (a few ms, never Overpass); a tile straddling its edge is answered with the
part inside at once and fetched whole from Overpass in the background. A cached tile answers
instantly; the tiles a request is missing are fetched in ONE Overpass query (the bbox of those
tiles, or a union of their boxes when they are scattered). Requests for different tiles
download side by side; one that needs a tile already being downloaded waits for it.

The same data serves `POST /api/route-path` (a route along an explicit node path, built from
the tiles in memory) and lets `/api/route` skip Overpass when cached tiles cover its corridor.

OSM data is (c) OpenStreetMap contributors, ODbL 1.0.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

from driftmaps.geo import offline as region_db
from driftmaps.geo.attributes import is_oneway
from driftmaps.geo.overpass import (
    OsmData,
    OverpassError,
    Way,
    build_bbox_query,
    parse_overpass,
    post_query,
)

__all__ = [
    "MAX_TILES",
    "TILE_ZOOM",
    "TileStore",
    "parse_tile",
    "tile_bounds",
    "tile_key",
    "tile_xy",
    "tiles_for_box",
]

log = logging.getLogger(__name__)

TILE_ZOOM = 14
MAX_TILES = 16  # tiles per /api/roads request
TILE_FORMAT = 1
MEMORY_TILES = 192  # tiles kept parsed in memory (route-path looks for its nodes there)
# Interactive Overpass calls: a short server-side limit and quick retries.
# On 2026-10-16 the one mirror answering from the dev sandbox (maps.mail.ru) took 15-30 s for
# 1 to 16 tiles, and timed out once at 45 s.
QUERY_TIMEOUT_S = 60
POST_KWARGS = {"rounds": 3, "backoff_s": 2.0, "timeout_s": 70.0}  # 504s come fast: retry
WAIT_OTHER_S = 300.0  # longest wait for another request's download (post_query ends sooner)
_TILE_RE = re.compile(r"^(\d{1,2})/(\d{1,7})/(\d{1,7})$")

Tile = tuple[int, int, int]


def tile_xy(lat: float, lon: float, z: int = TILE_ZOOM) -> tuple[int, int]:
    """The (x, y) of the z tile holding (lat, lon)."""
    xf, yf = _tile_float(np.array([lon], np.float64), np.array([lat], np.float64), z)
    n = 2**z
    return min(n - 1, max(0, int(xf[0]))), min(n - 1, max(0, int(yf[0])))


def _tile_float(lon: np.ndarray, lat: np.ndarray, z: int) -> tuple[np.ndarray, np.ndarray]:
    n = 2.0**z
    lat_r = np.radians(np.clip(lat, -85.05112878, 85.05112878))
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - np.log(np.tan(lat_r) + 1.0 / np.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """(south, west, north, east) degrees of a tile."""
    n = 2.0**z

    def lat(yy: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * yy / n))))

    return lat(y + 1), x / n * 360.0 - 180.0, lat(y), (x + 1) / n * 360.0 - 180.0


def tile_key(tile: Tile) -> str:
    return f"{tile[0]}/{tile[1]}/{tile[2]}"


def parse_tile(value: object) -> Tile:
    """Parse "14/x/y" -> (14, x, y); ValueError for anything else (only z14 is served)."""
    m = _TILE_RE.match(value) if isinstance(value, str) else None
    if not m:
        raise ValueError(f"not a map tile: {value!r}")
    z, x, y = (int(g) for g in m.groups())
    if z != TILE_ZOOM or x >= 2**z or y >= 2**z:
        raise ValueError(f"not a zoom-{TILE_ZOOM} map tile: {value!r}")
    return z, x, y


def tiles_for_box(
    south: float, west: float, north: float, east: float, z: int = TILE_ZOOM
) -> list[Tile]:
    """Every z tile overlapping the lat/lon box, row by row."""
    x0, y0 = tile_xy(north, west, z)
    x1, y1 = tile_xy(south, east, z)
    return [(z, x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def _way_tiles(lonlat: np.ndarray, wanted: set[Tile], z: int) -> set[Tile]:
    """The tiles in `wanted` that a way's segments (their bounding boxes) touch."""
    xf, yf = _tile_float(lonlat[:, 0], lonlat[:, 1], z)
    xi, yi = np.floor(xf).astype(np.int64), np.floor(yf).astype(np.int64)
    if len(xi) > 1:
        lo_x = np.minimum(xi[:-1], xi[1:])
        hi_x = np.maximum(xi[:-1], xi[1:])
        lo_y = np.minimum(yi[:-1], yi[1:])
        hi_y = np.maximum(yi[:-1], yi[1:])
    else:
        lo_x = hi_x = xi
        lo_y = hi_y = yi
    out: set[Tile] = set()
    simple = (lo_x == hi_x) & (lo_y == hi_y)
    for x, y in set(zip(lo_x[simple].tolist(), lo_y[simple].tolist(), strict=True)):
        if (z, x, y) in wanted:
            out.add((z, x, y))
    for i in np.flatnonzero(~simple).tolist():
        for x in range(int(lo_x[i]), int(hi_x[i]) + 1):
            for y in range(int(lo_y[i]), int(hi_y[i]) + 1):
                if (z, x, y) in wanted:
                    out.add((z, x, y))
    return out


class _TileData:
    """One tile: its ways (whole) and every node of them."""

    __slots__ = ("nodes", "osm_base", "ways")

    def __init__(self, ways: list[dict], osm_base: str | None) -> None:
        self.ways = ways  # {"id", "nodes", "geom": [[lon, lat], ...], "tags"}
        self.osm_base = osm_base
        self.nodes: dict[int, tuple[float, float]] = {}
        for w in ways:
            for nid, (lon, lat) in zip(w["nodes"], w["geom"], strict=True):
                self.nodes.setdefault(nid, (lon, lat))

    def to_json(self) -> bytes:
        doc = {"format": TILE_FORMAT, "osm_base": self.osm_base, "ways": self.ways}
        return json.dumps(doc, separators=(",", ":"), sort_keys=True).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> _TileData:
        doc = json.loads(raw)
        if not isinstance(doc, dict) or doc.get("format") != TILE_FORMAT:
            raise ValueError("unknown tile format")
        ways = [
            {
                "id": int(w["id"]),
                "nodes": [int(n) for n in w["nodes"]],
                "geom": [(float(p[0]), float(p[1])) for p in w["geom"]],
                "tags": {str(k): str(v) for k, v in w["tags"].items()},
            }
            for w in doc["ways"]
        ]
        return cls(ways, doc.get("osm_base"))


def _way_record(w: Way) -> dict:
    return {
        "id": w.id,
        "nodes": list(w.nodes),
        "geom": [(round(lon, 7), round(lat, 7)) for lon, lat in w.lonlat.tolist()],
        "tags": dict(w.tags),
    }


def _api_way(w: dict) -> dict:
    tags = w["tags"]
    return {
        "id": w["id"],
        "name": tags.get("name") or None,
        "ref": tags.get("ref") or None,
        "highway": tags.get("highway") or None,
        "oneway": is_oneway(tags),
        "nodes": w["nodes"],
    }


Poster = Callable[[str], bytes]


class TileStore:
    """Road tiles from memory, the disk cache or Overpass (one query per request)."""

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        post: Poster | None = None,
        include_tracks: str = "paved",
        memory: int = MEMORY_TILES,
        use_region_db: bool = True,
    ) -> None:
        self.dir = Path(cache_dir) / "tiles"
        self.use_region_db = use_region_db
        self.post = post
        self.include_tracks = include_tracks
        self.memory = memory
        self._mem: OrderedDict[Tile, _TileData] = OrderedDict()
        self._lock = threading.Lock()
        # Tiles being downloaded right now -> set when that download is over. A request waits
        # for another's download of its tiles instead of asking again, and never for others.
        self._inflight: dict[Tile, threading.Event] = {}
        # Tiles straddling a downloaded region's edge, still to fetch whole (`_fill_later`).
        self._fill: set[Tile] = set()
        self._fill_worker: threading.Thread | None = None

    # -- cache -------------------------------------------------------------------------------

    def _path(self, tile: Tile) -> Path:
        z, x, y = tile
        return self.dir / str(z) / str(x) / f"{y}.json.gz"

    def _remember(self, tile: Tile, data: _TileData) -> None:
        with self._lock:
            self._mem[tile] = data
            self._mem.move_to_end(tile)
            while len(self._mem) > self.memory:
                self._mem.popitem(last=False)

    def cached(self, tile: Tile) -> _TileData | None:
        """The tile from memory, the downloaded region database or disk, or None."""
        with self._lock:
            data = self._mem.get(tile)
            if data is not None:
                self._mem.move_to_end(tile)
                return data
        region = self._from_region(tile)
        if region is not None and region[1]:
            self._remember(tile, region[0])
            return region[0]
        data = self._from_disk(tile)
        if data is not None:
            self._remember(tile, data)
            return data
        if region is not None:  # straddles the region's edge: the inside now, all of it later
            self._remember(tile, region[0])
            self._fill_later(tile)
            return region[0]
        return None

    def _from_disk(self, tile: Tile) -> _TileData | None:
        path = self._path(tile)
        try:
            data = _TileData.from_json(gzip.decompress(path.read_bytes()))
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
            log.warning("ignoring a broken tile cache file %s: %s", path, exc)
            return None
        return data

    def _from_region(self, tile: Tile) -> tuple[_TileData, bool] | None:
        """(the tile from the downloaded region database (`geo.offline`), complete) if one
        holds any of it; `complete` is False for a tile straddling the region's edge."""
        if not self.use_region_db or tile[0] != TILE_ZOOM:
            return None
        try:
            found = region_db.tile_ways(tile[1], tile[2], self.include_tracks)
        except Exception as exc:  # the database is only a shortcut
            log.warning("offline road data failed for tile %s: %s", tile_key(tile), exc)
            return None
        if found is None:
            return None
        ways, osm_base, complete = found
        return _TileData(ways, osm_base), complete

    def _fill_later(self, tile: Tile) -> None:
        """Download a tile the region database holds only part of, in the background (one
        worker, MAX_TILES per Overpass query); it replaces the partial tile when it arrives."""
        with self._lock:
            if tile in self._fill:
                return
            self._fill.add(tile)
            if self._fill_worker is not None and self._fill_worker.is_alive():
                return
            self._fill_worker = threading.Thread(
                target=self._fill_run, name="road-tiles-fill", daemon=True
            )
            self._fill_worker.start()

    def _fill_run(self) -> None:
        while True:
            with self._lock:
                todo = sorted(self._fill)[:MAX_TILES]
                if not todo:
                    self._fill_worker = None
                    return
            try:
                self._download(todo)
            except Exception as exc:  # noqa: BLE001 - the partial tiles stay; next time again
                log.warning("background road tiles download failed: %s", exc)
            with self._lock:
                self._fill.difference_update(todo)

    def _save(self, tile: Tile, data: _TileData) -> None:
        self._remember(tile, data)
        path = self._path(tile)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + f".{threading.get_ident()}.tmp")
            tmp.write_bytes(gzip.compress(data.to_json(), compresslevel=6, mtime=0))
            tmp.replace(path)
        except OSError as exc:  # memory still has it
            log.warning("could not save road tile %s: %s", tile_key(tile), exc)

    # -- fetching ------------------------------------------------------------------------------

    def _query(self, tiles: Sequence[Tile]) -> str:
        xs = [t[1] for t in tiles]
        ys = [t[2] for t in tiles]
        span = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
        if span <= 2 * len(tiles) + 2:  # compact: one box round them all
            s_, w, _, _ = tile_bounds(TILE_ZOOM, min(xs), max(ys))
            _, _, n, e = tile_bounds(TILE_ZOOM, max(xs), min(ys))
            boxes = [(s_, w, n, e)]
        else:  # scattered: one box each, still one query
            boxes = [tile_bounds(*t) for t in tiles]
        return build_bbox_query(boxes, self.include_tracks, timeout_s=QUERY_TIMEOUT_S)

    def _download(self, tiles: Sequence[Tile]) -> None:
        query = self._query(tiles)
        body = self.post(query) if self.post is not None else post_query(query, **POST_KWARGS)
        osm = parse_overpass(body)
        wanted = set(tiles)
        per_tile: dict[Tile, list[dict]] = {t: [] for t in tiles}
        for way in osm.ways:
            hit = _way_tiles(way.lonlat, wanted, TILE_ZOOM)
            if hit:
                rec = _way_record(way)
                for t in hit:
                    per_tile[t].append(rec)
        for t in tiles:
            self._save(t, _TileData(per_tile[t], osm.osm_base))

    def get(
        self, tiles: Sequence[Tile], stats: dict | None = None
    ) -> tuple[dict[Tile, _TileData], str | None]:
        """The tiles, from the caches or one Overpass query for the missing ones.

        Returns (tile -> data for every tile that is available, error text or None). Tiles
        the download failed for are missing from the result. `stats`, if given, gets
        "download_s": how long this request's Overpass download took (0 without one).
        """
        out: dict[Tile, _TileData] = {}
        missing: list[Tile] = []
        for t in dict.fromkeys(tiles):
            data = self.cached(t)
            if data is None:
                missing.append(t)
            else:
                out[t] = data
        if not missing:
            return out, None
        error = None
        done = threading.Event()
        with self._lock:
            theirs = {self._inflight[t] for t in missing if t in self._inflight}
            mine = [t for t in missing if t not in self._inflight]
            for t in mine:
                self._inflight[t] = done
        try:
            still = [t for t in mine if self.cached(t) is None]  # saved in the meantime?
            if still:
                t0 = time.monotonic()
                try:
                    self._download(still)
                except (OverpassError, OSError, ValueError) as exc:
                    log.warning("road tiles download failed: %s", exc)
                    error = str(exc)
                if stats is not None:
                    stats["download_s"] = time.monotonic() - t0
        finally:
            with self._lock:
                for t in mine:
                    self._inflight.pop(t, None)
            done.set()
        for other in theirs:  # another request is downloading some of ours
            other.wait(WAIT_OTHER_S)
        for t in missing:
            data = self.cached(t)
            if data is not None:
                out[t] = data
        if error is None and len(out) < len(dict.fromkeys(tiles)):
            error = "the road data download failed"
        return out, error

    # -- views -----------------------------------------------------------------------------------

    @staticmethod
    def api_payload(datas: Iterable[_TileData]) -> tuple[dict[str, list[float]], list[dict]]:
        """(nodes {"id": [lon, lat]}, ways) of the tiles, each way once."""
        ways: dict[int, dict] = {}
        nodes: dict[str, list[float]] = {}
        for data in datas:
            for w in data.ways:
                if w["id"] in ways:
                    continue
                ways[w["id"]] = _api_way(w)
                for nid, (lon, lat) in zip(w["nodes"], w["geom"], strict=True):
                    nodes.setdefault(str(nid), [lon, lat])
        return nodes, [ways[k] for k in sorted(ways)]

    @staticmethod
    def osm_data(datas: Iterable[_TileData], only_nodes: set[int] | None = None) -> OsmData:
        """The tiles' ways as OsmData (each once; with `only_nodes`, ways through them)."""
        ways: dict[int, Way] = {}
        bases: list[str] = []
        for data in datas:
            if data.osm_base:
                bases.append(data.osm_base)
            for w in data.ways:
                if w["id"] in ways or len(w["nodes"]) < 2:
                    continue
                if only_nodes is not None and only_nodes.isdisjoint(w["nodes"]):
                    continue
                lonlat = np.array(w["geom"], dtype=np.float64)
                ways[w["id"]] = Way(w["id"], tuple(w["nodes"]), lonlat, dict(w["tags"]))
        return OsmData(
            tuple(ways[k] for k in sorted(ways)), min(bases) if bases else None, "tiles-z14"
        )

    def for_nodes(self, node_ids: Iterable[int]) -> tuple[OsmData, set[int]]:
        """Ways through any of the nodes, from the tiles in memory; and the nodes not found.

        The cached tiles (memory or disk) holding the nodes found are added too, so every road
        meeting the path at a junction is there (a road is only in the tiles it crosses).
        """
        want = {int(n) for n in node_ids}
        with self._lock:
            datas = list(reversed(self._mem.values()))
        hits = {id(d): d for d in datas if not want.isdisjoint(d.nodes.keys())}
        found: dict[int, tuple[float, float]] = {}
        for d in hits.values():
            for n in want & d.nodes.keys():
                found.setdefault(n, d.nodes[n])
        if found:
            lonlat = np.array(list(found.values()), dtype=np.float64)
            xf, yf = _tile_float(lonlat[:, 0], lonlat[:, 1], TILE_ZOOM)
            xs, ys = np.floor(xf).astype(int).tolist(), np.floor(yf).astype(int).tolist()
            cells = set(zip(xs, ys, strict=True))
            for x, y in sorted(cells):
                data = self.cached((TILE_ZOOM, x, y))
                if data is not None:
                    hits.setdefault(id(data), data)
        return self.osm_data(hits.values(), want), want - found.keys()

    def covering(
        self, south: float, west: float, north: float, east: float, max_tiles: int = 48
    ) -> OsmData | None:
        """All roads in the box from cached tiles, or None if a tile isn't cached (or the
        box needs more than `max_tiles`)."""
        tiles = tiles_for_box(south, west, north, east)
        if len(tiles) > max_tiles:
            return None
        datas = []
        for t in tiles:
            data = self.cached(t)
            if data is None:
                return None
            datas.append(data)
        osm = self.osm_data(datas)
        return osm if osm.ways else None
