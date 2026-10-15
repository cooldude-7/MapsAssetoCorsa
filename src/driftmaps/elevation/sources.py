"""Elevation sources: sample DEM heights at lon/lat points through a per-location source chain.

`sample_elevation(lon, lat)` picks a chain of sources for each point's region and tries them in
order; a point that a source can't answer (outside its coverage, a missing tile, NoData, or the
service unreachable) falls through to the next source (docs/research/geo.md §2, PLAN §5.2):

- Canada: NRCan HRDEM 1 m mosaic -> HRDEM lidar projects -> HRDEM 2 m mosaic -> MRDEM-30 ->
  Copernicus GLO-30. HRDEM/MRDEM are found through the NRCan datacube STAC API
  (`hrdem-mosaic-1m`, `hrdem-lidar`, `hrdem-mosaic-2m`, `mrdem-30`), and are bare-earth (DTM)
  cloud-optimised GeoTIFFs in EPSG:3979 on `canelevation-dem.s3.ca-central-1.amazonaws.com`
  (VERIFIED by live STAC search + windowed reads near Toronto, 2026-10-15).
- US: USGS 3DEP 1 m project DEMs (newest first, via the TNM Access API) -> 3DEP 1/3 arc-second
  1x1 degree tiles -> GLO-30. All public COGs on `prd-tnm.s3.amazonaws.com` (VERIFIED, geo.md).
- Near the Canada/US border both countries' lidar is tried before either country's coarse DEM.
- France: IGN RGE ALTI 1 m through the Geoplateforme WMS-R, which returns float32 heights as
  GeoTIFF (`FORMAT=image/geotiff`, VERIFIED live 2026-10-15, NoData = -99999) -> GLO-30.
- Everywhere else: Copernicus GLO-30 (a DSM: includes trees and buildings).

Reads are windowed and cached: every raster is read in blocks aligned to its own grid (the
COG's internal tiles), and each block is cached on disk as a float32 `.npy`, so a road only ever
downloads the blocks it touches and a rebuild (or the terrain grid after the profile) reuses
them. Catalogue answers (STAC / TNM) are cached per 0.1 degree cell as JSON.

Heights are bilinear between pixel centres. NoData pixels are dropped and the remaining
neighbours renormalised; a point with no valid neighbour is NaN.

Vertical datums differ (CGVD2013 for HRDEM/MRDEM, NAVD88 for 3DEP, EGM2008 for GLO-30,
NGF-IGN69 for RGE ALTI). ASSUMED fine to treat all as orthometric metres; mixing sources inside
one track can still leave a step of up to a few metres where coverage changes, so callers should
check `ElevationSample.source` and prefer one source per build (geo.md §2.3.6).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.crs import CRS
from rasterio.errors import RasterioIOError
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

__all__ = [
    "CHAINS",
    "ENV_CACHE_DIR",
    "CatalogItem",
    "CatalogSource",
    "CogSource",
    "ElevationSample",
    "ElevationSource",
    "SourceInfo",
    "TiledCogSource",
    "WmsSource",
    "default_cache_dir",
    "default_sources",
    "local_dem_source",
    "region_of",
    "sample_elevation",
    "sample_grid",
    "source_chain",
]

log = logging.getLogger(__name__)

ENV_CACHE_DIR = "DRIFTMAPS_CACHE_DIR"
USER_AGENT = "driftmaps/0.1 (Assetto Corsa track generator)"
WGS84 = CRS.from_epsg(4326)

# Plausible land heights; anything outside is treated as NoData (Dead Sea shore ~ -430 m).
_MIN_VALID_M = -500.0
_MAX_VALID_M = 9000.0

FloatArray = npt.NDArray[np.float64]


def default_cache_dir() -> Path:
    """Root cache folder: $DRIFTMAPS_CACHE_DIR, else `<repo>/.cache` in a dev checkout, else the
    per-user cache folder (%LOCALAPPDATA%/driftmaps or ~/.cache/driftmaps)."""
    env = os.environ.get(ENV_CACHE_DIR)
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[3]
    if (repo / "pyproject.toml").is_file() and (repo / "src" / "driftmaps").is_dir():
        return repo / ".cache"
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "driftmaps"


# ---------------------------------------------------------------------------------------------
# Results and source descriptions
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceInfo:
    """What a source is, for choosing smoothing and for the track's LICENSE-DATA.txt."""

    name: str
    title: str
    resolution_m: float
    surface: str  # "DTM" (bare earth) or "DSM" (includes trees and buildings)
    licence: str
    attribution: str


@dataclass(frozen=True)
class ElevationSample:
    """Heights for N points. `heights` is NaN and `source` is "" where nothing had data."""

    heights: FloatArray  # (N,) metres
    source: npt.NDArray[np.str_]  # (N,) source name per point
    resolution_m: FloatArray  # (N,) nominal resolution of that point's source (NaN if none)
    info: Mapping[str, SourceInfo] = field(default_factory=dict)  # sources actually used

    @property
    def valid(self) -> npt.NDArray[np.bool_]:
        return np.isfinite(self.heights)

    @property
    def coarsest_resolution_m(self) -> float:
        """Worst resolution among the points that got a height (NaN if none did)."""
        r = self.resolution_m[np.isfinite(self.resolution_m)]
        return float(r.max()) if r.size else math.nan

    @property
    def any_dsm(self) -> bool:
        """True if any height came from a surface model (trees/buildings included)."""
        return any(i.surface == "DSM" for i in self.info.values())

    @property
    def attributions(self) -> list[str]:
        """Attribution lines for every source used, for LICENSE-DATA.txt / ui_track.json."""
        return [i.attribution for i in self.info.values()]


class SourceUnavailable(RuntimeError):
    """A source could not be reached right now (network, server error). Not cached."""


class ElevationSource(ABC):
    """A DEM that can answer heights at lon/lat points (NaN where it has no data)."""

    def __init__(self, info: SourceInfo) -> None:
        self.info = info

    @property
    def name(self) -> str:
        return self.info.name

    @abstractmethod
    def sample(self, lon: FloatArray, lat: FloatArray, cache_dir: Path) -> FloatArray:
        """Heights (N,) in metres, NaN outside coverage or on NoData. `cache_dir` is this
        source's own cache folder. May raise SourceUnavailable."""


# ---------------------------------------------------------------------------------------------
# HTTP + GDAL plumbing
# ---------------------------------------------------------------------------------------------


def _gdal_env() -> rasterio.Env:
    opts: dict[str, object] = dict(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        GDAL_HTTP_MAX_RETRY=3,
        GDAL_HTTP_RETRY_DELAY=1,
        GDAL_HTTP_TIMEOUT=60,
        GDAL_HTTP_USERAGENT=USER_AGENT,
        VSI_CACHE=True,
    )
    # GDAL's bundled curl doesn't always find the system CA store (proxies, some Windows
    # setups); hand it the bundle the environment already names, if any.
    for var in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        if os.environ.get(var):
            opts["GDAL_CURL_CA_BUNDLE"] = os.environ[var]
            break
    return rasterio.Env(**opts)


def _gdal_path(url: str) -> str:
    return "/vsicurl/" + url if url.startswith(("http://", "https://")) else url


def _is_missing_error(exc: Exception) -> bool:
    """True if a rasterio open error means 'there is no such file' (cacheable), not an outage."""
    msg = str(exc)
    return any(k in msg for k in ("404", "No such file", "does not exist"))


def _http_get(url: str, params: Mapping[str, object] | None = None, *, tries: int = 3) -> bytes:
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params, safe=",|:")
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code < 500 and e.code != 429:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        time.sleep(1.5 * 2**attempt)
    raise SourceUnavailable(f"GET {url} failed: {last}")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _save_block(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def _clean(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    bad = ~np.isfinite(arr) | (arr < _MIN_VALID_M) | (arr > _MAX_VALID_M)
    if nodata is not None and np.isfinite(nodata):
        bad |= arr == np.float32(nodata)
    return np.where(bad, np.float32(np.nan), arr)


# ---------------------------------------------------------------------------------------------
# Block-cached raster sampling
# ---------------------------------------------------------------------------------------------


class _Raster:
    """One georeferenced grid, read block by block; every block is cached on disk.

    Subclasses provide metadata and block reads; the bilinear sampling and caching are shared.
    """

    def __init__(self, key: str, cache_dir: Path) -> None:
        self.key = key
        self.dir = cache_dir / hashlib.sha1(key.encode()).hexdigest()[:20]
        self._meta: dict | None = None
        self._blocks: dict[tuple[int, int], np.ndarray] = {}

    # -- to override -------------------------------------------------------------------------
    def _read_meta(self) -> dict:
        """{crs, transform, width, height, block, nodata} or {"missing": True}."""
        raise NotImplementedError

    def _read_blocks(self, keys: list[tuple[int, int]]) -> dict[tuple[int, int], np.ndarray]:
        raise NotImplementedError

    # -- shared ------------------------------------------------------------------------------
    def meta(self) -> dict:
        if self._meta is None:
            path = self.dir / "meta.json"
            if path.is_file():
                self._meta = json.loads(path.read_text())
            else:
                self._meta = self._read_meta()
                self._meta["key"] = self.key
                _atomic_write(path, json.dumps(self._meta).encode())
        return self._meta

    @property
    def missing(self) -> bool:
        return bool(self.meta().get("missing"))

    def _block_path(self, bi: int, bj: int) -> Path:
        return self.dir / f"b{bi}_{bj}.npy"

    def _ensure_blocks(self, keys: list[tuple[int, int]]) -> None:
        todo = []
        for k in keys:
            if k in self._blocks:
                continue
            p = self._block_path(*k)
            if p.is_file():
                self._blocks[k] = np.load(p)
            else:
                todo.append(k)
        if todo:
            for k, arr in self._read_blocks(todo).items():
                _save_block(self._block_path(*k), arr)
                self._blocks[k] = arr

    def _gather(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        m = self.meta()
        b, h, w = int(m["block"]), int(m["height"]), int(m["width"])
        out = np.full(i.shape, np.nan, np.float32)
        inside = (i >= 0) & (j >= 0) & (i < h) & (j < w)
        if not inside.any():
            return out
        idx = np.flatnonzero(inside)
        bi, bj = i[idx] // b, j[idx] // b
        pairs, inv = np.unique(np.stack([bi, bj], axis=1), axis=0, return_inverse=True)
        keys = [(int(p[0]), int(p[1])) for p in pairs]
        self._ensure_blocks(keys)
        inv = inv.ravel()
        for n, k in enumerate(keys):
            sel = idx[inv == n]
            out[sel] = self._blocks[k][i[sel] - k[0] * b, j[sel] - k[1] * b]
        return out

    def sample(self, lon: FloatArray, lat: FloatArray) -> FloatArray:
        """Bilinear heights at lon/lat (WGS84 degrees); NaN outside or on NoData."""
        lon = np.asarray(lon, np.float64)
        lat = np.asarray(lat, np.float64)
        out = np.full(lon.shape, np.nan)
        if lon.size == 0 or self.missing:
            return out
        m = self.meta()
        crs = CRS.from_user_input(m["crs"])
        if crs == WGS84:
            x, y = lon, lat
        else:
            xs, ys = warp_transform(WGS84, crs, lon.tolist(), lat.tolist())
            x, y = np.asarray(xs), np.asarray(ys)
        inv = ~Affine(*m["transform"])
        col = inv.a * x + inv.b * y + inv.c - 0.5  # pixel-centre coordinates
        row = inv.d * x + inv.e * y + inv.f - 0.5
        j0 = np.floor(col).astype(np.int64)
        i0 = np.floor(row).astype(np.int64)
        fx, fy = col - j0, row - i0
        ii = np.concatenate([i0, i0, i0 + 1, i0 + 1])
        jj = np.concatenate([j0, j0 + 1, j0, j0 + 1])
        wts = np.concatenate([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy])
        vals = self._gather(ii, jj).astype(np.float64)
        ok = np.isfinite(vals)
        n = lon.size
        wsum = np.where(ok, wts, 0.0).reshape(4, n).sum(axis=0)
        vsum = np.where(ok, wts * np.where(ok, vals, 0.0), 0.0).reshape(4, n).sum(axis=0)
        # Keep a point while at least a fifth of its bilinear weight lands on valid pixels:
        # any point inside the raster footprint passes, points deep in NoData don't.
        good = wsum > 0.2
        out[good] = vsum[good] / wsum[good]
        return out


class _CogRaster(_Raster):
    """A (cloud-optimised) GeoTIFF, local path or https URL, read in its own internal blocks.
    `key` (default: the URL) names its cache folder."""

    def __init__(self, url: str, cache_dir: Path, key: str | None = None) -> None:
        super().__init__(key or url, cache_dir)
        self.url = url

    def _read_meta(self) -> dict:
        try:
            with _gdal_env(), rasterio.open(_gdal_path(self.url)) as ds:
                bh, bw = ds.block_shapes[0]
                # Read in the file's own tiles (COGs: 256-1024 px); striped files in 512 px.
                block = int(min(max(bh, 16), 1024)) if bh == bw else 512
                return dict(
                    crs=ds.crs.to_wkt(),
                    transform=list(ds.transform)[:6],
                    width=ds.width,
                    height=ds.height,
                    block=block,
                    nodata=ds.nodata,
                )
        except RasterioIOError as e:
            if _is_missing_error(e):
                return {"missing": True}
            raise SourceUnavailable(f"cannot open {self.url}: {e}") from e

    def _read_blocks(self, keys: list[tuple[int, int]]) -> dict[tuple[int, int], np.ndarray]:
        m = self.meta()
        b, h, w = int(m["block"]), int(m["height"]), int(m["width"])
        out = {}
        try:
            with _gdal_env(), rasterio.open(_gdal_path(self.url)) as ds:
                for bi, bj in keys:
                    r0, c0 = bi * b, bj * b
                    nh, nw = min(b, h - r0), min(b, w - c0)
                    arr = np.full((b, b), np.nan, np.float32)
                    arr[:nh, :nw] = ds.read(1, window=Window(c0, r0, nw, nh))
                    out[(bi, bj)] = _clean(arr, m.get("nodata"))
        except RasterioIOError as e:
            raise SourceUnavailable(f"cannot read {self.url}: {e}") from e
        return out


class _WmsRaster(_Raster):
    """A WMS GetMap layer that returns float GeoTIFF heights, seen as one fixed projected grid
    (origin, pixel size and extent given) whose blocks are fetched as GetMap requests."""

    _last_request = 0.0

    def __init__(
        self,
        url: str,
        layer: str,
        crs_code: str,
        origin: tuple[float, float],
        size: tuple[int, int],
        pixel_m: float,
        nodata: float,
        cache_dir: Path,
        *,
        block: int = 512,
        min_interval_s: float = 0.25,
    ) -> None:
        super().__init__(f"{url}|{layer}|{crs_code}|{pixel_m}", cache_dir)
        self.url, self.layer, self.crs_code = url, layer, crs_code
        self.origin, self.size, self.pixel_m = origin, size, pixel_m
        self.nodata, self.block, self.min_interval_s = nodata, block, min_interval_s

    def _read_meta(self) -> dict:
        x0, y0 = self.origin
        t = Affine(self.pixel_m, 0.0, x0, 0.0, -self.pixel_m, y0)
        return dict(
            crs=self.crs_code,
            transform=list(t)[:6],
            width=self.size[0],
            height=self.size[1],
            block=self.block,
            nodata=self.nodata,
        )

    def _read_blocks(self, keys: list[tuple[int, int]]) -> dict[tuple[int, int], np.ndarray]:
        b, px = self.block, self.pixel_m
        x0, y0 = self.origin
        out = {}
        for bi, bj in keys:
            left, top = x0 + bj * b * px, y0 - bi * b * px
            wait = _WmsRaster._last_request + self.min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)  # IGN asks for <= 5 requests/s per IP (geo.md §2.1)
            _WmsRaster._last_request = time.monotonic()
            data = _http_get(
                self.url,
                dict(
                    SERVICE="WMS",
                    VERSION="1.3.0",
                    REQUEST="GetMap",
                    LAYERS=self.layer,
                    STYLES="",
                    CRS=self.crs_code,
                    BBOX=f"{left},{top - b * px},{left + b * px},{top}",
                    WIDTH=b,
                    HEIGHT=b,
                    FORMAT="image/geotiff",
                ),
            )
            try:
                with MemoryFile(data) as mf, mf.open() as ds:
                    arr = ds.read(1)
            except RasterioIOError as e:
                snippet = data[:200].decode("utf-8", "replace")
                raise SourceUnavailable(f"WMS returned no GeoTIFF: {snippet}") from e
            if arr.shape != (b, b):
                raise SourceUnavailable(f"WMS returned shape {arr.shape}, wanted {(b, b)}")
            out[(bi, bj)] = _clean(arr, self.nodata)
        return out


# ---------------------------------------------------------------------------------------------
# Source kinds
# ---------------------------------------------------------------------------------------------


def _guard(src: ElevationSource, fn: Callable[[], FloatArray], n: int) -> FloatArray:
    try:
        return fn()
    except SourceUnavailable as e:
        warnings.warn(f"elevation source {src.name} unavailable, falling through: {e}",
                      stacklevel=3)
        return np.full(n, np.nan)


class CogSource(ElevationSource):
    """A single GeoTIFF/COG (local path or https URL) covering the whole source area.
    `cache_key` (default: the URL) names its block cache; give a local file one that changes
    with the file's contents."""

    def __init__(self, info: SourceInfo, url: str, cache_key: str | None = None) -> None:
        super().__init__(info)
        self.url = url
        self.cache_key = cache_key

    def sample(self, lon: FloatArray, lat: FloatArray, cache_dir: Path) -> FloatArray:
        raster = _CogRaster(self.url, cache_dir, self.cache_key)
        return _guard(self, lambda: raster.sample(lon, lat), len(lon))


def local_dem_source(path: Path | str, name: str = "dem-file") -> CogSource:
    """A user-supplied GeoTIFF as an elevation source (`drift-maps build --dem-file`).

    The cache key is the resolved path plus the file's size and modification time, so an
    edited or replaced file is read again. The resolution is the file's real pixel size in
    metres (the coarser axis; degrees are converted at the raster's centre latitude).
    """
    path = Path(path).resolve()
    st = path.stat()
    with rasterio.open(path) as ds:
        rx, ry = abs(float(ds.res[0])), abs(float(ds.res[1]))
        crs = ds.crs
        if crs is not None and crs.is_geographic:
            lat = 0.5 * (ds.bounds.bottom + ds.bounds.top)
            rx *= 111_320.0 * math.cos(math.radians(lat))
            ry *= 111_320.0
        elif crs is not None:
            try:
                factor = float(crs.linear_units_factor[1])
            except Exception:  # noqa: BLE001 - unknown units: assume metres
                factor = 1.0
            rx, ry = rx * factor, ry * factor
    info = SourceInfo(
        name,
        f"Local DEM {path.name}",
        round(max(rx, ry), 3),
        "DTM",
        "user supplied",
        f"Elevation from the user-supplied file {path.name}.",
    )
    return CogSource(info, str(path), cache_key=f"{path}|{st.st_size}|{st.st_mtime_ns}")


class TiledCogSource(ElevationSource):
    """1 x 1 degree COG tiles; `tile_url(lat_floor, lon_floor)` names the tile holding a point.
    A tile that doesn't exist (ocean, outside coverage) is remembered as missing."""

    def __init__(self, info: SourceInfo, tile_url: Callable[[int, int], str]) -> None:
        super().__init__(info)
        self.tile_url = tile_url

    def sample(self, lon: FloatArray, lat: FloatArray, cache_dir: Path) -> FloatArray:
        out = np.full(len(lon), np.nan)
        keys = np.stack([np.floor(lat), np.floor(lon)], axis=1).astype(np.int64)
        for k in np.unique(keys, axis=0):
            sel = np.flatnonzero((keys == k).all(axis=1))
            raster = _CogRaster(self.tile_url(int(k[0]), int(k[1])), cache_dir)
            out[sel] = _guard(self, lambda r=raster, s=sel: r.sample(lon[s], lat[s]), len(sel))
        return out


@dataclass(frozen=True)
class CatalogItem:
    """One raster found by a catalogue search: where it is, what it covers, how new it is."""

    url: str
    bbox: tuple[float, float, float, float]  # lon/lat min x, min y, max x, max y
    date: str  # ISO date; newer items are tried first


class CatalogSource(ElevationSource):
    """Rasters found by a catalogue search (STAC, TNM Access) per `cell_deg` cell, newest first.
    Search answers are cached per cell as JSON."""

    def __init__(
        self,
        info: SourceInfo,
        search: Callable[[tuple[float, float, float, float]], list[CatalogItem]],
        *,
        cell_deg: float = 0.1,
    ) -> None:
        super().__init__(info)
        self.search = search
        self.cell_deg = cell_deg

    def items(self, cell: tuple[int, int], cache_dir: Path) -> list[CatalogItem]:
        path = cache_dir / "catalog" / f"{cell[0]}_{cell[1]}.json"
        if path.is_file():
            raw = json.loads(path.read_text())
        else:
            d = self.cell_deg
            bbox = (cell[1] * d, cell[0] * d, (cell[1] + 1) * d, (cell[0] + 1) * d)
            found = self.search(tuple(round(v, 6) for v in bbox))
            raw = [dict(url=i.url, bbox=list(i.bbox), date=i.date) for i in found]
            _atomic_write(path, json.dumps(raw).encode())
        items = [CatalogItem(r["url"], tuple(r["bbox"]), r["date"]) for r in raw]
        return sorted(items, key=lambda i: i.date, reverse=True)

    def sample(self, lon: FloatArray, lat: FloatArray, cache_dir: Path) -> FloatArray:
        out = np.full(len(lon), np.nan)
        cells = np.stack(
            [np.floor(lat / self.cell_deg), np.floor(lon / self.cell_deg)], axis=1
        ).astype(np.int64)
        for c in np.unique(cells, axis=0):
            in_cell = (cells == c).all(axis=1)
            try:
                items = self.items((int(c[0]), int(c[1])), cache_dir)
            except SourceUnavailable as e:
                warnings.warn(f"elevation source {self.name}: search failed: {e}", stacklevel=2)
                continue
            for it in items:
                x0, y0, x1, y1 = it.bbox
                sel = np.flatnonzero(
                    in_cell & np.isnan(out) & (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
                )
                if sel.size == 0:
                    continue
                raster = _CogRaster(it.url, cache_dir)
                out[sel] = _guard(self, lambda r=raster, s=sel: r.sample(lon[s], lat[s]), len(sel))
        return out


class WmsSource(ElevationSource):
    """A WMS layer returning float GeoTIFF heights on a fixed projected grid (IGN WMS-R)."""

    def __init__(self, info: SourceInfo, raster_args: Mapping[str, object]) -> None:
        super().__init__(info)
        self.raster_args = dict(raster_args)

    def sample(self, lon: FloatArray, lat: FloatArray, cache_dir: Path) -> FloatArray:
        raster = _WmsRaster(cache_dir=cache_dir, **self.raster_args)  # type: ignore[arg-type]
        return _guard(self, lambda: raster.sample(lon, lat), len(lon))


# ---------------------------------------------------------------------------------------------
# The concrete sources
# ---------------------------------------------------------------------------------------------

NRCAN_STAC = "https://datacube.services.geo.ca/stac/api/search"
TNM_PRODUCTS = "https://tnmaccess.nationalmap.gov/api/v1/products"
IGN_WMS_R = "https://data.geopf.fr/wms-r"

# Open Government Licence - Canada: the STAC collections say "OGL-Canada-2.0" (VERIFIED live);
# this is the licence's own attribution sentence.
_OGL_CANADA = (
    "Contains information licensed under the Open Government Licence - Canada "
    "(https://open.canada.ca/en/open-government-licence-canada)."
)
_USGS = (
    "Elevation: U.S. Geological Survey, 3D Elevation Program (public domain; "
    "https://www.usgs.gov/3d-elevation-program)."
)


def stac_search(
    collection: str, asset: str = "dtm", api: str = NRCAN_STAC
) -> Callable[[tuple[float, float, float, float]], list[CatalogItem]]:
    """A catalogue search over one STAC collection, returning the `asset` COG of each item."""

    def search(bbox: tuple[float, float, float, float]) -> list[CatalogItem]:
        params = dict(collections=collection, bbox=",".join(f"{v:.6f}" for v in bbox), limit=200)
        doc = json.loads(_http_get(api, params))
        out = []
        for f in doc.get("features", []):
            a = f.get("assets", {}).get(asset)
            if not a or not f.get("bbox"):
                continue
            p = f.get("properties", {})
            date = p.get("datetime") or p.get("end_datetime") or p.get("start_datetime") or ""
            out.append(CatalogItem(a["href"], tuple(f["bbox"][:4]), date))
        return out

    return search


def tnm_search(
    dataset: str = "Digital Elevation Model (DEM) 1 meter",
) -> Callable[[tuple[float, float, float, float]], list[CatalogItem]]:
    """A catalogue search over the USGS TNM Access API (GeoTIFF products of one dataset)."""

    def search(bbox: tuple[float, float, float, float]) -> list[CatalogItem]:
        params = dict(
            datasets=dataset,
            bbox=",".join(f"{v:.6f}" for v in bbox),
            prodFormats="GeoTIFF",
            max=200,
        )
        doc = json.loads(_http_get(TNM_PRODUCTS, params))
        out = []
        for it in doc.get("items", []):
            url, bb = it.get("downloadURL", ""), it.get("boundingBox") or {}
            if not url.lower().endswith(".tif") or not bb:
                continue
            box = (bb["minX"], bb["minY"], bb["maxX"], bb["maxY"])
            out.append(CatalogItem(url, box, it.get("publicationDate") or ""))
        return out

    return search


def _glo30_url(lat: int, lon: int) -> str:
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    name = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def _usgs13_url(lat: int, lon: int) -> str:
    # 1/3" tiles are named by their NW corner (geo.md §2.1).
    ns, ew = ("n" if lat + 1 >= 0 else "s"), ("w" if lon < 0 else "e")
    tile = f"{ns}{abs(lat + 1):02d}{ew}{abs(lon):03d}"
    return (
        "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/"
        f"{tile}/USGS_13_{tile}.tif"
    )


@cache
def default_sources() -> dict[str, ElevationSource]:
    """The built-in sources by name."""
    hrdem = dict(surface="DTM", licence="OGL-Canada-2.0", attribution=_OGL_CANADA
                 + " Source: Natural Resources Canada, High Resolution Digital Elevation Model.")
    srcs: list[ElevationSource] = [
        CatalogSource(
            SourceInfo("hrdem-1m", "NRCan HRDEM mosaic 1 m (lidar DTM)", 1.0, **hrdem),
            stac_search("hrdem-mosaic-1m"),
        ),
        CatalogSource(
            SourceInfo("hrdem-lidar", "NRCan HRDEM lidar projects (DTM)", 1.0, **hrdem),
            stac_search("hrdem-lidar"),
        ),
        CatalogSource(
            SourceInfo("hrdem-2m", "NRCan HRDEM mosaic 2 m (lidar DTM)", 2.0, **hrdem),
            stac_search("hrdem-mosaic-2m"),
        ),
        CogSource(
            SourceInfo(
                "mrdem-30", "NRCan MRDEM-30 (DTM)", 30.0, "DTM", "OGL-Canada-2.0",
                _OGL_CANADA + " Source: Natural Resources Canada, Medium Resolution DEM (MRDEM).",
            ),
            "https://canelevation-dem.s3.ca-central-1.amazonaws.com/mrdem-30/mrdem-30-dtm.tif",
        ),
        CatalogSource(
            SourceInfo("3dep-1m", "USGS 3DEP 1 m project DEM (lidar DTM)", 1.0, "DTM",
                       "Public domain", _USGS),
            tnm_search(),
        ),
        TiledCogSource(
            SourceInfo("3dep-13", "USGS 3DEP 1/3 arc-second DEM", 10.0, "DTM", "Public domain",
                       _USGS),
            _usgs13_url,
        ),
        WmsSource(
            SourceInfo(
                "ign-rge-alti", "IGN RGE ALTI 1 m (France)", 1.0, "DTM",
                "Licence Ouverte / Etalab 2.0",
                "Elevation: IGN - RGE ALTI (Licence Ouverte / Etalab 2.0, "
                "https://geoservices.ign.fr/rgealti).",
            ),
            dict(
                url=IGN_WMS_R,
                layer="ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES",
                crs_code="EPSG:2154",  # Lambert-93, metres
                origin=(0.0, 7_200_000.0),
                size=(1_300_000, 1_250_000),  # covers metropolitan France and Corsica
                pixel_m=1.0,
                nodata=-99999.0,
            ),
        ),
        TiledCogSource(
            SourceInfo(
                "glo30", "Copernicus DEM GLO-30 (DSM)", 30.0, "DSM", "Copernicus DEM licence",
                "Produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) Airbus "
                "Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European "
                "Union and ESA; all rights reserved. The organisations in charge of the "
                "Copernicus programme by law or by delegation do not incur any liability for "
                "any use of the Copernicus WorldDEM-30.",
            ),
            _glo30_url,
        ),
    ]
    return {s.name: s for s in srcs}


# ---------------------------------------------------------------------------------------------
# Regions and chains
# ---------------------------------------------------------------------------------------------

CHAINS: dict[str, tuple[str, ...]] = {
    "CA": ("hrdem-1m", "hrdem-lidar", "hrdem-2m", "mrdem-30", "glo30"),
    # Near the border: both countries' lidar first, then the home country's coarse DEM.
    "CA-border": ("hrdem-1m", "hrdem-lidar", "hrdem-2m", "3dep-1m", "mrdem-30", "3dep-13", "glo30"),
    "US-border": ("3dep-1m", "hrdem-1m", "hrdem-lidar", "3dep-13", "mrdem-30", "glo30"),
    "US": ("3dep-1m", "3dep-13", "glo30"),
    "FR": ("ign-rge-alti", "glo30"),
    "WORLD": ("glo30",),
}

# Coarse outline of Canada (lon, lat). Only its southern edge matters: it follows the
# Canada-US border to ~10-20 km (ASSUMED good enough: near the border both countries' lidar is
# tried anyway, this only orders the coarse fallbacks).
_CANADA = np.array([
    (-141.0, 84.0), (-141.0, 60.3), (-137.5, 59.0), (-135.5, 59.8), (-133.4, 58.4),
    (-131.0, 56.2), (-130.0, 55.9), (-130.0, 55.2), (-130.7, 54.7), (-134.0, 54.4),
    (-130.0, 50.0), (-125.5, 48.4), (-123.2, 48.3), (-123.0, 48.8), (-122.75, 49.0),
    (-95.15, 49.0), (-95.15, 49.38), (-94.82, 49.32), (-94.6, 48.72), (-93.4, 48.6),
    (-92.0, 48.35), (-90.8, 48.2), (-89.6, 48.0), (-88.4, 48.3), (-84.9, 46.9),
    (-84.5, 46.48), (-84.1, 46.2), (-83.6, 46.0), (-82.5, 45.35), (-82.15, 43.6),
    (-82.42, 43.0), (-82.5, 42.62), (-82.9, 42.35), (-83.1, 42.28), (-83.13, 42.05),
    (-82.7, 41.68), (-81.25, 42.2), (-79.75, 42.5), (-78.92, 42.9), (-79.05, 43.27),
    (-77.5, 43.6), (-76.45, 44.0), (-76.2, 44.25), (-75.3, 44.85), (-74.65, 45.0),
    (-71.5, 45.01), (-71.1, 45.3), (-70.6, 45.6), (-70.25, 46.1), (-70.0, 46.7),
    (-69.25, 47.45), (-68.3, 47.35), (-67.8, 47.07), (-67.8, 45.7), (-67.45, 45.25),
    (-66.9, 44.75), (-67.0, 44.3), (-65.5, 42.5), (-52.0, 42.5), (-52.0, 84.0),
])  # fmt: skip


def _in_polygon(x: np.ndarray, y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    inside = np.zeros(x.shape, bool)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for a, b, c, d in zip(x1, y1, x2, y2, strict=True):
        crosses = (b > y) != (d > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = (c - a) * (y - b) / (d - b) + a
        inside ^= crosses & (x < xc)
    return inside


def _in_box(lon, lat, x0, y0, x1, y1):
    return (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)


def region_of(lon: npt.ArrayLike, lat: npt.ArrayLike) -> npt.NDArray[np.str_]:
    """Region code (a `CHAINS` key) for each point: CA, CA-border, US, US-border, FR or WORLD."""
    lon = np.atleast_1d(np.asarray(lon, np.float64))
    lat = np.atleast_1d(np.asarray(lat, np.float64))
    canada = _in_polygon(lon, lat, _CANADA)
    us = ~canada & (
        _in_box(lon, lat, -125.0, 24.0, -66.5, 49.5)  # CONUS
        | _in_box(lon, lat, -180.0, 51.0, -129.9, 72.0)  # Alaska
        | _in_box(lon, lat, -161.0, 18.5, -154.0, 22.5)  # Hawaii
    )
    border = _in_box(lon, lat, -125.0, 41.6, -66.5, 49.5)
    france = _in_box(lon, lat, -5.3, 41.3, 9.7, 51.2)
    out = np.full(lon.shape, "WORLD", dtype="<U9")
    out[france] = "FR"
    out[us] = "US"
    out[us & border] = "US-border"
    out[canada] = "CA"
    out[canada & border] = "CA-border"
    return out


def _with_prefer(chain: Sequence[str], prefer: str | Sequence[str] | None) -> tuple[str, ...]:
    if prefer is None:
        return tuple(chain)
    first = (prefer,) if isinstance(prefer, str) else tuple(prefer)
    return first + tuple(n for n in chain if n not in first)


def source_chain(
    lon: float, lat: float, prefer: str | Sequence[str] | None = None
) -> tuple[str, ...]:
    """Ordered source names tried for one point; `prefer` names go first."""
    return _with_prefer(CHAINS[str(region_of(lon, lat)[0])], prefer)


# ---------------------------------------------------------------------------------------------
# Public sampling API
# ---------------------------------------------------------------------------------------------


def sample_elevation(
    lon: npt.ArrayLike,
    lat: npt.ArrayLike,
    *,
    cache_dir: Path | str | None = None,
    prefer: str | Sequence[str] | None = None,
    sources: Mapping[str, ElevationSource] | None = None,
    chain: Sequence[str] | None = None,
) -> ElevationSample:
    """Heights at N lon/lat points (WGS84 degrees), each from the best source that covers it.

    - `cache_dir`: cache root (default `default_cache_dir()`); data goes in `<root>/elevation/`.
    - `prefer`: source name(s) tried before the regional chain (e.g. "glo30" to compare).
    - `sources` / `chain`: override the source registry and use one chain for every point
      (tests and power users). Without `chain` each point uses `CHAINS[region_of(point)]`.
    """
    lon = np.asarray(lon, np.float64).ravel()
    lat = np.asarray(lat, np.float64).ravel()
    if lon.shape != lat.shape:
        raise ValueError(f"lon and lat differ in shape: {lon.shape} vs {lat.shape}")
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError("lon/lat must be finite")
    sources = default_sources() if sources is None else sources
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    root = root / "elevation"

    n = lon.size
    heights = np.full(n, np.nan)
    names = np.full(n, "", dtype=object)
    res = np.full(n, np.nan)
    used: dict[str, SourceInfo] = {}

    if chain is not None:
        groups = [(_with_prefer(chain, prefer), np.arange(n))]
    else:
        regions = region_of(lon, lat) if n else np.array([], dtype="<U9")
        groups = [
            (_with_prefer(CHAINS[str(r)], prefer), np.flatnonzero(regions == r))
            for r in np.unique(regions)
        ]
    for names_in_chain, idx in groups:
        for name in names_in_chain:
            if name not in sources:
                raise ValueError(f"unknown elevation source {name!r}; have {sorted(sources)}")
            todo = idx[np.isnan(heights[idx])]
            if todo.size == 0:
                break
            src = sources[name]
            z = np.asarray(src.sample(lon[todo], lat[todo], root / src.name), np.float64)
            got = np.isfinite(z)
            if got.any():
                heights[todo[got]] = z[got]
                names[todo[got]] = name
                res[todo[got]] = src.info.resolution_m
                used[name] = src.info
    missing = int(np.isnan(heights).sum())
    if missing:
        log.warning("no elevation for %d of %d points", missing, n)
    return ElevationSample(heights, names.astype(str), res, used)


def sample_grid(
    lon_grid: npt.ArrayLike,
    lat_grid: npt.ArrayLike,
    **kwargs: object,
) -> ElevationSample:
    """`sample_elevation` on a grid of points (any shape); the arrays come back in that shape.
    For terrain heightfields; keyword arguments as for `sample_elevation`."""
    lon_grid = np.asarray(lon_grid, np.float64)
    lat_grid = np.asarray(lat_grid, np.float64)
    if lon_grid.shape != lat_grid.shape:
        raise ValueError(f"grid shapes differ: {lon_grid.shape} vs {lat_grid.shape}")
    s = sample_elevation(lon_grid.ravel(), lat_grid.ravel(), **kwargs)  # type: ignore[arg-type]
    shape = lon_grid.shape
    return ElevationSample(
        s.heights.reshape(shape), s.source.reshape(shape), s.resolution_m.reshape(shape), s.info
    )
