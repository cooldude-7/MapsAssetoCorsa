"""Tests for driftmaps.elevation.sources: source chains by location, windowed + cached reads,
NoData / missing-tile / outage fallthrough, catalogue and WMS sources.

Offline: real DEM windows live in tests/fixtures/dem (see tests/fixtures/README.md), synthetic
rasters are written to tmp_path. Tests marked `network` hit the live services and are skipped
unless selected with `pytest -m network`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform

from driftmaps.elevation import sources as srcmod
from driftmaps.elevation.sources import (
    CHAINS,
    CatalogItem,
    CatalogSource,
    CogSource,
    SourceInfo,
    TiledCogSource,
    WmsSource,
    default_sources,
    region_of,
    sample_elevation,
    sample_grid,
    source_chain,
)

FIXTURES = Path(__file__).parent / "fixtures" / "dem"

# Fixture windows and the heights the LIVE pipeline returned at their centre points
# (2026-10-15). The fixtures are exact copies of the source pixels, so they must agree.
TORONTO = (-79.3957, 43.7046)
AURORA = (-88.29689, 41.71712)
OLLIOULES = (5.85, 43.14)
FIXTURE_VALUES = {
    "hrdem_toronto_1m.tif": (TORONTO, 156.425),  # NRCan HRDEM 1 m mosaic, EPSG:3979
    "usgs13_aurora.tif": (AURORA, 203.173),  # USGS 3DEP 1/3", EPSG:4269
    "glo30_ollioules.tif": (OLLIOULES, 63.927),  # Copernicus GLO-30, EPSG:4326
}


@pytest.fixture(autouse=True)
def _network_opt_in(request):
    if request.node.get_closest_marker("network") and "network" not in (
        request.config.option.markexpr or ""
    ):
        pytest.skip("live network test: run with `pytest -m network`")


def _info(name: str, res: float = 1.0, surface: str = "DTM") -> SourceInfo:
    return SourceInfo(name, name, res, surface, "test", f"attribution for {name}")


def _write_tif(path: Path, arr: np.ndarray, *, west: float, north: float, px: float,
               nodata: float | None = -9999.0, crs: str = "EPSG:4326", block: int = 16) -> None:
    h, w = arr.shape
    with rasterio.open(
        path, "w", driver="GTiff", width=w, height=h, count=1, dtype="float32", crs=crs,
        transform=from_origin(west, north, px, px), nodata=nodata, tiled=True,
        blockxsize=block, blockysize=block,
    ) as dst:
        dst.write(arr.astype(np.float32), 1)


# ---------------------------------------------------------------------------------------------
# Regions and chains
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lon, lat, region, first",
    [
        (*TORONTO, "CA-border", "hrdem-1m"),
        (-75.6972, 45.4215, "CA-border", "hrdem-1m"),  # Ottawa
        (-82.60, 42.05, "CA-border", "hrdem-1m"),  # Leamington, south of Detroit's latitude
        (-80.65, 51.27, "CA", "hrdem-1m"),  # Moosonee
        (-123.12, 49.28, "CA-border", "hrdem-1m"),  # Vancouver
        (-135.05, 60.72, "CA", "hrdem-1m"),  # Whitehorse
        (*AURORA, "US-border", "3dep-1m"),
        (-122.33, 47.61, "US-border", "3dep-1m"),  # Seattle
        (-104.99, 39.74, "US", "3dep-1m"),  # Denver
        (-149.90, 61.20, "US", "3dep-1m"),  # Anchorage
        (*OLLIOULES, "FR", "ign-rge-alti"),
        (139.70, 35.68, "WORLD", "glo30"),  # Tokyo
    ],
)
def test_region_and_first_source(lon, lat, region, first):
    assert region_of(lon, lat)[0] == region
    chain = source_chain(lon, lat)
    assert chain[0] == first
    assert chain[-1] == "glo30"  # global fallback everywhere


def test_chains_name_real_sources_and_canada_tries_hrdem_before_coarse():
    known = default_sources()
    for chain in CHAINS.values():
        assert set(chain) <= set(known)
        assert len(set(chain)) == len(chain)
    for region in ("CA", "CA-border"):
        chain = CHAINS[region]
        assert chain.index("hrdem-1m") < chain.index("mrdem-30") < chain.index("glo30")
    # Near the border both countries' lidar is tried before any coarse DEM.
    for region in ("CA-border", "US-border"):
        chain = CHAINS[region]
        coarse = min(chain.index("mrdem-30"), chain.index("3dep-13"))
        assert chain.index("hrdem-1m") < coarse and chain.index("3dep-1m") < coarse
    assert source_chain(*OLLIOULES) == ("ign-rge-alti", "glo30")
    assert source_chain(139.7, 35.68) == ("glo30",)


def test_prefer_goes_first_without_duplicates():
    chain = source_chain(*TORONTO, prefer="glo30")
    assert chain[0] == "glo30" and chain.count("glo30") == 1
    assert source_chain(*AURORA, prefer=["3dep-13", "glo30"])[:3] == ("3dep-13", "glo30", "3dep-1m")


def test_unknown_preferred_source_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="unknown elevation source"):
        sample_elevation([0.0], [0.0], cache_dir=tmp_path, prefer="nope", sources={}, chain=[])


def test_licences_are_set():
    for src in default_sources().values():
        assert src.info.attribution and src.info.licence
        assert src.info.surface in ("DTM", "DSM")
    srcs = default_sources()
    assert "Open Government Licence - Canada" in srcs["hrdem-1m"].info.attribution
    assert srcs["glo30"].info.surface == "DSM"


def test_tile_names():
    assert srcmod._glo30_url(43, 5).endswith(
        "Copernicus_DSM_COG_10_N43_00_E005_00_DEM/Copernicus_DSM_COG_10_N43_00_E005_00_DEM.tif"
    )
    assert "Copernicus_DSM_COG_10_S23_00_W047_00_DEM" in srcmod._glo30_url(-23, -47)
    # 3DEP 1/3" tiles are named by their NW corner: Aurora (41.7 N, 88.3 W) -> n42w089.
    assert srcmod._usgs13_url(41, -89).endswith("/n42w089/USGS_13_n42w089.tif")


# ---------------------------------------------------------------------------------------------
# Windowed, cached reads
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURE_VALUES))
def test_fixture_window_matches_live_value(tmp_path, name):
    (lon, lat), expected = FIXTURE_VALUES[name]
    src = CogSource(_info("fx"), str(FIXTURES / name))
    z = src.sample(np.array([lon]), np.array([lat]), tmp_path)
    assert z[0] == pytest.approx(expected, abs=2e-3)


def test_windowed_read_touches_only_needed_blocks_and_caches_them(tmp_path):
    tif = tmp_path / "src" / "hrdem.tif"
    tif.parent.mkdir()
    shutil.copy(FIXTURES / "hrdem_toronto_1m.tif", tif)
    with rasterio.open(tif) as ds:
        assert ds.block_shapes[0] == (32, 32) and ds.shape == (96, 96)  # 3 x 3 blocks
        data = ds.read(1)
        # Pixel centres (row, col) inside block (0, 0) and block (2, 1).
        rc = [(5, 5), (6, 9), (70, 40)]
        xs, ys = zip(*(ds.xy(r, c) for r, c in rc), strict=True)
        lon, lat = warp_transform(ds.crs, "EPSG:4326", list(xs), list(ys))
        # Halfway between two horizontally adjacent pixel centres.
        mx, my = ds.xy(20, 10)
        nx, _ = ds.xy(20, 11)
        mlon, mlat = warp_transform(ds.crs, "EPSG:4326", [(mx + nx) / 2], [my])
    cache = tmp_path / "cache"
    src = CogSource(_info("fx"), str(tif))
    z = src.sample(np.array(lon), np.array(lat), cache)
    np.testing.assert_allclose(z, [data[r, c] for r, c in rc], atol=1e-3)
    zm = src.sample(np.array(mlon), np.array(mlat), cache)
    assert zm[0] == pytest.approx((data[20, 10] + data[20, 11]) / 2, abs=1e-3)

    cached = sorted(p.name for p in cache.rglob("b*.npy"))
    assert cached == ["b0_0.npy", "b2_1.npy"]  # 2 of the 9 blocks read
    assert len(list(cache.rglob("meta.json"))) == 1

    tif.unlink()  # everything needed is cached now
    z2 = src.sample(np.array(lon), np.array(lat), cache)
    np.testing.assert_array_equal(z, z2)


def test_points_outside_raster_are_nan(tmp_path):
    src = CogSource(_info("fx"), str(FIXTURES / "glo30_ollioules.tif"))
    z = src.sample(np.array([5.85, 6.5, -100.0]), np.array([43.14, 43.14, 40.0]), tmp_path)
    assert np.isfinite(z[0]) and np.isnan(z[1:]).all()


# ---------------------------------------------------------------------------------------------
# Fallthrough
# ---------------------------------------------------------------------------------------------


def test_nodata_falls_through_to_next_source(tmp_path):
    # "fine": 1 x 1 degree at 0.01 deg, 100 m, with a NoData hole in its middle.
    fine = np.full((100, 100), 100.0)
    fine[40:60, 40:60] = -9999.0
    _write_tif(tmp_path / "fine.tif", fine, west=10.0, north=46.0, px=0.01)
    # "coarse": 2 x 2 degrees at 0.1 deg, 50 m everywhere.
    _write_tif(tmp_path / "coarse.tif", np.full((20, 20), 50.0), west=9.5, north=46.5, px=0.1,
               nodata=None)
    sources = {
        "fine": CogSource(_info("fine", 10.0), str(tmp_path / "fine.tif")),
        "coarse": CogSource(_info("coarse", 30.0, "DSM"), str(tmp_path / "coarse.tif")),
    }
    lon = np.array([10.2, 10.5, 9.7, 20.0])  # in fine, in the hole, coarse only, nowhere
    lat = np.array([45.2, 45.5, 45.0, 45.0])
    r = sample_elevation(lon, lat, cache_dir=tmp_path / "c", sources=sources,
                         chain=["fine", "coarse"])
    np.testing.assert_allclose(r.heights[:3], [100.0, 50.0, 50.0])
    assert np.isnan(r.heights[3])
    assert list(r.source) == ["fine", "coarse", "coarse", ""]
    np.testing.assert_array_equal(r.resolution_m[:3], [10.0, 30.0, 30.0])
    assert r.coarsest_resolution_m == 30.0 and r.any_dsm
    assert r.attributions == ["attribution for fine", "attribution for coarse"]

    # Bilinear next to the hole renormalises over valid pixels instead of mixing in -9999.
    edge = sample_elevation([10.395], [45.6], cache_dir=tmp_path / "c", sources=sources,
                            chain=["fine", "coarse"])
    assert edge.heights[0] == pytest.approx(100.0) and edge.source[0] == "fine"


def test_missing_tile_is_remembered_and_falls_through(tmp_path):
    _write_tif(tmp_path / "t_45_10.tif", np.full((10, 10), 7.0), west=10.0, north=46.0, px=0.1)
    calls = []

    def tile_url(lat: int, lon: int) -> str:
        calls.append((lat, lon))
        return str(tmp_path / f"t_{lat}_{lon}.tif")

    sources = {
        "tiles": TiledCogSource(_info("tiles"), tile_url),
        "flat": CogSource(_info("flat"), str(tmp_path / "flat.tif")),
    }
    _write_tif(tmp_path / "flat.tif", np.full((10, 20), 3.0), west=10.0, north=46.0, px=0.1)
    r = sample_elevation([10.5, 11.5], [45.5, 45.5], cache_dir=tmp_path / "c",
                         sources=sources, chain=["tiles", "flat"])
    np.testing.assert_allclose(r.heights, [7.0, 3.0])
    assert sorted(calls) == [(45, 10), (45, 11)]
    metas = [p.read_text() for p in (tmp_path / "c").rglob("meta.json")]
    assert sum('"missing": true' in m for m in metas) == 1


def test_unreachable_server_warns_and_falls_through(tmp_path):
    _write_tif(tmp_path / "flat.tif", np.full((10, 10), 3.0), west=0.0, north=1.0, px=0.1)
    sources = {
        # Nothing listens on port 9 of localhost: connection refused, no internet needed.
        "down": CogSource(_info("down"), "http://127.0.0.1:9/dem.tif"),
        "flat": CogSource(_info("flat"), str(tmp_path / "flat.tif")),
    }
    with pytest.warns(UserWarning, match="down unavailable"):
        r = sample_elevation([0.5], [0.5], cache_dir=tmp_path / "c", sources=sources,
                             chain=["down", "flat"])
    assert r.heights[0] == pytest.approx(3.0) and r.source[0] == "flat"
    assert not list((tmp_path / "c" / "elevation" / "down").rglob("meta.json"))  # not cached


def test_catalog_source_tries_newest_item_first_and_caches_search(tmp_path):
    old = np.full((10, 10), 10.0)
    new = np.full((10, 10), 20.0)
    new[:, 5:] = -9999.0  # the newer survey only covers the western half
    _write_tif(tmp_path / "old.tif", old, west=5.0, north=43.1, px=0.01)
    _write_tif(tmp_path / "new.tif", new, west=5.0, north=43.1, px=0.01)
    searches = []

    def search(bbox):
        searches.append(bbox)
        return [
            CatalogItem(str(tmp_path / "old.tif"), (5.0, 43.0, 5.1, 43.1), "2015-01-01"),
            CatalogItem(str(tmp_path / "new.tif"), (5.0, 43.0, 5.1, 43.1), "2023-06-01"),
            CatalogItem(str(tmp_path / "far.tif"), (7.0, 44.0, 7.1, 44.1), "2025-01-01"),
        ]

    sources = {"cat": CatalogSource(_info("cat"), search)}
    lon, lat = np.array([5.02, 5.08]), np.array([43.05, 43.05])
    r = sample_elevation(lon, lat, cache_dir=tmp_path / "c", sources=sources, chain=["cat"])
    np.testing.assert_allclose(r.heights, [20.0, 10.0])
    assert len(searches) == 1
    assert searches[0] == pytest.approx((5.0, 43.0, 5.1, 43.1))
    sample_elevation(lon, lat, cache_dir=tmp_path / "c", sources=sources, chain=["cat"])
    assert len(searches) == 1  # second run answered from the cached search


def test_wms_source_fetches_blocks_on_its_grid(tmp_path, monkeypatch):
    # A fake WMS that returns the plane z = 100 + 0.001 (x - 935000) + 0.002 (y - 6235000)
    # in Lambert-93 for whatever BBOX it is asked for. Bilinear sampling of a plane is exact.
    requests = []

    def fake_get(url, params=None, **_):
        requests.append(params)
        x0, y0, x1, y1 = map(float, params["BBOX"].split(","))
        w, h = int(params["WIDTH"]), int(params["HEIGHT"])
        px = (x1 - x0) / w
        xs = x0 + (np.arange(w) + 0.5) * px
        ys = y1 - (np.arange(h) + 0.5) * px
        z = 100 + 0.001 * (xs[None, :] - 935000) + 0.002 * (ys[:, None] - 6235000)
        with MemoryFile() as mf:
            with mf.open(driver="GTiff", width=w, height=h, count=1, dtype="float32",
                         crs="EPSG:2154", transform=from_origin(x0, y1, px, px)) as ds:
                ds.write(z.astype(np.float32), 1)
            return mf.read()

    monkeypatch.setattr(srcmod, "_http_get", fake_get)
    src = WmsSource(_info("wms"), dict(
        url="https://example.invalid/wms", layer="L", crs_code="EPSG:2154",
        origin=(0.0, 7_200_000.0), size=(1_300_000, 1_250_000), pixel_m=1.0, nodata=-99999.0,
        block=64, min_interval_s=0.0,
    ))
    lon, lat = np.array([5.85, 5.8501]), np.array([43.14, 43.1401])
    z = src.sample(lon, lat, tmp_path)
    x, y = warp_transform("EPSG:4326", "EPSG:2154", lon.tolist(), lat.tolist())
    expected = 100 + 0.001 * (np.array(x) - 935000) + 0.002 * (np.array(y) - 6235000)
    np.testing.assert_allclose(z, expected, atol=2e-3)
    assert all(p["CRS"] == "EPSG:2154" and p["FORMAT"] == "image/geotiff" for p in requests)
    n = len(requests)
    src.sample(lon, lat, tmp_path)
    assert len(requests) == n  # cached blocks


def test_sample_grid_keeps_shape(tmp_path):
    _write_tif(tmp_path / "flat.tif", np.full((10, 10), 3.0), west=0.0, north=1.0, px=0.1)
    sources = {"flat": CogSource(_info("flat"), str(tmp_path / "flat.tif"))}
    lon, lat = np.meshgrid(np.linspace(0.1, 0.9, 4), np.linspace(0.1, 0.9, 3))
    g = sample_grid(lon, lat, cache_dir=tmp_path, sources=sources, chain=["flat"])
    assert g.heights.shape == (3, 4) and g.source.shape == (3, 4)
    np.testing.assert_allclose(g.heights, 3.0)


# ---------------------------------------------------------------------------------------------
# Live services (pytest -m network)
# ---------------------------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.parametrize(
    "lon, lat, source, expected",
    [
        (*TORONTO, "hrdem-1m", 156.43),  # NRCan HRDEM 1 m mosaic
        (-75.6972, 45.4215, "hrdem-1m", 71.03),  # Ottawa, Parliament Hill area
        (*AURORA, "3dep-1m", 203.23),  # USGS EPQS gives 203.24 here (geo.md)
        (*OLLIOULES, "ign-rge-alti", 65.52),  # IGN point API gives 65.47 here
        (139.70, 35.68, "glo30", 46.53),  # Tokyo
    ],
)
def test_live_default_chain(tmp_path, lon, lat, source, expected):
    r = sample_elevation([lon], [lat], cache_dir=tmp_path)
    assert r.source[0] == source
    assert r.heights[0] == pytest.approx(expected, abs=0.5)
