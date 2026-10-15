"""Tests for the surface-model (DSM) side of driftmaps.elevation.sources: SURFACE_CHAINS,
`sample_surface_elevation` and `canopy_height_grid` (DSM - DTM, for trees and buildings).

Offline: a 64 x 64 m DSM/DTM pair cut from the NRCan HRDEM 1 m mosaic beside Snake Road
(tests/fixtures/dem/hrdem_snake_{dsm,dtm}_1m.tif, see tests/fixtures/README.md), plus synthetic
rasters in tmp_path. The `network` test reads the live COGs and is skipped unless selected with
`pytest -m network`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from driftmaps.elevation import sources as srcmod
from driftmaps.elevation.sources import (
    CHAINS,
    DSM_TO_DTM,
    SURFACE_CHAINS,
    CogSource,
    ElevationSource,
    SourceInfo,
    canopy_height_grid,
    default_sources,
    sample_elevation,
    sample_surface_elevation,
)

FIXTURES = Path(__file__).parent / "fixtures" / "dem"
DSM_TIF = FIXTURES / "hrdem_snake_dsm_1m.tif"
DTM_TIF = FIXTURES / "hrdem_snake_dtm_1m.tif"

# Pixel centres inside the fixture and what the LIVE default pipeline returned there on
# 2026-10-15 (sample_surface_elevation / sample_elevation / canopy_height_grid, sources
# hrdem-1m-dsm and hrdem-1m). The fixtures are exact copies of the source pixels.
# Coordinates are the exact pixel centres (10 decimals: at 7 decimals the bilinear weights
# already pull in ~1 mm from neighbouring pixels).
#                  lon              lat            DSM      DTM      canopy
REF = {
    "roof": (-79.8897903099, 43.3102417704, 159.447, 150.955, 8.491),  # OSM building
    "tallest_tree": (-79.8900168277, 43.3104533266, 170.473, 150.825, 19.648),
    "bare_ground": (-79.8900496508, 43.3102424927, 149.926, 149.905, 0.021),
}


@pytest.fixture(autouse=True)
def _network_opt_in(request):
    if request.node.get_closest_marker("network") and "network" not in (
        request.config.option.markexpr or ""
    ):
        pytest.skip("live network test: run with `pytest -m network`")


class _Boom(ElevationSource):
    """A source that must never be asked for heights."""

    def __init__(self, name: str) -> None:
        super().__init__(SourceInfo(name, name, 1.0, "DSM", "test", "test"))

    def sample(self, lon, lat, cache_dir):
        raise AssertionError(f"source {self.name} should not have been read")


def _fixture_sources() -> dict[str, ElevationSource]:
    """The default registry's names, with the 1 m mosaic DSM/DTM served from the fixtures and
    every other source forbidden."""
    real = default_sources()
    srcs: dict[str, ElevationSource] = {name: _Boom(name) for name in real}
    srcs["hrdem-1m-dsm"] = CogSource(real["hrdem-1m-dsm"].info, str(DSM_TIF))
    srcs["hrdem-1m"] = CogSource(real["hrdem-1m"].info, str(DTM_TIF))
    return srcs


def _info(name: str, surface: str) -> SourceInfo:
    return SourceInfo(name, name, 1.0, surface, "test", f"attribution for {name}")


def _write_tif(path: Path, arr: np.ndarray, *, west: float = -80.0, north: float = 43.5,
               px: float = 1e-4, nodata: float = -32767.0) -> None:
    h, w = arr.shape
    with rasterio.open(
        path, "w", driver="GTiff", width=w, height=h, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_origin(west, north, px, px), nodata=nodata,
        tiled=True, blockxsize=16, blockysize=16,
    ) as dst:
        dst.write(arr.astype(np.float32), 1)


# ---------------------------------------------------------------------------------------------
# Registry and chains
# ---------------------------------------------------------------------------------------------


def test_surface_chains_use_lidar_dsms_paired_with_same_collection_dtms():
    srcs = default_sources()
    assert set(SURFACE_CHAINS) == set(CHAINS)  # every region has an entry
    assert SURFACE_CHAINS["CA"][0] == "hrdem-1m-dsm"
    for region in ("US", "FR", "WORLD"):
        assert SURFACE_CHAINS[region] == ()  # no coarse/radar DSM fallback
    for chain in SURFACE_CHAINS.values():
        for name in chain:
            assert srcs[name].info.surface == "DSM"
            dtm = srcs[DSM_TO_DTM[name]].info
            assert dtm.surface == "DTM"
            assert dtm.resolution_m == srcs[name].info.resolution_m
            assert "Open Government Licence - Canada" in srcs[name].info.attribution
    # The terrain chains never pick up a lidar DSM by accident.
    for chain in CHAINS.values():
        assert not set(chain) & set(DSM_TO_DTM)


def test_dsm_sources_search_the_dsm_asset_of_the_same_stac_items(monkeypatch):
    calls = []

    def fake_get(url, params=None, *, tries=3):
        calls.append(params["collections"])
        item = {
            "bbox": [-82.8, 42.6, -75.1, 48.0],
            "properties": {"datetime": "2025-11-14T12:00:00Z"},
            "assets": {
                "dtm": {"href": "https://example.test/8_2-mosaic-1m-dtm.tif"},
                "dsm": {"href": "https://example.test/8_2-mosaic-1m-dsm.tif"},
            },
        }
        return json.dumps({"features": [item]}).encode()

    monkeypatch.setattr(srcmod, "_http_get", fake_get)
    srcs = default_sources()
    bbox = (-79.9, 43.3, -79.8, 43.4)
    assert srcs["hrdem-1m-dsm"].search(bbox)[0].url.endswith("-dsm.tif")
    assert srcs["hrdem-1m"].search(bbox)[0].url.endswith("-dtm.tif")
    assert srcs["hrdem-lidar-dsm"].search(bbox)[0].url.endswith("-dsm.tif")
    assert calls == ["hrdem-mosaic-1m", "hrdem-mosaic-1m", "hrdem-lidar"]


# ---------------------------------------------------------------------------------------------
# sample_surface_elevation
# ---------------------------------------------------------------------------------------------


def test_surface_elevation_from_the_snake_road_fixture(tmp_path):
    lon = [v[0] for v in REF.values()]
    lat = [v[1] for v in REF.values()]
    r = sample_surface_elevation(lon, lat, cache_dir=tmp_path, sources=_fixture_sources())
    np.testing.assert_allclose(r.heights, [v[2] for v in REF.values()], atol=1e-3)
    assert list(r.source) == ["hrdem-1m-dsm"] * 3
    assert r.any_dsm and r.coarsest_resolution_m == 1.0
    # Second call is served from the block cache.
    again = sample_surface_elevation(lon, lat, cache_dir=tmp_path, sources=_fixture_sources())
    np.testing.assert_array_equal(again.heights, r.heights)
    assert any((tmp_path / "elevation" / "hrdem-1m-dsm").rglob("*.npy"))


def test_surface_elevation_is_nan_outside_canada_without_reading_anything(tmp_path):
    srcs = {name: _Boom(name) for name in default_sources()}
    lon = [5.85, 139.70, -100.0]  # France, Tokyo, Kansas
    lat = [43.14, 35.68, 38.5]
    r = sample_surface_elevation(lon, lat, cache_dir=tmp_path, sources=srcs)
    assert np.isnan(r.heights).all()
    assert list(r.source) == ["", "", ""]
    assert r.info == {}


def test_surface_elevation_falls_through_the_chain(tmp_path):
    first = np.full((20, 20), 200.0)
    first[:, 10:] = -32767.0  # NoData on the east half
    _write_tif(tmp_path / "a.tif", first)
    _write_tif(tmp_path / "b.tif", np.full((20, 20), 300.0))
    srcs = {
        "a": CogSource(_info("a", "DSM"), str(tmp_path / "a.tif")),
        "b": CogSource(_info("b", "DSM"), str(tmp_path / "b.tif")),
    }
    lon, lat = [-79.99975, -79.99875], [43.49905, 43.49905]  # west half, east half
    r = sample_surface_elevation(lon, lat, cache_dir=tmp_path / "c", sources=srcs,
                                 chain=["a", "b"])
    np.testing.assert_allclose(r.heights, [200.0, 300.0])
    assert list(r.source) == ["a", "b"]


# ---------------------------------------------------------------------------------------------
# canopy_height_grid
# ---------------------------------------------------------------------------------------------


def test_canopy_height_grid_from_the_snake_road_fixture(tmp_path):
    srcs = _fixture_sources()
    # A 2-D grid over the inner +-22 m of the fixture (EPSG:3979 is rotated ~12 degrees from
    # lon/lat here, so this stays inside the 64 m window); steps ~2 m.
    lo, la = np.meshgrid(np.linspace(-79.89006, -79.88952, 24),
                         np.linspace(43.31004, 43.31044, 18))
    c = canopy_height_grid(lo, la, cache_dir=tmp_path, sources=srcs)
    assert c.heights.shape == lo.shape and c.source.shape == lo.shape
    assert np.isfinite(c.heights).all() and (c.heights >= 0).all()
    assert set(c.source.ravel()) == {"hrdem-1m-dsm"}
    assert set(c.info) == {"hrdem-1m-dsm", "hrdem-1m"}  # both go in LICENSE-DATA.txt
    # It is exactly DSM - DTM (both bilinear at the same points), clamped at 0.
    dsm = sample_surface_elevation(lo, la, cache_dir=tmp_path, sources=srcs).heights
    dtm = sample_elevation(lo, la, cache_dir=tmp_path, sources=srcs, chain=["hrdem-1m"]).heights
    np.testing.assert_allclose(c.heights.ravel(), np.maximum(dsm - dtm, 0.0), atol=1e-9)
    # Trees, roof and open ground are all in this window.
    assert (c.heights > 5).mean() > 0.2 and (c.heights < 0.5).mean() > 0.1
    assert c.heights.max() > 15


@pytest.mark.parametrize("name", list(REF))
def test_canopy_height_at_reference_points(tmp_path, name):
    lon, lat, _, _, canopy = REF[name]
    c = canopy_height_grid(np.array([[lon]]), np.array([[lat]]), cache_dir=tmp_path,
                           sources=_fixture_sources())
    assert c.heights.shape == (1, 1)
    assert c.heights[0, 0] == pytest.approx(canopy, abs=1e-3)


def test_canopy_pairs_each_dsm_with_its_own_dtm_and_clamps(tmp_path):
    # Two "collections": A covers the west half only, B everything. Each DSM must be paired
    # with its own DTM, never with the other collection's.
    dsm_a = np.full((20, 20), 112.0)
    dsm_a[:, 10:] = -32767.0
    _write_tif(tmp_path / "dsm_a.tif", dsm_a)
    _write_tif(tmp_path / "dtm_a.tif", np.full((20, 20), 100.0))
    dsm_b = np.full((20, 20), 205.0)
    dsm_b[:10, 10:] = 199.5  # DSM below DTM (north-east quarter): must clamp to 0
    _write_tif(tmp_path / "dsm_b.tif", dsm_b)
    _write_tif(tmp_path / "dtm_b.tif", np.full((20, 20), 200.0))
    srcs = {n: CogSource(_info(n, "DSM" if n.startswith("dsm") else "DTM"),
                         str(tmp_path / f"{n}.tif"))
            for n in ("dsm_a", "dtm_a", "dsm_b", "dtm_b")}
    lon = np.array([-79.99975, -79.99875, -79.99875])
    lat = np.array([43.49905, 43.49805, 43.49955])  # west, south-east, north-east
    c = canopy_height_grid(lon, lat, cache_dir=tmp_path / "c", sources=srcs,
                           chain=["dsm_a", "dsm_b"], pairs={"dsm_a": "dtm_a", "dsm_b": "dtm_b"})
    np.testing.assert_allclose(c.heights, [12.0, 5.0, 0.0])
    assert list(c.source) == ["dsm_a", "dsm_b", "dsm_b"]
    assert set(c.info) == {"dsm_a", "dtm_a", "dsm_b", "dtm_b"}


def test_canopy_is_nan_where_the_paired_dtm_is_missing(tmp_path):
    _write_tif(tmp_path / "dsm.tif", np.full((20, 20), 150.0))
    dtm = np.full((20, 20), 140.0)
    dtm[:, 10:] = -32767.0  # the matching DTM has a hole in the east half
    _write_tif(tmp_path / "dtm.tif", dtm)
    # A coarse DTM that covers everything must NOT be used to fill the hole.
    _write_tif(tmp_path / "coarse.tif", np.full((20, 20), 0.0))
    srcs = {
        "dsm": CogSource(_info("dsm", "DSM"), str(tmp_path / "dsm.tif")),
        "dtm": CogSource(_info("dtm", "DTM"), str(tmp_path / "dtm.tif")),
        "coarse": CogSource(_info("coarse", "DTM"), str(tmp_path / "coarse.tif")),
    }
    lon, lat = np.array([-79.99975, -79.99875]), np.array([43.49905, 43.49905])
    c = canopy_height_grid(lon, lat, cache_dir=tmp_path / "c", sources=srcs, chain=["dsm"],
                           pairs={"dsm": "dtm"})
    assert c.heights[0] == pytest.approx(10.0)
    assert np.isnan(c.heights[1]) and c.source[1] == "" and np.isnan(c.resolution_m[1])


def test_canopy_errors(tmp_path):
    _write_tif(tmp_path / "dsm.tif", np.full((20, 20), 150.0))
    srcs = {"dsm": CogSource(_info("dsm", "DSM"), str(tmp_path / "dsm.tif"))}
    with pytest.raises(ValueError, match="no paired DTM"):
        canopy_height_grid([-79.99975], [43.49905], cache_dir=tmp_path, sources=srcs,
                           chain=["dsm"], pairs={})
    with pytest.raises(ValueError, match="grid shapes differ"):
        canopy_height_grid(np.zeros((2, 2)), np.zeros(4), cache_dir=tmp_path, sources=srcs)


# ---------------------------------------------------------------------------------------------
# Live services (pytest -m network)
# ---------------------------------------------------------------------------------------------


@pytest.mark.network
def test_live_dsm_and_canopy_beside_snake_road(tmp_path):
    lon = np.array([v[0] for v in REF.values()])
    lat = np.array([v[1] for v in REF.values()])
    dsm = sample_surface_elevation(lon, lat, cache_dir=tmp_path)
    assert list(dsm.source) == ["hrdem-1m-dsm"] * 3
    np.testing.assert_allclose(dsm.heights, [v[2] for v in REF.values()], atol=0.01)
    c = canopy_height_grid(lon, lat, cache_dir=tmp_path)
    np.testing.assert_allclose(c.heights, [v[4] for v in REF.values()], atol=0.01)
