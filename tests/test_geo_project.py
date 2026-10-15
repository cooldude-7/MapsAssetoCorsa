"""Tests for driftmaps.geo.project: the local transverse Mercator and its AC mapping."""

from __future__ import annotations

import numpy as np
import pytest

from driftmaps.geo.project import WGS84_A, WGS84_F, LocalProjection

E2 = WGS84_F * (2.0 - WGS84_F)


def meridian_arc(lat_deg: float) -> float:
    """Equator -> lat along the WGS84 meridian, by numerical integration of M(phi)."""
    phi = np.linspace(0.0, np.radians(lat_deg), 200_001)
    m = WGS84_A * (1.0 - E2) / (1.0 - E2 * np.sin(phi) ** 2) ** 1.5
    return float(np.sum((m[1:] + m[:-1]) * 0.5 * np.diff(phi)))


def test_origin_maps_to_zero():
    p = LocalProjection(43.8106, -79.98185)
    e, n = p.to_enu(-79.98185, 43.8106)
    assert abs(e) < 1e-9 and abs(n) < 1e-9


def test_quarter_meridian():
    p = LocalProjection(0.0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        e, n = p.to_enu(0.0, 90.0)
    # WGS84 quarter meridian: 10 001 965.729 m.
    assert n == pytest.approx(10_001_965.729, abs=1e-3)
    assert e == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("lat", [10.0, 43.81, 45.0, 60.0])
def test_meridian_arc_matches_integration(lat):
    p = LocalProjection(0.0, 7.0)
    _, n = p.to_enu(7.0, lat)
    assert n == pytest.approx(meridian_arc(lat), abs=1e-3)


def test_short_distances_along_parallel_and_meridian():
    lat0 = 43.81
    p = LocalProjection(lat0, -79.98)
    phi = np.radians(lat0)
    w = np.sqrt(1.0 - E2 * np.sin(phi) ** 2)
    prime_vertical = WGS84_A / w  # N(phi)
    meridional = WGS84_A * (1.0 - E2) / w**3  # M(phi)
    d = 0.01  # degrees, ~800 m / ~1.1 km
    e, n = p.to_enu(-79.98 + d, lat0)
    assert e == pytest.approx(prime_vertical * np.cos(phi) * np.radians(d), rel=1e-6)
    assert abs(n) < 0.1  # parallels curve away from the grid only slightly near the origin
    e, n = p.to_enu(-79.98, lat0 + d)
    assert abs(e) < 1e-9
    assert n == pytest.approx(meridional * np.radians(d), rel=1e-6)


def test_round_trip_vectorised():
    rng = np.random.default_rng(0)
    for lat0, lon0 in [(43.81, -79.98), (36.48, 138.85), (-33.9, 151.2), (43.14, 5.85)]:
        p = LocalProjection(lat0, lon0)
        lon = lon0 + rng.uniform(-0.5, 0.5, (40, 3))
        lat = lat0 + rng.uniform(-0.5, 0.5, (40, 3))
        e, n = p.to_enu(lon, lat)
        assert e.shape == (40, 3)
        lon2, lat2 = p.to_lonlat(e, n)
        np.testing.assert_allclose(lon2, lon, atol=1e-11)
        np.testing.assert_allclose(lat2, lat, atol=1e-11)
        ll = np.stack([lon, lat], axis=-1)
        np.testing.assert_allclose(p.from_xy(p.to_xy(ll)), ll, atol=1e-11)


def test_antimeridian_wraps():
    p = LocalProjection(-17.0, 179.99)
    e, _ = p.to_enu(-179.99, -17.0)
    assert 2000.0 < e < 2200.0  # 0.02 deg east, not 360 deg west
    lon, _ = p.to_lonlat(e, 0.0)
    assert lon == pytest.approx(-179.99, abs=1e-9)


def test_lonlat_to_ac_axes():
    """North of the origin is -Z in AC, east is +X, height is +Y."""
    p = LocalProjection(43.81, -79.98)
    north = p.lonlat_to_ac(-79.98, 43.82, 12.0)
    assert north[0] == pytest.approx(0.0, abs=1e-6)
    assert north[1] == 12.0
    assert north[2] < -1000.0
    east = p.lonlat_to_ac([-79.97, -79.96], [43.81, 43.81])
    assert east.shape == (2, 3)
    assert np.all(east[:, 0] > 700.0)
    np.testing.assert_allclose(east[:, 1], 0.0)


def test_rejects_bad_origin():
    with pytest.raises(ValueError):
        LocalProjection(91.0, 0.0)
