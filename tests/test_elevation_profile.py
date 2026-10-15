"""Tests for driftmaps.elevation.profile: spike rejection, hard grade/curvature limits, straight
bridges and tunnels, and leaving an already smooth road alone."""

from __future__ import annotations

import numpy as np
import pytest

from driftmaps.elevation.profile import (
    cross_road_median,
    cutoff_wavelength_m,
    road_profile,
)

S = np.arange(0.0, 3000.0, 1.0)
BASE = 120.0 + 5.0 * np.sin(2 * np.pi * S / 2000.0) + 0.02 * S  # gentle real-ish road


def grades(s, z):
    return np.diff(z) / np.diff(s)


def curvatures(s, z):
    return 2.0 * np.diff(grades(s, z)) / (s[2:] - s[:-2])


@pytest.mark.parametrize("res", [1.0, 10.0, 30.0])
def test_smooth_input_is_left_alone(res):
    z = road_profile(S, BASE, source_resolution_m=res)
    assert np.abs(z - BASE).max() < 0.02


def test_dsm_tree_spikes_are_removed():
    rng = np.random.default_rng(1)
    y = BASE + rng.normal(0.0, 0.3, S.size)
    for start in (400, 1100, 1900, 2500):  # 40 m of +15 m forest canopy
        y[start : start + 40] += 15.0
    z = road_profile(S, y, source_resolution_m=30.0)
    assert np.abs(z - BASE).max() < 1.0
    # Without the robust/asymmetric weights a plain smoother would sit metres high there.
    assert np.abs(z[400:440] - BASE[400:440]).max() < 1.0


def test_lidar_outlier_is_removed_and_lidar_is_smoothed_less():
    rng = np.random.default_rng(2)
    y = BASE + rng.normal(0.0, 0.05, S.size)
    y[1500:1503] += 8.0  # a stray return
    z = road_profile(S, y, source_resolution_m=1.0)
    assert np.abs(z - BASE).max() < 0.3
    assert cutoff_wavelength_m(1.0) < cutoff_wavelength_m(10.0) < cutoff_wavelength_m(30.0)
    assert cutoff_wavelength_m(1.0) == pytest.approx(45.0)
    assert cutoff_wavelength_m(30.0) == pytest.approx(200.0)


@pytest.mark.parametrize("max_grade, kappa", [(0.20, 1 / 150), (0.10, 1 / 400)])
def test_grade_and_vertical_curvature_limits_hold(max_grade, kappa):
    # Uneven station spacing and a 40 % ramp with sharp corners.
    rng = np.random.default_rng(3)
    s = np.cumsum(rng.uniform(0.5, 1.5, 2500))
    y = np.interp(s, [0, 900, 1000, 1100, 4000], [100, 100, 140, 140, 150])
    z = road_profile(s, y, source_resolution_m=1.0, max_grade=max_grade,
                     max_vertical_curvature=kappa)
    assert np.abs(grades(s, z)).max() <= max_grade + 1e-6
    assert np.abs(curvatures(s, z)).max() <= kappa + 1e-6
    # It still climbs the hill rather than ignoring it.
    assert z[0] < 105 and z[-1] > 140


def test_bridge_is_straight_and_ignores_the_valley():
    y = BASE.copy()
    y[1400:1600] -= 25.0  # bare-earth DTM shows the river valley under the deck
    w = np.ones_like(S)
    w[1400:1600] = 0.0
    z = road_profile(S, y, w, source_resolution_m=1.0)
    deck = z[1399:1601]
    assert np.abs(np.diff(deck, 2)).max() < 1e-6  # constant grade across the bridge
    assert np.abs(z[1400:1600] - BASE[1400:1600]).max() < 0.5  # not down in the valley
    assert abs(z[1399] - BASE[1399]) < 0.1 and abs(z[1600] - BASE[1600]) < 0.1


def test_tunnel_at_route_start_is_straight():
    w = np.ones_like(S)
    w[:300] = 0.0
    y = BASE.copy()
    y[:300] += 80.0  # the mountain above the tunnel
    z = road_profile(S, y, w, source_resolution_m=1.0)
    assert np.abs(np.diff(z[:301], 2)).max() < 1e-6
    # Through the mountain, not over it; a straight line can't follow the curved valley
    # exactly, but it stays close and joins the open road smoothly.
    assert (z[:300] < y[:300] - 75.0).all()
    assert np.abs(z[:300] - BASE[:300]).max() < 2.0
    assert np.abs(z[300:] - BASE[300:]).max() < 0.1


def test_nodata_stations_and_cross_road_median():
    k = np.repeat(BASE[:, None], 5, axis=1)
    k[::7, 0] = BASE[::7] + 30.0  # one wild sample out of five
    k[1000:1050, :] = np.nan  # no data at all
    k[2000:2010, 1:] = np.nan  # only one sample left
    y = cross_road_median(k)
    assert np.isnan(y[1000:1050]).all()
    np.testing.assert_allclose(y[:1000], BASE[:1000])
    z = road_profile(S, k, source_resolution_m=1.0)
    assert np.isfinite(z).all()
    assert np.abs(z - BASE).max() < 0.05


def test_bad_inputs():
    with pytest.raises(ValueError, match="strictly increasing"):
        road_profile(np.array([0.0, 1.0, 1.0]), np.zeros(3), source_resolution_m=1.0)
    with pytest.raises(ValueError, match="no usable"):
        road_profile(S[:10], np.zeros(10), np.zeros(10), source_resolution_m=1.0)
    with pytest.raises(ValueError, match="weights"):
        road_profile(S[:10], np.zeros(10), np.ones(9), source_resolution_m=1.0)
    assert road_profile(np.zeros(0), np.zeros(0), source_resolution_m=1.0).shape == (0,)
    assert road_profile(np.array([5.0]), np.array([42.0]), source_resolution_m=1.0)[0] == 42.0
