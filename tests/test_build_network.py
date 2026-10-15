"""Live end-to-end build: Overpass + the real elevation chain (NRCan HRDEM in Ontario), with
the live scenery sources (HRDEM lidar DSM trees, NRCan land cover, OSM/Microsoft buildings,
OSM barriers).

Skipped unless run with `-m network` or DRIFTMAPS_NETWORK=1. Uses the normal download cache
(`DRIFTMAPS_CACHE_DIR` or .cache/), so a second run is offline-fast.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from driftmaps.ac.ai_line import read_ai
from driftmaps.ac.kn5 import read_kn5, validate
from driftmaps.build import BuildOptions, build_track
from driftmaps.geo.router import route_from_pins
from driftmaps.geometry.terrain import TerrainOptions

FORKS_PINS = [(43.80410, -79.99231), (43.81060, -79.98185)]


def _network_enabled(config) -> bool:
    return os.environ.get("DRIFTMAPS_NETWORK") == "1" or "network" in (config.option.markexpr or "")


@pytest.mark.network
def test_live_build_ontario(request, tmp_path):
    if not _network_enabled(request.config):
        pytest.skip("live network test: run with -m network or DRIFTMAPS_NETWORK=1")
    route = route_from_pins(FORKS_PINS, buffer_m=400)
    options = BuildOptions(
        name="Forks Live",
        track_id="dm_forks_live",
        terrain=TerrainOptions(rings=((15.0, 3.0), (60.0, 8.0), (150.0, 20.0)), near_m=60.0),
        debug_png=False,
    )
    result = build_track(route, tmp_path, options)
    assert result.stats["elevation_source"].startswith("hrdem")  # Canadian lidar, not GLO-30
    kn5 = read_kn5(result.track_dir / "dm_forks_live.kn5")
    assert validate(kn5) == []
    ai = read_ai(result.track_dir / "ai" / "fast_lane.ai")
    assert ai.n_points > 1000
    # The Credit River valley: the road drops and climbs by tens of metres, gently.
    z = result.geometry.model.centre[:, 1]
    assert np.ptp(z) > 5.0 and result.stats["max_grade"] < 0.2
    # Scenery from the live sources: lidar trees (forested valley), the river bridge with its
    # structure and no terrain in the deck; credits for the lidar and the land cover.
    sc = result.stats["scenery"]
    assert sc["trees"]["total"] > 1000
    # (< 5 cm, the build's warning threshold: with this test's coarse 3 m terrain grid the
    # triangle from the last road-edge point before the deck can graze its seat by a hair.)
    assert sc["bridges"] and sc["deck_intrusion_m"] < 0.05
    names = {n.name for n in kn5.root.children}
    assert any(n.startswith("trees_near_") for n in names)
    assert "bridge00_concrete" in names
    lic = (result.track_dir / "LICENSE-DATA.txt").read_text(encoding="utf-8")
    assert "lidar DSM" in lic and "Land Cover of Canada 2020" in lic
