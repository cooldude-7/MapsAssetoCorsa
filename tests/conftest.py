"""Shared test setup.

Every test runs without a downloaded offline region database (`driftmaps.geo.offline`): the
lookup folder is an empty temporary one, so a developer's real `<app data>/offline/ontario.sqlite`
never changes what the Overpass-path tests see. Tests that want a database build one and point
`DRIFTMAPS_OFFLINE_DIR` (or `offline.configure`) at it.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def _empty_offline_dir(tmp_path_factory: pytest.TempPathFactory):
    return tmp_path_factory.mktemp("no-offline-db")


@pytest.fixture(autouse=True)
def _no_offline_region_db(_empty_offline_dir, monkeypatch: pytest.MonkeyPatch):
    from driftmaps.geo import offline

    monkeypatch.setenv(offline.ENV_OFFLINE_DIR, str(_empty_offline_dir))
    offline.configure(None)
    yield
    offline.configure(None)
