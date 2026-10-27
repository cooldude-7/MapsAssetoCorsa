"""Shared test setup.

Every test runs without a downloaded offline region database (`driftmaps.geo.offline`): the
lookup folder is an empty temporary one, so a developer's real `<app data>/offline/ontario.sqlite`
never changes what the Overpass-path tests see. Tests that want a database build one and point
`DRIFTMAPS_OFFLINE_DIR` (or `offline.configure`) at it.

No test uses a Mapillary token unless it brings its own: the app's built-in one
(`geo.mapillary.DEFAULT_CLIENT_TOKEN`) is blanked and `$DRIFTMAPS_MAPILLARY_TOKEN` unset.
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


@pytest.fixture(autouse=True)
def _no_mapillary_token(monkeypatch: pytest.MonkeyPatch):
    from driftmaps.geo import mapillary

    monkeypatch.setattr(mapillary, "DEFAULT_CLIENT_TOKEN", "")
    monkeypatch.delenv(mapillary.ENV_TOKEN, raising=False)
    mapillary.reset_builtin_state()
    yield
    mapillary.reset_builtin_state()
