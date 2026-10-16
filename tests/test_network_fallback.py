"""The road network is never silently skipped (`build._fetch_network`).

Overpass is forced to fail. The network then comes from the app's z14 road tiles (the tile
store's cache, else its own download); only when those fail too is the network missing, and
that is an alert the app shows with the build result.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from driftmaps import build as build_mod
from driftmaps.app import tiles as tiles_mod
from driftmaps.app.jobs import BuildRequest, BuildRunner
from driftmaps.build import NETWORK_MISSING_WARNING, BuildOptions, _fetch_network
from driftmaps.geo.overpass import OverpassError, parse_overpass
from driftmaps.geo.router import route_from_pins

FIXTURES = Path(__file__).parent / "fixtures"
FORKS = FIXTURES / "osm" / "forks_of_the_credit.json"
FORKS_PINS = [(43.80410, -79.99231), (43.81060, -79.98185)]


@pytest.fixture(scope="module")
def route():
    return route_from_pins(FORKS_PINS, osm=parse_overpass(FORKS.read_bytes()))


@pytest.fixture
def no_overpass(monkeypatch):
    def fail(*args, **kwargs):
        raise OverpassError("all Overpass mirrors failed")

    monkeypatch.setattr(build_mod, "fetch_network", fail)


def _fetch(route, options):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        net = _fetch_network(route, options)
    return net, [str(w.message) for w in caught]


def test_tiles_stand_in_for_a_failed_overpass(route, tmp_path, monkeypatch, no_overpass):
    queries: list[str] = []

    def post(query: str, **kwargs) -> bytes:
        queries.append(query)
        return FORKS.read_bytes()

    monkeypatch.setattr(tiles_mod, "post_query", post)
    options = BuildOptions(cache_dir=tmp_path)
    net, notes = _fetch(route, options)
    assert net is not None and net.ways and not net.nodes_known  # rules decide the nodes
    assert queries and all("way[" in q for q in queries)
    assert any("from the map tiles" in n for n in notes)
    assert not any(n.startswith(NETWORK_MISSING_WARNING[:40]) for n in notes)
    # Every road reaches within the network radius (+ the tile margin) of the route.
    route_ways = set(route.ways)
    assert route_ways <= {w.id for w in net.ways}
    # Cached now: the next build reads the tiles without asking again, also offline.
    n = len(queries)
    again, _ = _fetch(route, BuildOptions(cache_dir=tmp_path, offline=True))
    assert len(queries) == n and {w.id for w in again.ways} == {w.id for w in net.ways}


def test_no_network_at_all_is_an_alert(route, tmp_path, monkeypatch, no_overpass):
    def fail(query: str, **kwargs) -> bytes:
        raise OverpassError("all Overpass mirrors failed")

    monkeypatch.setattr(tiles_mod, "post_query", fail)
    net, notes = _fetch(route, BuildOptions(cache_dir=tmp_path))
    assert net is None
    missing = [n for n in notes if n.startswith(NETWORK_MISSING_WARNING[:40])]
    assert len(missing) == 1 and "mirrors failed" in missing[0]


def test_offline_without_cached_tiles_is_an_alert(route, tmp_path, monkeypatch, no_overpass):
    def never(query: str, **kwargs) -> bytes:  # offline: the tile store must not download
        raise AssertionError("downloaded while offline")

    monkeypatch.setattr(tiles_mod, "post_query", never)
    net, notes = _fetch(route, BuildOptions(cache_dir=tmp_path, offline=True))
    assert net is None and any("not cached" in n for n in notes)


def test_alerts_reach_the_app_result(route, tmp_path):
    req = BuildRequest(route_id="0" * 16, route=route, name="T", track_id="dm_t")
    result = SimpleNamespace(
        track_dir=tmp_path, track_id="dm_t", stats={"kn5_bytes": 1}, alerts=["No side roads"]
    )
    out = BuildRunner._result(req, result, None)
    assert out["warnings"] == ["No side roads"]
    del result.alerts  # an older build function: no alerts
    assert BuildRunner._result(req, result, None)["warnings"] == []
