"""Tests for driftmaps.geo.overpass: query building, parsing, retries and the disk cache.

Everything here is offline except `test_live_fetch`, which needs `-m network` (or
DRIFTMAPS_NETWORK=1) and talks to the real Overpass mirrors.
"""

from __future__ import annotations

import gzip
import json
import os
import threading
import time
import urllib.error
from pathlib import Path

import pytest

from driftmaps.geo.overpass import (
    OverpassError,
    build_query,
    fetch_corridor,
    load_cached,
    parse_overpass,
    post_query,
    query_cache_key,
)

FIXTURE = Path(__file__).parent / "fixtures" / "osm" / "forks_of_the_credit.json"
FORKS_PINS = [(43.80410, -79.99231), (43.81060, -79.98185), (43.83180, -79.95940)]
FORKS_BUFFER_M = 400.0


def test_build_query_corridor():
    q = build_query(FORKS_PINS, 400)
    assert q.startswith("[out:json][timeout:120];")
    assert "(around:400,43.8041,-79.99231,43.8106,-79.98185,43.8318,-79.9594)" in q
    assert "motorway|trunk|primary|secondary|tertiary)(_link)?" in q
    assert 'way["highway"="track"]["surface"~' in q
    assert '["service"!~"^(parking_aisle|driveway' in q
    assert q.rstrip().endswith("out body geom;")
    assert 'way["highway"="track"]' not in build_query(FORKS_PINS, include_tracks="none")
    assert 'way["highway"="track"]["area"' in build_query(FORKS_PINS, include_tracks="all")
    # The cache key is a pure function of the query text.
    assert query_cache_key(q) == query_cache_key(build_query(FORKS_PINS, 400.0))
    assert query_cache_key(q) != query_cache_key(build_query(FORKS_PINS, 401))
    with pytest.raises(ValueError):
        build_query([])


def test_parse_fixture():
    osm = parse_overpass(FIXTURE.read_bytes(), "key")
    assert osm.osm_base == "2026-10-15T13:51:13Z"
    assert osm.cache_key == "key"
    assert len(osm.ways) == 37
    ids = [w.id for w in osm.ways]
    assert ids == sorted(ids)
    for w in osm.ways:
        assert len(w.nodes) == len(w.lonlat) >= 2
        assert "highway" in w.tags
    names = {w.tags.get("name") for w in osm.ways}
    assert "Forks of the Credit Road" in names
    # Shared nodes have one position.
    ll = osm.node_lonlat()
    assert len(ll) < sum(len(w.nodes) for w in osm.ways)


def test_parse_skips_bad_ways_and_rejects_errors():
    payload = {
        "osm3s": {"timestamp_osm_base": "T"},
        "elements": [
            {"type": "node", "id": 1, "lat": 0, "lon": 0},
            {"type": "way", "id": 9, "nodes": [1], "geometry": [{"lat": 0, "lon": 0}]},
            {"type": "way", "id": 8, "nodes": [1, 2], "geometry": [{"lat": 0, "lon": 0}, None]},
            {
                "type": "way",
                "id": 7,
                "nodes": [1, 2],
                "geometry": [{"lat": 0, "lon": 0}, {"lat": 0, "lon": 0.001}],
                "tags": {"highway": "service"},
            },
        ],
    }
    osm = parse_overpass(json.dumps(payload))
    assert [w.id for w in osm.ways] == [7]
    with pytest.raises(OverpassError):
        parse_overpass(b"<html>504 Gateway Time-out</html>")
    with pytest.raises(OverpassError):
        parse_overpass({"elements": [], "remark": "runtime error: Query timed out"})


class FakeOpener:
    """Replays a script of responses: bytes are returned, exceptions are raised."""

    def __init__(self, script):
        self.script = list(script)
        self.urls: list[str] = []

    def __call__(self, request, timeout):
        self.urls.append(request.full_url)
        assert request.get_method() == "POST"
        assert request.get_header("User-agent").startswith("driftmaps/")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "err", {}, None)


def test_post_query_falls_through_mirrors():
    body = FIXTURE.read_bytes()
    opener = FakeOpener([http_error(504), urllib.error.URLError("reset"), b"<html>", body])
    sleeps: list[float] = []
    out = post_query("q", ["https://a", "https://b"], opener=opener, sleep=sleeps.append)
    assert out == body
    # Each pass asks both mirrors at once (threads), so only the per-pass set is fixed.
    assert sorted(opener.urls[:2]) == sorted(opener.urls[2:]) == ["https://a", "https://b"]
    assert sleeps == [5.0]  # one backoff between the two passes


def test_post_query_races_mirrors():
    """The first good answer wins without waiting for a slow mirror."""
    body = FIXTURE.read_bytes()
    release = threading.Event()

    def opener(request, timeout):
        if request.full_url == "https://slow":
            release.wait(5)
            raise urllib.error.URLError("too late")
        return body

    t0 = time.monotonic()
    out = post_query("q", ["https://slow", "https://fast"], opener=opener, sleep=lambda s: None)
    assert out == body and time.monotonic() - t0 < 2.0
    release.set()


def test_post_query_backoff_and_failure():
    opener = FakeOpener([http_error(504)] * 3)
    sleeps: list[float] = []
    with pytest.raises(OverpassError, match="all Overpass mirrors failed"):
        post_query("q", ["https://a"], rounds=3, backoff_s=2.0, opener=opener, sleep=sleeps.append)
    assert sleeps == [2.0, 4.0]


def test_post_query_rate_limit_pause_and_bad_query():
    body = FIXTURE.read_bytes()
    sleeps: list[float] = []
    post_query("q", ["https://a"], opener=FakeOpener([http_error(429), body]), sleep=sleeps.append)
    assert sleeps == [30.0]
    with pytest.raises(OverpassError, match="rejected"):
        post_query("q", ["https://a"], opener=FakeOpener([http_error(400)]), sleep=sleeps.append)


def test_fetch_corridor_caches(tmp_path):
    body = FIXTURE.read_bytes()
    opener = FakeOpener([body])
    osm = fetch_corridor(
        FORKS_PINS, tmp_path, buffer_m=FORKS_BUFFER_M, mirrors=["https://a"], opener=opener
    )
    key = query_cache_key(build_query(FORKS_PINS, FORKS_BUFFER_M))
    assert osm.cache_key == key
    assert (tmp_path / "overpass" / f"{key}.json.gz").is_file()
    # Second call: served from the cache, no request (the opener script is empty now).
    again = fetch_corridor(
        FORKS_PINS, tmp_path, buffer_m=FORKS_BUFFER_M, mirrors=["https://a"], opener=opener
    )
    assert [w.id for w in again.ways] == [w.id for w in osm.ways]
    assert load_cached(tmp_path, key).osm_base == osm.osm_base
    with pytest.raises(OverpassError, match="no cached"):
        fetch_corridor(FORKS_PINS, tmp_path, buffer_m=999, offline=True)


def test_corrupt_cache_entry_is_ignored(tmp_path):
    key = "abc"
    path = tmp_path / "overpass" / f"{key}.json.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(gzip.compress(b"not json"))
    assert load_cached(tmp_path, key) is None


def _network_enabled(config) -> bool:
    return os.environ.get("DRIFTMAPS_NETWORK") == "1" or "network" in (config.option.markexpr or "")


@pytest.mark.network
def test_live_fetch(request, tmp_path):
    """Live Overpass: a small corridor on Forks of the Credit Road (mirrors + retries)."""
    if not _network_enabled(request.config):
        pytest.skip("live network test: run with -m network or DRIFTMAPS_NETWORK=1")
    pins = [(43.8106, -79.98185), (43.8125, -79.9805)]
    osm = fetch_corridor(pins, tmp_path, buffer_m=150, rounds=4, backoff_s=10.0, timeout_s=120)
    assert osm.osm_base
    assert any(w.tags.get("name") == "Forks of the Credit Road" for w in osm.ways)
