"""The local app's HTTP API, end to end against a real server on a random port (offline).

`route_from_pins` returns the Forks of the Credit fixture route (routed on the saved OSM
fixture), `build_track` is a fake that writes a minimal marked track folder, and elevation
sampling is a fake. Assetto Corsa is a temporary folder with acs.exe and content/tracks.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from driftmaps.ac.scene import MARKER_FILE, MARKER_TEXT
from driftmaps.app import server as server_mod
from driftmaps.app.jobs import NETWORK_MESSAGE, NO_AC_MESSAGE
from driftmaps.app.server import (
    COARSE_WARNING,
    MAP_SERVERS_MESSAGE,
    STALE_MAP_WARNING,
    App,
    AppServer,
    display_line,
    read_static,
    route_error_message,
    static_root,
)
from driftmaps.app.settings import APP_NAME, Settings
from driftmaps.elevation import sources as elevation_sources
from driftmaps.elevation.sources import ElevationSample, SourceInfo
from driftmaps.geo import router
from driftmaps.geo.overpass import OverpassError, parse_overpass
from driftmaps.geo.router import Route, RouteError

FIXTURES = Path(__file__).parent / "fixtures"
PIN_A = [43.80410, -79.99231]
PIN_B = [43.81060, -79.98185]
REAL_ROUTE_FROM_PINS = router.route_from_pins  # tests replace router.route_from_pins


@pytest.fixture(scope="module")
def forks_route() -> Route:
    osm = parse_overpass((FIXTURES / "osm" / "forks_of_the_credit.json").read_bytes())
    return router.route_from_pins([tuple(PIN_A), tuple(PIN_B)], osm=osm)


def make_ac(root: Path) -> Path:
    (root / "content" / "tracks").mkdir(parents=True)
    (root / "acs.exe").write_bytes(b"")
    return root


def write_track(folder: Path, name: str, ours: bool = True, length: str = "1363") -> Path:
    (folder / "ui").mkdir(parents=True, exist_ok=True)
    if ours:
        (folder / MARKER_FILE).write_text(MARKER_TEXT, encoding="utf-8")
    tags = ["A2B", "drift-maps"] if ours else ["circuit"]
    ui = {"name": name, "tags": tags, "length": length}
    (folder / "ui" / "ui_track.json").write_text(json.dumps(ui), encoding="utf-8")
    (folder / f"{folder.name}.kn5").write_bytes(b"x" * 2_500_000)
    return folder


class FakeBuild:
    """Stands in for build_track: reports progress, writes a marked folder, can block/fail."""

    def __init__(self) -> None:
        self.calls: list[tuple[Route, Path, object]] = []
        self.gate: threading.Event | None = None
        self.error: Exception | None = None
        self.running = 0
        self.max_running = 0
        self._lock = threading.Lock()

    def __call__(self, route, out_dir, options):
        with self._lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        try:
            self.calls.append((route, Path(out_dir), options))
            options.progress("Getting elevation and shaping the road", 0.0)
            if self.gate is not None:
                assert self.gate.wait(10)
            if self.error is not None:
                raise self.error
            options.progress("Planting trees", 0.7)
            track_dir = write_track(Path(out_dir) / options.track_id, options.name)
            options.progress("Track built", 1.0)
            stats = {
                "length_m": route.length_m,
                "kn5_bytes": 2_500_000,
                "scenery": {"trees": {"total": 1234}, "buildings": {"buildings": 56}},
            }
            return SimpleNamespace(
                track_dir=track_dir,
                track_id=options.track_id,
                stats=stats,
                warnings=[],
                summary=lambda: f"track    {track_dir}\nKN5      2.5 MB",
            )
        finally:
            with self._lock:
                self.running -= 1


class Client:
    def __init__(self, srv: AppServer) -> None:
        self.srv = srv

    def call(self, method: str, path: str, body=None, headers=None, raw: bytes | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.srv.port, timeout=20)
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        conn.request(method, path, body=data, headers=h)
        resp = conn.getresponse()
        payload = resp.read()
        conn.close()
        ctype = resp.getheader("Content-Type") or ""
        out = json.loads(payload) if ctype.startswith("application/json") else payload
        return resp.status, out, resp

    def get(self, path, **kw):
        return self.call("GET", path, **kw)[:2]

    def post(self, path, body=None, **kw):
        return self.call("POST", path, body, **kw)[:2]

    def wait_job(self, job_id: str, states=("done", "error"), timeout: float = 15.0) -> dict:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            status, job = self.get(f"/api/jobs/{job_id}")
            assert status == 200
            if job["state"] in states:
                return job
            time.sleep(0.02)
        raise AssertionError(f"job {job_id} still {job['state']}")


@pytest.fixture
def env(tmp_path, monkeypatch, forks_route):
    """A running server with a fake AC, fake routing/build/elevation."""
    monkeypatch.delenv("DRIFTMAPS_CACHE_DIR", raising=False)
    ac = make_ac(tmp_path / "assettocorsa")
    route_calls: list[dict] = []

    def fake_route(pins, cache_dir=None, **kw):
        route_calls.append({"pins": pins, "cache_dir": cache_dir, **kw})
        return forks_route

    monkeypatch.setattr(router, "route_from_pins", fake_route)
    build = FakeBuild()
    settings = Settings(tmp_path / "appdata", discover=lambda: ac)
    srv = AppServer(App(settings, build_fn=build), port=0).start()
    ns = SimpleNamespace(
        client=Client(srv),
        srv=srv,
        ac=ac,
        tracks=ac / "content" / "tracks",
        out=settings.output_dir,
        settings=settings,
        build=build,
        route_calls=route_calls,
        tmp=tmp_path,
    )
    yield ns
    if build.gate is not None:
        build.gate.set()
    srv.stop()


def make_route(c: Client, reverse: bool = False) -> dict:
    status, body = c.post("/api/route", {"pins": [PIN_A, PIN_B], "reverse": reverse})
    assert status == 200, body
    return body


# ---------------------------------------------------------------------------------------------
# status / settings
# ---------------------------------------------------------------------------------------------


def test_status(env):
    status, body = env.client.get("/api/status")
    assert status == 200
    assert body["app"] == APP_NAME == "Assetto Corsa Map Generator"
    assert body["version"]
    assert body["ac_found"] is True
    assert Path(body["ac_root"]) == env.ac
    assert Path(body["tracks_dir"]) == env.tracks


def test_status_without_ac(tmp_path):
    srv = AppServer(App(Settings(tmp_path / "d", discover=lambda: None)), port=0).start()
    try:
        status, body = Client(srv).get("/api/status")
    finally:
        srv.stop()
    assert status == 200
    assert body["ac_found"] is False and body["ac_root"] is None and body["tracks_dir"] is None


def test_settings_validates_and_persists(env):
    other = make_ac(env.tmp / "D" / "SteamLibrary" / "assettocorsa")
    # A folder inside AC (content/tracks) is taken as its root.
    status, body = env.client.post("/api/settings", {"ac_root": str(other / "content" / "tracks")})
    assert status == 200, body
    assert Path(body["ac_root"]) == other and body["ac_found"] is True
    saved = json.loads(env.settings.config_path.read_text(encoding="utf-8"))
    assert Path(saved["ac_root"]) == other
    # A new app reading the same data folder uses it.
    assert Settings(env.settings.data_dir, discover=lambda: None).ac_root() == other

    status, body = env.client.post("/api/settings", {"ac_root": str(env.tmp)})
    assert status == 400 and "acs.exe" in body["error"]
    status, body = env.client.post("/api/settings", {"ac_root": str(env.tmp / "nope")})
    assert status == 400 and "doesn't exist" in body["error"]
    status, body = env.client.post("/api/settings", {"ac_root": 5})
    assert status == 400

    # Empty forgets the saved folder and searches again.
    status, body = env.client.post("/api/settings", {"ac_root": ""})
    assert status == 200 and Path(body["ac_root"]) == env.ac


# ---------------------------------------------------------------------------------------------
# route / profile
# ---------------------------------------------------------------------------------------------


def test_route(env, forks_route):
    body = make_route(env.client)
    assert set(body) == {"route_id", "length_m", "line", "snapped", "names", "warnings", "loop"}
    assert body["loop"] is False
    assert len(body["route_id"]) == 16
    assert body["length_m"] == pytest.approx(forks_route.length_m, abs=0.1)
    line = np.array(body["line"])
    assert 2 <= len(line) <= 2000 and line.shape[1] == 2
    # [lon, lat], starting at A and ending at B
    assert line[0] == pytest.approx([PIN_A[1], PIN_A[0]], abs=1e-3)
    assert line[-1] == pytest.approx([PIN_B[1], PIN_B[0]], abs=1e-3)
    assert body["snapped"][0] == pytest.approx(PIN_A, abs=1e-3)
    assert body["snapped"][1] == pytest.approx(PIN_B, abs=1e-3)
    assert body["names"] == ["Forks of the Credit Road"]
    assert body["warnings"] == []
    call = env.route_calls[0]
    assert call["pins"] == [tuple(PIN_A), tuple(PIN_B)]
    assert "loop" not in call  # A->B routing is called exactly as before
    assert Path(call["cache_dir"]) == env.settings.data_dir / "cache"
    # The route is kept on disk for builds after a restart.
    saved = Route.load(env.settings.routes_dir / f"{body['route_id']}.json")
    assert saved.length_m == pytest.approx(forks_route.length_m)


def test_route_reverse(env):
    fwd = make_route(env.client)
    rev = make_route(env.client, reverse=True)
    assert rev["route_id"] != fwd["route_id"]
    assert rev["line"][0] == pytest.approx(fwd["line"][-1], abs=1e-6)
    assert rev["line"][-1] == pytest.approx(fwd["line"][0], abs=1e-6)
    assert rev["snapped"] == fwd["snapped"]  # request order, not driving order
    assert len(env.route_calls) == 1  # same pins: routed once
    stored = env.srv.app.routes.get(rev["route_id"])
    assert stored.pins[0].snap_lat == pytest.approx(PIN_B[0], abs=1e-3)


def test_route_bad_pins(env):
    for pins in (None, [PIN_A], [PIN_A, "x"], [PIN_A, [95.0, 0.0]], [PIN_A, [True, 1.0]]):
        status, body = env.client.post("/api/route", {"pins": pins})
        assert status == 400 and body["error"], pins
    status, body = env.client.post("/api/route", {"pins": [[43.0, -80.0], [45.0, -75.0]]})
    assert status == 400 and "too far" in body["error"]
    assert env.route_calls == []


def test_route_errors_are_plain(env, monkeypatch):
    def far(pins, cache_dir=None, **kw):
        lat, lon = pins[1]
        raise RouteError(f"pin ({lat:.6f}, {lon:.6f}) is 412 m from the nearest road (max 150 m)")

    monkeypatch.setattr(router, "route_from_pins", far)
    status, body = env.client.post("/api/route", {"pins": [PIN_A, PIN_B]})
    assert status == 400
    assert body["error"] == "No road found near point B - click closer to a road."

    def no_net(pins, cache_dir=None, offline=False, **kw):
        raise OverpassError("all Overpass mirrors failed")

    monkeypatch.setattr(router, "route_from_pins", no_net)
    status, body = env.client.post("/api/route", {"pins": [PIN_A, [43.81, -79.98]]})
    assert status == 502 and body["error"] == MAP_SERVERS_MESSAGE

    def boom(pins, cache_dir=None, **kw):
        raise KeyError("bad data")

    monkeypatch.setattr(router, "route_from_pins", boom)
    status, body = env.client.post("/api/route", {"pins": [PIN_A, [43.812, -79.98]]})
    assert status == 500 and "finding the route" in body["error"]


def test_route_uses_saved_map_data_when_offline(env, monkeypatch, forks_route):
    calls = []

    def flaky(pins, cache_dir=None, offline=False, **kw):
        calls.append(offline)
        if not offline:
            raise OverpassError("all Overpass mirrors failed")
        return forks_route

    monkeypatch.setattr(router, "route_from_pins", flaky)
    body = make_route(env.client)
    assert calls == [False, True]
    assert STALE_MAP_WARNING in body["warnings"]


def test_route_warnings(env, monkeypatch, forks_route):
    from dataclasses import replace

    long_route = replace(forks_route, length_m=31_400.0)
    monkeypatch.setattr(router, "route_from_pins", lambda pins, cache_dir=None, **kw: long_route)
    body = make_route(env.client)
    assert any("31.4 km" in w for w in body["warnings"])
    assert COARSE_WARNING not in body["warnings"]  # Ontario has lidar sources

    # Somewhere with only the 30 m world DEM (Japan): shift every node there.
    nodes = {k: (lon + 218.9, lat - 7.4) for k, (lon, lat) in forks_route.nodes.items()}
    origin = (forks_route.origin[0] - 7.4, forks_route.origin[1] + 218.9)
    japan = replace(forks_route, nodes=nodes, origin=origin)
    monkeypatch.setattr(router, "route_from_pins", lambda pins, cache_dir=None, **kw: japan)
    status, body = env.client.post("/api/route", {"pins": [[36.4, 138.9], [36.41, 138.91]]})
    assert status == 200
    assert COARSE_WARNING in body["warnings"]


def test_display_line_is_simplified(forks_route):
    full = display_line(forks_route)
    small = display_line(forks_route, max_points=8)
    assert len(small) <= 8 < len(full)
    assert small[0] == full[0] and small[-1] == full[-1]


def test_route_error_message_labels_waypoints():
    pins = [(43.0, -80.0), (43.1, -80.1), (43.2, -80.2)]
    exc = RouteError("pin (43.100000, -80.100000) is 300 m from the nearest road (max 150 m)")
    assert route_error_message(exc, pins) == (
        "No road found near point B - click closer to a road."
    )
    assert "U-turns" in route_error_message(RouteError("no route joins the pins in order"), pins)


def test_profile(env, monkeypatch, forks_route):
    calls = []

    def fake_sample(lon, lat, cache_dir=None, **kw):
        calls.append((np.asarray(lon), np.asarray(lat), cache_dir))
        n = len(lon)
        z = 300.0 + np.arange(n, dtype=float)
        z[0] = np.nan
        info = SourceInfo("hrdem-1m", "NRCan HRDEM 1 m", 1.0, "DTM", "OGL", "NRCan")
        src = np.array([""] + ["hrdem-1m"] * (n - 1))
        res = np.array([np.nan] + [1.0] * (n - 1))
        return ElevationSample(z, src, res, {"hrdem-1m": info})

    monkeypatch.setattr(elevation_sources, "sample_elevation", fake_sample)
    route_id = make_route(env.client)["route_id"]
    status, body = env.client.post("/api/profile", {"route_id": route_id})
    assert status == 200, body
    s = np.array(body["s"])
    assert s[0] == 0.0 and s[-1] == pytest.approx(forks_route.length_m, abs=1.0)
    assert np.diff(s) == pytest.approx(np.full(len(s) - 1, 20.0), abs=1.0)
    assert len(body["z"]) == len(s) and body["z"][0] is None and body["z"][1] == 301.0
    assert body["source"] == "NRCan HRDEM 1 m" and body["resolution_m"] == 1.0
    lon, lat, cache_dir = calls[0]
    assert lon[0] == pytest.approx(PIN_A[1], abs=1e-3) and lat[0] == pytest.approx(
        PIN_A[0], abs=1e-3
    )
    assert Path(cache_dir) == env.settings.cache_dir
    # Cached: no second download.
    assert env.client.post("/api/profile", {"route_id": route_id}) == (200, body)
    assert len(calls) == 1

    status, body = env.client.post("/api/profile", {"route_id": "0123456789abcdef"})
    assert status == 404 and "draw it again" in body["error"]


def test_profile_without_data(env, monkeypatch):
    def nothing(lon, lat, cache_dir=None, **kw):
        n = len(lon)
        return ElevationSample(np.full(n, np.nan), np.full(n, ""), np.full(n, np.nan), {})

    monkeypatch.setattr(elevation_sources, "sample_elevation", nothing)
    route_id = make_route(env.client)["route_id"]
    status, body = env.client.post("/api/profile", {"route_id": route_id})
    assert status == 502 and "internet" in body["error"]


# ---------------------------------------------------------------------------------------------
# build / jobs / tracks
# ---------------------------------------------------------------------------------------------


def test_build_installs_and_lists(env, forks_route):
    route_id = make_route(env.client)["route_id"]
    status, body = env.client.post(
        "/api/build",
        {
            "route_id": route_id,
            "name": "Forks of the Credit",
            "track_id": None,
            "options": {"trees": False, "buildings": True, "guardrails": False,
                        "scenery_density": 5},
        },
    )  # fmt: skip
    assert status == 200, body
    job = env.client.wait_job(body["job_id"])
    assert job["state"] == "done", job
    assert job["progress"] == 1.0 and job["error"] is None
    tid = "dm_forks_of_the_credit"
    assert job["result"] == {
        "track_id": tid,
        "name": "Forks of the Credit",
        "track_dir": str(env.out / tid),
        "installed_to": str(env.tracks / tid),
        "length_m": pytest.approx(forks_route.length_m, abs=0.1),
        "kn5_mb": 2.5,
        "trees": 1234,
        "buildings": 56,
        "loop": False,
    }
    assert "Planting trees" in job["log"] and any("KN5" in line for line in job["log"])
    assert (env.tracks / tid / MARKER_FILE).is_file()
    _, _, options = env.build.calls[0]
    assert options.track_id == tid and options.name == "Forks of the Credit"
    assert options.trees is False and options.buildings is True and options.guardrails is False
    assert options.scenery_density == 2.0  # clamped
    assert options.cache_dir == env.settings.cache_dir and options.debug_png is False

    status, tracks = env.client.get("/api/tracks")
    assert status == 200
    assert [(t["track_id"], t["name"], t["installed"]) for t in tracks] == [
        (tid, "Forks of the Credit", True)
    ]
    assert tracks[0]["length_m"] == 1363.0 and tracks[0]["built_at"]

    # Building the same id again replaces our own folders.
    status, body = env.client.post(
        "/api/build", {"route_id": route_id, "name": "Forks of the Credit"}
    )
    assert status == 200
    assert env.client.wait_job(body["job_id"])["state"] == "done"


def test_build_without_install_and_custom_id(env):
    route_id = make_route(env.client)["route_id"]
    status, body = env.client.post(
        "/api/build",
        {"route_id": route_id, "name": "Forks", "track_id": "My Forks", "install": False},
    )
    assert status == 200
    job = env.client.wait_job(body["job_id"])
    assert job["result"]["track_id"] == "dm_my_forks"
    assert job["result"]["installed_to"] is None
    assert not (env.tracks / "dm_my_forks").exists()
    status, tracks = env.client.get("/api/tracks")
    assert [(t["track_id"], t["installed"]) for t in tracks] == [("dm_my_forks", False)]


def test_build_rejects(env, tmp_path):
    route_id = make_route(env.client)["route_id"]
    post = env.client.post
    assert post("/api/build", {"route_id": "nope", "name": "X"})[0] == 404
    assert post("/api/build", {"route_id": route_id, "name": "  "})[0] == 400
    assert post("/api/build", {"route_id": route_id, "name": "X", "options": "all"})[0] == 400
    # Somebody else's track of that id in AC: refuse up front, touch nothing.
    write_track(env.tracks / "dm_kunos_road", "Kunos Road", ours=False)
    status, body = post("/api/build", {"route_id": route_id, "name": "Kunos Road"})
    assert status == 409 and "didn't make" in body["error"]
    assert not (env.tracks / "dm_kunos_road" / MARKER_FILE).exists()
    assert env.build.calls == []

    # No AC found: installing is refused with a plain message; building alone still works.
    srv = AppServer(App(Settings(tmp_path / "d2", discover=lambda: None)), port=0).start()
    try:
        c = Client(srv)
        rid = make_route(c)["route_id"]
        status, body = c.post("/api/build", {"route_id": rid, "name": "X"})
        assert status == 400 and body["error"] == NO_AC_MESSAGE
    finally:
        srv.stop()


def test_build_error_path(env):
    from driftmaps.elevation.sources import SourceUnavailable

    route_id = make_route(env.client)["route_id"]
    env.build.error = SourceUnavailable("HRDEM: HTTP 503")
    status, body = env.client.post("/api/build", {"route_id": route_id, "name": "Broken"})
    assert status == 200
    job = env.client.wait_job(body["job_id"])
    assert job["state"] == "error" and job["result"] is None
    assert job["error"] == NETWORK_MESSAGE
    assert job["log"][-1] == f"error: {NETWORK_MESSAGE}"
    assert not (env.tracks / "dm_broken").exists()

    env.build.error = ValueError("mesh exploded")
    _, body = env.client.post("/api/build", {"route_id": route_id, "name": "Broken"})
    job = env.client.wait_job(body["job_id"])
    assert job["error"] == "The build failed: mesh exploded"


def test_builds_run_one_at_a_time(env):
    route_id = make_route(env.client)["route_id"]
    env.build.gate = threading.Event()
    ids = []
    for name in ("One", "Two", "Three"):
        status, body = env.client.post(
            "/api/build", {"route_id": route_id, "name": name, "install": False}
        )
        assert status == 200
        ids.append(body["job_id"])
    end = time.monotonic() + 10
    while (
        env.client.get(f"/api/jobs/{ids[0]}")[1]["step"] != "Getting elevation and shaping the road"
    ):
        assert time.monotonic() < end
        time.sleep(0.02)
    first = env.client.get(f"/api/jobs/{ids[0]}")[1]
    assert first["state"] == "running" and 0.0 < first["progress"] < 0.1
    assert env.client.get(f"/api/jobs/{ids[1]}")[1]["state"] == "queued"
    assert env.client.get(f"/api/jobs/{ids[2]}")[1]["state"] == "queued"
    # A track being built can't be deleted.
    status, body = env.client.call("DELETE", "/api/tracks/dm_one")[:2]
    assert status == 409
    env.build.gate.set()
    for job_id in ids:
        assert env.client.wait_job(job_id)["state"] == "done"
    assert env.build.max_running == 1
    assert [o.name for _, _, o in env.build.calls] == ["One", "Two", "Three"]
    assert env.client.get("/api/jobs/does-not-exist")[0] == 404


def test_delete_only_our_tracks(env):
    ours = write_track(env.tracks / "dm_mine", "Mine")
    ours_out = write_track(env.out / "dm_mine", "Mine")
    theirs = write_track(env.tracks / "ks_nordschleife", "Nordschleife", ours=False)
    clash = write_track(env.tracks / "dm_clash", "Clash", ours=False)

    status, tracks = env.client.get("/api/tracks")
    assert {t["track_id"] for t in tracks} == {"dm_mine"}

    delete = lambda tid: env.client.call("DELETE", f"/api/tracks/{tid}")[:2]  # noqa: E731
    assert delete("ks_nordschleife")[0] == 403
    assert delete("dm_clash")[0] == 403
    assert theirs.is_dir() and clash.is_dir() and (theirs / "ks_nordschleife.kn5").is_file()
    assert delete("dm_nothing")[0] == 404
    assert delete("Bad-Id")[0] == 400
    assert delete("..%2F..%2Fx")[0] in (400, 404)

    assert delete("dm_mine") == (200, {"ok": True})
    assert not ours.exists() and not ours_out.exists()
    assert env.client.get("/api/tracks") == (200, [])
    assert theirs.is_dir() and clash.is_dir()


# ---------------------------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------------------------


def test_static_files(env, tmp_path, monkeypatch):
    static = tmp_path / "static"
    (static / "js").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>t</title>", encoding="utf-8")
    (static / "js" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("no", encoding="utf-8")
    monkeypatch.setattr(server_mod, "static_root", lambda: static)

    status, body, resp = env.client.call("GET", "/")
    assert status == 200 and body.startswith(b"<!doctype html>")
    assert resp.getheader("Content-Type") == "text/html; charset=utf-8"
    for path in ("/js/app.js", "/static/js/app.js"):
        status, body, resp = env.client.call("GET", path)
        assert status == 200 and body == b"console.log(1)"
        assert resp.getheader("Content-Type").startswith("text/javascript")
    assert env.client.get("/some/page")[1].startswith(b"<!doctype html>")  # SPA fallback
    assert env.client.get("/missing.png")[0] == 404
    for evil in ("/../secret.txt", "/%2e%2e/secret.txt", "/static/..%2Fsecret.txt"):
        status, body = env.client.get(evil)
        assert status == 404 and body != b"no", evil


def test_static_missing_page(env, tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "static_root", lambda: tmp_path / "nothing")
    status, body = env.client.get("/")
    assert status == 503 and "missing" in body["error"]


def test_static_root_is_in_the_package():
    root = static_root()
    assert root.name == "static"
    assert Path(str(root)).parent == Path(server_mod.__file__).parent
    assert read_static("/../server.py") is None


def test_api_errors(env):
    c = env.client
    assert c.get("/api/nope")[0] == 404
    assert c.get("/api/route")[0] == 405
    assert c.call("PUT", "/api/status")[0] == 405
    status, body = c.call("POST", "/api/route", raw=b"{not json")[:2]
    assert status == 400 and "JSON" in body["error"]
    assert c.call("POST", "/api/route", raw=b"[1, 2]")[0] == 400


def test_foreign_pages_are_refused(env):
    c = env.client
    # DNS rebinding: a page on another host name pointed at 127.0.0.1.
    assert c.call("GET", "/api/status", headers={"Host": f"evil.example:{env.srv.port}"})[0] == 403
    # Another website posting to the API from the user's browser.
    status, body = c.call(
        "POST", "/api/settings", {"ac_root": ""}, headers={"Origin": "https://evil.example"}
    )[:2]
    assert status == 403
    assert c.call("DELETE", "/api/tracks/dm_x", headers={"Origin": "null"})[0] == 403
    assert c.call("OPTIONS", "/api/build")[0] == 405
    # The app's own page is fine (either host name).
    for host in ("127.0.0.1", "localhost"):
        origin = f"http://{host}:{env.srv.port}"
        status, _ = c.call("POST", "/api/settings", {"ac_root": ""}, headers={"Origin": origin})[:2]
        assert status == 200


# ---------------------------------------------------------------------------------------------
# loops ("Back to start")
# ---------------------------------------------------------------------------------------------


def _block_loop(size: float = 1.0) -> Route:
    """A closed route round a small block (300 x 200 m at size 1) near the Forks pins."""
    from driftmaps.geo.overpass import OsmData, Way
    from driftmaps.geo.project import LocalProjection

    proj = LocalProjection(*PIN_A)
    xy = np.array([(0, 0), (300, 0), (300, 200), (0, 200), (0, 0)], float) * size
    lon, lat = proj.to_lonlat(xy[:, 0], xy[:, 1])
    way = Way(
        7,
        (1, 2, 3, 4, 1),
        np.stack([lon, lat], -1),
        {"highway": "residential", "name": "Block Road"},
    )
    osm = OsmData((way,), "2026-01-01T00:00:00Z", "block")
    a = proj.to_lonlat(150 * size, 0.0)
    b = proj.to_lonlat(300 * size, 100 * size)
    return REAL_ROUTE_FROM_PINS([(a[1], a[0]), (b[1], b[0])], osm=osm, loop=True)


def test_route_loop(env, monkeypatch, forks_route):
    loop = _block_loop()
    calls = []

    def fake(pins, cache_dir=None, **kw):
        calls.append(kw)
        return loop if kw.get("loop") else forks_route

    monkeypatch.setattr(router, "route_from_pins", fake)
    pins = [[p.lat, p.lon] for p in loop.pins[:-1]]
    status, body = env.client.post("/api/route", {"pins": pins, "loop": True})
    assert status == 200, body
    assert body["loop"] is True and calls == [{"loop": True, "max_length_m": float("inf")}]
    assert body["line"][0] == body["line"][-1]  # the line on the map closes
    assert len(body["snapped"]) == 2  # one per point sent (the closing pin isn't repeated)
    assert body["length_m"] == pytest.approx(1000.0, abs=0.5)
    stored = env.srv.app.routes.get(body["route_id"])
    assert stored.closed
    # The same points as A->B are a different route (routed separately).
    status, ab = env.client.post("/api/route", {"pins": pins})
    assert status == 200 and ab["loop"] is False and len(calls) == 2
    # Reversed: the other way round from the same start.
    status, rev = env.client.post("/api/route", {"pins": pins, "loop": True, "reverse": True})
    assert status == 200 and rev["loop"] is True and len(calls) == 2
    assert rev["line"][0] == pytest.approx(body["line"][0], abs=1e-6)
    assert rev["line"][1] == pytest.approx(body["line"][-2], abs=1e-6)

    # Build: a loop route builds as a loop; `loop` must match the route if it is sent.
    status, res = env.client.post(
        "/api/build", {"route_id": body["route_id"], "name": "Block", "loop": False}
    )
    assert status == 400 and "Back to start" in res["error"]
    status, res = env.client.post(
        "/api/build", {"route_id": ab["route_id"], "name": "Block", "loop": True}
    )
    assert status == 400 and "isn't a loop" in res["error"]
    status, res = env.client.post(
        "/api/build", {"route_id": body["route_id"], "name": "Block", "loop": True}
    )
    assert status == 200, res
    job = env.client.wait_job(res["job_id"])
    assert job["state"] == "done" and job["result"]["loop"] is True
    assert env.build.calls[-1][0].closed


def test_route_loop_refusals(env, monkeypatch):
    from driftmaps.geo.router import LOOP_MESSAGE

    def no_loop(pins, cache_dir=None, **kw):
        raise RouteError(f"no loop: {LOOP_MESSAGE}")

    monkeypatch.setattr(router, "route_from_pins", no_loop)
    status, body = env.client.post("/api/route", {"pins": [PIN_A, PIN_B], "loop": True})
    assert status == 400
    assert body["error"] == (
        "Can't make a loop here - add a point on a different road back to the start."
    )
    tiny = _block_loop(size=0.2)  # 200 m round
    monkeypatch.setattr(router, "route_from_pins", lambda pins, cache_dir=None, **kw: tiny)
    status, body = env.client.post("/api/route", {"pins": [PIN_A, PIN_B], "loop": True})
    assert status == 400 and "too short" in body["error"]


def test_tracks_say_which_are_loops(env):
    folder = write_track(env.out / "dm_block", "Block")
    ui = json.loads((folder / "ui" / "ui_track.json").read_text())
    ui["tags"] = ["circuit", "street", "drift-maps"]
    (folder / "ui" / "ui_track.json").write_text(json.dumps(ui))
    write_track(env.out / "dm_road", "Road")
    status, tracks = env.client.get("/api/tracks")
    assert status == 200
    assert {t["track_id"]: t["loop"] for t in tracks} == {"dm_block": True, "dm_road": False}


def test_pin_letters():
    from driftmaps.app.server import pin_letter

    assert [pin_letter(i) for i in (0, 1, 2, 25, 26, 27, 49)] == [
        "A", "B", "C", "Z", "AA", "AB", "AX"
    ]  # fmt: skip
