"""End to end: the local app ("Assetto Corsa Map Generator") as the player uses it.

Offline (always run):
- `drift-maps ui` passes its options to the app;
- the app started through the `ui` command (settings from the environment, its data folder, a
  real HTTP server on 127.0.0.1) walks the player's whole flow over HTTP: page -> status ->
  route -> elevation profile -> build + install into a fake Assetto Corsa -> My tracks ->
  delete. Routing, elevation and the build itself are fakes, so nothing is downloaded;
- the search helpers in app.js, under Node (one result per road, road-name-only fallback);
- the built track credits the app by name.

In a browser (opt-in: needs the internet for MapLibre, the map tiles and place search, plus
Playwright and a Chromium for it):

    DRIFTMAPS_E2E_BROWSER=1 PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers \\
        uv run --with playwright==1.56.0 pytest tests/test_app_e2e.py -k browser

clicks A and B on the map, waits for the route and the elevation profile, builds, checks the
install and My tracks, and deletes the track, saving screenshots in out/app_screens/. With
DRIFTMAPS_E2E_LIVE=1 as well it uses the real router and builder on Snake Road (Overpass and
elevation downloads; point DRIFTMAPS_CACHE_DIR at a warm cache to make it quick).
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from driftmaps import build as build_mod
from driftmaps import cli
from driftmaps.ac.scene import MARKER_FILE, MARKER_TEXT, is_driftmaps_track
from driftmaps.ac.track_files import GENERATOR_NAME, UiTrack
from driftmaps.app import server as server_mod
from driftmaps.app.server import App, AppServer
from driftmaps.app.settings import APP_NAME, Settings
from driftmaps.elevation import sources as elevation_sources
from driftmaps.elevation.sources import ElevationSample, SourceInfo
from driftmaps.geo import router
from driftmaps.geo.overpass import parse_overpass
from driftmaps.geo.router import Route

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
APP_JS = ROOT / "src" / "driftmaps" / "app" / "static" / "app.js"
SCREENS = ROOT / "out" / "app_screens"
FORKS_A = (43.80410, -79.99231)
FORKS_B = (43.81060, -79.98185)
NAME = "Snake Road (App Test)"
TRACK_ID = "dm_snake_road_app_test"


@pytest.fixture(scope="module")
def forks_route() -> Route:
    osm = parse_overpass((FIXTURES / "osm" / "forks_of_the_credit.json").read_bytes())
    return router.route_from_pins([FORKS_A, FORKS_B], osm=osm)


def make_ac(root: Path) -> Path:
    (root / "content" / "tracks").mkdir(parents=True)
    (root / "acs.exe").write_bytes(b"")
    return root


def fake_build(route, out_dir, options):
    """Stands in for build_track: reports progress and writes a small marked track folder."""
    options.progress("Getting elevation and shaping the road", 0.0)
    track_dir = Path(out_dir) / options.track_id
    (track_dir / "ui").mkdir(parents=True, exist_ok=True)
    (track_dir / MARKER_FILE).write_text(MARKER_TEXT, encoding="utf-8")
    ui = UiTrack(name=options.name, tags=["A2B", "drift-maps"], length_m=route.length_m)
    (track_dir / "ui" / "ui_track.json").write_text(json.dumps(ui.to_json_dict()), "utf-8")
    (track_dir / f"{options.track_id}.kn5").write_bytes(b"k" * 1_500_000)
    options.progress("Writing the track files", 0.87)
    options.progress("Track built", 1.0)
    stats = {
        "length_m": route.length_m,
        "kn5_bytes": 1_500_000,
        "scenery": {"trees": {"total": 321}, "buildings": {"buildings": 12}},
    }
    return SimpleNamespace(
        track_dir=track_dir, track_id=options.track_id, stats=stats, summary=lambda: "fake build"
    )


def fake_sample(lon, lat, cache_dir=None, **kw):
    n = len(lon)
    z = 150.0 + 40.0 * np.sin(np.linspace(0.0, 3.0, n))
    info = SourceInfo("hrdem-1m", "NRCan HRDEM mosaic 1 m (lidar DTM)", 1.0, "DTM", "OGL", "NRCan")
    return ElevationSample(z, np.array(["hrdem-1m"] * n), np.ones(n), {"hrdem-1m": info})


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(port: int, method: str, path: str, body: object = None) -> tuple[int, object]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=data, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    ctype = resp.getheader("Content-Type") or ""
    return resp.status, json.loads(raw) if ctype.startswith("application/json") else raw


# ---------------------------------------------------------------------------------------------
# The `drift-maps ui` command
# ---------------------------------------------------------------------------------------------


def test_ui_command_passes_its_options(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("driftmaps.app.run", lambda **kw: calls.append(kw))
    runner = CliRunner()
    res = runner.invoke(
        cli.app, ["ui", "--no-browser", "--port", "8799", "--data-dir", str(tmp_path)]
    )
    assert res.exit_code == 0, res.output
    assert calls == [{"port": 8799, "open_browser": False, "data_dir": tmp_path}]
    res = runner.invoke(cli.app, ["ui"])
    assert res.exit_code == 0, res.output
    assert calls[-1] == {"port": None, "open_browser": True, "data_dir": None}


def test_cli_help_names_the_app():
    res = CliRunner().invoke(cli.app, ["--help"])
    assert res.exit_code == 0
    assert "Assetto Corsa Map Generator" in res.output
    assert "ui" in res.output


def test_built_tracks_credit_the_app():
    ui = UiTrack(name="Snake Road").to_json_dict()
    assert ui["author"] == GENERATOR_NAME == APP_NAME
    assert build_mod.BuildOptions().name != "Drift Maps track"


# ---------------------------------------------------------------------------------------------
# The player's whole flow over HTTP, through `drift-maps ui`
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def running_ui(tmp_path, monkeypatch, forks_route):
    """`drift-maps ui --no-browser --port N --data-dir D` running on a thread, with fakes."""
    ac = make_ac(tmp_path / "assettocorsa")
    monkeypatch.setenv("DRIFTMAPS_AC_ROOT", str(ac))
    monkeypatch.setenv("DRIFTMAPS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("DRIFTMAPS_APP_DIR", raising=False)
    route_calls = []

    def fake_route(pins, cache_dir=None, **kw):
        route_calls.append((pins, cache_dir))
        return forks_route

    monkeypatch.setattr(router, "route_from_pins", fake_route)
    monkeypatch.setattr(build_mod, "build_track", fake_build)
    monkeypatch.setattr(elevation_sources, "sample_elevation", fake_sample)
    monkeypatch.setattr(server_mod, "_setup_logging", lambda data_dir: None)
    opened = []
    monkeypatch.setattr(server_mod.webbrowser, "open", lambda url: opened.append(url))
    servers: list[AppServer] = []

    class Recorded(AppServer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            servers.append(self)

    monkeypatch.setattr(server_mod, "AppServer", Recorded)
    port = free_port()
    data = tmp_path / "appdata"
    errors: list[BaseException] = []

    def serve():
        try:
            cli.ui(port=port, browser=False, data_dir=data)
        except BaseException as exc:  # surfaced by the test below
            errors.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    end = time.monotonic() + 15
    while not servers and not errors and time.monotonic() < end:
        time.sleep(0.02)
    assert servers and not errors, errors
    yield SimpleNamespace(
        port=port,
        ac=ac,
        tracks=ac / "content" / "tracks",
        data=data,
        route_calls=route_calls,
        opened=opened,
        cache=tmp_path / "cache",
    )
    servers[0].httpd.shutdown()
    thread.join(10)
    assert not thread.is_alive()


def test_whole_flow_through_the_ui_command(running_ui, forks_route):
    ui = running_ui
    port = ui.port

    status, page = call(port, "GET", "/")
    assert status == 200 and b"<title>Assetto Corsa Map Generator</title>" in page
    assert call(port, "GET", "/app.js")[0] == 200
    assert ui.opened == []  # --no-browser

    status, st = call(port, "GET", "/api/status")
    assert status == 200
    assert st["app"] == APP_NAME and st["ac_found"] is True
    assert Path(st["tracks_dir"]) == ui.tracks

    # 1. Pick the road
    status, route = call(port, "POST", "/api/route", {"pins": [FORKS_A, FORKS_B], "reverse": False})
    assert status == 200, route
    assert route["length_m"] == pytest.approx(forks_route.length_m, abs=0.5)
    assert 2 <= len(route["line"]) <= 2000 and route["names"]
    assert Path(ui.route_calls[0][1]) == ui.cache  # the download cache from the environment
    assert (ui.data / "routes" / f"{route['route_id']}.json").is_file()

    status, prof = call(port, "POST", "/api/profile", {"route_id": route["route_id"]})
    assert status == 200, prof
    assert len(prof["s"]) == len(prof["z"]) >= 2
    assert prof["source"].startswith("NRCan HRDEM") and prof["resolution_m"] == 1.0

    # 2. Build it (and install into Assetto Corsa)
    body = {
        "route_id": route["route_id"],
        "name": NAME,
        "track_id": None,
        "install": True,
        "options": {"trees": True, "buildings": True, "guardrails": True, "scenery_density": 1.0},
    }
    status, started = call(port, "POST", "/api/build", body)
    assert status == 200, started
    end = time.monotonic() + 20
    while True:
        status, job = call(port, "GET", f"/api/jobs/{started['job_id']}")
        assert status == 200
        if job["state"] in ("done", "error") or time.monotonic() > end:
            break
        time.sleep(0.05)
    assert job["state"] == "done", job
    result = job["result"]
    assert result["track_id"] == TRACK_ID and result["name"] == NAME
    assert Path(result["installed_to"]) == ui.tracks / TRACK_ID
    assert result["kn5_mb"] == 1.5 and result["trees"] == 321 and result["buildings"] == 12
    assert any("Installed to" in line for line in job["log"])
    installed = ui.tracks / TRACK_ID
    assert is_driftmaps_track(installed) and (installed / f"{TRACK_ID}.kn5").is_file()
    assert (ui.data / "tracks" / TRACK_ID).is_dir()  # the app keeps its own copy

    # My tracks
    status, tracks = call(port, "GET", "/api/tracks")
    assert status == 200
    assert [(t["track_id"], t["name"], t["installed"]) for t in tracks] == [(TRACK_ID, NAME, True)]
    assert tracks[0]["length_m"] == pytest.approx(round(forks_route.length_m), abs=1)

    # Delete: gone from Assetto Corsa and from the app's folder; other tracks are never touched.
    kunos = ui.tracks / "ks_nordschleife"
    (kunos / "ui").mkdir(parents=True)
    assert call(port, "DELETE", "/api/tracks/ks_nordschleife")[0] == 403
    assert call(port, "DELETE", f"/api/tracks/{TRACK_ID}") == (200, {"ok": True})
    assert not installed.exists() and not (ui.data / "tracks" / TRACK_ID).exists()
    assert kunos.is_dir()
    assert call(port, "GET", "/api/tracks") == (200, [])


# ---------------------------------------------------------------------------------------------
# app.js search helpers (Node)
# ---------------------------------------------------------------------------------------------

NODE = shutil.which("node")

SEARCH_PROBE = r"""
const road = (title, ext, c) => ({ title, subtitle: "Burlington, Ontario, Canada", center: c,
  props: { extent: ext } });
// Photon for "Snake Road Waterdown": Waterdown Road three times (three OSM ways).
const full = [
  road("Waterdown Road", [-79.8533, 43.3072, -79.8527, 43.3068], [-79.853, 43.307]),
  road("Waterdown Road", [-79.8580, 43.3108, -79.8561, 43.3093], [-79.857, 43.31]),
  road("Waterdown Road", [-79.8596, 43.3119, -79.8589, 43.3114], [-79.859, 43.311]),
];
// ...and for "Snake Road": Snake Road in two pieces, plus a far-away road.
const nameOnly = [
  road("Snake Road", [-79.8892, 43.3094, -79.8842, 43.2977], [-79.887, 43.303]),
  road("Snake Road", [-79.8937, 43.3253, -79.8889, 43.3094], [-79.891, 43.318]),
  { title: "Snake Forest Road", subtitle: "New York", center: [-79.51, 42.1], props: {} },
];
process.stdout.write(JSON.stringify({
  queries: [searchQueries(" Snake  Road Waterdown "), searchQueries("Snake Road"),
    searchQueries("")],
  merged: mergeSearchResults([full, nameOnly], "Snake Road Waterdown")
    .map((i) => [i.title, i.props.extent || null]),
  capped: mergeSearchResults([nameOnly], "Snake Road", 1).map((i) => i.title),
  source: [sourceLabel("NRCan HRDEM mosaic 1 m (lidar DTM)", 1),
    sourceLabel("Copernicus GLO-30", 30.4), sourceLabel("", 1)],
}));
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_search_helpers_under_node(tmp_path):
    script = tmp_path / "probe.js"
    script.write_text(APP_JS.read_text("utf-8") + "\n;" + SEARCH_PROBE, encoding="utf-8")
    proc = subprocess.run([NODE, str(script)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["queries"] == [["Snake Road Waterdown", "Snake Road"], ["Snake Road"], []]
    # One entry per road, covering all its pieces; the road the player meant comes first.
    assert out["merged"] == [
        ["Snake Road", [-79.8937, 43.3253, -79.8842, 43.2977]],
        ["Waterdown Road", [-79.8596, 43.3119, -79.8527, 43.3068]],
        ["Snake Forest Road", None],
    ]
    assert out["capped"] == ["Snake Road"]
    assert out["source"] == [
        "Source: NRCan HRDEM mosaic 1 m (lidar DTM)",
        "Source: Copernicus GLO-30 (30 m)",
        "",
    ]


# ---------------------------------------------------------------------------------------------
# In a real browser (opt-in)
# ---------------------------------------------------------------------------------------------

LIVE = os.environ.get("DRIFTMAPS_E2E_LIVE") == "1"
SCENARIO = (
    {
        "search": "Snake Road Waterdown",
        "a": (43.29771, -79.88439),
        "b": (43.32555, -79.88968),
        "km": (3.0, 4.2),
    }
    if LIVE
    else {"search": "Forks of the Credit Road", "a": FORKS_A, "b": FORKS_B, "km": (0.5, 3.0)}
)


@pytest.fixture
def browser_page():
    if os.environ.get("DRIFTMAPS_E2E_BROWSER") != "1":
        pytest.skip("browser test: set DRIFTMAPS_E2E_BROWSER=1 (needs internet and Playwright)")
    sync_api = pytest.importorskip("playwright.sync_api")
    args = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:  # Playwright's own proxy option would send 127.0.0.1 through the proxy too
        args += [f"--proxy-server={proxy}", "--proxy-bypass-list=127.0.0.1;localhost"]
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(args=args)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        problems: list[str] = []
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        yield page, problems
        browser.close()


def test_browser_flow(browser_page, tmp_path, monkeypatch, forks_route):
    page, problems = browser_page
    ac = make_ac(tmp_path / "assettocorsa")
    tracks = ac / "content" / "tracks"
    if LIVE:
        settings = Settings(tmp_path / "appdata", discover=lambda: ac)
        app = App(settings)
    else:
        monkeypatch.setattr(router, "route_from_pins", lambda pins, *a, **kw: forks_route)
        monkeypatch.setattr(elevation_sources, "sample_elevation", fake_sample)
        settings = Settings(tmp_path / "appdata", discover=lambda: ac)
        app = App(settings, build_fn=fake_build)
    srv = AppServer(app, port=0).start()
    SCREENS.mkdir(parents=True, exist_ok=True)

    def shot(name: str) -> None:
        page.screenshot(path=str(SCREENS / f"{name}.png"))

    def idle() -> None:
        page.wait_for_function(
            "() => window.acmg.map && window.acmg.map.loaded() && !window.acmg.map.isMoving()",
            timeout=60_000,
        )
        page.wait_for_timeout(500)

    def click(lat: float, lon: float) -> None:
        x, y = page.evaluate(
            "([lon, lat]) => { const p = window.acmg.map.project([lon, lat]); return [p.x, p.y]; }",
            [lon, lat],
        )
        box = page.locator("#map").bounding_box()
        page.mouse.click(box["x"] + x, box["y"] + y)

    try:
        page.goto(srv.url)
        page.wait_for_function("() => window.acmg && window.acmg.state.mapReady", timeout=60_000)
        idle()
        page.wait_for_function("() => document.getElementById('ac-pill').dataset.kind === 'ok'")
        shot("01_home")

        page.fill("#search", SCENARIO["search"])
        page.wait_for_selector("#search-results li[role=option]", timeout=30_000)
        shot("02_search")
        page.locator("#search-results li[role=option]").first.dispatch_event("mousedown")
        page.wait_for_timeout(1200)
        idle()
        shot("03_search_result")

        a, b = SCENARIO["a"], SCENARIO["b"]
        page.evaluate(
            "([a, b]) => window.acmg.map.fitBounds([[Math.min(a[1], b[1]), Math.min(a[0], b[0])],"
            " [Math.max(a[1], b[1]), Math.max(a[0], b[0])]], {padding: 80, duration: 0})",
            [list(a), list(b)],
        )
        idle()
        click(*a)
        page.wait_for_timeout(400)
        click(*b)
        page.wait_for_timeout(300)
        shot("04_finding_road")
        page.wait_for_function(
            "() => !document.getElementById('route-info').hidden"
            " || document.getElementById('route-status').classList.contains('error')",
            timeout=300_000,
        )
        assert page.text_content("#route-status") == ""
        km = page.evaluate("() => window.acmg.state.route.length_m") / 1000
        assert SCENARIO["km"][0] < km < SCENARIO["km"][1]
        assert page.text_content("#route-length").endswith("km")

        page.wait_for_function(
            "() => document.getElementById('profile-stats').textContent.length > 0"
            " || document.getElementById('profile-empty').textContent.includes('unavailable')",
            timeout=300_000,
        )
        assert page.text_content("#profile-stats").startswith("Climb")
        box = page.locator("#profile-canvas").bounding_box()
        page.mouse.move(box["x"] + box["width"] * 0.45, box["y"] + box["height"] * 0.5)
        page.wait_for_timeout(400)
        assert page.is_visible("#profile-tip")
        idle()
        shot("05_route_and_profile")

        page.fill("#track-name", NAME)
        page.click("#btn-build")
        page.wait_for_timeout(2000)
        assert page.locator("#job").is_visible()
        shot("06_building")
        page.wait_for_function(
            "() => ['done', 'error'].includes(document.getElementById('job').dataset.state)",
            timeout=1_800_000,
        )
        page.wait_for_timeout(800)
        shot("07_built")
        assert page.get_attribute("#job", "data-state") == "done", page.text_content("#job")
        assert "Installed!" in page.text_content("#job-result")
        assert is_driftmaps_track(tracks / TRACK_ID)

        page.wait_for_selector("#track-list li", timeout=10_000)
        assert NAME in page.text_content("#track-list")
        assert "Installed" in page.text_content("#track-list")
        page.locator("#track-list li").first.scroll_into_view_if_needed()
        shot("08_my_tracks")
        page.locator("#track-list li .icon-btn.danger").first.click()
        page.wait_for_selector("#confirm-dialog[open]")
        shot("09_confirm_delete")
        page.click("#confirm-ok")
        page.wait_for_function(
            "() => document.querySelectorAll('#track-list li').length === 0", timeout=10_000
        )
        page.wait_for_timeout(300)
        shot("10_deleted")
        assert not (tracks / TRACK_ID).exists()
        assert problems == []
    finally:
        try:
            shot("zz_last")
        finally:
            srv.stop()
