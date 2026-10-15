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

The loop scenario (`test_browser_loop`, opt-in with DRIFTMAPS_E2E_LOOP=1 as well) is always
live: it clicks four points round a block of streets in Waterdown (A to D), presses "Back to
start", checks the loop on the map (closed line, labels, toggling back to A to B, Shift+click
in between and Undo), creates the map with the real builder into a fake Assetto Corsa
(DRIFTMAPS_AC_ROOT) and checks it is a circuit. Screenshots go to out/app_screens/loop_*.png,
the route and the result to loop_route.json / loop_result.json beside them.

The route options scenario (`test_browser_route_options`, opt-in with DRIFTMAPS_E2E_OPTIONS=1
as well) is live too (real router and elevation, fake build): A and B on Snake Road, the points
move onto the road, at least two route options, each profile from A's height to B's, picking
option 2 (list, [ and ], clicking its grey line) and building it, dragging the line to add a
point, and a click far from any road that is refused. Screenshots: out/app_screens/options_*.png.
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


# ---------------------------------------------------------------------------------------------
# In a real browser: a loop ("Back to start"), live (opt-in)
# ---------------------------------------------------------------------------------------------

# Round a block in Waterdown: Flamboro St (A), Dundas St E (B), Mill St S (C), Main St S (D),
# then back to A along Flamboro St: about 1.4 km.
LOOP_POINTS = [
    (43.33140, -79.89173),
    (43.33355, -79.89272),
    (43.33302, -79.88976),
    (43.33059, -79.88994),
]
LOOP_BETWEEN = (43.33206, -79.89010)  # on Union Street, between C and D
LOOP_NAME = "Waterdown Loop (App Test)"
LOOP_ID = "dm_waterdown_loop_app_test"


def test_browser_loop(browser_page, tmp_path, monkeypatch):
    if os.environ.get("DRIFTMAPS_E2E_LOOP") != "1":
        pytest.skip("live loop test: set DRIFTMAPS_E2E_LOOP=1 (downloads map and elevation data)")
    page, problems = browser_page
    ac = make_ac(tmp_path / "assettocorsa")
    monkeypatch.setenv("DRIFTMAPS_AC_ROOT", str(ac))
    tracks = ac / "content" / "tracks"
    # Assetto Corsa from DRIFTMAPS_AC_ROOT, downloads cached in DRIFTMAPS_CACHE_DIR if set.
    srv = AppServer(App(Settings(tmp_path / "appdata")), port=0).start()
    SCREENS.mkdir(parents=True, exist_ok=True)
    js_state = "window.acmg.state"

    def shot(name: str) -> None:
        page.screenshot(path=str(SCREENS / f"loop_{name}.png"))

    def idle() -> None:
        page.wait_for_function(
            "() => window.acmg.map && window.acmg.map.loaded() && !window.acmg.map.isMoving()",
            timeout=60_000,
        )
        page.wait_for_timeout(500)

    def click(lat: float, lon: float, shift: bool = False) -> None:
        x, y = page.evaluate(
            "([lon, lat]) => { const p = window.acmg.map.project([lon, lat]); return [p.x, p.y]; }",
            [lon, lat],
        )
        box = page.locator("#map").bounding_box()
        if shift:
            page.keyboard.down("Shift")
        page.mouse.click(box["x"] + x, box["y"] + y)
        if shift:
            page.keyboard.up("Shift")
        page.wait_for_timeout(450)

    def wait_route(loop: bool) -> dict:
        page.wait_for_function(
            f"(loop) => {{ const s = {js_state}; return !s.routeStale && "
            "((s.route && !!s.route.loop === loop) || !!s.routeError); }",
            arg=loop,
            timeout=300_000,
        )
        assert page.text_content("#route-status") == "", page.text_content("#route-status")
        return page.evaluate(f"() => {js_state}.route")

    def labels() -> list[str]:
        """The map markers' letters, in the app's point order (not DOM order)."""
        return page.evaluate(
            f"() => {js_state}.pins.map(p => p.el.querySelector('.dm-pin-label').textContent)"
        )

    try:
        page.goto(srv.url)
        page.wait_for_function("() => window.acmg && window.acmg.state.mapReady", timeout=60_000)
        page.wait_for_function("() => document.getElementById('ac-pill').dataset.kind === 'ok'")
        assert page.text_content("#btn-build").strip() == "Create map"
        assert not page.locator("#opt-trees").is_visible()  # under the closed "More options"
        lats = [p[0] for p in LOOP_POINTS]
        lons = [p[1] for p in LOOP_POINTS]
        page.evaluate(
            "(b) => window.acmg.map.fitBounds(b, {padding: 110, duration: 0})",
            [[min(lons), min(lats)], [max(lons), max(lats)]],
        )
        idle()

        click(*LOOP_POINTS[0])
        assert page.is_disabled("#btn-loop")  # needs two points
        shot("01_start_A")
        for p in LOOP_POINTS[1:]:
            click(*p)
        ab = wait_route(False)
        assert labels() == ["A", "B", "C", "D"]
        assert page.eval_on_selector_all(
            "#pin-list .pin-badge", "els => els.map(e => e.textContent)"
        ) == ["A", "B", "C", "D"]
        assert "Finish" in page.text_content("#pin-list li:nth-child(4)")
        assert page.is_enabled("#btn-loop")
        page.click("#btn-fit")
        idle()
        shot("02_points_A_to_D")

        # Back to start: the loop closes on the map.
        page.click("#btn-loop")
        loop = wait_route(True)
        assert page.get_attribute("#btn-loop", "aria-pressed") == "true"
        assert loop["line"][0] == loop["line"][-1]
        assert 1.1 < loop["length_m"] / 1000 < 1.8, loop["length_m"]
        assert loop["length_m"] != ab["length_m"]
        assert page.text_content("#route-length-label") == "Lap length"
        assert "Start and finish" in page.text_content("#pin-list li:nth-child(1)")
        assert "Back to the start" in page.text_content("#pin-list")
        assert labels() == ["A", "B", "C", "D"]
        page.wait_for_function(
            "() => document.getElementById('profile-stats').textContent.length > 0",
            timeout=300_000,
        )
        page.click("#btn-fit")
        idle()
        shot("03_back_to_start")

        # Off again: A to B, D is the finish; on again: the same loop.
        page.click("#btn-loop")
        again = wait_route(False)
        assert again["length_m"] == pytest.approx(ab["length_m"], abs=0.5)
        idle()
        shot("04_toggled_off")
        page.click("#btn-loop")
        assert wait_route(True)["route_id"] == loop["route_id"]

        # Shift+click on the loop between C and D inserts a point there (the old D becomes
        # E); Undo takes it out again.
        click(*LOOP_BETWEEN, shift=True)
        between = wait_route(True)
        assert labels() == ["A", "B", "C", "D", "E"]
        pins = page.evaluate(f"() => {js_state}.pins.map(p => p.lngLat)")
        assert pins[3] == pytest.approx([LOOP_BETWEEN[1], LOOP_BETWEEN[0]], abs=2e-4)
        assert between["length_m"] == pytest.approx(loop["length_m"], abs=25.0)
        idle()
        shot("05_shift_click_between")
        page.click("#btn-undo")
        assert wait_route(True)["route_id"] == loop["route_id"]
        assert labels() == ["A", "B", "C", "D"]

        page.click("#more-options summary")
        page.wait_for_timeout(300)
        assert page.locator("#opt-trees").is_visible()
        shot("06_more_options")
        page.click("#more-options summary")

        page.fill("#track-name", LOOP_NAME)
        page.click("#btn-build")
        page.wait_for_timeout(2500)
        assert page.locator("#job").is_visible()
        shot("07_creating")
        page.wait_for_function(
            "() => ['done', 'error'].includes(document.getElementById('job').dataset.state)",
            timeout=1_800_000,
        )
        page.wait_for_timeout(800)
        shot("08_created")
        assert page.get_attribute("#job", "data-state") == "done", page.text_content("#job")
        assert "Installed!" in page.text_content("#job-result")
        assert "Lap" in page.text_content("#job-result")
        track = tracks / LOOP_ID
        assert is_driftmaps_track(track)
        ui = json.loads((track / "ui" / "ui_track.json").read_text(encoding="utf-8"))
        assert "circuit" in ui["tags"] and ui["run"] in ("clockwise", "anticlockwise")
        assert Route.load(track / "route.json").closed
        page.wait_for_selector("#track-list li", timeout=10_000)
        assert "Loop" in page.text_content("#track-list")
        page.locator("#track-list li").first.scroll_into_view_if_needed()
        shot("09_my_tracks")

        # Keep what was built for a closer look (debug render, zip size).
        shutil.copyfile(track / "route.json", SCREENS / "loop_route.json")
        result = page.evaluate("() => fetch('/api/tracks').then(r => r.json())")
        (SCREENS / "loop_result.json").write_text(
            json.dumps(
                {
                    "tracks": result,
                    "job_result": page.text_content("#job-result"),
                    "ui_track": ui,
                    "kn5_bytes": (track / f"{LOOP_ID}.kn5").stat().st_size,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        assert problems == []
    finally:
        try:
            shot("zz_last")
        finally:
            srv.stop()


# ---------------------------------------------------------------------------------------------
# In a real browser: route options, snapping and dragging the line, live (opt-in)
# ---------------------------------------------------------------------------------------------

# Snake Road, Waterdown: up the escarpment (about 3.5 km). The other way round is Old York
# Road and Plains Road. OPTIONS_FAR is in the woods between them, 400 m from any road.
OPTIONS_A = (43.29771, -79.88439)
OPTIONS_B = (43.32555, -79.88968)
OPTIONS_FAR = (43.31121, -79.88439)


def _xy(line, lat0: float) -> np.ndarray:
    """[lon, lat] points -> local metres (equirectangular; fine over a few km)."""
    ll = np.asarray(line, float)
    return np.stack([ll[:, 0] * 111_320.0 * np.cos(np.radians(lat0)), ll[:, 1] * 110_540.0], -1)


def _dist_to_line(line, p) -> float:
    """Metres from [lon, lat] point p to a [lon, lat] line."""
    xy = _xy(list(line) + [p], p[1])
    pts, q = xy[:-1], xy[-1]
    a, ab = pts[:-1], pts[1:] - pts[:-1]
    t = np.clip(((q - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + ab * t[:, None] - q, axis=1)))


def _far_from(line, other, min_m: float = 150.0) -> list[float] | None:
    """A vertex of `line` near its middle that is at least `min_m` from `other`."""
    n = len(line)
    for f in (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8):
        p = line[int(f * (n - 1))]
        if _dist_to_line(other, p) > min_m:
            return p
    return None


def test_browser_route_options(browser_page, tmp_path):
    if os.environ.get("DRIFTMAPS_E2E_OPTIONS") != "1":
        pytest.skip("live route options test: set DRIFTMAPS_E2E_OPTIONS=1 (downloads map data)")
    page, problems = browser_page
    ac = make_ac(tmp_path / "assettocorsa")
    # Real routing and elevation profiles (downloads cached in DRIFTMAPS_CACHE_DIR if set); the
    # build itself is a fake (this test checks which route it gets).
    app = App(Settings(tmp_path / "appdata", discover=lambda: ac), build_fn=fake_build)
    srv = AppServer(app, port=0).start()
    SCREENS.mkdir(parents=True, exist_ok=True)
    js_state = "window.acmg.state"

    def shot(name: str) -> None:
        page.screenshot(path=str(SCREENS / f"options_{name}.png"))

    def idle() -> None:
        page.wait_for_function(
            "() => window.acmg.map && window.acmg.map.loaded() && !window.acmg.map.isMoving()",
            timeout=60_000,
        )
        page.wait_for_timeout(500)

    def screen(lon: float, lat: float) -> tuple[float, float]:
        x, y = page.evaluate(
            "([lon, lat]) => { const p = window.acmg.map.project([lon, lat]); return [p.x, p.y]; }",
            [lon, lat],
        )
        box = page.locator("#map").bounding_box()
        return box["x"] + x, box["y"] + y

    def click(lat: float, lon: float) -> None:
        page.mouse.click(*screen(lon, lat))
        page.wait_for_timeout(450)

    def wait_route() -> dict:
        page.wait_for_function(
            f"() => {{ const s = {js_state};"
            " return !s.routeStale && (!!s.route || !!s.routeError); }",
            timeout=300_000,
        )
        page.wait_for_timeout(300)
        return page.evaluate(f"() => {js_state}")

    def selected_id() -> str:
        return page.evaluate(f"() => {js_state}.route.route_id")

    def assert_pins_drawn_where_they_are() -> None:
        """Each marker's tip is on its point (every marker, not just A)."""
        off = page.evaluate(
            """() => {
              const s = window.acmg.state, m = window.acmg.map;
              const box = document.getElementById('map').getBoundingClientRect();
              return s.pins.map(p => {
                const r = p.el.getBoundingClientRect(), q = m.project(p.lngLat);
                return [r.left + r.width / 2 - box.left - q.x, r.bottom - box.top - q.y];
              });
            }"""
        )
        for dx, dy in off:
            assert abs(dx) < 2 and abs(dy) < 3, off

    try:
        page.goto(srv.url)
        page.wait_for_function("() => window.acmg && window.acmg.state.mapReady", timeout=60_000)
        a, b = OPTIONS_A, OPTIONS_B
        page.evaluate(
            "(b) => window.acmg.map.fitBounds(b, {padding: 120, duration: 0})",
            [[b[1] - 0.012, a[0] - 0.002], [a[1] + 0.012, b[0] + 0.002]],
        )
        idle()

        # A clicked 25 m off the road: the pin moves onto the road it grabbed.
        click(a[0], a[1] + 0.0003)
        click(*b)
        st = wait_route()
        assert st["routeError"] == "", st["routeError"]
        options = st["routeOptions"]
        assert len(options) >= 2, [o["names"] for o in options]
        assert st["optionIndex"] == 0 and st["route"]["route_id"] == options[0]["route_id"]
        pins = page.evaluate(f"() => {js_state}.pins.map(p => p.lngLat)")
        snapped = st["routeSnapped"]
        for (lon, lat), (slat, slon) in zip(pins, snapped, strict=True):
            assert (lon, lat) == pytest.approx((slon, slat), abs=1e-6)
        assert abs(pins[0][0] - (a[1] + 0.0003)) > 1e-4  # moved: it was clicked off the road
        assert_pins_drawn_where_they_are()
        # Every option's elevation profile runs from A's height to B's (Snake Road climbs the
        # escarpment, about 95 m).
        (lat_a, lon_a), (lat_b, lon_b) = snapped
        ends = elevation_sources.sample_elevation(
            [lon_a, lon_b], [lat_a, lat_b], cache_dir=app.settings.cache_dir
        ).heights
        assert ends[1] - ends[0] > 60.0, ends
        for opt in options[:2]:
            prof = app.profile({"route_id": opt["route_id"]})
            assert prof["z"][0] == pytest.approx(ends[0], abs=4.0), (opt["names"], prof["z"][:3])
            assert prof["z"][-1] == pytest.approx(ends[1], abs=4.0), (opt["names"], prof["z"][-3:])
        assert page.is_visible("#route-options")
        labels = page.eval_on_selector_all(
            "#route-option-list button", "els => els.map(e => e.textContent)"
        )
        assert labels[0].startswith("1 · ") and "Snake Road" in labels[0]
        assert labels[1].startswith("2 · ")
        pressed = "#route-option-list li:nth-child({}) button"
        assert page.get_attribute(pressed.format(1), "aria-pressed") == "true"
        page.wait_for_function(
            "() => document.getElementById('profile-stats').textContent.length > 0"
        )
        idle()
        shot("01_two_options")

        # Option 2 from the list: the blue line, length, profile and name follow it.
        page.click("#route-option-list li:nth-child(2) button")
        assert selected_id() == options[1]["route_id"] != options[0]["route_id"]
        assert page.get_attribute(pressed.format(2), "aria-pressed") == "true"
        option_2 = page.text_content(pressed.format(2))
        assert option_2.split(" · ")[1] == page.text_content("#route-length")
        assert options[1]["names"][0] in page.input_value("#track-name")
        page.wait_for_function(
            "(id) => window.acmg.state.profile && window.acmg.state.profile.route_id === id",
            arg=options[1]["route_id"],
        )
        idle()
        shot("02_option_2_selected")

        # [ and ] cycle through the options.
        page.keyboard.press("[")
        assert selected_id() == options[0]["route_id"]
        page.keyboard.press("]")
        assert selected_id() == options[1]["route_id"]
        page.keyboard.press("[")

        # Hovering a grey line highlights it; clicking it selects it.
        grey = _far_from(options[1]["line"], options[0]["line"])
        assert grey is not None
        page.mouse.move(*screen(*grey))
        page.wait_for_timeout(300)
        assert page.evaluate(f"() => {js_state}.altHover") == 1
        assert page.evaluate("() => window.acmg.map.getCanvas().style.cursor") == "pointer"
        shot("03_hover_grey_line")
        page.mouse.click(*screen(*grey))
        page.wait_for_timeout(300)
        assert selected_id() == options[1]["route_id"]
        assert page.evaluate(f"() => {js_state}.pins.length") == 2  # selected, no point added

        # Create map builds the selected option.
        page.fill("#track-name", "Snake Road Option 2 (App Test)")
        with page.expect_request("**/api/build") as req:
            page.click("#btn-build")
        sent = json.loads(req.value.post_data)
        assert sent["route_id"] == options[1]["route_id"]
        page.wait_for_function(
            "() => ['done', 'error'].includes(document.getElementById('job').dataset.state)",
            timeout=60_000,
        )
        assert page.get_attribute("#job", "data-state") == "done"
        shot("04_built_option_2")

        # Drag the blue line (option 1, Snake Road) over to the other road: a point goes in
        # between A and B there and the route goes that way.
        page.keyboard.press("[")
        assert selected_id() == options[0]["route_id"]
        grab = _far_from(options[0]["line"], options[1]["line"])
        drop = _far_from(options[1]["line"], options[0]["line"], 250.0)
        assert grab is not None and drop is not None
        page.mouse.move(*screen(*grab))
        page.wait_for_timeout(200)
        assert page.evaluate("() => window.acmg.map.getCanvas().style.cursor") == "move"
        page.mouse.down()
        (x0, y0), (x1, y1) = screen(*grab), screen(*drop)
        for k in range(1, 11):
            page.mouse.move(x0 + (x1 - x0) * k / 10, y0 + (y1 - y0) * k / 10)
            page.wait_for_timeout(30)
        shot("05_dragging_line")
        page.mouse.up()
        st = wait_route()
        assert st["routeError"] == ""
        pins = page.evaluate(f"() => {js_state}.pins.map(p => p.lngLat)")
        assert len(pins) == 3
        assert page.evaluate(
            f"() => {js_state}.pins.map(p => p.el.querySelector('.dm-pin-label').textContent)"
        ) == ["A", "B", "C"]
        assert pins[1] == pytest.approx(drop, abs=5e-4)  # the new B, snapped onto that road
        assert _dist_to_line(st["route"]["line"], pins[1]) < 5.0
        assert_pins_drawn_where_they_are()
        assert st["route"]["route_id"] not in {o["route_id"] for o in options}
        idle()
        shot("06_dragged_line_new_point")

        # A click far from any road: no point is added, and the app says why.
        before = st["route"]["route_id"]
        click(*OPTIONS_FAR)
        page.wait_for_function(
            "() => !document.getElementById('toast').hidden"
            " && document.getElementById('toast').textContent.includes('No road found')",
            timeout=300_000,
        )
        assert "point D" in page.text_content("#toast")
        shot("07_far_click_refused")
        st = wait_route()
        assert len(page.evaluate(f"() => {js_state}.pins")) == 3
        assert st["route"]["route_id"] == before
        assert problems == []
    finally:
        try:
            shot("zz_last")
        finally:
            srv.stop()
