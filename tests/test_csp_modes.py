"""Touge Battle CSP New Mode (csp/new-modes/acmg-touge) and its installer (driftmaps.csp_modes).

- The mode's files: manifest rules, settings.ini defaults matching the script, and every CSP
  API name the script uses being one checked in the CSP 0.2.11 New Modes SDK.
- `install_touge` / `drift-maps install-touge`: copies, replaces only its own folder, the
  Track Day variant, uninstall.
- `data/acmg_touge.json` from a real (fixture) A->B build agrees with the KN5's dummies.
- With `lupa` installed (`uv run --with lupa pytest tests/test_csp_modes.py`), mode.lua is
  compiled with LuaJIT and its state machine is driven through whole races against a stubbed
  CSP API on a straight 1 km road: AC's race start, the finish at B, braking both cars, the
  result screen, reruns from our countdown, a chase start, a session restart, a stuck AI, the
  gate-dummy fallback and the error screens.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from driftmaps.build import SPAWN_HEIGHT_M, TOUGE_CHASE_GAP_M, TOUGE_FILE, build_track, lane_offset
from driftmaps.cli import app
from driftmaps.csp_modes import (
    MODE_AUTHOR,
    TOUGE_MODE_ID,
    TOUGE_TRACKDAY_ID,
    ModeInstallError,
    install_touge,
    is_ours,
    manifest_value,
    modes_source_dir,
    new_modes_dir,
    trackday_manifest,
    uninstall_touge,
)

MODE_DIR = Path(__file__).parents[1] / "csp" / "new-modes" / TOUGE_MODE_ID
MODE_LUA = MODE_DIR / "mode.lua"

# Every CSP API name mode.lua may use. Each one was checked in the generated SDK shipped with
# CSP 0.2.11 (extension/internal/lua-sdk/ac_new_modes/lib.lua, 2025-06-11): functions as
# `function <name>(`, enum values inside their table. Add a name here only after checking it
# there (or in https://github.com/ac-custom-shaders-patch/acc-lua-sdk for hand-written parts).
VERIFIED_0211_API = {
    "JSON.parse",
    "ac.ControlButton",
    "ac.INIConfig.scriptSettings",
    "ac.SessionType.Race",
    "ac.debug",
    "ac.endSession",
    "ac.findNodes",
    "ac.getCar",
    "ac.getSim",
    "ac.getTrackDataFilename",
    "ac.getUI",
    "ac.lapTimeToString",
    "ac.log",
    "ac.onSessionStart",
    "ac.overrideCarControls",
    "ac.setMessage",
    "ac.tryToRestartSession",
    "ac.worldCoordinateToTrackProgress",
    "io.load",
    "physics.disableCarCollisions",
    "physics.lockUserControlsFor",
    "physics.preventAIFromRetiring",
    "physics.resetCarState",
    "physics.setAIAggression",
    "physics.setAICarPosition",
    "physics.setAILevel",
    "physics.setAINoInput",
    "physics.setAIStopCounter",
    "physics.setCarPosition",
    "physics.setCarVelocity",
    "physics.setExtraAIGrip",
    "ui.Font.Huge",
    "ui.Font.Title",
    "ui.KeyIndex.R",
    "ui.button",
    "ui.popFont",
    "ui.pushFont",
    "ui.radioButton",
    "ui.sameLine",
    "ui.separator",
    "ui.setNextItemWidth",
    "ui.slider",
    "ui.text",
    "ui.textAligned",
    "ui.toolWindow",
    "ui.transparentWindow",
}
# Methods on SDK objects (SceneReference, ControlButton, INIConfig, vec3), also checked in 0.2.11.
VERIFIED_0211_METHODS = {"clone", "empty", "getPosition", "mapSection", "pressed"}


def _code(path: Path) -> str:
    """Lua source without comments (good enough for our own files: no '--' inside strings)."""
    text = re.sub(r"--\[\[.*?\]\]", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", text)


# --------------------------------------------------------------------------------------------
# Mode files
# --------------------------------------------------------------------------------------------


def test_manifest_rules():
    text = (MODE_DIR / "manifest.ini").read_text(encoding="utf-8")
    assert manifest_value(text, "AUTHOR") == MODE_AUTHOR
    assert manifest_value(text, "BASE_MODE") == "RACE"
    assert manifest_value(text, "AI_LIMIT") == "1"
    assert manifest_value(text, "ALLOW_PHYSICS_ALTERATIONS") == "1"
    assert "#touge" in manifest_value(text, "ALLOWED_TRACKS")
    for name in ("manifest.ini", "mode.lua", "settings.ini", "description.txt"):
        assert (MODE_DIR / name).is_file(), name


def test_settings_match_script_defaults():
    """settings.ini (what Content Manager shows) and mapSection's defaults agree."""
    ini = {}
    for line in (MODE_DIR / "settings.ini").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z_]+)=([^;]*);", line)
        if m:
            ini[m.group(1)] = m.group(2).strip()
    block = re.search(r"mapSection\('SETTINGS', \{(.*?)\}\)", _code(MODE_LUA), re.DOTALL)
    assert block
    lua = dict(re.findall(r"(\w+) = ([^,\n]+),", block.group(1)))
    assert set(ini) == set(lua)
    for key, value in ini.items():
        expected = {"true": "1", "false": "0"}.get(lua[key], lua[key].strip("'"))
        assert value == expected, key


def test_mode_uses_only_verified_csp_api():
    code = _code(MODE_LUA)
    used = set(
        re.findall(r"\b(?:ac|physics|ui|io|JSON|render)\.[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", code)
    )
    assert used, "no API calls found"
    assert used <= VERIFIED_0211_API, sorted(used - VERIFIED_0211_API)
    methods = set(re.findall(r":([A-Za-z_]\w*)\(", code))
    assert methods <= VERIFIED_0211_METHODS, sorted(methods - VERIFIED_0211_METHODS)
    # Newer-than-0.2.11 or invented names we must not drift into.
    for name in ("addMainMenuSettings", "setAIPedals", "setAISteer", "physics.setAITimeToStart"):
        assert name not in code
    assert "function script.update(" in code and "function script.drawUI(" in code


# --------------------------------------------------------------------------------------------
# Installer
# --------------------------------------------------------------------------------------------


def _ac_root(tmp_path: Path, csp: bool = True) -> Path:
    root = tmp_path / "assettocorsa"
    (root / "content" / "tracks").mkdir(parents=True)
    (root / "acs.exe").write_bytes(b"")
    if csp:
        (root / "extension" / "config").mkdir(parents=True)
    return root


def test_source_dir_is_the_repo_folder():
    assert modes_source_dir(env={}) == MODE_DIR.parent


def test_install_replace_and_uninstall(tmp_path):
    root = _ac_root(tmp_path)
    (folder,) = install_touge(root)
    assert folder == new_modes_dir(root) / TOUGE_MODE_ID
    for name in ("manifest.ini", "mode.lua", "settings.ini", "description.txt"):
        assert (folder / name).read_bytes() == (MODE_DIR / name).read_bytes(), name
    assert is_ours(folder)
    (folder / "stale.lua").write_text("old")
    install_touge(root)  # our own folder: replaced
    assert not (folder / "stale.lua").exists()

    variant = install_touge(root, trackday_variant=True)[1]
    assert variant.name == TOUGE_TRACKDAY_ID
    text = (variant / "manifest.ini").read_text(encoding="utf-8")
    assert manifest_value(text, "BASE_MODE") == "TRACK_DAY"
    assert manifest_value(text, "NAME").endswith("- Track Day test")
    assert (variant / "mode.lua").read_bytes() == MODE_LUA.read_bytes()

    assert set(uninstall_touge(root)) == {folder, variant}
    assert not folder.exists() and not variant.exists()
    assert uninstall_touge(root) == []


def test_install_refuses_foreign_folder_and_missing_csp(tmp_path):
    root = _ac_root(tmp_path)
    foreign = new_modes_dir(root) / TOUGE_MODE_ID
    foreign.mkdir(parents=True)
    (foreign / "manifest.ini").write_text("[ABOUT]\nAUTHOR = someone else\n")
    with pytest.raises(ModeInstallError, match="not installed by"):
        install_touge(root)
    assert uninstall_touge(root) == []  # never removes somebody else's folder
    install_touge(root, force=True)
    assert is_ours(foreign)

    bare = _ac_root(tmp_path / "nocsp", csp=False)
    with pytest.raises(ModeInstallError, match="Custom Shaders Patch"):
        install_touge(bare)


def test_trackday_manifest_needs_one_base_mode():
    with pytest.raises(ModeInstallError):
        trackday_manifest("[RULES]\nAI_LIMIT=1\n")


def test_cli_install_touge(tmp_path):
    root = _ac_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["install-touge", "--ac-root", str(root)])
    assert result.exit_code == 0, result.output
    assert (new_modes_dir(root) / TOUGE_MODE_ID / "mode.lua").is_file()
    assert "docs/checklists/touge.md" in result.output
    result = runner.invoke(app, ["install-touge", "--ac-root", str(root), "--uninstall"])
    assert result.exit_code == 0 and "removed" in result.output
    bare = _ac_root(tmp_path / "nocsp", csp=False)
    result = runner.invoke(app, ["install-touge", "--ac-root", str(bare)])
    assert result.exit_code == 1


# --------------------------------------------------------------------------------------------
# data/acmg_touge.json from a real build
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from test_build import FIXTURES, FORKS_PINS, fixture_options

    from driftmaps.geo.overpass import parse_overpass
    from driftmaps.geo.router import route_from_pins

    osm = parse_overpass((FIXTURES / "osm" / "forks_of_the_credit.json").read_bytes())
    root = tmp_path_factory.mktemp("touge")
    return build_track(
        route_from_pins(FORKS_PINS, osm=osm), root / "out", fixture_options(root / "cache")
    )


def test_touge_json_matches_the_track(built):
    from driftmaps.ac.kn5 import read_kn5
    from driftmaps.ac.model import Dummy

    data = json.loads((built.track_dir / "data" / TOUGE_FILE).read_text(encoding="utf-8"))
    ui = json.loads((built.track_dir / "ui" / "ui_track.json").read_text(encoding="utf-8"))
    assert "touge" in ui["tags"]
    assert data["format"] == 1 and data["track_id"] == built.track_id
    geo = built.geometry
    assert data["length_m"] == pytest.approx(geo.length_m, abs=0.1)
    assert data["run_off_m"] > 100.0

    kn5 = read_kn5(built.track_dir / f"{built.track_id}.kn5")
    d = {
        n.name: n.transform[3, :3].astype(float) for n in kn5.root.children if isinstance(n, Dummy)
    }
    for key, prefix in (("start_gate", "AC_AB_START"), ("finish_gate", "AC_AB_FINISH")):
        g = data[key]
        left, right = d[f"{prefix}_L"], d[f"{prefix}_R"]
        mid = (left + right) / 2
        assert np.allclose(np.array(g["centre"])[[0, 2]], mid[[0, 2]], atol=0.01), key
        # the dummies are reach_m either side, L on the driver's left
        assert np.linalg.norm(left - right) / 2 == pytest.approx(g["reach_m"], abs=0.01)
        assert (left - right) @ np.array(g["left"]) > 0
        assert np.array(g["forward"]) @ np.array(g["left"]) == pytest.approx(0, abs=1e-3)
    grid = data["grid"]
    assert [p["name"] for p in grid] == ["AC_START_0", "AC_START_1"]
    for pose in grid:
        assert np.allclose(pose["position"], d[pose["name"]], atol=0.01)
    lead, chase = data["chase"]["leader"], data["chase"]["chaser"]
    assert np.allclose(lead["position"], grid[0]["position"], atol=0.01)
    gap = np.linalg.norm(np.array(lead["position"]) - np.array(chase["position"]))
    assert gap == pytest.approx(TOUGE_CHASE_GAP_M, abs=0.2)
    # the chaser is behind the leader on the driving lane, SPAWN_HEIGHT_M above the road
    m = geo.model
    s = m.s[np.argmin(np.linalg.norm(m.centre - np.array(chase["position"]), axis=1))]
    pos, fwd, left = m.at(s)
    assert (np.array(chase["position"]) - pos) @ left == pytest.approx(
        m.interp(lane_offset(geo), s), abs=0.3
    )
    assert np.array(chase["position"])[1] - pos[1] == pytest.approx(SPAWN_HEIGHT_M, abs=0.2)
    assert (np.array(lead["position"]) - np.array(chase["position"])) @ fwd > 9.0


# --------------------------------------------------------------------------------------------
# mode.lua under LuaJIT with a stubbed CSP API
# --------------------------------------------------------------------------------------------

# A straight road heading north (-z): A at z = 0, B at z = -1000, the AI spline from z = 100
# to z = -1100. Cars accelerate towards `target` (m/s) and brake at 8 m/s^2 when held.
HARNESS = r"""
H = { calls = {}, logs = {}, debug = {}, messages = {}, buttons = {}, pressed = false }
local V = {}
V.__index = V
function vec3(x, y, z) return setmetatable({ x = x or 0, y = y or 0, z = z or 0 }, V) end
V.__add = function (a, b) return vec3(a.x + b.x, a.y + b.y, a.z + b.z) end
V.__sub = function (a, b) return vec3(a.x - b.x, a.y - b.y, a.z - b.z) end
V.__mul = function (a, b)
  if type(a) == 'number' then a, b = b, a end
  return vec3(a.x * b, a.y * b, a.z * b)
end
function V:clone() return vec3(self.x, self.y, self.z) end
function vec2(x, y) return { x = x or 0, y = y or 0 } end
local function rec(name, ...) H.calls[#H.calls + 1] = { name, ... } end
local function progress(p) return (100 - p.z) / 1200 end

H.sim = { carsCount = 2, raceSessionType = 3, isSessionStarted = false, dt = 0.05,
  isReplayActive = false, isInMainMenu = false, trackLengthM = 1200 }
H.cars = {}
for i = 0, 1 do
  H.cars[i] = { position = vec3(i == 0 and 2 or -2, 0, 6), look = vec3(0, 0, -1), speedKmh = 0,
    splinePosition = 0, physicsAvailable = true, v = 0, target = 0, stop = 0, parked = false }
end

ac = { SessionType = { Race = 3, Practice = 1 } }
function ac.getSim() return H.sim end
function ac.getCar(i) if i < H.sim.carsCount then return H.cars[i] end return nil end
function ac.getUI() return { windowSize = vec2(1920, 1080) } end
ac.INIConfig = { scriptSettings = function ()
  return { mapSection = function (self, section, defaults)
    local t = {}
    for k, v in pairs(defaults) do t[k] = v end
    for k, v in pairs(H.settings or {}) do t[k] = v end
    return t
  end }
end }
function ac.getTrackDataFilename(name) return 'content/tracks/t/data/' .. name end
io.load = function (path) rec('io.load', path) return H.json end
JSON = { parse = function (s) return H.parseJson(s) end }
function ac.findNodes(name)
  return {
    empty = function (self) return H.nodes == nil or H.nodes[name] == nil end,
    getPosition = function (self) return H.nodes[name] end,
  }
end
function ac.worldCoordinateToTrackProgress(p) return progress(p) end
function ac.ControlButton(id, defaults)
  rec('ControlButton', id)
  return { pressed = function (self) local p = H.pressed; H.pressed = false; return p end }
end
function ac.log(s) H.logs[#H.logs + 1] = s end
function ac.debug(k, v) H.debug[k] = v end
function ac.setMessage(title, text) H.messages[#H.messages + 1] = text end
function ac.overrideCarControls(i)
  rec('overrideCarControls', i)
  H.controls = H.controls or { brake = 0 }
  return H.controls
end
function ac.tryToRestartSession() rec('tryToRestartSession') return true end
function ac.onSessionStart(callback) H.onSessionStart = callback end
function ac.endSession(message, ok, results)
  H.ended = { message = message, ok = ok, results = results }
end
function ac.lapTimeToString(ms) return string.format('%.3f', ms / 1000) end

physics = {}
for _, n in ipairs({ 'setAILevel', 'setAIAggression', 'setExtraAIGrip', 'setCarVelocity',
    'disableCarCollisions', 'resetCarState', 'preventAIFromRetiring' }) do
  physics[n] = function (...) rec(n, ...) end
end
local function teleport(name, i, pos, dir)
  rec(name, i, pos.x, pos.z)
  H.cars[i].position = pos:clone()
  H.cars[i].v = 0
end
function physics.setCarPosition(i, pos, dir) teleport('setCarPosition', i, pos, dir) end
function physics.setAICarPosition(i, pos, dir) teleport('setAICarPosition', i, pos, dir) end
function physics.setAIStopCounter(i, t) rec('setAIStopCounter', i, t) H.cars[i].stop = t end
function physics.setAINoInput(i, active, stall)
  rec('setAINoInput', i, active)
  H.cars[i].parked = active ~= false
end
function physics.lockUserControlsFor(t) rec('lockUserControlsFor', t) H.lock = t end

ui = { KeyIndex = { R = 82 }, Font = { Huge = 7, Title = 6 } }
function ui.transparentWindow(id, pos, size, noPadding, inputs, content) content() end
ui.toolWindow = ui.transparentWindow
for _, n in ipairs({ 'pushFont', 'popFont', 'textAligned', 'text', 'separator',
    'setNextItemWidth', 'sameLine' }) do
  ui[n] = function () end
end
function ui.slider(label, value) return value, false end
function ui.radioButton(label, checked)
  local b = H.buttons[label]; H.buttons[label] = nil; return b == true
end
function ui.button(label) local b = H.buttons[label]; H.buttons[label] = nil; return b == true end
script = {}

function H.step()
  local dt = H.sim.dt
  for i = 0, 1 do
    local c = H.cars[i]
    local held
    if i == 1 then
      held = c.parked or c.stop > 0
      c.stop = math.max(0, c.stop - dt)
    else
      held = (H.lock or 0) > 0 or (H.controls ~= nil and H.controls.brake > 0)
    end
    if H.sim.raceSessionType == 3 and not H.sim.isSessionStarted then
      c.v = 0 -- AC holds the grid until the green light
    elseif held then
      c.v = math.max(0, c.v - 8 * dt)
    else
      c.v = c.v + math.max(-8 * dt, math.min(4 * dt, c.target - c.v))
    end
    c.position = vec3(c.position.x, c.position.y, c.position.z - c.v * dt)
    c.speedKmh = c.v * 3.6
    c.splinePosition = progress(c.position)
  end
  if H.lock then H.lock = math.max(0, H.lock - dt) end
  script.update(dt)
  script.drawUI()
end

function H.runUntil(state, maxSteps)
  for k = 1, maxSteps do
    H.step()
    if H.debug['touge.state'] == state then return k end
  end
  return nil
end
"""

TRACK_JSON = {
    "format": 1,
    "finish_gate": {"centre": [0, 0, -1000], "forward": [0, 0, -1], "reach_m": 9},
    "grid": [
        {"name": "AC_START_0", "position": [-2, 1, 6], "forward": [0, 0, -1]},
        {"name": "AC_START_1", "position": [2, 1, 6], "forward": [0, 0, -1]},
    ],
    "chase": {
        "leader": {"position": [-2, 1, 6], "forward": [0, 0, -1]},
        "chaser": {"position": [-2, 1, 16], "forward": [0, 0, -1]},
    },
}


class Mode:
    """mode.lua loaded into a fresh LuaJIT state with the stubbed API."""

    def __init__(self, lupa, track_json: dict | None = TRACK_JSON, nodes=None, **settings):
        self.lua = lupa.LuaRuntime(unpack_returned_tuples=True)
        self.lua.execute(HARNESS)
        self.H = self.lua.globals().H
        self.H.parseJson = lambda s: self.lua.table_from(json.loads(s), recursive=True)
        self.H.json = json.dumps(track_json) if track_json is not None else None
        if nodes:  # track dummies: name -> (x, y, z)
            vec3 = self.lua.globals().vec3
            self.H.nodes = self.lua.table_from({k: vec3(*v) for k, v in nodes.items()})
        if settings:
            self.H.settings = self.lua.table_from(settings)
        self.lua.execute(MODE_LUA.read_text(encoding="utf-8"))

    @property
    def state(self) -> str:
        return self.H.debug["touge.state"]

    def run_until(self, state: str, max_steps: int = 4000) -> None:
        assert self.H.runUntil(state, max_steps) is not None, (state, self.state, self.logs())

    def car(self, i: int):
        return self.H.cars[i]

    def drive(self, player: float, ai: float) -> None:
        self.car(0).target, self.car(1).target = player, ai

    def logs(self) -> list[str]:
        return list(self.H.logs.values())

    def calls(self, name: str) -> list[tuple]:
        out = []
        for call in self.H.calls.values():
            args = list(call.values())
            if args[0] == name:
                out.append(tuple(args[1:]))
        return out

    def click(self, label: str) -> None:
        """Click a button: drawUI sees it this frame, update acts on it the next."""
        self.H.buttons[label] = True
        self.H.step()
        self.H.step()


@pytest.fixture(scope="module")
def lupa():
    module = pytest.importorskip("lupa")
    for name in ("luajit21", "luajit20"):  # CSP runs LuaJIT
        try:
            return __import__(f"lupa.{name}", fromlist=["LuaRuntime"])
        except ImportError:
            continue
    return module


def test_mode_lua_compiles(lupa):
    rt = lupa.LuaRuntime(unpack_returned_tuples=True)
    check = rt.eval("function (s) local f, e = loadstring(s, '@mode.lua') return f ~= nil, e end")
    ok, err = check(MODE_LUA.read_text(encoding="utf-8"))
    assert ok, err


def test_race_start_finish_and_reruns(lupa):
    mode = Mode(lupa)
    mode.H.step()
    assert mode.state == "grid"  # AC's own start lights; no teleport for side by side
    assert not mode.calls("setCarPosition")
    assert mode.calls("setAILevel") == [(1, 0.95)]
    assert mode.calls("setExtraAIGrip") == [(1, 1.2)]

    mode.H.sim.isSessionStarted = True
    mode.drive(30, 28)
    mode.run_until("finishing")
    assert any("car 0 passed B" in s for s in mode.logs())
    assert (1, True) in mode.calls("disableCarCollisions")  # no rear-ending after B
    mode.run_until("result")
    assert any(re.search(r"result: you won, gap \d+\.\d{3} s", s) for s in mode.logs())
    for _ in range(200):  # both cars come to a stop before the end of the run-off
        mode.H.step()
    assert mode.car(1).parked and mode.car(1).speedKmh < 3
    assert mode.car(0).speedKmh < 3 and mode.H.controls.brake == 1
    assert mode.car(0).position.z > -1000 - 220 and mode.car(1).position.z > -1000 - 220

    # Race again: both repaired and on the grid (AI on its driving-lane slot), held, then GO.
    mode.click("Race again (Shift+R)")
    assert mode.state == "countdown"
    assert mode.calls("resetCarState")[-2:] == [(0,), (1,)]
    assert (1, False) in mode.calls("setAINoInput") and (1, False) in mode.calls(
        "disableCarCollisions"
    )
    assert mode.calls("setCarPosition")[-1] == (0, 2, 6)
    assert mode.calls("setAICarPosition")[-1] == (1, -2, 6)
    mode.drive(28, 32)
    for _ in range(20):
        mode.H.step()
    assert mode.state == "countdown" and mode.car(0).speedKmh == 0 and mode.car(1).speedKmh == 0
    mode.run_until("racing")
    assert mode.H.controls.brake == 0 and mode.H.lock == 0
    mode.run_until("result")
    assert any("result: AI won" in s and "score you 1, AI 1" in s for s in mode.logs())

    # Chase start: AI ahead on the driving lane, you one car length behind.
    mode.click("Chase (AI leads)")
    mode.click("Race again (Shift+R)")
    assert mode.calls("setAICarPosition")[-1] == (1, -2, 6)
    assert mode.calls("setCarPosition")[-1] == (0, -2, 16)

    mode.run_until("result")
    assert mode.car(1).parked
    # Restart session: AC resets to its grid (the script may or may not be reloaded by CSP).
    mode.click("Restart session")
    assert mode.calls("tryToRestartSession")
    mode.H.sim.isSessionStarted = False
    vec3 = mode.lua.globals().vec3
    mode.car(0).position, mode.car(1).position = vec3(2, 0, 6), vec3(-2, 0, 6)  # AC's grid
    mode.H.onSessionStart(0, True)
    mode.H.step()
    mode.H.step()
    assert mode.state == "grid" and not mode.car(1).parked
    assert mode.calls("disableCarCollisions")[-1] == (1, False)
    mode.H.sim.isSessionStarted = True
    mode.drive(30, 28)
    mode.run_until("result")
    assert any("result: you won" in s and "score you 1, AI 0" in s for s in mode.logs())
    mode.click("End")
    assert mode.H.ended is not None and mode.H.ended.ok


def test_loser_far_behind_and_stuck_ai(lupa):
    mode = Mode(lupa, BRAKE_AFTER_FINISH=False)
    mode.H.step()
    mode.H.sim.isSessionStarted = True
    mode.drive(30, 0)  # the AI never gets going
    mode.run_until("finishing", 400)
    assert any("AI stuck" in s for s in mode.logs())
    mode.run_until("result")
    assert any("gap AI did not finish" in s for s in mode.logs())

    mode = Mode(lupa, BRAKE_AFTER_FINISH=False)
    mode.H.step()
    mode.H.sim.isSessionStarted = True
    mode.drive(10, 40)  # the AI wins by far more than the grace time
    mode.run_until("result")
    assert any(re.search(r"result: AI won, gap more than 8 s \(\d+ m short of B\)", s)
               for s in mode.logs())  # fmt: skip
    assert mode.car(1).parked
    assert mode.H.controls is None  # BRAKE_AFTER_FINISH=0: your pedals are never touched


def test_trackday_session_with_gate_dummies(lupa):
    """No acmg_touge.json: the B gate comes from AC_AB_FINISH_L/R; not a race session, so the
    first run already uses our countdown from where the cars spawned."""
    nodes = {"AC_AB_FINISH_L": (-8, 0, -1000), "AC_AB_FINISH_R": (8, 0, -1000)}
    mode = Mode(lupa, track_json=None, nodes=nodes)
    mode.H.sim.raceSessionType = 1
    mode.H.sim.isSessionStarted = True
    mode.H.step()
    assert mode.state == "countdown"
    assert mode.calls("setCarPosition")[-1] == (0, 2, 6)  # back where it spawned
    mode.drive(30, 25)
    mode.run_until("result")
    assert any("AC_AB_FINISH dummies" in s for s in mode.logs())
    assert any("result: you won" in s for s in mode.logs())


@pytest.mark.parametrize("problem", ["no_ai", "no_gate"])
def test_error_screens(lupa, problem):
    mode = Mode(lupa, track_json=None if problem == "no_gate" else TRACK_JSON)
    if problem == "no_ai":
        mode.H.sim.carsCount = 1
    mode.H.step()
    mode.H.step()
    assert mode.state == "error"
    expected = "one AI opponent" if problem == "no_ai" else "No B gate"
    assert any(expected in m for m in mode.H.messages.values())
    assert not mode.calls("setCarPosition")
