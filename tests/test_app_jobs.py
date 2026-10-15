"""The app's build jobs and settings: queue lifecycle, track ids, error messages, folders, and
the real `build_track` progress callback (offline Forks of the Credit fixture build)."""

from __future__ import annotations

import errno
import json
import logging
import threading
import time
from pathlib import Path

import pytest

from driftmaps.ac.scene import MARKER_FILE, TrackFolderError
from driftmaps.app import settings as settings_mod
from driftmaps.app.jobs import (
    NETWORK_MESSAGE,
    BuildRequest,
    BuildRunner,
    JobError,
    JobQueue,
    app_track_id,
    describe_error,
)
from driftmaps.app.settings import APP_DIR_NAME, Settings, SettingsError, normalize_ac_root
from driftmaps.build import BuildOptions, build_track
from driftmaps.elevation.sources import CogSource, SourceInfo
from driftmaps.geo.overpass import OverpassError, parse_overpass
from driftmaps.geo.router import route_from_pins
from driftmaps.geometry.terrain import TerrainOptions

FIXTURES = Path(__file__).parent / "fixtures"


def wait(job, states=("done", "error"), timeout=10.0):
    end = time.monotonic() + timeout
    while job.snapshot()["state"] not in states:
        assert time.monotonic() < end, job.snapshot()
        time.sleep(0.01)
    return job.snapshot()


# ---------------------------------------------------------------------------------------------
# JobQueue
# ---------------------------------------------------------------------------------------------


def test_job_lifecycle():
    gate = threading.Event()

    def runner(job):
        job.update("Step one", 0.25)
        job.update("Step one", 0.1)  # progress never goes backwards; no duplicate log line
        assert gate.wait(5)
        logging.getLogger("driftmaps.elevation.sources").warning("no height for 3 points")
        job.add_log("extra line")
        return {"track_id": "dm_x"}

    q = JobQueue(runner)
    try:
        job = q.submit("request")
        assert job.snapshot()["state"] in ("queued", "running")
        end = time.monotonic() + 5
        while job.snapshot()["step"] != "Step one":
            assert time.monotonic() < end
            time.sleep(0.01)
        snap = job.snapshot()
        assert snap["state"] == "running" and snap["progress"] == 0.25
        assert q.active() == [job]
        gate.set()
        snap = wait(job)
    finally:
        q.close(5)
    assert snap == {
        "job_id": job.id,
        "state": "done",
        "step": "Done",
        "progress": 1.0,
        "log": ["Step one", "note: no height for 3 points", "extra line", "Done"],
        "result": {"track_id": "dm_x"},
        "error": None,
    }
    assert q.get(job.id) is job and q.get("nope") is None and q.active() == []


def test_job_errors_and_queue_order():
    order = []

    def runner(job):
        order.append(job.request)
        if job.request == "net":
            raise OverpassError("all Overpass mirrors failed")
        if job.request == "plain":
            raise JobError("Pick another name.")
        return {}

    q = JobQueue(runner)
    try:
        jobs = [q.submit(r) for r in ("net", "plain", "ok")]
        snaps = [wait(j) for j in jobs]
    finally:
        q.close(5)
    assert order == ["net", "plain", "ok"]
    assert [s["state"] for s in snaps] == ["error", "error", "done"]
    assert snaps[0]["error"] == NETWORK_MESSAGE and snaps[0]["step"] == "Failed"
    assert snaps[1]["error"] == "Pick another name."
    assert snaps[1]["log"][-1] == "error: Pick another name."
    assert snaps[0]["result"] is None


def test_describe_error():
    assert describe_error(TrackFolderError("x")).startswith("A track folder of that id")
    assert describe_error(TimeoutError()) == NETWORK_MESSAGE
    assert "Content Manager" in describe_error(PermissionError(13, "denied", "C:/x.kn5"))
    assert "disk is full" in describe_error(OSError(errno.ENOSPC, "No space left"))
    assert describe_error(RuntimeError("bad")) == "The build failed: bad"


def test_app_track_id():
    assert app_track_id("Forks of the Credit") == "dm_forks_of_the_credit"
    assert app_track_id("Forks", "") == "dm_forks"
    assert app_track_id("Forks", "my road") == "dm_my_road"
    assert app_track_id("Forks", "dm_already") == "dm_already"
    assert app_track_id("Forks", "DM_Caps") == "dm_caps"
    assert len(app_track_id("x" * 80)) <= 32


# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------


def make_ac(root: Path) -> Path:
    (root / "content" / "tracks").mkdir(parents=True)
    (root / "acs.exe").write_bytes(b"")
    return root


def test_app_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings_mod.sys, "platform", "win32")
    env = {"LOCALAPPDATA": str(tmp_path / "Local")}
    assert settings_mod.app_data_dir(env) == tmp_path / "Local" / APP_DIR_NAME
    monkeypatch.setattr(settings_mod.sys, "platform", "linux")
    assert settings_mod.app_data_dir({"XDG_DATA_HOME": str(tmp_path)}) == tmp_path / APP_DIR_NAME
    assert settings_mod.app_data_dir({}) == Path.home() / ".local" / "share" / APP_DIR_NAME
    assert settings_mod.app_data_dir({"DRIFTMAPS_APP_DIR": str(tmp_path / "x")}) == tmp_path / "x"


def test_settings_folders_and_cache(tmp_path):
    s = Settings(tmp_path / "data", discover=lambda: None, env={})
    assert s.cache_dir == tmp_path / "data" / "cache"
    assert s.output_dir == tmp_path / "data" / "tracks"
    assert s.routes_dir == tmp_path / "data" / "routes"
    s2 = Settings(tmp_path / "data", env={"DRIFTMAPS_CACHE_DIR": str(tmp_path / "c")})
    assert s2.cache_dir == tmp_path / "c"


def test_settings_ac_discovery(tmp_path):
    found = make_ac(tmp_path / "steam" / "assettocorsa")
    calls = []

    def discover():
        calls.append(1)
        return found

    s = Settings(tmp_path / "data", discover=discover)
    assert s.ac_root() == found and s.ac_root() == found
    assert len(calls) == 1  # remembered
    assert s.tracks_dir() == found / "content" / "tracks"
    # A saved folder that stopped being valid falls back to the search.
    picked = make_ac(tmp_path / "other")
    s.set_ac_root(str(picked))
    assert s.ac_root() == picked
    (picked / "acs.exe").unlink()
    assert s.ac_root() == found
    # A broken search never breaks the app.
    broken = Settings(tmp_path / "d2", discover=lambda: 1 / 0)
    assert broken.ac_root() is None and broken.status()["ac_found"] is False
    # A corrupt settings file is ignored.
    (tmp_path / "d3").mkdir()
    (tmp_path / "d3" / "settings.json").write_text("{nope", encoding="utf-8")
    assert Settings(tmp_path / "d3", discover=lambda: None).ac_root_override is None


def test_normalize_ac_root(tmp_path):
    ac = make_ac(tmp_path / "assettocorsa")
    assert normalize_ac_root(str(ac)) == ac
    assert normalize_ac_root(f'"{ac / "content"}"') == ac
    assert normalize_ac_root(str(ac / "content" / "tracks")) == ac
    assert normalize_ac_root(str(ac / "acs.exe")) == ac
    for bad in ("", str(tmp_path), str(tmp_path / "missing")):
        with pytest.raises(SettingsError):
            normalize_ac_root(bad)


# ---------------------------------------------------------------------------------------------
# BuildRunner with the real build_track (offline fixture)
# ---------------------------------------------------------------------------------------------


def fixture_dem() -> dict[str, CogSource]:
    info = SourceInfo(
        "hrdem-fixture", "NRCan HRDEM (5 m test window)", 5.0, "DTM", "OGL-Canada-2.0",
        "Contains information licensed under the Open Government Licence - Canada.",
    )  # fmt: skip
    return {"hrdem-fixture": CogSource(info, str(FIXTURES / "dem" / "hrdem_forks_5m.tif"))}


def test_build_progress_callback_and_install(tmp_path):
    osm = parse_overpass((FIXTURES / "osm" / "forks_of_the_credit.json").read_bytes())
    route = route_from_pins([(43.80410, -79.99231), (43.81060, -79.98185)], osm=osm)
    ac = make_ac(tmp_path / "assettocorsa")
    settings = Settings(tmp_path / "data", discover=lambda: ac)
    steps: list[tuple[str, float]] = []

    def build_fn(route, out_dir, options: BuildOptions):
        app_progress = options.progress

        def progress(step, fraction):
            steps.append((step, fraction))
            app_progress(step, fraction)

        # The app's options, with the offline fixture data a test needs.
        options.progress = progress
        options.elevation_sources = fixture_dem()
        options.elevation_chain = ["hrdem-fixture"]
        options.terrain = TerrainOptions(
            rings=((15.0, 3.0), (60.0, 8.0), (150.0, 20.0)), near_m=60.0
        )
        options.osm_barriers = ()
        options.offline = True
        return build_track(route, out_dir, options)

    q = JobQueue(BuildRunner(settings, build_fn))
    try:
        req = BuildRequest(
            route_id="0" * 16, route=route, name="Forks Test", track_id="dm_forks_test",
            trees=False, buildings=False,
        )  # fmt: skip
        snap = wait(q.submit(req), timeout=120)
    finally:
        q.close(5)
    assert snap["state"] == "done", snap
    fractions = [f for _, f in steps]
    assert fractions == sorted(fractions) and fractions[0] == 0.0 and fractions[-1] == 1.0
    names = [s for s, _ in steps]
    assert names[0] == "Getting elevation and shaping the road" and names[-1] == "Track built"
    assert "Placing buildings" not in names and "Writing the track files" in names
    assert all(n in snap["log"] for n in names)
    result = snap["result"]
    installed = ac / "content" / "tracks" / "dm_forks_test"
    assert result["installed_to"] == str(installed)
    assert (installed / MARKER_FILE).is_file() and (installed / "dm_forks_test.kn5").is_file()
    assert result["track_dir"] == str(settings.output_dir / "dm_forks_test")
    assert result["length_m"] == pytest.approx(route.length_m, rel=0.01)  # smoothed A->B
    assert 0.0 < result["kn5_mb"] < 50.0
    assert result["trees"] == 0 and result["buildings"] == 0
    ui = json.loads((installed / "ui" / "ui_track.json").read_text(encoding="utf-8"))
    assert ui["name"] == "Forks Test"
