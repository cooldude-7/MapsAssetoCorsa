"""The new-version check and one-click updater (driftmaps.app.updates), with fake HTTP."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from driftmaps.app import updates as up
from driftmaps.app.server import ApiError, App
from driftmaps.app.settings import Settings
from driftmaps.geo import mapillary

REPO = up.REPO
SETUP = b"MZ fake installer bytes"


def release_json(version: str = "v0.5.0", setup: bytes = SETUP) -> dict:
    base = f"https://github.com/{REPO}/releases/download/{version}/"
    name = f"AssettoCorsaMapGenerator-Setup-{version.lstrip('v')}.exe"
    return {
        "tag_name": version,
        "html_url": f"https://github.com/{REPO}/releases/tag/{version}",
        "draft": False,
        "prerelease": False,
        "assets": [
            {"name": "evil.exe", "browser_download_url": "https://evil.example/evil.exe"},
            {"name": name, "browser_download_url": base + name, "size": len(setup)},
            {"name": "SHA256SUMS.txt", "browser_download_url": base + "SHA256SUMS.txt"},
        ],
    }


class Resp(io.BytesIO):
    headers: dict = {}


class FakeGitHub:
    """`open_url` stand-in: the release API plus its assets, recording each URL."""

    def __init__(self, release: dict | None = None, sums: str | None = None, setup=SETUP):
        self.release = release or release_json()
        name = self.release["assets"][1]["name"]
        self.sums = sums if sums is not None else f"{hashlib.sha256(setup).hexdigest()}  {name}\n"
        self.setup = setup
        self.urls: list[str] = []

    def __call__(self, url: str, timeout: float, kind: str):
        self.urls.append(url)
        if url == up.LATEST_API:
            return Resp(json.dumps(self.release).encode())
        if url.endswith("SHA256SUMS.txt"):
            return Resp(self.sums.encode())
        if url.endswith(".exe"):
            return Resp(self.setup)
        raise AssertionError(url)


def make_updater(tmp_path: Path, gh: FakeGitHub, *, current="v0.4.0", installed=True, **kw):
    launched: list[list[str]] = []
    kw.setdefault("open_url", gh)
    u = up.Updater(
        tmp_path,
        current,
        launch=launched.append,
        installed=lambda: tmp_path if installed else None,
        sleep=lambda s: None,
        **kw,
    )
    return u, launched


# -- versions and addresses ---------------------------------------------------------------------


def test_version_compare():
    assert up.parse_version("v1.2.3") == (1, 2, 3) and up.parse_version("0.4") == (0, 4, 0)
    for bad in ("dev", "", None, "Update 3", "v1.2.3-rc1", "v1.2.3.4", "1.x"):
        assert up.parse_version(bad) is None, bad
    assert up.is_newer("v0.10.0", "v0.9.9") and up.is_newer("v1.0", "v0.99.99")
    assert not up.is_newer("v0.4.0", "v0.4.0") and not up.is_newer("v0.3.9", "0.4.0")
    assert not up.is_newer("v9.9.9", "dev") and not up.is_newer("dev", "v0.1.0")


def test_host_allow_list():
    ok = f"https://github.com/{REPO}/releases/download/v0.5.0/AssettoCorsaMapGenerator-Setup-0.5.0.exe"
    assert up.asset_url_allowed(ok)
    assert up.asset_url_allowed(
        ok.replace("cooldude-7/MapsAssetoCorsa", "COOLDUDE-7/mapsassetocorsa")
    )
    signed = "https://objects.githubusercontent.com/github-production-release-asset/1?sig=x"
    assert up.asset_url_allowed(signed, redirect=True) and not up.asset_url_allowed(signed)
    assert up.asset_url_allowed("https://release-assets.githubusercontent.com/x", redirect=True)
    for bad in (
        ok.replace("https://", "http://"),  # HTTPS only
        "https://github.com/someone/else/releases/download/v1/x.exe",  # another repository
        f"https://github.com/{REPO}/archive/main.zip",  # not a release asset
        f"https://github.com/{REPO}/releases/download/../../../evil/x/releases/download/a.exe",
        f"https://github.com/{REPO}/releases/download/%2e%2e/x.exe",
        "https://github.com.evil.example/" + REPO + "/releases/download/v1/x.exe",
        f"https://user:pw@github.com/{REPO}/releases/download/v1/x.exe",
        f"https://github.com:8443/{REPO}/releases/download/v1/x.exe",
        "https://evil.example/x.exe",
        "file:///C:/Windows/notepad.exe",
    ):
        assert not up.asset_url_allowed(bad, redirect=True), bad
    assert up.api_url_allowed(up.LATEST_API) and not up.api_url_allowed("http://api.github.com/x")
    assert up.release_page("https://evil.example/") == up.RELEASES_PAGE
    page = f"https://github.com/{REPO}/releases/tag/v0.5.0"
    assert up.release_page(page) == page


def test_redirects_only_to_allowed_hosts():
    handler = up.AllowListRedirects(lambda u: up.asset_url_allowed(u, redirect=True))
    req = urllib.request.Request(f"https://github.com/{REPO}/releases/download/v1/a.exe")
    good = "https://objects.githubusercontent.com/a?sig=1"
    assert handler.redirect_request(req, None, 302, "Found", {}, good).full_url == good
    for bad in ("https://evil.example/a.exe", "http://objects.githubusercontent.com/a"):
        with pytest.raises(up.UpdateError, match="isn't GitHub"):
            handler.redirect_request(req, None, 302, "Found", {}, bad)
    with pytest.raises(up.UpdateError):  # never even asked for
        up.open_url("https://evil.example/a.exe", 1.0, "asset")
    with pytest.raises(up.UpdateError):
        up.open_url("http://api.github.com/repos/x/y/releases/latest", 1.0, "api")


def test_parse_sums():
    a, b = "a" * 64, "B" * 64
    text = f"{a}  one.exe\n{b} *two.zip\nnot a line\n{a}  dup\n{b}  dup\n"
    assert up.parse_sums(text) == {"one.exe": a, "two.zip": b.lower()}


# -- the check ----------------------------------------------------------------------------------


def test_check_finds_a_newer_version_and_caches_it_for_a_day(tmp_path):
    gh = FakeGitHub()
    clock = [1_000_000.0]
    u, _ = make_updater(tmp_path, gh, now=lambda: clock[0])
    st = u.check()
    assert st["available"] and st["latest"] == "v0.5.0" and st["can_install"]
    assert st["page"].endswith("/releases/tag/v0.5.0") and gh.urls == [up.LATEST_API]
    # A new run within 24 h answers from the cache; after that it asks again.
    clock[0] += 23 * 3600
    u2, _ = make_updater(tmp_path, gh, now=lambda: clock[0])
    assert u2.check()["available"] and gh.urls == [up.LATEST_API]
    clock[0] += 2 * 3600
    u2.check()
    assert gh.urls == [up.LATEST_API] * 2
    # The same version, or a development build: nothing to offer.
    assert not make_updater(tmp_path, gh, current="v0.5.0")[0].check()["available"]
    dev = make_updater(tmp_path, gh, current="dev")[0].check()
    assert dev["dev"] and not dev["available"]


def test_check_fails_silently_and_respects_the_setting(tmp_path):
    def offline(url, timeout, kind):
        raise urllib.error.URLError("no internet")

    u, _ = make_updater(tmp_path, FakeGitHub(), open_url=offline)
    st = u.check()
    assert not st["available"] and not st["checked"] and "GitHub" in st["check_error"]
    gh = FakeGitHub()
    off, _ = make_updater(tmp_path / "off", gh, enabled=lambda: False)
    off.start_background_check()
    off.wait_for_check(1)
    assert gh.urls == [] and not off.state()["enabled"]


# -- installing ---------------------------------------------------------------------------------


def run_install(u: up.Updater, busy=lambda: None):
    closed: list[bool] = []
    u.install(busy=busy, shutdown=lambda: closed.append(True), port=8766)
    u.join(5)
    return closed


def test_install_verifies_runs_the_installer_and_closes_the_app(tmp_path):
    u, launched = make_updater(tmp_path, FakeGitHub())
    u.check()
    closed = run_install(u)
    (args,) = launched
    exe = Path(args[0])
    assert exe.parent == tmp_path / "temp" / "updates" and exe.read_bytes() == SETUP
    for flag in ("/VERYSILENT", "/SUPPRESSMSGBOXES", "/CURRENTUSER", "/NORESTART", "/relaunch=1"):
        assert flag in args
    assert "/port=8766" in args and closed == [True] and u.state()["phase"] == "installing"
    assert "matches SHA256SUMS.txt" in (tmp_path / "updates.log").read_text(encoding="utf-8")


def test_checksum_mismatch_deletes_the_file_and_runs_nothing(tmp_path):
    gh = FakeGitHub(sums=f"{'0' * 64}  AssettoCorsaMapGenerator-Setup-0.5.0.exe\n")
    u, launched = make_updater(tmp_path, gh)
    u.check()
    closed = run_install(u)
    st = u.state()
    assert st["phase"] == "error" and "checksum" in st["error"]
    assert launched == [] and closed == []
    assert list((tmp_path / "temp" / "updates").iterdir()) == []
    assert "MISMATCH" in (tmp_path / "updates.log").read_text(encoding="utf-8")
    # Not listed in SHA256SUMS.txt at all: refused too.
    u2, launched2 = make_updater(tmp_path / "b", FakeGitHub(sums=""))
    u2.check()
    run_install(u2)
    assert u2.state()["phase"] == "error" and launched2 == []


def test_portable_copy_and_dev_build_dont_self_update(tmp_path):
    u, _ = make_updater(tmp_path, FakeGitHub(), installed=False)
    st = u.check()
    assert st["available"] and not st["can_install"] and not st["installed"]
    with pytest.raises(up.UpdateError, match="Setup program"):
        u.install(busy=lambda: None, shutdown=lambda: None)
    dev, _ = make_updater(tmp_path / "d", FakeGitHub(), current="dev")
    dev.check()
    with pytest.raises(up.UpdateError, match="development"):
        dev.install(busy=lambda: None, shutdown=lambda: None)


# -- through the app ----------------------------------------------------------------------------


@pytest.fixture
def app(tmp_path):
    a = App(Settings(tmp_path / "data", discover=lambda: None, env={}))
    yield a
    a.close()


def test_app_refuses_to_update_while_a_build_runs(app, tmp_path, monkeypatch):
    gh = FakeGitHub()
    app.updater, launched = make_updater(tmp_path / "u", gh)
    app.updater.check()
    app.shutdown_hook = lambda: None
    monkeypatch.setattr(app.jobs, "active", lambda: [object()])
    with pytest.raises(ApiError) as exc:
        app.update_install()
    assert exc.value.status == 409 and "being built" in exc.value.message
    assert launched == [] and not any(url.endswith(".exe") for url in gh.urls)
    # And while an update runs, builds are refused.
    monkeypatch.setattr(app.updater, "busy", lambda: True)
    with pytest.raises(ApiError) as exc:
        app.build({})
    assert exc.value.status == 409 and "update" in exc.value.message


def test_settings_switches_and_status_never_show_a_token(app, monkeypatch):
    builtin = "MLY|1234|builtinbuiltinbuiltinbuiltin1308"
    monkeypatch.setattr(mapillary, "DEFAULT_CLIENT_TOKEN", builtin)
    st = app.status()
    assert st["mapillary_source"] == "built-in" and st["mapillary_token_hint"] == "built-in"
    assert st["mapillary_token_set"] is False and builtin not in json.dumps(st)
    assert app.settings.mapillary_token() == builtin  # builds get it
    st = app.set_settings({"use_mapillary": False, "check_updates": False})
    assert st["mapillary_source"] == "off" and app.settings.mapillary_token() is None
    assert st["check_updates"] is False and st["update"]["enabled"] is False
    saved = json.loads(app.settings.config_path.read_text(encoding="utf-8"))
    assert saved["use_mapillary"] is False and saved["check_updates"] is False
    again = Settings(app.settings.data_dir, discover=lambda: None, env={})
    assert not again.use_mapillary and not again.check_updates
    st = app.set_settings({"use_mapillary": True, "mapillary_token": "MLY|own0000000000001308"})
    assert st["mapillary_source"] == "own" and st["mapillary_token_hint"] == "MLY|…1308"
    assert "own0000" not in json.dumps(st) and builtin not in json.dumps(st)
    with pytest.raises(ApiError):
        app.set_settings({"check_updates": "yes"})
