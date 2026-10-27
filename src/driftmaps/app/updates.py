"""New-version check and one-click update of the installed app.

- **Check** (`Updater.check`): one HTTPS GET of GitHub's public "latest release" of
  `REPO` (no account, no cookies, nothing about this PC beyond the request itself), 3 s
  timeout, remembered for 24 h in `<app data>/update-check.json`; any failure is silent. The app
  starts one in the background unless the `check_updates` setting is off (released versions
  only); "Check for updates" in the settings panel asks again.
- **Install** (`Updater.install`, only when the user clicks Install, only for the installed
  app): downloads the release's Setup .exe and SHA256SUMS.txt (HTTPS, that repository's
  release assets only, redirects only to GitHub's asset hosts) into `<app data>/temp/updates`,
  deletes the installer and stops if its SHA-256 isn't the published one, else starts it
  silently (`INSTALLER_ARGS`: it waits for this app to quit, installs over it and starts the
  new version) and shuts the app down. Refused while a track builds. A portable copy (no
  uninstaller next to the exe) only links to the release page.
- Every step is logged, and also appended to `<app data>/updates.log` (app.log starts afresh
  each run).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "REPO",
    "UpdateError",
    "Updater",
    "asset_url_allowed",
    "installed_app_dir",
    "is_newer",
    "parse_sums",
    "parse_version",
]

log = logging.getLogger(__name__)

REPO = "cooldude-7/MapsAssetoCorsa"
LATEST_API = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
USER_AGENT = "AssettoCorsaMapGenerator-update-check"
CHECK_TIMEOUT_S = 3.0
DOWNLOAD_TIMEOUT_S = 30.0
CACHE_TTL_S = 24 * 3600.0
CACHE_FILE = "update-check.json"
LOG_FILE = "updates.log"
SUMS_NAME = "SHA256SUMS.txt"
SETUP_RE = re.compile(r"^AssettoCorsaMapGenerator-Setup-[0-9A-Za-z._-]+\.exe$")
# Where github.com sends release-asset downloads (signed, short-lived links).
ASSET_HOSTS = frozenset({"objects.githubusercontent.com", "release-assets.githubusercontent.com"})
MAX_API_BYTES = 2_000_000
MAX_SUMS_BYTES = 65_536
MAX_SETUP_BYTES = 2_000_000_000
# Inno Setup: no wizard, no message boxes, per-user, never reboot; `/relaunch=1` makes
# packaging/installer.iss start the app again when it is done (and `/port=N` keeps its port).
INSTALLER_ARGS = ("/VERYSILENT", "/SUPPRESSMSGBOXES", "/CURRENTUSER", "/NORESTART", "/relaunch=1")
SHUTDOWN_DELAY_S = 1.5  # lets the page see "installing" before the app goes away
_VERSION_RE = re.compile(r"^v?(\d{1,4})\.(\d{1,4})(?:\.(\d{1,6}))?$")
_BUSY_PHASES = ("downloading", "verifying", "installing")


class UpdateError(Exception):
    """The update can't go ahead; the message is shown to the user as is."""


# ---------------------------------------------------------------------------------------------
# Versions, URLs, checksums
# ---------------------------------------------------------------------------------------------


def parse_version(text: object) -> tuple[int, int, int] | None:
    """(major, minor, patch) of "v1.2.3" / "1.2" / "1.2.3"; None for anything else ("dev")."""
    m = _VERSION_RE.match(str(text or "").strip())
    return (int(m[1]), int(m[2]), int(m[3] or 0)) if m else None


def is_newer(latest: object, current: object) -> bool:
    """True if release `latest` is a later version than `current` (both must be versions)."""
    a, b = parse_version(latest), parse_version(current)
    return a is not None and b is not None and a > b


def _https_parts(url: str) -> urllib.parse.SplitResult | None:
    try:
        parts = urllib.parse.urlsplit(str(url))
        port = parts.port
    except ValueError:
        return None
    if parts.scheme != "https" or parts.username or parts.password or port not in (None, 443):
        return None
    path = urllib.parse.unquote(parts.path)
    if "\\" in path or any(seg in ("..", ".") for seg in path.split("/")):
        return None
    return parts


def asset_url_allowed(url: str, *, redirect: bool = False) -> bool:
    """True for an HTTPS download of one of `REPO`'s release assets on github.com; after a
    redirect, also GitHub's asset storage hosts (`ASSET_HOSTS`). Nothing else."""
    parts = _https_parts(url)
    if parts is None:
        return False
    host = (parts.hostname or "").lower()
    if host == "github.com":
        return parts.path.lower().startswith(f"/{REPO.lower()}/releases/download/")
    return redirect and host in ASSET_HOSTS


def api_url_allowed(url: str) -> bool:
    """The release API on api.github.com over HTTPS (a renamed repository redirects there)."""
    parts = _https_parts(url)
    return parts is not None and (parts.hostname or "").lower() == "api.github.com"


def release_page(url: object) -> str:
    """`url` if it is a page under `REPO`'s releases on github.com, else the latest release."""
    parts = _https_parts(str(url or ""))
    if parts is not None and (parts.hostname or "").lower() == "github.com":
        if parts.path.lower().startswith(f"/{REPO.lower()}/releases/"):
            return str(url)
    return RELEASES_PAGE


def parse_sums(text: str) -> dict[str, str]:
    """{file name: sha256 hex} from `sha256sum` output; a name listed twice differently is
    left out."""
    out: dict[str, str] = {}
    bad: set[str] = set()
    for line in text.splitlines():
        m = re.match(r"^([0-9a-fA-F]{64})\s+\*?(\S.*?)\s*$", line.strip())
        if not m:
            continue
        name, digest = m[2], m[1].lower()
        if out.setdefault(name, digest) != digest:
            bad.add(name)
    return {k: v for k, v in out.items() if k not in bad}


def _clean_release(data: object) -> dict | None:
    """The fields the app uses from a release (GitHub's JSON or our cache), checked."""
    if not isinstance(data, dict) or data.get("draft") or data.get("prerelease"):
        return None
    version = str(data.get("tag_name") or data.get("version") or "").strip()
    if parse_version(version) is None:
        return None
    setup = sums = None
    assets = data.get("assets")
    if isinstance(assets, list):  # GitHub's answer
        for a in assets:
            if not isinstance(a, dict):
                continue
            name, url = str(a.get("name") or ""), str(a.get("browser_download_url") or "")
            try:
                size = max(0, int(a.get("size") or 0))
            except (TypeError, ValueError):
                size = 0
            if not asset_url_allowed(url):
                continue
            if setup is None and SETUP_RE.match(name):
                setup = {"name": name, "url": url, "size": size}
            elif name == SUMS_NAME:
                sums = {"name": name, "url": url}
    else:  # the cache
        s, c = data.get("setup"), data.get("sums")
        if isinstance(s, dict) and SETUP_RE.match(str(s.get("name") or "")):
            if asset_url_allowed(str(s.get("url") or "")):
                setup = {"name": s["name"], "url": s["url"], "size": int(s.get("size") or 0)}
        if isinstance(c, dict) and asset_url_allowed(str(c.get("url") or "")):
            sums = {"name": SUMS_NAME, "url": c["url"]}
    page = release_page(data.get("html_url") or data.get("page"))
    return {"version": version, "page": page, "setup": setup, "sums": sums}


# ---------------------------------------------------------------------------------------------
# HTTP (allow-listed redirects) and launching
# ---------------------------------------------------------------------------------------------


class AllowListRedirects(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to a URL `allowed` accepts; anything else is an UpdateError."""

    max_redirections = 5

    def __init__(self, allowed: Callable[[str], bool]) -> None:
        self.allowed = allowed

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        if not self.allowed(newurl):
            host = urllib.parse.urlsplit(newurl).hostname or newurl[:60]
            raise UpdateError(f"The download was redirected to {host}, which isn't GitHub.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_url(url: str, timeout: float, kind: str) -> Any:
    """GET `url` ("api": the release API; "asset": a release download), following only
    allow-listed redirects. Sends nothing but a User-Agent (GitHub's API requires one)."""
    allowed = api_url_allowed if kind == "api" else (lambda u: asset_url_allowed(u, redirect=True))
    if not (api_url_allowed(url) if kind == "api" else asset_url_allowed(url)):
        raise UpdateError("Refused to download from an address that isn't this app's GitHub.")
    accept = "application/vnd.github+json" if kind == "api" else "application/octet-stream"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    opener = urllib.request.build_opener(AllowListRedirects(allowed))
    return opener.open(req, timeout=timeout)


def launch_detached(args: list[str]) -> None:
    """Start `args` on its own (no console, not a child that dies with this app)."""
    kw: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
        "cwd": str(Path(args[0]).parent),
    }
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        try:  # out of a job object that might close with this app, when that is allowed
            subprocess.Popen(args, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kw)
            return
        except OSError:
            pass
        subprocess.Popen(args, creationflags=flags, **kw)
    else:
        subprocess.Popen(args, start_new_session=True, **kw)


def installed_app_dir() -> Path | None:
    """The app's folder if the Setup program installed it (its uninstaller unins000.exe sits
    next to the exe); None for a portable copy or a development checkout."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return None
    folder = Path(sys.executable).resolve().parent
    return folder if any(folder.glob("unins*.exe")) else None


# ---------------------------------------------------------------------------------------------
# The updater
# ---------------------------------------------------------------------------------------------


class Updater:
    """Update state for one running app. `open_url`, `launch`, `installed` and `sleep` are
    the network, process, install-folder and delay hooks (tests replace them)."""

    def __init__(
        self,
        data_dir: Path | str,
        current: str,
        *,
        enabled: Callable[[], bool] = lambda: True,
        open_url: Callable[[str, float, str], Any] = open_url,
        launch: Callable[[list[str]], None] = launch_detached,
        installed: Callable[[], Path | None] = installed_app_dir,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.current = str(current or "dev")
        self._enabled = enabled
        self._open = open_url
        self._launch = launch
        self._installed = installed() is not None
        self._now = now
        self._sleep = sleep
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._checked_at: float | None = None
        self._check_error: str | None = None
        self._phase = "idle"
        self._progress: float | None = None
        self._error: str | None = None
        self._check_thread: threading.Thread | None = None
        self._install_thread: threading.Thread | None = None

    @property
    def folder(self) -> Path:
        return self.data_dir / "temp" / "updates"

    def record(self, message: str) -> None:
        """Log `message` and append it to `<app data>/updates.log`."""
        log.info("update: %s", message)
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
            with open(self.data_dir / LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{stamp}  {message}\n")
        except OSError:
            pass

    # -- state ----------------------------------------------------------------------------------

    def _set(self, **kw: Any) -> None:
        with self._lock:
            for key, value in kw.items():
                setattr(self, f"_{key}", value)

    def busy(self) -> bool:
        """True while an update downloads or installs (builds are refused then)."""
        with self._lock:
            return self._phase in _BUSY_PHASES

    def state(self) -> dict:
        """The `/api/update` body."""
        with self._lock:
            latest = self._latest
            available = bool(latest and is_newer(latest["version"], self.current))
            return {
                "enabled": bool(self._enabled()),
                "current": self.current,
                "dev": parse_version(self.current) is None,
                "checked": self._checked_at is not None,
                "latest": latest["version"] if latest else None,
                "available": available,
                "page": latest["page"] if latest else RELEASES_PAGE,
                "installed": self._installed,
                "can_install": bool(
                    available and self._installed and latest["setup"] and latest["sums"]
                ),
                "phase": self._phase,
                "progress": self._progress,
                "error": self._error,
                "check_error": self._check_error,
            }

    # -- check ----------------------------------------------------------------------------------

    def _read_cache(self) -> dict | None:
        try:
            data = json.loads((self.data_dir / CACHE_FILE).read_text(encoding="utf-8"))
            age = self._now() - float(data["checked_at"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if not 0 <= age < CACHE_TTL_S:
            return None
        release = _clean_release(data.get("release"))
        return {"release": release, "checked_at": float(data["checked_at"])} if release else None

    def _write_cache(self, release: dict, checked_at: float) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            path = self.data_dir / CACHE_FILE
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"checked_at": checked_at, "release": release}), "utf-8")
            tmp.replace(path)
        except OSError:
            pass

    def check(self, *, force: bool = False) -> dict:
        """Ask GitHub for the latest release (unless the 24 h cache answers); never raises."""
        cached = None if force else self._read_cache()
        if cached is not None:
            self._set(latest=cached["release"], checked_at=cached["checked_at"], check_error=None)
            return self.state()
        try:
            with self._open(LATEST_API, CHECK_TIMEOUT_S, "api") as resp:
                data = json.loads(resp.read(MAX_API_BYTES).decode("utf-8"))
            release = _clean_release(data)
            if release is None:
                raise UpdateError("the latest release has no version number")
        except Exception as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                message = "No version has been published on GitHub yet."
            else:
                message = "Couldn't reach GitHub to check for a new version."
            log.info("update check failed: %s", exc)
            self._set(check_error=message)
            return self.state()
        now = self._now()
        self._set(latest=release, checked_at=now, check_error=None)
        self._write_cache(release, now)
        log.info("update check: latest %s, running %s", release["version"], self.current)
        return self.state()

    def start_background_check(self) -> None:
        """The start-up check (off when the setting is off, or in a development build)."""
        if not self._enabled() or parse_version(self.current) is None:
            return
        t = threading.Thread(target=self.check, name="acmg-update-check", daemon=True)
        self._check_thread = t
        t.start()

    def wait_for_check(self, timeout: float) -> None:
        t = self._check_thread
        if t is not None:
            t.join(timeout)

    # -- install --------------------------------------------------------------------------------

    def install(
        self,
        *,
        busy: Callable[[], str | None],
        shutdown: Callable[[], None],
        port: int | None = None,
    ) -> dict:
        """Start installing the newer release (the user clicked Install). Raises UpdateError
        when it can't; else returns the state while a background thread does the work."""
        if self.busy():
            return self.state()
        st = self.state()
        if st["dev"]:
            raise UpdateError("This is a development build: get a released version instead.")
        if not st["available"]:
            raise UpdateError("You already have the newest version.")
        if not st["installed"]:
            raise UpdateError(
                "This copy wasn't installed with the Setup program, so it can't update itself. "
                "Download the new version from the release page."
            )
        if not st["can_install"]:
            raise UpdateError(
                "The new version's installer isn't on GitHub yet. Try again in a few minutes."
            )
        why = busy()
        if why:
            raise UpdateError(why)
        with self._lock:
            release = self._latest
            self._phase, self._progress, self._error = "downloading", 0.0, None
        assert release is not None
        self.record(f"installing {release['version']} (running {self.current}) - user clicked")
        t = threading.Thread(
            target=self._install, args=(release, busy, shutdown, port), name="acmg-update"
        )
        t.daemon = True
        self._install_thread = t
        t.start()
        return self.state()

    def join(self, timeout: float | None = None) -> None:
        t = self._install_thread
        if t is not None:
            t.join(timeout)

    def _install(
        self,
        release: dict,
        busy: Callable[[], str | None],
        shutdown: Callable[[], None],
        port: int | None,
    ) -> None:
        setup = release["setup"]
        dest = self.folder / setup["name"]
        part = dest.with_name(dest.name + ".part")
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            for old in self.folder.glob("*.exe*"):  # earlier updates' installers
                old.unlink(missing_ok=True)
            with self._open(release["sums"]["url"], DOWNLOAD_TIMEOUT_S, "asset") as resp:
                sums = parse_sums(resp.read(MAX_SUMS_BYTES).decode("utf-8", "replace"))
            expected = sums.get(setup["name"])
            if not expected:
                raise UpdateError(f"{SUMS_NAME} doesn't list the installer, so it wasn't run.")
            digest = self._download(setup["url"], part, setup.get("size") or 0)
            self._set(phase="verifying")
            if digest != expected:
                self.record(f"checksum MISMATCH for {setup['name']}: got {digest}, "
                            f"published {expected}; deleted it")  # fmt: skip
                raise UpdateError(
                    "The downloaded update doesn't match its published checksum, so it was "
                    "deleted and not run. Try again later."
                )
            part.replace(dest)
            self.record(f"downloaded {setup['name']}, SHA-256 {digest} matches {SUMS_NAME}")
            why = busy()
            if why:
                raise UpdateError(why)
            args = [str(dest), *INSTALLER_ARGS, f"/LOG={self.folder / 'install.log'}"]
            if port:
                args.append(f"/port={int(port)}")
            self._set(phase="installing", progress=1.0)
            self.record("starting the installer: " + " ".join(args[1:]))
            self._launch(args)
        except Exception as exc:
            for f in (part, dest):
                f.unlink(missing_ok=True)
            message = str(exc) if isinstance(exc, UpdateError) else (
                f"The update didn't work ({type(exc).__name__}: {exc}). Try again later."
            )  # fmt: skip
            self.record(f"update failed: {message}")
            self._set(phase="error", error=message, progress=None)
            return
        self._sleep(SHUTDOWN_DELAY_S)
        self.record("closing the app so the installer can replace it")
        shutdown()

    def _download(self, url: str, part: Path, size: int) -> str:
        """Stream `url` into `part`; returns its SHA-256 (hex)."""
        h = hashlib.sha256()
        done = 0
        with self._open(url, DOWNLOAD_TIMEOUT_S, "asset") as resp, open(part, "wb") as f:
            headers = getattr(resp, "headers", None)
            total = size or int((headers.get("Content-Length") if headers else 0) or 0)
            while chunk := resp.read(1 << 20):
                done += len(chunk)
                if done > MAX_SETUP_BYTES:
                    raise UpdateError("The download is far bigger than the app; stopped.")
                h.update(chunk)
                f.write(chunk)
                if total:
                    self._set(progress=min(done / total, 1.0))
        if size and done != size:
            raise UpdateError("The download stopped before the end. Try again.")
        return h.hexdigest()
