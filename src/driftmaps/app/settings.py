"""Where the local app keeps its files, its settings file, and where Assetto Corsa is.

- **App data folder** (`app_data_dir`): `%LOCALAPPDATA%\\AssettoCorsaMapGenerator` on Windows,
  `$XDG_DATA_HOME/AssettoCorsaMapGenerator` (default `~/.local/share/...`) elsewhere, or
  `$DRIFTMAPS_APP_DIR` when set. It holds `settings.json`, routed roads (`routes/`), built
  tracks (`tracks/`) and, unless `$DRIFTMAPS_CACHE_DIR` is set, the download cache (`cache/`).
- **Assetto Corsa**: the folder saved in `settings.json` (`ac_root`, set from the app) wins
  while it is still a valid AC folder; otherwise `config.find_ac_root()` searches (env var,
  registry, Steam libraries, common folders). The search result is remembered for the session.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from driftmaps import config

__all__ = [
    "APP_DIR_NAME",
    "APP_NAME",
    "ENV_APP_DIR",
    "Settings",
    "SettingsError",
    "app_data_dir",
    "app_version",
    "normalize_ac_root",
]

APP_NAME = "Assetto Corsa Map Generator"
APP_DIR_NAME = "AssettoCorsaMapGenerator"
ENV_APP_DIR = "DRIFTMAPS_APP_DIR"
ENV_CACHE_DIR = "DRIFTMAPS_CACHE_DIR"
SETTINGS_FILE = "settings.json"


class SettingsError(ValueError):
    """A setting the user gave can't be used; the message is shown to them as is."""


def app_version() -> str:
    """The installed driftmaps version (a frozen app may have no metadata: "0.1.0")."""
    try:
        from importlib.metadata import version

        return version("driftmaps")
    except Exception:  # PackageNotFoundError, or no metadata at all when frozen
        return "0.1.0"


def app_data_dir(env: dict[str, str] | None = None) -> Path:
    """The app's own folder (not created here); see the module docstring."""
    env = os.environ if env is None else env
    override = env.get(ENV_APP_DIR, "").strip()
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = env.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_DIR_NAME
    base = env.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_DIR_NAME


def normalize_ac_root(text: str) -> Path:
    """The AC root for a folder the user picked: the root itself, or a folder inside it
    (`content`, `content/tracks`), or `acs.exe` itself. Raises `SettingsError` with a plain
    explanation if it isn't Assetto Corsa."""
    text = (text or "").strip().strip('"')
    if not text:
        raise SettingsError("Pick the Assetto Corsa folder (the one with acs.exe in it).")
    path = Path(text).expanduser()
    candidates = [path, *list(path.parents)[:2]]
    if path.suffix.lower() == ".exe":
        candidates = [path.parent]
    for c in candidates:
        if config.is_ac_root(c):
            return Path(os.path.abspath(c))
    if not path.exists():
        raise SettingsError(f"The folder {text} doesn't exist.")
    raise SettingsError(
        f"{text} doesn't look like Assetto Corsa: it should contain acs.exe and a "
        "content\\tracks folder (usually ...\\steamapps\\common\\assettocorsa)."
    )


class Settings:
    """The app's folders and settings file. Thread-safe; one instance per running app.

    `discover` finds AC when no valid folder is saved (default `config.find_ac_root`).
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        discover: Callable[[], Path | None] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._env = os.environ if env is None else env
        self.data_dir = Path(data_dir) if data_dir is not None else app_data_dir(self._env)
        self._discover = discover or config.find_ac_root
        self._lock = threading.Lock()
        self._found: Path | None = None
        self._searched = False
        self.ac_root_override: str | None = None
        self._load()

    # -- folders ------------------------------------------------------------------------------

    @property
    def config_path(self) -> Path:
        return self.data_dir / SETTINGS_FILE

    @property
    def cache_dir(self) -> Path:
        """Download cache: `$DRIFTMAPS_CACHE_DIR`, else `<app data>/cache`."""
        env = self._env.get(ENV_CACHE_DIR, "").strip()
        return Path(env).expanduser() if env else self.data_dir / "cache"

    @property
    def output_dir(self) -> Path:
        """Where tracks are built (and kept, installed or not)."""
        return self.data_dir / "tracks"

    @property
    def routes_dir(self) -> Path:
        return self.data_dir / "routes"

    # -- settings file --------------------------------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict) and isinstance(data.get("ac_root"), str) and data["ac_root"]:
            self.ac_root_override = data["ac_root"]

    def _save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.config_path.with_suffix(".tmp")
        data = {"ac_root": self.ac_root_override}
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.config_path)

    # -- Assetto Corsa ------------------------------------------------------------------------

    def ac_root(self) -> Path | None:
        """The AC folder: the saved one if still valid, else the (remembered) search result."""
        with self._lock:
            if self.ac_root_override and config.is_ac_root(Path(self.ac_root_override)):
                return Path(self.ac_root_override)
            if not self._searched:
                try:
                    self._found = self._discover()
                except Exception:  # a broken registry or drive must not break the app
                    self._found = None
                self._searched = True
            return self._found

    def tracks_dir(self) -> Path | None:
        root = self.ac_root()
        return config.tracks_dir(root) if root is not None else None

    def set_ac_root(self, text: str | None) -> Path | None:
        """Save the AC folder the user picked (validated); empty/None forgets it and searches
        again. Returns the AC folder now in use."""
        if text is None or not str(text).strip():
            with self._lock:
                self.ac_root_override = None
                self._searched = False
                self._save()
            return self.ac_root()
        root = normalize_ac_root(str(text))
        with self._lock:
            self.ac_root_override = str(root)
            self._save()
        return root

    def status(self) -> dict[str, object]:
        """The `/api/status` body."""
        root = self.ac_root()
        return {
            "app": APP_NAME,
            "version": app_version(),
            "ac_found": root is not None,
            "ac_root": str(root) if root is not None else None,
            "tracks_dir": str(config.tracks_dir(root)) if root is not None else None,
        }
