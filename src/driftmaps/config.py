"""Find the user's Assetto Corsa install (the folder holding `content/tracks`).

Search order (docs/research/geo.md §5, "Finding the AC install"); the first folder that passes
`is_ac_root` wins:

1. The `DRIFTMAPS_AC_ROOT` environment variable (an invalid value is skipped with a warning).
2. On Windows, AC's own uninstall registry entry ("Steam App 244210" InstallLocation).
3. Steam: the Steam folder from the Windows registry (HKCU SteamPath, then HKLM InstallPath) or
   the usual per-platform locations (incl. Linux/Proton, Flatpak, Snap and WSL), then every
   library listed in its `steamapps/libraryfolders.vdf`, then
   `<library>/steamapps/appmanifest_244210.acf` -> `installdir` ->
   `<library>/steamapps/common/<installdir>` (normally `assettocorsa`). AC's Steam app id is
   244210 (VERIFIED: store page, and AcTools `CommonAcConsts.AppId`).
4. Common install folders on each drive (`<drive>:/SteamLibrary/...` and the like).

Content Manager's `_ac_root` value is only stored in its compressed `Values.data`, so it isn't
read here (format undocumented).

Steam's `.vdf`/`.acf` files are Valve KeyValues text; `parse_vdf` is a small reader for it.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

__all__ = [
    "AC_APP_ID",
    "ENV_AC_ROOT",
    "ac_roots_in_library",
    "candidate_ac_roots",
    "default_steam_roots",
    "find_ac_root",
    "is_ac_root",
    "parse_vdf",
    "steam_library_folders",
    "tracks_dir",
]

AC_APP_ID = "244210"
ENV_AC_ROOT = "DRIFTMAPS_AC_ROOT"
AC_INSTALL_DIR = "assettocorsa"
# acs.exe is the game; AssettoCorsa.exe the Kunos launcher. Either one marks the root.
AC_EXECUTABLES = ("acs.exe", "AssettoCorsa.exe")

VdfDict = dict[str, "str | VdfDict"]


def tracks_dir(ac_root: Path) -> Path:
    """`<ac_root>/content/tracks`, where track folders go."""
    return Path(ac_root) / "content" / "tracks"


def is_ac_root(path: Path) -> bool:
    """True if `path` looks like an AC install: `content/tracks` plus acs.exe/AssettoCorsa.exe."""
    try:
        path = Path(path)
        return tracks_dir(path).is_dir() and any((path / exe).is_file() for exe in AC_EXECUTABLES)
    except OSError:
        return False


# ---------------------------------------------------------------------------------------------
# Valve KeyValues ("VDF") text
# ---------------------------------------------------------------------------------------------

_ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", '"': '"'}


def _vdf_tokens(text: str) -> Iterator[tuple[str, str]]:
    """Yield (kind, value): kind "str" for quoted/bare words, "{" or "}" for braces.

    Skips whitespace, `//` comments and `[$PLATFORM]` conditions.
    """
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j + 1
        elif c in "{}":
            yield c, c
            i += 1
        elif c == "[":
            j = text.find("]", i)
            i = n if j < 0 else j + 1
        elif c == '"':
            out: list[str] = []
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\" and i + 1 < n:
                    nxt = text[i + 1]
                    # Unknown escapes (e.g. an unescaped "C:\Steam") keep their backslash.
                    out.append(_ESCAPES.get(nxt, "\\" + nxt))
                    i += 2
                else:
                    out.append(text[i])
                    i += 1
            i += 1  # closing quote
            yield "str", "".join(out)
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in '{}"':
                j += 1
            yield "str", text[i:j]
            i = j


def parse_vdf(text: str) -> VdfDict:
    """Parse Valve KeyValues text into nested dicts of strings.

    Keys keep their spelling (KeyValues keys are case-insensitive; look them up with
    `_get_ci`). A repeated key keeps its last value. Malformed input raises ValueError.
    """
    tokens = list(_vdf_tokens(text.lstrip("\ufeff")))
    pos = 0

    def parse_block(nested: bool) -> VdfDict:
        nonlocal pos
        out: VdfDict = {}
        while pos < len(tokens):
            kind, value = tokens[pos]
            if kind == "}":
                if not nested:
                    raise ValueError("unbalanced '}' in VDF")
                pos += 1
                return out
            if kind != "str":
                raise ValueError(f"expected a key in VDF, got {value!r}")
            pos += 1
            if pos >= len(tokens):
                raise ValueError(f"VDF key {value!r} has no value")
            nkind, nvalue = tokens[pos]
            pos += 1
            if nkind == "{":
                out[value] = parse_block(nested=True)
            elif nkind == "str":
                out[value] = nvalue
            else:
                raise ValueError(f"unexpected '}}' after VDF key {value!r}")
        if nested:
            raise ValueError("unterminated '{' in VDF")
        return out

    return parse_block(nested=False)


def _get_ci(d: Mapping[str, object], key: str) -> object | None:
    lower = key.lower()
    for k, v in d.items():
        if k.lower() == lower:
            return v
    return None


def _read_vdf(path: Path) -> VdfDict | None:
    try:
        return parse_vdf(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------
# Steam libraries
# ---------------------------------------------------------------------------------------------


def _native_path(text: str) -> Path:
    """A path from a Steam file. On WSL a Windows path "D:\\X" becomes "/mnt/d/X"."""
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if m and os.name != "nt":
        wsl = Path("/mnt") / m.group(1).lower()
        if wsl.is_dir():
            return wsl.joinpath(*re.split(r"[\\/]+", m.group(2)))
    return Path(text).expanduser()


def _steamapps(library: Path) -> list[Path]:
    # Old Steam clients used "SteamApps"; it matters on case-sensitive file systems.
    return [library / "steamapps", library / "SteamApps"]


def steam_library_folders(steam_root: Path) -> list[Path]:
    """The Steam root plus every library in its libraryfolders.vdf, AC's library first.

    Reads `steamapps/libraryfolders.vdf` (and `config/libraryfolders.vdf`), both the current
    format (numbered blocks with "path" and an "apps" map of app id -> size) and the old one
    (numbered keys whose value is the path).
    """
    steam_root = Path(steam_root)
    with_ac: list[Path] = []
    others: list[Path] = [steam_root]
    files = [d / "libraryfolders.vdf" for d in _steamapps(steam_root)]
    files.append(steam_root / "config" / "libraryfolders.vdf")
    for vdf_path in files:
        data = _read_vdf(vdf_path)
        if data is None:
            continue
        root = _get_ci(data, "libraryfolders")
        if not isinstance(root, dict):
            continue
        for key, entry in root.items():
            if not key.isdigit():
                continue
            if isinstance(entry, str):
                others.append(_native_path(entry))
            elif isinstance(entry, dict):
                path = _get_ci(entry, "path")
                if not isinstance(path, str) or not path:
                    continue
                apps = _get_ci(entry, "apps")
                has_ac = isinstance(apps, dict) and AC_APP_ID in apps
                (with_ac if has_ac else others).append(_native_path(path))
    return _dedupe(with_ac + others)


def ac_roots_in_library(library: Path) -> list[Path]:
    """Where AC would be inside one Steam library (whether or not it exists).

    `appmanifest_244210.acf`'s `installdir` comes first, then the default `assettocorsa`.
    """
    out: list[Path] = []
    for steamapps in _steamapps(Path(library)):
        manifest = _read_vdf(steamapps / f"appmanifest_{AC_APP_ID}.acf")
        state = _get_ci(manifest, "AppState") if manifest else None
        installdir = _get_ci(state, "installdir") if isinstance(state, dict) else None
        if isinstance(installdir, str) and installdir:
            out.append(steamapps / "common" / installdir)
        out.append(steamapps / "common" / AC_INSTALL_DIR)
    return _dedupe(out)


def _dedupe(paths: Sequence[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            out.append(Path(p))
    return out


# ---------------------------------------------------------------------------------------------
# Platform guesses
# ---------------------------------------------------------------------------------------------


def _registry_value(hive_name: str, subkey: str, value: str) -> str | None:
    if sys.platform != "win32":
        return None
    try:
        import winreg
    except ImportError:
        return None
    hive = getattr(winreg, hive_name)
    for view in (0, getattr(winreg, "KEY_WOW64_64KEY", 0)):
        try:
            with winreg.OpenKey(hive, subkey, 0, winreg.KEY_READ | view) as key:
                result, _ = winreg.QueryValueEx(key, value)
        except OSError:
            continue
        if isinstance(result, str) and result:
            return result
    return None


def _registry_steam_roots() -> list[Path]:
    """Steam folder from the registry: HKCU SteamPath (ASSUMED, widely used), HKLM InstallPath."""
    queries = [
        ("HKEY_CURRENT_USER", r"Software\Valve\Steam", "SteamPath"),
        ("HKEY_LOCAL_MACHINE", r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
        ("HKEY_LOCAL_MACHINE", r"SOFTWARE\Valve\Steam", "InstallPath"),
    ]
    out = []
    for hive, sub, name in queries:
        value = _registry_value(hive, sub, name)
        if value:
            out.append(Path(value))
    return out


def _registry_ac_roots() -> list[Path]:
    """AC's Steam uninstall entry (standard Steam practice for every app; ASSUMED for AC)."""
    out = []
    for sub in (
        rf"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Steam App {AC_APP_ID}",
        rf"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Steam App {AC_APP_ID}",
    ):
        value = _registry_value("HKEY_LOCAL_MACHINE", sub, "InstallLocation")
        if value:
            out.append(Path(value))
    return out


def _windows_drives() -> list[str]:
    if sys.platform != "win32":
        return []
    listdrives = getattr(os, "listdrives", None)  # Python 3.12+, Windows only
    drives = listdrives() if listdrives else [f"{c}:\\" for c in "CDEFGH"]
    return [d for d in drives if d[:1].upper() not in ("A", "B")]


def default_steam_roots(home: Path | None = None) -> list[Path]:
    """Usual Steam install folders for this platform (they may not exist)."""
    home = Path.home() if home is None else Path(home)
    if sys.platform == "win32":
        out = []
        for env in ("ProgramFiles(x86)", "ProgramFiles"):
            base = os.environ.get(env)
            if base:
                out.append(Path(base) / "Steam")
        out += [Path("C:/Program Files (x86)/Steam"), Path("C:/Program Files/Steam")]
        return _dedupe(out)
    if sys.platform == "darwin":
        return [home / "Library" / "Application Support" / "Steam"]
    return [
        home / ".steam" / "steam",
        home / ".steam" / "root",
        home / ".local" / "share" / "Steam",
        home / ".var" / "app" / "com.valvesoftware.Steam" / ".local" / "share" / "Steam",
        home / ".var" / "app" / "com.valvesoftware.Steam" / "data" / "Steam",
        home / "snap" / "steam" / "common" / ".local" / "share" / "Steam",
        Path("/mnt/c/Program Files (x86)/Steam"),  # WSL
        Path("/mnt/c/Program Files/Steam"),
    ]


def _default_libraries() -> list[Path]:
    """Common Steam library folders on each Windows drive."""
    out = []
    for drive in _windows_drives():
        root = Path(drive)
        for sub in ("SteamLibrary", "Steam", "Games/SteamLibrary", "Games/Steam"):
            out.append(root / sub)
        out += [root / "Program Files (x86)" / "Steam", root / "Program Files" / "Steam"]
    return out


# ---------------------------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------------------------


def candidate_ac_roots(
    *,
    env: Mapping[str, str] | None = None,
    steam_roots: Sequence[Path] | None = None,
    search_defaults: bool = True,
) -> Iterator[Path]:
    """Every folder that might be the AC root, in search order (duplicates possible).

    `env` defaults to `os.environ`. `steam_roots` are searched before the registry and default
    Steam folders; `search_defaults=False` skips the registry and all built-in guesses (tests).
    """
    env = os.environ if env is None else env
    value = env.get(ENV_AC_ROOT, "").strip()
    if value:
        yield Path(value).expanduser()
    if search_defaults:
        yield from _registry_ac_roots()
    roots = list(steam_roots or [])
    if search_defaults:
        roots += _registry_steam_roots() + default_steam_roots()
    for steam_root in _dedupe(roots):
        for library in steam_library_folders(steam_root):
            yield from ac_roots_in_library(library)
    if search_defaults:
        for library in _default_libraries():
            yield from ac_roots_in_library(library)


def find_ac_root(
    *,
    env: Mapping[str, str] | None = None,
    steam_roots: Sequence[Path] | None = None,
    search_defaults: bool = True,
) -> Path | None:
    """The Assetto Corsa install folder, or None if it can't be found.

    See the module docstring for the search order and `candidate_ac_roots` for the arguments.
    """
    env = os.environ if env is None else env
    explicit = env.get(ENV_AC_ROOT, "").strip()
    seen: set[str] = set()
    for candidate in candidate_ac_roots(
        env=env, steam_roots=steam_roots, search_defaults=search_defaults
    ):
        key = os.path.normcase(os.path.abspath(candidate))
        if key in seen:
            continue
        seen.add(key)
        if is_ac_root(candidate):
            return Path(os.path.abspath(candidate))
        if explicit and candidate == Path(explicit).expanduser():
            warnings.warn(
                f"{ENV_AC_ROOT}={explicit!r} is not an Assetto Corsa folder (expected "
                f"content/tracks and acs.exe or AssettoCorsa.exe); searching elsewhere",
                stacklevel=2,
            )
    return None
