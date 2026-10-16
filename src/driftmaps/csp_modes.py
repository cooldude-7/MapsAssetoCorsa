"""Install Assetto Corsa Map Generator's CSP New Modes into Assetto Corsa.

A CSP New Mode is a folder `<AC>/extension/lua/new-modes/<id>/` holding `manifest.ini`,
`mode.lua`, `settings.ini` and so on (docs/research/bots.md B.4). Ours live in the repo's
`csp/new-modes/` (a frozen app looks for them in `<bundle>/csp/new-modes`, or in the folder
named by `DRIFTMAPS_CSP_MODES`). The one mode so far is **Touge Battle** (`acmg-touge`): 1v1
against one AI car on an A->B track we built (tag "touge", `data/acmg_touge.json`); see
docs/checklists/touge.md for the in-game steps.

`install_touge(ac_root)` copies the mode over, replacing only a folder this tool installed
before (its manifest names us as AUTHOR) unless `force`. With `trackday_variant` it also
installs `acmg-touge-trackday`: the same script on CSP's Track Day base mode instead of Race,
a fallback to try if AC's race session misbehaves on an A->B road (spike S3 in bots.md).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

__all__ = [
    "ENV_MODES_DIR",
    "MODE_AUTHOR",
    "TOUGE_MODE_ID",
    "TOUGE_TRACKDAY_ID",
    "ModeInstallError",
    "has_csp",
    "install_touge",
    "is_ours",
    "manifest_value",
    "modes_source_dir",
    "new_modes_dir",
    "trackday_manifest",
    "uninstall_touge",
]

TOUGE_MODE_ID = "acmg-touge"
TOUGE_TRACKDAY_ID = "acmg-touge-trackday"
MODE_AUTHOR = "Assetto Corsa Map Generator"
ENV_MODES_DIR = "DRIFTMAPS_CSP_MODES"
_IGNORE = shutil.ignore_patterns(".*", "__pycache__", "*.pyc")


class ModeInstallError(Exception):
    """The mode can't be installed (no source files, no CSP, or a folder that isn't ours)."""


def modes_source_dir(env: dict[str, str] | None = None) -> Path:
    """The folder holding our New Mode folders (`csp/new-modes`)."""
    env = os.environ if env is None else env
    candidates: list[Path] = []
    if env.get(ENV_MODES_DIR, "").strip():
        candidates.append(Path(env[ENV_MODES_DIR]).expanduser())
    bundle = getattr(sys, "_MEIPASS", None)  # PyInstaller
    if bundle:
        candidates.append(Path(bundle) / "csp" / "new-modes")
    candidates.append(Path(__file__).resolve().parents[2] / "csp" / "new-modes")
    for folder in candidates:
        if (folder / TOUGE_MODE_ID / "manifest.ini").is_file():
            return folder
    raise ModeInstallError(
        "the CSP mode files were not found (looked in "
        + ", ".join(str(c) for c in candidates)
        + f"); set {ENV_MODES_DIR} to the csp/new-modes folder"
    )


def new_modes_dir(ac_root: Path) -> Path:
    """`<ac_root>/extension/lua/new-modes`, where CSP looks for New Modes."""
    return Path(ac_root) / "extension" / "lua" / "new-modes"


def has_csp(ac_root: Path) -> bool:
    """True if Custom Shaders Patch looks installed (it brings the `extension` folder)."""
    return (Path(ac_root) / "extension").is_dir()


def manifest_value(text: str, key: str) -> str | None:
    """A `KEY = value` from manifest.ini text (first match, comments after ';' dropped)."""
    m = re.search(rf"^[ \t]*{re.escape(key)}[ \t]*=([^\r\n]*)", text, re.MULTILINE)
    return m.group(1).split(";", 1)[0].strip() if m else None


def is_ours(folder: Path) -> bool:
    """True if `folder` holds a New Mode this tool installed (manifest AUTHOR is us)."""
    try:
        text = (Path(folder) / "manifest.ini").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return manifest_value(text, "AUTHOR") == MODE_AUTHOR


def trackday_manifest(text: str) -> str:
    """The Touge Battle manifest on Track Day (practice with AI) instead of Race."""
    out, n = re.subn(r"^BASE_MODE[ \t]*=[^\r\n]*", "BASE_MODE=TRACK_DAY", text, flags=re.MULTILINE)
    if n != 1:
        raise ModeInstallError("manifest.ini has no single BASE_MODE line to change")
    return re.sub(
        r"^(NAME[ \t]*=[^\r\n]*?)[ \t]*$", r"\1 - Track Day test", out, count=1, flags=re.MULTILINE
    )


def _copy_mode(
    source: Path, dest: Path, *, force: bool, manifest: Callable[[str], str] | None = None
) -> Path:
    if dest.exists() and not is_ours(dest) and not force:
        raise ModeInstallError(
            f"{dest} exists and was not installed by {MODE_AUTHOR}; remove it or use --force"
        )
    tmp = dest.with_name(dest.name + ".acmg-tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(source, tmp, ignore=_IGNORE)
    if manifest is not None:
        path = tmp / "manifest.ini"
        path.write_text(manifest(path.read_text(encoding="utf-8")), encoding="utf-8")
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    return dest


def install_touge(
    ac_root: Path,
    *,
    force: bool = False,
    trackday_variant: bool = False,
    source_dir: Path | None = None,
) -> list[Path]:
    """Copy Touge Battle into `<ac_root>/extension/lua/new-modes/` and return the folders.

    Raises `ModeInstallError` if CSP isn't installed (unless `force`) or a folder of the same
    name exists that this tool didn't install (unless `force`)."""
    ac_root = Path(ac_root)
    if not has_csp(ac_root) and not force:
        raise ModeInstallError(
            f"Custom Shaders Patch is not installed in {ac_root} (no extension folder); "
            "install CSP 0.2.11 or newer first (Content Manager: Settings > Custom Shaders Patch)"
        )
    source = (source_dir or modes_source_dir()) / TOUGE_MODE_ID
    if not (source / "mode.lua").is_file():
        raise ModeInstallError(f"{source} has no mode.lua")
    target = new_modes_dir(ac_root)
    target.mkdir(parents=True, exist_ok=True)
    out = [_copy_mode(source, target / TOUGE_MODE_ID, force=force)]
    if trackday_variant:
        out.append(
            _copy_mode(source, target / TOUGE_TRACKDAY_ID, force=force, manifest=trackday_manifest)
        )
    return out


def uninstall_touge(ac_root: Path) -> list[Path]:
    """Remove the Touge Battle folders this tool installed; returns the removed folders."""
    removed = []
    for mode_id in (TOUGE_MODE_ID, TOUGE_TRACKDAY_ID):
        folder = new_modes_dir(ac_root) / mode_id
        if folder.is_dir() and is_ours(folder):
            shutil.rmtree(folder)
            removed.append(folder)
    return removed
