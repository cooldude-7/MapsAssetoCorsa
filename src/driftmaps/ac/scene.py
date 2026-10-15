"""Shared track-assembly helpers: one KN5 scene, materials, spawn/gate dummies, AI files, zips.

Used by the M0 test tracks (`driftmaps.spike`) and the real track builder (`driftmaps.build`).
Everything here is format glue that was proven in game at M0 (docs/checklists/M0-results.md):

- `Scene` collects textures, materials and root children; every node (meshes and `AC_*`
  dummies) is a direct child of the root dummy (ac_format.md §2).
- `ks_per_pixel` is the plain `ksPerPixel` material with one diffuse texture (ac_format.md §5).
- `spawn_dummy` / `gate_dummies`: local +Z along the road, `_L` on the driver's left
  (`coords.left_of`), verified with dm_t12 (S4, S6).
- `zip_tracks` packs track folders deterministically (sorted entries, fixed dates and modes).
- `clear_track_folder` / `install_track` only ever delete a folder drift-maps made (a
  `.driftmaps` marker, or "drift-maps" in `ui/ui_track.json` tags), unless forced: an id
  that clashes with a Kunos or mod track must never wipe it.
"""

from __future__ import annotations

import json
import shutil
import time
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Unpack

import numpy as np

from driftmaps.ac.ai_line import AiLine, write_ai
from driftmaps.ac.coords import WORLD_UP, frame_transform
from driftmaps.ac.model import (
    Dummy,
    Kn5,
    Material,
    MaterialProperty,
    Mesh,
    Node,
    Texture,
    TextureMapping,
)
from driftmaps.ac.track_files import validate_track_id
from driftmaps.geometry.primitives import MeshData, MeshFlags, to_kn5_mesh

__all__ = [
    "DECAL_FLAGS",
    "HIDDEN_WALL_FLAGS",
    "MARKER_FILE",
    "MARKER_TEXT",
    "marker_built_at",
    "stamp_marker",
    "Scene",
    "TrackFolderError",
    "check_track_folder",
    "clear_track_folder",
    "gate_dummies",
    "install_track",
    "is_driftmaps_track",
    "ks_per_pixel",
    "spawn_dummy",
    "write_ai_files",
    "zip_tracks",
]

# Invisible collision walls (ac_format.md §3.2) and visual-only markings that should not cast
# shadows (painted lines, decals).
HIDDEN_WALL_FLAGS: MeshFlags = {"is_renderable": False, "cast_shadows": False}
DECAL_FLAGS: MeshFlags = {"cast_shadows": False}


def ks_per_pixel(name: str, texture_name: str, *, bright: bool = False) -> Material:
    """A `ksPerPixel` material with one diffuse texture (ac_format.md §5 "v1" values).

    `bright` lifts ambient so signs and markers read well in shadow.
    """
    return Material(
        name=name,
        shader="ksPerPixel",
        properties=[
            MaterialProperty("ksAmbient", a=0.6 if bright else 0.45),
            MaterialProperty("ksDiffuse", a=0.6),
            MaterialProperty("ksSpecular", a=0.08),
            MaterialProperty("ksSpecularEXP", a=12.0),
            MaterialProperty("ksEmissive", c=(0.0, 0.0, 0.0)),
            MaterialProperty("ksAlphaRef", a=0.0),
        ],
        textures=[TextureMapping("txDiffuse", 0, texture_name)],
    )


class Scene:
    """Collects the textures, materials and root children of one KN5 file.

    `texture_factory(name)` makes the embedded texture for a texture file name the first time a
    material uses it.
    """

    def __init__(self, root_name: str, texture_factory: Callable[[str], Texture]) -> None:
        self.root_name = root_name
        self.texture_factory = texture_factory
        self.textures: dict[str, Texture] = {}
        self.materials: list[Material] = []
        self.nodes: list[Node] = []

    def material(self, name: str, texture_name: str, *, bright: bool = False) -> int:
        """Index of material `name` (one diffuse texture), adding it on first use."""
        for i, mat in enumerate(self.materials):
            if mat.name == name:
                return i
        if texture_name not in self.textures:
            self.textures[texture_name] = self.texture_factory(texture_name)
        self.materials.append(ks_per_pixel(name, texture_name, bright=bright))
        return len(self.materials) - 1

    def mesh(self, data: MeshData, name: str, material: int, **flags: Unpack[MeshFlags]) -> Mesh:
        mesh = to_kn5_mesh(data, name, material, **flags)
        self.nodes.append(mesh)
        return mesh

    def add(self, *nodes: Node) -> None:
        self.nodes.extend(nodes)

    def kn5(self, version: int = 5) -> Kn5:
        return Kn5(
            root=Dummy(self.root_name, children=list(self.nodes)),
            textures=list(self.textures.values()),
            materials=list(self.materials),
            version=version,
            extra=0,
        )


# --------------------------------------------------------------------------------------------
# Dummies
# --------------------------------------------------------------------------------------------


def spawn_dummy(name: str, position: np.ndarray, forward: np.ndarray) -> Dummy:
    """A spawn dummy at `position` facing `forward` (+Z forward, +X = driver's left)."""
    return Dummy(name, frame_transform(position, forward))


def gate_dummies(
    prefix: str,
    position: np.ndarray,
    forward: np.ndarray,
    left: np.ndarray,
    reach: float,
    height: float,
    *,
    swapped: bool = False,
) -> list[Dummy]:
    """`<prefix>_L` on the driver's left and `<prefix>_R` on the right (swapped: reversed),
    each `reach` metres from `position` along `left`, `height` up, oriented along `forward`."""
    up = height * WORLD_UP
    sides = {"L": position + reach * left + up, "R": position - reach * left + up}
    if swapped:
        sides = {"L": sides["R"], "R": sides["L"]}
    return [Dummy(f"{prefix}_{k}", frame_transform(sides[k], forward)) for k in ("L", "R")]


# --------------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------------


def write_ai_files(ai: AiLine, track_dir: Path) -> Path:
    """Write ai/fast_lane.ai and the same bytes as data/ideal_line.ai.

    COMMUNITY: data/ideal_line.ai is "the preferred ideal line shown in-game" (ai_line.md §2.2).
    """
    track_dir = Path(track_dir)
    ai_path = write_ai(ai, track_dir / "ai" / "fast_lane.ai")
    (track_dir / "data").mkdir(parents=True, exist_ok=True)
    (track_dir / "data" / "ideal_line.ai").write_bytes(ai_path.read_bytes())
    return ai_path


# Fixed metadata so the zip is byte-identical on every build and platform.
_ZIP_DATE = (2026, 1, 1, 0, 0, 0)
_ZIP_FILE_MODE = 0o100644


def zip_tracks(track_dirs: Sequence[Path], zip_path: Path) -> Path:
    """Zip track folders with each folder at the zip root, so extracting the zip into
    `content/tracks/` installs them. Deterministic: sorted entries, fixed dates and modes."""
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[str, Path]] = []
    for track_dir in track_dirs:
        track_dir = Path(track_dir)
        for file in track_dir.rglob("*"):
            if file.is_file():
                rel = file.relative_to(track_dir.parent).as_posix()
                entries.append((rel, file))
    entries.sort(key=lambda e: e[0])
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        for rel, file in entries:
            info = zipfile.ZipInfo(rel, date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3  # Unix, so the mode below means the same everywhere
            info.external_attr = _ZIP_FILE_MODE << 16
            zf.writestr(info, file.read_bytes(), compresslevel=6)
    tmp.replace(zip_path)
    return zip_path


# Written into every track folder drift-maps builds; together with the "drift-maps" ui tag it
# marks a folder that drift-maps may replace.
MARKER_FILE = ".driftmaps"
MARKER_TEXT = "Made by drift-maps. drift-maps build/install may replace this folder.\n"


_BUILT_AT = "built_at: "


def stamp_marker(folder: Path, when: float | None = None) -> Path:
    """Write the marker with the build time (`built_at: <unix seconds>`): zips carry a fixed
    date (`_ZIP_DATE`), so file times can't say when a track was built."""
    path = Path(folder) / MARKER_FILE
    when = time.time() if when is None else float(when)
    path.write_text(f"{MARKER_TEXT}{_BUILT_AT}{when:.0f}\n", encoding="utf-8", newline="\n")
    return path


def marker_built_at(folder: Path) -> float | None:
    """The build time written by `stamp_marker` (unix seconds), or None."""
    try:
        text = (Path(folder) / MARKER_FILE).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith(_BUILT_AT):
            try:
                value = float(line[len(_BUILT_AT) :].strip())
            except ValueError:
                return None
            return value if value > 0 else None
    return None


def is_zip_time(ts: float) -> bool:
    """True if a file time is the fixed zip date (the file came out of one of our zips)."""
    return time.localtime(ts)[:6] == _ZIP_DATE


class TrackFolderError(FileExistsError):
    """A track folder of that id exists and was not made by drift-maps."""


def is_driftmaps_track(folder: Path) -> bool:
    """True if `folder` was made by drift-maps (marker file, or the ui_track.json tag)."""
    folder = Path(folder)
    if (folder / MARKER_FILE).is_file():
        return True
    try:
        ui = json.loads((folder / "ui" / "ui_track.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    tags = ui.get("tags") if isinstance(ui, dict) else None
    return isinstance(tags, list) and "drift-maps" in tags


def check_track_folder(folder: Path, *, force: bool = False) -> None:
    """Raise `TrackFolderError` if `folder` exists and may not be replaced."""
    folder = Path(folder)
    if not folder.exists() or force:
        return
    if not folder.is_dir() or not is_driftmaps_track(folder):
        raise TrackFolderError(
            f"{folder} already exists and was not made by drift-maps (another game or mod "
            "track?). Refusing to replace it: pick another --id, or pass --force to replace it."
        )


def clear_track_folder(folder: Path, *, force: bool = False) -> None:
    """Delete an existing track folder before it is rebuilt, but only one drift-maps made
    (or anything with `force`); otherwise raise `TrackFolderError` and touch nothing."""
    folder = Path(folder)
    check_track_folder(folder, force=force)
    if folder.is_dir() and not folder.is_symlink():
        shutil.rmtree(folder)
    elif folder.exists() or folder.is_symlink():
        folder.unlink()


def install_track(track_dir: Path, tracks_root: Path, *, force: bool = False) -> Path:
    """Copy a built track folder into `tracks_root` (an AC `content/tracks` folder), replacing
    an existing folder of the same id only if drift-maps made it (or `force`). Returns the
    installed folder."""
    track_dir = Path(track_dir)
    dest = Path(tracks_root) / validate_track_id(track_dir.name)
    clear_track_folder(dest, force=force)
    shutil.copytree(track_dir, dest)
    return dest
