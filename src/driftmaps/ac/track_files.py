"""Writers for the non-KN5 files of an Assetto Corsa track folder.

Folder layout (docs/research/ac_format.md §1.1), for a single-layout track `<id>`:

    content/tracks/<id>/
        <id>.kn5, models.ini, map.png, LICENSE-DATA.txt
        data/   surfaces.ini, cameras.ini, map.ini, lighting.ini, groove.ini, crew.ini
        ui/     ui_track.json, preview.png (355x200), outline.png (365x192)

With layouts, `map.png` and `data/` move into `<id>/<layout>/`, which is why the map writer
takes the data folder and the map.png folder separately.

Sources: Content Manager / AcTools (gro-ove/actools, MS-PL) for map.ini, ui_track.json fields
and the preview/outline sizes; real track files for the INI layouts. Each writer's docstring
names its source and marks what is ASSUMED. No AcTools code is copied.

All coordinates are AC world space (X = east, Y = up, Z = south; `driftmaps.ac.coords`).
INI files are written as ASCII with CRLF line endings, like Kunos' files.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "DATA_SOURCES",
    "DEFAULT_SURFACES",
    "DEFAULT_TAGS",
    "GENERATOR_NAME",
    "KUNOS_NOTICE",
    "MAP_INI_MAX_SIZE",
    "OUTLINE_SIZE",
    "PREVIEW_SIZE",
    "Camera",
    "MapInfo",
    "Surface",
    "UiTrack",
    "attribution_text",
    "cameras_along",
    "compute_map_info",
    "draw_map",
    "make_track_id",
    "read_map_ini",
    "render_map",
    "validate_track_id",
    "write_cameras_ini",
    "write_crew_ini",
    "write_data_license",
    "write_groove_ini",
    "write_lighting_ini",
    "write_map_ini",
    "write_models_ini",
    "write_outline_png",
    "write_preview_png",
    "write_surfaces_ini",
    "write_ui_track_json",
]

# CM constants (AcTools/CommonAcConsts.cs): TrackPreviewWidth/Height, TrackOutlineWidth/Height.
PREVIEW_SIZE = (355, 200)
OUTLINE_SIZE = (365, 192)

# map.ini MAX_SIZE: Kunos files and Hagnhofer's docs both have 1600 ("unknown, default 1600").
# CM's own map writer omits the key. We always write the Kunos default; it is NOT our pixel cap.
MAP_INI_MAX_SIZE = 1600

_TRACK_ID_RE = re.compile(r"^[a-z0-9_]{1,32}$")
_SURFACE_KEY_RE = re.compile(r"^[A-Z0-9_-]+$")


# ---------------------------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------------------------


def _num(x: float, decimals: int = 6) -> str:
    """Fixed-point number without trailing zeros or exponent: 1.0 -> "1", 0.25 -> "0.25"."""
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"cannot write non-finite number {x!r}")
    s = f"{x:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _vec(v: Sequence[float] | np.ndarray, sep: str = ",", decimals: int = 6) -> str:
    return sep.join(_num(c, decimals) for c in np.asarray(v, dtype=np.float64).ravel())


def _check_ascii(text: str, what: str) -> str:
    if not text.isascii():
        raise ValueError(f"{what} must be ASCII: {text!r}")
    if any(c in text for c in "\r\n"):
        raise ValueError(f"{what} must be a single line: {text!r}")
    return text


def _write_text(path: Path, text: str, encoding: str = "ascii") -> Path:
    """Write `text` with CRLF line endings, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode(encoding))
    return path


def _ini(sections: Iterable[tuple[str, Iterable[tuple[str, object]]]]) -> str:
    out: list[str] = []
    for name, items in sections:
        if out:
            out.append("")
        out.append(f"[{name}]")
        for key, value in items:
            out.append(f"{key}={value}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------
# Track id
# ---------------------------------------------------------------------------------------------


def validate_track_id(track_id: str) -> str:
    """Return `track_id` if it is a valid track folder name, else raise ValueError.

    Rule (assettocorsamods basic guide, ac_format.md §1.1): lowercase a-z, 0-9 and "_",
    at most 32 characters. The main KN5 must be named `<id>.kn5`.
    """
    if not isinstance(track_id, str) or not _TRACK_ID_RE.match(track_id):
        raise ValueError(
            f"invalid track id {track_id!r}: use 1-32 characters from lowercase a-z, 0-9 and _"
        )
    return track_id


def make_track_id(name: str, prefix: str = "dm_") -> str:
    """A valid track id from a free-text name, e.g. "Akina Downhill" -> "dm_akina_downhill"."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_") or "track"
    return validate_track_id((prefix + slug)[:32].rstrip("_") or "track")


# ---------------------------------------------------------------------------------------------
# data/surfaces.ini
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Surface:
    """One `[SURFACE_n]` entry of data/surfaces.ini (ac_format.md §1.5).

    A physical mesh named `<digits><KEY>...` (e.g. `1ROAD_c003`) uses the surface with that KEY.
    `extra` holds additional keys such as CSP's `_EXT_SURFACE_TYPE` (ignored by vanilla AC,
    ASSUMED).
    """

    key: str
    friction: float = 1.0
    damping: float = 0.0
    wav: str = ""
    wav_pitch: float = 0.0
    ff_effect: str = "NULL"
    dirt_additive: float = 0.0
    black_flag_time: float = 0.0
    is_valid_track: bool = True
    sin_height: float = 0.0
    sin_length: float = 0.0
    is_pitlane: bool = False
    vibration_gain: float = 0.0
    vibration_length: float = 0.0
    extra: tuple[tuple[str, str], ...] = ()

    def items(self) -> list[tuple[str, str]]:
        """KEY=VALUE pairs in the order of Kunos' system/data/surfaces.ini."""
        items = [
            ("KEY", self.key),
            ("FRICTION", _num(self.friction)),
            ("DAMPING", _num(self.damping)),
            ("WAV", self.wav),
            ("WAV_PITCH", _num(self.wav_pitch)),
            ("FF_EFFECT", self.ff_effect),
            ("DIRT_ADDITIVE", _num(self.dirt_additive)),
            ("BLACK_FLAG_TIME", _num(self.black_flag_time)),
            ("IS_VALID_TRACK", str(int(self.is_valid_track))),
            ("SIN_HEIGHT", _num(self.sin_height)),
            ("SIN_LENGTH", _num(self.sin_length)),
            ("IS_PITLANE", str(int(self.is_pitlane))),
            ("VIBRATION_GAIN", _num(self.vibration_gain)),
            ("VIBRATION_LENGTH", _num(self.vibration_length)),
        ]
        items.extend(self.extra)
        return items


# ROAD and GRASS copy vanilla system/data/surfaces.ini (VERIFIED (doc), ac_format.md §1.5).
# GRAVEL is our own key for shoulders: FRICTION 0.75, DIRT_ADDITIVE 1 and the CSP surface type
# come from the research plan; DAMPING, WAV (vanilla sand.wav), SIN_* and VIBRATION_* are
# ASSUMED (between vanilla GRASS and SAND) and need tuning in game.
# There is deliberately no WALL key: walls are meshes with WALL in the name, and a WALL surface
# would make them driveable (KinG-InFeT note, ASSUMED).
DEFAULT_SURFACES: tuple[Surface, ...] = (
    Surface(key="ROAD", friction=1.0),
    Surface(
        key="GRASS",
        friction=0.6,
        wav="grass.wav",
        dirt_additive=1.0,
        is_valid_track=False,
        sin_height=0.03,
        sin_length=0.5,
        vibration_gain=0.2,
        vibration_length=0.6,
    ),
    Surface(
        key="GRAVEL",
        friction=0.75,
        damping=0.03,
        wav="sand.wav",
        dirt_additive=1.0,
        is_valid_track=False,
        sin_height=0.02,
        sin_length=0.4,
        vibration_gain=0.2,
        vibration_length=0.3,
        extra=(("_EXT_SURFACE_TYPE", "GRAVEL"),),
    ),
)


def write_surfaces_ini(path: Path, surfaces: Sequence[Surface] = DEFAULT_SURFACES) -> Path:
    """Write data/surfaces.ini with one `[SURFACE_n]` section per surface, n from 0."""
    seen: set[str] = set()
    for s in surfaces:
        if not _SURFACE_KEY_RE.match(s.key):
            raise ValueError(f"surface key {s.key!r} must be uppercase A-Z, 0-9, _ or -")
        if "WALL" in s.key:
            raise ValueError("don't define a WALL surface: WALL meshes are collision barriers")
        if s.key in seen:
            raise ValueError(f"duplicate surface key {s.key!r}")
        seen.add(s.key)
        _check_ascii(s.wav, "WAV")
        for k, v in s.extra:
            _check_ascii(k, "surface option name")
            _check_ascii(v, "surface option value")
    return _write_text(path, _ini((f"SURFACE_{i}", s.items()) for i, s in enumerate(surfaces)))


# ---------------------------------------------------------------------------------------------
# ui/ui_track.json
# ---------------------------------------------------------------------------------------------

# Credited as the author in ui_track.json and in LICENSE-DATA.txt (the "drift-maps" ui tag and the
# .driftmaps marker stay as they are: they are how our own tracks are recognised).
GENERATOR_NAME = "Assetto Corsa Map Generator"
DEFAULT_TAGS: tuple[str, ...] = ("A2B", "touge", "point-to-point", "generated")


@dataclass
class UiTrack:
    """Fields of ui/ui_track.json (ac_format.md §1.9; CM `TrackObjectBase`).

    Kunos stores numbers as strings: `length` in metres ("17619"), `width` ("8m"), `pitboxes`
    ("4"). `geotags` (lat, lon) are written as plain signed decimal strings, which CM's
    `GeoTagsEntry` parses. `year` stays a JSON number. Quick Drive modes other than a 1-lap race
    warn unless the track has the "circuit" tag, which A->B tracks deliberately don't have.
    """

    name: str
    description: str = ""
    tags: list[str] = field(default_factory=lambda: list(DEFAULT_TAGS))
    geotags: tuple[float, float] | None = None
    country: str = ""
    city: str = ""
    length_m: float = 0.0
    width: str | float = ""
    pitboxes: int = 1
    run: str = "A2B"
    author: str = GENERATOR_NAME
    version: str = "1.0"
    year: int | None = None
    url: str = ""

    def to_json_dict(self) -> dict[str, object]:
        width = self.width if isinstance(self.width, str) else f"{_num(self.width, 1)}m"
        d: dict[str, object] = {
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
        }
        if self.geotags is not None:
            lat, lon = self.geotags
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                raise ValueError(f"geotags must be (lat, lon) in degrees, got {self.geotags}")
            d["geotags"] = [_num(lat, 6), _num(lon, 6)]
        d.update(
            {
                "country": self.country,
                "city": self.city,
                "length": str(int(round(self.length_m))),
                "width": width,
                "pitboxes": str(int(self.pitboxes)),
                "run": self.run,
                "author": self.author,
                "version": self.version,
            }
        )
        if self.url:
            d["url"] = self.url
        if self.year is not None:
            d["year"] = int(self.year)
        return d


def write_ui_track_json(path: Path, ui: UiTrack) -> Path:
    """Write ui/ui_track.json (UTF-8; non-ASCII text is \\u-escaped so the bytes are ASCII).

    ASSUMED: AC's own JSON reader decodes \\u escapes (CM's Newtonsoft reader does).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(ui.to_json_dict(), indent=2, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="ascii", newline="\n")
    return path


# ---------------------------------------------------------------------------------------------
# map.png + data/map.ini
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MapInfo:
    """data/map.ini `[PARAMETERS]`: how world positions map onto map.png pixels.

    `px = (X + x_offset) / scale_factor`, `py = (Z + z_offset) / scale_factor`, in continuous
    image coordinates (pixel (i, j) covers [i, i+1) x [j, j+1)). Image up = world -Z = north.
    VERIFIED (code): CM's writer `TrackMapRenderer.TrackMapInformation.SaveTo` / `Prepare`
    (WIDTH = image width incl. margins, X_OFFSET = -min.X + Margin/Scale, SCALE_FACTOR =
    1/Scale with Scale in px per metre, ortho camera Up = (0, 0, -1)) and its consumer
    `ServerPresetRunningStatus.MapParams` (relX = (X + X_OFFSET) / (WIDTH * SCALE_FACTOR)).
    How AC itself uses the file is ASSUMED to match (in-game spike S11).
    """

    width: float
    height: float
    margin: float
    scale_factor: float
    x_offset: float
    z_offset: float
    max_size: int = MAP_INI_MAX_SIZE
    drawing_size: float = 10.0

    @property
    def size_px(self) -> tuple[int, int]:
        return int(round(self.width)), int(round(self.height))

    def world_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """AC points (..., 3), or (X, Z) pairs (..., 2), -> continuous pixel coords (..., 2)."""
        p = np.asarray(points, dtype=np.float64)
        x, z = (p[..., 0], p[..., 2]) if p.shape[-1] == 3 else (p[..., 0], p[..., 1])
        return np.stack(
            [(x + self.x_offset) / self.scale_factor, (z + self.z_offset) / self.scale_factor],
            axis=-1,
        )

    def pixel_to_world(self, pixels: np.ndarray) -> np.ndarray:
        """Continuous pixel coords (..., 2) -> world (X, Z) pairs (..., 2)."""
        p = np.asarray(pixels, dtype=np.float64)
        return np.stack(
            [
                p[..., 0] * self.scale_factor - self.x_offset,
                p[..., 1] * self.scale_factor - self.z_offset,
            ],
            axis=-1,
        )

    def ini_items(self) -> list[tuple[str, str]]:
        # Same key order and number style as Kunos' map.ini files.
        return [
            ("WIDTH", f"{self.width:.6f}"),
            ("HEIGHT", f"{self.height:.6f}"),
            ("MARGIN", _num(self.margin)),
            ("SCALE_FACTOR", f"{self.scale_factor:.6f}"),
            ("MAX_SIZE", str(int(self.max_size))),
            ("X_OFFSET", f"{self.x_offset:.6f}"),
            ("Z_OFFSET", f"{self.z_offset:.6f}"),
            ("DRAWING_SIZE", _num(self.drawing_size)),
        ]


def write_map_ini(path: Path, info: MapInfo) -> Path:
    return _write_text(path, _ini([("PARAMETERS", info.ini_items())]))


def read_map_ini(path: Path) -> MapInfo:
    """Parse a map.ini (ours or Kunos'); missing keys get CM's defaults."""
    values: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="ascii", errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if "=" in line and not line.startswith("["):
            k, v = line.split("=", 1)
            values[k.strip().upper()] = v.strip()

    def get(key: str, default: float) -> float:
        return float(values[key]) if key in values else default

    return MapInfo(
        width=get("WIDTH", 1000.0),
        height=get("HEIGHT", 1000.0),
        margin=get("MARGIN", 20.0),
        scale_factor=get("SCALE_FACTOR", 1.0),
        x_offset=get("X_OFFSET", 0.0),
        z_offset=get("Z_OFFSET", 0.0),
        max_size=int(get("MAX_SIZE", MAP_INI_MAX_SIZE)),
        drawing_size=get("DRAWING_SIZE", 10.0),
    )


def _as_polylines(polylines: Sequence[np.ndarray]) -> list[np.ndarray]:
    out = []
    for p in polylines:
        arr = np.asarray(p, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
            raise ValueError(f"each polyline must be a non-empty (N, 3) array, got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError("polyline contains NaN or inf")
        out.append(arr)
    if not out:
        raise ValueError("need at least one polyline")
    return out


def _as_widths(widths: float | Sequence[float], n: int) -> list[float]:
    arr = np.asarray(widths, dtype=np.float64)
    w = [float(arr)] * n if arr.ndim == 0 else [float(x) for x in arr.ravel()]
    if len(w) != n:
        raise ValueError(f"got {len(w)} widths for {n} polylines")
    if not all(math.isfinite(x) and x >= 0 for x in w):
        raise ValueError(f"road widths must be finite and >= 0, got {w}")
    return w


def _ceil6(x: float) -> float:
    """Round up to 6 decimals (map.ini precision), so the rounded scale still fits."""
    return math.ceil(x * 1e6 - 1e-6) / 1e6


def compute_map_info(
    polylines: Sequence[np.ndarray],
    widths: float | Sequence[float],
    *,
    meters_per_px: float = 1.0,
    margin: int = 20,
    max_size: int = 2048,
    drawing_size: float = 10.0,
) -> MapInfo:
    """Choose map.png size and the map.ini constants for these roads.

    The roads' bounding box (centrelines grown by half their width) is drawn at
    `meters_per_px`, plus `margin` px on every side. If that exceeds `max_size` px on either
    side, the scale grows so the larger side is exactly `max_size`. Values are rounded to the 6
    decimals map.ini stores, and the image is drawn with those rounded values.
    """
    lines = _as_polylines(polylines)
    ws = _as_widths(widths, len(lines))
    if meters_per_px <= 0:
        raise ValueError("meters_per_px must be positive")
    if max_size <= 2 * margin + 1:
        raise ValueError("max_size must leave room inside the margins")
    lo = np.min([ln[:, [0, 2]].min(0) - w / 2 for ln, w in zip(lines, ws, strict=True)], axis=0)
    hi = np.max([ln[:, [0, 2]].max(0) + w / 2 for ln, w in zip(lines, ws, strict=True)], axis=0)
    extent = np.maximum(hi - lo, 1e-3)
    sf = _ceil6(meters_per_px)
    if float(extent.max()) / sf + 2 * margin > max_size:
        sf = _ceil6(float(extent.max()) / (max_size - 2 * margin))
    size = np.minimum(np.ceil(extent / sf + 2 * margin - 1e-9), max_size)
    return MapInfo(
        width=float(size[0]),
        height=float(size[1]),
        margin=float(margin),
        scale_factor=sf,
        x_offset=round(-float(lo[0]) + margin * sf, 6),
        z_offset=round(-float(lo[1]) + margin * sf, 6),
        drawing_size=drawing_size,
    )


def _supersample(size: tuple[int, int], limit: int = 8192, best: int = 4) -> int:
    return max(1, min(best, limit // max(size)))


def _stroke(
    draw: ImageDraw.ImageDraw, pts: np.ndarray, width: float, fill: int | tuple[int, ...]
) -> None:
    """A polyline with round joins and caps; `pts` are Pillow coordinates.

    Pillow's wide-line joins leave hairline see-through seams between short segments on
    curves, so a disc is drawn on every vertex (round joins everywhere), not only at the ends.
    """
    wpx = max(1, int(round(width)))
    r = wpx / 2.0
    flat = [(float(x), float(y)) for x, y in pts]
    if len(flat) > 1:
        draw.line(flat, fill=fill, width=wpx, joint="curve")
    for x, y in flat:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)


def _mask(
    size: tuple[int, int], strokes: Sequence[tuple[np.ndarray, float]], ss: int
) -> Image.Image:
    """Anti-aliased coverage mask: draw at `ss`x and box-filter down.

    `strokes` hold (continuous pixel coords (N, 2), width in px). A continuous coordinate u
    lands at Pillow coordinate u * ss - 0.5, since Pillow puts pixel centres on integers.
    """
    w, h = size
    big = Image.new("L", (w * ss, h * ss), 0)
    draw = ImageDraw.Draw(big)
    for pts, width in strokes:
        _stroke(draw, pts * ss - 0.5, width * ss, 255)
    return big.resize((w, h), Image.Resampling.BOX) if ss > 1 else big


def draw_map(
    info: MapInfo,
    polylines: Sequence[np.ndarray],
    widths: float | Sequence[float],
    *,
    min_line_px: float = 2.0,
    outline_px: float = 0.0,
) -> Image.Image:
    """The map.png image: white roads on transparent, north-up, positioned by `info`.

    `outline_px` > 0 adds a dark border around the roads for contrast.
    """
    lines = _as_polylines(polylines)
    ws = _as_widths(widths, len(lines))
    size = info.size_px
    ss = _supersample(size)
    px_widths = [max(min_line_px, w / info.scale_factor) for w in ws]
    pix = [info.world_to_pixel(ln) for ln in lines]
    road = _mask(size, list(zip(pix, px_widths, strict=True)), ss)
    white = Image.new("RGBA", size, (255, 255, 255, 0))
    white.putalpha(road)
    if outline_px <= 0:
        return white
    border = _mask(size, [(p, w + 2 * outline_px) for p, w in zip(pix, px_widths, strict=True)], ss)
    out = Image.new("RGBA", size, (0, 0, 0, 0))
    out.putalpha(border)
    return Image.alpha_composite(out, white)


def render_map(
    polylines: Sequence[np.ndarray],
    widths: float | Sequence[float],
    data_dir: Path,
    track_dir: Path,
    *,
    meters_per_px: float = 1.0,
    margin: int = 20,
    max_size: int = 2048,
    min_line_px: float = 2.0,
    outline_px: float = 0.0,
    drawing_size: float = 10.0,
) -> MapInfo:
    """Write `<track_dir>/map.png` and `<data_dir>/map.ini` for the given road centrelines.

    `polylines` are (N, 3) AC-world centrelines and `widths` their road widths in metres (one
    number for all, or one per polyline). See `compute_map_info` for the sizing rules and
    `MapInfo` for the formula. Returns the constants written to map.ini.
    """
    info = compute_map_info(
        polylines,
        widths,
        meters_per_px=meters_per_px,
        margin=margin,
        max_size=max_size,
        drawing_size=drawing_size,
    )
    img = draw_map(info, polylines, widths, min_line_px=min_line_px, outline_px=outline_px)
    track_dir = Path(track_dir)
    track_dir.mkdir(parents=True, exist_ok=True)
    img.save(track_dir / "map.png", format="PNG")
    write_map_ini(Path(data_dir) / "map.ini", info)
    return info


# ---------------------------------------------------------------------------------------------
# ui/outline.png and ui/preview.png
# ---------------------------------------------------------------------------------------------


def _fit(
    lines: Sequence[np.ndarray], size: tuple[int, int], padding: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Uniform north-up fit of the lines' (X, Z) bounds into `size` minus `padding`.

    Returns (px per metre, world (X, Z) centre, image centre); a point maps to
    `(p - centre) * k + image_centre`.
    """
    xz = np.concatenate([ln[:, [0, 2]] for ln in lines])
    lo, hi = xz.min(0), xz.max(0)
    extent = np.maximum(hi - lo, 1e-6)
    avail = np.maximum(np.array(size, dtype=np.float64) - 2 * padding, 1.0)
    k = float(np.min(avail / extent))
    return k, (lo + hi) / 2, np.array(size, dtype=np.float64) / 2


def write_outline_png(
    path: Path,
    polylines: Sequence[np.ndarray],
    *,
    size: tuple[int, int] = OUTLINE_SIZE,
    padding: float = 12.0,
    line_px: float = 3.0,
    shadow: bool = True,
) -> Path:
    """ui/outline.png: the layout as a white line on transparent, north-up, fitted to `size`.

    CM draws its outlines as a white line with a soft dark shadow (`TrackOutlineRenderer`);
    `shadow` adds a faint dark halo so the line also reads on light backgrounds.
    """
    lines = _as_polylines(polylines)
    k, centre, mid = _fit(lines, size, padding)
    pix = [(ln[:, [0, 2]] - centre) * k + mid for ln in lines]
    ss = _supersample(size)
    out = Image.new("RGBA", size, (0, 0, 0, 0))
    if shadow:
        halo = _mask(size, [(p, line_px + 3.0) for p in pix], ss)
        out.putalpha(halo.point(lambda a: a * 90 // 255))
    white = Image.new("RGBA", size, (255, 255, 255, 0))
    white.putalpha(_mask(size, [(p, line_px) for p in pix], ss))
    out = Image.alpha_composite(out, white)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, format="PNG")
    return path


def _nice_step(span: float, target_lines: int = 6) -> float:
    raw = span / max(target_lines, 1)
    mag = 10 ** math.floor(math.log10(max(raw, 1e-9)))
    for m in (1, 2, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def write_preview_png(
    path: Path,
    polylines: Sequence[np.ndarray],
    *,
    size: tuple[int, int] = PREVIEW_SIZE,
    padding: float = 22.0,
    line_px: float = 5.0,
    title: str | None = None,
) -> Path:
    """ui/preview.png: a clean top-down render, north-up, on a dark background.

    Shows a metre grid, the roads (grey with light edges), a green "A" at the start of the first
    polyline and a red "B" at the end of the last one, and an optional title. CM can replace it
    with an in-game shot later.
    """
    lines = _as_polylines(polylines)
    w, h = size
    k, centre, mid = _fit(lines, size, padding)
    ss = _supersample(size)
    big = Image.new("RGB", (w * ss, h * ss), (24, 27, 32))
    draw = ImageDraw.Draw(big)

    def to_pil(p: np.ndarray) -> np.ndarray:
        return ((p - centre) * k + mid) * ss - 0.5

    # Grid in world metres.
    step = _nice_step(max(w, h) / k)
    x_world = (np.array([0, w]) - mid[0]) / k + centre[0]
    z_world = (np.array([0, h]) - mid[1]) / k + centre[1]
    for gx in np.arange(math.floor(x_world[0] / step) * step, x_world[1] + step, step):
        px = to_pil(np.array([gx, 0.0]))[0]
        draw.line((px, 0, px, h * ss), fill=(38, 43, 51), width=ss)
    for gz in np.arange(math.floor(z_world[0] / step) * step, z_world[1] + step, step):
        py = to_pil(np.array([0.0, gz]))[1]
        draw.line((0, py, w * ss, py), fill=(38, 43, 51), width=ss)

    pix = [to_pil(ln[:, [0, 2]]) for ln in lines]
    for p in pix:
        _stroke(draw, p, (line_px + 2.0) * ss, (225, 228, 232))
    for p in pix:
        _stroke(draw, p, line_px * ss, (88, 92, 100))

    marker_r = 7.0 * ss
    font = ImageFont.load_default(size=int(10 * ss))
    for (x, y), colour, letter in (
        (pix[0][0], (46, 184, 92), "A"),
        (pix[-1][-1], (220, 60, 50), "B"),
    ):
        draw.ellipse((x - marker_r, y - marker_r, x + marker_r, y + marker_r), fill=colour)
        draw.text((x, y), letter, font=font, fill=(255, 255, 255), anchor="mm")
    if title:
        tfont = ImageFont.load_default(size=int(12 * ss))
        draw.text((8 * ss, (h - 8) * ss), title, font=tfont, fill=(230, 230, 230), anchor="ls")
    img = big.resize(size, Image.Resampling.BOX) if ss > 1 else big
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path


# ---------------------------------------------------------------------------------------------
# data/cameras.ini
# ---------------------------------------------------------------------------------------------


@dataclass
class Camera:
    """One `[CAMERA_n]` of a replay TV camera set (cameras.ini `VERSION=3`, ac_format.md §1.6).

    Field set and defaults copied from a real exported file (FSAEJ_Endurance/data/cameras.ini)
    and Hagnhofer's docs. `in_point`/`out_point` are the normalised track progress where this
    camera takes over and hands over (ASSUMED: progress along ai/fast_lane.ai). TV cameras
    follow the car between `min_fov` and `max_fov` (ASSUMED from the key names; the real file
    uses 15-30 degrees).
    """

    name: str
    position: tuple[float, float, float]
    forward: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    in_point: float = 0.0
    out_point: float = 1.0
    min_fov: float = 20.0
    max_fov: float = 45.0
    shadow_splits: tuple[float, float, float] = (1.8, 20.0, 180.0)
    near_plane: float = 0.1
    far_plane: float = 5000.0
    min_exposure: float = 0.0
    max_exposure: float = 10000.0
    dof_factor: float = 10.0
    dof_range: float = 10000.0
    dof_focus: float = 0.0
    dof_manual: bool = False
    spline: str = ""
    spline_rotation: float = 0.0
    fov_gamma: float = 1.0
    spline_animation_length: float = 15.0
    is_fixed: bool = False

    def items(self) -> list[tuple[str, str]]:
        # Kunos' exporter separates vector parts with " ," (e.g. "-115.486 ,1.19716 ,35.4616").
        return [
            ("NAME", _check_ascii(self.name, "camera name")),
            ("POSITION", _vec(self.position, " ,", 4)),
            ("FORWARD", _vec(self.forward, " ,", 6)),
            ("UP", _vec(self.up, " ,", 6)),
            ("MIN_FOV", _num(self.min_fov, 3)),
            ("MAX_FOV", _num(self.max_fov, 3)),
            ("IN_POINT", _num(self.in_point, 6)),
            ("OUT_POINT", _num(self.out_point, 6)),
            ("SHADOW_SPLIT0", _num(self.shadow_splits[0])),
            ("SHADOW_SPLIT1", _num(self.shadow_splits[1])),
            ("SHADOW_SPLIT2", _num(self.shadow_splits[2])),
            ("NEAR_PLANE", _num(self.near_plane)),
            ("FAR_PLANE", _num(self.far_plane)),
            ("MIN_EXPOSURE", _num(self.min_exposure)),
            ("MAX_EXPOSURE", _num(self.max_exposure)),
            ("DOF_FACTOR", _num(self.dof_factor)),
            ("DOF_RANGE", _num(self.dof_range)),
            ("DOF_FOCUS", _num(self.dof_focus)),
            ("DOF_MANUAL", str(int(self.dof_manual))),
            ("SPLINE", _check_ascii(self.spline, "camera spline")),
            ("SPLINE_ROTATION", _num(self.spline_rotation)),
            ("FOV_GAMMA", _num(self.fov_gamma)),
            ("SPLINE_ANIMATION_LENGTH", _num(self.spline_animation_length)),
            ("IS_FIXED", str(int(self.is_fixed))),
        ]


def write_cameras_ini(path: Path, cameras: Sequence[Camera], set_name: str = "TV 1") -> Path:
    """Write one camera set (data/cameras.ini; extra sets go in cameras_1.ini, cameras_2.ini)."""
    header = [
        ("VERSION", "3"),
        ("CAMERA_COUNT", str(len(cameras))),
        ("SET_NAME", _check_ascii(set_name, "camera set name")),
    ]
    sections: list[tuple[str, Iterable[tuple[str, object]]]] = [("HEADER", header)]
    sections += [(f"CAMERA_{i}", cam.items()) for i, cam in enumerate(cameras)]
    return _write_text(path, _ini(sections))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def cameras_along(
    centreline: np.ndarray,
    spacing_m: float = 200.0,
    *,
    road_half_width_m: float = 4.0,
    offset_m: float = 6.0,
    height_m: float = 3.0,
    look_ahead_m: float = 30.0,
    min_fov: float = 20.0,
    max_fov: float = 45.0,
    spline_length_m: float | None = None,
    spline_offset_m: float = 0.0,
    name_prefix: str = "cam",
) -> list[Camera]:
    """TV cameras along a road, each covering one equal slice of it (ac_format.md §1.6).

    The road is cut into round(length / spacing_m) equal slices (at least one). Each camera
    stands at the middle of its slice, `offset_m` beyond the road edge on the outside of the
    bend there (alternating sides on straights), `height_m` above the road, aimed at the road
    point `look_ahead_m` further on. IN/OUT_POINT tile [0, 1] exactly: the first camera starts
    at 0, each one starts where the previous one ends, and the last ends at 1.

    Progress is arc length / road length, unless the AI spline is longer (e.g. it runs on past
    B): then pass its `spline_length_m` and the spline distance of the centreline's first point
    (`spline_offset_m`); the first and last cameras still stretch to 0 and 1.
    """
    pts = np.asarray(centreline, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise ValueError("centreline must be an (N >= 2, 3) array")
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])
    pts = pts[keep]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    length = float(s[-1])
    if length <= 0:
        raise ValueError("centreline has zero length")
    total = float(spline_length_m) if spline_length_m else length
    count = max(1, int(round(length / spacing_m)))
    bounds = np.linspace(0.0, length, count + 1)

    def at(d: float) -> np.ndarray:
        d = min(max(d, 0.0), length)
        return np.array([np.interp(d, s, pts[:, i]) for i in range(3)])

    def heading(d: float, h: float = 5.0) -> np.ndarray:
        a, b = at(d - h), at(d + h)
        t = np.array([b[0] - a[0], 0.0, b[2] - a[2]])
        return _unit(t)

    cams: list[Camera] = []
    for i in range(count):
        mid_s = 0.5 * (bounds[i] + bounds[i + 1])
        p = at(mid_s)
        t = heading(mid_s)
        left = np.array([t[2], 0.0, -t[0]])  # up x forward (coords.left_of)
        reach = min(25.0, 0.5 * (bounds[i + 1] - bounds[i]))
        turn = float(np.dot(heading(mid_s + reach) - heading(mid_s - reach), left))
        if abs(turn) > 0.02:
            side = -left if turn > 0 else left  # outside of a left bend is on the right
        else:
            side = left if i % 2 == 0 else -left
        pos = p + side * (road_half_width_m + offset_m) + np.array([0.0, height_m, 0.0])
        target = at(mid_s + look_ahead_m) + np.array([0.0, 0.5, 0.0])
        fwd = _unit(target - pos)
        up = _unit(np.array([0.0, 1.0, 0.0]) - fwd[1] * fwd)
        in_p = 0.0 if i == 0 else (spline_offset_m + bounds[i]) / total
        out_p = 1.0 if i == count - 1 else (spline_offset_m + bounds[i + 1]) / total
        cams.append(
            Camera(
                name=f"{name_prefix}{i}",
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                forward=(float(fwd[0]), float(fwd[1]), float(fwd[2])),
                up=(float(up[0]), float(up[1]), float(up[2])),
                in_point=float(np.clip(in_p, 0.0, 1.0)),
                out_point=float(np.clip(out_p, 0.0, 1.0)),
                min_fov=min_fov,
                max_fov=max_fov,
            )
        )
    return cams


# ---------------------------------------------------------------------------------------------
# data/lighting.ini, groove.ini, crew.ini; models.ini
# ---------------------------------------------------------------------------------------------


def write_lighting_ini(
    path: Path, sun_pitch_angle: float = 45.0, sun_heading_angle: float = 45.0
) -> Path:
    """data/lighting.ini (Hagnhofer; real files hold only these two keys).

    Per Hagnhofer, pitch 0 = zenith and 90 = horizon; heading turns counter-clockwise around
    the track (both ASSUMED, from docs only).
    """
    items = [
        ("SUN_PITCH_ANGLE", _num(sun_pitch_angle)),
        ("SUN_HEADING_ANGLE", _num(sun_heading_angle)),
    ]
    return _write_text(path, _ini([("LIGHTING", items)]))


def write_groove_ini(path: Path, grooves: int = 0) -> Path:
    """data/groove.ini; real files contain only `[HEADER] GROOVES_NUMBER=0`."""
    return _write_text(path, _ini([("HEADER", [("GROOVES_NUMBER", str(int(grooves)))])]))


def write_crew_ini(path: Path, side: int = -1) -> Path:
    """data/crew.ini: pit crew side, 1 = left, -1 = right (comment in a real Kunos file)."""
    if side not in (-1, 1):
        raise ValueError("crew side must be 1 (left) or -1 (right)")
    # The real file has the comment after the value; we put it on its own line so strict INI
    # readers (Python's configparser) see a clean value.
    text = "[HEADER]\n; 1 = left , -1 = right\n" + f"SIDE={side}\n"
    return _write_text(path, text)


def write_models_ini(path: Path, kn5_files: Sequence[str]) -> Path:
    """models.ini in the track root: one `[MODEL_n]` per KN5, all at the origin, unrotated.

    Format VERIFIED (code) from CM's `TrackComplexModelDescription` (FILE, POSITION, ROTATION as
    yaw, pitch, roll). Whether `<id>.kn5` must be listed too is spike S8; list it as MODEL_0.
    """
    if not kn5_files:
        raise ValueError("models.ini needs at least one KN5")
    sections = []
    for i, name in enumerate(kn5_files):
        _check_ascii(name, "KN5 file name")
        if not name.lower().endswith(".kn5") or "/" in name or "\\" in name:
            raise ValueError(f"expected a bare *.kn5 file name, got {name!r}")
        sections.append(
            (f"MODEL_{i}", [("FILE", name), ("POSITION", "0,0,0"), ("ROTATION", "0,0,0")])
        )
    return _write_text(path, _ini(sections))


# ---------------------------------------------------------------------------------------------
# LICENSE-DATA.txt and attribution
# ---------------------------------------------------------------------------------------------

KUNOS_NOTICE = (
    "Not affiliated with or endorsed by Kunos Simulazioni. Assetto Corsa is a trademark of "
    "Kunos Simulazioni, named only to state compatibility."
)

# Notices per data source (docs/research/geo.md §6). Copernicus wording is verbatim from the
# licence (Article 6, the "adapted or modified" form plus the required liability sentence).
DATA_SOURCES: dict[str, str] = {
    "osm": (
        "Road data: © OpenStreetMap contributors. Contains information from "
        "OpenStreetMap, which is made available here under the Open Database License (ODbL): "
        "https://opendatacommons.org/licenses/odbl/1-0/ . See "
        "https://www.openstreetmap.org/copyright . The route data used to build this track "
        "(geometry, OSM way IDs and tags) is included as route_osm.geojson."
    ),
    "copernicus": (
        "Elevation: produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and "
        "© Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the "
        "European Union and ESA; all rights reserved. The organisations in charge of the "
        "Copernicus programme by law or by delegation do not incur any liability for any use "
        "of the Copernicus WorldDEM-30."
    ),
    "usgs_3dep": (
        "Elevation: U.S. Geological Survey, 3D Elevation Program (public domain, "
        "https://www.usgs.gov/3d-elevation-program)."
    ),
    "ign": (
        "Elevation: IGN (Institut national de l'information géographique et "
        "forestière), RGE ALTI / LiDAR HD, Licence Ouverte / Open Licence 2.0 (Etalab)."
    ),
}

# One OSM credit when the buildings, barriers and street furniture come from OSM too (their
# ways and nodes are listed in route_osm.geojson next to the road's).
DATA_SOURCES["osm_scenery"] = (
    DATA_SOURCES["osm"]
    .replace("Road data:", "Road, barrier, street furniture and building data:")
    .replace(
        "The route data used to build this track (geometry, OSM way IDs and tags)",
        "The OSM data used to build this track (the road, buildings, barriers, signs and "
        "poles: geometry, OSM IDs and tags)",
    )
)

# ASCII one-liners for ui_track.json descriptions.
_SHORT_CREDITS: dict[str, str] = {
    "osm": "Road data (c) OpenStreetMap contributors, ODbL (openstreetmap.org/copyright).",
    "osm_scenery": (
        "Road and building data (c) OpenStreetMap contributors, ODbL (openstreetmap.org/copyright)."
    ),
    "copernicus": (
        "Elevation produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) Airbus "
        "Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and "
        "ESA; all rights reserved."
    ),
    "usgs_3dep": "Elevation: U.S. Geological Survey, 3D Elevation Program.",
    "ign": "Elevation: IGN, Licence Ouverte 2.0.",
}


def _check_sources(sources: Sequence[str]) -> list[str]:
    unknown = [s for s in sources if s not in DATA_SOURCES]
    if unknown:
        raise ValueError(f"unknown data sources {unknown}; known: {sorted(DATA_SOURCES)}")
    return list(dict.fromkeys(sources))


def attribution_text(sources: Sequence[str], generator: str = GENERATOR_NAME) -> str:
    """ASCII credit line for the ui_track.json description, e.g. for ["osm", "usgs_3dep"]."""
    parts = [_SHORT_CREDITS[s] for s in _check_sources(sources)]
    parts.append(f"Generated with {generator}. Not affiliated with Kunos Simulazioni.")
    return " ".join(parts)


def write_data_license(
    track_dir: Path,
    sources: Sequence[str],
    *,
    track_name: str | None = None,
    generator: str = GENERATOR_NAME,
    extra_notices: Sequence[str] = (),
) -> Path:
    """Write `<track_dir>/LICENSE-DATA.txt` (UTF-8) with every required notice.

    `sources` are keys of `DATA_SOURCES` ("osm", "copernicus", "usgs_3dep", "ign"); an empty
    list is fine for synthetic test tracks, which then carry only the generator credit and the
    Kunos non-affiliation line. Not legal advice (docs/research/geo.md §6).
    """
    keys = _check_sources(sources)
    title = (
        f"Data licences and credits: {track_name}" if track_name else "Data licences and credits"
    )
    lines = [title, "=" * len(title), ""]
    lines.append(
        f"This track was generated by {generator}. Its 3D model, textures, AI line and other "
        "files were produced by the generator; no Assetto Corsa game files are included."
    )
    if not keys:
        lines.append("It is a synthetic test track and uses no third-party map or elevation data.")
    lines.append("")
    for key in keys:
        lines += [DATA_SOURCES[key], ""]
    for notice in extra_notices:
        lines += [notice, ""]
    lines += [KUNOS_NOTICE, ""]
    path = Path(track_dir) / "LICENSE-DATA.txt"
    return _write_text(path, "\n".join(lines), encoding="utf-8")
