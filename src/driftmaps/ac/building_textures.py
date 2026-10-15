"""Procedural building atlas: siding, brick, barn boards, shingles, metal and flat roofs.

docs/research/scenery.md §6 ("Buildings: procedural siding, brick and shingle textures from
Pillow, one 512-1024 BC1 atlas") and §7 ("Textures"). Every pixel is generated here from seeded
noise with numpy, so we own the texture outright and the same size + seed gives the same bytes.

**Layout.** The atlas is `size` x `size` (default 1024, BC1) and holds `BAND_COUNT` (16)
horizontal bands, one per `BuildingStyle`, each the full atlas width and `size / 16` px tall.
A band repeats seamlessly left to right, so a mesh can let u run past 1 (wrap addressing, as the
road textures rely on) while v stays inside the band. That is what lets one material, one
texture and one draw call per chunk carry every building colour:

- **Walls**: one band = one storey (`STOREY_M`, 3 m) tall and `ATLAS_U_M` (48 m) wide, so the
  pixels are square (4.7 cm). u = metres along the wall / 48, v = position inside the storey;
  `geometry.buildings` cuts walls at storey lines so v never leaves the band. House bands carry
  windows (white trim, dark glass) at irregular spacing; barns have a few small ones; the shed
  band has none.
- **Roofs**: band top = ridge, band bottom = eave; u = metres along the contour / 48. A slope of
  any length is stretched to the band (shingle courses are not to scale; at scenery distance
  only the colour and the stripes read).
- `band_v(style)` returns the band's v range, inset by `INSET_PX` so bilinear filtering and the
  first mip levels do not bleed in the neighbouring band.

Image orientation: row 0 is the top of the image (UV v = 0 in DirectX/AC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from driftmaps.ac.model import Texture
from driftmaps.ac.textures import kn5_texture

__all__ = [
    "ATLAS_SEED",
    "ATLAS_SIZE",
    "ATLAS_U_M",
    "BAND_COUNT",
    "INSET_PX",
    "STOREY_M",
    "STYLES",
    "BuildingStyle",
    "band_v",
    "building_atlas",
    "building_atlas_texture",
    "style",
    "styles_for",
]

ATLAS_SIZE = 1024
ATLAS_SEED = 20261016
BAND_COUNT = 16
STOREY_M = 3.0  # a wall band is one storey tall
ATLAS_U_M = 48.0  # metres of wall/roof per atlas width (square pixels at 1024 px: 16 x 3 m)
INSET_PX = 1.5  # v inset at each band edge (texels at the top level)

Surface = Literal["wall", "roof"]


@dataclass(frozen=True)
class BuildingStyle:
    """One band of the atlas: `name`, its `band` index (0 = top), and what it is for."""

    name: str
    band: int
    surface: Surface


# Band order is part of the texture contract (meshes store v ranges): append, never reorder.
STYLES: tuple[BuildingStyle, ...] = (
    BuildingStyle("siding_white", 0, "wall"),
    BuildingStyle("siding_cream", 1, "wall"),
    BuildingStyle("siding_grey", 2, "wall"),
    BuildingStyle("siding_blue", 3, "wall"),
    BuildingStyle("brick_red", 4, "wall"),
    BuildingStyle("brick_brown", 5, "wall"),
    BuildingStyle("brick_buff", 6, "wall"),
    BuildingStyle("stucco", 7, "wall"),
    BuildingStyle("barn_red", 8, "wall"),
    BuildingStyle("shed_wood", 9, "wall"),
    BuildingStyle("commercial", 10, "wall"),
    BuildingStyle("shingle_charcoal", 11, "roof"),
    BuildingStyle("shingle_brown", 12, "roof"),
    BuildingStyle("metal_grey", 13, "roof"),
    BuildingStyle("metal_green", 14, "roof"),
    BuildingStyle("flat_membrane", 15, "roof"),
)
_BY_NAME = {s.name: s for s in STYLES}


def style(name: str) -> BuildingStyle:
    """The style called `name` (KeyError if unknown)."""
    return _BY_NAME[name]


def styles_for(surface: Surface) -> tuple[BuildingStyle, ...]:
    return tuple(s for s in STYLES if s.surface == surface)


def band_v(which: BuildingStyle | str | int, size: int = ATLAS_SIZE) -> tuple[float, float]:
    """(v_top, v_bottom) of a style's band in UV space, inset by `INSET_PX` texels."""
    if isinstance(which, str):
        band = style(which).band
    elif isinstance(which, BuildingStyle):
        band = which.band
    else:
        band = int(which)
    if not 0 <= band < BAND_COUNT:
        raise ValueError(f"band must be 0..{BAND_COUNT - 1}, got {band}")
    h = size / BAND_COUNT
    return (band * h + INSET_PX) / size, ((band + 1) * h - INSET_PX) / size


# ---------------------------------------------------------------------------------------------
# Painting helpers (all arrays are one band: (h, w) or (h, w, 3), float 0..255)
# ---------------------------------------------------------------------------------------------


def _noise(shape: tuple[int, int], rng: np.random.Generator, beta: float) -> np.ndarray:
    """Periodic 1/f^beta noise (zero mean, unit std): seamless left to right."""
    h, w = shape
    white = rng.standard_normal((h, w))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    amp = f ** (-beta / 2.0)
    amp[0, 0] = 0.0
    out = np.fft.ifft2(np.fft.fft2(white) * amp).real
    std = out.std()
    return (out - out.mean()) / (std if std > 0 else 1.0)


def _rgb(*c: float) -> np.ndarray:
    return np.array(c, dtype=np.float64)


def _shade(colour: np.ndarray, lum: np.ndarray) -> np.ndarray:
    """colour (3,) scaled by a luminance factor (h, w) -> (h, w, 3)."""
    return lum[..., None] * colour


def _window_spans(
    w: int, rng: np.random.Generator, px_per_m: float, gap_m: tuple[float, float],
    width_m: tuple[float, float],
) -> list[tuple[int, int]]:
    """Horizontal pixel spans [x0, x1) of windows across a band of width `w` (none crosses the
    wrap seam, so the band still tiles)."""
    spans: list[tuple[int, int]] = []
    x = rng.uniform(*gap_m) * px_per_m * 0.5
    while True:
        ww = rng.uniform(*width_m) * px_per_m
        if x + ww > w - 2:
            break
        spans.append((int(round(x)), int(round(x + ww))))
        x += ww + rng.uniform(*gap_m) * px_per_m
    return spans


def _paint_windows(
    rgb: np.ndarray,
    rng: np.random.Generator,
    px_per_m: float,
    *,
    gap_m: tuple[float, float] = (2.2, 5.5),
    width_m: tuple[float, float] = (0.9, 1.6),
    sill_m: float = 0.9,
    head_m: float = 2.2,
    trim: np.ndarray | None = None,
    glass: np.ndarray | None = None,
    skip: float = 0.0,
) -> None:
    """Paint windows (trim frame, dark glass with a sky reflection and a mullion) in place.

    Heights are metres above the storey floor (the band's bottom row)."""
    h, w = rgb.shape[:2]
    trim = _rgb(236, 234, 226) if trim is None else trim
    glass = _rgb(42, 50, 60) if glass is None else glass
    y_top = int(round(h - head_m * px_per_m))
    y_bot = int(round(h - sill_m * px_per_m))
    t = max(1, int(round(0.09 * px_per_m)))  # trim width ~9 cm
    for x0, x1 in _window_spans(w, rng, px_per_m, gap_m, width_m):
        if rng.random() < skip:
            continue
        rgb[y_top:y_bot, x0:x1] = trim
        gy = np.linspace(0.0, 1.0, max(1, y_bot - y_top - 2 * t))[:, None, None]
        pane = glass * (0.75 + 0.6 * (1.0 - gy)) + rng.normal(0.0, 4.0)  # sky glint at the top
        rgb[y_top + t : y_bot - t, x0 + t : x1 - t] = pane
        mid = (x0 + x1) // 2
        rgb[y_top + t : y_bot - t, mid : mid + max(1, t // 2)] = trim
        rgb[y_bot : y_bot + t, x0 - t : x1 + t] = trim * 0.92  # sill


def _siding(shape, rng, colour: np.ndarray, px_per_m: float) -> np.ndarray:
    """Horizontal lap siding: ~20 cm boards, each darker at its lower edge (shadow line)."""
    h, w = shape
    board = 0.2 * px_per_m
    y = np.arange(h)[:, None] + 0.5
    phase = (y / board) % 1.0
    lum = 1.0 - 0.16 * phase**3 - 0.10 * (phase < 0.12)
    lum = lum + 0.025 * _noise(shape, rng, 2.5) + 0.015 * _noise(shape, rng, 0.5)
    rgb = _shade(colour, np.broadcast_to(lum, shape))
    _paint_windows(rgb, rng, px_per_m)
    return rgb


def _brick(shape, rng, colour: np.ndarray, px_per_m: float, *, windows: bool = True):
    """Running-bond brick with light mortar; per-brick tone jitter. Bricks are only ~2 px tall
    at the default size, so the pattern is painted at 4x and box-filtered down (no aliasing)."""
    h, w = shape
    ss = 4
    course = 0.075 * px_per_m * ss
    n_cols = max(1, int(round(w / (0.2 * px_per_m))))  # whole bricks per row: seamless wrap
    length = w * ss / n_cols
    y = np.arange(h * ss)[:, None] + 0.5
    x = np.arange(w * ss)[None, :] + 0.5
    row = np.floor(y / course).astype(np.int64)
    xs = x + (row % 2) * length * 0.5
    col = np.floor(xs / length).astype(np.int64) % n_cols
    tone = rng.normal(0.0, 0.07, (int(row.max()) + 1, n_cols))
    lum_hi = 1.0 + tone[row, col]
    mortar = ((y / course) % 1.0 < 0.2) | ((xs / length) % 1.0 < 0.05)
    mix = np.where(mortar, 1.0, 0.0)
    lum = lum_hi.reshape(h, ss, w, ss).mean(axis=(1, 3))
    m = mix.reshape(h, ss, w, ss).mean(axis=(1, 3))
    lum = lum + 0.03 * _noise(shape, rng, 1.0)
    rgb = _shade(colour, lum) * (1.0 - m[..., None]) + _rgb(176, 170, 160) * m[..., None]
    if windows:
        _paint_windows(rgb, rng, px_per_m, trim=_rgb(225, 222, 212))
    return rgb


def _stucco(shape, rng, colour: np.ndarray, px_per_m: float) -> np.ndarray:
    lum = 1.0 + 0.035 * _noise(shape, rng, 2.8) + 0.03 * _noise(shape, rng, 0.2)
    rgb = _shade(colour, lum)
    _paint_windows(rgb, rng, px_per_m, trim=_rgb(214, 206, 190))
    return rgb


def _boards(shape, rng, colour: np.ndarray, px_per_m: float, *, windows: bool) -> np.ndarray:
    """Vertical board-and-batten barn siding, weathered."""
    h, w = shape
    board = 0.3 * px_per_m
    x = np.arange(w)[None, :] + 0.5
    phase = (x / board) % 1.0
    n_boards = int(np.ceil(w / board)) + 1
    tone = rng.normal(0.0, 0.06, n_boards)[np.floor(x / board).astype(np.int64) % n_boards]
    lum = 1.0 + tone - 0.22 * (phase < 0.1) + 0.06 * (phase > 0.9)
    streaks = _noise(shape, rng, 1.2) * np.linspace(0.5, 1.0, h)[:, None]
    lum = lum + 0.05 * streaks + 0.02 * _noise(shape, rng, 3.0)
    rgb = _shade(colour, np.broadcast_to(lum, shape))
    if windows:
        _paint_windows(
            rgb, rng, px_per_m, gap_m=(7.0, 14.0), width_m=(0.6, 0.9), sill_m=1.4, head_m=2.2,
            trim=_rgb(228, 224, 214), skip=0.3,
        )
    return rgb


def _commercial(shape, rng, px_per_m: float) -> np.ndarray:
    """Concrete block / panel wall with a band of wide windows."""
    h, w = shape
    y = np.arange(h)[:, None] + 0.5
    x = np.arange(w)[None, :] + 0.5
    block_h, block_w = 0.2 * px_per_m, 0.4 * px_per_m
    joint = ((y / block_h) % 1.0 < 0.1) | (((x + (np.floor(y / block_h) % 2) * block_w / 2)
                                            / block_w) % 1.0 < 0.05)
    lum = 1.0 + 0.04 * _noise(shape, rng, 2.0) + 0.02 * _noise(shape, rng, 0.3) - 0.08 * joint
    rgb = _shade(_rgb(178, 176, 170), lum)
    _paint_windows(
        rgb, rng, px_per_m, gap_m=(1.0, 4.0), width_m=(2.5, 5.0), sill_m=1.0, head_m=2.5,
        trim=_rgb(120, 124, 128), glass=_rgb(48, 60, 72),
    )
    return rgb


def _shingles(shape, rng, colour: np.ndarray, courses: int = 12) -> np.ndarray:
    """Asphalt shingle courses (band top = ridge): tab slots offset per course, granule noise."""
    h, w = shape
    course = h / courses
    y = np.arange(h)[:, None] + 0.5
    x = np.arange(w)[None, :] + 0.5
    row = np.floor(y / course)
    tab = 1.6 * course
    slots = ((x + (row % 2) * tab * 0.5) / tab) % 1.0 < 0.05
    phase = (y / course) % 1.0
    lum = 1.0 - 0.18 * phase**2 - 0.15 * slots * (phase > 0.4)
    n_rows = courses + 1
    n_tabs = int(np.ceil(w / tab)) + 2
    tone = rng.normal(0.0, 0.05, (n_rows, n_tabs))
    lum = lum + tone[row.astype(np.int64) % n_rows,
                     np.floor((x + (row % 2) * tab * 0.5) / tab).astype(np.int64) % n_tabs]
    lum = lum + 0.06 * _noise(shape, rng, 0.0) + 0.03 * _noise(shape, rng, 2.5)
    return _shade(colour, lum)


def _metal(shape, rng, colour: np.ndarray, px_per_m: float) -> np.ndarray:
    """Standing-seam / ribbed steel: seams down the slope every ~0.45 m."""
    h, w = shape
    pitch = 0.45 * px_per_m
    x = np.arange(w)[None, :] + 0.5
    phase = (x / pitch) % 1.0
    lum = 1.0 + 0.12 * (phase < 0.08) - 0.07 * ((phase > 0.08) & (phase < 0.16))
    lum = lum + 0.025 * _noise(shape, rng, 3.0) + 0.012 * _noise(shape, rng, 0.5)
    lum = lum + 0.04 * np.linspace(-1.0, 1.0, h)[:, None]  # brighter towards the eave
    return _shade(colour, np.broadcast_to(lum, shape))


def _membrane(shape, rng) -> np.ndarray:
    """Flat roof: grey membrane / ballast gravel with large stains."""
    lum = 1.0 + 0.05 * _noise(shape, rng, 3.0) + 0.06 * _noise(shape, rng, 0.0)
    return _shade(_rgb(150, 148, 142), lum)


def _paint_band(s: BuildingStyle, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    px_per_m = h / STOREY_M
    n = s.name
    if n.startswith("siding_"):
        colour = {
            "siding_white": _rgb(226, 224, 216),
            "siding_cream": _rgb(214, 200, 166),
            "siding_grey": _rgb(160, 162, 160),
            "siding_blue": _rgb(132, 150, 166),
        }[n]
        return _siding(shape, rng, colour, px_per_m)
    if n.startswith("brick_"):
        colour = {
            "brick_red": _rgb(150, 70, 52),
            "brick_brown": _rgb(118, 82, 62),
            "brick_buff": _rgb(196, 168, 120),
        }[n]
        return _brick(shape, rng, colour, px_per_m)
    if n == "stucco":
        return _stucco(shape, rng, _rgb(206, 194, 170), px_per_m)
    if n == "barn_red":
        return _boards(shape, rng, _rgb(128, 46, 36), px_per_m, windows=True)
    if n == "shed_wood":
        return _boards(shape, rng, _rgb(122, 116, 106), px_per_m, windows=False)
    if n == "commercial":
        return _commercial(shape, rng, px_per_m)
    if n == "shingle_charcoal":
        return _shingles(shape, rng, _rgb(64, 64, 66))
    if n == "shingle_brown":
        return _shingles(shape, rng, _rgb(92, 72, 58))
    if n == "metal_grey":
        return _metal(shape, rng, _rgb(150, 154, 156), px_per_m)
    if n == "metal_green":
        return _metal(shape, rng, _rgb(62, 92, 66), px_per_m)
    if n == "flat_membrane":
        return _membrane(shape, rng)
    raise ValueError(f"no painter for style {n!r}")


def building_atlas(size: int = ATLAS_SIZE, seed: int = ATLAS_SEED) -> Image.Image:
    """The RGB atlas: `BAND_COUNT` horizontal bands, one per style (module docstring)."""
    if size < 64 or size & (size - 1):
        raise ValueError(f"atlas size must be a power of two >= 64, got {size}")
    band_h = size // BAND_COUNT
    out = np.zeros((size, size, 3))
    for s in STYLES:
        rng = np.random.default_rng([seed, s.band])
        out[s.band * band_h : (s.band + 1) * band_h] = _paint_band(s, (band_h, size), rng)
    arr = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def building_atlas_texture(
    name: str = "dm_buildings.dds", size: int = ATLAS_SIZE, seed: int = ATLAS_SEED
) -> Texture:
    """The atlas as a KN5 texture: BC1 DDS with mips for "*.dds" names, PNG for "*.png"."""
    return kn5_texture(name, building_atlas(size, seed), "BC1")
