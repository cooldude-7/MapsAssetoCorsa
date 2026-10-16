"""Procedural building atlas for Ontario: vinyl siding, brick, barn boards, steel cladding,
asphalt shingles, metal roofs, and doors.

docs/research/scenery.md §6 ("Buildings: procedural siding, brick and shingle textures from
Pillow, one BC1 atlas") and the builders' buildings.md §7 ("keep procedural for asphalt
shingles and vinyl siding"). Every pixel is generated here from seeded noise with numpy, so we
own the texture outright and the same size + seed gives the same bytes. Colours are ASSUMED
typical Ontario finishes (no cladding statistics exist): vinyl in white, almond, grey, clay,
blue-grey and sage; red, brown, buff, orange and charcoal brick; fieldstone; red and grey barn
boards; white ribbed steel for drive sheds; charcoal, grey, weathered-wood and black asphalt
shingles; galvanised, green, red and charcoal steel roofs.

**Layout.** The atlas is `size` wide and 2 x `size` tall (default 1024 x 2048, BC1, 1.4 MB
with mips) and holds `BAND_COUNT` (32) horizontal bands, one per `BuildingStyle`, each the full
atlas width and `size / 16` px tall. A band repeats seamlessly left to right, so a mesh can let
u run past 1 (wrap addressing) while v stays inside the band. One material, one texture and one
draw call per batch carry every building colour:

- **Walls**: one band = one storey (`STOREY_M`, 3 m) tall and `ATLAS_U_M` (48 m) wide, so the
  pixels are square (4.7 cm). u = metres along the wall / 48, v = position inside the storey;
  `geometry.buildings` cuts walls at storey lines so v never leaves the band. Windowed bands
  (siding, brick, stone, stucco) share ONE window layout: a window centred every
  `WINDOW_PERIOD_M` (4 m) over the first `WINDOW_ZONE_M` (36 m), then 12 m of blank wall, so
  the mesh can centre whole windows on any wall (`wall_u_start`) or give a garage side a
  blank wall. Barns have a few small windows, steel and shed bands none.
- **Roofs**: band top = ridge, band bottom = eave; u = metres along the contour / 48.
- **Doors** (`DOORS`): one band holds front doors, single and double garage doors, a sliding
  barn door and a steel overhead door, each a u range (metres) and a painted height from the
  band bottom; `geometry.buildings` maps door quads onto them.
- `band_v(style)` returns the band's v range, inset by `INSET_PX` so bilinear filtering and the
  first mip levels do not bleed in the neighbouring band.

Image orientation: row 0 is the top of the image (UV v = 0 in DirectX/AC).
"""

from __future__ import annotations

import math
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
    "DOORS",
    "INSET_PX",
    "STOREY_M",
    "STYLES",
    "WINDOW_PERIOD_M",
    "WINDOW_ZONE_M",
    "WINDOWED",
    "BuildingStyle",
    "DoorStyle",
    "band_v",
    "building_atlas",
    "building_atlas_texture",
    "door_uv",
    "style",
    "styles_for",
    "wall_u_start",
]

ATLAS_SIZE = 1024  # width; the atlas is twice as tall
ATLAS_SEED = 20261016
BAND_COUNT = 32
STOREY_M = 3.0  # a wall band is one storey tall
ATLAS_U_M = 48.0  # metres of wall/roof per atlas width (square pixels at 1024 px: 16 x 3 m)
INSET_PX = 1.5  # v inset at each band edge (texels at the top level)
WINDOW_PERIOD_M = 4.0
WINDOW_ZONE_M = 36.0  # 9 windows, then blank wall to 48 m
_WINDOW_WIDTHS_M = (1.2, 1.5, 1.0, 1.7, 1.2, 1.4, 0.9, 1.8, 1.3)  # one per period
_SILL_M, _HEAD_M = 0.9, 2.2

Surface = Literal["wall", "roof", "detail"]


@dataclass(frozen=True)
class BuildingStyle:
    """One band of the atlas: `name`, its `band` index (0 = top), and what it is for."""

    name: str
    band: int
    surface: Surface


_STYLE_NAMES: tuple[tuple[str, Surface], ...] = (
    ("siding_white", "wall"),
    ("siding_cream", "wall"),  # almond / beige vinyl
    ("siding_grey", "wall"),
    ("siding_blue", "wall"),  # blue-grey vinyl
    ("siding_sage", "wall"),
    ("siding_tan", "wall"),  # clay vinyl
    ("siding_brown", "wall"),  # dark brown vinyl / stained wood
    ("board_white", "wall"),  # white vertical board-and-batten (farmhouse)
    ("brick_red", "wall"),
    ("brick_brown", "wall"),
    ("brick_buff", "wall"),
    ("brick_orange", "wall"),
    ("brick_charcoal", "wall"),
    ("stone", "wall"),  # fieldstone / ledgestone
    ("stucco", "wall"),
    ("barn_red", "wall"),
    ("barn_grey", "wall"),  # weathered grey boards
    ("shed_wood", "wall"),  # brown stained boards, no windows
    ("steel_white", "wall"),  # ribbed steel cladding: drive sheds, pole barns
    ("steel_red", "wall"),
    ("commercial", "wall"),
    ("shingle_charcoal", "roof"),
    ("shingle_grey", "roof"),  # weathered grey
    ("shingle_brown", "roof"),  # weathered wood
    ("shingle_black", "roof"),
    ("metal_grey", "roof"),  # galvanised / galvalume
    ("metal_green", "roof"),
    ("metal_red", "roof"),
    ("metal_charcoal", "roof"),
    ("metal_blue", "roof"),
    ("flat_membrane", "roof"),
    ("doors", "detail"),
)
STYLES: tuple[BuildingStyle, ...] = tuple(
    BuildingStyle(n, i, s) for i, (n, s) in enumerate(_STYLE_NAMES)
)
_BY_NAME = {s.name: s for s in STYLES}
WINDOWED = frozenset({
    "siding_white", "siding_cream", "siding_grey", "siding_blue", "siding_sage", "siding_tan",
    "siding_brown", "board_white", "brick_red", "brick_brown", "brick_buff", "brick_orange",
    "brick_charcoal", "stone", "stucco",
})  # fmt: skip


@dataclass(frozen=True)
class DoorStyle:
    """A door on the doors band: u range `u0..u1` (metres, the painted door + its trim) and
    the painted `height_m` from the band bottom; `width_m` = u1 - u0."""

    name: str
    u0: float
    u1: float
    height_m: float

    @property
    def width_m(self) -> float:
        return self.u1 - self.u0


DOORS: dict[str, DoorStyle] = {
    d.name: d
    for d in (
        DoorStyle("door_white", 0.2, 1.4, 2.3),
        DoorStyle("door_dark", 1.8, 3.0, 2.3),
        DoorStyle("door_red", 3.4, 4.6, 2.3),
        DoorStyle("garage1_white", 5.2, 8.2, 2.5),
        DoorStyle("garage2_white", 8.8, 14.4, 2.5),
        DoorStyle("garage1_almond", 15.0, 18.0, 2.5),
        DoorStyle("garage2_almond", 18.6, 24.2, 2.5),
        DoorStyle("barn_door", 25.0, 29.4, 3.0),
        DoorStyle("steel_door", 30.0, 34.0, 3.0),
    )
}


def style(name: str) -> BuildingStyle:
    """The style called `name` (KeyError if unknown)."""
    return _BY_NAME[name]


def styles_for(surface: Surface) -> tuple[BuildingStyle, ...]:
    return tuple(s for s in STYLES if s.surface == surface)


def band_v(which: BuildingStyle | str | int, size: int = ATLAS_SIZE) -> tuple[float, float]:
    """(v_top, v_bottom) of a style's band in UV space, inset by `INSET_PX` texels. `size` is
    the atlas WIDTH (its height is 2 x size)."""
    if isinstance(which, str):
        band = style(which).band
    elif isinstance(which, BuildingStyle):
        band = which.band
    else:
        band = int(which)
    if not 0 <= band < BAND_COUNT:
        raise ValueError(f"band must be 0..{BAND_COUNT - 1}, got {band}")
    height = 2 * size
    h = height / BAND_COUNT
    return (band * h + INSET_PX) / height, ((band + 1) * h - INSET_PX) / height


def door_uv(name: str, size: int = ATLAS_SIZE) -> tuple[float, float, float, float]:
    """(u0, u1, v_top, v_bottom) of a door's painted rectangle (`DOORS[name]`)."""
    d = DOORS[name]
    v_top, v_bot = band_v("doors", size)
    span = v_bot - v_top
    return d.u0 / ATLAS_U_M, d.u1 / ATLAS_U_M, v_bot - span * d.height_m / STOREY_M, v_bot


def wall_u_start(length_m: float, key: float, windows: bool = True) -> float:
    """Where (metres along a windowed band) a wall of `length_m` should start so that it
    shows whole windows, centred (`key` in [0, 1) picks which of the band's windows), or,
    with `windows=False` or a wall too short for one (< 2.4 m), only blank wall."""
    if not windows or length_m < 2.4:
        if length_m <= ATLAS_U_M - WINDOW_ZONE_M - 0.4:
            return WINDOW_ZONE_M + 0.5 * (ATLAS_U_M - WINDOW_ZONE_M - length_m)
        return WINDOW_ZONE_M
    slots = int(WINDOW_ZONE_M // WINDOW_PERIOD_M)
    n = max(1, min(slots, int(length_m // WINDOW_PERIOD_M)))
    if length_m > WINDOW_ZONE_M:
        return 0.0
    k0 = min(slots - n, int(key * (slots - n + 1)))
    centre = WINDOW_PERIOD_M * (k0 + 0.5 * n)
    return centre - 0.5 * length_m


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


def _house_window_spans(px_per_m: float) -> list[tuple[int, int]]:
    """The shared window layout (module docstring): pixel spans [x0, x1)."""
    spans = []
    for k, wm in enumerate(_WINDOW_WIDTHS_M):
        c = (k + 0.5) * WINDOW_PERIOD_M * px_per_m
        spans.append((int(round(c - 0.5 * wm * px_per_m)), int(round(c + 0.5 * wm * px_per_m))))
    return spans


def _random_spans(
    w: int,
    rng: np.random.Generator,
    px_per_m: float,
    gap_m: tuple[float, float],
    width_m: tuple[float, float],
) -> list[tuple[int, int]]:
    """Irregular spans [x0, x1) (none crosses the wrap seam, so the band still tiles)."""
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
    spans: list[tuple[int, int]],
    *,
    sill_m: float = _SILL_M,
    head_m: float = _HEAD_M,
    trim: np.ndarray | None = None,
    glass: np.ndarray | None = None,
    skip: float = 0.0,
    grid: bool = False,
) -> None:
    """Paint windows (trim frame, dark glass with a sky reflection, a mullion, a sill) in
    place. Heights are metres above the storey floor (the band's bottom row)."""
    h = rgb.shape[0]
    trim = _rgb(236, 234, 226) if trim is None else trim
    glass = _rgb(42, 50, 60) if glass is None else glass
    y_top = int(round(h - head_m * px_per_m))
    y_bot = int(round(h - sill_m * px_per_m))
    t = max(1, int(round(0.09 * px_per_m)))  # trim width ~9 cm
    for x0, x1 in spans:
        if rng.random() < skip:
            continue
        rgb[y_top:y_bot, x0:x1] = trim
        gy = np.linspace(0.0, 1.0, max(1, y_bot - y_top - 2 * t))[:, None, None]
        pane = glass * (0.75 + 0.6 * (1.0 - gy)) + rng.normal(0.0, 4.0)  # sky glint at the top
        rgb[y_top + t : y_bot - t, x0 + t : x1 - t] = pane
        mid = (x0 + x1) // 2
        rgb[y_top + t : y_bot - t, mid : mid + max(1, t // 2)] = trim
        if grid:  # a transom bar across the upper third
            yy = y_top + (y_bot - y_top) // 3
            rgb[yy : yy + max(1, t // 2), x0 + t : x1 - t] = trim
        rgb[y_bot : y_bot + t, max(0, x0 - t) : x1 + t] = trim * 0.92  # sill


def _siding(shape, rng, colour: np.ndarray, px_per_m: float, *, trim=None) -> np.ndarray:
    """Horizontal vinyl lap siding: ~18 cm boards, each darker at its lower edge."""
    h, w = shape
    board = 0.18 * px_per_m
    y = np.arange(h)[:, None] + 0.5
    phase = (y / board) % 1.0
    lum = 1.0 - 0.14 * phase**3 - 0.09 * (phase < 0.12)
    lum = lum + 0.02 * _noise(shape, rng, 2.5) + 0.012 * _noise(shape, rng, 0.5)
    rgb = _shade(colour, np.broadcast_to(lum, shape))
    _paint_windows(rgb, rng, px_per_m, _house_window_spans(px_per_m), trim=trim)
    return rgb


def _board_batten(shape, rng, colour: np.ndarray, px_per_m: float) -> np.ndarray:
    h, w = shape
    period = 0.4 * px_per_m
    x = np.arange(w)[None, :] + 0.5
    n = max(1, int(round(w / period)))
    phase = (x * n / w) % 1.0
    lum = 1.0 - 0.07 * (phase < 0.12) + 0.04 * ((phase > 0.12) & (phase < 0.2))
    lum = lum + 0.015 * _noise(shape, rng, 2.0)
    rgb = _shade(colour, np.broadcast_to(lum, shape))
    _paint_windows(rgb, rng, px_per_m, _house_window_spans(px_per_m), trim=_rgb(40, 40, 42),
                   glass=_rgb(36, 42, 50), grid=True)  # fmt: skip
    return rgb


def _brick(shape, rng, colour: np.ndarray, px_per_m: float, *, spread: float = 0.07):
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
    tone = rng.normal(0.0, spread, (int(row.max()) + 1, n_cols))
    lum_hi = 1.0 + tone[row, col]
    mortar = ((y / course) % 1.0 < 0.2) | ((xs / length) % 1.0 < 0.05)
    mix = np.where(mortar, 1.0, 0.0)
    lum = lum_hi.reshape(h, ss, w, ss).mean(axis=(1, 3))
    m = mix.reshape(h, ss, w, ss).mean(axis=(1, 3))
    lum = lum + 0.03 * _noise(shape, rng, 1.0)
    rgb = _shade(colour, lum) * (1.0 - m[..., None]) + _rgb(176, 170, 160) * m[..., None]
    _paint_windows(rgb, rng, px_per_m, _house_window_spans(px_per_m), trim=_rgb(225, 222, 212))
    return rgb


def _stone(shape, rng, px_per_m: float) -> np.ndarray:
    """Fieldstone: irregular stones (Voronoi-ish cells from jittered rows) in grey-brown."""
    h, w = shape
    ss = 2
    hh, ww = h * ss, w * ss
    row_h = 0.22 * px_per_m * ss
    n_rows = int(math.ceil(hh / row_h)) + 1
    y = np.arange(hh)[:, None] + 0.5
    x = np.arange(ww)[None, :] + 0.5
    row = np.floor(y / row_h).astype(np.int64)
    n_cols = max(4, int(round(w / (0.45 * px_per_m))))
    cell = ww / n_cols
    jitter = rng.uniform(0.0, 1.0, (n_rows, n_cols))
    xs = (x + jitter[row % n_rows, 0:1].reshape(-1, 1) * cell) % ww
    col = np.floor(xs / cell).astype(np.int64) % n_cols
    tones = rng.normal(0.0, 0.13, (n_rows, n_cols))
    hue = rng.uniform(0.0, 1.0, (n_rows, n_cols))
    lum = 1.0 + tones[row % n_rows, col]
    warm = hue[row % n_rows, col]
    base = _rgb(150, 144, 134)[None, None, :] * (1 - warm[..., None] * 0.25) + _rgb(
        168, 140, 110)[None, None, :] * (warm[..., None] * 0.25)  # fmt: skip
    mortar = ((y / row_h) % 1.0 < 0.12) | ((xs / cell) % 1.0 < 0.06)
    rgb = base * lum[..., None]
    rgb = np.where(mortar[..., None], _rgb(118, 114, 108), rgb)
    rgb = rgb.reshape(h, ss, w, ss, 3).mean(axis=(1, 3))
    rgb *= (1.0 + 0.04 * _noise(shape, rng, 1.0))[..., None]
    _paint_windows(rgb, rng, px_per_m, _house_window_spans(px_per_m), trim=_rgb(222, 218, 206))
    return rgb


def _stucco(shape, rng, colour: np.ndarray, px_per_m: float) -> np.ndarray:
    lum = 1.0 + 0.035 * _noise(shape, rng, 2.8) + 0.03 * _noise(shape, rng, 0.2)
    rgb = _shade(colour, lum)
    _paint_windows(rgb, rng, px_per_m, _house_window_spans(px_per_m), trim=_rgb(214, 206, 190))
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
        spans = _random_spans(w, rng, px_per_m, (7.0, 14.0), (0.6, 0.9))
        _paint_windows(rgb, rng, px_per_m, spans, sill_m=1.4, head_m=2.2,
                       trim=_rgb(228, 224, 214), skip=0.3)  # fmt: skip
    return rgb


def _steel_wall(shape, rng, colour: np.ndarray, px_per_m: float) -> np.ndarray:
    """Ribbed steel cladding (vertical ribs every ~23 cm) with a dark base trim."""
    h, w = shape
    n = max(1, int(round(w / (0.229 * px_per_m))))
    x = np.arange(w)[None, :] + 0.5
    phase = (x * n / w) % 1.0
    lum = 1.0 + 0.08 * (phase < 0.12) - 0.08 * ((phase > 0.12) & (phase < 0.24))
    lum = lum + 0.015 * _noise(shape, rng, 3.0) + 0.012 * _noise(shape, rng, 0.5)
    rgb = _shade(colour, np.broadcast_to(lum, shape))
    rgb[h - max(1, int(0.12 * px_per_m)) :] *= 0.55  # base trim
    return rgb


def _commercial(shape, rng, px_per_m: float) -> np.ndarray:
    """Concrete block / panel wall with a band of wide windows."""
    h, w = shape
    y = np.arange(h)[:, None] + 0.5
    x = np.arange(w)[None, :] + 0.5
    block_h, block_w = 0.2 * px_per_m, 0.4 * px_per_m
    joint = ((y / block_h) % 1.0 < 0.1) | (
        ((x + (np.floor(y / block_h) % 2) * block_w / 2) / block_w) % 1.0 < 0.05
    )
    lum = 1.0 + 0.04 * _noise(shape, rng, 2.0) + 0.02 * _noise(shape, rng, 0.3) - 0.08 * joint
    rgb = _shade(_rgb(178, 176, 170), lum)
    spans = _random_spans(w, rng, px_per_m, (1.0, 4.0), (2.5, 5.0))
    _paint_windows(rgb, rng, px_per_m, spans, sill_m=1.0, head_m=2.5,
                   trim=_rgb(120, 124, 128), glass=_rgb(48, 60, 72))  # fmt: skip
    return rgb


def _shingles(shape, rng, colour: np.ndarray, courses: int = 12, blend: float = 0.06):
    """Asphalt shingle courses (band top = ridge): tab slots offset per course, granule noise
    and a few lighter / darker granule blends."""
    h, w = shape
    course = h / courses
    y = np.arange(h)[:, None] + 0.5
    x = np.arange(w)[None, :] + 0.5
    row = np.floor(y / course)
    tab = 1.6 * course
    slots = ((x + (row % 2) * tab * 0.5) / tab) % 1.0 < 0.05
    phase = (y / course) % 1.0
    lum = 1.0 - 0.16 * phase**2 - 0.15 * slots * (phase > 0.4)
    n_rows = courses + 1
    n_tabs = int(np.ceil(w / tab)) + 2
    tone = rng.normal(0.0, blend, (n_rows, n_tabs))
    lum = (
        lum
        + tone[
            row.astype(np.int64) % n_rows,
            np.floor((x + (row % 2) * tab * 0.5) / tab).astype(np.int64) % n_tabs,
        ]
    )
    lum = lum + 0.05 * _noise(shape, rng, 0.0) + 0.03 * _noise(shape, rng, 2.5)
    return _shade(colour, lum)


def _metal(shape, rng, colour: np.ndarray, px_per_m: float) -> np.ndarray:
    """Ribbed steel roofing: ribs down the slope every ~0.23 m, a major rib every 0.9 m."""
    h, w = shape
    n = max(1, int(round(w / (0.914 * px_per_m))))
    x = np.arange(w)[None, :] + 0.5
    phase = (x * n / w) % 1.0
    lum = 1.0 + 0.12 * (phase < 0.06) - 0.07 * ((phase > 0.06) & (phase < 0.12))
    minor = ((phase * 4.0) % 1.0) < 0.06
    lum = lum + 0.04 * minor
    lum = lum + 0.025 * _noise(shape, rng, 3.0) + 0.012 * _noise(shape, rng, 0.5)
    lum = lum + 0.04 * np.linspace(-1.0, 1.0, h)[:, None]  # brighter towards the eave
    return _shade(colour, np.broadcast_to(lum, shape))


def _membrane(shape, rng) -> np.ndarray:
    """Flat roof: grey membrane / ballast gravel with large stains."""
    lum = 1.0 + 0.05 * _noise(shape, rng, 3.0) + 0.06 * _noise(shape, rng, 0.0)
    return _shade(_rgb(150, 148, 142), lum)


def _doors(shape, rng, px_per_m: float) -> np.ndarray:
    """The doors band (`DOORS`): neutral wall between the painted doors."""
    h, w = shape
    rgb = _shade(_rgb(150, 148, 144), 1.0 + 0.02 * _noise(shape, rng, 2.0))

    def px(m: float) -> int:
        return int(round(m * px_per_m))

    def box(d: DoorStyle) -> tuple[int, int, int, int]:
        return px(d.u0), px(d.u1), h - px(d.height_m), h

    trim_w = max(1, px(0.1))
    for name, colour in (("door_white", _rgb(232, 230, 224)), ("door_dark", _rgb(52, 46, 42)),
                         ("door_red", _rgb(128, 36, 34))):  # fmt: skip
        x0, x1, y0, y1 = box(DOORS[name])
        rgb[y0:y1, x0:x1] = _rgb(238, 236, 230)  # trim
        dx0, dx1, dy0 = x0 + trim_w, x1 - trim_w, y0 + trim_w
        rgb[dy0:y1, dx0:dx1] = colour
        # Four raised panels and a small window at the top.
        pw = (dx1 - dx0) // 2
        for k in range(2):
            for j, (a, b) in enumerate(((0.12, 0.42), (0.52, 0.92))):
                ya, yb = dy0 + int(a * (y1 - dy0)), dy0 + int(b * (y1 - dy0))
                xa, xb = dx0 + k * pw + max(1, pw // 5), dx0 + (k + 1) * pw - max(1, pw // 5)
                if j == 0:
                    rgb[ya:yb, xa:xb] = _rgb(60, 70, 82)  # glass
                else:
                    rgb[ya:yb, xa:xb] = colour * 0.88
        knob_y, knob_x = dy0 + int(0.55 * (y1 - dy0)), dx1 - max(2, pw // 4)
        rgb[knob_y : knob_y + 2, knob_x : knob_x + 2] = _rgb(200, 180, 120)
    for name, colour in (("garage1_white", _rgb(236, 236, 232)),
                         ("garage2_white", _rgb(236, 236, 232)),
                         ("garage1_almond", _rgb(214, 204, 182)),
                         ("garage2_almond", _rgb(214, 204, 182))):  # fmt: skip
        x0, x1, y0, y1 = box(DOORS[name])
        rgb[y0:y1, x0:x1] = _rgb(240, 238, 232)  # trim
        dx0, dx1, dy0 = x0 + trim_w, x1 - trim_w, y0 + trim_w
        door = np.full((y1 - dy0, dx1 - dx0, 3), colour)
        sec = (y1 - dy0) / 4.0
        yy = np.arange(y1 - dy0)[:, None] + 0.5
        joint = (yy % sec) < 1.0
        door *= np.where(joint, 0.78, 1.0)[..., None]
        n_pan = max(2, int(round((dx1 - dx0) / px(0.7))))
        pwid = (dx1 - dx0) / n_pan
        xx = np.arange(dx1 - dx0)[None, :] + 0.5
        inner = (((xx % pwid) > 0.12 * pwid) & ((xx % pwid) < 0.88 * pwid)
                 & ((yy % sec) > 0.2 * sec) & ((yy % sec) < 0.8 * sec))  # fmt: skip
        door *= np.where(inner, 0.94, 1.0)[..., None]
        top = int(sec)  # windows in the top section
        door[2 : max(3, top - 2)] = np.where(
            inner[2 : max(3, top - 2), :, None], _rgb(58, 66, 78), door[2 : max(3, top - 2)]
        )
        rgb[dy0:y1, dx0:dx1] = door
    x0, x1, y0, y1 = box(DOORS["barn_door"])
    bw = _boards((y1 - y0, x1 - x0), rng, _rgb(118, 38, 30), px_per_m, windows=False)
    yy, xx = np.mgrid[0 : y1 - y0, 0 : x1 - x0]
    hh, ww = y1 - y0, x1 - x0
    t = max(1, px(0.12))
    mid = np.abs(xx - ww // 2) < t // 2 + 1
    edge = (yy < t) | (yy >= hh - t) | (xx < t) | (xx >= ww - t) | mid
    diag = (np.abs(yy / hh - (xx % (ww / 2)) / (ww / 2)) < 0.05) | (
        np.abs(yy / hh - 1.0 + (xx % (ww / 2)) / (ww / 2)) < 0.05)  # fmt: skip
    bw[edge | diag] = _rgb(232, 228, 218)
    rgb[y0:y1, x0:x1] = bw
    x0, x1, y0, y1 = box(DOORS["steel_door"])
    sd = np.full((y1 - y0, x1 - x0, 3), _rgb(222, 222, 218))
    yy = np.arange(y1 - y0)[:, None]
    sd *= np.where((yy % max(2, px(0.15))) < 1, 0.8, 1.0)[..., None]
    sd[:t] = sd[:, :t] = sd[:, -t:] = _rgb(70, 70, 72)
    rgb[y0:y1, x0:x1] = sd
    return rgb


_SIDING = {
    "siding_white": _rgb(230, 229, 222),
    "siding_cream": _rgb(216, 204, 174),
    "siding_grey": _rgb(158, 160, 157),
    "siding_blue": _rgb(126, 140, 152),
    "siding_sage": _rgb(146, 156, 132),
    "siding_tan": _rgb(182, 164, 136),
    "siding_brown": _rgb(104, 84, 66),
}
_BRICK = {
    "brick_red": _rgb(148, 66, 50),
    "brick_brown": _rgb(112, 78, 60),
    "brick_buff": _rgb(198, 170, 122),
    "brick_orange": _rgb(170, 88, 56),
    "brick_charcoal": _rgb(92, 84, 82),
}
_SHINGLE = {
    "shingle_charcoal": _rgb(62, 62, 64),
    "shingle_grey": _rgb(108, 108, 106),
    "shingle_brown": _rgb(98, 80, 64),
    "shingle_black": _rgb(40, 40, 42),
}
_METAL = {
    "metal_grey": _rgb(160, 164, 166),
    "metal_green": _rgb(62, 92, 66),
    "metal_red": _rgb(128, 42, 36),
    "metal_charcoal": _rgb(66, 68, 70),
    "metal_blue": _rgb(64, 84, 112),
}


def _paint_band(s: BuildingStyle, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    px_per_m = h / STOREY_M
    n = s.name
    if n in _SIDING:
        return _siding(shape, rng, _SIDING[n], px_per_m)
    if n == "board_white":
        return _board_batten(shape, rng, _rgb(232, 232, 228), px_per_m)
    if n in _BRICK:
        return _brick(shape, rng, _BRICK[n], px_per_m)
    if n == "stone":
        return _stone(shape, rng, px_per_m)
    if n == "stucco":
        return _stucco(shape, rng, _rgb(206, 194, 170), px_per_m)
    if n == "barn_red":
        return _boards(shape, rng, _rgb(128, 46, 36), px_per_m, windows=True)
    if n == "barn_grey":
        return _boards(shape, rng, _rgb(128, 124, 116), px_per_m, windows=True)
    if n == "shed_wood":
        return _boards(shape, rng, _rgb(120, 96, 72), px_per_m, windows=False)
    if n == "steel_white":
        return _steel_wall(shape, rng, _rgb(226, 226, 222), px_per_m)
    if n == "steel_red":
        return _steel_wall(shape, rng, _rgb(130, 40, 34), px_per_m)
    if n == "commercial":
        return _commercial(shape, rng, px_per_m)
    if n in _SHINGLE:
        return _shingles(shape, rng, _SHINGLE[n])
    if n in _METAL:
        return _metal(shape, rng, _METAL[n], px_per_m)
    if n == "flat_membrane":
        return _membrane(shape, rng)
    if n == "doors":
        return _doors(shape, rng, px_per_m)
    raise ValueError(f"no painter for style {n!r}")


def building_atlas(size: int = ATLAS_SIZE, seed: int = ATLAS_SEED) -> Image.Image:
    """The RGB atlas, `size` wide and 2 x `size` tall: `BAND_COUNT` horizontal bands, one per
    style (module docstring)."""
    if size < 64 or size & (size - 1):
        raise ValueError(f"atlas size must be a power of two >= 64, got {size}")
    band_h = 2 * size // BAND_COUNT
    out = np.zeros((2 * size, size, 3))
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
