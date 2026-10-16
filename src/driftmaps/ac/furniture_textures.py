"""The street furniture atlas: Ontario sign faces and the colours of poles, lights and hydrants.

One 1024 x 1024 image drawn procedurally with PIL (supersampled 4x, then downsampled) and
embedded as a BC1 DDS with mips. Sign plates are shaped by their geometry (octagon, triangle,
diamond, rectangle; `geometry.furniture`), so the atlas needs no alpha: each sign tile is the
plate's bounding box with the face drawn edge to edge, and whatever lies outside the shape is the
border colour (so mip bleeding at the plate's edge stays the border colour).

Faces follow the Ontario Traffic Manual (OTM) Book 5 (regulatory) and Book 6 (warning) designs
and the TAC Manual of Uniform Traffic Control Devices for Canada:

- Ra-1 STOP: red octagon, white border and legend. Ra-1t ALL WAY tab: white legend on red.
- Ra-2 YIELD: white triangle with a red border, no legend (the Canadian design).
- Rb-1 MAXIMUM speed: white rectangle, black border, "MAXIMUM", the number and "km/h".
- Rb-25 keep right: white, black arrow passing right of a black island.
- Wa-14 stop ahead: yellow diamond, red octagon below a black arrow.
- Wa-?? roundabout ahead: yellow diamond, three circular arrows (counter-clockwise).
- Wa-? dead end: yellow diamond, "DEAD END".
- Chevron alignment (Wa-9): black chevron on yellow; the right-pointing one is the left one's
  mirror (UVs flipped by the geometry).
- Street name blades: white legend on green. ASSUMED: Burlington's and Oakville's own blade
  designs could not be confirmed (2026-10-16), so the generic Ontario white-on-green is used.
- Signal lenses (`geometry.signals`): a green left-turn arrow on black, and the pedestrian
  signal's orange hand and white walking person on black (OTM Book 12 / TAC symbols); signal
  heads are black with black backplates bordered in fluorescent yellow.

The legend font is Pillow's bundled one (Aileron), stretched to each legend box: close to the
narrow FHWA/Clearview lettering of the real signs, and the same on every PC.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from driftmaps.ac.model import Texture
from driftmaps.ac.textures import encode_dds, encode_png

__all__ = [
    "ATLAS_SIZE",
    "SPEEDS",
    "SWATCHES",
    "Tile",
    "furniture_atlas",
    "furniture_atlas_texture",
    "name_tile",
    "sign_tile",
    "swatch_uv",
]

ATLAS_SIZE = 1024
SS = 4  # supersampling for the sign faces

# Colours (sRGB). OTM/TAC retroreflective sheeting colours, approximated.
RED = (190, 22, 34)
WHITE = (240, 240, 236)
BLACK = (18, 18, 18)
YELLOW = (250, 196, 0)
GREEN = (0, 108, 70)
SPEEDS = (30, 40, 50, 60, 70, 80, 90, 100)


@dataclass(frozen=True)
class Tile:
    """A pixel box in the atlas and its UVs: u0..u1 left to right, v0..v1 top to bottom."""

    x: int
    y: int
    w: int
    h: int

    @property
    def uv(self) -> tuple[float, float, float, float]:
        s = float(ATLAS_SIZE)
        return self.x / s, self.y / s, (self.x + self.w) / s, (self.y + self.h) / s


# Sign faces: the plate's bounding box (aspect = the plate's width / height).
_SIGNS: dict[str, Tile] = {
    "stop": Tile(0, 0, 128, 128),
    "yield": Tile(128, 0, 128, 111),
    "stop_ahead": Tile(256, 0, 128, 128),
    "roundabout_ahead": Tile(384, 0, 128, 128),
    "dead_end": Tile(512, 0, 128, 128),
    "chevron": Tile(640, 0, 96, 128),
    "keep_right": Tile(736, 0, 102, 128),
    "all_way": Tile(840, 0, 180, 60),
    # Signal lenses (square; under the swatches, beside the wood).
    "lens_arrow_left": Tile(128, 352, 64, 64),
    "ped_hand": Tile(192, 352, 64, 64),
    "ped_walk": Tile(256, 352, 64, 64),
}
for _k, _v in enumerate(SPEEDS):
    _SIGNS[f"speed_limit:{_v}"] = Tile(128 * _k, 128, 128, 160)

# Solid colours, 32 x 32 px each, sampled at their centre (no bleeding).
SWATCHES: dict[str, tuple[int, int, int]] = {
    "sign_back": (150, 152, 150),
    "galvanized": (165, 168, 166),
    "dark_metal": (70, 72, 74),
    "crossarm": (96, 84, 68),
    "insulator": (190, 196, 190),
    "wire": (28, 28, 28),
    "lamp_head": (176, 178, 176),
    "lamp_lens": (250, 246, 220),
    "signal_yellow": (232, 186, 20),
    "signal_black": (22, 22, 22),
    "lens_red": (200, 30, 20),
    "lens_amber": (230, 140, 20),
    "lens_green": (30, 190, 110),
    "hydrant_red": (196, 30, 28),
    "hydrant_cap": (236, 196, 30),
    "mailbox": (34, 36, 38),
    "post_wood": (120, 98, 72),
    "concrete": (172, 170, 164),
    "transformer": (120, 128, 124),
    "paint_white": (235, 235, 230),
    "backplate_yellow": (222, 228, 40),  # fluorescent yellow retroreflective border
    "lens_off": (34, 34, 32),
}
_SWATCH_Y = 288
_SWATCH = 32
WOOD = Tile(0, 352, 128, 128)  # the hydro pole: weathered cedar, grain along v
NAME_Y, NAME_W, NAME_H, NAME_COLS, NAME_ROWS = 480, 512, 64, 2, 8
MAX_NAMES = NAME_COLS * NAME_ROWS


def sign_tile(name: str) -> Tile:
    """The tile of a sign face ("stop", "speed_limit:70", ...); KeyError if there is none."""
    return _SIGNS[name]


def swatch_uv(name: str) -> tuple[float, float]:
    """The (u, v) centre of a colour swatch."""
    k = list(SWATCHES).index(name)
    x = (k % 32) * _SWATCH + _SWATCH / 2
    y = _SWATCH_Y + (k // 32) * _SWATCH + _SWATCH / 2
    return x / ATLAS_SIZE, y / ATLAS_SIZE


def name_tile(index: int) -> Tile:
    """The tile of street name blade number `index` (< MAX_NAMES)."""
    if not 0 <= index < MAX_NAMES:
        raise IndexError(index)
    return Tile(
        (index % NAME_COLS) * NAME_W, NAME_Y + (index // NAME_COLS) * NAME_H, NAME_W, NAME_H
    )


# --------------------------------------------------------------------------------------------
# Drawing helpers (on a supersampled canvas of the tile's size x SS)
# --------------------------------------------------------------------------------------------


@functools.cache
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=max(8, size))


def _legend(
    img: Image.Image,
    box: tuple[float, float, float, float],
    text: str,
    fill: tuple[int, int, int],
    weight: float = 0.07,
) -> None:
    """Draw `text` stretched to fill `box` (x0, y0, x1, y1): rendered large in its own mask
    (emboldened by a stroke of `weight` x the font size), cropped to the ink and resized."""
    size = 160
    font = _font(size)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    stroke = max(1, int(size * weight))
    try:
        x0, y0, x1, y1 = probe.textbbox((0, 0), text, font=font, stroke_width=stroke)
    except TypeError:  # very old Pillow without stroke support in textbbox
        x0, y0, x1, y1 = probe.textbbox((0, 0), text, font=font)
    mask = Image.new("L", (x1 - x0 + 4, y1 - y0 + 4), 0)
    ImageDraw.Draw(mask).text((2 - x0, 2 - y0), text, font=font, fill=255, stroke_width=stroke,
                              stroke_fill=255)  # fmt: skip
    bbox = mask.getbbox()
    if bbox is None:
        return
    mask = mask.crop(bbox)
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    mask = mask.resize((max(1, x1 - x0), max(1, y1 - y0)), Image.LANCZOS)
    img.paste(Image.new("RGB", mask.size, fill), (x0, y0), mask)


def _polygon(draw: ImageDraw.ImageDraw, pts: np.ndarray, fill) -> None:
    draw.polygon([tuple(p) for p in np.asarray(pts, float).tolist()], fill=fill)


def _octagon(cx: float, cy: float, r_flat: float) -> np.ndarray:
    """Octagon with flat top/bottom, `r_flat` = centre to flat side."""
    r = r_flat / math.cos(math.pi / 8)
    a = np.radians(22.5 + 45.0 * np.arange(8))
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def _arrow_up(draw, cx, top, bottom, shaft_w, head_w, head_h, fill) -> None:
    draw.rectangle([cx - shaft_w / 2, top + head_h * 0.9, cx + shaft_w / 2, bottom], fill=fill)
    _polygon(draw, np.array([[cx, top], [cx + head_w / 2, top + head_h],
                             [cx - head_w / 2, top + head_h]]), fill)  # fmt: skip


def _diamond(size: int, fill, border=BLACK) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """A warning diamond face: corners outside the diamond are the border colour."""
    img = Image.new("RGB", (size, size), border)
    d = ImageDraw.Draw(img)
    c, r = size / 2, size / 2 * 0.93
    _polygon(d, np.array([[c, c - r], [c + r, c], [c, c + r], [c - r, c]]), fill)
    return img, d


# --------------------------------------------------------------------------------------------
# Sign faces
# --------------------------------------------------------------------------------------------


def _stop(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(img)
    _polygon(d, _octagon(w / 2, h / 2, w / 2 * 0.955), RED)  # OTM: border ~1/40 of the width
    _legend(img, (w * 0.17, h * 0.36, w * 0.83, h * 0.64), "STOP", WHITE, 0.05)
    return img


def _yield(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), RED)
    d = ImageDraw.Draw(img)
    # Inverted triangle; the white inside is inset by the red band (~1/6 of the side).
    k = 0.21
    cx = w / 2
    dx, dy = w * k * 0.866 + 2, h * k * 0.5 + 2
    inner = np.array([[dx, dy], [w - dx, dy], [cx, h - h * k - 2]])
    _polygon(d, inner, WHITE)
    return img


def _speed(w: int, h: int, value: int) -> Image.Image:
    img = Image.new("RGB", (w, h), BLACK)
    d = ImageDraw.Draw(img)
    b = w * 0.035
    d.rounded_rectangle([b, b, w - b, h - b], radius=w * 0.05, fill=WHITE)
    _legend(img, (w * 0.12, h * 0.09, w * 0.88, h * 0.21), "MAXIMUM", BLACK, 0.04)
    _legend(img, (w * 0.14, h * 0.29, w * 0.86, h * 0.72), str(value), BLACK, 0.06)
    _legend(img, (w * 0.32, h * 0.78, w * 0.68, h * 0.89), "km/h", BLACK, 0.04)
    return img


def _chevron(w: int, h: int) -> Image.Image:
    """Points left: the road turns left (the right one is this face mirrored)."""
    img = Image.new("RGB", (w, h), YELLOW)
    d = ImageDraw.Draw(img)
    t = w * 0.30
    pts = np.array(
        [
            [w * 0.80, h * 0.10],
            [w * 0.80 - t, h * 0.10],
            [w * 0.20, h * 0.50],
            [w * 0.80 - t, h * 0.90],
            [w * 0.80, h * 0.90],
            [w * 0.20 + t, h * 0.50],
        ]
    )
    _polygon(d, pts, BLACK)  # fmt: skip
    return img


def _keep_right(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), BLACK)
    d = ImageDraw.Draw(img)
    b = w * 0.035
    d.rounded_rectangle([b, b, w - b, h - b], radius=w * 0.05, fill=WHITE)
    # The island (left), the arrow sweeping down past its right side.
    d.rounded_rectangle([w * 0.18, h * 0.40, w * 0.42, h * 0.86], radius=w * 0.08, fill=BLACK)
    sw = w * 0.11
    d.line([(w * 0.36, h * 0.12), (w * 0.62, h * 0.40), (w * 0.62, h * 0.70)], fill=BLACK,
           width=int(sw), joint="curve")  # fmt: skip
    _polygon(d, np.array([[w * 0.62, h * 0.90], [w * 0.80, h * 0.66], [w * 0.44, h * 0.66]]),
             BLACK)  # fmt: skip
    return img


def _stop_ahead(w: int, h: int) -> Image.Image:
    img, d = _diamond(w, YELLOW)
    _polygon(d, _octagon(w / 2, h * 0.63, w * 0.15), RED)
    _arrow_up(d, w / 2, h * 0.17, h * 0.47, w * 0.07, w * 0.22, h * 0.13, BLACK)
    return img


def _roundabout(w: int, h: int) -> Image.Image:
    img, d = _diamond(w, YELLOW)
    c, r, sw = w / 2, w * 0.17, w * 0.055
    for k in range(3):  # three arrows chasing each other counter-clockwise (seen from above)
        a0 = 90 + 120 * k
        d.arc([c - r, c - r, c + r, c + r], start=a0 + 18, end=a0 + 100, fill=BLACK,
              width=int(sw))  # fmt: skip
        a = math.radians(a0 + 18)
        tip = np.array([c + r * math.cos(a), c + r * math.sin(a)])
        tang = np.array([math.sin(a), -math.cos(a)])  # pointing counter-clockwise on screen
        norm = np.array([math.cos(a), math.sin(a)])
        _polygon(d, np.array([tip + tang * sw * 1.6, tip - norm * sw * 1.3,
                              tip + norm * sw * 1.3]), BLACK)  # fmt: skip
    return img


def _dead_end(w: int, h: int) -> Image.Image:
    img, d = _diamond(w, YELLOW)
    _legend(img, (w * 0.30, h * 0.34, w * 0.70, h * 0.48), "DEAD", BLACK, 0.05)
    _legend(img, (w * 0.30, h * 0.52, w * 0.70, h * 0.66), "END", BLACK, 0.05)
    return img


def _all_way(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(img)
    b = h * 0.06
    d.rectangle([b, b, w - b, h - b], fill=RED)
    _legend(img, (w * 0.12, h * 0.24, w * 0.88, h * 0.76), "ALL WAY", WHITE, 0.05)
    return img


def _name(w: int, h: int, text: str) -> Image.Image:
    img = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(img)
    b = h * 0.05
    d.rectangle([b, b, w - b, h - b], fill=GREEN)
    text = text.strip() or " "
    # Narrow letters: about 0.45 of the letter height per character, centred.
    width = min(w * 0.92, h * 0.62 * 0.5 * max(len(text), 3))
    _legend(img, ((w - width) / 2, h * 0.2, (w + width) / 2, h * 0.8), text, WHITE, 0.03)
    return img


def _lens_arrow_left(w: int, h: int) -> Image.Image:
    """A lit green left-turn arrow on a black lens."""
    img = Image.new("RGB", (w, h), (16, 16, 16))
    d = ImageDraw.Draw(img)
    d.ellipse([w * 0.04, h * 0.04, w * 0.96, h * 0.96], fill=(20, 40, 30))
    g = (60, 235, 150)
    d.rectangle([w * 0.42, h * 0.42, w * 0.80, h * 0.58], fill=g)
    _polygon(d, np.array([[w * 0.16, h * 0.50], [w * 0.46, h * 0.24], [w * 0.46, h * 0.76]]), g)
    return img


def _ped_hand(w: int, h: int) -> Image.Image:
    """The lit upraised hand (don't walk), Portland orange on black."""
    img = Image.new("RGB", (w, h), (16, 16, 16))
    d = ImageDraw.Draw(img)
    o = (255, 120, 30)
    d.rounded_rectangle([w * 0.30, h * 0.42, w * 0.70, h * 0.84], radius=w * 0.10, fill=o)
    for k, top in enumerate((0.22, 0.16, 0.18, 0.26)):  # four fingers
        x0 = w * (0.30 + 0.10 * k)
        d.rounded_rectangle([x0, h * top, x0 + w * 0.085, h * 0.55], radius=w * 0.04, fill=o)
    _polygon(d, np.array([[w * 0.30, h * 0.62], [w * 0.16, h * 0.44], [w * 0.22, h * 0.40],
                          [w * 0.34, h * 0.52]]), o)  # fmt: skip
    return img


def _ped_walk(w: int, h: int) -> Image.Image:
    """The walking person (unlit here: dark grey on black)."""
    img = Image.new("RGB", (w, h), (16, 16, 16))
    d = ImageDraw.Draw(img)
    c = (70, 70, 66)
    lw = int(w * 0.09)
    d.ellipse([w * 0.44, h * 0.10, w * 0.58, h * 0.24], fill=c)
    d.line([(w * 0.50, h * 0.28), (w * 0.47, h * 0.58)], fill=c, width=lw)
    d.line([(w * 0.47, h * 0.58), (w * 0.34, h * 0.88)], fill=c, width=lw)
    d.line([(w * 0.47, h * 0.58), (w * 0.62, h * 0.88)], fill=c, width=lw)
    d.line([(w * 0.36, h * 0.48), (w * 0.50, h * 0.32), (w * 0.64, h * 0.46)], fill=c, width=lw)
    return img


_FACES = {
    "lens_arrow_left": _lens_arrow_left,
    "ped_hand": _ped_hand,
    "ped_walk": _ped_walk,
    "stop": _stop,
    "yield": _yield,
    "stop_ahead": _stop_ahead,
    "roundabout_ahead": _roundabout,
    "dead_end": _dead_end,
    "chevron": _chevron,
    "keep_right": _keep_right,
    "all_way": _all_way,
}


def _wood(w: int, h: int, seed: int = 11) -> Image.Image:
    """Weathered cedar: grey-brown with vertical grain and a few checks."""
    rng = np.random.default_rng(seed)
    cols = rng.normal(0.0, 1.0, w)
    cols = np.convolve(cols, np.ones(3) / 3, mode="same")
    rows = np.convolve(rng.normal(0.0, 1.0, h), np.ones(9) / 9, mode="same")
    base = np.array([108, 92, 74], float)
    shade = 1.0 + 0.10 * cols[None, :] + 0.03 * rows[:, None]
    img = base[None, None, :] * shade[:, :, None]
    for x in rng.integers(0, w, 5):  # drying checks
        img[:, x, :] *= 0.72
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def furniture_atlas(names: Sequence[str] = ()) -> Image.Image:
    """The atlas image (RGB). `names` are street name blades (at most MAX_NAMES)."""
    atlas = Image.new("RGB", (ATLAS_SIZE, ATLAS_SIZE), (128, 128, 128))
    for key, tile in _SIGNS.items():
        w, h = tile.w * SS, tile.h * SS
        if key.startswith("speed_limit:"):
            face = _speed(w, h, int(key.split(":")[1]))
        else:
            face = _FACES[key](w, h)
        atlas.paste(face.resize((tile.w, tile.h), Image.LANCZOS), (tile.x, tile.y))
    for k, (_, rgb) in enumerate(SWATCHES.items()):
        x, y = (k % 32) * _SWATCH, _SWATCH_Y + (k // 32) * _SWATCH
        atlas.paste(Image.new("RGB", (_SWATCH, _SWATCH), rgb), (x, y))
    atlas.paste(_wood(WOOD.w, WOOD.h).filter(ImageFilter.SMOOTH), (WOOD.x, WOOD.y))
    for k, text in enumerate(list(names)[:MAX_NAMES]):
        tile = name_tile(k)
        face = _name(tile.w * 2, tile.h * 2, text)
        atlas.paste(face.resize((tile.w, tile.h), Image.LANCZOS), (tile.x, tile.y))
    return atlas


def furniture_atlas_texture(name: str = "dm_furniture.dds", names: Sequence[str] = ()) -> Texture:
    """The atlas as a KN5 `Texture`: BC1 DDS with mips ("*.dds") or PNG ("*.png")."""
    if not name.isascii():
        raise ValueError(f"texture names must be ASCII: {name!r}")
    return _atlas_texture(name, tuple(names))


@functools.cache
def _atlas_texture(name: str, names: tuple[str, ...]) -> Texture:
    img = furniture_atlas(names)
    lower = name.lower()
    if lower.endswith(".dds"):
        return Texture(name=name, data=encode_dds(img, "BC1", True))
    if lower.endswith(".png"):
        return Texture(name=name, data=encode_png(img))
    raise ValueError(f"texture name must end in .png or .dds: {name!r}")
