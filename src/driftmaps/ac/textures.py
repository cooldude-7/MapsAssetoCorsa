"""Procedural textures and their PNG / DDS (BC1, BC3) encodings for KN5 files.

Every texture here is generated from scratch with numpy and Pillow, so we own it outright and can
ship it in generated tracks (never ship Kunos textures; docs/research/geo.md §6). Generators are
deterministic: the same arguments and seed give the same pixels.

Encodings (docs/research/ac_format.md §4.6):
- `encode_png`: plain PNG bytes. AC accepts PNG inside a KN5; whether it builds mips for them is
  ASSUMED (in-game spike S2).
- `encode_dds`: a DX9-style DDS ("DXT1" = BC1 for opaque, "DXT5" = BC3 for alpha) with a full mip
  chain down to 1x1. Vanilla AC can't read BC7, so we never write it. Our own vectorised block
  encoder is used (principal-axis endpoints, nearest palette index, one least-squares endpoint
  refinement); tests decode the result with Pillow to check the error.

DDS header layout: Microsoft's DDS_HEADER reference
(https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header). The flags follow
DirectXTex's writer `_EncodeDDSHeader` (github.com/microsoft/DirectXTex, DirectXTexDDS.cpp):
MIPMAPCOUNT is always set with the real level count, COMPLEX|MIPMAP caps only when there is more
than one level, LINEARSIZE = byte size of the top level, depth = 1. Levels smaller than 4x4 still
take one whole 4x4 block (their pixels are edge-padded before encoding).

Image orientation: row 0 is the top of the image, which is UV v = 0 in DirectX/AC.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from driftmaps.ac.model import Texture

__all__ = [
    "Arrow",
    "Color",
    "DdsFormat",
    "DdsInfo",
    "DdsLevel",
    "asphalt_texture",
    "decode_dds_level",
    "debug_grid_texture",
    "encode_bc1",
    "encode_bc3",
    "encode_dds",
    "encode_png",
    "grass_texture",
    "gravel_texture",
    "kn5_texture",
    "label_texture",
    "mip_chain",
    "read_dds_info",
    "solid_texture",
]

Color = tuple[int, int, int] | tuple[int, int, int, int]
DdsFormat = Literal["BC1", "BC3"]
Arrow = Literal["up", "down", "left", "right"]

# ---------------------------------------------------------------------------------------------
# DDS constants (Microsoft DDS_HEADER / DDS_PIXELFORMAT docs).
# ---------------------------------------------------------------------------------------------

DDS_MAGIC = b"DDS "
DDS_HEADER_SIZE = 124
DDS_PIXELFORMAT_SIZE = 32

DDSD_CAPS = 0x1
DDSD_HEIGHT = 0x2
DDSD_WIDTH = 0x4
DDSD_PITCH = 0x8
DDSD_PIXELFORMAT = 0x1000
DDSD_MIPMAPCOUNT = 0x20000
DDSD_LINEARSIZE = 0x80000

DDPF_ALPHAPIXELS = 0x1
DDPF_FOURCC = 0x4

DDSCAPS_COMPLEX = 0x8
DDSCAPS_TEXTURE = 0x1000
DDSCAPS_MIPMAP = 0x400000

_FOURCC: dict[str, bytes] = {"BC1": b"DXT1", "BC3": b"DXT5"}
_BLOCK_BYTES: dict[str, int] = {"BC1": 8, "BC3": 16}
_FORMAT_OF_FOURCC = {v: k for k, v in _FOURCC.items()}

# magic, size, flags, height, width, pitchOrLinearSize, depth, mipMapCount, reserved1[11],
# pf.size, pf.flags, pf.fourCC, pf.rgbBitCount, pf.r/g/b/a masks, caps, caps2, caps3, caps4,
# reserved2.
_DDS_HEADER = struct.Struct("<4s7I11I2I4s5I5I")
assert _DDS_HEADER.size == 4 + DDS_HEADER_SIZE

_BC1_DTYPE = np.dtype([("c0", "<u2"), ("c1", "<u2"), ("idx", "<u4")])
_BC3_DTYPE = np.dtype(
    [
        ("a0", "u1"),
        ("a1", "u1"),
        ("abits", "u1", (6,)),
        ("c0", "<u2"),
        ("c1", "<u2"),
        ("idx", "<u4"),
    ]
)
assert _BC1_DTYPE.itemsize == 8 and _BC3_DTYPE.itemsize == 16

# Weight of endpoint 0 for colour indices 0..3 in 4-colour mode (c0 > c1):
# index 0 = c0, 1 = c1, 2 = (2*c0 + c1) / 3, 3 = (c0 + 2*c1) / 3.
_COLOR_W0 = np.array([1.0, 0.0, 2.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
# Swapping the endpoints (to keep c0 > c1) swaps index 0 <-> 1 and 2 <-> 3.
_SWAP_INDEX = np.array([1, 0, 3, 2], dtype=np.int64)
# Weight of alpha endpoint 0 for alpha codes 0..7 in 8-value mode (a0 > a1).
_ALPHA_W0 = np.array([1, 0, 6 / 7, 5 / 7, 4 / 7, 3 / 7, 2 / 7, 1 / 7], dtype=np.float32)

# Blocks per encoder batch; bounds peak memory (~16 MB of temporaries per 16k blocks).
_BATCH_BLOCKS = 16384


# ---------------------------------------------------------------------------------------------
# Tileable noise
# ---------------------------------------------------------------------------------------------


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _spectral_noise(
    shape: tuple[int, int], rng: np.random.Generator, beta: float, cutoff: float = 0.5
) -> np.ndarray:
    """Tileable noise with a 1/f^beta power spectrum, zero mean and unit standard deviation.

    Filtering white noise in the frequency domain keeps it periodic, so the texture tiles with
    no seam. `beta` 0 = white grain, 2 = cloudy, 3+ = large soft patches. `cutoff` (cycles per
    pixel) softly removes detail finer than that.
    """
    h, w = shape
    white = rng.standard_normal((h, w))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    amp = f ** (-beta / 2.0) * np.exp(-((f / cutoff) ** 2))
    amp[0, 0] = 0.0
    out = np.fft.ifft2(np.fft.fft2(white) * amp).real
    out -= out.mean()
    std = out.std()
    return (out / std if std > 0 else out).astype(np.float32)


def _worley(
    size: int, cells: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tileable cellular (Worley) noise: one jittered point per cell on a wrapped grid.

    Returns (f1, f2, cell_id) per pixel: distance to the nearest and second-nearest point (in
    cell units) and the flat index of the nearest point's cell.
    """
    pts = rng.random((cells, cells, 2))
    coord = (np.arange(size) + 0.5) * (cells / size)
    yy, xx = np.meshgrid(coord, coord, indexing="ij")
    cy = np.floor(yy).astype(np.int64)
    cx = np.floor(xx).astype(np.int64)
    f1 = np.full((size, size), np.inf)
    f2 = np.full((size, size), np.inf)
    cell_id = np.zeros((size, size), dtype=np.int64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ny = (cy + dy) % cells
            nx = (cx + dx) % cells
            px = cx + dx + pts[ny, nx, 0]
            py = cy + dy + pts[ny, nx, 1]
            d = np.hypot(xx - px, yy - py)
            closer = d < f1
            f2 = np.where(closer, f1, np.minimum(f2, d))
            cell_id = np.where(closer, ny * cells + nx, cell_id)
            f1 = np.where(closer, d, f1)
    return f1.astype(np.float32), f2.astype(np.float32), cell_id


def _to_image(rgb: np.ndarray) -> Image.Image:
    """Float RGB(A) in 0..255 -> uint8 Pillow image."""
    arr = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA" if arr.shape[-1] == 4 else "RGB")


def _check_size(size: int) -> None:
    if size < 4:
        raise ValueError(f"texture size must be at least 4 px, got {size}")


# ---------------------------------------------------------------------------------------------
# Material textures (all tileable, RGB)
# ---------------------------------------------------------------------------------------------


def asphalt_texture(size: int = 512, seed: int = 1) -> Image.Image:
    """Tileable dark-grey asphalt: soft tonal patches, fine grain and light/dark aggregate."""
    _check_size(size)
    rng = _rng(seed)
    shape = (size, size)
    patches = _spectral_noise(shape, rng, beta=3.0)
    grain = _spectral_noise(shape, rng, beta=0.3)
    specks = _spectral_noise(shape, rng, beta=0.0)
    lum = 72.0 + 5.0 * patches + 9.0 * grain
    lum = lum + np.where(specks > 2.4, 38.0, 0.0) - np.where(specks < -2.5, 22.0, 0.0)
    tint = np.array([1.0, 1.0, 1.03], dtype=np.float32)
    return _to_image(lum[..., None] * tint)


def grass_texture(size: int = 512, seed: int = 2) -> Image.Image:
    """Tileable grass: mid green with large lighter/drier patches and fine blade noise."""
    _check_size(size)
    rng = _rng(seed)
    shape = (size, size)
    patches = _spectral_noise(shape, rng, beta=3.2)
    mid = _spectral_noise(shape, rng, beta=2.0)
    blades = _spectral_noise(shape, rng, beta=0.4)
    base = np.array([58.0, 92.0, 36.0], dtype=np.float32)
    rgb = (
        base
        + patches[..., None] * np.array([7.0, 10.0, 4.0], dtype=np.float32)
        + mid[..., None] * np.array([5.0, 8.0, 3.0], dtype=np.float32)
        + blades[..., None] * np.array([9.0, 14.0, 6.0], dtype=np.float32)
    )
    # Dry, yellowish patches where the large-scale noise is high.
    dry = np.clip((patches - 0.8) / 1.2, 0.0, 1.0)[..., None] * 0.45
    rgb = rgb * (1.0 - dry) + np.array([112.0, 108.0, 62.0], dtype=np.float32) * dry
    return _to_image(rgb)


def gravel_texture(size: int = 512, seed: int = 3, stones: int | None = None) -> Image.Image:
    """Tileable gravel: rounded stones of mixed grey/brown tones with dark gaps between them.

    `stones` is the number of stones across the tile (default ~ one per 12 px).
    """
    _check_size(size)
    rng = _rng(seed)
    cells = stones if stones is not None else max(2, size // 12)
    f1, f2, cell_id = _worley(size, cells, rng)
    # Per-stone colour: lightness and a grey/brown tint.
    lightness = rng.uniform(0.45, 0.85, cells * cells).astype(np.float32)
    warmth = rng.uniform(0.0, 1.0, cells * cells).astype(np.float32)
    grey = np.array([128.0, 126.0, 122.0], dtype=np.float32)
    brown = np.array([134.0, 116.0, 92.0], dtype=np.float32)
    stone_rgb = (grey * (1.0 - warmth[:, None]) + brown * warmth[:, None]) * lightness[:, None]
    rgb = stone_rgb[cell_id]
    shade = np.clip(1.0 - (f1 / 0.75) ** 2, 0.0, 1.0)  # rounded top lit, edges darker
    gap = np.clip((f2 - f1) / 0.12, 0.0, 1.0)  # dark crevices along cell borders
    grain = _spectral_noise((size, size), rng, beta=0.2)
    factor = (0.55 + 0.45 * shade) * (0.35 + 0.65 * gap)
    rgb = rgb * factor[..., None] + grain[..., None] * 6.0
    return _to_image(rgb)


def solid_texture(color: Color, size: int | tuple[int, int] = 4) -> Image.Image:
    """A single-colour texture. A 4-tuple colour gives an RGBA image, a 3-tuple an RGB one."""
    w, h = (size, size) if isinstance(size, int) else size
    if len(color) not in (3, 4):
        raise ValueError(f"colour must have 3 or 4 components, got {color!r}")
    mode = "RGBA" if len(color) == 4 else "RGB"
    return Image.new(mode, (w, h), tuple(int(c) for c in color))


# ---------------------------------------------------------------------------------------------
# Text and debug textures
# ---------------------------------------------------------------------------------------------


def _font(size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Pillow's bundled default font (Aileron, scalable when Pillow has FreeType)."""
    return ImageFont.load_default(size=max(1, int(size_px)))


def _text_box(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont
) -> tuple[float, float, float, float]:
    """(width, height, left, top) of `text` drawn at the origin."""
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    return right - left, bottom - top, left, top


def _draw_text_fit(
    img: Image.Image, text: str, box: tuple[float, float, float, float], fill: Color
) -> None:
    """Draw `text` as large as fits, centred in `box` = (x0, y0, x1, y1)."""
    if not text:
        return
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if bw <= 1 or bh <= 1:
        return
    draw = ImageDraw.Draw(img)
    probe = _font(100)
    if not isinstance(probe, ImageFont.FreeTypeFont):
        _draw_text_bitmap(img, text, box, fill)
        return
    # The bbox scales ~linearly with font size: estimate, then step down until it fits.
    tw, th, _, _ = _text_box(draw, text, probe)
    size = int(100 * min(bw / max(tw, 1), bh / max(th, 1)))
    while size > 1:
        tw, th, _, _ = _text_box(draw, text, _font(size))
        if tw <= bw and th <= bh:
            break
        size = int(size * 0.95)
    font = _font(size)
    tw, th, left, top = _text_box(draw, text, font)
    x = x0 + (bw - tw) / 2 - left
    y = y0 + (bh - th) / 2 - top
    draw.multiline_text((x, y), text, font=font, fill=fill, align="center")


def _draw_text_bitmap(
    img: Image.Image, text: str, box: tuple[float, float, float, float], fill: Color
) -> None:
    """Fallback when Pillow lacks FreeType: render the small bitmap font, then scale it up."""
    font = ImageFont.load_default()
    tw, th, left, top = _text_box(ImageDraw.Draw(Image.new("L", (1, 1))), text, font)
    mask = Image.new("L", (max(int(tw), 1), max(int(th), 1)), 0)
    ImageDraw.Draw(mask).multiline_text((-left, -top), text, font=font, fill=255, align="center")
    x0, y0, x1, y1 = box
    k = min((x1 - x0) / mask.width, (y1 - y0) / mask.height)
    size = (max(1, int(mask.width * k)), max(1, int(mask.height * k)))
    mask = mask.resize(size, Image.Resampling.NEAREST)
    layer = Image.new(img.mode, size, fill)
    img.paste(layer, (int(x0 + (x1 - x0 - size[0]) / 2), int(y0 + (y1 - y0 - size[1]) / 2)), mask)


# An arrow pointing up, in a [-1, 1] square (y down, like image rows).
_ARROW_UP = [(0.0, -1.0), (0.8, -0.1), (0.3, -0.1), (0.3, 1.0), (-0.3, 1.0), (-0.3, -0.1)]
_ARROW_UP.append((-0.8, -0.1))
_ARROW_ROTATE = {
    "up": lambda x, y: (x, y),
    "down": lambda x, y: (x, -y),
    "right": lambda x, y: (-y, x),
    "left": lambda x, y: (y, -x),
}


def _draw_arrow(
    img: Image.Image, direction: Arrow, box: tuple[float, float, float, float], fill: Color
) -> None:
    if direction not in _ARROW_ROTATE:
        raise ValueError(f"arrow must be one of {sorted(_ARROW_ROTATE)}, got {direction!r}")
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = min(x1 - x0, y1 - y0) / 2
    rot = _ARROW_ROTATE[direction]
    pts = [(cx + px * half, cy + py * half) for px, py in (rot(x, y) for x, y in _ARROW_UP)]
    ImageDraw.Draw(img).polygon(pts, fill=fill)


def label_texture(
    text: str,
    size: int | tuple[int, int] = 512,
    *,
    fg: Color = (255, 255, 255),
    bg: Color = (32, 32, 32),
    arrow: Arrow | None = None,
    border_px: int = 0,
) -> Image.Image:
    """Big centred text (e.g. "N", "A", "B", "PNG", "DDS", "NPOT") with an optional arrow.

    For road markings and signs. The arrow points towards the image top for "up" (UV v = 0),
    and is drawn below the text when both are given. A 4-component `bg` with alpha < 255 gives
    an RGBA image (transparent background for decals); otherwise the image is RGB. `border_px`
    draws a frame in the foreground colour.
    """
    w, h = (size, size) if isinstance(size, int) else size
    mode = "RGBA" if len(bg) == 4 or len(fg) == 4 else "RGB"
    bg_full = tuple(bg) if mode == "RGB" or len(bg) == 4 else (*bg, 255)
    fg_full = tuple(fg) if mode == "RGB" or len(fg) == 4 else (*fg, 255)
    img = Image.new(mode, (w, h), bg_full)
    pad = 0.06 * min(w, h) + border_px
    inner = (pad, pad, w - pad, h - pad)
    if arrow is not None and text:
        split = inner[1] + (inner[3] - inner[1]) * 0.55
        _draw_text_fit(img, text, (inner[0], inner[1], inner[2], split), fg_full)
        _draw_arrow(img, arrow, (inner[0], split + pad * 0.5, inner[2], inner[3]), fg_full)
    elif arrow is not None:
        _draw_arrow(img, arrow, inner, fg_full)
    else:
        _draw_text_fit(img, text, inner, fg_full)
    if border_px > 0:
        ImageDraw.Draw(img).rectangle((0, 0, w - 1, h - 1), outline=fg_full, width=int(border_px))
    return img


def debug_grid_texture(
    size: int = 1024,
    *,
    cells: int = 8,
    metres_per_cell: float = 1.0,
    light: Color = (205, 205, 200),
    dark: Color = (150, 152, 158),
    line: Color = (40, 40, 40),
) -> Image.Image:
    """A checkerboard with grid lines and metre labels, for checking UV scale and direction.

    Each of the `cells` x `cells` squares is labelled "u,v" with the position of its top-left
    corner in metres (`metres_per_cell` per square), so a mesh with 1 UV unit = `cells *
    metres_per_cell` metres shows true distances. A red bar runs along the top edge (the +u
    direction) and a green bar down the left edge (the +v direction).
    """
    if cells < 1:
        raise ValueError("cells must be >= 1")
    _check_size(size)
    img = Image.new("RGB", (size, size), tuple(light))
    draw = ImageDraw.Draw(img)
    edges = np.rint(np.linspace(0, size, cells + 1)).astype(int)
    for row in range(cells):
        for col in range(cells):
            if (row + col) % 2:
                draw.rectangle(
                    (edges[col], edges[row], edges[col + 1] - 1, edges[row + 1] - 1),
                    fill=tuple(dark),
                )
    lw = max(1, size // 256)
    for e in edges:
        draw.line((e, 0, e, size), fill=tuple(line), width=lw)
        draw.line((0, e, size, e), fill=tuple(line), width=lw)
    for row in range(cells):
        for col in range(cells):
            x0, x1, y0, y1 = edges[col], edges[col + 1], edges[row], edges[row + 1]
            m = 0.18 * (x1 - x0)
            label = f"{col * metres_per_cell:g},{row * metres_per_cell:g}"
            _draw_text_fit(img, label, (x0 + m, y0 + m, x1 - m, y1 - m), (20, 20, 20))
    bar = max(2, size // 64)
    draw.rectangle((0, 0, size - 1, bar - 1), fill=(220, 30, 30))
    draw.rectangle((0, 0, bar - 1, size - 1), fill=(30, 170, 40))
    return img


# ---------------------------------------------------------------------------------------------
# PNG
# ---------------------------------------------------------------------------------------------


def encode_png(img: Image.Image) -> bytes:
    """PNG file bytes (no timestamp or metadata chunks, so the output is deterministic)."""
    if img.mode not in ("RGB", "RGBA", "L", "LA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()


# ---------------------------------------------------------------------------------------------
# Mip chain
# ---------------------------------------------------------------------------------------------


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


def _half(x: np.ndarray) -> np.ndarray:
    """2x2 box filter (2x1 or 1x2 once one side is 1 px). Sides > 1 must be even."""
    if x.shape[0] > 1:
        x = 0.5 * (x[0::2] + x[1::2])
    if x.shape[1] > 1:
        x = 0.5 * (x[:, 0::2] + x[:, 1::2])
    return x


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def mip_chain(img: Image.Image, *, srgb: bool = True) -> list[np.ndarray]:
    """All mip levels of `img` (power-of-two sides) down to 1x1, as uint8 (h, w, c) arrays.

    Colour is averaged in linear light when `srgb` (correct for colour textures; pass False for
    masks and normal maps). With alpha, colour is alpha-weighted so transparent pixels don't
    bleed their (invisible) colour into the edges of lower levels. Each level is filtered from
    the previous one at float precision.
    """
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    w, h = img.size
    if not (_is_pow2(w) and _is_pow2(h)):
        raise ValueError(f"mipmapped textures need power-of-two sides, got {w}x{h}")
    top = np.asarray(img)
    has_alpha = top.shape[-1] == 4
    f = top.astype(np.float64) / 255.0
    if srgb:
        f[..., :3] = _srgb_to_linear(f[..., :3])
    levels = [top]
    while f.shape[0] > 1 or f.shape[1] > 1:
        if has_alpha:
            a = f[..., 3:]
            a_half = _half(a)
            premul = _half(f[..., :3] * a)
            plain = _half(f[..., :3])
            rgb = np.where(a_half > 1e-6, premul / np.maximum(a_half, 1e-6), plain)
            f = np.concatenate([rgb, a_half], axis=-1)
        else:
            f = _half(f)
        out = f.copy()
        if srgb:
            out[..., :3] = _linear_to_srgb(out[..., :3])
        levels.append(np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8))
    return levels


# ---------------------------------------------------------------------------------------------
# BC1 / BC3 block encoder
# ---------------------------------------------------------------------------------------------


def _to_blocks(arr: np.ndarray) -> np.ndarray:
    """(h, w, c) -> (n_blocks, 16, c), blocks in row-major order, texels row-major per block.

    Sides that aren't multiples of 4 (e.g. the 2x2 and 1x1 mips) are edge-padded first.
    """
    h, w = arr.shape[:2]
    ph, pw = (-h) % 4, (-w) % 4
    if ph or pw:
        arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="edge")
    hh, ww, c = arr.shape
    return arr.reshape(hh // 4, 4, ww // 4, 4, c).swapaxes(1, 2).reshape(-1, 16, c)


def _quantise_565(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Float RGB (n, 3) -> (packed RGB565 (n,), the colour a decoder expands it to (n, 3))."""
    scale = np.array([31.0, 63.0, 31.0], dtype=np.float32) / 255.0
    q = np.rint(np.clip(c, 0.0, 255.0) * scale).astype(np.int64)
    r, g, b = q[:, 0], q[:, 1], q[:, 2]
    packed = (r << 11) | (g << 5) | b
    decoded = np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], axis=1)
    return packed, decoded.astype(np.float32)


def _nearest_colour(
    px: np.ndarray, d0: np.ndarray, d1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Best 4-colour palette index per texel, and the summed squared error per block."""
    w = _COLOR_W0[None, :, None]
    palette = w * d0[:, None, :] + (1.0 - w) * d1[:, None, :]  # (n, 4, 3)
    dist = ((px[:, :, None, :] - palette[:, None, :, :]) ** 2).sum(-1)  # (n, 16, 4)
    idx = dist.argmin(-1)
    err = np.take_along_axis(dist, idx[..., None], -1)[..., 0].sum(1)
    return idx, err


def _principal_endpoints(px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Endpoints at the extremes of each block's colours along its principal axis."""
    mean = px.mean(1)
    d = px - mean[:, None, :]
    cov = np.einsum("nki,nkj->nij", d, d)
    # Power iteration, started from the covariance column of the most varying channel (never
    # orthogonal to the principal axis unless that channel doesn't vary along it at all).
    start = np.argmax(np.einsum("nii->ni", cov), axis=1)
    v = np.take_along_axis(cov, start[:, None, None], 2)[..., 0]
    for _ in range(8):
        v = np.einsum("nij,nj->ni", cov, v)
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / np.where(norm > 1e-12, norm, 1.0)
    proj = np.einsum("nki,ni->nk", d, v)
    hi = mean + proj.max(1)[:, None] * v
    lo = mean + proj.min(1)[:, None] * v
    return hi, lo


def _refine_endpoints(
    px: np.ndarray, idx: np.ndarray, e0: np.ndarray, e1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares endpoints for fixed indices (keeps the old ones where it's degenerate)."""
    a = _COLOR_W0[idx]
    b = 1.0 - a
    aa, bb, ab = (a * a).sum(1), (b * b).sum(1), (a * b).sum(1)
    ap = np.einsum("nk,nkc->nc", a, px)
    bp = np.einsum("nk,nkc->nc", b, px)
    det = aa * bb - ab * ab
    ok = (det > 1e-6)[:, None]
    safe = np.where(det > 1e-6, det, 1.0)[:, None]
    n0 = (bb[:, None] * ap - ab[:, None] * bp) / safe
    n1 = (aa[:, None] * bp - ab[:, None] * ap) / safe
    return np.where(ok, n0, e0), np.where(ok, n1, e1)


def _encode_colour_blocks(px: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BC1-style colour blocks for (n, 16, 3) float texels -> (c0, c1, packed indices)."""
    e0, e1 = _principal_endpoints(px)
    v0, d0 = _quantise_565(e0)
    v1, d1 = _quantise_565(e1)
    idx, err = _nearest_colour(px, d0, d1)
    for _ in range(2):
        r0, r1 = _refine_endpoints(px, idx, e0, e1)
        rv0, rd0 = _quantise_565(r0)
        rv1, rd1 = _quantise_565(r1)
        ridx, rerr = _nearest_colour(px, rd0, rd1)
        better = rerr < err
        e0 = np.where(better[:, None], r0, e0)
        e1 = np.where(better[:, None], r1, e1)
        v0, v1 = np.where(better, rv0, v0), np.where(better, rv1, v1)
        idx = np.where(better[:, None], ridx, idx)
        err = np.where(better, rerr, err)
    # 4-colour mode needs c0 > c1 as uint16. Equal endpoints = one flat colour: use index 0
    # (in BC1 c0 == c1 selects 3-colour mode where index 3 is transparent black).
    swap = v0 < v1
    idx = np.where(swap[:, None], _SWAP_INDEX[idx], idx)
    v0, v1 = np.where(swap, v1, v0), np.where(swap, v0, v1)
    idx = np.where((v0 == v1)[:, None], 0, idx)
    shifts = (2 * np.arange(16, dtype=np.uint64))[None, :]
    bits = (idx.astype(np.uint64) << shifts).sum(1, dtype=np.uint64)
    return v0.astype(np.uint16), v1.astype(np.uint16), bits.astype(np.uint32)


def _encode_alpha_blocks(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BC3 alpha blocks for (n, 16) float alpha -> (a0, a1, 6 index bytes per block).

    Always 8-value mode (a0 = block max > a1 = block min), which reproduces 0 and 255 exactly
    whenever they are the block extremes.
    """
    a0 = np.rint(alpha.max(1)).astype(np.int64)
    a1 = np.rint(alpha.min(1)).astype(np.int64)
    w = _ALPHA_W0[None, :]
    palette = w * a0[:, None] + (1.0 - w) * a1[:, None]  # (n, 8)
    codes = np.abs(alpha[:, :, None] - palette[:, None, :]).argmin(-1)
    codes = np.where((a0 == a1)[:, None], 0, codes)
    shifts = (3 * np.arange(16, dtype=np.uint64))[None, :]
    bits = (codes.astype(np.uint64) << shifts).sum(1, dtype=np.uint64)
    as_bytes = bits.astype("<u8").view(np.uint8).reshape(-1, 8)[:, :6]
    return a0.astype(np.uint8), a1.astype(np.uint8), as_bytes


def _as_channels(arr: np.ndarray, channels: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"expected an (h, w, 3|4) image array, got shape {arr.shape}")
    if arr.shape[-1] == 3 and channels == 4:
        arr = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, arr.dtype)], axis=-1)
    return arr[..., :channels]


def encode_bc1(rgb: np.ndarray) -> bytes:
    """BC1 (DXT1) blocks for a uint8 (h, w, 3|4) array; alpha is ignored (opaque).

    Sides that aren't multiples of 4 are edge-padded to whole blocks.
    """
    blocks = _to_blocks(_as_channels(rgb, 3))
    out = np.empty(len(blocks), dtype=_BC1_DTYPE)
    for s in range(0, len(blocks), _BATCH_BLOCKS):
        chunk = blocks[s : s + _BATCH_BLOCKS].astype(np.float32)
        c0, c1, idx = _encode_colour_blocks(chunk)
        out["c0"][s : s + len(chunk)] = c0
        out["c1"][s : s + len(chunk)] = c1
        out["idx"][s : s + len(chunk)] = idx
    return out.tobytes()


def encode_bc3(rgba: np.ndarray) -> bytes:
    """BC3 (DXT5) blocks for a uint8 (h, w, 3|4) array (RGB input is treated as opaque)."""
    blocks = _to_blocks(_as_channels(rgba, 4))
    out = np.empty(len(blocks), dtype=_BC3_DTYPE)
    for s in range(0, len(blocks), _BATCH_BLOCKS):
        chunk = blocks[s : s + _BATCH_BLOCKS].astype(np.float32)
        a0, a1, abits = _encode_alpha_blocks(chunk[..., 3])
        c0, c1, idx = _encode_colour_blocks(chunk[..., :3])
        sl = slice(s, s + len(chunk))
        out["a0"][sl], out["a1"][sl], out["abits"][sl] = a0, a1, abits
        out["c0"][sl], out["c1"][sl], out["idx"][sl] = c0, c1, idx
    return out.tobytes()


# ---------------------------------------------------------------------------------------------
# DDS container
# ---------------------------------------------------------------------------------------------


def _level_size(width: int, height: int, fmt: str) -> int:
    return max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * _BLOCK_BYTES[fmt]


def _dds_header(width: int, height: int, fmt: str, mip_count: int) -> bytes:
    flags = (
        DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_MIPMAPCOUNT | DDSD_LINEARSIZE
    )
    caps = DDSCAPS_TEXTURE | (DDSCAPS_COMPLEX | DDSCAPS_MIPMAP if mip_count > 1 else 0)
    return _DDS_HEADER.pack(
        DDS_MAGIC,
        DDS_HEADER_SIZE,
        flags,
        height,
        width,
        _level_size(width, height, fmt),  # linear size = bytes in the top level
        1,  # depth (unused for 2D; DirectXTex writes 1)
        mip_count,
        *([0] * 11),
        DDS_PIXELFORMAT_SIZE,
        DDPF_FOURCC,
        _FOURCC[fmt],
        0,  # rgbBitCount
        0,  # r mask
        0,  # g mask
        0,  # b mask
        0,  # a mask
        caps,
        0,  # caps2
        0,  # caps3
        0,  # caps4
        0,  # reserved2
    )


def encode_dds(
    img: Image.Image, fmt: DdsFormat = "BC1", mipmaps: bool = True, *, srgb: bool = True
) -> bytes:
    """A complete .dds file: DX9 header (FourCC DXT1/DXT5) + all mip levels.

    BC1 for opaque textures, BC3 when alpha matters. Both sides must be multiples of 4, and
    powers of two when `mipmaps` is set (the chain then runs down to 1x1, with the 2x2 and 1x1
    levels stored as one padded 4x4 block each). `srgb` selects gamma-correct mip filtering.
    """
    if fmt not in _FOURCC:
        raise ValueError(f"fmt must be 'BC1' or 'BC3', got {fmt!r}")
    w, h = img.size
    if w % 4 or h % 4:
        raise ValueError(f"BC1/BC3 textures need sides divisible by 4, got {w}x{h}")
    mode = "RGBA" if fmt == "BC3" else "RGB"
    if img.mode != mode:
        img = img.convert(mode)
    levels = mip_chain(img, srgb=srgb) if mipmaps else [np.asarray(img)]
    encode = encode_bc1 if fmt == "BC1" else encode_bc3
    parts = [_dds_header(w, h, fmt, len(levels))]
    parts.extend(encode(level) for level in levels)
    return b"".join(parts)


@dataclass(frozen=True)
class DdsLevel:
    width: int
    height: int
    offset: int
    size: int


@dataclass(frozen=True)
class DdsInfo:
    """Parsed header of a block-compressed DDS file (as written by `encode_dds`)."""

    width: int
    height: int
    flags: int
    caps: int
    pf_flags: int
    fourcc: bytes
    linear_size: int
    mip_count: int
    depth: int
    levels: tuple[DdsLevel, ...]

    @property
    def format(self) -> str:
        return _FORMAT_OF_FOURCC.get(self.fourcc, self.fourcc.decode("ascii", "replace"))


def read_dds_info(data: bytes) -> DdsInfo:
    """Parse a DXT1/DXT5 DDS header and locate each mip level's bytes.

    Raises ValueError for anything else (bad magic or size, other FourCCs, truncated data).
    """
    if len(data) < _DDS_HEADER.size:
        raise ValueError("data too short for a DDS header")
    fields = _DDS_HEADER.unpack_from(data, 0)
    magic, size, flags, height, width, linear, depth, mips = fields[:8]
    pf_size, pf_flags, fourcc = fields[19:22]
    caps = fields[27]  # dwCaps; fields[22:27] are the pixel format's bit count and masks
    if magic != DDS_MAGIC or size != DDS_HEADER_SIZE or pf_size != DDS_PIXELFORMAT_SIZE:
        raise ValueError("not a DDS file (bad magic or header size)")
    if not pf_flags & DDPF_FOURCC or fourcc not in _FORMAT_OF_FOURCC:
        raise ValueError(f"unsupported DDS pixel format {fourcc!r}")
    fmt = _FORMAT_OF_FOURCC[fourcc]
    count = mips if (flags & DDSD_MIPMAPCOUNT and mips > 0) else 1
    levels = []
    offset = _DDS_HEADER.size
    w, h = width, height
    for _ in range(count):
        n = _level_size(w, h, fmt)
        levels.append(DdsLevel(w, h, offset, n))
        offset += n
        w, h = max(1, w // 2), max(1, h // 2)
    if offset > len(data):
        raise ValueError(f"DDS data truncated: need {offset} bytes, have {len(data)}")
    return DdsInfo(
        width=width,
        height=height,
        flags=flags,
        caps=caps,
        pf_flags=pf_flags,
        fourcc=fourcc,
        linear_size=linear,
        mip_count=mips,
        depth=depth,
        levels=tuple(levels),
    )


def decode_dds_level(data: bytes, level: int = 0) -> Image.Image:
    """Decode one mip level of a DXT1/DXT5 DDS with Pillow (RGBA image), for checks/debugging."""
    info = read_dds_info(data)
    lv = info.levels[level]
    header = _dds_header(lv.width, lv.height, info.format, 1)
    img = Image.open(io.BytesIO(header + data[lv.offset : lv.offset + lv.size]))
    img.load()
    return img.convert("RGBA")


def kn5_texture(
    name: str, img: Image.Image, fmt: DdsFormat = "BC1", *, mipmaps: bool = True
) -> Texture:
    """A KN5 `Texture` from an image: PNG bytes for "*.png" names, DDS for "*.dds" names."""
    if not name.isascii():
        raise ValueError(f"texture names must be ASCII: {name!r}")
    lower = name.lower()
    if lower.endswith(".png"):
        return Texture(name=name, data=encode_png(img))
    if lower.endswith(".dds"):
        return Texture(name=name, data=encode_dds(img, fmt, mipmaps))
    raise ValueError(f"texture name must end in .png or .dds: {name!r}")
