"""Procedural tree atlas for the KN5 card trees (docs/research/scenery.md §4.1, §6, §7).

One RGBA atlas holds every silhouette the vegetation builder (`driftmaps.geometry.vegetation`)
uses: four broadleaf crowns (maple, oak, poplar, birch), three conifers (spruce, white pine,
cedar) and two shrubs. Everything is painted here from seeded noise with numpy, so we own the
pixels outright (no licence work) and the same size + seed always gives the same bytes.

How a tile is painted (at 2x supersampling, then box-filtered down):
- Crowns are built from many small "leaf clusters": blobs whose edges are eaten by fine leaf
  noise, shaded as little spheres lit from the upper left, darker towards the back and the bottom
  of the crown (self-shadowing), with per-cluster colour jitter. Conifers are stacks of drooping
  branch tiers; shrubs are domes that reach the ground.
- Trunks and branches are tapered capsules with a cylinder shading and bark streaks.
- Each tile keeps a transparent margin at its left, right and top edges (so mips don't bleed
  between tiles); the trunk or foliage reaches the bottom edge, which is the ground line.
- The colour of transparent pixels is filled with the blurred colour of nearby foliage, so
  bilinear filtering at the alpha-test edge never pulls in black (no dark halos).

Encoding: `encode_alpha_tested_dds` writes BC3 (DXT5) with a full mip chain whose alpha is
rescaled per level and per tile so the fraction of texels passing the alpha test stays what it is
at the top level. Plain box-filtered mips make alpha-tested foliage thin out and vanish with
distance (the usual "alpha coverage" fix; the scale factor is found by bisection).

Image orientation: row 0 is the top of the image (UV v = 0 in DirectX/AC). In each tile the
ground line is the bottom row; a card maps the tile's bottom edge to the tree base.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from driftmaps.ac.model import Texture
from driftmaps.ac.textures import _dds_header, encode_bc3, encode_png, mip_chain

__all__ = [
    "ATLAS_SEED",
    "ATLAS_SIZE",
    "BUSH",
    "CONIFER",
    "DECIDUOUS",
    "KIND_NAMES",
    "TREE_TILES",
    "TreeTile",
    "alpha_coverage",
    "alpha_tested_mips",
    "encode_alpha_tested_dds",
    "tile_pixel_box",
    "tile_uv",
    "tiles_of_kind",
    "tree_atlas",
    "tree_atlas_texture",
]

# Tree kinds (what the land cover decides; docs/research/scenery.md §7 "Tree species").
DECIDUOUS = 0
CONIFER = 1
BUSH = 2
KIND_NAMES = ("deciduous", "conifer", "bush")

ATLAS_SIZE = 1024
ATLAS_SEED = 20261015
_SS = 2  # supersampling factor while painting
_MARGIN = 0.035  # transparent margin at the left/right/top of a tile (fraction of tile width)
_LIGHT = np.array([-0.35, -0.75, 0.56]) / np.linalg.norm([-0.35, -0.75, 0.56])  # image x, y, z


@dataclass(frozen=True)
class TreeTile:
    """One silhouette in the atlas.

    `rect` = (x0, y0, x1, y1) as fractions of the atlas (y down). `fill` = the fraction of the
    tile width the crown spans at its widest (a card is made 2 * crown radius / fill wide so
    the visible crown matches the crown measured in the lidar). `crown_base` = the fraction of
    the tile height, from the bottom, where foliage starts (bare trunk below it).
    """

    name: str
    kind: int
    rect: tuple[float, float, float, float]
    fill: float
    crown_base: float

    @property
    def aspect(self) -> float:
        """Width / height of the tile in pixels (the atlas is square)."""
        x0, y0, x1, y1 = self.rect
        return (x1 - x0) / (y1 - y0)


TREE_TILES: tuple[TreeTile, ...] = (
    TreeTile("maple", DECIDUOUS, (0.00, 0.00, 0.25, 0.50), 0.92, 0.30),
    TreeTile("oak", DECIDUOUS, (0.25, 0.00, 0.50, 0.50), 0.93, 0.28),
    TreeTile("poplar", DECIDUOUS, (0.50, 0.00, 0.75, 0.50), 0.62, 0.16),
    TreeTile("spruce", CONIFER, (0.75, 0.00, 1.00, 0.50), 0.90, 0.03),
    TreeTile("white_pine", CONIFER, (0.00, 0.50, 0.25, 1.00), 0.92, 0.30),
    TreeTile("birch", DECIDUOUS, (0.25, 0.50, 0.50, 1.00), 0.80, 0.34),
    TreeTile("cedar", CONIFER, (0.50, 0.50, 0.75, 1.00), 0.78, 0.02),
    TreeTile("shrub_a", BUSH, (0.75, 0.50, 1.00, 0.75), 0.92, 0.0),
    TreeTile("shrub_b", BUSH, (0.75, 0.75, 1.00, 1.00), 0.92, 0.0),
)


def tiles_of_kind(kind: int) -> tuple[int, ...]:
    """Indices into `TREE_TILES` of every tile of one kind."""
    return tuple(i for i, t in enumerate(TREE_TILES) if t.kind == kind)


def tile_pixel_box(index: int, size: int = ATLAS_SIZE) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) pixel bounds of tile `index` in a `size` x `size` atlas (end-exclusive)."""
    x0, y0, x1, y1 = TREE_TILES[index].rect
    return round(x0 * size), round(y0 * size), round(x1 * size), round(y1 * size)


def tile_uv(index: int, size: int = ATLAS_SIZE) -> tuple[float, float, float, float]:
    """(u0, v0, u1, v1) of tile `index`, inset by half a texel so bilinear filtering at the
    card's edges never reads the neighbouring tile."""
    x0, y0, x1, y1 = tile_pixel_box(index, size)
    h = 0.5
    return (x0 + h) / size, (y0 + h) / size, (x1 - h) / size, (y1 - h) / size


# ---------------------------------------------------------------------------------------------
# Painting
# ---------------------------------------------------------------------------------------------


def _noise(shape: tuple[int, int], rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Smooth noise with zero mean and unit standard deviation (blurred white noise)."""
    n = ndi.gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    s = n.std()
    return (n - n.mean()) / (s if s > 0 else 1.0)


def _rgb(*c: int) -> np.ndarray:
    return np.array(c, dtype=np.float64) / 255.0


class _Painter:
    """A supersampled RGBA canvas for one tile (float colours 0..1, binary coverage)."""

    def __init__(self, width: int, height: int, rng: np.random.Generator) -> None:
        self.w, self.h = width * _SS, height * _SS
        self.rng = rng
        self.rgb = np.zeros((self.h, self.w, 3))
        self.a = np.zeros((self.h, self.w))
        self.leaf = _noise((self.h, self.w), rng, 1.3 * _SS)  # leaf-sized grain
        self.mottle = _noise((self.h, self.w), rng, 7.0 * _SS)  # sun patches
        self.margin = _MARGIN * self.w

    def _window(self, cx: float, cy: float, rx: float, ry: float):
        x0 = max(int(cx - rx) - 1, 0)
        x1 = min(int(cx + rx) + 2, self.w)
        y0 = max(int(cy - ry) - 1, 0)
        y1 = min(int(cy + ry) + 2, self.h)
        if x0 >= x1 or y0 >= y1:
            return None
        ys, xs = np.mgrid[y0:y1, x0:x1] + 0.5
        return (slice(y0, y1), slice(x0, x1)), xs, ys

    def _clip_margins(self, xs: np.ndarray, ys: np.ndarray, inside: np.ndarray) -> None:
        inside &= (xs > self.margin) & (xs < self.w - self.margin) & (ys > self.margin)

    def blob(
        self,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        colour: np.ndarray,
        *,
        edge: float = 0.35,
        shade: float = 1.0,
    ) -> None:
        """A leaf cluster: an ellipse with a leafy ragged edge, shaded as a lit sphere."""
        win = self._window(cx, cy, rx * (1 + edge), ry * (1 + edge))
        if win is None:
            return
        sl, xs, ys = win
        dx = (xs - cx) / rx
        dy = (ys - cy) / ry
        d2 = dx * dx + dy * dy
        leaf = self.leaf[sl]
        inside = np.sqrt(d2) + edge * 0.5 * leaf < 1.0
        self._clip_margins(xs, ys, inside)
        if not inside.any():
            return
        nz = np.sqrt(np.clip(1.0 - d2, 0.0, 1.0))
        norm = np.sqrt(d2 + nz * nz) + 1e-9
        lam = np.clip((dx * _LIGHT[0] + dy * _LIGHT[1] + nz * _LIGHT[2]) / norm, 0.0, 1.0)
        tone = shade * (0.58 + 0.52 * lam) * (1.0 + 0.10 * leaf + 0.07 * self.mottle[sl])
        col = colour[None, None, :] * tone[..., None]
        self.rgb[sl][inside] = col[inside]
        self.a[sl][inside] = 1.0

    def limb(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        r0: float,
        r1: float,
        colour: np.ndarray,
        *,
        marks: float = 0.0,
    ) -> None:
        """A tapered trunk or branch from (x0, y0) radius r0 to (x1, y1) radius r1, shaded as a
        cylinder lit from the left. `marks` > 0 adds dark bark marks (birch)."""
        r = max(r0, r1)
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        half = 0.5 * math.hypot(x1 - x0, y1 - y0)
        win = self._window(cx, cy, half + r, half + r)
        if win is None:
            return
        sl, xs, ys = win
        vx, vy = x1 - x0, y1 - y0
        ll = vx * vx + vy * vy + 1e-9
        t = np.clip(((xs - x0) * vx + (ys - y0) * vy) / ll, 0.0, 1.0)
        px, py = x0 + t * vx - xs, y0 + t * vy - ys
        dist = np.sqrt(px * px + py * py)
        rad = r0 + (r1 - r0) * t
        inside = dist < rad
        self._clip_margins(xs, ys, inside)
        if not inside.any():
            return
        # Signed offset across the limb (+ = right of the direction of travel) for shading.
        side = np.sign(px * vy - py * vx) * dist / np.maximum(rad, 1e-6)
        tone = 0.78 - 0.30 * side + 0.08 * self.leaf[sl]
        streak = ndi.gaussian_filter1d(self.leaf[sl], 3.0, axis=0)
        tone = tone * (1.0 + 0.08 * streak)
        col = colour[None, None, :] * tone[..., None]
        if marks > 0:
            dark = (self.mottle[sl] > 1.15 - marks) | (np.abs(streak) > 1.4)
            col = np.where(dark[..., None], col * 0.25, col)
        self.rgb[sl][inside] = col[inside]
        self.a[sl][inside] = 1.0

    def result(self) -> np.ndarray:
        """(h, w, 4) float RGBA at final resolution (box filter, alpha-weighted colour)."""
        s = _SS
        h, w = self.h // s, self.w // s
        a = self.a.reshape(h, s, w, s).mean(axis=(1, 3))
        pre = (self.rgb * self.a[..., None]).reshape(h, s, w, s, 3).sum(axis=(1, 3))
        cnt = self.a.reshape(h, s, w, s).sum(axis=(1, 3))
        rgb = np.where(cnt[..., None] > 0, pre / np.maximum(cnt, 1e-9)[..., None], 0.0)
        return np.concatenate([rgb, a[..., None]], axis=-1)


def _jitter(rng: np.random.Generator, colour: np.ndarray, amount: float) -> np.ndarray:
    return np.clip(colour * (1.0 + rng.normal(0.0, amount, 3)) * rng.normal(1.0, amount), 0, 1)


def _broadleaf(
    p: _Painter,
    tile: TreeTile,
    leaf: np.ndarray,
    bark: np.ndarray,
    *,
    clusters: int,
    cluster_r: float,
    top: float = 0.02,
    squareness: float = 2.0,
    airy: float = 0.0,
    trunk_r: float = 0.038,
    marks: float = 0.0,
) -> None:
    """A broadleaf tree: trunk + branches into an elliptical crown of clustered leaf blobs."""
    rng = p.rng
    w, h = p.w, p.h
    cx = w / 2
    ground = h + 2.0
    crown_bottom = h * (1.0 - tile.crown_base)
    crown_top = h * top
    ecy = 0.5 * (crown_top + crown_bottom)
    erx = 0.5 * w * tile.fill
    ery = 0.5 * (crown_bottom - crown_top)
    # Trunk, slightly leaning, then branches fanning into the crown.
    lean = rng.normal(0.0, 0.02 * w)
    trunk_top = (cx + lean, ecy + 0.15 * ery)
    p.limb(cx, ground, *trunk_top, trunk_r * w, 0.45 * trunk_r * w, bark, marks=marks)
    for _ in range(rng.integers(5, 8)):
        f = rng.uniform(0.25, 0.85)
        sx = cx + lean * f
        sy = ground + (trunk_top[1] - ground) * rng.uniform(0.75, 1.0)
        ang = rng.uniform(-0.9 * math.pi, -0.1 * math.pi)  # upwards (image y down)
        reach = rng.uniform(0.45, 0.85)
        ex = cx + math.cos(ang) * erx * reach
        ey = ecy + math.sin(ang) * ery * reach * 0.8
        p.limb(sx, sy, ex, ey, 0.5 * trunk_r * w, 0.15 * trunk_r * w, bark, marks=marks)
    # Leaf clusters: centres inside the (super-)ellipse, shrunk by the blob size.
    r_blob = cluster_r * w
    pts = []
    while len(pts) < clusters:
        u, v = rng.uniform(-1, 1, 2)
        if abs(u) ** squareness + abs(v) ** squareness > 1.0:
            continue
        if airy and rng.random() < airy * (1.0 - abs(u) ** 2 - abs(v) ** 2):
            continue  # thin the inside of airy crowns (sky gaps)
        pts.append((cx + u * (erx - r_blob), ecy + v * (ery - 0.7 * r_blob), rng.uniform(-1, 1)))
    pts.sort(key=lambda q: q[2])  # back to front
    for x, y, z in pts:
        base = _jitter(rng, leaf, 0.05)
        yrel = (y - crown_top) / max(crown_bottom - crown_top, 1.0)
        shade = (0.70 + 0.30 * (z + 1) / 2) * (1.08 - 0.30 * yrel)
        for _ in range(rng.integers(3, 6)):
            ox, oy = rng.normal(0.0, 0.45 * r_blob, 2)
            rr = r_blob * rng.uniform(0.45, 0.8)
            p.blob(x + ox, y + oy, rr, rr * rng.uniform(0.8, 1.0), base, edge=0.55, shade=shade)


def _conifer_tiers(
    p: _Painter,
    tile: TreeTile,
    leaf: np.ndarray,
    bark: np.ndarray,
    *,
    tiers: int,
    droop: float,
    needle_r: float,
    profile: float,
    top: float = 0.01,
    trunk_r: float = 0.03,
) -> None:
    """A conical conifer: a trunk and drooping branch tiers from the top down to `crown_base`.
    `profile` < 1 makes the cone convex (bushier), > 1 concave (spikier)."""
    rng = p.rng
    w, h = p.w, p.h
    cx = w / 2
    ground = h + 2.0
    y_top = h * top
    y_base = h * (1.0 - tile.crown_base)
    half = 0.5 * w * tile.fill
    p.limb(cx, ground, cx, y_top + 0.05 * h, trunk_r * w, 0.25 * trunk_r * w, bark)
    r_n = needle_r * w
    # A dark core so the cone doesn't look hollow between tiers.
    for k in range(tiers * 2):
        f = (k + 0.5) / (tiers * 2)
        y = y_top + (y_base - y_top) * f
        hw = half * max(f, 0.05) ** profile * 0.35
        p.blob(cx, y, max(hw, r_n), 1.6 * r_n, _jitter(rng, leaf, 0.04), edge=0.6, shade=0.55)
    # Tiers from the bottom up, so upper tiers' drooping tips overlap the ones below.
    for k in reversed(range(tiers)):
        f = (k + 0.5 + rng.uniform(-0.25, 0.25)) / tiers
        y = y_top + (y_base - y_top) * f
        hw = half * f**profile * rng.uniform(0.88, 1.06)
        hw = max(hw, 1.5 * r_n)
        base = _jitter(rng, leaf, 0.05)
        for side in (-1.0, 1.0):
            n = max(3, int(hw / (0.9 * r_n)))
            for j in range(n):
                t = (j + 0.5) / n
                bx = cx + side * hw * t
                by = y + droop * hw * t * t + rng.normal(0.0, 0.25 * r_n)
                rr = r_n * (1.25 - 0.55 * t) * rng.uniform(0.85, 1.15)
                shade = (0.72 + 0.35 * t) * (1.05 - 0.25 * f)
                p.blob(bx, by, rr * 1.35, rr * 0.85, base, edge=0.7, shade=shade)


def _white_pine(p: _Painter, tile: TreeTile, leaf: np.ndarray, bark: np.ndarray) -> None:
    """An eastern white pine: tall bare trunk, irregular horizontal layers of soft needles."""
    rng = p.rng
    w, h = p.w, p.h
    cx = w / 2
    ground = h + 2.0
    y_top = 0.03 * h
    y_base = h * (1.0 - tile.crown_base)
    half = 0.5 * w * tile.fill
    p.limb(cx, ground, cx, y_top + 0.06 * h, 0.04 * w, 0.012 * w, bark)
    layers = 7
    for k in reversed(range(layers)):
        f = (k + 0.5) / layers
        y = y_top + (y_base - y_top) * f
        reach = half * (0.35 + 0.65 * math.sin(math.pi * min(0.15 + 0.85 * f, 1.0) / 1.15))
        reach *= rng.uniform(0.75, 1.05)
        side = -1.0 if k % 2 else 1.0
        span = (cx - reach * rng.uniform(0.5, 1.0), cx + reach * rng.uniform(0.5, 1.0))
        if side < 0:
            span = (cx - reach, cx + reach * rng.uniform(0.3, 0.8))
        p.limb(cx, y + 0.02 * h, span[0] + 0.1 * reach, y, 0.012 * w, 0.004 * w, bark)
        p.limb(cx, y + 0.02 * h, span[1] - 0.1 * reach, y, 0.012 * w, 0.004 * w, bark)
        base = _jitter(rng, leaf, 0.05)
        pad_h = (y_base - y_top) / layers * 0.75
        for _ in range(int(14 + 22 * f)):
            bx = rng.uniform(*span)
            by = y + rng.normal(0.0, 0.3 * pad_h)
            rr = 0.055 * w * rng.uniform(0.7, 1.2)
            shade = rng.uniform(0.75, 1.0) * (1.05 - 0.2 * f)
            p.blob(bx, by, rr * 1.4, rr * 0.75, base, edge=0.65, shade=shade)


def _shrub(p: _Painter, leaf: np.ndarray, *, blobs: int, flat: float) -> None:
    """A dome-shaped shrub that reaches the ground."""
    rng = p.rng
    w, h = p.w, p.h
    cx = w / 2
    erx = 0.5 * w * 0.92
    ery = h * (0.92 - flat)
    cy = h + 0.05 * h
    r_b = 0.09 * w
    pts = []
    while len(pts) < blobs:
        u, v = rng.uniform(-1, 1), rng.uniform(-1, 0)
        if u * u + v * v > 1.0:
            continue
        pts.append((cx + u * (erx - r_b), cy + v * (ery - r_b), rng.uniform(-1, 1)))
    pts.sort(key=lambda q: q[2])
    for x, y, z in pts:
        yrel = (y - (cy - ery)) / ery
        shade = (0.70 + 0.3 * (z + 1) / 2) * (1.08 - 0.35 * yrel)
        rr = r_b * rng.uniform(0.6, 1.1)
        p.blob(x, y, rr, rr * 0.9, _jitter(rng, leaf, 0.06), edge=0.6, shade=shade)


def _paint_tile(index: int, width: int, height: int, seed: int) -> np.ndarray:
    tile = TREE_TILES[index]
    rng = np.random.default_rng([seed, index])
    p = _Painter(width, height, rng)
    bark = _rgb(92, 78, 64)
    match tile.name:
        case "maple":
            _broadleaf(p, tile, _rgb(78, 112, 42), bark, clusters=60, cluster_r=0.075)
        case "oak":
            _broadleaf(
                p, tile, _rgb(62, 92, 38), _rgb(80, 70, 58), clusters=52, cluster_r=0.085,
                top=0.05, squareness=3.0, trunk_r=0.048,
            )
        case "poplar":
            _broadleaf(
                p, tile, _rgb(88, 118, 48), bark, clusters=70, cluster_r=0.09, squareness=1.6,
                trunk_r=0.042,
            )
        case "birch":
            _broadleaf(
                p, tile, _rgb(104, 132, 52), _rgb(222, 220, 208), clusters=52,
                cluster_r=0.07, airy=0.45, trunk_r=0.032, marks=0.35,
            )
        case "spruce":
            _conifer_tiers(
                p, tile, _rgb(40, 66, 46), bark, tiers=17, droop=0.22, needle_r=0.045,
                profile=0.95,
            )
        case "cedar":
            _conifer_tiers(
                p, tile, _rgb(70, 92, 44), bark, tiers=22, droop=0.10, needle_r=0.05,
                profile=0.6, top=0.02,
            )
        case "white_pine":
            _white_pine(p, tile, _rgb(52, 78, 50), bark)
        case "shrub_a":
            _shrub(p, _rgb(72, 102, 44), blobs=55, flat=0.15)
        case "shrub_b":
            _shrub(p, _rgb(88, 106, 50), blobs=45, flat=0.3)
        case _:  # pragma: no cover - every tile has a painter
            raise ValueError(f"no painter for tile {tile.name!r}")
    return p.result()


def _fill_transparent(rgba: np.ndarray) -> np.ndarray:
    """Give fully transparent texels the blurred colour of nearby foliage (no dark halos)."""
    a = rgba[..., 3]
    rgb = rgba[..., :3]
    out = rgb.copy()
    todo = a <= 0
    if not todo.any() or not (~todo).any():
        return rgba
    mean = (rgb * a[..., None]).sum(axis=(0, 1)) / a.sum()
    filled = np.broadcast_to(mean, rgb.shape).copy()
    for sigma in (24.0, 8.0, 2.5):  # coarse to fine: nearer foliage wins
        den = ndi.gaussian_filter(a, sigma)
        num = np.stack([ndi.gaussian_filter(rgb[..., c] * a, sigma) for c in range(3)], -1)
        ok = den > 1e-3
        filled[ok] = num[ok] / den[ok][:, None]
    out[todo] = filled[todo]
    return np.concatenate([out, a[..., None]], axis=-1)


def tree_atlas(size: int = ATLAS_SIZE, seed: int = ATLAS_SEED) -> Image.Image:
    """The tree atlas as a `size` x `size` RGBA image (power of two, >= 64).

    Tiles are laid out as in `TREE_TILES`. Deterministic: same size and seed, same pixels.
    """
    if size < 64 or size & (size - 1):
        raise ValueError(f"atlas size must be a power of two >= 64, got {size}")
    atlas = np.zeros((size, size, 4))
    for i in range(len(TREE_TILES)):
        x0, y0, x1, y1 = tile_pixel_box(i, size)
        atlas[y0:y1, x0:x1] = _paint_tile(i, x1 - x0, y1 - y0, seed)
    atlas = _fill_transparent(atlas)
    arr = np.clip(np.rint(atlas * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


# ---------------------------------------------------------------------------------------------
# Alpha-tested mips and encodings
# ---------------------------------------------------------------------------------------------


def alpha_coverage(alpha: np.ndarray, alpha_ref: float = 0.5) -> float:
    """Fraction of texels whose alpha (uint8 0..255 or float 0..1) passes the alpha test."""
    alpha = np.asarray(alpha)
    ref = alpha_ref * 255.0 if alpha.dtype == np.uint8 else alpha_ref
    return float((alpha.astype(np.float64) >= ref).mean()) if alpha.size else 0.0


def _coverage_scale(alpha: np.ndarray, target: float, ref: float) -> float:
    """The factor s (0.25..16) for which coverage(alpha * s) is closest to `target`."""
    lo, hi = 0.25, 16.0
    if target <= 0.0:
        return 1.0
    for _ in range(24):
        mid = math.sqrt(lo * hi)
        if (alpha * mid >= ref).mean() < target:
            lo = mid
        else:
            hi = mid
    return hi


def alpha_tested_mips(
    img: Image.Image,
    alpha_ref: float = 0.5,
    rects: tuple[tuple[float, float, float, float], ...] | None = None,
) -> list[np.ndarray]:
    """The mip chain of an RGBA image (as `textures.mip_chain`), with each level's alpha
    rescaled so the share of texels passing `alpha_ref` matches the top level.

    `rects` (fractions of the image, (x0, y0, x1, y1)) are scaled separately, so every atlas
    tile keeps its own coverage; default = the whole image. Regions smaller than 2 x 2 texels
    at a level are left as filtered.
    """
    img = img.convert("RGBA")
    levels = mip_chain(img, srgb=True)
    rects = rects or ((0.0, 0.0, 1.0, 1.0),)
    ref = alpha_ref * 255.0

    def region(arr: np.ndarray, rect) -> tuple[slice, slice]:
        h, w = arr.shape[:2]
        x0, y0, x1, y1 = rect
        return slice(round(y0 * h), round(y1 * h)), slice(round(x0 * w), round(x1 * w))

    top = levels[0][..., 3].astype(np.float64)
    targets = [float((top[region(top, r)] >= ref).mean()) for r in rects]
    out = [levels[0]]
    for lv in levels[1:]:
        lv = lv.copy()
        a = lv[..., 3].astype(np.float64)
        new = a.copy()
        for rect, target in zip(rects, targets, strict=True):
            sl = region(a, rect)
            sub = a[sl]
            if sub.shape[0] < 2 or sub.shape[1] < 2:
                continue
            new[sl] = np.clip(sub * _coverage_scale(sub, target, ref), 0.0, 255.0)
        lv[..., 3] = np.rint(new).astype(np.uint8)
        out.append(lv)
    return out


def encode_alpha_tested_dds(
    img: Image.Image,
    alpha_ref: float = 0.5,
    rects: tuple[tuple[float, float, float, float], ...] | None = None,
) -> bytes:
    """A BC3 (DXT5) .dds with a full mip chain from `alpha_tested_mips` (same header layout as
    `textures.encode_dds`, so `read_dds_info` / `decode_dds_level` read it)."""
    w, h = img.size
    levels = alpha_tested_mips(img, alpha_ref, rects)
    return _dds_header(w, h, "BC3", len(levels)) + b"".join(encode_bc3(lv) for lv in levels)


def tree_atlas_texture(
    name: str = "dm_trees.dds",
    size: int = ATLAS_SIZE,
    seed: int = ATLAS_SEED,
    alpha_ref: float = 0.5,
) -> Texture:
    """The atlas as a KN5 `Texture`: BC3 DDS with coverage-preserving mips for "*.dds" names,
    PNG for "*.png" names."""
    if not name.isascii():
        raise ValueError(f"texture names must be ASCII: {name!r}")
    img = tree_atlas(size, seed)
    lower = name.lower()
    if lower.endswith(".dds"):
        rects = tuple(t.rect for t in TREE_TILES)
        return Texture(name=name, data=encode_alpha_tested_dds(img, alpha_ref, rects))
    if lower.endswith(".png"):
        return Texture(name=name, data=encode_png(img))
    raise ValueError(f"texture name must end in .png or .dds: {name!r}")
