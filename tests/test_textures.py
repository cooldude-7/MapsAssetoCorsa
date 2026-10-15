"""Tests for driftmaps.ac.textures: procedural textures, PNG and DDS (BC1/BC3 + mips) encoding.

DDS output is checked three ways: our header fields against the Microsoft DDS_HEADER layout
(parsed here with `struct`, independently of `read_dds_info`), Pillow's DDS decoder for the
pixels of every mip level, and the error against Pillow's own BC1/BC3 encoder as a baseline.
"""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest
from PIL import Image

from driftmaps.ac import textures as tx

DDSD_REQUIRED = 0x1 | 0x2 | 0x4 | 0x1000  # CAPS | HEIGHT | WIDTH | PIXELFORMAT
DDSD_MIPMAPCOUNT = 0x20000
DDSD_LINEARSIZE = 0x80000
DDSCAPS_COMPLEX, DDSCAPS_TEXTURE, DDSCAPS_MIPMAP = 0x8, 0x1000, 0x400000
DDPF_FOURCC = 0x4


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((np.asarray(a, float) - np.asarray(b, float)) ** 2).mean()))


def parse_header(data: bytes) -> dict[str, object]:
    """DDS_HEADER fields by offset (Microsoft docs), without using the module under test."""
    assert data[:4] == b"DDS "
    size, flags, height, width, linear, depth, mips = struct.unpack_from("<7I", data, 4)
    pf_size, pf_flags = struct.unpack_from("<2I", data, 76)
    fourcc = data[84:88]
    bitcount, *masks = struct.unpack_from("<5I", data, 88)
    caps, caps2, caps3, caps4, reserved2 = struct.unpack_from("<5I", data, 108)
    return {
        "size": size,
        "flags": flags,
        "height": height,
        "width": width,
        "linear": linear,
        "depth": depth,
        "mips": mips,
        "pf_size": pf_size,
        "pf_flags": pf_flags,
        "fourcc": fourcc,
        "bitcount": bitcount,
        "masks": masks,
        "caps": caps,
        "caps2": caps2,
        "reserved1": data[32:76],
    }


def pillow_dds(img: Image.Image, pixel_format: str) -> np.ndarray:
    """Pillow's own encoder (top level only), decoded again: the quality baseline."""
    buf = io.BytesIO()
    img.save(buf, format="DDS", pixel_format=pixel_format)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGBA"))


# ---------------------------------------------------------------------------------------------
# Procedural textures
# ---------------------------------------------------------------------------------------------

GENERATORS = [tx.asphalt_texture, tx.grass_texture, tx.gravel_texture]


@pytest.mark.parametrize("gen", GENERATORS)
def test_generators_are_deterministic_and_seeded(gen):
    a = gen(128, seed=7)
    b = gen(128, seed=7)
    c = gen(128, seed=8)
    assert a.mode == "RGB" and a.size == (128, 128)
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != c.tobytes()


@pytest.mark.parametrize("gen", GENERATORS)
def test_generators_tile_without_a_seam(gen):
    arr = np.asarray(gen(128), dtype=float)
    inner_x = np.abs(np.diff(arr, axis=1)).mean()
    inner_y = np.abs(np.diff(arr, axis=0)).mean()
    seam_x = np.abs(arr[:, 0] - arr[:, -1]).mean()
    seam_y = np.abs(arr[0] - arr[-1]).mean()
    assert seam_x < 1.5 * inner_x
    assert seam_y < 1.5 * inner_y


def test_material_colours_are_plausible():
    asphalt = np.asarray(tx.asphalt_texture(64), float).mean((0, 1))
    grass = np.asarray(tx.grass_texture(64), float).mean((0, 1))
    gravel = np.asarray(tx.gravel_texture(64), float).mean((0, 1))
    assert 50 < asphalt.mean() < 100 and np.ptp(asphalt) < 6  # dark, neutral grey
    assert grass[1] > grass[0] and grass[1] > grass[2]  # green
    assert 40 < gravel.mean() < 140


def test_solid_texture():
    rgb = tx.solid_texture((10, 20, 30), 8)
    rgba = tx.solid_texture((10, 20, 30, 40), (4, 8))
    assert rgb.mode == "RGB" and rgb.size == (8, 8)
    assert rgba.mode == "RGBA" and rgba.size == (4, 8)
    assert (np.asarray(rgba) == (10, 20, 30, 40)).all()
    with pytest.raises(ValueError):
        tx.solid_texture((1, 2), 4)


def _fg_mask(img: Image.Image, fg=(255, 255, 255)) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), int)
    return np.abs(arr - np.array(fg)).sum(-1) < 60


@pytest.mark.parametrize("text", ["N", "A", "B", "PNG", "DDS", "NPOT"])
def test_label_texture_draws_big_centred_text(text):
    img = tx.label_texture(text, 256)
    mask = _fg_mask(img)
    ys, xs = np.nonzero(mask)
    assert img.mode == "RGB" and img.size == (256, 256)
    assert mask.mean() > 0.05  # big, not a speck
    assert abs(xs.mean() - 128) < 20 and abs(ys.mean() - 128) < 25
    assert tx.label_texture(text, 256).tobytes() == img.tobytes()


@pytest.mark.parametrize("direction", ["up", "down", "left", "right"])
def test_label_arrow_points_the_right_way(direction):
    mask = _fg_mask(tx.label_texture("", 200, arrow=direction))
    # The arrow head (widest part) sits on the side it points to.
    axis = 1 if direction in ("up", "down") else 0
    widest = int(np.argmax(mask.sum(axis=axis)))
    if direction in ("up", "left"):
        assert widest < 100
    else:
        assert widest > 100


def test_label_with_transparent_background_is_rgba():
    img = tx.label_texture("B", 64, bg=(0, 0, 0, 0), arrow="up")
    arr = np.asarray(img)
    assert img.mode == "RGBA"
    assert arr[0, 0, 3] == 0 and arr[-1, -1, 3] == 0
    assert arr[..., 3].max() == 255


def test_debug_grid_texture():
    img = tx.debug_grid_texture(256, cells=4, metres_per_cell=2.0)
    arr = np.asarray(img, int)
    assert img.size == (256, 256)
    assert arr[1, 128, 0] > 180 and arr[1, 128, 1] < 80  # red bar along the top (+u)
    assert arr[128, 1, 1] > 140 and arr[128, 1, 0] < 80  # green bar down the left (+v)
    # Checker: cell (0, 0) light, cell (0, 1) dark, read away from the labels.
    assert arr[50, 10].sum() > arr[50, 74].sum()
    assert tx.debug_grid_texture(256, cells=4, metres_per_cell=2.0).tobytes() == img.tobytes()


# ---------------------------------------------------------------------------------------------
# PNG
# ---------------------------------------------------------------------------------------------


def test_encode_png_round_trips_and_is_deterministic():
    img = tx.label_texture("PNG", (100, 60), bg=(10, 20, 30, 128))
    data = tx.encode_png(img)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data == tx.encode_png(img)
    back = Image.open(io.BytesIO(data))
    assert back.mode == "RGBA" and back.tobytes() == img.tobytes()


# ---------------------------------------------------------------------------------------------
# DDS
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("fmt", "fourcc", "block"), [("BC1", b"DXT1", 8), ("BC3", b"DXT5", 16)])
def test_dds_header_fields_and_mip_chain(fmt, fourcc, block):
    img = tx.asphalt_texture(256)
    data = tx.encode_dds(img, fmt)
    h = parse_header(data)
    assert h["size"] == 124 and h["pf_size"] == 32
    assert h["flags"] & DDSD_REQUIRED == DDSD_REQUIRED
    assert h["flags"] & DDSD_MIPMAPCOUNT and h["flags"] & DDSD_LINEARSIZE
    assert not h["flags"] & 0x8  # no PITCH flag for compressed data
    assert (h["width"], h["height"]) == (256, 256)
    assert h["linear"] == (256 // 4) * (256 // 4) * block
    assert h["mips"] == 9  # 256, 128, ..., 2, 1
    assert h["pf_flags"] == DDPF_FOURCC and h["fourcc"] == fourcc
    assert h["bitcount"] == 0 and h["masks"] == [0, 0, 0, 0]
    assert h["caps"] == DDSCAPS_TEXTURE | DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
    assert h["caps2"] == 0 and h["reserved1"] == bytes(44)
    # Levels: max(1, ceil(side / 4))^2 blocks each; 2x2 and 1x1 take one padded block.
    sides = [256 >> i for i in range(9)]
    expected = sum(max(1, (s + 3) // 4) ** 2 * block for s in sides)
    assert len(data) == 128 + expected
    info = tx.read_dds_info(data)
    assert [(lv.width, lv.height) for lv in info.levels] == [(s, s) for s in sides]
    assert info.levels[-1].size == block and info.levels[-2].size == block
    assert info.format == fmt


def test_dds_header_matches_kn5_texture_info():
    """Cross-check with the KN5 module's independent DDS sniffer (used by kn5-dump)."""
    kn5 = pytest.importorskip("driftmaps.ac.kn5")
    info = kn5.texture_info(tx.encode_dds(tx.grass_texture(64), "BC1"))
    assert "64x64" in info and "DXT1" in info and "mips=7" in info


@pytest.mark.parametrize("gen", GENERATORS)
def test_bc1_decodes_with_pillow_and_beats_pillows_encoder(gen):
    img = gen(256)
    data = tx.encode_dds(img, "BC1")
    decoded = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))  # Pillow reads level 0
    assert decoded.shape == (256, 256, 4)
    assert (decoded[..., 3] == 255).all()  # opaque: never BC1's 3-colour transparent index
    ours = rmse(decoded[..., :3], np.asarray(img))
    baseline = rmse(pillow_dds(img, "DXT1")[..., :3], np.asarray(img))
    assert ours < 6.0
    assert ours <= baseline * 1.05


def test_every_mip_level_decodes_and_keeps_the_average_colour():
    img = tx.grass_texture(64)
    data = tx.encode_dds(img, "BC1")
    info = tx.read_dds_info(data)
    lin = (np.asarray(img, float) / 255.0) ** 2.2
    expected = 255.0 * lin.mean((0, 1)) ** (1 / 2.2)  # mips average in linear light
    for i, lv in enumerate(info.levels):
        level = tx.decode_dds_level(data, i)
        assert level.size == (lv.width, lv.height)
        mean = np.asarray(level, float)[..., :3].mean((0, 1))
        assert np.abs(mean - expected).max() < 6.0, (i, mean, expected)


def test_bc3_preserves_alpha():
    img = tx.label_texture("DDS", 128, bg=(0, 0, 0, 0), arrow="up")
    data = tx.encode_dds(img, "BC3")
    decoded = np.asarray(tx.decode_dds_level(data, 0))
    src = np.asarray(img)
    assert rmse(decoded[..., 3], src[..., 3]) < 2.0
    assert (decoded[..., 3][src[..., 3] == 0] == 0).all()
    assert (decoded[..., 3][src[..., 3] == 255] == 255).all()
    visible = src[..., 3] == 255
    assert rmse(decoded[..., :3][visible], src[..., :3][visible]) < 8.0
    assert rmse(decoded, src) <= rmse(pillow_dds(img, "DXT5"), src) * 1.05


@pytest.mark.parametrize("colour", [(0, 0, 0), (255, 255, 255), (200, 100, 50), (8, 4, 8)])
def test_bc1_flat_colour_blocks_stay_opaque(colour):
    """Equal endpoints switch BC1 to 3-colour mode, where index 3 is transparent black."""
    img = tx.solid_texture(colour, 8)
    data = tx.encode_dds(img, "BC1", mipmaps=False)
    decoded = np.asarray(tx.decode_dds_level(data, 0), int)
    assert (decoded[..., 3] == 255).all()
    assert np.abs(decoded[..., :3] - np.array(colour)).max() <= 8  # RGB565 precision


def test_bc1_exact_for_two_565_colours():
    a, b = (255, 0, 0), (0, 0, 255)  # both exactly representable in RGB565
    arr = np.zeros((8, 8, 3), np.uint8)
    arr[...] = a
    arr[::2, ::2] = b
    img = Image.fromarray(arr, "RGB")
    decoded = np.asarray(tx.decode_dds_level(tx.encode_dds(img, "BC1", mipmaps=False)))
    assert (decoded[..., :3] == arr).all()


def test_dds_size_rules():
    with pytest.raises(ValueError, match="power-of-two"):
        tx.encode_dds(tx.solid_texture((1, 2, 3), (12, 8)), "BC1")
    with pytest.raises(ValueError, match="divisible by 4"):
        tx.encode_dds(tx.solid_texture((1, 2, 3), (6, 8)), "BC1", mipmaps=False)
    with pytest.raises(ValueError):
        tx.encode_dds(tx.solid_texture((1, 2, 3), 8), "BC7")  # type: ignore[arg-type]
    # Non-power-of-two (multiple of 4) without mips is fine.
    img = tx.label_texture("NPOT", (20, 12))
    data = tx.encode_dds(img, "BC1", mipmaps=False)
    h = parse_header(data)
    assert (h["width"], h["height"], h["mips"]) == (20, 12, 1)
    assert h["caps"] == DDSCAPS_TEXTURE  # no COMPLEX/MIPMAP for a single surface
    assert len(data) == 128 + 5 * 3 * 8
    assert Image.open(io.BytesIO(data)).size == (20, 12)


def test_non_square_mip_chain_runs_to_1x1():
    data = tx.encode_dds(tx.asphalt_texture(64).resize((64, 16)), "BC1")
    info = tx.read_dds_info(data)
    dims = [(lv.width, lv.height) for lv in info.levels]
    assert dims == [(64, 16), (32, 8), (16, 4), (8, 2), (4, 1), (2, 1), (1, 1)]
    for i in range(len(dims)):
        assert tx.decode_dds_level(data, i).size == dims[i]


def test_mips_do_not_bleed_transparent_colour():
    arr = np.zeros((4, 4, 4), np.uint8)
    arr[:, :2] = (255, 0, 0, 255)  # opaque red
    arr[:, 2:] = (0, 255, 0, 0)  # invisible green
    levels = tx.mip_chain(Image.fromarray(arr, "RGBA"))
    assert [lv.shape[:2] for lv in levels] == [(4, 4), (2, 2), (1, 1)]
    last = levels[-1][0, 0]
    assert tuple(last[:3]) == (255, 0, 0)
    assert abs(int(last[3]) - 128) <= 1


def test_gamma_correct_mips():
    arr = np.zeros((2, 2, 3), np.uint8)
    arr[0] = 255  # half white, half black
    levels = tx.mip_chain(Image.fromarray(arr, "RGB"))
    assert 185 <= levels[-1][0, 0, 0] <= 189  # linear 0.5 in sRGB, not 128
    assert tx.mip_chain(Image.fromarray(arr, "RGB"), srgb=False)[-1][0, 0, 0] in (127, 128)


def test_dds_is_deterministic():
    img = tx.gravel_texture(64)
    assert tx.encode_dds(img, "BC3") == tx.encode_dds(img, "BC3")


def test_read_dds_info_rejects_bad_data():
    good = tx.encode_dds(tx.solid_texture((1, 2, 3), 8), "BC1")
    with pytest.raises(ValueError):
        tx.read_dds_info(b"PNG " + good[4:])
    with pytest.raises(ValueError):
        tx.read_dds_info(good[:-1])


def test_kn5_texture():
    img = tx.asphalt_texture(32)
    png = tx.kn5_texture("asphalt.png", img)
    dds = tx.kn5_texture("asphalt.dds", img, "BC1")
    assert png.name == "asphalt.png" and png.data[:4] == b"\x89PNG"
    assert dds.data[:4] == b"DDS " and dds.data[84:88] == b"DXT1"
    with pytest.raises(ValueError):
        tx.kn5_texture("asphalt.jpg", img)
    with pytest.raises(ValueError):
        tx.kn5_texture("asphält.png", img)
