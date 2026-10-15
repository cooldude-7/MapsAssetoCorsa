"""Tests for driftmaps.ac.track_files: every writer's output is parsed back.

INI files go through Python's configparser (a strict reader: one value per key, no inline
comments) and JSON through the json module. The map tests check CM's formula
(`px = (X + X_OFFSET) / SCALE_FACTOR`, `py = (Z + Z_OFFSET) / SCALE_FACTOR`, north up).
"""

from __future__ import annotations

import configparser
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from driftmaps.ac import track_files as tf


def read_ini(path: Path) -> configparser.ConfigParser:
    cp = configparser.ConfigParser(interpolation=None, strict=True)
    cp.optionxform = str  # keep KEY case
    with open(path, encoding="ascii", newline="") as f:
        text = f.read()
    assert "\r\n" in text and "\n" not in text.replace("\r\n", "")  # CRLF only
    cp.read_string(text)
    return cp


def vec(text: str) -> np.ndarray:
    return np.array([float(p) for p in text.split(",")])


def l_road(step: float = 1.0) -> np.ndarray:
    """300 m north, a 30 m-radius left turn, then 200 m west (AC frame: north = -Z)."""
    north = np.stack([np.zeros(301), np.zeros(301), -np.arange(0, 301.0, step)[:301]], 1)
    t = np.linspace(0, np.pi / 2, 40)[1:]
    arc = np.stack([-30 + 30 * np.cos(t), np.zeros_like(t), -300 - 30 * np.sin(t)], 1)
    west = np.stack([-30 - np.arange(1, 201.0), np.zeros(200), np.full(200, -330.0)], 1)
    return np.concatenate([north, arc, west])


# ---------------------------------------------------------------------------------------------
# Track ids
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("good", ["a", "dm_t1_minimal", "x" * 32, "0_9", "akina"])
def test_valid_track_ids(good):
    assert tf.validate_track_id(good) == good


@pytest.mark.parametrize(
    "bad", ["", "x" * 33, "Akina", "dm-test", "dm test", "dm.test", "café", "a/b"]
)
def test_invalid_track_ids(bad):
    with pytest.raises(ValueError):
        tf.validate_track_id(bad)


def test_make_track_id():
    assert tf.make_track_id("Akina Downhill") == "dm_akina_downhill"
    assert tf.make_track_id("Gorges d'Ollioules — D N8!") == "dm_gorges_d_ollioules_d_n8"
    assert tf.make_track_id("") == "dm_track"
    long_id = tf.make_track_id("A very long road name that keeps going and going")
    assert len(long_id) <= 32 and not long_id.endswith("_")
    tf.validate_track_id(long_id)


# ---------------------------------------------------------------------------------------------
# surfaces.ini
# ---------------------------------------------------------------------------------------------


def test_default_surfaces_ini(tmp_path):
    path = tf.write_surfaces_ini(tmp_path / "data" / "surfaces.ini")
    cp = read_ini(path)
    assert cp.sections() == ["SURFACE_0", "SURFACE_1", "SURFACE_2"]
    keys = [cp[s]["KEY"] for s in cp.sections()]
    assert keys == ["ROAD", "GRASS", "GRAVEL"]
    assert not any("WALL" in k for k in keys)
    road, grass, gravel = (cp[s] for s in cp.sections())
    # Vanilla system/data/surfaces.ini values for ROAD and GRASS.
    assert float(road["FRICTION"]) == 1.0 and road["IS_VALID_TRACK"] == "1"
    assert road["FF_EFFECT"] == "NULL" and road["WAV"] == ""
    assert float(grass["FRICTION"]) == 0.6 and grass["WAV"] == "grass.wav"
    assert grass["IS_VALID_TRACK"] == "0" and float(grass["SIN_HEIGHT"]) == 0.03
    assert float(gravel["FRICTION"]) == 0.75 and float(gravel["DIRT_ADDITIVE"]) == 1
    assert gravel["_EXT_SURFACE_TYPE"] == "GRAVEL"
    expected_order = [
        "KEY", "FRICTION", "DAMPING", "WAV", "WAV_PITCH", "FF_EFFECT", "DIRT_ADDITIVE",
        "BLACK_FLAG_TIME", "IS_VALID_TRACK", "SIN_HEIGHT", "SIN_LENGTH", "IS_PITLANE",
        "VIBRATION_GAIN", "VIBRATION_LENGTH",
    ]  # fmt: skip
    assert list(road.keys()) == expected_order


def test_surfaces_reject_wall_and_duplicates(tmp_path):
    with pytest.raises(ValueError, match="WALL"):
        tf.write_surfaces_ini(tmp_path / "s.ini", [tf.Surface("ROAD"), tf.Surface("WALL")])
    with pytest.raises(ValueError, match="duplicate"):
        tf.write_surfaces_ini(tmp_path / "s.ini", [tf.Surface("ROAD"), tf.Surface("ROAD")])
    with pytest.raises(ValueError):
        tf.write_surfaces_ini(tmp_path / "s.ini", [tf.Surface("road")])


# ---------------------------------------------------------------------------------------------
# ui_track.json
# ---------------------------------------------------------------------------------------------


def test_ui_track_json_uses_kunos_string_fields(tmp_path):
    ui = tf.UiTrack(
        name="Aurora Test",
        description="Generated. " + tf.attribution_text(["osm", "usgs_3dep"]),
        geotags=(41.71712, -88.29689),
        country="USA",
        city="Aurora",
        length_m=3612.6,
        width=7.5,
        pitboxes=8,
        year=2026,
    )
    path = tf.write_ui_track_json(tmp_path / "ui" / "ui_track.json", ui)
    raw = path.read_bytes()
    assert raw.isascii()
    d = json.loads(raw)
    assert d["name"] == "Aurora Test"
    assert d["length"] == "3613" and d["width"] == "7.5m" and d["pitboxes"] == "8"
    assert d["geotags"] == ["41.71712", "-88.29689"]
    assert [float(x) for x in d["geotags"]] == [41.71712, -88.29689]
    assert d["tags"] == ["A2B", "touge", "point-to-point", "generated"]
    assert "circuit" not in d["tags"]
    assert d["run"] == "A2B" and d["year"] == 2026 and isinstance(d["year"], int)
    for key in ("author", "version", "country", "city", "description"):
        assert isinstance(d[key], str)
    assert "OpenStreetMap" in d["description"]


def test_ui_track_json_optional_fields(tmp_path):
    ui = tf.UiTrack(name="Zürich loop", width="6m-8m", tags=["A2B"])
    d = json.loads(tf.write_ui_track_json(tmp_path / "ui_track.json", ui).read_text("ascii"))
    assert "geotags" not in d and "year" not in d and "url" not in d
    assert d["name"] == "Zürich loop" and d["width"] == "6m-8m" and d["length"] == "0"
    with pytest.raises(ValueError):
        tf.UiTrack(name="x", geotags=(-88.3, 141.7 + 100)).to_json_dict()


# ---------------------------------------------------------------------------------------------
# map.png + map.ini
# ---------------------------------------------------------------------------------------------


def test_map_formula_round_trip(tmp_path):
    road = l_road()
    info = tf.render_map([road], 8.0, tmp_path / "data", tmp_path, meters_per_px=0.5)
    img = Image.open(tmp_path / "map.png")
    assert img.mode == "RGBA" and img.size == info.size_px

    # map.ini parses back to the same constants, and WIDTH/HEIGHT are the image size.
    cp = read_ini(tmp_path / "data" / "map.ini")
    p = cp["PARAMETERS"]
    assert list(p.keys()) == [
        "WIDTH", "HEIGHT", "MARGIN", "SCALE_FACTOR", "MAX_SIZE", "X_OFFSET", "Z_OFFSET",
        "DRAWING_SIZE",
    ]  # fmt: skip
    assert (float(p["WIDTH"]), float(p["HEIGHT"])) == img.size
    assert float(p["SCALE_FACTOR"]) == 0.5 and float(p["MARGIN"]) == 20
    assert tf.read_map_ini(tmp_path / "data" / "map.ini") == info

    sf, xo, zo = float(p["SCALE_FACTOR"]), float(p["X_OFFSET"]), float(p["Z_OFFSET"])
    alpha = np.asarray(img)[..., 3]

    def pixel(x: float, z: float) -> tuple[int, int]:
        return int((x + xo) / sf), int((z + zo) / sf)  # (column, row)

    # Known world points on the road land on opaque white pixels.
    for x, z in [(0.0, 0.0), (0.0, -150.0), (0.0, -300.0), (-130.0, -330.0), (-230.0, -330.0)]:
        col, row = pixel(x, z)
        assert alpha[row, col] == 255, (x, z)
        assert tuple(np.asarray(img)[row, col, :3]) == (255, 255, 255)
    # Off the road it's transparent (9 m east of the 8 m road, and inside the bend).
    for x, z in [(9.0, -50.0), (-100.0, -200.0)]:
        col, row = pixel(x, z)
        assert alpha[row, col] == 0
    # The road (width 8 m) starts after the margin: margin * sf metres of empty border.
    assert (alpha[:, :19] == 0).all() and (alpha[:19] == 0).all()
    np.testing.assert_allclose(info.pixel_to_world(info.world_to_pixel(road)), road[:, [0, 2]])


def test_map_is_north_up_and_not_mirrored():
    info = tf.compute_map_info([l_road()], 8.0, meters_per_px=1.0)
    start, north_end, west_end = info.world_to_pixel(
        np.array([[0, 0, 0], [0, 0, -300], [-230, 0, -330]])
    )
    assert north_end[1] < start[1]  # north (-Z) is up the image
    assert west_end[0] < north_end[0]  # west (-X) is left
    # CM's live-map consumer: relX = (X + X_OFFSET) / (WIDTH * SCALE_FACTOR) in [0, 1].
    rel = (np.array([0.0, -230.0]) + info.x_offset) / (info.width * info.scale_factor)
    assert ((rel > 0) & (rel < 1)).all()
    np.testing.assert_allclose(rel * info.width, [start[0], west_end[0]])


def test_map_respects_max_size(tmp_path):
    long_road = np.stack([np.linspace(0, 9000, 200), np.zeros(200), np.linspace(0, -3000, 200)], 1)
    info = tf.render_map(
        [long_road], [7.0], tmp_path / "data", tmp_path, meters_per_px=1.0, max_size=1024
    )
    assert max(info.size_px) == 1024 and info.scale_factor > 1.0
    assert Image.open(tmp_path / "map.png").size == info.size_px
    assert info.max_size == tf.MAP_INI_MAX_SIZE  # the ini key stays Kunos' default
    px = info.world_to_pixel(long_road)
    assert px.min() >= info.margin - 1 and (px.max(0) <= np.array(info.size_px)).all()


def test_map_multiple_polylines_and_widths(tmp_path):
    a = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    b = np.array([[0.0, 5.0, -50.0], [100.0, 5.0, -50.0]])
    info = tf.render_map([a, b], [4.0, 20.0], tmp_path, tmp_path, meters_per_px=0.5)
    alpha = np.asarray(Image.open(tmp_path / "map.png"))[..., 3]
    col = int((50 + info.x_offset) / info.scale_factor)
    thin = (alpha[:, col] > 127)[int((-10 + info.z_offset) / 0.5) :].sum()
    wide = (alpha[:, col] > 127)[: int((-10 + info.z_offset) / 0.5)].sum()
    assert abs(thin - 8) <= 2 and abs(wide - 40) <= 2  # 4 m and 20 m at 0.5 m/px
    with pytest.raises(ValueError):
        tf.render_map([a, b], [4.0], tmp_path, tmp_path)


def test_outline_png_is_north_up(tmp_path):
    path = tf.write_outline_png(tmp_path / "ui" / "outline.png", [l_road()])
    img = Image.open(path)
    assert img.size == (365, 192) and img.mode == "RGBA"
    arr = np.asarray(img)
    white = (arr[..., 3] > 200) & (arr[..., :3].min(-1) > 200)
    assert arr[0, 0, 3] == 0 and white.any()
    rows, cols = np.nonzero(white)
    # The west leg is the top row of the drawing, and lies left of the north leg.
    top_cols = cols[rows <= rows.min() + 2]
    bottom_cols = cols[rows >= rows.max() - 2]
    assert top_cols.mean() < bottom_cols.mean()


def test_preview_png(tmp_path):
    path = tf.write_preview_png(tmp_path / "ui" / "preview.png", [l_road()], title="dm_test")
    img = Image.open(path)
    assert img.size == (355, 200)
    arr = np.asarray(img.convert("RGB"), int)
    green = (arr[..., 1] > 150) & (arr[..., 0] < 100)
    red = (arr[..., 0] > 180) & (arr[..., 1] < 100)
    assert green.any() and red.any()
    # A (start, south) is below B (end, after going north then west) and to its right.
    gy, gx = np.argwhere(green).mean(0)
    ry, rx = np.argwhere(red).mean(0)
    assert gy > ry and gx > rx


# ---------------------------------------------------------------------------------------------
# cameras.ini
# ---------------------------------------------------------------------------------------------


def test_cameras_ini_round_trip(tmp_path):
    road = l_road()
    cams = tf.cameras_along(road, spacing_m=150.0)
    path = tf.write_cameras_ini(tmp_path / "data" / "cameras.ini", cams, set_name="TV 1")
    cp = read_ini(path)
    assert cp["HEADER"]["VERSION"] == "3"
    assert int(cp["HEADER"]["CAMERA_COUNT"]) == len(cams) == 4
    assert cp["HEADER"]["SET_NAME"] == "TV 1"
    keys = list(cp["CAMERA_0"].keys())
    assert keys[:8] == [
        "NAME", "POSITION", "FORWARD", "UP", "MIN_FOV", "MAX_FOV", "IN_POINT", "OUT_POINT"
    ]  # fmt: skip
    assert keys[-1] == "IS_FIXED" and len(keys) == 24
    ins = [float(cp[f"CAMERA_{i}"]["IN_POINT"]) for i in range(len(cams))]
    outs = [float(cp[f"CAMERA_{i}"]["OUT_POINT"]) for i in range(len(cams))]
    assert ins[0] == 0.0 and outs[-1] == 1.0
    assert ins[1:] == outs[:-1]  # contiguous tiling of [0, 1]
    assert all(a < b for a, b in zip(ins, outs, strict=True))
    for i, cam in enumerate(cams):
        sec = cp[f"CAMERA_{i}"]
        np.testing.assert_allclose(vec(sec["POSITION"]), cam.position, atol=1e-4)
        f, u = vec(sec["FORWARD"]), vec(sec["UP"])
        assert abs(np.linalg.norm(f) - 1) < 1e-5 and abs(np.linalg.norm(u) - 1) < 1e-5
        assert abs(f @ u) < 1e-5 and u[1] > 0.9


def test_cameras_stand_off_the_road_and_look_ahead():
    road = l_road()
    half, offset = 4.0, 6.0
    cams = tf.cameras_along(road, 120.0, road_half_width_m=half, offset_m=offset, height_m=3.0)
    for cam in cams:
        pos = np.array(cam.position)
        d = np.hypot(road[:, 0] - pos[0], road[:, 2] - pos[2])
        nearest = int(d.argmin())
        assert d.min() > half + offset - 0.5  # off the road
        assert abs(pos[1] - road[nearest, 1] - 3.0) < 1e-6
        ahead = road[min(nearest + 10, len(road) - 1)] - road[max(nearest - 10, 0)]
        assert np.dot(cam.forward, ahead) > 0  # aimed down the road, the way cars drive
        assert cam.forward[1] < 0  # looking slightly down at the road


def test_camera_at_a_bend_is_on_the_outside():
    road = l_road()  # the bend's centre is (-30, -300); radius 30 m
    cams = tf.cameras_along(road, spacing_m=40.0)
    centre = np.array([-30.0, -300.0])
    bend = [c for c in cams if c.position[0] < -2 and c.position[2] < -302 and c.position[0] > -28]
    assert bend
    for cam in bend:
        assert np.hypot(cam.position[0] - centre[0], cam.position[2] - centre[1]) > 30.0 + 4.0


def test_cameras_with_a_longer_ai_spline():
    road = np.stack([np.zeros(101), np.zeros(101), -np.arange(101.0) * 10], 1)  # 1 km
    cams = tf.cameras_along(road, 250.0, spline_length_m=1500.0, spline_offset_m=100.0)
    assert len(cams) == 4
    assert cams[0].in_point == 0.0 and cams[-1].out_point == 1.0
    assert cams[1].in_point == pytest.approx((100 + 250) / 1500)
    assert all(cams[i].out_point == cams[i + 1].in_point for i in range(3))


def test_cameras_short_road_and_bad_input():
    cams = tf.cameras_along(np.array([[0.0, 0, 0], [0.0, 0, -50]]), 200.0)
    assert len(cams) == 1 and (cams[0].in_point, cams[0].out_point) == (0.0, 1.0)
    with pytest.raises(ValueError):
        tf.cameras_along(np.zeros((1, 3)))
    with pytest.raises(ValueError):
        tf.cameras_along(np.zeros((5, 3)))


# ---------------------------------------------------------------------------------------------
# lighting / groove / crew / models
# ---------------------------------------------------------------------------------------------


def test_small_data_files(tmp_path):
    cp = read_ini(tf.write_lighting_ini(tmp_path / "lighting.ini", 30.0, 120.5))
    assert dict(cp["LIGHTING"]) == {"SUN_PITCH_ANGLE": "30", "SUN_HEADING_ANGLE": "120.5"}
    cp = read_ini(tf.write_groove_ini(tmp_path / "groove.ini"))
    assert dict(cp["HEADER"]) == {"GROOVES_NUMBER": "0"}
    cp = read_ini(tf.write_crew_ini(tmp_path / "crew.ini"))
    assert dict(cp["HEADER"]) == {"SIDE": "-1"}
    assert read_ini(tf.write_crew_ini(tmp_path / "crew2.ini", 1))["HEADER"]["SIDE"] == "1"
    with pytest.raises(ValueError):
        tf.write_crew_ini(tmp_path / "crew3.ini", 0)


def test_models_ini(tmp_path):
    cp = read_ini(tf.write_models_ini(tmp_path / "models.ini", ["dm_x.kn5", "dm_x_terrain.kn5"]))
    assert cp.sections() == ["MODEL_0", "MODEL_1"]
    assert cp["MODEL_0"]["FILE"] == "dm_x.kn5" and cp["MODEL_1"]["FILE"] == "dm_x_terrain.kn5"
    assert vec(cp["MODEL_1"]["POSITION"]).tolist() == [0, 0, 0]
    assert vec(cp["MODEL_1"]["ROTATION"]).tolist() == [0, 0, 0]
    for bad in ([], ["x.fbx"], ["sub/x.kn5"]):
        with pytest.raises(ValueError):
            tf.write_models_ini(tmp_path / "bad.ini", bad)


# ---------------------------------------------------------------------------------------------
# LICENSE-DATA.txt
# ---------------------------------------------------------------------------------------------


def test_license_for_synthetic_track(tmp_path):
    path = tf.write_data_license(tmp_path, [], track_name="dm_t1_minimal")
    assert path == tmp_path / "LICENSE-DATA.txt"
    text = path.read_text("utf-8")
    assert tf.KUNOS_NOTICE in text
    assert "generated by drift-maps" in text
    assert "OpenStreetMap" not in text and "Copernicus" not in text


def test_license_with_sources(tmp_path):
    text = tf.write_data_license(tmp_path, ["osm", "copernicus", "osm"]).read_text("utf-8")
    assert "© OpenStreetMap contributors" in text and "Open Database License" in text
    assert (
        "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus "
        "Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and "
        "ESA; all rights reserved"
    ) in text
    assert "do not incur any liability for any use of the Copernicus WorldDEM-30" in text
    assert text.count("Open Database License") == 1  # duplicates dropped
    assert text.rstrip().endswith(tf.KUNOS_NOTICE)
    with pytest.raises(ValueError):
        tf.write_data_license(tmp_path, ["srtm"])


def test_attribution_text_is_ascii():
    text = tf.attribution_text(["osm", "copernicus", "usgs_3dep", "ign"])
    assert text.isascii() and "OpenStreetMap" in text and "Kunos" in text
