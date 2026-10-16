"""Building heights from measured sources, categories, front doors, budget and LOD batches."""

from __future__ import annotations

import math

import numpy as np
import pytest

from driftmaps.ac.building_textures import (
    ATLAS_U_M,
    DOORS,
    WINDOW_PERIOD_M,
    WINDOW_ZONE_M,
    band_v,
    door_uv,
    wall_u_start,
)
from driftmaps.geo.buildings import Footprint
from driftmaps.geo.project import LocalProjection
from driftmaps.geometry.buildings import (
    ROOF_FLAT,
    BuildingOptions,
    building_meshes,
    extrude_building,
    mesh_bytes,
    ontario_pitch,
    place_buildings,
)
from driftmaps.geometry.primitives import face_normals
from driftmaps.geometry.road import RoadModel

PROJ = LocalProjection(43.43, -79.85)


def _fp(fid, xz, source="osm", tags=None, **kw) -> Footprint:
    r = np.asarray(xz, np.float64)
    r = np.vstack([r, r[:1]])
    lon, lat = PROJ.to_lonlat(r[:, 0], -r[:, 1])
    tags = {"building": "house"} if tags is None else tags
    return Footprint(fid, source, np.stack([lon, lat], 1), tags, **kw)


def _rect(x0, z0, w, d):
    return [(x0, z0), (x0 + w, z0), (x0 + w, z0 + d), (x0, z0 + d)]


def _road(length=800.0) -> RoadModel:
    x = np.arange(-length / 2, length / 2 + 1e-9, 2.0)
    return RoadModel(x - x[0], np.stack([x, 0 * x, 0 * x], 1), 3.0, 1.0, True, False)


def _flat(xz):
    return np.full(len(xz), 100.0)


def _one(fp, **kw):
    out = place_buildings([fp], PROJ, _road(), _flat, seed="t", **kw)
    assert len(out) == 1
    return out[0]


def test_measured_heights_win_and_microsoft_is_mid_roof():
    house = _rect(-6, 30, 12, 9)
    b = _one(_fp("burlington:1", house, "burlington", peak_m=6.6, eave_m=3.1))
    assert b.height_source == "burlington"
    assert b.ridge_h == pytest.approx(6.6) and b.eave_h == pytest.approx(3.1)
    flat = _one(_fp("burlington:2", house, "burlington", peak_m=4.0, eave_m=3.8))
    assert flat.roof.shape == ROOF_FLAT and flat.eave_h == pytest.approx(4.0)
    nr = _one(_fp("nrcan:1", house, "nrcan", peak_m=7.0, lidar=True))
    assert nr.height_source == "nrcan" and nr.ridge_h == pytest.approx(7.0)
    assert 0.30 * 7.0 - 1e-9 <= nr.roof.rise <= 0.45 * 7.0 + 1e-9  # Burlington-fitted share
    ms = _one(_fp("ms:q:1", house, "microsoft", tags={}, ms_height_m=5.0))
    assert ms.height_source == "microsoft"
    assert 0.5 * (ms.eave_h + ms.ridge_h) == pytest.approx(5.0)  # the ML height is mid-roof


def test_categories_and_typical_heights():
    shed = _one(_fp("nrcan:2", _rect(0, 30, 4, 4), "nrcan", tags={}))
    garage = _one(_fp("nrcan:3", _rect(0, 30, 6, 8), "nrcan", tags={}))
    barn = _one(_fp("nrcan:4", _rect(0, 30, 20, 30), "nrcan", tags={"building": "barn"}))
    assert (shed.category, garage.category, barn.category) == ("shed", "garage", "barn")
    assert shed.ridge_h == pytest.approx(3.2) and barn.ridge_h == pytest.approx(8.5)
    assert barn.roof_style.startswith(("metal", "shingle")) and barn.roof.shape != ROOF_FLAT
    assert ontario_pitch(0.5) == pytest.approx(22.5) and ontario_pitch(0.1) == pytest.approx(14.8)
    assert ontario_pitch(0.9) == pytest.approx(33.7)
    assert ontario_pitch(0.5, "barn") > ontario_pitch(0.5) > ontario_pitch(0.5, "shed")


def test_front_door_faces_the_road_and_doors_fit_the_wall():
    # Road along X at z = 0; the house sits at z = 30..40, its road side is z = 30.
    fp = _fp("osm:w1", _rect(-8, 30, 16, 10))
    b = _one(fp)
    n = b.front_normal
    assert n is not None and n @ np.array([0.0, -1.0]) > 0.99
    assert b.doors and all(d.edge == b.front_edge for d in b.doors)
    for d in b.doors:
        assert d.centre[1] == pytest.approx(30.0)  # on the front wall
        assert b.ground_y < d.y0 + 0.5 and d.y0 + d.height <= b.eave_y
    # The door quads face the road, 3 cm proud of the wall, inside their atlas rectangles.
    m = extrude_building(b)
    fn = face_normals(m)
    proud = m.positions[m.triangles][..., 2].max(axis=1) < 30.0 - 0.02
    assert proud.any() and np.allclose(fn[proud][:, [0, 2]], [0.0, -1.0], atol=1e-9)
    v0, v1 = band_v("doors")
    uv = m.uvs[m.triangles[proud]]
    assert uv[..., 1].min() >= v0 - 1e-9 and uv[..., 1].max() <= v1 + 1e-9
    garage = _one(_fp("osm:w2", _rect(-4, -40, 8, 7), tags={"building": "garage"}))
    assert [d.style.startswith("garage") for d in garage.doors] == [True]
    assert garage.front_normal @ np.array([0.0, 1.0]) > 0.99  # the road is north (z < 0)


def test_window_layout_is_centred_and_garages_are_blank():
    for length in (4.0, 9.0, 12.5, 20.0):
        a = wall_u_start(length, 0.37)
        centres = WINDOW_PERIOD_M * (np.arange(9) + 0.5)
        inside = centres[(centres > a) & (centres < a + length)]
        assert len(inside) == min(9, int(length // WINDOW_PERIOD_M))
        assert inside.mean() - a == pytest.approx(length / 2)  # symmetric on the wall
    assert wall_u_start(6.0, 0.5, windows=False) >= WINDOW_ZONE_M
    assert wall_u_start(1.5, 0.5) + 1.5 <= ATLAS_U_M  # too short for a window: blank
    u0, u1, vt, vb = door_uv("garage2_white")
    assert u1 - u0 == pytest.approx(DOORS["garage2_white"].width_m / ATLAS_U_M) and vt < vb


def test_budget_lod_batches_and_distance():
    road = _road(1200.0)
    fps = [_fp(f"osm:w{i:03d}", _rect(-500 + 9 * (i % 100), 20 + 30 * (i // 100), 7, 6))
           for i in range(300)]  # fmt: skip
    fps += [_fp("osm:s1", _rect(0, -20, 4, 4), tags={"building": "shed"})]
    fps += [_fp("osm:far", _rect(0, 320, 10, 10))]
    allb = place_buildings(fps, PROJ, road, _flat, seed="t")
    assert "osm:far" not in {b.id for b in allb}  # beyond 300 m
    total = sum(mesh_bytes(b.mesh) for b in allb)
    small = place_buildings(fps, PROJ, road, _flat, seed="t",
                            options=BuildingOptions(budget_bytes=total // 3))  # fmt: skip
    assert 0 < len(small) < len(allb)
    near = max(b.road_distance_m for b in small)
    kept = {b.id for b in small}
    assert all(b.road_distance_m >= near - 1e-9 for b in allb if b.id not in kept)
    meshes = building_meshes(allb)
    names = {m.name for m in meshes}
    assert any(n.startswith("buildings_s") for n in names)  # the shed's short-range batch
    lods = {m.flags.get("lod_out") for m in meshes if not m.collision}
    assert lods == {BuildingOptions().lod_out_m, BuildingOptions().detail_lod_out_m}
    assert math.isclose(sum(m.building_count for m in meshes if not m.collision), len(allb))
