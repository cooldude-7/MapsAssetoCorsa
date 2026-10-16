"""Building heights from measured sources, categories, front doors, budget and LOD batches."""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely

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
    big = _one(_fp("burlington:3", _rect(-20, 30, 40, 25), "burlington", peak_m=12.0, eave_m=4.0))
    assert big.roof.shape != ROOF_FLAT and big.eave_h == pytest.approx(4.0)  # a big barn
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
                            options=BuildingOptions(budget_bytes=total // 3, frontage_m=0.0,
                                                    town_budget_bytes=total // 3))  # fmt: skip
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


def _subdivision(n_rows=2) -> list:
    """Rows of 13 detached houses (12 x 9 m, 15 m apart) at z = 20 and z = 90: 26 within
    100 m of the lot at z = 50-70 between them."""
    return [
        _fp(f"osm:h{r}{i:02d}", _rect(-96 + 15 * i, 20 + 70 * r, 12, 9), tags={})
        for r in range(n_rows)
        for i in range(13)
    ]


def _place(fps, **kw):
    return {b.id: b for b in place_buildings(fps, PROJ, _road(), _flat, seed="t", **kw)}


def _cover(cls):
    return lambda xz: np.full(len(xz), cls, np.uint8)


def test_barns_only_in_the_country():
    big = _fp("ms:q:big", _rect(-14, 50, 28, 20), "microsoft", tags={})  # 560 m2, untyped
    alone = _place([big])["ms:q:big"]
    assert alone.category == "barn" and not alone.town
    town = _place([big, *_subdivision()])
    assert town["ms:q:big"].category != "barn" and town["ms:q:big"].town
    assert town["ms:q:big"].wall_style not in ("barn_red", "barn_grey", "steel_white")
    # An OSM-tagged barn stays a barn wherever it stands.
    tagged = _fp("osm:barn", _rect(-15, 50, 30, 20), tags={"building": "barn"})
    assert _place([tagged, *_subdivision()])["osm:barn"].category == "barn"
    # A hamlet (12 houses within 100 m): in the country among fields and woods, else a town.
    hamlet = _subdivision(1)[:12]
    assert _place([big, *hamlet], landcover=_cover(15))["ms:q:big"].category == "barn"
    assert _place([big, *hamlet], landcover=_cover(6))["ms:q:big"].category == "barn"
    assert _place([big, *hamlet], landcover=_cover(17))["ms:q:big"].category != "barn"
    assert _place([big, *hamlet])["ms:q:big"].category != "barn"  # no land cover: a town
    # A farm: a house or two among barns and sheds (which do not count), even where NRCan maps
    # the farmstead as urban.
    farm = [_fp(f"ms:q:b{i}", _rect(-60 + 30 * i, 100, 25, 20), "microsoft", tags={})
            for i in range(5)] + [_fp(f"ms:q:s{i}", _rect(-60 + 8 * i, 20, 4, 4), "microsoft",
                                      tags={}) for i in range(12)]  # fmt: skip
    got = _place([big, *farm, *_subdivision(1)[:3]], landcover=_cover(15))
    assert got["ms:q:big"].category == "barn" and got["ms:q:b0"].category == "barn"
    # A district of big buildings on all-urban land cover (shops by the lake) is a town.
    got = _place([big, *farm, *_subdivision(1)[:3]], landcover=_cover(17))
    assert got["ms:q:big"].category != "barn" and got["ms:q:big"].town


def test_town_houses_are_brick_rows_with_a_door_per_unit():
    from driftmaps.geometry.buildings import _TOWN_HOUSE_WALLS, _TOWNHOUSE_WALLS

    row = _fp("ms:q:row", _rect(-16, 50, 32, 10), "microsoft", tags={})  # 5 units of 6.4 m
    got = _place([row, *_subdivision()])
    b = got["ms:q:row"]
    assert b.category == "townhouse" and b.roof.shape != ROOF_FLAT
    assert b.wall_style in dict(_TOWNHOUSE_WALLS) and b.wall_style.startswith(("brick", "stone"))
    fronts = [d for d in b.doors if d.style.startswith("door_")]
    assert len(fronts) == 5 and all(d.edge == b.front_edge for d in b.doors)
    houses = [h for h in got.values() if h.category == "house"]
    assert len(houses) == 26 and all(h.wall_style in dict(_TOWN_HOUSE_WALLS) for h in houses)
    assert all(h.roof_style.startswith("shingle") for h in houses)
    styles = {h.wall_style for h in houses}
    assert not styles & {"board_white", "barn_red", "barn_grey", "steel_white", "shed_wood"}
    # The same footprint in the country is one long house, not a row of units.
    assert _place([row])["ms:q:row"].category == "house"


def test_garages_wear_their_house_colours():
    house = _fp("osm:w10", _rect(-6, 30, 12, 9))
    garages = [
        _fp(
            f"osm:g{i}",
            _rect(10 + 9 * (i % 2), 32 + 9 * (i // 2), 6, 7),
            tags={"building": "garage"},
        )
        for i in range(6)
    ]
    got = _place([house, *garages])
    assert all(got[g.id].wall_style == got["osm:w10"].wall_style for g in garages)


def test_street_front_buildings_are_always_placed():
    # A downtown block: shops 3 m from the road's edge (inside the 4 m wall corridor), and
    # 200 more buildings behind them; a budget too small for even the street front.
    road = _road(1200.0)
    front = [_fp(f"osm:f{i:02d}", _rect(-300 + 15 * i, 7, 12, 12), tags={"building": "retail"})
             for i in range(40)]  # fmt: skip
    back = [_fp(f"osm:b{i:03d}", _rect(-300 + 12 * (i % 50), 80 + 25 * (i // 50), 8, 8))
            for i in range(200)]  # fmt: skip
    corridor = shapely.box(-600, -8, 600, 8)  # road (4 m) + 4 m verge each side
    rep: dict = {}
    opts = BuildingOptions(budget_bytes=1000, town_budget_bytes=1000)
    got = place_buildings(front + back, PROJ, road, _flat, seed="t", corridor=corridor,
                          options=opts, report=rep)  # fmt: skip
    ids = {b.id for b in got}
    assert all(f.id in ids for f in front)  # in the wall corridor and over budget: placed
    assert not any(b.id in ids for b in back)  # beyond 60 m, over budget: dropped
    assert rep["street_front"] == 40 and rep["placed"] == 40 and rep["budget"] == 200
    assert rep["town"] == 240 and rep["corridor"] == 0
    # In the country the wall corridor still keeps buildings away from the road.
    lone = _fp("osm:lone", _rect(0, 7, 12, 12), tags={"building": "house"})
    rep = {}
    assert place_buildings([lone], PROJ, road, _flat, corridor=corridor, report=rep) == []
    assert rep["corridor"] == 1
    # A footprint over a side road (the network's paved surface) is dropped in town too.
    paved = shapely.box(-306, 0, -294, 300)  # a side road through the first shop
    got = place_buildings(front + back, PROJ, road, _flat, seed="t", corridor=corridor,
                          options=opts, paved=paved)  # fmt: skip
    assert "osm:f00" not in {b.id for b in got} and "osm:f01" in {b.id for b in got}


def test_commercial_roofs_stay_flat_and_gables_keep_brick_proportions():

    retail = {"building": "retail"}
    tower = _one(_fp("burlington:9", _rect(-10, 30, 20, 15), "burlington", tags=retail,
                     peak_m=34.0, eave_m=30.0))  # fmt: skip
    assert tower.roof.shape == ROOF_FLAT and tower.eave_h == pytest.approx(30.0)  # clamp 30
    shop = _one(_fp("burlington:8", _rect(-10, 30, 20, 15), "burlington", tags=retail,
                    peak_m=5.0, eave_m=4.0))  # fmt: skip
    assert shop.roof.shape == ROOF_FLAT and shop.eave_h == pytest.approx(4.6)  # + parapet
    plaza = _one(_fp("burlington:7", _rect(-10, 30, 20, 15), "burlington", tags=retail,
                     peak_m=7.5, eave_m=4.0))  # fmt: skip
    assert plaza.roof.shape != ROOF_FLAT  # a low building with a real pitched roof
    # In town a wide untyped building whose peak is 11 m above its eave (a garage's stair
    # tower) is flat, not a barn-like 11 m gable; in the country it keeps its big roof.
    big = _fp("burlington:11", _rect(-20, 40, 40, 20), "burlington", tags={}, peak_m=18.0,
              eave_m=7.0)  # fmt: skip
    got = _place([big, *_subdivision()])["burlington:11"]
    assert got.roof.shape == ROOF_FLAT and got.eave_h == pytest.approx(7.6)
    assert _place([big])["burlington:11"].roof.shape != ROOF_FLAT
    # A 3 m gable end of a brick house is mapped 1:1 on the band's window-free end (u in the
    # last 12 m), not on the 0.75 m strip stretched 4x into wood grain.
    house = _one(_fp("burlington:10", _rect(-7, 30, 14, 8), "burlington", peak_m=8.0, eave_m=5.0))
    house.wall_style = "brick_grey"
    m = extrude_building(house, doors=False)
    v0, v1 = band_v("brick_grey")
    gable = (m.positions[:, 1] > house.eave_y + 0.5) & (np.abs(m.normals[:, 1]) < 1e-6)  # walls
    assert gable.any()
    ys, uv = m.positions[gable, 1], m.uvs[gable]
    assert np.all((uv[:, 0] * ATLAS_U_M) % ATLAS_U_M >= WINDOW_ZONE_M - 1e-6)
    v_per_m = (v1 - v0) / 3.0  # one storey of band per 3 m
    top = np.argmax(ys)
    assert uv[top, 1] == pytest.approx(v1 - (ys[top] - house.eave_y) * v_per_m, abs=1e-4)
