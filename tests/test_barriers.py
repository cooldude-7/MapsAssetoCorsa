"""Roadside barriers: OSM fetch/matching (geo/barriers.py) and meshes (geometry/barriers.py).

Offline except `test_live_fetch_finds_the_mill_street_rails` (marker `network`; skipped unless
run with `-m network` or DRIFTMAPS_NETWORK=1).
"""

from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from driftmaps.ac.model import MAX_MESH_VERTICES
from driftmaps.geo.barriers import (
    BarrierSpan,
    OsmBarrier,
    build_barrier_query,
    fetch_barriers,
    load_user_spans,
    match_barriers,
    merge_spans,
    parse_barriers,
    parse_user_spans,
)
from driftmaps.geo.overpass import OverpassError, Way, parse_overpass, query_cache_key
from driftmaps.geo.project import LocalProjection
from driftmaps.geometry.barriers import (
    TEXTURES,
    barrier_path,
    barrier_pieces,
    concrete_texture,
    drop_off_spans,
    plan_barriers,
    span_meshes,
    steel_texture,
    sweep,
)
from driftmaps.geometry.primitives import face_normals, to_kn5_mesh
from driftmaps.geometry.road import RoadModel

FIXTURE = Path(__file__).parent / "fixtures" / "osm" / "mill_st_waterdown_barriers.json"
# Mill Street South, Waterdown (Hamilton): ways in driving order (north-west -> south-east).
MILL_ST_WAYS = (34632407, 24486613, 24486612)
MILL_ST_BRIDGE = 24486613
MILL_ST_RAILS = (427465490, 427465491, 427465492)


def model_from(xz: np.ndarray, y: np.ndarray | None = None, bridge=False) -> RoadModel:
    y = np.zeros(len(xz)) if y is None else y
    centre = np.stack([xz[:, 0], y, xz[:, 1]], axis=1)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))])
    return RoadModel(
        s=s, centre=centre, half_width=3.5, shoulder=1.0, centre_line=True, bridge=bridge
    )


def straight(length: float = 300.0) -> RoadModel:
    """A flat road heading east (+X): the driver's left is north (-Z)."""
    x = np.arange(0.0, length + 0.5, 1.0)
    return model_from(np.stack([x, np.zeros(len(x))], axis=1))


def densify(xz: np.ndarray, step: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    parts = [xz[:1]]
    for a, b in zip(xz[:-1], xz[1:], strict=True):
        n = max(1, math.ceil(np.linalg.norm(b - a) / step))
        parts.append(a + np.linspace(0.0, 1.0, n + 1)[1:, None] * (b - a))
    pts = np.concatenate(parts)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return pts, s


def mill_street():
    osm = parse_overpass(FIXTURE.read_text(encoding="utf-8"))
    ways = {w.id: w for w in osm.ways}
    ll = np.concatenate([ways[i].lonlat[(1 if k else 0) :] for k, i in enumerate(MILL_ST_WAYS)])
    proj = LocalProjection(float(ll[0, 1]), float(ll[0, 0]))
    e, n = proj.to_enu(ll[:, 0], ll[:, 1])
    centre, s = densify(np.stack([e, -n], axis=1))
    return osm, ways, proj, centre, s


def station_of(proj, centre, s, lonlat) -> np.ndarray:
    e, n = proj.to_enu(lonlat[:, 0], lonlat[:, 1])
    pts = np.stack([e, -n], axis=1)
    d = np.linalg.norm(pts[:, None, :] - centre[None, :, :], axis=2)
    return s[np.argmin(d, axis=1)]


# --------------------------------------------------------------------------------------------
# OSM: query, cache, parsing, matching
# --------------------------------------------------------------------------------------------


def test_query_lists_the_barrier_kinds_and_radius():
    q = build_barrier_query([(43.33, -79.88), (43.331, -79.885)], around_m=15)
    assert '"barrier"~"^(guard_rail|jersey_barrier|retaining_wall|wall)$"' in q
    assert "(around:15,43.33,-79.88,43.331,-79.885)" in q
    assert q.rstrip().endswith("out body geom;")


def test_fetch_reads_the_cache_offline_and_never_hits_the_network(tmp_path):
    route = np.array([[-79.8890, 43.3315], [-79.8860, 43.3302], [-79.8840, 43.3290]])
    with pytest.raises(OverpassError):
        fetch_barriers(route, cache_dir=tmp_path, offline=True)
    # Put the fixture where fetch_barriers looks for this exact query.
    from driftmaps.geo.barriers import _simplified_latlon

    q = build_barrier_query(_simplified_latlon(route, 5.0), 15.0 + 5.0)
    path = tmp_path / "overpass" / f"{query_cache_key(q)}.json.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(gzip.compress(FIXTURE.read_bytes()))
    got = fetch_barriers(route, cache_dir=tmp_path, offline=True)
    assert [b.id for b in got] == list(MILL_ST_RAILS)


def test_parse_keeps_only_barrier_ways():
    osm, *_ = mill_street()
    bars = parse_barriers(osm)
    assert [(b.id, b.kind) for b in bars] == [(i, "guard_rail") for i in MILL_ST_RAILS]
    assert all(b.height_m == pytest.approx(1.0) for b in bars)  # mapped height=1
    assert parse_barriers(osm, kinds=("wall",)) == []


def test_mill_street_rails_match_the_right_side_along_their_extent():
    osm, ways, proj, centre, s = mill_street()
    spans = match_barriers(parse_barriers(osm), proj, centre, s)
    assert [sp.osm_id for sp in spans] == list(MILL_ST_RAILS)
    assert all(sp.side == "right" and sp.kind == "guard_rail" for sp in spans)
    for sp in spans:
        st = station_of(proj, centre, s, ways[sp.osm_id].lonlat)
        assert sp.s0 == pytest.approx(st.min(), abs=3.0)
        assert sp.s1 == pytest.approx(st.max(), abs=3.0)
        assert 4.0 < sp.offset_m < 8.0  # a few metres off the OSM centreline, as mapped
    # The first rail runs over the bridge and on past its east end.
    bridge = station_of(proj, centre, s, ways[MILL_ST_BRIDGE].lonlat)
    first = spans[0]
    assert first.s0 < bridge.max() < first.s1
    # Driving the other way, every rail is on the left at mirrored stations.
    rev = match_barriers(parse_barriers(osm), proj, centre[::-1], s[-1] - s[::-1])
    assert {sp.side for sp in rev} == {"left"}
    for a in spans:
        b = next(r for r in rev if r.osm_id == a.osm_id)
        assert b.s0 == pytest.approx(s[-1] - a.s1, abs=1.5)
        assert b.s1 == pytest.approx(s[-1] - a.s0, abs=1.5)


def _barrier(proj, xz, kind="guard_rail", id_=1, tags=None) -> OsmBarrier:
    xz = np.asarray(xz, float)
    lon, lat = proj.to_lonlat(xz[:, 0], -xz[:, 1])
    return OsmBarrier(id_, kind, np.stack([lon, lat], axis=1), tags or {})


def test_matching_rejects_crossings_far_ways_and_the_carriageway():
    proj = LocalProjection(43.3, -79.9)
    x = np.arange(0.0, 201.0)
    centre = np.stack([x, np.zeros(len(x))], axis=1)  # heading east; left = north = -Z
    bars = [
        _barrier(proj, [[20, -5.0], [60, -5.0]], id_=1),  # left, 40 m
        _barrier(proj, [[80, 6.0], [120, 6.0]], kind="jersey_barrier", id_=2),  # right
        _barrier(proj, [[100, -14.0], [100, -40.0]], kind="wall", id_=3),  # at right angles
        _barrier(proj, [[130, -20.0], [170, -20.0]], id_=4),  # 20 m away
        _barrier(proj, [[140, 0.3], [180, 0.3]], id_=5),  # on the centreline
        _barrier(proj, [[190, 5.0], [240, 5.0]], id_=6),  # runs past the end of the road
        _barrier(proj, [[30, 8.0], [32, 8.0]], id_=7),  # 2 m stub
    ]
    spans = match_barriers(bars, proj, centre, x)
    got = {sp.osm_id: sp for sp in spans}
    assert set(got) == {1, 2, 6}
    assert got[1].side == "left" and got[1].s0 == pytest.approx(20, abs=0.1)
    assert got[1].s1 == pytest.approx(60, abs=0.1) and got[1].offset_m == pytest.approx(5.0)
    assert got[2].side == "right" and got[2].kind == "jersey_barrier"
    assert got[6].s0 == pytest.approx(190, abs=0.1) and got[6].s1 == pytest.approx(200, abs=0.1)


def test_matching_follows_one_leg_of_a_hairpin():
    proj = LocalProjection(43.3, -79.9)
    r = 12.0
    up = np.stack([np.zeros(100), -np.arange(100.0)], axis=1)
    a = np.linspace(math.pi, 0.0, 40)
    arc = np.stack([r + r * np.cos(a), -100 - r * np.sin(a)], axis=1)
    down = np.stack([np.full(100, 2 * r), -np.arange(99.0, -1.0, -1.0)], axis=1)
    centre, s = densify(np.concatenate([up, arc[1:-1], down]))
    # A rail on the outside of the down leg (east of it): the driver's left going south.
    bar = _barrier(proj, [[2 * r + 5, -90.0], [2 * r + 5, -20.0]])
    (span,) = match_barriers([bar], proj, centre, s)
    assert span.side == "left"
    assert span.s0 > 100 + math.pi * r - 1 and span.s1 - span.s0 == pytest.approx(70, abs=1)


def test_merge_spans_joins_pieces_of_one_rail():
    a = BarrierSpan("left", 0.0, 10.0, osm_id=1)
    b = BarrierSpan("left", 10.5, 30.0, osm_id=2)
    c = BarrierSpan("right", 5.0, 8.0)
    merged = merge_spans([b, c, a])
    assert [(m.side, m.s0, m.s1, m.osm_id) for m in merged] == [
        ("left", 0.0, 30.0, 1),
        ("right", 5.0, 8.0, None),
    ]


def test_user_spans_are_route_distances_from_a(tmp_path):
    data = [
        {"side": "left", "from_m": 100, "to_m": 150},
        {"side": "right", "from_m": 90, "to_m": 40, "kind": "jersey_barrier"},
    ]
    spans = parse_user_spans({"guardrails": data}, s_origin=60.0)
    assert [(sp.side, sp.s0, sp.s1, sp.kind, sp.source) for sp in spans] == [
        ("left", 160.0, 210.0, "guard_rail", "user"),
        ("right", 100.0, 150.0, "jersey_barrier", "user"),
    ]
    assert load_user_spans(tmp_path / "guardrails.json") == []
    (tmp_path / "guardrails.json").write_text(json.dumps(data), encoding="utf-8")
    assert len(load_user_spans(tmp_path / "guardrails.json", s_origin=60.0)) == 2
    # Windows editors save UTF-8 with a BOM.
    bom = tmp_path / "bom" / "guardrails.json"
    bom.parent.mkdir()
    bom.write_bytes(b"\xef\xbb\xbf" + json.dumps(data).encode("utf-8"))
    assert load_user_spans(bom, s_origin=60.0) == spans
    for bad in (
        [{"side": "middle", "from_m": 0, "to_m": 5}],
        [{"side": "left", "from_m": 3, "to_m": 3}],
        [{"side": "left", "from_m": "x", "to_m": 3}],
        [{"side": "left", "from_m": 0, "to_m": 3, "kind": "hedge"}],
        {"side": "left"},
    ):
        with pytest.raises(ValueError):
            parse_user_spans(bad)


def test_user_spans_are_mirrored_for_the_other_direction(tmp_path):
    """Spans are measured as driven along route.json; a --reverse build (or a file that says
    "reversed") swaps the sides and measures from the other end."""
    data = [{"side": "left", "from_m": 100, "to_m": 150}]
    same = parse_user_spans(data, s_origin=60.0, reverse=False, length_m=1000.0)
    assert [(sp.side, sp.s0, sp.s1) for sp in same] == [("left", 160.0, 210.0)]
    rev = parse_user_spans(data, s_origin=60.0, reverse=True, length_m=1000.0)
    assert [(sp.side, sp.s0, sp.s1) for sp in rev] == [("right", 910.0, 960.0)]
    flagged = {"reversed": True, "guardrails": data}
    assert parse_user_spans(flagged, s_origin=60.0, reverse=True, length_m=1000.0) == same
    assert parse_user_spans(flagged, s_origin=60.0, length_m=1000.0) == rev
    path = tmp_path / "guardrails.json"
    path.write_text(json.dumps(flagged), encoding="utf-8")
    assert load_user_spans(path, s_origin=60.0, reverse=True, length_m=1000.0) == same
    with pytest.raises(ValueError):
        parse_user_spans(data, reverse=True)  # mirroring needs the length
    with pytest.raises(ValueError):
        parse_user_spans({"reversed": "yes", "guardrails": data})


# --------------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------------


def cliff(xz: np.ndarray) -> np.ndarray:
    """Ground falling 10 m away just north of the road (the driver's left heading east)."""
    return np.where(xz[:, 1] < -5.0, -10.0, 0.0)


def test_nothing_is_invented_by_default():
    m = straight()
    assert plan_barriers(m) == []
    assert plan_barriers(m, bridges=[(100.0, 140.0)]) == []  # no approach rails either
    # The opt-in heuristic does see the drop on the left, but only when asked for.
    found = drop_off_spans(m, cliff)
    assert [(sp.side, sp.source) for sp in found] == [("left", "heuristic")]
    assert found[0].s1 - found[0].s0 == pytest.approx(300, abs=1)
    planned = plan_barriers(m, heuristic=found)
    assert [sp.source for sp in planned] == ["heuristic"]


def test_plan_priorities_bridges_and_approaches():
    m = straight(400)
    bridges = [(200.0, 240.0)]
    osm = [BarrierSpan("right", 100.0, 220.0, source="osm", osm_id=9)]
    user = [BarrierSpan("right", 150.0, 170.0, "jersey_barrier", "user")]
    plan = plan_barriers(m, osm=osm, user=user, bridges=bridges, approach_m=20.0)
    got = [(sp.side, sp.s0, sp.s1, sp.kind, sp.source) for sp in plan]
    assert got == [
        ("left", 180.0, 200.0, "guard_rail", "bridge"),
        ("left", 240.0, 260.0, "guard_rail", "bridge"),
        ("right", 100.0, 150.0, "guard_rail", "osm"),  # user span wins 150-170
        ("right", 150.0, 170.0, "jersey_barrier", "user"),
        ("right", 170.0, 200.0, "guard_rail", "osm"),  # clipped at the deck
        ("right", 240.0, 260.0, "guard_rail", "bridge"),
    ]


# --------------------------------------------------------------------------------------------
# Meshes
# --------------------------------------------------------------------------------------------


def _consistent_winding(mesh) -> bool:
    """Every triangle's geometric normal agrees with its vertices' stored normals."""
    fn = face_normals(mesh)
    vn = mesh.normals[mesh.triangles].mean(axis=1)
    ok = np.linalg.norm(fn, axis=1) > 0
    return bool(np.all(np.einsum("ij,ij->i", fn[ok], vn[ok]) > 0.0))


def test_sweep_of_a_ccw_profile_faces_outward():
    path = barrier_path(straight(), "left", 10.0, 20.0)
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])  # CCW in (d, h)
    for side in ("left", "right"):
        p = barrier_path(straight(), side, 10.0, 20.0)
        mesh = sweep(p.columns(square), p.s, closed=True)
        centre = p.at(0.5, 0.5)
        tri_c = mesh.positions[mesh.triangles].mean(axis=1)
        nearest = np.clip(np.round(tri_c[:, 0] - 10.0).astype(int), 0, len(p) - 1)
        away = tri_c - centre[nearest]
        away[:, 0] = 0.0
        assert np.all(np.einsum("ij,ij->i", face_normals(mesh), away) > 0)
        assert _consistent_winding(mesh)
        # UVs are not mirrored seen from the front: negative signed area (v grows down).
        uv = mesh.uvs[mesh.triangles]
        e1, e2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
        assert np.all(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] < 1e-12)
    assert path.side == "left" and path.out[0] @ np.array([0.0, 0.0, -1.0]) == pytest.approx(1)


@pytest.mark.parametrize("side", ["left", "right"])
def test_w_beam_faces_the_road_and_stands_at_the_shoulder_edge(side):
    m = straight()
    span = BarrierSpan(side, 50.0, 90.0)
    ((visual, coll),) = span_meshes(m, span, None, free_ends=(False, False))
    steel = visual["steel"]
    assert _consistent_winding(steel)
    sign = -1.0 if side == "left" else 1.0  # Z of the left side is negative (north)
    road_dir = np.array([0.0, 0.0, -sign])  # from the barrier towards the road
    fn = face_normals(steel)
    c = steel.positions[steel.triangles].mean(axis=1)
    # The rail's peaks (closest to the road, at rail height) face the road.
    peak = (np.abs(np.abs(c[:, 2]) - 4.5) < 0.001) & (c[:, 1] > 0.4) & (c[:, 1] < 0.8)
    assert peak.sum() > 20
    assert np.all(fn[peak] @ road_dir > 0.99)
    # Nothing of the barrier is on the road side of the shoulder edge (4.5 m).
    assert np.abs(steel.positions[:, 2]).min() >= 4.5 - 1e-6
    assert steel.positions[:, 1].max() == pytest.approx(0.75, abs=1e-6)
    # Back faces exist (seen from the field side).
    assert np.any(fn @ road_dir < -0.99)


@pytest.mark.parametrize(
    ("kind", "inside"),  # a point (d, h) inside each profile
    [("jersey_barrier", (0.3, 0.3)), ("wall", (0.15, 0.5)), ("parapet", (0.175, 0.4))],
)
def test_solid_barriers_have_outward_faces_and_end_caps(kind, inside):
    m = straight()
    span = BarrierSpan("right", 40.0, 60.0, kind)
    ((visual, _),) = span_meshes(m, span, None)
    mesh = visual["concrete"]
    assert _consistent_winding(mesh)
    fn = face_normals(mesh)
    c = mesh.positions[mesh.triangles].mean(axis=1)
    centre = np.array([50.0, inside[1], 4.5 + inside[0]])  # right side: +Z is away
    away = c - centre
    away[:, 0] = np.where(np.abs(fn[:, 0]) > 0.99, away[:, 0], 0.0)  # caps look along X
    assert np.all(np.einsum("ij,ij->i", fn, away) > 0)
    assert np.any(fn[:, 0] > 0.99) and np.any(fn[:, 0] < -0.99)  # both end caps


def test_collision_strip_present_continuous_and_hidden():
    t = np.linspace(0.0, math.pi, 700)
    m = model_from(np.stack([150 * np.cos(t), 150 * np.sin(t)], axis=1))  # ~470 m arc
    span = BarrierSpan("left", 20.0, m.s[-1] - 20.0)
    pieces = barrier_pieces(m, [span], lambda xz: np.full(len(xz), -0.4))
    coll = [p for p in pieces if p.name.startswith("1WALL_GR_")]
    vis = [p for p in pieces if p.name.startswith("guardrail_")]
    assert len(coll) == len(vis) == 3  # ~431 m in chunks of at most 150 m
    for p in coll:
        assert p.physical and p.flags == {"is_renderable": False, "cast_shadows": False}
        assert p.material == "wall_hidden"
    for p in vis:
        assert not p.physical and p.flags["lod_out"] > 0
    # The bottoms of consecutive strips meet exactly, covering the whole span.
    lines = []
    for p in coll:
        pos = p.mesh.positions
        bottom = pos[np.isclose(pos[:, 1], pos[:, 1].min())]
        lines.append(bottom[:, [0, 2]])
    ang = [np.sort(np.arctan2(seg[:, 1], seg[:, 0])) for seg in lines]
    for a, b in zip(ang[:-1], ang[1:], strict=True):
        assert b[0] == pytest.approx(a[-1], abs=1e-9)
    r_face = 150 + 4.5  # heading south round the origin, the left is the outside
    assert np.allclose(np.linalg.norm(np.concatenate(lines), axis=1), r_face, atol=0.05)
    covered = (ang[-1][-1] - ang[0][0]) * r_face
    assert covered == pytest.approx((span.s1 - span.s0) * r_face / 150, abs=0.5)
    # Every strip reaches below the ground and well above the road, and is double-sided.
    for p in coll:
        assert p.mesh.positions[:, 1].min() <= -0.4 - 0.5 + 1e-6
        fn = face_normals(p.mesh)
        assert np.isclose(fn.sum(axis=0), 0.0, atol=1e-6).all()
    tops = np.concatenate([p.mesh.positions[:, 1] for p in coll])
    assert tops.max() == pytest.approx(1.2)


def test_posts_reach_the_ground_and_free_ends_turn_down():
    m = straight()

    def ground(xz):  # fill slope falling 0.6 m per metre beyond the shoulder edge
        return -0.6 * np.maximum(np.abs(xz[:, 1]) - 4.5, 0.0)

    span = BarrierSpan("right", 100.0, 140.0)
    ((visual, _),) = span_meshes(m, span, ground, free_ends=(True, False))
    pos = visual["steel"].positions
    posts = pos[np.abs(pos[:, 2] - (4.5 + 0.083 + 0.15 + 0.075)) < 0.076]
    for x in np.linspace(100.0, 140.0, 21)[3:]:  # full-height posts
        here = posts[np.abs(posts[:, 0] - x) < 0.08]
        # embed_m below the ground at the post's centre, so well below it at every corner
        assert here[:, 1].min() == pytest.approx(ground(np.array([[x, 4.808]]))[0] - 0.5)
        assert here[:, 1].min() < ground(here[:, [0, 2]]).min() - 0.4
    rail = pos[np.abs(pos[:, 2] - 4.5) < 1e-6]  # the rail's front peaks
    start = rail[np.isclose(rail[:, 0], 100.0)]
    end = rail[np.isclose(rail[:, 0], 140.0)]
    assert start[:, 1].max() < 0.0  # turned down into the ground at the free start
    assert end[:, 1].max() == pytest.approx(0.75 - 0.052, abs=1e-6)  # joined end: full height


def test_osm_rails_near_the_edge_snap_onto_it_walls_keep_their_offset():
    m = straight()

    def fill(xz):  # ground falling away beyond the shoulder edge
        return -0.6 * np.maximum(np.abs(xz[:, 1]) - 4.5, 0.0)

    rail = BarrierSpan("left", 10.0, 50.0, offset_m=6.3)  # as traced in OSM (Mill St)
    ((visual, coll),) = span_meshes(m, rail, fill)
    assert np.abs(visual["steel"].positions[:, 2]).min() == pytest.approx(4.5)
    assert coll.positions[:, 1].max() == pytest.approx(1.2)  # collision above the road
    far = BarrierSpan("left", 10.0, 50.0, offset_m=8.0)  # really off the edge: on the ground
    ((visual, _),) = span_meshes(m, far, fill)
    pos = visual["steel"].positions
    assert np.abs(pos[:, 2]).min() == pytest.approx(8.0)
    assert pos[:, 1].max() == pytest.approx(-0.6 * 3.5 + 0.75)
    wall = BarrierSpan("left", 10.0, 50.0, "wall", offset_m=6.3)
    ((visual, _),) = span_meshes(m, wall, fill)
    assert np.abs(visual["concrete"].positions[:, 2]).min() == pytest.approx(6.3)


def test_long_barriers_stay_under_the_vertex_limit():
    m = straight(2000)
    spans = [
        BarrierSpan("left", 0.0, 2000.0),
        BarrierSpan("right", 0.0, 2000.0, "jersey_barrier"),
        BarrierSpan("right", 0.0, 2000.0, "wall", offset_m=9.0),
    ]
    pieces = barrier_pieces(m, spans, None)
    names = [p.name for p in pieces]
    assert len(names) == len(set(names))
    for p in pieces:
        assert p.mesh.vertex_count <= MAX_MESH_VERTICES
        to_kn5_mesh(p.mesh, p.name, 0, **p.flags)
    walls = [p for p in pieces if p.name.startswith("barrier_wall_")]
    assert np.abs(walls[0].mesh.positions[:, 2]).min() == pytest.approx(9.0)  # OSM offset


def test_textures_are_deterministic():
    assert set(TEXTURES) == {"concrete", "steel"}
    a, b = concrete_texture(64), concrete_texture(64)
    assert a.size == (64, 64) and a.mode == "RGB" and a.tobytes() == b.tobytes()
    s = steel_texture(32)
    assert s.size == (32, 32) and s.tobytes() == steel_texture(32).tobytes()
    assert 100 < np.asarray(a).mean() < 200 and 100 < np.asarray(s).mean() < 200


def test_parse_accepts_osm_data_and_reads_heights():
    from driftmaps.geo.overpass import OsmData

    lonlat = np.array([[-79.9, 43.3], [-79.9, 43.301]])
    ways = (
        Way(5, (1, 2), lonlat, {"barrier": "wall", "height": "4 ft"}),
        Way(6, (2, 3), lonlat, {"barrier": "fence"}),
        Way(7, (3, 4), lonlat, {"highway": "tertiary"}),
    )
    (wall,) = parse_barriers(OsmData(ways))
    assert (wall.id, wall.kind) == (5, "wall")
    assert wall.height_m == pytest.approx(4 * 0.3048)


def _network_enabled(config) -> bool:
    return os.environ.get("DRIFTMAPS_NETWORK") == "1" or "network" in (config.option.markexpr or "")


@pytest.mark.network
def test_live_fetch_finds_the_mill_street_rails(request, tmp_path):
    if not _network_enabled(request.config):
        pytest.skip("live network test: run with -m network or DRIFTMAPS_NETWORK=1")
    _, ways, *_ = mill_street()
    route = np.concatenate([ways[i].lonlat for i in MILL_ST_WAYS])
    got = fetch_barriers(route, cache_dir=tmp_path)
    assert set(MILL_ST_RAILS) <= {b.id for b in got}
    # The second call is served from the cache.
    again = fetch_barriers(route, cache_dir=tmp_path, offline=True)
    assert [b.id for b in again] == [b.id for b in got]
