"""Bridge structures (geometry/bridges.py): deck, railings, abutments, piers, terrain checks."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from driftmaps.ac.model import MAX_MESH_VERTICES
from driftmaps.geometry.barriers import plan_barriers
from driftmaps.geometry.bridges import (
    BridgeOptions,
    _deck,
    _pier,
    bridge_pieces,
    bridge_railing_spans,
    bridge_spans,
    deck_ceiling,
    deck_half_width,
    plan_bridge,
    terrain_through_deck,
)
from driftmaps.geometry.primitives import face_normals, to_kn5_mesh
from driftmaps.geometry.road import RoadModel, build_road_meshes
from driftmaps.geometry.terrain import build_terrain

DEPTH = 12.0  # valley depth below the road


def valley_model(depth: float = DEPTH, grade: float = 0.02) -> RoadModel:
    """A road heading east (+X) with a gentle curve, crossing a V valley at x = 200..300 on a
    bridge from x = 205 to 295 (stations ~ x)."""
    x = np.arange(0.0, 501.0)
    z = 0.0005 * (x - 300.0) ** 2
    centre = np.stack([x, grade * x, z], axis=1)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))])
    return RoadModel(
        s=s,
        centre=centre,
        half_width=3.5,
        shoulder=1.0,
        centre_line=True,
        bridge=(x >= 205) & (x <= 295),
    )


def valley_ground(depth: float = DEPTH, grade: float = 0.02):
    def ground(xz: np.ndarray) -> np.ndarray:
        x = xz[:, 0]
        return grade * x - depth * np.clip(1.0 - np.abs(x - 250.0) / 50.0, 0.0, 1.0)

    return ground


def nearest(m: RoadModel, pts: np.ndarray) -> np.ndarray:
    """Index of the station nearest to each point (in XZ)."""
    return cKDTree(m.centre[:, [0, 2]]).query(pts[:, [0, 2]])[1]


def test_bridge_spans_are_the_runs_of_bridge_stations():
    m = valley_model()
    ((s0, s1),) = bridge_spans(m)
    assert m.centre[m.index_at(s0), 0] == pytest.approx(205.0)
    assert m.centre[m.index_at(s1), 0] == pytest.approx(295.0)
    m.bridge[:] = False
    assert bridge_spans(m) == []


def test_plan_places_abutment_faces_and_piers_15_to_25_m_apart():
    m = valley_model()
    ((s0, s1),) = bridge_spans(m)
    plan = plan_bridge(m, s0, s1, valley_ground())
    assert s0 + 1.0 <= plan.face0 < s0 + 10 and s1 - 10 < plan.face1 <= s1 - 1.0
    stations = [plan.face0, *plan.piers, plan.face1]
    gaps = np.diff(stations)
    assert len(plan.piers) >= 3
    assert np.all(gaps >= 15.0 - 1e-9) and np.all(gaps <= 25.0 + 1e-9)


def test_piers_reach_the_ground():
    m = valley_model()
    ground = valley_ground()
    ((s0, s1),) = bridge_spans(m)
    for style in ("columns", "wall"):
        opt = BridgeOptions(pier_style=style)
        plan = plan_bridge(m, s0, s1, ground, opt)
        for sp in plan.piers:
            boxes = _pier(m, sp, plan, ground, opt)
            supports = boxes[1:] if style == "columns" else boxes  # skip the cap beam
            assert supports
            for b in supports:
                p = b.positions
                bottom = p[np.isclose(p[:, 1], p[:, 1].min())]
                assert np.all(bottom[:, 1] <= ground(bottom[:, [0, 2]]) - opt.embed_m + 1e-6)
                # ...and up to the deck structure.
                deck_y = np.interp(sp, m.s, m.centre[:, 1])
                assert p[:, 1].max() >= deck_y - plan.thickness - opt.cap_height_m


def test_no_piers_where_the_gap_is_small_and_solid_when_ground_never_clears():
    m = valley_model(depth=4.0)
    ((s0, s1),) = bridge_spans(m)
    plan = plan_bridge(m, s0, s1, valley_ground(depth=4.0))
    assert not plan.solid and plan.piers == ()  # underside only ~3 m up at the deepest
    flat = plan_bridge(m, s0, s1, valley_ground(depth=0.5))
    assert flat.solid and flat.piers == ()
    pieces, _ = bridge_pieces(m, [(s0, s1)], valley_ground(depth=0.5))
    assert {p.name for p in pieces} == {
        "bridge00_concrete",
        "1WALL_BRIDGE00_L00",
        "1WALL_BRIDGE00_R00",
    }


def test_deck_slab_faces_point_outward_and_down():
    m = valley_model()
    ((s0, s1),) = bridge_spans(m)
    plan = plan_bridge(m, s0, s1, valley_ground())
    deck = _deck(m, plan, BridgeOptions())
    fn = face_normals(deck)
    c = deck.positions[deck.triangles].mean(axis=1)
    centre = m.centre[nearest(m, c)]
    under = fn[:, 1] < -0.99
    assert under.any() and np.all(c[under, 1] < centre[under, 1] - 0.99)
    side = ~under
    away = c[side] - centre[side]
    away[:, 1] = 0.0
    assert np.all(np.einsum("ij,ij->i", fn[side], away) > 0)
    # The slab reaches the railing's outer face (shoulder edge + parapet width).
    half = deck_half_width(m)[0]
    j = nearest(m, deck.positions)
    lateral = np.abs(np.einsum("ij,ij->i", deck.positions - m.centre[j], m.left[j]))
    assert lateral.max() == pytest.approx(half, abs=0.02)


@pytest.mark.parametrize("railing", ["parapet", "steel"])
def test_railing_and_collision_on_both_sides_of_the_whole_deck(railing):
    m = valley_model()
    ((s0, s1),) = bridge_spans(m)
    opt = BridgeOptions(railing=railing)
    pieces, (plan,) = bridge_pieces(m, [(s0, s1)], valley_ground(), opt)
    names = {p.name: p for p in pieces}
    assert ("bridge00_steel" in names) == (railing == "steel")
    assert names["bridge00_concrete"].flags == {"lod_out": opt.lod_out_m}
    for side, sign in (("L", 1.0), ("R", -1.0)):
        coll = names[f"1WALL_BRIDGE00_{side}00"]
        assert coll.physical and coll.flags["is_renderable"] is False
        pos = coll.mesh.positions
        i = nearest(m, pos)
        lateral = np.einsum("ij,ij->i", pos - m.centre[i], m.left[i])
        assert np.allclose(lateral, sign * m.outer[i], atol=0.05)  # on the railing face
        st = m.s[i]
        assert st.min() <= s0 + 1.0 and st.max() >= s1 - 1.0  # the whole deck
        top = pos[:, 1] - m.centre[i, 1]
        assert top.max() >= 1.0  # higher than the parapet: no driving off
    kind = "parapet" if railing == "parapet" else "bridge_rail"
    assert bridge_railing_spans([plan], opt)[0].kind == kind
    for p in pieces:
        assert p.mesh.vertex_count <= MAX_MESH_VERTICES
        to_kn5_mesh(p.mesh, p.name, 0, **p.flags)


def test_approach_guardrails_meet_the_parapet_collision():
    m = valley_model()
    spans = bridge_spans(m)
    planned = plan_barriers(m, bridges=spans)
    (s0, s1) = spans[0]
    left = sorted((sp.s0, sp.s1) for sp in planned if sp.side == "left")
    assert left == [(s0 - 20.0, s0), (s1, s1 + 20.0)]  # continuous with the railing [s0, s1]


def test_terrain_from_build_terrain_stays_out_of_the_deck():
    m = valley_model()
    roads = build_road_meshes(m)
    terrain = build_terrain(
        m, roads.left_outer, roads.right_outer, lambda xz, far: valley_ground()(xz)
    )

    def ground(xz):
        h = terrain.height_at(xz)
        return np.where(np.isfinite(h), h, valley_ground()(xz))

    ((s0, s1),) = bridge_spans(m)
    plan = plan_bridge(m, s0, s1, ground)
    check = terrain_through_deck(m, plan, ground)
    assert check.ok, f"terrain {check.max_intrusion:.2f} m into the deck at {check.points[:3]}"


def test_deck_check_finds_a_bump_and_the_ceiling_clamps_it():
    m = valley_model()
    base = valley_ground()

    def bumpy(xz):  # a knoll rising to the road mid-span
        bump = 14.0 * np.exp(-(((xz[:, 0] - 262.0) / 4.0) ** 2))
        return base(xz) + bump

    ((s0, s1),) = bridge_spans(m)
    plan = plan_bridge(m, s0, s1, base)
    check = terrain_through_deck(m, plan, bumpy)
    assert not check.ok and check.max_intrusion > 1.0
    assert np.all(np.abs(check.points[:, 0] - 262.0) < 8.0)

    def clamped(xz):
        return np.minimum(bumpy(xz), deck_ceiling(m, [plan], xz))

    assert terrain_through_deck(m, plan, clamped).ok
    # Away from the deck the ceiling is +inf.
    far = np.array([[262.0, 60.0], [100.0, 0.0]])
    assert np.all(np.isinf(deck_ceiling(m, [plan], far)))
