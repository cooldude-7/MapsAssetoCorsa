"""Automatic guardrails where the design rules require one (geometry.warrants,
docs/research/guardrails.md §7): the rule tables, the hazards on synthetic roads and terrain,
length of need, joining, cutting at side roads, minimum lengths, the OSM roadside parser, the
guardrails.json switches and the build wiring (offline, Forks of the Credit fixture)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import shapely

from driftmaps.build import BuildOptions, build_track
from driftmaps.elevation.sources import CogSource, SourceInfo
from driftmaps.geo.barriers import (
    BarrierSpan,
    RoadsideFeatures,
    parse_roadside,
    parse_user_file,
    parse_user_spans,
)
from driftmaps.geo.overpass import parse_overpass
from driftmaps.geo.project import LocalProjection
from driftmaps.geo.router import Route, route_from_pins
from driftmaps.geometry.barriers import BarrierOptions, plan_barriers, span_meshes
from driftmaps.geometry.road import RoadModel
from driftmaps.geometry.terrain import TerrainOptions
from driftmaps.geometry.warrants import (
    COL_FLAT,
    Connection,
    WarrantInputs,
    WarrantOptions,
    clear_zone,
    connections_from_route,
    curve_factor,
    h_max,
    parse_maxspeed,
    runout_length,
    warrant_spans,
)

HW = 3.25  # asphalt half width
SHOULDER = 1.0
FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------------------------
# Rule tables
# --------------------------------------------------------------------------------------------


def test_critical_slope_heights():
    assert h_max([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]).tolist() == [0.6, 0.6, 0.9, 1.8, 2.7, 6.0]
    assert float(h_max(1.75)) == pytest.approx(1.35)  # linear between rows
    assert math.isinf(float(h_max(3.2)))  # flatter than 3:1: never, for the slope alone


def test_clear_zone_curve_factor_and_runout_tables():
    # Snake Rd (guardrails.md §3.3): tertiary, 60 posted -> 70 design, AADT >= 750.
    assert clear_zone(70.0, 1200.0, COL_FLAT) == 5.0
    assert clear_zone(80.0, 1200.0, COL_FLAT) == 5.0  # the same 70-80 row
    assert clear_zone(60.0, 3000.0, COL_FLAT) == 4.5  # Forks of the Credit
    assert clear_zone(110.0, 8000.0, 0) == 14.0  # fill 4H:1V-5H:1V, top row
    assert curve_factor(1000.0, 60.0) == 1.0  # R > 900 m: no factor
    assert curve_factor(450.0, 60.0) == 1.2
    assert curve_factor(470.0, 80.0) == 1.3  # nearest radius at or below R
    assert curve_factor(120.0, 70.0) == 1.5  # a "—" cell: the table's ceiling
    assert curve_factor(40.0, 60.0) == 1.5  # hairpin
    assert runout_length(60.0, 3000.0) == 34.0
    assert runout_length(70.0, 1200.0) == 42.0
    assert runout_length(40.0, 500.0) == 21.0  # <= 50 row
    assert runout_length(100.0, 20000.0) == 91.0


def test_parse_maxspeed():
    assert parse_maxspeed("50") == 50.0
    assert parse_maxspeed("40 mph") == pytest.approx(64.37, abs=0.01)
    assert parse_maxspeed("60;80") == 60.0
    assert parse_maxspeed("none") is None
    assert parse_maxspeed(None) is None


# --------------------------------------------------------------------------------------------
# Synthetic roads
# --------------------------------------------------------------------------------------------


def straight_model(length: float = 600.0) -> RoadModel:
    """A level road along +X (AC: driver's left is -Z), stations every 1 m."""
    s = np.arange(0.0, length + 0.5, 1.0)
    centre = np.stack([s, np.zeros_like(s), np.zeros_like(s)], axis=1)
    return RoadModel(s, centre, HW, SHOULDER, True, np.zeros(len(s), bool))


def fall(s0: float, s1: float, drop: float, ratio: float, side: str = "left", start: float = 0.0):
    """Ground (AC Y) of a straight road with one fill: flat at road level to `start` beyond
    the shoulder edge, then down at `ratio` H:V by `drop` metres, flat below, for stations
    s0..s1 on `side` (the left is -Z)."""

    def fn(xz: np.ndarray) -> np.ndarray:
        xz = np.asarray(xz, np.float64).reshape(-1, 2)
        x, lat = xz[:, 0], -xz[:, 1] if side == "left" else xz[:, 1]
        beyond = lat - (HW + SHOULDER + start)
        z = -np.clip(beyond / ratio, 0.0, drop)
        return np.where((x >= s0) & (x <= s1), z, 0.0)

    return fn


def inputs(model: RoadModel, dtm, **kw) -> WarrantInputs:
    n = len(model)
    args = dict(
        model=model,
        dtm=dtm,
        highway=np.full(n, "tertiary"),
        maxspeed_kmh=np.full(n, 60.0),
        sidewalk=np.zeros(n, bool),
        oneway=np.zeros(n, np.int64),
        s_range=(0.0, float(model.s[-1])),
    )
    return WarrantInputs(**(args | kw))


def spans_of(result, side: str) -> list[tuple[float, float]]:
    return [(sp.s0, sp.s1) for sp in result.spans if sp.side == side]


def test_high_steep_fill_gets_a_rail_with_its_length_of_need():
    m = straight_model()
    res = warrant_spans(inputs(m, fall(200.0, 320.0, 4.0, 1.5)))
    assert spans_of(res, "right") == []  # flat side
    [(a, b)] = spans_of(res, "left")
    [hz] = [h for h in res.hazards if h.kind == "fall"]
    assert hz.side == "left" and hz.h_m == pytest.approx(4.0, abs=0.3)
    # Posted 60 -> V_d 80, AADT 1,200 -> E = 49 m; B = DCZ 5.0, A = 1.0 (rail at the shoulder
    # edge): La = 49 (1 - 1/5) = 39.2 m for the near (westbound) traffic, downstream of +s on
    # the left; La' = 49 (1 - 4.25/5) = 7.35 m for the opposing traffic.
    assert hz.la_m == pytest.approx(39.2, abs=0.1)
    assert hz.la2_m == pytest.approx(7.35, abs=0.1)
    assert a == pytest.approx(200.0 - 7.35, abs=4.0)
    assert b == pytest.approx(320.0 + 39.2, abs=4.0)
    assert res.spans[0].source == "warrant" and res.spans[0].kind == "guard_rail"
    assert "fill" in res.report[0]["reasons"][0]


def test_mirror_side_extends_the_other_way():
    m = straight_model()
    res = warrant_spans(inputs(m, fall(200.0, 320.0, 4.0, 1.5, side="right")))
    [(a, b)] = spans_of(res, "right")
    # Right side: near traffic drives +s (right-hand traffic), so La goes below s0.
    assert a == pytest.approx(200.0 - 39.2, abs=4.0)
    assert b == pytest.approx(320.0 + 7.35, abs=4.0)


@pytest.mark.parametrize(
    ("drop", "ratio", "warranted"),
    [
        (1.0, 2.0, False),  # 2:1, 1 m < 1.8 m
        (2.5, 2.0, True),  # 2:1, 2.5 m > 1.8 m
        (5.0, 3.0, False),  # 3:1 up to 6 m is fine
        (8.0, 3.0, True),
        (8.0, 4.0, False),  # flatter than 3:1: never for the slope alone
        (1.0, 0.5, True),  # a 1 m drop-off (steeper than 1:1)
    ],
)
def test_embankment_warrant_heights(drop, ratio, warranted):
    m = straight_model()
    res = warrant_spans(inputs(m, fall(200.0, 320.0, drop, ratio)))
    assert bool(spans_of(res, "left")) is warranted


def test_h_scale_raises_the_heights():
    m = straight_model()
    dtm = fall(200.0, 320.0, 2.5, 2.0)
    assert spans_of(warrant_spans(inputs(m, dtm)), "left")
    assert not spans_of(warrant_spans(inputs(m, dtm), WarrantOptions(h_scale=1.5)), "left")


def test_fall_beyond_the_clear_zone_is_not_a_hazard():
    m = straight_model()
    # Flat for 8 m beyond the shoulder edge (9 m from the lane edge > DCZ 5 m).
    res = warrant_spans(inputs(m, fall(200.0, 320.0, 4.0, 1.5, start=8.0)))
    assert spans_of(res, "left") == []


def test_short_hazards_are_ignored_unless_big():
    m = straight_model()
    small = warrant_spans(inputs(m, fall(200.0, 206.0, 1.5, 1.0)))  # 6 m long, 1.5 m drop
    assert spans_of(small, "left") == []
    big = warrant_spans(inputs(m, fall(200.0, 206.0, 3.0, 1.0)))  # >= 2 m drops are exempt
    [(a, b)] = spans_of(big, "left")
    assert b - a >= 25.0


def test_v_ditch_is_not_a_hazard():
    m = straight_model()
    edge = HW + SHOULDER

    def ditch(xz):
        xz = np.asarray(xz, np.float64).reshape(-1, 2)
        lat = -xz[:, 1] - edge
        z = -np.clip(np.minimum(lat, 4.0 - lat), 0.0, 1.8)  # 1.8 m deep, 1:1 sides
        return np.where((xz[:, 0] >= 200) & (xz[:, 0] <= 320), z, 0.0)

    assert spans_of(warrant_spans(inputs(m, ditch)), "left") == []
    loose = WarrantOptions(ditch_max_depth_m=0.0)
    assert spans_of(warrant_spans(inputs(m, ditch), loose), "left")


def test_rails_are_cut_at_side_roads_and_not_joined_across_them():
    m = straight_model()
    road = Connection(260.0, "left", 8.0, 5.0, 1.0, "tertiary")
    res = warrant_spans(inputs(m, fall(200.0, 320.0, 4.0, 1.5), connections=[road]))
    left = spans_of(res, "left")
    assert len(left) == 2
    (a0, b0), (a1, b1) = left
    assert b0 == pytest.approx(251.0, abs=0.01) and a1 == pytest.approx(269.0, abs=0.01)
    # The other side is untouched by a left-side driveway.
    assert spans_of(res, "right") == []


def test_nearby_rails_join_and_short_pieces_grow():
    m = straight_model(800.0)

    def two(xz):
        return np.minimum(fall(200, 260, 4.0, 1.5)(xz), fall(330, 390, 4.0, 1.5)(xz))

    res = warrant_spans(inputs(m, two))
    assert len(spans_of(res, "left")) == 1  # gap 70 - La - La' < 50 m: one rail
    # A driveway in the gap keeps them apart.
    drive = Connection(300.0, "left", 4.0, 2.0, 1.0, "driveway")
    res = warrant_spans(inputs(m, two, connections=[drive]))
    assert len(spans_of(res, "left")) == 2


def test_bridge_ends_get_rails_on_all_four_corners():
    m = straight_model()
    res = warrant_spans(inputs(m, lambda xz: np.zeros(len(xz)), decks=[(300.0, 340.0)]))
    for side in ("left", "right"):
        pieces = spans_of(res, side)
        assert len(pieces) == 2
        (a0, b0), (a1, b1) = pieces
        assert b0 == pytest.approx(300.0) and a1 == pytest.approx(340.0)  # meet the deck
        assert b0 - a0 >= 25.0 - 1e-6 and b1 - a1 >= 25.0 - 1e-6
    # Right side, before the deck: the approach of the near traffic, La = 39.2 m.
    (a0, _), _ = spans_of(res, "right")
    assert 300.0 - a0 == pytest.approx(39.2, abs=0.1)
    # plan_barriers never puts them on the deck.
    plan = plan_barriers(m, bridges=[(300.0, 340.0)], heuristic=res.spans)
    assert all(sp.s1 <= 300.0 + 1e-6 or sp.s0 >= 340.0 - 1e-6 for sp in plan)


def test_deep_water_inside_the_clear_zone():
    m = straight_model()
    flat = lambda xz: np.zeros(len(xz))  # noqa: E731
    near = shapely.LineString([(150.0, -(HW + 3.0)), (260.0, -(HW + 3.0))])  # 3 m off, left
    far = shapely.LineString([(150.0, HW + 20.0), (260.0, HW + 20.0)])  # 20 m off, right
    res = warrant_spans(inputs(m, flat, water=[near, far]))
    [(a, b)] = spans_of(res, "left")
    assert a < 150.0 and b > 260.0
    assert spans_of(res, "right") == []
    assert any("water" in r for r in res.report[0]["reasons"])


def test_urban_curbed_streets_keep_only_big_hazards():
    m = straight_model()
    n = len(m)
    urban = dict(sidewalk=np.ones(n, bool), maxspeed_kmh=np.full(n, 50.0))
    small = warrant_spans(inputs(m, fall(200.0, 320.0, 1.5, 1.0), **urban))
    assert spans_of(small, "left") == []
    big = warrant_spans(inputs(m, fall(200.0, 320.0, 3.0, 1.0), **urban))
    assert spans_of(big, "left")
    slow = dict(maxspeed_kmh=np.full(n, 40.0))
    assert warrant_spans(inputs(m, fall(200.0, 320.0, 3.0, 1.0), **slow)).spans == []


def test_one_way_road_extends_only_upstream():
    m = straight_model()
    n = len(m)
    res = warrant_spans(inputs(m, fall(200.0, 320.0, 4.0, 1.5), oneway=np.ones(n, np.int64)))
    [(a, b)] = spans_of(res, "left")
    assert a == pytest.approx(200.0 - 39.2, abs=4.0)
    assert b == pytest.approx(320.0, abs=4.0)


def arc_model(radius: float = 100.0, turn: str = "left"):
    """A level arc of `radius` (centre on the driver's `turn` side), 250 m long."""
    t = np.arange(0.0, 250.5, 1.0) / radius
    sign = 1.0 if turn == "left" else -1.0
    # Start at the origin heading +X; the centre of curvature is to the left (-Z) or right.
    cz = -sign * radius
    x = radius * np.sin(t)
    z = cz + sign * radius * np.cos(t)
    centre = np.stack([x, np.zeros_like(x), z], axis=1)
    m = RoadModel(t * radius, centre, HW, SHOULDER, True, np.zeros(len(t), bool))
    return m, np.array([0.0, cz])


def test_outside_of_curves_widens_the_clear_zone():
    # A 4 m 1.5:1 drop starting 6.5 m beyond the lane edge: outside the 5 m tangent DCZ, inside
    # 5.0 x 1.5 = 7.5 m on the outside of an R = 100 m curve.
    for turn in ("left", "right"):
        m, c = arc_model(turn=turn)
        start = 6.5 - SHOULDER

        def outside(xz, c=c, start=start):
            r = np.linalg.norm(np.asarray(xz).reshape(-1, 2) - c, axis=1)
            beyond = r - (100.0 + HW + SHOULDER + start)
            return -np.clip(beyond / 1.5, 0.0, 4.0)

        res = warrant_spans(inputs(m, outside))
        out_side = "right" if turn == "left" else "left"
        assert spans_of(res, out_side), turn
    # The same drop on a straight road is beyond the clear zone.
    straight = inputs(straight_model(), fall(0, 600, 4.0, 1.5, start=6.5 - SHOULDER))
    assert not spans_of(warrant_spans(straight), "left")


def test_deterministic():
    m = straight_model()
    dtm = fall(200.0, 320.0, 4.0, 1.5)
    a = warrant_spans(inputs(m, dtm))
    b = warrant_spans(inputs(m, dtm))
    assert a.spans == b.spans and a.report == b.report


# --------------------------------------------------------------------------------------------
# Joining ways, OSM roadside features, guardrails.json
# --------------------------------------------------------------------------------------------


def test_connections_from_stubs_and_driveways():
    proj = LocalProjection(43.8, -80.0)
    m = straight_model()

    def ll(x: float, z: float) -> tuple[float, float]:
        lon, lat = proj.to_lonlat(np.array([x]), np.array([-z]))
        return float(lon[0]), float(lat[0])

    route = SimpleNamespace(
        node_path=lambda: [1, 2, 3],
        edges=[SimpleNamespace(u=1, v=2, way=10), SimpleNamespace(u=2, v=3, way=10)],
        stubs=[(2, 20, 4)],
        nodes={1: ll(0, 0), 2: ll(300, 0), 3: ll(600, 0), 4: ll(300, 50)},  # side road right
        ways={10: {"highway": "tertiary"}, 20: {"highway": "residential"}},
    )
    drive = RoadsideFeatures(
        access=(
            # A driveway ending at the left road edge at 150 m, no shared node.
            SimpleNamespace(
                id=30,
                tags={"highway": "service", "service": "driveway"},
                nodes=(5, 6),
                lonlat=np.array([ll(150, -4.5), ll(150, -40)]),
            ),
        )
    )
    conns = connections_from_route(m, proj, route, drive)
    by_side = {c.side: c for c in conns}
    assert by_side["right"].s == pytest.approx(300.0, abs=0.5)
    assert by_side["right"].kind == "residential" and by_side["right"].half_cut == 8.5
    assert by_side["left"].s == pytest.approx(150.0, abs=0.5)
    assert by_side["left"].kind == "driveway" and by_side["left"].half_cut == 4.0


def test_parse_roadside_features():
    payload = {
        "elements": [
            {"type": "way", "id": 1, "nodes": [1, 2, 3, 1], "tags": {"natural": "water"},
             "geometry": [{"lat": 43.0, "lon": -80.0}, {"lat": 43.001, "lon": -80.0},
                          {"lat": 43.001, "lon": -80.001}, {"lat": 43.0, "lon": -80.0}]},
            {"type": "way", "id": 2, "nodes": [4, 5], "tags": {"waterway": "river", "width": "20"},
             "geometry": [{"lat": 43.0, "lon": -80.0}, {"lat": 43.01, "lon": -80.0}]},
            {"type": "way", "id": 3, "nodes": [6, 7],
             "tags": {"natural": "water", "intermittent": "yes"},
             "geometry": [{"lat": 43.0, "lon": -80.0}, {"lat": 43.01, "lon": -80.0}]},
            {"type": "way", "id": 4, "nodes": [8, 9, 10],
             "tags": {"highway": "service", "service": "driveway"},
             "geometry": [{"lat": 43.0, "lon": -80.0}, None, {"lat": 43.01, "lon": -80.0}]},
            {"type": "way", "id": 5, "nodes": [11, 12], "tags": {"highway": "footway"},
             "geometry": [{"lat": 43.0, "lon": -80.0}, {"lat": 43.01, "lon": -80.0}]},
            {"type": "way", "id": 6, "nodes": [13, 14, 15, 16], "tags": {"highway": "track"},
             "geometry": [{"lat": 43.0, "lon": -80.0}, {"lat": 43.001, "lon": -80.0},
                          {"lat": 43.002, "lon": -80.0}, {"lat": 43.003, "lon": -80.0}]},
            {"type": "relation", "id": 7, "tags": {"natural": "water", "type": "multipolygon"},
             "members": [{"type": "way", "ref": 99, "role": "outer",
                          "geometry": [{"lat": 43.0, "lon": -80.0}, {"lat": 43.1, "lon": -80.0},
                                       None, {"lat": 43.2, "lon": -80.0}]}]},
            {"type": "way", "id": 8, "nodes": [17, 18], "tags": {"natural": "cliff"},
             "geometry": [{"lat": 43.0, "lon": -80.0}, {"lat": 43.01, "lon": -80.0}]},
        ]
    }  # fmt: skip
    f = parse_roadside(payload)
    assert sorted((w.id, w.kind) for w in f.water) == [(1, "water"), (2, "river"), (7, "water")]
    assert next(w for w in f.water if w.id == 2).half_width_m == 10.0
    assert [a.id for a in f.access] == [6]  # the clipped 2-piece driveway has 1-point pieces
    assert f.access[0].nodes == (13, 14, 15, 16)
    assert [r.id for r in f.rock] == [8]


def test_guardrails_json_auto_switch_and_erase_spans():
    data = {
        "auto": False,
        "guardrails": [
            {"side": "left", "from_m": 10, "to_m": 50},
            {"side": "right", "from_m": 100, "to_m": 300, "kind": "none"},
        ],
    }
    parsed = parse_user_file(data, s_origin=60.0)
    assert parsed.auto is False
    assert [(s.side, s.s0, s.s1) for s in parsed.spans] == [("left", 70.0, 110.0)]
    assert [(s.side, s.s0, s.s1, s.kind) for s in parsed.erase] == [("right", 160.0, 360.0, "none")]
    assert len(parse_user_spans(data, s_origin=60.0)) == 1  # the old API: real spans only
    assert parse_user_file({"auto": True}).auto is True
    with pytest.raises(ValueError):
        parse_user_file({"auto": "no"})


def test_type_m_rail_geometry_and_flared_ends():
    opt = BarrierOptions()
    assert (opt.rail_top_m, opt.post_spacing_m, opt.blockout_m) == (0.785, 1.905, 0.20)
    m = straight_model()
    span = BarrierSpan("left", 100.0, 200.0, "guard_rail", "warrant")
    [(vis, coll), *_] = span_meshes(m, span, None, opt, chunk_m=500.0)
    pos = vis["steel"].positions
    # The middle stands at the shoulder edge (left = -Z); the free ends flare 0.3 m out.
    mid = pos[np.abs(pos[:, 0] - 150.0) < 1.0]
    end = pos[np.abs(pos[:, 0] - 100.5) < 0.6]
    edge = HW + SHOULDER
    assert -mid[:, 2].max() == pytest.approx(edge, abs=0.01)  # the rail face
    assert -end[:, 2].max() > edge + 0.25


# --------------------------------------------------------------------------------------------
# Build wiring (offline)
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def forks_route() -> Route:
    osm = parse_overpass((FIXTURES / "osm" / "forks_of_the_credit.json").read_bytes())
    return route_from_pins([(43.80410, -79.99231), (43.81060, -79.98185)], osm=osm)


def _options(root: Path, **kw) -> BuildOptions:
    info = SourceInfo("hrdem-fixture", "HRDEM 5 m test window", 5.0, "DTM", "OGL-Canada-2.0", "x")
    dem = CogSource(info, str(FIXTURES / "dem" / "hrdem_forks_5m.tif"))
    args = dict(
        name="Forks Warrants",
        track_id="dm_forks_warrants",
        cache_dir=root / "cache",
        elevation_sources={"hrdem-fixture": dem},
        elevation_chain=["hrdem-fixture"],
        terrain=TerrainOptions(rings=((15.0, 3.0), (60.0, 8.0), (150.0, 20.0)), near_m=60.0),
        offline=True,
        trees=False,
        buildings=False,
        debug_png=False,
        osm_barriers=(),
        roadside_features=RoadsideFeatures(),
    )
    return BuildOptions(**(args | kw))


def test_build_places_automatic_rails_and_reports_why(forks_route, tmp_path):
    res = build_track(forks_route, tmp_path / "out", _options(tmp_path))
    sc = res.scenery
    auto = [sp for sp in sc.spans if sp.source == "warrant"]
    assert auto, "the Credit River bridge ends alone warrant rails"
    decks = [(p.s0, p.s1) for p in sc.plans if p.railing]
    assert decks
    for sp in auto:
        assert sp.s1 - sp.s0 >= 12.0 - 1e-6
        assert all(sp.s1 <= a + 1e-6 or sp.s0 >= b - 1e-6 for a, b in decks)
    report = json.loads((res.track_dir / "data" / "guardrails_auto.json").read_text())
    assert report["spans"] and all(s["reasons"] for s in report["spans"])
    assert len(report["built"]) == len(auto)
    assert res.stats["scenery"]["warrant_spans"] == len(auto)
    assert any("bridge end" in r for s in report["spans"] for r in s["reasons"])
    # Off: the flag, or "auto": false in guardrails.json.
    off = build_track(
        forks_route, tmp_path / "off", _options(tmp_path, auto_guardrails=False, force=True)
    )
    assert not [sp for sp in off.scenery.spans if sp.source == "warrant"]
    user = tmp_path / "guardrails.json"
    user.write_text(json.dumps({"auto": False, "guardrails": []}), encoding="utf-8")
    off2 = build_track(forks_route, tmp_path / "off2", _options(tmp_path, guardrails_file=user))
    assert not [sp for sp in off2.scenery.spans if sp.source == "warrant"]
    # "kind": "none" erases them where the user says.
    a, b = auto[0].s0 - res.geometry.s_a, auto[0].s1 - res.geometry.s_a
    user.write_text(
        json.dumps([{"side": auto[0].side, "from_m": a - 1, "to_m": b + 1, "kind": "none"}]),
        encoding="utf-8",
    )
    erased = build_track(forks_route, tmp_path / "erased", _options(tmp_path, guardrails_file=user))
    left = [
        sp
        for sp in erased.scenery.spans
        if sp.source == "warrant" and sp.side == auto[0].side and sp.s0 < auto[0].s1
        and sp.s1 > auto[0].s0
    ]  # fmt: skip
    assert left == []
