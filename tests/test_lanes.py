"""Lane layout from OSM lane tags and the lines Ontario paints (`geometry.lanes`), the town
signal rule (`street_items.approach_stops`) and turn arrows (`furniture.arrow_kind`)."""

from __future__ import annotations

import numpy as np
import pytest

from driftmaps.geometry.furniture import arrow_kind
from driftmaps.geometry.lanes import (
    BROKEN_GAP_M,
    BROKEN_LINE_M,
    WayLanes,
    paint_lines,
    road_lanes,
    way_lanes,
)
from driftmaps.geometry.street_items import approach_stops, signalized


def test_way_lanes_from_tags():
    assert way_lanes({"highway": "residential"}) == WayLanes(1, 1, tagged=False)
    two = way_lanes({"highway": "primary", "lanes": "4"})
    assert (two.forward, two.backward, two.tagged) == (2, 2, True)
    tags = {"highway": "secondary", "lanes": "3", "lanes:forward": "2", "lanes:backward": "1",
            "turn:lanes:forward": "left|through"}  # fmt: skip
    along = way_lanes(tags)
    assert (along.forward, along.backward, along.turn_forward) == (2, 1, ("left", "through"))
    against = way_lanes(tags, along=False)  # driven against the way: the directions swap
    assert (against.forward, against.backward, against.turn_backward) == (1, 2, ("left", "through"))
    one = way_lanes({"highway": "primary", "oneway": "yes", "lanes": "4",
                     "turn:lanes": "left|through|through|through"})  # fmt: skip
    assert (one.forward, one.backward) == (4, 0)
    assert one.turn_forward == ("left", "through", "through", "through")
    # turn:lanes that does not match the lane count is ignored; "none" means no arrow.
    assert way_lanes({"oneway": "yes", "lanes": "2", "turn:lanes": "left|"}).turn_forward == (
        "left",
        "",
    )
    assert way_lanes({"oneway": "yes", "lanes": "3", "turn:lanes": "left|"}).turn_forward == ()


def test_lane_layout_spans_and_centres():
    n = 5
    hw = np.full(n, 7.2)  # 3 lanes of 4.8 m: two forward (right), one back (left)
    lanes = road_lanes(hw, [WayLanes(2, 1)] * n)
    assert lanes.lane_w[0] == pytest.approx(4.8)
    assert lanes.centre[0] == pytest.approx(2.4)  # 2.4 m left of the centreline
    assert lanes.span(0, True) == pytest.approx((-7.2, 2.4))
    assert lanes.span(0, False) == pytest.approx((2.4, 7.2))
    assert lanes.lane_centres(0, True) == pytest.approx([0.0, -4.8])  # left to right
    one = road_lanes(np.full(n, 5.4), [WayLanes(3, 0)] * n)
    assert one.span(0, False) is None  # a one-way: no lane the other way
    assert one.lane_centres(0, True) == pytest.approx([3.6, 0.0, -3.6])
    # Left-hand traffic mirrors it.
    lh = road_lanes(hw, [WayLanes(2, 1)] * n, right_hand=False)
    assert lh.centre[0] == pytest.approx(-2.4) and lh.span(0, True) == pytest.approx((-2.4, 7.2))


def _lines(lanes, n, *, urban, highway, oneway=0, centre_ok=True):
    s = np.arange(n, dtype=float)
    return paint_lines(
        s, lanes, urban=np.full(n, urban), highway=[highway] * n, centre_ok=np.full(n, centre_ok),
        edge_ok=np.ones(n, bool), oneway=np.full(n, oneway),
    )  # fmt: skip


def test_rural_two_lane_road_keeps_its_lines():
    n = 40
    lanes = road_lanes(np.full(n, 3.4), [WayLanes(1, 1)] * n)
    lines = _lines(lanes, n, urban=False, highway="tertiary")
    centre = [ln for ln in lines if ln.role == "centre"]
    edges = [ln for ln in lines if ln.role == "edge"]
    assert len(centre) == 2 and all(ln.colour == "yellow" and ln.keep.all() for ln in centre)
    assert sorted(float(ln.offset[0]) for ln in centre) == pytest.approx([-0.12, 0.12])
    assert {ln.colour for ln in edges} == {"white"}
    assert sorted(float(ln.offset[0]) for ln in edges) == pytest.approx([-3.2, 3.2])
    assert not [ln for ln in lines if ln.role == "lane"]


def test_urban_one_way_lanes_and_turn_lane_line():
    n = 60
    turns = ("left", "", "")
    lanes = road_lanes(np.full(n, 5.4), [WayLanes(3, 0, turns)] * n)
    lines = _lines(lanes, n, urban=True, highway="primary", oneway=1)
    assert not [ln for ln in lines if ln.role in ("edge", "centre")]  # curbs, no centre
    lane = sorted((ln for ln in lines if ln.role == "lane"), key=lambda ln: -ln.offset[0])
    assert len(lane) == 2 and all(ln.colour == "white" for ln in lane)
    assert [float(ln.offset[0]) for ln in lane] == pytest.approx([1.8, -1.8])
    assert lane[0].keep.all()  # beside the left-turn lane: solid
    period = BROKEN_LINE_M + BROKEN_GAP_M
    assert lane[1].keep.mean() == pytest.approx(BROKEN_LINE_M / period, abs=0.05)  # 3 m / 6 m


def test_urban_centre_lines_by_class():
    n = 30
    two = road_lanes(np.full(n, 3.3), [WayLanes(1, 1)] * n)
    assert not _lines(two, n, urban=True, highway="residential")  # local street: no lines
    collector = [ln for ln in _lines(two, n, urban=True, highway="tertiary") if ln.role == "centre"]
    assert len(collector) == 1 and not collector[0].keep.all()  # single broken yellow
    four = road_lanes(np.full(n, 7.0), [WayLanes(2, 2)] * n)
    arterial = [ln for ln in _lines(four, n, urban=True, highway="secondary")]
    assert len([ln for ln in arterial if ln.role == "centre"]) == 2  # double solid
    assert len([ln for ln in arterial if ln.role == "lane"]) == 2  # one per direction


def test_one_way_rural_edges_yellow_left():
    n = 20
    lanes = road_lanes(np.full(n, 3.6), [WayLanes(2, 0)] * n)
    edges = [ln for ln in _lines(lanes, n, urban=False, highway="primary", oneway=1)
             if ln.role == "edge"]  # fmt: skip
    left = [ln for ln in edges if ln.offset[0] > 0]
    right = [ln for ln in edges if ln.offset[0] < 0]
    assert [ln.colour for ln in left] == ["yellow"] and [ln.colour for ln in right] == ["white"]


def test_town_signals_rule():
    assert signalized("secondary", ["primary"]) and signalized("tertiary", ["primary"])
    assert not signalized("residential", ["primary"])  # a local street stops
    assert not signalized("tertiary", ["tertiary"])  # two collectors: no rule-made signal
    assert not signalized("primary_link", ["primary"])
    assert approach_stops("secondary", ["primary", "secondary"], urban=True,
                          crossing=["primary"]) == "signal"  # fmt: skip
    assert approach_stops("secondary", ["primary"]) == "stop"  # rural: the old rule
    assert approach_stops("residential", ["primary"], urban=True) == "stop"
    assert approach_stops("secondary", ["primary"], urban=True, mapped="stop") == "stop"


def test_arrow_kinds():
    assert arrow_kind("left") == "arrow:left"
    assert arrow_kind("through;right") == "arrow:straight_right"
    assert arrow_kind("left;through") == "arrow:left_straight"
    assert arrow_kind("") is None and arrow_kind("none") is None
