from driftmaps.geometry.street_items import StreetItem, approach_stops, highway_rank


def test_highway_rank_orders_classes_and_links():
    assert highway_rank("secondary") > highway_rank("tertiary") > highway_rank("residential")
    assert highway_rank("primary_link") == highway_rank("primary") - 1
    assert highway_rank("footway") == 0


def test_approach_stops_rules():
    assert approach_stops("residential", ["tertiary", "tertiary"]) == "stop"
    assert approach_stops("tertiary", ["residential", "tertiary"]) is None
    assert approach_stops("residential", ["residential"]) is None  # no invented all-way stops
    assert approach_stops("residential", ["residential"], mapped="all_way") == "stop"
    assert approach_stops("tertiary", ["residential"], mapped="signal") == "signal"
    assert approach_stops("secondary", ["secondary"], roundabout=True) == "yield"
    assert approach_stops("residential", []) is None


def test_street_item_is_hashable_and_defaults_to_rule():
    item = StreetItem("sign:stop", (1.0, 0.0, 2.0), (0.0, 1.0))
    assert item.source == "rule" and len({item, item}) == 1
