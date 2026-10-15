"""Tests for driftmaps.geo.attributes: width cascade, lanes, surfaces, driving side."""

from __future__ import annotations

import pytest

from driftmaps.geo.attributes import (
    DEFAULT_WIDTH_M,
    driving_side,
    is_bridge,
    is_oneway,
    is_tunnel,
    lane_count,
    layer_of,
    parse_width,
    region_for,
    road_width,
    route_tags,
    surface_class,
)


@pytest.mark.parametrize(
    ("text", "metres"),
    [
        ("6", 6.0),
        ("6.5 m", 6.5),
        ("6,5", 6.5),
        ("7m", 7.0),
        ("20'", 6.096),
        ("20 ft", 6.096),
        ("12'6\"", 3.81),
        ("6;7", 6.0),
        ("~6", 6.0),
        ("", None),
        ("narrow", None),
        ("0", None),
        (None, None),
    ],
)
def test_parse_width(text, metres):
    assert parse_width(text) == (pytest.approx(metres) if metres is not None else None)


def test_width_cascade():
    # 1. explicit width wins over lanes
    assert road_width({"highway": "secondary", "width": "8", "lanes": "4"}) == 8.0
    assert road_width({"highway": "secondary", "width:carriageway": "7.2", "width": "12"}) == 7.2
    # 2. lanes x lane width (region dependent)
    assert road_width({"highway": "secondary", "lanes": "2"}, "na") == pytest.approx(6.8)
    assert road_width({"highway": "secondary", "lanes": "2"}, "intl") == pytest.approx(6.5)
    assert road_width({"highway": "primary_link", "lanes": "1"}) == pytest.approx(4.0)
    # 3. class default
    assert road_width({"highway": "tertiary"}) == DEFAULT_WIDTH_M["tertiary"]
    assert road_width({"highway": "motorway_link"}) == DEFAULT_WIDTH_M["link"]
    assert road_width({"highway": "weird"}) == 5.0
    # clamps
    assert road_width({"highway": "service"}, min_width=5.0) == 5.0
    assert road_width({"highway": "primary", "width": "60"}) == 20.0


def test_lanes_and_oneway():
    assert lane_count({"highway": "residential"}) == 2
    assert lane_count({"highway": "residential", "oneway": "yes"}) == 1
    assert lane_count({"highway": "motorway"}) == 2
    assert lane_count({"highway": "primary", "lanes": "3"}) == 3
    assert lane_count({"highway": "service"}) == 1
    assert is_oneway({"oneway": "yes"}) == 1
    assert is_oneway({"oneway": "-1"}) == -1
    assert is_oneway({"highway": "motorway"}) == 1
    assert is_oneway({"highway": "motorway", "oneway": "no"}) == 0
    assert is_oneway({"junction": "roundabout"}) == 1
    assert is_oneway({"highway": "secondary"}) == 0


def test_surface_bridge_tunnel_layer():
    assert surface_class({"surface": "asphalt"}) == "asphalt"
    assert surface_class({"surface": "concrete:plates"}) == "asphalt"
    assert surface_class({"surface": "sett"}) == "setts"
    assert surface_class({"surface": "gravel"}) == "gravel"
    assert surface_class({"highway": "track"}) == "gravel"
    assert surface_class({"highway": "tertiary"}) == "asphalt"
    assert is_bridge({"bridge": "yes"}) and is_bridge({"bridge": "viaduct"})
    assert not is_bridge({"bridge": "no"}) and not is_bridge({})
    assert is_tunnel({"tunnel": "building_passage"}) and is_tunnel({"covered": "yes"})
    assert layer_of({"layer": "-1"}) == -1 and layer_of({}) == 0 and layer_of({"layer": "x"}) == 0


def test_route_tags_keeps_useful_keys_only():
    tags = {
        "highway": "secondary",
        "name": "Forks of the Credit Road",
        "lanes:forward": "1",
        "tiger:cfcc": "A41",
        "source": "survey",
    }
    assert route_tags(tags) == {
        "highway": "secondary",
        "lanes:forward": "1",
        "name": "Forks of the Credit Road",
    }


@pytest.mark.parametrize(
    ("lat", "lon", "side"),
    [
        (43.8106, -79.9818, "right"),  # Forks of the Credit, Ontario
        (43.6532, -79.3832, "right"),  # Toronto
        (41.71712, -88.29689, "right"),  # Aurora, IL
        (25.76, -80.19, "right"),  # Miami
        (43.14, 5.85, "right"),  # Gorges d'Ollioules
        (50.95, 1.85, "right"),  # Calais
        (50.72, 1.61, "right"),  # Boulogne-sur-Mer
        (35.10, 129.04, "right"),  # Busan
        (25.03, 121.56, "right"),  # Taipei
        (36.48, 138.85, "left"),  # Akina / Mt. Haruna
        (35.68, 139.69, "left"),  # Tokyo
        (43.06, 141.35, "left"),  # Sapporo
        (51.51, -0.13, "left"),  # London
        (51.13, 1.31, "left"),  # Dover
        (53.35, -6.26, "left"),  # Dublin
        (55.95, -3.19, "left"),  # Edinburgh
        (-33.87, 151.21, "left"),  # Sydney
        (-36.85, 174.76, "left"),  # Auckland
        (-26.20, 28.05, "left"),  # Johannesburg
        (13.75, 100.50, "left"),  # Bangkok
        (1.35, 103.82, "left"),  # Singapore
        (22.28, 114.16, "left"),  # Hong Kong
        (22.54, 114.06, "right"),  # Shenzhen
    ],
)
def test_driving_side(lat, lon, side):
    assert driving_side(lat, lon) == side


def test_region_for():
    assert region_for(43.81, -79.98) == "na"
    assert region_for(41.7, -88.3) == "na"
    assert region_for(43.14, 5.85) == "intl"
    assert region_for(36.48, 138.85) == "intl"
