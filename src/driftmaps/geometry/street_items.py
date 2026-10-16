"""Shared contract for roadside items (signs, poles, lights, mailboxes ...) and junction control.

Several builders place roadside items: the road network (`geometry.network`, side roads,
junction mouths, roundabouts) and the street furniture layer (`geometry.furniture`, from OSM
nodes, Mapillary detections and Ontario rules). They exchange `StreetItem`s; the furniture layer
renders every item (merged meshes per cell) and de-duplicates items of the same kind that
stand close together, preferring the better `source` (`SOURCE_PRIORITY`).

Junction control (who stops at a junction) is decided once, here, by `approach_stops`, so the
stop sign (furniture) and the stop bar (network markings) always agree.

Coordinates: AC world frame (X east, Y up, Z south), metres, like every mesh in the KN5.
"""

from __future__ import annotations

from dataclasses import dataclass

# Item kinds. Signs use "sign:<name>", speed limits "sign:speed_limit:<km/h>".
SIGN_STOP = "sign:stop"
SIGN_STOP_AHEAD = "sign:stop_ahead"
SIGN_ALL_WAY = "sign:all_way"  # the "ALL WAY" tab under a stop sign
SIGN_YIELD = "sign:yield"
SIGN_ROUNDABOUT_AHEAD = "sign:roundabout_ahead"
SIGN_KEEP_RIGHT = "sign:keep_right"
SIGN_CHEVRON_LEFT = "sign:chevron_left"  # points left: the road turns left
SIGN_CHEVRON_RIGHT = "sign:chevron_right"
SIGN_DEAD_END = "sign:dead_end"
SIGN_SPEED_LIMIT = "sign:speed_limit"  # + ":<km/h>"
SIGN_STREET_NAME = "sign:street_name"  # `text` holds the name
POLE_UTILITY = "pole:utility"  # wooden hydro pole
LIGHT_STREET = "light:street"
SIGNAL_TRAFFIC = "signal:traffic"
MAILBOX = "mailbox"
HYDRANT = "hydrant"

# Where an item came from, best first: a measured position beats an inferred one.
SOURCE_PRIORITY = ("municipal", "osm", "mapillary", "rule")


@dataclass(frozen=True)
class StreetItem:
    """One roadside item standing on the ground.

    `position` is the base of the item on the ground (AC world coordinates). `facing` is the
    horizontal unit vector (x, z) the item's front points to: for a sign, towards the drivers
    it is meant for (opposite to their direction of travel). `text` carries a street name or
    similar; `ref` an id from the source (OSM node, Mapillary feature) for de-duplication and
    credits.
    """

    kind: str
    position: tuple[float, float, float]
    facing: tuple[float, float]
    source: str = "rule"
    text: str = ""
    ref: str = ""


# OSM highway classes from most to least important; `_link` ranks one below its road.
HIGHWAY_RANK = {
    "motorway": 9,
    "trunk": 8,
    "primary": 7,
    "secondary": 6,
    "tertiary": 5,
    "unclassified": 4,
    "residential": 3,
    "living_street": 2,
    "service": 1,
    "track": 1,
}


def highway_rank(highway: str) -> int:
    """Rank of an OSM highway value (0 for unknown); a `_link` ranks one below its road."""
    if highway.endswith("_link"):
        return max(HIGHWAY_RANK.get(highway[:-5], 0) - 1, 0)
    return HIGHWAY_RANK.get(highway, 0)


def approach_stops(
    approach_highway: str,
    other_highways: list[str],
    *,
    mapped: str | None = None,
    roundabout: bool = False,
) -> str | None:
    """Control for traffic arriving at a junction on one approach: "stop", "yield",
    "signal" or None (it has the right of way).

    `mapped` is what the data says for this approach ("stop", "yield", "signal", "all_way"
    from OSM `highway=stop|give_way|traffic_signals`, `stop=all`, a Mapillary stop sign facing
    this approach, a municipal sign inventory); it always wins ("all_way" returns "stop").
    Without data: an approach entering a roundabout yields; otherwise, as on Ontario roads
    (Highway Traffic Act s.136/138 practice, OTM Book 5), the approach of a lower class than
    the busiest other road at the junction stops, and equal classes get no inferred control
    (all-way stops are only placed where the data says so).
    """
    if mapped:
        return "stop" if mapped == "all_way" else mapped
    if roundabout:
        return "yield"
    if not other_highways:
        return None
    if highway_rank(approach_highway) < max(highway_rank(h) for h in other_highways):
        return "stop"
    return None
