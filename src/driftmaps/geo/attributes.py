"""OSM tags -> road parameters: carriageway width, lanes, surface class, driving side.

Width cascade (docs/research/geo.md §1.5, PLAN §5.1):
1. A `width`, `width:carriageway` or `est_width` tag (first value of a `;` list; metres unless
   it says ft / ' / ").
2. Otherwise `lanes` x the lane width for the road class and region.
3. Otherwise a per-`highway` default carriageway width.

Lane widths: the "na" region (US, Canada, Mexico) follows the AASHTO-based FHWA guidance quoted
in geo.md §1.5 (10-12 ft majors, 12 ft high-speed rural, 9 ft low-volume): VERIFIED there via
https://www.fhwa.dot.gov/planning/css/resources/lanewidth. Canada's TAC Geometric Design Guide
uses similar 3.0-3.7 m lanes (ASSUMED, not re-checked). The "intl" values (3.5 m primary down
to 2.75 m minor) are typical European national guidance: ASSUMED (geo.md says the same). Japan's
Road Structure Ordinance classes (3.5 / 3.25 / 3.0 / 2.75 m) match "intl" closely (ASSUMED).

`road_width` returns the real-world estimate. The geometry stage clamps it to its minimum
driveable width (`MIN_DRIVE_WIDTH_M`, configurable there) and ramps changes over ~30 m.

Driving side comes from a rough bounding-box table of left-hand-traffic countries; everything
else drives on the right. The boxes are hand-drawn and ASSUMED (good enough for whole touge
roads; border areas can be wrong, see the notes in `_LEFT_HAND_BOXES`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping

__all__ = [
    "DEFAULT_LANES",
    "DEFAULT_WIDTH_M",
    "LANE_WIDTH_M",
    "MAX_WIDTH_M",
    "MIN_DRIVE_WIDTH_M",
    "ROUTE_TAG_KEYS",
    "driving_side",
    "is_bridge",
    "is_oneway",
    "is_tunnel",
    "layer_of",
    "lane_count",
    "lane_width",
    "parse_width",
    "region_for",
    "road_width",
    "route_tags",
    "surface_class",
]

MIN_DRIVE_WIDTH_M = 5.0  # geo.md §1.5: clamp floor used by the geometry stage
MAX_WIDTH_M = 20.0

# Default carriageway width (m) when a way has no width or lanes tag (geo.md §1.5 table;
# motorway/trunk values are per one-way carriageway).
DEFAULT_WIDTH_M: dict[str, float] = {
    "motorway": 7.5,
    "trunk": 7.5,
    "primary": 7.0,
    "secondary": 6.5,
    "tertiary": 6.0,
    "unclassified": 5.0,
    "residential": 5.5,
    "living_street": 4.0,
    "service": 4.0,
    "track": 3.5,
    "link": 4.5,  # any *_link
}
_FALLBACK_WIDTH_M = 5.0

# Default total lane count (both directions) when `lanes` is missing (geo.md §1.5).
DEFAULT_LANES: dict[str, int] = {
    "motorway": 2,
    "trunk": 2,
    "primary": 2,
    "secondary": 2,
    "tertiary": 2,
    "unclassified": 2,  # geo.md says 1-2; Ontario concession roads are two-lane (ASSUMED)
    "residential": 2,
    "living_street": 1,
    "service": 1,
    "track": 1,
    "link": 1,
}

# Lane width (m) by region and class; see the module docstring for sources.
LANE_WIDTH_M: dict[str, dict[str, float]] = {
    "na": {
        "motorway": 3.65,
        "trunk": 3.65,
        "primary": 3.6,
        "secondary": 3.4,
        "tertiary": 3.3,
        "unclassified": 3.0,
        "residential": 3.0,
        "living_street": 3.5,
        "service": 3.5,
        "track": 3.0,
        "link": 4.0,
    },
    "intl": {
        "motorway": 3.75,
        "trunk": 3.5,
        "primary": 3.5,
        "secondary": 3.25,
        "tertiary": 3.0,
        "unclassified": 2.75,
        "residential": 2.75,
        "living_street": 3.5,
        "service": 3.5,
        "track": 3.0,
        "link": 4.0,
    },
}

_PAVED = {"asphalt", "paved", "concrete", "chipseal", "metal", "wood"}
_SETTS = {"paving_stones", "sett", "cobblestone", "unhewn_cobblestone", "bricks"}
_GRAVEL = {
    "compacted",
    "fine_gravel",
    "gravel",
    "pebblestone",
    "unpaved",
    "dirt",
    "ground",
    "earth",
    "grass",
    "mud",
    "sand",
    "rock",
}
_FEET_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:'|ft|feet|foot)\s*(?:(\d+(?:\.\d+)?)\s*(?:\"|in))?")
_ONEWAY_YES = {"yes", "true", "1"}
_ONEWAY_REVERSE = {"-1", "reverse"}


# Tags copied into route.json (geo.md §1.2 "Tags to keep", plus name/ref). Keys starting with
# one of ROUTE_TAG_PREFIXES are kept too.
ROUTE_TAG_KEYS = frozenset(
    "highway name ref oneway junction lanes width est_width surface smoothness tracktype bridge "
    "tunnel covered layer level cutting embankment maxspeed sidewalk shoulder barrier access "
    "motor_vehicle area incline man_made lit".split()
)
ROUTE_TAG_PREFIXES = ("lanes:", "width:", "shoulder:", "bridge:", "maxspeed:", "turn:lanes")


def route_tags(tags: Mapping[str, str]) -> dict[str, str]:
    """The subset of a way's tags worth keeping in route.json, sorted by key."""
    return {
        k: tags[k] for k in sorted(tags) if k in ROUTE_TAG_KEYS or k.startswith(ROUTE_TAG_PREFIXES)
    }


def is_bridge(tags: Mapping[str, str]) -> bool:
    return tags.get("bridge", "no") not in ("no", "") or tags.get("man_made") == "bridge"


def is_tunnel(tags: Mapping[str, str]) -> bool:
    return tags.get("tunnel", "no") not in ("no", "") or tags.get("covered") == "yes"


def layer_of(tags: Mapping[str, str]) -> int:
    """`layer` as an int (0 if missing or unparsable)."""
    try:
        return int(float(tags.get("layer", "0").split(";")[0]))
    except ValueError:
        return 0


def road_class(highway: str | None) -> str:
    """Normalised class key for the tables: `*_link` -> "link", unknown -> as given."""
    hw = (highway or "").strip()
    return "link" if hw.endswith("_link") else hw


def parse_width(value: str | None) -> float | None:
    """Parse an OSM width value to metres, or None if unusable.

    Accepts "6", "6.5 m", "6,5", "20'", "20 ft", "12'6\"", "6;7" (first value), "~6".
    """
    if value is None:
        return None
    text = str(value).split(";")[0].strip().lower().lstrip("~").strip()
    if not text:
        return None
    feet = _FEET_RE.fullmatch(text)
    if feet:
        inches = float(feet.group(2) or 0.0)
        return round((float(feet.group(1)) * 12.0 + inches) * 0.0254, 4)
    inch = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:\"|in)", text)
    if inch:
        return round(float(inch.group(1)) * 0.0254, 4)
    metres = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*(?:m|metres?|meters?)?", text)
    if metres:
        width = float(metres.group(1).replace(",", "."))
        return width if width > 0.0 else None
    return None


def region_for(lat: float, lon: float) -> str:
    """Lane-width region: "na" for North America and the Caribbean, "intl" elsewhere."""
    return "na" if 7.0 <= lat <= 84.0 and -170.0 <= lon <= -50.0 else "intl"


def lane_width(highway: str | None, region: str = "intl") -> float:
    table = LANE_WIDTH_M.get(region, LANE_WIDTH_M["intl"])
    return table.get(road_class(highway), 3.0)


def is_oneway(tags: Mapping[str, str]) -> int:
    """+1 one-way along the way, -1 against it, 0 two-way.

    Implied one-ways: `junction=roundabout|circular` and `highway=motorway` (OSM convention).
    """
    value = tags.get("oneway", "").strip().lower()
    if value in _ONEWAY_YES:
        return 1
    if value in _ONEWAY_REVERSE:
        return -1
    if value == "no":
        return 0
    if tags.get("junction") in ("roundabout", "circular") or tags.get("highway") == "motorway":
        return 1
    return 0


def lane_count(tags: Mapping[str, str]) -> int:
    """Total lanes (both directions): the `lanes` tag, else the class default (1 if one-way)."""
    raw = tags.get("lanes", "").split(";")[0].strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    cls = road_class(tags.get("highway"))
    default = DEFAULT_LANES.get(cls, 1)
    if is_oneway(tags) and cls not in ("motorway", "trunk"):
        return 1
    return default


def road_width(
    tags: Mapping[str, str],
    region: str = "intl",
    min_width: float = 0.0,
    max_width: float = MAX_WIDTH_M,
) -> float:
    """Carriageway width in metres from the tag cascade, clamped to [min_width, max_width]."""
    width = None
    for key in ("width:carriageway", "width", "est_width"):
        width = parse_width(tags.get(key))
        if width is not None:
            break
    if width is None:
        raw = tags.get("lanes", "").split(";")[0].strip()
        if raw.isdigit() and int(raw) > 0:
            width = int(raw) * lane_width(tags.get("highway"), region)
    if width is None:
        width = DEFAULT_WIDTH_M.get(road_class(tags.get("highway")), _FALLBACK_WIDTH_M)
    return float(min(max(width, min_width), max_width))


def surface_class(tags: Mapping[str, str]) -> str:
    """ "asphalt", "setts" or "gravel" (geo.md §1.5). Missing: gravel for tracks, else asphalt."""
    surface = tags.get("surface", "").strip().lower()
    if surface in _PAVED or surface.startswith("concrete"):
        return "asphalt"
    if surface in _SETTS:
        return "setts"
    if surface in _GRAVEL:
        return "gravel"
    return "gravel" if tags.get("highway") == "track" else "asphalt"


# (south, west, north, east) degrees. Rough boxes of left-hand-traffic countries, ASSUMED.
# Known border misses are accepted: e.g. Kuril Is. read as left; Isaan (NE Thailand), western
# Uganda/Tanzania, Sabah and the Channel Islands read as right.
_LEFT_HAND_BOXES: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    # Japan, split to stay clear of Korea (Busan, Ulleungdo), Taiwan and Sakhalin.
    ("JP", (24.0, 122.9, 30.0, 131.4)),  # Ryukyu
    ("JP", (30.0, 128.4, 34.0, 132.2)),  # Kyushu
    ("JP", (34.0, 129.1, 34.75, 129.55)),  # Tsushima
    ("JP", (33.7, 130.75, 35.8, 132.2)),  # western Chugoku
    ("JP", (30.0, 132.2, 35.8, 142.0)),  # Shikoku, Kansai, Tokai, Kanto, Izu
    ("JP", (35.8, 132.5, 41.6, 142.2)),  # northern Honshu
    ("JP", (41.3, 139.3, 45.6, 145.9)),  # Hokkaido
    # Great Britain + Ireland; the southern box stops at 1.45 E to leave out Calais/Boulogne.
    ("GB", (49.85, -6.5, 51.42, 1.45)),
    ("GB/IE", (51.42, -10.7, 61.0, 1.8)),
    ("MT", (35.78, 14.18, 36.09, 14.58)),
    ("CY", (34.55, 32.25, 35.72, 34.6)),
    ("AU", (-44.0, 112.0, -10.0, 154.0)),
    ("NZ", (-47.5, 166.0, -34.0, 179.0)),
    ("ID/TL/MY-Borneo", (-11.0, 95.0, 4.0, 141.0)),
    ("PG", (-11.7, 141.0, -1.0, 155.5)),
    ("MY/SG", (1.1, 99.6, 6.5, 104.6)),
    ("TH", (12.5, 99.0, 17.5, 102.0)),
    ("TH", (17.5, 98.0, 20.5, 100.0)),
    ("TH", (5.6, 98.2, 12.5, 102.1)),
    ("HK", (22.15, 113.83, 22.5, 114.45)),
    ("MO", (22.1, 113.52, 22.22, 113.6)),
    ("IN/PK/BD/LK", (5.5, 67.0, 26.0, 92.3)),
    ("IN/NP", (26.0, 69.5, 32.5, 88.2)),
    ("Southern Africa", (-35.0, 11.5, -17.5, 41.0)),
    ("East Africa", (-17.2, 30.5, 4.3, 41.9)),
    ("JM", (17.6, -78.5, 18.6, -76.1)),
    ("BS", (23.4, -79.4, 27.3, -74.0)),
    ("BS/TC", (20.9, -74.4, 23.4, -71.0)),
    ("TT", (10.0, -61.95, 11.4, -60.5)),
    ("BB", (13.0, -59.7, 13.4, -59.4)),
    ("GY/SR", (1.8, -60.2, 8.6, -54.0)),
)


def driving_side(lat: float, lon: float) -> str:
    """ "left" or "right" for a location (rough bounding boxes, ASSUMED)."""
    for _, (south, west, north, east) in _LEFT_HAND_BOXES:
        if south <= lat <= north and west <= lon <= east:
            return "left"
    return "right"
