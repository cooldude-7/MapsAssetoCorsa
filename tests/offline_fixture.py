"""A small OSM region for the offline-database tests, made from the committed Overpass fixtures.

`write_region(dir)` writes `region.osm` (OSM XML: the Forks of the Credit roads, the Snake Road
buildings and the Mill St barriers, plus a made-up lake relation, a pond, a river, a cliff and a
driveway near Mill St) and `region.poly` (a box round all of it). The roads in the Overpass
fixtures are complete ways, so the file is an extract of the same OSM data (ODbL 1.0, (c)
OpenStreetMap contributors; tests/fixtures/README.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from driftmaps.geo.project import LocalProjection

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "osm"
ROADS = FIXTURES / "forks_of_the_credit.json"
BUILDINGS = FIXTURES / "buildings_snake_rd.json"
BARRIERS = FIXTURES / "mill_st_waterdown_barriers.json"
# (south, west, north, east) of the region: everything above, nothing of Toronto.
BOX = (43.25, -80.05, 43.90, -79.85)
POLY = f"""region
1
   {BOX[1]} {BOX[0]}
   {BOX[3]} {BOX[0]}
   {BOX[3]} {BOX[2]}
   {BOX[1]} {BOX[2]}
   {BOX[1]} {BOX[0]}
END
END
"""
# Made-up roadside features beside Mill St, Waterdown: ids far above real ones.
LAKE_REL = 9_000_000_001
OSM_BASE = "2026-10-15T13:51:13Z"


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["elements"]


def _mill_frame():
    """(anchor (lat, lon), at(along_m, left_m) -> (lat, lon)) on Mill St's bridge way."""
    g = next(el for el in _load(BARRIERS) if el["id"] == 24486613)["geometry"]
    a, b = g[0], g[-1]
    proj = LocalProjection((a["lat"] + b["lat"]) / 2, (a["lon"] + b["lon"]) / 2)
    d = proj.to_xy([b["lon"], b["lat"]]) - proj.to_xy([a["lon"], a["lat"]])
    t = d / np.hypot(*d)
    left = np.array([-t[1], t[0]])

    def at(along: float, side: float) -> tuple[float, float]:
        lon, lat = proj.from_xy(t * along + left * side).tolist()
        return round(lat, 7), round(lon, 7)

    return at(0.0, 0.0), at


MILL_ST, _at = _mill_frame()


def _square(nid0: int, along: float, side: float, half: float) -> tuple[list[int], dict]:
    pts = [(along - half, side - half), (along + half, side - half),
           (along + half, side + half), (along - half, side + half)]  # fmt: skip
    nodes = {nid0 + k: _at(*p) for k, p in enumerate(pts)}
    return [*nodes, nid0], nodes


def region_elements() -> tuple[dict[int, tuple[float, float]], list[dict], list[dict]]:
    """(nodes {id: (lat, lon)}, ways [{id, nodes, tags}], relations [{id, members, tags}])."""
    nodes: dict[int, tuple[float, float]] = {}
    ways: list[dict] = []
    for path in (ROADS, BUILDINGS, BARRIERS):
        for el in _load(path):
            if el["type"] != "way":
                continue
            for nid, g in zip(el["nodes"], el["geometry"], strict=True):
                nodes[nid] = (g["lat"], g["lon"])
            ways.append({"id": el["id"], "nodes": el["nodes"], "tags": el.get("tags", {})})
    nid = 9_100_000_000
    # A lake (multipolygon relation, untagged member way) 20 m left of the road.
    outer, pts = _square(nid, 0.0, 30.0, 10.0)
    nodes.update(pts)
    ways.append({"id": 9_200_000_001, "nodes": outer, "tags": {}})
    # A pond way (natural=water) and a cliff, both closed, ~20 m and ~15 m to the right.
    pond, pts = _square(nid + 10, 40.0, -25.0, 5.0)
    nodes.update(pts)
    ways.append({"id": 9_200_000_002, "nodes": pond, "tags": {"natural": "water"}})
    cliff, pts = _square(nid + 20, -40.0, -20.0, 5.0)
    nodes.update(pts)
    ways.append({"id": 9_200_000_003, "nodes": cliff, "tags": {"natural": "cliff"}})
    # A river crossing under the road and a driveway leaving it to the left.
    nodes[nid + 30], nodes[nid + 31] = _at(60.0, -40.0), _at(60.0, 40.0)
    ways.append(
        {"id": 9_200_000_004, "nodes": [nid + 30, nid + 31],
         "tags": {"waterway": "river", "name": "Test Creek"}}
    )  # fmt: skip
    nodes[nid + 40], nodes[nid + 41] = _at(-20.0, 4.0), _at(-20.0, 20.0)
    ways.append(
        {"id": 9_200_000_005, "nodes": [nid + 40, nid + 41],
         "tags": {"highway": "service", "service": "driveway"}}
    )  # fmt: skip
    relations = [
        {"id": LAKE_REL, "members": [("w", 9_200_000_001, "outer")],
         "tags": {"type": "multipolygon", "natural": "water", "name": "Test Lake"}},
    ]  # fmt: skip
    # Street furniture (FORMAT 3): tagged nodes beside the road, a power line on two poles.
    for k, (along, side, _) in enumerate(STREET_NODES):
        nodes[STREET_NODE0 + k] = _at(along, side)
    nodes[STREET_NODE0 + 10], nodes[STREET_NODE0 + 11] = _at(-30.0, -8.0), _at(30.0, -8.0)
    ways.append(
        {"id": POWER_LINE, "nodes": [STREET_NODE0 + 10, STREET_NODE0 + 11],
         "tags": {"power": "minor_line"}}
    )  # fmt: skip
    return nodes, ways, relations


# (along, left, tags) of the made-up furniture nodes near Mill St, from STREET_NODE0 on; the last
# one (a bench) is not furniture the builder reads and must not be stored.
STREET_NODE0 = 9_300_000_000
POWER_LINE = 9_200_000_010
STREET_NODES = (
    (-15.0, -4.0, {"highway": "stop", "direction": "forward"}),
    (5.0, 6.0, {"highway": "street_lamp"}),
    (10.0, -8.0, {"power": "pole"}),
    (12.0, 5.0, {"emergency": "fire_hydrant"}),
    (20.0, -5.0, {"traffic_sign": "CA:rb-1", "maxspeed": "50"}),
    (25.0, 5.0, {"amenity": "bench"}),
)


def node_tags() -> dict[int, dict[str, str]]:
    return {STREET_NODE0 + k: tags for k, (_, _, tags) in enumerate(STREET_NODES)}


def osm_xml() -> str:
    nodes, ways, relations = region_elements()
    lines = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        f'<osm version="0.6" generator="driftmaps tests" timestamp="{OSM_BASE}">',
    ]
    tagged = node_tags()
    for nid in sorted(nodes):
        lat, lon = nodes[nid]
        head = f'  <node id="{nid}" version="1" lat="{lat:.7f}" lon="{lon:.7f}"'
        if nid not in tagged:
            lines.append(head + "/>")
            continue
        lines.append(head + ">")
        lines += [f"    <tag k={quoteattr(k)} v={quoteattr(v)}/>" for k, v in tagged[nid].items()]
        lines.append("  </node>")
    for w in sorted(ways, key=lambda w: w["id"]):
        lines.append(f'  <way id="{w["id"]}" version="1">')
        lines += [f'    <nd ref="{n}"/>' for n in w["nodes"]]
        lines += [f"    <tag k={quoteattr(k)} v={quoteattr(v)}/>" for k, v in w["tags"].items()]
        lines.append("  </way>")
    for r in relations:
        lines.append(f'  <relation id="{r["id"]}" version="1">')
        lines += [
            f'    <member type="{"way" if t == "w" else t}" ref="{ref}" role="{role}"/>'
            for t, ref, role in r["members"]
        ]
        lines += [f"    <tag k={quoteattr(k)} v={quoteattr(v)}/>" for k, v in r["tags"].items()]
        lines.append("  </relation>")
    lines.append("</osm>")
    return "\n".join(lines) + "\n"


def write_region(folder: Path) -> tuple[Path, Path]:
    """Write region.osm and region.poly into `folder`; returns their paths."""
    folder.mkdir(parents=True, exist_ok=True)
    osm = folder / "region.osm"
    osm.write_text(osm_xml(), encoding="utf-8")
    poly = folder / "region.poly"
    poly.write_text(POLY, encoding="utf-8")
    return osm, poly
