"""Offline checks for the map page in src/driftmaps/app/static.

The page is plain HTML/CSS/JS with no build step, so these tests check what a build step would:
the files exist, nothing loads from an unexpected host, every element the script looks up exists,
the script parses, and its pure helpers (run under Node when it is installed) give the right
answers. Nothing here touches the network.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "driftmaps" / "app" / "static"
FILES = ("index.html", "app.js", "style.css", "favicon.svg")

# Hosts the page may load from or link to. Anything else is a mistake (or a privacy leak).
ALLOWED_HOSTS = {
    "unpkg.com",  # MapLibre GL JS, pinned version
    "tiles.openfreemap.org",  # map styles and tiles
    "photon.komoot.io",  # place search
    "www.openstreetmap.org",  # the attribution link in the footer
    "www.w3.org",  # the SVG XML namespace, not a request
}
URL_RE = re.compile(r"""(?:https?:)?//[A-Za-z0-9.-]+\.[A-Za-z]{2,}[^\s"'`<>)]*""")
API_PATHS = (
    "/api/status",
    "/api/settings",
    "/api/route",
    "/api/roads",
    "/api/route-path",
    "/api/profile",
    "/api/build",
    "/api/jobs/",
    "/api/tracks",
)

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.scripts: list[dict[str, str | None]] = []
        self.links: list[dict[str, str | None]] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if a.get("id"):
            self.ids.append(a["id"] or "")
        if tag == "script":
            self.scripts.append(a)
        elif tag == "link":
            self.links.append(a)
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data


def parse_index() -> _Collector:
    c = _Collector()
    c.feed(read("index.html"))
    return c


def test_files_exist_and_are_not_empty():
    for name in FILES:
        path = STATIC / name
        assert path.is_file(), f"missing {path}"
        assert path.stat().st_size > 100, f"{name} looks empty"


def test_only_allowed_external_hosts():
    for name in FILES:
        for url in URL_RE.findall(read(name)):
            host = urlparse(url if url.startswith("http") else "https:" + url).hostname
            assert host in ALLOWED_HOSTS, f"{name} references {url}"


def test_maplibre_is_pinned_to_one_exact_version():
    html = read("index.html")
    pattern = r"https://unpkg\.com/maplibre-gl@([^/\"']+)/dist/maplibre-gl\.(?:js|css)"
    versions = re.findall(pattern, html)
    assert len(versions) == 2, "expected the MapLibre script and stylesheet"
    assert len(set(versions)) == 1
    assert re.fullmatch(r"\d+\.\d+\.\d+", versions[0]), f"not an exact version: {versions[0]}"


def test_index_wires_up_local_assets_and_required_text():
    c = parse_index()
    html = read("index.html")
    assert c.title.strip() == "Assetto Corsa Map Generator"
    assert any(s.get("src") == "app.js" for s in c.scripts)
    assert any(s.get("href") == "style.css" for s in c.links)
    assert any(s.get("href") == "favicon.svg" and s.get("rel") == "icon" for s in c.links)
    assert 'name="viewport"' in html
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html))
    assert "Road data © OpenStreetMap contributors." in text
    assert "Not affiliated with Kunos Simulazioni." in text


def test_html_ids_are_unique_and_cover_every_id_the_script_uses():
    ids = parse_index().ids
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate ids: {sorted(dupes)}"
    used = set(re.findall(r"""\$\(\s*["']([\w-]+)["']\s*\)""", read("app.js")))
    used |= set(re.findall(r"""container:\s*["']([\w-]+)["']""", read("app.js")))
    assert used, "app.js should look elements up with $(id)"
    missing = used - set(ids)
    assert not missing, f"app.js uses ids that index.html lacks: {sorted(missing)}"


def test_script_follows_the_api_contract_and_map_sources():
    js = read("app.js")
    for path in API_PATHS:
        assert f'"{path}"' in js, f"app.js never calls {path}"
    assert "https://tiles.openfreemap.org/styles/liberty" in js
    assert "https://tiles.openfreemap.org/styles/dark" in js
    assert "https://photon.komoot.io/api/" in js
    for field in (
        "route_id", "track_id", "install", "scenery_density", "reverse", "loop", "ac_root",
        "alternatives", "snapped", "far_pin", "tiles", "path", "start_t", "end_t",
        "auto_guardrails", "build",
    ):  # fmt: skip
        assert re.search(rf"\b{field}\b", js), f"app.js never sends/reads {field}"


def test_markers_keep_maplibre_classes():
    # Replacing a marker element's className drops MapLibre's `maplibregl-marker` class (its
    # position: absolute), and every point after A is then drawn a pin's height too low.
    assert not re.search(r"\bpin\.el\.className\s*=", read("app.js"))


def test_local_storage_is_always_guarded():
    lines = [ln for ln in read("app.js").splitlines() if "localStorage" in ln]
    assert lines, "expected the guarded store helper"
    for ln in lines:
        assert "try" in ln and "catch" in ln, f"unguarded localStorage access: {ln.strip()}"


def test_stylesheet_has_dark_and_light_tokens():
    css = read("style.css")
    assert ":root" in css and "--accent" in css
    assert "prefers-color-scheme: light" in css
    assert css.count("{") == css.count("}"), "unbalanced braces in style.css"


@needs_node
def test_script_parses():
    proc = subprocess.run(
        [NODE, "--check", str(STATIC / "app.js")], capture_output=True, text=True, encoding="utf-8"
    )
    assert proc.returncode == 0, proc.stderr


HELPER_PROBE = r"""
const ELL = [[0, 0], [0.01, 0], [0.01, 0.01]];
const out = {
  len: [formatLength(850), formatLength(3421), formatLength(12345), formatLength(NaN)],
  names: [
    defaultTrackName(["Snake Road"]), defaultTrackName([]), defaultTrackName(["A Rd", "B Rd"]),
    defaultTrackName(["A Rd", "B Rd", "C Rd"]), defaultTrackName([null, " Kerns Road "]),
  ],
  insertMid: bestInsertIndex([[0, 0], [0, 1], [0, 2]], [0.01, 1.5]),
  insertFirst: bestInsertIndex([[0, 0], [0, 1], [0, 2]], [0.01, 0.4]),
  insertShort: bestInsertIndex([[0, 0]], [0, 1]),
  // On a loop the gap from the last point back to A counts too (index n: insert at the end).
  insertLoopClose: bestInsertIndex([[0, 0], [0, 1], [1, 1]], [0.8, 0.5], true),
  insertLoopOpen: bestInsertIndex([[0, 0], [0, 1], [1, 1]], [0.8, 0.5], false),
  pins: apiPins([[-79.8, 43.2], [-79.75, 43.25], [-79.7, 43.3]], false),
  pinsRev: apiPins([[-79.8, 43.2], [-79.75, 43.25], [-79.7, 43.3]], true),
  pinsLoopRev: apiPins([[-79.8, 43.2], [-79.75, 43.25], [-79.7, 43.3]], true, true),
  reversedLoop: reversedPins(["A", "B", "C", "D"], true),
  reversedOpen: reversedPins(["A", "B", "C", "D"], false),
  letters: [0, 1, 2, 25, 26, 27, 49].map(pinLetter),
  roles: [[0, 3, false], [1, 3, false], [2, 3, false], [0, 3, true], [2, 3, true]].map(
    ([i, n, loop]) => pinRole(i, n, loop)),
  loopNames: [defaultTrackName(["Mill St"], true),
    defaultTrackName(["Mill St", "Main St", "X"], true), defaultTrackName([], true)],
  degree: distanceM([0, 0], [0, 1]),
  along: pointAlong(
    [[0, 0], [0, 1], [0, 2]],
    cumulativeM([[0, 0], [0, 1], [0, 2]]),
    distanceM([0, 0], [0, 1]) * 1.5,
  ),
  alongEnd: pointAlong([[0, 0], [0, 1]], cumulativeM([[0, 0], [0, 1]]), 1e9),
  ids: [sanitizeTrackId("Snake Road!"), sanitizeTrackId("  __Kerns--Rd__ "), sanitizeTrackId("")],
  coords: [
    parseLatLon("43.25, -79.87"), parseLatLon("43.25 -79.87"), parseLatLon("hamilton"),
    parseLatLon("95, 10"),
  ],
  density: [densityLabel(0), densityLabel(0.3), densityLabel(1), densityLabel(2)],
  step: [niceStep(87, 3), niceStep(4200, 5)],
  stats: profileStats([0, 20, 40, 60, 80, 100], [100, 102, 104, 103, 110, 110]),
  nearest: [
    nearestIndex([0, 20, 40, 60], 29),
    nearestIndex([0, 20, 40, 60], 31),
    nearestIndex([0, 20], 500),
  ],
  label: photonLabel({ name: "Snake Road", city: "Hamilton", state: "Ontario", country: "Canada" }),
  labelStreet: photonLabel({ housenumber: "12", street: "King St", city: "Hamilton" }),
  // Snapped points come back in request order: flip them back like the points were.
  snapOpen: displaySnapped([[1, 10], [2, 20], [3, 30]], 3, false, false),
  snapRev: displaySnapped([[1, 10], [2, 20], [3, 30]], 3, true, false),
  snapLoopRev: displaySnapped([[1, 10], [2, 20], [3, 30]], 3, true, true),
  snapBad: [displaySnapped([[1, 10]], 2, false, false), displaySnapped(null, 0, false, false),
    displaySnapped([[1, 10], [95, 0]], 2, false, false)],
  farIdx: [[0, 3, false, false], [0, 3, true, false], [2, 3, true, false], [1, 4, true, true],
    [0, 4, true, true]].map(([i, n, rev, loop]) => displayIndex(i, n, rev, loop)),
  optLabels: [
    routeOptionLabel({ length_m: 3512, names: ["Snake Road"] }, 0),
    routeOptionLabel({ length_m: 4120, names: ["Waterdown Road", "Mill Street"] }, 1),
    routeOptionLabel({ length_m: 900, names: ["A", "B", "C", "D"] }, 2),
    routeOptionLabel({ length_m: 900, names: [] }, 3),
  ],
  // A line east then north; points A (start), B (corner), C (end).
  onLine: nearestOnLine(ELL, cumulativeM(ELL), [0.005, 0.001]),
  insAB: insertIndexAlong(ELL, ELL, [0.004, 0.0003], false),
  insBC: insertIndexAlong(ELL, ELL, [0.0102, 0.006], false),
  // A loop round a square from A at the south-west corner, B at the south-east, C at the
  // north-east: a drag on the west side (after C, back to A) goes at the end.
  insLoopEnd: insertIndexAlong([[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01], [0, 0]],
    [[0, 0], [0.01, 0], [0.01, 0.01]], [-0.0002, 0.004], true),
  insLoopMid: insertIndexAlong([[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01], [0, 0]],
    [[0, 0], [0.01, 0], [0.01, 0.01]], [0.006, 0.0002], true),
};
process.stdout.write(JSON.stringify(out));
"""


@needs_node
def test_pure_helpers_under_node(tmp_path: Path):
    # app.js only boots when `document` exists, so under Node it just defines its helpers.
    script = tmp_path / "probe.js"
    script.write_text(read("app.js") + "\n;" + HELPER_PROBE, encoding="utf-8")
    proc = subprocess.run(
        [NODE, str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)

    assert out["len"] == ["850 m", "3.42 km", "12.3 km", "–"]
    assert out["names"] == ["Snake Road", "", "A Rd & B Rd", "A Rd to C Rd", "Kerns Road"]
    assert out["insertMid"] == 2
    assert out["insertFirst"] == 1
    assert out["insertShort"] == 1
    assert out["pins"] == [[43.2, -79.8], [43.25, -79.75], [43.3, -79.7]]
    assert out["pinsRev"] == [[43.3, -79.7], [43.25, -79.75], [43.2, -79.8]]
    # Loops: A stays the start whichever way round; letters in click order.
    assert out["insertLoopClose"] == 3 and out["insertLoopOpen"] == 2
    assert out["pinsLoopRev"] == [[43.2, -79.8], [43.3, -79.7], [43.25, -79.75]]
    assert out["reversedLoop"] == ["A", "D", "C", "B"]
    assert out["reversedOpen"] == ["D", "C", "B", "A"]
    assert out["letters"] == ["A", "B", "C", "Z", "AA", "AB", "AX"]
    assert out["roles"] == ["start", "via", "finish", "start", "via"]
    assert out["loopNames"] == ["Mill St Loop", "Mill St & Main St Loop", ""]
    assert out["degree"] == pytest.approx(111195, rel=1e-3)
    assert out["along"] == pytest.approx([0, 1.5], abs=1e-9)
    assert out["alongEnd"] == [0, 1]
    assert out["ids"] == ["snake_road", "kerns_rd", ""]
    assert out["coords"] == [[-79.87, 43.25], [-79.87, 43.25], None, None]
    assert out["density"] == ["None", "Light", "Normal", "Dense"]
    assert out["step"] == [50, 1000]
    st = out["stats"]
    assert st["climb"] == pytest.approx(11)
    assert st["drop"] == pytest.approx(1)
    assert (st["minZ"], st["maxZ"]) == (100, 110)
    assert st["maxGrade"] == pytest.approx(8 / 60)
    assert out["nearest"] == [1, 2, 1]
    assert out["label"] == {"title": "Snake Road", "subtitle": "Hamilton, Ontario, Canada"}
    assert out["labelStreet"] == {"title": "12 King St", "subtitle": "Hamilton"}
    assert out["snapOpen"] == [[10, 1], [20, 2], [30, 3]]
    assert out["snapRev"] == [[30, 3], [20, 2], [10, 1]]
    assert out["snapLoopRev"] == [[10, 1], [30, 3], [20, 2]]
    assert out["snapBad"] == [None, None, None]
    assert out["farIdx"] == [0, 2, 0, 3, 0]
    assert out["optLabels"] == [
        "1 · 3.51 km · Snake Road",
        "2 · 4.12 km · Waterdown Road, Mill Street",
        "3 · 900 m · A, B, C, …",
        "4 · 900 m",
    ]
    on = out["onLine"]
    assert on["d"] == pytest.approx(556, rel=0.01) and on["dist"] == pytest.approx(110.5, rel=0.01)
    assert on["point"] == pytest.approx([0.005, 0], abs=1e-9)
    assert (out["insAB"], out["insBC"]) == (1, 2)
    assert (out["insLoopEnd"], out["insLoopMid"]) == (3, 1)


# A small road network (in /api/roads' shape) round (0, 0); 0.001 degrees is about 111 m.
#
#   10 ------------- 7            Loop Rd: 7 - 10 - 1 (back to Main St's west end)
#   |                |            North Rd: 3 - 6 - 7
#   |                6     9      One Way: 9 -> 4 only (oneway 1)
#   |                |     |      a driveway (unnamed service road) off node 2
#   1 ----- 2 ------ 3 --- 4 --- 5   Main St
#           |
#           8 (driveway)
ROADS = {
    "tiles": {"14/8192/8191": "ok"},
    "nodes": {
        "1": [-0.002, 0], "2": [-0.001, 0], "3": [0, 0], "4": [0.001, 0], "5": [0.002, 0],
        "6": [0, 0.001], "7": [0, 0.002], "8": [-0.001, -0.0005], "9": [0.001, 0.001],
        "10": [-0.002, 0.002],
    },
    "ways": [
        {"id": 1, "name": "Main St", "ref": "", "highway": "residential", "oneway": 0,
         "nodes": [1, 2, 3, 4, 5]},
        {"id": 2, "name": "North Rd", "ref": "", "highway": "residential", "oneway": 0,
         "nodes": [3, 6, 7]},
        {"id": 3, "name": "", "ref": "", "highway": "service", "oneway": 0, "nodes": [2, 8]},
        {"id": 4, "name": "One Way", "ref": "", "highway": "residential", "oneway": 1,
         "nodes": [9, 4]},
        {"id": 5, "name": "Loop Rd", "ref": "", "highway": "tertiary", "oneway": 0,
         "nodes": [7, 10, 1]},
    ],
    "error": None,
}  # fmt: skip

ROADS_PROBE = r"""
const g = createRoadGraph();
const added = addRoads(g, ROADS);
const again = addRoads(g, ROADS); // the same ways twice: added once
const all = () => true;
const br = (path) =>
  branchesAt(g, path, all).map((b) => [b.name, Math.round(b.angle), b.allowed, b.run]);
const east = branchesAt(g, ["1", "2", "3"], all);
const at4 = branchesAt(g, ["2", "3", "4"], all);
const keys = keyBranches(east);
const ahead = findAhead(g, ["1", "2", "3"], { a: "6", b: "7" }, all);
const ahead2 = findAhead(g, ["1", "2", "3"], { a: "5", b: "4" }, all);
const notAhead = findAhead(g, ["1", "2", "3"], { a: "2", b: "8" }, all);
// A grid of 120 x 120 nodes (about 28 000 segments): snapping stays fast.
const big = createRoadGraph();
const nodes = {};
const ways = [];
const N = 120;
for (let i = 0; i < N; i++) {
  for (let j = 0; j < N; j++) nodes[`${i}_${j}`] = [-79.9 + i * 0.0005, 43.3 + j * 0.0005];
}
const ids = (f) => [...Array(N).keys()].map(f);
for (let i = 0; i < N; i++) {
  ways.push({ id: `r${i}`, name: `R${i}`, highway: "residential", nodes: ids((j) => `${i}_${j}`) });
  ways.push({ id: `c${i}`, name: `C${i}`, highway: "residential", nodes: ids((j) => `${j}_${i}`) });
}
let t0 = Date.now();
addRoads(big, { nodes, ways });
const loadMs = Date.now() - t0;
t0 = Date.now();
let hits = 0;
for (let k = 0; k < 5000; k++) {
  const p = [-79.9 + Math.random() * 0.06, 43.3 + Math.random() * 0.06];
  if (nearestSegment(big, p, 25)) hits++;
}
const snapMs = Date.now() - t0;
const out = {
  added, again, segments: g.segments,
  // Driving east from 1: the driveway at 2 is passed, the run stops at the junction at 3.
  run12: runFrom(g, "1", "2", all),
  runAround: runAround(g, "2", "3"),
  through2: !!throughEdge(g, "1", "2"), through3: !!throughEdge(g, "2", "3"),
  driveway: !!throughEdge(g, "8", "2"),
  east: br(["1", "2", "3"]),
  at4: br(["2", "3", "4"]),
  keys: [keys.left, keys.straight, keys.right].map((b) => b && b.name),
  labels: branchKeyLabels(at4), labelsEast: branchKeyLabels(east),
  byKey: ["ArrowLeft", "ArrowUp", "ArrowRight", "1", "2", "3"].map((k) => {
    const b = branchForKey(east, k);
    return b && b.name;
  }),
  ahead: ahead && [ahead.nodes, ahead.cut, ahead.forward, ahead.branch.name],
  ahead2: ahead2 && [ahead2.nodes, ahead2.cut, ahead2.forward],
  notAhead,
  // Back to start: round the loop to A's road (1 -> 2) again.
  loop: closeLoopPath(g, ["1", "2", "3"], all),
  loopFromRing: closeLoopPath(g, ["1", "2", "3", "6", "7"], all),
  noLoop: closeLoopPath(g, ["1", "2", "3", "4"], all),
  wrongWay: closeLoopPath(g, ["2", "1", "10", "7"], all),
  nearest: (() => {
    const s = nearestSegment(g, [-0.0015, 0.00005], 30);
    return s && [s.a, s.b, s.w, Math.round(s.t * 100) / 100, Math.round(s.dist)];
  })(),
  far: nearestSegment(g, [-0.0015, 0.001], 30),
  line: pathLine(g, ["1", "2", "3"], 0.5, 0.5),
  lineLen: Math.round(lineLengthM(pathLine(g, ["1", "2", "3"], 0.5, 0.5))),
  head: lineHead([[0, 0], [0.001, 0], [0.002, 0]], 150)
    .map((p) => p.map((x) => Math.round(x * 1e6) / 1e6)),
  labelsLeg: [
    legLabel({ kind: "start", name: "Snake Road" }),
    legLabel({ kind: "turn", name: "Mountain Brow Road", angle: -88 }),
    legLabel({ kind: "turn", name: "Snake Road", angle: 3 }, "Snake Road"),
    legLabel({ kind: "turn", name: "Snake Road", angle: 3 }, "Old York Road"),
    legLabel({ kind: "turn", name: "", angle: 40 }),
    legLabel({ kind: "turn", name: "Kerns Road", angle: 150 }),
    legLabel({ kind: "close", name: "Snake Road" }),
  ],
  build: [
    buildLabel({ version: "3", commit: "abcdef1234567", date: "2026-10-15T21:58:39Z" }),
    buildLabel({ version: "Update 4", date: "2026-10-16" }),
    buildLabel({ version: "dev" }), buildLabel(null), buildLabel({}),
  ],
  tile: tileKeyAt(-79.885, 43.31),
  tileBox: tileBounds("14/4556/6000"),
  tilesView: tilesForBounds(-79.89, 43.305, -79.88, 43.315, 1),
  spread: spreadAngles([0, 10, 180], 40).map(Math.round),
  spreadWrap: spreadAngles([355, 5], 40).map(Math.round),
  spreadFine: spreadAngles([0, 90, 180, 270], 40),
  compass: [0, 44, 90, 200, 315].map(compassWord),
  keyFor: [10, 80, 170, 280, 350].map(keyForBearing),
  optKey: [
    optionForKey([45, 225], "ArrowUp"), optionForKey([45, 225], "ArrowDown"),
    optionForKey([45, 225], "ArrowLeft"), optionForKey([90, 270], "ArrowUp"),
    optionForKey([null, 225], "ArrowUp"), optionForKey([0, 180], "Enter"),
  ],
  big: { segments: big.segments, loadMs, snapMs, hits },
};
process.stdout.write(JSON.stringify(out));
"""


@needs_node
def test_turn_by_turn_helpers_under_node(tmp_path: Path):
    script = tmp_path / "roads_probe.js"
    script.write_text(
        read("app.js") + f"\n;const ROADS = {json.dumps(ROADS)};\n" + ROADS_PROBE,
        encoding="utf-8",
    )
    proc = subprocess.run(
        [NODE, str(script)], capture_output=True, text=True, encoding="utf-8", timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)

    assert out["added"] == 10 and out["again"] == 0 and out["segments"] == 10
    assert out["run12"] == ["2", "3"]
    # The stretch round 2-3: the whole ring, as 3 is its only junction (7 and 1 are plain road).
    assert out["runAround"] == ["3", "6", "7", "10", "1", "2", "3"]
    assert (out["through2"], out["through3"], out["driveway"]) == (True, False, False)
    # Heading east at 3: North Rd on the left (round the ring back to 3), Main St straight on
    # (no U-turn back to 2).
    ring = ["6", "7", "10", "1", "2", "3"]
    assert out["east"] == [["North Rd", -90, True, ring], ["Main St", 0, True, ["4"]]]
    # At 4 One Way comes in from the north: you may not drive up it.
    assert out["at4"] == [["One Way", -90, False, ["9"]], ["Main St", 0, True, ["5"]]]
    assert out["keys"] == ["North Rd", "Main St", None]
    assert out["labels"] == ["", "↑"] and out["labelsEast"] == ["←", "↑"]
    assert out["byKey"] == ["North Rd", "Main St", None, "North Rd", "Main St", None]
    # Clicking further along: up North Rd; along Main St through the junction at 4.
    assert out["ahead"] == [ring, 1, True, "North Rd"]
    assert out["ahead2"] == [["4", "5"], 1, False]
    assert out["notAhead"] is None
    assert out["loop"] == ["1", "2", "3", "6", "7", "10", "1"]
    assert out["loopFromRing"] == ["1", "2", "3", "6", "7", "10", "1"]
    assert out["noLoop"] is None
    assert out["wrongWay"] is None  # back on A's road but driving it the other way
    assert out["nearest"] == ["1", "2", "1", 0.5, 6]
    assert out["far"] is None
    assert out["line"] == [[-0.0015, 0], [-0.001, 0], [-0.0005, 0]]
    assert out["lineLen"] == 111
    assert out["head"] == [[0, 0], [0.001, 0], [0.001349, 0]]
    assert out["labelsLeg"] == [
        "Start on Snake Road",
        "Left onto Mountain Brow Road",
        "Continue on Snake Road",
        "Straight on Snake Road",
        "Slight right onto an unnamed road",
        "Sharp right onto Kerns Road",
        "Back to A on Snake Road",
    ]
    assert out["build"] == [
        "Update 3 · 2026-10-15 · abcdef1",
        "Update 4 · 2026-10-16",
        "dev build",
        "dev build",
        "dev build",
    ]
    assert out["tile"] == "14/4556/6000"
    w, s_, e, n = out["tileBox"]
    assert w < -79.885 < e and s_ < 43.31 < n
    # The box spans tiles 4556/6000-6001; one tile round them: 3 x 4, nearest the middle first.
    assert out["tilesView"][0] == "14/4556/6000" and len(set(out["tilesView"])) == 12
    assert out["spread"] == [345, 25, 180]
    assert out["spreadWrap"] == [340, 20]
    assert out["spreadFine"] == [0, 90, 180, 270]
    assert out["compass"] == ["north", "northeast", "east", "south", "northwest"]
    assert out["keyFor"] == ["ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp"]
    assert out["optKey"] == [0, 1, 1, -1, -1, -1]
    big = out["big"]
    assert big["segments"] == 2 * 120 * 119
    assert big["hits"] > 1000
    # Generous bounds (CI machines are slow); in practice both take a few tens of ms.
    assert big["loadMs"] < 3000 and big["snapMs"] < 3000, big
