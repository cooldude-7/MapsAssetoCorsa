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
        "alternatives", "snapped", "far_pin",
    ):  # fmt: skip
        assert re.search(rf"\b{field}\b", js), f"app.js never sends/reads {field}"


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
