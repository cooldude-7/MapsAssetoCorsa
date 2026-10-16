/* Assetto Corsa Map Generator - the map page.
 *
 * Plain modern JS, no build step, no framework. Talks to the local app server on the same origin
 * (GET /api/status, POST /api/route, POST /api/build, ...). The pure helpers come first so the
 * tests can run them under Node; boot() only runs in a browser.
 *
 * Two ways to pick the road: turn by turn (the default: click the road you start on, pick a
 * direction, then a road at every junction; roads come from POST /api/roads in z14 tiles and
 * the junctions are worked out here, then POST /api/route-path makes the route) and quick
 * A to B (click points; POST /api/route finds the roads between them).
 */
"use strict";

const API = {
  status: "/api/status",
  settings: "/api/settings",
  route: "/api/route",
  roads: "/api/roads",
  routePath: "/api/route-path",
  profile: "/api/profile",
  build: "/api/build",
  jobs: "/api/jobs/",
  tracks: "/api/tracks",
};

const MAP_STYLES = {
  light: "https://tiles.openfreemap.org/styles/liberty",
  dark: "https://tiles.openfreemap.org/styles/dark",
};
const PHOTON_URL = "https://photon.komoot.io/api/";
const HOME_VIEW = { center: [-79.87, 43.25], zoom: 10.5 }; // Hamilton, Ontario
const ROUTE_COLOR = "#19d3ff";
const CASING_COLOR = "#06121a";
const ALT_COLOR = "#7f8a9c"; // route options that aren't selected
const ALT_HOVER_COLOR = "#9be9ff";
const LINE_DRAG_PX = 4; // a press on the route line that moves further than this is a drag
const STORE_KEYS = {
  view: "acmg.view",
  style: "acmg.style",
  pins: "acmg.pins",
  job: "acmg.job",
  options: "acmg.options",
  mode: "acmg.mode",
  tbt: "acmg.tbt",
};
const EMPTY_FC = { type: "FeatureCollection", features: [] };
const EARTH_RADIUS_M = 6371008.8;

/* ------------------------------------------------------------------------------------------
 * Pure helpers (no DOM). Points are [lon, lat] unless the name says otherwise.
 * ---------------------------------------------------------------------------------------- */

function toRad(deg) {
  return (deg * Math.PI) / 180;
}

/** Great-circle distance in metres between two [lon, lat] points. */
function distanceM(a, b) {
  const dLat = toRad(b[1] - a[1]);
  const dLon = toRad(b[0] - a[0]);
  const s =
    Math.sin(dLat / 2) ** 2 + Math.cos(toRad(a[1])) * Math.cos(toRad(b[1])) * Math.sin(dLon / 2) ** 2;
  return 2 * EARTH_RADIUS_M * Math.asin(Math.min(1, Math.sqrt(s)));
}

/** Running distance (m) at each vertex of a [lon, lat] line. */
function cumulativeM(line) {
  const out = [0];
  for (let i = 1; i < line.length; i++) out.push(out[i - 1] + distanceM(line[i - 1], line[i]));
  return out;
}

/** The point `d` metres along `line` (with `cum` from cumulativeM). */
function pointAlong(line, cum, d) {
  if (!line || !line.length) return null;
  const total = cum[cum.length - 1];
  if (!(d > 0)) return line[0].slice();
  if (d >= total) return line[line.length - 1].slice();
  let lo = 0;
  let hi = cum.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (cum[mid] <= d) lo = mid;
    else hi = mid;
  }
  const seg = cum[hi] - cum[lo];
  const t = seg > 0 ? (d - cum[lo]) / seg : 0;
  return [line[lo][0] + (line[hi][0] - line[lo][0]) * t, line[lo][1] + (line[hi][1] - line[lo][1]) * t];
}

/** Where a point `p` added in between goes in the chain: the gap it lengthens the least
 * (1..n-1; on a loop also n, the gap from the last point back to A). */
function bestInsertIndex(points, p, loop) {
  if (points.length < 2) return points.length;
  let best = 1;
  let bestCost = Infinity;
  const n = points.length;
  for (let i = 1; i <= (loop ? n : n - 1); i++) {
    const a = points[i - 1];
    const b = points[i % n];
    const cost = distanceM(a, p) + distanceM(p, b) - distanceM(a, b);
    if (cost < bestCost) {
      bestCost = cost;
      best = i;
    }
  }
  return best;
}

/** The label of the i-th point in click order: A, B, ..., Z, AA, AB, ... */
function pinLetter(i) {
  let out = "";
  for (let k = i + 1; k > 0; k = Math.floor((k - 1) / 26)) out = String.fromCharCode(65 + ((k - 1) % 26)) + out;
  return out;
}

/** What a point is: "start" (A), "finish" (the last point of an A->B route) or "via". On a
 * loop A is both the start and the finish, and the last point is just another point. */
function pinRole(i, n, loop) {
  if (i === 0) return "start";
  if (i === n - 1 && !loop) return "finish";
  return "via";
}

/** Pins for POST /api/route: [lat, lon] in routing order. The displayed order is always the
 * driving order. A reversed route is routed in the other order and then driven backwards by
 * the server (same roads), so the displayed order is flipped back first; on a loop A stays
 * the start, so only the points after it are flipped. */
function apiPins(lngLats, reversed, loop) {
  const pins = lngLats.map((p) => [p[1], p[0]]);
  if (!reversed) return pins;
  if (loop) return pins.slice(0, 1).concat(pins.slice(1).reverse());
  return pins.reverse();
}

/** The points after reversing the driving direction (on a loop A stays first). */
function reversedPins(items, loop) {
  return loop ? items.slice(0, 1).concat(items.slice(1).reverse()) : items.slice().reverse();
}

/** A friendly default track name from the road names along the route. */
function defaultTrackName(names, loop) {
  const clean = (names || []).map((n) => String(n == null ? "" : n).trim()).filter(Boolean);
  if (!clean.length) return "";
  let name;
  if (loop) {
    name = clean.length === 1 ? `${clean[0]} Loop` : `${clean[0]} & ${clean[1]} Loop`;
    if (name.length > 48) name = `${clean[0]} Loop`;
  } else {
    if (clean.length === 1) name = clean[0];
    else if (clean.length === 2) name = `${clean[0]} & ${clean[1]}`;
    else name = `${clean[0]} to ${clean[clean.length - 1]}`;
    if (name.length > 48) name = clean[0];
  }
  return name.slice(0, 60);
}

function formatLength(m) {
  if (!Number.isFinite(m)) return "–";
  if (m < 1000) return `${Math.round(m)} m`;
  return `${(m / 1000).toFixed(m < 10000 ? 2 : 1)} km`;
}

function formatInt(n) {
  return Number.isFinite(n) ? Math.round(n).toLocaleString("en-CA") : "–";
}

/** Folder-safe track id: lowercase letters, digits and underscores. */
function sanitizeTrackId(s) {
  return String(s || "")
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9_]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 60);
}

function densityLabel(v) {
  const x = Number(v);
  if (!(x > 0)) return "None";
  if (x < 0.6) return "Light";
  if (x < 1.4) return "Normal";
  return "Dense";
}

/** A round axis step (1, 2 or 5 x 10^k) giving about `target` ticks over `range`. */
function niceStep(range, target) {
  if (!(range > 0)) return 1;
  const raw = range / Math.max(1, target);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const n = raw / mag;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * mag;
}

/** Climb, drop, steepest grade (over >= 50 m) and the height range of a profile. */
function profileStats(s, z) {
  let climb = 0;
  let drop = 0;
  let minZ = Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i < z.length; i++) {
    if (z[i] < minZ) minZ = z[i];
    if (z[i] > maxZ) maxZ = z[i];
    if (i > 0) {
      const dz = z[i] - z[i - 1];
      if (dz > 0) climb += dz;
      else drop -= dz;
    }
  }
  let maxGrade = 0;
  let j = 0;
  for (let i = 0; i < s.length; i++) {
    while (j < s.length && s[j] - s[i] < 50) j++;
    if (j >= s.length) break;
    const g = Math.abs(z[j] - z[i]) / (s[j] - s[i]);
    if (g > maxGrade) maxGrade = g;
  }
  return { climb, drop, maxGrade, minZ, maxZ };
}

/** Index of the value in sorted `arr` closest to `v`. */
function nearestIndex(arr, v) {
  let lo = 0;
  let hi = arr.length - 1;
  if (hi < 0) return -1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] <= v) lo = mid;
    else hi = mid;
  }
  return Math.abs(arr[hi] - v) < Math.abs(arr[lo] - v) ? hi : lo;
}

/** "43.25, -79.87" -> [lon, lat], or null. */
function parseLatLon(q) {
  const m = /^\s*(-?\d{1,2}(?:\.\d+)?)\s*[,;\s]\s*(-?\d{1,3}(?:\.\d+)?)\s*$/.exec(String(q || ""));
  if (!m) return null;
  const lat = Number(m[1]);
  const lon = Number(m[2]);
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return [lon, lat];
}

function isLngLat(p) {
  return (
    Array.isArray(p) &&
    p.length === 2 &&
    Number.isFinite(p[0]) &&
    Number.isFinite(p[1]) &&
    Math.abs(p[0]) <= 180 &&
    Math.abs(p[1]) <= 90
  );
}

/** Title and subtitle for a Photon search result. */
function photonLabel(props) {
  const p = props || {};
  const street = p.street ? `${p.housenumber ? `${p.housenumber} ` : ""}${p.street}` : "";
  const title = p.name || street || p.city || p.county || p.state || p.country || "Unnamed place";
  const parts = [];
  for (const part of [street, p.city || p.district, p.state, p.country]) {
    if (part && part !== title && !parts.includes(part)) parts.push(part);
  }
  return { title, subtitle: parts.join(", ") };
}

/** The texts to search for. People type "<road> <town>", but OpenStreetMap often files a road
 * under a different town (Snake Road near Waterdown is in Burlington), so a query of 3+ words is
 * also searched without its last word. */
function searchQueries(q) {
  const words = String(q || "").trim().split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  const out = [words.join(" ")];
  if (words.length >= 3) out.push(words.slice(0, -1).join(" "));
  return out;
}

/** Photon extent [minLon, maxLat, maxLon, minLat] of an item (its point if it has none). */
function itemExtent(item) {
  const e = item.props && Array.isArray(item.props.extent) ? item.props.extent.map(Number) : null;
  if (e && e.length === 4 && e.every(Number.isFinite)) return e;
  const [lon, lat] = item.center;
  return [lon, lat, lon, lat];
}

/** Lower-case words of a text ("Snake Rd, Burlington" -> ["snake", "rd", "burlington"]). */
function wordsOf(text) {
  return String(text || "").toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(Boolean);
}

/** Search results from each query (`groups[i]` = results for `searchQueries(q)[i]`), merged:
 * one entry per title + subtitle (a road split into several OSM ways becomes one result whose
 * extent covers every piece), ranked by how many of the typed words they contain (a word in the
 * name counts double, a name word nobody typed counts against). Ties keep the road-name-only
 * search first, then Photon's order. */
function mergeSearchResults(groups, q, limit = 6) {
  const words = wordsOf(q);
  const merged = new Map();
  groups.forEach((items, g) => {
    (items || []).forEach((item, i) => {
      if (!item || !isLngLat(item.center)) return;
      const key = `${item.title}\u0000${item.subtitle || ""}`.toLowerCase();
      const seen = merged.get(key);
      if (!seen) {
        merged.set(key, { item: { ...item, props: { ...(item.props || {}) } }, group: g, order: i });
        return;
      }
      const a = itemExtent(seen.item);
      const b = itemExtent(item);
      seen.item.props.extent = [
        Math.min(a[0], b[0]),
        Math.max(a[1], b[1]),
        Math.max(a[2], b[2]),
        Math.min(a[3], b[3]),
      ];
      seen.group = Math.max(seen.group, g);
    });
  });
  const score = (it) => {
    const title = wordsOf(it.title);
    const rest = wordsOf(it.subtitle);
    const typed = words.reduce((n, w) => n + (title.includes(w) ? 2 : rest.includes(w) ? 1 : 0), 0);
    return typed - 0.5 * title.filter((w) => !words.includes(w)).length;
  };
  return [...merged.values()]
    .map((m) => ({ ...m, score: score(m.item) }))
    .sort((x, y) => y.score - x.score || y.group - x.group || x.order - y.order)
    .slice(0, limit)
    .map((m) => m.item);
}

/** "Source: NRCan HRDEM (1.0 m)" for the profile; no resolution if the title already has one. */
function sourceLabel(source, resolutionM) {
  if (!source) return "";
  const res = Number(resolutionM);
  const hasRes = /\b\d+(?:\.\d+)?\s?m\b/.test(source);
  const resText = Number.isFinite(res) && res > 0 && !hasRes ? ` (${res < 10 ? res.toFixed(1) : Math.round(res)} m)` : "";
  return `Source: ${source}${resText}`;
}

function formatDate(v) {
  if (v == null || v === "") return "";
  const d = typeof v === "number" ? new Date(v < 1e12 ? v * 1000 : v) : new Date(v);
  if (Number.isNaN(d.getTime())) return String(v);
  return d.toLocaleDateString("en-CA", { year: "numeric", month: "short", day: "numeric" });
}

function clamp(x, lo, hi) {
  return Math.min(hi, Math.max(lo, x));
}

/* ------------------------------------------------------------------------------------------
 * Turn by turn: road tiles and the client-side road graph (pure, no DOM)
 *
 * Roads come from POST /api/roads in z14 slippy-map tiles. The graph has a node per OSM node
 * ([lon, lat]), a half-edge per drivable direction of every way segment, and a coarse grid of
 * segments for snapping clicks. A junction is a node where 3+ segment ends meet, or a dead end.
 * ---------------------------------------------------------------------------------------- */

const TILE_Z = 14;
const ROADS_MIN_ZOOM = 13;
const ROADS_BATCH = 16;
const GRID_DEG = 0.002; // spatial grid cell for snapping (about 160-220 m)

function lonToTileX(lon, z = TILE_Z) {
  const n = 2 ** z;
  return clamp(Math.floor(((lon + 180) / 360) * n), 0, n - 1);
}

function latToTileY(lat, z = TILE_Z) {
  const n = 2 ** z;
  const r = toRad(clamp(lat, -85.05112878, 85.05112878));
  return clamp(Math.floor(((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * n), 0, n - 1);
}

function tileKeyAt(lon, lat, z = TILE_Z) {
  return `${z}/${lonToTileX(lon, z)}/${latToTileY(lat, z)}`;
}

/** [west, south, east, north] of tile "z/x/y". */
function tileBounds(key) {
  const [z, x, y] = String(key).split("/").map(Number);
  const n = 2 ** z;
  const lat = (yy) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * yy) / n))) * 180) / Math.PI;
  return [(x / n) * 360 - 180, lat(y + 1), ((x + 1) / n) * 360 - 180, lat(y)];
}

/** The z14 tiles covering a box plus `margin` tiles round it, nearest the middle first. */
function tilesForBounds(west, south, east, north, margin = 1, z = TILE_Z) {
  const n = 2 ** z;
  const x0 = Math.max(0, lonToTileX(west, z) - margin);
  const x1 = Math.min(n - 1, lonToTileX(east, z) + margin);
  const y0 = Math.max(0, latToTileY(north, z) - margin);
  const y1 = Math.min(n - 1, latToTileY(south, z) + margin);
  const cx = (lonToTileX(west, z) + lonToTileX(east, z)) / 2;
  const cy = (latToTileY(north, z) + latToTileY(south, z)) / 2;
  const out = [];
  for (let x = x0; x <= x1; x++) for (let y = y0; y <= y1; y++) out.push({ k: `${z}/${x}/${y}`, d: Math.hypot(x - cx, y - cy) });
  out.sort((a, b) => a.d - b.d);
  return out.map((t) => t.k);
}

function createRoadGraph() {
  return { nodes: new Map(), ways: new Map(), adj: new Map(), grid: new Map(), segments: 0 };
}

function wayName(w) {
  return (w && (String(w.name || "").trim() || String(w.ref || "").trim())) || "";
}

function gridKey(i, j) {
  return `${i},${j}`;
}

/** Merge a /api/roads response into the graph (ways already there are skipped). Returns the
 * number of segments added. */
function addRoads(g, data) {
  const nodes = (data && data.nodes) || {};
  for (const [id, ll] of Object.entries(nodes)) {
    if (g.nodes.has(id) || !Array.isArray(ll)) continue;
    const p = [Number(ll[0]), Number(ll[1])];
    if (isLngLat(p)) g.nodes.set(id, p);
  }
  let added = 0;
  for (const raw of (data && Array.isArray(data.ways) ? data.ways : [])) {
    if (!raw || raw.id == null) continue;
    const id = String(raw.id);
    if (g.ways.has(id)) continue;
    const ids = (Array.isArray(raw.nodes) ? raw.nodes : []).map(String);
    const oneway = Number(raw.oneway) === 1 ? 1 : Number(raw.oneway) === -1 ? -1 : 0;
    const way = { id, name: raw.name || "", ref: raw.ref || "", highway: raw.highway || "", oneway, nodes: ids };
    g.ways.set(id, way);
    for (let i = 0; i + 1 < ids.length; i++) {
      const a = ids[i];
      const b = ids[i + 1];
      const pa = g.nodes.get(a);
      const pb = g.nodes.get(b);
      if (a === b || !pa || !pb) continue;
      for (const [from, e] of [[a, { to: b, way: id, i, fwd: true }], [b, { to: a, way: id, i, fwd: false }]]) {
        let list = g.adj.get(from);
        if (!list) g.adj.set(from, (list = []));
        list.push(e);
      }
      const seg = { a, b, w: id };
      const i0 = Math.floor(Math.min(pa[0], pb[0]) / GRID_DEG);
      const i1 = Math.floor(Math.max(pa[0], pb[0]) / GRID_DEG);
      const j0 = Math.floor(Math.min(pa[1], pb[1]) / GRID_DEG);
      const j1 = Math.floor(Math.max(pa[1], pb[1]) / GRID_DEG);
      for (let gi = i0; gi <= i1; gi++) {
        for (let gj = j0; gj <= j1; gj++) {
          const k = gridKey(gi, gj);
          let cell = g.grid.get(k);
          if (!cell) g.grid.set(k, (cell = []));
          cell.push(seg);
        }
      }
      g.segments++;
      added++;
    }
  }
  return added;
}

function nodeDegree(g, id) {
  const list = g.adj.get(id);
  return list ? list.length : 0;
}

/** Unnamed service roads and tracks: driveways, parking aisles, lanes into fields. */
function isMinorWay(w) {
  return !!w && (w.highway === "service" || w.highway === "track") && !wayName(w);
}

/** The one way on from `cur` (arrived from `prev`) when `cur` isn't a junction worth stopping
 * at: plain road (two segment ends), or a road passing only driveways and other minor roads
 * (see isMinorWay; not when you are on one). Null at a junction or a dead end. */
function throughEdge(g, prev, cur) {
  const edges = g.adj.get(cur) || [];
  const out = edges.filter((e) => e.to !== prev);
  if (edges.length === 2) return out.length === 1 ? out[0] : null;
  if (edges.length < 2 || out.length !== edges.length - 1) return null;
  const inEdge = edges.find((e) => e.to === prev);
  if (isMinorWay(g.ways.get(inEdge.way))) return null;
  const major = out.filter((e) => !isMinorWay(g.ways.get(e.way)));
  return major.length === 1 ? major[0] : null;
}

/** A junction: where the route stops to ask which way (see throughEdge). */
function isJunction(g, prev, id) {
  return !throughEdge(g, prev, id);
}

/** May a car drive half-edge `e` (respects one-way streets)? */
function canDrive(g, e) {
  const w = g.ways.get(e.way);
  if (!w || !w.oneway) return true;
  return w.oneway === 1 ? e.fwd : !e.fwd;
}

function findEdge(g, from, to) {
  const list = g.adj.get(from) || [];
  return list.find((e) => e.to === to && canDrive(g, e)) || list.find((e) => e.to === to) || null;
}

/** The nearest segment to p within maxM metres: {a, b, w, t (0..1 from a to b), point, dist}. */
function nearestSegment(g, p, maxM) {
  const kx = Math.cos(toRad(p[1])) * 111320;
  const ky = 110540;
  const dLon = maxM / Math.max(kx, 1);
  const dLat = maxM / ky;
  const i0 = Math.floor((p[0] - dLon) / GRID_DEG);
  const i1 = Math.floor((p[0] + dLon) / GRID_DEG);
  const j0 = Math.floor((p[1] - dLat) / GRID_DEG);
  const j1 = Math.floor((p[1] + dLat) / GRID_DEG);
  let best = null;
  for (let gi = i0; gi <= i1; gi++) {
    for (let gj = j0; gj <= j1; gj++) {
      for (const s of g.grid.get(gridKey(gi, gj)) || []) {
        const a = g.nodes.get(s.a);
        const b = g.nodes.get(s.b);
        const ax = (a[0] - p[0]) * kx;
        const ay = (a[1] - p[1]) * ky;
        const bx = (b[0] - p[0]) * kx;
        const by = (b[1] - p[1]) * ky;
        const vx = bx - ax;
        const vy = by - ay;
        const len2 = vx * vx + vy * vy;
        const t = len2 > 0 ? clamp(-(ax * vx + ay * vy) / len2, 0, 1) : 0;
        const dist = Math.hypot(ax + t * vx, ay + t * vy);
        if (dist <= maxM && (!best || dist < best.dist)) {
          best = { a: s.a, b: s.b, w: s.w, t, dist, point: [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])] };
        }
      }
    }
  }
  return best;
}

/** Compass bearing (degrees, 0 = north, clockwise) from a to b; flat-earth, fine for metres. */
function bearingDeg(a, b) {
  const dx = (b[0] - a[0]) * Math.cos(toRad((a[1] + b[1]) / 2));
  const dy = b[1] - a[1];
  return ((Math.atan2(dx, dy) * 180) / Math.PI + 360) % 360;
}

/** Signed turn from heading `inB` to heading `outB`: -180..180, negative = left. */
function turnAngle(inB, outB) {
  let d = (((outB - inB) % 360) + 540) % 360 - 180;
  if (d === -180) d = 180;
  return d;
}

function turnWord(angle) {
  const a = Math.abs(angle);
  if (a < 20) return "straight";
  const side = angle < 0 ? "left" : "right";
  if (a < 60) return `slight ${side}`;
  if (a < 135) return side;
  return `sharp ${side}`;
}

/** Drive on from `cur` (arrived from `prev`) through plain road nodes (and past driveways, see
 * throughEdge) to the next junction. Returns the nodes after `cur`, ending at the junction.
 * Stops early at a node whose roads aren't loaded yet (`isLoaded`), a one-way the wrong way
 * (unless `anyWay`), or after going all round a ring. */
function continueRun(g, prev, cur, isLoaded, anyWay) {
  const out = [];
  const seen = new Set();
  for (let guard = 0; guard < 100000; guard++) {
    if (isLoaded && !isLoaded(cur)) break;
    if (seen.has(cur)) break;
    const next = throughEdge(g, prev, cur);
    if (!next || next.to === prev || (!anyWay && !canDrive(g, next))) break;
    seen.add(cur);
    prev = cur;
    cur = next.to;
    out.push(cur);
  }
  return out;
}

/** Nodes driving from `from` onto `to` and on to the next junction: [to, ..., junction]. */
function runFrom(g, from, to, isLoaded) {
  return [to, ...continueRun(g, from, to, isLoaded)];
}

/** The whole stretch of road between two junctions that segment a-b is on, as node ids. */
function runAround(g, a, b) {
  const back = continueRun(g, b, a, null, true).reverse();
  const fwd = continueRun(g, a, b, null, true);
  return [...back, a, b, ...fwd];
}

function coordsOf(g, ids) {
  const out = [];
  for (const id of ids) {
    const p = g.nodes.get(id);
    if (p) out.push(p);
  }
  return out;
}

/** The point `d` metres along a line of [lon, lat] (the last point if it is shorter). */
function pointAtDistance(line, d) {
  return pointAlong(line, cumulativeM(line), d);
}

/** The point `d` metres back from the end of a line. */
function pointBack(line, d) {
  return pointAtDistance(line.slice().reverse(), d);
}

/** The roads you can take at the end of `path` (node ids in driving order): every way out
 * except straight back (the U-turn), ordered left to right, each {to, way, name, angle,
 * allowed (false: a one-way the wrong way), run (nodes to the next junction), coords}. */
function branchesAt(g, path, isLoaded) {
  const n = path.length;
  if (n < 2) return [];
  const prev = path[n - 2];
  const cur = path[n - 1];
  const pc = g.nodes.get(cur);
  if (!pc) return [];
  const back = pointBack(coordsOf(g, path.slice(Math.max(0, n - 40))), 20);
  const inB = bearingDeg(back, pc);
  const out = [];
  for (const e of g.adj.get(cur) || []) {
    if (e.to === prev) continue;
    const run = runFrom(g, cur, e.to, isLoaded);
    const coords = [pc, ...coordsOf(g, run)];
    const ahead = pointAtDistance(coords, 20);
    const bearing = bearingDeg(pc, ahead);
    const w = g.ways.get(e.way);
    out.push({ to: e.to, way: e.way, name: wayName(w), angle: turnAngle(inB, bearing), bearing, allowed: canDrive(g, e), run, coords });
  }
  out.sort((a, b) => a.angle - b.angle);
  return out;
}

/** Which branch the Left / Up / Right keys take: the leftmost, the straightest, the rightmost
 * of the ones you may drive (null when there is none that way). */
function keyBranches(branches) {
  const ok = branches.filter((b) => b.allowed);
  if (!ok.length) return { left: null, straight: null, right: null };
  let straight = ok.reduce((best, b) => (Math.abs(b.angle) < Math.abs(best.angle) ? b : best), ok[0]);
  if (Math.abs(straight.angle) > 50 && ok.length > 1) straight = null;
  const left = ok[0].angle < -15 || ok.length === 1 ? ok[0] : null;
  const right = ok[ok.length - 1].angle > 15 || ok.length === 1 ? ok[ok.length - 1] : null;
  return { left, straight, right };
}

/** Keep going "along the same road" at a junction: the way you're on, else a road of the same
 * name, else the straightest road within 45 degrees. */
function sameRoadBranch(g, branches, wayId) {
  const ok = branches.filter((b) => b.allowed);
  const same = ok.find((b) => b.way === wayId);
  if (same) return same;
  const name = wayName(g.ways.get(wayId));
  const named = name ? ok.filter((b) => b.name === name) : [];
  if (named.length === 1) return named[0];
  const straight = ok.filter((b) => Math.abs(b.angle) <= 45).sort((a, b) => Math.abs(a.angle) - Math.abs(b.angle));
  return straight[0] || null;
}

function edgeWay(g, a, b) {
  const e = findEdge(g, a, b);
  return e ? e.way : null;
}

/** Driving on from the end of `path` (any road out of its junction, then along that road
 * through up to `maxJunctions` junctions), where is segment {a, b}? Returns {nodes (to append,
 * up to the junction after the segment), cut (index in nodes of the segment's end node),
 * forward (driving a to b), branch} or null. */
function findAhead(g, path, seg, isLoaded, maxJunctions = 40) {
  const hit = (from, to) => (from === seg.a && to === seg.b ? 1 : from === seg.b && to === seg.a ? -1 : 0);
  for (const br of branchesAt(g, path, isLoaded)) {
    if (!br.allowed) continue;
    const nodes = [];
    let tail = path.slice(-2);
    let run = br.run;
    let way = br.way;
    for (let j = 0; j <= maxJunctions && run.length; j++) {
      let from = tail[tail.length - 1];
      for (let k = 0; k < run.length; k++) {
        const to = run[k];
        nodes.push(to);
        const h = hit(from, to);
        if (h) {
          const cut = nodes.length - 1;
          nodes.push(...run.slice(k + 1));
          return { nodes, cut, forward: h > 0, branch: br };
        }
        from = to;
      }
      tail = [tail[tail.length - 1], ...run].slice(-2);
      if (nodes.length > 50000) break;
      const next = sameRoadBranch(g, branchesAt(g, [...path.slice(-1), ...nodes], isLoaded), way);
      if (!next) break;
      way = next.way;
      run = next.run;
    }
  }
  return null;
}

/** "Back to start": the path closed back to A, if the route returns to A's road and drives
 * through the start the same way; null if it doesn't (yet). The closed path ends at path[0]. */
function closeLoopPath(g, path, isLoaded) {
  if (path.length < 3) return null;
  const p0 = path[0];
  const p1 = path[1];
  for (let i = 2; i + 1 < path.length; i++) if (path[i] === p0 && path[i + 1] === p1) return path.slice(0, i + 1);
  const tip = path[path.length - 1];
  if (tip === p0) return path.slice();
  for (const br of branchesAt(g, path, isLoaded)) {
    if (!br.allowed) continue;
    const seq = [tip, ...br.run];
    for (let j = 0; j + 1 < seq.length; j++) {
      if (seq[j] === p0 && seq[j + 1] === p1) return path.concat(seq.slice(1, j + 1));
    }
  }
  return null;
}

/** The drawn line of a path: from `startT` along its first segment to `endT` along its last. */
function pathLine(g, path, startT, endT) {
  const pts = coordsOf(g, path);
  if (pts.length < 2) return pts;
  const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  const n = pts.length;
  const first = lerp(pts[0], pts[1], startT || 0);
  const last = lerp(pts[n - 2], pts[n - 1], endT == null ? 1 : endT);
  if (n === 2) return [first, last];
  return [first, ...pts.slice(1, n - 1), last];
}

function lineLengthM(line) {
  let s = 0;
  for (let i = 1; i < line.length; i++) s += distanceM(line[i - 1], line[i]);
  return s;
}

/** One line per leg for the route panel: "Start on Snake Road", "Left onto Thomson Drive". */
function legLabel(leg, prevName) {
  const name = leg.name || "an unnamed road";
  if (leg.kind === "start") return `Start on ${name}`;
  if (leg.kind === "close") return `Back to A on ${name}`;
  const word = turnWord(leg.angle || 0);
  if (word === "straight") return leg.name && leg.name === prevName ? `Continue on ${name}` : `Straight on ${name}`;
  return `${word[0].toUpperCase()}${word.slice(1)} onto ${name}`;
}

/** "Update 3 · 2026-10-15 · abc1234", or "dev build". */
function buildLabel(build) {
  if (!build || typeof build !== "object") return "dev build";
  const v = String(build.version == null ? "" : build.version).trim();
  if (!v || v === "dev") return "dev build";
  const parts = [/^update\b/i.test(v) ? v : `Update ${v}`];
  if (build.date) parts.push(String(build.date).slice(0, 10));
  if (build.commit) parts.push(String(build.commit).slice(0, 7));
  return parts.join(" · ");
}

/** The first `m` metres of a [lon, lat] line (all of it if it is shorter). */
function lineHead(line, m) {
  if (!Array.isArray(line) || line.length < 2) return Array.isArray(line) ? line.slice() : [];
  const out = [line[0]];
  let s = 0;
  for (let i = 1; i < line.length; i++) {
    const d = distanceM(line[i - 1], line[i]);
    if (s + d >= m) {
      const f = d > 0 ? (m - s) / d : 0;
      out.push([line[i - 1][0] + (line[i][0] - line[i - 1][0]) * f, line[i - 1][1] + (line[i][1] - line[i - 1][1]) * f]);
      return out;
    }
    s += d;
    out.push(line[i]);
  }
  return out;
}

/** Display angles for arrows round one point: the bearings (degrees), nudged apart so no two
 * are closer than `minGap` (same order as given). Keeps arrows for roads that leave a junction
 * at nearly the same angle from sitting on top of each other. */
function spreadAngles(bearings, minGap = 40) {
  const norm = (b) => ((b % 360) + 360) % 360;
  const n = bearings.length;
  if (n < 2) return bearings.map(norm);
  const gap = Math.min(minGap, 360 / n);
  const order = [...bearings.keys()].sort((i, j) => norm(bearings[i]) - norm(bearings[j]));
  const a = order.map((i) => norm(bearings[i]));
  for (let pass = 0; pass < 50; pass++) {
    let moved = false;
    for (let k = 0; k < n; k++) {
      const k2 = (k + 1) % n;
      const d = a[k2] - a[k] + (k2 === 0 ? 360 : 0);
      if (d < gap - 1e-6) {
        const push = (gap - d) / 2;
        a[k] -= push;
        a[k2] += push;
        moved = true;
      }
    }
    if (!moved) break;
  }
  const out = new Array(n);
  order.forEach((i, k) => {
    out[i] = norm(a[k]);
  });
  return out;
}

const COMPASS = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"];

function compassWord(bearing) {
  return COMPASS[Math.round((((bearing % 360) + 360) % 360) / 45) % 8];
}

const KEY_BEARING = { ArrowUp: 0, ArrowRight: 90, ArrowDown: 180, ArrowLeft: 270 };
const KEY_GLYPH = { ArrowUp: "↑", ArrowRight: "→", ArrowDown: "↓", ArrowLeft: "←" };

/** The arrow key that points most nearly along `bearing` ("ArrowUp" for north...). */
function keyForBearing(bearing) {
  return ["ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft"][Math.round((((bearing % 360) + 360) % 360) / 90) % 4];
}

/** Which of the options (bearings; null = not allowed) an arrow key points at: the nearest
 * within 89 degrees of the key's direction, or -1. */
function optionForKey(bearings, key) {
  const kb = KEY_BEARING[key];
  if (kb == null) return -1;
  let best = -1;
  let bestD = 89;
  bearings.forEach((b, i) => {
    if (b == null) return;
    const d = Math.abs(turnAngle(kb, b));
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  });
  return best;
}

/** The key shown on each road's arrow at a junction: ←, ↑ or → (what keyBranches gives those
 * keys), or its number among the roads you may take (always numbers when there are more than
 * three). Returns one label per branch ("" for a road you may not take). */
function branchKeyLabels(branches) {
  const ok = branches.filter((b) => b.allowed);
  const k = keyBranches(branches);
  return branches.map((b) => {
    const n = ok.indexOf(b);
    if (n < 0) return "";
    if (ok.length > 3) return String(n + 1);
    if (b === k.straight) return "↑";
    if (b === k.left) return "←";
    if (b === k.right) return "→";
    return String(n + 1);
  });
}

/** The branch a key takes at a junction: arrows via keyBranches, digits by number. */
function branchForKey(branches, key) {
  const ok = branches.filter((b) => b.allowed);
  if (/^[1-9]$/.test(key)) return ok[Number(key) - 1] || null;
  const k = keyBranches(branches);
  if (key === "ArrowLeft") return k.left;
  if (key === "ArrowUp") return k.straight;
  if (key === "ArrowRight") return k.right;
  return null;
}

/* ------------------------------------------------------------------------------------------
 * Small browser utilities
 * ---------------------------------------------------------------------------------------- */

const $ = (id) => document.getElementById(id);

/** Browser storage is a convenience only: every access is guarded; nothing depends on it. */
const store = {
  get(key) { try { const v = window.localStorage.getItem(key); return v == null ? null : JSON.parse(v); } catch { return null; } },
  set(key, value) { try { window.localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ } },
  remove(key) { try { window.localStorage.removeItem(key); } catch { /* storage unavailable */ } },
};

/** Tiny element builder. Strings become text nodes (never HTML). */
function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : String(c));
  }
  return el;
}

class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data || null; // the error response's other fields (e.g. far_pin)
  }
}

/** JSON request to the local app. Errors carry the server's plain-English message. */
async function api(path, { method = "GET", body, signal } = {}) {
  let res;
  try {
    res = await fetch(path, {
      method,
      headers: body === undefined ? { Accept: "application/json" } : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if (err && err.name === "AbortError") throw err;
    throw new ApiError(
      "Can't reach the Map Generator app. Make sure its window is still open, then try again.",
      0,
    );
  }
  let data = null;
  try {
    data = await res.json();
  } catch {
    data = null;
  }
  if (!res.ok) {
    const msg = data && typeof data.error === "string" && data.error ? data.error : `Something went wrong (HTTP ${res.status}).`;
    throw new ApiError(msg, res.status, data);
  }
  return data;
}

let toastTimer = 0;
function toast(message, kind) {
  const el = $("toast");
  el.textContent = message;
  el.dataset.kind = kind || "info";
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.hidden = true;
  }, 4500);
}

/** Confirm with the <dialog>, falling back to window.confirm. Resolves true on OK. */
function confirmDialog(title, text, okLabel) {
  const dlg = $("confirm-dialog");
  if (!dlg || typeof dlg.showModal !== "function") {
    return Promise.resolve(window.confirm(`${title}\n\n${text}`));
  }
  $("confirm-title").textContent = title;
  $("confirm-text").textContent = text;
  $("confirm-ok").textContent = okLabel || "Delete";
  return new Promise((resolve) => {
    const onClose = () => {
      dlg.removeEventListener("close", onClose);
      resolve(dlg.returnValue === "ok");
    };
    dlg.returnValue = "cancel";
    dlg.addEventListener("close", onClose);
    dlg.showModal();
    $("confirm-ok").focus();
  });
}

/** Where the server snapped each point, as [lon, lat] in the displayed (driving) order.
 * `snapped` is [[lat, lon], ...] in request order (see apiPins); null if it doesn't fit. */
function displaySnapped(snapped, n, reversed, loop) {
  if (!Array.isArray(snapped) || snapped.length !== n) return null;
  const pts = snapped.map((p) => (Array.isArray(p) && p.length === 2 ? [Number(p[1]), Number(p[0])] : null));
  if (!pts.every(isLngLat)) return null;
  return reversed ? reversedPins(pts, loop) : pts;
}

/** The displayed index of request pin `i` (apiPins' order is its own inverse). */
function displayIndex(i, n, reversed, loop) {
  const order = [...Array(n).keys()];
  return (reversed ? reversedPins(order, loop) : order)[i];
}

/** "1 · 3.5 km · Snake Road" for route option i (at most three road names). */
function routeOptionLabel(opt, i) {
  const names = (Array.isArray(opt && opt.names) ? opt.names : []).map((x) => String(x || "").trim()).filter(Boolean);
  const roads = names.length > 3 ? `${names.slice(0, 3).join(", ")}, …` : names.join(", ");
  return [String(i + 1), formatLength(Number(opt && opt.length_m)), roads].filter(Boolean).join(" · ");
}

/** The point of a [lon, lat] line nearest to p: {d: metres along the line, dist: metres off it,
 * point}. Each segment is flattened around its own start (fine at map-click scale). With
 * `fromD`, only the part of the line from `fromD` metres on is searched. */
function nearestOnLine(line, cum, p, fromD = 0) {
  let best = null;
  for (let i = 1; i < line.length; i++) {
    if (cum[i] < fromD) continue;
    const a = line[i - 1];
    const b = line[i];
    const kx = Math.cos(toRad(a[1])) * 111320;
    const ky = 110540;
    const bx = (b[0] - a[0]) * kx;
    const by = (b[1] - a[1]) * ky;
    const px = (p[0] - a[0]) * kx;
    const py = (p[1] - a[1]) * ky;
    const len2 = bx * bx + by * by;
    let t = len2 > 0 ? clamp((px * bx + py * by) / len2, 0, 1) : 0;
    const segLen = cum[i] - cum[i - 1];
    if (cum[i - 1] + t * segLen < fromD && segLen > 0) t = (fromD - cum[i - 1]) / segLen;
    const dist = Math.hypot(px - t * bx, py - t * by);
    if (!best || dist < best.dist - 1e-9) {
      best = { d: cum[i - 1] + t * segLen, dist, point: [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])] };
    }
  }
  return best;
}

/** Where a point dragged off the route line at `p` goes in the chain of points: between the
 * two points it lies between along the route (1..n-1; on a loop n is the stretch back to A). */
function insertIndexAlong(line, points, p, loop) {
  const n = points.length;
  if (!Array.isArray(line) || line.length < 2 || n < 2) return bestInsertIndex(points, p, loop);
  const cum = cumulativeM(line);
  const at = nearestOnLine(line, cum, p);
  if (!at) return bestInsertIndex(points, p, loop);
  let from = 0;
  for (let i = 1; i < n; i++) {
    // Each point is looked for after the one before it, so a route that runs over itself
    // still finds the points in driving order.
    const pin = nearestOnLine(line, cum, points[i], from);
    if (!pin) break;
    if (at.d < pin.d) return i;
    from = pin.d;
  }
  return loop ? n : n - 1;
}

/* ------------------------------------------------------------------------------------------
 * App state
 * ---------------------------------------------------------------------------------------- */

const state = {
  map: null,
  mapReady: false,
  styleKey: "light",
  mode: "tbt", // "tbt" (turn by turn) or "quick" (A to B points)
  tbt: null, // turn-by-turn state (see tbtNew)
  pins: [], // [{lngLat: [lon, lat], marker, el}] in driving order: A, B, C, ... (click order)
  reversed: false, // drive the routed road backwards (sent as `reverse` to /api/route)
  loop: false, // "Back to start": the route comes back to A (sent as `loop` to /api/route)
  history: [],
  dragSnapshot: null,
  lastDragEnd: 0,
  waypointMode: false,
  route: null, // the selected route option (from the last /api/route response) + `snapped`
  routeOptions: [], // the response's `alternatives`: the best route first
  optionIndex: 0, // which of routeOptions is selected (drawn blue, built, profiled)
  routeSnapped: null, // the response's `snapped` (request order)
  altHover: -1, // route option highlighted on the map (hovered line or list item)
  lastEdit: null, // {pin}: the point the last click/drag placed (taken back if it is off-road)
  lineDrag: null, // dragging the route line: {index, start, moved}
  routeCum: null,
  routeStale: false,
  routeError: "",
  routeSeq: 0,
  routeTimer: 0,
  routeAbort: null,
  profile: null, // last /api/profile response (+ route_id)
  profileGeom: null,
  hoverIndex: -1,
  status: null,
  nameTouched: false,
  jobId: null,
  jobName: "",
  jobTimer: 0,
  jobFailures: 0,
  jobLogCount: -1,
  jobResumed: false, // polling a job started before this page loaded
  resultTrackId: null, // track shown as built in the build panel
  search: { items: [], active: -1, timer: 0, abort: null },
};

/* ------------------------------------------------------------------------------------------
 * Map
 * ---------------------------------------------------------------------------------------- */

function initMap() {
  if (!window.maplibregl) {
    showMapError("The map couldn't load. Check your internet connection, then reload the page.");
    return;
  }
  const saved = store.get(STORE_KEYS.view);
  const view =
    saved && isLngLat(saved.center) && Number.isFinite(saved.zoom) ? saved : HOME_VIEW;
  const map = new window.maplibregl.Map({
    container: "map",
    style: MAP_STYLES[state.styleKey],
    center: view.center,
    zoom: view.zoom,
    attributionControl: false,
    boxZoom: false, // Shift+click adds waypoints
    doubleClickZoom: false, // two quick clicks are two pins
    dragRotate: false,
    pitchWithRotate: false,
    touchPitch: false,
    maxPitch: 0,
  });
  state.map = map;
  map.touchZoomRotate.disableRotation();
  map.addControl(new window.maplibregl.NavigationControl({ showCompass: false }), "top-right");
  map.addControl(new window.maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
  map.addControl(new window.maplibregl.AttributionControl({ compact: false }), "bottom-right");

  map.on("style.load", () => {
    state.mapReady = true;
    $("map-error").hidden = true;
    addRouteLayers();
  });
  map.on("error", (e) => {
    console.warn("map error", e && e.error);
    if (!state.mapReady) {
      showMapError("The map tiles couldn't load. Check your internet connection, then reload the page.");
    }
  });
  map.on("click", onMapClick);
  map.on("mousemove", onMapMouseMove);
  map.on("mouseout", () => {
    if (state.mode === "tbt") tbtHoverOff();
    else if (!state.lineDrag) setMapHover(-1, false);
  });
  if (state.mode === "tbt") map.keyboard.disable(); // the arrow keys take turns
  let roadsTimer = 0;
  map.on("moveend", () => {
    clearTimeout(roadsTimer);
    roadsTimer = setTimeout(() => {
      if (state.mode !== "tbt") return;
      tbtWantTiles();
      if (tbtPhase() === "start") tbtRender();
    }, 200);
  });
  map.on("zoomend", () => tbtRenderChip());
  map.on("mousedown", "dm-route-hit", onRouteLineDown);
  map.on("mouseup", onRouteLineUp);
  window.addEventListener("mouseup", () => setTimeout(cancelLineDrag, 0));
  let viewTimer = 0;
  map.on("moveend", () => {
    clearTimeout(viewTimer);
    viewTimer = setTimeout(() => {
      const c = map.getCenter();
      store.set(STORE_KEYS.view, { center: [c.lng, c.lat], zoom: map.getZoom() });
    }, 400);
  });
}

function showMapError(message) {
  const el = $("map-error");
  el.textContent = message;
  el.hidden = false;
}

function arrowImage() {
  const size = 32;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const chevron = () => {
    ctx.beginPath();
    ctx.moveTo(12, 9);
    ctx.lineTo(20, 16);
    ctx.lineTo(12, 23);
  };
  chevron();
  ctx.strokeStyle = CASING_COLOR;
  ctx.lineWidth = 7;
  ctx.stroke();
  chevron();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 3.5;
  ctx.stroke();
  return ctx.getImageData(0, 0, size, size);
}

/** (Re)add our sources and layers on top of the base style. Runs on every style load. */
function addRouteLayers() {
  const map = state.map;
  if (!map) return;
  if (!map.hasImage("dm-arrow")) map.addImage("dm-arrow", arrowImage(), { pixelRatio: 2 });
  const sources = {
    "dm-route": routeData(),
    "dm-alts": altsData(),
    "dm-snap": snapData(),
    "dm-hover": hoverData(),
    "dm-drag": EMPTY_FC,
    "dm-tbt-hover": EMPTY_FC,
    "dm-tbt-branches": tbtBranchData(),
  };
  for (const [id, data] of Object.entries(sources)) {
    if (!map.getSource(id)) map.addSource(id, { type: "geojson", data });
  }
  const add = (layer) => {
    if (!map.getLayer(layer.id)) map.addLayer(layer);
  };
  const isLine = ["==", ["geometry-type"], "LineString"];
  const isPoint = ["==", ["geometry-type"], "Point"];
  add({
    id: "dm-snap-line",
    type: "line",
    source: "dm-snap",
    filter: isLine,
    paint: { "line-color": "#ffffff", "line-width": 1.5, "line-dasharray": [2, 2], "line-opacity": 0.85 },
  });
  // Turn by turn: the road under the mouse, the roads out of the junction, a click's preview.
  add({
    id: "dm-tbt-road",
    type: "line",
    source: "dm-tbt-hover",
    filter: ["==", ["get", "kind"], "road"],
    layout: { "line-join": "round", "line-cap": "round" },
    paint: {
      "line-color": ROUTE_COLOR,
      "line-width": ["interpolate", ["linear"], ["zoom"], 12, 5, 17, 14],
      "line-opacity": 0.45,
    },
  });
  add({
    id: "dm-tbt-stub-casing",
    type: "line",
    source: "dm-tbt-branches",
    filter: ["all", isLine, ["==", ["get", "allowed"], true]],
    layout: { "line-join": "round", "line-cap": "round" },
    paint: { "line-color": CASING_COLOR, "line-width": ["case", ["get", "hot"], 10, 8], "line-opacity": 0.7 },
  });
  add({
    id: "dm-tbt-stub",
    type: "line",
    source: "dm-tbt-branches",
    filter: ["all", isLine, ["==", ["get", "allowed"], true]],
    layout: { "line-join": "round", "line-cap": "round" },
    paint: { "line-color": ["case", ["get", "hot"], ALT_HOVER_COLOR, "#ffffff"], "line-width": ["case", ["get", "hot"], 6, 4] },
  });
  add({
    id: "dm-tbt-stub-no",
    type: "line",
    source: "dm-tbt-branches",
    filter: ["all", isLine, ["==", ["get", "allowed"], false]],
    paint: { "line-color": "#f05252", "line-width": 3, "line-dasharray": [1.5, 1.5], "line-opacity": 0.8 },
  });
  // The other route options: thinner grey lines under the selected one.
  add({
    id: "dm-alt-casing",
    type: "line",
    source: "dm-alts",
    layout: { "line-join": "round", "line-cap": "round" },
    paint: {
      "line-color": CASING_COLOR,
      "line-width": ["interpolate", ["linear"], ["zoom"], 8, 4, 13, 6, 17, 10],
      "line-opacity": 0.45,
    },
  });
  add({
    id: "dm-alt-line",
    type: "line",
    source: "dm-alts",
    layout: { "line-join": "round", "line-cap": "round" },
    paint: {
      "line-color": ALT_COLOR,
      "line-width": ["interpolate", ["linear"], ["zoom"], 8, 2, 13, 3.5, 17, 6.5],
    },
  });
  add({
    id: "dm-alt-hover",
    type: "line",
    source: "dm-alts",
    filter: ["==", ["get", "index"], -1],
    layout: { "line-join": "round", "line-cap": "round" },
    paint: {
      "line-color": ALT_HOVER_COLOR,
      "line-width": ["interpolate", ["linear"], ["zoom"], 8, 3, 13, 5, 17, 9],
    },
  });
  add({
    id: "dm-route-casing",
    type: "line",
    source: "dm-route",
    layout: { "line-join": "round", "line-cap": "round" },
    paint: {
      "line-color": CASING_COLOR,
      "line-width": ["interpolate", ["linear"], ["zoom"], 8, 5, 13, 9, 17, 16],
      "line-opacity": 0.85,
    },
  });
  add({
    id: "dm-route-line",
    type: "line",
    source: "dm-route",
    layout: { "line-join": "round", "line-cap": "round" },
    paint: {
      "line-color": ROUTE_COLOR,
      "line-width": ["interpolate", ["linear"], ["zoom"], 8, 3, 13, 5.5, 17, 10],
    },
  });
  add({
    id: "dm-route-arrows",
    type: "symbol",
    source: "dm-route",
    layout: {
      "symbol-placement": "line",
      "symbol-spacing": 110,
      "icon-image": "dm-arrow",
      "icon-size": 0.85,
      "icon-allow-overlap": true,
      "icon-ignore-placement": true,
      "icon-rotation-alignment": "map",
    },
  });
  add({
    id: "dm-snap-point",
    type: "circle",
    source: "dm-snap",
    filter: isPoint,
    paint: {
      "circle-radius": 4,
      "circle-color": "#ffffff",
      "circle-stroke-color": CASING_COLOR,
      "circle-stroke-width": 2,
    },
  });
  add({
    id: "dm-hover",
    type: "circle",
    source: "dm-hover",
    paint: {
      "circle-radius": 7,
      "circle-color": "#ffffff",
      "circle-stroke-color": ROUTE_COLOR,
      "circle-stroke-width": 3,
    },
  });
  add({
    id: "dm-tbt-ahead",
    type: "line",
    source: "dm-tbt-hover",
    filter: ["==", ["get", "kind"], "ahead"],
    layout: { "line-join": "round", "line-cap": "round" },
    paint: { "line-color": ROUTE_COLOR, "line-width": 4, "line-dasharray": [1.2, 1.2], "line-opacity": 0.9 },
  });
  add({
    id: "dm-tbt-junction",
    type: "circle",
    source: "dm-tbt-branches",
    filter: isPoint,
    paint: { "circle-radius": 6, "circle-color": "#ffffff", "circle-stroke-color": CASING_COLOR, "circle-stroke-width": 2.5 },
  });
  add({
    id: "dm-tbt-finish",
    type: "circle",
    source: "dm-tbt-hover",
    filter: isPoint,
    paint: { "circle-radius": 6, "circle-color": "#ef4444", "circle-stroke-color": "#ffffff", "circle-stroke-width": 2.5 },
  });
  // Invisible, wider copies of the lines: what the mouse hovers, clicks and drags.
  const hit = (id, source) =>
    add({
      id,
      type: "line",
      source,
      layout: { "line-join": "round", "line-cap": "round" },
      paint: { "line-color": "#000000", "line-width": 18, "line-opacity": 0 },
    });
  hit("dm-alt-hit", "dm-alts");
  hit("dm-route-hit", "dm-route");
  // Dragging the route line: dashed legs from the points either side to the new point.
  add({
    id: "dm-drag-line",
    type: "line",
    source: "dm-drag",
    filter: isLine,
    paint: { "line-color": ROUTE_COLOR, "line-width": 3, "line-dasharray": [1.5, 1.5], "line-opacity": 0.9 },
  });
  add({
    id: "dm-drag-point",
    type: "circle",
    source: "dm-drag",
    filter: isPoint,
    paint: {
      "circle-radius": 7,
      "circle-color": "#ffffff",
      "circle-stroke-color": ROUTE_COLOR,
      "circle-stroke-width": 3,
    },
  });
  applyAltHover();
  applyStaleStyle();
}

/** The route options that aren't selected (feature property `index` = option index). */
function altsData() {
  const features = [];
  state.routeOptions.forEach((opt, i) => {
    if (i === state.optionIndex || !opt || !Array.isArray(opt.line) || opt.line.length < 2) return;
    features.push({ type: "Feature", properties: { index: i }, geometry: { type: "LineString", coordinates: opt.line } });
  });
  return features.length ? { type: "FeatureCollection", features } : EMPTY_FC;
}

function routeData() {
  let line = state.route && Array.isArray(state.route.line) ? state.route.line : [];
  if (state.mode === "tbt") line = state.tbt ? tbtLine() : []; // drawn as you drive, no wait
  if (line.length < 2) return EMPTY_FC;
  return {
    type: "FeatureCollection",
    features: [{ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: line } }],
  };
}

/** Snapped points (where the pins landed on the road) plus dashed leaders from far-off pins. */
function snapData() {
  if (state.mode === "tbt") return EMPTY_FC;
  const snapped = state.route && Array.isArray(state.route.snapped) ? state.route.snapped : [];
  const pts = snapped.filter((p) => Array.isArray(p) && p.length === 2).map((p) => [p[1], p[0]]);
  if (!pts.length) return EMPTY_FC;
  const features = pts.map((p) => ({ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: p } }));
  for (const pin of state.pins) {
    let best = null;
    let bestD = Infinity;
    for (const p of pts) {
      const d = distanceM(pin.lngLat, p);
      if (d < bestD) {
        bestD = d;
        best = p;
      }
    }
    if (best && bestD > 12) {
      features.push({
        type: "Feature",
        properties: {},
        geometry: { type: "LineString", coordinates: [pin.lngLat, best] },
      });
    }
  }
  return { type: "FeatureCollection", features };
}

function hoverData(point) {
  if (!point) return EMPTY_FC;
  return { type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: point } }] };
}

function setSourceData(id, data) {
  const src = state.map && state.map.getSource(id);
  if (src) src.setData(data);
}

function drawRoute() {
  setSourceData("dm-route", routeData());
  setSourceData("dm-alts", altsData());
  setSourceData("dm-snap", snapData());
  applyAltHover();
  applyStaleStyle();
}

/** Highlight route option `state.altHover` (a grey line) on the map and in the list. */
function applyAltHover() {
  const map = state.map;
  if (map && map.getLayer("dm-alt-hover")) map.setFilter("dm-alt-hover", ["==", ["get", "index"], state.altHover]);
  const list = $("route-option-list");
  if (list) [...list.querySelectorAll("button")].forEach((b, i) => b.classList.toggle("hover", i === state.altHover));
}

function setAltHover(i) {
  const k = Number.isInteger(i) && i >= 0 && i !== state.optionIndex ? i : -1;
  if (k === state.altHover) return;
  state.altHover = k;
  applyAltHover();
}

function applyStaleStyle() {
  const map = state.map;
  if (!map || !map.getLayer("dm-route-line")) return;
  const stale = state.routeStale && state.mode !== "tbt";
  map.setPaintProperty("dm-route-line", "line-opacity", stale ? 0.4 : 1);
  map.setPaintProperty("dm-route-casing", "line-opacity", stale ? 0.3 : 0.85);
  map.setPaintProperty("dm-route-arrows", "icon-opacity", stale ? 0.3 : 1);
}

function fitToRoute() {
  const map = state.map;
  if (!map) return;
  let coords =
    state.route && state.route.line && state.route.line.length >= 2
      ? state.route.line
      : state.pins.map((p) => p.lngLat);
  if (state.mode === "tbt") {
    coords = tbtLine();
    if (!coords.length && state.tbt.start) coords = [state.tbt.start.point];
  }
  if (!coords.length) return;
  if (coords.length === 1) {
    map.flyTo({ center: coords[0], zoom: Math.max(map.getZoom(), 14) });
    return;
  }
  const b = new window.maplibregl.LngLatBounds(coords[0], coords[0]);
  for (const c of coords) b.extend(c);
  map.fitBounds(b, { padding: { top: 80, bottom: 50, left: 50, right: 60 }, maxZoom: 16, duration: 700 });
}

function setMapStyle(key) {
  state.styleKey = key === "dark" ? "dark" : "light";
  store.set(STORE_KEYS.style, state.styleKey);
  const btn = $("style-toggle");
  btn.textContent = state.styleKey === "dark" ? "Light map" : "Dark map";
  btn.setAttribute("aria-label", `Switch to the ${state.styleKey === "dark" ? "light" : "dark"} map`);
}

function toggleMapStyle() {
  setMapStyle(state.styleKey === "dark" ? "light" : "dark");
  if (state.map) state.map.setStyle(MAP_STYLES[state.styleKey], { diff: false });
}

/* ------------------------------------------------------------------------------------------
 * Pins
 * ---------------------------------------------------------------------------------------- */

const ICONS = {
  locate:
    '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">' +
    '<circle cx="12" cy="12" r="6"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/></svg>',
  close:
    '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round">' +
    '<path d="M6 6l12 12M18 6L6 18"/></svg>',
  trash:
    '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 11v6M14 11v6"/></svg>',
};

/** A button showing one of the static ICONS (markup is constant, never user data). */
function iconButton(icon, attrs) {
  const btn = h("button", { type: "button", ...attrs });
  btn.innerHTML = ICONS[icon];
  return btn;
}

const PIN_SVG =
  '<svg viewBox="0 0 30 40" width="30" height="40" aria-hidden="true">' +
  '<path d="M15 38.5S2.5 24 2.5 14.5a12.5 12.5 0 0 1 25 0C27.5 24 15 38.5 15 38.5z"/></svg>';

function createPin(lngLat) {
  const shape = h("span", { class: "dm-pin-shape" });
  shape.innerHTML = PIN_SVG; // static markup, no user data
  const el = h("div", { class: "dm-pin" }, shape, h("span", { class: "dm-pin-label" }));
  const pin = { lngLat: lngLat.slice(), marker: null, el };
  el.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    e.stopPropagation();
    removePin(state.pins.indexOf(pin));
  });
  el.addEventListener("click", (e) => e.stopPropagation());
  if (state.map) {
    const marker = new window.maplibregl.Marker({ element: el, draggable: true, anchor: "bottom" })
      .setLngLat(lngLat)
      .addTo(state.map);
    marker.on("dragstart", () => {
      state.dragSnapshot = snapshot();
      el.classList.add("dragging");
    });
    marker.on("dragend", () => {
      el.classList.remove("dragging");
      state.lastDragEnd = Date.now();
      const ll = marker.getLngLat();
      pushHistory(state.dragSnapshot || snapshot());
      state.dragSnapshot = null;
      pin.lngLat = [ll.lng, ll.lat];
      state.lastEdit = { pin };
      afterPinsChanged();
    });
    pin.marker = marker;
  }
  return pin;
}

function snapshot() {
  return { pins: state.pins.map((p) => p.lngLat.slice()), reversed: state.reversed, loop: state.loop };
}

function pushHistory(snap) {
  state.lastEdit = null; // set again by a click or drag that placed one point
  state.history.push(snap || snapshot());
  if (state.history.length > 100) state.history.shift();
}

function setPins(lngLats) {
  for (const p of state.pins) if (p.marker) p.marker.remove();
  state.pins = lngLats.map((ll) => createPin(ll));
}

function pinTitle(role, loop) {
  if (role === "start") return loop ? "Start and finish" : "Start";
  if (role === "finish") return "Finish";
  return "Then through here";
}

function addPin(lngLat, between, at) {
  pushHistory();
  const points = state.pins.map((p) => p.lngLat);
  let index = points.length;
  if (Number.isInteger(at)) index = clamp(at, 0, points.length);
  else if (between && points.length >= 2) index = bestInsertIndex(points, lngLat, state.loop);
  const pin = createPin(lngLat);
  state.pins.splice(index, 0, pin);
  state.lastEdit = { pin };
  afterPinsChanged();
}

/** Put the points where the server snapped them, so each visibly sits on its road. */
function snapPinsToRoad(snapped) {
  if (state.dragSnapshot) return; // a point is being dragged: it re-snaps when dropped
  const n = state.pins.length;
  const pts = displaySnapped(snapped, n, state.reversed, state.loop && n >= 2);
  if (!pts) return;
  let moved = false;
  state.pins.forEach((pin, i) => {
    if (distanceM(pin.lngLat, pts[i]) < 0.05) return;
    pin.lngLat = pts[i];
    if (pin.marker) pin.marker.setLngLat(pts[i]);
    moved = true;
  });
  if (moved) {
    renderPins();
    store.set(STORE_KEYS.pins, snapshot());
  }
}

/** A click or drag put point `farPin` (request index) too far from any road: take that edit
 * back and say why. False if that point wasn't the one just placed. */
function takeBackFarPin(farPin, message) {
  const n = state.pins.length;
  if (!Number.isInteger(farPin) || farPin < 0 || farPin >= n || !state.lastEdit) return false;
  const pin = state.pins[displayIndex(farPin, n, state.reversed, state.loop && n >= 2)];
  if (pin !== state.lastEdit.pin || !state.history.length) return false;
  const snap = state.history.pop();
  state.lastEdit = null;
  setPins(snap.pins);
  state.reversed = !!snap.reversed;
  state.loop = !!snap.loop;
  afterPinsChanged();
  toast(message, "error");
  return true;
}

function removePin(index) {
  if (index < 0 || index >= state.pins.length) return;
  pushHistory();
  const [pin] = state.pins.splice(index, 1);
  if (pin.marker) pin.marker.remove();
  afterPinsChanged();
}

function undo() {
  const snap = state.history.pop();
  if (!snap) return;
  setPins(snap.pins);
  state.reversed = !!snap.reversed;
  state.loop = !!snap.loop;
  afterPinsChanged();
}

function reverseRoute() {
  if (state.pins.length < 2) return;
  pushHistory();
  state.pins = reversedPins(state.pins, state.loop);
  state.reversed = !state.reversed;
  afterPinsChanged();
}

/** "Back to start": close the route back to A (a loop), or open it again (A to B). */
function toggleLoop() {
  if (state.pins.length < 2) return;
  pushHistory();
  state.loop = !state.loop;
  afterPinsChanged();
}

function clearPins() {
  if (!state.pins.length) return;
  pushHistory();
  setPins([]);
  setWaypointMode(false);
  afterPinsChanged();
}

function afterPinsChanged() {
  if (state.pins.length < 2) {
    setWaypointMode(false);
    state.reversed = false;
    state.loop = false;
  }
  renderPins();
  store.set(STORE_KEYS.pins, snapshot());
  scheduleRoute();
}

/** Insert mode (W key): plain clicks add points in between, like Shift+click. */
function setWaypointMode(on) {
  state.waypointMode = !!on && state.pins.length >= 2;
  $("map-mode").hidden = !state.waypointMode;
}

function onMapClick(e) {
  const target = e.originalEvent && e.originalEvent.target;
  if (target && target.closest && target.closest(".dm-pin")) return;
  if (target && target.closest && target.closest(".tbt-arrow")) return;
  if (Date.now() - state.lastDragEnd < 300) return;
  closeSearch();
  if (state.mode === "tbt") {
    tbtClick(e);
    return;
  }
  const shift = !!(e.originalEvent && e.originalEvent.shiftKey);
  const between = state.waypointMode || shift;
  const alt = between ? -1 : altAt(e.point);
  if (alt >= 0) {
    selectOption(alt);
    return;
  }
  addPin([e.lngLat.lng, e.lngLat.lat], between);
}

/** The route option whose grey line is under screen point `pt` (-1: none, or the selected
 * route's line is on top there). */
function altAt(pt) {
  const map = state.map;
  if (!map || !map.getLayer("dm-alt-hit") || state.routeStale) return -1;
  if (map.queryRenderedFeatures(pt, { layers: ["dm-route-hit"] }).length) return -1;
  const hits = map.queryRenderedFeatures(pt, { layers: ["dm-alt-hit"] });
  const i = hits.length ? Number(hits[0].properties.index) : -1;
  return Number.isInteger(i) && i < state.routeOptions.length ? i : -1;
}

function setMapHover(alt, onRoute) {
  setAltHover(alt);
  const canvas = state.map && state.map.getCanvas();
  if (canvas) canvas.style.cursor = state.lineDrag ? "grabbing" : alt >= 0 ? "pointer" : onRoute ? "move" : "";
}

function onMapMouseMove(e) {
  const map = state.map;
  if (!map) return;
  if (state.mode === "tbt") {
    tbtMouseMove(e);
    return;
  }
  const drag = state.lineDrag;
  if (drag) {
    if (!drag.moved && Math.hypot(e.point.x - drag.start.x, e.point.y - drag.start.y) <= LINE_DRAG_PX) return;
    drag.moved = true;
    const cur = [e.lngLat.lng, e.lngLat.lat];
    const n = state.pins.length;
    const prev = state.pins[drag.index - 1];
    const next = state.pins[drag.index % n];
    const legs = [prev, null, next].map((p) => (p ? p.lngLat : cur));
    setSourceData("dm-drag", {
      type: "FeatureCollection",
      features: [
        { type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: legs } },
        { type: "Feature", properties: {}, geometry: { type: "Point", coordinates: cur } },
      ],
    });
    setMapHover(-1, true);
    return;
  }
  if (!map.getLayer("dm-route-hit")) return;
  const onRoute = !!state.route && !state.routeStale && map.queryRenderedFeatures(e.point, { layers: ["dm-route-hit"] }).length > 0;
  setMapHover(onRoute ? -1 : altAt(e.point), onRoute);
}

/** Press on the selected route's line: dragging it adds a point there (between the points
 * either side along the route), Google Maps style. */
function onRouteLineDown(e) {
  const ev = e.originalEvent;
  const r = state.route;
  if (state.mode !== "quick") return;
  if (!r || state.routeStale || state.pins.length < 2 || !Array.isArray(r.line) || r.line.length < 2) return;
  if ((ev && ev.button !== 0) || (ev && ev.shiftKey) || state.waypointMode) return;
  e.preventDefault(); // no map panning while dragging the line
  const loop = state.loop && state.pins.length >= 2;
  const index = insertIndexAlong(r.line, state.pins.map((p) => p.lngLat), [e.lngLat.lng, e.lngLat.lat], loop);
  state.lineDrag = { index, start: e.point, moved: false };
}

function onRouteLineUp(e) {
  const drag = state.lineDrag;
  if (!drag) return;
  state.lineDrag = null;
  setSourceData("dm-drag", EMPTY_FC);
  setMapHover(-1, false);
  if (!drag.moved) return; // a plain click: handled as a click
  state.lastDragEnd = Date.now();
  closeSearch();
  addPin([e.lngLat.lng, e.lngLat.lat], true, drag.index);
}

function cancelLineDrag() {
  if (!state.lineDrag) return;
  state.lineDrag = null;
  setSourceData("dm-drag", EMPTY_FC);
  setMapHover(-1, false);
}

function renderPins() {
  const n = state.pins.length;
  const loop = state.loop && n >= 2;
  const list = $("pin-list");
  list.replaceChildren();
  state.pins.forEach((pin, i) => {
    const role = pinRole(i, n, loop);
    const label = pinLetter(i);
    const title = pinTitle(role, loop);
    // Only our own classes change: MapLibre's (maplibregl-marker: position absolute) stay, or
    // every point after A is drawn a pin's height lower than where it is.
    const cls = pin.el.classList;
    for (const c of [...cls]) if (c.startsWith("dm-pin--")) cls.remove(c);
    cls.add("dm-pin", `dm-pin--${role}`);
    if (loop && role === "start") cls.add("dm-pin--loop");
    pin.el.querySelector(".dm-pin-label").textContent = label;
    pin.el.title = `${label}: ${title}. Drag to move, right-click to remove.`;
    const coords = `${pin.lngLat[1].toFixed(5)}, ${pin.lngLat[0].toFixed(5)}`;
    list.append(
      h(
        "li",
        { class: "pin-item" },
        h("span", { class: `pin-badge ${role}${loop && role === "start" ? " loop" : ""}`, "aria-hidden": "true" }, label),
        h("span", { class: "pin-text" }, h("strong", {}, title), h("span", {}, coords)),
        h(
          "span",
          { class: "pin-actions" },
          iconButton("locate", {
            class: "icon-btn",
            title: "Show on map",
            "aria-label": `Show point ${label} on the map`,
            onclick: () =>
              state.map && state.map.flyTo({ center: pin.lngLat, zoom: Math.max(state.map.getZoom(), 15) }),
          }),
          iconButton("close", {
            class: "icon-btn danger",
            title: "Remove",
            "aria-label": `Remove point ${label}`,
            onclick: () => removePin(state.pins.indexOf(pin)),
          }),
        ),
      ),
    );
  });
  if (loop) {
    list.append(
      h(
        "li",
        { class: "pin-item pin-back" },
        h("span", { class: "pin-badge start loop", "aria-hidden": "true" }, "A"),
        h("span", { class: "pin-text" }, h("strong", {}, "Back to the start"), h("span", {}, "A loop: lap it as many times as you like")),
      ),
    );
  }

  let hint;
  if (n === 0) hint = "Click the map where your road starts (A). Each click adds the next point: B, C, D…";
  else if (n === 1) hint = "Now click further along the road for point B.";
  else if (loop)
    hint = `Click to add point ${pinLetter(n)}; the road comes back to A after it. Shift+click or drag the blue line to add a point in between; drag points to move them, right-click to remove one.`;
  else
    hint = `Click to add point ${pinLetter(n)}; the last point is the finish. Shift+click or drag the blue line to add a point in between; drag points to move them, right-click to remove one.`;
  $("route-hint").textContent = hint;

  const loopBtn = $("btn-loop");
  loopBtn.disabled = n < 2;
  loopBtn.setAttribute("aria-pressed", String(loop));
  $("loop-state").textContent = loop ? "On" : "Off";
  loopBtn.title = n < 2 ? "Add at least two points first" : loop ? "Make it A to B again" : "Come back to A along the roads: a loop you can lap";
  $("btn-reverse").disabled = n < 2;
  renderHeadTools();
}

/* ------------------------------------------------------------------------------------------
 * Route
 * ---------------------------------------------------------------------------------------- */

function scheduleRoute() {
  clearTimeout(state.routeTimer);
  if (state.routeAbort) state.routeAbort.abort();
  state.routeAbort = null;
  state.routeSeq++;
  state.routeOptions = []; // the old options don't go through the new points
  state.optionIndex = 0;
  state.altHover = -1;
  setSourceData("dm-alts", EMPTY_FC);
  renderRouteOptions();
  if (state.pins.length < 2) {
    state.route = null;
    state.routeCum = null;
    state.routeStale = false;
    state.routeError = "";
    setBusy(false);
    drawRoute();
    renderRouteInfo();
    resetProfile();
    updateBuildPanel();
    return;
  }
  state.routeStale = true;
  state.routeError = "";
  applyStaleStyle();
  renderRouteInfo();
  updateBuildPanel();
  state.routeTimer = setTimeout(requestRoute, 400);
}

async function requestRoute() {
  const seq = state.routeSeq;
  const ctrl = new AbortController();
  state.routeAbort = ctrl;
  setBusy(true);
  const loop = state.loop && state.pins.length >= 2;
  const body = {
    pins: apiPins(state.pins.map((p) => p.lngLat), state.reversed, loop),
    reverse: state.reversed,
    loop,
  };
  try {
    const route = await api(API.route, { method: "POST", body, signal: ctrl.signal });
    if (seq !== state.routeSeq) return;
    const alts = Array.isArray(route.alternatives)
      ? route.alternatives.filter((o) => o && o.route_id && Array.isArray(o.line))
      : [];
    state.routeOptions = alts.length ? alts : [route];
    state.routeSnapped = route.snapped;
    state.routeStale = false;
    state.routeError = "";
    snapPinsToRoad(route.snapped);
    selectOption(0, true);
  } catch (err) {
    if ((err && err.name === "AbortError") || seq !== state.routeSeq) return;
    // A point just clicked or dragged off the roads: put things back as they were.
    if (takeBackFarPin(err.data && err.data.far_pin, err.message)) return;
    state.route = null;
    state.routeCum = null;
    state.routeStale = false;
    state.routeError = err.message || "Couldn't find a route.";
    drawRoute();
    renderRouteInfo();
    renderRouteOptions();
    resetProfile("No route yet, so no elevation profile.");
  } finally {
    if (seq === state.routeSeq) {
      state.routeAbort = null;
      setBusy(false);
      updateBuildPanel();
    }
  }
}

/** Show route option i (wraps round): the blue line, the profile, the name and the build all
 * use it. */
function selectOption(i, force) {
  const opts = state.routeOptions;
  if (!opts.length || (state.routeStale && !force)) return;
  const k = ((i % opts.length) + opts.length) % opts.length;
  if (!force && k === state.optionIndex && state.route && state.route.route_id === opts[k].route_id) return;
  const opt = opts[k];
  state.optionIndex = k;
  state.altHover = -1;
  state.route = { ...opt, snapped: state.routeSnapped || [] };
  state.routeCum = cumulativeM(Array.isArray(opt.line) ? opt.line : []);
  drawRoute();
  renderRouteInfo();
  renderRouteOptions();
  applyDefaultName();
  loadProfile(opt.route_id);
  updateBuildPanel();
}

function renderRouteOptions() {
  const box = $("route-options");
  const list = $("route-option-list");
  const opts = state.routeOptions;
  box.hidden = opts.length < 2;
  list.replaceChildren(
    ...opts.map((opt, i) => {
      const selected = i === state.optionIndex;
      const parts = routeOptionLabel(opt, i).split(" · ");
      const btn = h(
        "button",
        {
          type: "button",
          class: `route-option${selected ? " selected" : ""}`,
          "aria-pressed": String(selected),
          title: selected ? "The route you'll build" : "Use this route",
          onclick: () => selectOption(i),
          onmouseenter: () => setAltHover(i),
          onmouseleave: () => setAltHover(-1),
        },
        h("span", { class: "opt-num" }, parts[0]),
        " · ",
        h("span", { class: "opt-len" }, parts[1]),
        ...(parts.length > 2 ? [" · ", h("span", { class: "opt-names" }, parts.slice(2).join(" · "))] : []),
      );
      return h("li", {}, btn);
    }),
  );
}

function setBusy(on) {
  $("map-busy").hidden = !on;
}

function renderRouteInfo() {
  const status = $("route-status");
  status.classList.toggle("error", !!state.routeError);
  status.textContent = state.routeError
    ? state.routeError
    : state.routeStale && state.mode !== "tbt"
      ? "Finding the road…"
      : "";
  const r = state.route;
  $("route-info").hidden = !r;
  if (!r) return;
  $("route-length-label").textContent = r.loop ? "Lap length" : "Length";
  $("route-length").textContent = formatLength(Number(r.length_m));
  const names = Array.isArray(r.names) ? r.names.filter(Boolean) : [];
  const shown = names.slice(0, 10);
  $("route-names").replaceChildren(
    ...shown.map((n) => h("li", {}, n)),
    ...(names.length > shown.length ? [h("li", {}, `+${names.length - shown.length} more`)] : []),
  );
  const warnings = Array.isArray(r.warnings) ? r.warnings.filter(Boolean) : [];
  $("route-warnings").replaceChildren(...warnings.map((w) => h("li", {}, w)));
}

function applyDefaultName() {
  const input = $("track-name");
  const name = state.route ? defaultTrackName(state.route.names, !!state.route.loop) : "";
  input.placeholder = name || "Snake Road";
  if (!state.nameTouched) input.value = name;
}

/* ------------------------------------------------------------------------------------------
 * Turn by turn (in the browser): roads in z14 tiles, the route as legs, arrows at junctions
 *
 * The route is the start (a point on one road segment) plus legs: the first leg's nodes are the
 * start segment's two ends in the chosen driving direction and on to the next junction; each
 * later leg is the nodes after the junction it leaves from. `end` finishes it (t along the last
 * segment) or closes it back to A (`loop`). Every edit pushes a snapshot, so Backspace undoes.
 * ---------------------------------------------------------------------------------------- */

const TBT_PICK_PX = 14; // a click this close (screen pixels) to a road picks it
const TBT_ARROW_PX = 58; // the arrows sit this far from the junction
const TBT_STUB_M = 80; // the start of each road out of a junction, drawn under its arrow
const TILE_RETRY_MS = 20000; // a tile that failed to load is asked for again after this

function tbtNew() {
  return {
    g: createRoadGraph(),
    tiles: new Map(), // "14/x/y" -> "loading" | "ok" | "error"
    failedAt: new Map(),
    want: [], // tiles to load next, nearest first
    loading: null, // {since, keys}: the /api/roads request in flight
    chipTimer: 0,
    version: 0, // bumped whenever roads arrive
    start: null, // {a, b, w, t, point}: segment a-b of way w, t from a
    legs: [], // [{kind: "start" | "turn" | "close", nodes, way, name, angle}]
    end: null, // {t, loop}
    history: [],
    branchCache: null,
    markers: [],
    markerSig: "",
    pins: { a: null, b: null },
    hot: -1, // the arrow under the mouse: its whole road is highlighted
    hoverPlan: null,
    hoverRaf: 0,
    lastMouse: null,
    errorShown: 0,
    retried: false,
    saved: null, // the route saved in the browser, waiting for its roads (see tbtTryRestore)
  };
}

function tbtIsLoaded(id) {
  const t = state.tbt;
  const p = t.g.nodes.get(id);
  return !!p && t.tiles.get(tileKeyAt(p[0], p[1])) === "ok";
}

function tbtPath() {
  const out = [];
  for (const leg of state.tbt.legs) for (const id of leg.nodes) out.push(id);
  return out;
}

function tbtPhase() {
  const t = state.tbt;
  if (!t || !t.start) return "start";
  if (!t.legs.length) return "direction";
  return t.end ? "done" : "drive";
}

/** How far along the first segment (in driving order) the route starts. */
function tbtStartT() {
  const t = state.tbt;
  if (!t.start || !t.legs.length) return 0;
  return t.legs[0].nodes[0] === t.start.a ? t.start.t : 1 - t.start.t;
}

/** The route as drawn: from the start point to the finish, or round to the start again. */
function tbtLine() {
  const t = state.tbt;
  if (!t || !t.legs.length) return [];
  const st = tbtStartT();
  const line = pathLine(t.g, tbtPath(), st, t.end ? t.end.t : 1);
  if (t.end && t.end.loop && st > 0 && line.length) line.push(t.start.point.slice());
  return line;
}

function tbtTip() {
  const path = tbtPath();
  return path.length ? path[path.length - 1] : null;
}

function tbtSnapshot() {
  const t = state.tbt;
  return JSON.parse(JSON.stringify({ start: t.start, legs: t.legs, end: t.end }));
}

function tbtPush() {
  const t = state.tbt;
  t.history.push(tbtSnapshot());
  if (t.history.length > 300) t.history.shift();
}

function tbtRestore(snap) {
  const t = state.tbt;
  t.start = snap.start;
  t.legs = snap.legs;
  t.end = snap.end;
}

function tbtUndo() {
  const snap = state.tbt.history.pop();
  if (snap) tbtRestore(snap);
  else if (!tbtStepBack()) return;
  tbtChanged({ keepView: true });
}

/** Undo with no history (a route put back after a reload): one step back along the route. */
function tbtStepBack() {
  const t = state.tbt;
  if (t.end) {
    const last = t.legs[t.legs.length - 1];
    if (t.end.loop && t.legs.length > 1 && last.kind === "close") t.legs.pop();
    t.end = null;
    tbtSettle(); // a finish mid-road drives on to the junction again
  } else if (t.legs.length) {
    t.legs.pop();
  } else if (t.start) {
    t.start = null;
  } else {
    return false;
  }
  return true;
}

function tbtClear() {
  if (!state.tbt.start) return;
  tbtPush();
  tbtRestore({ start: null, legs: [], end: null });
  tbtChanged({ keepView: true });
}

/** Cut the legs so the path has `len` nodes. */
function tbtTruncate(len) {
  const t = state.tbt;
  const legs = [];
  let count = 0;
  for (const leg of t.legs) {
    if (count >= len) break;
    const take = Math.min(leg.nodes.length, len - count);
    legs.push({ ...leg, nodes: leg.nodes.slice(0, take) });
    count += take;
  }
  t.legs = legs;
}

/** The two ways you can drive off from the start point. */
function tbtDirections() {
  const t = state.tbt;
  const s = t.start;
  if (!s) return [];
  const g = t.g;
  const name = wayName(g.ways.get(s.w));
  return [
    [s.a, s.b, s.t],
    [s.b, s.a, 1 - s.t],
  ].map(([from, to, tt]) => {
    const e = (g.adj.get(from) || []).find((x) => x.to === to && x.way === s.w);
    const nodes = [from, to, ...continueRun(g, from, to, tbtIsLoaded)];
    const coords = pathLine(g, nodes, tt, 1);
    const bearing = bearingDeg(s.point, pointAtDistance(coords, 25));
    return { nodes, way: s.w, name, allowed: !e || canDrive(g, e), bearing, coords };
  });
}

/** The roads you can take at the junction the route has reached (cached until it changes). */
function tbtBranches() {
  const t = state.tbt;
  if (tbtPhase() !== "drive") return [];
  const path = tbtPath();
  const n = path.length;
  const sig = `${t.version}|${t.legs.length}|${n}|${path[n - 2]}|${path[n - 1]}`;
  if (t.branchCache && t.branchCache.sig === sig) return t.branchCache.list;
  const list = tbtIsLoaded(path[n - 1]) ? branchesAt(t.g, path, tbtIsLoaded) : [];
  t.branchCache = { sig, list };
  return list;
}

/** A leg that stopped where the roads weren't loaded yet drives on to the real junction once
 * they are. True if the route grew. */
function tbtSettle() {
  const t = state.tbt;
  if (t.end || !t.legs.length) return false;
  let grew = false;
  for (let guard = 0; guard < 50; guard++) {
    const path = tbtPath();
    const n = path.length;
    const tip = path[n - 1];
    if (!tbtIsLoaded(tip)) break;
    const more = continueRun(t.g, path[n - 2], tip, tbtIsLoaded);
    if (!more.length) break;
    t.legs[t.legs.length - 1].nodes.push(...more);
    grew = true;
  }
  return grew;
}

function tbtAddLeg(leg, end) {
  tbtPush();
  state.tbt.legs.push(leg);
  if (end) state.tbt.end = end;
  tbtSettle();
  tbtChanged();
}

function tbtLegFor(br, nodes) {
  return { kind: "turn", nodes, way: br.way, name: br.name, angle: Math.round(br.angle) };
}

function tbtTake(br) {
  if (!br || !br.allowed || tbtPhase() !== "drive") return;
  tbtAddLeg(tbtLegFor(br, br.run.slice()));
}

function tbtPickDirection(d) {
  if (!d || !d.allowed || tbtPhase() !== "direction") return;
  tbtAddLeg({ kind: "start", nodes: d.nodes.slice(), way: d.way, name: d.name, angle: 0 });
}

function tbtSetStart(seg) {
  const t = state.tbt;
  const pa = t.g.nodes.get(seg.a);
  const pb = t.g.nodes.get(seg.b);
  const tt = clamp(seg.t, 0.001, 0.999);
  tbtPush();
  t.start = { a: seg.a, b: seg.b, w: seg.w, t: tt, point: [pa[0] + (pb[0] - pa[0]) * tt, pa[1] + (pb[1] - pa[1]) * tt] };
  t.legs = [];
  t.end = null;
  const ok = tbtDirections().filter((d) => d.allowed);
  if (ok.length === 1) {
    // A one-way street: there is only one way to go.
    t.legs.push({ kind: "start", nodes: ok[0].nodes.slice(), way: ok[0].way, name: ok[0].name, angle: 0 });
    tbtSettle();
    toast(`${ok[0].name || "This road"} is one-way, so you drive it the way it goes.`);
  }
  tbtChanged();
}

function tbtFinish() {
  if (tbtPhase() !== "drive") return;
  tbtPush();
  state.tbt.end = { t: 1, loop: false };
  tbtChanged({ keepView: true });
}

/** The path closed back to A, when the route has come back round to A's road. */
function tbtLoopPath() {
  if (tbtPhase() !== "drive") return null;
  return closeLoopPath(state.tbt.g, tbtPath(), tbtIsLoaded);
}

function tbtCloseLoop() {
  const t = state.tbt;
  if (t.end && t.end.loop) {
    tbtUndo(); // pressed again: open the loop (closing it was the last change)
    return;
  }
  const closed = tbtLoopPath();
  if (!closed) return;
  const path = tbtPath();
  tbtPush();
  if (closed.length > path.length) {
    const nodes = closed.slice(path.length);
    const br = tbtBranches().find((b) => b.to === nodes[0]);
    t.legs.push({
      kind: "close",
      nodes,
      way: br ? br.way : edgeWay(t.g, path[path.length - 1], nodes[0]),
      name: br ? br.name : wayName(t.g.ways.get(t.start.w)),
      angle: br ? Math.round(br.angle) : 0,
    });
  } else {
    tbtTruncate(closed.length);
  }
  t.end = { t: 1, loop: true };
  tbtChanged({ keepView: true });
}

/** Metres per screen pixel at latitude `lat` at the map's zoom (512 px tiles). */
function metresPerPixel(lat) {
  const z = state.map ? state.map.getZoom() : 15;
  return (40075016.686 * Math.cos(toRad(lat))) / (512 * 2 ** z);
}

function tbtSegAt(lngLat) {
  const p = [lngLat.lng, lngLat.lat];
  return nearestSegment(state.tbt.g, p, clamp(TBT_PICK_PX * metresPerPixel(p[1]), 3, 300));
}

/** Where segment `seg` is from the end of the route: on the last leg (`behind`: cut = its
 * index in the path) or ahead ({nodes to add, cut = the segment's end in nodes, branch});
 * `t` is how far along the segment the click is, in driving order. Null: not reachable. */
function tbtPlanTo(seg) {
  const t = state.tbt;
  const path = tbtPath();
  const n = path.length;
  const last = t.legs[t.legs.length - 1];
  const from = Math.max(1, n - last.nodes.length);
  for (let k = n - 1; k >= from; k--) {
    const fwd = path[k - 1] === seg.a && path[k] === seg.b;
    if (fwd || (path[k - 1] === seg.b && path[k] === seg.a)) return { behind: true, cut: k, t: fwd ? seg.t : 1 - seg.t };
  }
  const ahead = findAhead(t.g, path, seg, tbtIsLoaded);
  if (!ahead) return null;
  return { behind: false, nodes: ahead.nodes, cut: ahead.cut, t: ahead.forward ? seg.t : 1 - seg.t, branch: ahead.branch };
}

function tbtDriveClick(seg, shift) {
  const plan = tbtPlanTo(seg);
  if (!plan) {
    toast("You can't get there from the end of your route. Pick the next road with the arrows, or click further along a road you can turn onto.");
    return;
  }
  const t = clamp(plan.t, 0.001, 1);
  if (plan.behind) {
    if (!shift) {
      toast("That's on your route already. Shift+click it to finish there.");
      return;
    }
    if (plan.cut === 1 && t <= tbtStartT() + 0.01) {
      toast("Finish after the start (A).");
      return;
    }
    tbtPush();
    tbtTruncate(plan.cut + 1);
    state.tbt.end = { t, loop: false };
    tbtChanged({ keepView: true });
    return;
  }
  if (shift) tbtAddLeg(tbtLegFor(plan.branch, plan.nodes.slice(0, plan.cut + 1)), { t, loop: false });
  else tbtAddLeg(tbtLegFor(plan.branch, plan.nodes));
}

function tbtClick(e) {
  const t = state.tbt;
  const shift = !!(e.originalEvent && e.originalEvent.shiftKey);
  const phase = tbtPhase();
  const seg = tbtSegAt(e.lngLat);
  if (!seg) {
    if (state.map.getZoom() < ROADS_MIN_ZOOM) toast("Zoom in closer to pick a road.");
    else if (t.loading) toast("The roads here are still loading. Try again in a moment.");
    else if (phase === "start" || phase === "direction") toast("Click right on a road to start there.");
    return;
  }
  if (phase === "start" || phase === "direction") {
    tbtSetStart(seg);
  } else if (phase === "done") {
    toast(t.end.loop ? "The loop is closed. Press Backspace to open it again." : "Your route is finished. Press Backspace to keep driving.");
  } else {
    tbtDriveClick(seg, shift);
  }
}

/* -- hover: the road under the mouse, or where a click would take the route ---------------- */

function tbtMouseMove(e) {
  const t = state.tbt;
  t.lastMouse = { lngLat: e.lngLat, shift: !!(e.originalEvent && e.originalEvent.shiftKey) };
  if (!t.hoverRaf) t.hoverRaf = requestAnimationFrame(tbtUpdateHover);
}

function tbtHoverPlan(seg, phase) {
  const t = state.tbt;
  const key = `${t.version}|${t.history.length}|${t.legs.length}|${phase}|${seg.a}|${seg.b}`;
  if (t.hoverPlan && t.hoverPlan.key === key) return t.hoverPlan;
  let plan = { key, kind: "none" };
  if (phase === "start" || phase === "direction") {
    plan = { key, kind: "road", coords: coordsOf(t.g, runAround(t.g, seg.a, seg.b)) };
  } else if (phase === "drive") {
    const p = tbtPlanTo(seg);
    if (p) plan = { key, kind: p.behind ? "behind" : "ahead", p };
  }
  t.hoverPlan = plan;
  return plan;
}

function tbtUpdateHover() {
  const t = state.tbt;
  t.hoverRaf = 0;
  const m = t.lastMouse;
  const features = [];
  let pointer = false;
  const phase = tbtPhase();
  if (m && state.mode === "tbt" && phase !== "done" && state.map) {
    const seg = tbtSegAt(m.lngLat);
    if (seg) {
      const plan = tbtHoverPlan(seg, phase);
      const line = (coords, kind) => ({ type: "Feature", properties: { kind }, geometry: { type: "LineString", coordinates: coords } });
      const dot = (p) => ({ type: "Feature", properties: { kind: "finish" }, geometry: { type: "Point", coordinates: p } });
      if (plan.kind === "road" && plan.coords.length >= 2) {
        features.push(line(plan.coords, "road"));
        pointer = true;
      } else if (plan.kind === "ahead") {
        const g = t.g;
        const tip = g.nodes.get(tbtTip());
        const p = plan.p;
        const coords = m.shift
          ? [tip, ...coordsOf(g, p.nodes.slice(0, p.cut)), seg.point]
          : [tip, ...coordsOf(g, p.nodes)];
        features.push(line(coords, "ahead"));
        if (m.shift) features.push(dot(seg.point));
        pointer = true;
      } else if (plan.kind === "behind" && m.shift) {
        features.push(dot(seg.point));
        pointer = true;
      }
    }
  }
  setSourceData("dm-tbt-hover", features.length ? { type: "FeatureCollection", features } : EMPTY_FC);
  const canvas = state.map && state.map.getCanvas();
  if (canvas && state.mode === "tbt") canvas.style.cursor = pointer ? "pointer" : "";
}

function tbtHoverOff() {
  const t = state.tbt;
  if (!t) return;
  t.lastMouse = null;
  setSourceData("dm-tbt-hover", EMPTY_FC);
}

/* -- the arrows: the ways to go from the start point or the junction ---------------------- */

/** What the arrows offer now: {at: [lon, lat], items: [{key, title, allowed, bearing, coords,
 * act}]}. */
function tbtChoices() {
  const t = state.tbt;
  const phase = tbtPhase();
  if (phase === "direction") {
    const dirs = tbtDirections();
    return {
      at: t.start.point,
      items: dirs.map((d, i) => ({
        key: d.allowed ? KEY_GLYPH[keyForBearing(d.bearing)] : "",
        num: i + 1,
        title: d.allowed ? `Drive ${compassWord(d.bearing)}${d.name ? ` on ${d.name}` : ""}` : `${d.name || "This road"} is one-way the other way`,
        icon: d.bearing,
        allowed: d.allowed,
        bearing: d.bearing,
        coords: d.coords,
        act: () => tbtPickDirection(d),
      })),
    };
  }
  if (phase === "drive") {
    const brs = tbtBranches();
    const keys = branchKeyLabels(brs);
    const at = t.g.nodes.get(tbtTip());
    return {
      at,
      items: brs.map((b, i) => ({
        key: keys[i],
        num: i + 1,
        title: b.allowed
          ? legLabel({ kind: "turn", name: b.name, angle: b.angle }, wayName(t.g.ways.get(t.legs[t.legs.length - 1].way)))
          : `${b.name || "This road"}: one-way, no entry`,
        icon: b.angle,
        allowed: b.allowed,
        bearing: b.bearing,
        coords: b.coords,
        act: () => tbtTake(b),
      })),
    };
  }
  return { at: null, items: [] };
}

function tbtClearMarkers() {
  const t = state.tbt;
  for (const m of t.markers) m.remove();
  t.markers = [];
  t.markerSig = "";
}

const ARROW_SVG =
  '<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">' +
  '<path d="M12 20V5M5.5 11.5L12 5l6.5 6.5"/></svg>';
const NO_ENTRY_SVG =
  '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round">' +
  '<path d="M6 12h12"/></svg>';

function tbtArrowIcon(deg, allowed) {
  const span = h("span", { class: "tbt-icon", style: allowed ? `transform: rotate(${Math.round(deg)}deg)` : null });
  span.innerHTML = allowed ? ARROW_SVG : NO_ENTRY_SVG; // static markup, no user data
  return span;
}

function tbtSetHot(i) {
  const t = state.tbt;
  if (t.hot === i) return;
  t.hot = i;
  setSourceData("dm-tbt-branches", tbtBranchData());
  for (const [k, m] of t.markers.entries()) m.getElement().classList.toggle("hot", k === i);
  const list = $("tbt-choice-list");
  [...list.querySelectorAll("button")].forEach((b, k) => b.classList.toggle("hot", k === i));
}

/** The arrow buttons on the map round the junction (or the start point). */
function tbtRenderMarkers() {
  const t = state.tbt;
  const map = state.map;
  if (!map || !window.maplibregl) return;
  const ch = state.mode === "tbt" ? tbtChoices() : { at: null, items: [] };
  const sig = ch.at ? `${ch.at.join(",")}|${ch.items.map((c) => `${c.bearing.toFixed(1)}/${c.allowed}/${c.key}`).join(";")}` : "";
  if (sig === t.markerSig) return;
  tbtClearMarkers();
  t.markerSig = sig;
  t.hot = -1;
  if (!ch.at) return;
  const angles = spreadAngles(ch.items.map((c) => c.bearing), 42);
  // Round the start point the arrows sit a little further out, clear of the A pin.
  const r = tbtPhase() === "direction" ? TBT_ARROW_PX + 10 : TBT_ARROW_PX;
  ch.items.forEach((c, i) => {
    const a = toRad(angles[i]);
    const el = h(
      "button",
      {
        type: "button",
        class: `tbt-arrow${c.allowed ? "" : " no"}`,
        title: c.allowed ? `${c.title}${c.key ? ` (${c.key})` : ""}` : c.title,
        "aria-label": c.title,
        disabled: !c.allowed,
        onclick: (e) => {
          e.stopPropagation();
          c.act();
        },
        onmouseenter: () => tbtSetHot(i),
        onmouseleave: () => tbtSetHot(-1),
      },
      tbtArrowIcon(angles[i], c.allowed),
      c.key ? h("span", { class: "tbt-key" }, c.key) : null,
    );
    el.addEventListener("mousedown", (e) => e.stopPropagation());
    const marker = new window.maplibregl.Marker({
      element: el,
      anchor: "center",
      offset: [Math.sin(a) * r, -Math.cos(a) * r],
    })
      .setLngLat(ch.at)
      .addTo(map);
    t.markers.push(marker);
  });
}

/** The first stretch of each road out of the junction (and the whole of the hovered one), plus
 * the junction itself. */
function tbtBranchData() {
  const t = state.tbt;
  if (!t || state.mode !== "tbt") return EMPTY_FC;
  const ch = tbtChoices();
  if (!ch.at) return EMPTY_FC;
  const features = ch.items.map((c, i) => ({
    type: "Feature",
    properties: { i, allowed: c.allowed, hot: i === t.hot },
    geometry: { type: "LineString", coordinates: i === t.hot ? c.coords : lineHead(c.coords, TBT_STUB_M) },
  }));
  features.push({ type: "Feature", properties: { i: -1 }, geometry: { type: "Point", coordinates: ch.at } });
  return { type: "FeatureCollection", features };
}

function tbtPin(role, label) {
  const shape = h("span", { class: "dm-pin-shape" });
  shape.innerHTML = PIN_SVG; // static markup, no user data
  const el = h("div", { class: `dm-pin dm-pin--fixed dm-pin--${role}` }, shape, h("span", { class: "dm-pin-label" }, label));
  return new window.maplibregl.Marker({ element: el, anchor: "bottom" });
}

/** A at the start, B at the finish (A doubles as the finish on a loop). */
function tbtRenderPins() {
  const t = state.tbt;
  const map = state.map;
  if (!map || !window.maplibregl) return;
  const show = state.mode === "tbt";
  if (show && t.start) {
    if (!t.pins.a) t.pins.a = tbtPin("start", "A");
    const loop = !!(t.end && t.end.loop);
    const el = t.pins.a.getElement();
    el.classList.toggle("dm-pin--loop", loop);
    el.title = loop ? "A: start and finish" : "A: start";
    t.pins.a.setLngLat(t.start.point);
    if (!el.isConnected) t.pins.a.addTo(map);
  } else if (t.pins.a) {
    t.pins.a.remove();
  }
  const line = tbtLine();
  if (show && t.end && !t.end.loop && line.length) {
    if (!t.pins.b) t.pins.b = tbtPin("finish", "B");
    t.pins.b.getElement().title = "B: finish";
    t.pins.b.setLngLat(line[line.length - 1]);
    if (!t.pins.b.getElement().isConnected) t.pins.b.addTo(map);
  } else if (t.pins.b) {
    t.pins.b.remove();
  }
}

function tbtRenderMap() {
  drawRoute();
  tbtRenderMarkers();
  tbtRenderPins();
  setSourceData("dm-tbt-branches", tbtBranchData());
  tbtUpdateHover();
}

function tbtClearMap() {
  const t = state.tbt;
  tbtClearMarkers();
  if (t.pins.a) t.pins.a.remove();
  if (t.pins.b) t.pins.b.remove();
  tbtHoverOff();
  setSourceData("dm-tbt-branches", EMPTY_FC);
}

/** Pan so the junction you're at stays on screen while you drive with the keys. */
function tbtKeepTipInView() {
  const map = state.map;
  const t = state.tbt;
  if (!map || !t.start || t.end) return;
  const p = t.legs.length ? t.g.nodes.get(tbtTip()) : t.start.point;
  if (!p) return;
  const q = map.project(p);
  const c = map.getCanvas();
  const pad = t.legs.length ? 90 : 120; // room for the arrows round the start point
  if (q.x < pad || q.y < pad || q.x > c.clientWidth - pad || q.y > c.clientHeight - pad) map.easeTo({ center: p, duration: 450 });
}

/* -- the sidebar --------------------------------------------------------------------------- */

const LEG_ICONS = {
  loop:
    '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M20 12a8 8 0 1 1-2.6-5.9"/><path d="M20 4v5h-5"/></svg>',
};

function tbtLegLengths() {
  const t = state.tbt;
  const g = t.g;
  const st = tbtStartT();
  const out = [];
  let prev = null;
  t.legs.forEach((leg, i) => {
    const last = i === t.legs.length - 1;
    const ids = i === 0 ? leg.nodes : [prev, ...leg.nodes];
    let len = lineLengthM(pathLine(g, ids, i === 0 ? st : 0, last && t.end ? t.end.t : 1));
    if (last && t.end && t.end.loop && st > 0) len += distanceM(g.nodes.get(leg.nodes[leg.nodes.length - 1]) || t.start.point, t.start.point);
    out.push(len);
    prev = leg.nodes[leg.nodes.length - 1];
  });
  return out;
}

function tbtRender() {
  const t = state.tbt;
  const phase = tbtPhase();
  const zoomedOut = !state.map || state.map.getZoom() < ROADS_MIN_ZOOM;
  let hint;
  if (phase === "start" && t.saved) {
    hint = "Loading your route…";
  } else if (phase === "start") {
    hint = zoomedOut
      ? "Zoom in on the map to where you want to drive, then click the road you start on."
      : "Click the road you want to start on.";
  } else if (phase === "direction") {
    hint = "Which way? Click an arrow on the map, or press the arrow key that points that way.";
  } else if (phase === "drive") {
    const brs = tbtBranches();
    const tip = t.g.nodes.get(tbtTip());
    if (!tbtIsLoaded(tbtTip())) {
      hint =
        tip && t.tiles.get(tileKeyAt(tip[0], tip[1])) === "error"
          ? "Couldn't load the roads ahead: the map servers are busy. Move the map a little to try again, or Finish here."
          : "Loading the roads ahead…";
    } else if (!brs.some((b) => b.allowed)) hint = "Dead end: there's no road you may take from here. Finish here, or press Backspace.";
    else hint = "Pick the next road: click an arrow, press ← ↑ →, or click further along a road to drive straight through junctions.";
  } else {
    hint = t.end.loop
      ? "Loop closed: the road comes back round to A. Create your map, or press Backspace to change it."
      : "Finished at B. Create your map, or press Backspace to keep driving.";
  }
  $("tbt-hint").textContent = hint;

  // The legs: "Start on Snake Road · 420 m", "Left onto Mountain Brow Road · 1.2 km", ...
  const lengths = tbtLegLengths();
  const legsEl = $("tbt-legs");
  let prevName = "";
  const groups = []; // straight on along the same road: one row
  t.legs.forEach((leg, i) => {
    const last = groups[groups.length - 1];
    const straight = leg.kind === "turn" && turnWord(leg.angle || 0) === "straight";
    if (last && straight && leg.name && leg.name === last.leg.name) last.len += lengths[i];
    else groups.push({ leg, len: lengths[i] });
  });
  const rows = groups.map(({ leg, len }) => {
    const label = legLabel(leg, prevName);
    prevName = leg.name;
    let icon;
    if (leg.kind === "start") icon = h("span", { class: "pin-badge start", "aria-hidden": "true" }, "A");
    else if (leg.kind === "close") {
      icon = h("span", { class: "leg-icon", "aria-hidden": "true" });
      icon.innerHTML = LEG_ICONS.loop; // static markup
    } else {
      icon = h("span", { class: "leg-icon", "aria-hidden": "true" }, tbtArrowIcon(leg.angle || 0, true));
    }
    return h("li", { class: `leg leg-${leg.kind}` }, icon, h("span", { class: "leg-text" }, label), h("span", { class: "leg-len" }, formatLength(len)));
  });
  if (t.end && !t.end.loop) {
    rows.push(h("li", { class: "leg leg-finish" }, h("span", { class: "pin-badge finish", "aria-hidden": "true" }, "B"), h("span", { class: "leg-text" }, "Finish")));
  }
  legsEl.replaceChildren(...rows);
  legsEl.scrollTop = legsEl.scrollHeight;

  // The choices, as buttons too (the same as the arrows on the map).
  const ch = tbtChoices();
  $("tbt-choices").hidden = !ch.items.length;
  $("tbt-choices-label").textContent = phase === "direction" ? "Which way?" : "Next road";
  $("tbt-choice-list").replaceChildren(
    ...ch.items.map((c, i) =>
      h(
        "li",
        {},
        h(
          "button",
          {
            type: "button",
            class: "tbt-choice",
            disabled: !c.allowed,
            title: c.title,
            onclick: () => c.act(),
            onmouseenter: () => tbtSetHot(i),
            onmouseleave: () => tbtSetHot(-1),
          },
          tbtArrowIcon(c.icon, c.allowed),
          h("span", { class: "tbt-choice-text" }, c.title),
          c.key ? h("kbd", {}, c.key) : null,
        ),
      ),
    ),
  );

  $("tbt-tools").hidden = phase === "start" || phase === "direction";
  const loopPath = tbtLoopPath();
  const closed = !!(t.end && t.end.loop);
  $("tbt-finish").disabled = phase !== "drive";
  const loopBtn = $("tbt-loop");
  loopBtn.disabled = !loopPath && !closed;
  loopBtn.setAttribute("aria-pressed", String(closed));
  loopBtn.title = loopPath ? "Close the loop back to A" : closed ? "Open the loop again" : "Drive back to A's road first";
  const loopHint = $("tbt-loop-hint");
  loopHint.textContent =
    phase !== "drive" ? "" : loopPath ? "You're back on A's road: Back to start closes the loop." : "Drive back to A to close the loop.";
  loopHint.classList.toggle("ready", !!loopPath);
  renderHeadTools();
}

/* -- roads: z14 tiles round the view and round the end of the route ------------------------ */

function tbtWantTiles() {
  const t = state.tbt;
  const map = state.map;
  if (!t || !map || state.mode !== "tbt") return;
  const want = t.saved ? t.saved.tiles.slice() : [];
  const tip = !t.end && t.legs.length ? t.g.nodes.get(tbtTip()) : null;
  if (tip) want.push(...tilesForBounds(tip[0], tip[1], tip[0], tip[1], 1));
  const z = map.getZoom();
  if (z >= ROADS_MIN_ZOOM) {
    const b = map.getBounds();
    want.push(...tilesForBounds(b.getWest(), b.getSouth(), b.getEast(), b.getNorth(), z >= 14 ? 1 : 0));
  }
  const now = Date.now();
  t.want = [...new Set(want)].filter((k) => {
    const s = t.tiles.get(k);
    return !s || (s === "error" && now - (t.failedAt.get(k) || 0) > TILE_RETRY_MS);
  });
  tbtPump();
  tbtRenderChip();
}

async function tbtPump() {
  const t = state.tbt;
  if (t.loading || state.mode !== "tbt") return;
  const keys = t.want.filter((k) => t.tiles.get(k) !== "ok" && t.tiles.get(k) !== "loading").slice(0, ROADS_BATCH);
  t.want = t.want.filter((k) => !keys.includes(k));
  if (!keys.length) {
    tbtRenderChip();
    return;
  }
  for (const k of keys) t.tiles.set(k, "loading");
  t.loading = { since: Date.now(), keys };
  tbtRenderChip();
  let problem = "";
  try {
    const res = await api(API.roads, { method: "POST", body: { tiles: keys } });
    addRoads(t.g, res);
    const got = (res && res.tiles) || {};
    for (const k of keys) {
      if (got[k] === "ok") t.tiles.set(k, "ok");
      else {
        t.tiles.set(k, "error");
        t.failedAt.set(k, Date.now());
      }
    }
    if (res && res.error) problem = String(res.error);
  } catch (err) {
    for (const k of keys) {
      t.tiles.set(k, "error");
      t.failedAt.set(k, Date.now());
    }
    problem = err.message;
  }
  t.loading = null;
  t.version++;
  if (problem && Date.now() - t.errorShown > 15000) {
    t.errorShown = Date.now();
    toast(problem, "error");
  }
  if (state.mode === "tbt") {
    if (tbtTryRestore()) {
      /* drawn by tbtChanged */
    } else if (tbtSettle()) tbtChanged();
    else {
      tbtRender();
      tbtRenderMap();
    }
  }
  tbtPump();
}

function tbtRenderChip() {
  const t = state.tbt;
  const chip = $("roads-chip");
  const stop = () => {
    clearInterval(t.chipTimer);
    t.chipTimer = 0;
  };
  if (state.mode !== "tbt" || !state.map) {
    chip.hidden = true;
    stop();
    return;
  }
  if (t.loading) {
    const s = Math.floor((Date.now() - t.loading.since) / 1000);
    $("roads-chip-text").textContent = `Loading roads… ${s} s`;
    $("roads-spinner").hidden = false;
    chip.dataset.kind = "loading";
    chip.hidden = false;
    if (!t.chipTimer) t.chipTimer = setInterval(tbtRenderChip, 1000);
    return;
  }
  stop();
  if (state.map.getZoom() < ROADS_MIN_ZOOM && tbtPhase() === "start") {
    $("roads-chip-text").textContent = "Zoom in to pick a road";
    $("roads-spinner").hidden = true;
    chip.dataset.kind = "zoom";
    chip.hidden = false;
    return;
  }
  chip.hidden = true;
}

/* -- the route: POST /api/route-path for the path as it stands ----------------------------- */

/** The route is kept in the browser, so a reload (or an app restart) doesn't lose it. */
function tbtSave() {
  const t = state.tbt;
  if (!t.start) {
    if (!t.saved) store.remove(STORE_KEYS.tbt);
    return;
  }
  const keys = new Set([tileKeyAt(t.start.point[0], t.start.point[1])]);
  for (const id of tbtPath()) {
    const p = t.g.nodes.get(id);
    if (p) keys.add(tileKeyAt(p[0], p[1]));
  }
  store.set(STORE_KEYS.tbt, { start: t.start, legs: t.legs, end: t.end, tiles: [...keys].slice(0, 400) });
}

/** At boot: load the saved route's tiles first; restored once they are in (see tbtTryRestore). */
function tbtLoadSaved() {
  const saved = store.get(STORE_KEYS.tbt);
  const t = state.tbt;
  if (!saved || !saved.start || !Array.isArray(saved.legs) || !Array.isArray(saved.tiles)) return;
  if (!isLngLat(saved.start.point) || !saved.legs.every((l) => l && Array.isArray(l.nodes))) return;
  t.saved = saved;
}

/** Put the saved route back when its roads are loaded and it still follows them. */
function tbtTryRestore() {
  const t = state.tbt;
  const saved = t.saved;
  if (!saved) return false;
  if (t.start) {
    t.saved = null; // you started another route meanwhile
    return false;
  }
  const states = saved.tiles.map((k) => t.tiles.get(k));
  if (states.some((st) => st !== "ok" && st !== "error")) return false; // still loading
  t.saved = null;
  const path = [];
  for (const leg of saved.legs) path.push(...leg.nodes.map(String));
  const s0 = saved.start;
  const joined = (a, b) => (t.g.adj.get(a) || []).some((e) => e.to === b);
  const ok =
    states.every((st) => st === "ok") &&
    joined(String(s0.a), String(s0.b)) &&
    path.every((id, i) => i === 0 || joined(path[i - 1], id));
  if (!ok) {
    store.remove(STORE_KEYS.tbt);
    return false;
  }
  tbtRestore(JSON.parse(JSON.stringify({ start: s0, legs: saved.legs, end: saved.end || null })));
  tbtChanged({ keepView: true });
  return true;
}

function tbtChanged(opts = {}) {
  const t = state.tbt;
  tbtSave();
  t.hoverPlan = null;
  t.hot = -1;
  t.retried = false;
  tbtRender();
  tbtRenderMap();
  tbtWantTiles();
  if (!opts.keepView) tbtKeepTipInView();
  tbtScheduleRoute();
}

function tbtScheduleRoute() {
  clearTimeout(state.routeTimer);
  if (state.routeAbort) state.routeAbort.abort();
  state.routeAbort = null;
  state.routeSeq++;
  state.routeOptions = [];
  state.optionIndex = 0;
  state.altHover = -1;
  setBusy(false);
  if (tbtPath().length < 2) {
    state.route = null;
    state.routeCum = null;
    state.routeStale = false;
    state.routeError = "";
    drawRoute();
    renderRouteInfo();
    resetProfile();
    updateBuildPanel();
    return;
  }
  state.routeStale = true;
  state.routeError = "";
  renderRouteInfo();
  updateBuildPanel();
  state.routeTimer = setTimeout(requestTbtRoute, 300);
}

async function requestTbtRoute() {
  const seq = state.routeSeq;
  const ctrl = new AbortController();
  state.routeAbort = ctrl;
  const t = state.tbt;
  const path = tbtPath();
  const loop = !!(t.end && t.end.loop);
  const round6 = (x) => Math.round(x * 1e6) / 1e6;
  const body = {
    path: path.map(Number),
    start_t: round6(tbtStartT()),
    end_t: loop ? 1 : round6(t.end ? t.end.t : 1),
    loop,
  };
  try {
    const route = await api(API.routePath, { method: "POST", body, signal: ctrl.signal });
    if (seq !== state.routeSeq) return;
    state.routeOptions = [route];
    state.routeSnapped = route.snapped;
    state.routeStale = false;
    state.routeError = "";
    selectOption(0, true);
  } catch (err) {
    if ((err && err.name === "AbortError") || seq !== state.routeSeq) return;
    if (err.status === 409 && !t.retried) {
      // The app forgot the roads (restarted?): send it the tiles again, then try once more.
      t.retried = true;
      const keys = [...new Set(path.map((id) => t.g.nodes.get(id)).filter(Boolean).map((p) => tileKeyAt(p[0], p[1])))];
      try {
        for (let i = 0; i < keys.length; i += ROADS_BATCH) {
          await api(API.roads, { method: "POST", body: { tiles: keys.slice(i, i + ROADS_BATCH) } });
        }
      } catch {
        /* the retry reports it */
      }
      if (seq === state.routeSeq) requestTbtRoute();
      return;
    }
    state.route = null;
    state.routeCum = null;
    state.routeStale = false;
    state.routeError = err.message || "Couldn't make the route.";
    drawRoute();
    renderRouteInfo();
    resetProfile("No route yet, so no elevation profile.");
  } finally {
    if (seq === state.routeSeq) {
      state.routeAbort = null;
      updateBuildPanel();
    }
  }
}

/** Keys while driving. True if the key was used. */
function tbtKey(e) {
  const phase = tbtPhase();
  if (e.key === "Backspace") {
    tbtUndo();
    return true;
  }
  if (e.key === "Enter" && phase === "drive" && !(e.target && e.target.closest && e.target.closest("button"))) {
    tbtFinish();
    return true;
  }
  const isArrow = e.key in KEY_BEARING;
  const isDigit = /^[1-9]$/.test(e.key);
  if (!isArrow && !isDigit) return false;
  if (phase === "direction") {
    const dirs = tbtDirections();
    const i = isDigit ? Number(e.key) - 1 : optionForKey(dirs.map((d) => (d.allowed ? d.bearing : null)), e.key);
    if (dirs[i] && dirs[i].allowed) tbtPickDirection(dirs[i]);
    return true;
  }
  if (phase === "drive") {
    const br = branchForKey(tbtBranches(), e.key);
    if (br) tbtTake(br);
    return true;
  }
  return isArrow && phase !== "start";
}

/* -- switching between turn by turn and quick A to B --------------------------------------- */

function applyModeUi() {
  const m = state.mode;
  $("mode-tbt").setAttribute("aria-pressed", String(m === "tbt"));
  $("mode-quick").setAttribute("aria-pressed", String(m === "quick"));
  $("tbt-panel").hidden = m !== "tbt";
  $("quick-panel").hidden = m !== "quick";
  $("btn-reverse").hidden = m === "tbt";
  $("map").setAttribute("aria-label", m === "tbt" ? "Map. Click the road you start on, then pick each turn." : "Map. Click to add points.");
  const map = state.map;
  if (map && map.keyboard) {
    if (m === "tbt") map.keyboard.disable(); // the arrow keys take turns
    else map.keyboard.enable();
  }
}

function setMode(mode) {
  const m = mode === "quick" ? "quick" : "tbt";
  if (m === state.mode) return;
  state.mode = m;
  store.set(STORE_KEYS.mode, m);
  closeSearch();
  if (m === "tbt") {
    setWaypointMode(false);
    cancelLineDrag();
    setPins([]); // the points stay saved for when you come back
    setMapHover(-1, false);
  } else {
    tbtClearMap();
    restorePins();
  }
  applyModeUi();
  enterMode();
}

function enterMode() {
  if (state.mode === "tbt") {
    tbtChanged({ keepView: true });
  } else {
    renderPins();
    scheduleRoute();
  }
  tbtRenderChip();
  updateBuildPanel();
}

/** The Undo and Clear buttons in the card's head, for whichever mode is on. */
function renderHeadTools() {
  const tbt = state.mode === "tbt";
  const undoBtn = $("btn-undo");
  const clearBtn = $("btn-clear");
  undoBtn.disabled = tbt ? !state.tbt.history.length && !state.tbt.start : state.history.length === 0;
  undoBtn.title = tbt ? "Undo the last turn (Backspace)" : "Undo (Ctrl+Z)";
  clearBtn.disabled = tbt ? !state.tbt.start : state.pins.length === 0;
  clearBtn.title = tbt ? "Start again" : "Remove all points";
}

/* ------------------------------------------------------------------------------------------
 * Elevation profile
 * ---------------------------------------------------------------------------------------- */

async function loadProfile(routeId) {
  state.profile = null;
  state.hoverIndex = -1;
  $("profile-stats").textContent = "";
  $("profile-source").textContent = "";
  showProfileMessage("Loading elevation…");
  drawProfile();
  try {
    const p = await api(API.profile, { method: "POST", body: { route_id: routeId } });
    if (!state.route || state.route.route_id !== routeId) return;
    const s = Array.isArray(p.s) ? p.s.map(Number) : [];
    const z = Array.isArray(p.z) ? p.z.map(Number) : [];
    if (s.length < 2 || s.length !== z.length) {
      showProfileMessage("No elevation data for this road.");
      return;
    }
    state.profile = { ...p, s, z, route_id: routeId };
    const st = profileStats(s, z);
    $("profile-stats").textContent =
      `Climb ${formatInt(st.climb)} m · Drop ${formatInt(st.drop)} m · ` +
      `${formatInt(st.minZ)}–${formatInt(st.maxZ)} m · Steepest ${(st.maxGrade * 100).toFixed(0)}%`;
    $("profile-source").textContent = sourceLabel(p.source, p.resolution_m);
    showProfileMessage("");
    drawProfile();
  } catch (err) {
    if (state.route && state.route.route_id === routeId) {
      showProfileMessage(`Elevation unavailable: ${err.message}`);
    }
  }
}

function resetProfile(message) {
  state.profile = null;
  state.hoverIndex = -1;
  $("profile-stats").textContent = "";
  $("profile-source").textContent = "";
  $("profile-tip").hidden = true;
  showProfileMessage(
    message || (state.mode === "tbt" ? "Start on a road to see the elevation profile." : "Click two points to see the elevation profile."),
  );
  setSourceData("dm-hover", EMPTY_FC);
  drawProfile();
}

function showProfileMessage(message) {
  const el = $("profile-empty");
  el.textContent = message || "";
  el.hidden = !message;
}

function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function drawProfile() {
  const canvas = $("profile-canvas");
  const box = $("profile-body");
  const w = box.clientWidth;
  const ht = box.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(ht * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(ht * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, ht);
  state.profileGeom = null;
  const p = state.profile;
  if (!p || w < 50 || ht < 40) return;

  const { s, z } = p;
  const pad = { l: 48, r: 14, t: 8, b: 20 };
  const s0 = s[0];
  const s1 = s[s.length - 1];
  let zmin = Infinity;
  let zmax = -Infinity;
  for (const v of z) {
    if (v < zmin) zmin = v;
    if (v > zmax) zmax = v;
  }
  const span = Math.max(zmax - zmin, 12);
  const lo = zmin - span * 0.12;
  const hi = zmax + span * 0.18;
  const pw = w - pad.l - pad.r;
  const ph = ht - pad.t - pad.b;
  const X = (sv) => pad.l + ((sv - s0) / (s1 - s0 || 1)) * pw;
  const Y = (zv) => pad.t + ((hi - zv) / (hi - lo)) * ph;
  state.profileGeom = { left: pad.l, width: pw, s0, s1 };

  const muted = cssVar("--muted", "#8d97a8");
  const grid = cssVar("--grid", "rgba(255,255,255,0.07)");
  const line = cssVar("--profile-line", "#19d3ff");
  const fill = cssVar("--profile-fill", "rgba(25,211,255,0.2)");
  const font = cssVar("--font", "system-ui, sans-serif");

  // grid + labels
  ctx.font = `11px ${font}`;
  ctx.fillStyle = muted;
  ctx.strokeStyle = grid;
  ctx.lineWidth = 1;
  let zStep = niceStep(hi - lo, 3);
  if (Math.floor(hi / zStep) - Math.ceil(lo / zStep) < 1) zStep = niceStep(hi - lo, 6); // >= 2 labels
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let v = Math.ceil(lo / zStep) * zStep; v <= hi; v += zStep) {
    const y = Math.round(Y(v)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillText(`${Math.round(v)} m`, pad.l - 6, y);
  }
  const total = s1 - s0;
  const useKm = total >= 1500;
  const sStep = niceStep(total, Math.max(2, Math.floor(pw / 90)));
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let v = 0; v <= total + 1e-6; v += sStep) {
    const x = X(s0 + v);
    const label = useKm ? `${+(v / 1000).toFixed(2)} km` : `${Math.round(v)} m`;
    ctx.fillText(label, clamp(x, pad.l + 14, w - pad.r - 18), ht - pad.b + 5);
  }

  // area + line
  const path = new Path2D();
  path.moveTo(X(s[0]), Y(z[0]));
  for (let i = 1; i < s.length; i++) path.lineTo(X(s[i]), Y(z[i]));
  const area = new Path2D(path);
  area.lineTo(X(s[s.length - 1]), pad.t + ph);
  area.lineTo(X(s[0]), pad.t + ph);
  area.closePath();
  ctx.fillStyle = fill;
  ctx.fill(area);
  ctx.strokeStyle = line;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.stroke(path);

  // start (A) and finish (B) dots
  for (const [k, color] of [[0, "#22c55e"], [s.length - 1, "#ef4444"]]) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(X(s[k]), Y(z[k]), 4, 0, Math.PI * 2);
    ctx.fill();
  }

  // hover
  const i = state.hoverIndex;
  if (i >= 0 && i < s.length) {
    const x = Math.round(X(s[i])) + 0.5;
    ctx.strokeStyle = muted;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, pad.t + ph);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = line;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(x, Y(z[i]), 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

function onProfileMove(e) {
  const p = state.profile;
  const g = state.profileGeom;
  if (!p || !g) return;
  const rect = $("profile-canvas").getBoundingClientRect();
  const x = e.clientX - rect.left;
  const sv = g.s0 + clamp((x - g.left) / g.width, 0, 1) * (g.s1 - g.s0);
  const i = nearestIndex(p.s, sv);
  if (i < 0) return;
  state.hoverIndex = i;
  drawProfile();

  const a = Math.max(0, i - 2);
  const b = Math.min(p.s.length - 1, i + 2);
  const grade = p.s[b] > p.s[a] ? (p.z[b] - p.z[a]) / (p.s[b] - p.s[a]) : 0;
  const gradeText = Math.abs(grade) < 0.005 ? "flat" : `${(Math.abs(grade) * 100).toFixed(0)}% ${grade > 0 ? "up" : "down"}`;
  const tip = $("profile-tip");
  tip.textContent = `${formatLength(p.s[i] - p.s[0])} · ${Math.round(p.z[i])} m · ${gradeText}`;
  tip.hidden = false;
  const tipW = tip.offsetWidth;
  const left = g.left + ((p.s[i] - g.s0) / (g.s1 - g.s0 || 1)) * g.width;
  tip.style.left = `${clamp(left - tipW / 2, 4, rect.width - tipW - 4)}px`;

  const r = state.route;
  if (r && state.routeCum && Array.isArray(r.line) && r.line.length >= 2) {
    const lineLen = state.routeCum[state.routeCum.length - 1];
    const frac = (p.s[i] - p.s[0]) / (p.s[p.s.length - 1] - p.s[0] || 1);
    setSourceData("dm-hover", hoverData(pointAlong(r.line, state.routeCum, frac * lineLen)));
  }
}

function onProfileLeave() {
  state.hoverIndex = -1;
  $("profile-tip").hidden = true;
  setSourceData("dm-hover", EMPTY_FC);
  drawProfile();
}

/* ------------------------------------------------------------------------------------------
 * Search (Photon)
 * ---------------------------------------------------------------------------------------- */

function initSearch() {
  const input = $("search");
  input.addEventListener("input", () => {
    clearTimeout(state.search.timer);
    const q = input.value.trim();
    if (q.length < 3) {
      closeSearch();
      return;
    }
    state.search.timer = setTimeout(() => runSearch(q), 300);
  });
  input.addEventListener("keydown", (e) => {
    const items = state.search.items;
    const open = !$("search-results").hidden;
    if (e.key === "ArrowDown" && open && items.length) {
      e.preventDefault();
      setActiveResult((state.search.active + 1) % items.length);
    } else if (e.key === "ArrowUp" && open && items.length) {
      e.preventDefault();
      setActiveResult((state.search.active - 1 + items.length) % items.length);
    } else if (e.key === "Enter") {
      e.preventDefault();
      const q = input.value.trim();
      const coord = parseLatLon(q);
      if (open && items.length) chooseResult(items[Math.max(0, state.search.active)]);
      else if (coord) chooseResult({ title: q, center: coord });
      else if (q.length >= 2) {
        clearTimeout(state.search.timer);
        runSearch(q);
      }
    } else if (e.key === "Escape") {
      if (open) closeSearch();
      else input.blur();
    }
  });
  input.addEventListener("blur", () => setTimeout(closeSearch, 150));
}

async function runSearch(q) {
  const coord = parseLatLon(q);
  if (coord) {
    renderResults([{ title: `Go to ${coord[1].toFixed(5)}, ${coord[0].toFixed(5)}`, subtitle: "Coordinates", center: coord }]);
    return;
  }
  if (state.search.abort) state.search.abort.abort();
  const ctrl = new AbortController();
  state.search.abort = ctrl;
  const c = state.map ? state.map.getCenter() : { lng: HOME_VIEW.center[0], lat: HOME_VIEW.center[1] };
  const fetchOne = async (text) => {
    const params = new URLSearchParams({ q: text, limit: "8", lat: c.lat.toFixed(4), lon: c.lng.toFixed(4), lang: "en" });
    const res = await fetch(`${PHOTON_URL}?${params}`, { signal: ctrl.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return (data.features || [])
      .filter((f) => f && f.geometry && isLngLat(f.geometry.coordinates))
      .map((f) => ({ ...photonLabel(f.properties), center: f.geometry.coordinates, props: f.properties || {} }));
  };
  try {
    const [first, ...rest] = searchQueries(q).map(fetchOne);
    // The extra (road name only) search is a bonus: if it fails, show the main results.
    const extra = rest.map((p) => p.catch(() => []));
    const groups = [await first];
    for (const p of extra) groups.push(await p);
    if (ctrl !== state.search.abort || $("search").value.trim() !== q) return;
    const items = mergeSearchResults(groups, q);
    renderResults(items, items.length ? "" : "No places found. Try a road name and a town.");
  } catch (err) {
    if (err && err.name === "AbortError") return;
    renderResults([], "Search isn't available right now. You can still pan the map.");
  }
}

function renderResults(items, emptyText) {
  const list = $("search-results");
  state.search.items = items;
  state.search.active = -1;
  list.replaceChildren();
  items.forEach((item, i) => {
    const li = h(
      "li",
      { id: `search-opt-${i}`, role: "option", "aria-selected": "false" },
      h("span", { class: "r-title" }, item.title),
      item.subtitle ? h("span", { class: "r-sub" }, item.subtitle) : null,
    );
    li.addEventListener("mousedown", (e) => {
      e.preventDefault();
      chooseResult(item);
    });
    li.addEventListener("mousemove", () => {
      if (state.search.active !== i) setActiveResult(i);
    });
    list.append(li);
  });
  if (!items.length && emptyText) list.append(h("li", { class: "empty", "aria-disabled": "true" }, emptyText));
  const show = items.length > 0 || !!emptyText;
  list.hidden = !show;
  $("search").setAttribute("aria-expanded", String(show));
}

function setActiveResult(i) {
  state.search.active = i;
  const list = $("search-results");
  [...list.children].forEach((li, j) => li.setAttribute("aria-selected", String(j === i)));
  const input = $("search");
  if (i >= 0) {
    input.setAttribute("aria-activedescendant", `search-opt-${i}`);
    const li = list.children[i];
    if (li && li.scrollIntoView) li.scrollIntoView({ block: "nearest" });
  } else input.removeAttribute("aria-activedescendant");
}

function closeSearch() {
  const list = $("search-results");
  if (!list) return;
  list.hidden = true;
  $("search").setAttribute("aria-expanded", "false");
  $("search").removeAttribute("aria-activedescendant");
}

function chooseResult(item) {
  if (!item || !item.center) return;
  closeSearch();
  $("search").value = item.title.replace(/^Go to /, "");
  const map = state.map;
  if (!map) return;
  const ext = item.props && Array.isArray(item.props.extent) ? item.props.extent.map(Number) : null;
  if (ext && ext.length === 4 && ext.every(Number.isFinite) && ext[0] !== ext[2]) {
    // Photon extent: [minLon, maxLat, maxLon, minLat]
    map.fitBounds(
      [
        [Math.min(ext[0], ext[2]), Math.min(ext[1], ext[3])],
        [Math.max(ext[0], ext[2]), Math.max(ext[1], ext[3])],
      ],
      { padding: 60, maxZoom: 16, duration: 900 },
    );
  } else {
    const isRoad = item.props && item.props.osm_key === "highway";
    map.flyTo({ center: item.center, zoom: isRoad ? 15 : 13.5, duration: 900 });
  }
}

/* ------------------------------------------------------------------------------------------
 * Assetto Corsa status and settings
 * ---------------------------------------------------------------------------------------- */

async function loadStatus() {
  try {
    applyStatus(await api(API.status));
  } catch (err) {
    setPill("bad", "App not reachable", err.message);
  }
}

function setPill(kind, text, title) {
  const pill = $("ac-pill");
  pill.dataset.kind = kind;
  pill.textContent = text;
  pill.title = title || "";
}

function applyStatus(st) {
  state.status = st || null;
  const label = $("build-label");
  label.textContent = buildLabel(st && st.build);
  const b = (st && st.build) || {};
  label.title = b.commit ? `Version ${b.version || "dev"}, commit ${b.commit}${b.date ? `, built ${b.date}` : ""}` : "A development build";
  if (st && st.ac_found) {
    setPill("ok", "Assetto Corsa found", `${st.ac_root || ""}\nClick to change the folder.`);
  } else {
    setPill("warn", "Assetto Corsa not found", "Click to tell the app where Assetto Corsa is.");
    openAcPanel(false);
  }
  updateBuildPanel();
}

function openAcPanel(byUser) {
  const st = state.status;
  const found = !!(st && st.ac_found);
  $("ac-message").textContent = found
    ? "Tracks are installed into this Assetto Corsa folder. Paste a different one to change it."
    : "We couldn't find Assetto Corsa on this PC. Paste your assettocorsa folder so finished tracks install themselves. (You can still build without it.)";
  const input = $("ac-root");
  if (!input.value && st && st.ac_root) input.value = st.ac_root;
  $("ac-error").hidden = true;
  $("ac-panel").hidden = false;
  $("ac-pill").setAttribute("aria-expanded", "true");
  if (byUser) input.focus();
}

function closeAcPanel() {
  $("ac-panel").hidden = true;
  $("ac-pill").setAttribute("aria-expanded", "false");
}

async function saveAcRoot(e) {
  e.preventDefault();
  const input = $("ac-root");
  const value = input.value.trim().replace(/^["']+|["']+$/g, "").trim(); // "Copy as path" adds quotes
  const errEl = $("ac-error");
  if (!value) {
    errEl.textContent = "Paste the path of your assettocorsa folder first.";
    errEl.hidden = false;
    return;
  }
  const btn = $("ac-save");
  btn.disabled = true;
  errEl.hidden = true;
  try {
    const st = await api(API.settings, { method: "POST", body: { ac_root: value } });
    state.status = st;
    if (st && st.ac_found) {
      applyStatus(st);
      closeAcPanel();
      toast("Saved. Finished tracks will be installed into Assetto Corsa.", "ok");
      loadTracks();
    } else {
      applyStatus(st);
      errEl.textContent =
        "That folder doesn't look like Assetto Corsa. Pick the folder that has acs.exe and a content folder inside.";
      errEl.hidden = false;
    }
  } catch (err) {
    errEl.textContent = err.message;
    errEl.hidden = false;
  } finally {
    btn.disabled = false;
  }
}

/* ------------------------------------------------------------------------------------------
 * Build
 * ---------------------------------------------------------------------------------------- */

const OPTION_IDS = ["opt-trees", "opt-buildings", "opt-install"];
const GUARDRAIL_MODES = ["auto", "mapped", "off"];
const GUARDRAIL_HINTS = {
  auto: "Rails where Ontario's road rules call for them (steep drops, high embankments, deep water, bridge ends), plus any mapped in OpenStreetMap.",
  mapped: "Only the guardrails and barriers mapped in OpenStreetMap.",
  off: "No guardrails at all.",
};

function loadOptions() {
  const saved = store.get(STORE_KEYS.options);
  if (saved && typeof saved === "object") {
    for (const id of OPTION_IDS) if (typeof saved[id] === "boolean") $(id).checked = saved[id];
    const d = Number(saved.density);
    if (Number.isFinite(d)) $("opt-density").value = String(clamp(d, 0, 2));
    if (GUARDRAIL_MODES.includes(saved.guardrails)) $("opt-guardrails").value = saved.guardrails;
    else if (saved["opt-guardrails"] === false) $("opt-guardrails").value = "off"; // the old on/off switch
  }
  renderDensity();
}

function saveOptions() {
  const out = { density: Number($("opt-density").value), guardrails: $("opt-guardrails").value };
  for (const id of OPTION_IDS) out[id] = $(id).checked;
  store.set(STORE_KEYS.options, out);
}

function renderDensity() {
  const v = Number($("opt-density").value);
  const trees = $("opt-trees").checked;
  $("opt-density").disabled = !trees;
  $("density-out").textContent = trees ? `${densityLabel(v)} (${v.toFixed(1)}x)` : "No trees";
  $("guardrails-hint").textContent = GUARDRAIL_HINTS[$("opt-guardrails").value] || "";
}

function buildOptions() {
  return {
    trees: $("opt-trees").checked,
    buildings: $("opt-buildings").checked,
    guardrails: $("opt-guardrails").value !== "off",
    auto_guardrails: $("opt-guardrails").value === "auto",
    scenery_density: Number($("opt-density").value),
  };
}

function updateBuildPanel() {
  const btn = $("btn-build");
  const ready = !!state.route && !state.routeStale;
  const busy = state.jobId != null;
  btn.disabled = !ready || busy;
  btn.textContent = busy ? "Creating map…" : "Create map";
  let hint;
  const tbt = state.mode === "tbt";
  if (busy) hint = "You can keep exploring. The map is made while the app is open.";
  else if (tbt && !(state.tbt && state.tbt.legs.length)) hint = "Pick your road on the map first.";
  else if (!tbt && state.pins.length < 2) hint = "Click at least two points on the map first.";
  else if (state.routeStale) hint = "Waiting for the route…";
  else if (!state.route) hint = "Fix the route above first.";
  else if ($("opt-install").checked && state.status && !state.status.ac_found)
    hint = "Assetto Corsa wasn't found, so the track is saved in the app's folder instead.";
  else if (state.route.loop) hint = "A loop: you can race it lap after lap. Takes a few minutes.";
  else if (tbt && !state.tbt.end) hint = "The track ends where the blue line ends. Keep picking turns, or create it now.";
  else hint = "Takes a few minutes. The first map in a new area downloads map data.";
  $("build-hint").textContent = hint;
}

async function startBuild() {
  const r = state.route;
  if (!r || state.routeStale || state.jobId != null) return;
  const name = $("track-name").value.trim() || defaultTrackName(r.names, !!r.loop) || "My Road";
  const trackId = sanitizeTrackId($("track-id").value) || null;
  state.jobLogCount = -1;
  renderJob({ state: "queued", step: "Sending to the builder…", progress: 0, log: [] });
  $("btn-build").disabled = true;
  try {
    const res = await api(API.build, {
      method: "POST",
      body: {
        route_id: r.route_id,
        name,
        track_id: trackId,
        install: $("opt-install").checked,
        loop: !!r.loop,
        options: buildOptions(),
      },
    });
    if (!res || !res.job_id) throw new ApiError("The app didn't start the build. Try again.", 0);
    state.jobId = String(res.job_id);
    state.jobResumed = false;
    state.jobName = name;
    state.jobFailures = 0;
    store.set(STORE_KEYS.job, { id: state.jobId, name });
    updateBuildPanel();
    pollJob();
  } catch (err) {
    renderJob({ state: "error", step: "Couldn't start the build", progress: 0, log: [], error: err.message });
    updateBuildPanel();
  }
}

async function pollJob() {
  const id = state.jobId;
  if (id == null) return;
  clearTimeout(state.jobTimer);
  try {
    const job = await api(API.jobs + encodeURIComponent(id));
    if (id !== state.jobId) return;
    state.jobFailures = 0;
    renderJob(job);
    if (job.state === "done" || job.state === "error") {
      finishJob(job);
      return;
    }
  } catch (err) {
    if (id !== state.jobId) return;
    if (err.status === 404) {
      const resumed = state.jobResumed;
      finishJob(null);
      if (resumed) {
        // The app restarted since that build; whatever it made is in My tracks.
        $("job").hidden = true;
        loadTracks();
        return;
      }
      renderJob({
        state: "error",
        step: "Build lost",
        progress: 0,
        log: [],
        error: "The app was restarted, so this map wasn't finished. Press Create map to start it again.",
      });
      return;
    }
    state.jobFailures++;
    if (state.jobFailures >= 3) $("job-step").textContent = "Lost contact with the app. Retrying…";
  }
  state.jobTimer = setTimeout(pollJob, 1000);
}

function finishJob(job) {
  state.jobId = null;
  state.jobResumed = false;
  store.remove(STORE_KEYS.job);
  updateBuildPanel();
  if (job && job.state === "done") {
    toast(job.result && job.result.installed_to ? "Track installed!" : "Track built!", "ok");
    loadTracks();
  }
  if (job) revealJob();
}

/** Scroll the build panel so the progress (or result) is on screen: at 1280x800 it sits
 * below the Build button, out of view. */
function revealJob() {
  const box = $("job");
  if (box.hidden || typeof box.scrollIntoView !== "function") return;
  requestAnimationFrame(() => box.scrollIntoView({ block: "nearest", behavior: "smooth" }));
}

function renderJob(job) {
  const box = $("job");
  const appearing = box.hidden;
  box.hidden = false;
  box.dataset.state = job.state || "running";
  const done = job.state === "done";
  const pct = done ? 100 : Math.round(clamp(Number(job.progress) || 0, 0, 1) * 100);
  let step;
  if (job.state === "queued") step = job.step || "Waiting for the previous build to finish…";
  else if (done) step = "Done";
  else if (job.state === "error") step = job.step ? `Failed: ${job.step}` : "Build failed";
  else step = job.step || "Building…";
  $("job-step").textContent = step;
  $("job-pct").textContent = `${pct}%`;
  $("job-bar").style.width = `${pct}%`;
  $("job-progress").setAttribute("aria-valuenow", String(pct));

  const lines = Array.isArray(job.log) ? job.log : [];
  if (lines.length !== state.jobLogCount) {
    state.jobLogCount = lines.length;
    const pre = $("job-log");
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 12;
    pre.textContent = lines.join("\n");
    if (atBottom) pre.scrollTop = pre.scrollHeight;
  }
  $("job-log-wrap").hidden = lines.length === 0;
  renderJobResult(job);
  if (appearing) revealJob();
}

function renderJobResult(job) {
  const el = $("job-result");
  el.replaceChildren();
  el.className = "job-result";
  state.resultTrackId = job.state === "done" && job.result ? job.result.track_id || null : null;
  if (job.state === "done" && job.result) {
    const r = job.result;
    const name = r.name || state.jobName || "your track";
    el.classList.add("ok");
    if (r.installed_to) {
      el.append(h("p", { class: "big" }, "Installed! Open Content Manager and look for ", h("strong", {}, name), "."));
    } else {
      el.append(
        h("p", { class: "big" }, "Built! ", h("strong", {}, name), " is ready."),
        h(
          "p",
          {},
          "It wasn't installed. Copy this folder into assettocorsa\\content\\tracks to drive it: ",
          h("code", {}, r.track_dir || ""),
        ),
      );
    }
    const stat = (label, value) => h("div", {}, h("dt", {}, label), h("dd", {}, value));
    el.append(
      h(
        "dl",
        { class: "job-stats" },
        stat(r.loop ? "Lap" : "Length", formatLength(Number(r.length_m))),
        stat("Size", Number.isFinite(Number(r.kn5_mb)) ? `${Number(r.kn5_mb).toFixed(1)} MB` : "–"),
        stat("Trees", formatInt(Number(r.trees))),
        stat("Buildings", formatInt(Number(r.buildings))),
      ),
    );
    if (r.installed_to) el.append(h("p", { class: "muted" }, "Folder: ", h("code", {}, r.installed_to)));
  } else if (job.state === "error") {
    el.classList.add("bad");
    el.append(h("p", {}, job.error || "The build failed. Open the log below for details."));
  }
  el.hidden = el.childNodes.length === 0;
}

function resumeJob() {
  const saved = store.get(STORE_KEYS.job);
  if (!saved || typeof saved.id !== "string" || !saved.id) return;
  state.jobId = saved.id;
  state.jobResumed = true;
  state.jobName = typeof saved.name === "string" ? saved.name : "";
  state.jobLogCount = -1;
  renderJob({ state: "running", step: "Checking on your build…", progress: 0, log: [] });
  updateBuildPanel();
  pollJob();
}

/* ------------------------------------------------------------------------------------------
 * My tracks
 * ---------------------------------------------------------------------------------------- */

async function loadTracks() {
  const list = $("track-list");
  try {
    const tracks = await api(API.tracks);
    renderTracks(Array.isArray(tracks) ? tracks : []);
  } catch (err) {
    list.replaceChildren();
    $("tracks-empty").textContent = `Couldn't load your tracks: ${err.message}`;
    $("tracks-empty").hidden = false;
  }
}

function builtAtValue(v) {
  if (typeof v === "number") return v < 1e12 ? v * 1000 : v;
  const t = Date.parse(v);
  return Number.isNaN(t) ? 0 : t;
}

function renderTracks(tracks) {
  const list = $("track-list");
  const sorted = tracks.slice().sort((a, b) => builtAtValue(b.built_at) - builtAtValue(a.built_at));
  list.replaceChildren(
    ...sorted.map((t) => {
      const name = t.name || t.track_id;
      const meta = [formatLength(Number(t.length_m)), t.loop ? "Loop" : "", formatDate(t.built_at)].filter((x) => x && x !== "–");
      return h(
        "li",
        { class: "track-item" },
        h(
          "div",
          { class: "track-main" },
          h("strong", { title: `${name} (${t.track_id})` }, name),
          h("span", { class: "track-meta" }, meta.join(" · ")),
        ),
        h(
          "span",
          { class: t.installed ? "badge ok" : "badge", title: t.installed ? "In your Assetto Corsa tracks folder" : "Only in the app's folder" },
          t.installed ? "Installed" : "Not installed",
        ),
        iconButton("trash", {
          class: "icon-btn danger",
          title: "Delete",
          "aria-label": `Delete ${name}`,
          onclick: () => deleteTrack(t),
        }),
      );
    }),
  );
  $("tracks-empty").textContent = "Tracks you build show up here.";
  $("tracks-empty").hidden = sorted.length > 0;
}

async function deleteTrack(t) {
  const name = t.name || t.track_id;
  const ok = await confirmDialog(
    `Delete "${name}"?`,
    "This removes the track from Assetto Corsa and from the app's folder. It can't be undone.",
    "Delete",
  );
  if (!ok) return;
  try {
    await api(API.tracks + "/" + encodeURIComponent(t.track_id), { method: "DELETE" });
    toast(`Deleted ${name}.`, "ok");
    // Don't leave "Installed! Look for <name>" on screen for a track that's gone.
    if (state.jobId == null && state.resultTrackId === t.track_id) {
      $("job").hidden = true;
      state.resultTrackId = null;
    }
  } catch (err) {
    toast(err.message, "error");
  }
  loadTracks();
}

/* ------------------------------------------------------------------------------------------
 * Wiring
 * ---------------------------------------------------------------------------------------- */

function onKeyDown(e) {
  if (e.key === "Escape" && state.waypointMode) {
    setWaypointMode(false);
    return;
  }
  const t = e.target;
  const typing = t && t.closest && t.closest("input, textarea, select, [contenteditable='true']");
  if (e.key === "Shift" && state.mode === "tbt" && state.tbt.lastMouse) {
    state.tbt.lastMouse.shift = true;
    tbtUpdateHover();
  }
  if (typing || e.altKey) return;
  if ($("confirm-dialog").open) return;
  const key = e.key.toLowerCase();
  const tbt = state.mode === "tbt";
  if (tbt && !e.ctrlKey && !e.metaKey && tbtKey(e)) {
    e.preventDefault();
  } else if ((e.ctrlKey || e.metaKey) && !e.shiftKey && key === "z") {
    e.preventDefault();
    if (tbt) tbtUndo();
    else undo();
  } else if (!e.ctrlKey && !e.metaKey && e.key === "/") {
    e.preventDefault();
    $("search").focus();
  } else if (!tbt && !e.ctrlKey && !e.metaKey && key === "w" && state.pins.length >= 2) {
    setWaypointMode(!state.waypointMode);
  } else if (!tbt && !e.ctrlKey && !e.metaKey && (e.key === "[" || e.key === "]") && state.routeOptions.length > 1) {
    e.preventDefault();
    selectOption(state.optionIndex + (e.key === "]" ? 1 : -1));
  }
}

function bindUi() {
  $("style-toggle").addEventListener("click", toggleMapStyle);
  $("btn-loop").addEventListener("click", toggleLoop);
  $("btn-undo").addEventListener("click", () => (state.mode === "tbt" ? tbtUndo() : undo()));
  $("btn-reverse").addEventListener("click", reverseRoute);
  $("btn-clear").addEventListener("click", () => (state.mode === "tbt" ? tbtClear() : clearPins()));
  $("mode-tbt").addEventListener("click", () => setMode("tbt"));
  $("mode-quick").addEventListener("click", () => setMode("quick"));
  $("tbt-finish").addEventListener("click", tbtFinish);
  $("tbt-loop").addEventListener("click", tbtCloseLoop);
  $("opt-guardrails").addEventListener("change", () => {
    renderDensity();
    saveOptions();
  });
  $("btn-fit").addEventListener("click", fitToRoute);
  $("btn-build").addEventListener("click", startBuild);
  $("btn-tracks-refresh").addEventListener("click", loadTracks);
  $("ac-pill").addEventListener("click", () => ($("ac-panel").hidden ? openAcPanel(true) : closeAcPanel()));
  $("ac-cancel").addEventListener("click", closeAcPanel);
  $("ac-form").addEventListener("submit", saveAcRoot);
  $("track-name").addEventListener("input", (e) => {
    state.nameTouched = e.target.value.trim() !== "";
  });
  $("track-id").addEventListener("change", (e) => {
    e.target.value = sanitizeTrackId(e.target.value);
  });
  for (const id of OPTION_IDS) {
    $(id).addEventListener("change", () => {
      renderDensity();
      saveOptions();
      updateBuildPanel();
    });
  }
  $("opt-density").addEventListener("input", () => {
    renderDensity();
    saveOptions();
  });
  const canvas = $("profile-canvas");
  canvas.addEventListener("mousemove", onProfileMove);
  canvas.addEventListener("mouseleave", onProfileLeave);
  if (typeof ResizeObserver === "function") new ResizeObserver(() => drawProfile()).observe($("profile-body"));
  else window.addEventListener("resize", drawProfile);
  document.addEventListener("keydown", onKeyDown);
  document.addEventListener("keyup", (e) => {
    if (e.key === "Shift" && state.mode === "tbt" && state.tbt.lastMouse) {
      state.tbt.lastMouse.shift = false;
      tbtUpdateHover();
    }
  });
  initSearch();
}

function restorePins() {
  const snap = store.get(STORE_KEYS.pins);
  if (!snap || !Array.isArray(snap.pins) || !snap.pins.every(isLngLat)) return;
  setPins(snap.pins.slice(0, 50));
  state.reversed = !!snap.reversed && state.pins.length >= 2;
  state.loop = !!snap.loop && state.pins.length >= 2;
}

function initialStyleKey() {
  const saved = store.get(STORE_KEYS.style);
  if (saved === "dark" || saved === "light") return saved;
  try {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  } catch {
    return "light";
  }
}

function boot() {
  // Read-only handle for troubleshooting from the browser console and for the end-to-end test
  // (tests/test_app_e2e.py clicks map positions through map.project).
  window.acmg = Object.freeze({
    state,
    get map() { return state.map; },
    tbt: Object.freeze({ phase: tbtPhase, branches: tbtBranches, path: tbtPath, line: tbtLine, loopPath: tbtLoopPath }),
  });
  state.tbt = tbtNew();
  tbtLoadSaved();
  state.mode = store.get(STORE_KEYS.mode) === "quick" ? "quick" : "tbt";
  setMapStyle(initialStyleKey());
  bindUi();
  initMap();
  applyModeUi();
  if (state.mode === "quick") restorePins();
  loadOptions();
  renderRouteInfo();
  resetProfile();
  enterMode();
  loadStatus();
  loadTracks();
  resumeJob();
}

if (typeof document !== "undefined" && typeof window !== "undefined") {
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
}
