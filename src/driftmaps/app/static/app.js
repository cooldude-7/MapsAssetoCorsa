/* Assetto Corsa Map Generator - the map page.
 *
 * Plain modern JS, no build step, no framework. Talks to the local app server on the same origin
 * (GET /api/status, POST /api/route, POST /api/build, ...). The pure helpers come first so the
 * tests can run them under Node; boot() only runs in a browser.
 */
"use strict";

const API = {
  status: "/api/status",
  settings: "/api/settings",
  route: "/api/route",
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
    if (!state.lineDrag) setMapHover(-1, false);
  });
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
  const line = state.route && Array.isArray(state.route.line) ? state.route.line : [];
  if (line.length < 2) return EMPTY_FC;
  return {
    type: "FeatureCollection",
    features: [{ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: line } }],
  };
}

/** Snapped points (where the pins landed on the road) plus dashed leaders from far-off pins. */
function snapData() {
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
  const stale = state.routeStale;
  map.setPaintProperty("dm-route-line", "line-opacity", stale ? 0.4 : 1);
  map.setPaintProperty("dm-route-casing", "line-opacity", stale ? 0.3 : 0.85);
  map.setPaintProperty("dm-route-arrows", "icon-opacity", stale ? 0.3 : 1);
}

function fitToRoute() {
  const map = state.map;
  if (!map) return;
  const coords =
    state.route && state.route.line && state.route.line.length >= 2
      ? state.route.line
      : state.pins.map((p) => p.lngLat);
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
  if (Date.now() - state.lastDragEnd < 300) return;
  closeSearch();
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
    pin.el.className = `dm-pin dm-pin--${role}${loop && role === "start" ? " dm-pin--loop" : ""}`;
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
  $("btn-undo").disabled = state.history.length === 0;
  $("btn-reverse").disabled = n < 2;
  $("btn-clear").disabled = n === 0;
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
    : state.routeStale
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
  showProfileMessage(message || "Click two points to see the elevation profile.");
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

const OPTION_IDS = ["opt-trees", "opt-buildings", "opt-guardrails", "opt-install"];

function loadOptions() {
  const saved = store.get(STORE_KEYS.options);
  if (saved && typeof saved === "object") {
    for (const id of OPTION_IDS) if (typeof saved[id] === "boolean") $(id).checked = saved[id];
    const d = Number(saved.density);
    if (Number.isFinite(d)) $("opt-density").value = String(clamp(d, 0, 2));
  }
  renderDensity();
}

function saveOptions() {
  const out = { density: Number($("opt-density").value) };
  for (const id of OPTION_IDS) out[id] = $(id).checked;
  store.set(STORE_KEYS.options, out);
}

function renderDensity() {
  const v = Number($("opt-density").value);
  const trees = $("opt-trees").checked;
  $("opt-density").disabled = !trees;
  $("density-out").textContent = trees ? `${densityLabel(v)} (${v.toFixed(1)}x)` : "No trees";
}

function buildOptions() {
  return {
    trees: $("opt-trees").checked,
    buildings: $("opt-buildings").checked,
    guardrails: $("opt-guardrails").checked,
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
  if (busy) hint = "You can keep exploring. The map is made while the app is open.";
  else if (state.pins.length < 2) hint = "Click at least two points on the map first.";
  else if (state.routeStale) hint = "Waiting for the route…";
  else if (!state.route) hint = "Fix the route above first.";
  else if ($("opt-install").checked && state.status && !state.status.ac_found)
    hint = "Assetto Corsa wasn't found, so the track is saved in the app's folder instead.";
  else if (state.route.loop) hint = "A loop: you can race it lap after lap. Takes a few minutes.";
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
  if (typing || e.altKey) return;
  const key = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && !e.shiftKey && key === "z") {
    e.preventDefault();
    undo();
  } else if (!e.ctrlKey && !e.metaKey && e.key === "/") {
    e.preventDefault();
    $("search").focus();
  } else if (!e.ctrlKey && !e.metaKey && key === "w" && state.pins.length >= 2) {
    setWaypointMode(!state.waypointMode);
  } else if (!e.ctrlKey && !e.metaKey && (e.key === "[" || e.key === "]") && state.routeOptions.length > 1) {
    e.preventDefault();
    selectOption(state.optionIndex + (e.key === "]" ? 1 : -1));
  }
}

function bindUi() {
  $("style-toggle").addEventListener("click", toggleMapStyle);
  $("btn-loop").addEventListener("click", toggleLoop);
  $("btn-undo").addEventListener("click", undo);
  $("btn-reverse").addEventListener("click", reverseRoute);
  $("btn-clear").addEventListener("click", clearPins);
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
  window.acmg = Object.freeze({ state, get map() { return state.map; } });
  setMapStyle(initialStyleKey());
  bindUi();
  initMap();
  restorePins();
  loadOptions();
  renderPins();
  renderRouteInfo();
  resetProfile();
  scheduleRoute();
  updateBuildPanel();
  loadStatus();
  loadTracks();
  resumeJob();
}

if (typeof document !== "undefined" && typeof window !== "undefined") {
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
}
