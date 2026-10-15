"""The local app's web server: the map page plus a small JSON API (standard library only).

Binds 127.0.0.1 only. Endpoints (all JSON; errors are `{"error": "<plain English>"}`):

- `GET /` and other GETs: the single-page app from `driftmaps/app/static/` (read through
  `importlib.resources`, so it also works when frozen by PyInstaller).
- `GET /api/status`, `POST /api/settings {ac_root}`: app name/version and where AC is.
- `POST /api/route {pins: [[lat, lon], ...], reverse, loop}`: route over real roads; the route
  is kept by `route_id` (in memory and as `<app data>/routes/<route_id>.json`).
  `snapped[i]` is where request pin i landed on the road (request order, even with
  `reverse`); `line` is the display line `[[lon, lat], ...]` in driving order. With
  `loop: true` the route comes back to the first pin (a circuit; `line` ends where it
  starts) and `loop` in the answer says so; `reverse` then drives the loop the other way
  round from the same start. `loop` defaults to false (A to B), as before.
  `alternatives` lists up to ROUTE_OPTIONS ways through the same pins, each
  `{route_id, length_m, line, names, warnings, loop}`; the first is the best route and the
  top-level fields are the same as its. A pin too far from any road answers 400 with
  `far_pin` (its request index) beside `error`.
- `POST /api/profile {route_id}`: ground height every ~20 m along the route.
- `POST /api/build {route_id, name, track_id, install, options, loop}` -> `{job_id}`;
  `GET /api/jobs/<job_id>`: one build at a time, the rest queued (see `jobs.py`). A loop route
  builds a circuit; `loop` is optional and, if sent, must match the route.
- `GET /api/tracks`, `DELETE /api/tracks/<id>`: tracks this app made (marker file or the
  "drift-maps" tag), in AC's `content/tracks` and the app's output folder. Other tracks are
  never listed or touched.

Requests must come from this machine's own page: the Host header must name 127.0.0.1 or
localhost (DNS-rebinding guard) and a POST/DELETE with an Origin header must come from it too
(so another website can't drive the API).
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import logging
import math
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

from driftmaps.ac.scene import clear_track_folder, is_driftmaps_track
from driftmaps.ac.track_files import validate_track_id
from driftmaps.app.jobs import (
    NO_AC_MESSAGE,
    BuildRequest,
    BuildRunner,
    JobQueue,
    app_track_id,
    folder_conflict,
    is_network_error,
)
from driftmaps.app.settings import APP_NAME, Settings, SettingsError

if TYPE_CHECKING:
    from driftmaps.geo.router import Route

__all__ = ["DEFAULT_PORT", "ApiError", "App", "AppServer", "run"]

log = logging.getLogger(__name__)

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
PORT_TRIES = 10  # 8765..8774, then any free port
MAX_BODY_BYTES = 1_000_000
MAX_PINS = 50
MAX_LINE_POINTS = 2000
LONG_ROUTE_M = 25_000.0  # PLAN §5.1: warn above this
MIN_LOOP_M = 250.0  # a lap needs room for the start grid (the builder refuses under 200 m)
# ASSUMED: an Overpass corridor query over pins spread wider than this is too big to be useful.
MAX_PIN_SPAN_M = 100_000.0
ROUTE_OPTIONS = 4  # route options offered per request (the best one and up to 3 others)
PROFILE_STEP_M = 20.0
ROUTE_MEMORY = 64  # routes (and profiles) kept in memory; all routes also go to disk
ROUTE_FILES_KEPT = 200
_ROUTE_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}

MAP_SERVERS_MESSAGE = "Couldn't reach the map servers - check your internet and try again."
STALE_MAP_WARNING = (
    "Couldn't reach the map servers, so this route uses map data saved earlier (it may be "
    "out of date)."
)
COARSE_WARNING = (
    "No detailed elevation data here - elevation will be coarse (30 m), so small bumps and "
    "dips won't be in the road."
)
ROUTE_GONE = "That route isn't available any more - draw it again."

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".txt": "text/plain; charset=utf-8",
    ".pbf": "application/x-protobuf",
}


class ApiError(Exception):
    """An HTTP error with a message for the user (and `extra` fields for the page)."""

    def __init__(self, status: int, message: str, extra: dict | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.extra = dict(extra or {})


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def pin_letter(i: int) -> str:
    """The label of the i-th point in click order: A, B, ..., Z, AA, AB, ... (as on the map)."""
    out = ""
    i += 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        out = chr(65 + r) + out
    return out


def _pin_label(i: int, n: int) -> str:
    return pin_letter(i)


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def parse_pins(value: object) -> list[tuple[float, float]]:
    """Validate `[[lat, lon], ...]` from a request (2..MAX_PINS finite points)."""
    if not isinstance(value, list) or len(value) < 2:
        raise ApiError(400, "Click at least two points on the map: A (the start) and B.")
    if len(value) > MAX_PINS:
        raise ApiError(400, f"Too many points - use at most {MAX_PINS}.")
    pins = []
    for i, p in enumerate(value):
        bad = ApiError(400, f"Point {_pin_label(i, len(value))} isn't a map position.")
        if not isinstance(p, list) or len(p) != 2 or any(isinstance(v, bool) for v in p):
            raise bad
        try:
            lat, lon = float(p[0]), float(p[1])
        except (TypeError, ValueError):
            raise bad from None
        if not (math.isfinite(lat) and math.isfinite(lon)):
            raise bad
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise bad
        pins.append((lat, lon))
    return pins


def far_pin_index(exc: Exception, pins: list[tuple[float, float]]) -> int | None:
    """The index of the pin a `RouteError` says is too far from any road, else None."""
    m = re.search(r"pin \(([-\d.]+), ([-\d.]+)\) is (\d+) m from the nearest road", str(exc))
    if not m or not pins:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    return min(range(len(pins)), key=lambda k: _haversine_m(pins[k], (lat, lon)))


def route_error(exc: Exception, pins: list[tuple[float, float]]) -> ApiError:
    """The 400 answer for a `RouteError`: its message, and `far_pin` when a pin is off-road."""
    far = far_pin_index(exc, pins)
    return ApiError(400, route_error_message(exc, pins), None if far is None else {"far_pin": far})


def route_error_message(exc: Exception, pins: list[tuple[float, float]]) -> str:
    """A plain-English message for a `RouteError`."""
    from driftmaps.geo.router import LOOP_MESSAGE

    text = str(exc)
    if "no loop" in text:
        return LOOP_MESSAGE
    i = far_pin_index(exc, pins)
    if i is not None:
        return f"No road found near point {_pin_label(i, len(pins))} - click closer to a road."
    if "same point" in text:
        return "Your points land on the same spot of road - put them further apart."
    if "no route joins" in text:
        return (
            "Couldn't find a way along the roads that joins your points in order without "
            "U-turns. Try moving a point, or add a point on the road you want to take."
        )
    if "no road segments" in text:
        return "No drivable roads found around your points - click on or near a road."
    if "at least two" in text:
        return "Click at least two points on the map: A (the start) and B."
    return f"Couldn't make a route: {text}"


def display_line(route: Route, max_points: int = MAX_LINE_POINTS) -> list[list[float]]:
    """The route as `[[lon, lat], ...]`, Douglas-Peucker simplified to <= `max_points`."""
    import numpy as np
    import shapely

    pl = route.polyline(step_m=1e9)  # one point per OSM edge (+ the end): the node path
    lonlat = np.asarray(pl.lonlat, np.float64)
    if len(lonlat) > 2:
        proj = route.projection()
        xy = np.asarray(proj.to_xy(lonlat), np.float64)
        line = shapely.LineString(xy)
        tol = 0.5
        out = xy
        while True:
            simple = line.simplify(tol, preserve_topology=False)
            out = np.asarray(simple.coords, np.float64)
            if len(out) <= max_points or tol > 1e5:
                break
            tol *= 2.0
        if len(out) >= 2:
            lonlat = np.asarray(proj.from_xy(out), np.float64)
    return [[round(float(lon), 6), round(float(lat), 6)] for lon, lat in lonlat]


def road_names(route: Route) -> list[str]:
    """Road names (or refs) along the route in driving order, each once."""
    names: list[str] = []
    for e in route.edges:
        tags = route.ways.get(e.way, {})
        name = (tags.get("name") or tags.get("ref") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _coarse_only(route: Route) -> bool:
    """True if no detailed DEM is planned anywhere along the route (only 30 m GLO-30)."""
    from driftmaps.elevation.sources import CHAINS, region_of

    lonlat = [route.nodes[n] for n in route.node_path() if n in route.nodes]
    if not lonlat:
        return False
    regions = region_of([p[0] for p in lonlat], [p[1] for p in lonlat])
    return all(tuple(CHAINS[str(r)]) == ("glo30",) for r in set(regions.tolist()))


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def _track_info(folder: Path) -> dict[str, object]:
    name, length, loop = folder.name, 0.0, False
    ui_path = folder / "ui" / "ui_track.json"
    try:
        ui = json.loads(ui_path.read_text(encoding="utf-8-sig"))
        if isinstance(ui, dict):
            name = str(ui.get("name") or name)
            length = float(re.sub(r"[^\d.]", "", str(ui.get("length") or "0")) or 0)
            tags = ui.get("tags")
            loop = isinstance(tags, list) and "circuit" in tags
    except (OSError, ValueError):
        pass
    built = 0.0
    for p in (folder / ".driftmaps", ui_path, folder):
        try:
            built = p.stat().st_mtime
            break
        except OSError:
            continue
    return {
        "track_id": folder.name,
        "name": name,
        "length_m": length,
        "built_at": _iso(built),
        "loop": loop,
    }


def _json_default(o: object) -> object:
    item = getattr(o, "item", None)  # numpy scalars
    if callable(item):
        return item()
    tolist = getattr(o, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


# ---------------------------------------------------------------------------------------------
# Routes kept by id
# ---------------------------------------------------------------------------------------------


class RouteStore:
    """Routes by `route_id` (first 16 hex digits of the route.json SHA-256): an in-memory LRU
    backed by `<dir>/<route_id>.json`, plus each route's elevation profile once computed."""

    def __init__(self, routes_dir: Path) -> None:
        self.dir = Path(routes_dir)
        self._routes: OrderedDict[str, Route] = OrderedDict()
        self._profiles: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _touch(d: OrderedDict, key: str, value: object) -> None:
        d[key] = value
        d.move_to_end(key)
        while len(d) > ROUTE_MEMORY:
            d.popitem(last=False)

    def put(self, route: Route) -> str:
        text = route.to_json()
        route_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        with self._lock:
            self._touch(self._routes, route_id, route)
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            path = self.dir / f"{route_id}.json"
            if not path.is_file():
                route.save(path)
                self._prune()
            else:
                path.touch()
        except OSError as exc:  # memory still has it; disk is for app restarts
            log.warning("could not save route %s: %s", route_id, exc)
        return route_id

    def _prune(self) -> None:
        files = sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[ROUTE_FILES_KEPT:]:
            old.unlink(missing_ok=True)

    def get(self, route_id: object) -> Route | None:
        if not isinstance(route_id, str) or not _ROUTE_ID_RE.match(route_id):
            return None
        with self._lock:
            route = self._routes.get(route_id)
            if route is not None:
                self._routes.move_to_end(route_id)
                return route
        from driftmaps.geo.router import Route as RouteCls

        try:
            route = RouteCls.load(self.dir / f"{route_id}.json")
        except (OSError, ValueError, KeyError, TypeError):
            return None
        with self._lock:
            self._touch(self._routes, route_id, route)
        return route

    def profile(self, route_id: str) -> dict | None:
        with self._lock:
            return self._profiles.get(route_id)

    def set_profile(self, route_id: str, profile: dict) -> None:
        with self._lock:
            self._touch(self._profiles, route_id, profile)


# ---------------------------------------------------------------------------------------------
# The app (request handling without HTTP)
# ---------------------------------------------------------------------------------------------


class App:
    """Everything behind the API. Methods take the decoded JSON body and return the JSON
    response, or raise `ApiError`.

    `route_fn` defaults to `driftmaps.geo.router.route_from_pins` and `build_fn` to
    `driftmaps.build.build_track`, both looked up at call time (tests patch them).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        route_fn: Callable[..., Route] | None = None,
        build_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.route_fn = route_fn
        self.routes = RouteStore(self.settings.routes_dir)
        self.jobs = JobQueue(BuildRunner(self.settings, build_fn))
        self._by_pins: OrderedDict[tuple, Route] = OrderedDict()
        self._lock = threading.Lock()

    def close(self) -> None:
        self.jobs.close(timeout=0)

    # -- status and settings ------------------------------------------------------------------

    def status(self) -> dict:
        return self.settings.status()

    def set_settings(self, body: dict) -> dict:
        if "ac_root" not in body:
            raise ApiError(400, "Nothing to change: send ac_root.")
        value = body["ac_root"]
        if value is not None and not isinstance(value, str):
            raise ApiError(400, "ac_root must be a folder path.")
        try:
            self.settings.set_ac_root(value)
        except SettingsError as exc:
            raise ApiError(400, str(exc)) from None
        except OSError as exc:
            raise ApiError(500, f"Couldn't save the settings: {exc}") from None
        return self.status()

    # -- routing --------------------------------------------------------------------------------

    def _call_route(self, pins: list[tuple[float, float]], **kw) -> Route:
        if self.route_fn is not None:
            fn = self.route_fn
        else:
            from driftmaps.geo import router

            fn = router.route_from_pins
        # Our own warnings replace route_from_pins' length warning. The route options come
        # back in the best route's `alternatives`.
        return fn(
            pins, self.settings.cache_dir, max_length_m=math.inf, alternatives=ROUTE_OPTIONS, **kw
        )

    def _base_route(
        self, pins: list[tuple[float, float]], loop: bool = False
    ) -> tuple[Route, list[str]]:
        """The route through the pins as given (not reversed; with `loop`, back to the first
        pin), and data warnings."""
        from driftmaps.geo.router import RouteError

        key = (tuple(pins), loop)
        kw = {"loop": True} if loop else {}  # A->B calls stay exactly as they were
        with self._lock:
            cached = self._by_pins.get(key)
        if cached is not None:
            return cached, []
        notes: list[str] = []
        try:
            route = self._call_route(pins, **kw)
        except RouteError as exc:
            raise route_error(exc, pins) from None
        except Exception as exc:
            if not is_network_error(exc):
                raise
            # Overpass trouble: route on map data saved by an earlier download, if any.
            log.warning("route download failed, trying saved map data: %s", exc)
            try:
                route = self._call_route(pins, offline=True, **kw)
            except RouteError as exc2:
                raise route_error(exc2, pins) from None
            except Exception:
                raise ApiError(502, MAP_SERVERS_MESSAGE) from None
            notes.append(STALE_MAP_WARNING)
            return route, notes  # not remembered: try the map servers again next time
        with self._lock:
            self._by_pins[key] = route
            while len(self._by_pins) > ROUTE_MEMORY:
                self._by_pins.popitem(last=False)
        return route, notes

    def route(self, body: dict) -> dict:
        pins = parse_pins(body.get("pins"))
        reverse = bool(body.get("reverse", False))
        loop = bool(body.get("loop", False))
        legs = pins + [pins[0]] if loop else pins
        span = sum(_haversine_m(a, b) for a, b in zip(legs, legs[1:], strict=False))
        if span > MAX_PIN_SPAN_M:
            raise ApiError(
                400,
                f"Your points are {span / 1000:.0f} km apart - that's too far for one track. "
                f"Keep tracks under about {LONG_ROUTE_M / 1000:.0f} km.",
            )
        try:
            base, notes = self._base_route(pins, loop)
        except ApiError:
            raise
        except Exception as exc:
            log.exception("routing failed")
            raise ApiError(500, f"Something went wrong while finding the route: {exc}") from None
        found = [base, *getattr(base, "alternatives", ())][:ROUTE_OPTIONS]
        routes = [r.reverse() if reverse else r for r in found]
        if routes[0].closed and routes[0].length_m < MIN_LOOP_M:
            raise ApiError(
                400,
                f"That loop is only {routes[0].length_m:.0f} m long - too short for a track. "
                "Add a point further away to make it bigger.",
            )
        options = [
            self._route_option(r, notes)
            for r in routes
            if not (r.closed and r.length_m < MIN_LOOP_M)
        ]
        return {
            **options[0],
            "snapped": [
                [round(p.snap_lat, 7), round(p.snap_lon, 7)] for p in base.pins[: len(pins)]
            ],
            "alternatives": options,
        }

    def _route_option(self, route: Route, notes: list[str]) -> dict:
        """One route option as the page shows it (the route is kept by its id)."""
        route_id = self.routes.put(route)
        warnings_out = list(notes)
        if route.length_m > LONG_ROUTE_M:
            warnings_out.append(
                f"This road is {route.length_m / 1000:.1f} km long - over "
                f"{LONG_ROUTE_M / 1000:.0f} km, the build will take a long time and the track "
                "will be big."
            )
        if route.repeated_nodes():
            warnings_out.append(
                "The loop crosses itself at a junction (a figure 8) - check the line on the map."
                if route.closed
                else "The route crosses or runs over itself - check the line on the map."
            )
        try:
            if _coarse_only(route):
                warnings_out.append(COARSE_WARNING)
        except Exception:  # only a hint
            pass
        return {
            "route_id": route_id,
            "length_m": round(float(route.length_m), 1),
            "line": display_line(route),
            "names": road_names(route),
            "warnings": warnings_out,
            "loop": bool(route.closed),
        }

    def _route_for(self, body: dict) -> tuple[str, Route]:
        route_id = body.get("route_id")
        route = self.routes.get(route_id)
        if route is None:
            raise ApiError(404, ROUTE_GONE)
        return str(route_id), route

    def profile(self, body: dict) -> dict:
        import numpy as np

        from driftmaps.elevation import sources

        route_id, route = self._route_for(body)
        cached = self.routes.profile(route_id)
        if cached is not None:
            return cached
        pl = route.polyline(step_m=5.0)
        total = float(pl.s[-1])
        n = max(2, int(round(total / PROFILE_STEP_M)) + 1)
        s = np.linspace(0.0, total, n)
        lon = np.interp(s, pl.s, pl.lonlat[:, 0])
        lat = np.interp(s, pl.s, pl.lonlat[:, 1])
        try:
            smp = sources.sample_elevation(lon, lat, cache_dir=self.settings.cache_dir)
        except Exception as exc:
            if is_network_error(exc) or isinstance(exc, OSError):
                log.warning("elevation download failed: %s", exc)
                raise ApiError(
                    502, "Couldn't get elevation data - check your internet and try again."
                ) from None
            log.exception("elevation sampling failed")
            raise ApiError(500, f"Something went wrong reading elevation: {exc}") from None
        z = np.asarray(smp.heights, np.float64)
        ok = np.isfinite(z)
        if not ok.any():
            raise ApiError(
                502,
                "Couldn't get elevation data for this road - check your internet and try again.",
            )
        names = [str(x) for x in np.asarray(smp.source)[ok]]
        primary = max(set(names), key=names.count)
        res = np.asarray(smp.resolution_m, np.float64)[ok][np.array(names) == primary]
        info = smp.info.get(primary) if smp.info else None
        resolution = float(np.median(res)) if res.size else float("nan")
        result = {
            "s": [round(float(v), 1) for v in s],
            "z": [round(float(v), 2) if math.isfinite(v) else None for v in z],
            "source": getattr(info, "title", None) or primary,
            "resolution_m": resolution if math.isfinite(resolution) else None,
        }
        self.routes.set_profile(route_id, result)
        return result

    # -- builds ---------------------------------------------------------------------------------

    def build(self, body: dict) -> dict:
        route_id, route = self._route_for(body)
        loop = body.get("loop")
        if loop is not None and bool(loop) != bool(route.closed):
            raise ApiError(
                400,
                "That route isn't a loop - press Back to start and wait for the route first."
                if loop
                else "That route is a loop - turn off Back to start and wait for the route first.",
            )
        name = str(body.get("name") or "").strip()
        if not name:
            raise ApiError(400, "Give the track a name.")
        if len(name) > 80:
            raise ApiError(400, "That name is too long - keep it under 80 characters.")
        raw_id = body.get("track_id")
        if raw_id is not None and not isinstance(raw_id, str):
            raise ApiError(400, "The track id must be text.")
        try:
            track_id = app_track_id(name, raw_id)
        except ValueError:
            raise ApiError(
                400, "That track id can't be used - use letters, numbers and _ only."
            ) from None
        install = body.get("install", True)
        install = True if install is None else bool(install)
        opts = body.get("options") or {}
        if not isinstance(opts, dict):
            raise ApiError(400, "options must be an object.")
        try:
            density = float(opts.get("scenery_density", 1.0))
        except (TypeError, ValueError):
            raise ApiError(400, "scenery_density must be a number.") from None
        density = min(2.0, max(0.0, density)) if math.isfinite(density) else 1.0

        folders = [self.settings.output_dir]
        if install:
            tracks = self.settings.tracks_dir()
            if tracks is None:
                raise ApiError(400, NO_AC_MESSAGE)
            folders.append(tracks)
        conflict = folder_conflict(track_id, folders)
        if conflict:
            raise ApiError(409, conflict)
        job = self.jobs.submit(
            BuildRequest(
                route_id=route_id,
                route=route,
                name=name,
                track_id=track_id,
                install=install,
                trees=bool(opts.get("trees", True)),
                buildings=bool(opts.get("buildings", True)),
                guardrails=bool(opts.get("guardrails", True)),
                scenery_density=density,
            )
        )
        return {"job_id": job.id}

    def job(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            raise ApiError(404, "That build isn't known (the app may have been restarted).")
        return job.snapshot()

    # -- tracks -----------------------------------------------------------------------------------

    def _track_folders(self) -> list[tuple[Path, bool]]:
        out: list[tuple[Path, bool]] = []
        tracks = self.settings.tracks_dir()
        if tracks is not None:
            out.append((tracks, True))
        out.append((self.settings.output_dir, False))
        return out

    def tracks(self) -> list[dict]:
        found: dict[str, dict] = {}
        for base, installed in self._track_folders():
            try:
                folders = sorted(p for p in base.iterdir() if p.is_dir())
            except OSError:
                continue
            for folder in folders:
                try:
                    validate_track_id(folder.name)
                except ValueError:
                    continue
                if not is_driftmaps_track(folder):
                    continue
                entry = found.get(folder.name)
                if entry is None:
                    found[folder.name] = _track_info(folder) | {"installed": installed}
                elif installed:
                    entry["installed"] = True
        return sorted(found.values(), key=lambda t: str(t["built_at"]), reverse=True)

    def delete_track(self, track_id: str) -> dict:
        try:
            validate_track_id(track_id)
        except ValueError:
            raise ApiError(400, "That isn't a valid track id.") from None
        for job in self.jobs.active():
            if getattr(job.request, "track_id", None) == track_id:
                raise ApiError(409, "That track is being built right now - wait for it to finish.")
        ours: list[Path] = []
        foreign = False
        for base, _ in self._track_folders():
            folder = base / track_id
            if not (folder.exists() or folder.is_symlink()):
                continue
            if folder.is_dir() and is_driftmaps_track(folder):
                ours.append(folder)
            else:
                foreign = True
        if not ours:
            if foreign:
                raise ApiError(403, f"{track_id} wasn't made by this app, so it won't be deleted.")
            raise ApiError(404, f"There's no track called {track_id}.")
        for folder in ours:
            try:
                clear_track_folder(folder)
            except OSError as exc:
                log.warning("could not delete %s: %s", folder, exc)
                raise ApiError(
                    500,
                    f"Couldn't delete {folder} - close Assetto Corsa and Content Manager and "
                    "try again.",
                ) from None
        return {"ok": True}


# ---------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------


def static_root():
    """The `static/` folder inside the package, as an importlib.resources Traversable."""
    return importlib.resources.files("driftmaps.app").joinpath("static")


def read_static(path: str) -> tuple[bytes, str] | None:
    """(bytes, content type) for a request path, or None. `/` is index.html; a `/static/`
    prefix is optional; unknown paths without a file extension fall back to index.html."""
    # Decode first, then split: an encoded "%2F" must not hide a ".." inside one segment.
    parts = [p for p in urllib.parse.unquote(path).split("/") if p]
    if parts and parts[0] == "static":
        parts = parts[1:]
    if not parts:
        parts = ["index.html"]
    if any(p in (".", "..") or "\\" in p or ":" in p or "\0" in p for p in parts):
        return None
    candidates = [parts]
    if "." not in parts[-1]:
        candidates.append(["index.html"])
    for cand in candidates:
        res = static_root()
        try:
            for p in cand:
                res = res.joinpath(p)
            if res.is_file():
                suffix = "." + cand[-1].rsplit(".", 1)[-1].lower() if "." in cand[-1] else ""
                ctype = _CONTENT_TYPES.get(suffix, "application/octet-stream")
                return res.read_bytes(), ctype
        except (OSError, ValueError):
            continue
    return None


class _Handler(BaseHTTPRequestHandler):
    server: AppHTTPServer
    server_version = "AssettoCorsaMapGenerator"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib name
        log.debug("%s - %s", self.address_string(), format % args)

    # -- responses ------------------------------------------------------------------------------

    def _send(self, status: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, default=_json_default, allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    # -- request checks -------------------------------------------------------------------------

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        name = host.rsplit(":", 1)[0] if not host.endswith("]") else host
        return name in _LOCAL_HOSTS

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        parts = urllib.parse.urlsplit(origin.strip().lower())
        try:
            port = parts.port
        except ValueError:
            return False
        return (
            parts.scheme == "http"
            and (parts.hostname or "") in {"127.0.0.1", "localhost", "::1"}
            and port == self.server.server_port
        )

    def _body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ApiError(400, "Bad request.") from None
        if length > MAX_BODY_BYTES:
            raise ApiError(413, "Request too big.")
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ApiError(400, "Bad request: the body isn't JSON.") from None
        if not isinstance(data, dict):
            raise ApiError(400, "Bad request: send a JSON object.")
        return data

    # -- dispatch -------------------------------------------------------------------------------

    def _handle(self, method: str) -> None:
        path = urllib.parse.urlsplit(self.path).path
        try:
            if not self._host_ok():
                raise ApiError(403, "This app only answers on 127.0.0.1.")
            if method in ("POST", "DELETE") and not self._origin_ok():
                raise ApiError(403, "Requests must come from the app's own page.")
            if path == "/api" or path.startswith("/api/"):
                self._api(method, path)
            elif method in ("GET", "HEAD"):
                self._static(path)
            else:
                raise ApiError(405, "Not allowed.")
        except ApiError as exc:
            self._json(exc.status, {**exc.extra, "error": exc.message})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            log.exception("error handling %s %s", method, path)
            try:
                self._error(500, f"Something went wrong in the app: {exc}")
            except OSError:
                pass

    def _api(self, method: str, path: str) -> None:
        app = self.server.app
        segs = [urllib.parse.unquote(s) for s in path.split("/")[2:] if s]
        key = (method, segs[0] if segs else "", len(segs))
        if key == ("GET", "status", 1):
            self._json(200, app.status())
        elif key == ("POST", "settings", 1):
            self._json(200, app.set_settings(self._body()))
        elif key == ("POST", "route", 1):
            self._json(200, app.route(self._body()))
        elif key == ("POST", "profile", 1):
            self._json(200, app.profile(self._body()))
        elif key == ("POST", "build", 1):
            self._json(200, app.build(self._body()))
        elif key == ("GET", "jobs", 2):
            self._json(200, app.job(segs[1]))
        elif key == ("GET", "tracks", 1):
            self._json(200, app.tracks())
        elif key == ("DELETE", "tracks", 2):
            self._json(200, app.delete_track(segs[1]))
        else:
            known = {"status", "settings", "route", "profile", "build", "jobs", "tracks"}
            if segs and segs[0] in known:
                raise ApiError(405, "Not allowed.")
            raise ApiError(404, "Unknown API address.")

    def _static(self, path: str) -> None:
        found = read_static(path)
        if found is None:
            if path in ("/", "/index.html"):
                raise ApiError(503, "The app's page files are missing - reinstall the app.")
            self._send(404, b"Not found\n", "text/plain; charset=utf-8")
            return
        body, ctype = found
        self._send(200, body, ctype, cache="no-cache")

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        self._handle("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle("HEAD")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle("DELETE")

    def do_PUT(self) -> None:  # noqa: N802
        self._handle("PUT")

    def do_OPTIONS(self) -> None:  # noqa: N802 - no CORS: other sites may not call the API
        self._handle("OPTIONS")


class AppHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second server bind a port already in use; never set it.
    allow_reuse_address = sys.platform != "win32"

    def __init__(self, address: tuple[str, int], app: App) -> None:
        self.app = app
        super().__init__(address, _Handler)

    def server_bind(self) -> None:
        # HTTPServer.server_bind calls socket.getfqdn(), which can stall for seconds on Windows.
        import socketserver

        socketserver.TCPServer.server_bind(self)
        self.server_name = HOST
        self.server_port = self.server_address[1]


class AppServer:
    """The HTTP server for an `App` on 127.0.0.1. `port=None` tries 8765..8774, then any free
    port; `port=0` takes any free port; another number must be free."""

    def __init__(self, app: App | None = None, port: int | None = None) -> None:
        self.app = app or App()
        self.httpd = self._bind(port)
        self._thread: threading.Thread | None = None

    def _bind(self, port: int | None) -> AppHTTPServer:
        if port is not None:
            return AppHTTPServer((HOST, port), self.app)
        for p in range(DEFAULT_PORT, DEFAULT_PORT + PORT_TRIES):
            try:
                return AppHTTPServer((HOST, p), self.app)
            except OSError:
                continue
        return AppHTTPServer((HOST, 0), self.app)

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}/"

    def start(self) -> AppServer:
        """Serve on a background thread (tests, embedding)."""
        self._thread = threading.Thread(
            target=self.httpd.serve_forever, name="drift-maps-http", daemon=True
        )
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def stop(self) -> None:
        if self._thread is not None:
            self.httpd.shutdown()
            self._thread.join(5)
            self._thread = None
        self.httpd.server_close()
        self.app.close()


def running_instance(port: int = DEFAULT_PORT, timeout: float = 1.5) -> str | None:
    """The URL of this app already running on `port`, if it is (else None)."""
    url = f"http://{HOST}:{port}/"
    try:
        req = urllib.request.Request(url + "api/status", headers={"Accept": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    return url if isinstance(data, dict) and data.get("app") == APP_NAME else None


def _say(text: str) -> None:
    if sys.stdout is not None:  # a windowed (no console) frozen app has no stdout
        try:
            print(text, flush=True)
        except OSError:
            pass


def _setup_logging(data_dir: Path) -> None:
    root = logging.getLogger()
    if getattr(root, "_driftmaps_app", False):
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(data_dir / "app.log", mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)
        root.addHandler(fh)
    except OSError:
        pass
    if sys.stderr is not None:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        sh.setLevel(logging.WARNING)
        root.addHandler(sh)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    root._driftmaps_app = True  # type: ignore[attr-defined]


def run(
    port: int | None = None,
    open_browser: bool = True,
    data_dir: Path | str | None = None,
) -> None:
    """Start the app and serve until Ctrl+C: print the URL and open it in the browser.

    With `port=None`, if this app already runs on 8765 its page is just opened again.
    """
    settings = Settings(data_dir)
    _setup_logging(settings.data_dir)
    if port is None:
        existing = running_instance(DEFAULT_PORT)
        if existing:
            _say(f"{APP_NAME} is already running at {existing}")
            if open_browser:
                webbrowser.open(existing)
            return
    try:
        server = AppServer(App(settings), port)
    except OSError as exc:
        _say(f"Couldn't start the app on port {port}: {exc}")
        raise SystemExit(1) from None
    url = server.url
    _say(f"{APP_NAME} is running at {url}")
    _say("Keep this window open while you use the app. Press Ctrl+C to stop it.")
    log.info("serving at %s (data folder %s)", url, settings.data_dir)
    if open_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _say("Stopping.")
    finally:
        server.stop()
