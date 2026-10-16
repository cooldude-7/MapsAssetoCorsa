"""Render a built Assetto Corsa track (KN5) to PNG images, without Assetto Corsa.

The KN5 is converted to a GLB (cached next to it), loaded into three.js in headless Chromium,
and each requested view is saved as a PNG. Loading is the slow part, so ask for all views in one
run. Examples (from any checkout of the repo):

    uv run python tools/kn5view/render.py TRACK_DIR --station 0 --out shots/
    uv run python tools/kn5view/render.py TRACK_DIR --stations 0,2984,7282 --look forward,back
    uv run python tools/kn5view/render.py TRACK_DIR --at 43.4547,-79.8876 --aerial 120 --tilt 30
    uv run python tools/kn5view/render.py TRACK_DIR --topdown 43.4547,-79.8876 --size 300

TRACK_DIR is a built track folder (with route.json and ai/fast_lane.ai) or a .kn5 file.
Stations are metres along the AI line measured from the A start gate (AC_AB_START_*).
"""

from __future__ import annotations

import argparse
import base64
import functools
import json
import math
import re
import sys
import threading
import time
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

import kn5glb  # noqa: E402
from chromium import Chromium, ChromiumError  # noqa: E402

LOOKS = ("forward", "back", "left", "right")


# --------------------------------------------------------------------------------------------
# Track discovery
# --------------------------------------------------------------------------------------------


@dataclass
class Track:
    kn5: Path
    folder: Path
    route: dict | None
    ai_path: Path | None
    lighting: dict[str, float]


def find_kn5(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise SystemExit(f"not found: {path}")
    same = path / f"{path.name}.kn5"
    if same.exists():
        return same
    kn5s = sorted(path.glob("*.kn5"), key=lambda p: p.stat().st_size, reverse=True)
    if not kn5s:
        raise SystemExit(f"no .kn5 in {path}")
    return kn5s[0]


def _read_lighting(folder: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for ini in (folder / "data" / "lighting.ini", *folder.glob("*/data/lighting.ini")):
        if not ini.exists():
            continue
        for line in ini.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"\s*(SUN_PITCH_ANGLE|SUN_HEADING_ANGLE)\s*=\s*([-\d.]+)", line)
            if m:
                out[m.group(1)] = float(m.group(2))
        break
    return out


def load_track(path: Path) -> Track:
    kn5 = find_kn5(path.resolve())
    folder = kn5.parent
    route_path = folder / "route.json"
    route = json.loads(route_path.read_text(encoding="utf-8")) if route_path.exists() else None
    ai_path = None
    for cand in (
        folder / "ai" / "fast_lane.ai",
        folder / "data" / "ideal_line.ai",
        *sorted(folder.glob("*/ai/fast_lane.ai")),
    ):
        if cand.exists():
            ai_path = cand
            break
    return Track(kn5, folder, route, ai_path, _read_lighting(folder))


# --------------------------------------------------------------------------------------------
# Route geometry
# --------------------------------------------------------------------------------------------


class Route:
    """The AI line as a polyline with arc length, plus the station of the A gate."""

    def __init__(self, ai_path: Path, dummies: dict[str, list[float]]) -> None:
        from driftmaps.ac.ai_line import read_ai

        ai = read_ai(ai_path)
        self.pos = ai.positions.astype(np.float64)
        seg = np.linalg.norm(np.diff(self.pos, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.side_left = ai.side_left.astype(np.float64)
        self.side_right = ai.side_right.astype(np.float64)
        self.origin = 0.0
        self.origin_name = "AI line start"
        for left, right in (("AC_AB_START_L", "AC_AB_START_R"), ("AC_TIME_0_L", "AC_TIME_0_R")):
            if left in dummies and right in dummies:
                mid = (np.array(dummies[left]) + np.array(dummies[right])) / 2
                self.origin = self.project(mid)
                self.origin_name = left[:-2]
                break
        else:
            if "AC_START_0" in dummies:
                self.origin = self.project(np.array(dummies["AC_START_0"]))
                self.origin_name = "AC_START_0"

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def project(self, p: np.ndarray) -> float:
        """Arc length of the point on the line nearest to p (in plan, XZ)."""
        a = self.pos[:-1][:, [0, 2]]
        b = self.pos[1:][:, [0, 2]]
        ab = b - a
        t = np.clip(((p[[0, 2]] - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
        d = np.linalg.norm(a + ab * t[:, None] - p[[0, 2]], axis=1)
        i = int(np.argmin(d))
        return float(self.s[i] + t[i] * np.linalg.norm(ab[i]))

    def at(self, station: float) -> tuple[np.ndarray, np.ndarray, float, float]:
        """(position, unit forward in plan, side_left, side_right) at a station from A."""
        s = float(np.clip(self.origin + station, 0.0, self.length))
        p = np.array([np.interp(s, self.s, self.pos[:, k]) for k in range(3)])
        lo = np.array([np.interp(max(0.0, s - 3), self.s, self.pos[:, k]) for k in range(3)])
        hi = np.array(
            [np.interp(min(self.length, s + 3), self.s, self.pos[:, k]) for k in range(3)]
        )
        f = hi - lo
        f[1] = 0.0
        f /= max(np.linalg.norm(f), 1e-9)
        return (
            p,
            f,
            float(np.interp(s, self.s, self.side_left)),
            float(np.interp(s, self.s, self.side_right)),
        )


def latlon_to_xz(route: dict | None, text: str) -> tuple[float, float]:
    if not route or "origin" not in route:
        raise SystemExit("lat/lon needs route.json (with an origin) in the track folder")
    from driftmaps.geo.project import LocalProjection

    lat, lon = (float(v) for v in text.split(","))
    proj = LocalProjection(route["origin"]["lat"], route["origin"]["lon"])
    p = proj.lonlat_to_ac(lon, lat)
    return float(p[0]), float(p[2])


def _floats(text: str, n: int) -> list[float]:
    vals = [float(v) for v in text.replace(" ", "").split(",")]
    if len(vals) != n:
        raise SystemExit(f"expected {n} comma-separated numbers, got {text!r}")
    return vals


# --------------------------------------------------------------------------------------------
# View specs (consumed by viewer.js render())
# --------------------------------------------------------------------------------------------


def driver_view(route: Route, station: float, side: str, look: str, height: float, a) -> dict:
    p, f, sl, sr = route.at(station)
    left = np.array([f[2], 0.0, -f[0]])  # up x forward (driftmaps.ac.coords.left_of)
    # Road centre and half width from the AI line's side distances; sit mid-lane on `side`.
    centre = p + left * (sl - sr) / 2
    half = (sl + sr) / 2
    lane = centre - left * half / 2 if side == "right" else centre + left * half / 2
    d = {"forward": f, "back": -f, "left": left, "right": -left}[look]
    pitch = math.radians(a.pitch)
    look_vec = d * math.cos(pitch) * 100 + np.array([0.0, math.sin(pitch) * 100 + height, 0.0])
    return {
        "anchor": lane.tolist(),
        "snap": {"up": 1.5, "down": 6.0},
        "eyeOffset": [0.0, height, 0.0],
        "lookOffset": look_vec.tolist(),
        "fov": a.fov,
        "near": 0.1,
        "far": 8000,
        "fog": [a.fog_near, a.fog_far],
        "shadowRadius": 200,
    }


def aerial_view(x: float, y: float | None, z: float, up_m: float, tilt: float, heading: float, a):
    t = math.radians(min(max(tilt, 0.0), 85.0))
    h = math.radians(heading)
    f = np.array([math.sin(h), 0.0, -math.cos(h)])
    eye = -f * up_m * math.tan(t) + np.array([0.0, up_m, 0.0])
    dist = float(np.linalg.norm(eye))
    view = {
        "anchor": [x, 0.0 if y is None else y, z],
        "eyeOffset": eye.tolist(),
        "lookOffset": [0.0, 0.0, 0.0],
        "up": f.tolist(),
        "fov": a.fov,
        "near": max(0.5, dist / 1000),
        "far": dist + 8000,
        "fog": [dist + 400, dist + 4000],
        "shadowRadius": max(150.0, dist * 1.2),
    }
    if y is None:
        view["snap"] = {"up": 5000.0, "down": 10000.0}
    return view


def topdown_view(x: float, z: float, size: float, a) -> dict:
    return {
        "anchor": [x, 0.0, z],
        "eyeOffset": [0.0, 3000.0, 0.0],
        "lookOffset": [0.0, 0.0, 0.0],
        "up": [0.0, 0.0, -1.0],  # north up
        "ortho": True,
        "orthoWidth": size,
        "far": 6000,
        "shadowRadius": size * 0.75,
    }


def _tag(v: float, digits: int = 5) -> str:
    return f"{v:.{digits}f}".rstrip("0").rstrip(".")


def build_views(args, track: Track, meta: dict) -> list[tuple[str, dict]]:
    views: list[tuple[str, dict]] = []
    stations: list[float] = []
    for text in args.station + args.stations:
        stations += [float(v) for v in text.split(",") if v.strip()]
    looks = [lk.strip() for lk in args.look.split(",")]
    for lk in looks:
        if lk not in LOOKS:
            raise SystemExit(f"--look must be one of {', '.join(LOOKS)}")
    if stations:
        if not track.ai_path:
            raise SystemExit("--station needs ai/fast_lane.ai in the track folder")
        route = Route(track.ai_path, meta.get("dummies", {}))
        print(
            f"route: AI line {route.length:.0f} m, station 0 = {route.origin_name} "
            f"at {route.origin:.0f} m along it"
        )
        side = args.side or (track.route or {}).get("driving_side", "right")
        for st in stations:
            for lk in looks:
                name = f"station_{st:06.0f}m_{side}_{lk}"
                views.append((name, driver_view(route, st, side, lk, args.height, args)))
    for text in args.at:
        x, z = latlon_to_xz(track.route, text)
        lat, lon = text.split(",")
        name = f"aerial_{_tag(float(lat))}_{_tag(float(lon))}_{args.aerial:.0f}m_t{args.tilt:.0f}"
        views.append((name, aerial_view(x, None, z, args.aerial, args.tilt, args.heading, args)))
    for text in args.at_xyz:
        x, y, z = _floats(text, 3)
        name = f"aerial_xyz_{x:.0f}_{y:.0f}_{z:.0f}_{args.aerial:.0f}m_t{args.tilt:.0f}"
        views.append((name, aerial_view(x, y, z, args.aerial, args.tilt, args.heading, args)))
    for text in args.topdown:
        x, z = latlon_to_xz(track.route, text)
        lat, lon = text.split(",")
        name = f"topdown_{_tag(float(lat))}_{_tag(float(lon))}_{args.size:.0f}m"
        views.append((name, topdown_view(x, z, args.size, args)))
    for text in args.topdown_xyz:
        x, z = _floats(text, 2)
        views.append(
            (f"topdown_xyz_{x:.0f}_{z:.0f}_{args.size:.0f}m", topdown_view(x, z, args.size, args))
        )
    w, h = (int(v) for v in args.res.lower().split("x"))
    for _, v in views:
        v["width"], v["height"] = w, h
    return views


# --------------------------------------------------------------------------------------------
# Local HTTP server: the viewer, three.js and the GLB (fetch() can't read file:// URLs)
# --------------------------------------------------------------------------------------------


class _Handler(SimpleHTTPRequestHandler):
    glb_path: Path = Path()

    def translate_path(self, path: str) -> str:
        if path.split("?", 1)[0] == "/scene.glb":
            return str(self.glb_path)
        return super().translate_path(path)

    def log_message(self, format, *args) -> None:  # noqa: A002 - signature from the base class
        pass


def serve(glb_path: Path) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"glb_path": glb_path})
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(handler, directory=str(TOOL_DIR))
    )
    return server


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        prog="render.py",
        description="Render a built AC track (KN5) to PNGs with three.js in headless Chromium.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    p.add_argument("track", type=Path, help="track folder (with route.json, ai/) or a .kn5 file")
    p.add_argument("--out", type=Path, default=Path("kn5view_out"), help="output folder")
    g = p.add_argument_group("driver views")
    g.add_argument("--station", action="append", default=[], help="metres from A (repeatable)")
    g.add_argument("--stations", action="append", default=[], help="comma list, e.g. 0,500,1000")
    g.add_argument("--side", choices=("right", "left"), help="lane (default: route driving side)")
    g.add_argument("--look", default="forward", help="forward|back|left|right, or a comma list")
    g.add_argument("--height", type=float, default=1.2, help="eye height above the road (m)")
    g.add_argument("--pitch", type=float, default=-2.0, help="gaze pitch in degrees (+ is up)")
    g = p.add_argument_group("aerial / plan views")
    g.add_argument("--at", action="append", default=[], metavar="LAT,LON", help="aerial target")
    g.add_argument("--at-xyz", action="append", default=[], metavar="X,Y,Z", help="AC coords")
    g.add_argument("--aerial", type=float, default=120.0, help="camera height above target (m)")
    g.add_argument("--tilt", type=float, default=35.0, help="degrees from straight down")
    g.add_argument("--heading", type=float, default=0.0, help="aerial view direction, deg from N")
    g.add_argument("--topdown", action="append", default=[], metavar="LAT,LON")
    g.add_argument("--topdown-xyz", action="append", default=[], metavar="X,Z")
    g.add_argument("--size", type=float, default=300.0, help="top-down view width (m)")
    g = p.add_argument_group("rendering")
    g.add_argument("--res", default="1280x720", help="image size WxH")
    g.add_argument("--fov", type=float, default=60.0, help="vertical field of view (deg)")
    g.add_argument("--fog-near", type=float, default=250.0)
    g.add_argument("--fog-far", type=float, default=3000.0)
    g.add_argument("--shadows", action="store_true", help="sun shadows (slower)")
    g.add_argument("--show-collision", action="store_true", help="show hidden walls in red")
    g.add_argument("--double-sided", action="store_true", help="draw every face both sides")
    g.add_argument("--hide", help="regex of mesh names to hide, e.g. '^far|^tree'")
    g.add_argument("--only", help="regex of mesh names to keep (all others hidden)")
    g = p.add_argument_group("cache / debugging")
    g.add_argument("--reconvert", action="store_true", help="rebuild the cached GLB")
    g.add_argument("--glb-only", action="store_true", help="convert to GLB and stop")
    g.add_argument("--cache-dir", type=Path, help="where to cache the GLB (default: by the KN5)")
    g.add_argument("--max-texture", type=int, default=2048, help="downscale bigger textures")
    g.add_argument("--chrome", help="Chromium binary (default: Playwright's, then PATH)")
    g.add_argument("--verbose", action="store_true", help="print the page's console")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t_start = time.perf_counter()
    track = load_track(args.track)
    glb_path, meta = kn5glb.cached_glb(
        track.kn5, cache_dir=args.cache_dir, force=args.reconvert, max_texture=args.max_texture
    )
    t_conv = time.perf_counter()
    print(f"glb: {glb_path} ({glb_path.stat().st_size / 1e6:.1f} MB, {t_conv - t_start:.1f} s)")
    if args.glb_only:
        return 0
    views = build_views(args, track, meta)
    if not views:
        print("no views requested: use --station/--stations, --at/--at-xyz or --topdown")
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    server = serve(glb_path)
    with Chromium(args.chrome, verbose=args.verbose) as browser:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/viewer/index.html"
            browser.open_page(url)
            w, h = (int(v) for v in args.res.lower().split("x"))
            cfg = {
                "glb": "/scene.glb",
                "width": w,
                "height": h,
                "showCollision": args.show_collision,
                "doubleSided": args.double_sided,
                "hide": args.hide,
                "only": args.only,
                "shadows": args.shadows,
                "sunPitch": track.lighting.get("SUN_PITCH_ANGLE", 45.0),
                "sunHeading": track.lighting.get("SUN_HEADING_ANGLE", 135.0),
            }
            t0 = time.perf_counter()
            stats = browser.call("load", cfg, timeout=600)
            t_load = time.perf_counter() - t0
            print(
                f"scene: {stats['meshes']} meshes ({stats['hiddenMeshes']} hidden), "
                f"{stats['triangles']:,.0f} triangles drawn, loaded in {t_load:.1f} s "
                f"[{stats['renderer']}]"
            )
            for name, view in views:
                t0 = time.perf_counter()
                res = browser.call("render", view, timeout=600)
                png = base64.b64decode(res["png"].split(",", 1)[1])
                path = args.out / f"{name}.png"
                path.write_bytes(png)
                info = res["info"]
                ground = "none" if info["ground"] is None else f"{info['ground']:.2f}"
                print(
                    f"{path}  ({time.perf_counter() - t0:.1f} s; eye {info['eye']}, "
                    f"ground y {ground}, {info['drawCalls']} draws)"
                )
        except ChromiumError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        finally:
            server.shutdown()
    print(f"done in {time.perf_counter() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
