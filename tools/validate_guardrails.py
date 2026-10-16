"""Validate the automatic (warrant) guardrails against guard rails mapped in OpenStreetMap.

    uv run python tools/validate_guardrails.py                # all areas, default options
    uv run python tools/validate_guardrails.py --sweep        # + a small parameter sweep
    uv run python tools/validate_guardrails.py --area governors_road --h-scale 1.5

For each area (an Ontario road stretch where OSM has `barrier=guard_rail` mapped, within NRCan
HRDEM lidar coverage; see AREAS): find the named road and the rails beside it (Overpass), put
pins on the road 500 m before the first and after the last rail (at most `MAX_ROUTE_M`), route
with the project's router (`geo.router.route_from_pins`, read-only use), shape the road like a
build (`build.prepare_geometry`), run `geometry.warrants` on it, and compare the predicted spans
with the mapped ones per side, by length, inside A..B and off railed bridge decks:

- precision = predicted length that is also mapped / predicted length,
- recall    = mapped length that is also predicted / mapped length,
both strict and with a 10 m tolerance at span ends. OSM is incomplete (a missing rail is not
necessarily a false positive), so precision is a lower bound.

Writes `out/guardrails_validate/<area>.png` (top-down: lidar hillshade, road, mapped rails blue
and outside, predicted rails yellow and inside, hazard labels) plus detail crops of the worst
disagreements, and `out/guardrails_validate/results.json`. Everything downloaded is cached
(Overpass, HRDEM), so a second run is offline-fast.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import math
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from driftmaps.build import BuildOptions, dtm_height_fn, prepare_geometry, railed_decks
from driftmaps.geo.barriers import (
    BarrierSpan,
    fetch_barriers,
    fetch_roadside,
    match_barriers,
    merge_spans,
)
from driftmaps.geo.overpass import (
    OverpassError,
    default_cache_dir,
    post_query,
    query_cache_key,
)
from driftmaps.geo.router import Route, route_from_pins
from driftmaps.geometry.bridges import BridgeOptions, bridge_spans, plan_bridge
from driftmaps.geometry.road import road_columns
from driftmaps.geometry.warrants import (
    WarrantInputs,
    WarrantOptions,
    WarrantResult,
    warrant_inputs,
    warrant_spans,
)

OUT = Path("out/guardrails_validate")
MAX_ROUTE_M = 6000.0
MARGIN_M = 500.0
TOL_M = 10.0


@dataclass(frozen=True)
class Area:
    name: str
    road: str  # OSM name of the road
    bbox: tuple[float, float, float, float]  # south, west, north, east
    note: str = ""


# Found 2026-10-15 with an Overpass census of `barrier=guard_rail` in southern Ontario (568
# ways, mostly Toronto expressways): the rural or escarpment roads with rails mapped beside
# them. All inside the HRDEM 1 m mosaic (checked per run: areas whose road is not on lidar are
# skipped).
AREAS: tuple[Area, ...] = (
    Area("hwy15_seeleys_bay", "Highway 15", (44.34, -76.36, 44.42, -76.28),
         "primary 80 km/h, Rideau lakes, rock cuts"),
    Area("hwy15_elgin", "Highway 15", (44.59, -76.29, 44.67, -76.20),
         "primary 80 km/h, lakes and wetlands"),
    Area("governors_road", "Governors Road", (43.235, -80.04, 43.265, -79.98),
         "secondary 60 km/h, climbs out of the Dundas valley"),
    Area("black_bridge_road", "Black Bridge Road", (43.43, -80.34, 43.47, -80.29),
         "tertiary 50 km/h, Speed River crossing, Cambridge"),
    Area("mountainview_grimsby", "Mountainview Road North", (43.17, -79.52, 43.21, -79.48),
         "tertiary, Niagara escarpment climb"),
    Area("mountain_brow", "Mountain Brow Boulevard", (43.20, -79.85, 43.23, -79.79),
         "secondary, Hamilton escarpment brow (suburban)"),
)  # fmt: skip


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------


def _overpass(query: str, tries: int = 4) -> dict:
    key = query_cache_key(query)
    path = default_cache_dir() / "overpass" / f"{key}.json.gz"
    if path.is_file():
        return json.loads(gzip.decompress(path.read_bytes()))
    last: Exception | None = None
    for attempt in range(tries):
        try:
            body = post_query(query, rounds=2, timeout_s=200)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gzip.compress(body, mtime=0))
            return json.loads(body)
        except OverpassError as e:
            last = e
            time.sleep(20 * (attempt + 1))
    raise OverpassError(f"Overpass failed {tries} times: {last}")


def _retry(fn: Callable, tries: int = 4):
    last: Exception | None = None
    for attempt in range(tries):
        try:
            return fn()
        except OverpassError as e:
            last = e
            time.sleep(20 * (attempt + 1))
    raise OverpassError(str(last))


def _xy(lonlat: np.ndarray, lat0: float) -> np.ndarray:
    k = 111_320.0
    return np.stack([lonlat[:, 0] * k * math.cos(math.radians(lat0)), lonlat[:, 1] * k], axis=1)


def _lonlat(xy: np.ndarray, lat0: float) -> np.ndarray:
    k = 111_320.0
    return np.stack([xy[:, 0] / (k * math.cos(math.radians(lat0))), xy[:, 1] / k], axis=1)


def find_pins(area: Area) -> list[tuple[float, float]]:
    """Two (lat, lon) pins on the named road around the rails mapped beside it."""
    s, w, n, e = area.bbox
    q = (
        f"[out:json][timeout:90];"
        f'way["highway"]["name"="{area.road}"]({s},{w},{n},{e})->.h;'
        f'way["barrier"="guard_rail"](around.h:20)->.r;'
        ".h out geom;.r out geom;"
    )
    d = _overpass(q)
    lat0 = 0.5 * (s + n)
    roads, rails = [], []
    for el in d["elements"]:
        g = el.get("geometry") or []
        if len(g) < 2 or any(p is None for p in g):
            continue
        xy = _xy(np.array([[p["lon"], p["lat"]] for p in g]), lat0)
        (rails if el.get("tags", {}).get("barrier") == "guard_rail" else roads).append(xy)
    if not roads or not rails:
        raise RuntimeError(f"{area.name}: no road '{area.road}' or no rails beside it")
    merged = shapely.line_merge(shapely.MultiLineString([shapely.LineString(r) for r in roads]))
    lines = list(getattr(merged, "geoms", [merged]))
    # The merged line with the most rail beside it.
    best, best_len, best_d = None, -1.0, None
    for line in lines:
        d_along = []
        total = 0.0
        for r in rails:
            if line.distance(shapely.LineString(r)) < 25.0:
                for p in r:
                    d_along.append(line.project(shapely.Point(p)))
                total += float(np.linalg.norm(np.diff(r, axis=0), axis=1).sum())
        if d_along and total > best_len:
            best, best_len, best_d = line, total, d_along
    if best is None:
        raise RuntimeError(f"{area.name}: no rail within 25 m of '{area.road}'")
    a = max(0.0, min(best_d) - MARGIN_M)
    b = min(best.length, max(best_d) + MARGIN_M)
    if b - a > MAX_ROUTE_M:  # keep the densest MAX_ROUTE_M window of rail points
        pts = np.sort(np.asarray(best_d))
        starts = np.clip(pts - MARGIN_M, 0.0, None)
        counts = [np.sum((pts >= x) & (pts <= x + MAX_ROUTE_M)) for x in starts]
        a = float(starts[int(np.argmax(counts))])
        b = min(best.length, a + MAX_ROUTE_M)
    pa = np.array(best.interpolate(a).coords[0])[None, :]
    pb = np.array(best.interpolate(b).coords[0])[None, :]
    la, lb = _lonlat(pa, lat0)[0], _lonlat(pb, lat0)[0]
    return [(float(la[1]), float(la[0])), (float(lb[1]), float(lb[0]))]


class _MemoDtm:
    """Caches DTM samples by query so a parameter sweep samples the lidar once."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self.fn = fn
        self.memo: dict[str, np.ndarray] = {}

    def __call__(self, xz: np.ndarray) -> np.ndarray:
        xz = np.ascontiguousarray(np.asarray(xz, np.float64).reshape(-1, 2))
        key = hashlib.sha1(xz.tobytes()).hexdigest()
        if key not in self.memo:
            self.memo[key] = np.asarray(self.fn(xz), np.float64)
        return self.memo[key].copy()


@dataclass
class Prepared:
    area: Area
    route: Route
    geo: object
    inputs: WarrantInputs
    mapped: list[BarrierSpan]
    decks: list[tuple[float, float]]
    source: str


def prepare(area: Area, cache: Path) -> Prepared:
    rfile = cache / f"{area.name}_route.json"
    if rfile.is_file():
        route = Route.load(rfile)
    else:
        pins = find_pins(area)
        route = _retry(lambda: route_from_pins(pins))
        route.save(rfile)
    opts = BuildOptions(name=area.name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geo = prepare_geometry(route, opts)
    m = geo.model
    dtm = _MemoDtm(dtm_height_fn(geo))
    plans = [
        plan_bridge(m, a, b, dtm, BridgeOptions(), k) for k, (a, b) in enumerate(bridge_spans(m))
    ]
    decks = railed_decks(plans)
    c = m.centre[::5]
    lon, lat = geo.projection.to_lonlat(c[:, 0], -c[:, 2])
    lonlat = np.stack([np.asarray(lon), np.asarray(lat)], axis=1)
    features = _retry(lambda: fetch_roadside(lonlat))
    osm = _retry(lambda: fetch_barriers(lonlat, kinds=("guard_rail",)))
    mapped = merge_spans(
        [sp for sp in match_barriers(osm, geo.projection, m.centre[:, [0, 2]], m.s)], join_m=2.0
    )
    inputs = warrant_inputs(
        m,
        dtm,
        geo.projection,
        route,
        geo.way_id,
        geo.oneway,
        decks=decks,
        features=features,
        s_range=(geo.s_a, geo.s_b),
    )
    inputs.driving_side = geo.driving_side
    return Prepared(area, route, geo, inputs, mapped, decks, str(geo.elevation.primary))


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------


def _intervals(spans: Sequence[BarrierSpan], side: str) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted((sp.s0, sp.s1) for sp in spans if sp.side == side):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _clip(iv, lo, hi, holes) -> list[tuple[float, float]]:
    pieces = [(max(a, lo), min(b, hi)) for a, b in iv if min(b, hi) > max(a, lo)]
    for h0, h1 in holes:
        nxt = []
        for a, b in pieces:
            if h1 <= a or h0 >= b:
                nxt.append((a, b))
                continue
            if h0 > a:
                nxt.append((a, h0))
            if h1 < b:
                nxt.append((h1, b))
        pieces = nxt
    return pieces


def _length(iv) -> float:
    return float(sum(b - a for a, b in iv))


def _overlap(a_iv, b_iv) -> float:
    total = 0.0
    for a0, a1 in a_iv:
        for b0, b1 in b_iv:
            total += max(0.0, min(a1, b1) - max(a0, b0))
    return total


def _dilate(iv, d):
    out: list[list[float]] = []
    for a, b in sorted((a - d, b + d) for a, b in iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def score(p: Prepared, predicted: Sequence[BarrierSpan]) -> dict:
    lo, hi = p.geo.s_a, p.geo.s_b
    holes = [(a - 2.0, b + 2.0) for a, b in p.decks]
    res = {"pred_m": 0.0, "mapped_m": 0.0, "tp_m": 0.0, "tp_pred_tol_m": 0.0, "tp_map_tol_m": 0.0}
    for side in ("left", "right"):
        pr = _clip(_intervals(predicted, side), lo, hi, holes)
        mp = _clip(_intervals(p.mapped, side), lo, hi, holes)
        res["pred_m"] += _length(pr)
        res["mapped_m"] += _length(mp)
        res["tp_m"] += _overlap(pr, mp)
        res["tp_pred_tol_m"] += _overlap(pr, _dilate(mp, TOL_M))
        res["tp_map_tol_m"] += _overlap(mp, _dilate(pr, TOL_M))
    return res


def summarise(rows: dict[str, dict]) -> dict:
    tot = {k: sum(r[k] for r in rows.values()) for k in next(iter(rows.values()))}
    return {
        **tot,
        "precision": tot["tp_m"] / tot["pred_m"] if tot["pred_m"] else float("nan"),
        "recall": tot["tp_m"] / tot["mapped_m"] if tot["mapped_m"] else float("nan"),
        "precision_tol": tot["tp_pred_tol_m"] / tot["pred_m"] if tot["pred_m"] else float("nan"),
        "recall_tol": tot["tp_map_tol_m"] / tot["mapped_m"] if tot["mapped_m"] else float("nan"),
    }


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


# --------------------------------------------------------------------------------------------
# Pictures
# --------------------------------------------------------------------------------------------

_MAPPED = (30, 90, 230)
_PRED = (255, 215, 0)
_BOTH = (40, 200, 70)


def render(p: Prepared, result: WarrantResult, path: Path, window=None, px_max=1800) -> None:
    """Top-down picture: lidar hillshade within 70 m of the road, the road, mapped rails
    (blue, drawn outside) and predicted rails (yellow, inside), hazard labels, km ticks."""
    m = p.geo.model
    lo, hi = window or (p.geo.s_a, p.geo.s_b)
    sel = (m.s >= lo - 50) & (m.s <= hi + 50)
    xz = m.centre[sel][:, [0, 2]]
    a, b = xz.min(axis=0) - 80.0, xz.max(axis=0) + 80.0
    scale = max(float(np.max(b - a)) / px_max, 0.4)
    w, h = (np.ceil((b - a) / scale).astype(int) + 1).tolist()
    gx, gz = np.meshgrid(a[0] + (np.arange(w) + 0.5) * scale, a[1] + (np.arange(h) + 0.5) * scale)
    grid = np.stack([gx.ravel(), gz.ravel()], axis=1)
    near = cKDTree(xz).query(grid, distance_upper_bound=70.0)[0] < 70.0
    img = np.full((h * w, 3), 225.0)
    if near.any():
        z = np.full(len(grid), np.nan)
        z[near] = p.inputs.dtm(grid[near])
        zz = z.reshape(h, w)
        dzx = np.gradient(zz, scale, axis=1)
        dzz = np.gradient(zz, scale, axis=0)
        # Light from the north-west (AC: -X, -Z), elevation 45 deg.
        nrm = np.stack([-dzx, np.ones_like(zz), -dzz], axis=-1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        light = np.array([-0.5, 0.7, -0.5])
        shade = np.clip(nrm @ (light / np.linalg.norm(light)), 0.0, 1.0).reshape(-1)
        slope = np.hypot(dzx, dzz).reshape(-1)
        base = np.where(slope[:, None] > 1 / 3, [205, 160, 120], [150, 190, 130])
        ok = near & np.isfinite(shade)
        img[ok] = base[ok] * (0.35 + 0.8 * shade[ok, None])
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8).reshape(h, w, 3))
    dr = ImageDraw.Draw(im)

    def px(pts: np.ndarray) -> list[tuple[float, float]]:
        return [tuple(v) for v in ((pts[:, [0, 2]] - a) / scale).tolist()]

    for poly in p.inputs.water:  # deep-water edges
        for g in getattr(poly, "geoms", [poly]):
            c = np.asarray(g.coords)
            if len(c) > 1:
                dr.line(
                    [tuple(v) for v in ((c - a) / scale).tolist()], fill=(40, 120, 255), width=1
                )
    hw = m.half_width
    road = np.flatnonzero(sel)
    lc = m.centre + hw[:, None] * m.left
    rc = m.centre - hw[:, None] * m.left
    for i in road[:-1]:
        dr.polygon(px(np.stack([lc[i], lc[i + 1], rc[i + 1], rc[i]])), fill=(70, 70, 74))
    for d0, d1 in p.decks:
        on = road[(m.s[road] >= d0) & (m.s[road] <= d1)]
        if len(on) > 1:
            dr.line(px(m.centre[on]), fill=(255, 255, 255), width=max(2, int(6 / scale)))
    for c in p.inputs.connections:
        if lo <= c.s <= hi:
            pos, _, left = m.at(c.s)
            sgn = 1.0 if c.side == "left" else -1.0
            q = np.stack([pos + sgn * 3 * left, pos + sgn * 12 * left])
            dr.line(px(q), fill=(255, 0, 255), width=2)

    def draw(spans, extra: float, colour, width: int) -> None:
        for sp in spans:
            sgn = 1.0 if sp.side == "left" else -1.0
            col = road_columns(m, sgn * (m.outer + extra))
            on = (m.s >= max(sp.s0, lo)) & (m.s <= min(sp.s1, hi))
            if on.sum() > 1:
                dr.line(px(col[on]), fill=colour, width=width)

    lw = max(2, int(round(1.6 / scale)))
    draw(p.mapped, 4.0, _MAPPED, lw)
    draw(result.spans, 1.0, _PRED, lw)
    for hz in result.hazards:
        if not (lo <= hz.s0 <= hi):
            continue
        pos, _, left = m.at(hz.s0)
        sgn = 1.0 if hz.side == "left" else -1.0
        x, y = px((pos + sgn * (float(m.interp(m.outer, hz.s0)) + 9.0) * left)[None, :])[0]
        label = {"fall": f"{hz.h_m:.1f}m@{hz.ratio:.1f}:1", "water": "water",
                 "bridge": "bridge", "rock": "rock"}[hz.kind]  # fmt: skip
        dr.text((x, y), label, fill=(0, 0, 0))
    for d in np.arange(0.0, p.geo.s_b - p.geo.s_a + 1.0, 250.0):
        s = p.geo.s_a + d
        if not lo <= s <= hi:
            continue
        pos, _, left = m.at(s)
        x, y = px((pos + 20.0 * left)[None, :])[0]
        dr.text((x, y), f"{d / 1000:.2f}", fill=(120, 0, 0))
    items = [
        ("mapped (OSM)", _MAPPED),
        ("predicted", _PRED),
        ("side road/driveway", (255, 0, 255)),
        ("deep water edge", (40, 120, 255)),
        ("bridge deck", (255, 255, 255)),
        ("slope > 3:1", (205, 160, 120)),
    ]
    dr.rectangle((4, 4, 170, 22 + 13 * len(items)), fill=(255, 255, 255), outline=(0, 0, 0))
    dr.text((8, 6), f"{p.area.name} {lo - p.geo.s_a:.0f}-{hi - p.geo.s_a:.0f} m", fill=(0, 0, 0))
    for k, (label, colour) in enumerate(items):
        y = 20 + 13 * k
        dr.rectangle((8, y + 1, 18, y + 10), fill=colour, outline=(0, 0, 0))
        dr.text((24, y), label, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def worst_windows(p: Prepared, predicted, n: int = 2, size: float = 500.0):
    """Up to `n` windows (station ranges) with the most disagreement."""
    lo, hi = p.geo.s_a, p.geo.s_b
    s = np.arange(lo, hi, 5.0)
    bad = np.zeros(len(s))
    for side in ("left", "right"):
        pr = _intervals(predicted, side)
        mp = _intervals(p.mapped, side)
        inp = np.array([any(a <= x <= b for a, b in pr) for x in s])
        inm = np.array([any(a <= x <= b for a, b in mp) for x in s])
        bad += inp != inm
    out = []
    k = int(size / 5.0)
    score_ = np.convolve(bad, np.ones(k), mode="same")
    for _ in range(n):
        if score_.max() <= 0:
            break
        i = int(np.argmax(score_))
        c = s[i]
        out.append((max(lo, c - size / 2), min(hi, c + size / 2)))
        score_[max(0, i - k) : i + k] = 0
    return out


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------


def run(prepared: list[Prepared], opt: WarrantOptions) -> tuple[dict, dict[str, WarrantResult]]:
    rows, results = {}, {}
    for p in prepared:
        res = warrant_spans(p.inputs, opt)
        results[p.area.name] = res
        rows[p.area.name] = score(p, res.spans)
    return rows, results


def _fmt_row(name: str, r: dict) -> str:
    pr = r["tp_m"] / r["pred_m"] if r["pred_m"] else float("nan")
    rc = r["tp_m"] / r["mapped_m"] if r["mapped_m"] else float("nan")
    prt = r["tp_pred_tol_m"] / r["pred_m"] if r["pred_m"] else float("nan")
    rct = r["tp_map_tol_m"] / r["mapped_m"] if r["mapped_m"] else float("nan")
    return (
        f"{name:24s} pred {r['pred_m']:6.0f} m  mapped {r['mapped_m']:6.0f} m  "
        f"P {pr:5.2f} R {rc:5.2f}  (tol {TOL_M:.0f} m: P {prt:5.2f} R {rct:5.2f})"
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--area", action="append", help="Only these areas (repeatable).")
    ap.add_argument("--h-scale", type=float, default=None)
    ap.add_argument("--min-hazard", type=float, default=None)
    ap.add_argument("--v-margin", type=float, default=None)
    ap.add_argument("--sweep", action="store_true", help="Also sweep the calibration knobs.")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    out: Path = args.out
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    areas = [a for a in AREAS if not args.area or a.name in args.area]
    prepared: list[Prepared] = []
    skipped: dict[str, str] = {}
    for area in areas:
        t0 = time.perf_counter()
        try:
            p = prepare(area, cache)
        except Exception as e:  # noqa: BLE001 - report and go on with the other areas
            skipped[area.name] = f"{type(e).__name__}: {e}"
            print(f"{area.name:24s} SKIPPED ({skipped[area.name][:160]})", flush=True)
            continue
        if not p.source.startswith("hrdem"):
            skipped[area.name] = f"not on lidar (DEM {p.source})"
            print(f"{area.name:24s} SKIPPED ({skipped[area.name]})", flush=True)
            continue
        prepared.append(p)
        print(
            f"{area.name:24s} {p.geo.length_m / 1000:.2f} km, DEM {p.source}, "
            f"{len(p.mapped)} mapped spans, {len(p.decks)} railed decks, "
            f"{len(p.inputs.water)} water edges, {len(p.inputs.connections)} joins "
            f"({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )
    if not prepared:
        print("no areas to score")
        return 1

    base = WarrantOptions()
    chosen = dataclasses.replace(
        base,
        **{
            k: v
            for k, v in (
                ("h_scale", args.h_scale),
                ("min_hazard_m", args.min_hazard),
                ("v_margin_kmh", args.v_margin),
            )
            if v is not None
        },
    )
    report: dict = {"areas": {}, "skipped": skipped}
    for label, opt in (("default", base), ("chosen", chosen)):
        if label == "chosen" and opt == base:
            continue
        rows, results = run(prepared, opt)
        print(f"\n== {label}: h_scale {opt.h_scale}, min_hazard {opt.min_hazard_m} m, "
              f"V margin {opt.v_margin_kmh} km/h")  # fmt: skip
        for name, r in rows.items():
            print(_fmt_row(name, r))
        tot = summarise(rows)
        print(_fmt_row("TOTAL", tot))
        report[label] = {
            "options": {"h_scale": opt.h_scale, "min_hazard_m": opt.min_hazard_m,
                        "v_margin_kmh": opt.v_margin_kmh},
            "rows": rows,
            "total": tot,
        }  # fmt: skip
        if not args.no_png:
            for p in prepared:
                res = results[p.area.name]
                render(p, res, out / f"{p.area.name}_{label}.png")
                for k, win in enumerate(worst_windows(p, res.spans)):
                    render(p, res, out / f"{p.area.name}_{label}_detail{k}.png", window=win,
                           px_max=1200)  # fmt: skip
                report["areas"].setdefault(p.area.name, {})[label] = res.report
    if args.sweep:
        print("\n== sweep (TOTAL, 10 m tolerance)")
        sweep = []
        for v_margin in (0.0, 10.0, 20.0):
            for h_scale in (0.75, 1.0, 1.25, 1.5, 2.0, 2.5):
                for min_hazard in (10.0, 20.0, 30.0):
                    opt = dataclasses.replace(
                        base, h_scale=h_scale, min_hazard_m=min_hazard, v_margin_kmh=v_margin
                    )
                    tot = summarise(run(prepared, opt)[0])
                    f1 = _f1(tot["precision_tol"], tot["recall_tol"])
                    sweep.append(
                        {"v_margin_kmh": v_margin, "h_scale": h_scale,
                         "min_hazard_m": min_hazard, **tot, "f1_tol": f1}
                    )  # fmt: skip
                    print(
                        f"V+{v_margin:<4.0f} h_scale {h_scale:4.2f} min_hazard {min_hazard:4.0f}:"
                        f" pred {tot['pred_m']:6.0f} P {tot['precision_tol']:.2f}"
                        f" R {tot['recall_tol']:.2f} F1 {f1:.2f}",
                        flush=True,
                    )
        report["sweep"] = sweep
    (out / "results.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"\nwrote {out}/results.json and the PNGs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
