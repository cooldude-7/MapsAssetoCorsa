"""The urban street: where a road is curbed, and its cross-section there.

**Urban or rural** (`UrbanContext`), per road, first match wins:
1. OSM says so: a sidewalk on either side (`sidewalk=both|left|right|yes|separate`, also the
   `sidewalk:*` forms) or `lit=yes` with `maxspeed` <= 70 -> urban; `sidewalk=no` (and not lit)
   or `lit=no` -> rural. A multi-lane arterial (4+ lanes, or a 2+ lane one-way carriageway of
   a divided road) signed 70 km/h or less is a town arterial -> urban (Ontario's rural
   multi-lane highways are signed 80 km/h and more).
2. A municipal sidewalk line (Oakville open data, `geo.sidewalks`) beside it -> urban.
3. The neighbourhood: at least `URBAN_STREET_M` of residential streets within
   `URBAN_RADIUS_M`, and `maxspeed` <= 60 (or untagged) -> urban (a subdivision).
4. Otherwise rural (Ontario's rural cross-section: gravel shoulders and ditches).
Along a road the classification is smoothed (`smooth_runs`, `URBAN_MIN_RUN_M` = 150 m): rural
gaps shorter than that between urban stretches are filled, then urban stretches shorter than
that are dropped (the user's Oakville check, 2026-10-16: "some sidewalks breaking/stopping").

**Sidewalks** (`walk_sides`) follow OSM `sidewalk*` tags, then the municipal lines beside the
road (`SidewalkIndex`, up to `SidewalkIndex.FAR_M` from the asphalt edge: arterial
boulevards are wide), then the rule; smoothed the same way, except that a stretch OSM tags
`sidewalk=no` is never filled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import shapely
from scipy.spatial import cKDTree

__all__ = [
    "URBAN_MIN_RUN_M",
    "URBAN_RADIUS_M",
    "URBAN_STREET_M",
    "UrbanContext",
    "sidewalk_sides",
    "smooth_runs",
    "tag_context",
]

URBAN_RADIUS_M = 150.0
URBAN_STREET_M = 500.0  # residential streets within the radius that make a subdivision
URBAN_MIN_RUN_M = 150.0  # rural gaps / urban stretches shorter than this are smoothed away
_STREET_STEP_M = 10.0
_LOCAL_CLASSES = frozenset({"residential", "living_street"})
_YES_SIDES = frozenset({"both", "left", "right", "yes", "separate"})


def _speed(tags: Mapping[str, str]) -> float | None:
    raw = (tags.get("maxspeed") or "").split(";")[0].strip().lower()
    num = "".join(ch for ch in raw if ch.isdigit() or ch == ".")
    try:
        v = float(num)
    except ValueError:
        return None
    return v * 1.609 if "mph" in raw else v


def sidewalk_sides(tags: Mapping[str, str]) -> tuple[bool | None, bool | None]:
    """(left, right) sidewalk per OSM, relative to the way's direction: True / False, or None
    where OSM says nothing. `separate` counts as a sidewalk (mapped as its own way)."""
    left = right = None
    v = (tags.get("sidewalk") or "").strip().lower()
    if v in ("both", "yes", "separate"):
        left = right = True
    elif v == "left":
        left, right = True, False
    elif v == "right":
        left, right = False, True
    elif v in ("no", "none"):
        left = right = False
    both = (tags.get("sidewalk:both") or "").strip().lower()
    if both:
        left = right = both in _YES_SIDES
    for key, side in (("sidewalk:left", "l"), ("sidewalk:right", "r")):
        vv = (tags.get(key) or "").strip().lower()
        if vv:
            if side == "l":
                left = vv in _YES_SIDES
            else:
                right = vv in _YES_SIDES
    return left, right


def tag_context(tags: Mapping[str, str]) -> str:
    """ "urban", "rural" or "" (the tags do not say)."""
    left, right = sidewalk_sides(tags)
    speed = _speed(tags)
    lit = (tags.get("lit") or "").strip().lower()
    if left or right:
        return "urban"
    if lit == "yes" and (speed is None or speed <= 70):
        return "urban"
    if lit == "no" or (left is False and right is False):
        return "rural"
    lanes = (tags.get("lanes") or "").split(";")[0].strip()
    n = int(lanes) if lanes.isdigit() else 0
    oneway = (tags.get("oneway") or "").strip().lower() in ("yes", "1", "true", "-1")
    if speed is not None and speed <= 70 and (n >= 4 or (oneway and n >= 2)):
        return "urban"
    return ""


class UrbanContext:
    """Urban or rural for any road near the route (see the module docstring).

    `streets`: (xz (K, 2), tags) of the roads around (the network's ways, AC XZ); `sidewalks`:
    municipal sidewalk lines (M, 2) in AC XZ, if any."""

    def __init__(
        self,
        streets: Sequence[tuple[np.ndarray, Mapping[str, str]]] = (),
        sidewalks: Sequence[np.ndarray] = (),
    ) -> None:
        pts = []
        for xz, tags in streets:
            if tags.get("highway") in _LOCAL_CLASSES and len(xz) >= 2:
                line = shapely.LineString(np.asarray(xz, np.float64))
                d = np.arange(0.0, line.length, _STREET_STEP_M) + _STREET_STEP_M / 2
                pts.append(shapely.get_coordinates(shapely.line_interpolate_point(line, d)))
        allp = np.concatenate(pts) if pts else np.zeros((0, 2))
        self._streets = cKDTree(allp) if len(allp) else None
        sw = []
        for line in sidewalks:
            ln = shapely.LineString(np.asarray(line, np.float64))
            if ln.length > 0:
                d = np.arange(0.0, ln.length, 5.0)
                sw.append(shapely.get_coordinates(shapely.line_interpolate_point(ln, d)))
        swp = np.concatenate(sw) if sw else np.zeros((0, 2))
        self._sidewalks = cKDTree(swp) if len(swp) else None

    def street_m(self, xz: np.ndarray) -> np.ndarray:
        """Metres of residential street within `URBAN_RADIUS_M` of each point."""
        xz = np.asarray(xz, np.float64).reshape(-1, 2)
        if self._streets is None:
            return np.zeros(len(xz))
        counts = self._streets.query_ball_point(xz, URBAN_RADIUS_M, return_length=True)
        return np.asarray(counts, np.float64) * _STREET_STEP_M

    def sidewalk_near(self, xz: np.ndarray, reach_m: float) -> np.ndarray:
        """Whether a municipal sidewalk line runs within `reach_m` of each point."""
        xz = np.asarray(xz, np.float64).reshape(-1, 2)
        if self._sidewalks is None:
            return np.zeros(len(xz), bool)
        reach = np.broadcast_to(np.asarray(reach_m, np.float64), (len(xz),))
        d, _ = self._sidewalks.query(xz, distance_upper_bound=float(reach.max()) if len(xz) else 0)
        return np.isfinite(d) & (d <= reach)

    def urban(
        self,
        xz: np.ndarray,
        tags_per_point: Sequence[Mapping[str, str]],
        half_width: np.ndarray | float = 4.0,
        s: np.ndarray | None = None,
    ) -> np.ndarray:
        """(N,) urban flags for points along one road with their tags (smoothed along `s`)."""
        xz = np.asarray(xz, np.float64).reshape(-1, 2)
        n = len(xz)
        hw = np.broadcast_to(np.asarray(half_width, np.float64), (n,))
        out = np.zeros(n, bool)
        known = np.zeros(n, bool)
        cache: dict[int, str] = {}
        speeds = np.full(n, np.nan)
        for i, t in enumerate(tags_per_point):
            k = id(t)
            if k not in cache:
                cache[k] = tag_context(t)
            ctx = cache[k]
            if ctx:
                out[i] = ctx == "urban"
                known[i] = True
            sp = _speed(t)
            speeds[i] = np.nan if sp is None else sp
        todo = np.flatnonzero(~known)
        if len(todo):
            walk = self.sidewalk_near(xz[todo], 12.0 + hw[todo])
            dense = self.street_m(xz[todo]) >= URBAN_STREET_M
            slow = ~(speeds[todo] > 60.0)
            out[todo] = walk | (dense & slow)
        if s is not None and n > 2:
            out = smooth_runs(out, np.asarray(s, np.float64), URBAN_MIN_RUN_M)
        return out


def _runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """(start, end, value) of each run of equal values."""
    d = np.flatnonzero(np.diff(mask.astype(np.int8)) != 0) + 1
    edges = np.concatenate([[0], d, [len(mask)]])
    return [(int(a), int(b), bool(mask[a])) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def smooth_runs(
    mask: np.ndarray,
    s: np.ndarray,
    min_m: float = URBAN_MIN_RUN_M,
    fixed: np.ndarray | None = None,
) -> np.ndarray:
    """Continuity along a road (stations at `s`): False gaps shorter than `min_m` between True
    stretches are filled (unless a station of the gap is `fixed`: real data says no), then
    True stretches shorter than `min_m` that do not reach an end of the road are dropped."""
    out = np.asarray(mask, bool).copy()
    s = np.asarray(s, np.float64)
    n = len(out)
    if n < 3:
        return out
    hold = np.zeros(n, bool) if fixed is None else np.asarray(fixed, bool)

    def length(a: int, b: int) -> float:  # metres covered by stations a..b-1 (to the next)
        return float(s[min(b, n - 1)] - s[a]) if b < n else float(s[n - 1] - s[a])

    for a, b, v in _runs(out):
        if not v and 0 < a and b < n and length(a, b) < min_m and not hold[a:b].any():
            out[a:b] = True
    for a, b, v in _runs(out):
        if v and 0 < a and b < n and length(a, b) < min_m:
            out[a:b] = False
    return out


# --------------------------------------------------------------------------------------------
# The urban cross-section (docs/research/intersections.md, streetscape research §4-§5)
# --------------------------------------------------------------------------------------------

# Going out from the asphalt edge: a concrete gutter at road level, the barrier curb (OPSD
# 600-series; Hamilton RD-103: 150 mm reveal, 150 mm top), a grass boulevard rising at 2 %
# (Hamilton RD-113.01: 1.75 m), the concrete sidewalk (1.5 m on local streets, 1.8 m on
# collectors and arterials; OPSD 310.010 crossfall 2 %), then 0.3 m of grass behind it.
GUTTER_M = 0.30  # Hamilton RD-113.01 allows 0.5 m per side for curb and gutter
CURB_REVEAL_M = 0.15
# The curb face rises over 180 mm instead of standing vertical: AC tests tyre contact at one
# point per wheel, so a vertical step makes the wheel jump (as the roundabouts' curbs).
CURB_FACE_RUN_M = 0.18
CURB_TOP_M = 0.15
BOULEVARD_M = 1.75
BOULEVARD_SLOPE = 0.02
SIDEWALK_LOCAL_M = 1.5
SIDEWALK_MAJOR_M = 1.8
SIDEWALK_SLOPE = 0.02
BACK_M = 0.30
DEPRESSED_CURB_M = 0.025  # driveway curb cut: a 25 mm lip (Hamilton RD-106)


def section_offsets(sidewalk_m: np.ndarray | float) -> tuple[list, list]:
    """(offsets beyond the asphalt edge, lifts above the road) of the section's seven column
    lines: asphalt edge, gutter end, curb face top, curb back, boulevard back, sidewalk back,
    the outer edge. `sidewalk_m` may be per station (N,)."""
    w = np.asarray(sidewalk_m, np.float64)
    o1 = GUTTER_M
    o2 = o1 + CURB_FACE_RUN_M
    o3 = o2 + CURB_TOP_M
    o4 = o3 + BOULEVARD_M
    o5 = o4 + w
    o6 = o5 + BACK_M
    r = CURB_REVEAL_M
    l4 = r + BOULEVARD_M * BOULEVARD_SLOPE
    l5 = l4 + w * SIDEWALK_SLOPE
    return [0.0, o1, o2, o3, o4, o5, o6], [0.0, 0.0, r, r, l4, l5, l5]


def section_width(sidewalk_m: np.ndarray | float) -> np.ndarray:
    """Asphalt edge to the outer edge of the urban section."""
    return np.asarray(section_offsets(sidewalk_m)[0][-1], np.float64)


def sidewalk_width(highway: str) -> float:
    """1.8 m on collectors and arterials, 1.5 m on local streets."""
    from driftmaps.geometry.street_items import highway_rank

    return (
        SIDEWALK_MAJOR_M if highway_rank(highway) >= highway_rank("tertiary") else SIDEWALK_LOCAL_M
    )


def section_lift(d: np.ndarray, sidewalk_m: float = SIDEWALK_LOCAL_M) -> np.ndarray:
    """Lift above the road at distance `d` beyond the asphalt edge (the section's profile)."""
    offs, lifts = section_offsets(sidewalk_m)
    return np.interp(np.asarray(d, np.float64), [float(o) for o in offs],
                     [float(v) for v in lifts])  # fmt: skip


class SidewalkIndex:
    """Municipal sidewalk lines (AC XZ) for asking which side of a road has one."""

    PROBE_M = 4.0  # probe this far beyond the asphalt edge ...
    REACH_M = 3.5  # ... for a sidewalk line within this distance (0.5-7.5 m from the edge)
    FAR_M = 10.5  # and, where `far` allows it, here for one 7.5-13.5 m out (wide boulevards)
    FAR_REACH_M = 3.0

    def __init__(self, lines: Sequence[np.ndarray] = ()) -> None:
        pts = []
        for line in lines:
            ln = shapely.LineString(np.asarray(line, np.float64))
            if ln.length > 0:
                d = np.arange(0.0, ln.length + 1e-9, 2.0)
                pts.append(shapely.get_coordinates(shapely.line_interpolate_point(ln, d)))
        allp = np.concatenate(pts) if pts else np.zeros((0, 2))
        self.tree = cKDTree(allp) if len(allp) else None

    def __bool__(self) -> bool:
        return self.tree is not None

    def sides(
        self,
        xz: np.ndarray,
        left: np.ndarray,
        half_width: np.ndarray,
        far: tuple[np.ndarray | bool, np.ndarray | bool] = (True, True),
    ) -> tuple[np.ndarray, np.ndarray]:
        """(left, right) (N,) bools: a sidewalk line runs beside the road there. `far`
        (left, right; per station or one value) allows the far probe: not on a divided
        road's median side, where it would find the other carriageway's sidewalk."""
        xz = np.asarray(xz, np.float64).reshape(-1, 2)
        if self.tree is None:
            z = np.zeros(len(xz), bool)
            return z, z.copy()
        hw = np.asarray(half_width, np.float64).reshape(-1)[:, None]
        out = []
        for sign, far_ok in zip((1.0, -1.0), far, strict=True):
            probe = xz + sign * left * (hw + self.PROBE_M)
            d, _ = self.tree.query(probe, distance_upper_bound=self.REACH_M)
            got = np.isfinite(d)
            ok = np.broadcast_to(np.asarray(far_ok, bool), got.shape) & ~got
            if ok.any():
                probe2 = xz[ok] + sign * left[ok] * (hw[ok] + self.FAR_M)
                d2, _ = self.tree.query(probe2, distance_upper_bound=self.FAR_REACH_M)
                got[np.flatnonzero(ok)] = np.isfinite(d2)
            out.append(got)
        return out[0], out[1]


class RouteSection:
    """Per route station (N,): urban (curbed), sidewalk left / right, sidewalk width."""

    def __init__(self, urban, walk_left, walk_right, walk_w) -> None:
        self.urban = np.asarray(urban, bool)
        self.walk_left = np.asarray(walk_left, bool) & self.urban
        self.walk_right = np.asarray(walk_right, bool) & self.urban
        self.walk_w = np.asarray(walk_w, np.float64)


def walk_sides(
    tags: Sequence[Mapping[str, str]],
    along: np.ndarray,
    urban: np.ndarray,
    municipal: tuple[np.ndarray, np.ndarray] | None,
    s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sidewalk (left, right) per station of a road driven along `s`: OSM `sidewalk*` tags
    first (`along`: the road runs along its way's direction there), then the municipal lines
    (a way they miss on both sides over its whole length is taken as not surveyed: a new
    road), then the rule (urban collectors and arterials both sides, a one-way carriageway of
    a divided road its outer (right) side only, local streets none). Only on `urban`
    stations; short gaps and stretches smoothed away (`smooth_runs`)."""
    from driftmaps.geometry.street_items import highway_rank

    n = len(tags)
    left = np.zeros(n, bool)
    right = np.zeros(n, bool)
    no_left = np.zeros(n, bool)  # OSM says there is none: never filled
    no_right = np.zeros(n, bool)
    cache: dict[int, tuple] = {}
    major = highway_rank("tertiary")
    surveyed = np.ones(n, bool)
    if municipal is not None:
        hit = np.asarray(municipal[0], bool) | np.asarray(municipal[1], bool)
        keys = np.array([id(t) for t in tags])
        for k in set(keys.tolist()):
            sel = keys == k
            if not hit[sel].any():
                surveyed[sel] = False
    for i, t in enumerate(tags):
        k = id(t)
        if k not in cache:
            one = str(t.get("oneway", "")).strip().lower() in ("yes", "1", "true", "-1")
            cache[k] = (sidewalk_sides(t), highway_rank(t.get("highway", "")) >= major, one)
        (ol, orr), is_major, one = cache[k]
        if not along[i]:
            ol, orr = orr, ol
        ml = bool(municipal[0][i]) if municipal is not None else False
        mr = bool(municipal[1][i]) if municipal is not None else False
        rule = municipal is None or not surveyed[i]
        left[i] = ol if ol is not None else (ml or (rule and is_major and not one))
        right[i] = orr if orr is not None else (mr or (rule and is_major))
        no_left[i], no_right[i] = ol is False, orr is False
    urban = np.asarray(urban, bool)
    if s is not None and n > 2:
        s = np.asarray(s, np.float64)
        left = smooth_runs(left, s, URBAN_MIN_RUN_M, fixed=no_left | ~urban)
        right = smooth_runs(right, s, URBAN_MIN_RUN_M, fixed=no_right | ~urban)
    return left & urban, right & urban


# --------------------------------------------------------------------------------------------
# The urban section around the network's side roads (polygons)
# --------------------------------------------------------------------------------------------

# Bands outward from the asphalt edge: (name, mesh material, index of its outer column in
# `section_offsets`). Fewer than the route's columns, to keep the vertex count down (the
# network has kilometres of them): the curb face and top are one band (its lift still follows
# `section_lift` at every vertex) and the terrain meets the sidewalk's back edge directly.
BANDS: tuple[tuple[str, str, int], ...] = (
    ("gutter", "kerb", 1),
    ("curb", "kerb", 3),
    ("boulevard", "boulevard", 4),
    ("walk", "sidewalk", 5),
)
BAND_SEG_M = 3.0  # longest boundary edge of the bands


def urban_bands(
    base: shapely.Geometry,
    exclude: shapely.Geometry,
    no_walk: shapely.Geometry | None = None,
    sidewalk_m: float = SIDEWALK_LOCAL_M,
) -> dict[str, shapely.Geometry]:
    """The section's bands (`BANDS`) around the curbed asphalt `base`, as polygons, minus
    `exclude` (other surfaces). Buffers of the whole base, so the curb follows every corner's
    radius (curb returns) and every cul-de-sac bulb. Inside `no_walk` the sidewalk band is
    boulevard grass."""
    offs, _ = section_offsets(sidewalk_m)
    # Simplified once (5 cm), so neighbouring bands share the very same boundary vertices.
    rings = [base] + [base.buffer(float(offs[i]), quad_segs=6).simplify(0.05) for _, _, i in BANDS]
    out: dict[str, shapely.Geometry] = {}
    taken = exclude.buffer(0.01) if exclude is not None and not exclude.is_empty else None
    for k, (name, _, _) in enumerate(BANDS):
        band = rings[k + 1].difference(rings[k])
        if taken is not None:
            band = band.difference(taken)
        out[name] = band
    if no_walk is not None and not no_walk.is_empty and not out["walk"].is_empty:
        grass = out["walk"].intersection(no_walk)
        out["walk"] = out["walk"].difference(no_walk)
        out["boulevard"] = shapely.union(out["boulevard"], grass)
    return out


def band_heights(
    base: shapely.Geometry,
    height,
    sidewalk_m: float = SIDEWALK_LOCAL_M,
    mouths: shapely.Geometry | None = None,
):
    """height(xz) + the section's lift at the distance from `base`; inside `mouths` (driveway
    curb cuts) the curb drops to `DEPRESSED_CURB_M` and the boulevard ramps up from it."""
    offs, lifts = section_offsets(sidewalk_m)
    o3, o4 = float(offs[3]), float(offs[4])
    l4 = float(lifts[4])
    edge = base.boundary
    shapely.prepare(edge)
    if mouths is not None and not mouths.is_empty:
        shapely.prepare(mouths)

    def fn(xz: np.ndarray) -> np.ndarray:
        xz = np.asarray(xz, np.float64).reshape(-1, 2)
        pts = shapely.points(xz)
        d = shapely.distance(edge, pts)
        lift = section_lift(d, sidewalk_m)
        if mouths is not None and not mouths.is_empty:
            cut = shapely.contains_xy(mouths, xz[:, 0], xz[:, 1])
            ramp = DEPRESSED_CURB_M + (l4 - DEPRESSED_CURB_M) * np.clip((d - o3) / (o4 - o3), 0, 1)
            low = np.where(d <= o3, np.minimum(lift, DEPRESSED_CURB_M), ramp)
            lift = np.where(cut & (d <= o4 + 1e-6), low, lift)
        return np.asarray(height(xz), np.float64) + lift

    return fn
