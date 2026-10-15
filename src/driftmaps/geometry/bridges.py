"""Bridge structures for OSM bridge spans: deck slab, railings, abutments and piers.

User feedback: bridges looked like "floating road sheets". The deck height is right (the
lidar DSM sees the Sydenham Rd deck ~11 m above the creek, scenery.md §1.2); what was missing
is the structure. The road meshes (`1ROAD`, `1GRAVEL`) already form the deck's driving
surface and stay level/straight as the profile made them; this module adds, per bridge span
[s0, s1] (model stations, from `bridge_spans(model)`):

- **Deck slab**: a U section under the road from the left fascia, along the underside, to
  the right fascia, `deck_thickness_m` deep, out to the railing's outer face
  (`outer + gap + parapet_width`). It runs `abutment_back_m` past each end, into the
  abutments, so no edge of it is seen floating.
- **Railing** on both sides over [s0, s1]: a concrete parapet (default) or a steel rail on a
  concrete curb (`geometry.barriers`), each with a hidden, double-sided `1WALL` collision
  strip on its road face so cars cannot drive off (scenery.md §5: bridges always get one).
  Except at-grade culverts (`BridgePlan.railing` False): a "solid" bridge (below) or one whose
  deck is never `min_railing_height_m` (2 m) above the ground under it is just road, with no
  parapets (the corridor's hidden walls still hold).
- **Abutments**: a concrete block at each end, from `abutment_back_m` under the approach to
  the abutment face, top just under the road, bottom `embed_m` below the lowest ground under
  it, so it blends into the terrain whatever its slope. The face is placed adaptively: the
  first station (at least `seat_m` in) where the ground across the whole deck lies
  `abutment_clearance_m` below the deck underside, so terrain that the cut/fill left high near
  the bridge ends is buried in the abutment instead of poking through the deck. If no station
  clears, the bridge is "solid" (an embankment with a culvert, e.g. OSM `bridge=yes` on a
  9 m culvert at Forks).
- **Piers** between the abutment faces every `pier_spacing_m` (spans kept within 15-25 m)
  wherever the deck underside is more than `pier_min_gap_m` above the ground: a cap beam with
  columns (default) or a solid wall pier, reaching `embed_m` below the ground under each
  column.

All visual meshes are concrete (one `bridge##_concrete` node per bridge; steel railings add
`bridge##_steel`), with `lod_out`; collision is `1WALL_BRIDGE##_<L|R>##`, hidden.

`terrain_through_deck` checks that no terrain rises into the deck (above the underside over
the open span, above the road surface over the abutment seats), and `deck_ceiling` gives the
maximum terrain height under the decks so a builder can clamp terrain heights with it.

AC frame: X east, Y up, Z south; left = up x forward. `ground_fn(xz (M, 2)) -> (M,)` is the
visible terrain height (NaN where unknown).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from driftmaps.ac.scene import HIDDEN_WALL_FLAGS
from driftmaps.geo.barriers import BarrierSpan
from driftmaps.geometry.barriers import (
    UP,
    BarrierOptions,
    GroundFn,
    ScenePiece,
    span_meshes,
    sweep,
)
from driftmaps.geometry.primitives import MeshData, box, merge
from driftmaps.geometry.road import RoadModel, road_columns

__all__ = [
    "BridgeOptions",
    "BridgePlan",
    "DeckCheck",
    "bridge_pieces",
    "bridge_railing_spans",
    "bridge_spans",
    "deck_ceiling",
    "deck_half_width",
    "plan_bridge",
    "terrain_through_deck",
]


@dataclass(frozen=True)
class BridgeOptions:
    deck_thickness_m: float = 1.0  # slab + girders (ASSUMED typical for 15-25 m spans)
    railing: Literal["parapet", "steel"] = "parapet"
    seat_m: float = 1.0  # minimum abutment seat under each deck end
    abutment_back_m: float = 4.0  # abutment and deck reach this far under the approach
    abutment_clearance_m: float = 0.5
    max_abutment_fraction: float = 0.4  # of the span length, per end
    wing_m: float = 0.4  # abutment wider than the deck by this on each side
    pier_spacing_m: float = 20.0
    min_span_m: float = 15.0
    max_span_m: float = 25.0
    pier_min_gap_m: float = 3.0
    pier_style: Literal["columns", "wall"] = "columns"
    pier_thickness_m: float = 1.0  # wall pier, along the road
    column_size_m: float = 0.9
    cap_height_m: float = 0.8
    embed_m: float = 1.0
    lod_out_m: float = 2500.0
    # Decks never this high above the ground under them (culverts) get no railing.
    min_railing_height_m: float = 2.0
    barrier: BarrierOptions = field(default_factory=BarrierOptions)

    @property
    def railing_kind(self) -> str:
        return "parapet" if self.railing == "parapet" else "bridge_rail"


@dataclass
class BridgePlan:
    """Where one bridge's parts go, in model stations. The open span (no abutment under the
    deck) is [face0, face1]; `solid` bridges have none. `piers` are pier stations. `height`
    is the deck's greatest height above the ground under it (inf: ground unknown); `railing`
    False for an at-grade culvert (no parapets)."""

    index: int
    s0: float
    s1: float
    face0: float
    face1: float
    piers: tuple[float, ...]
    thickness: float
    height: float = math.inf
    railing: bool = True

    @property
    def length(self) -> float:
        return self.s1 - self.s0

    @property
    def open_length(self) -> float:
        return max(0.0, self.face1 - self.face0)

    @property
    def solid(self) -> bool:
        return self.open_length <= 0.0


@dataclass
class DeckCheck:
    """Terrain points (M, 3) that rise into a deck, and by how much (M,) metres."""

    points: np.ndarray
    intrusion: np.ndarray

    @property
    def ok(self) -> bool:
        return len(self.intrusion) == 0

    @property
    def max_intrusion(self) -> float:
        return float(self.intrusion.max()) if len(self.intrusion) else 0.0


def bridge_spans(model: RoadModel, min_length_m: float = 1.0) -> list[tuple[float, float]]:
    """Station ranges (s0, s1) of the runs of bridge stations (`RoadModel.bridge`)."""
    flag = np.asarray(model.bridge, bool)
    if not flag.any():
        return []
    d = np.diff(np.concatenate([[0], flag.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    out = []
    for a, b in zip(starts, ends, strict=True):
        s0, s1 = float(model.s[a]), float(model.s[b])
        if s1 - s0 >= min_length_m:
            out.append((s0, s1))
    return out


def deck_half_width(model: RoadModel, options: BridgeOptions | None = None) -> np.ndarray:
    """Deck half width (N,) at every station: shoulder edge + gap + railing width."""
    opt = options or BridgeOptions()
    return model.outer + opt.barrier.gap_m + opt.barrier.parapet_width_m


# --------------------------------------------------------------------------------------------
# Sampling helpers
# --------------------------------------------------------------------------------------------


def _interp(model: RoadModel, values: np.ndarray, s: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return np.interp(s, model.s, values)
    return np.stack([np.interp(s, model.s, values[:, k]) for k in range(values.shape[1])], -1)


def _frame(model: RoadModel, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre, horizontal forward, left) at stations s (linear)."""
    c = _interp(model, model.centre, s)
    f = _interp(model, model.forward, s)
    f[..., 1] = 0.0
    f /= np.linalg.norm(f, axis=-1, keepdims=True)
    left = np.stack([f[..., 2], np.zeros(f.shape[:-1]), -f[..., 0]], axis=-1)
    return c, f, left


def _ground(ground_fn: GroundFn, pts: np.ndarray, fallback: np.ndarray | float) -> np.ndarray:
    """Ground under XYZ points (..., 3); NaN -> `fallback`."""
    flat = pts.reshape(-1, 3)
    g = np.asarray(ground_fn(flat[:, [0, 2]]), np.float64).reshape(pts.shape[:-1])
    return np.where(np.isfinite(g), g, fallback)


def _across(model: RoadModel, s: np.ndarray, half: np.ndarray, n: int) -> np.ndarray:
    """Points (len(s), n, 3) across the deck at stations s, from -half to +half."""
    c, _, left = _frame(model, s)
    lat = np.linspace(-1.0, 1.0, n)[None, :] * half[:, None]
    return c[:, None, :] + lat[..., None] * left[:, None, :]


def _rows(a: float, b: float, step: float) -> np.ndarray:
    n = max(1, math.ceil((b - a) / step - 1e-9))
    return np.linspace(a, b, n + 1)


# --------------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------------


def plan_bridge(
    model: RoadModel,
    s0: float,
    s1: float,
    ground_fn: GroundFn,
    options: BridgeOptions | None = None,
    index: int = 0,
) -> BridgePlan:
    """Abutment faces and pier stations for the bridge span [s0, s1] (see module docstring)."""
    opt = options or BridgeOptions()
    if not s1 > s0:
        raise ValueError(f"need s0 < s1, got {s0}, {s1}")
    t = opt.deck_thickness_m
    half_all = deck_half_width(model, opt)
    s = _rows(s0, s1, 0.5)
    deck = _interp(model, model.centre[:, 1], s)
    under = deck - t
    pts = _across(model, s, _interp(model, half_all, s), 9)
    g_all = _ground(ground_fn, pts, np.nan)
    g = np.where(np.isnan(g_all), under[:, None] - 1e3, g_all).max(axis=1)
    low = np.where(np.isnan(g_all), np.inf, g_all).min(axis=1)  # deepest ground across
    height = float(np.max(deck - low)) if np.isfinite(low).any() else math.inf
    clear = g <= under - opt.abutment_clearance_m
    length = s1 - s0
    seat = min(opt.seat_m, length / 2)
    cap = opt.max_abutment_fraction * length
    inner = (s >= s0 + seat - 1e-9) & (s <= s1 - seat + 1e-9)
    ok = np.flatnonzero(clear & inner)
    mid = 0.5 * (s0 + s1)
    if len(ok) == 0:  # solid: an embankment over a culvert, just road
        return BridgePlan(index, s0, s1, mid, mid, (), t, height, railing=False)
    face0 = min(float(s[ok[0]]), s0 + max(cap, seat))
    face1 = max(float(s[ok[-1]]), s1 - max(cap, seat))
    if face1 - face0 < 2.0:
        return BridgePlan(index, s0, s1, mid, mid, (), t, height, railing=False)
    railing = height >= opt.min_railing_height_m

    # Piers: spans as close to pier_spacing_m as possible, within [min_span_m, max_span_m].
    open_len = face1 - face0
    n = max(1, round(open_len / opt.pier_spacing_m))
    while open_len / n > opt.max_span_m:
        n += 1
    while n > 1 and open_len / n < opt.min_span_m and open_len / (n - 1) <= opt.max_span_m:
        n -= 1
    piers = []
    for k in range(1, n):
        sp = face0 + k * open_len / n
        sp_arr = np.array([sp])
        p = _across(model, sp_arr, _interp(model, half_all, sp_arr), 9)[0]
        u = float(np.interp(sp, model.s, model.centre[:, 1])) - t
        gp = _ground(ground_fn, p, np.nan)
        if np.isfinite(gp).any() and u - np.nanmax(gp) > opt.pier_min_gap_m:
            piers.append(float(sp))
    return BridgePlan(index, s0, s1, face0, face1, tuple(piers), t, height, railing)


def bridge_railing_spans(
    plans: Sequence[BridgePlan], options: BridgeOptions | None = None
) -> list[BarrierSpan]:
    """The railing spans (both sides, [s0, s1]) of the planned bridges that have one (not
    the at-grade culverts)."""
    opt = options or BridgeOptions()
    return [
        BarrierSpan(side, p.s0, p.s1, opt.railing_kind, "bridge")
        for p in plans
        if p.railing
        for side in ("left", "right")
    ]


# --------------------------------------------------------------------------------------------
# Meshes
# --------------------------------------------------------------------------------------------


def _metric_box(centre: np.ndarray, size: tuple[float, float, float], heading: float) -> MeshData:
    """`primitives.box` with UVs in metres (4 m tiles) instead of 0..1 per face."""
    m = box(centre, size, heading)
    sx, sy, sz = size
    dims = [(sz, sy), (sz, sy), (sx, sy), (sx, sy), (sx, sz), (sx, sz)]
    scale = np.repeat(np.array(dims) / 4.0, 4, axis=0)
    return dataclasses.replace(m, uvs=m.uvs * scale)


def _heading(f: np.ndarray) -> float:
    return math.degrees(math.atan2(float(f[0]), float(-f[2])))


def _deck(model: RoadModel, plan: BridgePlan, opt: BridgeOptions) -> MeshData | None:
    lo, hi = float(model.s[0]), float(model.s[-1])
    a = max(lo, plan.s0 - opt.abutment_back_m)
    b = min(hi, plan.s1 + opt.abutment_back_m)
    s = _rows(a, b, 1.0)
    half = deck_half_width(model, opt)
    t = plan.thickness
    c_left = _interp(model, road_columns(model, half), s)
    c_right = _interp(model, road_columns(model, -half), s)
    down = t * UP
    # Counter-clockwise in (left, up): right fascia down, underside, left fascia up.
    parts = [sweep([c_right, c_right - down, c_left - down, c_left], s, u_tile_m=4.0, v_tile_m=4.0)]
    gap = opt.barrier.gap_m
    if gap > 0.0:  # deck top between the shoulder edge and the railing
        edge, inner = model.outer, model.outer + gap
        rows = (s >= plan.s0) & (s <= plan.s1)
        if rows.sum() >= 2:
            sr = s[rows]
            for lat_a, lat_b in ((inner, edge), (-edge, -inner)):
                ca = _interp(model, road_columns(model, lat_a), sr)
                cb = _interp(model, road_columns(model, lat_b), sr)
                parts.append(sweep([ca, cb], sr, u_tile_m=4.0, v_tile_m=4.0))
    parts = [p for p in parts if p is not None]
    return merge(parts) if parts else None


def _abutment(
    model: RoadModel, a: float, b: float, ground_fn: GroundFn, opt: BridgeOptions
) -> MeshData | None:
    """A block from station a to b, top just under the road, bottom below the ground."""
    if b - a < 0.1:
        return None
    s = _rows(a, b, 1.0)
    half = deck_half_width(model, opt)
    width = float(_interp(model, half, s).max()) + opt.wing_m
    top = float(_interp(model, model.centre[:, 1], s).min()) - 0.05
    pts = _across(model, s, np.full(len(s), width), 7)
    g = _ground(ground_fn, pts, top)
    bottom = min(float(g.min()), top - 0.5) - opt.embed_m
    p0, _, _ = _frame(model, np.array([a]))
    p1, _, _ = _frame(model, np.array([b]))
    chord = p1[0] - p0[0]
    chord[1] = 0.0
    length = float(np.linalg.norm(chord))
    if length < 0.1:
        return None
    mid = 0.5 * (p0[0] + p1[0])
    centre = np.array([mid[0], 0.5 * (top + bottom), mid[2]])
    return _metric_box(centre, (2.0 * width, top - bottom, length), _heading(chord / length))


def _pier(
    model: RoadModel, sp: float, plan: BridgePlan, ground_fn: GroundFn, opt: BridgeOptions
) -> list[MeshData]:
    s = np.array([sp])
    c, f, left = _frame(model, s)
    c, f, left = c[0], f[0], left[0]
    half = float(_interp(model, deck_half_width(model, opt), s)[0])
    under = c[1] - plan.thickness
    heading = _heading(f)
    if opt.pier_style == "wall":
        across = max(1.0, 2.0 * half - 1.0)
        pts = _across(model, s, np.array([across / 2]), 7)[0]
        pts = np.concatenate(
            [pts + 0.5 * opt.pier_thickness_m * f, pts - 0.5 * opt.pier_thickness_m * f]
        )
        bottom = float(_ground(ground_fn, pts, under - 10.0).min()) - opt.embed_m
        top = under + 0.02
        centre = np.array([c[0], 0.5 * (top + bottom), c[2]])
        return [_metric_box(centre, (across, top - bottom, opt.pier_thickness_m), heading)]
    cap_h = opt.cap_height_m
    cap_c = np.array([c[0], under - cap_h / 2 + 0.01, c[2]])
    out = [_metric_box(cap_c, (max(1.0, 2.0 * half - 0.6), cap_h + 0.02, 1.2), heading)]
    count = max(2, math.ceil(2.0 * half / 5.0))
    reach = max(0.0, half - 1.2)
    col = opt.column_size_m
    top = under - cap_h + 0.05
    for lat in np.linspace(-reach, reach, count):
        base = c + lat * left
        corners = np.array(
            [base + (i * left + j * f) * col / 2 for i in (-1, 0, 1) for j in (-1, 0, 1)]
        )
        bottom = float(_ground(ground_fn, corners, under - 10.0).min()) - opt.embed_m
        if top - bottom < 0.2:
            continue
        centre = np.array([base[0], 0.5 * (top + bottom), base[2]])
        out.append(_metric_box(centre, (col, top - bottom, col), heading))
    return out


def bridge_pieces(
    model: RoadModel,
    spans: Sequence[tuple[float, float]],
    ground_fn: GroundFn,
    options: BridgeOptions | None = None,
) -> tuple[list[ScenePiece], list[BridgePlan]]:
    """KN5 pieces and plans for each bridge span (s0, s1) (e.g. `bridge_spans(model)`):
    `bridge##_concrete` (deck, abutments, piers, concrete railing parts), `bridge##_steel`
    (steel railing, if chosen) and hidden collision `1WALL_BRIDGE##_<L|R>##` (no railing and
    no collision on an at-grade culvert, `BridgePlan.railing`)."""
    opt = options or BridgeOptions()
    pieces: list[ScenePiece] = []
    plans: list[BridgePlan] = []
    visual_flags = {"lod_out": opt.lod_out_m}
    for b, (s0, s1) in enumerate(spans):
        plan = plan_bridge(model, s0, s1, ground_fn, opt, index=b)
        plans.append(plan)
        concrete: list[MeshData] = []
        steel: list[MeshData] = []
        deck = _deck(model, plan, opt)
        if deck is not None:
            concrete.append(deck)
        lo, hi = float(model.s[0]), float(model.s[-1])
        a0, a1 = max(lo, s0 - opt.abutment_back_m), min(hi, s1 + opt.abutment_back_m)
        if plan.solid:
            blocks = [(a0, a1)]
        else:
            blocks = [(a0, plan.face0), (plan.face1, a1)]
        for a, bb in blocks:
            m = _abutment(model, a, bb, ground_fn, opt)
            if m is not None:
                concrete.append(m)
        for sp in plan.piers:
            concrete += _pier(model, sp, plan, ground_fn, opt)
        for span in bridge_railing_spans([plan], opt):
            chunks = span_meshes(
                model, span, ground_fn, opt.barrier, free_ends=(False, False), chunk_m=150.0
            )
            for k, (visual, coll) in enumerate(chunks):
                concrete += [m for mat, m in visual.items() if mat == "concrete"]
                steel += [m for mat, m in visual.items() if mat == "steel"]
                side = "L" if span.side == "left" else "R"
                pieces.append(
                    ScenePiece(
                        f"1WALL_BRIDGE{b:02d}_{side}{k:02d}",
                        "wall_hidden",
                        coll,
                        dict(HIDDEN_WALL_FLAGS),
                    )
                )
        if concrete:
            pieces.append(
                ScenePiece(
                    f"bridge{b:02d}_concrete", "concrete", merge(concrete), dict(visual_flags)
                )
            )
        if steel:
            pieces.append(
                ScenePiece(f"bridge{b:02d}_steel", "steel", merge(steel), dict(visual_flags))
            )
    return pieces, plans


# --------------------------------------------------------------------------------------------
# Terrain checks
# --------------------------------------------------------------------------------------------


def _deck_limit(model: RoadModel, plan: BridgePlan, s: np.ndarray, road_lift: float) -> np.ndarray:
    """Highest allowed terrain at stations s: the underside over the open span, the road
    surface minus `road_lift` over the abutment seats."""
    deck = np.interp(s, model.s, model.centre[:, 1])
    open_span = (s >= plan.face0) & (s <= plan.face1) & (not plan.solid)
    return np.where(open_span, deck - plan.thickness, deck - road_lift)


def terrain_through_deck(
    model: RoadModel,
    plan: BridgePlan,
    ground_fn: GroundFn,
    options: BridgeOptions | None = None,
    *,
    step_m: float = 0.5,
    lateral_step_m: float = 0.5,
    margin_m: float = 0.0,
) -> DeckCheck:
    """Terrain samples under the deck (every `step_m` along [s0, s1], every
    `lateral_step_m` across the deck width) that rise above the deck underside (open span) or
    within 5 cm of the road surface (abutment seats), less `margin_m`."""
    opt = options or BridgeOptions()
    s = _rows(plan.s0, plan.s1, step_m)
    half = _interp(model, deck_half_width(model, opt), s)
    n = max(3, math.ceil(2.0 * float(half.max()) / lateral_step_m) + 1)
    pts = _across(model, s, half, n)
    g = _ground(ground_fn, pts, -np.inf)
    limit = _deck_limit(model, plan, s, 0.05)[:, None] - margin_m
    bad = g > limit
    out = pts[bad].copy()
    out[:, 1] = g[bad]
    return DeckCheck(out, (g - limit)[bad])


def deck_ceiling(
    model: RoadModel,
    plans: Sequence[BridgePlan],
    xz: npt.ArrayLike,
    options: BridgeOptions | None = None,
    *,
    clearance_m: float = 0.3,
    margin_m: float = 0.5,
) -> np.ndarray:
    """Maximum terrain height (M,) at XZ points so it stays `clearance_m` clear of every deck
    (underside over the open span, road surface over the seats), within `margin_m` beyond
    the deck edges; +inf away from bridges. Clamp terrain heights with np.minimum."""
    opt = options or BridgeOptions()
    p = np.asarray(xz, np.float64).reshape(-1, 2)
    out = np.full(len(p), np.inf)
    half_all = deck_half_width(model, opt)
    for plan in plans:
        s = _rows(plan.s0, plan.s1, 0.5)
        c, f, left = _frame(model, s)
        tree = cKDTree(c[:, [0, 2]])
        reach = float(_interp(model, half_all, s).max()) + margin_m + 0.5
        d, i = tree.query(p, distance_upper_bound=reach + 1.0)
        near = np.isfinite(d)
        if not near.any():
            continue
        j = i[near]
        rel = np.stack([p[near, 0] - c[j, 0], np.zeros(near.sum()), p[near, 1] - c[j, 2]], 1)
        along = np.einsum("ij,ij->i", rel, f[j])
        lat = np.abs(np.einsum("ij,ij->i", rel, left[j]))
        sp = s[j] + along
        half = np.interp(sp, model.s, half_all)
        inside = (sp >= plan.s0) & (sp <= plan.s1) & (lat <= half + margin_m)
        lim = _deck_limit(model, plan, np.clip(sp, plan.s0, plan.s1), 0.05) - clearance_m
        idx = np.flatnonzero(near)[inside]
        out[idx] = np.minimum(out[idx], lim[inside])
    return out
