"""Lanes: the lane layout of a road from its OSM lane tags, and the lines Ontario paints on it.

Layout (`way_lanes`, `RoadLanes`): `lanes`, `lanes:forward` / `lanes:backward` (and
`lanes:both_ways`, a centre two-way left-turn lane), `turn:lanes` / `turn:lanes:forward` /
`turn:lanes:backward`, `oneway`. The carriageway width comes from `width` or lanes x the
Ontario lane width (`geo.attributes.LANE_WIDTH_M["na"]`: 3.6 m arterials, 3.3-3.4 m
collectors, 3.0 m locals; TAC GDG Table 4.1.x, docs/research/intersections.md); the lanes share
it equally. With right-hand traffic the lanes in the driving direction are on the right; the
line between the two directions sits `centre` metres left of the centreline.

Markings (OTM Book 11, docs/research/intersections.md):
- white broken lane lines (3 m line, 6 m gap; 9 m on rural roads) between lanes of the
  same direction; a solid
  white line where a turn-only lane runs beside a through lane (its storage length);
- yellow centre line between the two directions: double solid on multi-lane roads and urban
  arterials, single broken on urban two-lane collectors, none on urban local streets; rural
  two-lane roads keep the double solid line they had (`centre_ok` from the caller);
- edge lines on uncurbed roads: white on the right, yellow on the left of a one-way
  carriageway (a divided road's median side); none where a curb is the edge (urban).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from driftmaps.geo.attributes import is_oneway
from driftmaps.geometry.street_items import highway_rank

__all__ = [
    "BROKEN_GAP_M",
    "BROKEN_LINE_M",
    "LANE_LINE_WIDTH_M",
    "PaintLine",
    "RoadLanes",
    "WayLanes",
    "paint_lines",
    "road_lanes",
    "way_lanes",
]

# OTM Book 11 (2000) s.3.5: broken lane lines are 3 m segments with 6 m gaps on urban roads
# and 9 m gaps on rural roads, about 10 cm wide; a broken directional dividing (centre) line
# is 3 m / 6 m everywhere.
BROKEN_LINE_M = 3.0
BROKEN_GAP_M = 6.0
BROKEN_GAP_RURAL_M = 9.0
LANE_LINE_WIDTH_M = 0.10

_TURN_ONLY = frozenset({"left", "right", "sharp_left", "sharp_right", "slight_left",
                        "slight_right", "reverse", "merge_to_left", "merge_to_right"})  # fmt: skip


def _int(v: str | None) -> int | None:
    raw = (v or "").split(";")[0].strip()
    return int(raw) if raw.isdigit() else None


def _turns(v: str | None, n: int) -> tuple[str, ...]:
    """`turn:lanes` split per lane (left to right), "" for none; () if it does not match."""
    if not v:
        return ()
    parts = tuple(p.strip() for p in v.split("|"))
    if len(parts) != n:
        return ()
    return tuple("" if p in ("none", "") else p for p in parts)


@dataclass(frozen=True)
class WayLanes:
    """Lanes of one way as driven (`forward`: in the driving direction), turn lanes left to
    right as those drivers see them, and whether OSM said so (`tagged`)."""

    forward: int
    backward: int
    turn_forward: tuple[str, ...] = ()
    turn_backward: tuple[str, ...] = ()
    centre_turn: int = 0  # a two-way left-turn lane in the middle (lanes:both_ways)
    tagged: bool = False

    @property
    def total(self) -> int:
        return self.forward + self.backward + self.centre_turn


def way_lanes(tags: Mapping[str, str], along: bool = True) -> WayLanes:
    """The lanes of a way driven along its node order (`along`) or against it."""
    ow = is_oneway(tags)
    total = _int(tags.get("lanes"))
    hw = tags.get("highway", "")
    if ow != 0:
        n = total or _int(tags.get("lanes:forward")) or (2 if hw in ("motorway", "trunk") else 1)
        turns = _turns(tags.get("turn:lanes") or tags.get("turn:lanes:forward"), n)
        return WayLanes(n, 0, turns, (), 0, total is not None)
    f, b = _int(tags.get("lanes:forward")), _int(tags.get("lanes:backward"))
    both = _int(tags.get("lanes:both_ways")) or 0
    tagged = total is not None or f is not None or b is not None
    if f is None or b is None:
        rest = max((total or 2) - both, 2 if total is None else 1)
        if f is None and b is None:
            f = (rest + 1) // 2
            b = rest - f
        elif f is None:
            f = max(rest - b, 1)
        else:
            b = max(rest - f, 0)
    tf = _turns(tags.get("turn:lanes:forward"), f)
    tb = _turns(tags.get("turn:lanes:backward"), b)
    if not along:
        f, b, tf, tb = b, f, tb, tf
    return WayLanes(f, b, tf, tb, both, tagged)


@dataclass
class RoadLanes:
    """Per station (N,): lanes in the driving direction and against it, the lane width and
    the lateral offset (+ left) of the line between the two directions; turn lanes per
    station (left to right as each direction's drivers see them)."""

    n_fwd: np.ndarray
    n_bwd: np.ndarray
    lane_w: np.ndarray
    centre: np.ndarray
    half_width: np.ndarray
    turn_fwd: list[tuple[str, ...]]
    turn_bwd: list[tuple[str, ...]]
    tagged: np.ndarray
    right_hand: bool = True

    def __len__(self) -> int:
        return len(self.n_fwd)

    def span(self, i: int, forward: bool) -> tuple[float, float] | None:
        """(l0, l1), l0 < l1, the lateral span of the lanes in one direction at station `i`,
        or None when no lane runs that way (a one-way carriageway)."""
        hw, c = float(self.half_width[i]), float(self.centre[i])
        n = int(self.n_fwd[i] if forward else self.n_bwd[i])
        if n <= 0:
            return None
        right_side = forward == self.right_hand  # these lanes are right of the centre line
        return (-hw, c) if right_side else (c, hw)

    def lane_centres(self, i: int, forward: bool) -> list[float]:
        """Lateral offsets of the lanes in one direction, left to right as their drivers see
        them."""
        sp = self.span(i, forward)
        if sp is None:
            return []
        n = int(self.n_fwd[i] if forward else self.n_bwd[i])
        w = (sp[1] - sp[0]) / n
        centres = [sp[1] - (k + 0.5) * w for k in range(n)]  # left to right, driving +s
        return centres if forward else centres[::-1]


def road_lanes(
    half_width: np.ndarray,
    ways: Sequence[WayLanes],
    *,
    right_hand: bool = True,
) -> RoadLanes:
    """The lane layout per station from each station's `WayLanes` (as driven)."""
    hw = np.asarray(half_width, np.float64)
    nf = np.array([w.forward for w in ways], np.int64)
    nb = np.array([w.backward for w in ways], np.int64)
    ct = np.array([w.centre_turn for w in ways], np.int64)
    total = np.maximum(nf + nb + ct, 1)
    lw = 2.0 * hw / total
    # The line between the directions: forward lanes from the right edge (right-hand traffic);
    # a centre turn lane sits on it (the yellow lines are drawn either side, ASSUMED one line).
    centre = -hw + (nf + 0.5 * ct) * lw if right_hand else hw - (nf + 0.5 * ct) * lw
    return RoadLanes(
        nf, nb, lw, centre, hw,
        [w.turn_forward for w in ways], [w.turn_backward for w in ways],
        np.array([w.tagged for w in ways], bool), right_hand,
    )  # fmt: skip


@dataclass
class PaintLine:
    """One painted line: `colour` "white" / "yellow", lateral `offset` (N,) (+ left) of its
    centre, `width`, `keep` (N - 1,) the segments painted (dashes, where it exists), and its
    `role` ("centre", "lane", "edge")."""

    colour: str
    offset: np.ndarray
    width: float
    keep: np.ndarray
    role: str


def _broken(s: np.ndarray, gap: float = BROKEN_GAP_M) -> np.ndarray:
    mid = 0.5 * (s[:-1] + s[1:])
    return (mid % (BROKEN_LINE_M + gap)) < BROKEN_LINE_M


def _seg(mask: np.ndarray) -> np.ndarray:
    return mask[:-1] & mask[1:]


def paint_lines(
    s: np.ndarray,
    lanes: RoadLanes,
    *,
    urban: np.ndarray,
    highway: Sequence[str],
    centre_ok: np.ndarray,
    edge_ok: np.ndarray,
    oneway: np.ndarray,
    centre_width: float = 0.12,
    centre_gap: float = 0.12,
    edge_width: float = 0.15,
    edge_inset: float = 0.20,
) -> list[PaintLine]:
    """The lines on one road (see the module docstring). `urban` (N,): curbed urban section
    (no edge lines, the urban centre-line rules); `highway` (N,) the OSM class; `centre_ok`
    (N,): where the caller paints a centre line on rural roads; `edge_ok` (N,): where edge
    lines may be painted; `oneway` (N,) +1 / -1 / 0 as driven."""
    n = len(s)
    s = np.asarray(s, np.float64)
    urban = np.asarray(urban, bool)
    oneway = np.asarray(oneway)
    hw = lanes.half_width
    broken = _broken(s)
    rural_broken = _broken(s, BROKEN_GAP_RURAL_M)
    out: list[PaintLine] = []
    rank = np.array([highway_rank(h) for h in highway])
    two_way = (lanes.n_fwd > 0) & (lanes.n_bwd > 0) & (oneway == 0)
    total = lanes.n_fwd + lanes.n_bwd
    # Centre line style per station: 0 none, 1 double solid, 2 single broken.
    style = np.zeros(n, np.int64)
    rural_c = two_way & ~urban & centre_ok
    style[rural_c] = 1
    urban_c = two_way & urban
    multi = urban_c & ((total >= 3) | (rank >= highway_rank("secondary")))
    collector = urban_c & ~multi & ((rank >= highway_rank("tertiary")) | lanes.tagged & centre_ok)
    collector &= rank > highway_rank("residential")
    style[multi] = 1
    style[collector] = 2
    c = lanes.centre
    g = centre_gap / 2 + centre_width / 2
    double = _seg(style == 1)
    if double.any():
        out.append(PaintLine("yellow", c + g, centre_width, double, "centre"))
        out.append(PaintLine("yellow", c - g, centre_width, double, "centre"))
    single = _seg(style == 2) & broken
    if single.any():
        out.append(PaintLine("yellow", c.copy(), centre_width, single, "centre"))

    # Lane lines between lanes of one direction.
    for forward in (True, False):
        count = lanes.n_fwd if forward else lanes.n_bwd
        kmax = int(count.max()) if n else 0
        turns = lanes.turn_fwd if forward else lanes.turn_bwd
        right_side = forward == lanes.right_hand
        for k in range(1, kmax):
            exists = count > k
            off = np.zeros(n)
            # Boundary k counted from the centre line (or a one-way's left edge) outwards.
            if right_side:
                start = np.where(two_way, c, hw)
                off = start - k * lanes.lane_w
            else:
                start = np.where(two_way, c, -hw)
                off = start + k * lanes.lane_w
            # Solid where exactly one of the two lanes is turn-only (OTM Book 11: lane-use
            # lines are solid over a turn lane's storage length).
            solid = np.zeros(n, bool)
            for i in np.flatnonzero(exists):
                t = turns[i]
                if len(t) == count[i]:
                    # lanes left to right as their drivers see them: boundary k lies between
                    # lane k-1 and lane k counted from the centre (the drivers' left).
                    a, b = t[k - 1], t[k]
                    ta = bool(set(a.split(";")) <= _TURN_ONLY) if a else False
                    tb = bool(set(b.split(";")) <= _TURN_ONLY) if b else False
                    solid[i] = ta != tb
            seg = _seg(exists)
            dashes = np.where(urban[:-1], broken, rural_broken)
            keep = seg & (dashes | _seg(solid))
            if keep.any():
                out.append(PaintLine("white", off, LANE_LINE_WIDTH_M, keep, "lane"))

    # Edge lines on uncurbed roads.
    edge = _seg(edge_ok & ~urban)
    if edge.any():
        inset = hw - edge_inset
        one = _seg(oneway != 0)
        left_yellow = one & (oneway[:-1] > 0) if lanes.right_hand else one & (oneway[:-1] < 0)
        right_yellow = one & ~left_yellow
        out.append(PaintLine("white", inset.copy(), edge_width, edge & ~left_yellow, "edge"))
        out.append(PaintLine("yellow", inset.copy(), edge_width, edge & left_yellow, "edge"))
        out.append(PaintLine("white", -inset, edge_width, edge & ~right_yellow, "edge"))
        out.append(PaintLine("yellow", -inset, edge_width, edge & right_yellow, "edge"))
    return [p for p in out if p.keep.any()]
