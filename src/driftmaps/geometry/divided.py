"""Divided roads: pairing the two one-way carriageways of a dual road, and the median between.

OSM maps a divided road (Dundas Street West, Neyagawa Boulevard in Oakville) as two one-way
ways, one per direction, with the same `name` / `ref`. When the route drives one of them, the
other is a side road of the network (`geometry.network`); this module finds such pairs (the
route against a side road, or two side roads) and the strip between them:

- a pair: two one-way carriageways sharing a name or ref, running in opposite directions
  side by side (cosine of their travel directions below `PAIR_COS`), their asphalt edges
  `MEDIAN_MIN_M`..`MEDIAN_MAX_M` apart, over at least `MIN_PAIR_M`;
- the median: the strip between the two inner asphalt edges, minus every paved area (junction
  boxes, crossing roads), so it stops at each junction and opens where a crossing road or a
  left-turn bay takes its place;
- its form, by its width (`median_kind`): up to `RAISED_MEDIAN_MAX_M` a raised island behind a
  barrier curb (TAC GDG ch. 8 / Ontario urban arterials: raised medians 1.2-6 m, wider ones
  where a left-turn lane is cut into them), with a concrete top up to `CONCRETE_MEDIAN_MAX_M`
  (splash strips) and grass above; wider medians stay grass at grade (the terrain), as on rural
  divided highways (depressed medians 10 m and more).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.spatial import cKDTree

__all__ = [
    "CONCRETE_MEDIAN_MAX_M",
    "MEDIAN_MAX_M",
    "RAISED_MEDIAN_MAX_M",
    "Carriageway",
    "carriageway_key",
    "median_kind",
    "median_polygons",
    "median_width",
]

MEDIAN_MIN_M = 0.3  # asphalt edges closer than this are one carriageway (or a mapping error)
MEDIAN_MAX_M = 30.0  # ... further apart than this are two roads
PAIR_COS = -0.8  # opposite travel directions (within ~37 degrees)
MIN_PAIR_M = 12.0  # a pair runs side by side at least this far
# A median up to this wide is a raised, curbed island; wider ones are grass at grade. TAC GDG
# 8.x: urban raised medians 1.2-6.0 m (a left-turn lane needs >= 4.8 m); ASSUMED 8 m cut-off.
RAISED_MEDIAN_MAX_M = 8.0
CONCRETE_MEDIAN_MAX_M = 2.0  # narrower raised medians are concrete (ASSUMED, as built in Halton)
MIN_MEDIAN_AREA_M2 = 4.0  # smaller leftovers (noses cut by a junction) are dropped


def carriageway_key(tags: Mapping[str, str]) -> frozenset[str]:
    """What identifies a road for pairing: its name and its refs (lowercase)."""
    out = set()
    name = (tags.get("name") or "").strip().lower()
    if name:
        out.add("name:" + name)
    for r in (tags.get("ref") or "").split(";"):
        if r.strip():
            out.add("ref:" + r.strip().lower())
    return frozenset(out)


@dataclass
class Carriageway:
    """A one-way carriageway (or a stretch of the route): centreline (K, 2) in AC XZ, asphalt
    half widths (K,), unit travel directions (K, 2) and its pairing key."""

    xz: np.ndarray
    half_width: np.ndarray
    travel: np.ndarray
    key: frozenset[str]
    name: str = ""


def _travel(xz: np.ndarray, sign: float) -> np.ndarray:
    t = np.gradient(xz, axis=0) * sign
    return t / np.maximum(np.linalg.norm(t, axis=1), 1e-9)[:, None]


def carriageway(
    xz: np.ndarray, half_width: np.ndarray, oneway: int, tags: Mapping[str, str], name: str = ""
) -> Carriageway | None:
    """A Carriageway for a one-way road (`oneway` +1 along `xz`, -1 against); None for a
    two-way road or one without a name or ref."""
    key = carriageway_key(tags)
    if oneway == 0 or not key or len(xz) < 2:
        return None
    xz = np.asarray(xz, np.float64)
    hw = np.broadcast_to(np.asarray(half_width, np.float64), (len(xz),)).copy()
    return Carriageway(xz, hw, _travel(xz, float(np.sign(oneway))), key, name)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist(),
                    strict=True))  # fmt: skip


def _foot(xz: np.ndarray, hw: np.ndarray, p: np.ndarray):
    """(foot points on the polyline, half width there, distance) for points `p`."""
    a = xz[:-1]
    seg = xz[1:] - a
    ll = np.maximum(np.einsum("ij,ij->i", seg, seg), 1e-12)
    mid = a + 0.5 * seg
    _, near = cKDTree(mid).query(p, k=min(4, len(mid)))
    near = np.atleast_2d(near).reshape(len(p), -1)
    best = np.full(len(p), np.inf)
    foot = np.zeros_like(p)
    width = np.zeros(len(p))
    for c in range(near.shape[1]):
        i = near[:, c]
        t = np.clip(np.einsum("ij,ij->i", p - a[i], seg[i]) / ll[i], 0.0, 1.0)
        f = a[i] + t[:, None] * seg[i]
        d = np.linalg.norm(p - f, axis=1)
        better = d < best
        best = np.where(better, d, best)
        foot = np.where(better[:, None], f, foot)
        width = np.where(better, hw[i] + t * (hw[i + 1] - hw[i]), width)
    return foot, width, best


def median_polygons(ways: Sequence[Carriageway], step_m: float = 2.0) -> list[shapely.Polygon]:
    """The strips between every pair of carriageways (asphalt edge to asphalt edge), not yet
    cut by the paved areas. Each pair is found once, from the shorter-indexed carriageway."""
    out: list[shapely.Polygon] = []
    for ia, a in enumerate(ways):
        for b in ways[ia + 1 :]:
            if not (a.key & b.key) or len(b.xz) < 2:
                continue
            k = max(1, int(round(step_m)))
            pa, hwa, ta = a.xz[::k], a.half_width[::k], a.travel[::k]
            if len(pa) < 2:
                continue
            fb, hwb, d = _foot(b.xz, b.half_width, pa)
            _, ib = _nearest_index(b.xz, fb)
            opposite = np.einsum("ij,ij->i", ta, b.travel[ib]) < PAIR_COS
            gap = d - hwa - hwb
            v = fb - pa
            lateral = np.abs(np.einsum("ij,ij->i", v, ta)) < 0.5 * np.maximum(d, 1e-6)
            ok = opposite & lateral & (gap > MEDIAN_MIN_M) & (gap < MEDIAN_MAX_M)
            for i0, i1 in _runs(ok):
                if i1 - i0 < 1 or (i1 - i0) * k * 1.0 < MIN_PAIR_M:
                    continue
                n = v[i0 : i1 + 1] / np.maximum(d[i0 : i1 + 1], 1e-9)[:, None]
                ea = pa[i0 : i1 + 1] + n * hwa[i0 : i1 + 1, None]
                eb = fb[i0 : i1 + 1] - n * hwb[i0 : i1 + 1, None]
                ring = np.concatenate([ea, eb[::-1]])
                poly = shapely.make_valid(shapely.Polygon(ring))
                for p in _polys(poly):
                    if p.area > MIN_MEDIAN_AREA_M2:
                        out.append(p)
    return out


def _nearest_index(xz: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cKDTree(xz).query(p)


def _polys(g: shapely.Geometry) -> list[shapely.Polygon]:
    if g is None or g.is_empty:
        return []
    if isinstance(g, shapely.Polygon):
        return [g]
    return [p for p in getattr(g, "geoms", []) for p in _polys(p)]


def median_width(poly: shapely.Geometry) -> float:
    """The mean width of a long strip: 2 x area / perimeter."""
    per = float(poly.length)
    return 2.0 * float(poly.area) / per if per > 0 else 0.0


def median_kind(poly: shapely.Geometry) -> str:
    """ "raised_concrete", "raised_grass" or "grass" (at grade, the terrain) by the width."""
    w = median_width(poly)
    if w <= CONCRETE_MEDIAN_MAX_M:
        return "raised_concrete"
    if w <= RAISED_MEDIAN_MAX_M:
        return "raised_grass"
    return "grass"
