"""The road network around the route: side roads, junction mouths, roundabouts, cul-de-sacs.

The route itself stays the lattice ribbon of `geometry.road`. Everything else on the network is
built from **polygons** in the AC XZ plane, so overlaps, junction corners and islands are plain
boolean geometry, and meshed on one **height field**, so every piece meets its neighbours:

- **Side roads** (`SideRoad`): every public OSM road within `NetworkOptions.radius_m` of the
  route (not driveways, parking aisles or farm tracks), cut at that radius. Chains of OSM
  segments between junctions are smoothed like the route (fillets, then the Whittaker smoother
  within 1 m of OSM) and get their own height profile from the lidar DTM (`road_profile`, like
  the route), pinned to the road they leave: inside the route's asphalt, inside a roundabout or
  inside a side road built before them (side roads are profiled outward from the route), with
  the pin spread out over `PIN_TAPER_M`. Widths: OSM width, else lanes x lane width, else the
  Ontario class default (`ONTARIO_WIDTH_M`). Rural roads get gravel shoulders; unpaved roads
  are gravel. Markings: double yellow centre line on paved two-way roads with two lanes (on
  minor classes only where `lanes` says so), white edge lines on paved rural arterials and
  collectors, yellow left / white right edge lines on one-way ones.
- **Junctions**: at every node where side roads meet the route or each other, the corners are
  rounded with the class corner radius (`CORNER_RADIUS_M`): morphological closing of the union
  of the roads there (`buffer(R).buffer(-R)` fills each concave corner with an arc of radius R
  tangent to both edges). The route's gravel shoulder and edge line are removed where a mouth
  or island takes their place (`RouteMasks`); its centre line is broken across the junction
  where the route is not the through road. Junction control comes from
  `street_items.approach_stops` (OSM `highway=stop|give_way|traffic_signals` nodes win):
  stop bars on stop-controlled approaches, and `sign:stop` items beside them.
- **Height field** (`HeightField`): on the route's asphalt the route's own height; within
  `ROUTE_BLEND_M` of its edge a smooth blend to the other roads; elsewhere an inverse-distance
  blend of the side roads and roundabout planes nearby. Meshes sample it at every vertex, so a
  mouth meets the route edge exactly whatever the route's grade.
- **Meshing** (`triangulate`): each polygon is cut by a global `cell_m` grid and every piece
  is triangulated with a constrained Delaunay triangulation, so pieces sharing an edge (paved
  area, shoulders, islands) have the same vertices on it; boundaries are densified to 1.5 m.
- **Terrain, walls, trees**: `RoadNetwork.terrain_extra` gives the terrain its holes, edge points
  (the outer boundary vertices of every network mesh, with their heights) and batter stations;
  `RoadNetwork.corridor` (paved area + `verge_m`) joins the hidden-wall corridor, so side roads
  are drivable up to where they end and a wall closes their far end; islands get their own
  hidden walls.

Roundabouts (`geometry.roundabouts`) and driveways plug into the same polygons and height field.
Ontario values: docs in the scratchpad research (network.md, roundabouts.md) and the TAC
Geometric Design Guide; each constant says where it comes from.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.spatial import cKDTree

from driftmaps.ac.scene import DECAL_FLAGS
from driftmaps.elevation import road_profile
from driftmaps.geo.attributes import is_oneway, lane_width, parse_width, surface_class
from driftmaps.geometry.barriers import ScenePiece
from driftmaps.geometry.primitives import MeshData, compute_normals, compute_tangents, merge
from driftmaps.geometry.road import (
    PAINT_LIFT_M,
    RoadModel,
    fillet_corners,
    footprint_polygon,
    road_columns,
    simplify,
    smooth_centreline,
)
from driftmaps.geometry.street_items import (
    MAILBOX,
    SIGN_STOP,
    SIGN_YIELD,
    StreetItem,
    approach_stops,
    highway_rank,
)

__all__ = [
    "CORNER_RADIUS_M",
    "ONTARIO_WIDTH_M",
    "CentreLine",
    "HeightField",
    "NetNode",
    "NetWay",
    "NetworkOptions",
    "PlannedItem",
    "RoadNetwork",
    "RouteMasks",
    "SideRoad",
    "build_network",
    "corner_radius",
    "mesh_polygon",
    "side_road_width",
    "triangulate",
    "way_kind",
]

# --------------------------------------------------------------------------------------------
# Ontario values
# --------------------------------------------------------------------------------------------

# Carriageway width (m) when OSM has neither `width` nor `lanes`. Rural two-lane roads in
# Ontario: 3.5-3.75 m lanes on arterials, 3.25-3.5 m on collectors, 3.0-3.25 m on locals
# (TAC GDG Table 4.2.x; MTO Geometric Design Standards); urban locals 8.0-8.5 m curb to curb
# with parking (Halton Region / City of Burlington local road standard ~8.0 m). ASSUMED where
# research gives no better figure.
ONTARIO_WIDTH_M: dict[str, float] = {
    "motorway": 7.5,
    "trunk": 7.4,
    "primary": 7.2,
    "secondary": 7.0,
    "tertiary": 6.7,
    "unclassified": 6.0,
    "residential": 7.0,
    "living_street": 6.0,
    "road": 6.0,
    "busway": 7.0,
    "service": 4.5,
    "track": 3.5,
    "link": 4.8,
}
ONEWAY_WIDTH_M = 4.8  # one-way carriageway without tags (single-lane roundabout entry 4.3-4.9 m)
MIN_SIDE_WIDTH_M = 3.0

# Corner (curb-return) radius by the minor road's class at a corner. TAC: 7.5-9 m on local
# streets (passenger car / single unit), 10-12 m on collectors, 12-15 m where trucks turn
# (arterials, rural roads with farm traffic). ASSUMED mid-range values.
CORNER_RADIUS_M: dict[str, float] = {
    "motorway": 15.0,
    "trunk": 15.0,
    "primary": 15.0,
    "secondary": 13.0,
    "tertiary": 12.0,
    "unclassified": 10.0,
    "road": 10.0,
    "residential": 9.0,
    "busway": 12.0,
    "living_street": 6.0,
    "service": 6.0,
    "track": 6.0,
}

# Stop bar: white, 0.30-0.60 m wide across the approach lane(s) (OTM Book 11); placed at the
# desired stopping point, normally 1-3 m back from the edge of the crossing road. ASSUMED 0.45
# m wide and 2.0 m back.
STOP_BAR_WIDTH_M = 0.45
STOP_BAR_SETBACK_M = 2.0
SIGN_OFFSET_M = 2.5  # sign post this far beyond the asphalt edge (OTM Book 1: 2-4 m rural)
STOP_SIGN_BEHIND_M = 1.0  # sign this far back from the stop bar, towards the approaching car

EDGE_LINE_WIDTH_M = 0.10  # OTM Book 11: 10-15 cm edge lines
EDGE_LINE_INSET_M = 0.15
CENTRE_LINE_WIDTH_M = 0.10
CENTRE_GAP_M = 0.10  # between the two yellow lines

NOTCH_FILL_M = 2.5  # wedges and notches about this narrow become gravel ...
NOTCH_MAX_M2 = 40.0  # ... when no bigger than this
POLY_SIMPLIFY_M = 0.04  # side road outlines (from 1 m stations) are simplified this much
PIN_TAPER_M = 30.0  # a side road's pinned start blends into its own profile over this
ROUTE_BLEND_M = 5.0  # mouths blend from the route's height to the side road's over this

# Cul-de-sac bulb curb radius: standards say 13-15 m (Hamilton RD-116.01, Richmond Hill C-15),
# measured bulbs 9-14 m (median 11.7 m Brampton): 12 m, clamped 9-15 m (streetscape research).
TURNING_CIRCLE_RADIUS_M = 12.0
TURNING_LOOP_ISLAND_M = 5.0  # a turning loop's central island (R 4-6 m, ASSUMED)

_ROAD_CLASSES = frozenset(
    "motorway trunk primary secondary tertiary unclassified residential living_street road "
    "busway service track".split()
)
_RURAL_SHOULDER_CLASSES = frozenset(
    "motorway trunk primary secondary tertiary unclassified road".split()
)
_EDGE_LINE_CLASSES = frozenset("motorway trunk primary secondary tertiary".split())


def _base_class(highway: str) -> str:
    return highway[:-5] if highway.endswith("_link") else highway


def way_kind(tags: Mapping[str, str]) -> str:
    """ "road" (a public road the network builds), "ring" (a roundabout's circulatory road),
    "driveway", or "" (left out: parking aisles, drive-throughs, unpaved farm tracks, areas)."""
    hw = tags.get("highway", "")
    if tags.get("area") == "yes" or _base_class(hw) not in _ROAD_CLASSES:
        return ""
    if tags.get("junction") in ("roundabout", "circular") and hw not in ("service", "track"):
        return "ring"
    if hw == "service":
        svc = tags.get("service", "")
        if svc == "driveway":
            return "driveway"
        if svc in ("parking_aisle", "drive-through", "emergency_access", "parking"):
            return ""
        if tags.get("access", "") in ("private", "no"):
            return "driveway"
        return "road"
    if hw == "track":
        return "road" if surface_class(tags) == "asphalt" and "surface" in tags else ""
    return "road"


def side_road_width(tags: Mapping[str, str]) -> float:
    """Carriageway width (m): OSM `width:carriageway` / `width` / `est_width`, else `lanes` x
    the North American lane width, else the Ontario default for the class."""
    for key in ("width:carriageway", "width", "est_width"):
        w = parse_width(tags.get(key))
        if w is not None and w > 0:
            return float(np.clip(w, MIN_SIDE_WIDTH_M, 20.0))
    hw = tags.get("highway", "")
    raw = tags.get("lanes", "").split(";")[0].strip()
    if raw.isdigit() and int(raw) > 0:
        w = int(raw) * lane_width(hw, "na")
        if int(raw) == 1:
            w += 1.0  # a single lane has some edge to spare
        return float(np.clip(w, MIN_SIDE_WIDTH_M, 20.0))
    if is_oneway(tags) and hw not in ("motorway", "trunk"):
        return ONEWAY_WIDTH_M
    cls = "link" if hw.endswith("_link") else hw
    return ONTARIO_WIDTH_M.get(cls, 6.0)


def corner_radius(highway_a: str, highway_b: str) -> float:
    """Corner radius between two roads: that of the lower class of the two."""
    low = highway_a if highway_rank(highway_a) <= highway_rank(highway_b) else highway_b
    return CORNER_RADIUS_M.get(_base_class(low), 9.0)


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NetWay:
    """One OSM highway way in AC coordinates: `xz` (K, 2) aligned with `nodes`."""

    id: int
    nodes: tuple[int, ...]
    xz: np.ndarray = field(repr=False)
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NetNode:
    """One tagged OSM node (stop sign, turning circle ...) in AC coordinates."""

    id: int
    xz: np.ndarray = field(repr=False)
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NetworkOptions:
    radius_m: float = 150.0  # side roads are built this far from the route centreline
    shoulder_m: float = 1.0  # gravel shoulder on rural side roads
    verge_m: float = 4.0  # hidden walls stand this far beyond the paved area
    cell_m: float = 3.0  # mesh grid (interior vertices) of network polygons
    batch_m: float = 400.0  # one KN5 mesh per material per cell of this size
    max_grade: float = 0.20
    max_vertical_curvature: float = 1.0 / 150.0
    source_resolution_m: float = 1.0
    min_piece_m: float = 6.0  # side road pieces shorter than this are dropped
    turning_circles: bool = True
    stop_bars: bool = True


# --------------------------------------------------------------------------------------------
# Centre lines and the height field
# --------------------------------------------------------------------------------------------


class CentreLine:
    """A road centreline with heights and half widths per station, for projections."""

    def __init__(self, xz: np.ndarray, y: np.ndarray, half_width: np.ndarray, name: str = ""):
        self.xz = np.asarray(xz, np.float64).reshape(-1, 2)
        self.y = np.asarray(y, np.float64).reshape(-1)
        self.hw = np.broadcast_to(np.asarray(half_width, np.float64), (len(self.xz),)).copy()
        self.name = name
        self.tree = cKDTree(self.xz)
        self.lo = self.xz.min(axis=0)
        self.hi = self.xz.max(axis=0)

    def near(self, p: np.ndarray, reach: float) -> np.ndarray:
        """Mask of the points within the bounding box + `reach`."""
        return np.all((p >= self.lo - reach) & (p <= self.hi + reach), axis=1)

    def nearest(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(foot points (M, 2) on the centreline, unit tangents (M, 2), distances (M,), half
        widths there (M,)) of every point."""
        p = np.asarray(p, np.float64).reshape(-1, 2)
        _, i = self.tree.query(p)
        a, t, d = self._segments(p, i)
        n = len(self.xz)
        b = np.minimum(a + 1, n - 1)
        seg = self.xz[b] - self.xz[a]
        foot = self.xz[a] + t[:, None] * seg
        tan = seg / np.maximum(np.linalg.norm(seg, axis=1), 1e-9)[:, None]
        hw = self.hw[a] + t * (self.hw[b] - self.hw[a])
        return foot, tan, d, hw

    def _segments(self, q: np.ndarray, i: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(segment start index, parameter, distance) of the nearest of the two segments next to
        each point's nearest station `i`."""
        n = len(self.xz)
        best_d = np.full(len(q), np.inf)
        best_t = np.zeros(len(q))
        best_a = i.copy()
        for a in (np.maximum(i - 1, 0), i):
            b = np.minimum(a + 1, n - 1)
            seg = self.xz[b] - self.xz[a]
            ll = np.einsum("ij,ij->i", seg, seg)
            t = np.where(
                ll > 1e-12,
                np.einsum("ij,ij->i", q - self.xz[a], seg) / np.maximum(ll, 1e-12),
                0.0,
            )
            t = np.clip(t, 0.0, 1.0)
            d = np.linalg.norm(self.xz[a] + t[:, None] * seg - q, axis=1)
            better = d < best_d
            best_d = np.where(better, d, best_d)
            best_t = np.where(better, t, best_t)
            best_a = np.where(better, a, best_a)
        return best_a, best_t, best_d

    def project(self, p: np.ndarray, reach: float = np.inf) -> tuple[np.ndarray, np.ndarray]:
        """(distance beyond the asphalt edge (0 inside), height) of every point; inf / NaN for
        points further than `reach` from the centreline."""
        p = np.asarray(p, np.float64).reshape(-1, 2)
        d_out = np.full(len(p), np.inf)
        h_out = np.full(len(p), np.nan)
        if len(self.xz) == 0 or len(p) == 0:
            return d_out, h_out
        sel = np.flatnonzero(self.near(p, reach)) if np.isfinite(reach) else np.arange(len(p))
        if not len(sel):
            return d_out, h_out
        q = p[sel]
        dist, i = self.tree.query(q, distance_upper_bound=reach)
        ok = np.isfinite(dist)
        if not ok.any():
            return d_out, h_out
        q, i, sel = q[ok], i[ok], sel[ok]
        best_a, best_t, best_d = self._segments(q, i)
        b = np.minimum(best_a + 1, len(self.xz) - 1)
        h = self.y[best_a] + best_t * (self.y[b] - self.y[best_a])
        hw = self.hw[best_a] + best_t * (self.hw[b] - self.hw[best_a])
        d_out[sel] = np.maximum(best_d - hw, 0.0)
        h_out[sel] = h
        return d_out, h_out


@dataclass
class PlaneZone:
    """A disc where the network surface is a plane (a roundabout): y = a + g . (xz - c)."""

    centre: np.ndarray
    radius: float
    a: float
    grad: np.ndarray

    def height(self, p: np.ndarray) -> np.ndarray:
        return self.a + (np.asarray(p, np.float64).reshape(-1, 2) - self.centre) @ self.grad

    def distance(self, p: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(np.asarray(p, np.float64).reshape(-1, 2) - self.centre, axis=1)
        return np.maximum(r - self.radius, 0.0)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class HeightField:
    """Network surface height (AC Y) at XZ points; see the module docstring."""

    REACH_M = 60.0

    def __init__(
        self,
        route: CentreLine | Sequence[CentreLine] | None,
        owners: Sequence[CentreLine] = (),
        planes: Sequence[PlaneZone] = (),
        blend_m: float = ROUTE_BLEND_M,
    ):
        # The route: pieces whose `hw` is the outer edge (asphalt + shoulder), so the whole
        # footprint of its ribbon reads the route's own (flat cross-section) height.
        if route is None:
            self.route: list[CentreLine] = []
        elif isinstance(route, CentreLine):
            self.route = [route]
        else:
            self.route = list(route)
        self.owners = list(owners)
        self.planes = list(planes)
        self.blend_m = blend_m

    def route_height(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(distance beyond the route's outer edge, route height) from the nearest piece."""
        d_best = np.full(len(p), np.inf)
        h_best = np.full(len(p), np.nan)
        for piece in self.route:
            d, h = piece.project(p)
            better = d < d_best
            d_best = np.where(better, d, d_best)
            h_best = np.where(better, h, h_best)
        return d_best, h_best

    def others(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(blend of the side roads and planes, weight sum) at the points."""
        num = np.zeros(len(p))
        den = np.zeros(len(p))
        for o in self.owners:
            d, h = o.project(p, self.REACH_M)
            ok = np.isfinite(d) & np.isfinite(h)
            w = np.where(ok, 1.0 / (np.where(ok, d, 0.0) + 0.2) ** 3, 0.0)
            num += w * np.where(ok, h, 0.0)
            den += w
        for z in self.planes:
            d = z.distance(p)
            ok = d < self.REACH_M
            w = np.where(ok, 1.0 / (d + 0.2) ** 3, 0.0)
            num += w * z.height(p)
            den += w
        return np.where(den > 0, num / np.maximum(den, 1e-300), np.nan), den

    def __call__(self, xz: np.ndarray) -> np.ndarray:
        p = np.asarray(xz, np.float64).reshape(-1, 2)
        other, den = self.others(p)
        if not self.route:
            return other
        d_r, h_r = self.route_height(p)
        a = _smoothstep(d_r / self.blend_m)
        out = np.where(den > 0, (1.0 - a) * h_r + a * np.nan_to_num(other), h_r)
        return np.where(np.isfinite(out), out, other)


# --------------------------------------------------------------------------------------------
# Polygon meshing
# --------------------------------------------------------------------------------------------


def _polygons(geom: shapely.Geometry | None, min_area: float = 1e-4) -> list[shapely.Polygon]:
    if geom is None or geom.is_empty:
        return []
    parts = shapely.get_parts(geom)
    out = []
    for p in parts:
        if isinstance(p, shapely.Polygon) and p.area > min_area:
            out.append(p)
        elif isinstance(p, shapely.MultiPolygon | shapely.GeometryCollection):
            out += _polygons(p, min_area)
    return out


def triangulate(
    geom: shapely.Geometry, cell_m: float = 3.0, seg_m: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    """(vertices (V, 2), triangles (T, 3)) covering the polygon(s) exactly: each piece of the
    polygon in a global `cell_m` grid gets a constrained Delaunay triangulation, so meshes of
    neighbouring polygons share their vertices along common edges. Triangles are
    counter-clockwise in (x, z) (they face up in AC, Y = up, Z = south)."""
    polys = _polygons(geom)
    if not polys:
        return np.zeros((0, 2)), np.zeros((0, 3), np.int64)
    tri_xy = []
    for poly in polys:
        poly = shapely.segmentize(poly, seg_m)
        x0, z0, x1, z1 = poly.bounds
        ix = np.arange(math.floor(x0 / cell_m), math.ceil(x1 / cell_m))
        iz = np.arange(math.floor(z0 / cell_m), math.ceil(z1 / cell_m))
        gx, gz = np.meshgrid(ix * cell_m, iz * cell_m, indexing="ij")
        boxes = shapely.box(gx.ravel(), gz.ravel(), gx.ravel() + cell_m, gz.ravel() + cell_m)
        shapely.prepare(poly)
        boxes = boxes[shapely.intersects(poly, boxes)]
        inside = shapely.contains_properly(poly, boxes)
        pieces = list(boxes[inside]) + list(shapely.intersection(boxes[~inside], poly))
        pieces = [q for piece in pieces for q in _polygons(piece, 1e-6)]
        if not pieces:
            continue
        tris = shapely.get_parts(shapely.constrained_delaunay_triangles(np.array(pieces)))
        tris = tris[shapely.get_type_id(tris) == 3]
        if not len(tris):
            continue
        coords = shapely.get_coordinates(shapely.get_exterior_ring(tris)).reshape(-1, 4, 2)
        tri_xy.append(coords[:, :3])
    if not tri_xy:
        return np.zeros((0, 2)), np.zeros((0, 3), np.int64)
    t = np.concatenate(tri_xy)
    key = np.round(t.reshape(-1, 2) / 1e-5).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    verts = np.zeros((len(uniq), 2))
    np.add.at(verts, inv.reshape(-1), t.reshape(-1, 2))
    counts = np.bincount(inv.reshape(-1), minlength=len(uniq))
    verts /= counts[:, None]
    tri = inv.reshape(-1, 3)
    # The AC normal's Y is -(cross product in (x, z)): make it positive (facing up).
    a, b, c = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    tri = tri[np.abs(cross) > 1e-10]
    cross = cross[np.abs(cross) > 1e-10]
    flip = cross > 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return verts, tri


def mesh_polygon(
    geom: shapely.Geometry,
    height_fn: Callable[[np.ndarray], np.ndarray],
    *,
    cell_m: float = 3.0,
    uv_tile_m: float = 4.0,
    lift: float = 0.0,
    seg_m: float = 1.5,
) -> MeshData | None:
    """The polygon(s) as an up-facing mesh on `height_fn` (+ `lift`), planar UVs; `seg_m`:
    the longest boundary edge."""
    verts, tri = triangulate(geom, cell_m, seg_m)
    if not len(tri):
        return None
    used = np.unique(tri)
    verts, tri = verts[used], np.searchsorted(used, tri)
    y = np.asarray(height_fn(verts), np.float64) + lift
    if not np.isfinite(y).all():
        bad = ~np.isfinite(y)
        good = np.flatnonzero(~bad)
        if not len(good):
            return None
        _, nn = cKDTree(verts[good]).query(verts[bad])
        y[bad] = y[good[nn]]
    pos = np.stack([verts[:, 0], y, verts[:, 1]], axis=1)
    pos_up = _face_up(pos, tri)
    uv = np.stack([verts[:, 0], verts[:, 1]], axis=1) / uv_tile_m
    data = MeshData(pos, np.tile([0.0, 1.0, 0.0], (len(pos), 1)), uv, np.zeros_like(pos), pos_up)
    return compute_tangents(compute_normals(data))


def _face_up(pos: np.ndarray, tri: np.ndarray) -> np.ndarray:
    n = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
    tri = tri.copy()
    flip = n[:, 1] < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return tri.reshape(-1)


def boundary_points(mesh: MeshData, spacing_m: float = 1.0) -> np.ndarray:
    """(K, 3) points on the mesh's open boundary (edges used by one triangle): its vertices
    and points every `spacing_m` along longer edges (on the edge, so a terrain joined to them
    has no crack, and dense enough for its triangulation to follow the outline)."""
    tri = mesh.triangles
    if not len(tri):
        return np.zeros((0, 3))
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    pos = mesh.positions
    key = np.round(pos[:, [0, 2]] / 1e-5).astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    vid = vid.reshape(-1)
    ee = np.sort(vid[e], axis=1)
    _, inv, counts = np.unique(ee, axis=0, return_inverse=True, return_counts=True)
    open_edges = ee[counts[inv.reshape(-1)] == 1]
    if not len(open_edges):
        return np.zeros((0, 3))
    first = np.zeros(vid.max() + 1, np.int64)
    first[vid[::-1]] = np.arange(len(vid))[::-1]
    a, b = pos[first[open_edges[:, 0]]], pos[first[open_edges[:, 1]]]
    n = np.maximum(1, np.ceil(np.linalg.norm((b - a)[:, [0, 2]], axis=1) / spacing_m)).astype(int)
    seg = np.repeat(np.arange(len(a)), n - 1)
    t = (np.arange((n - 1).sum()) - np.repeat(np.cumsum(n - 1) - (n - 1), n - 1) + 1) / np.repeat(
        n, n - 1
    )
    mids = a[seg] + t[:, None] * (b - a)[seg]
    return np.concatenate([pos[first[np.unique(open_edges)]], mids])


# --------------------------------------------------------------------------------------------
# Decals and batching
# --------------------------------------------------------------------------------------------


def strip_mesh(
    left: np.ndarray,
    right: np.ndarray,
    keep: np.ndarray,
    height_fn: Callable[[np.ndarray], np.ndarray],
    lift: float = PAINT_LIFT_M,
) -> MeshData | None:
    """A painted strip between two (N, 2) XZ edge polylines, on `height_fn` + `lift`, with
    only the segments where `keep` (N - 1,) is True."""
    keep = np.asarray(keep, bool)
    if len(left) < 2 or not keep.any():
        return None
    xz = np.concatenate([left, right])
    y = np.asarray(height_fn(xz), np.float64) + lift
    n = len(left)
    pos = np.stack([xz[:, 0], y, xz[:, 1]], axis=1)
    i = np.flatnonzero(keep)
    a, b, c, d = i, i + 1, n + i + 1, n + i
    tri = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    seg = np.linalg.norm(np.diff(left, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    uv = np.concatenate([np.stack([np.zeros(n), s], 1), np.stack([np.ones(n), s], 1)])
    used = np.unique(tri)
    tri = np.searchsorted(used, tri)
    pos, uv = pos[used], uv[used]
    ok = np.isfinite(pos[:, 1])
    if not ok.all():
        return None
    data = MeshData(
        pos, np.tile([0.0, 1.0, 0.0], (len(pos), 1)), uv, np.zeros_like(pos), _face_up(pos, tri)
    )
    return compute_tangents(compute_normals(data))


# Material keys -> (KN5 node prefix, material name, texture stem, flags)
MATERIALS: dict[str, tuple[str, str, str, dict]] = {
    "asphalt": ("1ROAD_net", "road", "asphalt", {}),
    "gravel": ("1GRAVEL_net", "gravel", "gravel", {}),
    "paint_white": ("paint_white_net", "paint_white", "paint_white", dict(DECAL_FLAGS)),
    "paint_yellow": ("paint_yellow_net", "paint_yellow", "paint_yellow", dict(DECAL_FLAGS)),
    "curb": ("1ROAD_curb", "concrete", "concrete", {}),
    "pavers": ("1ROAD_apron", "pavers", "pavers", {}),
    "island_grass": ("1GRASS_isl", "grass", "grass", {}),
    "island_concrete": ("1ROAD_isl", "concrete", "concrete", {}),
}


class Batcher:
    """Collects meshes per (material, cell) and merges them into KN5 pieces."""

    def __init__(self, cell_m: float = 400.0):
        self.cell_m = cell_m
        self.parts: dict[tuple[str, int, int], list[MeshData]] = defaultdict(list)

    def add(self, key: str, mesh: MeshData | None) -> None:
        if mesh is None or mesh.vertex_count == 0:
            return
        c = mesh.positions[:, [0, 2]].mean(axis=0)
        cell = (key, int(math.floor(c[0] / self.cell_m)), int(math.floor(c[1] / self.cell_m)))
        self.parts[cell].append(mesh)

    def pieces(self) -> list[ScenePiece]:
        out: list[ScenePiece] = []
        counters: dict[str, int] = defaultdict(int)
        for key, cx, cz in sorted(self.parts):
            prefix, material, texture, flags = MATERIALS[key]
            k = counters[key]
            counters[key] += 1
            mesh = merge(self.parts[(key, cx, cz)])
            out.append(ScenePiece(f"{prefix}{k:03d}", material, mesh, dict(flags), texture))
        return out


# --------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------


@dataclass
class PlannedItem:
    """A street item before the terrain exists: `xz` on the ground, height filled later."""

    kind: str
    xz: tuple[float, float]
    facing: tuple[float, float]
    source: str = "rule"
    text: str = ""
    ref: str = ""


@dataclass
class RouteMasks:
    """Per route segment (N - 1,) what the network replaces on the route ribbon."""

    hidden: np.ndarray  # road, shoulders and paint left out (a roundabout replaces them)
    shoulder_left: np.ndarray  # True: keep
    shoulder_right: np.ndarray
    edge_left: np.ndarray
    edge_right: np.ndarray
    centre: np.ndarray

    @classmethod
    def keep_all(cls, n_seg: int) -> RouteMasks:
        ones = np.ones(n_seg, bool)
        return cls(~ones, ones.copy(), ones.copy(), ones.copy(), ones.copy(), ones.copy())


@dataclass
class SideRoad:
    """One built side road piece: smoothed centreline with its profile (`model`), its tags
    and how its ends attach ("route", "ring", "road" (another side road), "end" (dead end) or
    "cut" (the scenery radius))."""

    index: int
    model: RoadModel
    tags: Mapping[str, str]
    way_ids: np.ndarray  # (N,) OSM way per station
    oneway: int  # +1 traffic along s, -1 against, 0 two-way
    paved: bool
    shoulder: float
    start: tuple[str, int]
    end: tuple[str, int]
    start_node: int
    end_node: int

    @property
    def highway(self) -> str:
        return self.tags.get("highway", "")

    @property
    def length(self) -> float:
        return float(self.model.s[-1] - self.model.s[0])

    @property
    def xz(self) -> np.ndarray:
        return self.model.centre[:, [0, 2]]


@dataclass
class Junction:
    node: int
    xz: np.ndarray
    route_s: float | None  # station on the route (None: side roads only)
    legs: list[tuple[int, str]]  # (side road index, "start"/"end")
    radius: float = 9.0
    fillets: shapely.Geometry | None = None
    controls: list[tuple[str, str, str | None]] = field(default_factory=list)  # (leg, class, ctrl)


@dataclass
class RoadNetwork:
    """What the network stage made (see `build_network`)."""

    model: RoadModel  # the route model (heights adjusted to roundabout planes)
    masks: RouteMasks
    pieces: list[ScenePiece] = field(default_factory=list)
    side_roads: list[SideRoad] = field(default_factory=list)
    junctions: list[Junction] = field(default_factory=list)
    roundabouts: list = field(default_factory=list)
    planned_items: list[PlannedItem] = field(default_factory=list)
    footprint: shapely.Geometry = field(default_factory=shapely.Polygon)  # terrain holes
    paved: shapely.Geometry = field(default_factory=shapely.Polygon)  # drivable network area
    islands: list[shapely.Geometry] = field(default_factory=list)
    island_walls: list[np.ndarray] = field(default_factory=list)  # closed (K, 2) rings
    edge_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    height: HeightField | None = None
    polygons: dict[str, shapely.Geometry] = field(default_factory=dict)  # for debug images
    stats: dict[str, object] = field(default_factory=dict)
    # For the driveway stage (after buildings): carriageways, the route's asphalt edges and
    # outer edges (N, 3), junction nodes, OSM driveway ways, the driveways built.
    carriage: list[shapely.Geometry] = field(default_factory=list, repr=False)
    route_columns: tuple[np.ndarray, ...] = field(default=(), repr=False)
    junction_xz: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)), repr=False)
    driveway_ways: list[NetWay] = field(default_factory=list, repr=False)
    driveways: list = field(default_factory=list)

    def corridor(self, verge_m: float) -> shapely.Geometry:
        """The network's share of the hidden-wall corridor (simplified to 0.25 m: the paved
        area's outline has a vertex every metre, which the walls do not need)."""
        if self.paved.is_empty:
            return shapely.Polygon()
        grown = shapely.buffer(self.paved, verge_m, quad_segs=4)
        return shapely.simplify(grown, 0.25).buffer(0)

    def centre_points(self) -> np.ndarray:
        """(K, 3) side road centreline points (for nearest-road heights)."""
        if not self.side_roads:
            return np.zeros((0, 3))
        return np.concatenate([r.model.centre for r in self.side_roads])

    def street_items(self, ground: Callable[[np.ndarray], np.ndarray]) -> list[StreetItem]:
        """The planned items on the ground (`ground(xz)` = visible terrain height)."""
        if not self.planned_items:
            return []
        xz = np.array([p.xz for p in self.planned_items], np.float64)
        y = np.asarray(ground(xz), np.float64)
        if self.height is not None:
            bad = ~np.isfinite(y)
            if bad.any():
                y[bad] = self.height(xz[bad])
        y = np.nan_to_num(y)
        return [
            StreetItem(
                p.kind,
                (round(float(x), 3), round(float(h), 3), round(float(z), 3)),
                (round(float(p.facing[0]), 4), round(float(p.facing[1]), 4)),
                p.source,
                p.text,
                p.ref,
            )
            for p, (x, z), h in zip(self.planned_items, xz, y, strict=True)
        ]

    def terrain_stations(self, step: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """(points (K, 3), outer half widths (K,)) for the terrain's cut/fill batters: side
        road stations and the network's outer boundary."""
        pts = [r.model.centre[::step] for r in self.side_roads]
        outer = [r.model.outer[::step] for r in self.side_roads]
        if len(self.edge_points):
            pts.append(self.edge_points)
            outer.append(np.zeros(len(self.edge_points)))
        if not pts:
            return np.zeros((0, 3)), np.zeros(0)
        return np.concatenate(pts), np.concatenate(outer)


# --------------------------------------------------------------------------------------------
# Topology: chains of OSM segments between junctions
# --------------------------------------------------------------------------------------------


def _key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


class NetGraph:
    """The OSM network near the route: node positions, the side-road segments (not driven by
    the route, not roundabout rings) and every node's neighbours over all roads."""

    def __init__(
        self,
        ways: Sequence[NetWay],
        route_segments: set[tuple[int, int]],
        route_nodes: set[int],
    ):
        self.coords: dict[int, np.ndarray] = {}
        self.ways: dict[int, NetWay] = {}
        self.adj: dict[int, list[tuple[int, int, bool]]] = defaultdict(list)
        self.nbrs: dict[int, set[int]] = defaultdict(set)
        self.rings: list[NetWay] = []
        self.driveways: list[NetWay] = []
        self.route_nodes = set(route_nodes)
        self.route_segments = set(route_segments)
        self.kind: dict[int, str] = {}
        for way in sorted(ways, key=lambda w: w.id):
            kind = way_kind(way.tags)
            self.kind[way.id] = kind
            if not kind:
                continue
            self.ways[way.id] = way
            for k, nid in enumerate(way.nodes):
                self.coords.setdefault(int(nid), np.asarray(way.xz[k], np.float64))
            if kind == "driveway":
                self.driveways.append(way)
                continue
            if kind == "ring":
                self.rings.append(way)
            for a, b in zip(way.nodes[:-1], way.nodes[1:], strict=True):
                a, b = int(a), int(b)
                if a == b:
                    continue
                self.nbrs[a].add(b)
                self.nbrs[b].add(a)
                if kind == "road" and _key(a, b) not in self.route_segments:
                    self.adj[a].append((b, way.id, True))
                    self.adj[b].append((a, way.id, False))
        for a, b in route_segments:  # the route's own segments count for the degree
            self.nbrs[a].add(b)
            self.nbrs[b].add(a)
        self.ring_nodes = {int(n) for w in self.rings for n in w.nodes}

    def chains(
        self, extra_breaks: set[int] = frozenset()
    ) -> list[tuple[list[int], list[int], list[bool]]]:
        """(nodes, way per segment, along-the-way per segment) for every chain of side-road
        segments between break nodes (junctions, dead ends, route and ring nodes)."""
        breaks = {
            n
            for n in self.adj
            if len(self.nbrs[n]) != 2 or n in self.route_nodes or n in self.ring_nodes
        } | (set(extra_breaks) & set(self.adj))
        seen: set[tuple[int, int, int]] = set()
        out = []

        def walk(start: int, first: tuple[int, int, bool]) -> None:
            nodes, ways, along = [start], [], []
            prev, (cur, w, fw) = start, first
            seen.add((*_key(start, cur), w))
            nodes.append(cur)
            ways.append(w)
            along.append(fw)
            while cur not in breaks and cur != start:
                nxt = [e for e in self.adj[cur] if (*_key(cur, e[0]), e[1]) not in seen]
                if len(nxt) != 1:
                    break
                prev, (cur, w, fw) = cur, nxt[0]
                seen.add((*_key(prev, cur), w))
                nodes.append(cur)
                ways.append(w)
                along.append(fw)
            out.append((nodes, ways, along))

        for n in sorted(breaks):
            for e in sorted(self.adj[n]):
                if (*_key(n, e[0]), e[1]) not in seen:
                    walk(n, e)
        for n in sorted(self.adj):  # closed loops without a break node
            for e in sorted(self.adj[n]):
                if (*_key(n, e[0]), e[1]) not in seen:
                    walk(n, e)
        return out


# --------------------------------------------------------------------------------------------
# Side roads: geometry and profiles
# --------------------------------------------------------------------------------------------


def _resample(xz: np.ndarray, step: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    seg = np.linalg.norm(np.diff(xz, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(1, math.ceil(s[-1] / step))
    t = np.linspace(0.0, s[-1], n + 1)
    return np.stack([np.interp(t, s, xz[:, 0]), np.interp(t, s, xz[:, 1])], axis=1), t


def _smooth_chain(xz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stations every ~1 m along a chain and their distance along the OSM polyline."""
    seg = np.linalg.norm(np.diff(xz, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-3])
    xz = xz[keep]
    if len(xz) < 2:
        return xz, np.zeros(len(xz))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))])
    if s[-1] < 15.0 or len(xz) < 3:
        return _resample(xz)
    f_xz, f_p = fillet_corners(*simplify(xz, s, 0.3), radius_m=10.0)
    out, param = smooth_centreline(f_xz, f_p, max_deviation_m=1.0, cutoff_m=30.0)
    return out, np.clip(np.maximum.accumulate(param), 0.0, s[-1])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index ranges of the True runs."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _box(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values.astype(float)
    pad = window // 2
    k = np.ones(2 * pad + 1) / (2 * pad + 1)
    return np.convolve(np.pad(values.astype(float), pad, mode="edge"), k, mode="valid")


def _pinned_profile(
    s: np.ndarray,
    samples: np.ndarray,
    target: np.ndarray,
    options: NetworkOptions,
) -> np.ndarray:
    """`road_profile` of DTM `samples` (N, K) with the stations where `target` is finite held
    on it (exactly, the correction spread over `PIN_TAPER_M`)."""
    pinned = np.isfinite(target)
    z_in = samples.copy()
    z_in[pinned] = target[pinned, None]
    w = np.where(pinned, 25.0, 1.0)
    usable = pinned | np.isfinite(samples).any(axis=1)
    if usable.sum() < 2:
        base = np.nanmean(target) if pinned.any() else 0.0
        return np.full(len(s), base)
    if len(s) >= 4:
        try:
            z = road_profile(
                s,
                z_in,
                np.where(usable, w, 0.0),
                source_resolution_m=options.source_resolution_m,
                max_grade=options.max_grade,
                max_vertical_curvature=options.max_vertical_curvature,
            )
        except ValueError:
            z = np.interp(s, s[usable], np.nanmedian(z_in[usable], axis=1))
    else:
        z = np.interp(s, s[usable], np.nanmedian(z_in[usable], axis=1))
    if pinned.any():
        r = np.where(pinned, target - z, 0.0)
        idx = np.flatnonzero(pinned)
        near = idx[np.clip(np.searchsorted(idx, np.arange(len(s))), 0, len(idx) - 1)]
        near2 = idx[np.clip(np.searchsorted(idx, np.arange(len(s))) - 1, 0, len(idx) - 1)]
        pick = np.where(np.abs(s[near] - s) < np.abs(s[near2] - s), near, near2)
        dist = np.abs(s - s[pick])
        z = z + r[pick] * (1.0 - _smoothstep(dist / PIN_TAPER_M))
    return z


# --------------------------------------------------------------------------------------------
# The network stage
# --------------------------------------------------------------------------------------------


def _quads(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quad polygons (N - 1,) between two (N, 3) edge polylines, in XZ."""
    q = np.stack([a[:-1], a[1:], b[1:], b[:-1], a[:-1]], axis=1)[..., [0, 2]]
    polys = shapely.polygons(q)
    bad = ~shapely.is_valid(polys)
    if bad.any():
        polys[bad] = shapely.make_valid(polys[bad])
    return polys


def _union(geoms: Sequence[shapely.Geometry]) -> shapely.Geometry:
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return shapely.Polygon()
    return shapely.union_all(np.array(geoms, dtype=object))


def _mask_by_s(s: np.ndarray, ranges: Sequence[tuple[float, float]]) -> np.ndarray:
    """(N - 1,) True for segments whose midpoint lies in any [a, b]."""
    mid = 0.5 * (s[:-1] + s[1:])
    out = np.zeros(len(mid), bool)
    for a, b in ranges:
        out |= (mid >= min(a, b)) & (mid <= max(a, b))
    return out


@dataclass
class _Leg:
    """One road arm at a junction, pointing away from it."""

    direction: np.ndarray  # unit XZ vector away from the junction
    highway: str
    kind: str  # "route_in", "route_out", "side"
    road: int = -1  # side road index
    end: str = ""  # "start" / "end" of that side road
    approach: bool = True  # traffic arrives at the junction on it
    poly: shapely.Geometry | None = None  # its carriageway near the junction


def build_network(
    model: RoadModel,
    ways: Sequence[NetWay],
    nodes: Sequence[NetNode] = (),
    *,
    route_nodes: Sequence[int] = (),
    route_segments: set[tuple[int, int]] = frozenset(),
    route_highway: Sequence[str] | None = None,
    route_oneway: np.ndarray | None = None,
    dtm: Callable[[np.ndarray], np.ndarray] | None = None,
    driving_side: str = "right",
    options: NetworkOptions | None = None,
    nodes_known: bool = True,
) -> RoadNetwork:
    """Build the network around the route `model` (see the module docstring).

    `ways` / `nodes`: OSM highway ways and tagged nodes in AC coordinates; `route_nodes` and
    `route_segments` (sorted node pairs): the OSM nodes and segments the route drives
    (including the real run-up and run-off); `route_highway` / `route_oneway`: the route's
    highway class and one-way flag (+1 legal, -1 against, 0 two-way) per station; `dtm(xz)`:
    bare-earth heights in AC Y. Returns the network with the route model (heights moved onto
    roundabout planes) and what it replaces on the route ribbon."""
    from driftmaps.geometry import roundabouts as rbm

    options = options or NetworkOptions()
    n_st = len(model)
    route_highway = list(route_highway) if route_highway is not None else ["tertiary"] * n_st
    route_oneway = (
        np.asarray(route_oneway, np.int64) if route_oneway is not None else np.zeros(n_st, np.int64)
    )
    graph = NetGraph(ways, set(route_segments), set(route_nodes))
    masks = RouteMasks.keep_all(n_st - 1)
    net = RoadNetwork(model=model, masks=masks)
    if not graph.ways or dtm is None:
        return net

    # Roundabouts first: they move the route onto their planes and hide its ribbon there.
    rbs = rbm.find_roundabouts(graph, driving_side)
    model, hidden = rbm.settle_route(model, rbs, dtm)
    net.model, net.roundabouts = model, rbs
    masks.hidden = hidden
    route_xz = model.centre[:, [0, 2]]
    route_tree = cKDTree(route_xz)
    route_cl = CentreLine(route_xz, model.centre[:, 1], model.half_width, "route")
    planes = [rb.zone for rb in rbs]

    # ---- Side roads: chains, cut to the scenery radius ----------------------------------
    turning = {
        nd.id: nd for nd in nodes if nd.tags.get("highway") in ("turning_circle", "turning_loop")
    }
    sides: list[SideRoad] = []
    raw: list[dict] = []
    for chain_nodes, chain_ways, along in graph.chains(set(turning)):
        xz_osm = np.array([graph.coords[n] for n in chain_nodes])
        seg_len = np.linalg.norm(np.diff(xz_osm, axis=0), axis=1)
        if seg_len.sum() < 1.0:
            continue
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        st, param = _smooth_chain(xz_osm)
        if len(st) < 2:
            continue
        seg_i = np.clip(np.searchsorted(cum, param, side="right") - 1, 0, len(chain_ways) - 1)
        way_ids = np.asarray(chain_ways)[seg_i]
        dist, _ = route_tree.query(st)
        inside = dist <= options.radius_m
        for k, w in enumerate(way_ids.tolist()):  # no side-road bridges/tunnels (yet)
            t = graph.ways[w].tags
            if t.get("bridge", "no") not in ("no", "") or t.get("tunnel", "no") not in ("no", ""):
                inside[k] = False
        for i0, i1 in _runs(inside):
            if i1 - i0 < 1:
                continue
            piece = st[i0 : i1 + 1]
            if np.linalg.norm(np.diff(piece, axis=0), axis=1).sum() < options.min_piece_m:
                continue
            raw.append(
                dict(
                    xz=piece,
                    ways=way_ids[i0 : i1 + 1],
                    along=np.asarray(along)[seg_i[i0 : i1 + 1]],
                    start_node=chain_nodes[0] if i0 == 0 else -1,
                    end_node=chain_nodes[-1] if i1 == len(st) - 1 else -1,
                )
            )

    def attach(node: int) -> tuple[str, int]:
        if node < 0:
            return ("cut", -1)
        if node in graph.ring_nodes:
            return ("ring", node)
        if node in graph.route_nodes:
            return ("route", node)
        if len(graph.nbrs[node]) >= 3:
            return ("road", node)
        return ("end", node)

    for r in raw:
        r["start"], r["end"] = attach(r["start_node"]), attach(r["end_node"])
        # Roads leaving the route or a ring start there (profiles are pinned from the start).
        rank = {"ring": 0, "route": 1, "road": 2, "end": 3, "cut": 4}
        if rank[r["end"][0]] < rank[r["start"][0]]:
            for k in ("xz", "ways", "along"):
                r[k] = r[k][::-1]
            r["along"] = ~r["along"]
            r["start"], r["end"] = r["end"], r["start"]
            r["start_node"], r["end_node"] = r["end_node"], r["start_node"]

    # Profile order: outward from the route (roads attached to built ones come after them).
    order = sorted(
        range(len(raw)),
        key=lambda i: (
            {"ring": 0, "route": 0, "road": 1, "end": 2, "cut": 2}[raw[i]["start"][0]],
            i,
        ),
    )
    # DTM for all side roads in one go.
    models0 = []
    for r in raw:
        xz = r["xz"]
        tags = [graph.ways[w].tags for w in r["ways"].tolist()]
        width = _box(np.array([side_road_width(t) for t in tags]), 15)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))])
        centre = np.stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]], axis=1)
        m0 = RoadModel(
            s=s, centre=centre, half_width=width / 2, shoulder=0.0, centre_line=False, bridge=False
        )
        models0.append(m0)
    lat = np.array([-0.5, 0.0, 0.5])
    pts = [
        (
            m0.centre[:, None, [0, 2]]
            + (lat[None, :, None] * 2 * m0.half_width[:, None, None]) * m0.left[:, None, [0, 2]]
        ).reshape(-1, 2)
        for m0 in models0
    ]
    if pts:
        allz = np.asarray(dtm(np.concatenate(pts)), np.float64)
        bounds = np.cumsum([0] + [len(p) for p in pts])
    owners: list[CentreLine] = []
    built: dict[int, int] = {}
    for i in order:
        r, m0 = raw[i], models0[i]
        xz = m0.centre[:, [0, 2]]
        samples = allz[bounds[i] : bounds[i + 1]].reshape(-1, 3)
        target = np.full(len(xz), np.nan)
        d_r, h_r = route_cl.project(xz, 30.0)
        on_route = d_r <= 0.3
        target[on_route] = h_r[on_route]
        for z in planes:
            inz = z.distance(xz) <= 0.0
            target[inz] = z.height(xz[inz])
        for o in owners:
            d_o, h_o = o.project(xz, 30.0)
            hit = (d_o <= 0.0) & ~np.isfinite(target)
            target[hit] = h_o[hit]
        y = _pinned_profile(m0.s, samples, target, options)
        tags = graph.ways[int(r["ways"][len(r["ways"]) // 2])].tags
        paved = surface_class(tags) == "asphalt"
        hw_cls = _base_class(tags.get("highway", ""))
        rural = (
            tags.get("sidewalk", "no") in ("no", "none", "") and hw_cls in _RURAL_SHOULDER_CLASSES
        )
        shoulder = (
            options.shoulder_m if (rural and paved) else (0.5 if hw_cls != "service" else 0.3)
        )
        ow = is_oneway(tags)
        along = bool(r["along"][len(r["along"]) // 2])
        oneway = 0 if ow == 0 else (ow if along else -ow)
        lanes_raw = tags.get("lanes", "").split(";")[0].strip()
        lanes = int(lanes_raw) if lanes_raw.isdigit() else (1 if oneway else 2)
        centre_line = (
            paved
            and oneway == 0
            and lanes >= 2
            and (hw_cls in _EDGE_LINE_CLASSES or lanes_raw.isdigit())
        )
        m = RoadModel(
            s=m0.s,
            centre=np.stack([xz[:, 0], y, xz[:, 1]], axis=1),
            half_width=m0.half_width,
            shoulder=shoulder,
            centre_line=centre_line,
            bridge=False,
        )
        sr = SideRoad(
            index=len(sides),
            model=m,
            tags=dict(tags),
            way_ids=np.asarray(r["ways"], np.int64),
            oneway=oneway,
            paved=paved,
            shoulder=shoulder,
            start=r["start"],
            end=r["end"],
            start_node=r["start_node"],
            end_node=r["end_node"],
        )
        built[i] = sr.index
        sides.append(sr)
        owners.append(CentreLine(xz, y, m0.half_width, f"side{sr.index}"))
    net.side_roads = sides
    kept_station = np.concatenate([~hidden, [False]]) | np.concatenate([[False], ~hidden])
    pieces = [
        CentreLine(route_xz[a : b + 1], model.centre[a : b + 1, 1], model.outer[a : b + 1])
        for a, b in _runs(kept_station)
        if b > a
    ]
    height = HeightField(pieces, owners, planes)
    net.height = height

    # ---- Polygons ---------------------------------------------------------------------
    route_left = road_columns(model, model.half_width)
    route_right = road_columns(model, -model.half_width)
    left_outer = road_columns(model, model.outer)
    right_outer = road_columns(model, -model.outer)
    asphalt_q = _quads(route_left, route_right)
    sh_left_q = _quads(left_outer, route_left)
    sh_right_q = _quads(route_right, right_outer)

    carriage: list[shapely.Geometry] = []
    shoulders: list[shapely.Geometry] = []
    loop_islands: list[shapely.Geometry] = []  # turning loops' central islands
    edges: list[tuple[np.ndarray, np.ndarray]] = []
    for sr in sides:
        m = sr.model
        le, ri = road_columns(m, m.half_width), road_columns(m, -m.half_width)
        edges.append((le, ri))
        # Simplified (4 cm): built from 1 m quads, the outline has a vertex every metre.
        poly = footprint_polygon(le, ri).simplify(POLY_SIMPLIFY_M)
        if sr.end[0] == "end" and options.turning_circles:
            bulb = _bulb(sr, turning, nodes_known)
            if bulb is not None:
                poly = _union([poly, bulb])
                tc = turning.get(sr.end[1])
                if tc is not None and tc.tags.get("highway") == "turning_loop":
                    end = sr.model.centre[-1, [0, 2]]
                    loop_islands.append(shapely.Point(end).buffer(TURNING_LOOP_ISLAND_M, 16))
        carriage.append(poly)
        if sr.shoulder > 0:
            lo, ro = road_columns(m, m.outer), road_columns(m, -m.outer)
            sh = _union([footprint_polygon(lo, le), footprint_polygon(ri, ro)])
            # Grown into the carriageway, then cut by it: the two share one edge exactly.
            sh = _union([sh.simplify(POLY_SIMPLIFY_M), poly.buffer(0.06)]).difference(poly)
            if sr.end[0] == "end":
                sh = _union([sh, poly.buffer(sr.shoulder, quad_segs=6).difference(poly)])
            shoulders.append(sh)
        else:
            shoulders.append(shapely.Polygon())

    # A side road leaving the route carries on straight back through its node onto the route:
    # where the route's smoothed line cuts a corner (it turns off a road that goes on), the
    # node lies off the route's ribbon and the road it turned off would stop short of it.
    for sr in sides:
        for end, (kind, _) in (("start", sr.start), ("end", sr.end)):
            if kind != "route":
                continue
            m = sr.model
            p = m.centre[0 if end == "start" else -1, [0, 2]]
            d = m.forward[0 if end == "start" else -1, [0, 2]] * (1.0 if end == "start" else -1.0)
            d = d / max(np.linalg.norm(d), 1e-9)
            dist, i = route_tree.query(p)
            reach = float(dist + 1.0)  # to just past the route's centre line
            hw = float(m.half_width[0 if end == "start" else -1])
            ext = shapely.LineString([p - d * reach, p]).buffer(hw, cap_style="flat")
            carriage[sr.index] = _union([carriage[sr.index], ext])

    # ---- Junctions ----------------------------------------------------------------------
    by_node: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for sr in sides:
        for end, (kind, node) in (("start", sr.start), ("end", sr.end)):
            if kind in ("route", "road"):
                by_node[node].append((sr.index, end))
    junctions: list[Junction] = []
    for node in sorted(by_node):
        legs = by_node[node]
        xz = graph.coords[node]
        on_route = node in graph.route_nodes
        if not on_route and len(legs) < 2:
            continue
        rs = None
        if on_route:
            _, i = route_tree.query(xz)
            rs = float(model.s[i])
        junctions.append(Junction(node, xz, rs, legs))
    hidden_seg = masks.hidden
    for j in junctions:
        disc = shapely.Point(j.xz).buffer(40.0, quad_segs=16)
        parts = [carriage[k].intersection(disc) for k, _ in j.legs]
        if j.route_s is not None:
            i = model.index_at(j.route_s)
            lo, hi = max(0, i - 50), min(n_st - 1, i + 50)
            local = asphalt_q[lo:hi][~hidden_seg[lo:hi]]
            route_local = _union(list(local)) if len(local) else shapely.Polygon()
            radii = [corner_radius(route_highway[i], sides[k].highway) for k, _ in j.legs]
        else:
            route_local = shapely.Polygon()
            hws = [sides[k].highway for k, _ in j.legs]
            radii = [corner_radius(a, b) for a in hws for b in hws if a is not b] or [9.0]
        j.radius = max(radii)
        u = _union([*parts, route_local])
        if u.is_empty:
            continue
        closed = u.buffer(j.radius, quad_segs=12).buffer(-j.radius, quad_segs=12)
        fill = closed.difference(u).intersection(disc)
        legs = _legs(j, sides, carriage, model, route_highway, route_oneway, asphalt_q, hidden_seg)
        outside = _outside_sector(legs)  # the through road's far side: no curb return there
        keep = [
            p
            for p in _polygons(fill, 0.05)
            if any(p.distance(q) < 0.05 for q in parts if not q.is_empty)
            and not (outside is not None and _in_sector(p.centroid.coords[0], j.xz, outside))
        ]
        j.fillets = _union(keep)
        sh = max([sides[k].shoulder for k, _ in j.legs] + [0.0])
        if sh > 0 and not j.fillets.is_empty:  # gravel round the corner, like the roads
            shoulders.append(j.fillets.buffer(sh, quad_segs=8).difference(u))
    net.junctions = junctions

    # ---- Roundabouts ----------------------------------------------------------------------
    rb_parts = [
        rbm.build_roundabout(
            rb,
            model=model,
            hidden=hidden_seg,
            route_quads=(asphalt_q, sh_left_q, sh_right_q),
            sides=sides,
            carriage=carriage,
            graph=graph,
            height=height,
            driving_side=driving_side,
            route_oneway=route_oneway,
        )
        for rb in rbs
    ]

    # ---- What replaces the route's shoulders and lines ------------------------------------
    islands = [isl for p in rb_parts for isl, _ in p.islands] + loop_islands
    islands_all = _union(islands)
    paved_raw = _union(
        [*carriage, *(j.fillets for j in junctions), *(p.pavement for p in rb_parts)]
    )
    cover = _union([paved_raw, islands_all])
    near = np.zeros(n_st - 1, bool)
    if not cover.is_empty:
        mids = 0.5 * (route_xz[:-1] + route_xz[1:])
        near = shapely.dwithin(cover, shapely.points(mids), float(model.outer.max()) + 3.0)
    removed = {}
    for side, quads in (("left", sh_left_q), ("right", sh_right_q)):
        gone = np.zeros(n_st - 1, bool)
        idx = np.flatnonzero(near & ~hidden_seg)
        if len(idx):
            area = shapely.area(shapely.intersection(quads[idx], cover))
            gone[idx] = area > 0.05 * np.maximum(shapely.area(quads[idx]), 1e-9)
        removed[side] = gone
    masks.shoulder_left = ~(removed["left"] | hidden_seg)
    masks.shoulder_right = ~(removed["right"] | hidden_seg)
    grow = np.ones(3, bool)
    masks.edge_left = ~(np.convolve(removed["left"], grow, "same") > 0) & ~hidden_seg
    masks.edge_right = ~(np.convolve(removed["right"], grow, "same") > 0) & ~hidden_seg
    masks.centre = ~hidden_seg.copy()

    # ---- Final surfaces ---------------------------------------------------------------------
    near_idx = np.flatnonzero(near & ~hidden_seg)
    route_kept = _union(list(asphalt_q[near_idx]))
    kept_sh = [
        q
        for q, k in zip(
            np.concatenate([sh_left_q[near_idx], sh_right_q[near_idx]]),
            np.concatenate([masks.shoulder_left[near_idx], masks.shoulder_right[near_idx]]),
            strict=True,
        )
        if k
    ]
    route_sh_kept = _union(kept_sh)
    gone_sh = [
        q
        for q, g in zip(
            np.concatenate([sh_left_q[near_idx], sh_right_q[near_idx]]),
            np.concatenate([removed["left"][near_idx], removed["right"][near_idx]]),
            strict=True,
        )
        if g
    ]
    blockers = _union([route_kept, route_sh_kept, islands_all])
    paved_all = _union([paved_raw, *gone_sh]).difference(blockers)
    unpaved = _union([c for c, sr in zip(carriage, sides, strict=True) if not sr.paved])
    paved_hard = _union(
        [
            *(c for c, sr in zip(carriage, sides, strict=True) if sr.paved),
            *(j.fillets for j in junctions),
            *(p.pavement for p in rb_parts),
            *gone_sh,
        ]
    )
    gravel_road = unpaved.difference(paved_hard).intersection(paved_all)
    asphalt = paved_all.difference(gravel_road)
    shoulder_all = _union(shoulders).difference(_union([paved_all, blockers]))
    # Acute notches where the network meets the route (or itself) are filled with gravel: the
    # terrain's triangulation cannot follow a boundary into a sliver that thin.
    route_foot_near = _union(
        list(asphalt_q[near_idx]) + list(sh_left_q[near_idx]) + list(sh_right_q[near_idx])
    )
    whole = _union([route_foot_near, paved_all, shoulder_all, islands_all])
    notch = whole.buffer(NOTCH_FILL_M, quad_segs=4).buffer(-NOTCH_FILL_M, quad_segs=4)
    network_area = _union([paved_all, shoulder_all])
    fill = [
        p
        for p in _polygons(notch.difference(whole), 0.02)
        if p.area <= NOTCH_MAX_M2 and p.distance(network_area) < 0.01
    ]
    shoulder_all = _union([shoulder_all, *fill])
    net.paved = _union([paved_all, islands_all])
    net.islands = islands
    net.polygons = {
        "asphalt": asphalt,
        "gravel": _union([gravel_road, shoulder_all]),
        "islands": islands_all,
    }

    batch = Batcher(options.batch_m)
    meshes: list[MeshData] = []

    def add(
        key: str,
        geom: shapely.Geometry,
        lift: float = 0.0,
        tile: float = 4.0,
        cell: float | None = None,
        seg: float = 1.5,
    ) -> None:
        for poly in _polygons(geom, 0.01):
            mesh = mesh_polygon(
                poly, height, cell_m=cell or options.cell_m, uv_tile_m=tile, lift=lift, seg_m=seg
            )
            if mesh is not None:
                batch.add(key, mesh)
                if lift == 0.0:
                    meshes.append(mesh)

    add("asphalt", asphalt)
    add("gravel", gravel_road, tile=2.0)
    add("gravel", shoulder_all, tile=2.0, cell=6.0, seg=4.0)  # thin strips: few vertices
    for p in rb_parts:
        for key, mesh in p.meshes:
            batch.add(key, mesh)
            meshes.append(mesh)
    for isl in loop_islands:
        for key, mesh in rbm.island_meshes(isl, height, options.cell_m):
            batch.add(key, mesh)
            meshes.append(mesh)

    # ---- Junction control, stop bars, lines -------------------------------------------------
    controls = [
        nd for nd in nodes if nd.tags.get("highway") in ("stop", "give_way", "traffic_signals")
    ]
    items: list[PlannedItem] = []
    stop_paint: list[shapely.Geometry] = []
    side_cut: dict[tuple[int, str], float] = {}  # centre line ends at the stop bar
    centre_off: list[tuple[float, float]] = []
    for j in junctions:
        legs = _legs(j, sides, carriage, model, route_highway, route_oneway, asphalt_q, hidden_seg)
        if len(legs) < 2:
            continue
        _control(
            j,
            legs,
            controls,
            sides,
            model,
            carriage,
            route_kept,
            items,
            stop_paint,
            side_cut,
            centre_off,
            driving_side,
            options,
        )
    others_for_lines = _union(
        [
            route_kept,
            route_sh_kept,
            *(j.fillets for j in junctions),
            islands_all,
            *(p.pavement for p in rb_parts),
        ]
    ).buffer(0.3)
    shapely.prepare(others_for_lines)
    for j in junctions:  # the through road's centre line carries on across the junction
        legs = _legs(j, sides, carriage, model, route_highway, route_oneway, asphalt_q, hidden_seg)
        line = _crossing_line(j, legs, sides, model, others_for_lines, centre_off)
        if line is not None:
            g = CENTRE_GAP_M / 2 + CENTRE_LINE_WIDTH_M / 2
            for off in (g, -g):
                le = _offset_line(line, off + CENTRE_LINE_WIDTH_M / 2)
                ri = _offset_line(line, off - CENTRE_LINE_WIDTH_M / 2)
                keep = np.ones(len(line) - 1, bool)
                batch.add("paint_yellow", strip_mesh(le, ri, keep, height))
    if centre_off:
        masks.centre &= ~_mask_by_s(model.s, centre_off)
    for p in rb_parts:
        items += p.items
        stop_paint += [g for k, g in p.paint if k == "paint_white"]
    for g in stop_paint:
        add("paint_white", g, lift=PAINT_LIFT_M + 0.004, tile=1.0)
    for p in rb_parts:
        for key, g in p.paint:
            if key != "paint_white":
                add(key, g, lift=PAINT_LIFT_M + 0.004, tile=1.0)
    for sr, poly in zip(sides, carriage, strict=True):
        _side_lines(sr, poly, carriage, others_for_lines, side_cut, height, batch)

    # ---- Terrain and walls -----------------------------------------------------------------
    foot = _union([net.paved, shoulder_all])
    net.footprint = foot
    if meshes:
        bp = np.concatenate([boundary_points(mm) for mm in meshes])
        if len(bp):
            near_route = np.flatnonzero(near & ~hidden_seg)
            route_foot = (
                footprint_polygon(left_outer, right_outer, _seg_mask(near_route, n_st - 1))
                if len(near_route)
                else shapely.Polygon()
            )
            inner = _union([foot, route_foot]).buffer(-0.05)
            out = ~shapely.contains_xy(inner, bp[:, 0], bp[:, 2])
            bp = bp[out]
            _, first = np.unique(np.round(bp[:, [0, 2]], 4), axis=0, return_index=True)
            net.edge_points = bp[np.sort(first)]
    net.island_walls = [w for p in rb_parts for w in p.walls] + [
        np.asarray(shapely.segmentize(isl.buffer(-0.5).exterior, 2.0).coords)
        for isl in loop_islands
    ]
    net.pieces = batch.pieces()
    net.planned_items = items
    net.carriage = carriage
    net.route_columns = (route_left, route_right, left_outer, right_outer)
    net.junction_xz = np.array([j.xz for j in junctions]).reshape(-1, 2)
    net.driveway_ways = list(graph.driveways)
    net.stats = {
        "side_roads": len(sides),
        "side_road_km": round(sum(sr.length for sr in sides) / 1000.0, 3),
        "junctions": len(junctions),
        "roundabouts": len(rbs),
        "items": len(items),
        "stop_bars": sum(1 for it in items if it.kind == SIGN_STOP),
        "turning_circles": sum(1 for sr in sides if sr.end[0] == "end" and sr.end[1] in turning),
        "verge_m": options.verge_m,
    }
    return net


def _seg_mask(idx: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    m[idx] = True
    return m


def _bulb(
    sr: SideRoad, turning: Mapping[int, NetNode], nodes_known: bool
) -> shapely.Geometry | None:
    """A cul-de-sac bulb at a side road's dead end: an OSM turning circle (radius from its
    `diameter` tag, else `TURNING_CIRCLE_RADIUS_M`), or, where OSM nodes are unknown, at a
    residential dead end."""
    node = sr.end[1]
    tc = turning.get(node)
    if tc is None and (nodes_known or sr.highway not in ("residential", "living_street")):
        return None
    r = TURNING_CIRCLE_RADIUS_M
    if tc is not None:
        d = parse_width(tc.tags.get("diameter"))
        if d is not None and d > 0:
            r = float(np.clip(d / 2.0, 9.0, 15.0))
        if tc.tags.get("highway") == "turning_loop":
            r = max(r, 15.0)
    end = sr.model.centre[-1, [0, 2]]
    fwd = sr.model.forward[-1, [0, 2]]
    fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
    # OSM puts the turning circle node in the middle of the bulb: centre the disc on it and
    # round the corners where the road widens into it (radius `rr`).
    disc = shapely.Point(end).buffer(r, quad_segs=16)
    hw = float(sr.model.half_width[-1])
    rr = 8.0
    road = shapely.LineString([end - fwd * (r + rr + 5.0), end]).buffer(hw, cap_style="flat")
    closed = disc.union(road).buffer(rr, quad_segs=8).buffer(-rr, quad_segs=8)
    return closed.difference(road.difference(disc))


def _legs(
    j: Junction,
    sides: Sequence[SideRoad],
    carriage: Sequence[shapely.Geometry],
    model: RoadModel,
    route_highway: Sequence[str],
    route_oneway: np.ndarray,
    asphalt_q: np.ndarray,
    hidden: np.ndarray,
) -> list[_Leg]:
    legs: list[_Leg] = []
    if j.route_s is not None:
        i = model.index_at(j.route_s)
        fwd = model.forward[i, [0, 2]]
        fwd = fwd / np.linalg.norm(fwd)
        i_in = model.index_at(j.route_s - 15.0)
        i_out = model.index_at(j.route_s + 15.0)
        lo, hi = max(0, i - 40), min(len(model) - 1, i + 40)
        local = asphalt_q[lo:hi][~hidden[lo:hi]]
        poly = _union(list(local))
        ow_in, ow_out = int(route_oneway[i_in]), int(route_oneway[i_out])
        if i > 2:
            legs.append(_Leg(-fwd, route_highway[i_in], "route_in", approach=ow_in >= 0, poly=poly))
        if i < len(model) - 3:
            legs.append(
                _Leg(fwd, route_highway[i_out], "route_out", approach=ow_out <= 0, poly=poly)
            )
    for k, end in j.legs:
        sr = sides[k]
        m = sr.model
        if end == "start":
            t = min(8.0, sr.length)
            p = np.asarray(m.at(t)[0])[[0, 2]]
            approach = sr.oneway <= 0
        else:
            t = max(0.0, sr.length - 8.0)
            p = np.asarray(m.at(t)[0])[[0, 2]]
            approach = sr.oneway >= 0
        d = p - j.xz
        d = d / max(np.linalg.norm(d), 1e-9)
        legs.append(_Leg(d, sr.highway, "side", k, end, approach, carriage[k]))
    return legs


def _through(legs: Sequence[_Leg]) -> tuple[int, int] | None:
    """The pair of legs that forms the through road: highest class, then straightest."""
    best = None
    for a in range(len(legs)):
        for b in range(a + 1, len(legs)):
            cos = float(legs[a].direction @ legs[b].direction)
            if cos > -0.5:  # not a straight-ish pair
                continue
            rank = min(highway_rank(legs[a].highway), highway_rank(legs[b].highway))
            key = (rank, -cos)
            if best is None or key > best[0]:
                best = (key, (a, b))
    return None if best is None else best[1]


def _outside_sector(legs: Sequence[_Leg]) -> tuple[float, float] | None:
    """(start, end) angles of the sector between the two through legs that holds no other
    leg (the outside of the through road), or None (no through road, or other legs on both
    sides as at a crossroads)."""
    through = _through(legs)
    if through is None:
        return None
    ang = [math.atan2(lg.direction[1], lg.direction[0]) for lg in legs]
    a, b = (ang[n] for n in through)
    others = [ang[n] for n in range(len(legs)) if n not in through]
    for lo, hi in ((a, b), (b, a)):
        span = (hi - lo) % (2 * math.pi)
        if not any(0.0 < (o - lo) % (2 * math.pi) < span for o in others):
            return (lo, hi)
    return None


def _in_sector(p, centre: np.ndarray, sector: tuple[float, float]) -> bool:
    lo, hi = sector
    t = math.atan2(p[1] - centre[1], p[0] - centre[0])
    return 0.0 < (t - lo) % (2 * math.pi) < (hi - lo) % (2 * math.pi)


def _lane_quad(m: RoadModel, s: float, lo: float, hi: float, width: float) -> shapely.Geometry:
    """A painted bar across lateral offsets [lo, hi] (+ = left) at station `s`, `width` long."""
    p0, _, l0 = m.at(s - width / 2)
    p1, _, l1 = m.at(s + width / 2)
    pts = [p0 + l0 * lo, p1 + l1 * lo, p1 + l1 * hi, p0 + l0 * hi]
    return shapely.Polygon([(q[0], q[2]) for q in pts]).buffer(0)


def _clear_distance(
    m: RoadModel,
    s0: float,
    direction: float,
    offsets: tuple[float, float],
    obstacle: shapely.Geometry,
    limit: float,
) -> float:
    """How far from `s0` (stepping by `direction` = +-1) both lane edge points at `offsets`
    are clear of `obstacle` (the crossing road)."""
    if obstacle is None or obstacle.is_empty:
        return 0.0
    for d in np.arange(0.0, limit, 0.5):
        s = s0 + direction * d
        if s < m.s[0] or s > m.s[-1]:
            return d
        p, _, left = m.at(s)
        pts = [p + left * o for o in offsets]
        if not any(obstacle.contains(shapely.Point(q[0], q[2])) for q in pts):
            return float(d)
    return limit


def _control(
    j: Junction,
    legs: list[_Leg],
    controls: Sequence[NetNode],
    sides: Sequence[SideRoad],
    model: RoadModel,
    carriage: Sequence[shapely.Geometry],
    route_kept: shapely.Geometry,
    items: list[PlannedItem],
    paint: list[shapely.Geometry],
    side_cut: dict[tuple[int, str], float],
    centre_off: list[tuple[float, float]],
    driving_side: str,
    options: NetworkOptions,
) -> None:
    """Junction control for every approach (`approach_stops`): stop bars and stop signs."""
    right = driving_side == "right"
    mapped: dict[int, str] = {}
    for c in controls:
        v = c.xz - j.xz
        dist = float(np.linalg.norm(v))
        hw = c.tags.get("highway")
        at_node = dist < 3.0
        if dist > 40.0:
            continue
        kind = {"stop": "stop", "give_way": "yield", "traffic_signals": "signal"}[hw]
        if hw == "stop" and c.tags.get("stop") == "all":
            kind = "all_way"
        if at_node:
            if kind in ("signal", "all_way"):
                for k in range(len(legs)):
                    mapped.setdefault(k, kind)
            continue
        cos = [float(lg.direction @ (v / dist)) for lg in legs]
        k = int(np.argmax(cos))
        if cos[k] > 0.85:
            mapped[k] = kind
    through = _through(legs)
    route_through = through is not None and {legs[through[0]].kind, legs[through[1]].kind} == {
        "route_in",
        "route_out",
    }
    partner = {} if through is None else {through[0]: through[1], through[1]: through[0]}
    hw_route = float(model.half_width[model.index_at(j.route_s)]) if j.route_s is not None else 0.0
    for k, leg in enumerate(legs):
        # A leg never gives way to the rest of its own road (its through partner), so a class
        # change at the junction (primary to secondary) puts no stop on the road itself.
        others = [lg.highway for n, lg in enumerate(legs) if n != k and n != partner.get(k)]
        ctrl = approach_stops(leg.highway, others, mapped=mapped.get(k))
        if k not in mapped and highway_rank(leg.highway) <= highway_rank("service"):
            ctrl = None  # private and commercial entrances get no rule-made signs
        j.controls.append((leg.kind, leg.highway, ctrl))
        if ctrl not in ("stop", "yield") or not leg.approach:
            continue
        # The road crossed: the other legs' carriageways (the route's legs as straight bands
        # from the node, as the route ribbon also turns into them).
        crossing = _union(
            [
                lg.poly
                if lg.kind == "side"
                else shapely.LineString([j.xz, j.xz + lg.direction * 40.0]).buffer(
                    hw_route, cap_style="flat"
                )
                for n, lg in enumerate(legs)
                if n != k
            ]
            + ([route_kept] if leg.kind == "side" else [])
        )
        if leg.kind == "side":
            sr = sides[leg.road]
            m = sr.model
            hw = float(m.half_width[0 if leg.end == "start" else -1])
            if leg.end == "start":  # traffic moves towards s = 0
                lane = (0.0, hw) if sr.oneway == 0 else (-hw, hw)
                if not right and sr.oneway == 0:
                    lane = (-hw, 0.0)
                d = _clear_distance(m, m.s[0], 1.0, lane, crossing, min(40.0, sr.length - 1.0))
                s_bar = m.s[0] + d + STOP_BAR_SETBACK_M
                if s_bar + STOP_BAR_WIDTH_M > m.s[-1]:
                    continue
                side_cut[(leg.road, "start")] = s_bar
                sign_s = s_bar + STOP_SIGN_BEHIND_M
                sgn = 1.0 if right else -1.0
            else:  # traffic moves towards the end
                lane = (-hw, 0.0) if sr.oneway == 0 else (-hw, hw)
                if not right and sr.oneway == 0:
                    lane = (0.0, hw)
                d = _clear_distance(m, m.s[-1], -1.0, lane, crossing, min(40.0, sr.length - 1.0))
                s_bar = m.s[-1] - d - STOP_BAR_SETBACK_M
                if s_bar - STOP_BAR_WIDTH_M < m.s[0]:
                    continue
                side_cut[(leg.road, "end")] = s_bar
                sign_s = s_bar - STOP_SIGN_BEHIND_M
                sgn = -1.0 if right else 1.0
            lo, hi = lane
            if sr.oneway == 0 and sr.model.centre_line[0]:  # keep clear of the yellow lines
                lo, hi = (max(lo, 0.2), hi) if lo >= 0 else (lo, min(hi, -0.2))
            if options.stop_bars:
                paint.append(_lane_quad(m, s_bar, lo, hi, STOP_BAR_WIDTH_M))
            p, f, left = m.at(sign_s)
            hw_s = float(np.interp(sign_s, m.s, m.half_width))
            pos = p + left * sgn * (hw_s + sr.shoulder + SIGN_OFFSET_M)
            face = f if leg.end == "start" else -f
        else:
            m = model
            i = m.index_at(j.route_s)
            hw = float(m.half_width[i])
            if leg.kind == "route_in":  # traffic moves +s towards the junction
                lane = (-hw, 0.0) if right else (0.0, hw)
                d = _clear_distance(m, j.route_s, -1.0, lane, crossing, 40.0)
                s_bar = j.route_s - d - STOP_BAR_SETBACK_M
                sign_s = s_bar - STOP_SIGN_BEHIND_M
                sgn = -1.0 if right else 1.0
                face_dir = -1.0
            else:  # opposing traffic moves -s towards the junction
                lane = (0.0, hw) if right else (-hw, 0.0)
                d = _clear_distance(m, j.route_s, 1.0, lane, crossing, 40.0)
                s_bar = j.route_s + d + STOP_BAR_SETBACK_M
                sign_s = s_bar + STOP_SIGN_BEHIND_M
                sgn = 1.0 if right else -1.0
                face_dir = 1.0
            if s_bar < m.s[0] or s_bar > m.s[-1]:
                continue
            lo, hi = lane
            lo, hi = (max(lo, 0.25), hi) if lo >= 0 else (lo, min(hi, -0.25))
            if options.stop_bars:
                paint.append(_lane_quad(m, s_bar, lo, hi, STOP_BAR_WIDTH_M))
            p, f, left = m.at(sign_s)
            ii = m.index_at(sign_s)
            pos = p + left * sgn * (float(m.half_width[ii]) + float(m.shoulder[ii]) + SIGN_OFFSET_M)
            face = f * face_dir
            if not route_through:  # the route's centre line stops at its stop bars
                centre_off.append((min(s_bar, j.route_s), max(s_bar, j.route_s)))
        kind = SIGN_STOP if ctrl == "stop" else SIGN_YIELD
        fxz = np.array([face[0], face[2]])
        fxz = fxz / max(np.linalg.norm(fxz), 1e-9)
        items.append(
            PlannedItem(
                kind,
                (float(pos[0]), float(pos[2])),
                (float(fxz[0]), float(fxz[1])),
                "osm" if k in mapped else "rule",
                ref=f"junction:{j.node}",
            )
        )
    if j.route_s is not None and not route_through and through is not None:
        # The route crosses or joins a bigger road: no centre line across its carriageway.
        cross = _union([legs[n].poly for n in through if legs[n].kind == "side"])
        if not cross.is_empty:
            i = model.index_at(j.route_s)
            lo, hi = max(0, i - 40), min(len(model) - 1, i + 40)
            pts = shapely.points(model.centre[lo:hi, [0, 2]])
            inside = shapely.dwithin(cross, pts, 1.0)
            if inside.any():
                k = np.flatnonzero(inside)
                centre_off.append((float(model.s[lo + k[0]]), float(model.s[lo + k[-1]])))


def _offset_line(xz: np.ndarray, off: float) -> np.ndarray:
    """A polyline (K, 2) moved `off` metres to its left (+) or right (-), in AC XZ."""
    t = np.gradient(xz, axis=0)
    t = t / np.maximum(np.linalg.norm(t, axis=1), 1e-9)[:, None]
    left = np.stack([t[:, 1], -t[:, 0]], axis=1)  # up x forward, with X east and Z south
    return xz + off * left


def _crossing_line(
    j: Junction,
    legs: Sequence[_Leg],
    sides: Sequence[SideRoad],
    model: RoadModel,
    block: shapely.Geometry,
    centre_off: list[tuple[float, float]],
) -> np.ndarray | None:
    """Where a side road is the through road (the route is the stem, crosses it or turns
    onto it), its centre line across the junction: a curve (K, 2) joining where its two legs'
    own centre lines stop. The route's centre line is cut back across that road."""
    through = _through(legs)
    if through is None:
        return None
    a, b = (legs[n] for n in through)
    if a.kind != "side" and b.kind != "side":
        return None  # the route is the through road: its own line carries on
    ends = []
    i = model.index_at(j.route_s) if j.route_s is not None else 0
    reach = max((float(sides[lg.road].model.half_width[0]) for lg in legs if lg.kind == "side"),
                default=3.0) + float(model.shoulder[i]) + 1.5  # fmt: skip
    for lg in (a, b):
        if lg.kind == "side":
            sr = sides[lg.road]
            m = sr.model
            if sr.oneway != 0 or not m.centre_line[0]:
                return None
            ss = np.arange(0.0, min(30.0, sr.length), 0.5)
            ss = ss if lg.end == "start" else m.s[-1] - ss
            pts = np.array([np.asarray(m.at(float(s))[0])[[0, 2]] for s in ss])
            free = ~shapely.contains_xy(block, pts[:, 0], pts[:, 1])
            if not free.any():
                return None
            ends.append(pts[int(np.argmax(free))])
        else:
            if j.route_s is None or not model.centre_line[i]:
                return None
            s_end = j.route_s + (reach if lg.kind == "route_out" else -reach)
            s_end = float(np.clip(s_end, model.s[0], model.s[-1]))
            ends.append(np.asarray(model.at(s_end)[0])[[0, 2]])
            centre_off.append((min(s_end, j.route_s), max(s_end, j.route_s)))
    if j.route_s is not None:  # the route's own line also stops short of the through road
        centre_off.append((j.route_s - reach, j.route_s + reach))
    p0, p1 = ends
    if np.linalg.norm(p1 - p0) < 1.0:
        return None
    # A quadratic curve through the junction node's side of the chord: straight for a
    # straight road, bending with it otherwise.
    mid = 0.5 * (p0 + p1)
    ctrl = mid + 0.5 * (j.xz - mid) if np.linalg.norm(j.xz - mid) > 0.5 else mid
    t = np.linspace(0.0, 1.0, max(3, int(np.linalg.norm(p1 - p0) / 1.0) + 1))[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t**2 * p1


def _side_lines(
    sr: SideRoad,
    own: shapely.Geometry,
    carriage: Sequence[shapely.Geometry],
    others: shapely.Geometry,
    side_cut: Mapping[tuple[int, str], float],
    height: HeightField,
    batch: Batcher,
) -> None:
    """Centre and edge lines of one side road, left out where another road or a junction
    takes over and before its stop bars."""
    m = sr.model
    if not sr.paved or len(m) < 2:
        return
    hw = m.half_width
    cls = _base_class(sr.highway)
    rural = sr.tags.get("sidewalk", "no") in ("no", "none", "")
    lines: list[tuple[str, np.ndarray, float]] = []
    if sr.oneway == 0 and m.centre_line[0]:
        g = CENTRE_GAP_M / 2 + CENTRE_LINE_WIDTH_M / 2
        lines += [
            ("paint_yellow", np.full(len(m), g), CENTRE_LINE_WIDTH_M),
            ("paint_yellow", np.full(len(m), -g), CENTRE_LINE_WIDTH_M),
        ]
    if cls in _EDGE_LINE_CLASSES and rural:
        inset = hw - EDGE_LINE_INSET_M
        if sr.oneway == 0:
            lines += [
                ("paint_white", inset, EDGE_LINE_WIDTH_M),
                ("paint_white", -inset, EDGE_LINE_WIDTH_M),
            ]
        else:  # one-way: yellow on the drivers' left, white on their right
            left_key = "paint_yellow" if sr.oneway > 0 else "paint_white"
            right_key = "paint_white" if sr.oneway > 0 else "paint_yellow"
            lines += [(left_key, inset, EDGE_LINE_WIDTH_M), (right_key, -inset, EDGE_LINE_WIDTH_M)]
    if not lines:
        return
    other_roads = _union([c for k, c in enumerate(carriage) if k != sr.index])
    block = _union([others, other_roads.buffer(0.3)])
    shapely.prepare(block)
    s = m.s
    seg_ok = np.ones(len(m) - 1, bool)
    a, b = side_cut.get((sr.index, "start")), side_cut.get((sr.index, "end"))
    mid = 0.5 * (s[:-1] + s[1:])
    for key, off, width in lines:
        le = road_columns(m, off + width / 2)[:, [0, 2]]
        ri = road_columns(m, off - width / 2)[:, [0, 2]]
        c = 0.5 * (le + ri)
        ok = ~shapely.contains_xy(block, c[:, 0], c[:, 1])
        keep = seg_ok & ok[:-1] & ok[1:]
        if key == "paint_yellow" and abs(float(np.mean(off))) < 0.5:  # centre lines
            if a is not None:
                keep &= mid > a
            if b is not None:
                keep &= mid < b
        batch.add(key, strip_mesh(le, ri, keep, height))


# --------------------------------------------------------------------------------------------
# Driveways (after the buildings are placed)
# --------------------------------------------------------------------------------------------

# Ontario driveways (streetscape research, Hamilton RD-106 / RD-113, Oakville, Burlington
# zoning): rural entrances 3.6-6.0 m of gravel (or asphalt), flared to the road edge with R 3 m
# corners, straight to the house and perpendicular to the road for the last 15 m; subdivision
# driveways perpendicular to the road, constant width (3.0-6.0 m, 5.5 m for a double garage),
# no flares, asphalt.
DRIVEWAY_RURAL_WIDTH_M = 4.5
DRIVEWAY_URBAN_WIDTH_M = 5.5
DRIVEWAY_FLARE_M = 3.0
DRIVEWAY_PERPENDICULAR_M = 15.0
DRIVEWAY_MAX_M = 150.0
DRIVEWAY_MIN_M = 3.0
DRIVEWAY_LIFT_M = 0.06  # above the terrain (draped on a 3 m grid; the terrain has no hole)
DRIVEWAY_SHOULDER_LIFT_M = 0.015  # above a road shoulder it lies on
DRIVEWAY_JUNCTION_CLEAR_M = 15.0  # no inferred driveway mouth this close to a junction
DRIVEWAY_SERVED_M = 12.0  # a house with a driveway this close already has one
MAILBOX_SETBACK_M = 0.6  # beyond the shoulder (rural route mailbox on a post)
MAILBOX_SIDE_M = 1.0  # beside the driveway edge
HOUSE_KINDS = frozenset(
    "house detached residential semidetached_house bungalow yes farm cabin terrace".split()
)
HOUSE_AREA_M2 = (45.0, 800.0)
SUBDIVISION_NEIGHBOURS = 5  # this many houses within 60 m: a subdivision, not a rural lot


@dataclass
class Driveway:
    polygon: shapely.Geometry
    path: np.ndarray  # (K, 2) from the road edge to the house
    source: str  # "osm" or "inferred"
    rural: bool
    road: int  # -1: the route, else a side road index
    mouth: np.ndarray  # (2,) on the road's asphalt edge
    normal: np.ndarray  # (2,) away from the road


@dataclass
class DrivewayResult:
    driveways: list[Driveway] = field(default_factory=list)
    pieces: list[ScenePiece] = field(default_factory=list)
    items: list[PlannedItem] = field(default_factory=list)
    masks_changed: bool = False

    @property
    def stats(self) -> dict[str, int]:
        return {
            "driveways_osm": sum(1 for d in self.driveways if d.source == "osm"),
            "driveways_inferred": sum(1 for d in self.driveways if d.source == "inferred"),
            "mailboxes": len(self.items),
        }


def _roads_for_driveways(net: RoadNetwork) -> list[tuple[int, CentreLine, float]]:
    """(road index (-1 route), centre line with asphalt half widths, shoulder) per road."""
    m = net.model
    hidden = net.masks.hidden
    kept = np.concatenate([~hidden, [False]]) | np.concatenate([[False], ~hidden])
    out = []
    for a, b in _runs(kept):
        if b > a:
            cl = CentreLine(m.centre[a : b + 1, [0, 2]], m.centre[a : b + 1, 1],
                            m.half_width[a : b + 1])  # fmt: skip
            out.append((-1, cl, float(np.median(m.shoulder[a : b + 1]))))
    for sr in net.side_roads:
        if highway_rank(sr.highway) > highway_rank("service") or sr.highway == "service":
            out.append((sr.index, CentreLine(sr.xz, sr.model.centre[:, 1], sr.model.half_width),
                        sr.shoulder))  # fmt: skip
    return out


def _nearest_road(
    roads: Sequence[tuple[int, CentreLine, float]], p: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float, float] | None:
    """(road, edge point, tangent, normal away from the road, edge distance, shoulder) of the
    road whose asphalt edge is nearest to the point."""
    best = None
    for idx, cl, shoulder in roads:
        if not cl.near(p[None], 200.0).any():
            continue
        foot, tan, d, hw = cl.nearest(p[None])
        edge = float(d[0] - hw[0])
        if best is None or edge < best[4]:
            n = np.array([tan[0, 1], -tan[0, 0]])
            if n @ (p - foot[0]) < 0:
                n = -n
            best = (idx, foot[0] + n * hw[0], tan[0], n, edge, shoulder)
    if best is None:
        return None
    return best


def build_driveways(
    net: RoadNetwork,
    footprints: Sequence[tuple[shapely.Geometry, str]],
    ground: Callable[[np.ndarray], np.ndarray],
    options: NetworkOptions | None = None,
    *,
    infer: bool = True,
) -> DrivewayResult:
    """OSM driveways and one inferred driveway per detached house without one (`footprints`:
    the placed buildings' (polygon, building kind) in AC XZ), draped on the terrain
    (`ground(xz)`). Their mouths cut the route's edge line and replace the route shoulder
    they cover (`net.masks` is updated: rebuild the route meshes when `masks_changed`); rural
    driveways get a mailbox on the right of the mouth (seen from the road)."""
    options = options or NetworkOptions()
    res = DrivewayResult()
    if not net.route_columns:
        return res
    roads = _roads_for_driveways(net)
    if not roads:
        return res
    route_xz = net.model.centre[:, [0, 2]]
    route_tree = cKDTree(route_xz)
    blocked = _union(
        [*net.islands]
        + [shapely.Point(rb.centre).buffer(rb.zone_r + 10.0) for rb in net.roundabouts]
    )
    junction_tree = cKDTree(net.junction_xz) if len(net.junction_xz) else None
    houses = [
        (poly, kind)
        for poly, kind in footprints
        if kind in HOUSE_KINDS and HOUSE_AREA_M2[0] <= poly.area <= HOUSE_AREA_M2[1]
    ]
    all_fp = [poly for poly, _ in footprints]
    fp_tree = shapely.STRtree(all_fp) if all_fp else None
    centroids = np.array([p.centroid.coords[0] for p, _ in houses]).reshape(-1, 2)
    house_tree = cKDTree(centroids) if len(centroids) else None
    cls = {sr.index: sr.highway for sr in net.side_roads}
    cls[-1] = "secondary"
    drives: list[Driveway] = []

    def rural_at(p: np.ndarray, road: int) -> bool:
        if cls.get(road, "") in ("residential", "living_street"):
            return False
        if house_tree is None:
            return True
        return len(house_tree.query_ball_point(p, 60.0)) < SUBDIVISION_NEIGHBOURS

    # OSM driveways: oriented from the end nearest a road.
    for way in net.driveway_ways:
        xz = np.asarray(way.xz, np.float64)
        if len(xz) < 2 or route_tree.query(xz)[0].min() > options.radius_m:
            continue
        ends = [_nearest_road(roads, xz[0]), _nearest_road(roads, xz[-1])]
        k = 0 if (ends[1] is None or (ends[0] is not None and ends[0][4] <= ends[1][4])) else 1
        near = ends[k]
        if near is None or near[4] > 8.0:
            continue
        path = xz if k == 0 else xz[::-1]
        road, edge_pt, _, n, _, _ = near
        keep = np.linalg.norm(path - edge_pt, axis=1) > 0.5
        path = np.vstack([edge_pt - n * 0.3, path[keep]])
        rural = rural_at(path[-1], road)
        w = parse_width(way.tags.get("width")) or (
            DRIVEWAY_RURAL_WIDTH_M if rural else DRIVEWAY_URBAN_WIDTH_M
        )
        drives.append(Driveway(shapely.LineString(path), path, "osm", rural, road, edge_pt, n))
        drives[-1].polygon = _driveway_polygon(path, w, rural, near, None)

    served = _union([d.polygon for d in drives]).buffer(DRIVEWAY_SERVED_M) if drives else None
    if infer:
        order = np.argsort([route_tree.query(c)[0] for c in centroids]) if len(houses) else []
        for h in order:
            poly, _ = houses[int(h)]
            if route_tree.query(centroids[h])[0] > options.radius_m:
                continue
            if served is not None and served.intersects(poly):
                continue
            near = _nearest_road(roads, centroids[h])
            if near is None:
                continue
            road, edge_pt, tan, n, _, _ = near
            # The house front: its boundary point nearest the road.
            ring = poly.exterior
            front = np.asarray(ring.interpolate(ring.project(shapely.Point(edge_pt))).coords[0])
            near = _nearest_road(roads, front)
            if near is None:
                continue
            road, edge_pt, tan, n, edge, _ = near
            if not (DRIVEWAY_MIN_M <= edge <= DRIVEWAY_MAX_M):
                continue
            if (
                junction_tree is not None
                and junction_tree.query(edge_pt)[0] < DRIVEWAY_JUNCTION_CLEAR_M
            ):
                continue
            if (
                blocked is not None
                and not blocked.is_empty
                and blocked.distance(shapely.Point(edge_pt)) < 5.0
            ):
                continue
            rural = rural_at(centroids[h], road)
            depth = float((front - edge_pt) @ n)
            if rural:
                bend = edge_pt + n * min(DRIVEWAY_PERPENDICULAR_M, max(depth, 0.0) * 0.6)
                path = np.array([edge_pt - n * 0.3, bend, front])
            else:
                path = np.array([edge_pt - n * 0.3, edge_pt + n * max(depth, 1.0)])
            path = path[
                np.concatenate([[True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 0.3])
            ]
            if len(path) < 2:
                continue
            w = DRIVEWAY_RURAL_WIDTH_M if rural else DRIVEWAY_URBAN_WIDTH_M
            dw = _driveway_polygon(path, w, rural, near, poly)
            # Never through another building or another driveway.
            if fp_tree is not None:
                hits = fp_tree.query(dw, predicate="intersects")
                if any(all_fp[i] is not poly for i in hits):
                    continue
            if served is not None and served.intersects(dw):
                continue
            d = Driveway(dw, path, "inferred", rural, road, edge_pt, n)
            drives.append(d)
            grown = dw.buffer(DRIVEWAY_SERVED_M)
            served = grown if served is None else served.union(grown)

    # Clip, mesh and mark the mouths.
    carriage = _union([c for c in net.carriage])
    cut = _union(
        [carriage, net.paved.difference(_union(net.islands)) if net.islands else net.paved]
    )
    route_left, route_right, left_outer, right_outer = net.route_columns
    outer_cls = [CentreLine(cl.xz, cl.y, cl.hw + sh) for _, cl, sh in roads]

    def height(xz: np.ndarray) -> np.ndarray:
        g = np.asarray(ground(xz), np.float64)
        d = np.full(len(xz), np.inf)
        for cl in outer_cls:
            dd, _ = cl.project(xz, 30.0)
            d = np.minimum(d, dd)
        lift = DRIVEWAY_SHOULDER_LIFT_M + (
            DRIVEWAY_LIFT_M - DRIVEWAY_SHOULDER_LIFT_M
        ) * _smoothstep(d / 1.0)
        return g + lift

    batch = Batcher(options.batch_m)
    sh_l = _quads(left_outer, route_left)
    sh_r = _quads(route_right, right_outer)
    mid_l = 0.5 * (route_left[:-1, [0, 2]] + route_left[1:, [0, 2]])
    mid_r = 0.5 * (route_right[:-1, [0, 2]] + route_right[1:, [0, 2]])
    seg_tree = cKDTree(0.5 * (route_xz[:-1] + route_xz[1:]))
    for d in drives:
        poly = d.polygon.difference(cut) if not cut.is_empty else d.polygon
        poly = _union([p for p in _polygons(poly, 0.5)])
        if poly.is_empty:
            continue
        d.polygon = poly
        res.driveways.append(d)
        mesh = mesh_polygon(poly, height, cell_m=3.0, uv_tile_m=2.0 if d.rural else 4.0, seg_m=3.0)
        batch.add("gravel" if d.rural else "asphalt", mesh)
        if d.road == -1:
            near_seg = seg_tree.query_ball_point(d.mouth, 25.0)
            if near_seg:
                idx = np.array(sorted(near_seg))
                grown = poly.buffer(0.6)
                for mids, quads, edge_mask, sh_mask in (
                    (mid_l, sh_l, net.masks.edge_left, net.masks.shoulder_left),
                    (mid_r, sh_r, net.masks.edge_right, net.masks.shoulder_right),
                ):
                    hit = shapely.contains_xy(grown, mids[idx, 0], mids[idx, 1])
                    if hit.any():
                        edge_mask[idx[hit]] = False
                        res.masks_changed = True
                    covered = shapely.contains(poly.buffer(0.05), quads[idx])
                    if covered.any():
                        sh_mask[idx[covered]] = False
                        res.masks_changed = True
        if d.rural:
            # The mailbox: just beyond the shoulder, beside the driveway on its right (seen
            # from the road), facing the road.
            right = np.array([-d.normal[1], d.normal[0]])
            shoulder = next((sh for idx, _, sh in roads if idx == d.road), 1.0)
            base = d.mouth + d.normal * (shoulder + MAILBOX_SETBACK_M)
            across = shapely.LineString([base - right * 20.0, base + right * 20.0])
            half = 0.0
            for piece in _polygons(across.intersection(poly).buffer(0.01), 0.0):
                pts = np.asarray(piece.exterior.coords)
                half = max(half, float(np.max((pts - base) @ right)))
            pos = base + right * (max(half, DRIVEWAY_RURAL_WIDTH_M / 2) + MAILBOX_SIDE_M)
            face = (float(-d.normal[0]), float(-d.normal[1]))
            res.items.append(
                PlannedItem(
                    MAILBOX, (float(pos[0]), float(pos[1])), face, ref=f"driveway:{d.source}"
                )
            )
    res.pieces = batch.pieces()
    for p in res.pieces:  # their own names: they come after the network's pieces
        p.name = p.name.replace("_net", "_dw")
    return res


def _driveway_polygon(
    path: np.ndarray,
    width: float,
    rural: bool,
    near: tuple,
    house: shapely.Geometry | None,
) -> shapely.Geometry:
    """The driveway's surface: the path at `width`, a rural entrance flared into the road
    edge with `DRIVEWAY_FLARE_M` corners, stopping at the house."""
    _, edge_pt, tan, n, _, _ = near
    body = shapely.LineString(path).buffer(width / 2, cap_style="flat", join_style="round")
    if rural:
        road = shapely.LineString([edge_pt - tan * 20 - n * 3, edge_pt + tan * 20 - n * 3]).buffer(
            3.0, cap_style="flat"
        )
        r = DRIVEWAY_FLARE_M
        flared = body.union(road).buffer(r, quad_segs=6).buffer(-r, quad_segs=6)
        body = flared.difference(road).intersection(
            shapely.Point(edge_pt).buffer(width / 2 + r + 1.0).union(body)
        )
    if house is not None:
        body = body.difference(house)
    return body
