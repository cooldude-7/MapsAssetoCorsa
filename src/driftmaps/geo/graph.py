"""Unsimplified directed edge graph over OSM ways (docs/research/geo.md §1.3, PLAN §5.1).

Every consecutive node pair of every way is one *segment*; each segment gives a forward edge
(along the way's node order, `dir = +1`) and a backward edge (`dir = -1`). One-ways are ignored
by default because the result is a closed race road; `respect_oneway=True` drops the edges that
go against `oneway` (including implied one-ways, see `attributes.is_oneway`).

Edges and segments are numbered in sorted (way id, index in way, forward first) order, and the
outgoing edges of each node are listed in edge order, so anything iterating the graph is
deterministic. Lengths are integer millimetres on the local projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from driftmaps.geo.attributes import is_oneway
from driftmaps.geo.overpass import OsmData
from driftmaps.geo.project import LocalProjection

__all__ = ["EdgeGraph", "build_graph"]


@dataclass(frozen=True, eq=False)
class EdgeGraph:
    """Arrays describing nodes, undirected segments and directed edges.

    Nodes are indexed 0..N-1 in ascending OSM id (`node_ids`). Segment s joins `seg_a[s]` ->
    `seg_b[s]` in way order; `seg_fwd[s]` / `seg_bwd[s]` are its directed edges (-1 if dropped
    by one-way rules). Edge e goes `edge_u[e]` -> `edge_v[e]`; `out_start[x]:out_start[x+1]`
    slices `out_edges` to give node x's outgoing edges in ascending edge order.
    """

    proj: LocalProjection
    node_ids: np.ndarray
    node_xy: np.ndarray
    node_lonlat: np.ndarray
    node_degree: np.ndarray  # number of distinct neighbouring nodes
    seg_a: np.ndarray
    seg_b: np.ndarray
    seg_way: np.ndarray  # OSM way id
    seg_idx: np.ndarray  # index of the segment within its way
    seg_len_mm: np.ndarray
    seg_fwd: np.ndarray
    seg_bwd: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_seg: np.ndarray
    edge_dir: np.ndarray  # +1 along the way, -1 against
    edge_len_mm: np.ndarray
    out_start: np.ndarray
    out_edges: np.ndarray
    way_tags: dict[int, dict[str, str]] = field(repr=False)
    respect_oneway: bool = False

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return len(self.edge_u)

    def node_index(self, osm_id: int) -> int:
        i = int(np.searchsorted(self.node_ids, osm_id))
        if i >= len(self.node_ids) or self.node_ids[i] != osm_id:
            raise KeyError(f"node {osm_id} not in graph")
        return i

    def outgoing(self, node: int) -> np.ndarray:
        return self.out_edges[self.out_start[node] : self.out_start[node + 1]]

    def edge_vectors(self) -> np.ndarray:
        """(E, 2) unit direction of each edge on the projection (zero for zero-length edges)."""
        d = self.node_xy[self.edge_v] - self.node_xy[self.edge_u]
        n = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])
        return np.where(n[:, None] > 0.0, d / np.where(n > 0.0, n, 1.0)[:, None], 0.0)


def build_graph(osm: OsmData, proj: LocalProjection, respect_oneway: bool = False) -> EdgeGraph:
    """Edge graph of every way in `osm`, projected with `proj`."""
    node_lonlat_map = osm.node_lonlat()
    if not node_lonlat_map:
        raise ValueError("no ways in the OSM data")
    node_ids = np.array(sorted(node_lonlat_map), dtype=np.int64)
    node_lonlat = np.array([node_lonlat_map[int(n)] for n in node_ids], dtype=np.float64)
    node_xy = proj.to_xy(node_lonlat)
    index = {int(n): i for i, n in enumerate(node_ids)}

    seg_a: list[int] = []
    seg_b: list[int] = []
    seg_way: list[int] = []
    seg_idx: list[int] = []
    seg_oneway: list[int] = []
    for way in sorted(osm.ways, key=lambda w: w.id):
        ow = is_oneway(way.tags) if respect_oneway else 0
        for i in range(len(way.nodes) - 1):
            a, b = index[way.nodes[i]], index[way.nodes[i + 1]]
            if a == b:
                continue
            seg_a.append(a)
            seg_b.append(b)
            seg_way.append(way.id)
            seg_idx.append(i)
            seg_oneway.append(ow)
    sa = np.array(seg_a, dtype=np.int64)
    sb = np.array(seg_b, dtype=np.int64)
    d = node_xy[sb] - node_xy[sa]
    seg_len_mm = np.rint(np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) * 1000.0).astype(np.int64)

    eu: list[int] = []
    ev: list[int] = []
    eseg: list[int] = []
    edir: list[int] = []
    fwd = np.full(len(sa), -1, dtype=np.int64)
    bwd = np.full(len(sa), -1, dtype=np.int64)
    for s, ow in enumerate(seg_oneway):
        if ow >= 0:
            fwd[s] = len(eu)
            eu.append(seg_a[s])
            ev.append(seg_b[s])
            eseg.append(s)
            edir.append(1)
        if ow <= 0:
            bwd[s] = len(eu)
            eu.append(seg_b[s])
            ev.append(seg_a[s])
            eseg.append(s)
            edir.append(-1)
    edge_u = np.array(eu, dtype=np.int64)
    edge_v = np.array(ev, dtype=np.int64)
    edge_seg = np.array(eseg, dtype=np.int64)

    order = np.argsort(edge_u, kind="stable")  # by tail node, then edge index
    out_start = np.zeros(len(node_ids) + 1, dtype=np.int64)
    np.add.at(out_start, edge_u + 1, 1)
    out_start = np.cumsum(out_start)

    pairs = np.unique(np.sort(np.stack([sa, sb], axis=1), axis=1), axis=0)
    degree = np.bincount(pairs.ravel(), minlength=len(node_ids)).astype(np.int64)

    return EdgeGraph(
        proj=proj,
        node_ids=node_ids,
        node_xy=node_xy,
        node_lonlat=node_lonlat,
        node_degree=degree,
        seg_a=sa,
        seg_b=sb,
        seg_way=np.array(seg_way, dtype=np.int64),
        seg_idx=np.array(seg_idx, dtype=np.int64),
        seg_len_mm=seg_len_mm,
        seg_fwd=fwd,
        seg_bwd=bwd,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_seg=edge_seg,
        edge_dir=np.array(edir, dtype=np.int8),
        edge_len_mm=seg_len_mm[edge_seg],
        out_start=out_start,
        out_edges=order.astype(np.int64),
        way_tags={w.id: dict(w.tags) for w in osm.ways},
        respect_oneway=respect_oneway,
    )
