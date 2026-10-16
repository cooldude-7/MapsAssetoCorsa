"""Municipal sidewalk lines (open data), cached: where a town says its sidewalks are.

- **Oakville** (`OAKVILLE_SIDEWALKS`): the Town's Sidewalks layer (23,452 lines; `ASSET_TYPE`
  Sidewalk / Multi-use / Road Crossing), Open Government Licence - Town of Oakville: free to
  use with the attribution line below.
- **Burlington** (`BURLINGTON_SIDEWALKS`): the City's Pedestrian Network (4,828 lines, mostly
  1.5 m), under the City's Open Data Terms of Use, which are revocable: used only when the
  build's Burlington switch is on (the same one as its buildings), and the terms travel with
  the data (LICENSE-DATA.txt).

`fetch_sidewalks(src, boxes)` returns the lines (lon/lat) touching the boxes, cached per box
under `<cache>/sidewalks/<source>/`; `offline=True` reads only the cache. Only lines that are
sidewalks or multi-use paths are kept (road crossings are left out).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from driftmaps.geo.building_sources import BURLINGTON_TERMS_URL, SourceError, _get_json
from driftmaps.geo.overpass import default_cache_dir

__all__ = [
    "BURLINGTON_SIDEWALKS",
    "OAKVILLE_SIDEWALKS",
    "SIDEWALK_SOURCES",
    "SidewalkSource",
    "fetch_sidewalks",
]

CACHE_VERSION = 1
Box = tuple[float, float, float, float]  # west, south, east, north (degrees)


@dataclass(frozen=True)
class SidewalkSource:
    name: str
    title: str
    url: str  # the layer (".../MapServer/23")
    bbox: Box  # the layer's extent: boxes outside it are not queried
    attribution: str
    where: str = "1=1"
    out_fields: tuple[str, ...] = ("OBJECTID",)
    keep: Callable[[Mapping[str, object]], bool] | None = None
    page_size: int = 1000


def _oakville_keep(attrs: Mapping[str, object]) -> bool:
    return str(attrs.get("ASSET_TYPE") or "") in ("Sidewalk", "Multi-use")


def _burlington_keep(attrs: Mapping[str, object]) -> bool:
    kind = str(attrs.get("PN_TYPE") or "").lower()
    return not kind or "sidewalk" in kind or "multi" in kind


OAKVILLE_SIDEWALKS = SidewalkSource(
    name="oakville",
    title="Town of Oakville Sidewalks",
    url=("https://maps.oakville.ca/oakgis/rest/services/ArcGISOnline/AGO_OpenData/MapServer/23"),
    bbox=(-79.8133, 43.3707, -79.6247, 43.5284),
    attribution=(
        "Sidewalks: contains information licensed under the Open Government Licence - Town of "
        "Oakville (https://www.oakville.ca/getmedia/3cb7902a-2220-433e-8aed-24f326b745b9/"
        "town-hall-oakville-open-data-licence.pdf)."
    ),
    out_fields=("OBJECTID", "ASSET_TYPE"),
    keep=_oakville_keep,
)
BURLINGTON_SIDEWALKS = SidewalkSource(
    name="burlington",
    title="City of Burlington Pedestrian Network",
    url="https://mapping.burlington.ca/arcgisweb/rest/services/COB/RoadStructures/MapServer/40",
    bbox=(-79.9584, 43.2857, -79.7229, 43.4742),
    attribution=(
        "Sidewalks: contains City of Burlington data, used under the City of Burlington Open "
        f"Data Terms of Use: {BURLINGTON_TERMS_URL} . These terms apply to anyone receiving "
        "this data, and the City may change or revoke them."
    ),
    out_fields=("OBJECTID", "PN_TYPE"),
    keep=_burlington_keep,
)
SIDEWALK_SOURCES: dict[str, SidewalkSource] = {
    s.name: s for s in (OAKVILLE_SIDEWALKS, BURLINGTON_SIDEWALKS)
}


def _round_box(box: Box) -> Box:
    w, s, e, n = box
    return (
        math.floor(w * 1e3) / 1e3,
        math.floor(s * 1e3) / 1e3,
        math.ceil(e * 1e3) / 1e3,
        math.ceil(n * 1e3) / 1e3,
    )


def _cache_path(cache_dir: Path | str | None, src: SidewalkSource, box: Box) -> Path:
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    key = f"v{CACHE_VERSION}|{src.url}|{src.where}|{box}"
    return cache / "sidewalks" / src.name / f"{hashlib.sha256(key.encode()).hexdigest()}.json.gz"


def _touch(a: Box, b: Box) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def fetch_sidewalks(
    src: SidewalkSource,
    boxes: Sequence[Box],
    *,
    cache_dir: Path | str | None = None,
    offline: bool = False,
    get_json: Callable[[str, Mapping[str, object]], dict] = _get_json,
) -> list[np.ndarray]:
    """The source's sidewalk lines ((K, 2) lon/lat each) touching any of the boxes, each once.
    Raises `SourceError` when a box can be had neither from the cache nor the server."""
    found: dict[str, np.ndarray] = {}
    for box in boxes:
        if not _touch(box, src.bbox):
            continue
        rb = _round_box(box)
        path = _cache_path(cache_dir, src, rb)
        rows = None
        try:
            rows = json.loads(gzip.decompress(path.read_bytes()))
        except (OSError, ValueError, EOFError):
            rows = None
        if rows is None:
            if offline:
                raise SourceError(f"{src.title} for this area are not cached (offline)")
            rows = []
            offset = 0
            while True:
                data = get_json(src.url + "/query", {
                    "where": src.where,
                    "geometry": ",".join(f"{v:.6f}" for v in rb),
                    "geometryType": "esriGeometryEnvelope",
                    "inSR": 4326,
                    "spatialRel": "esriSpatialRelIntersects",
                    "outFields": ",".join(src.out_fields),
                    "returnGeometry": "true",
                    "outSR": 4326,
                    "geometryPrecision": 7,
                    "orderByFields": src.out_fields[0],
                    "resultOffset": offset,
                    "resultRecordCount": src.page_size,
                    "f": "json",
                })  # fmt: skip
                feats = data.get("features") or []
                for f in feats:
                    attrs = f.get("attributes") or {}
                    if src.keep is not None and not src.keep(attrs):
                        continue
                    oid = str(attrs.get("OBJECTID", len(rows)))
                    for k, p in enumerate((f.get("geometry") or {}).get("paths") or []):
                        if len(p) >= 2:
                            rows.append([f"{oid}.{k}", [[round(x, 7), round(y, 7)] for x, y in p]])
                offset += len(feats)
                if not feats or not data.get("exceededTransferLimit"):
                    break
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(gzip.compress(json.dumps(rows).encode(), mtime=0))
            tmp.replace(path)
        for key, pts in rows:
            found.setdefault(key, np.asarray(pts, np.float64))
    return [found[k] for k in sorted(found)]
