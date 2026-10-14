# Research: geodata, routing, elevation, geometry, packaging, legal

Research date: 2026-10-14. This covers sections B, C, D, F (local tool) and G of `docs/PLANNING_PROMPT.md`.
Claims are marked **VERIFIED (url)** when checked against a source or a live request, and **ASSUMED** otherwise.
Library versions come from the PyPI JSON API on 2026-10-14 (VERIFIED, https://pypi.org/pypi/<name>/json).

---

## 0. Recommendation (chosen stack per area)

| Area | Choice | Why |
|---|---|---|
| **OSM data** | **Overpass API** with a corridor `poly:` query, `out tags geom`, cached on disk, plus a configurable mirror list. Optional "offline mode" later: Geofabrik `.pbf` + `osmium` (pyosmium 4.3.1). | No server to run. The fair-use limits (about 10k requests and 1 GB per day) are far above what one person's builds need. Caching gives determinism. |
| **Routing** | **Our own Python router** over the fetched graph: `networkx`-style data, with a hand-written **edge-based, waypoint-layered Dijkstra** (turn costs, no U-turns), on **unsimplified** OSM edges so each edge keeps its way ID and tags. OSMnx 2.1.1 can be used to build the graph, but don't let it simplify. | It's deterministic and bit-for-bit identical between the preview and the build, because both call the same function on the same cached snapshot. It's tag-complete, with no daemon to run. OSRM, GraphHopper, Valhalla and BRouter all need a server and none returns full per-edge OSM tags as directly. |
| **Preview ≡ build** | The preview endpoint returns a `route.json` (snapshot timestamp, ordered directed edge list `(way_id, from_node, to_node)`, snapped pins). **The build consumes `route.json` and never re-routes.** | This is the strongest guarantee that "the rig routes the same pins the same way". |
| **Elevation (US)** | **USGS 3DEP**: (1) the 1 m project DEMs found through the TNM Access API, (2) the S1M seamless 1 m, (3) the 1/3 arc-second seamless VRT. All are public-domain COGs on `prd-tnm.s3.amazonaws.com`, read with rasterio windowed reads over `/vsicurl/`. | VERIFIED as COGs that need no authentication. 1 m lidar is bare earth (a DTM). |
| **Elevation (rest of world)** | **Copernicus GLO-30** COGs from `copernicus-dem-30m.s3.amazonaws.com` (no auth). Optional national high-resolution adapters, first **France IGN RGE ALTI 1 m / LiDAR HD** via the Géoplateforme. Fallbacks: AWS Terrain Tiles, then OpenTopography (SRTMGL1, NASADEM, AW3D30; needs an API key). | GLO-30 is global and free, but it is a **DSM**, so it needs robust filtering. |
| **Profile smoothing** | Resample at 1 m. Take a robust cross-road median. Weight out bridges and tunnels (weight 0). Fit with an **asymmetric Whittaker smoother**, then a **QP with hard grade and vertical-curvature limits (OSQP 1.1.3, Apache-2.0)**. | It handles DSM vegetation bias, interpolates across bridges and tunnels, and guarantees a driveable profile. |
| **Projection** | **pyproj 3.8.0** custom Transverse Mercator `+proj=tmerc +lat_0=<start> +lon_0=<start> +k=1 +x_0=0 +y_0=0 +ellps=WGS84`, with the origin at the start. AC axes: X=East, Y=Up, Z=−North (**ASSUMED, needs a spike**). | It avoids UTM zone edges, and the scale error is about 5 ppm at 20 km. |
| **Centreline** | Fillet sharp junction turns. Use a curvature-regularised smoother (Whittaker on x(s), y(s)) with a maximum deviation tolerance. Resample to **1 m**. Optional `pyclothoids` 0.2.0 for clothoid refinement later. | OSM polylines are coarse, and Catmull-Rom alone keeps the kinks. |
| **Meshes** | numpy + shapely 2.2.0 (buffers, unions, walls) + scipy 1.18.1 (cKDTree, sparse) + trimesh 5.1.1 (debug GLB/OBJ export, checks) + mapbox-earcut 2.1.0 (building and polygon caps). **Avoid `triangle`** because of the licence. | Every one of these has Windows wheels (VERIFIED). Triangle "may not be sold or included in commercial products without a license" (VERIFIED, https://www.cs.cmu.edu/~quake/triangle.html). |
| **Terrain fit** | A regular-grid corridor heightfield with LOD rings. A **cut/fill slope-clamp** formula is applied per road "leg" (this handles hairpins). Triangles under the road are removed and the road verges get skirts. v2: a sparse least-squares (biharmonic) solve. | It's simple, robust and vectorised. |
| **Walls** | The boundary of `unary_union(buffer(leg_i, half_width + shoulder + verge))`, split by layer and extruded. | Overlapping hairpin corridors merge automatically, so no wall ever cuts across another leg of the road. |
| **Local tool** | Python **3.12** package with a **Typer 0.27.3** CLI and a **FastAPI 0.143.1 + uvicorn** local web UI. Distribute with **`uv tool install` / pipx** first, then a **PyInstaller 6.22.3 one-folder** exe. | Python 3.12 is the floor for the current rasterio, pyproj and scipy releases. |
| **Map UI** | **MapLibre GL JS + OpenFreeMap "dark"** style: no key, no limits, commercial use OK. CARTO dark only with the user's own CARTO key. | CARTO's current terms **require an API key** (VERIFIED, see §6). |
| **Geocoder** | **Photon** (`photon.komoot.io/api`) for search-as-you-type, Nominatim only for explicit submit, at ≤1 request per second with an identifying User-Agent and a cache. | Nominatim forbids client-side autocomplete (VERIFIED). |
| **Finding AC** | Order: CLI flag or config, then Steam (registry → `libraryfolders.vdf` → `appmanifest_244210.acf`), then Content Manager's `_ac_root` as a hint only. | |
| **Textures** | ambientCG (CC0) and Poly Haven (CC0). Credits are optional but recommended. | VERIFIED, CC0. |

---

## 1. Routing (section B)

### 1.1 Options compared

| Option | Licence | Public server terms | Per-edge OSM tags? | Self-hosting | Verdict |
|---|---|---|---|---|---|
| **Overpass + own router** | Overpass data is ODbL. OSMnx is MIT (VERIFIED, https://raw.githubusercontent.com/gboeing/osmnx/main/LICENSE.txt) | overpass-api.de: <10,000 queries/day, <1 GB/day, no parallel scripts; "divide those numbers by 100" for regular app use; pause 30 s after a 429 or 406; send an identifying User-Agent or Referer (VERIFIED, https://wiki.openstreetmap.org/wiki/Overpass_API). Default timeout 180 s, default maxsize 512 MiB, 429 for rate limit, 504 for overload (VERIFIED, https://dev.overpass-api.de/overpass-doc/en/preface/commons.html) | **Yes, all tags** (`out tags geom`) | None needed | **Chosen** |
| OSRM | BSD-2 (VERIFIED, https://raw.githubusercontent.com/Project-OSRM/osrm-backend/master/LICENSE.TXT) | Demo: ≤1 req/s, "reasonable, non-commercial", no guarantees (VERIFIED, https://github.com/Project-OSRM/osrm-backend/wiki/Demo-server) | **No.** Annotations are `nodes, distance, duration, datasources, weight, speed`, so node IDs only and no way IDs or tags. Has `continue_straight` (no U-turns at waypoints), `bearings` and `approaches` (VERIFIED, https://project-osrm.org/docs/v5.24.0/api/) | Needs a C++ server and a pre-processed `.osrm` per region. Fine for a country; tens of GB of RAM for continents (ASSUMED) | Node IDs could be joined back to Overpass data, but that's 2 sources and 2 snapshots. Rejected. |
| GraphHopper | Apache-2.0 (VERIFIED, https://raw.githubusercontent.com/graphhopper/graphhopper/master/LICENSE.txt) | Free API plan: 500 credits/day, **non-commercial only** (VERIFIED, https://www.graphhopper.com/pricing/) | Path details give a subset of encoded values (road_class, surface, lanes…). Way IDs need configuration (ASSUMED) | A Java server and an import per region | Rejected (needs a server; the free API is too small for a hosted site later) |
| Valhalla | MIT. FOSSGIS demo at valhalla1.openstreetmap.de, fair use, rate-limited, asks for an `X-Client-Id` header and an announcement (VERIFIED, https://github.com/valhalla/valhalla) | | `trace_attributes` returns `edge.way_id, edge.surface, edge.lane_count, edge.road_class, edge.bridge, edge.tunnel`, … (VERIFIED, https://valhalla.github.io/valhalla/api/map-matching/) | A C++ server and tile build | **Best server option** if we ever want one (map matching plus attributes), but still not every raw tag (`width`, `layer`). Keep as a plan B. |
| BRouter | MIT (VERIFIED, https://github.com/abrensch/brouter) | brouter.de hosts brouter-web; no API policy published | Profile-driven messages with way tags (ASSUMED) | Java; offline 5×5° `rd5` segment files built from OSM (VERIFIED, same URL) | Interesting for offline use, but it's Java and the tag set is limited to what the profile encodes. Rejected. |

**Decision:** Overpass + own router. The router is about 300 lines of Python and is fully under our control, including tie-breaking, U-turn rules and allowed highway classes. That control is what makes preview ≡ build trivially true.

Note: a live test from this research sandbox got HTTP 504 (dispatcher error) from overpass-api.de and timeouts from two mirrors (the sandbox proxy may be a factor). So in practice, **plan for 504/429 retries with backoff, a mirror list (overpass-api.de, overpass.private.coffee, maps.mail.ru/osm/tools/overpass…) and a persistent cache** (ASSUMED that these mirrors behave similarly; the list of instances is VERIFIED, https://wiki.openstreetmap.org/wiki/Overpass_API).

### 1.2 Fetching the graph

- **Corridor query.** Build a polygon = `buffer(LineString(pins), max(1500 m, 0.3 × straight_line_length))` in the local metric CRS, simplify it to ≤ 50 vertices, and send it as `(poly:"lat lon lat lon …")`. Overpass QL details: `[date:"YYYY-MM-DDThh:mm:ssZ"]` attic queries, `[bbox:s,w,n,e]` order, `out geom`, `poly:` syntax, `[timeout:]`, `[maxsize:]`, and linestring `around:` (since 0.7.55) (VERIFIED, https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL).
  ```
  [out:json][timeout:120];
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service|track)(_link)?$"]
     ["area"!="yes"]["highway"!="construction"](poly:"...");
  out tags geom qt;   // tags + node coords (no separate node fetch)
  >; out skel qt;     // optional: node ids are in "nodes" if `out body geom` is used
  ```
  Use `out body geom` so each way has both `nodes` (IDs, needed for topology) and `geometry` (coordinates).
- **Expand on failure.** If no path exists, or the best path touches the corridor boundary, double the buffer and retry (max 2×).
- **Snapshot.** Store the raw JSON gzip'd in the cache, keyed by `sha256(query)`, together with `osm3s.timestamp_osm_base`. The `route.json` records that timestamp and cache key. A rebuild can use the cache, or re-fetch the identical state with `[date:"<timestamp>"]` (attic) (VERIFIED that the `date` setting exists, URL above).
- **OSMnx path (optional).** `ox.graph_from_polygon(poly, network_type=..., simplify=False, retain_all=True, truncate_by_edge=True, custom_filter=...)`. `graph_from_bbox(bbox)` takes `bbox=(left, bottom, right, top)` in v2. Set `ox.settings.useful_tags_way` to the full list below. Set `ox.settings.overpass_url`, `ox.settings.http_user_agent` and `ox.settings.cache_folder`. `use_cache` and `overpass_rate_limit` default to True (VERIFIED, https://osmnx.readthedocs.io/en/stable/user-reference.html). If you simplify, merged edges keep `geometry` and list-valued attributes (`osmid` becomes a list), and `edge_attrs_differ=[...]` can stop merges across tag changes (VERIFIED, same URL). **Recommendation:** don't simplify; one edge = one OSM node pair. OSMnx 2.1.1 needs Python ≥3.11 (VERIFIED, PyPI). It pulls in geopandas, which is heavy but has Windows wheels. A hand-rolled Overpass JSON → graph parser is about 80 lines and avoids that dependency. Either way is fine.
- **Tags to keep:** `highway, name, ref, oneway, junction, lanes, lanes:forward, lanes:backward, width, est_width, width:carriageway, surface, smoothness, tracktype, bridge, tunnel, covered, layer, level, cutting, embankment, maxspeed, sidewalk, shoulder, shoulder:*, barrier, access, motor_vehicle, area, incline, bridge:structure, man_made`.

### 1.3 The router

**Graph.** Use a directed multigraph with nodes = OSM node IDs. For each consecutive node pair in each way, add a forward edge, plus a reverse edge unless one-ways are respected. Each edge carries `(way_id, idx_in_way, length_m, tags_ref, bearing_in, bearing_out)`.
- **One-ways:** add a setting, `respect_oneway` (default **True** for realism and to keep motorway carriageways apart; the user can switch it off for "closed-road touge"). Divided carriageways (`oneway=yes` on dual carriageways) need respect for one-ways, or the route will mix both carriageways.

**Snapping clicks.**
1. Project the click to the local TM.
2. Query a shapely `STRtree` of edge segments: `tree.query_nearest(point, max_distance=R, return_distance=True)`, with R based on zoom (for example 60 m).
3. Prefer, in order: within 3 m of the nearest, higher highway class, then lower way ID (deterministic tie-break).
4. Result: `Snap(edge=(u,v,key), t∈[0,1], point_xy, dist)`.
5. Return the snapped point to the UI so the pin moves onto the road ("snapped to the real network").

**Waypoints + no U-turns: layered, edge-based Dijkstra.**
- State = `(directed_edge, k)`, where k = the number of pins already passed (0…n−1).
- A transition from edge e1 to edge e2 at node v costs `len(e2) + turn_penalty(angle)`.
  - A **U-turn** (e2 = reverse(e1), or a turn angle >170°) is **forbidden**. Optionally allow it only at dead ends with a big penalty.
  - Sharp turns get a mild penalty (for example 5 m × (angle/90°)²) to prefer "through" roads at junctions.
  - A "class change" penalty prefers staying on the same `ref`/`name` (it makes the route follow the named road like the original "follows real roads").
- **Snapped pins on an edge:** split the edge virtually at t. The layer increments when the path passes the virtual node of pin k+1 in either direction. Arrival direction and departure direction must be the same, because a U-turn is forbidden at the virtual node too.
- The start state is the virtual edge leaving pin 0 in either direction (two seeds). The goal is reaching pin n−1's virtual node at layer n−1.
- This gives a **global** optimum over all waypoints with no U-turns. It's equivalent to OSRM's `continue_straight=true` (VERIFIED, OSRM docs above), but better than chaining per-leg routes.
- **Determinism:**
  - Store edge costs as **integers (millimetres + integer penalties)**, so no float ties.
  - Heap items are `(cost, tie_counter_from_sorted_edge_order, state)`.
  - Iterate adjacency in sorted `(way_id, idx)` order.
  - Pure Python `heapq`, so the result doesn't depend on the networkx version.
  - (networkx 3.7 is BSD-3 if used: VERIFIED, PyPI.)

**Reverse.** Reverse the pin list and **re-route**, rather than reversing the geometry, because one-ways and turn penalties may differ. Show the new route.

**Undo / clear.** These are UI state only. Each change triggers `POST /api/route` with the pins and gets back a new `route.json`.

**Junctions.**
- The route passes through intersections. Record every side-road edge that touches a route node (`stub` list with bearing and width).
- The geometry stage can then (a) build short stubs of 10–25 m that end in a wall, so junctions look real and the terrain blends, or (b) just close them with the wall (v1).
- Mark "junction zones" (±15 m along s) so the AI line and the camber/smoothing code treat them specially (no superelevation, keep the corner fillet).

**Self-crossing / repeated sections.**
- Detect when a node is visited twice, and use shapely `LineString.is_simple`.
  - Crossing at a **different `layer`/bridge/tunnel** is OK. Keep both and check vertical clearance ≥ 4.5 m after elevation.
  - Crossing **at grade**, or overlapping (driving the same road twice): reject by default with a message ("route crosses itself at grade; move a waypoint"). An option could allow it, but AC's A→B spline would self-intersect, which the AI-line research must check.
- **Length limits (ASSUMED, to be tuned):**
  - Warn above 25 km and refuse above 60 km in the local tool.
  - For a hosted site later, cap around 20 km per build.
  - Also cap the corridor polygon area (≤ 400 km²) to keep Overpass happy.

### 1.4 Per-segment tags along the route

`route.json`:
```json
{ "version":1, "osm_base":"2026-10-14T07:12:03Z", "overpass_cache_key":"…",
  "pins":[{"lat":..,"lon":..,"snap":{"way":123,"u":1,"v":2,"t":0.37}}],
  "edges":[{"way":123,"u":1,"v":2,"dir":1}, ...],     // ordered, directed
  "length_m":3612.4 }
```
The build expands this into a **polyline with per-vertex arc length s** and an **interval table** `[(s0, s1, way_id, tags…)]`. Consecutive edges of the same way merge into one interval. Every later stage (width, surface, bridge mask, camber, AI speed hints, lanes for traffic) looks things up by s through `bisect` over the interval starts. This is also the exact "derivative database" to ship for ODbL compliance (§6).

### 1.5 Road attributes → parameters

- **Width:**
  - Parse `width` / `width:carriageway` / `est_width`: first number of a `;` list, units `m` by default, `ft`, `'`, `"` → convert.
  - Otherwise use `lanes × lane_width(class, region)`.
  - Otherwise use the class default.
  - Clamp to `[min_drive_width (default 5.0 m, configurable), 20 m]`.
  - Ramp width changes over 30 m.
- **Lanes:** `lanes` = total of both directions. If it's missing, assume 2 lanes for two-way residential, tertiary, secondary and primary roads (1 if one-way). Assume 1 lane for unclassified, service and track. Motorways and trunks are usually two one-way ways with ≥2 lanes each. `lanes:forward/backward` give the split (VERIFIED, https://wiki.openstreetmap.org/wiki/Key:lanes). Store per-direction lane counts in the interval table, because traffic (section H) needs them.
- **Lane widths.** AASHTO-based guidance: 10–12 ft (3.0–3.6 m) on major roads, 12 ft standard for high-speed rural roads, as low as 9 ft (2.7 m) on low-volume rural roads (VERIFIED via summary, https://www.fhwa.dot.gov/planning/css/resources/lanewidth and https://assets.lawrenceks.org/mpo/study/MemoLaneWidths.pdf). European values are ASSUMED (typical national guidance: 3.5 m primary, 3.0–3.25 m minor).

  | highway | default lanes | lane width US / EU (m) | default carriageway if no tags (m) |
  |---|---|---|---|
  | motorway, trunk (per one-way carriageway) | 2 | 3.65 / 3.5–3.75 | 7.5 |
  | primary | 2 | 3.6 / 3.5 | 7.0 |
  | secondary | 2 | 3.4 / 3.25 | 6.5 |
  | tertiary | 2 | 3.3 / 3.0 | 6.0 |
  | unclassified | 1–2 | 3.0 / 2.75 | 5.0 |
  | residential | 2 | 3.0 / 2.75 | 5.5 |
  | *_link | 1 | 4.0 | 4.5 |
  | service, living_street | 1 | 3.5 | 4.0 (clamped to min_drive_width) |
  | track | 1 | n/a | 3.5 (gravel), clamp → min_drive_width |

- **Shoulders:** `shoulder=*` tags if present. Otherwise 0.5 m paved + 1.5–2.5 m verge (grass or gravel), configurable. That gives the run-off between road edge and wall (ASSUMED values).
- **Surface:**
  - `asphalt|paved|concrete|concrete:*|chipseal` → road asphalt.
  - `paving_stones|sett|cobblestone|unhewn_cobblestone` → road with a different texture (and optionally lower grip).
  - `compacted|fine_gravel|gravel|unpaved|dirt|ground|earth|grass` → gravel surface.
  - Missing: asphalt for residential and above, gravel for `track`. The AC surfaces mapping (1ROAD, 1GRASS, …) belongs to section A.
- **Bridges / tunnels / layer:** `bridge=*` (≠no), `man_made=bridge`, `tunnel=*` (≠no), `covered=yes` and `layer` mark intervals. They control elevation weighting (§2.4), terrain (no cut/fill under bridges, no terrain fill in cuttings), walls (guardrails on bridges) and self-crossing checks.
- **One-way:** respect it in routing (default). Ignore it for driving and AI (racing is on a closed road), but record it for traffic generation.

---

## 2. Elevation (section C)

### 2.1 Sources

**USGS 3DEP (US)**: public domain, "available, free of charge and without use restrictions" (VERIFIED, https://www.usgs.gov/3d-elevation-program/about-3dep-products-services).
- **Products:**
  - 1 m project DEMs (lidar, UTM).
  - **S1M** seamless 1 m (10 km tiles, COG; production since mid-2025).
  - 1/3 arc-second seamless (~10 m).
  - 1 arc-second (~30 m).
  - 1/9 arc-second (legacy, ~25% of CONUS).
  - 5 m IfSAR (Alaska) (VERIFIED, same URL).
  - S1M vertical datum is NAVD88 (GEOID18), NoData = −999999 (VERIFIED via data.usgs.gov summary, https://data.usgs.gov/datacatalog/data/USGS:4f34caac-f28f-4ea0-8d82-eafb2b8f9a5d).
- **Bucket `prd-tnm`** (public HTTPS, no auth). Live listing of `StagedProducts/Elevation/` on 2026-10-14 shows `1/, 13/, 19/, 1m/, 2/, S1M/, DSM/, LPC/, OPR/, …` (VERIFIED, https://prd-tnm.s3.amazonaws.com/?list-type=2&prefix=StagedProducts/Elevation/&delimiter=/).
  - 1/3″: `StagedProducts/Elevation/13/TIFF/current/n42w089/USGS_13_n42w089.tif` (1°×1° tiles named by NW corner) plus a **seamless VRT `StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt`** (VERIFIED, HTTP 200).
  - 1 m projects: `StagedProducts/Elevation/1m/Projects/<project>/TIFF/USGS_1M_<utmzone>_x<e>y<n>_<project>.tif`. Example for the Aurora IL test route: `…/1m/Projects/IL_19County_D24/TIFF/USGS_1M_16_x39y462_IL_19County_D24.tif`. Several overlapping projects exist, so pick the newest `publicationDate` (VERIFIED via TNM API, below).
  - S1M: `StagedProducts/Elevation/S1M/n04e13/n0490e1390/S1M_n0490e1390_<yyyymmdd>.tif` (projected-grid tile names), index in `S1M/FullExtentSpatialMetadata/S1M_Products.gpkg` (VERIFIED by listing).
  - **All three are COGs** (`LAYOUT=IFDS_BEFORE_DATA` in the GDAL ghost header) (VERIFIED by range-reading the first 400 bytes).
- **TNM Access API** to discover tiles: `GET https://tnmaccess.nationalmap.gov/api/v1/products?datasets=Digital%20Elevation%20Model%20(DEM)%201%20meter&bbox=minx,miny,maxx,maxy&prodFormats=GeoTIFF` returns `items[].downloadURL, publicationDate` (VERIFIED, live call returned 10 items for Aurora IL).
- **Point and raster services:**
  - `https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer`: capabilities `Image,Metadata,Catalog`, max 8000×8000 export, `getSamples` returns a value plus `resolution` (1 m at Aurora) (VERIFIED by live call).
  - EPQS point query `https://epqs.nationalmap.gov/v1/json?x=..&y=..&units=Meters&wkid=4326` (VERIFIED, returned 203.24 m at the Aurora start).
  - Good for spot checks. Too slow for 10k+ points, so use COG reads.
- **py3dep 0.19.0** (MIT):
  - `get_dem(geometry, resolution)` (static 10/30/60 m or dynamic service).
  - `get_map`.
  - `static_3dep_dem`.
  - `elevation_bycoords(coords, source='tep'|'tnm')` (10 m).
  - `elevation_profile(lines, spacing)` (10 m).
  - `get_dem_vrt`.
  - `check_3dep_availability(bbox)` (1/3/5/10/30/60 m).
  - `query_3dep_sources(bbox)`.

  (VERIFIED, https://docs.hyriver.io/autoapi/py3dep/py3dep/index.html.) Useful for availability checks. For 1 m data, read COGs directly; py3dep's dynamic service also reaches 1 m but resamples server-side.
- **Lidar point clouds:** `s3://usgs-lidar-public` (EPT, no auth, us-west-2) and `s3://usgs-lidar` (LAZ, Requester Pays) (VERIFIED, https://registry.opendata.aws/usgs-lidar/). Not needed for v1.

**Copernicus GLO-30 (global)**
- Buckets: `s3://copernicus-dem-30m/` (GLO-30), `s3://copernicus-dem-90m/`, region eu-central-1, COG (VERIFIED, https://registry.opendata.aws/copernicus-dem/).
- Tile naming: `Copernicus_DSM_COG_10_<N|S>dd_00_<E|W>ddd_00_DEM/Copernicus_DSM_COG_10_..._DEM.tif`, 1°×1° tiles. `tileList.txt` sits in the bucket. Ocean = no tile (assume 0). Some countries are not in the public release. **It is a DSM**: "represents the surface of the Earth including buildings, infrastructure and vegetation" (VERIFIED, https://copernicus-dem-30m.s3.amazonaws.com/readme.html).
- The Gorges d'Ollioules tile, `Copernicus_DSM_COG_10_N43_00_E005_00_DEM.tif`, exists and is a COG (VERIFIED by range read).
- **Accuracy:**
  - Absolute vertical < 4 m LE90.
  - Relative < 2 m (slope ≤ 20%), < 4 m (slope > 20%).
  - Vertical datum EGM2008 (EPSG:3855).

  (VERIFIED, https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM and https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_DEM_GLO30.)
- Licence text: see §6.

**France (first national adapter, because of the Ollioules test route)**
- Géoplateforme altimetry API: `GET https://data.geopf.fr/altimetrie/1.0/calcul/alti/rest/elevation.json?lon=..&lat=..&resource=ign_rge_alti_wld&zonly=true` returned `{"elevations":[65.47]}` (VERIFIED by live call). There's also `elevationLine` for profiles. **Limit: 5 req/s per IP** (VERIFIED, https://www.data.gouv.fr/dataservices/api-geoplateforme-calcul-altimetrique).
- Resources listed at `https://data.geopf.fr/altimetrie/resources?format=json`: `ign_rge_alti_wld` (RGE ALTI France), `ign_rge_alti_par_territoires` (RGE Alti 1 m with dynamic source/precision), `ign_lidar_hd_mnt_mono_wld` / `_multi_wld` (LiDAR HD DTM, with or without interpolation; bounding boxes show partial coverage), `IGNF_LIDARHD_MNX_WLD` (DTM/DSM/canopy height) (VERIFIED by live call).
- For grids: WMS-R `https://data.geopf.fr/wms-r` lists layers `ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES` (RGE ALTI 1 m) and `…HIGHRES.MNS`, and the format `image/geotiff` (VERIFIED from GetCapabilities). A GetMap request with `FORMAT=image/geotiff` returning float32 heights is **ASSUMED**; the test request was cut by the sandbox proxy. Spike it.
- RGE ALTI precision: 0.2–0.5 m where lidar was used, **~7 m in mountains surveyed by radar** (VERIFIED, https://developers.google.cn/earth-engine/datasets/catalog/IGN_RGE_ALTI_1M_2_0). So check the per-tile source; LiDAR HD may cover the route.
- Licence: Etalab Licence Ouverte 2.0 since 2021-01-01 (VERIFIED via https://www.actuia.com/actualite/les-donnees-publiques-de-lign-relatives-a-la-topographie-au-relief-et-a-la-visualisation-du-territoire-libres-et-gratuites/ and the Earth Engine page above). Attribution: "IGN – RGE ALTI®" (ASSUMED wording).
- Other national lidar (ASSUMED, add later as adapters with the same interface): UK EA LIDAR composite DTM (OGL), Switzerland swissALTI3D (free), Spain CNIG MDT05, Japan GSI DEM5A, Austria/Germany state DGM1, Norway Høydedata, NZ LINZ.

**Fallbacks**
- **AWS Terrain Tiles** `s3://elevation-tiles-prod` (us-east-1, `-eu` replica): bare-earth terrain heights tiled as `terrarium/{z}/{x}/{y}.png` and `geotiff/{z}/{x}/{y}.tif` (both VERIFIED HTTP 200). The attribution file is https://github.com/tilezen/joerd/blob/master/docs/attribution.md (VERIFIED, https://registry.opendata.aws/terrain-tiles/). It mixes SRTM, ETOPO, national DEMs and more, so the attribution list is long.
- **OpenTopography** `https://portal.opentopography.org/API/globaldem?demtype=COP30|SRTMGL1|NASADEM|AW3D30|…&south=&north=&west=&east=&outputFormat=GTiff&API_Key=` needs a key (VERIFIED, the keyless call returned "API Key required"). Free key: 50 calls/24 h for non-academics, 200 for academics; 450,000 km² limit for 30 m data (VERIFIED via search summary of https://www.opentopography.org/node/3158; numbers may change).
- **SRTM** (public domain) and **ALOS AW3D30** (JAXA, free with "©JAXA" credit; it's a DSM) (ASSUMED terms). **EU-DEM** is superseded by Copernicus DEM (ASSUMED). Not worth separate adapters, since OpenTopography covers them.
- Python helpers: `elevation` 1.1.4 (Apache-2.0; downloads SRTM via GDAL/make, awkward on Windows) and `SRTM.py` 0.3.7 (2021). **Not recommended.**

### 2.2 Reading DEMs in Python

- **rasterio 1.5.2** (BSD-3, Python ≥3.12, ships win_amd64 wheels with GDAL bundled) (VERIFIED, PyPI).
- Read COGs over HTTP:
  ```python
  env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                     CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.vrt",
                     GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                     VSI_CACHE=True, GDAL_CACHEMAX=512)
  with env, rasterio.open("/vsicurl/https://prd-tnm.s3.amazonaws.com/.../USGS_13_n42w089.tif") as src:
      win = rasterio.windows.from_bounds(*bounds_in_src_crs, transform=src.transform)
      arr = src.read(1, window=win, boundless=True, fill_value=np.nan)
  ```
  For Copernicus, use `/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/...`, or `/vsis3/` with `AWS_NO_SIGN_REQUEST=YES`.
- Build a **corridor raster once**: mosaic the needed tiles (`rasterio.merge.merge`, or a VRT via `gdal.BuildVRT` if `osgeo` is installed). Then `rasterio.warp.reproject(..., dst_crs=local_tmerc, resampling=Resampling.cubic)` to the local grid (1 m near the road for 3DEP 1 m, 5–10 m for GLO-30). Cache it as GeoTIFF under the user cache dir.
- Optional extras: `rioxarray` 0.23.0 / `xarray` (Apache-2.0) and `rio-tiler` 9.4.12 (BSD-3). `rio-cogeo` 7.0.4 only matters if we write COGs. Not needed.
- **Sampling:** `scipy.ndimage.map_coordinates(arr, [rows, cols], order=1 or 3)` on the local grid is fast for 100k points.

### 2.3 Accuracy problems to handle

1. **DSM bias (Copernicus, AW3D30):** trees (+10–30 m), buildings, cars and bridges are in the surface. Roads through forest come out bumpy and too high. Fix with a lower-envelope fit (asymmetric weights) and a wide smoothing window. Optionally sample a lateral band (±15 m) and take a low percentile.
2. **Resolution (30 m):** a narrow road on a steep side slope samples the hillside above and below it. Cross-road sampling at ±w/2 and a median helps. Mountain roads in gorges (Ollioules) are the worst case, so expect artefacts. Check against a known profile (Google Earth or IGN) as the M1 acceptance test.
3. **Cuttings and embankments:** even 1 m lidar is fine here, because the DTM includes the road bed. For 30 m data, OSM `cutting=yes` / `embankment=yes` intervals get lower weight.
4. **Bridges:**
   - In 1 m 3DEP (bare-earth DTM), bridge decks are usually **removed** (hydro-flattened), so the DTM shows the valley floor. You **must** interpolate across.
   - In Copernicus (DSM), the deck may or may not be in the surface. Interpolate across in both cases.
5. **Tunnels:** the DEM shows the mountain on top, so interpolate between the portals.
6. **Datums:** NAVD88/GEOID18 (3DEP) vs EGM2008 (Copernicus) vs NGF-IGN69 (France). Never mix sources inside one track without blending the offset. Pick one source per build, using the best one that covers the whole corridor; otherwise fill holes from the next source with a smooth offset correction.

### 2.4 Recommended profile pipeline

1. **Stations:** s every **1 m** along the smoothed centreline (§3.2).
2. **Raw samples:** z_raw(s) = median of DEM at offsets {−w/2, −w/4, 0, +w/4, +w/2} along the normal (5 samples, bilinear). With 30 m data, also record `z_low(s)`, the 20th percentile in a ±10 m lateral band.
3. **Weights w(s):**
   - 1 by default.
   - 0 on `bridge`/`tunnel`/`covered` intervals, extended by 10 m at each end (abutment and portal effects).
   - 0.3 on `cutting`/`embankment` for 30 m data.
   - 0 where the DEM is NoData.
4. **Robust smoothing**: Whittaker–Eilers smoother, a sparse penalised least-squares fit. Minimise `Σ w_i (z_i − y_i)² + λ Σ (Δ² z_i)²`, solved with `scipy.sparse.linalg.spsolve((W + λ DᵀD), W y)`. That's a banded, O(n) solve; 50k points take a few ms.
   - Zero weights make it **interpolate smoothly across bridges and tunnels automatically**.
   - For DSMs, iterate 3–5 times with **asymmetric weights**: w_i ← p if y_i > z_i, else 1−p, with p ≈ 0.1. This pulls the fit toward the lower envelope and rejects tree and building spikes.
   - λ sets the cut-off wavelength. Aim for about **150–250 m for GLO-30, 30–60 m for 1 m lidar, ~80 m for 10 m data** (ASSUMED starting values; tune on the test routes). The Georgia Tech method (cubic smoothing spline over a 10 m DEM) reached grade RMSE 0.2–0.6% (VERIFIED, https://transportation.ce.gatech.edu/projects/moves-matrix-modeling-tools/road-grade-for-use-in-energy-and-emissions-modeling), so smoothing-spline-class fits are good enough.
5. **Hard limits (driveability)**: a QP with **OSQP 1.1.3** (Apache-2.0, win wheels: VERIFIED PyPI). Variables are z. Objective as in step 4, with weights from the final asymmetric iteration. Constraints:
   - **Grade** |Δz/Δs| ≤ g_max: default 0.20, warn above 0.15. Real extremes such as Hardknott reach ~33%, so the limit is configurable (ASSUMED).
   - **Vertical curvature** |Δ²z/Δs²| ≤ 1/R_v,min with **R_v,min ≈ 150 m sag / 200 m crest** for mountain roads. Vertical acceleration v²/R at 90 km/h with R=200 m is about 3.1 m/s² ≈ 0.32 g. AASHTO K-values (crest K≈2 at 30 km/h, ≈7 at 50 km/h, where R_v = 100·K m) are ASSUMED from the Green Book, not fetched.
   - **Bridges:** optionally clamp the grade to ≤ 6% and enforce a linear or parabolic shape (Δ² ≈ 0) on long bridges.
   - Fallback without OSQP: `scipy.optimize.lsq_linear` with a grade-variable substitution (z = z0 + cumsum(g·Δs), bounds on g) via a LinearOperator, or just clamp-and-resmooth iterations.
6. **Output:** z_road(s). Also store `dz_err(s) = z_road − z_raw` (positive means fill, negative means cut). The terrain stage uses it, and it's a QA plot (fail the build if |dz_err| > 15 m over 200 m+ outside bridges and tunnels, which signals a bad DEM).
7. **Self-crossings:** where the route passes over itself (bridge over its own earlier part), assert z difference ≥ 5 m. Otherwise raise the bridge interval toward that clearance, using a constraint in the QP.

### 2.5 Fitting the terrain to the road (cut/fill)

- **Corridor terrain** = DEM reprojected to the local grid. For DSM sources, apply an optional canopy-removal filter: grey-scale morphological opening (`scipy.ndimage.grey_opening`, size ~ 3×3 px at 30 m) or the same asymmetric smoothing in 2D. ASSUMED useful; test it.
- **Road "legs":**
  1. For every terrain vertex p, find the nearest centreline stations with a cKDTree on the 1 m stations.
  2. Group the candidates within the influence radius by arc length: stations more than 2·(influence radius) apart in s belong to different legs, for example the two sides of a hairpin.
  3. For each leg i, compute the lateral distance d_i from the paved edge (≥0), the road edge height z_i (centreline z plus camber at the edge), and the constraint:
  ```
  lower_i = z_i − m_cut  · d_i      (cut batter, e.g. m_cut = 1.0  → 1:1)
  upper_i = z_i + m_fill · d_i      (fill batter, e.g. m_fill = 0.5 → 1V:2H)
  h(p) = clamp(DEM(p), max_i lower_i, min_i upper_i)
  ```
  If `max lower > min upper` (two legs at different heights closer than the batters allow), take the height-weighted midpoint and flag a **retaining wall** segment for the wall generator.
  4. Then apply a smoothstep blend over the outer 20–40 m to avoid creases: `h = lerp(h_clamped, DEM, smoothstep(d / D_blend))` only where the clamp was not active. Batter slopes are ASSUMED typical earthworks values (cut 1:1 in rock up to 1:1.5, fill 1:2).
- **Under the road:**
  - Set h = z_road − 0.15 m inside the paved polygon (or remove those terrain triangles), so the terrain never z-fights or pokes through.
  - The road mesh has a **skirt**: its outer verge edge drops 0.3–0.5 m below the terrain to hide gaps.
- **Bridges:** no clamp to the road height under the deck. Instead enforce `h ≤ z_road − clearance (5 m)` within the deck footprint, plus abutment ramps at the ends.
- **Tunnels (v1):** treat them as a deep cutting (open to the sky). The terrain clamp lowers the mountain to the batters. **v2:** a real tube (walls + ceiling mesh) with the terrain left intact and holes at the portals.
- **v2 option:** a global sparse least-squares solve. Minimise `Σ w (h−DEM)² + μ|∇²h|²` with hard (or very heavy) constraints at road-edge cells (scipy.sparse, about 1M unknowns at 2 m over 2×2 km, solved with conjugate gradient). It gives smoother, more natural blends at junction stubs and hairpins.

---

## 3. Geometry (section D)

### 3.1 Projection and axes

- `pyproj.Transformer.from_crs("EPSG:4326", f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs", always_xy=True)`, where (lat0, lon0) is the route start. pyproj 3.8.0 is MIT, Python ≥3.12, win wheels (VERIFIED, PyPI).
- TM scale error ≈ x²/(2R²): 5×10⁻⁶ at 20 km, 3×10⁻⁵ at 50 km. That's negligible, and better than UTM (zone edges, large coordinates, non-unity scale k=0.9996).
- Heights: use DEM orthometric heights minus `z_start` (so the start sits at y≈0). AC uses float32 vertices, so keep coordinates within a few km of the origin for precision. If the track is over 20 km, consider putting the origin at the route midpoint (ASSUMED that this matters at the cm level: float32 has ~1 mm resolution at 8 km and ~4 mm at 32 km).
- **AC axes (ASSUMED, needs the M0 spike).** AC is Y-up, and forum evidence says start objects need "Z pointing in the start direction, Y pointing up" and that Blender FBX exports use "-Z forward, Y up" (VERIFIED as community statements, https://assettocorsamods.net/threads/first-track-spawning-in-the-air.972/latest, https://blenderartists.org/t/from-blender-to-assetto-corsa/1413256). Proposed mapping: **X_ac = East, Y_ac = Up, Z_ac = −North**, which is right-handed and Y-up like FBX/OpenGL. Spike test: put an asymmetric "N" arrow mesh plus a dummy at a known east offset, load it in CM's showroom or the game, and compare with `map.png` orientation. The section-A research owns the final answer.

### 3.2 Centreline

1. Concatenate route edge coordinates (dedupe repeated nodes) and project them.
2. **Junction corners:** at vertices with a turn angle above 30° that are real junctions (the route changes way at an intersection), insert a **circular fillet**: radius R = clamp(lane-based turning radius ~8–15 m, ≤ tangent length limit = 0.45 × min(adjacent segment lengths)/tan(θ/2)). This prevents the smoother from rounding a T-junction turn into a 40 m arc.
3. **Smoothing:** resample at 2 m, then run a Whittaker smoother on x(s) and y(s) (the same sparse solver as §2.4) with a **3rd-difference penalty** (curvature continuity). Iterate: raise the weights where the deviation from the original polyline is above a tolerance (default **min(1.5 m, 0.25 × width)**), so the curve stays inside the real road.
   - **Catmull-Rom (centripetal, α=0.5)** interpolates the OSM nodes exactly, so it keeps their kinks. Use it only to densify before smoothing.
   - **B-spline `scipy.interpolate.splprep(s=…)`** works too, but picking the smoothing parameter for varying node density is harder.
   - **Clothoids:** `pyclothoids` 0.2.0 (win wheels VERIFIED, licence not stated on PyPI; upstream Clothoids library is BSD-2, ASSUMED) does G1 Hermite clothoid fits through points/headings. It's a nice v2 refinement for curvature-continuous AI lines.
4. **Resample** uniformly at **1.0 m** (mesh stations can be decimated adaptively: 1 m where |κ| > 1/50, 2–4 m on straights). Compute T (tangent), N (left normal = rotate T +90° in the EN plane), and κ = signed curvature (finite differences on the smoothed spline). Store s, x, y, z, T, N, κ, width_L, width_R, bank.
5. **Validation:**
   - Minimum radius ≥ (half-width + shoulder + 1 m). Otherwise the inner edge offset folds.
   - Increase smoothing locally, or clamp the inner offset (see 3.6).

### 3.3 Road ribbon

- **Cross-section per station, left to right:**
  - wall base L
  - verge L (grass/gravel)
  - shoulder L (asphalt, slightly different texture or `1ROAD` as well)
  - edge line
  - lanes…
  - centre line
  - …lanes
  - edge line
  - shoulder R
  - verge R
  - wall base R
- Painted lines are optional, as a separate mesh 2 mm above the road with alpha (phase 2).
- **Camber / superelevation (ASSUMED typical design values):**
  - Crown 2.0–2.5% on straights (both halves slope outward).
  - On curves, superelevation e = clamp(v_design²/(127·R) − f, 0, e_max). Simpler: e = clamp(k/R, 0, e_max), with e_max 6–8% (mountain roads often less), applied as a rigid rotation about the centreline.
  - Transition (runoff) over 30–60 m, using smoothstep in s.
  - No superelevation inside junction zones.
  - Put an `exaggerate_bank` setting at 0 by default.
- **Triangulation:** quad strips between consecutive stations → 2 triangles each, split into **chunks of ~250 m** per surface type. That limits vertex counts (keep each mesh well under 65,535 vertices in case the kn5 uses 16-bit indices: **ASSUMED**, check in the section-A research) and helps culling.
- **UVs:**
  - u = lateral metres / texture width (for example a 7 m asphalt texture across the road, or 0–1 across each lane).
  - v = s / tile_length (for example 10 m).
  - On tight curves the inner and outer edges differ in length, so asphalt stretches slightly. That's acceptable. For verges, use per-edge arc length.
  - To avoid float precision loss with large v, **restart v at each chunk boundary** on a multiple of tile_length.
- **Normals:** compute from the mesh (area-weighted) and keep them smooth across chunk seams by computing them before splitting.

### 3.4 Terrain corridor heightfield

- **Width:**
  - Driveable world = road + verges + walls.
  - Visual terrain: high-detail band out to 60–100 m from the road edge, medium detail to ~300 m, low detail to ~1–2 km, then a **backdrop** (a ring of far mountains from a coarse DEM to 5–10 km, or a skybox panorama). All ranges ASSUMED; tune against the AC triangle budget.
- **Resolution rings (ASSUMED):** 1–2 m cells within 30 m of the edge, 4 m to 100 m, 10 m to 300 m, 30 m to 2 km.
- **Implementation:** a quadtree of regular grid tiles (64×64 cells per tile) whose level comes from the distance to the road polygon (`shapely.distance` vectorised, or a distance transform on a raster mask: `scipy.ndimage.distance_transform_edt`). T-junctions between levels get stitched with skirts or edge-collapse. Simplest: skirts.
- **Alternative:** `pydelatin` 0.4.0 (MIT, Delatin TIN from a heightmap, max-error driven) (VERIFIED, PyPI). Feed it the blended 1–2 m heightfield and a max error of ~0.1 m near the road and ~1 m far away. It gives an adaptive TIN in one call; cut the road footprint triangles out afterwards. A good v1 shortcut, but test it.
- **Road hole:** remove terrain triangles whose centroid lies inside the paved polygon (`shapely.contains_xy`, vectorised). The road skirt hides the gap. A constrained Delaunay (exact stitching) can come later; avoid `triangle` (licence), and if needed use `mapbox_earcut` on strips.
- **Physics vs visual:** the driveable area is road + verge + near terrain (a physical surface such as 1GRASS). Far terrain is visual only. Naming belongs to section A.

### 3.5 Hairpins and near-overlaps

- **Terrain:** the multi-leg clamp in §2.5 handles it (one constraint per leg, intersected intervals).
- **Road mesh inner edge on tight radii:** if R < half-width the offset curve folds. Detect it by the sign of `cross(edge_i+1 − edge_i, T)` flipping. Fix it by collapsing the folded vertices to the curvature centre (a fan) and by limiting curvature in §3.2 smoothing.
- **Corridor overlap between legs** (two legs closer than 2 × (half-width + verge)):
  - Verges overlap, so build the verge geometry from the **unioned corridor polygon** rather than per-leg ribbons.
  - Trim each leg's verge strip with `shapely.difference(verge_i, road_polygon_other_legs)`.
  - If the legs are at different heights (switchback stacks), the area between them becomes terrain at the clamp heights, with a retaining-wall flag if the slope is above the cut/fill limits.

### 3.6 Walls and barriers

1. `corridor = unary_union([buffer(leg_centreline, half_width + shoulder + verge, cap_style="flat", join_style="round") for leg in legs_by_layer])`, computed **per layer group**: ground, plus each bridge or tunnel interval separately.
2. Wall lines = `corridor.exterior` + `corridor.interiors` (holes appear inside hairpin loops). Densify them to 2 m, cut short gaps at junction stubs (or close them in v1), then remove slivers below 5 m.
3. **Heights:** base = terrain/road height at the wall line − 0.5 m; top = base + 1.5–3 m (invisible physics wall) or a 0.8 m visible guardrail on bridges and drop-offs (where terrain falls away by more than 3 m beyond the verge).
4. **Start and end:** close the A and B ends with a cap wall placed 30–50 m before A and after B, so the run-up and run-out are inside the world.
5. **World bounds:** a big box of walls at the corridor terrain extent, plus a catch plane below the lowest terrain, so cars never fall forever.
6. OSM `barrier=guard_rail` ways near the route (Overpass `way[barrier=guard_rail](around.route:15)`) can switch to the visible guardrail model in phase 2.
7. Generated walls never cross the road, because they come from the union boundary.

### 3.7 Libraries (all with Windows wheels, VERIFIED on PyPI 2026-10-14)

| Library | Version | Licence |
|---|---|---|
| numpy | 2.5.4 | BSD-3 |
| scipy | 1.18.1 | BSD-3 |
| shapely | 2.2.0 | BSD-3 |
| pyproj | 3.8.0 | MIT |
| rasterio | 1.5.2 | BSD-3 |
| trimesh | 5.1.1 | MIT, pure Python |
| mapbox-earcut | 2.1.0 | ISC (ASSUMED) |
| osqp | 1.1.3 | Apache-2.0 |
| pyclothoids | 0.2.0 | |
| pydelatin | 0.4.0 | MIT |
| osmnx (optional) | 2.1.1 | MIT |
| networkx (optional) | 3.7 | BSD-3 |
| osmium (optional, offline .pbf) | 4.3.1 | BSD-2 |
| pyrosm (optional) | 0.20.0 | MIT |
| geopandas (optional) | 1.2.0 | BSD-3 |

**Do not use `triangle` (20250106)**: the PyPI wrapper is LGPL-3, but Shewchuk's code is non-commercial without a licence (VERIFIED above).

---

## 4. Scenery (later, M6)

- **OSM inputs.** Overpass in the same corridor polygon:
  - `nwr[landuse~"^(forest|meadow|grass|farmland|orchard|vineyard|residential|industrial)$"]`
  - `nwr[natural~"^(wood|scrub|heath|grassland|bare_rock|scree|water)$"]`
  - `way[building]`
  - `node[natural=tree]`
  - `way[natural=tree_row]`
  - `way[barrier~"guard_rail|wall|fence"]`

  Use `out geom` and assemble multipolygon relations (shapely `polygonize`, or OSMnx `features_from_polygon`).
- **Trees:**
  - **Poisson-disk** sampling (Bridson, numpy) inside wood/forest polygons minus `buffer(road, half_width + verge + 2 m)`. Density about 1 per 25–60 m² in forest, sparser for scrub. Random scale ±20% and rotation, seeded from a hash of the route ID (deterministic).
  - Outside polygons, add an optional sparse "default vegetation" density by biome. Natural=tree nodes are placed as-is.
  - Render as **cross-quad billboards** (2–3 intersecting planes, alpha-tested) batched into one mesh per 250 m chunk, so draw calls stay low. The AC shader choice (ksTree / ksPerPixelAT) belongs to section A.
- **Buildings:**
  - Extrude footprints with height = `height`, else `building:levels × 3 m + 1 m`, else 6 m.
  - Base = min terrain height under the footprint − 0.5 m.
  - Roofs: flat, or simple gabled for small footprints.
  - Triangulate caps with mapbox-earcut. Merge per chunk.
  - Keep buildings outside the wall corridor, or optionally within it as a visual only, with no collision.
- **Grass:** ground textures by landuse (grass, farmland, rock for `bare_rock`/`scree`, slope > 35° → rock). CSP grass FX is optional (section A).
- **Textures** (all CC0, no attribution required, credit appreciated):
  - **ambientCG**: "All ambientCG assets are provided under the Creative Commons CC0 1.0 Universal License"; commercial use allowed; suggested credit "Created using <asset> from ambientCG.com, licensed under CC0 1.0" (VERIFIED, https://docs.ambientcg.com/license/). It has an API (`/api/v2/…`, downloads CSV) (VERIFIED as existing, same page). Use it for asphalt, gravel, grass, rock, dirt and concrete.
  - **Poly Haven**: assets CC0, no credit required. Site content (logos, renders) is not CC0, and scraping needs permission (use their API, which has its own API Terms) (VERIFIED, https://polyhaven.com/license). Use it for textures, HDRIs (sky reference) and a few CC0 tree and rock models. Render the trees offline into billboard atlases.
  - Ship the chosen textures **in the repo** (converted to DDS) rather than downloading at runtime. Record each asset ID in a `THIRD_PARTY.md`.

---

## 5. Local tool packaging on Windows (F, local part)

- **Python:** **3.12 or 3.13**. The current rasterio, pyproj and scipy releases require ≥3.12 (VERIFIED, PyPI `requires_python`). PyInstaller 6.22.3 supports <3.16 (VERIFIED).
- **Package layout:** `driftmaps` library + CLI entry point `driftmaps = driftmaps.cli:app`.
  - CLI: **Typer 0.27.3** (MIT).
  - Commands: `driftmaps route --pins "lat,lon;lat,lon;…" -o route.json`, `driftmaps build route.json --name "…" [--ac-root …]`, `driftmaps build --start … --end …` (route + build in one), `driftmaps serve` (local UI), `driftmaps doctor` (finds AC, checks network and caches).
- **Local web UI:** **FastAPI 0.143.1 + uvicorn 0.54.0** (MIT/BSD-3).
  - `driftmaps serve` binds **127.0.0.1** only, on a random free port, and opens the browser.
  - Endpoints:
    - `POST /api/snap`
    - `POST /api/route` (returns `route.json` + GeoJSON + length)
    - `POST /api/build` (background task)
    - `GET /api/build/{id}` (progress via polling or SSE)
    - `GET /api/geocode?q=` (server-side proxy with caching, so policy compliance lives in one place)
  - Static frontend: a single `index.html` with no build step. **Leaflet 1.9.x** (BSD-2, VERIFIED, https://raw.githubusercontent.com/Leaflet/Leaflet/main/LICENSE) or **MapLibre GL JS** (BSD-3, VERIFIED, https://raw.githubusercontent.com/maplibre/maplibre-gl-js/main/LICENSE.txt). Vendor the JS into the package, so it works offline and needs no CDN.
- **Basemap tiles:**
  - **CARTO:** keyless `https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png` currently returns 200 (VERIFIED by live request). But the **current CARTO Basemap Terms say "Customer must always use the Basemap Services with Customer's own unique API keys"** (§8), keyless requests "may receive a visible watermark" (§3.d), free use is 5M requests/month non-commercial and 1M commercial (§12), attribution must be "© OpenStreetMap contributors, © CARTO", and server-side proxying/caching is prohibited (§9.c.iii) (VERIFIED, https://carto.com/legal/basemap-terms/ and https://carto.com/basemaps). So for the "driftmaps look", support CARTO with a user-supplied key.
  - **Default to OpenFreeMap's "Dark" style** (MapLibre vector): free, "no limits on the number of map views or requests", no registration or keys, commercial use allowed, attribution required (OpenFreeMap optional, OpenMapTiles + OSM required) (VERIFIED, https://openfreemap.org/). Style URL `https://tiles.openfreemap.org/styles/dark` (ASSUMED path pattern, per their Quick Start).
  - **Stadia Alidade Smooth Dark** works keyless only from localhost, with strict rate limits (VERIFIED, https://docs.stadiamaps.com/authentication/). A good fit for the local tool, not for a hosted site.
- **Geocoding:**
  - **Nominatim** (OSMF servers): max 1 req/s (4/min for long-running scripts), an identifying User-Agent (stock library UAs are rejected), results must be cached, **client-side autocomplete is forbidden**, attribution required (VERIFIED, https://operations.osmfoundation.org/policies/nominatim/). So only on Enter.
  - **Photon** (`https://photon.komoot.io/api/?q=…&limit=5`, reverse at `/reverse?lon=&lat=`) is fair-use and throttles heavy use (VERIFIED, https://photon.komoot.io/). It's built for typeahead. Apache-2.0 (VERIFIED, https://raw.githubusercontent.com/komoot/photon/master/LICENSE). Debounce at 300 ms, minimum 3 characters, cache in memory and on disk.
  - Both are OSM-based, so attribution is covered by the map footer.
- **Finding the AC install** (checked in order, the first one that validates wins; validation = `<root>/AssettoCorsa.exe` and `<root>/content/tracks/` exist):
  1. `--ac-root` flag / `DRIFTMAPS_AC_ROOT` env / config file (`platformdirs.user_config_dir("driftmaps")/config.toml`; platformdirs 4.13.0, MIT).
  2. **Steam:**
     - Read the Steam root from the registry `HKCU\Software\Valve\Steam\SteamPath` (ASSUMED, widely used), else `HKLM\SOFTWARE\WOW6432Node\Valve\Steam\InstallPath` (VERIFIED as common practice, e.g. https://docs.rs/crate/steamlocate/1.2.1), via `winreg`.
     - Parse `<steam>/steamapps/libraryfolders.vdf` with **`vdf` 3.4** (MIT): numbered entries with `path` and an `apps` map of appid → size (VERIFIED in format summaries; v1 files have paths only, so also scan each library).
     - For each library, look for `steamapps/appmanifest_244210.acf` (AC app ID **244210**, VERIFIED, https://steamcommunity.com/app/244210). Read `installdir` (normally `assettocorsa`) → `<lib>/steamapps/common/<installdir>`.
  3. **Content Manager hint:** CM stores the AC root under the key **`_ac_root`** in its `ValuesStorage` (VERIFIED, https://github.com/gro-ove/actools/blob/master/AcManager.Tools/Managers/AcRootDirectory.cs). The file is `%LOCALAPPDATA%\AcTools Content Manager\Values.data`, **compressed by default** (VERIFIED, `AcManager/App.xaml.cs` → `ValuesStorage.Initialize(FilesStorage.Instance.GetFilename("Values.data"), …, DisableValuesCompression)`). The compression format is not documented, so reading it is optional or best-effort (ASSUMED). Actools is Ms-PL (VERIFIED).
  4. Ask in the UI and save the answer to config.
- **Distribution:**
  - **Phase 1:** `uv tool install git+https://github.com/<user>/MapsAssetoCorsa` (or `pipx install …`). One command, an isolated venv, all deps from wheels (no compiler, because rasterio, pyproj, shapely, scipy and osqp all ship win_amd64 wheels: VERIFIED).
  - **Phase 2:** **PyInstaller 6.22.3** (GPLv2+ with the bootloader exception, so shipping a closed or open exe is allowed) in **one-folder** mode. One-file mode unpacks GDAL/PROJ data on every start and triggers more antivirus false positives (ASSUMED). Needs hooks for rasterio, pyproj and shapely data files (pyinstaller-hooks-contrib covers most; ASSUMED, test it). Nuitka 4.3 is AGPL for the compiler, but its output isn't covered; it's an alternative but more complex (ASSUMED).
- **Caches:** `platformdirs.user_cache_dir("driftmaps")` → `overpass/`, `dem/<source>/`, `geocode/`, `builds/`. Use requests-cache 1.3.3 (BSD-2) or a simple sha256-keyed file cache.

---

## 6. Legal (section G)

**OSM / ODbL**
- A generated AC track (meshes, textures, AI line) is best treated as a **Produced Work**: "a work (such as an image, audiovisual material, text, or sounds) resulting from using … the Contents" (VERIFIED, ODbL 1.0 definitions, https://opendatacommons.org/licenses/odbl/1-0/). The OSMF guideline's test is whether "the published result is meant for extracting the original data". If not, it's a Produced Work (VERIFIED, https://osmfoundation.org/wiki/Licence/Community_Guidelines/Produced_Work_-_Guideline). Whether a 3D game asset qualifies is not addressed explicitly. ASSUMED yes, because it isn't designed for data extraction. **Not legal advice.**
- **§4.3 notice** when publicly using a Produced Work, e.g. "Contains information from OpenStreetMap, which is made available here under the Open Database License (ODbL)" (VERIFIED, ODbL §4.3).
- OSMF attribution guidelines: credit "OpenStreetMap" (e.g. "Map data from OpenStreetMap"); "© OpenStreetMap contributors" is still acceptable; link to https://www.openstreetmap.org/copyright. For **games**, attribution can go in a splash screen, credits or menu (VERIFIED, https://osmfoundation.org/wiki/Licence/Attribution_Guidelines). For an AC track: put it in `ui/ui_track.json` `description` (shown in CM and the game UI), plus a `LICENSE-DATA.txt` / `README.txt` in the track folder.
- **§4.6 share-alike:** if a Produced Work is made from a **Derivative Database** (our filtered, merged OSM ways + elevation-enriched profile could count), anyone who publicly uses it must offer the derivative database or "a file containing all of the alterations … or a description of the method" in machine-readable form, free over the internet (VERIFIED, ODbL §4.6). **Cheap compliance:** ship `data/route_osm.geojson` (route geometry with OSM way IDs, the tags used and the `osm_base` timestamp) inside every generated track. Keep the generator open-source, since that *is* the method.
- For the purely local, personal tool there is no "Public Use" until the user shares the track (ASSUMED). The notices are still auto-included, because the original's consent text says tracks are the user's to share.
- Suggested `ui_track.json` description footer: `Road data © OpenStreetMap contributors, ODbL (openstreetmap.org/copyright). Elevation: <source credit>. Generated with <tool>.`

**Copernicus DEM (GLO-30)** (VERIFIED verbatim from the licence PDF, https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/resources/license/License-COPDEM-30.pdf, Article 6)
- Rights: reproduction, distribution, communication to the public, and adaptation/modification/combination, free of charge, worldwide (Art. 4–5).
- Unmodified data: "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."
- **Adapted or modified** data (our case): **"produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved"**
- Required disclaimer sentence: **"The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30"**
- The AWS registry also suggests the citation "Copernicus Digital Elevation Model (DEM) was accessed on DATE from https://registry.opendata.aws/copernicus-dem." (VERIFIED, https://registry.opendata.aws/copernicus-dem/).
- The licence PDF shown is for the GLO-30 instance. The GLO-30 Public release on AWS points to the same Copernicus licence page (VERIFIED, registry link). Some countries are excluded from the public release (VERIFIED, readme).

**USGS 3DEP**
- Public domain: "All 3DEP products are available, free of charge and without use restrictions" (VERIFIED, https://www.usgs.gov/3d-elevation-program/about-3dep-products-services). No credit is required. A courtesy credit is recommended: "Elevation: U.S. Geological Survey, 3D Elevation Program".
- Lidar on AWS: US Government Public Domain (VERIFIED, https://registry.opendata.aws/usgs-lidar/).

**IGN (France)**
- Etalab Licence Ouverte 2.0 (VERIFIED, see §2.1). It requires attribution of the source and the date of last update (ASSUMED from the general Etalab 2.0 terms).

**Others**
- AWS Terrain Tiles: per-source attribution list (VERIFIED as linked, https://github.com/tilezen/joerd/blob/master/docs/attribution.md).
- OpenTopography: cite each dataset's source.

**Map tiles and geocoders (UI only, not in the track)**
- CARTO: key required by the terms; attribution "© OpenStreetMap contributors, © CARTO"; no server-side proxy or cache (VERIFIED, §5).
- OpenFreeMap: attribution OpenMapTiles + OSM (VERIFIED).
- Nominatim / Photon: as in §5.

**Kunos / Assetto Corsa**
- No official Kunos modding or trademark policy was found (searched, nothing authoritative).
- Recommended approach (ASSUMED, standard nominative fair use):
  - Use the name only to state compatibility, with exactly the original's style of wording: "Not affiliated with or endorsed by Kunos Simulazioni. Assetto Corsa is a trademark of Kunos Simulazioni, named only to state compatibility."
  - No AC logos, no game screenshots used as branding, and no "Assetto Corsa" in the product name.
- **Never ship Kunos assets.** That means no textures, shaders, car or track files, `ksEditor` binaries, or `.kn5` content copied from the game install. Generated tracks must use only our own and CC0 textures.
- The local tool may *read* the user's install (to find the folder, or to use CM for testing) but must not copy game files into outputs or repos.
- Content Manager (Ms-PL) and CSP are third-party. Don't redistribute CSP; just document it as a user-installed optional extra.

---

## 7. Open items for spikes

1. **AC axis/handedness**, plus kn5 index width (16- vs 32-bit) → section-A research, M0 spike.
2. Overpass reliability from the user's network: build the retry/mirror logic and record failure rates. The sandbox test got a 504 and timeouts.
3. IGN WMS-R `image/geotiff` float download for RGE ALTI / LiDAR HD in the Ollioules bbox (43.13–43.16N, 5.84–5.88E).
4. Tune smoothing λ, g_max and R_v on the two test routes. Compare the GLO-30 profile vs the IGN 1 m profile for Ollioules, and the 3DEP 1 m vs 1/3″ profile for Aurora.
5. pydelatin vs a regular-grid quadtree for terrain triangle budgets.
6. PyInstaller one-folder build with rasterio/GDAL on Windows.
