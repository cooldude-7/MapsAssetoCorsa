# Scenery research (Ontario): trees, buildings, guardrails, textures

Status: v1, 2026-10-15. Scope: **Canada / Ontario only**, for the three test roads (Snake Rd and
Sydenham Rd in Hamilton, Forks of the Credit Rd in Caledon). This answers the user's feedback:
"trees and the rest of the scenery so it looks accurate", and "guardrails only where they exist".
**VERIFIED** = checked live or in a primary source on the date given. **ASSUMED** = reasoned, not
yet tested. Scripts and raw outputs were kept out of the repo; the numbers below are their results.

**TL;DR**
- **Where the trees are, and how tall:** use NRCan HRDEM **lidar DSM - DTM** (1 m, OGL-Canada). It
  covers 100 % of all three roads, and the module is done: `sample_surface_elevation` and
  `canopy_height_grid` in `driftmaps.elevation.sources`.
- **What kind of tree:** use NRCan **Land Cover of Canada 2020** (30 m COG, OGL-Canada).
- **Buildings:** OSM footprints, plus **Microsoft Global ML Building Footprints**
  (CDLA-Permissive-2.0, *not* ODbL) where OSM is missing. Heights come from DSM - DTM.
- **Trees in AC:** batched crossed-quad cards in the KN5 with Kunos' own `ksTree` settings, so
  they work without CSP. CSP `[TREES]` comes later; it needs `.bin` models made by Content
  Manager's tree converter.
- **Guardrails:** no free automatic source has them for these roads. Read OSM `barrier=guard_rail`,
  add a per-track override file, and put parapets on OSM bridges. **Mapillary can't do it.**

---

## 1. Lidar surface model (DSM) → canopy and building heights

### 1.1 What exists (VERIFIED live, 2026-10-15)

I searched the NRCan STAC (`https://datacube.services.geo.ca/stac/api/search`) with each track's
bbox. **Every item that has a `dtm` COG also has a `dsm` COG.** The mosaics also have
`hillshade-dsm`.

| Track | `hrdem-mosaic-1m` | `hrdem-lidar` items (newest first) | `hrdem-mosaic-2m` |
|---|---|---|---|
| Snake Rd | `8_2-mosaic-1m` (dsm+dtm) | TRCA-GTA_2023-1m (2023-05-11), NRCAN-HALTON-1m (2018), ON-2017-LakeErie-1m | `8_2-mosaic-2m` |
| Sydenham Rd | `8_2-mosaic-1m` | NRCAN-Hamilton_Niagara_2021_2-1m (2021-05-25), ON-2017-LakeErie-1m | `8_2-mosaic-2m` |
| Forks of the Credit | `8_2-mosaic-1m` | TRCA-GTA_2023-1m (2023-05-11), NRCAN-PEEL-1m (2016) | `8_2-mosaic-2m` |

- **File.** `https://canelevation-dem.s3.ca-central-1.amazonaws.com/hrdem-mosaic-1m/8_2-mosaic-1m-dsm.tif`
  is a COG on the same grid as the DTM: EPSG:3979, 1 m, 500,000² px, 512 px tiles, float32,
  NoData -32767, overviews ×2-16. Each lidar project's DSM is `hrdem-lidar/<id>-dsm.tif`.
  The collection licence is `OGL-Canada-2.0` (VERIFIED).
- **How the DSM is made.** It is a "highest hit" surface. From the HRDEM Mosaic Product
  Specification, Ed. 1.1 (2022-11-15): "The DSM datasets are generated from the LiDAR data with
  only the highest points … triangulates the LiDAR point cloud in a temporary TIN … The small
  areas without data … are filled by the pixel values of the DTM."
  ([PDF](https://download-telecharger.services.geo.ca/pub/elevation/dem_mne/HRDEMmosaic_mosaiqueMNEHR/HRDEM_Mosaic_Product_Specification.pdf))
  So DSM ≥ DTM by construction. The lowest DSM - DTM seen anywhere was 0.0 m.
- **Mosaic vs. projects.** At every route node the 1 m mosaic DSM and the newest-first
  lidar-project DSM agree exactly (median |Δ| = 0.00 m). The mosaic is enough.
- **Coverage.** 1,310 sample points (route nodes, and 25 m N/S/E/W of each) all got a DSM and a DTM.

### 1.2 What it shows on our roads (VERIFIED, DSM - DTM)

| Track | On the road: canopy > 3 m | 25 m off: canopy > 3 m, p50 / p90 / max | Tree tops ≥ 3 m within ±120 m (2 m grid, 3×3 local max, road ±8 m excluded) | Of those within 40 m of the road | Tree-top height p50 / p95 |
|---|---|---|---|---|---|
| Snake Rd | 12 % (overhanging) | 64 %, 7.2 / 19.2 / 38.6 m | 5,234 (63/ha) | 1,680 | 16.4 / 28.2 m |
| Sydenham Rd | 10 % | 30 %, 0.2 / 11.3 / 20.1 m | 2,874 (34/ha) | 1,209 | 11.4 / 23.9 m |
| Forks of the Credit | 18 % | 80 %, 11.7 / 20.3 / 28.7 m | 9,173 (90/ha) | 2,706 | 18.0 / 28.7 m |

These tree-top counts are a *lower bound* on visible crowns. A 2 m grid merges neighbouring
crowns. At 1 m, with a height-dependent window (§7), expect roughly 1.5× more (ASSUMED).

**Bonus for the bridge builder (VERIFIED):** the DSM sees bridge decks; the DTM sees the creek.
DSM - DTM along each OSM `bridge=yes` way:

| Bridge | Length | Deck above valley floor (max) |
|---|---|---|
| Sydenham Rd, way 26200368 | 114 m | **11.1 m** (deck falls 156.8 → 147.6 m) |
| Snake Rd, way 23810128 | 20 m | 7.8 m |
| Forks, way 67757089 | 22 m | 3.2 m |
| Forks, ways 1538960657 / -659 | 9 m each | 0.0-0.1 m (culverts; kept in the DTM) |

So the "floating road sheet" on Sydenham really is about 10 m up. The height is right. What's
missing is the structure: abutments, piers or a solid fill, parapets, and the creek valley visible
under the deck.

### 1.3 API (implemented in `src/driftmaps/elevation/sources.py`)

```python
from driftmaps.elevation.sources import sample_surface_elevation, canopy_height_grid
dsm = sample_surface_elevation(lon, lat)       # ElevationSample; NaN outside Canada
chm = canopy_height_grid(lon_grid, lat_grid)   # ElevationSample shaped like the grid;
                                               # .heights = DSM - DTM >= 0 (m), NaN = no lidar
license_lines = chm.attributions               # DSM + DTM sources (OGL-Canada) for LICENSE-DATA.txt
```

- **Chains.** `SURFACE_CHAINS`: `hrdem-1m-dsm` → `hrdem-lidar-dsm` → `hrdem-2m-dsm` in CA,
  CA-border and US-border. The chain is empty elsewhere, so it returns NaN. There is deliberately
  no GLO-30 fallback: a 30 m radar DSM can't place single trees.
- **Pairing.** `canopy_height_grid` pairs each DSM with the DTM from the **same collection**
  (`DSM_TO_DTM`). A lidar DSM is never subtracted from a coarser fallback DTM.
- **Cache.** Blocks are cached like the DTM, under `.cache/elevation/hrdem-1m-dsm/`. A ±120 m
  corridor costs about 10-15 MB of DSM blocks per track (ASSUMED from the 45 MB cached for all
  three tracks plus the earlier probes). Once cached, sampling the ±120 m corridor took 4-7 s
  per track.
- **Caveats.**
  - DSM - DTM also contains buildings, bridges, power lines, poles and parked vehicles. Mask them
    with footprints and the road corridor before looking for tree tops.
  - The lidar is 2021/2023, flown in May. That's early leaf-out, so canopy *height* is reliable
    but canopy *density* may be under-read (ASSUMED).
  - The lidar is up to 5 years older than OSM. Trees cut since then will still appear.

---

## 2. Forest type: NRCan "Land Cover of Canada 2020" (VERIFIED live)

- **Source.** [open.canada.ca dataset ee1580ab-…](https://open.canada.ca/data/en/dataset/ee1580ab-a23d-4f86-a09b-79763677eb47),
  Natural Resources Canada, Open Government Licence - Canada. 30 m, from Landsat; part of
  NALCMS 2020. The geo.ca record reports 86.9 % overall accuracy (not re-checked).
- **Access.** One COG:
  `https://datacube-prod-data-public.s3.ca-central-1.amazonaws.com/store/land/landcover/landcover-2020-classification.tif`.
  - VERIFIED: `LAYOUT=COG`, LZW, EPSG:3979, 30 m, 190,001 × 160,001 px, 512 px tiles, uint8,
    NoData 0, colour table embedded.
  - **A windowed read works.** One 512 px tile covers 15 km, so a whole track is 1-4 HTTP range
    requests. Read it with the existing `_CogRaster` block cache, or plain `rasterio` +
    `/vsicurl/`, using nearest-neighbour sampling.
  - A WMS also exists (`datacube.services.geo.ca/web/landcover.xml`). We don't need it.
- **Classes**
  ([ClassIndex_v2.pdf](https://datacube-prod-data-public.s3.ca-central-1.amazonaws.com/store/land/landcover/ClassIndex_v2.pdf)):

  | Value | Class |
  |---|---|
  | 1 | temperate/sub-polar needleleaf forest |
  | 2 | sub-polar taiga needleleaf forest |
  | 5 | temperate/sub-polar broadleaf deciduous forest |
  | 6 | mixed forest |
  | 8 | shrubland |
  | 10 | grassland |
  | 11-13 | polar classes |
  | 14 | wetland |
  | 15 | cropland |
  | 16 | barren |
  | 17 | urban and built-up |
  | 18 | water |
  | 19 | snow and ice |

- **Shares around each route (VERIFIED, route bbox ± 500 m):**

  | Track | 1 conifer | 5 deciduous | 6 mixed | 8 shrub | 15 crop | 17 urban | 18 water | other |
  |---|---|---|---|---|---|---|---|---|
  | Snake Rd | 0.3 % | 13 % | 41 % | 3 % | 13 % | 24 % | 0.7 % | 4 % |
  | Sydenham Rd | 0.2 % | 7 % | 9 % | 1 % | 57 % | 24 % | 0.6 % | 1 % |
  | Forks of the Credit | 8.5 % | 15 % | 36 % | 1 % | 27 % | 10 % | 1.1 % | 2 % |

- **How to use it.** CHM decides *where* trees are and *how tall*. Land cover decides *what kind*.
  It is only 30 m, so never use it to place trees.

---

## 3. Buildings

**OSM footprints (VERIFIED, Overpass via maps.mail.ru, osm_base 2026-10-15T16:00Z, route bbox ±200 m):**

| Track | Buildings in bbox | Within 150 m of route | Within 50 m | With `height` / `building:levels` |
|---|---|---|---|---|
| Snake Rd | 281 | 97 | 44 | 0 |
| Sydenham Rd | 487 | 105 | 35 | 0 |
| Forks of the Credit | 11 | **6** | 1 | 0 |

**Microsoft footprints.** These count the same corridors in Microsoft Global ML Building
Footprints, Canada quadkey tiles (VERIFIED, dataset-links.csv of 2026-08-13):

| Track | Within 150 m | Within 50 m | With ML height > 0 |
|---|---|---|---|
| Snake Rd | 87 | 36 | 80 |
| Sydenham Rd | 99 | 34 | 87 |
| Forks of the Credit | **55** | 23 | 53 |

So OSM is fine in Hamilton but **misses most houses at Forks**. Microsoft fills the gap.

- **Access (VERIFIED).** The README says hosting moved in 2026:
  1. Fetch `https://bfppub.blob.core.windows.net/%24web/2026-08-13/dataset-links.csv`. Its columns
     are `Location, QuadKey, Url, Size`, with `Location = Canada`.
  2. Download the zoom-9 quadkey tile(s) that the route bbox touches: `030223132` (29.3 MB,
     Snake + Sydenham) and `030223130` (39.3 MB, Forks).
  3. Each tile is `.csv.gz`, but the contents are GeoJSON lines. Each polygon has the properties
     `height` (m, -1 = none) and `confidence`.
  4. Stream, bbox-filter and cache the corridor subset as JSON. Downloading and scanning both
     tiles took about 17 s.
- **Licence (VERIFIED).** The global set is **CDLA-Permissive-2.0**
  ([README](https://github.com/microsoft/GlobalMLBuildingFootprints)). The older
  [CanadianBuildingFootprints](https://github.com/microsoft/CanadianBuildingFootprints) (2019,
  one zip per province, Ontario 808 MB unzipped) is **ODbL**. Prefer the global set: it's newer,
  in smaller tiles, and has heights.
  - ASSUMED reading of CDLA-P-2.0: generated meshes are "Results", free to share. Put an
    attribution line and the licence URL in LICENSE-DATA.txt anyway.
- **Merge rule (ASSUMED).** Use OSM first. Add a Microsoft polygon only if it overlaps no OSM
  building by more than 30 % of its own area.
- **Heights.** Use DSM - DTM inside the footprint, eroded by 1 m:
  - p90 → roof top, median → eave.
  - Fallbacks, in order: Microsoft `height` > 0; OSM levels × 3 m + 1; otherwise 6 m.
  - VERIFIED example: the OSM house in the test fixture reads 8.5 m.
  - Roof shape (ASSUMED): if p90 - median < 1 m it's flat; otherwise it's a gable along the long
    axis.

---

## 4. Trees in AC

### 4.1 Without CSP: KN5 card trees (works everywhere, CSP renders them too)

- **Shader + material.** Use Kunos' own tree settings, read from the NJMP track's persistence
  file and posted by LilSKi on 2017-07-11 on
  [assettocorsamods "SHADER setting-up trees"](https://assettocorsamods.net/goto/post?id=5088)
  (VERIFIED as doc, 3rd-party). Settings: `SHADER=ksTree`, `ALPHATEST=1`, `ALPHABLEND=0`,
  `DEPTHMODE=0`, `ksAlphaRef 0.5`, `ksAmbient 0.4`, `ksDiffuse 0.25`, `ksSpecular 0`,
  `ksSpecularEXP 1`, `ksEmissive 0`.
  - In our writer that is
    `Material(shader="ksTree", blend_mode=BlendMode.OPAQUE, alpha_tested=True, depth_mode=DepthMode.NORMAL)`,
    with one `txDiffuse` (RGBA, DXT5/BC3 DDS with mips, via `encode_dds`).
  - Ambient and diffuse need tuning against our terrain brightness. The thread says
    "around 0.2-0.3".
  - `ksPerPixelAT` with the same properties is the fallback (same thread).
  - AlphaToCoverage only helps with MSAA. Try it in the spike, but don't default to it.
- **Geometry.**
  - 2 crossed vertical quads per tree; a 3rd for big deciduous trees.
  - AC has no double-sided materials ("faces must be mirrored", same forum), so add the reversed
    triangles. They **share the same 4 vertices**, giving 8 verts and 24 indices per tree, which
    is 400 B in the KN5.
  - Synthetic test: 20,000 trees = 8.0 MB raw, **2.3 MB deflated** (VERIFIED by compression
    test; tangents set to a constant, which `ksTree` ignores (ASSUMED)).
- **Normals (ASSUMED, spike).** ksEditor's "auto-normals" need the pivot at the tree base, but
  that happens at editor time. We bake our own: normal = normalize(vertex - crown centre)
  blended toward +Y (the forum: tree-wall normals "must always be pointed vertical"). This gives
  soft, volumetric lighting with no hard card edges.
- **Batching and LOD.**
  - Use one mesh per ~250 × 250 m cell and band, all trees in one atlas material. ≤ 65,535 verts
    allows up to 8,000 trees per mesh; aim for about 2,000 so culling stays tight.
  - **Near band** (0-150 m from the road footprint): individual trees, `lod_out` 700 m,
    `cast_shadows` on (spike the cost).
  - **Far band** (150-500 m): every 3rd tree, cards scaled ×1.5 as "clumps", `lod_out` 1,500 m,
    `cast_shadows` off.
  - No trees inside the wall corridor. Cards have no collision, and a car must never drive
    "through" a tree.
  - Name the meshes `trees_<band>_<cell>`, with no digit prefix, so they are visual only.
- **Draw calls.** A 4 km road × ±500 m is about 40-60 tree meshes, with about 10-20 in view.
  That's well inside the ~1,000 DIP budget (ac_format.md §6).

### 4.2 With CSP: what it adds (VERIFIED doc: [CSP trees](https://cup.acstuff.club/docs/csp/tracks/trees))

- **Config.**
  - `[TREES] LIST_0 = trees/trees.txt`, and optionally `SURFACE_MATERIALS = <terrain material>`
    (snaps trees to the surface). It goes in the track's `extension/ext_config.ini`
    (ac_format.md §7).
  - Lines look like `tree: name.bin; x, y, z` or
    `tree: t.bin; pos=x, y, z; angle = 180; size = 1; width = 1; color = 2, 2, 2`.
  - `configure:` lines set random `size variance`, `width variance`, `angle variance`,
    `color variance`, `brightness variance`, `fake shadow` and `seed`.
  - It needs CSP ≥ 0.1.79. CSP instances the trees and adds LODs plus distant billboards:
    "hundreds of thousands of trees" should be fine.
  - A text list works at runtime. `COMPILED_LIST` needs the VAO bakery, which is optional.
- **We can't generate the models yet.** A `.bin` is an archive (7-Zip/WinRAR) holding meshes,
  textures and a `tree.ini`.
  - It is made only by Content Manager's GUI tree converter, from an FBX: LOD A/B/C at about
    10k/6k/3k triangles, with diffuse + normal maps.
  - There is no public spec of the mesh format inside, so Python can't write it today. (VERIFIED:
    the docs and wiki don't document it.) Whether CSP ships stock `.bin` trees a track could
    reference is unknown.
- **Avoid double trees.** CSP shader replacements can't hide meshes, but they can set `LOD_OUT`
  per mesh ([shader replacements](https://cup.acstuff.club/docs/csp/general/shader-replacements),
  VERIFIED key list). So `[SHADER_REPLACEMENT_0_CARDS] MESHES = trees_* ; LOD_OUT = 0.01` hides
  our cards only when CSP is present and the CSP trees exist (ASSUMED; spike).
- **GrassFX** (`[GRASS_FX] GRASS_MATERIALS = <terrain grass material>`, ac_format.md §7) is
  still the cheapest big visual win for the meadows and crop fields between the trees.

---

## 5. Guardrails only where they really exist

- **(a) OSM: none (VERIFIED).** There is no `barrier=guard_rail` (nor jersey_barrier) anywhere
  in the three route bboxes ±200 m, matching what the user saw. The only barriers are a hedge
  143 m from Snake Rd, and a gate and fences near Sydenham Rd. Keep reading
  `barrier=guard_rail|jersey_barrier` ways within 15 m of the route (side = sign of the lateral
  offset), so they appear once someone maps them.
- **(b) Mapillary: not usable (VERIFIED, [API docs](https://www.mapillary.com/developer/api-documentation)).**
  - `map_features` holds only `traffic_sign` and `point`. All 42 point values are listed on the
    [points page](https://www.mapillary.com/developer/api-documentation/points); the only
    barrier is `construction--barrier--temporary`, and there is no guard rail.
  - A [Mapillary forum thread](https://forum.mapillary.com/t/dataset-to-detect-guard-rails/5922)
    says line features (guard rails) aren't extracted.
  - The `/{image_id}/detections` endpoint returns per-image segmentation polygons in image
    pixels. Turning those into road-side 3D positions needs camera poses and multi-view fusion:
    fragile, and a scraping-scale job.
  - Access: a client token from the developer dashboard. Limits: 60k entity calls/min, 10k
    search calls/min, 50k tiles/day.
  - Licence: imagery is CC BY-SA 4.0 (secondary sources; Mapillary's licence help page returned
    403). The terms for redistributing derived data are unclear, but Mapillary does explicitly
    allow using it to **edit OSM**.
  - Coverage near Hamilton: not checked (needs a token).
- **(c) Ontario open data: nothing found (VERIFIED negative, 2026-10-15).**
  - ArcGIS Online search for "guide rail Hamilton", "guiderail Hamilton", "guide rail Caledon",
    "guiderail Peel" and "guardrail Ontario" found no guide-rail layer.
  - The Open Hamilton catalogue API is behind a Cloudflare challenge for scripts.
  - The MTO publishes provincial-highway assets only (ASSUMED). Our three roads are municipal or
    regional.
- **(d) Lidar.** A 1 m DSM can't reliably see a 0.7 m high, 0.3 m wide rail (ASSUMED). Classified
  point clouds might, but they're heavy and out of scope.
- **Recommendation.**
  1. Default: **no guardrails** anywhere unless a source below says so.
  2. Read OSM `barrier=guard_rail` automatically.
  3. Add a per-track **override file** next to `route.json`: `guardrails.json`, a list of
     `{"side": "left|right", "from_m": s0, "to_m": s1}` in route distance. Later, the map picker
     can draw these. The user has driven these roads and knows where the rails are.
  4. Suggest the user adds them to OSM (iD editor plus Mapillary photos, which Mapillary allows
     for OSM editing). Every future build then gets them under the ODbL we already carry.
  5. Bridges: always build parapets or railings on OSM bridge ways (ASSUMED: real bridges here
     have them).

---

## 6. Textures we can ship

- **In the repo today (VERIFIED).** `ac/textures.py` already generates procedural textures
  (numpy + Pillow) and encodes BC1/BC3 DDS with full mips. We own them outright.
- **CC0 libraries (VERIFIED via their public APIs, 2026-10-15).**
  - **ambientCG** (CC0, geo.md §4): atlases `Foliage001-008`, `LeafSet001-030`, `PineNeedles001`
    (PNG/JPG, 1K-8K) and `Bark001-015`.
  - **Poly Haven** (CC0; its API has separate API terms): tree models `fir_tree_01`,
    `pine_tree_01`, `pine_sapling_*`, `fir_sapling*`, `tree_small_02` and `island_tree_01-03`;
    shrubs `shrub_01-04`. Using a 3D model as a card needs an offline render to an impostor atlas
    (Blender, dev time). The same models are the natural input for CSP `.bin` trees.
- **Decision.**
  - **v1: procedural tree atlas generated by us.** One 1024² RGBA atlas holding 2 deciduous
    crowns, 1 mixed, 2 conifers and 1 shrub. Each is built from layered noisy leaf clusters, with
    a top-lit gradient baked in and alpha cut at a threshold. The atlas is BC3 with mips:
    1.4 MB raw, about 1.2 MB zipped (ASSUMED). No licence work and deterministic.
  - **v2:** composite ambientCG Foliage/LeafSet cut-outs into the same atlas layout for more
    detail. Credit them in LICENSE-DATA.txt and list them in THIRD_PARTY.md.
  - **Buildings:** procedural siding, brick and shingle textures from Pillow (one
    512-1024 BC1 atlas).

---

## 7. Decisions for the builders

| Topic | Decision |
|---|---|
| **Canopy/height data** | `canopy_height_grid(lon, lat)` on a **1 m** local grid (AC XZ → lon/lat via `geo.project`) over the band from the road footprint out to 150 m (near) and 500 m (far, sample at 2 m). Attribution: `chm.attributions` (OGL-Canada, NRCan HRDEM). Outside Canada: NaN, so emit no lidar trees. |
| **Tree positions** | Mask out: road footprint, the wall corridor, building footprints buffered 2 m, water (land cover 18), and bridges. Tree tops are local maxima of the CHM with height ≥ 3 m (≥ 1.5 m in land cover 8/14 → shrubs). Window: 3 m if height < 8 m, 5 m for 8-15 m, 7 m above 15 m. Then fill gaps: anywhere CHM ≥ 3 m with no top within 4 m gets Poisson-disk points (r = 4.5 m). Crown radius = 0.6 × distance to the nearest top, clamped to 1.5-6 m. Card height = CHM at the top. Base = **our terrain mesh** height − 0.3 m (not the raw DTM; cut/fill changed it). Seed = hash(route id). |
| **Tree species** | Land Cover 2020 class at the tree, nearest-neighbour: 1/2 → 90 % conifer; 5 → 90 % deciduous; 6 → 50/50; 8/14 → shrub, plus 20 % small deciduous; 15/10/17/other → 80 % deciduous. Atlas variant chosen at random within the type. |
| **Tree mesh** | KN5, visual only. Names `trees_near_<cell>` / `trees_far_<cell>`, 250 m cells. 2 crossed quads with shared verts and mirrored indices (8 verts and 24 indices per tree), baked hemisphere normals, constant tangent. Material `ksTree`: alpha_tested, blend OPAQUE, depth NORMAL, ksAlphaRef 0.5, ksAmbient 0.4, ksDiffuse 0.25, ksSpecular 0, ksSpecularEXP 1, one BC3 atlas. Near: `lod_out` 700 m, shadows on. Far: every 3rd tree ×1.5, `lod_out` 1,500 m, shadows off. ≤ 2,000 trees per mesh. |
| **Budgets** | Track ZIP ≤ 28 MB. Today the tracks are 10.3 / 10.3 / 14.0 MB zipped, so scenery gets **≤ 12 MB zipped**. Trees: cap **40,000 per track** (about 4.6 MB zipped). Buildings ≤ 400 per track (< 0.5 MB). Textures: tree atlas 1024² BC3 (~1.4 MB), building atlas ≤ 1024² BC1 (~0.7 MB). If the cap bites, thin far trees first. |
| **Buildings** | OSM footprints, plus Microsoft Global ML tiles (CDLA-Permissive-2.0) where OSM is missing (overlap < 30 %). Height from CHM p90 inside the footprint eroded 1 m; eave = median. Fallbacks: MS height → OSM levels × 3 + 1 → 6 m. Flat roof if p90 − median < 1 m, otherwise a gable along the long axis. Base = min terrain under the footprint − 0.5 m. Visual only. Keep them outside the wall corridor. |
| **Guardrails** | Default none. Sources: OSM `barrier=guard_rail|jersey_barrier` within 15 m of the route, plus the per-track `guardrails.json` override (`side`, `from_m`, `to_m`). Bridges always get parapets. No Mapillary. |
| **CSP** | v1: KN5 cards only, plus GrassFX on the terrain grass material. v2: convert 3-4 Poly Haven CC0 trees to `.bin` once with CM's tree converter, ship them, and emit `extension/trees/trees.txt` from the **same** placement list. Hide the cards under CSP with `[SHADER_REPLACEMENT_…] MESHES = trees_* ; LOD_OUT = 0.01` (spike). |
| **Textures** | v1 procedural (Pillow) tree and building atlases. v2 ambientCG CC0 foliage composites. Never Google/Bing/Esri/Mapbox content. |
| **Licences → LICENSE-DATA.txt** | NRCan HRDEM (OGL-Canada); NRCan Land Cover 2020 (OGL-Canada: "Contains information licensed under the Open Government Licence - Canada"); OSM (ODbL, already there); Microsoft Building Footprints (CDLA-Permissive-2.0, https://cdla.dev/permissive-2-0/); ambientCG (CC0, credit appreciated). |
| **Spikes (in game)** | T1: `ksTree` vs `ksPerPixelAT` look, baked normals, and alpha-test edges with and without MSAA/A2C. T2: FPS with 20k vs 40k cards and near-band shadows on and off. T3: CSP `LOD_OUT` override hides the cards. T4: the CSP `.bin` converter round trip. |

### Open questions
- Does CSP ship stock tree `.bin` models a track can reference, so we could skip the CM converter?
- Mapillary coverage and licensing for derived data near Hamilton. Moot unless we revisit
  guardrail detection.
- Ontario's own land cover products (SOLRIS / Ontario Land Cover Compilation, OGL-Ontario,
  15 m) might separate hedgerows and treed swamp better. Not evaluated here.
