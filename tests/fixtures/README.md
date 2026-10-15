# Test fixtures

Small, committed snapshots of open data so the tests run offline. Each section lists its
files, where they came from and the licence/attribution.

## DEM windows (`dem/`, used by `tests/test_elevation_sources.py`)

96 x 96 px windows cut on 2026-10-15 from the live cloud-optimised GeoTIFFs, keeping the source
CRS, transform, NoData and float32 values, re-written as tiled (32 x 32) deflate GeoTIFFs so
windowed block reads are exercised. Each file's `SOURCE_URL` / `SOURCE_WINDOW` tags record
exactly where it was cut from.

| File | Source | Centre point | Height at centre |
|---|---|---|---|
| `hrdem_toronto_1m.tif` | NRCan HRDEM mosaic 1 m DTM, `hrdem-mosaic-1m/8_2-mosaic-1m-dtm.tif` (EPSG:3979) | -79.3957, 43.7046 | 156.425 m |
| `usgs13_aurora.tif` | USGS 3DEP 1/3 arc-second, `USGS_13_n42w089.tif` (EPSG:4269) | -88.29689, 41.71712 | 203.173 m |
| `glo30_ollioules.tif` | Copernicus GLO-30 DSM, `Copernicus_DSM_COG_10_N43_00_E005_00_DEM.tif` (EPSG:4326) | 5.85, 43.14 | 63.927 m |

`hrdem_forks_5m.tif` (used by `tests/test_build.py`, the offline end-to-end track build) is
different: a 351 x 329 px window at **5 m** around the first leg of the Forks of the Credit
test route (pins (43.80410, -79.99231) -> (43.81060, -79.98185), local box E -300..1150 m,
N -250..1050 m from the first pin), read on 2026-10-15 from the NRCan HRDEM 1 m mosaic
`hrdem-mosaic-1m/8_2-mosaic-1m-dtm.tif` (EPSG:3979) with 5 x 5 **average** resampling, heights
rounded to 0.01 m, written as a 64 x 64 tiled deflate GeoTIFF (float32, predictor 3, NoData
-32767; 248 KB). Tags `SOURCE_URL`, `SOURCE_BOUNDS` and `RESAMPLING` record the cut. Its
heights range 285-451 m (the Credit River valley).

Licences and attribution:
- HRDEM: Contains information licensed under the Open Government Licence - Canada
  (https://open.canada.ca/en/open-government-licence-canada). Source: Natural Resources Canada.
- 3DEP: U.S. Geological Survey, 3D Elevation Program; public domain.
- GLO-30: produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence
  and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights
  reserved. The organisations in charge of the Copernicus programme by law or by delegation do
  not incur any liability for any use of the Copernicus WorldDEM-30.

## OSM extracts (`osm/`, used by `tests/test_geo_*.py`)

| File | What | Fetched |
|---|---|---|
| `forks_of_the_credit.json` | Raw Overpass API JSON (`out body geom`) for the query `driftmaps.geo.overpass.build_query(FORKS_PINS, buffer_m=400)`: every drivable way within 400 m of the pin chain (43.80410, -79.99231) -> (43.81060, -79.98185) -> (43.83180, -79.95940) along Forks of the Credit Road, Caledon, Ontario (twisty, with a hairpin and side-road junctions). 37 ways, 36 KB, `timestamp_osm_base` 2026-10-15T13:51:13Z. | 2026-10-15 from maps.mail.ru/osm/tools/overpass |

Licence and attribution: map data (c) OpenStreetMap contributors, available under the Open
Database Licence (ODbL) 1.0, https://www.openstreetmap.org/copyright. The file is an unmodified
extract of the OSM database, so it stays under the ODbL.

## DSM/DTM pair (`dem/hrdem_snake_dsm_1m.tif`, `dem/hrdem_snake_dtm_1m.tif`, used by `tests/test_elevation_dsm.py`)

Two 64 x 64 px (64 x 64 m) windows cut on 2026-10-15 at the **same** pixel window
(`col_off=245907,row_off=491515,width=64,height=64`, EPSG:3979, 1 m) from the NRCan HRDEM 1 m
mosaic surface model `hrdem-mosaic-1m/8_2-mosaic-1m-dsm.tif` and terrain model
`hrdem-mosaic-1m/8_2-mosaic-1m-dtm.tif` (both on `canelevation-dem.s3.ca-central-1.amazonaws.com`).
Exact float32 source pixels, NoData -32767, written as 32 x 32 tiled deflate GeoTIFFs
(predictor 3; 11.7 KB and 9.6 KB). Tags `SOURCE_URL` / `SOURCE_WINDOW` record the cut.

The window is centred on an OSM building 28 m east of Snake Road, Flamborough (Hamilton), Ontario
(-79.88979, 43.31024). It holds the roof, deciduous canopy and open ground: DSM - DTM ranges
0.01-19.65 m and is >= 3 m on 45 % of pixels. Reference values (also returned by the live
pipeline on 2026-10-15): roof pixel DSM 159.447 / DTM 150.955 m, tallest tree 170.473 / 150.825 m,
bare ground 149.926 / 149.905 m.

Licence and attribution: Contains information licensed under the Open Government Licence -
Canada (https://open.canada.ca/en/open-government-licence-canada). Source: Natural Resources
Canada, High Resolution Digital Elevation Model (HRDEM) - CanElevation Series.

## OSM barriers + bridge (`osm/mill_st_waterdown_barriers.json`, used by `tests/test_barriers.py`)

Raw Overpass API JSON (`out body geom`, 5.4 KB, `timestamp_osm_base` 2026-10-15T16:30:45Z),
fetched on 2026-10-15 from maps.mail.ru/osm/tools/overpass with the query

```
[out:json][timeout:60];
(way["barrier"~"^(guard_rail|jersey_barrier|retaining_wall|wall)$"](around:200,43.3306,-79.8866);
 way["highway"~"^(primary|secondary|tertiary|unclassified|residential|trunk)$"](around:200,43.3306,-79.8866););
out body geom;
```

It holds Mill Street South in Waterdown (Hamilton, Ontario), where it drops off the Niagara
Escarpment: three road ways in driving order 34632407 -> 24486613 (`bridge=yes`, ~30 m) ->
24486612, and three mapped `barrier=guard_rail` ways (427465490, 427465491, 427465492, each
tagged `height=1`), all on the south/west side of the road (the driver's right heading
south-east). None of the three test tracks has a mapped guard rail (docs/research/scenery.md
§5), so this nearby road is the real-data fixture for barrier matching.

Licence and attribution: map data (c) OpenStreetMap contributors, available under the Open
Database Licence (ODbL) 1.0, https://www.openstreetmap.org/copyright. The file is an unmodified
extract of the OSM database, so it stays under the ODbL.

## Land cover (`landcover/landcover2020_snake_30m.tif`, used by `tests/test_vegetation.py`)

A 48 x 48 px (1.44 x 1.44 km) window cut on 2026-10-15 at
`col_off=128174,row_off=146858,width=48,height=48` from NRCan's Land Cover of Canada 2020 COG
`https://datacube-prod-data-public.s3.ca-central-1.amazonaws.com/store/land/landcover/landcover-2020-classification.tif`
(EPSG:3979, 30 m, uint8, NoData 0), centred on the same Snake Road house as the DSM/DTM pair
above (-79.88979, 43.31024). Exact source pixels, written as a 16 x 16 tiled deflate GeoTIFF
(1.9 KB); tags `SOURCE_URL` / `SOURCE_WINDOW` record the cut. Classes present: 1 (needleleaf
forest), 5 (broadleaf deciduous), 6 (mixed forest, most common), 8 (shrubland), 15 (cropland),
16 (barren), 17 (urban), 18 (water). The class codes are in NRCan's
[ClassIndex_v2.pdf](https://datacube-prod-data-public.s3.ca-central-1.amazonaws.com/store/land/landcover/ClassIndex_v2.pdf).

Licence and attribution: Contains information licensed under the Open Government Licence -
Canada (https://open.canada.ca/en/open-government-licence-canada). Source: Natural Resources
Canada, Land Cover of Canada 2020.

## OSM + Microsoft buildings (`osm/buildings_snake_rd.json`, `osm/buildings_snake_rd_ms.geojsonl.gz`, used by `tests/test_buildings.py`)

`osm/buildings_snake_rd.json`: raw Overpass API JSON (`out body geom`, 12.7 KB,
`timestamp_osm_base` 2026-10-15T17:04:49Z), fetched on 2026-10-15 from
maps.mail.ru/osm/tools/overpass with the query

```
[out:json][timeout:60];
(way["building"](around:120,43.31024,-79.88979);
 relation["building"]["type"="multipolygon"](around:120,43.31024,-79.88979);
 way["highway"~"^(secondary|tertiary|unclassified|residential)$"](around:120,43.31024,-79.88979););
out body geom;
```

It holds three `building=yes` ways beside Snake Road, Flamborough (Hamilton, Ontario): 832043557
(the 10-corner house at the centre of the `dem/hrdem_snake_{dsm,dtm}_1m.tif` lidar windows),
832043679 and 832043713, plus the Snake Road (24233890, 121184920) and Old York Road (23810117)
ways. No building has `height` or `building:levels` (as on all three test roads, scenery.md §3).

Licence and attribution: map data (c) OpenStreetMap contributors, available under the Open
Database Licence (ODbL) 1.0, https://www.openstreetmap.org/copyright. The file is an unmodified
extract of the OSM database, so it stays under the ODbL.

`osm/buildings_snake_rd_ms.geojsonl.gz`: 6 lines (670 bytes gzipped) cut unmodified from the
Microsoft Global ML Building Footprints release 2026-08-13, tile `RegionName=Canada/quadkey=
030223132/part-00000-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz` (listed in
`https://bfppub.blob.core.windows.net/%24web/2026-08-13/dataset-links.csv`), keeping every
feature whose outline touches lon -79.89149..-79.88809, lat 43.30904..43.31144 (about +-140 m
around the same point). Each line is a GeoJSON `Feature` with a `Polygon` and properties
`height` (m, -1 = unknown) and `confidence`. Three of them trace the three OSM buildings above;
the other three are not in OSM (the merge rule keeps those).

Licence and attribution: Microsoft Global ML Building Footprints, licensed under the Community
Data License Agreement - Permissive, Version 2.0 (CDLA-Permissive-2.0,
https://cdla.dev/permissive-2-0/), which allows sharing the data with that licence notice.
