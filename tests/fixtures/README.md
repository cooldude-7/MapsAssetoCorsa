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
