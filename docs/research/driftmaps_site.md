# What driftmap.us says about how it works

Read 2026-10-15 from the public pages https://www.driftmap.us/ and
https://www.driftmap.us/guides/real-road-to-assetto-corsa.html. Quotes are theirs; anything
under "Inferred" is our reading, not something they state.

## Stated

- **Input:** start and end coordinates (copied from Google Maps) plus optional waypoints. "Two
  coordinates in. A track mod out." Routes "follow the real road network" (OpenStreetMap).
- **Road:** path and width from OpenStreetMap; asphalt "traced from aerial imagery", with
  "real elevation and camber".
- **Elevation:** terrain models, with lidar "where it exists" (USGS 3DEP, Copernicus DEM).
- **Imagery:** USGS aerial imagery (public domain) is a listed data source, used to trace the
  asphalt edge and place scenery. Street View is a reference only on the guide page; the home page
  says signals, markings, poles and trees come "from imagery and Street View" for the
  intersection/area product.
- **Build steps:** queued → "Building the road" → "Checking the track" → packaging. One build per
  user in the queue. Most tracks arrive the same day, by email with a one-click Content Manager
  install link.
- **Output:** zip with a `.kn5`, `ui/ui_track.json`, preview, outline, pit and start spawns.
  **Optional `.blend` file (Blender 3.6+).** Plus servers use base-game cars only.
- **Products:** roads (priced per km, up to 50 km, 2–5 km recommended) and an
  "Intersections/Area" mode priced per hectare (up to ~50 ha). A track library of past builds.
- **Limits they state:** Google Maps / Google Earth roads and 3D buildings cannot be exported into
  the game.

## Inferred

- **Their pipeline is Blender-based** (the optional `.blend` download), with a Blender KN5
  exporter or FBX → ksEditor. That, plus "Checking the track" (likely launching AC), explains
  the hours-long queue. Ours writes KN5 directly in Python, so a build can take minutes.
- **Aerial imagery is what makes the scenery read like Google Maps:** real road width traced
  from it, and very likely the photo itself draped over distant terrain for true ground colours.
- Their "markings, poles, signals" for intersections suggest object detection on imagery.

## What we take from it

| Idea | Where it goes in our plan | Free data |
|---|---|---|
| Drape aerial imagery on terrain (far ground), keep tiling textures near the road | M6 (simple version could come earlier) | US: USGS NAIP (~0.6 m, public domain). France: IGN BD ORTHO (Licence Ouverte). Canada: varies by province/city open data; check licence per area. Esri/Google/Bing/Mapbox imagery terms forbid this use. |
| Real road width from imagery instead of OSM defaults | After M1 (refinement) | Same imagery; start with OSM `width`/`lanes` |
| Intersection/area tracks (a whole neighbourhood, free roam) | New milestone after M5 | OSM roads + buildings |
| Optional editable export for modders | Later: GLB/FBX export via trimesh (we already have the meshes) | — |
| Painted lines, cut-slope dirt/rock, barriers on drop-offs, trees from canopy data | M1 (lines, slope colouring), M6 (rest) | OSM tags, lidar DSM−DTM, ESA WorldCover / Meta canopy height |
