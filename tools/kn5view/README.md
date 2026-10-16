# kn5view: see a built track without Assetto Corsa

`render.py` converts a track's KN5 to a GLB (cached next to the KN5), loads it into three.js in
headless Chromium and writes one PNG per view: driver views along the route, aerial views and
top-down plans. Loading the scene takes about a second, and each view takes well under a second
after the first, so ask for all your views in one run.

Run it from any checkout of the repo; the `driftmaps` package comes from that checkout's venv:

    uv run python tools/kn5view/render.py TRACK [views...] --out DIR

`TRACK` is a built track folder (the one with `route.json`, `ai/fast_lane.ai` and the `.kn5`) or
a `.kn5` file. The output file names say what each view is.

## Views

| What | Flags |
|---|---|
| Driver view at a station | `--station 2984` (metres from the A gate; repeatable), or `--stations 0,500,1000` |
| ...lane, direction, eye | `--side right\|left` (default: the route's driving side), `--look forward,back,left,right` (comma list = several views), `--height 1.2`, `--pitch -2` |
| Aerial view of a place | `--at LAT,LON` or `--at-xyz X,Y,Z` (AC coordinates), with `--aerial 120` (metres above it), `--tilt 35` (degrees from straight down), `--heading 0` (direction the camera faces, degrees from north) |
| Top-down plan (orthographic, north up) | `--topdown LAT,LON --size 300` (image width in metres), or `--topdown-xyz X,Z` |

`--at` and `--topdown` can be repeated. Lat/lon is converted with the origin in `route.json`.
Top-down PNGs get a scale bar and a north arrow, and their file names end in the metres per
pixel (e.g. `topdown_43.4477_-79.8776_250m_0.195mpp.png`). Vertical cards (trees) are edge-on
from straight above, as in AC, so use `--at ... --tilt 40` to see them from the air.

## Rendering options

- `--res 1280x720`, `--fov 60`, `--fog-near 250 --fog-far 3000`
- `--shadows`: sun shadows (sun from `data/lighting.ini`), slower.
- `--show-collision`: draws the hidden physics walls (`1WALL_...`) in translucent red.
- `--hide REGEX` / `--only REGEX`: hide or keep meshes by name, e.g. `--hide '^far'`, `--only '^1ROAD|^paint'`.
- `--double-sided`: draw every face from both sides (to tell winding bugs from missing faces).
  By default faces are culled like AC, which has no double-sided materials.
- `--no-lod`: ignore the meshes' `lod_in`/`lod_out` (by default a mesh is drawn only while the
  camera is within its LOD band, measured to its bounding-sphere centre, roughly as AC does).
- `--reconvert`: rebuild the cached GLB; `--glb-only`: just write the GLB (open it in Blender or
  any glTF viewer); `--verbose`: print the page's console.

## Examples (No 2 Side Road test build)

    T=/tmp/claude-0/-home-user-MapsAssetoCorsa/c1e65687-537c-507b-b23f-20abf8c39e66/scratchpad/no2/dm_no2_test
    uv run python tools/kn5view/render.py $T --stations 0,2984,7282 --look forward,back --out shots
    uv run python tools/kn5view/render.py $T --at 43.4477,-79.8776 --aerial 150 --tilt 40 --out shots
    uv run python tools/kn5view/render.py $T --topdown 43.4477,-79.8776 --size 250 --out shots

(43.4477, -79.8776 is the Appleby Line x Britannia Road roundabout.) Run `--help` for all flags.

Look at the PNGs with the Read tool.

## How it works

- `kn5glb.py`: KN5 -> GLB with numpy and Pillow. AC's frame (X east, Y up, Z south) and the
  repo's counter-clockwise winding are glTF's, so geometry is copied as is. `txDiffuse` becomes
  the base colour (DDS BC1/BC3 decoded by Pillow). Alpha-tested materials (`alpha_tested`,
  `ksTree`, `..._AT`) become alpha-masked with ksAlphaRef as cutoff; alpha-blend ones become
  blended. Nothing is forced double-sided: tree cards carry their own back triangles, and
  forcing double sides makes them z-fight.
  Hidden meshes (`1WALL_...`, invisible or unrenderable) are kept but flagged, and hidden unless
  `--show-collision`. The cache is `.<kn5 stem>.kn5view.glb` + `.json`, keyed on the KN5's size
  and mtime (dotfiles are skipped by drift-maps' CSP install copy).
- `chromium.py`: drives headless Chromium over the DevTools protocol on a pipe (standard library
  only). Chromium comes from `$KN5VIEW_CHROME`, `$PLAYWRIGHT_BROWSERS_PATH` (here
  `/opt/pw-browsers`), `~/.cache/ms-playwright`, or the PATH. WebGL runs on SwiftShader.
- `viewer/viewer.js`: three.js r160 (vendored in `vendor/three`, MIT) with Lambert materials
  scaled by each material's `ksAmbient + ksDiffuse`, a hemisphere light, the sun, sky and fog.
  Driver views snap the eye to the first opaque surface under the lane point.

Stations are metres along `ai/fast_lane.ai`, starting where the A gate (`AC_AB_START_L/R`)
projects onto it. The lane centre comes from the AI line's side distances.
