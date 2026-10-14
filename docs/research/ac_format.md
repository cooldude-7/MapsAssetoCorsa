# Assetto Corsa track format: research for procedural A→B track generation

Scope: sections A (track format) and E (generated metadata) of `docs/PLANNING_PROMPT.md`.
Research date: 2026-10-14. All repos were cloned and read directly. Commit refs are given where they matter.

**Evidence labels used throughout**

| Label | Meaning |
|---|---|
| **VERIFIED (code)** | Read in the source code of a working, widely used tool (mostly Content Manager / AcTools by gro-ove). |
| **VERIFIED (in-game, 3rd party)** | A third-party project's test log says it was confirmed by loading or driving in AC. |
| **VERIFIED (doc)** | Stated in community documentation or a forum answer (not Kunos-official). Weaker than the two above. |
| **VERIFIED (spike)** | Confirmed by my own throwaway spike in this session. That was a byte-level round trip only, not in-game. |
| **ASSUMED** | My inference or unconfirmed community lore. Needs a spike. |

Kunos has never published an official track or KN5 specification. Everything here is community knowledge, reverse engineering, or tool source code.

---

## 0. Recommendation (TL;DR)

**Write KN5 directly from our own Python code, as version 5, with no Blender, ksEditor or FBX in the build.**

Why:
1. **The format is small and fully known.** It has three sections (textures, materials, node tree) with fixed field order and no compression and no checksums.
   - The authoritative field order is gro-ove's AcTools `Kn5Writer.cs` and `Kn5Basic.cs` reader (MS-PL), which is what Content Manager itself uses. VERIFIED (code): https://github.com/gro-ove/actools/blob/master/AcTools/Kn5File/Kn5Writer.cs
2. **At least four independent projects write KN5 from scratch, and two of them confirm in-game loading:**
   - Hemphill39/AssettoCorsaAutoXGenerator (TypeScript, no Blender or ksEditor). It reports "installs, loads and drives… lap and sector timing work". VERIFIED (in-game, 3rd party): https://github.com/Hemphill39/AssettoCorsaAutoXGenerator/blob/main/TESTING.md
   - KinG-InFeT/blender-assetto-corsa-track-generator (Python in headless Blender, Apache-2.0, writes v6 with embedded PNG).
   - Thomas Hagnhofer's Blender exporter (now moppius/blender-assetto-corsa-tools, GPL-3). It shipped a working sample track that supports practice, time attack, hotlap and quick race.
3. **My spike round-trips cleanly.** A ~100-line stdlib-only Python writer produced a test-strip KN5 (texture, material, `1ROAD`, hidden `1WALL`, `AC_START/PIT/HOTLAP/AB_*` dummies). An independent GPL parser (nazariyKul `kn5.py`) consumed it to exactly EOF with no warnings. VERIFIED (spike), file `scratchpad/scripts/spike/kn5_spike.py`.
4. **It keeps the whole build on Linux, deterministic and fast.** This fits "build workers without the game" and the later hosted-website idea.
5. **The fallback exists and is scriptable.** FBX + `.fbx.ini` go through `ksEditorAt.exe kn5track out.kn5 in.fbx`, a command-line build of ksEditor by x4fab (= gro-ove). It is used by se0lus/sam3_ac_track_mod_gen and leBluem/ac_tools_cmd. It needs Windows plus Kunos' ksEditor (AC SDK), so it is a fallback only.

Writer design rules:
- Version 5. CM's own `Kn5ActualVersion = 5`.
- World-space vertices under identity-transform dummies.
- AC frame: **X = east, Y = up, Z = south** (right-handed).
- ≤ 65,535 vertices per mesh node.
- One material per mesh node.
- All `AC_*` dummies as direct children of the root.
- Textures embedded as DDS (BC1/BC3 with mips), with PNG accepted as a v0 shortcut.
- Validate every build with an independent parser in CI.

---

## 0.1 Spike: what must be tested in game (M0 checklist)

Debug log: `Documents\Assetto Corsa\logs\log.txt`. An AC crash back to the menu with no dialog is usually explained there. VERIFIED (doc): https://steamcommunity.com/app/388980/discussions/0/485624149166109832

Run these in order. Each failure explains the next one.

| # | Test | Pass condition | Why it's uncertain |
|---|---|---|---|
| S1 | Flat 200 m strip: v5 KN5, `ksPerPixel`, PNG texture, `1ROAD`, `data/surfaces.ini`, `ui/ui_track.json`, no `ai/` | Appears in CM. Practice loads. Car sits on road. Road is visible from above. | Winding and normals. Minimal file set. |
| S2 | Same strip with DDS BC1+mips instead of PNG, and a non-power-of-two PNG | Both render. Compare distant shimmer (mips). | PNG mip generation is unconfirmed. |
| S3 | `1WALL_L/R` with `isRenderable=false` | Car collides with an invisible wall. Wall never drawn. | Flag semantics come from community docs. |
| S4 | **Handedness:** an L-shaped road that turns left in the real world (from OSM) | Turns left in game. Minimap is north-up and not mirrored. | Z = south is in-game confirmed by one project only. |
| S5 | Spawn orientation: `AC_PIT_0`, `AC_START_0`, `AC_HOTLAP_START_0` with local +Z along road, at 0.2 m / 1 m / 2 m above surface | Car faces down the road with no drop or jolt. | Height guidance conflicts (1–2 m vs 0 m). |
| S6 | **A→B timing:** `AC_AB_START_L/R`, `AC_AB_FINISH_L/R`, one split `AC_TIME_0_L/R` | Timer starts at A, split fires, finish stops timer in Practice, Hotlap and Time Attack. Record what happens after the finish. | Only community-documented. |
| S7 | Swap L/R deliberately on one gate | Gate stops firing. Confirms the "L on driver's left" rule. | Reported in-game by AutoX for `AC_TIME`. |
| S8 | `models.ini` with main `<id>.kn5` (dummies only) + 3 chunk KN5s | Loads. Dummies in main work. Also try dummies in a chunk. | Whether `<id>.kn5` is still required is contested. |
| S9 | A mesh with exactly 65,535 vertices, and one dummy with an unknown name (e.g. `GROUP_terrain`) holding meshes, plus one empty unknown dummy | Loads. Check if anything after the empty unknown dummy is missing. | Hagnhofer FAQ: unknown logical objects stop loading. |
| S10 | `lodIn/lodOut` on far terrain, `castShadows=false` | Objects pop at the given distance. | CM honours these; AC on tracks is assumed. |
| S11 | Minimap: `map.png` + `map.ini` from our formula | Car dot follows the road. | Formula from CM source; AC side not tested. |
| S12 | 10 km generated road + terrain: FPS, load time, DIP count | Within budget (§6). | Perf budgets are estimates. |
| S13 | KN5 v6 header with Extra = 0 (KinG style) | Loads (informational only). | |
| S14 | CSP on: `extension/ext_config.ini` with `[GRASS_FX]` on terrain material, `[RAIN_FX]` puddles on road | Grass appears on terrain only. Puddles on road. | Optional extras. |
| S15 | Road physics smoothness at 1 m vs 3 m longitudinal tessellation, and with banking | No "stair-step" bumps through the steering wheel. | Collision triangle density is not documented. |

---

## 1. Track folder layout (`content/tracks/<id>/`)

### 1.1 Single-layout track (what we should generate)

```
content/tracks/<id>/               # id: lowercase a-z0-9_, ≤ 32 chars
  <id>.kn5                         # main model, MUST be named after the folder
  models.ini                       # optional: list of KN5s to load (chunking)
  <chunk_*.kn5>                    # optional extra KN5s referenced by models.ini
  map.png                          # minimap image (track root, not data/)
  ai/
    fast_lane.ai                   # AI racing line (also drives the ideal-line assist)
    pit_lane.ai                    # optional
  data/
    surfaces.ini                   # physical surface definitions
    cameras.ini (cameras_1.ini …)  # replay TV camera sets
    map.ini                        # world→map.png transform
    lighting.ini                   # sun angles
    groove.ini                     # rubber groove (can be GROOVES_NUMBER=0)
    crew.ini                       # pit crew side
    audio_sources.ini              # links AC_AUDIO_* dummies to sounds (optional)
  ui/
    ui_track.json                  # name, tags, length, pitboxes, geotags…
    preview.png                    # selector preview (CM uses 355×200)
    outline.png                    # small layout outline (CM uses 365×192)
  extension/
    ext_config.ini                 # optional CSP config (grass, rain, lights)
  skins/                           # optional CM track skins (ignore)
```

Evidence:
- Hagnhofer's track tree lists `ai/fast_lane.ai`, `pit_lane.ai`, `data/{cameras,crew,lighting,map,surfaces}.ini`, `ui/{preview,outline}.png`, `ui/ui_track.json`, `map.png` and `<track>.kn5`. It says "The main .kn5 file must have the same name as the track folder name." VERIFIED (doc): https://sites.google.com/site/hagn99/assettocorsa/modding/tracks
- The assettocorsamods basic guide adds `audio_sources.ini`, `groove.ini` and `texture/` (export-only, not shipped). It sets the naming limits: "track folder ≤ 32 characters, layout folder ≤ 15 … lowercase letters, digits, and underscores". VERIFIED (doc): https://assettocorsamods.net/threads/build-your-first-track-basic-guide.12/
- CM resolves paths as follows. VERIFIED (code): `AcManager.Tools/Objects/TrackObjectBase.cs` L451–470 and `TrackObject.cs` L97–98, L417
  - `MapImage = <LayoutDataDirectory>/map.png`
  - `DataDirectory = <LayoutDataDirectory>/data`
  - `ai/fast_lane.ai`, `ai/pit_lane.ai`, `ai/fast_lane.ai.candidate`
  - `ui/ui_track.json`, `ui/preview.png`, `ui/outline.png`
- CM needs a `ui/` directory and throws `Data_UiDirectoryIsMissing` without it. VERIFIED (code): `TrackObject.cs` `GetLayouts()`.
- CM constants: track preview **355×200**, outline **365×192**. VERIFIED (code): `AcTools/CommonAcConsts.cs`.

### 1.2 Multi-layout tracks (for later: e.g. "downhill" + "uphill" of the same road)

- `ui/<layout>/ui_track.json` (+ preview/outline) per layout. A plain `ui/ui_track.json` can coexist as the main layout. VERIFIED (code): `TrackObject.GetLayouts()`.
- `<id>/<layout>/data/…`, `<id>/<layout>/ai/…`, `<id>/<layout>/map.png`. VERIFIED (code): `TrackExtraConfigurationObject.LayoutDataDirectory = <track>/<layoutId>`.
- `models_<layout>.ini` in the track root. VERIFIED (code): `TrackObjectBase.ModelsFilename`.
- The forum says the main `<id>.kn5` stays required with layouts and holds shared geometry, and per-layout KN5s hold layout-specific pieces (e.g. dummies). VERIFIED (doc): https://assettocorsamods.net/goto/post?id=21922
- KinG-InFeT's generator uses exactly this for forward/reverse layouts. VERIFIED (code, Apache-2.0): `scripts/setup_mod_folder.py`.
- **Uphill vs downhill of the same road is a natural use of layouts.** Geometry stays shared. Only dummies, AI line and UI differ.

### 1.3 `models.ini` (multiple KN5s)

```ini
[MODEL_0]
FILE=<id>.kn5
POSITION=0,0,0
ROTATION=0,0,0

[MODEL_1]
FILE=<id>_terrain.kn5
POSITION=0,0,0
ROTATION=0,0,0
```

- CM parses `[MODEL_n]` with `FILE`, `POSITION`, `ROTATION` (yaw, pitch, roll). Matrix = `Translation(POSITION) * RotationYawPitchRoll(rot.X, rot.Y, rot.Z)`. VERIFIED (code): `AcTools.Render/Kn5SpecificSpecial/TrackMapRenderer.cs` `TrackComplexModelDescription` L243–250.
- Contradiction on whether `<id>.kn5` must also exist:
  - Without layouts, KinG notes "No models.ini needed (AC auto-detects KN5 by folder name)". VERIFIED (code comment).
  - The forum says the main KN5 named after the folder is mandatory with layouts.
  - One Steam answer: keeping "only the .kn5, models.ini, map.png and UI folder" fixed a crash-to-menu.
  - **Plan:** always ship `<id>.kn5` and list it as `MODEL_0` → spike S8.

### 1.4 Which files are mandatory to load

| File | Needed to load? | Notes |
|---|---|---|
| `<id>.kn5` | **Yes** | VERIFIED (doc) Hagnhofer, assettocorsamods. |
| ≥1 spawn dummy for the chosen spawn set | **Yes** in practice | Practice spawns at `AC_PIT_n` by default. Hotlap at `AC_HOTLAP_START_0`. Race and Time Attack at `AC_START_n`. VERIFIED (code) CM `Game.Properties.cs`. Missing pits give "Cannot have 0 cars in race.ini". VERIFIED (doc) basic guide troubleshooting. |
| `ui/ui_track.json` | Yes for CM listing. The AC launcher probably also uses it. | CM requires the `ui/` dir. VERIFIED (code). AutoX: "Track appears in CM's track list" depends on it. VERIFIED (in-game, 3rd party). |
| `data/surfaces.ini` | Not strictly. The guide says `data` "may be empty for a basic track". | AC falls back to `system/data/surfaces.ini` (ROAD, GRASS, KERB, SAND). CM merges track + system surfaces by KEY. VERIFIED (code) `TrackMapRendererTools.xaml.cs` `LoadAll()`. AC's own merge is ASSUMED. Also needed by servers ("a server track folder needs only `data\surfaces.ini`", basic guide). **Always ship it.** |
| `ai/fast_lane.ai` | No | Needed for AI, ideal line, replay cameras and race position. A *broken* AI file can crash to menu. VERIFIED (doc) Steam thread. Hagnhofer's sample without AI: "AI will retire after a few seconds". |
| `map.png` + `data/map.ini` | No (ASSUMED) | Minimap only. |
| `cameras.ini`, `lighting.ini`, `groove.ini`, `crew.ini`, `audio_sources.ini` | No (ASSUMED, guide says data may be empty) | Ship sane defaults anyway. |
| `ui/preview.png`, `ui/outline.png` | No | Cosmetic. AutoX added them after the "missing" warnings. |

### 1.5 `data/surfaces.ini`

Vanilla `system/data/surfaces.ini` (full file, VERIFIED (doc), copy in a server repo: https://github.com/Ilcoups/RedLine-Souls-Assetto-Corsa-Server/blob/HEAD/system/data/surfaces.ini):

```ini
[SURFACE_0]
KEY=ROAD
FRICTION=1
DAMPING=0
WAV=
WAV_PITCH=0
FF_EFFECT=NULL
DIRT_ADDITIVE=0
BLACK_FLAG_TIME=0
IS_VALID_TRACK=1
SIN_HEIGHT=0
SIN_LENGTH=0
IS_PITLANE=0
VIBRATION_GAIN=0
VIBRATION_LENGTH=0

[SURFACE_1]
KEY=GRASS
FRICTION=0.6
DAMPING=0
WAV=grass.wav
WAV_PITCH=0
FF_EFFECT=NULL
DIRT_ADDITIVE=1
IS_VALID_TRACK=0
BLACK_FLAG_TIME=0
SIN_HEIGHT=0.03
SIN_LENGTH=0.5
IS_PITLANE=0
VIBRATION_GAIN=0.2
VIBRATION_LENGTH=0.6

[SURFACE_2]
KEY=KERB
FRICTION=0.92
… WAV=kerb.wav, WAV_PITCH=1.3, FF_EFFECT=1, VIBRATION_GAIN=1.0, VIBRATION_LENGTH=1.5

[SURFACE_3]
KEY=SAND
FRICTION=0.8
DAMPING=0.1
WAV=sand.wav
FF_EFFECT=0
DIRT_ADDITIVE=1
IS_VALID_TRACK=0
SIN_HEIGHT=0.04
SIN_LENGTH=0.5
VIBRATION_GAIN=0.2
VIBRATION_LENGTH=0.3
```

Kunos tracks redefine and add keys. For example `ks_laguna_seca` has `PITS` (IS_PITLANE=1, FRICTION=0.965), `KERB`, `CURB`, `CONCRETE`, `DRAIN`. VERIFIED (doc, copy of vanilla file): https://github.com/Sezynex/AssettoServer-Traffic/blob/HEAD/content/tracks/ks_laguna_seca/data/surfaces.ini

Key meanings (VERIFIED (doc), Hagnhofer: https://sites.google.com/site/hagn99/assettocorsa/modding/tracks/data/surfaces-ini, unless marked):

| Key | Meaning |
|---|---|
| `KEY` | Surface name. A mesh `<digits><KEY>…` uses it. |
| `FRICTION` | Grip multiplier. Kunos road ≈ 0.96–1.0. Grass 0.6. |
| `DAMPING` | Rolling drag / slow-down factor (sand 0.1). |
| `WAV`, `WAV_PITCH` | Surface sound and its playback rate. |
| `FF_EFFECT` | Force-feedback effect ("NULL", 0, 1). Exact meaning unknown. ASSUMED: 1 = kerb rumble. |
| `DIRT_ADDITIVE` | Dirt on tyres (grass/sand = 1). CSP SurfacesFX auto-enables at ≥ 0.7. VERIFIED (doc) CSP: https://cup.acstuff.club/docs/csp/tracks/surface-tweaks |
| `IS_VALID_TRACK` | 1 = legal track. 0 triggers cut and penalty logic. |
| `BLACK_FLAG_TIME` | Unknown. ASSUMED: seconds off-track before a black flag. |
| `SIN_HEIGHT`, `SIN_LENGTH` | Procedural bumps (height in m, frequency). |
| `IS_PITLANE` | Pit limiter zone. |
| `VIBRATION_GAIN`, `VIBRATION_LENGTH` | FFB and rumble vibration. Meaning unconfirmed. |
| CSP extras | `_EXT_SURFACE_TYPE` (EXTRATURF/GRASS/GRAVEL/KERB/OLD/SAND), `_EXT_SURFACE_TYPE_MODIFIER`, `_EXT_PERLIN_NOISE`, `_EXT_PERLIN_OCTAVES`, `_EXT_PERLIN_PERSISTENCE`. VERIFIED (doc) CSP surface-tweaks page. Ignored by vanilla AC (ASSUMED). |

- **WALL is not a surfaces.ini key in vanilla.** Meshes with `WALL` are hard collision. KinG warns: "defining KEY=WALL would … make walls driveable surfaces instead of collision barriers. Kunos tracks (e.g. ks_laguna_seca) also omit WALL." VERIFIED (code comment, Apache-2.0); the claim itself is ASSUMED.
- **Our generated `surfaces.ini`:** `ROAD` (asphalt, FRICTION ≈ 0.97–1.0), `GRASS`, `GRAVEL` (shoulder; custom key, FRICTION ≈ 0.75, DIRT 1, `_EXT_SURFACE_TYPE=GRAVEL`), optionally `ROAD_DIRT` for OSM `surface=unpaved`. Add `PIT` only if we generate a pit area. No `WALL` entry.

### 1.6 `data/cameras.ini` (replay TV cameras)

Format VERIFIED (doc): Hagnhofer https://sites.google.com/site/hagn99/assettocorsa/modding/tracks/data/cameras-ini, plus a real exported file https://github.com/JSAE-ARCHIVES/FSAEJ-simulator/blob/HEAD/FSAEJ_Endurance/data/cameras.ini

```ini
[HEADER]
VERSION=3
CAMERA_COUNT=17
SET_NAME=TV 1

[CAMERA_0]
NAME=cam0
POSITION=-115.486 ,1.19716 ,35.4616
FORWARD=-0.4767 ,-0.0531 ,-0.8775
UP=-0.0254 ,0.9986 ,-0.0467
MIN_FOV=30
MAX_FOV=30
IN_POINT=0.987        ; spline-progress (0..1) where this camera takes over
OUT_POINT=1           ; … and hands over
SHADOW_SPLIT0=1.8
SHADOW_SPLIT1=20
SHADOW_SPLIT2=180
NEAR_PLANE=0.1
FAR_PLANE=5000
MIN_EXPOSURE=0
MAX_EXPOSURE=10000
DOF_FACTOR=10
DOF_RANGE=10000
DOF_FOCUS=0
DOF_MANUAL=0
SPLINE=
SPLINE_ROTATION=0
FOV_GAMMA=1
SPLINE_ANIMATION_LENGTH=15
IS_FIXED=0
```

- Extra sets go in `cameras_1.ini`, `cameras_2.ini` and so on.
- `IN_POINT`/`OUT_POINT` are normalised track progress. ASSUMED: along `fast_lane.ai`, judging by values in (0,1]. Consecutive cameras should tile [0,1].
- **Generation plan:** every ~150–300 m of road, place a camera 4–8 m off the road edge on the outside of bends, 2–5 m up, aimed at the road point ~30 m ahead. Set `IN/OUT_POINT` from the arc-length fraction.
- KinG's cameras.ini uses a different `VERSION=2` schema (FOV/NEAR/FAR/MIN_DISTANCE). That does **not** match real files. Don't copy it. ASSUMED wrong.

### 1.7 `data/map.ini` + `map.png`

Format (real file, VERIFIED (doc) https://github.com/JSAE-ARCHIVES/FSAEJ-simulator/blob/HEAD/FSAEJ_Endurance/data/map.ini; Hagnhofer https://sites.google.com/site/hagn99/assettocorsa/modding/tracks/data/map-ini):

```ini
[PARAMETERS]
WIDTH=2048.000000      ; map.png width in px (incl. margins)
HEIGHT=2048.000000
MARGIN=20
SCALE_FACTOR=0.181117  ; metres per pixel
MAX_SIZE=1600
X_OFFSET=173.232940
Z_OFFSET=198.544997
DRAWING_SIZE=10
```

**World → pixel:** `px = (X + X_OFFSET) / SCALE_FACTOR`, `py = (Z + Z_OFFSET) / SCALE_FACTOR`.
- Image is top-down with image-up = world −Z = **north**, given AC's Z = south.
- VERIFIED (code), two independent sides in CM:
  - The writer `TrackMapRenderer.TrackMapInformation.SaveTo` sets `WIDTH = Width + 2·Margin`, `X_OFFSET = −box.Min.X + Margin/Scale`, `SCALE_FACTOR = 1/Scale`, where Scale = px per metre. Its ortho camera uses `Up = (0,0,−1)`.
  - The consumer, CM's live server map, uses `relX = (worldX + X_OFFSET) / (WIDTH·SCALE_FACTOR)` and `relY = (worldZ + Z_OFFSET) / (HEIGHT·SCALE_FACTOR)`. Source: `AcManager/Pages/ServerPreset/ServerPresetRunningStatus.xaml.cs` L50–67.
- The FSAE file is consistent with this: 2048 px × 0.181 m/px ≈ 371 m, a plausible FSAE site. AutoX's alternative `(world+OFFSET)*SCALE+MARGIN` is marked unverified by its author and contradicts CM. **Use CM's formula.** Spike S11.
- **Generation:** render the road ribbon (white, alpha background, maybe a darker outline) at ~0.5–2 m/px with `MAX_SIZE`-ish caps. CM caps at 8192/16384 px. For a 10 km touge route that spans e.g. 4 km, use ~2 m/px and ≤ 2048 px.
- `outline.png`: same shape, small (365×192 per CM const), white on transparent.
- `preview.png`: 355×200. Render it ourselves: top-down or an oblique matplotlib/pyrender shot. CM can also regenerate it in game.

### 1.8 `data/lighting.ini`, `groove.ini`, `crew.ini`, `audio_sources.ini`

```ini
; lighting.ini — VERIFIED (doc) Hagnhofer
[LIGHTING]
SUN_PITCH_ANGLE=45     ; 0 = zenith, 90 = horizon (per Hagnhofer)
SUN_HEADING_ANGLE=45   ; counter-clockwise around the track

; groove.ini — VERIFIED (doc), real files contain only this
[HEADER]
GROOVES_NUMBER=0

; crew.ini — VERIFIED (doc), real file
[HEADER]
SIDE=-1 ; 1 = left , -1 = right
```

`audio_sources.ini` links `AC_AUDIO_*` dummies to sounds (basic guide). Format not researched. Omit it (ASSUMED optional).

### 1.9 `ui/ui_track.json`

Fields CM knows (VERIFIED (code) `AcManager.Tools/AcErrors/JObjectRestorationSchemeProvider.cs` L50–67, `TrackObjectBase.cs`):

| Field | Type | Notes |
|---|---|---|
| `name` | string | |
| `description` | string | |
| `tags` | string[] | See below. |
| `geotags` | string[2] | `[lat, lon]`. Kunos ships literal `["lat","lon"]` placeholders. CM parses decimal strings (`"41.71712"`, `"-88.29689"`), DMS, or N/S/E/W suffixes. Use plain signed decimals. VERIFIED (code) `AcTools/Utils/Helpers/GeoTagsEntry.cs`. |
| `country`, `city` | string | Used for CM filtering. |
| `length` | string | Kunos uses metres as a string, e.g. `"17619"`. |
| `width` | string | e.g. `"8m"`, `"10m-15m"`. |
| `pitboxes` | string | Number of pits. Limits AI/grid size in the UI. VERIFIED (doc) Hagnhofer, VERIFIED (in-game, 3rd party) AutoX ("AC capped the field at one car"). |
| `run` | string | Free text shown on hover. Kunos uses `"clockwise"`, `"DRIFT"`, `"dragstrip"`. A forum suggests `"point-to-point"`. |
| `year` | number | |
| `author`, `url`, `version` | string | |
| `priority` | number | Layout ordering (CM). |

Real Kunos A→B example, Trento-Bondone (VERIFIED (doc), copy at https://github.com/TylerBre/acserver_manager/blob/HEAD/seed/content/tracks/trento-bondone/ui/ui_track.json):

```json
{ "name": "Trento-Bondone", "description": "Hillclimb course",
  "tags": ["A2B","Hillclimb","touge","original","Italy"],
  "geotags": ["lat","lon"], "country": "Italy", "city": "Trento",
  "length": "17619", "width": "8m", "pitboxes": "4", "run": "clockwise" }
```

**CM tag gating** (VERIFIED (code) `AcManager/Pages/Drive/QuickDrive_*.xaml.cs` + `QuickDriveModeViewModel.TagRequired`):
- These modes show a "tag is missing" warning with a one-click "add tag" fix unless the track has `"circuit"`:
  - Race with **more than 1 lap**
  - Time Attack
  - Trackday
  - Weekend
- **Race with 1 lap does not need it.** That is exactly the touge-battle case.
- Drag requires `"drag"`.

Recommended for generated tracks:

```json
{ "name": "Akina Downhill", "description": "Generated from OpenStreetMap…",
  "tags": ["A2B", "touge", "point-to-point", "generated", "Japan"],
  "geotags": ["36.4612", "138.8742"], "country": "Japan", "city": "Takasaki",
  "length": "3600", "width": "6m", "pitboxes": "8", "run": "A2B",
  "author": "drift-maps generator", "version": "1.0", "year": 2026 }
```

---

## 2. Dummy (empty) nodes, orientation, timing

### 2.1 Names

| Name | Purpose | Evidence |
|---|---|---|
| `AC_START_n` (n from 0, no leading zeros) | Race grid slot n. Also Time Attack start. | VERIFIED (doc) Hagnhofer, basic guide. VERIFIED (code) CM spawn sets. |
| `AC_PIT_n` | Pit box n. **Default Practice and Drift spawn.** | VERIFIED (code) CM `PracticeProperties.StartType = Pit`. VERIFIED (in-game, 3rd party) AutoX "Practice spawns at AC_PIT_0, not AC_START_0". |
| `AC_HOTLAP_START_0` | Hotlap spawn (`SPAWN_SET=HOTLAP_START`). | VERIFIED (code) CM. |
| `AC_TIME_n_L` / `AC_TIME_n_R` | Timing gate n. On circuits n=0 is start/finish. Gate = segment between the L and R dummies. `_L` must be on the driver's left. | VERIFIED (doc) Hagnhofer. VERIFIED (in-game, 3rd party) AutoX: mirrored frame swapped L/R and "stops the timing gates firing at all". |
| `AC_AB_START_L/R`, `AC_AB_FINISH_L/R` | Point-to-point start and finish gates. | VERIFIED (doc) basic guide + forum "this works correct": https://assettocorsamods.net/threads/need-help-with-timing-objects-for-a-to-b-type-stage.1056/ |
| `AC_TIME_0_L/R` on A→B | Optional single split ("use only one!"). | VERIFIED (doc) forum, page-91 of the guide thread. |
| `AC_OPEN_START_L/R`, `AC_OPEN_FINISH_L/R` | Kunos drag strip open timing. | VERIFIED (doc) forum; listed in moppius' known-name list. |
| `AC_POBJECT_*` | Movable physics props (cones). Keep ≤ ~15. | VERIFIED (doc) basic guide. |
| `AC_AUDIO_*` | Positional audio emitters (`audio_sources.ini`). | VERIFIED (doc). |
| `AC_CREW_n` | Pit crew positions. | In moppius' known list; meaning ASSUMED. |

Rules:
- **Put all `AC_*` dummies directly under the KN5 root.** CM counts pits with `RootNode.GetByName("AC_PIT_{x}")`, which only checks direct children. VERIFIED (code) `SelectedTrackPage.xaml.cs` L303.
- Don't emit empty dummies with non-`AC_` names. Hagnhofer: "Logical objects that aren't recognized by the engine will cause problems… it just stops loading". VERIFIED (doc), maybe outdated → spike S9.
- Meshes must not have children: "A mesh node with children will end up in an engine crash." VERIFIED (doc).
- Number of `AC_START_n`/`AC_PIT_n` and `pitboxes` caps the car count. Generate ≥ 8–12 slots.

### 2.2 Orientation

- "Pivot Y axis points up and Z axis points forward." VERIFIED (doc) basic guide.
- AutoX writes dummy matrices as yaw about Y: rows `[c,0,−s,0] [0,1,0,0] [s,0,c,0] [x,y,z,1]`, so local +Z = `(sinθ,0,cosθ)` = forward. Cars spawned correctly. VERIFIED (in-game, 3rd party).
- Matrix is row-major and row-vector, with translation at floats 12–14 (§4.3). For a spawn at road point `p` with unit tangent `f` and up `u` (use world up `(0,1,0)` for spawns, or the road normal on steep grades): `Z = f`, `Y = u`, `X = Y × Z`. Write rows `X, Y, Z, (p + 0.5…1 m·up)`.
- Timing dummies don't need orientation, only position. VERIFIED (doc) basic guide.
- Spawn height: the guide says 1–2 m above the surface. AutoX used 0 m on a flat lot. JostMK uses +1 m. ASSUMED: AC drops the car onto the surface → spike S5.

### 2.3 How point-to-point timing and sessions work

- A→B tracks use `AC_AB_START_*` and `AC_AB_FINISH_*` instead of `AC_TIME_0` start/finish. Crossing START starts the timer. FINISH stops it. One optional split `AC_TIME_0_L/R`. VERIFIED (doc).
- One modder: A→B works in practice, hotlap and time attack, "not in race mode". Another A→B author: many timing apps malfunction. VERIFIED (doc), low confidence: https://www.overtake.gg/downloads/short-point-to-point-testing-track.59454/
- AI on A→B "views the finish line as a stopping point". Section H owns this.
- Kunos ships an official A→B hillclimb (Trento-Bondone, tags `A2B`, `pitboxes` 4). Inspecting its KN5 node names with CM or our reader is a cheap way to see Kunos' own A→B dummy set. **Do this in spike S6.**
- Spawn placement for A→B: grid `AC_START_n` and pits `AC_PIT_n` in pairs behind `AC_AB_START`, on the road. `AC_HOTLAP_START_0` ~50–100 m before the start gate so hotlap starts rolling. Make gate segments ≥ road width + 2×5 m so wide cars still trip them. AutoX used ≥ 14 m half-width for that reason. VERIFIED (in-game, 3rd party).
- **Record lanes from day one** (section H): store per-sample left/right edge offsets and lane count in the generator's intermediate data. The AI file (`sideLeft`/`sideRight`) and traffic splines need them.

---

## 3. Physical surfaces, collision, invisible walls

### 3.1 Naming

- Physical mesh name = `<ID><KEY><optional suffix>`, with ID a positive integer and KEY a `surfaces.ini` key. Examples: `1ROAD`, `1ROAD_003`, `2GRASS_left`, `1SAND_003`. The ID does not need to be unique. VERIFIED (doc) Hagnhofer track-kn5, basic guide.
  - `0ROAD` is not valid. `01ROAD` is. VERIFIED (doc) AutoX PLAN.
  - Don't combine two keys in one name (`1GRASS_road`). VERIFIED (doc).
- CM's own matcher builds the regex `\d+(KEY1|KEY2|…)`, and `\d*` for keys that start with a digit. VERIFIED (code) `AcManager/CustomShowroom/TrackMapRendererTools.xaml.cs` L181–192.
- Mesh names without the prefix are visual only, with no collision. AutoX used this for drive-through cones. VERIFIED (in-game, 3rd party).
- **WALL:** "WALL anywhere in the name creates a collidable wall. Using a numbered form such as `1WALL` is recommended." VERIFIED (doc). AutoX used `nWall_perimeter`. Use `1WALL_<side>_<chunk>`.
- KEY matching is case-sensitive. ASSUMED. Use uppercase KEYs.
- Physics is per mesh node, so one physical mesh node = one material + one surface. To make the asphalt and the shoulders different surfaces, emit separate mesh nodes (`1ROAD_c012`, `1GRAVEL_c012`, `1GRASS_t034`).

### 3.2 How collision works and invisible walls

- Meshes with a surface name are turned into physics geometry automatically at load. Hagnhofer: "Physical objects are automatically generated when the mesh nodes have a special naming scheme". VERIFIED (doc).
- **Invisible wall** = `1WALL_*` mesh with `isRenderable = false`. Hagnhofer settings.json docs: "isRenderable: controls if this node is drawn or not. Can be used for hidden physic walls". VERIFIED (doc) https://sites.google.com/site/hagn99/assettocorsa/blender-kn5-exporter/settings-json
  - Also set `castShadows=false`.
  - CM's renderer skips a mesh if `!Active || !IsVisible || !IsRenderable`. VERIFIED (code) `Kn5RenderableObject.cs` L42.
  - Use `isVisible=true, isRenderable=false`, as in ksEditor's "renderable" checkbox. Spike S3.
- Wall geometry: a vertical ribbon 2–3 m high along each road edge (offset ~0.5–1 m beyond shoulder), plus world-bound walls around the terrain corridor.
  - KinG splits walls every ~25 m "for AC collision". VERIFIED (code comment). Whether that is required is ASSUMED.
  - Make walls double-sided: two triangles per quad facing both ways. Collision is likely one-sided. ASSUMED.
- **Road physics tessellation:** the car rides on the collision triangles, so faceting = bumps. ASSUMED guidance: longitudinal step ≤ 1–2 m in curves (≤ 3–5 m on straights), ≥ 3 lateral vertices across each lane, and no T-junction cracks between chunks. Spike S15.
- Other node flags (`castShadows`, `isTransparent`, `layer`, `lodIn/lodOut`) don't affect physics. ASSUMED.
- Optionally use a coarser *render* road and a separate invisible fine *physics* road. Not needed initially.

---

## 4. KN5 binary format

### 4.1 Overall layout (little-endian)

Sources:
- VERIFIED (code): AcTools `Kn5Writer.cs`, `Kn5Basic.cs` (reader `Factory.Kn5Reader`), `ExtendedBinaryWriter.cs`, `ReadAheadBinaryReader.cs` at gro-ove/actools@e5e5c94.
- Cross-checked against moppius (GPL-3), KinG (Apache-2.0), AutoX PLAN §2.1, and the nazariyKul parser (GPL-3).

```
magic      6 bytes  "sc6969"
version    int32    5 (CM default, Kn5ActualVersion=5) | 6
[extra]    int32    only if version > 5 (meaning undocumented; KinG writes 0)
textures   int32 count
  per texture:
    int32  active        (1)
    string name          (e.g. "asphalt.dds")
    uint32 length
    bytes  data[length]  (raw DDS or PNG file bytes)
materials  int32 count
  per material:
    string name
    string shaderName    (e.g. "ksPerPixel")
    byte   blendMode     0 Opaque, 1 AlphaBlend, 2 AlphaToCoverage
    bool   alphaTested   (1 byte)
    int32  depthMode     0 Normal, 1 NoWrite (read-only), 2 Off   [absent if version <= 4]
    int32  propertyCount
      per property: string name; float A; float2 B; float3 C; float4 D   (name + 40 bytes)
    int32  textureMappingCount
      per mapping:  string slotName ("txDiffuse"…); int32 slot; string textureName
root node  (exactly one; recursive, depth-first pre-order: node record, then its children)
```

**Strings:** `int32 byteLength` + bytes, no terminator. AcTools writes UTF-8. One AcTools reader path decodes ASCII (`ReadAheadBinaryReader.ReadString` override, L553). **Keep all names ASCII.**

**Bools:** 1 byte.

### 4.2 Node records

```
int32  nodeClass      1 = Base/Dummy, 2 = Mesh, 3 = SkinnedMesh
string name
int32  childCount     (meshes: always 0)
bool   active
-- class 1 (dummy):
float[16] transform   (see 4.3)
-- class 2 (mesh):
bool   castShadows
bool   isVisible
bool   isTransparent
uint32 vertexCount
  per vertex (44 bytes): float3 position; float3 normal; float2 uv; float3 tangent
uint32 indexCount     (multiple of 3; triangle list)
uint16 indices[indexCount]
uint32 materialId     (index into the materials array)
uint32 layer          (0; CM treats layer 1 like transparent — Kunos semantics unknown)
float  lodIn          (metres; 0 = no limit)
float  lodOut         (metres; 0 = no limit)
float3 boundingSphereCenter
float  boundingSphereRadius
bool   isRenderable
-- class 3 (skinned mesh): castShadows, isVisible, isTransparent; uint32 boneCount;
   per bone {string name; float[16] bindTransform}; uint32 vertexCount; per vertex 76 bytes
   (44 as above + float4 weights + float4 boneIndices-as-floats); uint32 indexCount; uint16[];
   uint32 materialId; uint32 layer; 8 bytes (lodIn, lodOut). No sphere/renderable.  (not needed by us)
```

- **Mesh nodes carry no transform.** They inherit the parent dummy's. Generators emit world-space vertices under an identity root. VERIFIED (code) Kn5Writer; VERIFIED (in-game, 3rd party) AutoX.
- **Index format is uint16. Max 65,536 vertices per mesh node.** AcTools `Kn5MeshBuilder` throws "Limit exceeded" past 65,535. moppius and Hagnhofer auto-split at that limit. VERIFIED (code).
- **Bounding sphere:** used for culling. If it's too small, geometry vanishes at angles. Exporters use a loose sphere: center = AABB center, radius = max distance to a vertex (KinG) or 2 × max half-extent (moppius). VERIFIED (code). Use the max-distance version.
- **lodIn/lodOut** are distances in metres. Both 0 = always drawn. CM's renderer squares and compares them against camera distance. VERIFIED (code) `Kn5RenderableObject.cs` L52–55. AC's track behaviour is ASSUMED to match → spike S10.
- **isVisible vs isRenderable:** CM hides the mesh if either is false. Kunos semantics: ksEditor's "Renderable" toggles drawing while physics stays. ASSUMED that `isVisible` relates to editor visibility. Set both true for normal meshes. Set `isRenderable=false` for invisible walls.
- **isTransparent:** set true for alpha-blended meshes (sort order).
- **Tangent:** a 3-vector per vertex. Needed for normal-mapped shaders. AcTools has `Kn5MeshUtils.RecalculateTangents()` for reference. VERIFIED (code) `AcTools.ExtraKn5Utils/Kn5Utils/Kn5MeshUtils.cs`.
- **UV origin is top-left (DirectX).** Blender exporters write `v' = 1 − v` (KinG, KN5Export) or `−v` (moppius). This only matters for non-tiling atlases.
- **Winding:** keep triangles counter-clockwise in a right-handed Y-up frame when seen from the visible side, i.e. `cross(p1−p0, p2−p0)` points along the visible normal. AutoX derived this and confirmed the ground "renders correctly from above". VERIFIED (in-game, 3rd party) `geometry/builders.ts orientedTriangle()`.

### 4.3 Transform matrix layout

- 16 float32 written `M11 M12 M13 M14 M21 … M44`: row-major, DirectX row-vector convention.
- **Translation in M41, M42, M43 = floats 12, 13, 14.**
- Rows 0, 1, 2 are the local X, Y, Z axes in parent space.
- VERIFIED (code): `ExtendedBinaryWriter.Write(Mat4x4)`, `AcTools/Numerics/Mat4x4.cs` (`Translation => (M41,M42,M43)`), KinG `obj_to_matrix`, AutoX `dummyMatrix`.

### 4.4 Coordinate frame

- AC world: **Y up, right-handed, X = east, Z = south** (so −Z = north). VERIFIED (in-game, 3rd party) AutoX: "Left-handed coordinate frame mirrored the world… Corrected to AC's right-handed frame (Z = south)".
- Consistent with the Blender→AC mapping `(x,y,z) → (x,z,−y)` used by all Blender exporters (det +1). JostMK uses `−z` for northing too.
- **ENU → AC: `X = E`, `Y = U`, `Z = −N`.**
- Use a local frame centred near the route start (or the route bbox centre), so float32 precision stays sub-millimetre. Tracks span ≤ ~20 km. ASSUMED fine.

### 4.5 Versions

- v5: the baseline. CM writes 5 for new files. VERIFIED (code) `CommonAcConsts.Kn5ActualVersion = 5`.
- Hagnhofer, moppius and AutoX write v5. AutoX is in-game confirmed.
- v6: adds one int32 after `version`, read and written as "Extra". Its meaning is undocumented. VERIFIED (code). KinG writes v6 with Extra=0 and reports working tracks.
- v ≤ 4: material record lacks the `depthMode` int32, per nazariyKul's parser (`count(22 if version > 4 else 18)`). VERIFIED (code, GPL).
- v7: LKOLA's addon lists versions 4–7 without evidence. ASSUMED irrelevant.
- CSP "protected/encrypted" KN5s exist, with a sidecar envelope (Kn5Decrypt, GPL-3). Irrelevant for us.
- **Write v5.**

### 4.6 Textures in KN5

- The blob is the raw file. **DDS and PNG are both accepted.**
  - Hagnhofer's exporter embedded PNG until v0.0.9 added DDS passthrough. Its sample track worked. VERIFIED (doc) changelog.
  - KinG and AutoX embed PNG. VERIFIED (in-game, 3rd party, AutoX).
- DDS formats:
  - **BC7 is not readable by vanilla AC or ksEditor.** CSP adds support. VERIFIED (doc, in CM source) `AcManager/Pages/ContentTools/Bc7Upgrader.xaml` L59–61.
  - "for BC7 (or DXT1/DXT5) to work, texture dimensions should be divisible by 4". ksEditor's normal output is "RGBA8888 encoding with MIPs". VERIFIED (doc, in CM source).
- **Recommendation:** ship a small fixed library of tiling textures (asphalt, asphalt normal, gravel, grass, dirt, rock, guardrail, tree atlas). Store them pre-encoded as DDS BC1 (opaque) and BC3 (alpha), power-of-two (512–2048), with full mip chains.
  - Encode once offline (texconv / Compressonator) or in Python.
  - Pillow `main` can *save* DXT1/DXT3/DXT5/BC2/BC3/BC5 DDS but writes `mipmaps=0`. VERIFIED (code) https://github.com/python-pillow/Pillow/blob/main/src/PIL/DdsImagePlugin.py `_save`. Mips would need a tiny custom DDS header writer that concatenates per-level Pillow encodes.
  - Generated per-track textures (if any, e.g. a terrain colour mask) can start as PNG.
  - Whether AC builds mips for PNG is ASSUMED. Spike S2.
- Texture names must be unique within a KN5. Each material references textures by name. Textures shared by many chunks are duplicated per KN5 unless one KN5 holds them. Does AC share textures across KN5s by name? ASSUMED no. Keep shared materials in few KN5s.

### 4.7 Existing readers and writers

| Project | Lang | Read | **Write** | Licence | Notes |
|---|---|---|---|---|---|
| gro-ove/actools (Content Manager) https://github.com/gro-ove/actools | C# | ✔ | ✔ (`Kn5Writer`, `Kn5Basic.Save`) | **MS-PL** | Authoritative. Also: FBX↔KN5 bits for car LODs (`CarLodGenerator.LodFbxToKn5`, `Kn5MeshBuilder`), Collada/FBX export + `.fbx.ini` writer, AI spline reader/writer (`AcTools/AiFile`), track map renderer, `CustomTracksBakery` (VAO) CLI, CSP tree model converter (`AcTreeModelConverter.cs`). **CM itself has no general FBX→KN5 converter.** Its UI says "Convert FBX to KN5 the usual way (with ksEditor)" (`Uv2ModelConverter.xaml`). |
| moppius/blender-assetto-corsa-tools https://github.com/moppius/blender-assetto-corsa-tools | Python (Blender) | – | ✔ v5 | GPL-3 | Fork of Hagnhofer's 2014 exporter. Splits >65k. Has JSON settings for node/material flags. |
| KinG-InFeT/blender-assetto-corsa-track-generator https://github.com/KinG-InFeT/blender-assetto-corsa-track-generator | Python (headless Blender) | – | ✔ v6, PNG | **Apache-2.0** | Full procedural *circuit* pipeline: SRTM elevation, banking, walls, surfaces.ini, ui_track.json, outline, AI line from centreline. Reusable with attribution. Its cameras.ini looks non-standard. |
| Hemphill39/AssettoCorsaAutoXGenerator https://github.com/Hemphill39/AssettoCorsaAutoXGenerator | TypeScript | ✔ | ✔ v5, PNG, `fast_lane.ai` | **none (all rights reserved)** | **In-game verified**, programmatic, no Blender. Read for facts only, don't copy code. Its PLAN.md has a field spec. |
| FatihEsen/KN5Export https://github.com/FatihEsen/KN5Export | Python (Blender) | ✔ | ✔ v5/v6 | **CC0** | Brand new (2026-10-10), single commit, likely AI-written. Also claims `.ai` read/write, csv splines, cameras ini, fbx.ini. Treat claims as unverified. |
| LKOLA/blender-kn5-addon https://github.com/LKOLA/blender-kn5-addon | Python (Blender) | ✔ | ✔ | README says MIT, no LICENSE file | Car-oriented, new. |
| nazariyKul/kn5-blender-tools https://github.com/nazariyKul/kn5-blender-tools | Python | ✔ (pure stdlib `kn5.py`) | – | GPL-3 | **Good independent test oracle** (strict bounds checks, EOF check). Use it as a dev/test-only dependency, not vendored into distributed code. |
| I-Rinka/kn5-editor https://github.com/I-Rinka/kn5-editor | Python + three.js | ✔ | ✔ (transform edits) | README says MIT, no LICENSE file | |
| MarvinSt/kn5-obj-converter https://github.com/MarvinSt/kn5-obj-converter | Python | ✔ | – | **none** | KN5→OBJ/FBX. |
| RaduMC/kn5-converter https://github.com/RaduMC/kn5-converter | C# | ✔ | – | **none** | The original community converter. |
| IOBYTE/kn5toac https://github.com/IOBYTE/kn5toac | C++ | ✔ | – | ? | KN5→AC3D (Speed Dreams). |
| leBluem/ac_tools_cmd https://github.com/leBluem/ac_tools_cmd | exe + cmd | ✔ | via ksEditorAt | not stated | Bundles **ksEditorAt** (CLI ksEditor), `kn5FixByINI`, `KN5Join` (merge KN5s per models.ini), `kn52fbx`. |
| ksEditorAt (x4fab = gro-ove) https://ascobash.wordpress.com/2015/07/22/kseditor/ | exe | – | ✔ FBX→KN5 | freeware mod of Kunos ksEditor | `ksEditorAt.exe kn5track destination.kn5 original.fbx`, `kn5 …`, `ksanim …`. Must sit next to Kunos' ksEditor files. "kn5track mode was not tested" (2015 note). The sam3 project uses it today. |

### 4.8 Is writing KN5 from Python realistic?

Yes. Roughly 300 lines including mesh splitting, bounding spheres and DDS packing, using `struct` and `numpy`. Write vertex blocks with `numpy.ndarray.tobytes()` on a `float32[N,11]` array (pos 3, normal 3, uv 2, tangent 3, which is exactly the 44-byte layout) and indices as `uint16`.

Proposed module `kn5/writer.py`:
- `Kn5Builder.add_texture(name, bytes)`
- `add_material(name, shader, props: dict[str, float | tuple], maps: dict[slot, tex], blend, alpha_tested, depth)`
- `add_mesh(name, verts[N,11], idx[M], material, cast_shadows, renderable, lod_in, lod_out)`, which auto-splits > 65,535 verts into `name`, `name_SUB1`, …
- `add_dummy(name, matrix4x4_rows)`
- `write(path, version=5)`

Tests:
1. Round-trip through an independent parser (nazariyKul `kn5.py`) to exact EOF.
2. Golden-byte tests for tiny cases.
3. CM Showroom opening the file (manual).
4. In-game S1.

### 4.9 FBX → ksEditor path (fallback), and headless use

- Normal workflow: FBX 2012/2013 + `.fbx.ini` persistence file → ksEditor GUI assigns shaders → "Save KN5". VERIFIED (doc) basic guide. **ksEditor itself has no CLI.** VERIFIED (doc), multiple threads; no evidence of a headless mode.
- **ksEditorAt** adds a CLI: `ksEditorAt.exe kn5track out.kn5 in.fbx`.
  - se0lus/sam3_ac_track_mod_gen automates it. It downloads KsEditorAt v6, runs it with cwd containing `texture/` and `<name>.fbx.ini`, then runs `kn5FixByINI.exe out.kn5` to restore alpha flags. It also assigns `NULL.dds` to texture-less materials ("ksEditorAt fails on materials without any texture reference"). VERIFIED (code): https://github.com/se0lus/sam3_ac_track_mod_gen/blob/master/script/stages/s10_model_export.py
  - Windows only. Needs ksEditor from AC's SDK (Kunos-licensed, ships with the game).
- `.fbx.ini` format (written by CM's `Kn5Basic.ExportIni`). VERIFIED (code) `Kn5Basic.cs` L960–1087:
  - `[HEADER] VERSION=3`
  - `[MATERIAL_LIST] COUNT=n`
  - `[MATERIAL_i] NAME, SHADER, ALPHABLEND, ALPHATEST, DEPTHMODE, VARCOUNT, VAR_k_NAME, VAR_k_FLOAT1..4, RESCOUNT, RES_k_NAME, RES_k_SLOT, RES_k_TEXTURE`
  - `[model_FBX: <fbx>_<node>] ACTIVE, PRIORITY, VISIBLE, TRANSPARENT, CAST_SHADOWS, LOD_IN, LOD_OUT, RENDERABLE`
- FBX writing without Autodesk's SDK: Blender headless (`bpy` FBX export), or AcTools' bundled FbxSharpie (MIT, C#) binary/ASCII FBX writer, or Python libs. More moving parts than writing KN5 directly.
- **Verdict:** keep it only as fallback F2 for "AC rejects our KN5 for a reason we can't find". With ksEditorAt we can diff a ksEditor-made KN5 of the same FBX against ours byte by byte. That is a very effective debugging tool, so set it up on the user's Windows PC during M0 if S1 fails.
- **3DSimED** (commercial) exports KN5 directly. Not considered.

---

## 5. Shaders and materials for generated content

Shader names refer to Kunos' compiled shaders (`system/shaders`). We only reference them by name and never ship them.

Property values: Hagnhofer's statistics over Kunos content, "Shader values template" spreadsheet (2015, so it pre-dates `ksMultilayer_fresnel_nm`). VERIFIED (doc): https://docs.google.com/spreadsheets/d/1kbLNglwh05kwLJV_f5jSqhKPjAkqdtsitZb4CjaJVIY

| Shader | Use for | Properties (A = float valueA unless noted) | Texture slots |
|---|---|---|---|
| `ksPerPixel` | Anything simple (v1 everywhere) | `ksAmbient` (avg 0.53), `ksDiffuse` (0.85), `ksSpecular` (0.42), `ksSpecularEXP` (avg 115; use 10–50 for matte), `ksEmissive` (**float3 in valueC**), `ksAlphaRef` | `txDiffuse` |
| `ksPerPixelNM` | Road / rock with normal map | + `fresnelC`, `fresnelEXP`, `fresnelMaxLevel`, `isAdditive`, `nmObjectSpace` (0) | `txDiffuse`, `txNormal` |
| `ksPerPixelMultiMap` (+`_NMDetail`) | Guardrails, signs | + `detailUVMultiplier`, `useDetail`, `shadowBiasMult`, fresnel* | `txDiffuse`, `txNormal`, `txMaps` (R spec, G gloss, B reflection/AO), `txDetail` (+`txNormalDetail`) |
| `ksPerPixelAT` / `ksPerPixelAT_NM` | Alpha-tested foliage cards, fences | as ksPerPixel, `ksAlphaRef` ≈ 0.3–0.5. Material `alphaTested=true`, often `blendMode=AlphaToCoverage(2)`. | `txDiffuse` (+`txNormal`) |
| `ksTree` | Tree billboards / cards | `ksAmbient` 0.2–0.7, `ksDiffuse` 0.1–0.55, `ksSpecular` 0, `ksSpecularEXP` 1–10, `ksAlphaRef` | `txDiffuse` (ASSUMED only one) |
| `ksGrass` | Vanilla 3D grass cards | `gain` 0.5–1, `scale` (valueB float2), `ksAlphaRef`, standard ks* | `txDiffuse` (+ maybe `txVariation`, ASSUMED) |
| `ksMultilayer` / `ksMultilayer_fresnel_nm` | Terrain/road blending 4 detail textures via an RGBA mask | `ksAmbient`, `ksDiffuse`, `ksSpecular`, `ksSpecularEXP`, `magicMult` (0.75–2.3), `multR/G/B/A` (valueA = detail tiling multiplier per channel), `ksAlphaRef`. `_fresnel_nm` adds fresnel* (ASSUMED). | `txDiffuse`, `txMask`, `txDetailR`, `txDetailG`, `txDetailB`, `txDetailA` (VERIFIED (code) CM `CustomTracksBakery/Kn5MaterialToBake.cs` reads exactly these for any shader containing "Multilayer"), + `txNormal` for `_nm` (ASSUMED) |

Plan:
- **v1 (M0/M1): `ksPerPixel` + `txDiffuse` everywhere.** The road uses a tiling asphalt texture with UV u = lateral metres / 4, v = distance / 8. This is the shader all working generators used, so it's the lowest-risk choice. Typical values: ksAmbient 0.4–0.5, ksDiffuse 0.4–0.7, ksSpecular 0.1–0.2, ksSpecularEXP 10–15. A 4-float property record with only valueA set (rest zero) is fine. VERIFIED (code) KinG/Hagnhofer defaults.
- **v2:** road → `ksPerPixelNM` (asphalt normal map). Terrain → `ksMultilayer` with a generated `txMask` per terrain tile (R grass, G rock, B dirt, A gravel from slope and OSM landuse). That gives far better visuals with few textures. Guardrail → `ksPerPixelMultiMap`.
- **Trees:** `ksTree` cards (alphaTested, AlphaToCoverage) or, preferably with CSP, CSP procedural trees (§7). Trees on `ksPerPixelAT` also work.
- `slot` in texture mappings: writers simply number 0,1,2… Hagnhofer: "unknown, seems to be best to just increase it". VERIFIED (doc). Slot semantics ASSUMED irrelevant.
- **Road markings:** a separate thin mesh slightly above the road (+2–5 mm), `ksPerPixelAT` or alpha-blended with `depthMode` 0 and `isTransparent`. Not physical (no digit prefix), or physical as `1ROAD` with its own material. CSP `[RAIN_FX] LINES_MATERIALS` makes them slippery when wet.

---

## 6. Performance and limits

| Item | Value | Evidence |
|---|---|---|
| Vertices per mesh node | ≤ 65,535 (uint16 indices) | VERIFIED (code). Practical target ≤ 20–30k so culling spheres stay tight. ASSUMED. |
| Draw calls | ≈ 1 per visible mesh node × material. A 10×10×10 cube array raised AC's DIP counter by exactly 1000. A track author found > 4000 DIP too heavy, especially with AI (CPU-bound). | VERIFIED (doc): https://www.assettocorsamods.net/threads/maximize-performance-merge-all-track-objects.777 |
| Our budget | ≤ ~1000–1500 DIP in view. Main menu FPS must stay > 100 on the user's PC. | ASSUMED. Measure in S12 with AC's render stats / CSP. |
| Road chunking (10+ km) | 150–250 m chunks per surface type: `1ROAD_c###`, `1GRAVEL_c###`, plus walls (`1WALL_L_c###`, `1WALL_R_c###`, non-renderable), plus visual guardrails. 10 km ≈ 50 chunks × ~4 nodes ≈ 200 nodes, mostly culled. | ASSUMED. Matches culling via bounding spheres. |
| Terrain | Tiles of ~250 × 250 m near the corridor at 2–5 m grid. Coarser rings further out (10–30 m grid, 500 m–1 km tiles) with `lodOut`. Distant backdrop/horizon mesh with `castShadows=false`. | ASSUMED. Corridor width is a section D decision. |
| Shadows | `castShadows=false` on terrain far from the road, backdrop, invisible walls and road markings. AC's shadow cascades come from `cameras.ini SHADOW_SPLIT*` (replay) and the game's own settings. | ASSUMED. |
| Objects / props | `AC_POBJECT_*` ≤ ~15. Scenery should be batched (many trees per mesh node, per chunk), not one node per tree. | VERIFIED (doc) guide for POBJECT. Batching ASSUMED. |
| KN5 size | No hard limit seen (uint32 lengths → < 4 GB). Split into several KN5s via `models.ini` (road+dummies / terrain / scenery) so a scenery rebuild doesn't touch physics. | ASSUMED. |
| Load time | Physics meshes are processed at load. Keep collision geometry lean (road + walls + terrain near road). Far terrain can be visual-only (no digit prefix) behind invisible world-bound walls. | ASSUMED, spike S12. |
| Float precision | Local origin at route centre, coordinates within ±10 km. | ASSUMED fine for float32. |

---

## 7. CSP (Custom Shaders Patch) extras (optional; vanilla must keep working)

- **Location:** `content/tracks/<id>/extension/ext_config.ini`. VERIFIED (doc) forum/moderator: https://www.gtplanet.net/forum/threads/assetto-corsa-pc-mods-general-discussion.307899/post-12962360
  - CSP also ships configs for known tracks via its own `extension/config/tracks/`. For our tracks we own the in-folder file.
  - Official docs index: https://cup.acstuff.club/docs/csp/tracks/grass-fx (sidebar lists every track feature).
- **GrassFX:** `[GRASS_FX] GRASS_MATERIALS = terrain_grass`, `OCCLUDING_MATERIALS = road, gravel`, mask thresholds `MASK_MAIN_THRESHOLD/RED/MIN_LUMINANCE/MAX_LUMINANCE`, shape `SHAPE_SIZE/TIDY/CUT/WIDTH`. Grass grows where the surface colour passes a "green-ish" mask. VERIFIED (doc): https://cup.acstuff.club/docs/csp/tracks/grass-fx. **Cheapest big visual win for generated terrain.**
- **RainFX:** `[RAIN_FX] PUDDLES_MATERIALS = road`, `SOAKING_MATERIALS = road`, `ROUGH_MATERIALS = terrain_grass, gravel`, `LINES_MATERIALS = road_lines`. `[TRACK_OCCLUDER_BOX_n]` for tunnels and bridges. VERIFIED (doc): https://cup.acstuff.club/docs/csp/tracks/rainfx
- **Trees (CSP ≥ 0.1.79):**
  - `[TREES] LIST_0 = trees/trees.txt`, `SURFACE_MATERIALS = terrain_grass`, with lines like `tree: pine.bin; x, y, z` and `configure: size variance = 0.8, 1.2`.
  - Trees snap to the surface and get LOD/billboards automatically.
  - `.bin` models come from FBX via CM's "tree converter". Its source is in AcTools `AcTreeModelConverter.cs` (MS-PL), so later we could even produce `.bin` ourselves.
  - VERIFIED (doc): https://cup.acstuff.club/docs/csp/tracks/trees
  - **Ideal for OSM-forest scenery: we only emit a text list of positions.** Vanilla users just see no trees, or we also emit cheap `ksTree` cards in KN5.
- **Lights:** `[LIGHT_SERIES_…] MESHES = lamppost? ; CONDITION = NightLights` or per-light `POSITION/DIRECTION/COLOR/RANGE/SPOT`. Use them for street lamps from OSM `highway=street_lamp`. VERIFIED (doc): https://cup.acstuff.club/docs/csp/tracks/lights
- **Surface extras** in surfaces.ini: `_EXT_SURFACE_TYPE=GRAVEL/GRASS` for SurfacesFX dust. VERIFIED (doc).
- **VAO (vertex AO):** CM's `CustomTracksBakery` CLI bakes a `.vao-patch` for a KN5 (Windows, DirectX). It's an optional post-process on the user's PC. VERIFIED (code) `CustomTracksBakery/Program.cs` options.
- Other CSP track features (collision parameters, geometric colliders, dynamic objects, local cubemaps, water shader) are listed in the docs index. Not needed early.

---

## 8. Existing procedural / real-world AC track generators

| Project | What it does | Output path | Licence | Reuse |
|---|---|---|---|---|
| **Hemphill39/AssettoCorsaAutoXGenerator** | GPS logs → autocross course (cones, walls, timing, `fast_lane.ai` from driven line, map/preview/outline PNGs, CM-installable zip). Electron app. | Direct KN5 v5 + PNG. **In-game verified.** | none | Facts only: format spec, in-game lessons (Z = south, L/R, PIT spawn, pitboxes cap, `circuit` tag). |
| **KinG-InFeT/blender-assetto-corsa-track-generator** | Draw circuit on satellite map → Blender build with SRTM elevation, banking, kerbs, grass, segmented walls → KN5 → mod folder (layouts, surfaces, ui, outline, AI line). | Headless Blender + own KN5 v6 writer. | **Apache-2.0** | Can reuse code with attribution: centreline AI line, outline drawing, folder setup. Closed circuits only. |
| **JostMK/track-mapper** | OSM road graph (Geofabrik pbf → `.fmi` via OsmGraphCreator) + Dijkstra routing + GDAL DEM → road/terrain mesh → **FBX** via Autodesk SDK. Web UI. Spawns `AC_PIT_0/AC_START_0`. | FBX, then ksEditor by hand. | GPL-3 | Closest prior art to "pick A→B on OSM". Concepts only, since GPL and C++/FBX-SDK. |
| **se0lus/sam3_ac_track_mod_gen** | Drone ortho + 3D tiles → SAM3 segmentation → Blender → FBX split → **ksEditorAt CLI** → KN5 → packaging. | FBX + ksEditorAt (Windows). | none seen | Shows the automated ksEditorAt fallback works. |
| YannThorimbert/PyTrackAssettoCorsa | STL generator for track design. | STL (manual import). | ? | Minor. |
| Race Track Builder (commercial) | GUI track builder with terrain/OSM-ish import and export to AC via FBX + ksEditor. | FBX→ksEditor. | Commercial | Not reusable. |
| BlenderGIS (domlysz) | Blender add-on that imports OSM + DEM (SRTM) into Blender. Common manual workflow with the moppius exporter. | Blender scene → exporter. | GPL-3 | Manual-pipeline reference only. |
| driftmap.us (the target) | Not open source. | – | – | – |

**Licence guidance:**
- AcTools is MS-PL. Reading it to learn the format is fine. If we *port* code, keep the MS-PL notice on those files, and MS-PL requires source distribution of those portions under MS-PL.
- GPL projects (moppius, nazariyKul, JostMK) are fine as reference and for dev-time tools, but don't vendor them into our package unless we go GPL.
- Apache-2.0 (KinG) and CC0 (KN5Export) code can be reused with notice.
- Hemphill and MarvinSt/RaduMC have no licence: read only, never copy.
- File formats themselves are facts.

---

## 9. Concrete generator outputs for M0/M1 (summary)

1. `<id>.kn5` (v5), root dummy `<id>` (identity) containing:
   - `1ROAD_c###`, `1GRAVEL_c###` (shoulders), `1GRASS_t###` (near-road terrain tiles, physical)
   - `terrain_far_t###` (visual only, `lodOut`)
   - `1WALL_L_c###` / `1WALL_R_c###` (`isRenderable=false`, `castShadows=false`), `1WALL_bound_##`
   - Dummies: `AC_START_0..7`, `AC_PIT_0..7`, `AC_HOTLAP_START_0`, `AC_AB_START_L/R`, `AC_AB_FINISH_L/R`, optional `AC_TIME_0_L/R`
   - Materials `road`, `gravel`, `grass`, `wall_hidden` (any shader; still needs a material), all `ksPerPixel`
   - Embedded textures
   - If the file grows large (> ~150 MB) or for rebuild speed: split into `<id>.kn5` (dummies + road + walls) and `<id>_terrain.kn5` via `models.ini`.
2. `data/surfaces.ini` (ROAD, GRAVEL, GRASS), `data/map.ini`, `data/cameras.ini`, `data/lighting.ini` (sun from lat/lon and a nominal date, optional), `data/groove.ini`, `data/crew.ini`.
3. `map.png`, `ui/ui_track.json`, `ui/preview.png`, `ui/outline.png`.
4. `ai/fast_lane.ai` from M2 (section H; v7 layout in AcTools `AcTools/AiFile/AiSpline.cs`). Do not ship a placeholder AI file: a broken one crashes to menu.
5. Optional `extension/ext_config.ini` (GrassFX + RainFX blocks referencing our material names).

---

## 10. Sources (main)

- gro-ove/actools @ e5e5c94 (MS-PL), https://github.com/gro-ove/actools:
  - `AcTools/Kn5File/{Kn5Writer.cs, Kn5Basic.cs, Kn5Node.cs, Kn5Material.cs, Kn5MaterialBlendMode.cs, Kn5MaterialDepthMode.cs}`
  - `AcTools/ExtendedBinaryWriter.cs`, `AcTools/ReadAheadBinaryReader.cs`, `AcTools/CommonAcConsts.cs`, `AcTools/Numerics/Mat4x4.cs`
  - `AcTools.ExtraKn5Utils/Kn5Utils/{Kn5MeshBuilder.cs, Kn5MeshUtils.cs}`, `AcTools.ExtraKn5Utils/LodGenerator/CarLodGenerator.cs`, `AcTools.ExtraKn5Utils/Helpers/AcTreeModelConverter.cs`
  - `AcTools.Render/Kn5SpecificSpecial/{TrackMapRenderer.cs, TrackOutlineRenderer.cs}`, `AcTools.Render/Kn5Specific/Objects/Kn5RenderableObject.cs`
  - `AcTools/Processes/Game.Properties.cs`, `AcTools/Utils/Helpers/GeoTagsEntry.cs`
  - `AcManager.Tools/Objects/{TrackObject.cs, TrackObjectBase.cs, TrackExtraConfigurationObject.cs}`
  - `AcManager/Pages/Drive/QuickDrive*.cs`, `AcManager/Pages/Selected/SelectedTrackPage.xaml.cs`, `AcManager/Pages/ServerPreset/ServerPresetRunningStatus.xaml.cs`
  - `AcManager/CustomShowroom/TrackMapRendererTools.xaml.cs`, `AcManager/Pages/ContentTools/{Bc7Upgrader.xaml, Uv2ModelConverter.xaml}`
  - `CustomTracksBakery/*`
- Hagnhofer docs: https://sites.google.com/site/hagn99/assettocorsa/modding/tracks (+ `/data/*`, `/track-kn5`, `/ui/ui_track-json`, `/kn5-files`), exporter FAQ, settings.json and changelog under https://sites.google.com/site/hagn99/assettocorsa/blender-kn5-exporter
- moppius exporter (GPL-3): https://github.com/moppius/blender-assetto-corsa-tools (`exporter/*.py`, `utils/constants.py`)
- KinG-InFeT generator (Apache-2.0): https://github.com/KinG-InFeT/blender-assetto-corsa-track-generator (`scripts/export_kn5.py`, `scripts/setup_mod_folder.py`)
- Hemphill39 AutoX generator: https://github.com/Hemphill39/AssettoCorsaAutoXGenerator (`PLAN.md`, `TESTING.md`, `packages/core/src/kn5/encode.ts`, `geometry/builders.ts`, `package/buildTrack.ts`)
- nazariyKul parser (GPL-3): https://github.com/nazariyKul/kn5-blender-tools (`kn5_blender_tools/kn5.py`, `LICENSE_AUDIT.md`)
- FatihEsen/KN5Export (CC0): https://github.com/FatihEsen/KN5Export
- se0lus/sam3 (ksEditorAt automation): https://github.com/se0lus/sam3_ac_track_mod_gen/blob/master/script/stages/s10_model_export.py
- ksEditorAt: https://ascobash.wordpress.com/2015/07/22/kseditor/
- ac_tools_cmd: https://github.com/leBluem/ac_tools_cmd
- JostMK/track-mapper (GPL-3): https://github.com/JostMK/track-mapper
- assettocorsamods basic guide: https://assettocorsamods.net/threads/build-your-first-track-basic-guide.12/
- A→B timing threads: https://assettocorsamods.net/threads/need-help-with-timing-objects-for-a-to-b-type-stage.1056/ and https://assettocorsamods.net/threads/timer-for-up-downhill.2260/
- Layouts: https://assettocorsamods.net/goto/post?id=21922
- Draw calls: https://www.assettocorsamods.net/threads/maximize-performance-merge-all-track-objects.777
- RTB crash/log thread: https://steamcommunity.com/app/388980/discussions/0/485624149166109832
- CSP docs: https://cup.acstuff.club/docs/csp/tracks/grass-fx, `/rainfx`, `/trees`, `/lights`, `/surface-tweaks`, `/general-options`
- Vanilla data file copies:
  - system surfaces: https://github.com/Ilcoups/RedLine-Souls-Assetto-Corsa-Server/blob/HEAD/system/data/surfaces.ini
  - Laguna Seca: https://github.com/Sezynex/AssettoServer-Traffic/blob/HEAD/content/tracks/ks_laguna_seca/data/surfaces.ini
  - Kunos ui_track.json: https://github.com/TylerBre/acserver_manager/tree/HEAD/seed/content/tracks
  - Real cameras/map ini: https://github.com/JSAE-ARCHIVES/FSAEJ-simulator/tree/HEAD/FSAEJ_Endurance/data
- Pillow DDS writer: https://github.com/python-pillow/Pillow/blob/main/src/PIL/DdsImagePlugin.py
- Spike (this session, throwaway): `/tmp/claude-0/-home-user-MapsAssetoCorsa/c1e65687-537c-507b-b23f-20abf8c39e66/scratchpad/scripts/spike/kn5_spike.py` → `test_strip.kn5` (1,613 bytes), parsed to EOF by nazariyKul `kn5.py`. **Note:** the spike's wall quad is degenerate (zero height) and its winding was not checked. It proves byte layout only.
