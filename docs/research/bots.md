# Bot cars on generated tracks: traffic and touge battles (research)

Research date: 2026-10-14. Scope: section H of `docs/PLANNING_PROMPT.md`.
Legend: **VERIFIED** means I read it in source code, shipped config or official docs (URL given).
**ASSUMED** means inference, community hearsay or memory that I did not confirm in a primary source.

Primary sources I read directly (cloned or downloaded, not summarised by a search engine):

| Source | Version read | What it gave |
|---|---|---|
| `compujuckel/AssettoServer` (GitHub, master) | commit `1365f3e`, 2026-10-06; latest tag `v0.0.55-pre44` | AI traffic implementation, `AiParams`, spline parser, lane detection, junction config |
| `compujuckel/assettoserver-docs` | master (0.0.54 = current stable docs) | Beginner's guide, junctions guide, FAQ |
| `ac-custom-shaders-patch/acc-lua-internal` | 2026-08-17 | Traffic Planner tool source, built-in New Modes (traffic, p2p-1v1, rally, revenant, splitscreen) |
| `ac-custom-shaders-patch/acc-lua-sdk` + generated SDK inside **CSP 0.2.11** zip (`extension/internal/lua-sdk/*/lib.lua`) | public CSP 0.2.11 (files dated 2025-06-11) | Real Lua API names and docs (physics.setAI*, ac.overrideCarControls ...) |
| CSP 0.2.11 `extension/config/new_behaviour.ini` | 0.2.11 | AI Flood, AI tweaks, Custom AI switch |
| `ac-custom-shaders-patch/acc-extension-config` wiki | 2026 | Custom AI (memory-mapped) interface, surfaces.ini permissions, car physics scripts |
| `C1XTZ/csp-logs` (CSP changelog archive) | newest = 0.3.0-preview675, 2026-10-07 | Versions, Revenant mode, AI spline changes |
| `AssettoCN/ACTougeBattle` | 2026-05-12 | Existing touge-battle server plugin (PvP) |
| `dasGringuen/assetto_corsa_gym` | master | RL in AC prior art (controls player car via vJoy) |

---

## Recommendation (staged path)

**Stage 1. Grip AI touge battle, offline (do this first).**
Ship a small **CSP "New Mode"** (a Lua game mode: `extension/lua/new-modes/<id>/manifest.ini + mode.lua`) called
for example "Touge Battle". The mode uses one stock AC AI car on our generated `fast_lane.ai`. Lua handles the countdown,
the side-by-side or rolling start, finish detection, cat-and-mouse rules and results, and tunes the AI with the CSP physics API
(`physics.setAILevel`, `setAIAggression`, `setExtraAIGrip`, `setAITopSpeed`, `setAIStopCounter` ...).
Do not rely on a vanilla AC race session on an A→B track. Community reports say the AI brakes and stops at the finish line,
and race/lap handling on non-looped tracks is flaky. Our track generator also controls the AI spline
(extend it 150–300 m past the finish into a run-off) and the spawn points (`AC_START_0/1` side by side at A).
Feasibility: **feasible**. Everything needed exists in the public CSP 0.2.11 API.

**Stage 2. Traffic.** Three tiers, all generated from the same OSM lane centrelines:
- 2a. **AssettoServer AI traffic** (SRP-style, real AC car models, two-way, junctions). This is the best match for
  "Shutoko-style" traffic. It runs fine as a local server on the player's own PC. We write a `fast_lane.aip`
  (one `.ai` per lane per direction plus a `config.yml`), `extra_cfg.yml` (`EnableAi: true`, `TwoWayTraffic: true`, `LaneWidthMeters`),
  `entry_list.ini` (`AI=fixed` slots) and `server_cfg.ini`. Feasibility: **feasible, highly automatable**.
- 2b. **CSP Traffic mode** (offline, single player, Lua kinematic cars, thousands of cars, two-way via directional lanes).
  We write `traffic.json` (lanes as 3D polylines) next to the layout's `surfaces.ini`, add the `traffic-planner-ready` tag, and enable
  extended physics plus `ALLOW_NEW_MODE_SCRIPTS=1` in `surfaces.ini`. The car models come from a separate download.
  Feasibility: **feasible, highly automatable**. Cars are simple LOD models, not real AC cars.
- 2c. **CSP "AI Flood"** (offline, Track Day mode, real AC AI cars, one direction only, uses `fast_lane.ai`). Zero extra
  data, but no oncoming traffic. Good free fallback.

**Stage 3. Drifting AI (stretch goal with its own research spike).** In order of effort:
- 3a. *Replay ghost opponent:* CSP 0.3.0-preview's **Revenant** mode plays a recorded human run as a ghost with real
  collisions. It is not reactive, the user must record the drift run, and it needs the Patreon preview build.
- 3b. *Lua drift controller* inside our New Mode. It steers the AI car with `physics.overrideSteering` / `ac.overrideCarControls`
  and controls body slip angle with throttle (PID / cascaded controller on a "drift line"). The open question is whether AI pedal
  inputs can be fully suppressed: override pedals are combined with the AI's own using `max()`.
- 3c. *External controller* via CSP's **Custom AI** memory-mapped interface. It gives full gas/brake/steer/handbrake/clutch/gear
  control at 333 Hz plus per-wheel slip-angle telemetry, and it is allowed only when the track's `surfaces.ini` permits it,
  which we control. Highest control fidelity, but needs a helper exe running next to AC.
Feasibility: 3a feasible now (preview only). 3b/3c are **hard / research project**: a robust drift controller per car class is
real control engineering, but every API it needs exists. Plan a time-boxed spike (below) before committing.

**Things to bake into the track generator now (cheap, unlocks everything above):**
1. Many pit boxes (`AC_PIT_0..N`, e.g. 24–40). AssettoServer entry lists cannot exceed the pit count, or clients crash (VERIFIED, AssettoServer beginner's guide).
2. Two grid spots side by side at A (`AC_START_0`, `AC_START_1`), plus `AC_AB_START_L/R` and `AC_AB_FINISH_L/R` timing gates for A→B (community naming, ASSUMED standard).
3. A run-off past B, and an AI spline extended past the finish line.
4. Per-lane centreline polylines in both directions (world coordinates, ~1 m spacing), kept as an intermediate artefact so the
   generator can emit `fast_lane*.ai`, `traffic.json` and OBJ lane files from one source.
5. `surfaces.ini` options: a "CSP-extended" variant with `WAV_PITCH=extended-0` plus `[_SCRIPTING_PHYSICS]` / `[_EXTRA_PERMISSIONS]`.
   This crashes vanilla AC without CSP, so make it optional (VERIFIED, CSP wiki).
6. `ui_track.json` tags: `traffic-planner-ready` (CSP Traffic mode) and a custom `touge` tag for our New Mode's `ALLOWED_TRACKS`.
7. Driving side per country (Japan/UK left, most others right) to place lanes correctly.

---

## Spike (time-boxed experiments before committing to the plan)

Each spike uses one generated test road (about 3–5 km, two lanes, a few hairpins). Pass criteria are in brackets.

| # | Spike | Steps | Pass criteria | Effort |
|---|---|---|---|---|
| S1 | AssettoServer two-way traffic | Write `fast_lane_fwd.ai` + `fast_lane_bwd.ai` (format v‑1, see A.1) at ±LaneWidth/2 from the centreline, zip as `fast_lane.aip` with `config.yml`. Run AssettoServer locally with 10 `AI=fixed` Kunos cars, `TwoWayTraffic: true`. Join via CM → Online → LAN. | Log shows "Adjacent lane detection" with no errors; oncoming cars spawn; cars despawn at spline ends; nothing floats or sinks (else tune `SplineHeightOffsetMeters`). | 1–2 days |
| S2 | CSP Traffic Planner data | Write `traffic.json` (2 lanes, no `aabb`, so CSP resamples and raycasts) next to `surfaces.ini`; enable `extended-0` + `ALLOW_TOOLS=1` + `ALLOW_NEW_MODE_SCRIPTS=1`; tag layout. Open Objects Inspector → Tools → Traffic planner in Practice; then run the Traffic New Mode. | Lanes appear on the road; cars run in both directions; the file survives a save/load round-trip in the planner (diff it to learn the canonical format). | 1 day |
| S3 | Touge New Mode MVP (grip) | Mode with `BASE_MODE=RACE` (also try `TRACK_DAY`), `AI_LIMIT=1`, side-by-side start, finish gate from track metadata, `ac.endSession()` with result. Compare with a plain vanilla 1-lap race on the same track. | AI drives A→B without stopping before B; the mode declares a winner; works on public CSP 0.2.11. | 2–4 days |
| S4 | Can Lua fully drive an AI car? | In a New Mode with `ALLOW_PHYSICS_ALTERATIONS=1`, take AI car 1 and test: (a) `physics.overrideSteering(1, x)`; (b) `ac.overrideCarControls(1)` gas/brake/handbrake after `physics.setAIThrottleLimit(1, 0)` and after `physics.setAINoInput(1, true, false)`; log `ac.getCar(1).gas/brake/steer`. | We find a combination where our pedal values are applied exactly (not max-combined with the AI's). If none, fall back to S5. | 1 day |
| S5 | Custom AI (external) | `surfaces.ini` `[_EXTRA_PERMISSIONS] ALLOW_CUSTOM_AI_MANIPULATION=1`; small Python/C# process creates `AcTools.CSP.NewBehaviour.CustomAI.CarControls1.v0`, reads `...Car1.v0`. | Full control of car 1 at ≥100 Hz; slip angles readable. | 1–2 days |
| S6 | Drift controller prototype | Using S4 or S5: open-loop donut, then closed-loop constant-radius drift on a skidpad (target β = 25°), then one hairpin on the test road. RWD drift car. | Holds β ±5° for 3 s on the skidpad; completes one hairpin sideways without spinning in 8 of 10 runs. | 1–3 weeks |
| S7 | Replay-ghost opponent | CSP 0.3.0-preview: Revenant mode on the generated track, recording a human drift run. | Ghost replays with collisions; decide whether "record your own rival" is an acceptable UX. | 0.5 day |

---

## A. TRAFFIC

### A.1 AssettoServer AI traffic (compujuckel/AssettoServer)

**How it works (VERIFIED, source):**
- AI cars are **server-side kinematic puppets**, not physics-simulated AC AI. Each tick the server moves an "AI state" along a
  spline with a Catmull-Rom interpolation and sends the result to clients as a normal remote car: position, velocity,
  rotation from the spline tangent, camber as roll, fixed steer angle, fake wheel speed, lights, hazards, horn and indicators
  (`AssettoServer/Server/Ai/AiState.cs`, `Update()`). Speed comes from the spline's curvature radius:
  `v_max² = g · CorneringSpeedFactor · radius` (`Utils/PhysicsUtils.cs`). Obstacle and collision logic brakes cars
  (`DefaultDeceleration`, `MinCollisionStopTimeSeconds`). When a car reaches the end of a non-looped spline it despawns
  ("reached spline end, despawning"), so **point-to-point splines are supported**.
- Spawning happens around each player: between `MinSpawnDistancePoints` (100) and `MaxSpawnDistancePoints` (400) spline points
  ahead. Cars outside `PlayerRadiusMeters` (200) despawn. A player only spawns traffic within `MaxPlayerDistanceToAiSplineMeters` (7 m)
  of a spline. AI "slots" are overbooked: one entry-list car can appear several times far apart (`StateSpawnDistanceMeters` 1000).
- Master switch: `EnableAi: true` in `cfg/extra_cfg.yml` (default false, `ACExtraConfiguration.cs`).

**Track data it needs (VERIFIED, `Splines/AiSplineLocator.cs`, `FastLaneParser.cs`):**
- Folder `content/tracks/<track>/ai/`, or `content/tracks/<track>/<layout>/ai/` if it exists, **in the server folder**
  (not the client's AC install). It falls back to `content~tmp` when CM's "disable integrity verification" is on.
- Either a `fast_lane.aip` (a **zip renamed to .aip** with all `*.ai` files plus `config.yml`; when present, every other file is ignored)
  or loose `fast_lane*.ai` files (all files matching `fast_lane*.ai`, sorted by name) plus an optional `config.yml`.
- On startup it builds a cache: `cache/<xxhash64>.aic<version>`, keyed on the `.aip` or on `config.yml` + all `fast_lane*.ai`
  ("Cached AI spline not found. Generating cache..."). There is no separate "map.aip"; the preprocessing output is this `.aic` cache.
- Spline file formats accepted (`FastLaneParser.cs`, `FastLaneUtils/FastLane.cs`):
  - **v7 (AC native)**: `int32 7, int32 count, int32 lapTime, int32 sampleCount`, then per point `float x,y,z, float length, int32 id`,
    then `int32 extraCount`, then per point 18 floats: `speed, gas, brake, obsoleteLatG, radius, sideLeft, sideRight, camber,
    direction, normal.xyz, detailLength, forward.xyz, tag, grade`. The server only uses position, radius and camber×direction.
  - **v‑1 ("optimized", AssettoServer-only)**: `int32 -1, int32 count`, then per point `float x, y, z, radius, camber`. **Trivially writable
    by our generator.** `FastLaneUtils --optimize` converts v7 to v‑1, `--legacy` writes v7, and `--reverse` reverses a spline.
  - A spline is a closed loop if its first and last points are < 50 m apart; otherwise it is point-to-point.
  - `radius` must be a positive curvature radius in metres (it drives corner speed). If the last point's radius is < 1 it is reset to 1000.
  - Point spacing: the defaults count spawn distance in points (100–400), which implies roughly 1 m spacing as in AC-recorded splines (ASSUMED). Generate about 1 m.

**Multiple lanes and opposite direction (VERIFIED, `AdjacentLaneDetector.cs`, `SplinePointOperations.GetLanes`):**
- There is no lane-count field. **Each lane is its own spline file.** At load the server finds neighbours by probing
  `LaneWidthMeters` (default 3.0) to the left and right of each point and accepting the nearest point within 2.0 m
  (`LaneDetectionRadius`). The minimum lane width is 2.1 m.
- Direction is taken from each spline's point order. If the neighbour's forward vector has a positive dot product with ours, it is
  a same-direction lane; otherwise it is an opposite-direction lane.
- `TwoWayTraffic: true` lets AI spawn on opposite-direction lanes relative to the player's direction. `WrongWayTraffic` (default true)
  spawns traffic when the player drives against the spline (only when TwoWayTraffic is false).
- So for a two-way road: write `fast_lane_a.ai` (A→B, offset to the driving side) and `fast_lane_b.ai` (B→A, reversed order, other
  side), spaced exactly `LaneWidthMeters` apart (±2 m tolerance). Because `LaneWidthMeters` is a **server-wide** setting, pick
  one spacing per track (for example the route's median lane width, clamped to 2.5–3.5 m) and write it into `extra_cfg.yml`.
- Related settings: `LaneCountSpecificOverrides` (per-lane-count speed and spacing tweaks), `RightLaneOffsetKph` (+10 km/h on
  right lanes), `CarSpecificOverrides` (incl. `AllowedLanes: Left/Middle/Right`).

**Junctions (VERIFIED, `MutableAiSpline.ApplyConfiguration`, docs `ai-splines/junctions_guide.md`):** `config.yml` in the ai folder:
```yaml
Track: <track folder name>        # optional; refuses to load on another track
Author: MapsAssetoCorsa
Version: 1
Splines:
- Name: fast_lane_main_a.ai
  Junctions:
  - Name: main -> side road
    Start: 1234                 # point index in this spline
    End: fast_lane_side_a.ai@0  # <file>@<point index>
    Probability: 0.2
    IndicateWhenTaken: Left     # None/Left/Right
- Name: fast_lane_side_a.ai
  ConnectEnd: fast_lane_main_a.ai@5678   # join end of this spline into another
```
For a single A→B touge route we need no junctions. They only matter if we later add side roads from OSM.

**Key `AiParams` (VERIFIED, `Configuration/Extra/AiParams.cs`; defaults in parentheses):** `MaxSpeedKph` (80), `RightLaneOffsetKph` (10),
`MaxSpeedVariationPercent` (0.15), `CorneringSpeedFactor` (0.65), `CorneringBrakeDistanceFactor` (3), `CorneringBrakeForceFactor` (0.5),
`MinAiSafetyDistanceMeters`/`Max…` (20/70), `Min/MaxSpawnDistancePoints` (100/400), `PlayerRadiusMeters` (200),
`AiPerPlayerTargetCount` (0 = auto = number of AI-capable slots), `MaxAiTargetCount` (0 = auto), `TrafficDensity` (1.0),
`HourlyTrafficDensity` (24 values), `LaneWidthMeters` (3.0), `TwoWayTraffic` (false), `WrongWayTraffic` (true),
`SplineHeightOffsetMeters` (0), `SmoothCamber` (true), `HideAiCars` (true), `NamePrefix` ("Traffic"), `EnableDaytimeLights`,
`IgnorePlayerObstacleSpheres`, `Debug`. The docs note that defaults are tuned for Shutoko Revival Project. For a narrow mountain road, lower `MaxSpeedKph` to about 50–60.

**Car requirements (VERIFIED):**
- `entry_list.ini` per car: `AI=fixed` (always AI, players can't take it), `AI=auto` (AI unless a player joins; then also set
  `MaxPlayerCount` in extra_cfg), or `AI=none` (default). With `AutoAssignTrafficCars: true` (default), any car whose model folder
  contains "traffic" becomes `AI=fixed` automatically (`ACServerConfiguration.ApplyConfigurationFixes`).
- Any installed car works as traffic, including Kunos cars, so no mods are strictly required. Clients must have the car models installed.
- **Entry count ≤ track pit boxes**, otherwise the game crashes on join (docs, beginner's guide). That is why the generator should emit many `AC_PIT_n`.
- Density depends mostly on how many AI cars are in the entry list (FAQ "dense traffic").
- Client CSP: the docs (0.0.54) say CSP 0.1.77 (1937) is required by default; current master defaults `MinimumCSPVersion` to CSP 0.2.0. Both are configurable.

**Running it locally as a single player (VERIFIED steps from docs, our automation ASSUMED):**
1. Download `assetto-server-win-x64.zip` from GitHub releases (stable v0.0.54, or the latest preview v0.0.55-pre44). Extract it **outside** the AC folder (Steam SDK conflict, per the docs).
2. Our tool writes `cfg/server_cfg.ini` (TRACK, CONFIG_TRACK, CARS, MAX_CLIENTS, a long practice session), `cfg/entry_list.ini`
   (1 player car `AI=none` + N traffic cars `AI=fixed`), `cfg/extra_cfg.yml` (`EnableAi: true`, `AiParams: {TwoWayTraffic: true, LaneWidthMeters: …, MaxSpeedKph: …}`),
   and `content/tracks/<track>/ai/fast_lane.aip`.
3. Run `AssettoServer.exe` (accept the Windows firewall prompt; no port forwarding is needed for local play).
4. In Content Manager: Drive → Online → **LAN** tab → select the server → pick the player car → Join. Drive out of the pits and traffic spawns.
   (Alternative from the docs: rename CM's `server/acServer.exe`, drop AssettoServer in as `acServer.exe`, and run presets from CM's Server tab.)
5. Licence: AGPL-3.0. We can tell the user where to download it, or redistribute the unmodified release with its licence (ASSUMED OK; check before bundling).

Effort for the user is moderate: one extra download and a double-click. Our tool can generate and launch the whole folder.

### A.2 CSP offline traffic

There are three different CSP things, all **offline only**.

**(a) Traffic Planner tool + "Traffic" New Mode (VERIFIED, `acc-lua-internal/included-tools/traffic`, `included-new-modes/traffic`; ships in CSP 0.2.11 as `extension/lua/tools/csp-traffic-tool` and `extension/lua/new-modes/traffic`):**
- The cars are **Lua-driven kinematic rigid bodies** (`physics.RigidBody`) using LOD C/D car meshes. They switch to physics on contact.
  The author writes that it "can simulate 2000 cars in less than 3 ms". They are not AC cars. The car models must be downloaded separately
  (`data.zip`, link in the tool's readme/manifest: `https://files.acstuff.club/TCKo/data.zip`) and extracted to `csp-traffic-tool/data`.
- Data file: **`traffic.json`**, path from `ac.getTrackDataFilename('traffic.json')`, i.e. next to the layout's `surfaces.ini` (readme).
  CSP also caches a binary version in `ExtCache/traffic/<track>.bin`, and can download per-track data from `https://acstuff.club/u/traffic/<trackid>.zip`.
- Schema (`src/common/SerializedTypes.lua`, `EditorLane:encode`, `TrafficLane:initialize`):
```jsonc
{
  "lanes": [
    { "id": 1, "name": "A->B", "loop": false,
      "points": [[x,y,z], ...],          // AC world coords, in driving order (direction = point order)
      "role": 2,                          // optional; index into rules.laneRoles (default 3 = "Main")
      "priorityOffset": 0,
      "params": { "allowLaneChanges": true, "allowUTurns": false } },   // params is required
    { "id": 2, "name": "B->A", "loop": false, "points": [...], "params": {...} }
  ],
  "intersections": [],   // polygons + trajectory rules; optional for a single road
  "areas": [],           // polygons overriding speed/role/lane-change rules
  "rules": { "laneRoles": [
      {"name":"Parking","priority":-8,"speedLimit":40}, {"name":"Secondary","priority":-4,"speedLimit":60},
      {"name":"Main","priority":0,"speedLimit":80},     {"name":"Highway","priority":4,"speedLimit":90} ] }
}
```
  Lanes without an `aabb` field are treated as "raw from editor": CSP resamples them every 3 m and **raycasts them onto the track
  surface** (`TrafficLane.lua`), so we can emit plain centreline points. Speed slows automatically in curves (lerp on straightness).
  When a car reaches a lane end it is released and gets a new random path, which suits A→B roads. Two-way traffic is simply two
  directional lanes. Lane changes happen between parallel lanes.
- The tool can also **import lanes from an OBJ** ("Replace lines with OBJ file": `o <name>` + `v` + `l` polylines, coordinate swap `vec3(x, z, -y)`).
  That is an alternative export path.
- To enable it on a track (readme + mode `description.txt`): in `surfaces.ini` set `[SURFACE_0] WAV_PITCH=extended-0`
  and `[_SCRIPTING_PHYSICS] ALLOW_NEW_MODE_SCRIPTS=1` (and `ALLOW_TOOLS=1` while editing). Add the tag
  **`traffic-planner-ready`** to the layout (the mode's manifest has `ALLOWED_TRACKS=#traffic-planner-ready`). Then pick the
  "Traffic" New Mode with density 100–5000 cars, a speed multiplier and "carnage mode".
- Caveats: x4fab calls it "a small experiment ... pretty rough shape". There is no replay support, and the cars look basic.
- `lua-shared/sim/traffic.lua` is "a library allowing to quickly add traffic to new modes" (`traffic.lookup()` + `traffic.run()`),
  so **our touge New Mode could add oncoming traffic** during a battle. The mode's source asks authors to contact x4fab first
  "so we could plan things in a way that wouldn't break in the future".

**(b) "AI Flood" (New AI behaviour module; VERIFIED in CSP 0.2.11 `extension/config/new_behaviour.ini`):**
- Works in **Track Day** mode with ordinary AC AI cars on the track's `fast_lane.ai`. It despawns AI behind you and respawns it ahead
  ("seemingly endless traffic"). Settings: `MIN_TRACK_LENGTH` (default **7000 m**, configurable 1000–13000, so short touge roads
  need it lowered in CM → Settings → Custom Shaders Patch → New AI behavior), `SHUFFLE_BEHAVIOUR`, `SPEED_LIMIT=80,160`, push force/speed.
- `[WRONG_WAY] ALLOW_IN_PRACTICE/ALLOW_IN_TRACKDAY=1` lets the player drive against the flow, which gives you "oncoming" traffic if you drive B→A.
- One spline means one direction. True two-way would need per-car alternate splines: `physics.setAISpline(carIndex, file)` exists, and
  CSP's own `ai_tweaks.lua` randomly assigns `ai/ext_alt_fast_lane_*.ai` to race AIs. Whether AI Flood respawning respects a
  per-car reversed spline is untested (ASSUMED, spike candidate).

**(c) Custom Lua traffic:** possible with the same APIs (RigidBody or AI cars plus `physics.setAICarPosition`), but (a) already exists.

**Can it be generated from lane centrelines?** Yes, for all of the above. (a) needs only a JSON list of directional polylines.
(b) needs only `fast_lane.ai`, which we generate anyway.

### A.3 Most automatable path from OSM, and what we write

From OSM per way: `highway`, `oneway`, `lanes`, `lanes:forward`, `lanes:backward`, `width`, `lane_markings`, `maxspeed`, plus the country's driving side.
Rules (ASSUMED defaults, tune later):
- Lanes per direction: `lanes:forward/backward` → else `lanes/2` → else 1 each way (typical touge: `highway=unclassified/tertiary`, often no `lanes` tag).
- Lane width: `width / total lanes` → else a class default (about 2.75 m minor roads, 3.0–3.25 m primary).
  For AssettoServer, snap the spacing to one global `LaneWidthMeters`.
- Lane centre offset `i` (1 = nearest the centreline) = `(i − 0.5) · w` toward the driving side (left in JP/UK/AU, right elsewhere).
  Reverse the point order for the opposite direction.
- Speed: `maxspeed` → lane `speedLimit` (CSP) / `MaxSpeedKph` (AssettoServer, global) / AI hints; curvature radius per point for the `.ai` file.

Formats to write:

| Target | Files | Where |
|---|---|---|
| AssettoServer traffic | `fast_lane_<dir>_<n>.ai` (v‑1, or v7 to also open in AC tools) + `config.yml` → zipped as `fast_lane.aip`; `cfg/extra_cfg.yml`, `cfg/entry_list.ini`, `cfg/server_cfg.ini` | server folder `content/tracks/<track>/ai/` + `cfg/` |
| CSP Traffic mode | `traffic.json` (optionally an OBJ of lanes); edits to `surfaces.ini` (`extended-0`, `[_SCRIPTING_PHYSICS] ALLOW_NEW_MODE_SCRIPTS=1`); `ui_track.json` tag `traffic-planner-ready` | track layout `data/` folder |
| AI Flood / race AI | `ai/fast_lane.ai` (v7, AC-native; generated by the AI-line work in section H), optional `data/ai_hints.ini` | track layout |

Note: the client-side `ai/fast_lane.ai` (racing line for offline AI) and the server-side traffic lane splines are **different files in different
folders**. The AssettoServer FAQ even recommends renaming the client's `ai` folder if loading a big single-player spline stalls joining ("Initialising AI spline").

---

## B. TOUGE BATTLE

### B.4 Point-to-point races with AI

- **Vanilla AC:** A→B tracks exist (hillclimbs, Kunos' Trento-Bondone) and use `AC_AB_START_L/R` / `AC_AB_FINISH_L/R` timing objects
  (community modding guides; ASSUMED canonical) instead of a start/finish loop. Practice, Hotlap and Time Attack work.
  Race sessions with AI *can* be started, but forum reports (GTPlanet AC mods thread) say AI on point-to-point tracks "treats the finish line as a
  stopping point" and veers off or brakes before it. The workaround reported there is an AI hint above 100% near the end.
  Race completion and results on A→B are unreliable (ASSUMED). **Verdict: don't depend on it; test it in spike S3.**
- **CSP New Modes (VERIFIED, `acc-lua-internal/included-new-modes`, CSP 0.2.11 zip):** a mode is a folder in `extension/lua/new-modes/<id>/`
  with `manifest.ini` (`[RULES] BASE_MODE=PRACTICE|HOTLAP|TRACK_DAY|RACE`, `AI_LIMIT`, `AI_LEVEL`, `START_TYPE=PIT|START`, `LAPS_COUNT`,
  `PENALTIES`, `ALLOWED_TRACKS=` names, `#tags` or `pits>1`, `ALLOW_PHYSICS_ALTERATIONS=1`; `[TWEAKS] IMMEDIATE_START`, `FINISH_DELAY`, `FIX_SPAWN_POINTS`, …),
  `mode.lua` (`script.simUpdate`, `script.drawUI`, `ac.addMainMenuSettings`, `ac.endSession(message, ok, {place=…})`) and `settings.ini`.
  CM writes `__CM_CUSTOM_MODE` into `race.ini` when launching one (referenced by the p2p-1v1 manifest).
  Relevant built-in examples:
  - **Rally Stage** (`BASE_MODE=HOTLAP`, `LAPS_COUNT=1`, `START_TYPE=START`, `ALLOWED_TRACKS=…, #rally, #hillclimb`) is a point-to-point
    stage mode with its own start and finish logic. Good template for A→B timing.
  - **Revenant** (`BASE_MODE=TRACK_DAY`, `AI_LIMIT=1`): one opponent car driven by a replay-based ghost with collisions (CSP 0.3.0-preview110+, 2025-10-17).
  - **Splitscreen** (`BASE_MODE=RACE`, `AI_LIMIT=1`): "might fit for pair drifting".
  - **P2P 1v1**: peer-to-peer online 1v1 with a friend, without a server.
  - **Overtake**, **Drift Challenge** (zones and scoring, by Neoned), **Traffic**.
- **Our Touge New Mode (proposed):** `BASE_MODE=RACE` (or `TRACK_DAY`), `AI_LIMIT=1`, `ALLOWED_TRACKS=#touge`, `ALLOW_PHYSICS_ALTERATIONS=1`.
  - Side-by-side start: the generator places `AC_START_0/1` side by side; a countdown, holding both with `physics.setAINoInput`/`lockUserControlsFor`, then release.
  - Rolling start: `physics.setCarPosition` + `physics.setCarVelocity` for both cars.
  - Initial D "cat and mouse": leader starts ~1 car length ahead. Rules from ACTougeBattle: the chaser wins by overtaking and holding the lead;
    the leader wins by a gap > 100 m; a draw goes to round 2 with roles swapped.
  - Finish: compare `car.splinePosition`/position with the finish gate from our track metadata, then `physics.setAIStopCounter`
    to brake the AI after B and `ac.endSession()` with results.
  - Optional oncoming traffic via `shared/sim/traffic`.
- **Fallback if A→B race handling is unworkable:** generate a loop variant (out-and-back with a turnaround) as a normal circuit layout.
- **Existing touge mods and scripts (online, PvP, not AI):**
  - `AssettoCN/ACTougeBattle` (https://github.com/AssettoCN/ACTougeBattle), an AssettoServer fork with `TougeBattlePlugin` (C#) + `lua/tougebattle.lua` UI.
    Leader/chaser, 2 rounds, gap win at 100 m, overtake win, jump-start detection, start positions and finish radius per track in YAML, ghosting of non-participants. VERIFIED (read source).
  - AssettoServer `RaceChallengePlugin` (in the main repo): SRP-style highway battles, where you flash lights to challenge; it ends when the leader is 750 m ahead or after 60 s without an overtake. VERIFIED (source).
  - AssettoServer Patreon `PatreonOvertakePlugin` (overtake scoring) (https://assettoserver.org/patreon-docs/plugins/PatreonOvertakePlugin/).
  - "Touge-Spotter" Lua app (audio warning for oncoming cars) and `n0tluc/assettoserver-lua-scripts` (overtake script), found via search; not read (ASSUMED as described).
  - I found **no existing offline "touge battle vs AI" New Mode**. That is a gap we can fill.

### B.5 AI driving style controls

**Vanilla files:**
- `race.ini` (written by CM): `[RACE] AI_LEVEL`, per-car `[CAR_n] AI_LEVEL`, `AI_AGGRESSION` (ASSUMED from memory; CSP docs confirm that
  "aggression level set in a launcher" exists and is multiplied by 0.95).
- Track `data/ai_hints.ini`: `[HINT_n] START=, END=, VALUE=` (fraction of spline; VALUE multiplies the corner radius the AI sees,
  so < 1 is slower and > 1 pushes harder); also `BRAKEHINT_n`, `DANGER_n` (community/forum; Stefano Casillo quote on GTPlanet; ASSUMED). We can generate these from our curvature analysis.
- Car `data/ai.ini`: `[GEARS] UP/DOWN/SLIP_THRESHOLD/GAS_CUTOFF_TIME`, `[PEDALS] GASGAIN/BRAKE_HINT/TRAIL_HINT`, `[LOOKAHEAD] BASE/GAS_BRAKE_LOOKAHEAD`,
  `[STEER] STEER_GAIN`, `[ULTRA_GRIP] VALUE` (forum example; section names confirmed indirectly because CSP docs reference
  `ai.ini` `LOOKAHEAD/BASE` (default 12), `LOOKAHEAD/GAS_BRAKE_LOOKAHEAD` (10) and `PEDALS/BRAKE_HINT` (1)). These are car files, so we shouldn't edit them (checksums/Kunos assets); override live via Lua instead.
- `fast_lane.ai` side widths (`sideLeft/sideRight`) bound how far the AI may use the road. For touge, the full road width.

**CSP config (VERIFIED, `new_behaviour.ini`):** `[AI_TWEAKS]` (FIX_LOW_LEVEL, CAUTIOUS_STEERING, ACT_IMPATIENT, TRAJECTORIES_FOR_DUMMIES,
STOP_TELEPORTING_STUCK …), `[AI_RACE_RUBBERBANDING]`, `[AI_RACE_ADAPTATION]`, `[AI_RACE_RETIREMENT]`, `[AI_FLOOD]`, `[CUSTOM_AI]`.
Track-side, alternative AI splines `ai/ext_alt_fast_lane_*.ai` are randomly assigned to race AIs (`lua-module/src/ai_tweaks.lua`).

**CSP Lua physics/AI API: real names (VERIFIED in CSP 0.2.11 generated SDK `ac_new_modes/lib.lua`; also present in
online-script, track-script, tools and apps SDKs).** All take a 0-based `carIndex`:

| Function | What it does |
|---|---|
| `physics.setAILevel(i, level)` | 0–2 (above 1 = faster than "100%") |
| `physics.setAIAggression(i, a)` | 0–1 (launcher value × 0.95) |
| `physics.setAICaution(i, c)` | 0–16, following distance (experimental) |
| `physics.setExtraAIGrip(i, v)` | AI gets **120% grip by default**; 1 = fair |
| `physics.setAITopSpeed(i, kmh)`, `physics.setAIThrottleLimit(i, 0..1)` | speed / gas caps |
| `physics.setAILookaheadBase/GasBrake(i, v)`, `physics.setAIBrakeHint(i, v)`, `physics.setAISteerMultiplier(i, m)` | live ai.ini overrides |
| `physics.setAISplineOffset(i, -1..1, override)`, `physics.setAISplineAbsoluteOffset(i, m, override)` | lateral line offset (block or pass) |
| `physics.setAISpline(i, file)` | per-car alternative spline (0.3.0-preview: + hints file arg) |
| `physics.setAIStopCounter(i, s)`, `physics.setAINoInput(i, active, stall)` | hold or park the AI |
| `physics.setAIBrakeMistake(i, t)`, `physics.setAITractionMistake(i, t, severity)` | scripted mistakes (Initial D drama) |
| `physics.setAICarPosition(i, pos, dir)`, `physics.setCarPosition(i,pos,dir)`, `physics.setCarVelocity(i, v)`, `physics.teleportCarTo(i, spawnSet)` | starts / resets |
| `physics.setAIShiftingThresholds(i, up, down)`, `physics.preventAIFromRetiring(i)`, `physics.setAIPitStopRequest`, `physics.setAITyres` | misc |
| `physics.overrideSteering(i, value)` | **full** steering override for any car (`math.nan` to release) |
| `ac.overrideCarControls(i)` → `ac.CarControlsInput` | "only controls for AIs and user cars can be overriden". `steer` replaces; `gas/brake/handbrake` = **max(original, ours)**; `clutch` = min; `requestedGearIndex`, gear up/down. Available in New Modes, apps and tools (not online/track scripts) |
| `physics.setGripDecrease(i, wheels, x)`, `physics.addForce(i, pos, local, force, local)`, `physics.addWheelTorque`, `physics.setWheelAngularVelocity` | physics cheats |
| `physics.startPhysicsWorker(script, data, cb)` | 333 Hz worker with limited physics API |
| `physics.setAITimeToStart(i, t)` | 0.3.0-preview (changelog) |

Read-back: `ac.getCar(i).aiLevel/aiAggression/isAIControlled/splinePosition/localVelocity/wheels[n].slipAngle/ndSlip/steer/gas/brake/handbrake`.
The physics API needs permission. New Modes can grant it themselves with `ALLOW_PHYSICS_ALTERATIONS=1` (comment in the Revenant/Splitscreen manifests:
"allows teleports and physics API to Lua apps, for modes without competitive aspect"). Track scripts, apps and tools need
`surfaces.ini` `[_SCRIPTING_PHYSICS] ALLOW_TRACK_SCRIPTS / ALLOW_APPS / ALLOW_TOOLS / ALLOW_NEW_MODE_SCRIPTS=1` on a track with
`extended-0` physics (CSP wiki "Tracks – General extended physics options"). Track scripts are declared as `[SCRIPT_...]` sections in the track's
CSP config (`ext_config.ini`) (SDK docs mention `[SCRIPT_...]`; exact keys ASSUMED).
Names in the brief that **do not exist**: `physics.setAIPedals`, `physics.setAISteer`. The real equivalents are listed above.

### B.6 Drifting AI

**Prior art (honest picture):**
- No public, working, general "drift AI" for AC that I could verify. Community attempts:
  - Tyre/`ai.ini` hacks (2018–2019 AssettoCorsaMods thread "Artificial intelligence drifter": tyres needing large slip angle,
    less front grip, more rear grip, higher slip in ai.ini). Also an overtake.gg "AI DRIFT Car Conversion" (2023, page blocked to me). Per search snippets,
    its author notes AC AI **never uses the handbrake** and **brakes instead of throttling at deep angles**. Both are car-data edits, ASSUMED quality.
  - "AI Whisperer" CSP Lua app (overtake.gg, recent) with a "drift sensitivity" for drifty or vintage cars, per search snippet. Couldn't read the page (403).
  - `paulizleet/Assetto-Corsa-Drifter` (TensorFlow, 1 commit, no results). `dasGringuen/assetto_corsa_gym` (NeurIPS 2024 RL racing benchmark;
    drives the **player** car via vJoy) proves that learning controllers in AC is practical, but it is grip racing, not drifting.
- Native AC AI is fundamentally a grip racer: speed from line curvature, a slip-limiting throttle (`SLIP_THRESHOLD`), no handbrake.
  Tuning knobs alone will give "loose" AI at best, not anime drifting.

**What it takes (control design, ASSUMED engineering):**
- A **drift line**: the racing line shifted to the outside on entry, with a late apex. Per corner, a target body slip angle β* (e.g. 20–35°) and target speed from radius.
- A **cascaded controller** at ≥ 60 Hz (better 100–333 Hz):
  - Initiation: lift, handbrake pulse or clutch kick (`clutch` override) plus a steering flick at the entry point.
  - Sustain: **throttle controls β** (more throttle, more rear slip; PID on β error), **steering = countersteer feedforward (≈ −β·k) + PD on
    path lateral error / yaw-rate error**, brake only for speed.
  - Exit: reduce β* to 0, straighten, grip mode.
  - This mirrors academic autonomous drifting (Stanford "MARTY"/Gerdes lab, which stabilises a drift equilibrium with steering + rear torque; citation ASSUMED from memory).
- Telemetry available: `ac.getCar(i).localVelocity` (gives β), angular velocity, `wheels[n].slipAngle`, `ndSlip`. The Custom AI interface adds 333 Hz `local_angular_velocity`, per-wheel `slip_angle_deg`, `slip_ratio`, `load`.
- Car choice matters: start with one RWD drift car with a locking diff; tune per car class.

**Is it possible for an AI car (not just the player)? (VERIFIED APIs, behaviour partly ASSUMED):**
- Steering: yes. `physics.overrideSteering(carIndex, v)` fully replaces steering for any car. `ac.overrideCarControls(carIndex)` explicitly supports AI cars.
- Pedals: partly. Override gas/brake/handbrake are combined with the AI's own using `max()`, so the AI's own gas must be suppressed:
  `physics.setAIThrottleLimit(i, 0)`, or `physics.setAINoInput(i, true)`. Whether overrides still apply in "parked" state is **unknown → spike S4**.
- Full control, guaranteed: the **CSP Custom AI interface** (since 0.1.78). An external process writes `cai_car_controls`
  (gas, brake, clutch, steer −1..1, handbrake, gears, teleport) to the memory-mapped file `AcTools.CSP.NewBehaviour.CustomAI.CarControls<N>.v0`.
  CSP then swaps that car's controller for it, reads at **333 Hz**, and publishes full car and wheel state in `...Car<N>.v0`. Other cars appear in `...CarPublic<N>.v0`.
  Sim control (`pause`, `restart_session`, `disable_collisions`, `time_scale` since 0.2.8) is available too, which is great for training and tuning.
  Requirements: `[CUSTOM_AI] ENABLED=1` in `new_behaviour.ini` (already `ENABLED=1` in CSP 0.2.11's shipped config) and the track's `surfaces.ini`:
  `extended-0` + `[_EXTRA_PERMISSIONS] ALLOW_CUSTOM_AI_MANIPULATION=1`. **We generate surfaces.ini, so we can grant this.**
  Example client: https://gist.github.com/gro-ove/11489f32b3eb3c9e3df1e7819bb3008e
- Car physics scripts (`data/script.lua` in a car with extended physics) can overwrite gas/brake/steer/handbrake at physics rate via
  `ac.accessCarPhysics()`, and run for AI instances too (ASSUMED). But that means editing car data, so it is not shippable for Kunos cars.
- "Fake" options: AssettoServer AI is kinematic, so a fork could add a yaw offset and steer angle in corners for a purely visual drift
  (online only, no physics). Offline, a Lua puppet via `physics.setCarPosition` each frame is janky. Low value.

**Feasibility rating:**

| Approach | Rating | Notes |
|---|---|---|
| Native AI + tuning (grip loss via `setGripDecrease` on rears, mistakes) | Easy to try, likely ugly | "Loose" AI, not controlled drifting |
| Replay ghost (Revenant, CSP 0.3.0-preview) | **Feasible now** | Perfect-looking drifts, not reactive; user must record a run; Patreon preview only |
| Lua drift controller in New Mode (S4 → S6) | **Hard** | Depends on pedal-suppression result; render-rate control loop; per-car tuning |
| External controller via Custom AI (S5 → S6), optionally RL-trained | **Research project** | Full control at 333 Hz plus time-scaling for training; needs a helper exe; most robust long term |

**Proposed staged approach for drifting:**
1. D0: Ship the grip AI touge (Stage 1). Add "style" presets via the AI API (aggression, extra grip off, mistakes).
2. D1: Optional "Rival ghost" using Revenant on CSP preview (record your own rival run on the generated road).
3. D2: Spike S4/S5. Pick Lua-in-New-Mode if pedals can be fully overridden; otherwise Custom AI with a small Python/C# helper bundled with our local tool.
4. D3: Skidpad controller (constant-radius β hold). Then a single hairpin. Then a whole generated road, using per-corner β*/speed tables our generator emits next to `fast_lane.ai` (e.g. `drift_line.json`).
5. D4 (optional): learn or tune controller gains automatically with Custom AI `time_scale` + `restart_session` (RL or Bayesian optimisation per car).

### B.7 What the user installs, per option

| Option | AC | CSP | Content Manager | Other |
|---|---|---|---|---|
| Grip AI touge (our New Mode) | AC (Steam) | public **0.2.11** or newer (New Modes + physics API present) | Full CM recommended; pick the mode in Quick Drive / CSP main menu (ASSUMED UI) | our mode folder → `assettocorsa/extension/lua/new-modes/<id>/`; generated track with `#touge` tag |
| AssettoServer traffic | AC | ≥ 0.1.77 (docs) / ≥ 0.2.0 (current master default `MinimumCSPVersion`) | CM → Drive → Online → LAN to join | AssettoServer win-x64 release (v0.0.54 stable or 0.0.55-pre), extracted outside the AC folder; traffic car models installed (Kunos cars are fine) |
| CSP Traffic mode | AC | 0.2.x (tool + mode ship in 0.2.11) | CM; select "Traffic" New Mode | traffic `data.zip` → `extension/lua/tools/csp-traffic-tool/data`; track with `extended-0` surfaces + tag (CSP-only track) |
| AI Flood | AC | any recent | CM → Settings → CSP → New AI behavior: AI Flood on, lower "Minimum track length" below route length; Track Day mode with N AI | none |
| Revenant ghost rival | AC | **0.3.0-preview110+** (Patreon) | CM | none |
| Drift AI via Custom AI | AC | ≥ 0.1.78 (0.2.8+ for time_scale) | CM; `new_behaviour.ini` `[CUSTOM_AI] ENABLED=1` (default in 0.2.11) | our helper exe running alongside AC; track `surfaces.ini` permission |

CSP versions (VERIFIED, acstuff.club/patch and csp-logs): public recommended **0.2.11**. Newest preview **0.3.0-preview675** (2026-10-07, Patreon).

---

## Open questions / risks
- Vanilla A→B race session behaviour with AI (finish-line braking, race end) needs hands-on testing (S3).
- `ac.overrideCarControls` pedal semantics with AI suppressed (S4).
- Extended-physics `surfaces.ini` makes the track **CSP-only** and changes online checksums. Offer "vanilla" and "CSP-extended" variants
  or a separate layout.
- CSP Traffic Planner is self-described experimental. Its JSON format could change. The binary cache and the online lookup suggest it is stabilising, but pin a CSP version in docs.
- AssettoServer AI is kinematic. Collisions with traffic feel like hitting a puppet that stops (by design). It is fine for SRP-style cruising.
- Pit box count drives traffic density online. Generate plenty of pits.
- x4fab requests coordination before building New Modes that embed traffic.

## Sources
- AssettoServer source: https://github.com/compujuckel/AssettoServer (files: `AssettoServer/Server/Configuration/Extra/AiParams.cs`, `Server/Ai/Splines/AiSplineLocator.cs`, `FastLaneParser.cs`, `AdjacentLaneDetector.cs`, `MutableAiSpline.cs`, `SplinePointOperations.cs`, `Server/Ai/AiState.cs`, `Server/Configuration/ACServerConfiguration.cs`, `FastLaneUtils/*`, `Plugins/RaceChallengePlugin/Race.cs`)
- AssettoServer docs: https://assettoserver.org/docs/intro, https://assettoserver.org/docs/thebeginnersguide, https://assettoserver.org/docs/faq, junctions guide (repo https://github.com/compujuckel/assettoserver-docs `docs/ai-splines/junctions_guide.md`)
- CSP traffic tool: https://github.com/ac-custom-shaders-patch/acc-lua-internal/tree/main/included-tools/traffic
- CSP new modes: https://github.com/ac-custom-shaders-patch/acc-lua-internal/tree/main/included-new-modes (traffic, rally, revenant, splitscreen, p2p-1v1, overtake)
- CSP shared traffic lib: https://github.com/ac-custom-shaders-patch/acc-lua-internal/blob/main/lua-shared/sim/traffic.lua
- CSP AI tweaks module: https://github.com/ac-custom-shaders-patch/acc-lua-internal/blob/main/lua-module/src/ai_tweaks.lua
- CSP Lua SDK: https://github.com/ac-custom-shaders-patch/acc-lua-sdk (generated definitions from CSP 0.2.11 zip, https://acstuff.club/patch/?get=0.2.11, `extension/internal/lua-sdk/ac_new_modes/lib.lua`)
- CSP wiki, Custom AI: https://github.com/ac-custom-shaders-patch/acc-extension-config/wiki/Other-Things-%E2%80%93-Custom-AI
- CSP wiki, extended physics options: https://github.com/ac-custom-shaders-patch/acc-extension-config/wiki/Tracks-%E2%80%93-General-extended-physics-options
- CSP wiki, car physics scripts: https://github.com/ac-custom-shaders-patch/acc-extension-config/wiki/Cars-%E2%80%93-Physics-scripts
- CSP changelogs: https://github.com/C1XTZ/csp-logs (0.3.0-preview110 Revenant; preview675 latest)
- ACTougeBattle: https://github.com/AssettoCN/ACTougeBattle (`PROJECT_CONTEXT.md`, `TougeBattlePlugin/TougeConfiguration.cs`)
- assetto_corsa_gym: https://github.com/dasGringuen/assetto_corsa_gym
- AI drifter thread (2018–19): https://assettocorsamods.net/threads/artificial-intelligence-drifter.1223/
- Forum (AI stops at finish on P2P; ai_hints/ai.ini knobs): https://www.gtplanet.net/forum/threads/assetto-corsa-pc-mods-general-discussion.307899/ (search snippets only, 403 on fetch)
- overtake.gg pages (AI Whisperer, AI DRIFT Car Conversion): search snippets only, 403 on fetch
