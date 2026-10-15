# AI line (`fast_lane.ai`) research: format, game usage, generation, and point-to-point

Researched 2026-10-14. Every claim is tagged:
- **VERIFIED**: read in source code or docs, with the URL (and commit where I cloned).
- **COMMUNITY**: a forum or modder statement I read directly. It's credible but not authoritative.
- **ASSUMED**: my inference or background knowledge, not confirmed. These go into the in-game spike.

Sources cloned and read:
- AssettoServer @ `1365f3e` (2026-10-06)
- gro-ove/actools (Content Manager) @ `e5e5c94` (2026-06-17)
- leBluem/io_import_accsv @ `8fa3a11`
- KinG-InFeT/blender-assetto-corsa-track-generator @ `419d3d7`
- ac-custom-shaders-patch/acc-extension-config @ `963a683`
- c1xtz/csp-logs (community CSP changelog archive)
- compujuckel/assettoserver-docs
- TUMFTM/trajectory_planning_helpers and TUMFTM/global_racetrajectory_optimization

GTPlanet and overtake.gg (RaceDepartment) returned HTTP 403 to the fetcher. Anything from those sites comes only through search-result snippets and is marked COMMUNITY or ASSUMED.

---

## Recommendation (read this first)

1. **Generate `ai/fast_lane.ai` fully in code. Don't record it in game.** The binary format is fully known from two independent open-source readers/writers (Content Manager/AcTools and AssettoServer), and they agree byte for byte (§1). Several tools already write `fast_lane.ai` from geometry and say it works in game: TreCorsa, RacetrackDesign, Lapwerks, KinG-InFeT's generator, leBluem's Blender add-on. The build worker therefore never needs the game.
2. **Fill every field of the "extra" (payload) block with physically sensible values.** Don't leave them at zero:
   - `side_left`/`side_right`: road edges minus a margin.
   - `radius`: from curvature.
   - `speed`, `gas`, `brake`: from our speed profile.
   - `camber`/`direction`, `normal`, `forward`, `grade`, segment `length`: from the road geometry.

   The game and CSP regenerate some of this at load ("Initialising AI spline", CSP "cache generated payloads", raycast "AI spline metadata"). But it isn't documented which fields they regenerate. Writing good values costs nothing and protects us either way. AssettoServer reads `radius` directly for traffic corner speeds, so `radius = 0` would stall traffic (§1.5).
3. **Write the trailing grid flag as `int32 0` (no grid).** With CSP's default `ALT_GRID=1`, CSP builds its own sparse lookup grid, so the original grid isn't needed (VERIFIED from the CSP config text). Vanilla (no CSP) behaviour with no grid is the #1 spike item. Fallback: generate the grid ourselves; its structure is known from AcTools (§1.4).
4. **v1 racing line:** smoothed centreline → minimum-curvature lateral offset within (road half-width − margin), solved as a bounded least-squares/QP with 2–3 re-linearisations. Then a forward–backward speed pass using a friction circle with grade and banking terms (§5). Resample at about 1 m (calibrate against Kunos files in the spike). Use a generic grip μ ≈ 1.0 (street car) and cap the top speed.
5. **Also ship `data/ideal_line.ai` as a copy of `fast_lane.ai`.** COMMUNITY: it's "the preferred ideal line shown in-game". Skip `pit_lane.ai` in v1 unless the spike shows a session type needs it. If one is needed, generate a short spur from `AC_PIT_0` into the fast lane (§2.4).
6. **Point-to-point is the real risk, not the file format.** Vanilla AC AI is built around laps. COMMUNITY (Mount Akina 2017 author): AC's AI "does not support point to point tracks as they hit the finish line and just divide by 0 and drive into walls". Plan:
   - (a) The A→B track works for practice, hotlap and ideal line.
   - (b) For AI races, either ship an extra **loop layout**, or build a **CSP Lua "touge" mode/track script** that starts both cars at A, decides the winner at B, and teleports or stops cars. CSP exposes `physics.setAICarPosition`, `physics.setAINoInput`, `physics.setAISpline` (VERIFIED in CSP code and changelogs).
   - (c) For AssettoServer traffic, use two opposite-direction lane splines whose ends are joined with `ConnectEnd` in `config.yml` (VERIFIED in the docs), so traffic U-turns at A and B instead of despawning.
7. **Fallback if the generated line misbehaves:** load the generated line in CSP's **AI Spline Editor** (App Shelf, CSP ≥ 0.2.3) to fix sides and trajectory, or re-record with the AI dev app and the CSP recorder. Steps are in §3.

---

## Spike: what to test in game (in priority order, about half a day on the user's PC)

Tooling to write first (about 1 hour):
- `ai_dump.py`: dumps any `fast_lane.ai` to CSV (all 5 + 18 fields) and prints stats: point spacing, speed range, radius range, side widths, camber/direction/grade ranges, tag values, grid flag/size.
- `ai_write.py`: our writer.

| # | Test | Pass criterion |
|---|------|----------------|
| S0 | **Calibrate units.** Run `ai_dump.py` on 3–4 Kunos tracks (e.g. `magione`, `drift`, `ks_nordschleife` (long), `ks_highlands`). Record: typical point spacing; `speed` units (m/s vs km/h: compare max with the known top speed); `radius` sign and value on straights; `camber` units (rad?); `direction` (±1: which way?); `grade` (rad or ratio?); `normal`; extra `length` (segment length vs cumulative?); `tag` values; whether `lap_time`/`sample_count` are non-zero; grid present and its size. | A filled-in table of field semantics. This replaces every ASSUMED in §1.3. |
| S1 | **Round-trip.** Read a Kunos `fast_lane.ai` and rewrite it with our writer, byte-identical (including grid passthrough). | `cmp` equal |
| S2 | **Generated file, grid flag 0, CSP on (default).** Our generated line on our generated track. Practice, 1 AI car, ideal line on. | Loads; ideal line drawn along the road; AI drives the whole route without wall hits |
| S3 | Same as S2 with **CSP disabled** (vanilla). | Loads without crash. If it crashes or the AI is stuck, generate the grid (§1.4) and retest |
| S4 | **Zeroed payload.** Same file with sides/radius/normal/etc. = 0 (positions only). | Learns whether the game recomputes sides/normals itself (compare via CSP AI Spline Editor or `ai_dump` of any re-saved file) |
| S5 | **Sides.** Generated sides = road edge − 0.5 m vs. sides = 0 with surfaces `IS_VALID_TRACK=1` only on `1ROAD`. View in CSP AI Spline Editor / CM map preview. | Sides follow the road edge |
| S6 | **Point-to-point sessions** (track with `AC_AB_START_L/R`, `AC_AB_FINISH_L/R`): Practice, Hotlap, Time Attack, Drift, Quick Race (1 AI), Track Day (AI). For each, note what happens at B for the player and for the AI (stop / teleport / drive into wall / session end / crash). Once with CSP, once without. | Matrix of what works → decides the loop layout vs. Lua touge mode question |
| S7 | Same as S6 with **no `pit_lane.ai`** vs a generated short pit spur. | Learns whether `pit_lane.ai` is needed |
| S8 | **AI quality.** On a twisty 3–5 km route, 3 AI levels (80/90/100%) in 2 cars (street car, sports car). Note off-track moments. Try `ai_hints.ini` `[DANGER_n]` and `[HINT_n]` on a bad corner. | AI completes the route at least 3/5 times without leaving the road at 90% |
| S9 | **Speed field effect.** Same track, payload speed ×0.7 vs ×1.3. Does AI pace or the ideal-line colour change? | Tells whether AC AI uses our speed or computes its own |
| S10 | **AssettoServer.** Two opposite lanes from the same generator with `config.yml` `ConnectEnd` U-turns; `TwoWayTraffic: true`, `LaneWidthMeters` = our lane offset. | Traffic flows both ways and turns at the ends |
| S11 | Long route (> 20 km) load time with CSP payload cache on/off. CSP skips the ideal line above 54 km (VERIFIED, §2.2). | Load time acceptable |

---

## 1. `fast_lane.ai` binary format (field level)

### 1.1 Sources that agree

- **AcTools (Content Manager)**: `AcTools/AiFile/AiSpline.cs`, `AiPoint.cs`, `AiPointExtra.cs`, `AiSplineGrid.cs`.
  https://github.com/gro-ove/actools/tree/master/AcTools/AiFile. **VERIFIED**. Reader only, v7 only (`if (Version != 7) throw`).
- **AssettoServer**: `AssettoServer/Server/Ai/Splines/FastLaneParser.cs` (reader).
  https://github.com/compujuckel/AssettoServer/blob/main/AssettoServer/Server/Ai/Splines/FastLaneParser.cs. **VERIFIED**.
- **AssettoServer FastLaneUtils**: `FastLaneUtils/FastLane.cs`, a full **reader + writer** for v7 and AssettoServer's own v-1. It can also cut, reverse and shift height. **VERIFIED**. This is the best reference writer to port.
- **leBluem io_import_accsv**: `import_ai.py`/`export_ai.py`, Python `struct` reader/writer.
  https://github.com/leBluem/io_import_accsv. **VERIFIED**. Its 18-float view is shifted by one int; see the note in §1.3.

All values are little-endian. `float` = IEEE float32, `int` = int32.

### 1.2 Header + main points

From AcTools `AiSpline.cs`:

```csharp
Version = reader.ReadInt32();                  // 7
if (Version != 7) throw new Exception($"Version {Version} not supported");
var points = new AiPoint[reader.ReadInt32()];   // N
LapTime = reader.ReadInt32();                   // "Extra values, usually zeroes"
SampleCount = reader.ReadInt32();
for (...) points[i].LoadFrom(reader);           // N x AiPoint
var pointsExtra = new AiPointExtra[reader.ReadInt32()];  // N again
for (...) pointsExtra[i].LoadFrom(reader);      // N x AiPointExtra
HasGrid = reader.ReadInt32() != 0;
if (HasGrid) Grid = new AiSplineGrid(reader);
```

```csharp
[StructLayout(LayoutKind.Sequential, Pack = 4)]
public struct AiPoint {          // 20 bytes
    public Vec3 Position;        // x, y, z (AC world, Y up)
    public float Length;         // cumulative distance from point 0, metres
    public int Id;               // point index 0..N-1
}
```

| Offset | Type | Field | Notes |
|---|---|---|---|
| 0 | int32 | version | `7` for AC. AssettoServer also accepts `-1`, its own compact format, which AC can't read (VERIFIED, FastLaneUtils `--optimize` "only works in AssettoServer"). |
| 4 | int32 | N (detail count) | |
| 8 | int32 | lap_time | AcTools comment: "Extra values, usually zeroes". Write 0. |
| 12 | int32 | sample_count | Same; write 0. |
| 16 | N × {f32 x, f32 y, f32 z, f32 length, i32 id} | points | `length` is cumulative distance: KinG writes `cum_dist` (VERIFIED, `generate_ai_line.py`) and leBluem v2.0 changelog says "fixed increasing distance not being saved in ai line". `id` = index. |

### 1.3 Extra ("payload"/detail) section: N × 18 floats = 72 bytes each

```csharp
public struct AiPointExtra {   // AcTools, identical order in AssettoServer FastLaneUtils
    public float Speed;
    public float Gas;
    public float Brake;
    public float ObsoleteLatG;
    public float Radius;
    public float SideLeft;
    public float SideRight;
    public float Camber;
    public float Direction;
    public Vec3  Normal;
    public float Length;
    public Vec3  ForwardVector;
    public float Tag;
    public float Grade;
    public float Width => SideLeft + SideRight;
}
```

Preceded by `int32 extraCount`, which must equal N. AssettoServer throws "Count of spline points does not match extra spline points" otherwise (VERIFIED).

| # | Field | Meaning, and what we write | Status |
|---|---|---|---|
| 0 | speed | Recorded speed at the point. Write our profile speed. Units **unknown**: KinG writes m/s (`speeds/3.6`), but that writer is itself suspect (§4.1). Calibrate in S0. | name VERIFIED; units ASSUMED |
| 1 | gas | 0..1 throttle. Write 1 when accelerating or at the speed cap, else partial. | ASSUMED semantics |
| 2 | brake | 0..1. Write decel/(μg) in braking zones. | ASSUMED |
| 3 | obsolete_lat_g | Name says obsolete. Write 0 (or v²κ/g). | name VERIFIED |
| 4 | radius | Corner radius in **metres**. AssettoServer computes `v_max² = g·μ·radius` directly from it (`PhysicsUtils.CalculateMaxCorneringSpeedSquared`). It resets the last point to 1000 if < 1 ("For point-to-point splines the last point might be completely off"). Write `min(1/|κ|, 1000)` (or a larger cap for straights; check S0). | VERIFIED (AssettoServer usage) |
| 5 | side_left | Distance in metres from the line to the left edge. | VERIFIED (CM `AiLaneObject` draws `v − s·SideLeft`, `s = normalize(cross(dir, UnitY))`) |
| 6 | side_right | Distance to the right edge (`v + s·SideRight`). `cross(forward, up)` = the driver's right in AC's right-handed Y-up frame. | VERIFIED |
| 7 | camber | Road bank/cross-slope magnitude. AssettoServer uses `camber * direction` as the car's roll. Probably radians (ASSUMED; check S0). AssettoServer smooths it with a moving average of 5 by default (`SmoothCamber`). | VERIFIED usage |
| 8 | direction | "either 1 or -1" (AssettoServer comment). The sign of camber/turn. FastLaneUtils `Reverse()` flips `Direction` and `Forward`. | VERIFIED |
| 9–11 | normal (x,y,z) | Road surface normal under the point. CSP "AI spline metadata raycasting" suggests the game recomputes it by raycasting the physics mesh. Write the road-surface normal. | ASSUMED meaning |
| 12 | length | Named `Length` (AcTools) / `DetailLength` (FastLaneUtils). Probably the segment length to the next point. Write segment length; check S0. | ASSUMED |
| 13–15 | forward (x,y,z) | Unit tangent. Flipped by `Reverse()`. | VERIFIED via Reverse() |
| 16 | tag | Unknown. Write 0. | ASSUMED |
| 17 | grade | Slope. Units unknown (rad or rise/run); check S0. | ASSUMED |

**leBluem note (VERIFIED by reading the code):** `export_ai.py`/`import_ai.py` read the 18-float records straight after the points **without** consuming `extraCount`. Their index *k* is AcTools field *k−1*: 1=speed, 2=gas, 3=brake, 4=latG, 5=radius, 6=border_left, 7=border_right, 8=camber, 9=direction, 10–12=normal, 13=length, 14–16=forward, 17=tag. That's why they write `1.40129846432481e-45 * detailCount` as "float 0": it is the int `N` reinterpreted as a float. The two layouts agree.

### 1.4 Optional lookup grid (after the extras)

`int32 hasGrid`. If non-zero, AcTools `AiSplineGrid.cs`:

```csharp
public Vec3 MaxExtreme; public Vec3 MinExtreme;
public int NeighborsConsideredNumber; public float SamplingDensity;
Items = new Item[reader.ReadInt32()];       // cells
//   Item:    int subCount; subCount x ItemSub
//   ItemSub: int count;    count x int (point indices)
```

- It's a spatial lookup grid (cells of `SamplingDensity` metres over the XZ bounds, listing nearby spline point indices). The cell ordering is unknown; decode a Kunos grid in S0 if we ever need to write one. **ASSUMED** semantics.
- **CSP replaces it.** `new_behaviour.ini [AI_SPLINES] ALT_GRID=1 ; Use sparse grid (replaces original AC AI grid. Improves precision, performance, loading time and memory consumption. Original grid is still stored when recording AI lanes for now, for compatibility.)` — **VERIFIED** (acc-extension-config `config/new_behaviour.ini`). The CSP 0.1.80p218 changelog: "New alternative implementation of AI spline lookup grid: more accurate with 1 m resolution…" (VERIFIED, csp-logs).
- COMMUNITY: the game/CSP write `fast_lane_payloads.bin`, `pit_lane_payloads.bin` and `pit_lane_with_grid.ai` caches into `ai/`. Deleting them makes them regenerate on next load (Steam thread https://steamcommunity.com/app/244210/discussions/0/3123787625596058139/). So something computes a grid and payload when they're missing. Whether **vanilla** AC does so for a file with `hasGrid=0` is spike S3.

Writer layout (Python, a direct port of FastLaneUtils `ToFileV7`, plus the grid flag that AcTools expects):

```python
import struct
def write_fast_lane(path, pts, ext, lap_time=0, sample_count=0):
    # pts: list of (x, y, z, cum_len); ext: list of 18-tuples in AcTools order
    n = len(pts)
    with open(path, "wb") as f:
        f.write(struct.pack("<4i", 7, n, lap_time, sample_count))
        for i, (x, y, z, L) in enumerate(pts):
            f.write(struct.pack("<4fi", x, y, z, L, i))
        f.write(struct.pack("<i", n))
        for e in ext:
            f.write(struct.pack("<18f", *e))
        f.write(struct.pack("<i", 0))      # hasGrid = 0 (spike S3)
```

### 1.5 Versions and variants

- **v7** is the only AC format (AcTools rejects others). **VERIFIED**.
- **v-1** is AssettoServer-only: N × {pos, radius, camber·direction}. **VERIFIED**. Never ship it to clients.
- **`fast_lane.aip`** is an AssettoServer zip (`config.yml` + `*.ai`) with junction config. **VERIFIED** (`FastLaneParser.FromFiles`, docs junctions_guide).
- AssettoServer's binary cache (`AiSplineHeader {Version=1, NumPoints, NumJunctions, NumKdTreePoints}`) is server-internal. **VERIFIED**.
- What **AssettoServer uses** from a v7 file: position, radius, camber×direction. Length is recomputed from positions. Speed, gas, brake, sides, normal, forward, tag and grade are all discarded (`FastLaneParser.FromFileV7`). The loop is closed if start and end are < 50 m apart; otherwise it's open (`PreviousId/NextId = -1`). **VERIFIED**.

### 1.6 Coordinates

AC is **Y-up, right-handed**. Both Blender exporters map Blender (x, y, z) → AC (x, z, −y): KinG `ac_pts[:,1]=bl[:,2]; ac_pts[:,2]=-bl[:,1]`, leBluem `pack(x, z, -y)`. **VERIFIED**. From a local ENU frame (E, N, U), use **AC = (E, U, −N)**. CM's side rendering (`cross(forward, Y)` = right) is consistent with this. **VERIFIED**.

### 1.7 Other open-source code found

- **KinG-InFeT/blender-assetto-corsa-track-generator** (Apache-2.0): `scripts/generate_ai_line.py` generates `fast_lane.ai` from a centreline: Catmull-Rom, resampled at 2 m, speed interpolated from normalised curvature between `default_speed` 80 and `min_corner_speed` 35 km/h. **VERIFIED code.** But it writes the extras as **four separate count-prefixed arrays** (speed, gas, brake, lateral), not N×18 interleaved floats, and writes no grid flag. That doesn't match AcTools/AssettoServer, so AC would mis-parse it. Whether AC tolerates that, by detecting a short file and recomputing, is unknown. **Don't copy that part.**
- **leBluem/io_import_accsv** (Blender): import/export of `.ai` and `side_l/side_r.csv`. Changelog v2.0: "added AI line export into new/empty ai-line, afterwards you MUST either load it into "ksEditor" and save from there again or use shift on AC start with "side_l/r.csv" in place". **VERIFIED** (README). Its "new file" writes only positions + sides (everything else 0), which strongly suggests ksEditor or the game fills in the rest when re-saving or baking (S4).
- **TreCorsa** (web editor, closed source): "Copy Centerline → Auto-Optimize", exports `fast_lane.ai` directly. Traffic tab exports `fast_lane.ai` + `fast_lane_2.ai` at 25%/75% road width for AssettoServer. **VERIFIED** (docs page https://trecorsa.com/docs/?cat=ailines).
- **RacetrackDesign**: "racing line computed from a physics-based lap simulation". **Lapwerks**: "AI line included". Both closed source; this is evidence that generated lines work. **VERIFIED** (marketing pages).
- Content Manager can render `map.png` from the AI lane (`TrackMapRendererWrapper.Run(track, aiLane: true)` uses `AiSpline.FromFile` + `pit_lane.ai`). **VERIFIED**. Useful as a quick validity check that CM parses our file.

---

## 2. How AC uses it

### 2.1 AI driver

- The AI follows the spline and uses `side_left/right` as the usable track width for overtaking and avoidance. CM draws exactly that band; CSP's AI Spline Editor lets you drag "trajectory and sides" (VERIFIED, CSP 0.2.3 changelog). How exactly the AI uses each field is **not documented**.
- CSP config text supports that the game *generates* parts of the payload at load:
  - `CACHE_PAYLOADS=1 ; Cache generated payloads (might improve loading speed)`
  - `SAFE_NORMALS_RAYCASTING=0 ; Use safer approach for AI spline preparation (might help prevent crashes on some tracks where there is no physical surface below an AI spline)`
  - CSP changelog 0.2.2: "AI payloads cache refreshes if KN5s have been changed"

  So the payload depends on the physics mesh, i.e. it is raycast-derived (normals/camber/sides?). **VERIFIED text, ASSUMED interpretation.**
- AssettoServer's client FAQ: clients "stuck on 'Initialising AI spline' during loading" are processing the local spline. **VERIFIED** (assettoserver-docs `faq.md`).
- **ASSUMED (widely believed):** AC AI computes its own per-car target speeds from line curvature, car grip/aero (`ai.ini`) and `ai_hints.ini`, rather than replaying recorded `speed/gas/brake`. COMMUNITY support: Esotic, "yanking the wheel … will negatively impact how the AI adapts to different cars" (only a search snippet). Test this in S9.
- Sides, when not provided: "Without border CSVs, AC generates borders from surface data, using surfaces flagged `IS_VALID_TRACK=1`". Recording "merges both, the ai-line and the borders into fast_lane.ai". **COMMUNITY** (fughettaboutit, https://assettocorsamods.net/threads/ai-problem.1761/page-2). Our track must therefore mark only road (and maybe shoulder) surfaces `IS_VALID_TRACK=1`.
- Side CSVs: `data/side_l.csv` and `data/side_r.csv` are baked into an existing `fast_lane.ai` by **holding SHIFT on the loading screen**, or automatically if they're newer than the `.ai`. **COMMUNITY** (same thread + leBluem README).

### 2.2 Ideal line assist

- COMMUNITY (luchian guide, https://assettocorsamods.net/threads/how-can-i-make-ai-path.234/): also copy the recorded line to **`data/ideal_line.ai`**, "the preferred ideal line shown in-game and used for AI fuel calculations". Without it the game falls back to `ai/fast_lane.ai` (**ASSUMED**; CSP calls it "Ideal line… for AI splines", see below).
- CSP 0.2.7/0.2.8: "Ideal line: reworked generation… generation process is skipped entirely for AI splines longer than **54 km** (limit from original AC)". **VERIFIED** (csp-logs). Our routes stay under 54 km, or the ideal line disappears.
- The colours (braking zones) presumably come from the speed profile versus current speed (**ASSUMED**). S9 checks whether they react to our `speed` field.
- CSP `[RAIN_FX] IDEAL_LINE_RAIN_FACTOR` (rubber/wet line follows the ideal line). **VERIFIED** (0.3.0p619).

### 2.3 Point-to-point specifics in the file

- AssettoServer treats a spline as a closed loop only if its ends are within 50 m. **VERIFIED**. AcTools `CalculateLength` skips the closing segment if it's > 20 m, and CM's renderer skips segments > 10 m (`LengthSquared() > 100`). **VERIFIED**. So an open A→B spline is a normal, supported shape in the tooling.
- leBluem's importer has a "don't connect first/last verts" option for A→B. AssettoServer docs: "Point-to-point splines: in Blender import with 'dont connect first/last verts'". **VERIFIED**.
- CSP 0.2.4/0.2.5: "Automatically flip AI spline on up/downhill tracks where creator forgot to do so" (`FIX_REVERSED=1`). **VERIFIED**. A downhill layout must have the spline running in the driving direction.

### 2.4 `pit_lane.ai`

- Same format, recorded with the AI app's "start pits" button, starting before pit entry and ending after pit exit back on the racing line. It "should be smaller than fast_lane.ai". **COMMUNITY**.
- Needed? Steam user: "a pit lane AI line is also needed for practice and qualifying" (**COMMUNITY**, Steam thread above). The luchian guide calls it optional. CSP `[AI_RACE_RETIREMENT] TELEPORT_WITH_DAMAGE` teleports AIs to pits. **Undecided → S7.** A generated pit spur is easy: a spline from `AC_PIT_0` merging into the fast lane after 100–200 m.

### 2.5 `data/ai_hints.ini`

There's no official spec. **COMMUNITY** (assettocorsamods "Build your first track" p.95, GTPlanet snippets):

```ini
[HINT_0]          ; speed multiplier over a stretch
START=0.08        ; normalised spline position 0..1 (0 = spline start, 1 = end/lap)
END=0.10
VALUE=0.595       ; 1 = normal, 0.9 = back off a little, 0.5 = much slower; >1.1 risky

[DANGER_0]        ; keep away from edges
START=0.001
END=0.99
LEFT=0.1          ; poster's reading: stay ~10% away from left edge
RIGHT=0.1
```

- `MAXSPEED` and `BRAKEHINT` sections are also reported. One user said MAXSPEED "is the only thing they obey". Their keys are unverified.
- Sections should be numbered sequentially (`HINT_0`, `HINT_1`, …).
- CSP 0.3.0p110: "`physics.setAISpline()` takes an argument for AI hints config file" (VERIFIED), so per-car spline + hints are possible from Lua.
- **Generator plan:** emit `ai_hints.ini` automatically with `[HINT_n]` VALUE<1 on our sharpest hairpins only if S8 shows problems. Keep it empty in v1.

### 2.6 CSP extras relevant to us (all VERIFIED in csp-logs / config)

- `ext_alt_fast_lane_?.ai` in `ai/`: alternative splines that CSP assigns to random AI cars (CSP 0.2.3/0.2.4).
- `TRAJECTORIES_FOR_DUMMIES=1`: simpler splines for AI below 90%. `ALT_DRAG_SPLINE`: parallel spline for AI starting off-line. Rain "AI splines" are auto-generated (skipped above 50 km / 100k points).
- **Track Day "AI flood"** (endless traffic offline): `[AI_FLOOD] MIN_TRACK_LENGTH=7000` by default, i.e. only on tracks ≥ 7 km unless the user changes it. **VERIFIED**. This matters for "traffic on the road" offline.
- Lua: `physics.setAICarPosition`, `physics.setAINoInput` (used in CSP's driver-interview mode, VERIFIED in `lua/new-modes/driver-interview/mode.lua`). `physics.setAISpline()`, `physics.setAISplineAbsoluteOffset()`, Lua control of the AI spline recorder (VERIFIED in the changelog). `ac.hasTrackSpline()`, `ac.worldCoordinateToTrackProgress()` (VERIFIED in AssettoServer FastTravel plugin Lua).

---

## 3. Recording in game (documented fallback)

### 3.1 Vanilla AI dev app (COMMUNITY, luchian guide + several threads)

1. Enable developer apps. Either:
   - In CM: **Settings → Assetto Corsa → System → "Developer apps"** (the checkbox writes `[AC_APPS] ENABLE_DEV_APPS=1` to `assetto_corsa.ini`; VERIFIED in CM `SystemSettings.cs` / `AcSettingsSystem.xaml`), or
   - Edit `Documents\Assetto Corsa\cfg\assetto_corsa.ini` (the guide says `system/cfg`) and set `[AC_APPS] ENABLE_DEV_APPS=1`.
2. Before recording, consider turning off **CSP → New AI behavior** (or at least its `CACHE_PAYLOADS`). Several users report lines not saving or being corrupted with it on. Also delete any `*_payloads.bin` / `*_with_grid.ai` in the track's `ai/` folder (COMMUNITY).
3. Make sure `surfaces.ini` has `IS_VALID_TRACK=1` only for the road. Otherwise the auto-borders spill onto grass (COMMUNITY).
4. CM → Drive → Quick drive: our track, a stable car, mode **Practice** (or Hotlap), session long enough.
5. In game, open the app sidebar → **AI** app → **Start recording** (for the pit lane: **Start pits**).
6. Drive **smoothly**, not at the limit. Jerky steering or brake taps get copied into the AI's behaviour. On a loop, drive ≥ 2 full laps; the first completed lap can't be used.
7. Exit (ESC → Exit) or Stop → **Save AI**. The game may freeze while it processes. You get `ai/fast_lane.ai.candidate` (or `pit_lane.ai.candidate`).
8. Rename to `fast_lane.ai`; optionally copy to `data/ideal_line.ai`.
9. Optional: put `data/side_l.csv` and `data/side_r.csv` (from Esotic's **AI Line Helper** app: "hit record left or right… your front tyre will record the limit") and **hold SHIFT while the track loads** to bake them in.
10. Optional cleanup in **ksEditor**: load the track, load `fast_lane.ai`, "resample with 12", save.

**Time:** real-time driving. A 5 km road at about 70 km/h average is about 4–5 min per pass, plus 1–3 min of processing freeze and set-up. About 15 min per track, more if retakes are needed.

### 3.2 Point-to-point recording

- The vanilla recorder updates the line "each time you complete a lap". On an A→B track there's no lap. COMMUNITY threads leave "where to stop recording on a point-to-point track" unanswered.
- **CSP 0.1.77+**: "Control state of AI spline recorder: start and stop recording (allows to create AI splines without crossing finishing line)" (Lua API, VERIFIED). AssettoServer docs: "Point-to-point splines should be recorded with the new AI spline recorder (CSP ≥ 0.1.77), from the pinned message in #ai-spline-discussion" on the AssettoServer Discord (VERIFIED, junctions_guide). It's a Discord-distributed Lua app, not in CSP itself.
- Step-by-step for a player (CM + CSP): install CSP ≥ 0.1.77, get the recorder app from the AssettoServer Discord pin, drop it into `assettocorsa/apps/lua/`, enable dev apps, Practice session at A, press record in the app, drive A→B, press stop, then rename the candidate as above. **ASSUMED UI details** (app not inspected).

### 3.3 Editing instead of recording

- **CSP AI Spline Editor** (App Shelf, CSP ≥ 0.2.3): "load an AI spline and tweak its trajectory and sides by dragging them around". **VERIFIED** (changelog).
- **CSP Traffic Planner** (CSP ≥ 0.2.3): "Option to export a line as a new AI spline", "copy a lane with an offset, specify vertical offset when exporting splines". **VERIFIED**.
- **JumpAILine** (CSP Lua app) is used together with the AI Spline Editor by modders. **COMMUNITY**.

This is the best manual fix-up path: generate, then nudge bad corners in the AI Spline Editor.

---

## 4. Generating from geometry

### 4.1 Prior art

- Yes, people do it (§1.7): TreCorsa, RacetrackDesign and Lapwerks (closed), KinG-InFeT (open, but a wrong extras layout), leBluem (open, positions + sides, then "must resave in ksEditor or SHIFT-bake").
- I found **no** open-source writer that fills all 18 extra fields "properly". AssettoServer FastLaneUtils is a correct *serializer* (it round-trips existing data).
- No "kn5 → ai" auto-generator was found in open source.

### 4.2 Mandatory vs approximate (my assessment)

| Field | Required? | Source |
|---|---|---|
| version=7, N, extraCount=N, layout | **Mandatory**. Readers throw on mismatch. | VERIFIED |
| positions | **Mandatory**, about 0.5 m above the road surface (y = road height + 0.5, `AI_HEIGHT_OFFSET_M`), with road under every point. OBSERVED: the game-recorded sepang `fast_lane.ai` and `pit_lane.ai` sit 0.46–0.49 m (median 0.47) above that track's road meshes. CSP's Traffic Planner exports splines with a default vertical offset of 0.5 m (`EditorTabLanes.lua`), and CSP's AI Spline Editor snaps at 0.3 m by default. CSP raycasts down for metadata, and `SAFE_NORMALS_RAYCASTING` exists because "no physical surface below" crashes. A line exactly on the road mesh is coplanar with it, so don't put it there. | OBSERVED + VERIFIED + ASSUMED |
| cumulative length, id | Write them properly (trivial). | VERIFIED layout |
| sides | **Write them.** The AI needs width. The game may derive them from `IS_VALID_TRACK` surfaces only when recording/baking. | COMMUNITY |
| radius | **Write it.** AssettoServer traffic speed = sqrt(g·μ·R); 0 → stop. | VERIFIED |
| camber·direction | Write it (AssettoServer car roll; AC probably recomputes). | VERIFIED usage |
| normal, forward, grade, extra length | Write geometric values (cheap). Possibly recomputed. | ASSUMED |
| speed, gas, brake | Write our profile. Possibly only used for the ideal line / not at all by the AI (S9). | ASSUMED |
| lap_time, sample_count, tag, obsolete_lat_g | 0. | VERIFIED "usually zeroes" for the header |
| grid | `hasGrid=0`. CSP builds its own; vanilla → S3. | VERIFIED (CSP), ASSUMED (vanilla) |

### 4.3 Is a crude centreline + curvature speed line good enough?

- **For the ideal-line assist:** probably yes. It mostly has to sit on the road and show braking zones.
- **For AI racing:** the AI steers to the spline with look-ahead. A centreline is drivable, just slower: it uses half the road on curves and turns in late. The minimum-curvature line within the road width is a cheap, large improvement. Keep a safety margin (≥ 0.5 m + half car width) because real roads have no run-off.
- **ASSUMED** from general AC modding experience. The AI's notorious failures (wall hits on hairpins) are handled with `ai_hints.ini` and CSP `CAUTIOUS_STEERING=1` (which mentions Trento-Bondone hairpins, VERIFIED text).
- **For AssettoServer traffic:** use lane centrelines (not a racing line) at a constant offset equal to `LaneWidthMeters` (adjacent-lane detection looks exactly `laneWidth` sideways with a 2 m tolerance, VERIFIED in `AdjacentLaneDetector.cs`). Radius comes from the lane curvature.

---

## 5. Algorithms: racing line + speed profile

### 5.1 Inputs we already have

- A smoothed 3D centreline c_i, resampled at Δs (target about 1 m; check S0 spacing).
- Unit tangents t_i, horizontal left normals n_i.
- Half-widths w_L,i and w_R,i (road edge, from the OSM width model).
- Cross-slope/bank φ_i and grade γ_i from our road builder.

### 5.2 Minimum-curvature line (v1)

Lateral offset α_i along n_i, with p_i = c_i + α_i n_i (2D, XZ plane).

1. Discrete curvature proxy: second difference d2_i = p_{i−1} − 2p_i + p_{i+1}. It's linear in α.
2. Solve `min_α Σ ||d2_i||² + λ Σ (α_{i+1} − α_i)²` subject to `−(w_R,i − m) ≤ α_i ≤ (w_L,i − m)`, with margin m = half car width (≈ 0.95 m) + 0.3–0.5 m.

   This is a **sparse bounded linear least-squares** problem:
   - `scipy.optimize.lsq_linear(A, b, bounds=(lb, ub))` with sparse A (pentadiagonal); no extra dependency. Or
   - OSQP/quadprog QP.

   For open (A→B) roads, fix α at both ends to 0 (or to the start-grid offset). For loops, use wrap-around differences.
3. Re-linearise. Recompute the normals of the new line? No: keep the reference normals and re-solve 2–3 times (the TUMFTM approach iterates as well).
4. Optional blend: α = 0.7·α_mincurv on public roads (less aggressive cutting), or full on wide roads.
5. Smooth the result, resample at uniform Δs, recompute the 3D height from the road surface (not the centreline height), then add the 0.5 m spline height (§4.2). On a banked or crowned road, lateral offset changes y.

**Reference implementation:** TUMFTM `trajectory_planning_helpers.opt_min_curv(reftrack[x,y,w_tr_right,w_tr_left], normvectors, A, kappa_bound, w_veh, closed=True|False, psi_s, psi_e, fix_s, fix_e)`. It's a QP via `quadprog` that "can be used for closed and unclosed tracks… for unclosed tracks the heading psi_s and psi_e is enforced". **VERIFIED** (docstring). LGPL-3.0, `pip install trajectory-planning-helpers`; deps numpy/scipy/quadprog/matplotlib.
- Wrapper project: TUMFTM/global_racetrajectory_optimization (LGPL-3.0). Also offers shortest-path, min-time (CasADi) and iterative min-curvature.
- Option: depend on `tph` for v1 (LGPL as an unmodified pip dependency is fine). Or reimplement the ~100-line lsq version to avoid quadprog wheels on Windows. **I'd reimplement**, using tph as the validation oracle in tests.
- Other repos seen (yessin-hamdi/racing-line-optimizer, PSO/GA/DP variants) are less mature or unlicensed; skip them.

### 5.3 Speed profile (forward–backward, friction circle)

Per point i, curvature κ_i (from the final line), bank φ_i (positive = banked into the turn, i.e. sign-aware), grade γ_i (positive = uphill), grip μ (default 1.0; street tyres ≈ 0.9–1.0, semi-slicks ≈ 1.15). Let a_max = μ·g.

1. **Lateral limit with banking:** v_lat,i² = g·R_i·(μ + tan φ_i) / (1 − μ·tan φ_i), with R_i = 1/|κ_i|. Cap at v_cap: road-type based, e.g. 200–250 km/h for "race", or the OSM `maxspeed` × factor for traffic.
2. **Forward pass (acceleration):** v_{i+1}² = min(v_lat,{i+1}², v_i² + 2Δs·a_acc,i), where:
   - a_acc,i = min(a_engine(v_i), sqrt(max(0, a_max² − (v_i² κ_i)²))) − g·sin γ_i − k_drag·v_i²
   - a_engine(v) ≈ P/(m·v), clipped to about 0.5 g (generic 200 kW / 1300 kg car).
3. **Backward pass (braking):** v_i² = min(v_i², v_{i+1}² + 2Δs·a_brk,i), where a_brk,i = sqrt(max(0, a_max² − (v_{i+1}² κ_{i+1})²)) + g·sin γ_i + k_drag·v².
4. **Closed loop:** run the passes twice around (or start at the global minimum-speed point). **Open A→B:** v_0 = 0 (standing start) or a rolling speed; v_N free (no end constraint), or ~0 if AI must stop at B.
5. Derive the fields:
   - gas = 1 where the forward pass is limiting or at v_cap, otherwise partial
   - brake = a_brk_used / a_max in backward-limited zones
   - obsolete_lat_g = v²κ/g (or 0)
   - radius = R_i (capped)
6. Light smoothing (moving average ≈ 5 points) on κ before step 1. Curvature from discrete points is noisy and would make spiky speed limits.

TUMFTM `calc_vel_profile(ax_max_machines, kappa, el_lengths, closed, drag_coeff, m_veh, ggv, loc_gg, v_max, dyn_model_exp, mu, v_start, v_end, …)` implements exactly this, with GGV diagrams and open/closed support. **VERIFIED** (signature). Our hand-rolled version just adds the explicit grade and bank terms.

### 5.4 Recommended v1 (simple and robust)

1. Centreline → uniform 1 m samples (calibrate in S0).
2. Min-curvature lsq with margin, 3 iterations, blend 0.7 → line.
3. Heights from the road surface. 3D tangents/normals.
4. κ smoothing → speed profile (μ = 1.0, generic car, v_cap 220 km/h).
5. Sides = distance from the line to each road edge − 0.2 m. Don't subtract the car width here; the AI accounts for its own width (ASSUMED; check S5).
6. Write `fast_lane.ai` (hasGrid=0) and copy it to `data/ideal_line.ai`.
7. Traffic lanes for AssettoServer: separate files with offset ±lane/2 from the centreline, each in its own driving direction, joined at the ends with `ConnectEnd`.
8. Validate offline:
   - round-trip through our reader
   - AcTools-equivalent checks: version 7, counts, length ≈ route length
   - no side < 1 m
   - radius > 0
   - every point within the road polygon

---

## 6. Point-to-point specifics

- **Timing objects** (COMMUNITY, several modder threads): `AC_AB_START_L/R`, `AC_AB_FINISH_L/R`. Splits use `AC_TIME_n_L/R` ("no need to place one at the start"). `ui_track.json` `"run": "point-to-point"` is seen on rally tracks. Spawn points (`AC_START_n`, `AC_PIT_n`, `AC_HOTLAP_START_0`) must sit **behind** the A gate, or the timer starts immediately.
- **AI on A→B in vanilla:** COMMUNITY (Mount Akina 2017 author, https://acmods.net/tracks/mount-akina-2017/): "AC's AI does not support point to point tracks as they hit the finish line and just divide by 0 and drive into walls". Steam users report some touge mods "work in practice mode only" and that quick race spawns them under the track. **Treat AI races on A→B as broken in vanilla until S6 proves otherwise.**
- **CSP:** I found **no** CSP changelog entry adding point-to-point race support. I grepped all archived CSP changelogs for "point-to-point", "A2B" and "AB": nothing beyond spectator counts for "rally and hillclimb tracks". CSP does add:
  - the P2P recording ability (0.1.77)
  - spline flip fixes
  - Lua tools that make a custom touge mode feasible: `physics.setAICarPosition`, `physics.setAINoInput`, `physics.setAISpline`, track scripts / New Modes
- **Session types expected to work on A→B** (ASSUMED, confirm in S6):
  - Practice and Hotlap/Time Attack for the player (timing A→B), Drift.
  - Ideal line works in any session if the spline exists.
  - Race / Quick Race / Weekend with AI: likely problematic at B.
  - Track Day: AI flood needs ≥ 7 km by default.
- **Workarounds, cheapest first:**
  1. **Loop layout**: for routes where the start and end can be joined (rare on real roads).
  2. **Out-and-back layout** (`layout_return`): the AI spline goes A→B, then the same road B→A on the other lane, joined at B with a turning bulb (a generated roundabout/paved area) and at A. The track becomes a closed "circuit" with a lap = there and back. Works with vanilla AI races (ASSUMED). Costs extra geometry at the ends. A good v1.5 option for touge-style races.
  3. **CSP Lua touge mode**: both cars start at A on an A→B spline. The script detects crossing B, declares the winner, then freezes (`setAINoInput`) or teleports cars back to A (`setAICarPosition`) for the next run. Needs CSP. Rated **feasible but needs research** (S6 decides).
  4. **AssettoServer traffic**: open lane splines are fine. Use `ConnectEnd` (docs: "To connect a spline end to another spline instead of using a junction, use ConnectEnd") from the A→B lane's end to the B→A lane's start and vice versa, so traffic loops forever. VERIFIED feature; untested here (S10).
- **Known ideal-line issues on A→B** (ASSUMED/COMMUNITY): the last point can be "completely off" (AssettoServer comment). Don't let the spline extend past the B gate into nothing, and keep it on the physics surface to the very end. Keep under 54 km.

---

## Source list

- AssettoServer FastLaneParser / FastLaneUtils / PhysicsUtils / AdjacentLaneDetector: https://github.com/compujuckel/AssettoServer (main @ 1365f3e)
- AssettoServer docs (junctions guide, FAQ, intro): https://github.com/compujuckel/assettoserver-docs , https://assettoserver.org/docs/ai-splines/junctions_guide/
- AcTools AiFile + CM AiLaneObject/TrackMapRendererWrapper/SystemSettings: https://github.com/gro-ove/actools (@ e5e5c94)
- leBluem io_import_accsv: https://github.com/leBluem/io_import_accsv
- KinG-InFeT generator: https://github.com/KinG-InFeT/blender-assetto-corsa-track-generator
- CSP config (`new_behaviour.ini`): https://github.com/ac-custom-shaders-patch/acc-extension-config
- CSP changelog archive: https://github.com/c1xtz/csp-logs (site https://c1xtz.github.io/csp-logs/)
- TUMFTM: https://github.com/TUMFTM/trajectory_planning_helpers , https://github.com/TUMFTM/global_racetrajectory_optimization
- TreCorsa AI docs: https://trecorsa.com/docs/?cat=ailines
- RacetrackDesign: https://racetrackdesign.com/guides/how-to-make-a-track-for-assetto-corsa ; Lapwerks: https://lapwerks.com/
- Community: https://assettocorsamods.net/threads/how-can-i-make-ai-path.234/ , https://assettocorsamods.net/threads/ai-problem.1761/page-2 , https://assettocorsamods.net/threads/there-is-a-part-of-my-ai-spline-boundaries-thats-missing.3270/ , https://assettocorsamods.net/threads/build-your-first-track-basic-guide.12/page-95 , https://assettocorsamods.net/threads/need-help-with-timing-objects-for-a-to-b-type-stage.1056/ , https://assettocorsamods.net/threads/timer-for-up-downhill.2260/ , https://steamcommunity.com/app/244210/discussions/0/3123787625596058139/ , https://acmods.net/tracks/mount-akina-2017/
- Not reachable (403), seen only via search snippets: overtake.gg AI Line Helper / ai_hints threads, GTPlanet mods thread.
