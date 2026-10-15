# M0 results (in game, user's PC)

Tested 2026-10-15 on the user's Windows PC: Assetto Corsa + Content Manager + Custom Shaders Patch
(CSP on). Car: Toyota GT86. Partial run; tracks not listed here are still untested.

| Test | Track | Mode | Result | Notes |
|---|---|---|---|---|
| S1 load | dm_t1_minimal | Practice, Race (2 laps, 3 AI) | **PASS** | KN5 v5 loads, PNG textures render, car sits on the road, no falling through. Race grid (`AC_START_n`) spawns 4 cars facing down the road. AI cars sit still: expected, t1 has no AI line. CM showed the "tag circuit is missing" warning for the 2-lap race: expected for a non-loop track. |
| S4 handedness | dm_t12_ai_ab | Race with AI | **PASS** | The road turns LEFT after the first straight, as generated (north, then west). Confirms X = east, Z = south. |
| S6 A→B timing | dm_t12_ai_ab | Race with AI | **PASS** | A timer ran with `AC_AB_START_L/R` and `AC_AB_FINISH_L/R` gates (L on the driver's left). |
| AI line format | dm_t12_ai_ab | Race with AI | **PASS** | AI cars followed our generated `fast_lane.ai` (v7, with grid) around the corner and raced to the end. |
| AI at the A→B finish | dm_t12_ai_ab | Race with AI | **As predicted (problem)** | AI drives past the finish to the end of the line, into the end wall, and the cars pile up honking. Matches the research risk (`ai_line.md` §6): vanilla AI has no point-to-point finish. The touge mode (M3) must stop or teleport cars at B with CSP Lua. |

## What this settles

- Writing KN5 directly from Python works (`docs/research/ac_format.md` §0, recommendation D2).
- AC frame X = east, Y = up, Z = south is correct (D6). `_L` gates on the driver's left work.
- Generated AI lines work for AI driving, at least with CSP on and the lookup grid present (D3).
- A→B AI races need our own finish handling (D7 / M3), confirmed in game.

## Still open from M0

T2–T11 individually (walls, swapped gates, spawn heights, DDS/NPOT textures, models.ini, vertex
limit/unknown dummies/LOD, KN5 v6, AI loop, no-grid AI file, CSP off), and the S0 AI-unit files.
