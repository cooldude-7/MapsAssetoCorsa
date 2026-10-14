# Prompt: Plan a recreation of Drift Maps (real roads → Assetto Corsa tracks)

> Paste everything below the line into a fresh Claude Code session in this repo.

---

You are planning, **not yet building**, a clone of an app called **Drift Maps** (driftmap.us).
The user picks a start point and an end point on a real-world road. The app follows real roads
between them and produces a driveable **Assetto Corsa** track. Your output is a detailed,
buildable plan that a later session can follow step by step.

The repo (`MapsAssetoCorsa`) is currently empty. Write the plan to `docs/PLAN.md`, then
commit and push it to the working branch.

## 1. What we know about the original app

All of this comes from two screen recordings of driftmaps' own Instagram Reels. Treat it as
ground truth for *what* the product does, not *how*.

**User flow (web app at driftmap.us):**
1. **Sign in or create an account.** Email plus a password of at least 8 characters, or *"Email me a sign-in link
   instead"* (a magic link). Footer: *"Road data © OpenStreetMap contributors (ODbL). Elevation from
   USGS 3DEP and Copernicus DEM. Drift Maps is not affiliated with or endorsed by Kunos Simulazioni;
   Assetto Corsa is their trademark, named only to state compatibility."* The consent text says
   generated tracks are the user's to share.
2. **"Pick your road."** A full-screen dark map (Leaflet + CARTO dark tiles + OSM attribution) with
   zoom buttons and a place search box (placeholder examples: "Mount Haruna · Angeles Crest · St. Cha…").
   Instructions say: *"Search the area, then click the road: first click sets the start, the last
   click is the finish, and clicks in between become waypoints."*
   - Pins: green = start, amber = waypoints, red = finish. Pins can be dragged.
   - An **amber line** shows the routed road, *"snapped to the real network. The rig routes the
     same pins the same way."*
   - Header nav: **Your builds · Upgrade · Sign out**, which implies build history and a paid tier.
3. **Route panel.** START/FINISH lat-lon pairs (for example `41.71712, -88.29689` →
   `41.72328, -88.27246`, Aurora IL), each removable with ×. Buttons: **Undo last / Reverse / Clear**.
   Stats: **Routed length 3.6 km**, **Build time 10–30 min**.
4. **Track name.** The placeholder is "Akina Downhill". Then **Build this track** shows
   *"Sending this road to the build rig…"* and the note *"The rig builds this road, drives it in
   Assetto Corsa and emails you the track — usually 10–30 minutes. You can close this tab."*

**What the output looks like in game (from their "Day 8: Gorges d'Ollioules", D N8, France reel):**
- The road layout and elevation match real life closely when compared side by side with Google Street View.
- Scenery is generic and procedural: repeated tree models and billboards, flat grass, a distant
  mountain backdrop, plain asphalt. There are no real road markings or buildings.
- It's point to point (an A→B mountain-road run), driven in a stock-style car (Corvette C7).

**What this implies (verify, don't assume):**
- Routing is done over OSM data, with the same engine on the frontend preview and on the backend.
- Elevation comes from a DEM: USGS 3DEP for the US (high resolution) and Copernicus GLO-30 everywhere else.
- A "build rig" runs **real Assetto Corsa**. That's probably for generating or recording the AI line
  (`fast_lane.ai`) and/or checking that the track loads. This is why a build takes 10–30 minutes and is
  emailed instead of downloaded instantly.

## 2. Decisions the user has already made

Don't ask about these again:
- **Local first, website later.** Build the generator as a library plus a local tool that writes the
  track straight into the user's `assettocorsa/content/tracks` folder. Design it so a
  driftmaps-style website can sit on top later, but don't build the website in the first
  milestones.
- **The user owns Assetto Corsa and Content Manager on Windows.** Prefer generating everything,
  including the AI line, directly in code so builds don't need the game running. The user's PC is
  for testing in game, and is the fallback if a step (for example AI-line recording) really needs
  the game. Content Manager is installed, so it's fine to rely on it for testing (its track
  tools, quick launch) but not as a required build step.
- **First working version = road + real elevation + terrain + walls.** Trees, buildings and other
  scenery come later.
- **Free, personal project.** No payments, no Upgrade tier, no account tiers. Accounts only matter
  if and when the website happens.
- **AI cars are a core goal, not an extra.** Every generated track must ship a working AI line
  (`ai/fast_lane.ai`), so the ideal-line assist works and the track is ready for bots. The user wants
  two kinds of bot cars eventually:
  1. **Traffic cars on the road**, Shutoko Revival style: cars cruising both directions while you drive.
  2. **Touge battles**, Initial D style: a 1v1 race down the mountain road against an AI car. The
     dream is an AI that **drifts the corners** like in the show.
  Be honest in the plan about what vanilla AC, Custom Shaders Patch (CSP) and AssettoServer can and
  can't do here. Vanilla AI drives grip lines and doesn't drift. Find out what it would actually
  take (CSP Lua AI scripting, custom AI controllers, physics tricks) and rate it as feasible,
  hard, or research project.

## 3. Your job

Research, decide and write the plan. Use subagents and web search for research where they help.
If a new decision that belongs to the user comes up, **ask with AskUserQuestion** and recommend
an option instead of guessing. Explain AC jargon (KN5, AI line, CSP…) in plain words when you ask:
the user is a player, not a modder.

## 4. Questions the plan must answer

### A. Assetto Corsa track format (the hardest part, so research it properly)
- The required folder layout under `content/tracks/<id>/`: `<id>.kn5` or `models.ini` + several kn5 files,
  `data/surfaces.ini`, `data/cameras.ini`, `data/map.ini`, `data/groove.ini`, `map.png`,
  `ui/ui_track.json`, `ui/preview.png`, `ui/outline.png`, `ai/fast_lane.ai`, `ai/pit_lane.ai`.
  Also layouts/configs (`ui/<layout>/`, `<layout>/data`) if needed.
- Required dummy/empty objects and naming: `AC_START_n`, `AC_PIT_n`, `AC_HOTLAP_START_n`,
  `AC_TIME_n_L/R`. For **point-to-point** tracks: `AC_AB_START_L/R` and `AC_AB_FINISH_L/R`.
  Explain how A→B timing and the session types (practice, hotlap, drift) work.
- Keep the bot-car goals (section H) in mind here: AI needs clean road boundaries, and two-way
  traffic needs lanes, so record lane data from the start.
- Physical surfaces: mesh naming conventions (for example `1ROAD`, `1GRASS`, `1KERB`, `1WALL`, `1SAND`)
  and how they map to `surfaces.ini` keys (friction, damping, `IS_VALID_TRACK`, etc.).
- **KN5 file format:** the binary structure (header, textures, materials and shader properties, node
  tree, mesh vertex layout). Can we write KN5 directly from code (Python/Rust/C#)? Survey existing
  open-source writers and readers, Blender add-ons, FBX→KN5 paths, and whether Kunos' ksEditor
  can be scripted or run headless. Recommend one path and explain why.
- Shaders and materials to use (`ksPerPixel`, `ksPerPixelMultiMap`, `ksTree`, `ksGrass`…), plus
  texture format (DDS) requirements.
- **AI line:** the `fast_lane.ai` binary format. Can it be generated directly from the route
  centreline (with a racing-line/speed solve), or does it need recording in game? This decides
  whether the build worker needs the game at all.
- Custom Shaders Patch (CSP) extras worth supporting optionally, such as `ext_config.ini`, grass FX,
  and lights. Keep in mind that vanilla AC must still work.

### B. Road data and routing
- Fetching OSM data: the Overpass API versus a local extract (Geofabrik `.pbf`). Cover rate limits
  and caching.
- Routing engine: OSRM vs GraphHopper vs Valhalla vs a custom shortest path over the fetched
  graph. It must match exactly between the frontend preview and the backend build. How do we snap
  clicks to roads, handle waypoints, reverse, and stop U-turns?
- Road attributes: width from `width`/`lanes`/`highway` tags with sensible defaults; `surface`
  (asphalt/gravel); bridges and tunnels (`bridge=yes`, `tunnel=yes`, `layer`); one-ways (ignore for
  racing?).
- Limits on route length, and what to do with junctions and roads that cross themselves.

### C. Elevation
- Sources: USGS 3DEP (1 m / 1/3 arc-sec, US), Copernicus GLO-30 (global, ~30 m). Fallbacks
  are SRTM and ALOS. Cover access methods (AWS open data, OpenTopography API) and licensing/attribution.
- Turning a noisy DEM into a driveable profile: sample along the centreline, smooth (for example
  Savitzky–Golay or spline fitting), limit grade and vertical curvature, keep bridges and tunnels
  level instead of following the terrain under them, and cut or fill the terrain so it meets the road.

### D. Geometry generation
- Projection: WGS84 to a local metric frame (ENU or UTM) with the origin at the route start. Use
  AC's axis conventions (Y-up; check the handedness).
- Centreline: resample evenly, smooth the corners (OSM nodes are polylines), compute tangents,
  normals and optional banking/camber.
- Road mesh: carriageway, shoulders/verges, optional painted lines as a separate material, and UVs
  that tile asphalt textures along the road.
- Terrain: a heightfield corridor around the road (how wide?) blended into the road edges, at lower
  resolution further away. Add a horizon/backdrop.
- Barriers: invisible walls or guardrails so cars can't fall off the world, and the outer
  world bounds.
- Scenery (procedural, phase 2+): trees from OSM `landuse`/`natural=wood`, buildings from footprints,
  grass. Use instanced billboard or low-poly trees within AC's object limits.
- Performance budgets: triangle and draw-call counts and kn5 size. Split into chunks via
  `models.ini` if needed.

### E. Generated metadata
- `map.png` plus `map.ini` (the minimap) and outline/preview images.
- `ui_track.json`: name, length, tags, geotags (lat/lon), country, run (A→B), pitboxes.
- Pit/start positions for a point-to-point run (start grid at A), `cameras.ini` for replays.

### F. Product and infrastructure (later phase: website)
- Frontend stack: Leaflet or MapLibre with CARTO dark tiles. Cover the pin UX (drag, undo,
  reverse, clear), place search (Nominatim or Photon, respecting usage policies) and live route
  length.
- Backend: API, auth (email/password plus magic link), "Your builds" history, a job queue, build
  workers, artifact storage (zip) and email delivery. No paid tier (this is a free project), but
  plan sensible abuse limits (route length, builds per day).
- How a worker runs. If it needs AC: a Windows VM, a licence, automation, and how many builds run
  at once. If not: containerised Linux workers.
- Hosting cost estimate (keep it cheap, since this is free) and the expected build-time target.
- For the local tool: how the user installs and runs it on Windows (a Python package, a single exe,
  or a small local web page with the map picker), and how it finds the AC install folder.

### G. Legal and attribution
- ODbL obligations for OSM-derived tracks (attribution; is a track a "produced work"?).
- DEM licences (USGS public domain, Copernicus attribution terms).
- Kunos/AC trademark use: a compatibility statement only, and never ship Kunos assets (cars,
  textures, shaders from the game files).

### H. Bot cars: traffic and touge battles
- **AI line generation:** produce `fast_lane.ai` (and `pit_lane.ai` if needed) from the road
  centreline. Cover the racing line, speed profile from curvature and grade, and the left/right
  boundaries the AI uses. Fallback: record it in game with the AI-line recording tools in AC or Content
  Manager on the user's PC, and document the exact steps.
- **Point-to-point limits:** can vanilla AC run a *race* session with AI on an A→B track, or only
  hotlap/time attack? What does CSP add? Workarounds if needed: generate an out-and-back layout, a
  loop variant, or a touge layout where both cars start side by side at A and finish at B.
- **Traffic:** how traffic works in (a) CSP offline (for example its traffic/"new AI" features and the
  Traffic Planner tool), and (b) AssettoServer's AI traffic for multiplayer (lane splines,
  `fast_lane.ai`-based splines, two-way traffic, spawn density). What data does each need from the
  track (lanes in both directions, junctions, lane widths from OSM `lanes` tags), and can we
  generate it automatically?
- **Touge battle AI:** what controls exist for AI aggression and driving style (AI level, `ai_hints.ini`,
  CSP AI tweaks, Lua scripting)? Is a **drifting AI** possible, and how? Look at existing community
  drift-AI or Lua-controlled car projects. Recommend a realistic path: start with a fast grip-driving
  AI opponent, then make drifting a stretch goal with its own research spike.
- What the user installs for each option (CSP version, AssettoServer, Content Manager settings).

## 5. Deliverable: `docs/PLAN.md`

Structure it like this:
1. **Summary.** What we're building, in one paragraph, plus the key decisions made, each with a reason.
2. **Architecture.** A component diagram (Mermaid) showing the data flow
   `pins → route → elevation → geometry → kn5 + data files → AI line → zip`.
3. **Tech stack**, with the reason for each choice.
4. **Repository layout** (directories and modules).
5. **Detailed design** for sections A–H above, including file formats with field-level detail
   wherever you found authoritative sources. Cite links.
6. **Milestones.** Each milestone is shippable, with concrete acceptance tests. For example:
   - **M0 (spike):** prove we can write a KN5 that AC loads: a flat test strip with a start
     point and the right surfaces, checked in Content Manager.
   - **M1 (first real version):** CLI `drift-maps build --start LAT,LON --end LAT,LON --name "Test"`
     outputs a track folder in `content/tracks` with the real road shape, **real elevation, terrain and
     walls**, and working A→B timing. The test routes are Gorges d'Ollioules D N8 (France, Copernicus)
     and the 3.6 km Aurora IL route above (3DEP).
   - **M2:** a generated AI line, so the ideal-line assist works and an AI car can drive the road from
     A to B in game.
   - **M3:** touge battle v1: race one AI opponent down the road (grip driving is fine).
   - **M4:** traffic cars on the road (two-way), via CSP and/or AssettoServer, whichever the
     research recommends.
   - **M5:** a local map picker (a small web page served by the tool) with pins, a route preview
     and a Build button.
   - **M6:** procedural scenery (trees, grass, buildings).
   - **Stretch:** Initial D-style drifting AI for touge battles.
   - **Later:** a hosted website with accounts, "Your builds" and email delivery.
7. **Testing strategy.** Unit tests for geometry and format writers, round-trip tests that read the
   KN5 back with an independent parser, golden files, and a manual in-game checklist.
8. **Risks and open questions**, ranked, each with a fallback (for example "a direct KN5 writer turns out to be
   too hard → fall back to the FBX + ksEditor pipeline").
9. **Very first steps.** The exact first 5–10 tasks for the implementation session.

## 6. Rules
- Mark every claim about AC's file formats as **verified** (with a source) or **assumed**. Don't
  invent binary layouts. If you can't confirm one, say so and plan a spike to reverse-engineer
  or test it.
- Keep the plan concrete enough that another engineer could start M1 the same day.
- Don't write application code in this session. Small throwaway spikes to check a format
  question are fine if you clearly mark them.
- Commit `docs/PLAN.md` with a clear message and push it to the working branch. Don't open a PR
  unless asked.
