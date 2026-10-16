# Assetto Corsa Map Generator

Turn real roads into Assetto Corsa tracks: pick your road on a map turn by turn (click the
road you start on, then the road you want at every junction) or by clicking points A, B, C...,
from A to B or round a loop "back to start", and the app builds a driveable track from
OpenStreetMap road data and real (lidar) elevation, with trees, buildings, bridges and
guardrails where Ontario's road rules call for them, and installs it so it shows up in Content
Manager (a loop is a circuit you can race lap after lap).

Status: **M5 (the map app)**. Players: get the Windows app from
[`docs/INSTALL_APP.md`](docs/INSTALL_APP.md) and test it with
[`docs/checklists/M5-app.md`](docs/checklists/M5-app.md). The full plan is in
[`docs/PLAN.md`](docs/PLAN.md); older in-game checklists are in [`docs/checklists/`](docs/checklists/).

## The app

The app runs on your own PC: a small local web server (127.0.0.1 only) serves a map page in
your browser, and builds run on your machine. Everything it keeps (settings, downloaded map
data, built tracks, `app.log`) is in `%LOCALAPPDATA%\AssettoCorsaMapGenerator` on Windows
(`~/.local/share/AssettoCorsaMapGenerator` elsewhere). It only lists or deletes tracks it made.
Roads for turn by turn load in z14 map tiles round the view (cached per tile), and the page
works out the junctions and turns itself, so picking each turn is instant. The version is shown
next to the app's name ("Update N · date · commit", or "dev build").

## Development

```bash
uv sync                           # Python 3.12 environment + dev tools
uv run pytest                     # tests (offline)
uv run drift-maps ui              # start the map app (--no-browser, --port N, --data-dir DIR)
uv run drift-maps --help          # the same steps from the command line: route, build, ...
uv run pyinstaller --noconfirm --clean packaging/acmg.spec   # the Windows app (on Windows)
```

`DRIFTMAPS_AC_ROOT` points the app at an Assetto Corsa folder, and `DRIFTMAPS_CACHE_DIR` at a
download cache. `tests/test_app_e2e.py` explains how to run the app in a real browser
(Playwright) against the live map servers.

Road data © OpenStreetMap contributors (ODbL). Not affiliated with or endorsed by Kunos
Simulazioni; Assetto Corsa is their trademark, named only to state compatibility.
