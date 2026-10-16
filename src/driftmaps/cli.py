"""Assetto Corsa Map Generator's command line: `drift-maps ...` (or `python -m driftmaps ...`).

`drift-maps ui` starts the map app; the other commands are the same steps, one at a time."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from driftmaps.geo.router import Route

app = typer.Typer(
    help="Assetto Corsa Map Generator: turn real roads into Assetto Corsa tracks. "
    "Run `drift-maps ui` to pick a road on a map.",
    no_args_is_help=True,
    add_completion=False,
)

# Zip sizes for sending a track (MB = 1,000,000 bytes): warn from 25 MB, error from 28 MB.
ZIP_WARN_MB = 25.0
ZIP_ERROR_MB = 28.0


def check_zip_size(zip_path: Path) -> float:
    """Print the zip's size; warn at >= `ZIP_WARN_MB`, error (exit 1) at >= `ZIP_ERROR_MB`.
    Returns the size in MB."""
    mb = Path(zip_path).stat().st_size / 1e6
    typer.echo(f"zip      {zip_path} ({mb:.1f} MB)")
    if mb >= ZIP_ERROR_MB:
        typer.echo(
            f"error: {zip_path} is {mb:.1f} MB, over the {ZIP_ERROR_MB:.0f} MB limit; build with "
            "fewer trees (--scenery-density 0.5) or without buildings",
            err=True,
        )
        raise typer.Exit(1)
    if mb >= ZIP_WARN_MB:
        typer.echo(
            f"warning: {zip_path} is {mb:.1f} MB, close to the {ZIP_ERROR_MB:.0f} MB limit",
            err=True,
        )
    return mb


@app.command()
def ui(
    port: int = typer.Option(None, help="Port on 127.0.0.1 (default: 8765, or the next free one)."),
    browser: bool = typer.Option(True, help="Open the page in your web browser."),
    data_dir: Path = typer.Option(None, help="App data folder (default: the per-user app folder)."),
) -> None:
    """Start the map app: click points on a map (A to B, or a loop), get the track installed."""
    from driftmaps.app import run

    run(port=port, open_browser=browser, data_dir=data_dir)


@app.command()
def spike(
    out: Path = typer.Option(Path("out/spike"), help="Folder to write the test tracks into."),
    only: list[str] = typer.Option(
        None, help="Build only these test tracks (ids like dm_t1_minimal). Repeatable."
    ),
    zip_path: Path = typer.Option(None, "--zip", help="Also pack the tracks into this zip."),
) -> None:
    """Build the M0 in-game test tracks (see docs/checklists/M0.md)."""
    from driftmaps.spike import build_spike_tracks

    built = build_spike_tracks(out, only=only or None, zip_path=zip_path)
    for track_dir in built:
        typer.echo(f"built {track_dir}")
    if zip_path:
        typer.echo(f"zip   {zip_path}")


@app.command("kn5-dump")
def kn5_dump(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, help="A .kn5 file."),
    meshes: bool = typer.Option(True, help="List mesh nodes as well as dummies."),
) -> None:
    """Print the textures, materials and node tree of a KN5 file."""
    from driftmaps.ac.kn5 import dump

    typer.echo(dump(path, include_meshes=meshes))


@app.command("ai-dump")
def ai_dump(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, help="A fast_lane.ai file."),
    csv: Path = typer.Option(None, help="Also write every point and payload field to a CSV."),
) -> None:
    """Print a summary of an AI line file (point spacing, speed, widths, units)."""
    from driftmaps.ac.ai_line import dump

    typer.echo(dump(path, csv_path=csv))


@app.command("find-ac")
def find_ac() -> None:
    """Print where Assetto Corsa is installed, if it can be found."""
    from driftmaps.config import find_ac_root

    root = find_ac_root()
    if root is None:
        typer.echo("Assetto Corsa not found. Pass the folder explicitly where a command asks.")
        raise typer.Exit(1)
    typer.echo(str(root))


@app.command("install-touge")
def install_touge(
    ac_root: Path = typer.Option(
        None, "--ac-root", help="Assetto Corsa folder (default: found automatically)."
    ),
    force: bool = typer.Option(
        False,
        help="Replace a same-named mode folder this tool did not install, or install without CSP.",
    ),
    trackday_variant: bool = typer.Option(
        False,
        "--trackday-variant",
        help="Also install a Track Day based copy (to try if the Race based one misbehaves).",
    ),
    uninstall: bool = typer.Option(False, help="Remove the mode instead."),
) -> None:
    """Install the Touge Battle CSP New Mode: 1v1 against one AI car on an A to B track."""
    from driftmaps.config import find_ac_root
    from driftmaps.csp_modes import ModeInstallError, uninstall_touge
    from driftmaps.csp_modes import install_touge as install_mode

    root = ac_root if ac_root is not None else find_ac_root()
    if root is None:
        typer.echo("Assetto Corsa not found; pass --ac-root or set DRIFTMAPS_AC_ROOT.", err=True)
        raise typer.Exit(1)
    if uninstall:
        removed = uninstall_touge(root)
        for folder in removed:
            typer.echo(f"removed  {folder}")
        if not removed:
            typer.echo("Touge Battle is not installed (nothing removed).")
        return
    try:
        folders = install_mode(root, force=force, trackday_variant=trackday_variant)
    except ModeInstallError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from None
    for folder in folders:
        typer.echo(f"install  {folder}")
    typer.echo(
        "Next: in Content Manager pick an A to B track built by this tool, choose the "
        '"Touge Battle (ACMG)" mode with one opponent, and drive. Steps and what to report: '
        "docs/checklists/touge.md"
    )


# --------------------------------------------------------------------------------------------
# M1: route + build
# --------------------------------------------------------------------------------------------


def _latlon(text: str) -> tuple[float, float]:
    """Parse "LAT,LON" (decimal degrees)."""
    try:
        lat_s, lon_s = text.replace(" ", "").split(",")
        lat, lon = float(lat_s), float(lon_s)
    except ValueError:
        raise typer.BadParameter(f"expected LAT,LON in decimal degrees, got {text!r}") from None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise typer.BadParameter(f"{text!r} is not a valid LAT,LON")
    return lat, lon


def _municipal(enabled: bool, burlington: bool) -> tuple[str, ...]:
    """The municipal building layers to use (geo.building_sources.MUNICIPAL_SOURCES)."""
    from driftmaps.geo.building_sources import MUNICIPAL_SOURCES

    if not enabled:
        return ()
    return tuple(n for n in MUNICIPAL_SOURCES if burlington or n != "burlington")


def _pins(start: str | None, end: str | None, via: list[str] | None) -> list[tuple[float, float]]:
    if not start or not end:
        raise typer.BadParameter("give --start and --end (or --route for build)")
    return [_latlon(start), *(_latlon(v) for v in via or []), _latlon(end)]


def _route(
    pins: list[tuple[float, float]],
    cache_dir: Path | None,
    osm_json: Path | None,
    buffer_m: float,
    offline: bool,
    refresh: bool,
    loop: bool = False,
) -> Route:
    from driftmaps.geo.overpass import parse_overpass
    from driftmaps.geo.router import route_from_pins

    osm = parse_overpass(osm_json.read_bytes()) if osm_json else None
    return route_from_pins(
        pins, cache_dir, osm=osm, buffer_m=buffer_m, offline=offline, refresh=refresh, loop=loop
    )


_START = typer.Option(None, help="Start pin A as LAT,LON.")
_END = typer.Option(None, help="End pin B as LAT,LON.")
_VIA = typer.Option(None, help="Waypoint LAT,LON the route must pass, in order. Repeatable.")
_CACHE = typer.Option(
    None, help="Download cache folder (default $DRIFTMAPS_CACHE_DIR, .cache in a checkout)."
)
_OSM_JSON = typer.Option(
    None, exists=True, dir_okay=False, help="Use this saved Overpass JSON instead of fetching."
)
_BUFFER = typer.Option(1500.0, help="Road corridor half-width fetched around the pins (m).")
_OFFLINE = typer.Option(False, help="Use cached road data only; never hit the network.")
_REFRESH = typer.Option(False, help="Re-download road data even if it is cached.")
_LOOP = typer.Option(False, help="Come back to the start: a loop (circuit) instead of A to B.")


@app.command()
def route(
    start: str = _START,
    end: str = _END,
    via: list[str] = _VIA,
    out: Path = typer.Option(Path("route.json"), help="Where to write route.json."),
    reverse: bool = typer.Option(False, help="Drive the route from B to A."),
    loop: bool = _LOOP,
    cache_dir: Path = _CACHE,
    osm_json: Path = _OSM_JSON,
    buffer_m: float = _BUFFER,
    offline: bool = _OFFLINE,
    refresh: bool = _REFRESH,
) -> None:
    """Route between pins over real roads and write route.json (what `build` reads)."""
    r = _route(_pins(start, end, via), cache_dir, osm_json, buffer_m, offline, refresh, loop)
    if reverse:
        r = r.reverse()
    r.save(out)
    typer.echo(f"route  {r.length_m / 1000:.2f} km, {len(r.edges)} OSM segments -> {out}")


@app.command()
def build(
    start: str = _START,
    end: str = _END,
    via: list[str] = _VIA,
    route_json: Path = typer.Option(
        None, "--route", exists=True, dir_okay=False, help="Build from this route.json."
    ),
    name: str = typer.Option(..., help="Track name shown in the game."),
    track_id: str = typer.Option(
        None, "--id", help="Track folder id (a-z, 0-9, _). Default: from the name."
    ),
    reverse: bool = typer.Option(False, help="Drive the route from B to A."),
    loop: bool = _LOOP,
    out: Path = typer.Option(Path("out/tracks"), help="Folder to write the track into."),
    zip_path: Path = typer.Option(None, "--zip", help="Also pack the track into this zip."),
    install: bool = typer.Option(
        False, help="Also copy the track into Assetto Corsa's content/tracks."
    ),
    force: bool = typer.Option(
        False,
        help="Replace an existing track folder of that id even if this tool did not make it.",
    ),
    splits: int = typer.Option(
        0, min=0, max=1, clamp=True, help="Intermediate timing splits (AC_TIME_0): 0 or 1."
    ),
    dem: str = typer.Option(
        None, help="Prefer this elevation source (e.g. glo30, hrdem-1m, 3dep-1m)."
    ),
    dem_file: Path = typer.Option(
        None, exists=True, dir_okay=False, help="Use this GeoTIFF as the only elevation source."
    ),
    dds: bool = typer.Option(False, help="Embed DDS textures instead of PNG."),
    debug_png: bool = typer.Option(True, help="Write top-down debug images to <out>/debug."),
    trees: bool = typer.Option(True, help="Trees from the lidar canopy (NRCan, Canada only)."),
    buildings: bool = typer.Option(
        True,
        help="Buildings: municipal footprints, OSM, NRCan lidar footprints and heights, then "
        "Microsoft ML footprints, merged into one building each.",
    ),
    nrcan_buildings: bool = typer.Option(
        True, help="NRCan lidar building footprints and heights (Canada; OGL-Canada)."
    ),
    municipal_buildings: bool = typer.Option(
        True,
        help="Municipal building footprints where they exist (Burlington, Milton, Oakville).",
    ),
    burlington_buildings: bool = typer.Option(
        True,
        help="City of Burlington buildings (exact outlines, eave and peak heights). Its open "
        "data terms are revocable and must travel with the track: LICENSE-DATA.txt carries "
        "their URL. --no-burlington-buildings builds without them.",
    ),
    guardrails: bool = typer.Option(
        True,
        help="Guardrails from guardrails.json and OSM, plus the automatic ones (bridges keep "
        "their parapets; at-grade culverts have none). --no-guardrails: no rails at all.",
    ),
    auto_guardrails: bool = typer.Option(
        True,
        "--auto-guardrails/--no-auto-guardrails",
        help="Also put guardrails where Ontario road-design rules require one: high or steep "
        "fills and drop-offs (lidar), deep water and bridge ends inside the clear zone, cut at "
        "side roads and driveways. data/guardrails_auto.json says why each one is there.",
    ),
    bridge_approach_rails: bool = typer.Option(
        False,
        help="Also put 20 m of guardrail on both sides of each bridge approach (invented: "
        "no source says they exist).",
    ),
    scenery_density: float = typer.Option(
        1.0, min=0.0, max=2.0, help="Tree budget multiplier (0 = no trees, 1 = default cap)."
    ),
    guardrails_file: Path = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Guardrail spans JSON ([{side, from_m, to_m}], metres from A as driven in the "
        "route's own direction; with --reverse they are mirrored). Default: guardrails.json "
        "next to --route, if there is one.",
    ),
    cache_dir: Path = _CACHE,
    osm_json: Path = _OSM_JSON,
    buffer_m: float = _BUFFER,
    offline: bool = _OFFLINE,
    refresh: bool = _REFRESH,
) -> None:
    """Build an Assetto Corsa track folder from pins (or a route.json)."""
    from driftmaps.ac.scene import TrackFolderError, check_track_folder, install_track, zip_tracks
    from driftmaps.ac.track_files import make_track_id, validate_track_id
    from driftmaps.build import BuildOptions, build_track
    from driftmaps.config import find_ac_root, tracks_dir
    from driftmaps.geo.barriers import USER_SPANS_FILE
    from driftmaps.geo.router import Route

    # Refuse to overwrite somebody else's track before any download or build work.
    tid = validate_track_id(track_id or make_track_id(name))
    targets = [out / tid]
    ac_tracks = None
    if install:
        root = find_ac_root()
        if root is None:
            typer.echo("Assetto Corsa not found; set DRIFTMAPS_AC_ROOT to install.", err=True)
            raise typer.Exit(1)
        ac_tracks = tracks_dir(root)
        targets.append(ac_tracks / tid)
    try:
        for target in targets:
            check_track_folder(target, force=force)
    except TrackFolderError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from None

    if route_json is not None:
        r = Route.load(route_json)
        if guardrails_file is None and (route_json.parent / USER_SPANS_FILE).is_file():
            guardrails_file = route_json.parent / USER_SPANS_FILE
    else:
        r = _route(_pins(start, end, via), cache_dir, osm_json, buffer_m, offline, refresh, loop)
    if reverse:
        r = r.reverse()
    options = BuildOptions(
        name=name,
        track_id=track_id,
        cache_dir=cache_dir,
        splits=splits,
        elevation_prefer=dem,
        texture_format="dds" if dds else "png",
        debug_png=debug_png,
        force=force,
        trees=trees,
        buildings=buildings,
        nrcan_buildings=nrcan_buildings,
        municipal_buildings=_municipal(municipal_buildings, burlington_buildings),
        guardrails=guardrails,
        auto_guardrails=auto_guardrails,
        bridge_approach_rails=bridge_approach_rails,
        scenery_density=scenery_density,
        guardrails_file=guardrails_file,
        route_reversed=reverse,
        offline=offline,
    )
    if dem_file is not None:
        from driftmaps.elevation.sources import local_dem_source

        source = local_dem_source(dem_file)
        options.elevation_sources = {source.name: source}
        options.elevation_chain = [source.name]
    result = build_track(r, out, options)
    typer.echo(result.summary())
    for image in result.debug_images:
        typer.echo(f"debug    {image}")
    if zip_path:
        zip_tracks([result.track_dir], zip_path)
        check_zip_size(zip_path)
    if ac_tracks is not None:
        typer.echo(f"install  {install_track(result.track_dir, ac_tracks, force=force)}")


# ---------------------------------------------------------------------------------------------
# Offline road data
# ---------------------------------------------------------------------------------------------

offline_app = typer.Typer(
    help="Offline road data: download a region's roads once (e.g. Ontario, ~1 GB) so road "
    "tiles, routes and builds there never wait for Overpass.",
    no_args_is_help=True,
)
app.add_typer(offline_app, name="offline")


def _offline_manager(data_dir: Path | None):
    from driftmaps.app.offline import OfflineManager
    from driftmaps.app.settings import Settings
    from driftmaps.geo import offline as region_db

    folder = Settings(data_dir).data_dir / "offline" if data_dir else region_db.offline_dir()
    return OfflineManager(folder)


def _region(region: str) -> str:
    from driftmaps.app.offline import REGIONS

    if region not in REGIONS:
        raise typer.BadParameter(f"unknown region {region!r}; known: {', '.join(REGIONS)}")
    return region


@offline_app.command("download")
def offline_download(
    region: str = typer.Option("ontario", help="Region to download."),
    source_file: Path = typer.Option(
        None, exists=True, dir_okay=False, help="Use this .osm.pbf instead of downloading."
    ),
    poly_file: Path = typer.Option(
        None, exists=True, dir_okay=False, help="With --source-file: its boundary (.poly)."
    ),
    data_dir: Path = typer.Option(None, help="App data folder (default: the app's)."),
) -> None:
    """Download a region's OSM extract (resumable) and build its offline road database."""
    import time

    from driftmaps.app.offline import Cancelled, DownloadError

    manager = _offline_manager(data_dir)
    shown = {"key": None}
    t0 = time.monotonic()

    def progress(state: str, fraction: float, text: str) -> None:
        key = (state, int(fraction * 20))
        if key != shown["key"]:
            shown["key"] = key
            typer.echo(
                f"{time.monotonic() - t0:7.0f} s  {state:<11} {fraction * 100:5.1f}%  {text}"
            )

    try:
        stats = manager.run(
            _region(region), progress=progress, source_file=source_file, poly_file=poly_file
        )
    except (DownloadError, Cancelled) as exc:
        typer.echo(f"error: {exc or 'cancelled'}", err=True)
        raise typer.Exit(1) from None
    counts = ", ".join(f"{v} {k}" for k, v in stats.counts.items())
    typer.echo(
        f"ready    {stats.path} ({stats.size_bytes / 1e6:.0f} MB, {counts}; "
        f"processed in {stats.seconds:.0f} s; data from {stats.osm_base or 'unknown date'})"
    )


@offline_app.command("status")
def offline_status(
    region: str = typer.Option("ontario", help="Region."),
    data_dir: Path = typer.Option(None, help="App data folder (default: the app's)."),
) -> None:
    """Show whether a region's offline road data is ready."""
    manager = _offline_manager(data_dir)
    st = manager.status(_region(region))
    typer.echo(f"folder   {manager.dir}")
    if st["state"] == "ready":
        counts = ", ".join(f"{v} {k}" for k, v in (st.get("counts") or {}).items())
        typer.echo(
            f"ready    {st['region_name']}: {st['size_mb']} MB, data from {st['updated']} "
            f"({counts})"
        )
    else:
        typer.echo(f"{st['state']:<8} {st['region_name']}: {st['message'] or 'not downloaded'}")


@offline_app.command("delete")
def offline_delete(
    region: str = typer.Option("ontario", help="Region."),
    data_dir: Path = typer.Option(None, help="App data folder (default: the app's)."),
) -> None:
    """Delete a region's offline road data (and any unfinished download)."""
    manager = _offline_manager(data_dir)
    manager.delete(_region(region))
    typer.echo(f"deleted  {region} offline road data from {manager.dir}")
