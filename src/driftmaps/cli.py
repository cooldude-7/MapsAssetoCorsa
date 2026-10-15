"""Command-line entry point: `drift-maps ...` (or `python -m driftmaps ...`)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from driftmaps.geo.router import Route

app = typer.Typer(
    help="Turn real roads into Assetto Corsa tracks.",
    no_args_is_help=True,
    add_completion=False,
)


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
) -> Route:
    from driftmaps.geo.overpass import parse_overpass
    from driftmaps.geo.router import route_from_pins

    osm = parse_overpass(osm_json.read_bytes()) if osm_json else None
    return route_from_pins(
        pins, cache_dir, osm=osm, buffer_m=buffer_m, offline=offline, refresh=refresh
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


@app.command()
def route(
    start: str = _START,
    end: str = _END,
    via: list[str] = _VIA,
    out: Path = typer.Option(Path("route.json"), help="Where to write route.json."),
    reverse: bool = typer.Option(False, help="Drive the route from B to A."),
    cache_dir: Path = _CACHE,
    osm_json: Path = _OSM_JSON,
    buffer_m: float = _BUFFER,
    offline: bool = _OFFLINE,
    refresh: bool = _REFRESH,
) -> None:
    """Route between pins over real roads and write route.json (what `build` reads)."""
    r = _route(_pins(start, end, via), cache_dir, osm_json, buffer_m, offline, refresh)
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
    out: Path = typer.Option(Path("out/tracks"), help="Folder to write the track into."),
    zip_path: Path = typer.Option(None, "--zip", help="Also pack the track into this zip."),
    install: bool = typer.Option(
        False, help="Also copy the track into Assetto Corsa's content/tracks."
    ),
    force: bool = typer.Option(
        False,
        help="Replace an existing track folder of that id even if drift-maps did not make it.",
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
    buildings: bool = typer.Option(True, help="Buildings from OSM + Microsoft footprints."),
    guardrails: bool = typer.Option(
        True,
        help="Guardrails where OSM or guardrails.json has them, and on bridge approaches "
        "(bridges always keep their parapets).",
    ),
    scenery_density: float = typer.Option(
        1.0, min=0.0, max=2.0, help="Tree budget multiplier (0 = no trees, 1 = default cap)."
    ),
    guardrails_file: Path = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Guardrail spans JSON ([{side, from_m, to_m}], metres from A). Default: "
        "guardrails.json next to --route, if there is one.",
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
        r = _route(_pins(start, end, via), cache_dir, osm_json, buffer_m, offline, refresh)
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
        guardrails=guardrails,
        scenery_density=scenery_density,
        guardrails_file=guardrails_file,
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
        typer.echo(f"zip      {zip_path}")
    if ac_tracks is not None:
        typer.echo(f"install  {install_track(result.track_dir, ac_tracks, force=force)}")
