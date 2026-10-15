"""Command-line entry point: `drift-maps ...` (or `python -m driftmaps ...`)."""

from __future__ import annotations

from pathlib import Path

import typer

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
