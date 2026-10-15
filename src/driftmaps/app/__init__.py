"""The local "Assetto Corsa Map Generator" app: a map page served on 127.0.0.1 that routes,
builds and installs tracks on this PC. Start it with `python -m driftmaps.app` or `run()`."""

from __future__ import annotations

from pathlib import Path

__all__ = ["run"]


def run(
    port: int | None = None,
    open_browser: bool = True,
    data_dir: Path | str | None = None,
) -> None:
    """Serve the app until Ctrl+C (see `driftmaps.app.server.run`)."""
    from driftmaps.app.server import run as _run

    _run(port=port, open_browser=open_browser, data_dir=data_dir)
