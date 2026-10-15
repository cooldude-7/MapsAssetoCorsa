"""`python -m driftmaps.app [--port N] [--no-browser] [--data-dir DIR]`: start the local app."""

from __future__ import annotations

import argparse

from driftmaps.app import run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m driftmaps.app",
        description="Assetto Corsa Map Generator: pick a road on a map, get an AC track.",
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Port on 127.0.0.1 (default: 8765 or the next free)."
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't open the page in the web browser."
    )
    parser.add_argument(
        "--data-dir", default=None, help="App data folder (default: the per-user app folder)."
    )
    args = parser.parse_args(argv)
    run(port=args.port, open_browser=not args.no_browser, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
