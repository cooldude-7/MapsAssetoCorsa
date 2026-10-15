"""Double-click entry point of the packaged app (AssettoCorsaMapGenerator.exe).

Starts the local app (`driftmaps.app.run()`): the map page opens in the web browser and this
console window stays open while the app runs; closing the window quits it.

    AssettoCorsaMapGenerator [--port N] [--no-browser] [--data-dir DIR]
    AssettoCorsaMapGenerator --self-test [--network]

`--self-test` checks that the build has everything it needs (every driftmaps module, the map
page, numpy/scipy/shapely, Pillow with FreeType, rasterio opening a GeoTIFF and reprojecting with
PROJ's data) and exits 0 or 1; `--network` also reads a real elevation tile and OpenStreetMap
over HTTPS. Built by packaging/acmg.spec; also runs unfrozen: `uv run python packaging/launcher.py`.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import traceback
from collections.abc import Callable

APP_TITLE = "Assetto Corsa Map Generator"
BANNER = "Assetto Corsa Map Generator is running - keep this window open. Close it to quit."
# Modules the packaged app leaves out (see SKIP_MODULES in acmg.spec).
SKIPPED_MODULES = {"driftmaps.cli", "driftmaps.__main__", "driftmaps.spike"}


def frozen() -> bool:
    """True inside the PyInstaller build."""
    return bool(getattr(sys, "frozen", False))


def _say(text: str = "") -> None:
    if sys.stdout is None:
        return
    try:
        print(text, flush=True)
    except (OSError, ValueError):
        pass


def _console_setup() -> None:
    """Make the console behave for a long-running app: never crash printing a road name the
    console code page can't show; on Windows, set the window title and turn off QuickEdit, so a
    stray click in the window doesn't freeze the app until a key is pressed."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetConsoleTitleW(APP_TITLE)
        handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = wintypes.DWORD()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            enable_quick_edit, enable_extended_flags = 0x0040, 0x0080
            kernel32.SetConsoleMode(
                handle, (mode.value & ~enable_quick_edit) | enable_extended_flags
            )
    except Exception:  # cosmetic only
        pass


def _interactive_console() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (OSError, ValueError):
        return False


def _pause(seconds: float | None = None) -> None:
    """Keep a double-clicked window open long enough to read it (no-op when not interactive)."""
    if not (frozen() and _interactive_console()):
        return
    if seconds is not None:
        time.sleep(seconds)
        return
    try:
        input("Press Enter to close this window.")
    except (EOFError, OSError, KeyboardInterrupt):
        pass


def _already_running() -> str | None:
    try:
        from driftmaps.app.server import DEFAULT_PORT, running_instance

        return running_instance(DEFAULT_PORT)
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------------------------


def _check_modules() -> str:
    import importlib
    import pkgutil

    import driftmaps

    names = sorted(
        m.name
        for m in pkgutil.walk_packages(driftmaps.__path__, "driftmaps.")
        if m.name not in SKIPPED_MODULES
    )
    for name in names:
        importlib.import_module(name)
    for name in ("driftmaps.build", "driftmaps.geo.router", "driftmaps.app.server"):
        importlib.import_module(name)
    return f"{len(names)} driftmaps modules"


def _check_static() -> str:
    from driftmaps.app.server import read_static
    from driftmaps.app.settings import app_version

    page = read_static("/")
    if page is None or b"<html" not in page[0].lower():
        raise RuntimeError("the map page (index.html) is missing")
    for name in ("app.js", "style.css"):
        if read_static("/" + name) is None:
            raise RuntimeError(f"{name} is missing")
    return f"map page ok, version {app_version()}"


def _check_science() -> str:
    import numpy as np
    import shapely
    from scipy import ndimage, sparse
    from scipy.linalg import cho_solve_banded, cholesky_banded, solveh_banded
    from scipy.spatial import Delaunay, cKDTree

    rng = np.random.default_rng(1)
    pts = rng.random((200, 2))
    d, _ = cKDTree(pts).query([0.5, 0.5])
    Delaunay(pts)
    ndimage.gaussian_filter(rng.random((32, 32)), 2.0)
    sparse.csr_matrix(np.eye(4))
    ab = np.array([[0.0, -1, -1, -1], [4.0, 4, 4, 4]])
    x = solveh_banded(ab, np.ones(4))
    cho_solve_banded((cholesky_banded(ab), False), np.ones(4))
    line = shapely.LineString(pts[:20]).simplify(0.01).buffer(0.1)
    if not (np.isfinite(d) and np.all(np.isfinite(x)) and line.area > 0):
        raise RuntimeError("numeric check gave nonsense")
    return f"numpy {np.__version__}, shapely {shapely.__version__}"


def _check_pillow() -> str:
    import io

    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=20)
    if not isinstance(font, ImageFont.FreeTypeFont):
        raise RuntimeError("Pillow has no FreeType (text on signs and maps would be tiny)")
    img = Image.new("RGBA", (64, 32), (0, 0, 0, 255))
    ImageDraw.Draw(img).text((2, 2), "AC", font=font, fill=(255, 255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=6)  # the builder writes PNG textures
    buf.seek(0)
    Image.open(buf).load()
    return "Pillow with FreeType and PNG"


def _check_rasterio() -> str:
    import numpy as np
    import rasterio
    from rasterio.crs import CRS
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import Window

    data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    profile = dict(
        driver="GTiff",
        width=64,
        height=64,
        count=1,
        dtype="float32",
        nodata=-32767.0,
        crs=CRS.from_epsg(3979),
        transform=from_origin(1_300_000, -300_000, 30, 30),
        tiled=True,
        blockxsize=32,
        blockysize=32,
        compress="deflate",
    )
    with tempfile.TemporaryDirectory(prefix="acmg-selftest-") as tmp:
        path = os.path.join(tmp, "test.tif")
        with rasterio.open(path, "w", **profile) as ds:
            ds.write(data, 1)
        with rasterio.open(path) as ds:
            back = ds.read(1, window=Window(8, 8, 4, 4))
            epsg = ds.crs.to_epsg()
    if epsg != 3979 or not np.array_equal(back, data[8:12, 8:12]):
        raise RuntimeError("a GeoTIFF didn't read back the same")
    with MemoryFile() as mf:
        with mf.open(**profile) as ds:
            ds.write(data, 1)
        with mf.open() as ds:
            ds.read(1)
    # Toronto, WGS84 -> Canada Atlas Lambert and NAD83(CSRS) UTM 17N: needs PROJ's proj.db.
    for code in (3979, 2958):
        xs, ys = warp_transform(CRS.from_epsg(4326), CRS.from_epsg(code), [-79.38], [43.65])
        if not (np.isfinite(xs[0]) and np.isfinite(ys[0]) and abs(xs[0]) > 1000):
            raise RuntimeError(f"reprojecting to EPSG:{code} failed")
    return f"rasterio {rasterio.__version__}, GDAL {rasterio.__gdal_version__}"


def _check_https_python() -> str:
    import urllib.request

    req = urllib.request.Request(
        "https://overpass-api.de/api/status", headers={"User-Agent": "AssettoCorsaMapGenerator"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read(200)
    return "OpenStreetMap (Overpass) reachable"


def _check_https_gdal() -> str:
    import rasterio
    from rasterio.windows import Window

    from driftmaps.elevation import sources

    url = "https://canelevation-dem.s3.ca-central-1.amazonaws.com/mrdem-30/mrdem-30-dtm.tif"
    with sources._gdal_env(), rasterio.open("/vsicurl/" + url) as ds:
        ds.read(1, window=Window(ds.width // 2, ds.height // 2, 1, 1))
    return "Canadian elevation tiles reachable"


def self_test(network: bool = False) -> int:
    """Run the build checks; print one line each. 0 if all passed."""
    checks: list[tuple[str, Callable[[], str]]] = [
        ("modules", _check_modules),
        ("map page", _check_static),
        ("numpy/scipy/shapely", _check_science),
        ("Pillow", _check_pillow),
        ("rasterio/GDAL/PROJ", _check_rasterio),
    ]
    if network:
        checks.append(("HTTPS (Python)", _check_https_python))
        checks.append(("HTTPS (GDAL)", _check_https_gdal))
    _say(
        f"{APP_TITLE} self-test (Python {sys.version.split()[0]}, frozen={frozen()}, "
        f"UTF-8 mode={bool(sys.flags.utf8_mode)})"
    )
    failed = 0
    for label, fn in checks:
        try:
            _say(f"[ ok ] {label}: {fn()}")
        except Exception as exc:
            failed += 1
            _say(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    _say("Self-test passed." if not failed else f"Self-test FAILED ({failed} problem(s)).")
    return 1 if failed else 0


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="AssettoCorsaMapGenerator",
        description=f"{APP_TITLE}: pick a road on a map, get an Assetto Corsa track.",
    )
    parser.add_argument("--port", type=int, default=None, help="Port on 127.0.0.1.")
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't open the page in the web browser."
    )
    parser.add_argument("--data-dir", default=None, help="App data folder (settings, cache).")
    parser.add_argument("--self-test", action="store_true", help="Check the build and exit.")
    parser.add_argument(
        "--network", action="store_true", help="With --self-test: also check internet access."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _console_setup()
    if args.self_test:
        return self_test(network=args.network)

    already = _already_running() if args.port is None else None
    if already:
        if not args.no_browser:
            _say(f"{APP_TITLE} is already open - showing it in your web browser again.")
    else:
        _say("=" * len(BANNER))
        _say(BANNER)
        _say("=" * len(BANNER))
        _say()
        if not args.no_browser:
            _say("Your web browser will open the map page in a moment.")
            _say()
    try:
        from driftmaps.app import run

        run(port=args.port, open_browser=not args.no_browser, data_dir=args.data_dir)
    except KeyboardInterrupt:
        return 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if code:
            _pause()
        return code
    except Exception:
        traceback.print_exc()
        _say()
        _say(f"Sorry - {APP_TITLE} hit a problem and has to stop (details above).")
        _pause()
        return 1
    if already:
        _pause(4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
