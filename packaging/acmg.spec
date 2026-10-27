# PyInstaller spec for the double-click app "Assetto Corsa Map Generator".
#
#   uv run pyinstaller --noconfirm --clean packaging/acmg.spec
#
# One-folder build in dist/AssettoCorsaMapGenerator/ with AssettoCorsaMapGenerator.exe (on Linux
# the same spec builds dist/AssettoCorsaMapGenerator/AssettoCorsaMapGenerator, used for smoke
# tests). The console window stays: it is how the player sees the app is running and closes it.
#
# What has to be bundled by hand (PyInstaller can't see it from the imports):
# - every driftmaps module: the app imports the builder, router and elevation code lazily;
# - driftmaps/app/static/** (the map page), read through importlib.resources;
# - csp/new-modes/** (the Touge Battle CSP mode the app installs; driftmaps.csp_modes looks in
#   <bundle>/csp/new-modes);
# - rasterio's Cython modules import each other from C, and GDAL/PROJ need their data folders
#   (rasterio/gdal_data, rasterio/proj_data, found next to rasterio's own files); on Windows the
#   GDAL DLLs live in site-packages/rasterio.libs (a delvewheel folder);
# - certifi (imported from rasterio's C code): the CA bundle GDAL uses for HTTPS downloads;
# - pyosmium (the offline road data): compiled submodules and, on Windows, osmium.libs DLLs;
# - driftmaps' metadata, so /api/status reports the real version.
# numpy, scipy, shapely and Pillow come with hooks (PyInstaller, pyinstaller-hooks-contrib).
#
# Check a build with `AssettoCorsaMapGenerator --self-test` (opens a GeoTIFF with rasterio,
# reprojects with PROJ, draws text with FreeType, reads the map page, builds a tiny offline road
# database with pyosmium; see packaging/launcher.py).

import os
import re
import sys
import tomllib
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_delvewheel_libs_directory,
    collect_submodules,
    copy_metadata,
)

APP_NAME = "AssettoCorsaMapGenerator"
APP_TITLE = "Assetto Corsa Map Generator"

HERE = Path(SPECPATH).resolve()  # noqa: F821 - SPECPATH is set by PyInstaller
ROOT = HERE.parent
SRC = ROOT / "src"
STATIC = SRC / "driftmaps" / "app" / "static"
LAUNCHER = HERE / "launcher.py"
ICON = HERE / "icon.ico"

# Modules the app never uses: the developer CLI (pulls in typer/rich) and old research spikes.
SKIP_MODULES = {"driftmaps.cli", "driftmaps.__main__", "driftmaps.spike"}
# rasterio's command line (`rio`) and its matplotlib plotting helpers aren't needed either.
SKIP_RASTERIO = ("rasterio.rio", "rasterio.plot")
SKIP_OSMIUM = ("osmium.replication", "osmium.tools")
EXCLUDES = [
    "tkinter",
    "_tkinter",
    "matplotlib",
    "IPython",
    "pytest",
    "_pytest",
    "typer",
    "rich",
    "PyInstaller",
]


def make_icon(path: Path) -> None:
    """Draw the app icon (a road winding to the horizon, a map pin at its end) as a .ico."""
    from PIL import Image, ImageDraw

    s = 1024  # drawn 4x, then scaled down for smooth edges
    horizon = 600
    img = Image.new("RGBA", (s, s), (30, 41, 51, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((0, horizon, s, s), fill=(56, 142, 60, 255))

    def bezier(t: float) -> tuple[float, float]:
        p = [(470, 1080), (180, 840), (820, 760), (600, horizon)]
        u = 1 - t
        return (
            u**3 * p[0][0] + 3 * u * u * t * p[1][0] + 3 * u * t * t * p[2][0] + t**3 * p[3][0],
            u**3 * p[0][1] + 3 * u * u * t * p[1][1] + 3 * u * t * t * p[2][1] + t**3 * p[3][1],
        )

    def band(t0: float, t1: float, w0: float, w1: float, n: int = 60) -> list:
        """Polygon of a strip along the curve from t0 to t1, width w0..w1 (at t=0..1)."""
        left, right = [], []
        for i in range(n + 1):
            t = t0 + (t1 - t0) * i / n
            x, y = bezier(t)
            (xa, ya), (xb, yb) = bezier(max(0.0, t - 1e-3)), bezier(min(1.0, t + 1e-3))
            dx, dy = xb - xa, yb - ya
            k = (dx * dx + dy * dy) ** 0.5 or 1.0
            half = (w0 + (w1 - w0) * t) / 2
            left.append((x - dy / k * half, y + dx / k * half))
            right.append((x + dy / k * half, y - dx / k * half))
        return left + right[::-1]

    d.polygon(band(0, 1, 380, 44), fill=(240, 240, 240, 255))  # edge lines
    d.polygon(band(0, 1, 330, 30), fill=(62, 68, 78, 255))  # asphalt
    for i in range(6):  # centre dashes, shorter towards the horizon
        t0 = i / 6.4
        d.polygon(band(t0, t0 + 0.07, 26, 4, 8), fill=(255, 202, 40, 255))
    # The map pin, its tip where the road meets the horizon.
    cx, cy, r = 600, 330, 150
    pin = (229, 57, 53, 255)
    d.polygon(
        [(cx - r * 0.86, cy + r * 0.5), (cx + r * 0.86, cy + r * 0.5), (cx, horizon - 20)], fill=pin
    )
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=pin)
    d.ellipse(
        (cx - r * 0.42, cy - r * 0.42, cx + r * 0.42, cy + r * 0.42), fill=(255, 255, 255, 255)
    )
    # Rounded-square tile.
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle((24, 24, s - 24, s - 24), radius=200, fill=255)
    tile = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    tile.paste(img, (0, 0), mask)
    big = tile.resize((256, 256), Image.Resampling.LANCZOS)
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    path.parent.mkdir(parents=True, exist_ok=True)
    big.save(path, format="ICO", sizes=sizes)


def _version() -> str:
    """$ACMG_VERSION (CI: the release tag's X.Y.Z), else pyproject.toml's version."""
    env = os.environ.get("ACMG_VERSION", "").strip().lstrip("v")
    if re.fullmatch(r"\d+(\.\d+){1,3}", env):
        return env
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def _windows_version_info(version: str):
    """File properties shown by Explorer (Details tab) for the exe."""
    from PyInstaller.utils.win32 import versioninfo as vi

    nums = tuple((list(int(p) for p in version.split(".")[:3] if p.isdigit()) + [0, 0, 0, 0])[:4])
    strings = [
        vi.StringStruct("CompanyName", "driftmaps"),
        vi.StringStruct("FileDescription", APP_TITLE),
        vi.StringStruct("FileVersion", version),
        vi.StringStruct("InternalName", APP_NAME),
        vi.StringStruct("OriginalFilename", APP_NAME + ".exe"),
        vi.StringStruct("ProductName", APP_TITLE),
        vi.StringStruct("ProductVersion", version),
    ]
    return vi.VSVersionInfo(
        ffi=vi.FixedFileInfo(filevers=nums, prodvers=nums),
        kids=[
            vi.StringFileInfo([vi.StringTable("040904B0", strings)]),
            vi.VarFileInfo([vi.VarStruct("Translation", [1033, 1200])]),
        ],
    )


if not ICON.is_file():
    make_icon(ICON)

hiddenimports = collect_submodules("driftmaps", filter=lambda name: name not in SKIP_MODULES)
hiddenimports += collect_submodules(
    "rasterio", filter=lambda name: not name.startswith(SKIP_RASTERIO)
)
# rasterio's compiled _env module imports certifi to hand GDAL's curl a CA bundle; without it
# the elevation downloads (/vsicurl/ over HTTPS) can't verify certificates on a player's PC.
hiddenimports += ["certifi"]
# pyosmium (reads the offline road data's .osm.pbf): its compiled modules import each other from
# C. Its replication/tools helpers (they pull in requests) are never used.
hiddenimports += collect_submodules(
    "osmium", filter=lambda name: not name.startswith(SKIP_OSMIUM)
)

datas = [(str(STATIC), "driftmaps/app/static")]
datas += [(str(ROOT / "csp" / "new-modes"), "csp/new-modes")]
datas += collect_data_files(
    "rasterio",
    excludes=["**/*.pyx", "**/*.pxd", "**/*.pxi", "**/*.h", "rio/**"],
)
datas += copy_metadata("driftmaps")
binaries = []
# Windows wheels keep GDAL & co. in site-packages/rasterio.libs, and pyosmium's zlib/bz2/expat/
# lz4/MSVC runtime in site-packages/osmium.libs (delvewheel folders; no-op elsewhere).
datas, binaries = collect_delvewheel_libs_directory("rasterio", datas=datas, binaries=binaries)
datas, binaries = collect_delvewheel_libs_directory("osmium", datas=datas, binaries=binaries)

a = Analysis(  # noqa: F821 - PyInstaller spec globals
    [str(LAUNCHER)],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    # UTF-8 mode, as on Linux where the code is developed: files written without an explicit
    # encoding (road names like "Chemin de la Côte") come out UTF-8, not the Windows code page.
    [("X utf8", None, "OPTION")],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX-packed exes trip antivirus scanners far more often
    console=True,  # keep the console window: "keep this window open, close it to quit"
    disable_windowed_traceback=False,
    icon=[str(ICON)],
    version=_windows_version_info(_version()) if sys.platform == "win32" else None,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)
