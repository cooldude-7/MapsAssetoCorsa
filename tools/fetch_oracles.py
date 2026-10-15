"""Download dev-only test oracles into tools/oracles/ (gitignored).

An oracle is an independent, third-party parser that our tests use to cross-check what our own
writers produce. They are GPL-licensed, so they are:

- downloaded on demand (never vendored or committed: tools/oracles/ is in .gitignore),
- loaded only by tests (e.g. tests/test_kn5.py), never imported by the driftmaps package,
- pinned to an exact commit and SHA-256, so a changed upstream file can't silently change tests.

Usage:
    uv run python tools/fetch_oracles.py            # fetch what is missing
    uv run python tools/fetch_oracles.py --force    # re-download everything

Standard library only. Honours HTTPS_PROXY and SSL_CERT_FILE like any urllib client.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT / "tools" / "oracles"


@dataclass(frozen=True)
class OracleFile:
    """One file to download, pinned by URL (with a commit hash) and SHA-256."""

    url: str
    dest_name: str
    sha256: str


@dataclass(frozen=True)
class Oracle:
    name: str
    description: str
    licence: str
    homepage: str
    files: tuple[OracleFile, ...]


_NAZARIYKUL_COMMIT = "502ab4861dd3f324a794db5874c0f8237dbeaa79"
_NAZARIYKUL_RAW = (
    f"https://raw.githubusercontent.com/nazariyKul/kn5-blender-tools/{_NAZARIYKUL_COMMIT}"
)

ORACLES: tuple[Oracle, ...] = (
    Oracle(
        name="nazariykul_kn5",
        description=(
            "Pure-stdlib KN5 parser from nazariyKul/kn5-blender-tools (kn5_blender_tools/kn5.py). "
            "Strict bounds checks; reports trailing bytes. Used by tests/test_kn5.py."
        ),
        licence="GPL-3.0-or-later",
        homepage="https://github.com/nazariyKul/kn5-blender-tools",
        files=(
            OracleFile(
                url=f"{_NAZARIYKUL_RAW}/kn5_blender_tools/kn5.py",
                dest_name="nazariykul_kn5.py",
                sha256="9dd5df5a8ed6e3f7f6baa8bb90660a247bbb4aa21f971dfb7f9b66b3048d1316",
            ),
            OracleFile(
                url=f"{_NAZARIYKUL_RAW}/LICENSE",
                dest_name="nazariykul_LICENSE.txt",
                sha256="4c559afeaaa19f1c941c9ca9ca406757c9ed726902e43cf9b39db81ac31338b7",
            ),
        ),
    ),
)


def _download(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "driftmaps-fetch-oracles"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(dest: Path = DEFAULT_DEST, *, force: bool = False, timeout: float = 30.0) -> list[Path]:
    """Download every oracle file into `dest`. Returns the paths now present.

    Raises RuntimeError if a download fails or a file's SHA-256 does not match its pin.
    """
    dest.mkdir(parents=True, exist_ok=True)
    present: list[Path] = []
    for oracle in ORACLES:
        for item in oracle.files:
            target = dest / item.dest_name
            if target.exists() and not force and _sha256(target.read_bytes()) == item.sha256:
                present.append(target)
                continue
            try:
                data = _download(item.url, timeout)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise RuntimeError(f"could not download {item.url}: {exc}") from exc
            digest = _sha256(data)
            if digest != item.sha256:
                raise RuntimeError(
                    f"SHA-256 mismatch for {item.url}: got {digest}, pinned {item.sha256}"
                )
            target.write_bytes(data)
            present.append(target)
    _write_readme(dest)
    return present


def _write_readme(dest: Path) -> None:
    lines = [
        "Dev-only test oracles, downloaded by tools/fetch_oracles.py. DO NOT COMMIT.",
        "These are third-party files under their own licences. They are used only by tests,",
        "never imported by or shipped with the driftmaps package.",
        "",
    ]
    for oracle in ORACLES:
        lines += [
            f"{oracle.name}: {oracle.description}",
            f"  licence:  {oracle.licence}",
            f"  homepage: {oracle.homepage}",
        ]
        lines += [f"  {f.dest_name} <- {f.url}" for f in oracle.files]
        lines.append("")
    (dest / "README.txt").write_text("\n".join(lines), encoding="ascii")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="target folder")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request timeout (s)")
    args = parser.parse_args(argv)
    try:
        paths = fetch(args.dest, force=args.force, timeout=args.timeout)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        print(f"ok  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
