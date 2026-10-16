"""Offline roads: download a region's OSM extract once and turn it into the region database
(`driftmaps.geo.offline`), as a background job (the app) or in the foreground (the CLI).

One job at a time: download (HTTP Range resume, so a cancelled or broken download continues
where it stopped; checked against the server's .md5 when it has one), then processing
(`build_db`), then the .pbf is deleted. A running job can be cancelled. The old database keeps
serving until the new one replaces it.

Sources: Geofabrik (https://download.geofabrik.de, the extract the app was designed around) and,
if Geofabrik can't be reached, the same extract from openstreetmap.fr. Both are OSM data, (c)
OpenStreetMap contributors, ODbL 1.0. Ontario was 1075 MB on openstreetmap.fr on 2026-10-15
(Geofabrik's download host reset connections from the dev sandbox, so its size is not checked).

Files in the offline folder (`<app data>/offline`): `<region>.sqlite` (the database),
`<region>.poly` (the extract's boundary), `<region>.osm.pbf.part` + `.part.json` (a download in
progress), `<region>.osm.pbf` (a finished download, until processed).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from driftmaps.geo import offline as region_db

__all__ = [
    "DEFAULT_REGION",
    "REGIONS",
    "Cancelled",
    "DownloadError",
    "OfflineManager",
    "Region",
    "Source",
    "download_file",
]

log = logging.getLogger(__name__)

USER_AGENT = "AssettoCorsaMapGenerator (offline road data; Python urllib)"
CHUNK = 1 << 20
TIMEOUT_S = 60.0
RETRIES = 5  # broken downloads resumed this many times in a row
# Free space needed, x the download: the .pbf, the node index while processing (~2x the .pbf)
# and the database (~0.8x).
DISK_FACTOR = 4.0


@dataclass(frozen=True)
class Source:
    name: str
    pbf_url: str
    poly_url: str


@dataclass(frozen=True)
class Region:
    key: str
    name: str
    approx_mb: int  # shown before the download starts
    bbox: tuple[float, float, float, float]  # (south, west, north, east), for the hint only
    sources: tuple[Source, ...]


REGIONS: dict[str, Region] = {
    "ontario": Region(
        "ontario",
        "Ontario",
        1075,
        (41.6, -95.2, 56.9, -74.3),
        (
            Source(
                "Geofabrik",
                "https://download.geofabrik.de/north-america/canada/ontario-latest.osm.pbf",
                "https://download.geofabrik.de/north-america/canada/ontario.poly",
            ),
            Source(
                "openstreetmap.fr",
                "https://download.openstreetmap.fr/extracts/north-america/canada/"
                "ontario-latest.osm.pbf",
                "https://download.openstreetmap.fr/polygons/north-america/canada/ontario.poly",
            ),
        ),
    )
}
DEFAULT_REGION = "ontario"


class Cancelled(Exception):
    """The job was cancelled."""


class DownloadError(RuntimeError):
    """The download failed (plain-English message)."""


Opener = Callable[[urllib.request.Request, float], object]  # -> an http.client.HTTPResponse


def _urlopen(request: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(request, timeout=timeout)


def _get(url: str, opener: Opener, headers: dict[str, str] | None = None):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return opener(req, TIMEOUT_S)


def fetch_small(url: str, opener: Opener = _urlopen, limit: int = 4_000_000) -> bytes:
    with _get(url, opener) as resp:
        return resp.read(limit)


def download_file(
    url: str,
    dest: Path,
    *,
    progress: Callable[[int, int | None], None] | None = None,
    cancel: threading.Event | None = None,
    opener: Opener = _urlopen,
    retries: int = RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download `url` to `dest`, resuming `dest.part` if an earlier try of the same URL left
    one (HTTP Range + If-Range on the ETag / Last-Modified, so a changed file starts over).
    `progress(done_bytes, total_bytes or None)`; `cancel` keeps the part file for next time."""
    part = dest.with_name(dest.name + ".part")
    meta_path = dest.with_name(dest.name + ".part.json")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    if meta.get("url") != url or not part.is_file():
        part.unlink(missing_ok=True)
        meta = {"url": url}
    failures = 0
    while True:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        have = part.stat().st_size if part.is_file() else 0
        headers = {}
        if have and meta.get("validator"):
            headers = {"Range": f"bytes={have}-", "If-Range": meta["validator"]}
        try:
            with _get(url, opener, headers) as resp:
                status = getattr(resp, "status", 200)
                if status == 206:
                    start = have
                    total = meta.get("total")
                else:  # the whole file (no resume, or the file changed)
                    start = 0
                    length = resp.headers.get("Content-Length")
                    total = int(length) if length and length.isdigit() else None
                    validator = resp.headers.get("ETag") or resp.headers.get("Last-Modified")
                    meta = {"url": url, "validator": validator, "total": total}
                    meta_path.write_text(json.dumps(meta), encoding="utf-8")
                done = start
                with open(part, "r+b" if start else "wb") as fh:
                    fh.seek(start)
                    fh.truncate()
                    while True:
                        if cancel is not None and cancel.is_set():
                            raise Cancelled()
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        failures = 0
                        if progress is not None:
                            progress(done, total)
            if total is not None and done < total:
                raise DownloadError(f"the download stopped at {done} of {total} bytes")
            break
        except Cancelled:
            raise
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and have:  # asked past the end: start over
                part.unlink(missing_ok=True)
                continue
            if exc.code < 500 and exc.code != 429:
                raise DownloadError(f"{url}: HTTP {exc.code}") from None
            failures += 1
            err: Exception = exc
        except (OSError, DownloadError) as exc:  # URLError, timeouts, resets, short reads
            failures += 1
            err = exc
        if failures > retries:
            raise DownloadError(f"{url}: {err}") from None
        log.warning("download of %s broke (%s); resuming", url, err)
        sleep(min(30.0, 2.0 * failures))
    part.replace(dest)
    meta_path.unlink(missing_ok=True)
    return dest


def _md5(path: Path, cancel: threading.Event | None) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK * 8):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            h.update(chunk)
    return h.hexdigest()


def _date(stamp: str | None) -> str | None:
    """ "2026-10-15T00:15:47Z" -> "2026-10-15"."""
    return stamp[:10] if stamp and len(stamp) >= 10 else None


class OfflineManager:
    """The offline road data in `directory`: status, and one download/processing job."""

    def __init__(
        self,
        directory: Path | str,
        *,
        opener: Opener = _urlopen,
        sources: dict[str, tuple[Source, ...]] | None = None,
    ) -> None:
        self.dir = Path(directory)
        self.opener = opener
        self.sources = sources or {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._job: dict | None = None  # {region, state, progress, message, download_mb}
        self._last: dict | None = None  # how the last job ended: {region, state, message}

    # -- status ---------------------------------------------------------------------------------

    def _db(self, region: str) -> region_db.OfflineDB | None:
        path = region_db.db_path(region, self.dir)
        for db in region_db.databases(self.dir):
            if db.path == path:
                return db
        return None

    def status(self, region: str = DEFAULT_REGION) -> dict:
        info = REGIONS[region]
        out = {
            "region": region,
            "region_name": info.name,
            "state": "none",
            "progress": 0.0,
            "message": "",
            "size_mb": None,
            "download_mb": info.approx_mb,
            "updated": None,
            "built": None,
        }
        db = self._db(region)
        if db is not None:
            out.update(
                state="ready",
                size_mb=round(db.path.stat().st_size / 1e6, 1),
                updated=_date(db.osm_base) or _date(db.built),
                built=db.built,
                counts=db.counts,
            )
        with self._lock:
            job = dict(self._job) if self._job else None
            last = dict(self._last) if self._last else None
        if job is not None and job["region"] == region:
            out.update(
                state=job["state"],
                progress=round(job["progress"], 3),
                message=job["message"],
                download_mb=job.get("download_mb") or info.approx_mb,
                ready=db is not None,  # the old data keeps working during an update
            )
        elif last is not None and last["region"] == region:
            if last["state"] == "error":
                out.update(state="error", message=last["message"], ready=db is not None)
            else:
                out["message"] = last["message"]
        if out["state"] == "none" and (self.dir / f"{region}.osm.pbf.part").is_file():
            out["message"] = out["message"] or "A download was interrupted; it resumes from there."
        return out

    def busy(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def hint(self, lat: float, lon: float) -> str | None:
        """The region worth downloading for a slow road load at (lat, lon), if not ready."""
        for key, info in REGIONS.items():
            s_, w, n, e = info.bbox
            if s_ <= lat <= n and w <= lon <= e and self._db(key) is None:
                return key
        return None

    # -- the job ----------------------------------------------------------------------------------

    def _set(self, **kw) -> None:
        with self._lock:
            if self._job is not None:
                self._job.update(kw)

    def start(self, region: str = DEFAULT_REGION) -> dict:
        """Start downloading + processing `region` in the background. RuntimeError if a job
        is already running."""
        if region not in REGIONS:
            raise KeyError(region)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("busy")
            self._cancel = threading.Event()
            self._job = {"region": region, "state": "downloading", "progress": 0.0,
                         "message": "", "download_mb": None}  # fmt: skip
            self._last = None
            self._thread = threading.Thread(
                target=self._run_job, args=(region,), name="offline-roads", daemon=True
            )
            self._thread.start()
        return self.status(region)

    def _run_job(self, region: str) -> None:
        try:
            self.run(region, cancel=self._cancel)
            ended = {"region": region, "state": "ready", "message": ""}
        except Cancelled:
            ended = {"region": region, "state": "none", "message": "Cancelled."}
        except Exception as exc:  # noqa: BLE001 - shown to the player
            log.exception("offline road data job failed")
            ended = {"region": region, "state": "error", "message": _plain(exc)}
        with self._lock:
            self._last = ended
            self._job = None

    def cancel(self) -> bool:
        """Ask the running job to stop; True if there was one."""
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            if running:
                self._cancel.set()
                if self._job is not None:
                    self._job["message"] = "Cancelling"
        return running

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the job to end (tests, CLI); True if none is running."""
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)
            return not thread.is_alive()
        return True

    def delete(self, region: str = DEFAULT_REGION) -> dict:
        """Remove the region's database and any download. RuntimeError while a job runs."""
        if region not in REGIONS:
            raise KeyError(region)
        if self.busy():
            raise RuntimeError("busy")
        base = region_db.db_path(region, self.dir)
        names = [
            base,
            base.with_name(base.name + ".building"),
            base.with_name(base.name + ".nodes"),
        ]
        names += [self.dir / f"{region}{s}" for s in
                  (".poly", ".osm.pbf", ".osm.pbf.part", ".osm.pbf.part.json")]  # fmt: skip
        with region_db.exclusive():
            region_db.release(base)
            for p in names:
                p.unlink(missing_ok=True)
        with self._lock:
            self._last = None
        return self.status(region)

    def run(
        self,
        region: str = DEFAULT_REGION,
        *,
        cancel: threading.Event | None = None,
        progress: Callable[[str, float, str], None] | None = None,
        source_file: Path | None = None,
        poly_file: Path | None = None,
    ) -> region_db.BuildStats:
        """Download (unless `source_file` is given) and process `region` in this thread.
        `progress(state, fraction, text)`. Raises Cancelled / DownloadError / others."""
        info = REGIONS[region]

        def report(state: str, fraction: float, text: str) -> None:
            self._set(state=state, progress=fraction, message=text)
            if progress is not None:
                progress(state, fraction, text)

        self.dir.mkdir(parents=True, exist_ok=True)
        pbf = self.dir / f"{region}.osm.pbf"
        poly_path = self.dir / f"{region}.poly"
        source_name = ""
        if source_file is None:
            free = shutil.disk_usage(self.dir).free
            need = info.approx_mb * 1e6 * DISK_FACTOR
            if free < need and not pbf.is_file():
                raise DownloadError(
                    f"Not enough free disk space: the {info.name} roads need about "
                    f"{need / 1e9:.1f} GB while downloading (only {free / 1e9:.1f} GB free)."
                )
            source_name = self._download(region, pbf, poly_path, cancel, report)
            source_file = pbf
        elif poly_file is not None:
            shutil.copyfile(poly_file, poly_path)
        poly = poly_path.read_text(encoding="utf-8") if poly_path.is_file() else None
        report("processing", 0.0, "Processing roads")
        stats = region_db.build_db(
            source_file,
            region_db.db_path(region, self.dir),
            region=region,
            poly=poly,
            source_name=source_name or Path(source_file).name,
            progress=lambda f, text: report("processing", f, text),
            cancel=cancel,
        )
        if source_file == pbf:
            pbf.unlink(missing_ok=True)  # 1 GB nobody needs any more
        return stats

    def _download(self, region, pbf, poly_path, cancel, report) -> str:
        """Fetch the boundary and the extract from the first source that answers."""
        errors = []
        for src in self.sources.get(region) or REGIONS[region].sources:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            try:
                if not pbf.is_file():
                    poly_text = fetch_small(src.poly_url, self.opener).decode("utf-8")
                    region_db.parse_poly(poly_text)  # a real boundary, not an error page
                    poly_path.write_text(poly_text, encoding="utf-8")
                    t0 = time.monotonic()

                    def step(done: int, total: int | None, t0=t0) -> None:
                        mb = round(total / 1e6) if total else None
                        self._set(download_mb=mb)
                        frac = done / total if total else 0.0
                        rate = done / max(time.monotonic() - t0, 1e-3) / 1e6
                        of = f" of {mb} MB" if mb else " MB"
                        report("downloading", frac, f"{done / 1e6:.0f}{of} · {rate:.1f} MB/s")

                    download_file(src.pbf_url, pbf, progress=step, cancel=cancel,
                                  opener=self.opener)  # fmt: skip
                    self._verify(src, pbf, cancel, report)
                return src.name
            except (Cancelled, KeyboardInterrupt):
                raise
            except Exception as exc:  # noqa: BLE001 - try the next source
                log.warning("offline download from %s failed: %s", src.name, exc)
                errors.append(f"{src.name}: {exc}")
        raise DownloadError(
            "Couldn't download the road data - check your internet and try again. ("
            + "; ".join(errors)
            + ")"
        )

    def _verify(self, src: Source, pbf: Path, cancel, report) -> None:
        """Compare with the server's .md5, when it has one (a broken file would fail later)."""
        try:
            want = fetch_small(src.pbf_url + ".md5", self.opener, 4096).decode("ascii").split()
        except Exception:  # noqa: BLE001 - optional
            return
        if not want or len(want[0]) != 32:
            return
        report("downloading", 1.0, "checking the file")
        if _md5(pbf, cancel) != want[0].lower():
            pbf.unlink(missing_ok=True)
            raise DownloadError("the downloaded file is damaged (MD5 mismatch); download again")


def _plain(exc: Exception) -> str:
    if isinstance(exc, DownloadError):
        return str(exc)
    if isinstance(exc, OSError):
        return f"Couldn't write the road data: {exc}"
    return f"Processing the road data failed: {exc}"
