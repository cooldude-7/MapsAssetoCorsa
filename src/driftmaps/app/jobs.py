"""Background builds for the local app: one at a time, the rest wait in a queue.

`JobQueue` runs jobs on a single worker thread and keeps their state for `/api/jobs/<id>`.
`BuildRunner` is what a build job does: check the target folders (never replacing a track
drift-maps didn't make), `build_track` into the app's output folder with progress reports,
then copy the track into Assetto Corsa's `content/tracks` (replacing our own older copy).
Failures become plain-English messages (`describe_error`); details go to the console.
"""

from __future__ import annotations

import errno
import itertools
import logging
import queue
import secrets
import threading
import time
import traceback
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from driftmaps.ac.scene import (
    MARKER_FILE,
    TrackFolderError,
    check_track_folder,
    install_track,
    stamp_marker,
)
from driftmaps.ac.track_files import make_track_id

if TYPE_CHECKING:
    from driftmaps.app.settings import Settings
    from driftmaps.geo.router import Route

__all__ = [
    "BuildRequest",
    "BuildRunner",
    "Job",
    "JobError",
    "JobQueue",
    "app_track_id",
    "describe_error",
    "folder_conflict",
    "is_network_error",
]

log = logging.getLogger(__name__)

MAX_KEPT_JOBS = 50  # finished jobs remembered for /api/jobs (oldest dropped first)
MAX_LOG_LINES = 500
NO_AC_MESSAGE = (
    "Couldn't find your Assetto Corsa folder. Set it in Settings (the folder with acs.exe in "
    "it), or turn off 'Install into Assetto Corsa' to just build the track."
)
NETWORK_MESSAGE = "Couldn't download the map or elevation data - check your internet and try again."


class JobError(RuntimeError):
    """A build problem whose message is already plain English for the user."""


def app_track_id(name: str, track_id: str | None = None) -> str:
    """The track folder id: `track_id` (cleaned up) or one made from the name, always with the
    `dm_` prefix, e.g. "Forks of the Credit" -> "dm_forks_of_the_credit"."""
    raw = (track_id or "").strip()
    if not raw:
        return make_track_id(name)
    raw = raw.lower()
    return make_track_id(raw, prefix="" if raw.startswith("dm_") else "dm_")


def folder_conflict(track_id: str, folders: list[Path]) -> str | None:
    """A plain message if a folder `<f>/<track_id>` exists that drift-maps didn't make."""
    for base in folders:
        try:
            check_track_folder(Path(base) / track_id)
        except TrackFolderError:
            return (
                f"There's already a track called {track_id} in {base} that this app didn't "
                "make, so it won't be replaced. Pick a different track name or id."
            )
    return None


def is_network_error(exc: BaseException) -> bool:
    """True for download failures: Overpass, elevation servers, no connection, timeouts."""
    from driftmaps.elevation.sources import SourceUnavailable
    from driftmaps.geo.overpass import OverpassError

    return isinstance(
        exc,
        OverpassError | SourceUnavailable | urllib.error.URLError | TimeoutError | ConnectionError,
    )


def describe_error(exc: BaseException) -> str:
    """A short plain-English message for a failed build."""
    if isinstance(exc, JobError):
        return str(exc)
    if isinstance(exc, TrackFolderError):
        return (
            "A track folder of that id already exists and wasn't made by this app, so it "
            "won't be replaced. Pick a different track name or id."
        )
    if is_network_error(exc):
        return NETWORK_MESSAGE
    if isinstance(exc, PermissionError):
        where = f" ({exc.filename})" if getattr(exc, "filename", None) else ""
        return (
            f"Windows wouldn't let the app write a file{where}. Close Assetto Corsa and "
            "Content Manager (they can lock track files) and try again."
        )
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return "Your disk is full - free up some space and try again."
    if isinstance(exc, MemoryError):
        return "The PC ran out of memory - try a shorter road or fewer trees."
    text = str(exc).strip() or type(exc).__name__
    return f"The build failed: {text}"


# ---------------------------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------------------------


@dataclass
class Job:
    """One queued/running/finished job. Read it with `snapshot()` (thread-safe)."""

    id: str
    request: Any = field(repr=False)
    state: str = "queued"  # queued | running | done | error
    step: str = "Waiting for another build to finish"
    progress: float = 0.0
    log: list[str] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, step: str, progress: float | None = None) -> None:
        """Set the current step (and overall progress 0..1, never going backwards)."""
        with self._lock:
            changed = step != self.step
            self.step = step
            if progress is not None:
                self.progress = max(self.progress, min(1.0, max(0.0, float(progress))))
        if changed:
            self.add_log(step)

    def add_log(self, line: str) -> None:
        with self._lock:
            self.log.append(str(line))
            if len(self.log) > MAX_LOG_LINES:
                del self.log[: len(self.log) - MAX_LOG_LINES]

    def snapshot(self) -> dict[str, object]:
        """The `/api/jobs/<id>` body."""
        with self._lock:
            return {
                "job_id": self.id,
                "state": self.state,
                "step": self.step,
                "progress": round(self.progress, 4),
                "log": list(self.log),
                "result": dict(self.result) if self.result is not None else None,
                "error": self.error,
            }


class _JobLogHandler(logging.Handler):
    """Copies driftmaps warnings logged on the worker thread into the running job's log."""

    def __init__(self, job: Job, thread_id: int) -> None:
        super().__init__(logging.WARNING)
        self.job = job
        self.thread_id = thread_id

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self.thread_id:
            try:
                self.job.add_log(f"note: {record.getMessage()}")
            except Exception:  # never let logging break a build
                pass


class JobQueue:
    """Runs `runner(job) -> result dict` for each submitted job, one at a time, in order."""

    def __init__(self, runner: Callable[[Job], dict]) -> None:
        self.runner = runner
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._counter = itertools.count(1)
        self._thread = threading.Thread(target=self._work, name="drift-maps-build", daemon=True)
        self._thread.start()

    def submit(self, request: object) -> Job:
        job = Job(id=f"{next(self._counter)}-{secrets.token_hex(4)}", request=request)
        with self._lock:
            self._jobs[job.id] = job
            self._prune()
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> list[Job]:
        """Queued and running jobs, oldest first."""
        with self._lock:
            return [j for j in self._jobs.values() if j.state in ("queued", "running")]

    def close(self, timeout: float | None = None) -> None:
        """Stop the worker after the jobs already queued."""
        self._queue.put(None)
        self._thread.join(timeout)

    def _prune(self) -> None:
        done = [j for j in self._jobs.values() if j.state in ("done", "error")]
        for j in sorted(done, key=lambda j: j.created)[: max(0, len(done) - MAX_KEPT_JOBS)]:
            del self._jobs[j.id]

    def _work(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            self._run(job)

    def _run(self, job: Job) -> None:
        with job._lock:
            job.state = "running"
            job.step = "Starting"
        handler = _JobLogHandler(job, threading.get_ident())
        pkg_log = logging.getLogger("driftmaps")
        pkg_log.addHandler(handler)
        try:
            result = self.runner(job)
        except Exception as exc:  # every failure must end the job with a message
            message = describe_error(exc)
            if not isinstance(exc, JobError):
                log.error("build job %s failed:\n%s", job.id, traceback.format_exc())
            job.add_log(f"error: {message}")
            with job._lock:
                job.state = "error"
                job.error = message
                job.step = "Failed"
        else:
            with job._lock:
                job.result = result
                job.state = "done"
                job.progress = 1.0
                job.step = "Done"
            job.add_log("Done")
        finally:
            pkg_log.removeHandler(handler)


# ---------------------------------------------------------------------------------------------
# Builds
# ---------------------------------------------------------------------------------------------


@dataclass
class BuildRequest:
    """What `/api/build` asked for (already validated by the server)."""

    route_id: str
    route: Route
    name: str
    track_id: str
    install: bool = True
    trees: bool = True
    buildings: bool = True
    burlington_buildings: bool = True  # City of Burlington layer (revocable terms, see build)
    guardrails: bool = True
    auto_guardrails: bool = True  # also where Ontario road rules call for one (needs guardrails)
    scenery_density: float = 1.0


BUILD_START, BUILD_END = 0.02, 0.94  # build_track's 0..1 maps onto this part of the bar


def _num(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


class BuildRunner:
    """The work of one build job (see the module docstring).

    `build_fn(route, out_dir, options)` defaults to `driftmaps.build.build_track`, looked up
    at call time (so tests can patch it).
    """

    def __init__(self, settings: Settings, build_fn: Callable[..., Any] | None = None) -> None:
        self.settings = settings
        self.build_fn = build_fn

    def __call__(self, job: Job) -> dict:
        from driftmaps import build as build_mod

        req: BuildRequest = job.request
        job.update(f"Getting ready to build {req.name}", 0.0)
        out_dir = self.settings.output_dir
        ac_tracks: Path | None = None
        if req.install:
            ac_tracks = self.settings.tracks_dir()
            if ac_tracks is None:
                raise JobError(NO_AC_MESSAGE)
        conflict = folder_conflict(
            req.track_id, [out_dir] + ([ac_tracks] if ac_tracks is not None else [])
        )
        if conflict:
            raise JobError(conflict)
        out_dir.mkdir(parents=True, exist_ok=True)

        def progress(step: str, fraction: float) -> None:
            job.update(step, BUILD_START + (BUILD_END - BUILD_START) * _num(fraction))

        options = build_mod.BuildOptions(
            name=req.name,
            track_id=req.track_id,
            cache_dir=self.settings.cache_dir,
            debug_png=False,
            trees=req.trees,
            buildings=req.buildings,
            municipal_buildings=tuple(
                n
                for n in build_mod.BuildOptions().municipal_buildings
                if req.burlington_buildings or n != "burlington"
            ),
            guardrails=req.guardrails,
            auto_guardrails=req.auto_guardrails,
            scenery_density=req.scenery_density,
            progress=progress,
        )
        build_fn = self.build_fn or build_mod.build_track
        result = build_fn(req.route, out_dir, options)
        summary = getattr(result, "summary", None)
        if callable(summary):
            for line in str(summary()).splitlines():
                job.add_log(line)

        track_dir = getattr(result, "track_dir", None)
        if track_dir is not None and (Path(track_dir) / MARKER_FILE).is_file():
            try:  # "My tracks" shows this date (file times can come from a zip)
                stamp_marker(Path(track_dir))
            except OSError as exc:
                log.warning("could not write the build time: %s", exc)
        installed_to = None
        if ac_tracks is not None:
            job.update("Installing into Assetto Corsa", BUILD_END + 0.01)
            installed_to = str(install_track(result.track_dir, ac_tracks))
            job.add_log(f"Installed to {installed_to}")
        return self._result(req, result, installed_to)

    @staticmethod
    def _result(req: BuildRequest, result: Any, installed_to: str | None) -> dict:
        stats = getattr(result, "stats", None) or {}
        scenery = stats.get("scenery") or {}
        track_dir = Path(result.track_dir)
        track_id = getattr(result, "track_id", None) or track_dir.name
        kn5_bytes = stats.get("kn5_bytes")
        if kn5_bytes is None:
            kn5 = track_dir / f"{track_id}.kn5"
            kn5_bytes = kn5.stat().st_size if kn5.is_file() else 0
        return {
            "track_id": track_id,
            "name": req.name,
            "track_dir": str(track_dir),
            "installed_to": installed_to,
            "length_m": round(_num(stats.get("length_m"), req.route.length_m), 1),
            "kn5_mb": round(_num(kn5_bytes) / 1e6, 2),
            "trees": int(_num((scenery.get("trees") or {}).get("total"))),
            "buildings": int(_num((scenery.get("buildings") or {}).get("buildings"))),
            "loop": bool(getattr(req.route, "closed", False)),
        }
