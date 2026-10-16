"""Minimal headless Chromium driver over the DevTools protocol, standard library only.

Chromium is started with `--remote-debugging-pipe`: it reads JSON commands from fd 3 and writes
replies and events to fd 4, each message terminated by a NUL byte. That avoids a websocket client
and any port juggling. POSIX only (it relies on fd inheritance).
"""

from __future__ import annotations

import fcntl
import glob
import json
import os
import re
import select
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


class ChromiumError(RuntimeError):
    pass


def _version_key(path: str) -> int:
    m = re.search(r"-(\d+)", path)
    return int(m.group(1)) if m else 0


def find_chromium() -> str:
    """Locate a Chromium binary: $KN5VIEW_CHROME, Playwright's browsers, then the PATH."""
    env = os.environ.get("KN5VIEW_CHROME")
    if env:
        return env
    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH"), str(Path.home() / ".cache/ms-playwright")]
    patterns = [
        "chromium_headless_shell-*/chrome-linux/headless_shell",
        "chromium_headless_shell-*/chrome-*/headless_shell",
        "chromium-*/chrome-linux/chrome",
        "chromium-*/chrome-*/chrome",
    ]
    for root in roots:
        if not root:
            continue
        for pattern in patterns:
            hits = sorted(glob.glob(os.path.join(root, pattern)), key=_version_key, reverse=True)
            if hits:
                return hits[0]
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        found = shutil.which(name)
        if found:
            return found
    raise ChromiumError(
        "no Chromium found: set KN5VIEW_CHROME or PLAYWRIGHT_BROWSERS_PATH "
        "(e.g. /opt/pw-browsers), or install one with `npx playwright install chromium`"
    )


class Chromium:
    """One headless Chromium process and one page (tab) session."""

    def __init__(self, exe: str | None = None, *, verbose: bool = False) -> None:
        self.exe = exe or find_chromium()
        self.verbose = verbose
        self.console: list[str] = []
        self._id = 0
        self._buf = b""
        self._profile = tempfile.mkdtemp(prefix="kn5view-chrome-")
        self._log = tempfile.TemporaryFile()
        # Our ends get CLOEXEC; the child dup2()s its ends onto fds 3 and 4 (inheritable).
        cmd_r, cmd_w = os.pipe()
        evt_r, evt_w = os.pipe()
        hi = [fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 10) for fd in (cmd_r, cmd_w, evt_r, evt_w)]
        for fd in (cmd_r, cmd_w, evt_r, evt_w):
            os.close(fd)
        child_cmd_r, self._w, self._r, child_evt_w = hi

        def _child_fds() -> None:
            os.dup2(child_cmd_r, 3)
            os.dup2(child_evt_w, 4)

        args = [
            self.exe,
            "--headless",
            "--no-sandbox",
            "--remote-debugging-pipe",
            f"--user-data-dir={self._profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--disable-background-networking",
            "--disable-dev-shm-usage",
            "--mute-audio",
            "--hide-scrollbars",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
            "--ignore-gpu-blocklist",
            "--window-size=1280,720",
            "about:blank",
        ]
        self.proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=self._log,
            preexec_fn=_child_fds,  # noqa: PLW1509 - started before any helper thread
            close_fds=False,
        )
        os.close(child_cmd_r)
        os.close(child_evt_w)
        self.session: str | None = None

    # -- transport ---------------------------------------------------------------------------

    def _log_tail(self) -> str:
        try:
            self._log.seek(0)
            return self._log.read().decode("utf-8", "replace")[-3000:]
        except OSError:
            return ""

    def _read_message(self, deadline: float) -> dict:
        while b"\0" not in self._buf:
            left = deadline - time.monotonic()
            if left <= 0:
                raise ChromiumError("timed out waiting for Chromium")
            ready, _, _ = select.select([self._r], [], [], min(left, 1.0))
            if not ready:
                if self.proc.poll() is not None:
                    raise ChromiumError(f"Chromium exited:\n{self._log_tail()}")
                continue
            chunk = os.read(self._r, 1 << 20)
            if not chunk:
                raise ChromiumError(f"Chromium closed the pipe:\n{self._log_tail()}")
            self._buf += chunk
        raw, self._buf = self._buf.split(b"\0", 1)
        return json.loads(raw)

    def _event(self, msg: dict) -> None:
        method = msg.get("method")
        params = msg.get("params", {})
        if method == "Runtime.consoleAPICalled":
            text = " ".join(
                str(a.get("value", a.get("description", ""))) for a in params.get("args", [])
            )
            line = f"[console.{params.get('type')}] {text}"
            self.console.append(line)
            if self.verbose or params.get("type") in ("error", "warning"):
                print(line)
        elif method == "Runtime.exceptionThrown":
            det = params.get("exceptionDetails", {})
            desc = det.get("exception", {}).get("description") or det.get("text")
            line = f"[page exception] {desc}"
            self.console.append(line)
            print(line)

    def send(self, method: str, params: dict | None = None, *, timeout: float = 60.0) -> dict:
        self._id += 1
        mid = self._id
        msg: dict = {"id": mid, "method": method, "params": params or {}}
        if self.session and not method.startswith(("Target.", "Browser.")):
            msg["sessionId"] = self.session
        data = json.dumps(msg).encode() + b"\0"
        while data:
            n = os.write(self._w, data)
            data = data[n:]
        deadline = time.monotonic() + timeout
        while True:
            reply = self._read_message(deadline)
            if reply.get("id") == mid:
                if "error" in reply:
                    raise ChromiumError(f"{method}: {reply['error'].get('message')}")
                return reply.get("result", {})
            self._event(reply)

    # -- page helpers ------------------------------------------------------------------------

    def open_page(self, url: str, *, timeout: float = 60.0) -> None:
        target = self.send("Target.createTarget", {"url": "about:blank"})["targetId"]
        self.session = self.send("Target.attachToTarget", {"targetId": target, "flatten": True})[
            "sessionId"
        ]
        self.send("Runtime.enable")
        self.send("Page.navigate", {"url": url})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.evaluate("!!(window.kn5view && window.kn5view.ready)", timeout=10):
                return
            time.sleep(0.05)
        raise ChromiumError(f"page did not get ready: {url}\n" + "\n".join(self.console[-20:]))

    def evaluate(self, expression: str, *, timeout: float = 60.0):
        res = self.send(
            "Runtime.evaluate",
            {"expression": expression, "awaitPromise": True, "returnByValue": True},
            timeout=timeout,
        )
        if "exceptionDetails" in res:
            det = res["exceptionDetails"]
            desc = det.get("exception", {}).get("description") or det.get("text")
            raise ChromiumError(f"page error: {desc}")
        return res.get("result", {}).get("value")

    def call(self, function: str, arg, *, timeout: float = 120.0):
        """Await window.kn5view.<function>(arg) and return its JSON result."""
        return self.evaluate(f"window.kn5view.{function}({json.dumps(arg)})", timeout=timeout)

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                try:
                    self.send("Browser.close", timeout=5)
                except (ChromiumError, OSError):
                    pass
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
        finally:
            for fd in (self._r, self._w):
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._log.close()
            shutil.rmtree(self._profile, ignore_errors=True)

    def __enter__(self) -> Chromium:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
