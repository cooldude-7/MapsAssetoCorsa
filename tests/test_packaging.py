"""The Windows app packaging: PyInstaller spec, launcher, icon, GitHub Actions workflow, guide.

The real PyInstaller build runs in CI (.github/workflows/windows-app.yml), which also runs the
frozen app's `--self-test` and an HTTP smoke test; these tests check the pieces offline.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging"
SPEC = PACKAGING / "acmg.spec"
LAUNCHER = PACKAGING / "launcher.py"
ICON = PACKAGING / "icon.ico"
WORKFLOW = ROOT / ".github" / "workflows" / "windows-app.yml"
GUIDE = ROOT / "docs" / "INSTALL_APP.md"
STATIC = ROOT / "src" / "driftmaps" / "app" / "static"
APP_NAME = "AssettoCorsaMapGenerator"


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def load_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location("acmg_launcher", LAUNCHER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    mod = load_launcher()
    monkeypatch.setattr(mod, "_console_setup", lambda: None)  # leave the test console alone
    return mod


def spec_tree() -> ast.Module:
    return ast.parse(SPEC.read_text(encoding="utf-8"), filename=str(SPEC))


def spec_constant(name: str) -> object:
    for node in spec_tree().body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {SPEC.name}")


def spec_call(func: str) -> ast.Call:
    for node in ast.walk(spec_tree()):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == func:
            return node
    raise AssertionError(f"{func}(...) not found in {SPEC.name}")


def call_kwargs(call: ast.Call) -> dict[str, ast.expr]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg}


class MiniYamlError(ValueError):
    pass


class MiniYaml:
    """A strict parser for the block-style YAML subset GitHub workflows use (mappings,
    sequences, plain/quoted scalars, `[a, b]` lists, `|` blocks, comments), so the workflow is
    checked without adding PyYAML. When PyYAML is installed a test cross-checks the two."""

    KEY = re.compile(r"""^([A-Za-z0-9_.-]+|"[^"]*"|'[^']*'):(?:[ ]+(.*))?$""")

    def __init__(self, text: str) -> None:
        self.lines = text.splitlines()
        self.i = 0
        for n, line in enumerate(self.lines, 1):
            if line[: len(line) - len(line.lstrip())].count("\t"):
                raise MiniYamlError(f"line {n}: tab in indentation")

    @classmethod
    def load(cls, text: str) -> object:
        p = cls(text)
        p._skip()
        if p.i >= len(p.lines):
            return None
        value = p._node(p._indent(p.lines[p.i]))
        p._skip()
        if p.i < len(p.lines):
            raise MiniYamlError(f"line {p.i + 1}: unexpected text {p.lines[p.i].strip()!r}")
        return value

    @staticmethod
    def _indent(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    def _skip(self) -> None:
        while self.i < len(self.lines):
            st = self.lines[self.i].strip()
            if st and not st.startswith("#"):
                return
            self.i += 1

    def _node(self, indent: int) -> object:
        st = self.lines[self.i].strip()
        return self._seq(indent) if st == "-" or st.startswith("- ") else self._map(indent)

    def _peek(self, indent: int) -> str | None:
        """The next content line if it belongs to a block at `indent`, else None."""
        self._skip()
        if self.i >= len(self.lines):
            return None
        line = self.lines[self.i]
        ind = self._indent(line)
        if ind < indent:
            return None
        if ind > indent:
            raise MiniYamlError(f"line {self.i + 1}: bad indentation")
        return line.strip()

    def _seq(self, indent: int) -> list:
        out: list = []
        while (st := self._peek(indent)) is not None and (st == "-" or st.startswith("- ")):
            rest = st[1:].lstrip()
            if not rest:
                self.i += 1
                self._skip()
                out.append(self._node(self._indent(self.lines[self.i])))
            elif self.KEY.match(rest):
                child = indent + len(st) - len(rest)
                self.lines[self.i] = " " * child + rest
                out.append(self._map(child))
            else:
                out.append(self._scalar(rest))
                self.i += 1
        return out

    def _map(self, indent: int) -> dict:
        out: dict = {}
        while (st := self._peek(indent)) is not None:
            if st.startswith("- "):
                raise MiniYamlError(f"line {self.i + 1}: list item where a key was expected")
            m = self.KEY.match(st)
            if not m:
                raise MiniYamlError(f"line {self.i + 1}: expected 'key: value', got {st!r}")
            key = m.group(1).strip("\"'")
            if key in out:
                raise MiniYamlError(f"line {self.i + 1}: duplicate key {key!r}")
            rest = self._strip_comment(m.group(2) or "").strip()
            self.i += 1
            if rest in ("|", "|-"):
                out[key] = self._block(indent, keep_newline=rest == "|")
            elif rest:
                out[key] = self._scalar(rest)
            else:
                self._skip()
                nxt = self.lines[self.i] if self.i < len(self.lines) else ""
                ind = self._indent(nxt)
                if nxt and (ind > indent or (ind == indent and nxt.strip().startswith("- "))):
                    out[key] = self._node(ind)
                else:
                    out[key] = None
        return out

    def _block(self, parent: int, keep_newline: bool) -> str:
        body: list[str] = []
        content_indent = None
        while self.i < len(self.lines):
            line = self.lines[self.i]
            if line.strip():
                ind = self._indent(line)
                if ind <= parent:
                    break
                content_indent = ind if content_indent is None else content_indent
                if ind < content_indent:
                    raise MiniYamlError(f"line {self.i + 1}: bad indentation in a | block")
                body.append(line[content_indent:])
            else:
                body.append("")
            self.i += 1
        while body and not body[-1]:
            body.pop()
        return "\n".join(body) + ("\n" if keep_newline and body else "")

    @staticmethod
    def _strip_comment(text: str) -> str:
        quote = None
        for k, ch in enumerate(text):
            if quote:
                quote = None if ch == quote else quote
            elif ch in "\"'" and (k == 0 or text[k - 1] in " [,"):
                quote = ch
            elif ch == "#" and (k == 0 or text[k - 1] == " "):
                return text[:k]
        return text

    def _scalar(self, text: str) -> object:
        text = self._strip_comment(text).strip()
        if text.startswith('"'):
            if not text.endswith('"') or len(text) < 2:
                raise MiniYamlError(f"line {self.i + 1}: unterminated string")
            return json.loads(text)
        if text.startswith("'"):
            if not text.endswith("'") or len(text) < 2:
                raise MiniYamlError(f"line {self.i + 1}: unterminated string")
            return text[1:-1].replace("''", "'")
        if text.startswith("["):
            if not text.endswith("]"):
                raise MiniYamlError(f"line {self.i + 1}: unterminated [list]")
            inner = text[1:-1].strip()
            return [self._scalar(p) for p in inner.split(",")] if inner else []
        if text.startswith(("{", "&", "*", "!", "%", "@", "`", "|", ">")):
            raise MiniYamlError(f"line {self.i + 1}: unsupported or invalid value {text!r}")
        if ": " in text or text.endswith(":"):
            raise MiniYamlError(f"line {self.i + 1}: ': ' inside a plain value (quote it)")
        if text in ("true", "false"):
            return text == "true"
        if text in ("null", "~"):
            return None
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        return text


def workflow() -> dict:
    data = MiniYaml.load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def steps() -> list[dict]:
    jobs = workflow()["jobs"]
    assert len(jobs) == 1
    return next(iter(jobs.values()))["steps"]


def step_with(text: str) -> dict:
    found = [s for s in steps() if text in json.dumps(s)]
    assert found, f"no workflow step mentions {text!r}"
    return found[0]


# ---------------------------------------------------------------------------------------------
# PyInstaller spec
# ---------------------------------------------------------------------------------------------


def test_spec_is_valid_python_and_names_the_app() -> None:
    compile(SPEC.read_text(encoding="utf-8"), str(SPEC), "exec")
    assert spec_constant("APP_NAME") == APP_NAME
    exe = call_kwargs(spec_call("EXE"))
    coll = call_kwargs(spec_call("COLLECT"))
    assert isinstance(exe["name"], ast.Name) and exe["name"].id == "APP_NAME"
    assert isinstance(coll["name"], ast.Name) and coll["name"].id == "APP_NAME"
    # One-folder build (binaries go to COLLECT) with the console window kept.
    assert ast.literal_eval(exe["exclude_binaries"]) is True
    assert ast.literal_eval(exe["console"]) is True
    assert ast.literal_eval(exe["upx"]) is False


def test_spec_references_exist() -> None:
    text = SPEC.read_text(encoding="utf-8")
    assert '"launcher.py"' in text and LAUNCHER.is_file()
    assert '"icon.ico"' in text and ICON.is_file()
    assert '"static"' in text and '"driftmaps/app/static"' in text
    # The touge mode the app installs (driftmaps.csp_modes looks in <bundle>/csp/new-modes).
    assert '"csp/new-modes"' in text
    assert (ROOT / "csp" / "new-modes" / "acmg-touge" / "manifest.ini").is_file()
    for name in ("index.html", "app.js", "style.css"):
        assert (STATIC / name).is_file(), name
    assert (ROOT / "src" / "driftmaps" / "app" / "__init__.py").is_file()
    assert (ROOT / "pyproject.toml").is_file()
    # The hand-collected pieces PyInstaller can't find on its own.
    assert 'collect_submodules("driftmaps"' in text
    assert 'collect_submodules(\n    "rasterio"' in text or 'collect_submodules("rasterio"' in text
    assert 'collect_data_files(\n    "rasterio"' in text or 'collect_data_files("rasterio"' in text
    assert 'collect_delvewheel_libs_directory("rasterio"' in text
    assert 'copy_metadata("driftmaps")' in text


def test_spec_skips_only_real_modules_and_matches_the_launcher(launcher: ModuleType) -> None:
    skipped = spec_constant("SKIP_MODULES")
    assert skipped == launcher.SKIPPED_MODULES
    for name in skipped:
        rel = Path(*name.split(".")).with_suffix(".py")
        assert (ROOT / "src" / rel).is_file(), name
    assert "driftmaps.app" not in skipped and "driftmaps.build" not in skipped


def test_pyinstaller_is_a_dev_dependency() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev = project["dependency-groups"]["dev"]
    assert any(re.match(r"pyinstaller\b", d, re.I) for d in dev)


# ---------------------------------------------------------------------------------------------
# Icon
# ---------------------------------------------------------------------------------------------


def _assert_good_icon(path: Path) -> None:
    from PIL import Image

    with Image.open(path) as im:
        assert im.format == "ICO"
        sizes = im.info["sizes"]
    for size in (16, 32, 48, 256):
        assert (size, size) in sizes


def test_icon_has_the_windows_sizes() -> None:
    _assert_good_icon(ICON)


def test_spec_can_redraw_the_icon(tmp_path: Path) -> None:
    fn = next(
        n for n in spec_tree().body if isinstance(n, ast.FunctionDef) and n.name == "make_icon"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    namespace: dict = {"Path": Path}
    exec(compile(module, str(SPEC), "exec"), namespace)
    out = tmp_path / "sub" / "icon.ico"
    namespace["make_icon"](out)
    _assert_good_icon(out)


# ---------------------------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------------------------


def test_launcher_imports_and_parses_flags(launcher: ModuleType) -> None:
    assert launcher.BANNER == (
        "Assetto Corsa Map Generator is running - keep this window open. Close it to quit."
    )
    assert not launcher.frozen()
    args = launcher.parse_args([])
    assert (args.port, args.no_browser, args.data_dir, args.self_test) == (None, False, None, False)
    args = launcher.parse_args(["--no-browser", "--port", "9000", "--data-dir", "x", "--self-test"])
    assert (args.port, args.no_browser, args.data_dir, args.self_test) == (9000, True, "x", True)


def _record_run(monkeypatch: pytest.MonkeyPatch, effect=None) -> list[dict]:
    import driftmaps.app

    calls: list[dict] = []

    def fake_run(**kw):
        calls.append(kw)
        if effect is not None:
            raise effect

    monkeypatch.setattr(driftmaps.app, "run", fake_run)
    return calls


def test_launcher_starts_the_app_with_the_banner(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys, tmp_path: Path
) -> None:
    calls = _record_run(monkeypatch)
    monkeypatch.setattr(launcher, "_already_running", lambda: None)
    code = launcher.main(["--no-browser", "--port", "0", "--data-dir", str(tmp_path)])
    assert code == 0
    assert calls == [{"port": 0, "open_browser": False, "data_dir": str(tmp_path)}]
    assert launcher.BANNER in capsys.readouterr().out


def test_launcher_default_opens_the_browser(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    calls = _record_run(monkeypatch)
    monkeypatch.setattr(launcher, "_already_running", lambda: None)
    assert launcher.main([]) == 0
    assert calls == [{"port": None, "open_browser": True, "data_dir": None}]
    out = capsys.readouterr().out
    assert launcher.BANNER in out and "browser" in out


def test_launcher_second_start_just_reopens_the_page(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    calls = _record_run(monkeypatch)
    monkeypatch.setattr(launcher, "_already_running", lambda: "http://127.0.0.1:8765/")
    assert launcher.main([]) == 0
    assert len(calls) == 1  # run() itself opens the existing page and returns
    out = capsys.readouterr().out
    assert "already open" in out and launcher.BANNER not in out


def test_launcher_reports_a_crash(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _record_run(monkeypatch, RuntimeError("boom"))
    monkeypatch.setattr(launcher, "_already_running", lambda: None)
    assert launcher.main(["--no-browser"]) == 1
    captured = capsys.readouterr()
    assert "has to stop" in captured.out and "boom" in captured.err


def test_launcher_passes_exit_codes_through(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_run(monkeypatch, SystemExit(1))
    monkeypatch.setattr(launcher, "_already_running", lambda: None)
    assert launcher.main(["--no-browser", "--port", "1"]) == 1
    _record_run(monkeypatch, KeyboardInterrupt())
    assert launcher.main(["--no-browser", "--port", "1"]) == 0


def test_port_free(launcher: ModuleType) -> None:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        assert not launcher.port_free(port)


def test_already_running_finds_the_app(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    import driftmaps.app.server as server

    monkeypatch.setattr(launcher, "port_free", lambda port: True)
    assert launcher._already_running() is None  # nothing listening: no HTTP probe at all
    seen: list[int] = []
    monkeypatch.setattr(launcher, "port_free", lambda port: False)
    monkeypatch.setattr(server, "running_instance", lambda port: seen.append(port) or "url")
    assert launcher._already_running() == "url" and seen == [server.DEFAULT_PORT]


def test_self_test_passes_here(launcher: ModuleType, capsys) -> None:
    assert launcher.main(["--self-test"]) == 0
    out = capsys.readouterr().out
    assert "Self-test passed." in out
    assert "[FAIL]" not in out and out.count("[ ok ]") == 5


def test_self_test_reports_failures(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def broken() -> str:
        raise RuntimeError("no FreeType")

    monkeypatch.setattr(launcher, "_check_pillow", broken)
    assert launcher.self_test() == 1
    out = capsys.readouterr().out
    assert "[FAIL] Pillow: RuntimeError: no FreeType" in out and "FAILED" in out


# ---------------------------------------------------------------------------------------------
# GitHub Actions workflow
# ---------------------------------------------------------------------------------------------


def test_workflow_triggers() -> None:
    on = workflow()["on"]
    assert on["push"]["branches"] == ["**"]
    assert on["push"]["tags"] == ["v*"]
    assert "workflow_dispatch" in on


def test_workflow_job_setup() -> None:
    (job,) = workflow()["jobs"].values()
    assert job["runs-on"] == "windows-latest"
    py = step_with("actions/setup-python")
    assert py["with"]["python-version"] == "3.12"
    step_with("astral-sh/setup-uv")
    assert "uv sync --locked" in step_with("uv sync")["run"]


def test_workflow_has_the_steps_in_order() -> None:
    names = [s.get("name", s.get("uses", "")) for s in steps()]
    runs = [str(s.get("run", "")) + str(s.get("uses", "")) for s in steps()]

    def index(text: str) -> int:
        hits = [i for i, r in enumerate(runs) if text in r]
        assert hits, f"no step runs {text!r} ({names})"
        return hits[0]

    order = [
        index("pytest"),
        index("pyinstaller --noconfirm --clean packaging/acmg.spec"),
        index("AssettoCorsaMapGenerator.exe --self-test"),
        index("--no-browser"),
        index("Compress-Archive"),
        index("actions/upload-artifact"),
        index("gh release"),
    ]
    assert order == sorted(order)


def test_workflow_smoke_test_starts_queries_and_stops_the_exe() -> None:
    run = step_with("--no-browser")["run"]
    assert "dist\\AssettoCorsaMapGenerator\\AssettoCorsaMapGenerator.exe" in run
    assert "Start-Process" in run and "Stop-Process" in run
    assert "curl.exe" in run and "/api/status" in run
    assert "Assetto Corsa Map Generator" in run  # checks the status JSON's app name


def test_workflow_zips_uploads_and_releases() -> None:
    wf = workflow()
    assert wf["env"]["ZIP"] == "AssettoCorsaMapGenerator-windows.zip"
    zip_step = step_with("Compress-Archive")
    assert "dist\\AssettoCorsaMapGenerator" in zip_step["run"]
    upload = step_with("actions/upload-artifact")
    assert upload["with"]["name"] == "AssettoCorsaMapGenerator-windows"
    assert upload["with"]["path"].startswith("dist/AssettoCorsaMapGenerator")
    release = step_with("gh release")
    assert "refs/tags/v" in release["if"]
    assert "$env:ZIP" in release["run"]
    assert wf["permissions"]["contents"] == "write"


def test_mini_yaml_is_strict() -> None:
    assert MiniYaml.load("a: 1\nb:\n  - x\n  - y: [p, 'q']\n") == {
        "a": 1,
        "b": ["x", {"y": ["p", "q"]}],
    }
    assert MiniYaml.load("r: |\n  one\n  # not a comment\n\nx: 'it''s' # c\n") == {
        "r": "one\n# not a comment\n",
        "x": "it's",
    }
    for bad in ("a: b: c\n", "a:\n\t- x\n", "a: 1\n   b: 2\n", "a: 1\na: 2\n", "a: 'x\n"):
        with pytest.raises(MiniYamlError):
            MiniYaml.load(bad)


def test_mini_yaml_agrees_with_pyyaml() -> None:
    yaml = pytest.importorskip("yaml")
    expected = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    expected["on"] = expected.pop(True)  # YAML 1.1 reads the key `on` as a boolean
    assert workflow() == expected


# ---------------------------------------------------------------------------------------------
# Player guide
# ---------------------------------------------------------------------------------------------


def test_guide_covers_the_player_steps(launcher: ModuleType) -> None:
    text = GUIDE.read_text(encoding="utf-8")
    assert launcher.BANNER in text
    assert f"actions/workflows/{WORKFLOW.name}" in text
    assert "/releases" in text
    for phrase in (
        "AssettoCorsaMapGenerator-windows.zip",
        "Extract All",
        "AssettoCorsaMapGenerator.exe",
        "More info",
        "Run anyway",
        "Allow",
        "Assetto Corsa not found",
        "busy",
        "%LOCALAPPDATA%\\AssettoCorsaMapGenerator",
        "content\\tracks",
        "Content Manager",
    ):
        assert phrase in text, phrase
    steps_section = text.split("## 4.", 1)[1].split("\n## ", 1)[0]
    assert re.findall(r"^\d\. ", steps_section, re.M) == [f"{i}. " for i in range(1, 7)]


def test_spec_windows_version_info() -> None:
    pytest.importorskip("pefile")  # PyInstaller's Windows-only dependency
    fn = next(
        n
        for n in spec_tree().body
        if isinstance(n, ast.FunctionDef) and n.name == "_windows_version_info"
    )
    namespace: dict = {"APP_NAME": APP_NAME, "APP_TITLE": "Assetto Corsa Map Generator"}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SPEC), "exec"), namespace)
    info = namespace["_windows_version_info"]("0.1.0")
    assert info.toRaw() and "AssettoCorsaMapGenerator.exe" in str(info)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console only")
def test_console_setup_is_harmless_on_windows() -> None:
    load_launcher()._console_setup()
