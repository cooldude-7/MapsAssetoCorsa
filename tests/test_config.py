"""Tests for driftmaps.config: the VDF reader and Assetto Corsa folder discovery.

Discovery runs against fake Steam folders in tmp_path with `search_defaults=False`, so the
registry and the real machine's folders never leak into the result.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from driftmaps import config

NEW_LIBRARYFOLDERS = r"""
"libraryfolders"
{
	"0"
	{
		"path"		"C:\\Program Files (x86)\\Steam"
		"label"		""
		"contentid"		"1234567890"
		"totalsize"		"0"
		"apps"
		{
			"228980"		"236524036"
		}
	}
	"1"
	{
		"path"		"D:\\SteamLibrary"
		"label"		"games"
		"apps"
		{
			"244210"		"12345678901"
			"1066890"		"1"
		}
	}
}
"""

OLD_LIBRARYFOLDERS = """
"LibraryFolders"
{
	"TimeNextStatsReport"		"1600000000"
	"ContentStatsID"		"-123"
	"1"		"E:\\\\Games\\\\Steam"
}
"""

APPMANIFEST = """
"AppState"
{
	"appid"		"244210"
	"Universe"		"1"
	"name"		"Assetto Corsa"
	"StateFlags"		"4"
	"installdir"		"%s"
	"UserConfig"
	{
		"language"		"english"
	}
}
"""


def vdf_text(path: Path) -> str:
    """A path as it appears inside a .vdf (backslashes escaped)."""
    return str(path).replace("\\", "\\\\")


def make_ac(root: Path, exe: str = "acs.exe") -> Path:
    (root / "content" / "tracks").mkdir(parents=True)
    (root / exe).write_bytes(b"")
    return root


def make_steam(tmp_path: Path, *, libraries: list[Path], ac_library: Path | None) -> Path:
    steam = tmp_path / "steam"
    (steam / "steamapps").mkdir(parents=True)
    entries = []
    for i, lib in enumerate([steam, *libraries]):
        apps = '"244210" "1"' if lib == ac_library else '"7" "1"'
        entries.append(f'"{i}" {{ "path" "{vdf_text(lib)}" "apps" {{ {apps} }} }}')
    (steam / "steamapps" / "libraryfolders.vdf").write_text(
        '"libraryfolders" {\n' + "\n".join(entries) + "\n}\n", encoding="utf-8"
    )
    return steam


def find(**kwargs) -> Path | None:
    kwargs.setdefault("env", {})
    return config.find_ac_root(search_defaults=False, **kwargs)


# ---------------------------------------------------------------------------------------------
# VDF parsing
# ---------------------------------------------------------------------------------------------


def test_parse_new_libraryfolders():
    data = config.parse_vdf(NEW_LIBRARYFOLDERS)
    libs = data["libraryfolders"]
    assert isinstance(libs, dict)
    assert libs["0"]["path"] == "C:\\Program Files (x86)\\Steam"  # \\ unescaped
    assert libs["1"]["label"] == "games"
    assert libs["1"]["apps"] == {"244210": "12345678901", "1066890": "1"}


def test_parse_old_libraryfolders():
    data = config.parse_vdf(OLD_LIBRARYFOLDERS)
    assert data["LibraryFolders"]["1"] == "E:\\Games\\Steam"


def test_parse_vdf_comments_conditions_escapes_and_bare_words():
    text = """
    // a comment
    "root" // trailing comment
    {
        "quoted"   "say \\"hi\\"\\tthere"
        bare       word
        "unknown"  "C:\\Steam"
        "cond"     "1" [$WIN32]
        "empty"    ""
        "nested"   { "a" { "b" "c" } }
        "dup" "1"
        "dup" "2"
    }
    """
    d = config.parse_vdf(text)["root"]
    assert d["quoted"] == 'say "hi"\tthere'
    assert d["bare"] == "word"
    assert d["unknown"] == "C:\\Steam"  # unknown escape keeps its backslash
    assert d["cond"] == "1" and d["empty"] == ""
    assert d["nested"] == {"a": {"b": "c"}}
    assert d["dup"] == "2"


def test_parse_vdf_with_bom():
    assert config.parse_vdf('\ufeff"a" { "b" "c" }') == {"a": {"b": "c"}}


@pytest.mark.parametrize("bad", ['"a" {', '"a" }', '"a"', '"a" { "b" }', "{"])
def test_parse_vdf_rejects_malformed(bad):
    with pytest.raises(ValueError):
        config.parse_vdf(bad)


# ---------------------------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------------------------


def test_finds_ac_in_a_second_library(tmp_path):
    lib2 = tmp_path / "lib2"
    ac = make_ac(lib2 / "steamapps" / "common" / "assettocorsa")
    (lib2 / "steamapps" / "appmanifest_244210.acf").write_text(
        APPMANIFEST % "assettocorsa", encoding="utf-8"
    )
    other = tmp_path / "lib1"
    (other / "steamapps").mkdir(parents=True)
    steam = make_steam(tmp_path, libraries=[other, lib2], ac_library=lib2)
    assert find(steam_roots=[steam]) == ac
    assert config.tracks_dir(ac) == ac / "content" / "tracks"
    # The library whose "apps" lists AC is searched first.
    libs = config.steam_library_folders(steam)
    assert libs[0] == lib2 and set(libs) == {steam, other, lib2}


def test_manifest_installdir_is_honoured(tmp_path):
    lib = tmp_path / "lib"
    ac = make_ac(lib / "steamapps" / "common" / "Assetto Corsa Custom", exe="AssettoCorsa.exe")
    (lib / "steamapps" / "appmanifest_244210.acf").write_text(
        APPMANIFEST % "Assetto Corsa Custom", encoding="utf-8"
    )
    steam = make_steam(tmp_path, libraries=[lib], ac_library=lib)
    assert find(steam_roots=[steam]) == ac
    assert config.ac_roots_in_library(lib)[0] == ac


def test_ac_in_the_steam_root_without_manifest(tmp_path):
    steam = make_steam(tmp_path, libraries=[], ac_library=None)
    ac = make_ac(steam / "steamapps" / "common" / "assettocorsa")
    assert find(steam_roots=[steam]) == ac


def test_old_style_libraryfolders(tmp_path):
    lib = tmp_path / "old lib"
    ac = make_ac(lib / "steamapps" / "common" / "assettocorsa")
    steam = tmp_path / "steam"
    (steam / "steamapps").mkdir(parents=True)
    (steam / "steamapps" / "libraryfolders.vdf").write_text(
        f'"LibraryFolders"\n{{\n\t"ContentStatsID"\t\t"-1"\n\t"1"\t\t"{vdf_text(lib)}"\n}}\n',
        encoding="utf-8",
    )
    assert find(steam_roots=[steam]) == ac


def test_config_libraryfolders_location(tmp_path):
    lib = tmp_path / "lib"
    ac = make_ac(lib / "steamapps" / "common" / "assettocorsa")
    steam = tmp_path / "steam"
    (steam / "config").mkdir(parents=True)
    (steam / "config" / "libraryfolders.vdf").write_text(
        f'"libraryfolders" {{ "0" {{ "path" "{vdf_text(lib)}" }} }}', encoding="utf-8"
    )
    assert find(steam_roots=[steam]) == ac


def test_env_var_wins(tmp_path):
    lib = tmp_path / "lib"
    make_ac(lib / "steamapps" / "common" / "assettocorsa")
    steam = make_steam(tmp_path, libraries=[lib], ac_library=lib)
    custom = make_ac(tmp_path / "my ac")
    assert find(env={config.ENV_AC_ROOT: str(custom)}, steam_roots=[steam]) == custom


def test_invalid_env_var_warns_and_falls_back(tmp_path):
    lib = tmp_path / "lib"
    ac = make_ac(lib / "steamapps" / "common" / "assettocorsa")
    steam = make_steam(tmp_path, libraries=[lib], ac_library=lib)
    with pytest.warns(UserWarning, match=config.ENV_AC_ROOT):
        found = find(env={config.ENV_AC_ROOT: str(tmp_path / "nope")}, steam_roots=[steam])
    assert found == ac


def test_nothing_found(tmp_path):
    steam = make_steam(tmp_path, libraries=[tmp_path / "missing"], ac_library=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert find(steam_roots=[steam]) is None
        assert find(steam_roots=[tmp_path / "no steam here"]) is None
        assert find() is None


def test_corrupt_steam_files_are_skipped(tmp_path):
    steam = tmp_path / "steam"
    (steam / "steamapps").mkdir(parents=True)
    (steam / "steamapps" / "libraryfolders.vdf").write_text('"libraryfolders" {', "utf-8")
    (steam / "steamapps" / "appmanifest_244210.acf").write_bytes(b"\xff\xfe garbage {{")
    ac = make_ac(steam / "steamapps" / "common" / "assettocorsa")
    assert find(steam_roots=[steam]) == ac


def test_is_ac_root(tmp_path):
    assert not config.is_ac_root(tmp_path / "missing")
    (tmp_path / "content" / "tracks").mkdir(parents=True)
    assert not config.is_ac_root(tmp_path)  # no executable
    (tmp_path / "acs.exe").write_bytes(b"")
    assert config.is_ac_root(tmp_path)
    exe_only = tmp_path / "exe_only"
    exe_only.mkdir()
    (exe_only / "AssettoCorsa.exe").write_bytes(b"")
    assert not config.is_ac_root(exe_only)  # no content/tracks


def test_default_steam_roots_are_paths():
    roots = config.default_steam_roots(home=Path("/home/someone"))
    assert roots and all(isinstance(p, Path) for p in roots)


def test_candidates_start_with_env(tmp_path):
    cands = list(
        config.candidate_ac_roots(
            env={config.ENV_AC_ROOT: str(tmp_path)}, steam_roots=[], search_defaults=False
        )
    )
    assert cands == [tmp_path]
