"""Contract tests for the structure, comment and type-error ratchet.

Objective:
- Verify that limits are strict without a baseline, that recorded debt may stay but never
  grow, and that `--tighten` only lowers recorded values.

Technical approach:
- Build throwaway git repositories containing small Python and Rust sources and run the real
  `QualityGate` (real `lizard`, real `mypy` for one case) against them.

Flow (pseudocode):
1. Create a repository with sources that exceed a limit.
2. Check strictly, record a baseline, then make the code equal, worse, better and new.
3. Compare the reported violations and the rewritten baselines.

Important criteria:
- Tests, vendored code and ignored files are never measured.
- A missing baseline means no debt is allowed.

Inputs/Outputs:
- Input: generated source files and baselines in a temporary directory.
- Output: exact violation messages, baseline contents and exit statuses.
"""

import json
import pytest
import subprocess
from pathlib import Path
from tools.quality_gate import QualityGate


def build(root: Path, files: dict[str, str]) -> QualityGate:
    """Create a git repository with the given files and return a gate for it.

    Args:
        root (Path): Empty directory to initialize.
        files (dict[str, str]): Relative path to file content.

    Returns:
        QualityGate: Gate bound to the repository.
    """
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    files = {"pyproject.toml": "[tool.quality]\ntype-paths = []\n", **files}
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return QualityGate(root)


def messy(parameters: int = 7, branches: int = 0) -> str:
    """Generate a function with the requested parameter and branch counts.

    Args:
        parameters (int): Number of parameters.
        branches (int): Number of `if` statements, each adding one to the complexity.

    Returns:
        str: Python source defining `work`.
    """
    names = ", ".join(f"a{index}: int" for index in range(parameters))
    body = "".join(f"    if a0 == {index}:\n        return {index}\n" for index in range(branches))
    return f"def work({names}) -> int:\n{body}    return 0\n"


def test_limits_are_strict_without_a_baseline(tmp_path: Path) -> None:
    """Parameters, complexity, function length and file length all fail when unrecorded.

    Args:
        tmp_path (Path): Temporary repository root.
    """
    long_body = "".join(f"    x{index} = {index}\n" for index in range(110))
    gate = build(
        tmp_path,
        {
            "params.py": messy(parameters=7),
            "branchy.py": messy(parameters=1, branches=12),
            "long_function.py": f"def work() -> None:\n{long_body}",
            "long_file.py": "x = 1\n" * 601,
        },
    )
    errors = gate.violations(gate.collect_structure(), gate.collect_comments())
    text = "\n".join(errors)
    assert "params.py::work" in text and "parameter_count=7 exceeds 6" in text
    assert "branchy.py::work" in text and "cyclomatic_complexity=13 exceeds 10" in text
    assert "long_function.py::work" in text and "exceeds 100" in text and "nloc=11" in text
    assert "long_file.py: 601 lines exceeds 600" in text
    assert gate.run(check_types=False) == 1


def test_rust_functions_and_files_are_measured(tmp_path: Path) -> None:
    """Rust sources get the same function and file limits as Python.

    Args:
        tmp_path (Path): Temporary repository root.
    """
    arguments = ", ".join(f"a{index}: i32" for index in range(8))
    gate = build(
        tmp_path,
        {
            "engine/lib.rs": f"pub fn work({arguments}) -> i32 {{ a0 }}\n",
            "engine/big.rs": "const X: i32 = 1;\n" * 601,
        },
    )
    errors = "\n".join(gate.violations(gate.collect_structure(), gate.collect_comments()))
    assert "engine/lib.rs::work" in errors and "parameter_count=8 exceeds 6" in errors
    assert "engine/big.rs: 601 lines exceeds 600" in errors


def test_tests_vendored_code_and_ignored_files_are_not_measured(tmp_path: Path) -> None:
    """Test modules, vendor directories and git-ignored files never count.

    Args:
        tmp_path (Path): Temporary repository root.
    """
    gate = build(
        tmp_path,
        {
            "pkg/test_things.py": messy(parameters=9),
            "pkg/conftest.py": messy(parameters=9),
            "pkg/tests/helpers.py": messy(parameters=9),
            "vendor/lib.py": messy(parameters=9),
            "ignored.py": messy(parameters=9),
            ".gitignore": "ignored.py\n",
            "real.py": messy(parameters=2),
        },
    )
    assert [path.as_posix() for path in gate.source_files()] == ["real.py"]
    assert gate.run(check_types=False) == 0


def test_baseline_allows_equal_debt_and_blocks_worse_or_new(tmp_path: Path) -> None:
    """Recorded debt may stay, any increase or new violation fails.

    Args:
        tmp_path (Path): Temporary repository root.
    """
    gate = build(tmp_path, {"legacy.py": messy(parameters=8), "other.py": messy(parameters=2)})
    assert gate.run(write_baseline=True) == 0
    assert gate.run(check_types=False) == 0
    recorded = json.loads(gate.structure_baseline.read_text())
    assert list(recorded["functions"].values()) == [{"parameter_count": 8}]
    (tmp_path / "legacy.py").write_text(messy(parameters=9), encoding="utf-8")
    assert any(
        "parameter_count=9 exceeds 8" in e
        for e in gate.violations(gate.collect_structure(), gate.collect_comments())
    )
    (tmp_path / "legacy.py").write_text(messy(parameters=8), encoding="utf-8")
    (tmp_path / "other.py").write_text(messy(parameters=7), encoding="utf-8")
    assert gate.run(check_types=False) == 1


def test_tighten_lowers_recorded_values_and_never_adds_debt(tmp_path: Path) -> None:
    """Improvements are locked in; new violations are not admitted by `--tighten`.

    Args:
        tmp_path (Path): Temporary repository root.
    """
    gate = build(
        tmp_path, {"a.py": messy(parameters=9), "b.py": messy(parameters=8), "c.py": "x = 1\n"}
    )
    gate.run(write_baseline=True)
    (tmp_path / "a.py").write_text(messy(parameters=7), encoding="utf-8")
    (tmp_path / "b.py").write_text(messy(parameters=2), encoding="utf-8")
    (tmp_path / "c.py").write_text(messy(parameters=11), encoding="utf-8")
    assert gate.run(tighten=True) == 0
    recorded = json.loads(gate.structure_baseline.read_text())["functions"]
    assert {key.split("::")[0]: value for key, value in recorded.items()} == {
        "a.py": {"parameter_count": 7}
    }
    assert gate.run(check_types=False) == 1


def test_same_named_functions_are_tracked_separately(tmp_path: Path) -> None:
    """Two functions with the same signature in one file get distinct baseline keys.

    Args:
        tmp_path (Path): Temporary repository root.
    """
    gate = build(tmp_path, {"twins.py": messy(parameters=7) + "\n\n" + messy(parameters=7)})
    keys = sorted(gate.collect_structure()["functions"])
    assert len(keys) == 2
    assert sum(key.endswith("#2") for key in keys) == 1
    comment_keys = gate.collect_comments()["functions"]
    assert set(comment_keys) == {"twins.py::work", "twins.py::work#2"}


def test_comment_volume_is_limited_but_pragmas_are_not_counted(tmp_path: Path) -> None:
    """Explanatory comments fail the gate; tool pragmas do not count.

    Args:
        tmp_path (Path): Temporary repository root.
    """
    chatty = (
        "def talk() -> int:\n" + "".join(f"    # note {n}\n" for n in range(9)) + "    return 1\n"
    )
    quiet = (
        "def quiet() -> int:\n"
        "    value = 1  # noqa: E501\n"
        "    other = 2  # type: ignore[assignment]\n"
        "    return value + other\n"
    )
    gate = build(tmp_path, {"chatty.py": chatty, "quiet.py": quiet})
    metrics = gate.collect_comments()["functions"]
    assert metrics["chatty.py::talk"]["comment_lines"] == 9
    assert metrics["chatty.py::talk"]["max_consecutive_comments"] == 9
    assert metrics["quiet.py::quiet"]["comment_lines"] == 0
    errors = gate.violations(gate.collect_structure(), gate.collect_comments())
    assert any("chatty.py::talk: comment_lines=9 exceeds 8" in e for e in errors)
    assert not any("quiet.py" in e for e in errors)
    gate.run(write_baseline=True)
    assert gate.run(check_types=False) == 0


def test_type_errors_are_ratcheted_per_file(tmp_path: Path) -> None:
    """A file may keep its recorded type errors but not gain more, using the real mypy.

    Args:
        tmp_path (Path): Temporary repository root.
    """
    bad = "def f(x: int) -> str:\n    return x\n"
    gate = build(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": bad,
            "pyproject.toml": '[tool.quality]\ntype-paths = ["pkg"]\n',
        },
    )
    assert gate.collect_types() == {"files": {"pkg/mod.py": 1}}
    assert gate.run() == 1
    assert gate.run(write_baseline=True) == 0
    assert gate.run() == 0
    (tmp_path / "pkg" / "mod.py").write_text(bad + "def g(y: int) -> str:\n    return y\n")
    assert gate.run() == 1
    assert gate.run(check_types=False) == 0
    (tmp_path / "pkg" / "mod.py").write_text("def f(x: int) -> int:\n    return x\n")
    assert gate.run(tighten=True) == 0
    assert json.loads(gate.types_baseline.read_text())["files"] == {}


def test_type_errors_outside_the_type_paths_are_not_counted(tmp_path: Path) -> None:
    """An error inside an imported third-party module never enters or fails the gate.

    Args:
        tmp_path (Path): Temporary repository root.
    """
    gate = build(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": "import vendored\n\nvalue: int = vendored.number\n",
            "vendored.py": "number: int = 1\n\n\ndef broken(x: int) -> str:\n    return x\n",
            "pyproject.toml": '[tool.quality]\ntype-paths = ["pkg"]\n',
        },
    )
    assert gate.collect_types() == {"files": {}}
    assert gate.run() == 0


def test_command_line_modes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The entry point returns statuses, writes baselines and rejects conflicting flags.

    Args:
        tmp_path (Path): Temporary repository root.
        capsys (pytest.CaptureFixture[str]): Captured standard output.
    """
    build(tmp_path, {"m.py": messy(parameters=7)})
    root = ["--root", str(tmp_path), "--skip-types"]
    assert QualityGate.main(root) == 1
    assert "parameter_count=7 exceeds 6" in capsys.readouterr().out
    assert QualityGate.main([*root, "--write-baseline"]) == 0
    assert QualityGate.main(root) == 0
    assert "Quality gate: OK" in capsys.readouterr().out
    (tmp_path / "m.py").write_text(messy(parameters=2), encoding="utf-8")
    assert QualityGate.main(root) == 0
    assert "improved; run --tighten" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        QualityGate.main([*root, "--write-baseline", "--tighten"])


def test_keys_use_class_and_function_names_and_survive_signature_changes(tmp_path: Path) -> None:
    """Methods of different classes are separate entries; renaming a parameter keeps the allowance.

    Args:
        tmp_path (Path): Temporary repository root.
    """
    tail = "b: int, c: int, d: int, e: int, f: int, g: int) -> int:"

    def render(first: str, extra: str = "") -> str:
        """Render the module with a chosen first parameter name for `A.run`.

        Args:
            first (str): Name of the first parameter of `A.run`.
            extra (str): Extra leading parameter of `outer`, or an empty string.

        Returns:
            str: Python source with four over-limit functions.
        """
        return (
            f"class A:\n    def run(self, {first}: int, {tail}\n        return 0\n\n"
            f"class B:\n    def run(self, a: int, {tail}\n        return 0\n\n"
            f"def outer({extra}a: int, {tail}\n"
            f"    def inner(a: int, {tail}\n        return 0\n"
            "    return 0\n"
        )

    gate = build(tmp_path, {"m.py": render("a")})
    expected = ["m.py::A.run", "m.py::B.run", "m.py::outer", "m.py::outer.inner"]
    assert sorted(gate.collect_structure()["functions"]) == expected
    assert sorted(gate.collect_comments()["functions"]) == expected
    gate.run(write_baseline=True)
    (tmp_path / "m.py").write_text(render("a", extra="z: int, "), encoding="utf-8")
    assert gate.run(check_types=False) == 1
    (tmp_path / "m.py").write_text(render("renamed"), encoding="utf-8")
    assert gate.run(check_types=False) == 0
