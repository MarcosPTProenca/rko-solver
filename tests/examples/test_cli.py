"""Terminal contracts for the Knapsack problem adapter.

Objective:
- Run a selected instance and external configuration without a Python driver.

Technical approach:
- Check argument boundaries and the module entry point in subprocesses.

Flow (pseudocode):
1. Parse single-instance and legacy batch selections.
2. Reject ambiguous or missing input paths.
3. Execute sequential/spawn searches and independently check exported accounting.

Important criteria:
- Preserve paths containing spaces and do not modify user parameter files.

Inputs/Outputs:
- Input: temporary instances, parameter TOMLs and CLI arguments.
- Output: parser assertions and checked CSV results.
"""

import csv
import os
import pytest
import shutil
import subprocess
import sys
from examples.run_kp_cpp_instances import default_instances_dir
from examples.run_kp_cpp_instances import parse_args
from pathlib import Path
from rko_solver.parameters.loader import default_params_dir

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "flags, expected", [([], None), (["--prefill-pool"], True), (["--no-prefill-pool"], False)]
)
def test_prefill_cli_override(
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
    expected: bool | None,
) -> None:
    """Keep TOML precedence unless initialization is explicitly enabled or disabled.

    Args:
        monkeypatch (pytest.MonkeyPatch): Supply the command-line override.
        flags (list[str]): Explicit opt-in, opt-out or absent option.
        expected (bool | None): Override forwarded to the public solver.
    """
    monkeypatch.setattr(sys, "argv", ["rko-solver", *flags])
    assert parse_args().prefill_pool is expected


def test_legacy_batch_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the existing no-argument bundled-instance selection.

    Args:
        monkeypatch (pytest.MonkeyPatch): Supply an empty command line.
    """
    monkeypatch.setattr(sys, "argv", ["rko-solver"])
    args = parse_args()
    assert args.instance is None
    assert args.instances_dir == default_instances_dir()
    assert args.pattern == "kp*.txt"


def test_positional_instance_preserves_spaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accept a direct instance path without glob-based selection.

    Args:
        tmp_path (Path): Temporary fixture directory.
        monkeypatch (pytest.MonkeyPatch): Supply a direct input argument.
    """
    instance = tmp_path / "tiny instance.txt"
    instance.write_text("3 5 6 1 10 2 12 3", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["rko-solver", str(instance)])
    assert parse_args().instance == instance


@pytest.mark.parametrize("selector", ["--instances-dir", "--pattern"])
def test_single_instance_rejects_batch_selectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
) -> None:
    """Reject conflicting selectors rather than silently ignoring user settings.

    Args:
        tmp_path (Path): Temporary input directory.
        monkeypatch (pytest.MonkeyPatch): Supply conflicting arguments.
        selector (str): Batch-only option.
    """
    instance = tmp_path / "tiny.txt"
    instance.write_text("1 1 1 1", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["rko-solver", str(instance), selector, str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("name", ["missing.txt", "."])
def test_instance_must_be_a_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """Report missing files and directories as command-line errors.

    Args:
        tmp_path (Path): Existing directory for the invalid path.
        monkeypatch (pytest.MonkeyPatch): Supply the invalid instance argument.
        name (str): Missing filename or existing directory marker.
    """
    monkeypatch.setattr(sys, "argv", ["rko-solver", str(tmp_path / name)])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("execution", ["sequential", "process"])
@pytest.mark.parametrize("selection", ["single", "batch"])
@pytest.mark.parametrize("quiet", [False, True])
def test_console_runs_external_instance_and_parameters(
    tmp_path: Path,
    execution: str,
    selection: str,
    quiet: bool,
) -> None:
    """Execute the real entry point with user-owned configuration and CSV output.

    Args:
        tmp_path (Path): Temporary instance/configuration/result directory.
        execution (str): Sequential or spawned execution mode.
        selection (str): Direct path or legacy directory/glob selection.
        quiet (bool): Suppress live progress but retain the final CSV summary.
    """
    instance = tmp_path / "tiny instance.txt"
    instance.write_text("3 5 6 1 10 2 12 3", encoding="utf-8")
    parameters = tmp_path / "my parameters"
    parameters.mkdir()
    for source in default_params_dir().glob("*.toml"):
        shutil.copy2(source, parameters / source.name)
    configuration = parameters / "config.toml"
    configuration.write_text(
        '[run]\ncontrol=0\npool_size=2\n[active_metaheuristics]\nlist=["MultiStart"]\n',
        encoding="utf-8",
    )
    before = {path.name: path.read_bytes() for path in parameters.iterdir()}
    output = tmp_path / "my results" / "result.csv"
    arguments = (
        [str(instance)]
        if selection == "single"
        else [
            "--instances-dir",
            str(tmp_path),
            "--pattern",
            instance.name,
        ]
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.run_kp_cpp_instances",
            *arguments,
            *(["--quiet"] if quiet else []),
            *(["--prefill-pool"] if selection == "single" else []),
            *(["--n-jobs", "1"] if execution == "process" else []),
            "--params-dir",
            str(parameters),
            "--limit-time",
            "0.05",
            "--execution",
            execution,
            "--random-state",
            "42",
            "--output-csv",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    row = rows[0]
    assert row["instance"] == instance.name
    assert int(row["n_items"]) == 3
    assert row["engine"] == "python" and row["execution"] == execution
    assert row["worker_budget"] == ("1" if execution == "process" else "")
    assert row["seed"] == "42"
    assert float(row["cost"]) == 100000 * max(0, int(row["weight"]) - 5) - int(row["profit"])
    assert "CSV saved to:" in completed.stdout
    assert ("NEW BEST" in completed.stdout) == (not quiet)
    assert ("RKO START" in completed.stdout) == (not quiet)
    assert ("RKO DONE" in completed.stdout) == (not quiet)
    assert before == {path.name: path.read_bytes() for path in parameters.iterdir()}


def test_backend_flag_is_a_deprecated_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Translate `--backend` to engine and execution and say it is deprecated.

    Args:
        tmp_path (Path): Directory holding the instance file.
        monkeypatch (pytest.MonkeyPatch): Supply command-line arguments.
        capsys (pytest.CaptureFixture[str]): Capture the warning on standard error.
    """
    instance = tmp_path / "tiny.txt"
    instance.write_text("3 5 6 1 10 2 12 3", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["rko-solver", str(instance), "--backend", "serial"])
    args = parse_args()
    assert (args.engine, args.execution) == ("python", "sequential")
    assert "--backend is deprecated" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", ["rko-solver", str(instance)])
    args = parse_args()
    assert (args.engine, args.execution) == ("python", None)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "flags",
    [
        ["--engine", "rust", "--execution", "thread"],
        ["--backend", "serial", "--engine", "python"],
        ["--backend", "native", "--execution", "process"],
    ],
)
def test_invalid_engine_flags_are_command_line_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str]
) -> None:
    """Reject an execution for a compiled engine and mixtures with the deprecated flag.

    Args:
        tmp_path (Path): Directory holding the instance file.
        monkeypatch (pytest.MonkeyPatch): Supply command-line arguments.
        flags (list[str]): Contradictory engine options.
    """
    instance = tmp_path / "tiny.txt"
    instance.write_text("3 5 6 1 10 2 12 3", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["rko-solver", str(instance), *flags])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


def test_console_runs_the_rust_engine_on_the_builtin_knapsack(tmp_path: Path) -> None:
    """Run the real entry point with `--engine rust` and record the engine in the CSV.

    Args:
        tmp_path (Path): Temporary instance and result directory.
    """
    pytest.importorskip("rko_solver._native")
    instance = tmp_path / "tiny.txt"
    instance.write_text("3 5 6 1 10 2 12 3", encoding="utf-8")
    output = tmp_path / "result.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.run_kp_cpp_instances",
            str(instance),
            "--quiet",
            "--engine",
            "rust",
            "--limit-time",
            "0.1",
            "--random-state",
            "42",
            "--output-csv",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["engine"] == "rust" and row["execution"] == ""
    assert float(row["cost"]) == 100000 * max(0, int(row["weight"]) - 5) - int(row["profit"])
