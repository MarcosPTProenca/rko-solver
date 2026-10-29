"""Repository layout and module-entry contracts for reproducible benchmark studies.

Objective:
- Keep benchmark code grouped by purpose and recorded evidence separate from executable scripts.

Technical approach:
- Inspect the repository layout and execute representative module entry points in fresh processes.
- Verify the native report resolves recorded evidence and output assets independently of its module.

Flow (pseudocode):
1. Assert that the benchmark root contains navigation rather than loose scripts or measurements.
2. Resolve representative grouped entry points and print their help without running a study.
3. Check that recorded evidence and report assets resolve to their maintained repository paths.

Important criteria:
- These checks neither rerun historical studies nor assert performance improvements.
- A module help command must finish without fetching datasets or invoking an optimizer.

Inputs/Outputs:
- Input: the repository benchmark tree and installed development dependencies.
- Output: assertions on source organization, entry-point usability and reference locations.
"""

import importlib
import pytest
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"


def test_benchmark_root_contains_navigation_only() -> None:
    """Require loose executable scripts and measurements to live in their dedicated directories."""
    files = {path.name for path in BENCHMARKS.iterdir() if path.is_file()}
    assert files == {"README.md", "__init__.py"}
    assert {
        "runtime",
        "algorithms",
        "engines",
        "modeling",
        "problems",
        "tuning",
        "self_evolve",
    }.issubset({path.name for path in BENCHMARKS.iterdir() if path.is_dir()})


@pytest.mark.parametrize(
    "module",
    [
        "benchmarks.runtime.performance_profile",
        "benchmarks.algorithms.ipr_diagnosis",
        "benchmarks.engines.cpp_python_knapsack",
        "benchmarks.modeling.validation",
    ],
)
def test_grouped_benchmark_module_has_a_usable_entry_point(module: str) -> None:
    """Run module help using only the repository root as the working directory.

    Args:
        module (str): Grouped study module exposing a guarded argument parser.

    Raises:
        OSError: If the interpreter cannot be launched.
        subprocess.TimeoutExpired: If help unexpectedly starts a long-running study.
    """
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower(), result.stdout


def test_report_resolves_maintained_reference_and_asset_directories() -> None:
    """Resolve native-study evidence and outputs without assuming the report lives at the root."""
    report = importlib.import_module("benchmarks.engines.native_report")
    assert report.EVIDENCE == BENCHMARKS / "references" / "native-evidence.csv.gz"
    assert report.EVIDENCE.is_file()
    assert report.README == BENCHMARKS / "README.md"
    assert report.README.is_file()
    assert report.FIGURES == BENCHMARKS / "figures"
    assert report.FIGURES.is_dir()


@pytest.mark.parametrize("explicit", [False, True])
def test_new_knapsack_measurements_do_not_default_to_the_source_directory(
    monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    """Keep generated default output under results while honoring caller-selected paths.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces execution, not argument parsing or configuration.
        explicit (bool): Supply an explicit output path instead of using the default.
    """
    runner = importlib.import_module("benchmarks.engines.cpp_python_knapsack")
    observed: list[Path] = []

    def observe_run(benchmark: object, pilot: bool = False) -> None:
        """Capture the configured output without downloading data or compiling a benchmark.

        Args:
            benchmark (object): Real benchmark configured by the module entry point.
            pilot (bool): Whether the entry point requested the pilot mode.
        """
        assert isinstance(benchmark, runner.KnapsackBenchmark)
        assert not pilot
        observed.append(benchmark.output_path)

    monkeypatch.setattr(runner.KnapsackBenchmark, "run", observe_run)
    arguments = ["--output", "custom-trials.csv"] if explicit else []
    assert runner.main(arguments) == 0
    expected = (
        Path("custom-trials.csv") if explicit else Path("results/cpp-python-knapsack-reference.csv")
    )
    assert observed == [expected]
