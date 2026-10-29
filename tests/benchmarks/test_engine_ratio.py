"""Contract tests for the Rust/Python throughput ratio gate.

Objective:
- Verify that the comparison rule fails only on a drop larger than the allowance and that the
  measurement produces a well-formed report.

Technical approach:
- Exercise the pure comparison with synthetic ratios, then run a very short real measurement.

Flow (pseudocode):
1. Compare synthetic reports at, inside and outside the allowed drop.
2. Measure one algorithm for a few milliseconds and check the report fields.
3. Run the command line against a reference written from that measurement.

Important criteria:
- A higher ratio never fails; a missing algorithm always does.
- The short real run asserts structure and positivity, never a speed.

Inputs/Outputs:
- Input: synthetic ratio mappings and a 0.05 second measurement.
- Output: exact comparison messages and report structure.
"""

import json
import pytest
from benchmarks.engines.engine_ratio import EngineRatio
from pathlib import Path

native = pytest.importorskip("rko_solver._native")


def report(ratio: float) -> dict[str, dict[str, float]]:
    """Build a one-algorithm report with the given ratio.

    Args:
        ratio (float): Rust over Python throughput.

    Returns:
        dict[str, dict[str, float]]: Report mapping.
    """
    return {"SA": {"ratio": ratio}}


def test_only_a_drop_beyond_the_allowance_is_a_regression() -> None:
    """The boundary is inclusive, improvements pass and a missing algorithm fails."""
    reference = report(40.0)
    assert EngineRatio.compare(report(40.0), reference, 0.30) == []
    assert EngineRatio.compare(report(28.0), reference, 0.30) == []
    assert EngineRatio.compare(report(100.0), reference, 0.30) == []
    failed = EngineRatio.compare(report(27.9), reference, 0.30)
    assert len(failed) == 1 and "SA" in failed[0] and "27.9" in failed[0]
    assert EngineRatio.compare({}, reference, 0.30) == ["SA: missing from the current measurement"]
    assert EngineRatio.compare(report(27.9), reference, 0.50) == []


def test_a_short_measurement_has_the_documented_fields_and_positive_rates() -> None:
    """The real measurement returns rates, a ratio and spreads for the requested algorithm."""
    result = EngineRatio().measure(["SA"], seconds=0.05, repeats=1)
    entry = result["SA"]
    assert set(entry) == {
        "python_per_second",
        "rust_per_second",
        "ratio",
        "python_spread",
        "rust_spread",
    }
    assert entry["python_per_second"] > 0 and entry["rust_per_second"] > 0
    assert entry["ratio"] == pytest.approx(entry["rust_per_second"] / entry["python_per_second"])


def test_simulated_annealing_is_measured_with_the_historical_parameters_the_reference_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reference stays valid when the packaged SA preset changes.

    Args:
        monkeypatch (pytest.MonkeyPatch): Capture the settings passed to the solver.
    """
    import benchmarks.engines.engine_ratio as module

    seen: list[dict[str, object]] = []

    class Spy:
        """Record solver settings and return a minimal search report."""

        def __init__(self, *args: object, **options: object) -> None:
            """Store the keyword settings.

            Args:
                args (object): Ignored problem arguments.
                options (object): Solver settings under inspection.
            """
            seen.append(options)

        def solve(self, seconds: float) -> object:
            """Return one finished search.

            Args:
                seconds (float): Ignored budget.

            Returns:
                object: Object with the fields `rate` reads.
            """
            from types import SimpleNamespace

            return SimpleNamespace(
                searches=[SimpleNamespace(n_evaluations=10, elapsed_seconds=1.0)]
            )

    monkeypatch.setattr(module, "RKO", Spy)
    ratio = EngineRatio()
    ratio.rate("rust", "SA", 0.01)
    ratio.rate("rust", "BRKGA", 0.01)
    assert seen[0]["hyperparameters"] == {
        "SA": {"sa_max": 1000, "t0": 1000000.0, "alpha": 0.99, "beta_min": 0.01, "beta_max": 0.05}
    }
    assert "hyperparameters" not in seen[1]


def test_command_line_gates_the_known_processor_and_only_notices_an_unknown_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The entry point exits 0 within tolerance, 1 on a regression and 0 with a notice otherwise.

    Args:
        tmp_path (Path): Directory for the reference files.
        capsys (pytest.CaptureFixture[str]): Captured output.
    """
    arguments = ["--seconds", "0.05", "--repeats", "1", "--algorithms", "SA"]
    assert EngineRatio.main(arguments) == 0
    measured = json.loads(capsys.readouterr().out)
    assert set(measured) == {"schema_version", "environment", "algorithms"}
    cpu = measured["environment"]["cpu_model"]

    def reference(name: str, model: str, ratio: float) -> Path:
        """Write a reference holding one processor model and one ratio.

        Args:
            name (str): File name.
            model (str): Processor model key.
            ratio (float): Reference ratio for SA.

        Returns:
            Path: The written file.
        """
        path = tmp_path / name
        content = {"models": {model: {"algorithms": {"SA": {"ratio": ratio}}}}}
        path.write_text(json.dumps(content), encoding="utf-8")
        return path

    assert (
        EngineRatio.main([*arguments, "--reference", str(reference("easy.json", cpu, 0.001))]) == 0
    )
    assert EngineRatio.main([*arguments, "--reference", str(reference("hard.json", cpu, 1e9))]) == 1
    assert "REGRESSION SA" in capsys.readouterr().err
    other = reference("other.json", "some other processor", 1e9)
    assert EngineRatio.main([*arguments, "--reference", str(other)]) == 0
    assert "NOTICE no reference for processor" in capsys.readouterr().err
