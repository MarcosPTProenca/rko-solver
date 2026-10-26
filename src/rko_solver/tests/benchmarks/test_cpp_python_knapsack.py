"""Statistical and reproducibility contracts for the Knapsack comparison.

Objective:
- Verify deterministic workload selection and comparison metrics.

Technical approach:
- Exercise pure benchmark helpers on small generated inputs and observations.

Flow (pseudocode):
1. Generate the pinned family sample twice and compare exact IDs.
2. Check objective normalization, sign-test and multiplicity-correction cases.
3. Parse duplicate target events and verify the earliest latency is retained.

Important criteria:
- Ties and malformed telemetry never create false statistical evidence.
- Input selection depends only on the fixed seed, not solver outcomes.

Inputs/Outputs:
- Input: deterministic seeds, metric arrays and process telemetry text.
- Output: assertions on exact selection and statistical results.
"""

import benchmarks.cpp_python_knapsack as benchmark_module
import csv
import hashlib
import pytest
import subprocess
from benchmarks.cpp_python_knapsack import CPP_PORTFOLIO
from benchmarks.cpp_python_knapsack import PYTHON_ALGORITHMS
from benchmarks.cpp_python_knapsack import InstanceCase
from benchmarks.cpp_python_knapsack import KnapsackBenchmark
from benchmarks.cpp_python_knapsack import SolverOutput
from benchmarks.cpp_python_knapsack import TrialRecord
from benchmarks.knapsack_utils import exact_profit
from pathlib import Path
from rko_solver.examples.run_kp_cpp_instances import KpInstance
from unittest.mock import Mock


def test_exact_profit_oracle_handles_capacity_constraints() -> None:
    """Compute a feasible optimum while selecting each item at most once.

    Returns:
        None: Verifies the exact profit for a small independent instance.
    """
    instance = KpInstance(3, 5, [3, 4, 5], [2, 3, 4], "toy")

    assert exact_profit(instance) == 7


def test_instance_selection_is_stable_and_spans_each_family() -> None:
    """Choose the same three valid file IDs from every family on each run.

    Returns:
        None: Verifies exact, deterministic sample membership.
    """
    first = KnapsackBenchmark.select_instance_ids(20261019)
    second = KnapsackBenchmark.select_instance_ids(20261019)

    assert first == second
    assert len(first) == 5
    assert all(len(ids) == 3 for ids in first.values())
    assert all(0 <= value < 100 for ids in first.values() for value in ids)
    assert first == {
        "00Uncorrelated": (41, 75, 92),
        "01WeaklyCorrelated": (15, 27, 60),
        "02StronglyCorrelated": (34, 37, 69),
        "05SubsetSum": (3, 44, 91),
        "07SpannerUncorrelated": (0, 17, 92),
    }


def test_normalized_objective_matches_feasible_optimality_gap() -> None:
    """Normalize the reference objective without discarding infeasible results.

    Returns:
        None: Verifies feasible and infeasible objective normalization.
    """
    feasible_score = KnapsackBenchmark.normalized_objective_difference(-90, 100)
    infeasible_score = KnapsackBenchmark.normalized_objective_difference(100010, 100)

    assert feasible_score == pytest.approx(10.0)
    assert infeasible_score == pytest.approx(100110.0)


def test_exact_sign_test_handles_ties_and_consistent_differences() -> None:
    """Use only non-tied instance blocks and compute exact two-sided p-values.

    Returns:
        None: Verifies tie handling and exact probability calculations.
    """
    tied = KnapsackBenchmark.exact_sign_test([0.0, 0.0])
    consistent = KnapsackBenchmark.exact_sign_test([1.0] * 15)
    mixed = KnapsackBenchmark.exact_sign_test([1.0] * 10 + [-1.0] * 5)

    assert tied == (1.0, 0, 0, 0)
    assert consistent == pytest.approx((2 / 2**15, 15, 15, 0))
    assert mixed[0] == pytest.approx(0.3017578125)
    assert mixed[1:] == (15, 10, 5)


def test_holm_adjustment_is_monotonic_and_preserves_names() -> None:
    """Correct a family of p-values while keeping the original solver labels.

    Returns:
        None: Verifies corrected values preserve comparison identifiers.
    """
    adjusted = KnapsackBenchmark.holm_adjust({"first": 0.01, "second": 0.04, "third": 0.03})

    assert adjusted == {
        "first": pytest.approx(0.03),
        "second": pytest.approx(0.06),
        "third": pytest.approx(0.06),
    }


def test_bootstrap_interval_is_reproducible() -> None:
    """Produce a stable percentile interval for a fixed paired-difference sample.

    Returns:
        None: Verifies deterministic confidence-interval endpoints.
    """
    values = [float(value) for value in range(15)]
    first = KnapsackBenchmark.bootstrap_median_interval(values, seed=17, samples=1000)
    second = KnapsackBenchmark.bootstrap_median_interval(values, seed=17, samples=1000)

    assert first == second
    assert first[0] <= 7.0 <= first[1]


def test_target_parser_keeps_the_earliest_event_and_ignores_other_output() -> None:
    """Parse target events from mixed process output without replacing earlier hits.

    Returns:
        None: Verifies only the earliest elapsed time for each target is retained.
    """
    output = (
        "startup line\n"
        "RKO_BENCH_TARGET gap=10 elapsed_seconds=0.800000000\n"
        "RKO_BENCH_TARGET gap=1 elapsed_seconds=1.200000000\n"
        "RKO_BENCH_TARGET gap=10 elapsed_seconds=0.500000000\n"
    )

    assert KnapsackBenchmark.parse_target_events(output) == {10: 0.5, 1: 1.2}


def test_target_parser_rejects_unknown_and_malformed_target_events() -> None:
    """Reject corrupt benchmark telemetry rather than silently scoring it.

    Returns:
        None: Verifies malformed and unrecognized events fail validation.
    """
    assert KnapsackBenchmark.parse_target_events("no target\n") == {}
    with pytest.raises(ValueError):
        KnapsackBenchmark.parse_target_events("RKO_BENCH_TARGET gap=2 elapsed_seconds=1.0\n")


def test_trial_record_recomputes_objective_and_retains_provenance(tmp_path: Path) -> None:
    """Verify an in-memory solution and serialize execution provenance.

    Args:
        tmp_path (Path): Isolated directory for the raw-results destination.

    Returns:
        None: Asserts objective, target, runtime, and host fields.
    """
    benchmark = KnapsackBenchmark(tmp_path / "trials.csv", tmp_path / "cache")
    instance = KpInstance(2, 3, [4, 5], [2, 3], "toy")
    case = InstanceCase(
        family="00Uncorrelated",
        relative_path="00Uncorrelated/toy.kp",
        path=tmp_path / "toy.kp",
        sha256="input-hash",
        instance=instance,
        optimum=5,
    )
    metadata = {
        "python_revision": "revision",
        "python_source_sha256": "source-hash",
        "python_version": "3.14",
        "numpy_version": "2.4",
        "platform": "Linux",
        "cpu_model": "test-cpu",
        "cpu_count": "4",
        "memory_total_gib": "16.00",
        "memory_available_gib": "8.00",
        "compiler": "g++ test",
        "cpp_flags": "-O3",
        "cpp_revision": "cpp-revision",
        "cpp_patch_sha256": "patch-hash",
    }
    output = SolverOutput(
        keys=(0.1, 0.9),
        objective=-5.0,
        events={10: 0.1, 1: 0.2, 0: 0.3},
        cpp_seed=None,
        elapsed_seconds=0.5,
        return_code=0,
        error=None,
    )

    record = benchmark._record_trial(
        output,
        case,
        "BRKGA",
        "python",
        0,
        17,
        2,
        "python/BRKGA/toy/r00",
        "signature",
        metadata,
    )
    first_signature = benchmark._run_signature([case], metadata, 2, 1)
    resumed_metadata = metadata | {
        "python_revision": "later-revision",
        "memory_available_gib": "3.00",
    }

    assert benchmark._run_signature([case], resumed_metadata, 2, 1) == first_signature
    assert record.status == "ok"
    assert record.profit == 5
    assert record.capacity == 3
    assert record.exact_optimum == 5
    assert record.feasible is True
    assert record.feasible_gap_percent == 0.0
    assert record.success_1_percent is True
    assert record.time_optimum_seconds == 0.3
    assert record.execution_mode == "python-serial"
    assert record.memory_total_gib == "16.00"
    assert record.cpp_flags == "-O3"


def test_preliminary_reference_remains_historical_and_complete() -> None:
    """Preserve all old two-second records without confusing them with the active long run."""
    source = Path(benchmark_module.__file__).with_name("cpp-python-knapsack-2s-preliminary.csv")
    content = source.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(content).hexdigest() == (
        "1663785c936720f6974fae4cff90bc9db669a9f21cfc2d405d1c003ff2b1e390"
    )
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({row["trial_id"] for row in rows}) == 10800
    assert {row["budget_seconds"] for row in rows} == {"2"}
    assert {row["status"] for row in rows} == {"ok"}
    assert {row["python_revision"] for row in rows} == {"759309a0d8a74324806faba6b41f806eb3721dbc"}


@pytest.mark.parametrize(
    "changed", ["src/rko_solver/parameters/config.toml", "benchmarks/knapsack_utils.py"]
)
def test_source_fingerprint_covers_parameters_and_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    """Reject resume identity when packaged parameters or the shared optimum helper changes.

    Args:
        tmp_path (Path): Isolated synthetic source tree.
        monkeypatch (pytest.MonkeyPatch): Point provenance collection at the synthetic runner.
        changed (str): Relevant source/configuration file to modify.
    """
    sources = (
        "src/rko_solver/solver.py",
        "src/rko_solver/parameters/config.toml",
        "benchmarks/__init__.py",
        "benchmarks/knapsack.py",
        "benchmarks/knapsack_utils.py",
        "benchmarks/cpp_python_knapsack.py",
        "pyproject.toml",
        "uv.lock",
    )
    for source in sources:
        target = tmp_path / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        benchmark_module, "__file__", str(tmp_path / "benchmarks/cpp_python_knapsack.py")
    )
    benchmark = KnapsackBenchmark(tmp_path / "trials.csv", tmp_path / "cache")
    before = benchmark._python_source_hash()
    (tmp_path / changed).write_text("value = 2\n", encoding="utf-8")
    assert benchmark._python_source_hash() != before


def test_process_timeout_scales_with_search_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Allow a budget-length trial plus bounded cooperative-stop overrun.

    Args:
        tmp_path (Path): Isolated working directory for the child process.
        monkeypatch (pytest.MonkeyPatch): Replaces the subprocess runner.

    Returns:
        None: Verifies the child timeout accommodates a 120-second trial.
    """
    benchmark = KnapsackBenchmark(tmp_path / "trials.csv", tmp_path, budget_seconds=120)
    process = subprocess.CompletedProcess(
        args=["fake"], returncode=0, stdout="", stderr="RKO_BENCH_SEED=17\n"
    )
    run = Mock(return_value=process)
    monkeypatch.setattr("benchmarks.cpp_python_knapsack.subprocess.run", run)

    result = benchmark._execute_process(["fake"], tmp_path, {}, started_ns=0)

    assert run.call_args.kwargs["timeout"] == 240
    assert result.return_code == 0


def test_cpp_configuration_uses_reference_algorithm_identifiers(tmp_path: Path) -> None:
    """Write solver names that the pinned C++ executable recognizes.

    Args:
        tmp_path (Path): Isolated directory for the generated config file.

    Returns:
        None: Verifies each single solver and the exact portfolio list.
    """
    benchmark = KnapsackBenchmark(tmp_path / "trials.csv", tmp_path / "cache")
    program_dir = tmp_path / "Program"
    (program_dir / "config").mkdir(parents=True)
    config_path = program_dir / "config" / "config_tests.conf"

    for solver, _ in PYTHON_ALGORITHMS:
        benchmark._write_cpp_config(program_dir, solver)
        assert config_path.read_text(encoding="utf-8").splitlines()[0] == solver

    benchmark._write_cpp_config(program_dir, "RKO-default")

    assert config_path.read_text(encoding="utf-8").splitlines()[: len(CPP_PORTFOLIO)] == list(
        CPP_PORTFOLIO
    )


def test_summary_counts_errors_as_rate_misses_and_budget_censoring(tmp_path: Path) -> None:
    """Retain failed trials as misses and cap their target times at budget.

    Args:
        tmp_path (Path): Isolated directory for raw and summary CSV files.

    Returns:
        None: Verifies failure rates, censored time, and trial counts.
    """
    benchmark = KnapsackBenchmark(tmp_path / "trials.csv", tmp_path / "cache")
    fieldnames = list(TrialRecord.__dataclass_fields__)
    rows = []
    for implementation, status in (("python", "ok"), ("cpp64", "error")):
        row = dict.fromkeys(fieldnames, "")
        row.update(
            solver="BRKGA",
            implementation=implementation,
            instance_sha256="same-instance",
            budget_seconds="2",
            status=status,
            elapsed_seconds="1.0" if implementation == "python" else "3.0",
            run_signature="same-plan",
        )
        if status == "ok":
            row.update(
                feasible="True",
                success_1_percent="True",
                time_1_percent_seconds="0.5",
                normalized_objective_difference="10.0",
            )
        rows.append(row)
    with benchmark.output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    benchmark._write_summary("same-plan")

    with benchmark.output_path.with_name("trials-summary.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        summary = {row["metric"]: row for row in csv.DictReader(stream)}
    assert summary["feasible_rate"]["cpp64_median_across_instances"] == "0.0"
    assert summary["success_1_percent_rate"]["cpp64_median_across_instances"] == "0.0"
    assert summary["time_1_percent_seconds"]["cpp64_median_across_instances"] == "2.0"
    assert summary["time_1_percent_seconds"]["cpp64_failed_trials"] == "1"
    assert summary["time_1_percent_seconds"]["python_successful_trials"] == "1"
    assert summary["elapsed_seconds"]["cpp64_median_across_instances"] == "3.0"
