"""Paired baseline-report contracts.

Objective:
- Reject misleading fixed-work evidence before summarizing diagnostic profiles.

Technical approach:
- Use hand-worked plain/profiled records with known times and fingerprints.

Flow (pseudocode):
1. Supply matching fixed-work records in either order.
2. Verify timing scopes remain separate and invalid evidence is rejected.

Important criteria:
- No profiling speedup or C++ performance claim is inferred.

Inputs/Outputs:
- Input: tiny versioned diagnostic records.
- Output: exact summary assertions and validation failures.
"""

import copy
import hashlib
import json
import pytest
import subprocess
import sys
from benchmarks.runtime.performance_report import PerformanceReport
from pathlib import Path


@pytest.fixture
def records() -> list[dict[str, object]]:
    """Provide one independently specified pair with distinct timing scopes.

    Returns:
        list[dict[str, object]]: Plain and profiled records for the same search.
    """
    plain = {
        "schema_version": 1,
        "scope": "serial_fixed_work_search_only",
        "profiled": False,
        "profile_scope": None,
        "profile_total_seconds": None,
        "seed_pool_evaluations": 0,
        "provenance": {"source_sha256": "a" * 64, "python": "3.14", "numpy": "2.4"},
        "result": {
            "algorithm": "SA",
            "workload": "sphere",
            "dimension": 20,
            "seed": 200,
            "evaluations": 1024,
            "seconds": 2.0,
            "cpu_seconds": 1.5,
            "cost": 0.25,
            "keys_sha256": "b" * 64,
            "pool_sha256": "c" * 64,
            "rng_sha256": "d" * 64,
        },
        "profile": [],
    }
    profiled = copy.deepcopy(plain)
    profiled.update(
        profiled=True,
        profile_scope="fixed_work_runner_including_construction_and_seed_pool",
        profile_total_seconds=10.0,
        profile=[
            {
                "file": "~",
                "line": 0,
                "function": "native_operation",
                "exclusive_seconds": 6.0,
                "inclusive_seconds": 8.0,
            }
        ],
    )
    profiled["result"].update(seconds=4.0, cpu_seconds=3.0)
    return [plain, profiled]


def test_report_separates_search_timing_from_diagnostic_fractions(
    records: list[dict[str, object]],
) -> None:
    """Summarize the unprofiled search separately from full-runner profiling.

    Args:
        records (list[dict[str, object]]): Known paired evidence.

    Raises:
        AssertionError: If timing scopes or function shares are conflated.
    """
    summary = PerformanceReport.summarize(list(reversed(records)))
    group = summary["groups"][0]

    assert summary["pairs"] == 1
    assert summary["trials"] == 2
    assert group["unprofiled_search_median_seconds"] == 2.0
    assert group["profile_runner_median_seconds"] == 10.0
    assert group["top_functions"][0]["median_exclusive_fraction"] == 0.6
    assert group["top_functions"][0]["observations"] == 1
    assert group["seeds"] == [200]


@pytest.mark.parametrize("field", ["cost", "keys_sha256", "pool_sha256", "rng_sha256"])
def test_changed_replay_is_not_summarized(records: list[dict[str, object]], field: str) -> None:
    """Reject different trajectories rather than producing a plausible report.

    Args:
        records (list[dict[str, object]]): Matching records to corrupt.
        field (str): Result fingerprint or objective to change.

    Raises:
        AssertionError: If mismatched replay is accepted.
    """
    records[1]["result"][field] = 0.5 if field == "cost" else "e" * 64
    with pytest.raises(ValueError, match="replay"):
        PerformanceReport.summarize(records)


@pytest.mark.parametrize("rows", [[], [0], [0, 0], [0, 1, 1]])
def test_missing_and_duplicate_pairs_fail(
    records: list[dict[str, object]], rows: list[int]
) -> None:
    """Require exactly one plain/profiled observation per workload and seed.

    Args:
        records (list[dict[str, object]]): Known matching observations.
        rows (list[int]): Empty, incomplete or duplicate selection.

    Raises:
        AssertionError: If incomplete evidence is accepted.
    """
    with pytest.raises(ValueError):
        PerformanceReport.summarize([records[index] for index in rows])


@pytest.mark.parametrize(
    "field,value", [("source_sha256", "e" * 64), ("python", "other"), ("numpy", "other")]
)
def test_mixed_sources_and_runtimes_fail(
    records: list[dict[str, object]], field: str, value: str
) -> None:
    """Reject data that differ in source or runtime identity.

    Args:
        records (list[dict[str, object]]): Matching records to alter.
        field (str): Provenance component to change.
        value (str): Different identity value.

    Raises:
        AssertionError: If incomparable records are pooled.
    """
    records[1]["provenance"][field] = value
    with pytest.raises(ValueError, match="Mixed"):
        PerformanceReport.summarize(records)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_times_fail(records: list[dict[str, object]], value: float) -> None:
    """Reject invalid timing values rather than computing plausible medians.

    Args:
        records (list[dict[str, object]]): Diagnostic observations to alter.
        value (float): Negative or nonfinite time.

    Raises:
        AssertionError: If invalid evidence reaches aggregation.
    """
    records[0]["result"]["seconds"] = value
    with pytest.raises(ValueError):
        PerformanceReport.summarize(records)


@pytest.mark.parametrize(
    "field,value",
    [
        ("profile_total_seconds", 0.0),
        ("profile_total_seconds", 5.0),
        ("profile_scope", "search_only"),
        ("profiled", 1),
    ],
)
def test_invalid_profile_accounting_fails(
    records: list[dict[str, object]], field: str, value: object
) -> None:
    """Require a valid full-runner denominator and explicit profile mode.

    Args:
        records (list[dict[str, object]]): Matching observations to alter.
        field (str): Profile metadata component to change.
        value (object): Invalid denominator, scope or mode.

    Raises:
        AssertionError: If invalid profile fractions could be reported.
    """
    records[1][field] = value
    with pytest.raises(ValueError):
        PerformanceReport.summarize(records)


def test_missing_truncated_functions_are_not_assigned_zero(
    records: list[dict[str, object]],
) -> None:
    """Rank only functions observed in every seed's truncated profile.

    Args:
        records (list[dict[str, object]]): One pair to extend with a different top list.

    Raises:
        AssertionError: If a missing observation becomes an invented zero cost.
    """
    second = copy.deepcopy(records)
    for row in second:
        row["result"]["seed"] = 201
    second[0]["result"]["seconds"] = 6.0
    second[1]["profile"][0]["function"] = "other_operation"
    summary = PerformanceReport.summarize(records + second)

    assert summary["groups"][0]["pairs"] == 2
    assert summary["groups"][0]["unprofiled_search_median_seconds"] == 4.0
    assert summary["groups"][0]["top_functions"] == []


def test_cli_reads_shards_and_writes_a_validated_summary(
    records: list[dict[str, object]], tmp_path: Path
) -> None:
    """Write an actual compact report through the public CLI.

    Args:
        records (list[dict[str, object]]): Known diagnostic observations.
        tmp_path (Path): Temporary input and output locations.

    Raises:
        AssertionError: If the emitted summary is incomplete.
        subprocess.CalledProcessError: If the report command fails.
    """
    inputs = tmp_path / "raw"
    inputs.mkdir()
    (inputs / "shard.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8"
    )
    output = tmp_path / "report" / "summary.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.runtime.performance_report",
            "--input-dir",
            str(inputs),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["pairs"] == 1
    assert summary["groups"][0]["unprofiled_search_median_seconds"] == 2.0
    assert (
        summary["inputs_sha256"]["shard.jsonl"]
        == hashlib.sha256((inputs / "shard.jsonl").read_bytes()).hexdigest()
    )
    assert len(summary["analysis_sha256"]) == 64
