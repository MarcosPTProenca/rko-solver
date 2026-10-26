"""Fixed-work diagnostic CLI contracts.

Objective:
- Verify usable baseline records without asserting hardware-dependent speed.

Technical approach:
- Execute the diagnostic CLI with a tiny real search and parse its JSON output.

Flow (pseudocode):
1. Execute four completed objective evaluations.
2. Check reproducibility metadata and explicitly restricted measurement scope.

Important criteria:
- Functional smoke tests are not performance comparisons.

Inputs/Outputs:
- Input: a bounded four-key sphere search.
- Output: assertions on the public JSON record.
"""

import benchmarks.performance_profile as benchmark_module
import hashlib
import json
import numpy as np
import pickle
import pytest
import random
import subprocess
import sys
from benchmarks.performance_profile import PerformanceProbe
from pathlib import Path


def test_cli_records_fixed_work_and_baseline_provenance() -> None:
    """Emit a source-identified search record with an explicit measurement scope.

    Raises:
        AssertionError: If work, provenance or measurement limits are missing.
        subprocess.CalledProcessError: If the diagnostic command fails.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.performance_profile",
            "--dimension",
            "4",
            "--evaluations",
            "4",
            "--seed",
            "7",
        ],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    row = json.loads(completed.stdout)

    assert row["schema_version"] == 1
    assert row["scope"] == "serial_fixed_work_search_only"
    assert row["profiled"] is False
    assert row["profile"] == []
    assert row["result"]["evaluations"] == 4
    assert row["result"]["dimension"] == 4
    assert row["result"]["seed"] == 7
    assert len(row["result"]["keys_sha256"]) == 64
    assert row["provenance"]["python"]
    assert row["provenance"]["numpy"]
    assert row["provenance"]["source_sha256"]
    assert row["provenance"]["files_sha256"]["uv.lock"]
    assert row["provenance"]["native_thread_environment"] is not None


def test_profile_preserves_replay_and_caller_random_streams() -> None:
    """Expose diagnostic function costs without changing replay or caller RNG state.

    Raises:
        AssertionError: If profiling changes results or consumes caller RNG streams.
    """
    before = pickle.dumps((random.getstate(), np.random.get_state()))
    plain = PerformanceProbe.run(dimension=4, evaluations=8, seed=7)
    profiled = PerformanceProbe.run(dimension=4, evaluations=8, seed=7, profile=True)

    for field in ("cost", "keys_sha256", "pool_sha256", "rng_sha256", "evaluations"):
        assert profiled["result"][field] == plain["result"][field]
    assert pickle.dumps((random.getstate(), np.random.get_state())) == before
    assert profiled["profiled"] is True
    assert plain["profile_total_seconds"] is None
    assert (
        profiled["profile_total_seconds"]
        >= sum(row["exclusive_seconds"] for row in profiled["profile"])
        > 0
    )
    assert profiled["profile_scope"] == "fixed_work_runner_including_construction_and_seed_pool"
    assert 1 <= len(profiled["profile"]) <= 20
    assert all(row["exclusive_seconds"] >= 0 for row in profiled["profile"])
    assert all(row["calls"] >= row["primitive_calls"] > 0 for row in profiled["profile"])


@pytest.mark.parametrize("algorithm", ["GA", "BRKGA", "GRASP", "MultiStart", "IPR"])
def test_supported_algorithms_obey_the_search_evaluation_cap(algorithm: str) -> None:
    """Stop real diagnostic algorithms at the same completed search work.

    Args:
        algorithm (str): Algorithm with distinct construction or pool behavior.

    Raises:
        AssertionError: If search work or separate seeded-pool accounting is wrong.
    """
    row = PerformanceProbe.run(algorithm=algorithm, dimension=4, evaluations=4, seed=7)

    assert row["result"]["evaluations"] == 4
    assert row["seed_pool_evaluations"] == (10 if algorithm == "IPR" else 0)


@pytest.mark.parametrize("workload", ["rugged", "ordering"])
def test_supported_workloads_replay_deterministically(workload: str) -> None:
    """Keep deterministic result fingerprints for each non-sphere objective.

    Args:
        workload (str): Rugged numeric or stable permutation workload.

    Raises:
        AssertionError: If equal source, work and seeds produce different results.
    """
    first = PerformanceProbe.run(workload=workload, dimension=4, evaluations=4, seed=7)
    second = PerformanceProbe.run(workload=workload, dimension=4, evaluations=4, seed=7)

    assert first["result"]["keys_sha256"] == second["result"]["keys_sha256"]
    assert first["result"]["cost"] == second["result"]["cost"]


@pytest.mark.parametrize(
    "arguments",
    [
        {"dimension": 0},
        {"dimension": True},
        {"evaluations": -1},
        {"evaluations": 1.5},
        {"seed": -1},
        {"seed": True},
        {"seed": 2**32},
        {"algorithm": "IPR", "seed": 2**32 - 1000},
        {"algorithm": "missing"},
        {"workload": "missing"},
        {"profile": 1},
    ],
)
def test_invalid_work_is_rejected(arguments: dict[str, object]) -> None:
    """Reject unsupported diagnostic inputs rather than silently altering work.

    Args:
        arguments (dict[str, object]): Invalid public diagnostic arguments.

    Raises:
        AssertionError: If an invalid workload reaches execution.
    """
    with pytest.raises(ValueError):
        PerformanceProbe.run(**arguments)


def test_provenance_detects_helper_and_lockfile_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identify execution-affecting helper and dependency changes without Git.

    Args:
        tmp_path (Path): Temporary source layout outside the repository.
        monkeypatch (pytest.MonkeyPatch): Relocates the script's filesystem boundary.

    Raises:
        AssertionError: If changed helper or lockfile bytes retain an identity.
    """
    scripts = tmp_path / "benchmarks"
    scripts.mkdir()
    helper = scripts / "fixed_work.py"
    helper.write_text("first helper", encoding="utf-8")
    lock = tmp_path / "uv.lock"
    lock.write_text("first lock", encoding="utf-8")
    monkeypatch.setattr(benchmark_module, "__file__", str(scripts / "performance_profile.py"))
    first = PerformanceProbe.provenance()
    assert first["git"]["revision"] is None
    assert first["files_sha256"]["uv.lock"] == hashlib.sha256(b"first lock").hexdigest()

    helper.write_text("second helper", encoding="utf-8")
    second = PerformanceProbe.provenance()
    lock.write_text("second lock", encoding="utf-8")
    third = PerformanceProbe.provenance()

    assert first["source_sha256"] != second["source_sha256"] != third["source_sha256"]
    assert third["source_sha256"] == PerformanceProbe.provenance()["source_sha256"]
