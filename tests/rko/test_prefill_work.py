"""Check the matched-work startup diagnostic independently of quality outcomes.

Objective:
- Ensure both policies pay for all startup and search observations equally.

Technical approach:
- Import the optional benchmark from the repository, not the runtime package.
- Repeat tiny real public serial trials and compare deterministic records.

Flow (pseudocode):
1. Run each policy against numerical, permutation and tied objectives.
2. Reconcile completed objectives, startup allocation and rejected decoder attempt.
3. Repeat paired trials to check fixed-clock reproducibility and seed controls.

Important criteria:
- No speed or statistical superiority assertion is encoded in these tests.
- Invalid bounds fail instead of creating a stalled or incomparable experiment.

Inputs/Outputs:
- Input: small fixed-work trial settings.
- Output: accounting, reproducibility and validation assertions.
"""

import importlib
import json
import numpy as np
import pickle
import pytest
import random
from pathlib import Path
from types import ModuleType


@pytest.fixture
def study(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the standalone runner without adding a runtime dependency.

    Args:
        monkeypatch (pytest.MonkeyPatch): Temporarily expose maintained benchmark code.

    Returns:
        ModuleType: Bounded matched-work study.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    return importlib.import_module("prefill_work")


@pytest.mark.parametrize("dimension", [1, 4])
@pytest.mark.parametrize("algorithm", ["IPR", "GRASP", "MultiStart"])
@pytest.mark.parametrize("workload", ["sphere", "rugged", "ordering", "flat"])
def test_trial_pays_for_all_work(
    study: ModuleType,
    algorithm: str,
    workload: str,
    dimension: int,
) -> None:
    """Keep budget accounting and seed controls independent of observed quality.

    Args:
        study (ModuleType): Diagnostic runner.
        algorithm (str): Stock search implementation.
        workload (str): Pure objective family.
        dimension (int): Single-gene edge case or a small vector.
    """
    before = pickle.dumps((random.getstate(), np.random.get_state()))
    pair = [
        study.Trial(algorithm, workload, 200, dimension, 64, 3, prefill).run()
        for prefill in (False, True)
    ]
    for row in pair:
        assert row["evaluations"] == row["startup_evaluations"] + row["search_evaluations"] == 64
        assert row["decoder_calls"] == 65 and row["unscored_decoder_calls"] == 1
        assert row["best_curve"][-1] == (64, row["cost"])
        assert row["gap"] >= 0
    assert pair[0]["startup_evaluations"] == 0
    assert 3 <= pair[1]["startup_evaluations"] <= 5
    assert pair[1]["cost"] <= pair[1]["startup_best_cost"]
    assert pair[1]["search_improved_startup"] == (pair[1]["cost"] < pair[1]["startup_best_cost"])
    assert pair[0]["first_search_sha256"] == pair[1]["first_search_sha256"]
    assert pair[0]["search_seed"] == pair[1]["search_seed"]
    assert pickle.dumps((random.getstate(), np.random.get_state())) == before


def test_trials_reproduce_without_timing_fields(study: ModuleType) -> None:
    """Repeat a real constructive search with identical frozen policy controls.

    Args:
        study (ModuleType): Diagnostic runner.
    """
    first = study.Trial("GRASP", "rugged", 201, 8, 128, 3, True).run()
    second = study.Trial("GRASP", "rugged", 201, 8, 128, 3, True).run()
    assert first == second
    assert not {"seconds", "cpu_seconds", "elapsed_seconds"}.intersection(first)


def test_reference_summaries_match_raw_paired_trials(study: ModuleType) -> None:
    """Validate the archived pilot's totals and outcome counts without rerunning it.

    Args:
        study (ModuleType): Runner location used to find the identified reference.
    """
    path = Path(study.__file__).with_name("prefill-work-reference.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    trials = [row for row in rows if row["kind"] == "trial"]
    summaries = [row for row in rows if row["kind"] == "summary"]
    assert len(rows) == 133 and len(trials) == 120 and len(summaries) == 12
    for row in trials:
        assert row["evaluations"] == row["startup_evaluations"] + row["search_evaluations"] == 256
        assert row["decoder_calls"] == 257 and row["termination"] == "evaluation_budget"
    for summary in summaries:
        selected = [
            row
            for row in trials
            if (row["algorithm"], row["workload"]) == (summary["algorithm"], summary["workload"])
        ]
        outcomes = {"better": 0, "tied": 0, "worse": 0}
        for seed in range(200, 205):
            pair = sorted(
                [row for row in selected if row["seed"] == seed], key=lambda row: row["prefill"]
            )
            assert len(pair) == 2 and [row["prefill"] for row in pair] == [False, True]
            assert pair[0]["first_search_sha256"] == pair[1]["first_search_sha256"]
            before, after = pair[0]["cost"], pair[1]["cost"]
            outcomes["better" if after < before else "worse" if after > before else "tied"] += 1
        assert all(summary[key] == count for key, count in outcomes.items())


def test_startup_prefix_matches_across_algorithm_and_workload(study: ModuleType) -> None:
    """Use the same raw starting slots before objective-dependent tie perturbations.

    Args:
        study (ModuleType): Diagnostic runner.
    """
    rows = [
        study.Trial(algorithm, workload, 200, 4, 64, 3, True).run()
        for algorithm, workload in (
            ("IPR", "sphere"),
            ("GRASP", "rugged"),
            ("MultiStart", "ordering"),
        )
    ]
    assert len({row["startup_seed_prefix_sha256"] for row in rows}) == 1


@pytest.mark.parametrize("dimension, budget, capacity", [(0, 64, 3), (4, 5, 3), (4, 64, 1)])
def test_invalid_study_bounds_fail(
    study: ModuleType,
    dimension: int,
    budget: int,
    capacity: int,
) -> None:
    """Reject configurations with no guaranteed post-startup objective budget.

    Args:
        study (ModuleType): Diagnostic runner.
        dimension (int): Candidate dimension.
        budget (int): Total completed-work cap.
        capacity (int): Shared pool capacity.
    """
    with pytest.raises(ValueError):
        study.Trial("IPR", "sphere", 200, dimension, budget, capacity, True)
