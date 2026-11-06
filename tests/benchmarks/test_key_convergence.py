"""Checks for the model decoder convergence bench (baseline measurement, issue 125 phase 0).

Objective:
- Keep the convergence bench deterministic and honest before any new decoder exists.

Technical approach:
- Run tiny evaluation-budgeted searches through `RKO.from_model`; assert deterministic facts only.
- Check the summary on synthetic records so aggregation does not depend on a search.

Flow (pseudocode):
1. Run one fixture, one algorithm and one seed at a small evaluation budget.
2. Assert exact evaluation counts, repeatability and that no run beats the known optimum.
3. Aggregate synthetic records and check the summary fields.

Important criteria:
- Budgets are evaluation counts, never wall-clock; no assertion about search quality.

Inputs/Outputs:
- Input: structure fixtures, tiny budgets and fixed seeds.
- Output: assertions on record schema, repeatability and aggregation.
"""

import pytest
from benchmarks.modeling.key_convergence import KeyConvergenceStudy
from typing import Any

pytest.importorskip("ortools.math_opt.python.mathopt")

TINY: dict[str, Any] = {
    "names": ("coupled_equality", "set_cover"),
    "algorithms": ("SA",),
    "budgets": (50,),
    "seeds": (0,),
}


def test_runs_use_the_exact_evaluation_budget_and_never_beat_the_optimum() -> None:
    """Each record spends exactly the budget and its gap is not below the known optimum."""
    records = KeyConvergenceStudy.run(**TINY)
    assert {r["case"] for r in records} == set(TINY["names"])
    for record in records:
        assert record["evaluations"] == 50
        assert record["algorithm"] == "SA" and record["budget"] == 50 and record["seed"] == 0
        if record["feasible"]:
            assert record["gap"] >= -1e-9
            assert record["seconds_to_first_feasible"] is not None
        else:
            assert record["gap"] is None


def test_same_seed_and_budget_repeat_exactly() -> None:
    """Evaluation-budgeted runs are deterministic for the objective found."""
    first = KeyConvergenceStudy.run(**TINY)
    second = KeyConvergenceStudy.run(**TINY)
    assert [(r["case"], r["feasible"], r["gap"]) for r in first] == [
        (r["case"], r["feasible"], r["gap"]) for r in second
    ]


def test_summary_aggregates_per_case_algorithm_and_budget() -> None:
    """Feasible-run counts and gap statistics ignore infeasible runs."""
    base = {"case": "c", "algorithm": "SA", "budget": 10, "evaluations": 10}
    records = [
        {**base, "seed": 0, "feasible": True, "gap": 0.5, "seconds_to_first_feasible": 0.1},
        {**base, "seed": 1, "feasible": True, "gap": 0.1, "seconds_to_first_feasible": 0.3},
        {**base, "seed": 2, "feasible": False, "gap": None, "seconds_to_first_feasible": None},
    ]
    (row,) = KeyConvergenceStudy.summarize(records)
    assert (row["case"], row["algorithm"], row["budget"]) == ("c", "SA", 10)
    assert (row["runs"], row["feasible_runs"]) == (3, 2)
    assert row["best_gap"] == pytest.approx(0.1)
    assert row["median_gap"] == pytest.approx(0.3)
    assert row["median_seconds_to_first_feasible"] == pytest.approx(0.2)


def test_summary_without_feasible_runs_has_no_gap() -> None:
    """A group where nothing was feasible reports no gap instead of a number."""
    records = [
        {
            "case": "c",
            "algorithm": "SA",
            "budget": 10,
            "seed": 0,
            "evaluations": 10,
            "feasible": False,
            "gap": None,
            "seconds_to_first_feasible": None,
        }
    ]
    (row,) = KeyConvergenceStudy.summarize(records)
    assert row["feasible_runs"] == 0
    assert row["best_gap"] is None and row["median_gap"] is None
