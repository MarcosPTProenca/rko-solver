"""Learned priority study contract tests.

Objective:
- Verify seed isolation, allowance accounting, the pre-registered decision rule and a complete
  tiny run through the real solver.

Technical approach:
- Check the design and accounting directly, feed synthetic paired statistics to the verdict, and
  run a minimal study end to end with sequential execution.

Flow (pseudocode):
1. Validate disjoint seed ranges and allowance arithmetic.
2. Build synthetic comparison tables for each decision branch.
3. Run a tiny study and check every record.

Important criteria:
- Overlapping seed ranges are rejected.
- Advancement needs every five-constraint interval below zero and no plain-Knapsack loss.
- Every decoded result is re-verified against the solver's reported cost.

Inputs/Outputs:
- Input: design settings, synthetic statistics and a tiny study.
- Output: assertions on validation, verdicts and trial records.
"""

import pytest
from benchmarks.engines.learned_priority_study import ARMS
from benchmarks.engines.learned_priority_study import MINIMUM_SEARCH
from benchmarks.engines.learned_priority_study import PAIRS
from benchmarks.engines.learned_priority_study import PriorityDecision
from benchmarks.engines.learned_priority_study import PriorityStudy
from benchmarks.engines.learned_priority_study import StudyDesign
from benchmarks.engines.packing_problems import PackingFamily
from typing import Any

BUDGETS = ("2s", "10s")


def entry(low: float, high: float) -> dict[str, float]:
    """Build one paired statistic with a difference at the interval midpoint.

    Args:
        low (float): Lower interval bound.
        high (float): Upper interval bound.

    Returns:
        dict[str, float]: Statistic record.
    """
    return {"difference": (low + high) / 2, "ci_low": low, "ci_high": high}


def tables(
    mkp: tuple[float, float], knapsack: tuple[float, float]
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Build identical statistics for every pair and budget of each family.

    Args:
        mkp (tuple[float, float]): Interval for every five-constraint comparison.
        knapsack (tuple[float, float]): Interval for every plain-Knapsack comparison.

    Returns:
        dict[str, dict[str, dict[str, dict[str, float]]]]: Nested statistics.
    """
    return {
        kind: {budget: {f"{a}-vs-{b}": entry(*interval) for a, b in PAIRS} for budget in BUDGETS}
        for kind, interval in (("mkp", mkp), ("knapsack", knapsack))
    }


def test_default_design_has_disjoint_seed_ranges() -> None:
    """The shipped training, validation and test seeds never overlap."""
    design = StudyDesign()
    train = range(*design.train_seeds)
    validation = range(design.validation_base, design.validation_base + design.validation_instances)
    test = range(design.test_base, design.test_base + design.test_instances)
    assert not set(train) & set(validation) and not set(train) & set(test)
    assert not set(validation) & set(test)


@pytest.mark.parametrize(
    "overrides",
    [
        {"train_seeds": (5900, 6100)},
        {"train_seeds": (6900, 7100)},
        {"test_base": 6003},
    ],
)
def test_overlapping_seed_ranges_are_rejected(overrides: dict[str, Any]) -> None:
    """Any train, validation or test overlap is an error.

    Args:
        overrides (dict[str, Any]): Design overrides that create an overlap.
    """
    with pytest.raises(ValueError, match="disjoint"):
        StudyDesign(**overrides)


def test_allowance_subtracts_preparation_with_a_floor() -> None:
    """Score computation is charged to the search; the search never reaches zero."""
    assert PriorityStudy.search_allowance(2.0, 0.25) == pytest.approx(1.75)
    assert PriorityStudy.search_allowance(2.0, 5.0) == MINIMUM_SEARCH


def test_verdict_advances_only_when_every_condition_holds() -> None:
    """All five-constraint intervals below zero and no Knapsack loss advance the study."""
    win, tie, loss = (-0.5, -0.1), (-0.2, 0.3), (0.1, 0.6)
    assert PriorityDecision.verdict(tables(win, tie))["advance"] is True
    assert PriorityDecision.verdict(tables(tie, tie))["advance"] is False
    assert PriorityDecision.verdict(tables(win, loss))["advance"] is False
    verdict = PriorityDecision.verdict(tables(win, loss))
    assert verdict["mkp_conditions_hold"] is True and verdict["knapsack_loss"] is True


def test_one_failed_five_constraint_interval_blocks_advancement() -> None:
    """A single comparison whose interval touches zero is enough to stop."""
    data = tables((-0.5, -0.1), (-0.2, 0.3))
    data["mkp"]["10s"]["learned_resid-vs-hand_resid"] = entry(-0.4, 0.05)
    verdict = PriorityDecision.verdict(data)
    assert verdict["advance"] is False and verdict["mkp_conditions_hold"] is False


def test_tiny_study_runs_end_to_end_through_the_real_solver() -> None:
    """A minimal study trains, validates and solves every arm with verified costs."""
    design = StudyDesign(
        n_items=20,
        budgets=(0.2, 0.3),
        test_instances=1,
        validation_instances=1,
        steps=3,
        batch=2,
        samples=2,
        hidden=4,
        execution="sequential",
        algorithms=("SA",),
    )
    report = PriorityStudy(design).run(0)
    rows = report["trials"]
    assert len(rows) == len(PackingFamily.KINDS) * len(design.budgets) * len(ARMS)
    assert all(row["verified"] for row in rows)
    assert all(row["allowance"] <= row["budget"] + 1e-9 for row in rows)
    assert all(row["gap"] >= 0.0 for row in rows)
    assert {r["arm"] for r in rows} == set(ARMS)
    plain = [row for row in rows if row["arm"] == "plain"]
    assert all(row["prep_seconds"] == 0.0 for row in plain)
    assert set(report["training"]) == set(PackingFamily.KINDS)
    assert "advance" in report["summary"]["verdict"]
