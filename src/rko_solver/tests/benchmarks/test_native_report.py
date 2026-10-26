"""Statistical contracts of the native benchmark report.

Objective:
- Prove that the exact test, correction, effect size, winner rule and generated outputs
  behave as documented, so a printed verdict can be trusted.

Technical approach:
- Hand-worked small samples with known exact p-values, plus synthetic evidence rows for
  the round logic, and a consistency check that the committed README block is current.

Flow (pseudocode):
1. Check the signed-rank test and Holm correction on worked examples.
2. Build synthetic rounds with a clear winner, a tie and too few instances.
3. Compare the committed outputs with a fresh render of the committed evidence.

Important criteria:
- A nonsignificant difference is never reported as a win.

Inputs/Outputs:
- Input: tiny arrays and synthetic evidence rows.
- Output: exact assertions.
"""

import numpy as np
import pytest
from benchmarks.native_report import EVIDENCE
from benchmarks.native_report import FIGURES
from benchmarks.native_report import README
from benchmarks.native_report import ROUNDS
from benchmarks.native_report import NativeReport


def synthetic(
    key: str, family: str, gaps: dict[str, list[float]], solvers: tuple[str, ...]
) -> list[dict[str, str]]:
    """Build evidence rows with one replicate per instance.

    Args:
        key (str): Round key.
        family (str): Problem family.
        gaps (dict[str, list[float]]): Score per instance for each implementation.
        solvers (tuple[str, ...]): Solvers sharing the same scores.

    Returns:
        list[dict[str, str]]: Evidence rows.
    """
    rows = []
    for implementation, values in gaps.items():
        for solver in solvers:
            for index, value in enumerate(values):
                rows.append(
                    {
                        "round": key,
                        "family": family,
                        "instance": f"i{index:02d}",
                        "solver": solver,
                        "implementation": implementation,
                        "replicate": "0",
                        "budget_seconds": "2",
                        "elapsed_seconds": "2.0",
                        "objective": str(value),
                        "score": str(value),
                    }
                )
    return rows


def test_signed_rank_matches_worked_exact_values() -> None:
    """All-positive samples give 2 / 2**n, and zeros are discarded."""
    p_value, effect = NativeReport.signed_rank_p(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert p_value == pytest.approx(2 / 32)
    assert effect == 1.0
    p_value, effect = NativeReport.signed_rank_p(np.array([0.0, 0.0, -1.0, -2.0, -3.0]))
    assert p_value == pytest.approx(2 / 8)
    assert effect == -1.0
    assert NativeReport.signed_rank_p(np.zeros(4)) == (1.0, 0.0)
    p_value, effect = NativeReport.signed_rank_p(np.array([1.0, -1.0]))
    assert p_value == 1.0
    assert effect == 0.0


def test_signed_rank_handles_tied_magnitudes_exactly() -> None:
    """Ties receive average ranks and the permutation distribution stays exact."""
    p_value, _ = NativeReport.signed_rank_p(np.array([2.0, 2.0, 2.0, -2.0]))
    assert p_value == pytest.approx(0.625)


def test_holm_is_monotone_and_capped() -> None:
    """Holm multiplies by the remaining count and enforces monotonicity."""
    adjusted = NativeReport.holm([0.01, 0.04, 0.03, 0.5])
    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.5])


def test_clear_winner_tie_and_underpowered_rounds() -> None:
    """The unbeaten contender wins, equal scores tie and five instances stay indicative."""
    spec = next(item for item in ROUNDS if item.key == "engines-2s")
    better = [float(value) for value in range(1, 13)]
    worse = [value + 5 for value in better]
    report = NativeReport(
        synthetic("engines-2s", "tsp", {"cpp64": worse, "rust": better}, ("A", "B"))
    )
    result = report.analyze(spec, "tsp")
    assert result.winner == "rust"
    assert all(item.verdict == -1 for item in result.comparisons)
    assert not result.underpowered
    report = NativeReport(
        synthetic("engines-2s", "tsp", {"cpp64": better, "rust": better}, ("A", "B"))
    )
    result = report.analyze(spec, "tsp")
    assert result.winner is None
    assert set(result.leaders) == {"cpp64", "rust"}
    assert all(item.verdict == 0 for item in result.comparisons)
    report = NativeReport(
        synthetic("engines-2s", "tsp", {"cpp64": worse[:5], "rust": better[:5]}, ("A",) * 1)
    )
    assert report.analyze(spec, "tsp").underpowered


def test_marginal_difference_is_a_tie_not_a_win() -> None:
    """A mixed, nonsignificant difference is never promoted to a verdict."""
    spec = next(item for item in ROUNDS if item.key == "engines-2s")
    first = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    second = [1.5, 1.0, 3.5, 3.0, 5.5, 5.0, 7.5, 7.0]
    report = NativeReport(
        synthetic("engines-2s", "kp", {"cpp64": first, "rust": second}, ("A", "B"))
    )
    result = report.analyze(spec, "kp")
    assert result.winner is None
    assert all(item.verdict == 0 for item in result.comparisons)


def test_committed_outputs_match_a_fresh_render_of_the_evidence() -> None:
    """README block and figures are exactly what the committed evidence produces."""
    if not EVIDENCE.exists():
        pytest.skip("evidence file not present")
    outputs = NativeReport.load(EVIDENCE).render(README, FIGURES)
    stale = [
        path.name
        for path, text in outputs.items()
        if not path.exists() or path.read_text(encoding="utf-8") != text
    ]
    assert not stale, f"run `python -m benchmarks.native_report render`: {stale}"
