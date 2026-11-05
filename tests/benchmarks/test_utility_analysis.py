"""Win-rule, Hoeffding and Holm contracts of the expanded utility study.

Objective:
- Prove the registered decision arithmetic on synthetic scores, with no optimizer run.

Technical approach:
- Hand-built score vectors for each rule and edge, plus seeded random properties.

Flow (pseudocode):
1. Check each benefit rule at, just above and just below its threshold.
2. Check the p-value bound, Holm adjustment and the claim labels.
3. Check invariances over generated inputs.

Important criteria:
- Ties, zeros, infeasibility and incomplete claims are never wins and never dropped.

Inputs/Outputs:
- Input: synthetic scores and win lists.
- Output: deterministic assertions.
"""

import math
import pytest
import random
from benchmarks.algorithms.utility_analysis import UtilityAnalysis as U


def five(value: float) -> list[float]:
    """Repeat one score for every solver seed.

    Args:
        value (float): Score.

    Returns:
        list[float]: Five copies.
    """
    return [value] * 5


@pytest.mark.parametrize(
    ("baseline", "candidate", "win"),
    [(100, 95, True), (100, 96, False), (100, 94, True), (101, 95, True), (101, 96, False)],
)
def test_item_rule_is_exact_integer_five_percent(baseline: int, candidate: int, win: bool) -> None:
    """Compare at the 5% boundary using integer arithmetic only.

    Args:
        baseline (int): Baseline gap.
        candidate (int): Candidate gap.
        win (bool): Expected outcome.
    """
    assert U.profile_win("item", five(baseline), five(candidate)) is win


@pytest.mark.parametrize(
    ("baseline", "candidate", "win"),
    [(1000, 990, True), (1000, 991, False), (100, 99, True), (99, 99, False), (50, 49, True)],
)
def test_route_rule_is_exact_integer_one_percent(baseline: int, candidate: int, win: bool) -> None:
    """Compare at the 1% route boundary.

    Args:
        baseline (int): Baseline tour length.
        candidate (int): Candidate tour length.
        win (bool): Expected outcome.
    """
    assert U.profile_win("route", five(baseline), five(candidate)) is win


def test_zero_baseline_ties_and_worse_scores_are_non_wins() -> None:
    """Never count a zero baseline, a tie or a regression as a win."""
    for kind in ("item", "route", "continuous"):
        assert U.profile_win(kind, five(0), five(0)) is False
        assert U.profile_win(kind, five(10), five(10)) is False
        assert U.profile_win(kind, five(10), five(11)) is False


def test_the_seed_median_decides_not_one_lucky_seed() -> None:
    """Use the median over five seeds, so two lucky seeds cannot win."""
    assert U.profile_win("item", five(100), [1, 1, 100, 100, 100]) is False
    assert U.profile_win("item", five(100), [1, 1, 1, 100, 100]) is True


def test_a_candidate_infeasible_in_any_seed_cannot_win() -> None:
    """Treat infeasibility as a non-win, not as a dropped instance."""
    feasible = [True, True, True, True, False]
    assert U.profile_win("item", five(100), five(10), feasible) is False
    assert U.profile_win("item", five(100), five(10), [True] * 5) is True


def test_continuous_rule_uses_multiplication_and_the_reproducibility_floor() -> None:
    """Require both 5% and a contrast above max(atol, rtol*baseline)."""
    assert U.profile_win("continuous", five(100.0), five(95.0)) is True
    assert U.profile_win("continuous", five(100.0), five(95.1)) is False
    assert U.profile_win("continuous", five(1e-9), five(0.0)) is False
    assert U.profile_win("continuous", five(2e-8), five(0.0)) is True
    assert U.profile_win("continuous", five(0.0), five(0.0)) is False


def test_tiny_negative_residuals_are_numerical_zero_but_larger_ones_are_invalid() -> None:
    """Clip a residual within the floor to zero and refuse one beyond it."""
    assert U.profile_win("continuous", five(1.0), five(-1e-9)) is True
    with pytest.raises(ValueError, match="invalid"):
        U.profile_win("continuous", five(1.0), five(-1e-6))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_scores_and_wrong_counts_are_errors(bad: float) -> None:
    """Abort on an invalid measurement instead of manufacturing a non-win.

    Args:
        bad (float): Invalid score.
    """
    with pytest.raises(ValueError):
        U.profile_win("item", [bad, 1, 1, 1, 1], five(1))
    with pytest.raises(ValueError, match="seeds"):
        U.profile_win("item", [1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError, match="kind"):
        U.profile_win("tour", five(1), five(1))


def test_an_instance_needs_every_profile() -> None:
    """Combine profiles conservatively."""
    assert U.instance_win({"evaluations": True, "time": True}) is True
    assert U.instance_win({"evaluations": True, "time": False}) is False
    assert U.instance_win({}) is False


def test_hoeffding_bound_matches_the_derivation_and_the_strictest_holm_step() -> None:
    """Reproduce exp(-2n(r-.5)^2) and the 42-of-60 threshold for six claims."""
    assert U.hoeffding_p(30, 60) == 1.0 and U.hoeffding_p(20, 60) == 1.0
    assert U.hoeffding_p(42, 60) == pytest.approx(math.exp(-2 * 60 * 0.2**2))
    assert U.hoeffding_p(42, 60) <= ALPHA_OVER_SIX < U.hoeffding_p(41, 60)
    with pytest.raises(ValueError):
        U.hoeffding_p(61, 60)
    with pytest.raises(ValueError):
        U.hoeffding_p(1, 0)


ALPHA_OVER_SIX = 0.05 / 6


def test_holm_adjustment_is_monotone_capped_and_order_free() -> None:
    """Match hand-computed Holm values and ignore claim order."""
    adjusted = U.holm({"a": 0.001, "b": 0.01, "c": 0.04, "d": 0.5})
    assert adjusted == pytest.approx({"a": 0.004, "b": 0.03, "c": 0.08, "d": 0.5})
    shuffled = U.holm({"d": 0.5, "c": 0.04, "a": 0.001, "b": 0.01})
    assert shuffled == adjusted
    assert U.holm({"a": 0.9, "b": 0.8})["a"] <= 1.0
    with pytest.raises(ValueError):
        U.holm({})
    with pytest.raises(ValueError):
        U.holm({"a": 1.5})


def test_holm_never_rejects_less_than_bonferroni_and_controls_the_family() -> None:
    """Hold over generated families: Holm dominates Bonferroni and stays monotone in p."""
    generator = random.Random(11)
    for _ in range(200):
        values = {str(i): generator.random() ** 3 for i in range(6)}
        adjusted = U.holm(values)
        bonferroni = {key: min(1.0, 6 * value) for key, value in values.items()}
        assert all(adjusted[key] <= bonferroni[key] + 1e-15 for key in values)
        order = sorted(values, key=values.get)
        assert all(
            adjusted[a] <= adjusted[b] + 1e-15 for a, b in zip(order, order[1:], strict=False)
        )
        assert all(adjusted[key] >= values[key] - 1e-15 for key in values)


def test_decide_labels_every_claim_and_keeps_incomplete_in_the_family() -> None:
    """Classify evidence, insufficient and incomplete claims over the fixed family."""
    claims = {
        "adaptive:item": [True] * 50 + [False] * 10,
        "adaptive:route": [True] * 30 + [False] * 30,
        "adaptive:continuous": [True] * 41 + [False] * 19,
        "temperature:item": [True] * 42 + [False] * 18,
        "temperature:route": [True] * 59,
        "temperature:continuous": [False] * 60,
    }
    status = U.decide(claims)
    assert status["adaptive:item"] == "evidence"
    assert status["adaptive:route"] == "insufficient"
    assert status["adaptive:continuous"] == "insufficient"
    assert status["temperature:item"] == "evidence"
    assert status["temperature:route"] == "incomplete"
    assert status["temperature:continuous"] == "insufficient"
    with pytest.raises(ValueError):
        U.decide({})


def test_an_incomplete_claim_still_counts_toward_the_correction() -> None:
    """Keep the family size fixed so an unfinished claim cannot loosen the others."""
    full = {f"c{i}": [True] * 42 + [False] * 18 for i in range(6)}
    assert set(U.decide(full).values()) == {"evidence"}
    partial = {**full, "c5": [True] * 10}
    status = U.decide(partial)
    assert status["c5"] == "incomplete" and status["c0"] == "evidence"
    seven = {**full, "c6": [True] * 42 + [False] * 18}
    assert U.decide(seven)["c0"] == "insufficient"


def test_win_rule_is_invariant_to_positive_integer_scaling() -> None:
    """Hold over generated integer scores: scaling both arms keeps the decision."""
    generator = random.Random(3)
    for _ in range(300):
        base = [generator.randrange(0, 400) for _ in range(5)]
        cand = [generator.randrange(0, 400) for _ in range(5)]
        factor = generator.randrange(1, 50)
        for kind in ("item", "route"):
            assert U.profile_win(kind, base, cand) is U.profile_win(
                kind, [factor * x for x in base], [factor * x for x in cand]
            )
