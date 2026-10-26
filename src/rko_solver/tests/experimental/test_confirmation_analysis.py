"""Independent numerical examples for paired instance-level research analysis.

Objective:
- Validate paired instance-level inference against independent numerical examples.
Technical approach:
- Analyze fixed nested seed observations and compare exact probabilities and adjustments.
Flow (pseudocode):
1. Construct constant-effect and balanced-effect paired panels.
2. Compute inference and Holm-adjusted probabilities.
3. Assert known results and reject incomplete or invalid observations.
Important criteria:
- Solver seeds do not increase independent instance counts; invalid panels cannot infer.
Inputs/Outputs:
- Input: nested floating-point loss panels and a fixed family of p-values.
- Output: assertions on effects, intervals, probabilities and validation failures.
"""

import pytest
from rko_solver.experimental.self_evolve.analysis import PairedAnalysis


def test_exact_paired_test_and_constant_difference_interval() -> None:
    """Four identical positive instance effects have exact one-sided probability 1/16."""
    result = PairedAnalysis(bootstrap_samples=200, permutation_samples=200).compare(
        [[1.0, 1.0]] * 4, [[0.0, 0.0]] * 4
    )
    assert result["n_instances"] == 4
    assert result["seeds_per_instance"] == 2
    assert result["effect_mean"] == 1.0
    assert result["confidence_interval_95"] == [1.0, 1.0]
    assert result["p_value"] == 1 / 16
    assert result["permutation_method"] == "exact_paired_sign_flip"


def test_solver_seeds_are_not_independent_instances() -> None:
    """Twenty seed observations still represent only two independent instance units."""
    result = PairedAnalysis(bootstrap_samples=200, permutation_samples=200).compare(
        [[1.0] * 10, [0.0] * 10], [[0.0] * 10, [1.0] * 10]
    )
    assert result["n_instances"] == 2
    assert result["effect_mean"] == 0.0
    assert result["p_value"] == 0.75


def test_holm_adjustment_has_known_monotone_values() -> None:
    """Holm correction uses the whole hypothesis family, not only favorable comparisons."""
    assert PairedAnalysis.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


@pytest.mark.parametrize(
    "baseline,candidate",
    [
        ([[0.0]], [[0.0]]),
        ([[0.0], [0.0]], [[0.0]]),
        ([[float("nan")], [0.0]], [[0.0], [0.0]]),
        ([[1.1], [0.0]], [[0.0], [0.0]]),
    ],
)
def test_incomplete_or_invalid_panels_do_not_produce_inference(
    baseline: list[list[float]], candidate: list[list[float]]
) -> None:
    """Missing/nonfinite/out-of-domain observations cannot silently disappear.

    Args:
        baseline (list[list[float]]): Parametrized baseline losses grouped by instance and seed.
        candidate (list[list[float]]): Paired candidate losses testing panel validation.
    """
    with pytest.raises(ValueError):
        PairedAnalysis().compare(baseline, candidate)
