"""Freshness, multiplicity and declared sampling-design guards for locked studies.

Objective:
- Protect frozen study design and paired statistical inference from optimistic shortcuts.
Technical approach:
- Build synthetic studies and compare known-effect panels with deterministic analysis.
Flow (pseudocode):
1. Attempt exposed-instance and unjustified confirmatory study creation.
2. Inspect paired execution blocks and analyze controlled stratified panels.
3. Check rejection, reproducibility and multiplicity bounds.
Important criteria:
- Preserve instance independence, declared strata and nonzero Monte Carlo p-values.
Inputs/Outputs:
- Input: temporary study paths and synthetic instance-level effect panels.
- Output: assertions on design validation, sampling and adjusted inference.
"""

import pytest
from pathlib import Path
from rko_solver.experimental.self_evolve import Problem
from rko_solver.experimental.self_evolve.analysis import PairedAnalysis
from rko_solver.experimental.self_evolve.study import ARMS
from rko_solver.experimental.self_evolve.study import ConfirmationStudy
from tests.experimental.test_confirmation_study import create_study


def test_previously_exposed_instances_cannot_enter_test_panel(tmp_path: Path) -> None:
    """An exact development/test overlap is rejected before creating the output directory.

    Args:
        tmp_path (Path): Temporary parent for the rejected study directory.
    """
    prior = ConfirmationStudy.instance_hash(Problem.generate("knapsack", 3001, 4))
    with pytest.raises(ValueError, match="exposed"):
        create_study(tmp_path / "study", prior_instance_hashes=[prior])
    assert not (tmp_path / "study").exists()


def test_confirmatory_label_requires_external_power_justification(tmp_path: Path) -> None:
    """A large-looking panel or small p-value cannot retroactively manufacture a powered design.

    Args:
        tmp_path (Path): Temporary parent for the study lacking a power plan.
    """
    with pytest.raises(ValueError, match="power plan"):
        create_study(tmp_path / "study", claim="confirmatory")


def test_execution_order_is_paired_and_balanced(tmp_path: Path) -> None:
    """Every randomized block contains the same instance/seed and all six frozen arms.

    Args:
        tmp_path (Path): Temporary parent for the frozen study whose jobs are inspected.
    """
    study = create_study(tmp_path / "study")
    for start in range(0, len(study.spec["jobs"]), len(ARMS)):
        block = study.spec["jobs"][start : start + len(ARMS)]
        assert len({tuple(job[:3]) for job in block}) == 1
        assert {job[3] for job in block} == set(ARMS)


def test_stratified_bootstrap_keeps_declared_weights() -> None:
    """Two constant-effect strata retain their exact mixture mean in every resample."""
    result = PairedAnalysis(bootstrap_samples=200, permutation_samples=200).compare(
        [[0.2], [0.2], [0.8], [0.8]], [[0.0]] * 4, strata=["small", "small", "large", "large"]
    )
    assert result["confidence_interval_95"] == pytest.approx([0.5, 0.5])
    assert result["strata_counts"] == {"small": 2, "large": 2}


def test_singleton_strata_cannot_create_degenerate_confidence() -> None:
    """One instance per size regime is insufficient for within-stratum bootstrap uncertainty."""
    with pytest.raises(ValueError, match="two independent"):
        PairedAnalysis().compare([[0.0], [0.1]], [[0.0], [0.0]], strata=["small", "large"])


def test_randomized_p_values_never_equal_zero_and_are_replayable() -> None:
    """Monte Carlo inference includes the observed statistic using the plus-one convention."""
    analysis = PairedAnalysis(bootstrap_samples=200, permutation_samples=100)
    first = analysis.compare([[1.0]] * 17, [[0.0]] * 17)
    second = analysis.compare([[1.0]] * 17, [[0.0]] * 17)
    assert first == second
    assert first["permutation_method"] == "randomized_paired_sign_flip"
    assert first["p_value"] >= 1 / 101


def test_multiplicity_lower_bound_is_not_more_optimistic() -> None:
    """The six-comparison lower bound cannot exceed its single-comparison counterpart."""
    baseline = [[0.1], [0.3], [0.4], [0.8]]
    candidate = [[0.0]] * 4
    single = PairedAnalysis(bootstrap_samples=200, family_size=1).compare(baseline, candidate)
    family = PairedAnalysis(bootstrap_samples=200, family_size=6).compare(baseline, candidate)
    assert family["familywise_lower_bound_95"] <= single["familywise_lower_bound_95"]
