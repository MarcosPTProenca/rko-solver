"""Tests for the traveling-salesman learned-scorer study."""

from __future__ import annotations

import numpy as np
import pytest
from benchmarks.engines.priority_scorer import PriorityScorer
from benchmarks.engines.tsp_learned_study import ARMS
from benchmarks.engines.tsp_learned_study import SPREADS
from benchmarks.engines.tsp_learned_study import LearnedStudyDesign
from benchmarks.engines.tsp_learned_study import LearnedTspDecision
from benchmarks.engines.tsp_learned_study import TspLearnedStudy
from benchmarks.engines.tsp_problems import KINDS

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
    seed_pair: tuple[float, float], resid_pair: tuple[float, float]
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Build identical statistics for every family and budget.

    Args:
        seed_pair (tuple[float, float]): Interval for `learned_seed32-vs-twoopt_seed32`.
        resid_pair (tuple[float, float]): Interval for `learned_resid-vs-twoopt_resid`.

    Returns:
        dict[str, dict[str, dict[str, dict[str, float]]]]: Nested statistics.
    """
    chosen = {
        "learned_seed32-vs-twoopt_seed32": seed_pair,
        "learned_resid-vs-twoopt_resid": resid_pair,
        "learned_resid-vs-plain": (-50.0, -10.0),
    }
    return {
        kind: {b: {pair: entry(*i) for pair, i in chosen.items()} for b in BUDGETS}
        for kind in KINDS
    }


def test_disjoint_seed_ranges_are_required() -> None:
    """Overlapping training, validation or test seeds raise."""
    LearnedStudyDesign()
    with pytest.raises(ValueError, match="disjoint"):
        LearnedStudyDesign(validation_base=100_500)


def test_verdict_needs_both_mechanisms_to_beat_the_twoopt_baseline() -> None:
    """Only a seeded-and-residual win in every family and budget advances."""
    win, tie = (-0.5, -0.1), (-0.2, 0.3)
    assert LearnedTspDecision.verdict(tables(win, win))["advance"] is True
    assert LearnedTspDecision.verdict(tables(win, tie))["advance"] is False
    assert LearnedTspDecision.verdict(tables(tie, win))["advance"] is False


def test_one_failed_interval_blocks_advancement() -> None:
    """A single comparison whose interval touches zero is enough to stop."""
    data = tables((-0.5, -0.1), (-0.5, -0.1))
    data["clustered"]["10s"]["learned_resid-vs-twoopt_resid"] = entry(-0.4, 0.05)
    assert LearnedTspDecision.verdict(data)["advance"] is False


def test_plain_and_twoopt_arms_use_the_2opt_tour_not_the_scorer() -> None:
    """The baseline arms are the 2-opt tour regardless of the scorer."""
    design = LearnedStudyDesign(n_items=20)
    study = TspLearnedStudy(design)
    instance = study.instance("uniform", 1)
    scorer = PriorityScorer.init(5, 4, 0)
    tour = study.tour_for("twoopt_resid", instance, scorer)
    from benchmarks.engines.tsp_decoder_form_study import TspDecoderFormStudy

    np.testing.assert_array_equal(tour, TspDecoderFormStudy.prior("twoopt", instance))


def test_learned_arms_use_the_scorers_own_order() -> None:
    """A learned arm's prior tour is the argsort of the scorer's output."""
    from benchmarks.engines.tsp_problems import TspFamily

    design = LearnedStudyDesign(n_items=20)
    study = TspLearnedStudy(design)
    instance = study.instance("uniform", 1)
    scorer = PriorityScorer.init(5, 4, 0)
    tour = study.tour_for("learned_resid", instance, scorer)
    scores = scorer.scores(TspFamily.features(instance))
    np.testing.assert_array_equal(tour, np.argsort(-scores, kind="stable"))


def test_setup_builds_plain_seed_and_residual_decoders() -> None:
    """The mechanism difference is exactly where the prior enters: keys versus the decoder."""
    design = LearnedStudyDesign(n_items=20)
    study = TspLearnedStudy(design)
    instance = study.instance("uniform", 1)
    scorer = PriorityScorer.init(5, 4, 0)
    spreads = {"learned_resid": 0.1, "twoopt_resid": 0.1}
    plain, keys_plain, prep_plain = study.setup("plain", instance, spreads, scorer)
    assert keys_plain is None and prep_plain == 0.0 and plain.tour is None
    seeded, keys_seeded, _ = study.setup("twoopt_seed32", instance, spreads, scorer)
    assert seeded.tour is None and keys_seeded is not None and len(keys_seeded) == 32
    resid, keys_resid, _ = study.setup("twoopt_resid", instance, spreads, scorer)
    assert keys_resid is None and resid.tour is not None and resid.spread == 0.1


def test_tiny_study_trains_tunes_and_runs_end_to_end() -> None:
    """A minimal study trains a scorer per family, tunes spreads and verifies every length."""
    design = LearnedStudyDesign(
        n_items=16,
        budgets=(0.2, 0.3),
        test_instances=2,
        validation_instances=2,
        steps=5,
        batch=4,
        samples=4,
        hidden=4,
        execution="sequential",
        algorithms=("SA",),
    )
    report = TspLearnedStudy(design).run()
    rows = report["trials"]
    assert len(rows) == len(KINDS) * 2 * len(design.budgets) * len(ARMS)
    assert all(row["verified"] and row["gap"] >= 0.0 and row["bound_gap"] >= -1e-9 for row in rows)
    assert all(row["allowance"] <= row["budget"] + 1e-9 for row in rows)
    assert {row["arm"] for row in rows} == set(ARMS)
    assert set(report["training"]) == set(KINDS)
    for kind in KINDS:
        assert report["training"][kind]["steps"] == 5
        assert "validation_reward_after" in report["training"][kind]
        tuned = report["spreads"][kind]
        assert set(tuned) == {"learned_resid", "twoopt_resid"}
        assert tuned["learned_resid"] in SPREADS and tuned["twoopt_resid"] in SPREADS
    assert "advance" in report["summary"]["verdict"]
