"""Tests for the traveling-salesman decoder-form study."""

from __future__ import annotations

import numpy as np
import pytest
from benchmarks.engines.form_decision import DecoderFormDecision
from benchmarks.engines.form_study import SEED_COUNT
from benchmarks.engines.form_study import DecoderFormDesign
from benchmarks.engines.tsp_decoder_form_study import ARMS
from benchmarks.engines.tsp_decoder_form_study import NOISES
from benchmarks.engines.tsp_decoder_form_study import PAIRS
from benchmarks.engines.tsp_decoder_form_study import SPREADS
from benchmarks.engines.tsp_decoder_form_study import TspDecoderFormStudy
from benchmarks.engines.tsp_problems import KINDS
from benchmarks.engines.tsp_problems import TspFamily

BUDGETS = ("2s", "10s")
TUNED = {"nn": 0.5, "twoopt": 0.25, "nn_noise": 0.03, "twoopt_noise": 0.1}


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
    strong: tuple[float, float], rough: tuple[float, float]
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Build identical statistics for every family and budget.

    Args:
        strong (tuple[float, float]): Interval for the strong-prior comparisons.
        rough (tuple[float, float]): Interval for the rough-prior comparisons.

    Returns:
        dict[str, dict[str, dict[str, dict[str, float]]]]: Nested statistics.
    """
    chosen = {
        f"{a}-vs-{b}": (
            (-0.1, 0.3) if a.endswith("rerun") else strong if a.startswith("twoopt") else rough
        )
        for a, b in PAIRS
    }
    return {
        kind: {b: {pair: entry(*i) for pair, i in chosen.items()} for b in BUDGETS}
        for kind in KINDS
    }


def test_arm_names_parse_and_unknown_names_are_rejected() -> None:
    """Every arm maps to its prior and mechanism; an unknown prior is rejected."""
    assert TspDecoderFormStudy.parse("plain") == (None, "none")
    assert TspDecoderFormStudy.parse("twoopt_resid_rerun") == ("twoopt", "resid_rerun")
    assert TspDecoderFormStudy.parse("nn_seed32") == ("nn", "seed32")
    assert {TspDecoderFormStudy.parse(arm)[1] for arm in ARMS} == {
        "none",
        "seed1",
        "seed32",
        "resid",
        "resid_rerun",
    }
    with pytest.raises(ValueError, match="Unknown arm"):
        TspDecoderFormStudy.parse("ratio_resid")


def test_priors_are_tours_and_the_strong_one_is_shorter() -> None:
    """Nearest neighbour gives a permutation; 2-opt shortens it; unknown priors raise."""
    instance = TspFamily.make("clustered", 80, 2)
    rough = TspDecoderFormStudy.scores("nn", instance)
    strong = TspDecoderFormStudy.scores("twoopt", instance)
    assert sorted(rough.tolist()) == list(range(80)) == sorted(strong.tolist())
    assert TspFamily.length(instance, strong) < TspFamily.length(instance, rough)
    with pytest.raises(ValueError, match="Unknown prior"):
        TspDecoderFormStudy.scores("learned", instance)


def test_seeded_arms_use_the_plain_decoder_and_residual_arms_carry_the_tour() -> None:
    """The mechanism difference is exactly where the tour enters; tuned values are used."""
    study = TspDecoderFormStudy(DecoderFormDesign(n_items=40))
    instance = TspFamily.make("uniform", 40, 3)
    plain, none, prep = study.setup("plain", instance, TUNED, 1)
    assert plain.tour is None and none is None and prep == 0.0
    one, keys_one, _ = study.setup("twoopt_seed1", instance, TUNED, 1)
    many, keys_many, _ = study.setup("twoopt_seed32", instance, TUNED, 1)
    assert one.tour is None and many.tour is None
    assert keys_one is not None and len(keys_one) == 1 and keys_many is not None
    assert len(keys_many) == SEED_COUNT and keys_many[0] == keys_one[0]
    tour = TspDecoderFormStudy.scores("twoopt", instance)
    assert keys_one[0] == TspFamily.position_keys(tour)
    resid, keys_resid, _ = study.setup("twoopt_resid", instance, TUNED, 1)
    assert keys_resid is None and resid.tour is not None and resid.spread == 0.25
    np.testing.assert_array_equal(resid.tour, tour)
    rough, _, _ = study.setup("nn_resid", instance, TUNED, 1)
    assert rough.spread == 0.5


def test_seeding_noise_comes_from_the_tuned_values() -> None:
    """Different tuned noise changes the perturbed rows but never the exact first one."""
    study = TspDecoderFormStudy(DecoderFormDesign(n_items=40))
    instance = TspFamily.make("uniform", 40, 3)
    _, quiet, _ = study.setup("nn_seed32", instance, TUNED | {"nn_noise": 0.0}, 1)
    _, loud, _ = study.setup("nn_seed32", instance, TUNED | {"nn_noise": 0.1}, 1)
    assert quiet is not None and loud is not None
    assert quiet[0] == loud[0] and quiet[1:] != loud[1:]
    assert all(row == quiet[0] for row in quiet)


def test_verdict_needs_both_priors_to_advance() -> None:
    """The strong prior alone is a partial result; both are required to advance."""
    win, tie = (-0.5, -0.1), (-0.2, 0.3)
    both = DecoderFormDecision.verdict(tables(win, win), "twoopt", "nn", True)
    assert both["advance"] is True and both["noise_floor"] >= 0.0
    partial = DecoderFormDecision.verdict(tables(win, tie), "twoopt", "nn", True)
    assert partial["primary_holds"] is True and partial["secondary_holds"] is False
    assert partial["advance"] is False
    assert DecoderFormDecision.verdict(tables(tie, win), "twoopt", "nn", True)["advance"] is False


def test_one_failed_interval_blocks_advancement() -> None:
    """A single comparison whose interval touches zero is enough to stop."""
    data = tables((-0.5, -0.1), (-0.5, -0.1))
    data["clustered"]["10s"]["nn_resid-vs-nn_seed32"] = entry(-0.4, 0.05)
    verdict = DecoderFormDecision.verdict(data, "twoopt", "nn", True)
    assert verdict["advance"] is False and verdict["secondary_holds"] is False


def test_normalize_scores_gap_to_the_shortest_tour_and_to_the_bound() -> None:
    """Lower length is better: the best arm scores zero and the bound gap is relative."""
    rows = [
        {"instance": "uniform-10-1", "budget": 2.0, "arm": arm, "length": length, "bound": 80.0}
        for arm, length in zip(ARMS, [110, 108, 106, 104, 102, 101, 100, 101], strict=True)
    ]
    done = TspDecoderFormStudy.normalize(rows)
    gaps = {row["arm"]: row["gap"] for row in done}
    assert gaps["twoopt_resid"] == pytest.approx(0.0)
    assert gaps["plain"] == pytest.approx(10.0)
    assert {row["arm"]: row["bound_gap"] for row in done}["twoopt_resid"] == pytest.approx(25.0)


def test_badness_is_length_over_bound() -> None:
    """The validation score is scale free and lower is better."""
    assert TspDecoderFormStudy.badness({"length": 105.0, "bound": 100.0}) == pytest.approx(1.05)


def test_tiny_study_tunes_spread_and_noise_and_runs_end_to_end() -> None:
    """A minimal study tunes both values, solves every arm and verifies every length."""
    design = DecoderFormDesign(
        n_items=14,
        budgets=(0.2, 0.3),
        test_instances=2,
        validation_instances=1,
        execution="sequential",
        algorithms=("SA",),
    )
    report = TspDecoderFormStudy(design).run()
    rows = report["trials"]
    assert len(rows) == len(KINDS) * 2 * len(design.budgets) * len(ARMS)
    assert all(row["verified"] and row["gap"] >= 0.0 and row["bound_gap"] >= -1e-9 for row in rows)
    assert all(row["allowance"] <= row["budget"] + 1e-9 for row in rows)
    assert {row["arm"] for row in rows} == set(ARMS)
    assert set(report["spreads"]) == set(KINDS)
    for kind in KINDS:
        tuned = report["spreads"][kind]
        assert set(tuned) == {"nn", "twoopt", "nn_noise", "twoopt_noise"}
        assert tuned["nn"] in SPREADS and tuned["twoopt"] in SPREADS
        assert tuned["nn_noise"] in NOISES and tuned["twoopt_noise"] in NOISES
    assert "advance" in report["summary"]["verdict"]
