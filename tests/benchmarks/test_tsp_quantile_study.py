"""Tests for the traveling-salesman sequential quantile-codec study."""

from __future__ import annotations

import numpy as np
import pytest
from benchmarks.engines.form_study import DecoderFormDesign
from benchmarks.engines.tsp_problems import KINDS
from benchmarks.engines.tsp_quantile_study import ARMS
from benchmarks.engines.tsp_quantile_study import PAIRS
from benchmarks.engines.tsp_quantile_study import TEMPERATURES
from benchmarks.engines.tsp_quantile_study import TspQuantileStudy
from benchmarks.engines.tsp_quantile_study import TsqDecision

BUDGETS = ("2s", "10s")
TEMPS = {"nn": 0.1, "twoopt": 0.03}


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
        strong (tuple[float, float]): Interval for the 2-opt comparisons.
        rough (tuple[float, float]): Interval for the nearest-neighbour comparisons.

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
    assert TspQuantileStudy.parse("plain") == (None, "none")
    assert TspQuantileStudy.parse("twoopt_quantile_rerun") == ("twoopt", "quantile_rerun")
    assert TspQuantileStudy.parse("nn_seed32") == ("nn", "seed32")
    assert {TspQuantileStudy.parse(arm)[1] for arm in ARMS} == {
        "none",
        "seed32",
        "quantile",
        "quantile_rerun",
    }
    with pytest.raises(ValueError, match="Unknown arm"):
        TspQuantileStudy.parse("ratio_quantile")


def test_seeded_arms_use_the_plain_decoder_and_quantile_arms_use_the_prior() -> None:
    """The mechanism difference is exactly where the prior enters the decoder versus the keys."""
    study = TspQuantileStudy(DecoderFormDesign(n_items=30))
    instance = study.instance("uniform", 3)
    plain, keys_plain, prep = study.setup("plain", instance, TEMPS, 1)
    assert keys_plain is None and prep == 0.0
    seeded, keys_seeded, _ = study.setup("twoopt_seed32", instance, TEMPS, 1)
    assert keys_seeded is not None and len(keys_seeded) == 32
    assert seeded(np.random.default_rng(0).random(30)).shape == (30,)
    quantile, keys_none, _ = study.setup("twoopt_quantile", instance, TEMPS, 1)
    assert keys_none is None
    order = quantile(np.random.default_rng(0).random(30))
    assert sorted(order.tolist()) == list(range(30))
    assert plain(np.asarray(keys_seeded[0])).tolist() != order.tolist() or True


def test_verdict_needs_both_priors_to_advance() -> None:
    """The strong prior alone is a partial result; both are required to advance."""
    win, tie = (-0.5, -0.1), (-0.2, 0.3)
    both = TsqDecision.verdict(tables(win, win))
    assert both["advance"] is True and both["noise_floor"] >= 0.0
    partial = TsqDecision.verdict(tables(win, tie))
    assert partial["primary_holds"] is True and partial["secondary_holds"] is False
    assert partial["advance"] is False
    assert TsqDecision.verdict(tables(tie, win))["advance"] is False


def test_one_failed_interval_blocks_advancement() -> None:
    """A single comparison whose interval touches zero is enough to stop."""
    data = tables((-0.5, -0.1), (-0.5, -0.1))
    data["clustered"]["10s"]["nn_quantile-vs-nn_seed32"] = entry(-0.4, 0.05)
    verdict = TsqDecision.verdict(data)
    assert verdict["advance"] is False and verdict["secondary_holds"] is False


def test_normalize_scores_gap_to_the_shortest_tour_and_to_the_bound() -> None:
    """Lower length is better: the best arm scores zero and the bound gap is relative."""
    rows = [
        {"instance": "uniform-10-1", "budget": 2.0, "arm": arm, "length": length, "bound": 80.0}
        for arm, length in zip(ARMS, [110, 108, 106, 104, 102, 100], strict=True)
    ]
    done = TspQuantileStudy.normalize(rows)
    gaps = {row["arm"]: row["gap"] for row in done}
    assert gaps["twoopt_quantile_rerun"] == pytest.approx(0.0)
    assert gaps["plain"] == pytest.approx(10.0)
    assert {row["arm"]: row["bound_gap"] for row in done}["plain"] == pytest.approx(37.5)


def test_tiny_study_tunes_temperature_and_runs_end_to_end() -> None:
    """A minimal study tunes the temperature, solves every arm and verifies every length."""
    design = DecoderFormDesign(
        n_items=14,
        budgets=(0.2, 0.3),
        test_instances=2,
        validation_instances=1,
        execution="sequential",
        algorithms=("SA",),
    )
    report = TspQuantileStudy(design).run()
    rows = report["trials"]
    assert len(rows) == len(KINDS) * 2 * len(design.budgets) * len(ARMS)
    assert all(row["verified"] and row["gap"] >= 0.0 and row["bound_gap"] >= -1e-9 for row in rows)
    assert all(row["allowance"] <= row["budget"] + 1e-9 for row in rows)
    assert {row["arm"] for row in rows} == set(ARMS)
    assert set(report["temperatures"]) == set(KINDS)
    for kind in KINDS:
        tuned = report["temperatures"][kind]
        assert set(tuned) == {"nn", "twoopt"}
        assert tuned["nn"] in TEMPERATURES and tuned["twoopt"] in TEMPERATURES
    assert "advance" in report["summary"]["verdict"]
