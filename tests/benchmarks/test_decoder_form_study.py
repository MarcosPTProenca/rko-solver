"""Decoder-form study contract tests.

Objective:
- Verify seed isolation, arm construction, the pre-registered decision rule and a complete tiny
  run through the real solver.

Technical approach:
- Check the design and arm setup directly, feed synthetic paired statistics to the verdict, and
  run a minimal study end to end with one algorithm and sequential execution.

Flow (pseudocode):
1. Validate disjoint seed ranges and arm parsing.
2. Check that seeded arms use the plain decoder and residual arms carry the prior.
3. Build synthetic comparison tables for each decision branch.
4. Run a tiny study and check every record.

Important criteria:
- Overlapping validation and test seeds are rejected.
- Advancement needs every dual-prior interval below zero at every family and budget.
- Every decoded result is re-verified against the solver's reported cost.

Inputs/Outputs:
- Input: design settings, synthetic statistics and a tiny study.
- Output: assertions on validation, construction, verdicts and trial records.
"""

import numpy as np
import pytest
from benchmarks.engines.decoder_form_study import ARMS
from benchmarks.engines.decoder_form_study import KINDS
from benchmarks.engines.decoder_form_study import PAIRS
from benchmarks.engines.decoder_form_study import SEED_COUNT
from benchmarks.engines.decoder_form_study import SPREADS
from benchmarks.engines.decoder_form_study import DecoderFormDecision
from benchmarks.engines.decoder_form_study import DecoderFormDesign
from benchmarks.engines.decoder_form_study import DecoderFormStudy
from benchmarks.engines.packing_problems import PackingFamily
from typing import Any

BUDGETS = ("2s", "10s")
SPREAD_MAP = {"ratio": 0.5, "dual": 0.25}


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
    dual: tuple[float, float], ratio: tuple[float, float], rerun: tuple[float, float] = (-0.1, 0.3)
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Build identical statistics for every family and budget.

    Args:
        dual (tuple[float, float]): Interval for the dual-prior comparisons.
        ratio (tuple[float, float]): Interval for the ratio-prior comparisons.
        rerun (tuple[float, float]): Interval for the rerun-noise comparison.

    Returns:
        dict[str, dict[str, dict[str, dict[str, float]]]]: Nested statistics.
    """
    chosen = {
        f"{a}-vs-{b}": (rerun if a.endswith("rerun") else dual if a.startswith("dual") else ratio)
        for a, b in PAIRS
    }
    return {
        kind: {b: {pair: entry(*i) for pair, i in chosen.items()} for b in BUDGETS}
        for kind in KINDS
    }


def test_default_design_has_disjoint_seed_ranges() -> None:
    """The shipped validation and test seeds never overlap."""
    design = DecoderFormDesign()
    validation = range(design.validation_base, design.validation_base + design.validation_instances)
    test = range(design.test_base, design.test_base + design.test_instances)
    assert not set(validation) & set(test)
    assert design.test_base != 7000


@pytest.mark.parametrize(
    "overrides", [{"test_base": 6503}, {"validation_base": 7995}, {"validation_instances": 2000}]
)
def test_overlapping_seed_ranges_are_rejected(overrides: dict[str, Any]) -> None:
    """Any validation and test overlap is an error.

    Args:
        overrides (dict[str, Any]): Design overrides that create an overlap.
    """
    with pytest.raises(ValueError, match="disjoint"):
        DecoderFormDesign(**overrides)


def test_arm_names_parse_into_prior_and_mechanism() -> None:
    """Every documented arm has a prior and mechanism; unknown arms are rejected."""
    expected = {
        "plain": (None, "none"),
        "ratio_seed1": ("ratio", "seed1"),
        "dual_seed32": ("dual", "seed32"),
        "dual_resid": ("dual", "resid"),
        "dual_resid_rerun": ("dual", "resid_rerun"),
    }
    for arm, parsed in expected.items():
        assert DecoderFormStudy.parse(arm) == parsed
    assert {DecoderFormStudy.parse(arm)[1] for arm in ARMS} == {
        "none",
        "seed1",
        "seed32",
        "resid",
        "resid_rerun",
    }
    with pytest.raises(ValueError, match="arm"):
        DecoderFormStudy.parse("learned_resid")


def test_scores_dispatch_to_the_named_prior_and_reject_unknown() -> None:
    """Each prior returns its own score vector."""
    instance = PackingFamily.make("mkp", 40, 2)
    np.testing.assert_array_equal(
        DecoderFormStudy.scores("ratio", instance), PackingFamily.hand_scores(instance)
    )
    np.testing.assert_array_equal(
        DecoderFormStudy.scores("dual", instance), PackingFamily.dual_scores(instance)
    )
    with pytest.raises(ValueError, match="prior"):
        DecoderFormStudy.scores("learned", instance)


def test_seeded_arms_use_the_plain_decoder_and_residual_arms_carry_the_prior() -> None:
    """The mechanism difference is exactly where the prior enters."""
    study = DecoderFormStudy(DecoderFormDesign(n_items=40))
    instance = PackingFamily.make("mkp", 40, 3)
    plain, none, prep = study.setup("plain", instance, SPREAD_MAP, 1)
    assert plain.scores is None and none is None and prep == 0.0
    one, keys_one, _ = study.setup("dual_seed1", instance, SPREAD_MAP, 1)
    many, keys_many, _ = study.setup("dual_seed32", instance, SPREAD_MAP, 1)
    assert one.scores is None and many.scores is None
    assert keys_one is not None and len(keys_one) == 1 and keys_many is not None
    assert len(keys_many) == SEED_COUNT and keys_many[0] == keys_one[0]
    resid, keys_resid, _ = study.setup("dual_resid", instance, SPREAD_MAP, 1)
    assert keys_resid is None and resid.scores is not None and resid.spread == 0.25
    ratio, _, _ = study.setup("ratio_resid", instance, SPREAD_MAP, 1)
    assert ratio.spread == 0.5
    np.testing.assert_array_equal(ratio.scores, PackingFamily.hand_scores(instance))


def test_verdict_advances_only_when_every_dual_interval_is_below_zero() -> None:
    """All dual comparisons below zero advance; the ratio prior and noise do not decide."""
    win, tie = (-0.5, -0.1), (-0.2, 0.3)
    assert DecoderFormDecision.verdict(tables(win, tie))["advance"] is True
    assert DecoderFormDecision.verdict(tables(tie, win))["advance"] is False
    verdict = DecoderFormDecision.verdict(tables(win, win))
    assert verdict["primary_holds"] and verdict["secondary_holds"]
    assert 0.0 <= verdict["noise_floor"]


def test_one_failed_dual_interval_blocks_advancement() -> None:
    """A single dual comparison whose interval touches zero is enough to stop."""
    data = tables((-0.5, -0.1), (-0.5, -0.1))
    data["mkp10"]["10s"]["dual_resid-vs-dual_seed32"] = entry(-0.4, 0.05)
    verdict = DecoderFormDecision.verdict(data)
    assert verdict["advance"] is False and verdict["primary_holds"] is False
    assert verdict["secondary_holds"] is True


def test_normalize_adds_best_gap_and_bound_gap() -> None:
    """Gaps are nonnegative; the best arm of an instance scores zero against the best profit."""
    rows = [
        {"instance": "mkp-10-1", "budget": 2.0, "arm": arm, "profit": profit, "bound": 100.0}
        for arm, profit in zip(ARMS, [90, 91, 92, 93, 94, 95, 96, 95], strict=True)
    ]
    done = DecoderFormStudy.normalize(rows)
    gaps = {row["arm"]: row["gap"] for row in done}
    assert gaps["dual_resid"] == pytest.approx(0.0)
    assert gaps["plain"] == pytest.approx(100.0 * 6 / 96)
    assert {row["arm"]: row["bound_gap"] for row in done}["plain"] == pytest.approx(10.0)


def test_tiny_study_runs_end_to_end_through_the_real_solver() -> None:
    """A minimal study chooses spreads and solves every arm with verified profits."""
    design = DecoderFormDesign(
        n_items=24,
        budgets=(0.2, 0.3),
        test_instances=2,
        validation_instances=1,
        execution="sequential",
        algorithms=("SA",),
    )
    report = DecoderFormStudy(design).run()
    rows = report["trials"]
    assert len(rows) == len(KINDS) * 2 * len(design.budgets) * len(ARMS)
    assert all(row["verified"] and row["gap"] >= 0.0 and row["bound_gap"] >= -1e-9 for row in rows)
    assert all(row["allowance"] <= row["budget"] + 1e-9 for row in rows)
    assert {row["arm"] for row in rows} == set(ARMS)
    assert all(row["prep_seconds"] == 0.0 for row in rows if row["arm"] == "plain")
    assert set(report["spreads"]) == set(KINDS)
    assert all(value in SPREADS for kind in KINDS for value in report["spreads"][kind].values())
    assert "advance" in report["summary"]["verdict"]
