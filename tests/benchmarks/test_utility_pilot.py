"""Plan, determinism, stability and cost contracts of the baseline-only pilot.

Objective:
- Prove the pilot is baseline-only, repeated with one seed and judged by the fixed rules.

Technical approach:
- Synthetic rows with known repeats, spreads and wall times; no solver runs.

Flow (pseudocode):
1. Plan trials and check composition and ordering.
2. Report rows with identical, differing and noisy repeats.
3. Combine mechanism reports against the ledger.

Important criteria:
- Evaluation-profile mismatch stops; timed instability only marks the class uninterpretable.

Inputs/Outputs:
- Input: synthetic entries and rows.
- Output: deterministic assertions.
"""

import pytest
from benchmarks.algorithms.utility_pilot import CEILING_HOURS
from benchmarks.algorithms.utility_pilot import PROFILES
from benchmarks.algorithms.utility_pilot import UtilityPilot
from benchmarks.algorithms.utility_study import SOLVER_SEED
from collections.abc import Callable
from typing import Any

STRATA = ("item-500", "item-1000", "route-100", "route-300", "continuous-100", "continuous-300")


def entries() -> list[dict[str, Any]]:
    """Build twelve pilot entries, two per stratum.

    Returns:
        list[dict[str, Any]]: Manifest-like entries.
    """
    return [
        {"class": stratum.split("-")[0], "stratum": stratum, "instance": f"{stratum}-{n}"}
        for stratum in STRATA
        for n in range(2)
    ]


def rows(
    evals: dict[str, list[int]] | None = None, wall: float = 1.0, score: float = 5.0
) -> list[dict[str, Any]]:
    """Build pilot rows with controllable repeats.

    Args:
        evals (dict[str, list[int]] | None): Evaluations per repeat by profile.
        wall (float): Wall seconds of every trial.
        score (float): Score of every trial.

    Returns:
        list[dict[str, Any]]: One row per instance, profile and repeat.
    """
    given = {"evaluations": [100] * 3, "time2": [1000] * 3, "time10": [5000] * 3, **(evals or {})}
    return [
        {
            "class": entry["class"],
            "stratum": entry["stratum"],
            "instance": entry["instance"],
            "profile": profile,
            "repeat": repeat,
            "score": score,
            "evaluations": given[profile][repeat],
            "wall_seconds": wall,
        }
        for entry in entries()
        for profile in PROFILES
        for repeat in range(3)
    ]


def test_plan_is_baseline_only_one_seed_three_repeats_seeded_order() -> None:
    """Plan 12 instances x 3 profiles x 3 identical-seed repeats."""
    trials = UtilityPilot(entries(), 4).plan()
    assert len(trials) == 12 * 3 * 3
    assert {t["arm"] for t in trials} == {"baseline"}
    assert {t["seed"] for t in trials} == {SOLVER_SEED}
    keys = {(t["instance"], t["profile"], t["repeat"]) for t in trials}
    assert len(keys) == len(trials) and {t["repeat"] for t in trials} == {0, 1, 2}
    assert trials == UtilityPilot(entries(), 4).plan()
    assert trials != UtilityPilot(entries(), 5).plan()
    assert [t["instance"] for t in trials[:6]] != [
        t["instance"] for t in sorted(trials, key=lambda t: t["instance"])[:6]
    ]


def test_execute_routes_by_profile_and_keeps_the_pilot_label() -> None:
    """Send time profiles to the time executor but label rows with the pilot profile."""
    seen: list[str] = []

    def make(name: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Build a recording executor.

        Args:
            name (str): Executor name.

        Returns:
            Callable[[dict[str, Any]], dict[str, Any]]: Executor recording its profile.
        """

        def run(trial: dict[str, Any]) -> dict[str, Any]:
            """Record and echo the trial.

            Args:
                trial (dict[str, Any]): Trial.

            Returns:
                dict[str, Any]: Row.
            """
            seen.append(f"{name}:{trial['profile']}")
            return {**trial, "score": 1}

        return run

    executors = {p: make(p) for p in PROFILES}
    trial = {"instance": "i", "profile": "time2", "repeat": 1, "arm": "baseline", "seed": 1}
    row = UtilityPilot.execute(executors, trial)
    assert seen == ["time2:time"] and row["profile"] == "time2" and row["repeat"] == 1
    row = UtilityPilot.execute(executors, {**trial, "profile": "evaluations"})
    assert seen[-1] == "evaluations:evaluations" and row["profile"] == "evaluations"
    with pytest.raises(KeyError):
        UtilityPilot.execute(executors, {**trial, "profile": "other"})


def test_identical_repeats_pass_and_a_mismatch_stops() -> None:
    """Require identical score and evaluations in the evaluation profile."""
    clean = UtilityPilot.report(rows())
    assert clean["deterministic"] is True and clean["mismatches"] == []
    bad = rows({"evaluations": [100, 101, 100]})
    report = UtilityPilot.report(bad)
    assert report["deterministic"] is False and len(report["mismatches"]) == 12
    drift = rows()
    drift[0] = {**drift[0], "score": 6.0}
    assert UtilityPilot.report(drift)["deterministic"] is False


def test_timed_stability_uses_the_median_cv_of_evaluations_per_class() -> None:
    """Call a class interpretable only when its median CV is at most 5%."""
    stable = UtilityPilot.report(rows({"time10": [5000, 5100, 4900]}))
    assert stable["interpretable"]["item"]["time10"] is True
    assert stable["cv"]["item"]["time10"] == pytest.approx(0.02)
    noisy = UtilityPilot.report(rows({"time10": [5000, 6000, 4000]}))
    assert noisy["cv"]["route"]["time10"] == pytest.approx(0.2)
    assert noisy["interpretable"]["route"]["time10"] is False
    assert noisy["interpretable"]["route"]["time2"] is True


def test_cost_extrapolation_uses_the_multiplier_and_the_matrix_size() -> None:
    """Estimate 6 strata x 30 x 5 x 2 x (eval wall + time10 wall) x 1.3 over 3600."""
    report = UtilityPilot.report(rows(wall=2.0))
    expected = 6 * 30 * 5 * 2 * (2.0 + 2.0) * 1.3 / 3600
    assert report["estimated_primary_hours"] == pytest.approx(expected)
    assert report["pilot_hours"] == pytest.approx(12 * 9 * 2.0 / 3600)


def test_wrong_repeat_counts_are_an_error() -> None:
    """Abort on missing repeats instead of reporting partial data."""
    with pytest.raises(ValueError, match="repeats"):
        UtilityPilot.report(rows()[:-1])


def test_verdict_passes_stops_on_mismatch_and_on_ledger_overrun() -> None:
    """Combine mechanisms: determinism, pilot hour and the 24-hour fit decide."""
    good = UtilityPilot.report(rows(wall=2.0))
    verdict = UtilityPilot.verdict({"adaptive": good, "temperature": good})
    assert verdict["status"] == "PASS" and verdict["fits"] is True
    assert verdict["estimated_total_hours"] == pytest.approx(2 * good["estimated_primary_hours"])
    bad = UtilityPilot.report(rows({"evaluations": [1, 2, 1]}, wall=2.0))
    stop = UtilityPilot.verdict({"adaptive": good, "temperature": bad})
    assert stop["status"] == "STOP" and any("temperature" in r for r in stop["reasons"])
    slow = UtilityPilot.report(rows(wall=200.0))
    over = UtilityPilot.verdict({"adaptive": slow, "temperature": slow})
    assert over["status"] == "STOP" and over["fits"] is False
    assert over["remaining_hours"] < CEILING_HOURS


def test_noisy_timed_classes_are_listed_but_do_not_stop() -> None:
    """Report uninterpretable timed classes as a fallback, not a failure."""
    noisy = UtilityPilot.report(rows({"time10": [5000, 6000, 4000]}, wall=2.0))
    verdict = UtilityPilot.verdict({"adaptive": noisy, "temperature": noisy})
    assert verdict["status"] == "PASS"
    assert {"adaptive:item", "temperature:route"} <= set(verdict["timed_uninterpretable"])
