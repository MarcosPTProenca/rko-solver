"""Baseline-only pilot of the expanded utility study.

Objective:
- Measure determinism, timed stability and cost on disjoint inputs before any candidate run.

Technical approach:
- Repeat each baseline trial three times with the same seed under the evaluation profile
  and the 2 s and 10 s timed profiles, then summarize per mechanism.
- Judge timed stability by the coefficient of variation of completed evaluations, which
  is what host contention changes when the time budget is fixed.

Flow (pseudocode):
1. Plan baseline trials in a seeded order, three identical repeats each.
2. Run them through the mechanism's executors and keep every row.
3. Report determinism, per-class stability, cost and fit against the machine-hour ledger.

Important criteria:
- A mismatch between evaluation-profile repeats is a STOP, never averaged away.
- A timed class whose median CV exceeds the bound is uninterpretable, not a failure.
- A matrix that does not fit the remaining ledger stops the study before candidate data.

Inputs/Outputs:
- Input: manifest entries, executors and pilot rows.
- Output: trials, rows and a verdict dictionary.
"""

from __future__ import annotations

import random
import statistics
from benchmarks.algorithms.utility_study import SOLVER_SEED
from benchmarks.algorithms.utility_study import UtilityRunner
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

REPEATS = 3
PROFILES = ("evaluations", "time2", "time10")
PILOT_SECONDS = {"time2": 2.0, "time10": 10.0}
CV_LIMIT = 0.05
COST_MULTIPLIER = 1.3
CEILING_HOURS = 24.0
PILOT_HOURS = 1.0
MATRIX_INSTANCES = 30
MATRIX_SEEDS = 5


class UtilityPilot(UtilityRunner):
    """Plan, route and summarize the baseline-only pilot.

    Responsibilities:
    - Plan three identical baseline repeats per instance and profile in a seeded order.
    - Route each trial to the executor of its profile.
    - Summarize determinism, stability and cost per mechanism and combine mechanisms.
    """

    def __init__(self, entries: Sequence[Mapping[str, Any]], order_seed: int) -> None:
        """Store the plan inputs.

        Args:
            entries (Sequence[Mapping[str, Any]]): Pilot manifest entries.
            order_seed (int): Seed of the execution order.
        """
        super().__init__(entries, PROFILES, order_seed)

    def plan(self) -> list[dict[str, Any]]:
        """List every pilot trial once, in a seeded order independent of results.

        Returns:
            list[dict[str, Any]]: Baseline trials with the fixed seed and a repeat index.
        """
        trials = [
            {
                "class": entry["class"],
                "stratum": entry["stratum"],
                "instance": entry["instance"],
                "profile": profile,
                "repeat": repeat,
                "seed": SOLVER_SEED,
                "arm": "baseline",
            }
            for entry in self.entries
            for profile in PROFILES
            for repeat in range(REPEATS)
        ]
        random.Random(f"pilot-order:{self.order_seed}").shuffle(trials)
        return trials

    @staticmethod
    def execute(
        executors: Mapping[str, Callable[[dict[str, Any]], dict[str, Any]]], trial: dict[str, Any]
    ) -> dict[str, Any]:
        """Run one trial on the executor of its profile and keep the pilot profile label.

        Args:
            executors (Mapping[str, Callable[[dict[str, Any]], dict[str, Any]]]): Executor
                per pilot profile.
            trial (dict[str, Any]): Planned pilot trial.

        Returns:
            dict[str, Any]: Row with the pilot profile and repeat.

        Raises:
            KeyError: If the profile is unknown.
        """
        timed = trial["profile"] in PILOT_SECONDS
        row = executors[trial["profile"]](
            {**trial, "profile": "time" if timed else trial["profile"]}
        )
        return {**row, "profile": trial["profile"], "repeat": trial["repeat"]}

    @staticmethod
    def report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Summarize one mechanism's pilot rows.

        Args:
            rows (Sequence[Mapping[str, Any]]): Pilot rows of one mechanism.

        Returns:
            dict[str, Any]: Determinism, per-class CV and interpretability, cost and hours.

        Raises:
            ValueError: If an instance and profile lack exactly the planned repeats.
        """
        groups = UtilityPilot._groups(rows)
        mismatches = UtilityPilot._mismatches(groups)
        cv = UtilityPilot._cv(groups)
        return {
            "deterministic": not mismatches,
            "mismatches": mismatches,
            "cv": cv,
            "interpretable": UtilityPilot._interpretable(cv),
            "estimated_primary_hours": UtilityPilot._estimate(rows),
            "pilot_hours": sum(row["wall_seconds"] for row in rows) / 3600.0,
        }

    @staticmethod
    def _groups(
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
        """Group rows by instance and profile and require exactly the planned repeats.

        Args:
            rows (Sequence[Mapping[str, Any]]): Pilot rows.

        Returns:
            dict[tuple[str, str], list[Mapping[str, Any]]]: Rows by instance and profile.

        Raises:
            ValueError: If a group lacks exactly the planned repeats.
        """
        groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(row["instance"], row["profile"])].append(row)
        if any(len(group) != REPEATS for group in groups.values()) or len(groups) % len(PROFILES):
            raise ValueError(f"every instance and profile needs exactly {REPEATS} repeats")
        return groups

    @staticmethod
    def _interpretable(cv: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, bool]]:
        """Flag timed profiles whose median CV is within the stability bound.

        Args:
            cv (Mapping[str, Mapping[str, float]]): Median CV per class and timed profile.

        Returns:
            dict[str, dict[str, bool]]: Interpretability flag per class and timed profile.
        """
        return {
            kind: {profile: value <= CV_LIMIT for profile, value in profiles.items()}
            for kind, profiles in cv.items()
        }

    @staticmethod
    def _mismatches(groups: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]) -> list[str]:
        """List instances whose evaluation-profile repeats differ in score or evaluations.

        Args:
            groups (Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]): Rows by instance
                and profile.

        Returns:
            list[str]: Instance names with differing repeats.
        """
        return sorted(
            instance
            for (instance, profile), group in groups.items()
            if profile == "evaluations"
            and len({(row["score"], row["evaluations"]) for row in group}) > 1
        )

    @staticmethod
    def _cv(
        groups: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    ) -> dict[str, dict[str, float]]:
        """Median per-instance CV of completed evaluations, per class and timed profile.

        Args:
            groups (Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]): Rows by instance
                and profile.

        Returns:
            dict[str, dict[str, float]]: Median CV per class and timed profile.
        """
        values: dict[tuple[str, str], list[float]] = defaultdict(list)
        for (_, profile), group in groups.items():
            if profile in PILOT_SECONDS:
                counts = [row["evaluations"] for row in group]
                values[(group[0]["class"], profile)].append(
                    statistics.stdev(counts) / statistics.mean(counts)
                )
        result: dict[str, dict[str, float]] = defaultdict(dict)
        for (kind, profile), series in values.items():
            result[kind][profile] = statistics.median(series)
        return dict(result)

    @staticmethod
    def _estimate(rows: Sequence[Mapping[str, Any]]) -> float:
        """Extrapolate the primary matrix cost in hours for one mechanism.

        Args:
            rows (Sequence[Mapping[str, Any]]): Pilot rows.

        Returns:
            float: Hours for the 30-instance, five-seed, two-arm matrix with the multiplier.
        """
        walls: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            walls[(row["stratum"], row["profile"])].append(row["wall_seconds"])
        seconds = sum(
            MATRIX_INSTANCES
            * MATRIX_SEEDS
            * 2
            * (
                statistics.mean(walls[(stratum, "evaluations")])
                + statistics.mean(walls[(stratum, "time10")])
            )
            for stratum in {stratum for stratum, _ in walls}
        )
        return seconds * COST_MULTIPLIER / 3600.0

    @staticmethod
    def verdict(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """Combine mechanism reports into PASS or STOP against the machine-hour ledger.

        Args:
            reports (Mapping[str, Mapping[str, Any]]): Report per mechanism.

        Returns:
            dict[str, Any]: Status, reasons, fit and the classes whose timed profile fails.
        """
        reasons = UtilityPilot._reasons(reports)
        spent = sum(report["pilot_hours"] for report in reports.values())
        estimated = sum(report["estimated_primary_hours"] for report in reports.values())
        remaining = CEILING_HOURS - spent
        return {
            "status": "STOP" if reasons else "PASS",
            "reasons": reasons,
            "fits": estimated <= remaining,
            "estimated_total_hours": estimated,
            "remaining_hours": remaining,
            "pilot_hours": spent,
            "timed_uninterpretable": UtilityPilot._uninterpretable(reports),
        }

    @staticmethod
    def _reasons(reports: Mapping[str, Mapping[str, Any]]) -> list[str]:
        """List why the ledger stops, or an empty list when it does not.

        Args:
            reports (Mapping[str, Mapping[str, Any]]): Report per mechanism.

        Returns:
            list[str]: Human-readable stop reasons.
        """
        reasons = [
            f"{name}: evaluation-profile repeats differ"
            for name, report in reports.items()
            if not report["deterministic"]
        ]
        spent = sum(report["pilot_hours"] for report in reports.values())
        estimated = sum(report["estimated_primary_hours"] for report in reports.values())
        if spent > PILOT_HOURS:
            reasons.append(f"pilot used {spent:.2f} machine-hours, above {PILOT_HOURS}")
        remaining = CEILING_HOURS - spent
        if estimated > remaining:
            reasons.append(f"estimated {estimated:.1f} h exceeds the remaining {remaining:.1f} h")
        return reasons

    @staticmethod
    def _uninterpretable(reports: Mapping[str, Mapping[str, Any]]) -> list[str]:
        """List mechanism/class pairs whose timed profile fails the stability bound.

        Args:
            reports (Mapping[str, Mapping[str, Any]]): Report per mechanism.

        Returns:
            list[str]: Sorted "mechanism:class" labels.
        """
        return sorted(
            f"{name}:{kind}"
            for name, report in reports.items()
            for kind, flags in report["interpretable"].items()
            if not flags.get("time10", True)
        )
