"""Bounded two-family search screen comparing the frozen model across four fixed arms.

Objective:
- Decide whether guided search over the frozen scorer's keys adds measurable value
  beyond construction quality alone, under equal checked starts and search budgets.

Technical approach:
- Freeze a 10-instance test panel per family from previously unseen seeds, after
  checking headroom on a small disjoint validation panel.
- Run every arm (`benchmarks/quantile/screen.py`) on every test instance, repeated
  for three independent search randomizations, at both the 10- and 60-second
  common wall-time allowances declared by the plan.
- Average randomizations per instance before any paired comparison, then pair by
  instance identity and apply the same bootstrap procedure used for F4.

Flow (pseudocode):
1. Check validation-panel headroom: the frozen scorer must beat the analytic
   construction reference before any test instance is touched.
2. Freeze the 10-instance-per-family test panel from unseen seeds.
3. Run every arm/instance/randomization/budget cell and record raw results.
4. Average per instance, pair by identity, and apply the prespecified exit gate.

Important criteria:
- Never reuse F4's tune/qualification seeds (600000-607, 610000-617) or training
  seeds (100000+, 200000+); this screen's seeds must be unseen by the checkpoint.
- A favorable screen permits confirmation planning; it is not a release claim.

Inputs/Outputs:
- Input: a frozen scorer per family and this module's fixed design constants.
- Output: `summary.json` (paired comparisons and the advance decision) and
  `trials.jsonl` (every raw observation) under an explicit output directory.
"""

from __future__ import annotations

import json
import numpy as np
import subprocess
import time
from benchmarks.quantile.bootstrap import paired_interval
from benchmarks.quantile.codec import TrajectoryCodec
from benchmarks.quantile.instances import Instances
from benchmarks.quantile.model import FrozenScorer
from benchmarks.quantile.screen import ARMS
from benchmarks.quantile.screen import ArmBuilder
from benchmarks.quantile.tasks import Cover
from benchmarks.quantile.tasks import Tour
from pathlib import Path
from typing import Any
from typing import TextIO

FAMILIES = ("tour", "cover")
BUDGETS = (10.0, 60.0)
RANDOMIZATIONS = 3
TEST_INSTANCES = 10
VALIDATION_SEED_START = 700000
TEST_SEED_START = 710000
SIZE = 100


class HeadroomFailure(RuntimeError):
    """Signal that the frozen scorer did not beat the analytic reference on validation data.

    Responsibilities:
    - Stop the screen before any test-panel instance is touched or frozen.
    """

    def __init__(self, family: str, difference: float) -> None:
        """Describe the failed family and its measured validation-panel difference.

        Args:
            family (str): Task family whose validation headroom check failed.
            difference (float): Mean neural-minus-analytic normalized construction cost.
        """
        super().__init__(
            f"Validation headroom check failed for family '{family}': "
            f"mean normalized difference {difference:.6f} is not negative."
        )
        self.family, self.difference = family, difference


class Trial:
    """Run and record one arm/instance/randomization/budget observation.

    Responsibilities:
    - Independently recheck every produced solution against the task's own check.
    """

    @staticmethod
    def run(
        task: Tour | Cover, scorer: FrozenScorer, arm: str, seed: int, budget: float
    ) -> dict[str, Any]:
        """Run one arm once and independently recheck its reported cost.

        Args:
            task (Tour | Cover): Fixed test-panel instance.
            scorer (FrozenScorer): Frozen checkpoint for this instance's family.
            arm (str): One of `benchmarks.quantile.screen.ARMS`.
            seed (int): Search randomization seed.
            budget (float): Common wall-time allowance in seconds.

        Returns:
            dict[str, Any]: Raw trial record, including an independent cost recheck.

        Raises:
            ValueError: If the reported solution does not independently recheck.
            RuntimeError: If a worker fails or exits unexpectedly.
        """
        builder = ArmBuilder(task, scorer)
        reference = builder.reference_trace()
        started = time.perf_counter()
        result = builder.run(arm, seed, budget, reference)
        elapsed = time.perf_counter() - started
        keys = np.asarray(result.keys)
        checked = task.check(builder.bridge(arm).codec.decode(keys).solution)
        if not np.isclose(checked, result.cost, rtol=1e-9, atol=1e-9):
            raise ValueError("A reported trial cost did not independently recheck.")
        return {
            "arm": arm,
            "seed": seed,
            "budget": budget,
            "cost": checked,
            "n_evaluations": result.n_evaluations,
            "wall_seconds": elapsed,
        }


class Panel:
    """Build disjoint validation and test instance panels from declared seed ranges.

    Responsibilities:
    - Keep validation and test seeds disjoint from each other and from F4's seeds.
    """

    @staticmethod
    def make(family: str, start: int, count: int) -> list[tuple[int, Tour | Cover, float]]:
        """Generate one panel of checked instances with their analytic reference cost.

        Args:
            family (str): 'tour' or 'cover'.
            start (int): First seed; seeds are consecutive from this value.
            count (int): Number of instances.

        Returns:
            list[tuple[int, Tour | Cover, float]]: Seed, instance and analytic
                greedy reference cost, in seed order.
        """
        rows = []
        for seed in range(start, start + count):
            task = Instances.make(family, SIZE, seed)
            reference = TrajectoryCodec(task).greedy().cost
            rows.append((seed, task, reference))
        return rows


class HeadroomCheck:
    """Confirm the frozen scorer beats the analytic construction reference on validation data.

    Responsibilities:
    - Stop the screen before freezing or touching any test instance on failed headroom.
    """

    @staticmethod
    def run(family: str, scorer: FrozenScorer, instances: int) -> float:
        """Compare mean normalized sampled-construction cost against the analytic reference.

        Uses the same methodology that qualified this checkpoint in F4: matched
        random decision keys decoded through each scorer in turn, averaged over
        several samples per instance, not a greedy/argmax construction. Both
        candidates see the identical key sequence per instance so the comparison
        isolates the scorer, not the random draw.

        Args:
            family (str): 'tour' or 'cover'.
            scorer (FrozenScorer): Frozen checkpoint under test.
            instances (int): Validation-panel size.

        Returns:
            float: Mean neural-minus-analytic normalized difference; negative favors neural.

        Raises:
            HeadroomFailure: If the mean difference is not negative.
        """
        panel = Panel.make(family, VALIDATION_SEED_START, instances)
        differences = []
        for seed, task, reference in panel:
            neural_rng = np.random.default_rng([0, seed, 319])
            neural = float(
                np.mean(
                    [
                        TrajectoryCodec(task, scorer=scorer)
                        .decode(neural_rng.random(task.horizon))
                        .cost
                        for _ in range(4)
                    ]
                )
            )
            analytic_rng = np.random.default_rng([0, seed, 319])
            analytic = float(
                np.mean(
                    [
                        TrajectoryCodec(task).decode(analytic_rng.random(task.horizon)).cost
                        for _ in range(4)
                    ]
                )
            )
            differences.append((neural - analytic) / max(reference, 1e-12))
        mean_difference = float(np.mean(differences))
        if mean_difference >= 0.0:
            raise HeadroomFailure(family, mean_difference)
        return mean_difference


class PairedComparison:
    """Pair randomization-averaged per-instance costs by instance identity.

    Responsibilities:
    - Average randomizations before any paired statistic, never treat them as instances.
    """

    @staticmethod
    def average(rows: list[dict[str, Any]], arm: str, budget: float) -> dict[int, float]:
        """Average every randomization's cost per seed for one arm and budget.

        Args:
            rows (list[dict[str, Any]]): Raw trial records for one family.
            arm (str): Study arm to select.
            budget (float): Common wall-time allowance to select.

        Returns:
            dict[int, float]: Mean cost per seed.
        """
        selected: dict[int, list[float]] = {}
        for row in rows:
            if row["arm"] == arm and row["budget"] == budget:
                selected.setdefault(row["seed_group"], []).append(row["cost"])
        return {seed: float(np.mean(costs)) for seed, costs in selected.items()}

    @staticmethod
    def compare(candidate: dict[int, float], control: dict[int, float]) -> dict[str, Any]:
        """Bootstrap the paired candidate-minus-control difference by instance identity.

        Args:
            candidate (dict[int, float]): Mean cost per seed for the candidate arm.
            control (dict[int, float]): Mean cost per seed for the control arm.

        Returns:
            dict[str, Any]: Mean difference and a paired 95 percent bootstrap interval.

        Raises:
            ValueError: If the two arms do not share exactly the same instance identities.
        """
        if set(candidate) != set(control):
            raise ValueError("Paired comparisons require matching instance identities.")
        seeds = sorted(candidate)
        differences = np.asarray([candidate[seed] - control[seed] for seed in seeds])
        return paired_interval(differences)


class Screen:
    """Run the complete F5 development screen and apply its prespecified exit gate.

    Responsibilities:
    - Freeze evidence before any measured comparison and never overwrite it.

    Attributes:
        output (Path): New output directory; existing screens are never overwritten.
        checkpoints (dict[str, FrozenScorer]): Frozen scorer per family.
    """

    def __init__(self, output: Path, checkpoints: dict[str, FrozenScorer]) -> None:
        """Store the output location and frozen checkpoints without running anything.

        Args:
            output (Path): New output directory.
            checkpoints (dict[str, FrozenScorer]): Frozen scorer per family in `FAMILIES`.

        Raises:
            ValueError: If a checkpoint is missing or its family does not match.
        """
        if set(checkpoints) != set(FAMILIES):
            raise ValueError("A frozen checkpoint is required for every declared family.")
        for family, scorer in checkpoints.items():
            if scorer.family != family:
                raise ValueError("A checkpoint's family must match its declared family key.")
        self.output, self.checkpoints = output, checkpoints

    def freeze_manifest(self, panels: dict[str, list[tuple[int, Tour | Cover, float]]]) -> None:
        """Persist the frozen test-panel identities and checkpoint fingerprints once.

        Args:
            panels (dict[str, list[tuple[int, Tour | Cover, float]]]): Test panel per family.

        Raises:
            OSError: If the manifest cannot be written.
        """
        manifest = {
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "checkpoint_fingerprints": {
                family: scorer.fingerprint for family, scorer in self.checkpoints.items()
            },
            "arms": ARMS,
            "budgets": BUDGETS,
            "randomizations": RANDOMIZATIONS,
            "test_panel": {
                family: [
                    {"seed": seed, "instance_hash": Instances.fingerprint(task)}
                    for seed, task, _ in rows
                ]
                for family, rows in panels.items()
            },
            "claim": "development screen, not release or confirmation evidence",
        }
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def run(self) -> dict[str, Any]:
        """Run the complete screen: headroom check, panel freeze, matrix and exit gate.

        Returns:
            dict[str, Any]: Per-family paired comparisons and the final advance decision.

        Raises:
            FileExistsError: If the output directory already exists.
            HeadroomFailure: If a family fails the validation-panel headroom check.
            ValueError: If a reported trial cost does not independently recheck.
            RuntimeError: If a worker fails or exits unexpectedly.
            OSError: If evidence cannot be persisted.
        """
        self.output.mkdir(parents=True, exist_ok=False)
        for family, scorer in self.checkpoints.items():
            HeadroomCheck.run(family, scorer, instances=4)
        panels = {
            family: Panel.make(family, TEST_SEED_START, TEST_INSTANCES) for family in FAMILIES
        }
        self.freeze_manifest(panels)
        rows: list[dict[str, Any]] = []
        with (self.output / "trials.jsonl").open("a", encoding="utf-8") as stream:
            for family, panel in panels.items():
                rows.extend(self._run_family_panel(family, panel, stream))
        summary = self.summarize(rows)
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    def _run_family_panel(
        self, family: str, panel: list[tuple[int, Tour | Cover, float]], stream: TextIO
    ) -> list[dict[str, Any]]:
        """Run and persist every arm/budget/randomization trial for one family's panel.

        Args:
            family (str): Task family owning this panel.
            panel (list[tuple[int, Tour | Cover, float]]): Frozen test instances.
            stream (TextIO): Open append-mode text stream for raw trial records.

        Returns:
            list[dict[str, Any]]: Every raw trial row produced for this family.

        Raises:
            ValueError: If a reported trial cost does not independently recheck.
            RuntimeError: If a worker fails or exits unexpectedly.
            OSError: If a raw trial record cannot be persisted.
        """
        rows = []
        for seed, task, _ in panel:
            for arm in ARMS:
                for budget in BUDGETS:
                    for replicate in range(RANDOMIZATIONS):
                        trial = Trial.run(
                            task, self.checkpoints[family], arm, seed * 10 + replicate, budget
                        )
                        row = {"family": family, "seed_group": seed, **trial}
                        rows.append(row)
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
        return rows

    @staticmethod
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Apply the prespecified exit gate over every family/budget cell.

        A cell wins when `genetic` or `portfolio` beats `independent` with a
        confidence interval strictly below zero, and neither arm is beaten by
        `classical` with a confidence interval strictly above zero. The screen
        advances only if every family/budget cell wins and no cell fails a
        correctness recheck (already enforced in `Trial.run`).

        Args:
            rows (list[dict[str, Any]]): Every raw trial record.

        Returns:
            dict[str, Any]: Per-cell paired comparisons and the overall advance decision.
        """
        cells = {
            f"{family}@{budget:g}s": Screen._evaluate_cell(rows, family, budget)
            for family in FAMILIES
            for budget in BUDGETS
        }
        wins = [key for key, cell in cells.items() if cell["won"]]
        return {
            "cells": cells,
            "wins": wins,
            "advance": len(wins) == len(FAMILIES) * len(BUDGETS),
            "rule": (
                "development screen: a structured-search arm must beat independent "
                "sampling with a negative interval and not be beaten by classical "
                "construction, in every family/budget cell"
            ),
        }

    @staticmethod
    def _evaluate_cell(rows: list[dict[str, Any]], family: str, budget: float) -> dict[str, Any]:
        """Apply the exit-gate rule to one family/budget cell's four-arm comparisons.

        Args:
            rows (list[dict[str, Any]]): Every raw trial record.
            family (str): Task family for this cell.
            budget (float): Common wall-time allowance for this cell.

        Returns:
            dict[str, Any]: This cell's paired comparisons and its win decision.
        """
        family_rows = [row for row in rows if row["family"] == family]
        averages = {arm: PairedComparison.average(family_rows, arm, budget) for arm in ARMS}
        comparisons = {
            "genetic_vs_independent": PairedComparison.compare(
                averages["genetic"], averages["independent"]
            ),
            "portfolio_vs_independent": PairedComparison.compare(
                averages["portfolio"], averages["independent"]
            ),
            "genetic_vs_classical": PairedComparison.compare(
                averages["genetic"], averages["classical"]
            ),
            "portfolio_vs_classical": PairedComparison.compare(
                averages["portfolio"], averages["classical"]
            ),
        }
        structured_beats_independent = (
            comparisons["genetic_vs_independent"]["ci_high"] < 0.0
            or comparisons["portfolio_vs_independent"]["ci_high"] < 0.0
        )
        beaten_by_classical = (
            comparisons["genetic_vs_classical"]["ci_low"] > 0.0
            and comparisons["portfolio_vs_classical"]["ci_low"] > 0.0
        )
        won = structured_beats_independent and not beaten_by_classical
        return {"comparisons": comparisons, "won": won}
