"""Tour-only confirmation of the frozen scorer against the strongest known seeded baseline.

Objective:
- Decide whether the frozen contextual scorer, searched with RKO, beats the nearest-neighbour plus
  2-opt tour seeded through `initial_keys`, the baseline that beat every earlier decoder-form,
  quantile and learned-scorer study on the traveling-salesman problem.

Technical approach:
- Four arms share one process worker and one wall allowance: the frozen scorer searched by BRKGA
  (`genetic`) and by the BRKGA, SA and ILS portfolio (`portfolio`), and the 2-opt tour with 31
  perturbed variants searched by the same two algorithm sets (`twoopt_genetic`,
  `twoopt_portfolio`). Only the starting tour, decoder and scorer differ inside each pair.
- Every arm is charged its own setup time (greedy reference or nearest-neighbour plus 2-opt and
  seed keys) out of the allowance, so neither side searches for free.
- The seed noise is the best of a small grid on four validation instances at a shorter allowance,
  so the baseline is tuned as much as the F5 neural arms were fixed.
- Test seeds are fresh: none was used by F4 training, validation or the F5 screen.

Flow (pseudocode):
1. Tune the seed noise on the validation panel with the 2-opt BRKGA arm.
2. Freeze the manifest with checkpoint fingerprint, noise, seeds and source revision.
3. Run every arm on every test instance, rechecking each tour independently.
4. Compare each neural arm with the 2-opt arm using the same algorithms, paired by instance.

Important criteria:
- Rule, committed before the real run: advance only if `genetic` beats `twoopt_genetic` or
  `portfolio` beats `twoopt_portfolio`, with a paired 95 percent bootstrap interval entirely below
  zero. Two comparisons are examined, so a single narrow win is weak evidence.
- Tour only, one randomization, development seeds. A win here is not release evidence.

Inputs/Outputs:
- Input: one frozen tour scorer and an output directory that must not exist.
- Output: `manifest.json`, `trials.jsonl` and `summary.json` under that directory.
"""

from __future__ import annotations

import json
import numpy as np
import subprocess
import time
from benchmarks.engines.form_study import SEED_COUNT
from benchmarks.engines.tsp_problems import TspCost
from benchmarks.engines.tsp_problems import TspDecoder
from benchmarks.engines.tsp_problems import TspFamily
from benchmarks.engines.tsp_problems import TspInstance
from benchmarks.quantile import study
from benchmarks.quantile.instances import Instances
from benchmarks.quantile.model import FrozenScorer
from benchmarks.quantile.screen import ARM_ALGORITHMS
from benchmarks.quantile.screen import ArmBuilder
from benchmarks.quantile.tasks import Tour
from pathlib import Path
from rko_solver import RKO
from typing import Any
from typing import TextIO
from typing import cast

ARMS = ("genetic", "portfolio", "twoopt_genetic", "twoopt_portfolio")
PAIRS = (
    ("genetic", "twoopt_genetic"),
    ("portfolio", "twoopt_portfolio"),
)
NOISES = (0.005, 0.01, 0.03, 0.1, 0.3)
BUDGET = 60.0
TUNING_BUDGET = 10.0
TEST_INSTANCES = 20
TEST_SEED_START = 720000
VALIDATION_INSTANCES = 4
MINIMUM_SEARCH = 0.1


class TwoOptArm:
    """Search a plain sort decoder seeded with a 2-opt tour and its perturbed variants.

    Responsibilities:
    - Build the nearest-neighbour plus 2-opt tour once and charge that time to every run.
    - Report a tour length rechecked by the task, not the solver's own number.

    Attributes:
        task (Tour): Fixed tour instance.
        noise (float): Seed-position noise as a fraction of the tour.
        instance (TspInstance): The same cities for the existing traveling-salesman helpers.
        tour (np.ndarray): 2-opt local optimum from the nearest-neighbour tour.
        prior_seconds (float): Time spent building that tour.
    """

    def __init__(self, task: Tour, noise: float) -> None:
        """Build the 2-opt tour and time it.

        Args:
            task (Tour): Fixed tour instance.
            noise (float): Non-negative seed-position noise.
        """
        started = time.perf_counter()
        self.task, self.noise = task, noise
        self.instance = TspInstance("tour", task.coordinates)
        self.tour = TspFamily.two_opt(self.instance, TspFamily.nearest_neighbour(self.instance))
        self.prior_seconds = time.perf_counter() - started

    @property
    def prior_cost(self) -> float:
        """Return the closed length of the 2-opt seed tour.

        Returns:
            float: Seed tour length.
        """
        return TspFamily.length(self.instance, self.tour)

    def run(
        self, algorithms: tuple[str, ...], seed: int, allowance: float
    ) -> tuple[float, float, int]:
        """Search from the seed keys under the allowance minus the setup time.

        Args:
            algorithms (tuple[str, ...]): RKO algorithm names.
            seed (int): Search and perturbation seed.
            allowance (float): Common wall allowance in seconds.

        Returns:
            tuple[float, float, int]: Rechecked tour length, charged setup seconds and
                evaluations.

        Raises:
            ValueError: If the reported cost does not recheck.
        """
        started = time.perf_counter()
        keys = TspFamily.perturbed_keys(self.tour, SEED_COUNT, self.noise, seed)
        setup = self.prior_seconds + time.perf_counter() - started
        decoder = TspDecoder(self.instance)
        result = RKO(
            decoder,
            TspCost(self.instance),
            self.instance.n_cities,
            algorithms=list(algorithms),
            adaptation="fixed",
            execution="process",
            n_jobs=1,
            initial_keys=keys,
            random_state=seed,
        ).solve(max(allowance - setup, MINIMUM_SEARCH))
        order = [int(city) for city in decoder(np.asarray(result.keys))]
        start = order.index(0)
        closed = order[start:] + order[:start]
        checked = self.task.check((*closed, 0))
        if not np.isclose(checked, result.cost, rtol=1e-9, atol=1e-9):
            raise ValueError("A reported trial cost did not independently recheck.")
        return checked, setup, result.n_evaluations


class NeuralArm:
    """Search the frozen scorer's construction keys from the shared greedy reference.

    Responsibilities:
    - Charge the reference construction time to the arm's allowance.
    """

    @staticmethod
    def run(
        task: Tour, scorer: FrozenScorer, arm: str, seed: int, allowance: float
    ) -> tuple[float, float, int]:
        """Search one neural arm under the allowance minus the reference time.

        Args:
            task (Tour): Fixed tour instance.
            scorer (FrozenScorer): Frozen tour scorer.
            arm (str): `genetic` or `portfolio`.
            seed (int): Search seed.
            allowance (float): Common wall allowance in seconds.

        Returns:
            tuple[float, float, int]: Rechecked tour length, charged setup seconds and
                evaluations.

        Raises:
            ValueError: If the reported cost does not recheck.
        """
        builder = ArmBuilder(task, scorer)
        started = time.perf_counter()
        reference = builder.reference_trace()
        setup = time.perf_counter() - started
        result = builder.run(arm, seed, max(allowance - setup, MINIMUM_SEARCH), reference)
        decoded = builder.bridge(arm).codec.decode(np.asarray(result.keys))
        checked = task.check(decoded.solution)
        if not np.isclose(checked, result.cost, rtol=1e-9, atol=1e-9):
            raise ValueError("A reported trial cost did not independently recheck.")
        return checked, setup, result.n_evaluations


class Confirmation:
    """Run the Tour-only confirmation and apply its pre-registered rule.

    Responsibilities:
    - Freeze evidence before any measured comparison and never overwrite an earlier run.

    Attributes:
        output (Path): New output directory.
        scorer (FrozenScorer): Frozen tour scorer.
    """

    def __init__(self, output: Path, scorer: FrozenScorer) -> None:
        """Store the output location and checkpoint without running anything.

        Args:
            output (Path): New output directory.
            scorer (FrozenScorer): Frozen scorer whose family is `tour`.

        Raises:
            ValueError: If the scorer is not a tour scorer.
        """
        if scorer.family != "tour":
            raise ValueError("The confirmation needs a tour scorer.")
        self.output, self.scorer = output, scorer

    def run(self) -> dict[str, Any]:
        """Tune the noise, freeze the manifest, run every trial and apply the rule.

        Returns:
            dict[str, Any]: Paired comparisons, arm means and the advance decision.

        Raises:
            FileExistsError: If the output directory already exists.
            ValueError: If a reported trial cost does not independently recheck.
            RuntimeError: If a worker fails or exits unexpectedly.
            OSError: If evidence cannot be persisted.
        """
        self.output.mkdir(parents=True, exist_ok=False)
        noise = self._tune_noise()
        panel = self._panel(TEST_SEED_START, TEST_INSTANCES)
        self._freeze_manifest(noise, panel)
        rows: list[dict[str, Any]] = []
        with (self.output / "trials.jsonl").open("a", encoding="utf-8") as stream:
            for seed, task, _ in panel:
                rows.extend(self._run_instance(seed, task, noise, stream))
        summary = {**self.summarize(rows), "noise": noise}
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    @staticmethod
    def _panel(start: int, count: int) -> list[tuple[int, Tour, float]]:
        """Generate consecutive tour instances with their analytic greedy reference cost.

        Args:
            start (int): First instance seed.
            count (int): Number of instances.

        Returns:
            list[tuple[int, Tour, float]]: Seed, instance and reference cost in seed order.
        """
        rows = study.Panel.make("tour", start, count)
        return [(seed, cast(Tour, task), reference) for seed, task, reference in rows]

    def _tune_noise(self) -> float:
        """Choose the seed noise with the lowest mean relative length on validation instances.

        Returns:
            float: Best noise from `NOISES`.
        """
        panel = self._panel(study.VALIDATION_SEED_START, VALIDATION_INSTANCES)
        scores = []
        for noise in NOISES:
            relative = []
            for seed, task, _ in panel:
                arm = TwoOptArm(task, noise)
                cost, _, _ = arm.run(ARM_ALGORITHMS["genetic"], seed, TUNING_BUDGET)
                relative.append(cost / arm.prior_cost)
            scores.append(float(np.mean(relative)))
        return NOISES[int(np.argmin(scores))]

    def _freeze_manifest(self, noise: float, panel: list[tuple[int, Tour, float]]) -> None:
        """Persist the checkpoint, tuned noise and test-panel identities once.

        Args:
            noise (float): Tuned seed noise.
            panel (list[tuple[int, Tour, float]]): Frozen test instances.

        Raises:
            OSError: If the manifest cannot be written.
        """
        manifest = {
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "checkpoint_fingerprint": self.scorer.fingerprint,
            "arms": ARMS,
            "budget": BUDGET,
            "noise": noise,
            "test_panel": [
                {"seed": seed, "instance_hash": Instances.fingerprint(task)}
                for seed, task, _ in panel
            ],
            "claim": "development confirmation on fresh seeds, not release evidence",
        }
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def _run_instance(
        self, seed: int, task: Tour, noise: float, stream: TextIO
    ) -> list[dict[str, Any]]:
        """Run and persist every arm on one test instance.

        Args:
            seed (int): Instance seed, also the search seed.
            task (Tour): Test instance.
            noise (float): Tuned seed noise.
            stream (TextIO): Open append-mode stream for raw records.

        Returns:
            list[dict[str, Any]]: One raw row per arm.

        Raises:
            ValueError: If a reported trial cost does not independently recheck.
            RuntimeError: If a worker fails or exits unexpectedly.
            OSError: If a raw record cannot be persisted.
        """
        seeded = TwoOptArm(task, noise)
        rows = []
        for arm in ARMS:
            started = time.perf_counter()
            if arm.startswith("twoopt_"):
                cost, setup, evaluations = seeded.run(ARM_ALGORITHMS[arm[7:]], seed, BUDGET)
            else:
                cost, setup, evaluations = NeuralArm.run(task, self.scorer, arm, seed, BUDGET)
            row = {
                "seed_group": seed,
                "arm": arm,
                "budget": BUDGET,
                "cost": cost,
                "setup_seconds": setup,
                "n_evaluations": evaluations,
                "wall_seconds": time.perf_counter() - started,
            }
            rows.append(row)
        stream.writelines(json.dumps(row) + "\n" for row in rows)
        stream.flush()
        return rows

    @staticmethod
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Apply the pre-registered rule to the paired comparisons.

        Args:
            rows (list[dict[str, Any]]): Raw trial records of every arm and instance.

        Returns:
            dict[str, Any]: Paired comparisons, arm mean costs and the advance decision.
        """
        averages = {arm: study.PairedComparison.average(rows, arm, BUDGET) for arm in ARMS}
        comparisons = {
            f"{neural}_vs_{baseline}": study.PairedComparison.compare(
                averages[neural], averages[baseline]
            )
            for neural, baseline in PAIRS
        }
        return {
            "comparisons": comparisons,
            "mean_costs": {arm: float(np.mean(list(averages[arm].values()))) for arm in ARMS},
            "advance": any(value["ci_high"] < 0.0 for value in comparisons.values()),
            "rule": (
                "advance only if a neural arm beats the 2-opt seeded arm with the same "
                "algorithms and a paired 95 percent interval entirely below zero"
            ),
        }
