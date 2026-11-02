"""Train a per-item scorer for the traveling-salesman problem and test it against hand priors.

Objective:
- Decide whether a REINFORCE-trained per-city scorer, with no hand-written geometric rule, beats
  the hand-built priors (nearest neighbour, 2-opt) already measured on the same problem and
  families, before any neural decoder is considered for this library.

Technical approach:
- `TspFamily.features` gives five instance-only per-city columns (centered coordinates, rank-
  normalized distance to the centroid and to the nearest other city, polar angle). A one-hidden-
  layer scorer (`PriorityScorer`, already used for packing) is trained offline per family with
  `ReinforceTrainer`, the same REINFORCE and leave-one-out machinery, on CPU only, no GPU.
- The training reward is the 2-opt tour's length divided by the sampled order's length (so the
  2-opt prior scores exactly 1 as its own reference), via `RewardCache`.
- Five arms per instance at one charged allowance: plain RKO; the learned score as a sort order
  seeded with 32 perturbed orders (`learned_seed32`); the learned score inside the residual
  decoder (`learned_resid`); the 2-opt prior seeded the same way (`twoopt_seed32`, the mechanism
  that beat the residual decoder in the earlier studies); and the 2-opt prior inside the residual
  decoder (`twoopt_resid`), both reused as the strongest known baselines.
- The residual spread for both resid arms is chosen on validation instances at the shorter budget,
  the same selection the earlier decoder-form study used.

Flow (pseudocode):
1. Train one scorer per family on training seeds, disjoint from validation and test.
2. Choose the residual spread per family and arm on validation instances.
3. Solve every test instance with every arm at every budget; verify every returned key vector.
4. Compare the learned arms with the 2-opt baselines, paired over instances.

Important criteria:
- Decision rule, fixed before any result: advance only if, on both families at both budgets, the
  learned score beats the 2-opt baseline with the same mechanism (seeded and residual), each with
  a paired 95 percent bootstrap interval entirely below zero.
- Training reward and wall time are reported per family, separate from the test comparison.
- Training, validation and test seeds are pairwise disjoint.

Inputs/Outputs:
- Input: design and training settings and an output path on the command line.
- Output: a JSON file with training reports, every trial, pair statistics and the decision.
"""

from __future__ import annotations

import argparse
import numpy as np
import time
from benchmarks.engines.form_study import SEED_COUNT
from benchmarks.engines.form_study import SEED_NOISE
from benchmarks.engines.priority_scorer import PriorityScorer
from benchmarks.engines.priority_scorer import ReinforceStudyDesign
from benchmarks.engines.priority_scorer import ReinforceTrainer
from benchmarks.engines.priority_scorer import RewardCache
from benchmarks.engines.report_sink import ReportSink
from benchmarks.engines.solver_run import SolverRun
from benchmarks.engines.tsp_decoder_form_study import TspDecoderFormStudy
from benchmarks.engines.tsp_problems import KINDS
from benchmarks.engines.tsp_problems import TspCost
from benchmarks.engines.tsp_problems import TspDecoder
from benchmarks.engines.tsp_problems import TspFamily
from benchmarks.engines.tsp_problems import TspInstance
from benchmarks.engines.tsp_problems import TspReport
from benchmarks.engines.tsp_problems import TspStudyBase
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARMS = ("plain", "learned_seed32", "learned_resid", "twoopt_seed32", "twoopt_resid")
PAIRS = (
    ("learned_seed32", "twoopt_seed32"),
    ("learned_resid", "twoopt_resid"),
    ("learned_resid", "plain"),
)
SPREADS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
MINIMUM_SEARCH = 0.05


@dataclass(slots=True)
class LearnedStudyDesign(ReinforceStudyDesign):
    """Fixed numerical design of the study, including training settings.

    Responsibilities:
    - Redeclare the base design's defaults for 100-city traveling-salesman instances and a
      larger training budget than the packing study uses.
    """

    n_items: int = 100
    validation_instances: int = 4
    train_seeds: tuple[int, int] = (100_000, 110_000)
    validation_base: int = 6500
    test_base: int = 8000
    steps: int = 3000
    batch: int = 16
    samples: int = 32
    hidden: int = 32


class LearnedTspDecision:
    """Apply the decision rule fixed before the study.

    Responsibilities:
    - Decide from paired statistics whether the learned scorer clearly beats the 2-opt baseline.
    """

    @staticmethod
    def verdict(tables: dict[str, dict[str, dict[str, dict[str, float]]]]) -> dict[str, Any]:
        """Evaluate the rule on paired comparisons.

        Args:
            tables (dict[str, dict[str, dict[str, dict[str, float]]]]): Statistics keyed by
                family, budget label and `arm-vs-arm` pair, each holding `ci_high`.

        Returns:
            dict[str, Any]: `advance`, true only if both `learned_seed32-vs-twoopt_seed32` and
            `learned_resid-vs-twoopt_resid` have `ci_high` below zero in every family and budget.
        """
        advance = all(
            budget[pair]["ci_high"] < 0
            for family in tables.values()
            for budget in family.values()
            for pair in ("learned_seed32-vs-twoopt_seed32", "learned_resid-vs-twoopt_resid")
        )
        return {"advance": advance}


class TspLearnedStudy(TspStudyBase):
    """Train a per-city scorer and compare it with the 2-opt baseline.

    Responsibilities:
    - Train one scorer per family and report its learning curve.
    - Build each arm's decoder and seed keys and verify every returned key vector.

    Attributes:
        design (LearnedStudyDesign): Fixed design.
    """

    def __init__(self, design: LearnedStudyDesign) -> None:
        """Store the design and start an empty bound cache.

        Args:
            design (LearnedStudyDesign): Study design.
        """
        super().__init__(design.n_items)
        self.design = design

    def train(self, kind: str, seed: int) -> tuple[PriorityScorer, dict[str, Any]]:
        """Train a scorer for one family and report its learning curve.

        Args:
            kind (str): Instance family.
            seed (int): Training seed.

        Returns:
            tuple[PriorityScorer, dict[str, Any]]: Fitted scorer and its report.
        """
        design = self.design
        scorer = PriorityScorer.init(5, design.hidden, seed)
        reward = RewardCache(
            lambda instance, order: 1.0 / TspFamily.length(instance, order),
            lambda instance: TspDecoderFormStudy.prior("twoopt", instance),
        )
        trainer = ReinforceTrainer(scorer, TspFamily.features, reward, design.samples, seed)

        def make(instance_seed: int) -> TspInstance:
            """Generate one training instance of the family.

            Args:
                instance_seed (int): Training seed.

            Returns:
                TspInstance: Generated instance.
            """
            return TspFamily.make(kind, design.n_items, instance_seed)

        held = [
            self.instance(kind, design.validation_base + k)
            for k in range(design.validation_instances)
        ]
        report = trainer.fit(
            kind, make, held, range(*design.train_seeds), (design.steps, design.batch)
        )
        return scorer, report

    def tour_for(self, arm: str, instance: TspInstance, scorer: PriorityScorer) -> np.ndarray:
        """Return the arm's prior tour: 2-opt for the baseline arms, the scorer's order otherwise.

        Args:
            arm (str): One of `ARMS`.
            instance (TspInstance): Instance.
            scorer (PriorityScorer): Trained scorer of the family.

        Returns:
            np.ndarray: Permutation of the cities.
        """
        if arm.startswith("learned"):
            scores = scorer.scores(TspFamily.features(instance))
            return np.argsort(-scores, kind="stable")
        return TspDecoderFormStudy.prior("twoopt", instance)

    def setup(
        self, arm: str, instance: TspInstance, spreads: dict[str, float], scorer: PriorityScorer
    ) -> tuple[TspDecoder, list[list[float]] | None, float]:
        """Build an arm's decoder and seed keys and time their preparation.

        Args:
            arm (str): One of `ARMS`.
            instance (TspInstance): Instance.
            spreads (dict[str, float]): Chosen residual spread per family.
            scorer (PriorityScorer): Trained scorer of the family.

        Returns:
            tuple[TspDecoder, list[list[float]] | None, float]: Decoder, initial keys or None,
            and preparation seconds.
        """
        if arm == "plain":
            return TspDecoder(instance), None, 0.0
        started = time.perf_counter()
        tour = self.tour_for(arm, instance, scorer)
        if arm.endswith("_resid"):
            decoder, keys = TspDecoder(instance, tour, spreads[arm]), None
        else:
            decoder = TspDecoder(instance)
            keys = TspFamily.perturbed_keys(tour, SEED_COUNT, SEED_NOISE, 0)
        return decoder, keys, time.perf_counter() - started

    def solve(
        self,
        instance: TspInstance,
        arm: str,
        context: tuple[dict[str, float], float],
        seed: int,
        scorer: PriorityScorer,
    ) -> dict[str, Any]:
        """Run one arm on one instance within one allowance.

        Args:
            instance (TspInstance): Instance.
            arm (str): One of `ARMS`.
            context (tuple[dict[str, float], float]): Chosen spreads and the allowance.
            seed (int): Instance seed.
            scorer (PriorityScorer): Trained scorer of the family.

        Returns:
            dict[str, Any]: Arm, budget, tour length, verification and phase timings.
        """
        spreads, budget = context
        decoder, keys, prep = self.setup(arm, instance, spreads, scorer)
        allowance = max(budget - prep, MINIMUM_SEARCH)
        started = time.perf_counter()
        length, verified = SolverRun.solve(
            decoder,
            TspCost(instance),
            instance.n_cities,
            (self.design.execution, self.design.algorithms),
            (seed, allowance),
            keys,
        )
        timing = (prep, allowance, time.perf_counter() - started + prep)
        return self.record(instance, (arm, budget), length, verified, timing)

    def choose_spread(self, kind: str, arm: str, scorer: PriorityScorer) -> float:
        """Pick the residual spread with the lowest mean length over the bound on validation.

        Args:
            kind (str): Instance family.
            arm (str): `learned_resid` or `twoopt_resid`.
            scorer (PriorityScorer): Trained scorer of the family.

        Returns:
            float: Chosen spread.
        """
        budget = min(self.design.budgets)
        means = {}
        for spread in SPREADS:
            ratios = []
            for offset in range(self.design.validation_instances):
                seed = self.design.validation_base + offset
                row = self.solve(
                    self.instance(kind, seed), arm, ({arm: spread}, budget), seed, scorer
                )
                ratios.append(row["length"] / row["bound"])
            means[spread] = float(np.mean(ratios))
        return min(means, key=lambda value: means[value])

    def evaluate(
        self, kind: str, spreads: dict[str, float], seed: int, scorer: PriorityScorer
    ) -> list[dict[str, Any]]:
        """Run every arm at every budget on one test instance.

        Args:
            kind (str): Instance family.
            spreads (dict[str, float]): Chosen residual spread per arm.
            seed (int): Test instance seed.
            scorer (PriorityScorer): Trained scorer of the family.

        Returns:
            list[dict[str, Any]]: One record per arm and budget.
        """
        instance = self.instance(kind, seed)
        rows = []
        for budget in self.design.budgets:
            for arm in ARMS:
                rows.append(self.solve(instance, arm, (spreads, budget), seed, scorer))
                self.progress(rows[-1])
        return rows

    def summarize(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Compare arm pairs per family and budget and apply the decision rule.

        Args:
            rows (list[dict[str, Any]]): Normalized trial records.

        Returns:
            dict[str, Any]: Mean gaps, paired statistics and the verdict.
        """
        return ReportSink.summarize_tables(
            (KINDS, ARMS, PAIRS, self.design.budgets), rows, LearnedTspDecision.verdict
        )

    def run(self) -> dict[str, Any]:
        """Train a scorer per family, choose spreads, evaluate and summarize.

        Returns:
            dict[str, Any]: Training reports, chosen spreads, raw trials and the summary.
        """
        scorers: dict[str, PriorityScorer] = {}
        training: dict[str, dict[str, Any]] = {}
        for kind in KINDS:
            scorers[kind], training[kind] = self.train(kind, 0)
        spreads = {
            kind: {
                arm: self.choose_spread(kind, arm, scorers[kind])
                for arm in ("learned_resid", "twoopt_resid")
            }
            for kind in KINDS
        }
        rows: list[dict[str, Any]] = []
        for kind in KINDS:
            for offset in range(self.design.test_instances):
                rows.extend(
                    self.evaluate(
                        kind, spreads[kind], self.design.test_base + offset, scorers[kind]
                    )
                )
        rows = TspReport.normalize(rows)
        return {
            "training": training,
            "spreads": spreads,
            "trials": rows,
            "summary": self.summarize(rows),
        }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the learned-scorer study and write its JSON report.

    Args:
        argv (Sequence[str] | None): Arguments, or None for `sys.argv`.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--n-items", type=int, default=100)
    parser.add_argument("--test-instances", type=int, default=12)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=Path("results/tsp-learned-study.json"))
    options = parser.parse_args(argv)
    design = LearnedStudyDesign(
        n_items=options.n_items, test_instances=options.test_instances, steps=options.steps
    )
    report = {"design": asdict(design)} | TspLearnedStudy(design).run()
    ReportSink.write(report, options.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
