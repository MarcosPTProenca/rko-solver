"""Compare a REINFORCE-trained item scorer with the hand-written prior at equal charged time.

Objective:
- Decide whether a per-family learned priority beats the hand-written ratio rule, which already
  captures most of the value of any prior, before any library feature is considered.

Technical approach:
- Train one scorer per family offline with REINFORCE on generated instances whose seeds are
  disjoint from validation and test seeds; training time is reported, not charged per instance.
- Run five arms on held-out instances with the default portfolio and one shared wall allowance
  per instance: plain sort decoder, hand prior as a seed, learned prior as a seed, hand prior in
  a residual decoder and learned prior in a residual decoder. Each arm's score computation is
  charged to its allowance. The residual spread of each decoder is chosen on validation seeds.
- Quality is the percentage gap to the best profit any arm reached on the instance.

Flow (pseudocode):
1. Train a scorer for each family and record its reward history and wall time.
2. Choose each residual spread on validation instances at the shorter budget.
3. Solve every test instance with every arm at every budget; re-evaluate decoded results.
4. Compare learned with hand priors using the same mechanism, paired over instances.

Important criteria:
- Decision rule, fixed before any result: advance only if, on the five-constraint family at both
  budgets, the learned prior beats the hand prior with the same mechanism (seeded and residual)
  and beats plain, each with a paired 95 percent bootstrap interval below zero, and on plain
  Knapsack no learned-minus-hand interval lies above zero.
- Best-known profit is the best any arm found, not a proven optimum.
- This is a development study; confirmation seeds are reserved and unused.

Inputs/Outputs:
- Input: design settings and an output path on the command line.
- Output: a JSON file with training reports, every trial and the decision, plus saved scorers.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import sys
import time
from benchmarks.engines.packing_problems import PackingDecoder
from benchmarks.engines.packing_problems import PackingFamily
from benchmarks.engines.packing_problems import PackingInstance
from benchmarks.engines.packing_problems import PackingRun
from benchmarks.engines.priority_decoder_study import PairedStatistics
from benchmarks.engines.priority_scorer import PriorityScorer
from benchmarks.engines.priority_scorer import ReinforceTrainer
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARMS = ("plain", "hand_seed", "learned_seed", "hand_resid", "learned_resid")
SPREADS = (0.25, 0.5, 1.0, 2.0)
PAIRS = (
    ("learned_seed", "hand_seed"),
    ("learned_resid", "hand_resid"),
    ("learned_seed", "plain"),
    ("learned_resid", "plain"),
)
MINIMUM_SEARCH = 0.05


@dataclass(slots=True)
class StudyDesign:
    """Fixed numerical design of the study.

    Responsibilities:
    - Keep instance size, budgets and the disjoint seed ranges in one validated place.

    Attributes:
        n_items (int): Items per instance.
        budgets (tuple[float, ...]): Per-instance wall allowances in seconds.
        test_instances (int): Held-out instances per family.
        validation_instances (int): Instances used to choose spreads.
        train_seeds (tuple[int, int]): Half-open range of training instance seeds.
        validation_base (int): First validation seed.
        test_base (int): First test seed.
        steps (int): REINFORCE updates.
        batch (int): Instances per update.
        samples (int): Sampled orders per instance.
        hidden (int): Scorer hidden width.
        execution (str): Python-engine execution mode passed to the solver.
        algorithms (tuple[str, ...] | None): Portfolio override; None keeps the default portfolio.
    """

    n_items: int = 250
    budgets: tuple[float, ...] = (2.0, 10.0)
    test_instances: int = 12
    validation_instances: int = 6
    train_seeds: tuple[int, int] = (100_000, 101_000)
    validation_base: int = 6000
    test_base: int = 7000
    steps: int = 500
    batch: int = 8
    samples: int = 16
    hidden: int = 16
    execution: str = "process"
    algorithms: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Require pairwise disjoint training, validation and test seeds.

        Raises:
            ValueError: If any two seed ranges overlap.
        """
        spans = (
            range(*self.train_seeds),
            range(self.validation_base, self.validation_base + self.validation_instances),
            range(self.test_base, self.test_base + self.test_instances),
        )
        for index, first in enumerate(spans):
            for second in spans[index + 1 :]:
                if first.start < second.stop and second.start < first.stop:
                    raise ValueError("Training, validation and test seeds must be disjoint.")


class PriorityDecision:
    """Apply the decision rule fixed before the study.

    Responsibilities:
    - Compare learned and hand priors with paired statistics.
    - Decide whether the learned prior justifies a further step.
    """

    @staticmethod
    def verdict(
        tables: dict[str, dict[str, dict[str, dict[str, float]]]],
    ) -> dict[str, Any]:
        """Evaluate the rule on paired comparisons.

        Args:
            tables (dict[str, dict[str, dict[str, dict[str, float]]]]): Statistics keyed by
                family, then budget label, then `learned-vs-reference` pair name, each holding
                the paired `difference`, `ci_low` and `ci_high`.

        Returns:
            dict[str, Any]: Whether the five-constraint conditions hold, whether plain
            Knapsack shows a significant learned loss, and the final `advance` flag.
        """
        holds = all(
            statistic["ci_high"] < 0
            for budget in tables["mkp"].values()
            for statistic in budget.values()
        )
        loss = any(
            statistic["ci_low"] > 0
            for budget in tables["knapsack"].values()
            for pair, statistic in budget.items()
            if pair.endswith(("-vs-hand_seed", "-vs-hand_resid"))
        )
        return {"mkp_conditions_hold": holds, "knapsack_loss": loss, "advance": holds and not loss}


class PriorityStudy:
    """Train scorers, choose spreads, run all arms and summarize.

    Responsibilities:
    - Train one scorer per family and keep its training report.
    - Charge each arm's score computation to its allowance.
    - Re-evaluate every decoded result independently of the solver.

    Attributes:
        design (StudyDesign): Fixed design.
    """

    def __init__(self, design: StudyDesign) -> None:
        """Store the design.

        Args:
            design (StudyDesign): Study design.
        """
        self.design = design

    @staticmethod
    def search_allowance(budget: float, prep_seconds: float) -> float:
        """Return the search time left after charging score computation, with a floor.

        Args:
            budget (float): Per-instance wall allowance.
            prep_seconds (float): Measured time spent computing the arm's scores.

        Returns:
            float: Allowance passed to the solver, never below the minimum search slice.
        """
        return max(budget - prep_seconds, MINIMUM_SEARCH)

    def train(self, kind: str, seed: int) -> tuple[PriorityScorer, dict[str, Any]]:
        """Train a scorer for one family and report its learning curve.

        Args:
            kind (str): Instance family.
            seed (int): Training seed.

        Returns:
            tuple[PriorityScorer, dict[str, Any]]: Fitted scorer and its report.
        """
        design = self.design
        scorer = PriorityScorer.init(2 + (1 if kind == "knapsack" else 5), design.hidden, seed)
        trainer = ReinforceTrainer(scorer, design.samples, seed)

        def make(instance_seed: int) -> PackingInstance:
            """Generate one training instance of the family.

            Args:
                instance_seed (int): Training seed.

            Returns:
                PackingInstance: Generated instance.
            """
            return PackingFamily.make(kind, design.n_items, instance_seed)

        held = [self.instance(kind, design.validation_base + k) for k in range(8)]
        before = trainer.evaluate(held)
        started = time.perf_counter()
        history = trainer.train(make, range(*design.train_seeds), design.steps, design.batch)
        report = {
            "kind": kind,
            "seconds": time.perf_counter() - started,
            "validation_reward_before": before,
            "validation_reward_after": trainer.evaluate(held),
            "reward_first_10": float(np.mean(history[:10])),
            "reward_last_10": float(np.mean(history[-10:])),
            "steps": design.steps,
        }
        return scorer, report

    def instance(self, kind: str, seed: int) -> PackingInstance:
        """Generate an instance at the design size.

        Args:
            kind (str): Instance family.
            seed (int): Instance seed.

        Returns:
            PackingInstance: Generated instance.
        """
        return PackingFamily.make(kind, self.design.n_items, seed)

    def scores_for(
        self, arm: str, instance: PackingInstance, scorer: PriorityScorer
    ) -> tuple[np.ndarray | None, float]:
        """Compute an arm's item scores and the time they took.

        Args:
            arm (str): Study arm.
            instance (PackingInstance): Packing instance.
            scorer (PriorityScorer): Trained scorer of the family.

        Returns:
            tuple[np.ndarray | None, float]: Scores, or None for plain, and elapsed seconds.
        """
        started = time.perf_counter()
        if arm == "plain":
            return None, 0.0
        if arm.startswith("hand"):
            scores = PackingFamily.hand_scores(instance)
        else:
            scores = scorer.scores(PackingFamily.features(instance))
        return scores, time.perf_counter() - started

    def solve(
        self,
        instance: PackingInstance,
        arm: str,
        context: tuple[PriorityScorer, float, float],
        seed: int,
    ) -> dict[str, Any]:
        """Run one arm on one instance within one allowance.

        Args:
            instance (PackingInstance): Packing instance.
            arm (str): Study arm.
            context (tuple[PriorityScorer, float, float]): Trained scorer, residual spread and
                the per-instance allowance in seconds.
            seed (int): Solver seed.

        Returns:
            dict[str, Any]: Arm, budget, profit, verification and phase timings.
        """
        scorer, spread, budget = context
        scores, prep = self.scores_for(arm, instance, scorer)
        seeded = arm.endswith("_seed")
        decoder = PackingDecoder(instance, None if seeded else scores, spread)
        keys = [PackingFamily.order_keys(scores)] if seeded and scores is not None else None
        allowance = self.search_allowance(budget, prep)
        started = time.perf_counter()
        observed, verified = PackingRun.solve(
            instance,
            decoder,
            keys,
            (seed, allowance),
            (self.design.execution, self.design.algorithms),
        )
        return {
            "instance": instance.name,
            "family": instance.name.split("-")[0],
            "arm": arm,
            "budget": budget,
            "profit": observed,
            "verified": verified,
            "prep_seconds": prep,
            "allowance": allowance,
            "wall_seconds": time.perf_counter() - started + prep,
            "spread": spread,
        }

    @staticmethod
    def reference_profit(instance: PackingInstance) -> float:
        """Return the profit of the hand-prior greedy fill, used only to normalize validation.

        Args:
            instance (PackingInstance): Packing instance.

        Returns:
            float: Hand-prior greedy profit.
        """
        order = np.argsort(-PackingFamily.hand_scores(instance), kind="stable")
        return PackingFamily.fill(order, instance)[1]

    def choose_spread(self, kind: str, scorer: PriorityScorer, arm: str) -> float:
        """Pick the residual spread with the best mean profit on validation instances.

        Args:
            kind (str): Instance family.
            scorer (PriorityScorer): Trained scorer.
            arm (str): `hand_resid` or `learned_resid`.

        Returns:
            float: Chosen spread.
        """
        budget = min(self.design.budgets)
        means = {}
        for spread in SPREADS:
            profits = []
            for k in range(self.design.validation_instances):
                seed = self.design.validation_base + k
                row = self.solve(self.instance(kind, seed), arm, (scorer, spread, budget), seed)
                profits.append(row["profit"] / self.reference_profit(self.instance(kind, seed)))
            means[spread] = float(np.mean(profits))
        return max(means, key=lambda value: means[value])

    def run(self, seed: int) -> dict[str, Any]:
        """Train, validate and evaluate both families.

        Args:
            seed (int): Training seed.

        Returns:
            dict[str, Any]: Training reports, spreads, raw trials and the verdict.
        """
        rows: list[dict[str, Any]] = []
        reports, spreads, scorers = {}, {}, {}
        for kind in PackingFamily.KINDS:
            scorers[kind], reports[kind] = self.train(kind, seed)
            spreads[kind] = {
                arm: self.choose_spread(kind, scorers[kind], arm)
                for arm in ("hand_resid", "learned_resid")
            }
            for k in range(self.design.test_instances):
                rows.extend(
                    self.evaluate(kind, scorers[kind], spreads[kind], self.design.test_base + k)
                )
        rows = self.normalize(rows)
        return {
            "training": reports,
            "spreads": spreads,
            "scorers": {
                k: {n: v.tolist() for n, v in s.weights.items()} for k, s in scorers.items()
            },
            "trials": rows,
            "summary": self.summarize(rows),
        }

    def evaluate(
        self, kind: str, scorer: PriorityScorer, spreads: dict[str, float], seed: int
    ) -> list[dict[str, Any]]:
        """Run every arm at every budget on one test instance.

        Args:
            kind (str): Instance family.
            scorer (PriorityScorer): Trained scorer.
            spreads (dict[str, float]): Chosen residual spread per residual arm.
            seed (int): Test instance and solver seed.

        Returns:
            list[dict[str, Any]]: One record per arm and budget.
        """
        instance = self.instance(kind, seed)
        rows = []
        for budget in self.design.budgets:
            for arm in ARMS:
                spread = spreads.get(arm, 1.0)
                rows.append(self.solve(instance, arm, (scorer, spread, budget), seed))
                print(f"trial {instance.name} {budget:g}s {arm}", file=sys.stderr, flush=True)
        return rows

    @staticmethod
    def normalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add the percentage gap to the best profit any arm found on the instance.

        Args:
            rows (list[dict[str, Any]]): Trial records.

        Returns:
            list[dict[str, Any]]: Records with a `gap` field.
        """
        best: dict[str, float] = {}
        for row in rows:
            best[row["instance"]] = max(best.get(row["instance"], 0.0), row["profit"])
        for row in rows:
            row["gap"] = 100.0 * (best[row["instance"]] - row["profit"]) / best[row["instance"]]
        return rows

    @staticmethod
    def gaps(rows: list[dict[str, Any]], kind: str, budget: float) -> dict[str, list[float]]:
        """Collect the gaps of every arm on one family and budget in instance order.

        Args:
            rows (list[dict[str, Any]]): Normalized trial records.
            kind (str): Instance family.
            budget (float): Per-instance allowance in seconds.

        Returns:
            dict[str, list[float]]: Gap values per arm.
        """
        group = [r for r in rows if (r["family"], r["budget"]) == (kind, budget)]
        return {arm: [r["gap"] for r in group if r["arm"] == arm] for arm in ARMS}

    @staticmethod
    def compare(
        rows: list[dict[str, Any]], kind: str, budget: float
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        """Compute mean gaps and the paired comparisons for one family and budget.

        Args:
            rows (list[dict[str, Any]]): Normalized trial records.
            kind (str): Instance family.
            budget (float): Per-instance allowance in seconds.

        Returns:
            tuple[dict[str, float], dict[str, dict[str, float]]]: Mean gap per arm and the
            paired statistics of every compared pair.
        """
        gaps = PriorityStudy.gaps(rows, kind, budget)
        paired = {
            f"{a}-vs-{b}": PairedStatistics.compare(gaps[a], gaps[b], 10000, 0) for a, b in PAIRS
        }
        return {arm: float(np.mean(values)) for arm, values in gaps.items()}, paired

    def summarize(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Compare learned with hand priors and apply the decision rule.

        Args:
            rows (list[dict[str, Any]]): Normalized trial records.

        Returns:
            dict[str, Any]: Mean gaps, paired comparisons and the verdict.
        """
        tables: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        means: dict[str, dict[str, float]] = {}
        for kind in PackingFamily.KINDS:
            tables[kind] = {}
            for budget in self.design.budgets:
                mean, paired = self.compare(rows, kind, budget)
                means[f"{kind}@{budget:g}s"] = mean
                tables[kind][f"{budget:g}s"] = paired
        return {"mean_gap": means, "paired": tables, "verdict": PriorityDecision.verdict(tables)}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the study and write its JSON report and scorers.

    Args:
        argv (Sequence[str] | None): Arguments, or None for `sys.argv`.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-items", type=int, default=250)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--test-instances", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/learned-priority-study.json"))
    options = parser.parse_args(argv)
    design = StudyDesign(
        n_items=options.n_items, steps=options.steps, test_instances=options.test_instances
    )
    report = {"design": asdict(design)} | PriorityStudy(design).run(options.seed)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
