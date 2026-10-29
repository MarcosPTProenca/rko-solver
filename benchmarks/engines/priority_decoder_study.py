"""Compare analytic priority decoders with the plain sort decoder at equal wall time.

Objective:
- Decide, before any network is trained, whether an analytic item prior inside the decoder
  improves the default RKO portfolio on held-out Knapsack instances.

Technical approach:
- Generate Pisinger-style instances (uncorrelated, weakly and strongly correlated) from seeds that
  are disjoint between validation and test, and solve each exactly with the NumPy DP oracle.
- Choose the single hyperparameter of each informed decoder on the validation seeds, then freeze it.
- Run the default portfolio with identical seeds and wall budgets for every decoder, decode the
  returned keys independently and report the optimality gap.
- Summarize paired differences against the sort decoder with a bootstrap interval over instances
  and count wins, losses and ties.

Flow (pseudocode):
1. Build validation and test instance sets and their exact optima.
2. For each informed decoder pick the validation-best hyperparameter at the shortest budget.
3. Run every decoder on every test instance at every budget and record the gap.
4. Report paired statistics and the kill criterion: worse than sort at every budget.

Important criteria:
- Budgets, seeds, portfolio and cost function are identical across decoders.
- Infeasible threshold-decoder solutions count as zero profit and are flagged.
- Losses and failures are reported, not filtered.

Inputs/Outputs:
- Input: sizes, budgets, seed counts and an output path on the command line.
- Output: a JSON file with every trial and the summary, and a console table.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import sys
import time
from benchmarks.problems.knapsack_utils import exact_profit
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from examples.priority_decoders import ItemScores
from examples.priority_decoders import PriorityDecoder
from examples.priority_decoders import QuantileDecoder
from examples.priority_decoders import SortDecoder
from examples.run_kp_cpp_instances import KpInstance
from examples.run_kp_cpp_instances import decode_knapsack
from examples.run_kp_cpp_instances import evaluate_solution
from examples.run_kp_cpp_instances import make_cost
from pathlib import Path
from rko_solver import RKO
from typing import Any

FAMILIES = ("uncorrelated", "weakly_correlated", "strongly_correlated")
SPREADS = (0.25, 0.5, 1.0, 2.0)
TEMPERATURES = (0.05, 0.1, 0.3, 1.0)
VALIDATION_SEED_BASE = 1000
TEST_SEED_BASE = 2000
RANGE_LIMIT = 100


class InstanceFactory:
    """Generate Pisinger-style Knapsack instances from seeds.

    Responsibilities:
    - Produce reproducible instances whose family, size and seed identify them.
    """

    @staticmethod
    def make(family: str, n_items: int, seed: int) -> KpInstance:
        """Create one instance with half of the total weight as capacity.

        Args:
            family (str): One of `FAMILIES`.
            n_items (int): Number of items.
            seed (int): Instance seed.

        Returns:
            KpInstance: Integer-weight instance.

        Raises:
            ValueError: If the family is unknown.
        """
        if family not in FAMILIES:
            raise ValueError(f"'family' must be one of {FAMILIES}.")
        rng = np.random.default_rng([seed, n_items, FAMILIES.index(family)])
        weights = rng.integers(1, RANGE_LIMIT + 1, n_items)
        noise = rng.integers(1, RANGE_LIMIT + 1, n_items)
        window = RANGE_LIMIT // 10
        benefits = {
            "uncorrelated": noise,
            "weakly_correlated": np.maximum(1, weights + noise % (2 * window + 1) - window),
            "strongly_correlated": weights + window,
        }[family]
        return KpInstance(
            n_items=n_items,
            capacity=max(int(weights.max()), int(weights.sum()) // 2),
            benefits=benefits.tolist(),
            weights=weights.tolist(),
            name=f"{family}-{n_items}-{seed}",
        )


@dataclass(slots=True)
class Candidate:
    """Name a decoder together with its frozen hyperparameter.

    Attributes:
        name (str): Decoder label.
        parameter (float | None): Spread or temperature, None for parameter-free decoders.
    """

    name: str
    parameter: float | None = None

    def decoder(self, instance: KpInstance) -> Callable[..., list[int]]:
        """Build the decoder for an instance.

        Args:
            instance (KpInstance): Knapsack data.

        Returns:
            Callable[..., list[int]]: Pickleable decoder.

        Raises:
            ValueError: If the candidate name is unknown.
        """
        if self.name == "threshold":
            return decode_knapsack
        if self.name in ("sort", "sort_seeded"):
            return SortDecoder(instance)
        if self.name == "priority" and self.parameter is not None:
            return PriorityDecoder.build(instance, self.parameter)
        if self.name == "quantile" and self.parameter is not None:
            return QuantileDecoder.build(instance, self.parameter)
        raise ValueError(f"Unknown candidate {self.name!r}.")


class GreedySeed:
    """Encode the value/weight greedy order as one random-key vector.

    Responsibilities:
    - Give the plain sort decoder the same prior the priority decoder has, through `initial_keys`.
    """

    @staticmethod
    def keys(instance: KpInstance) -> list[float]:
        """Return keys whose ascending order is the best-ratio-first order.

        Args:
            instance (KpInstance): Knapsack data.

        Returns:
            list[float]: One key in (0, 1) per item.
        """
        position = np.empty(instance.n_items)
        position[np.argsort(-ItemScores.ratio(instance), kind="stable")] = np.arange(
            instance.n_items
        )
        return ((position + 0.5) / instance.n_items).tolist()


class TrialRunner:
    """Run the default portfolio for one decoder and score the result independently.

    Responsibilities:
    - Solve with a wall budget and a seed, then decode the returned keys outside the solver.
    """

    @staticmethod
    def run(
        instance: KpInstance, candidate: Candidate, budget: float, seed: int, optimum: int
    ) -> dict[str, Any]:
        """Solve one instance and return the trial record.

        Args:
            instance (KpInstance): Knapsack data.
            candidate (Candidate): Decoder under test.
            budget (float): Search seconds.
            seed (int): Solver seed.
            optimum (int): Exact optimal profit.

        Returns:
            dict[str, Any]: Candidate, budget, profit, gap in percent, feasibility and seconds.
        """
        decoder = candidate.decoder(instance)
        started = time.perf_counter()
        seeds = [GreedySeed.keys(instance)] if candidate.name == "sort_seeded" else None
        result = RKO(
            decoder, make_cost(instance), instance.n_items, random_state=seed, initial_keys=seeds
        ).solve(budget)
        elapsed = time.perf_counter() - started
        profit, _, feasible = evaluate_solution(decoder(np.asarray(result.keys)), instance)
        counted = profit if feasible else 0
        return {
            "decoder": candidate.name,
            "parameter": candidate.parameter,
            "budget": budget,
            "instance": instance.name,
            "profit": profit,
            "feasible": feasible,
            "optimum": optimum,
            "gap": 100.0 * (optimum - counted) / optimum,
            "seconds": elapsed,
        }


class PairedStatistics:
    """Summarize paired gap differences against the sort decoder.

    Responsibilities:
    - Report the mean difference, a bootstrap interval and win/loss/tie counts.
    """

    @staticmethod
    def compare(
        candidate: Sequence[float], reference: Sequence[float], resamples: int, seed: int
    ) -> dict[str, float]:
        """Compare two aligned gap vectors; negative differences favor the candidate.

        Args:
            candidate (Sequence[float]): Candidate gaps per instance.
            reference (Sequence[float]): Reference gaps in the same instance order.
            resamples (int): Bootstrap resamples over instances.
            seed (int): Bootstrap seed.

        Returns:
            dict[str, float]: Mean candidate and reference gap, mean difference, interval,
            wins, losses and ties.
        """
        difference = np.asarray(candidate) - np.asarray(reference)
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, len(difference), (resamples, len(difference)))
        means = difference[picks].mean(axis=1)
        return {
            "candidate_gap": float(np.mean(candidate)),
            "reference_gap": float(np.mean(reference)),
            "difference": float(difference.mean()),
            "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)),
            "wins": int((difference < -1e-9).sum()),
            "losses": int((difference > 1e-9).sum()),
            "ties": int((np.abs(difference) <= 1e-9).sum()),
        }

    @classmethod
    def against_sort(
        cls, rows: Sequence[dict[str, Any]], budget: float, resamples: int
    ) -> dict[str, dict[str, float]]:
        """Compare every decoder with `sort` at one budget.

        Args:
            rows (Sequence[dict[str, Any]]): Trial records with `decoder`, `budget` and `gap`.
            budget (float): Budget in seconds to select.
            resamples (int): Bootstrap resamples.

        Returns:
            dict[str, dict[str, float]]: Statistics keyed `decoder@budget`, including `sort`.
        """
        gaps: dict[str, list[float]] = {}
        for row in rows:
            if row["budget"] == budget:
                gaps.setdefault(row["decoder"], []).append(row["gap"])
        return {
            f"{name}@{budget:g}s": cls.compare(values, gaps["sort"], resamples, 0)
            for name, values in gaps.items()
        }


class PriorityDecoderStudy:
    """Run validation, test and summary.

    Responsibilities:
    - Select hyperparameters on validation instances only.
    - Evaluate every decoder on disjoint test instances at every budget.
    - Apply the kill criterion.

    Attributes:
        sizes (tuple[int, ...]): Item counts.
        budgets (tuple[float, ...]): Wall budgets in seconds.
        validation_seeds (int): Validation instances per family and size.
        test_seeds (int): Test instances per family and size.
        resamples (int): Bootstrap resamples.
    """

    def __init__(
        self,
        sizes: Sequence[int],
        budgets: Sequence[float],
        validation_seeds: int,
        test_seeds: int,
        resamples: int,
    ) -> None:
        """Store the study design.

        Args:
            sizes (Sequence[int]): Item counts.
            budgets (Sequence[float]): Wall budgets in seconds.
            validation_seeds (int): Validation instances per family and size.
            test_seeds (int): Test instances per family and size.
            resamples (int): Bootstrap resamples.
        """
        self.sizes = tuple(sizes)
        self.budgets = tuple(budgets)
        self.validation_seeds = validation_seeds
        self.test_seeds = test_seeds
        self.resamples = resamples

    def instances(self, base: int, count: int) -> list[tuple[KpInstance, int, int]]:
        """Build instances with their exact optimum and solver seed.

        Args:
            base (int): First instance seed of the split.
            count (int): Instances per family and size.

        Returns:
            list[tuple[KpInstance, int, int]]: Instance, optimum and seed.
        """
        built = []
        for family in FAMILIES:
            for size in self.sizes:
                for offset in range(count):
                    instance = InstanceFactory.make(family, size, base + offset)
                    built.append((instance, exact_profit(instance), base + offset))
        return built

    def validate(self, name: str, grid: Sequence[float]) -> tuple[float, dict[float, float]]:
        """Pick the grid value with the lowest mean gap at the shortest budget.

        Args:
            name (str): `priority` or `quantile`.
            grid (Sequence[float]): Hyperparameter values to try.

        Returns:
            tuple[float, dict[float, float]]: Best value and the mean gap of every value.
        """
        split = self.instances(VALIDATION_SEED_BASE, self.validation_seeds)
        budget = min(self.budgets)
        means = {}
        for value in grid:
            candidate = Candidate(name, value)
            gaps = [
                TrialRunner.run(instance, candidate, budget, seed, optimum)["gap"]
                for instance, optimum, seed in split
            ]
            means[value] = float(np.mean(gaps))
        return min(means, key=lambda value: means[value]), means

    def evaluate(self, candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
        """Run every candidate on every test instance and budget.

        Args:
            candidates (Sequence[Candidate]): Decoders with frozen hyperparameters.

        Returns:
            list[dict[str, Any]]: One record per trial.
        """
        rows = []
        for instance, optimum, seed in self.instances(TEST_SEED_BASE, self.test_seeds):
            for budget in self.budgets:
                for candidate in candidates:
                    rows.append(TrialRunner.run(instance, candidate, budget, seed, optimum))
                    print(
                        f"trial {len(rows)} {instance.name} {budget:g}s {candidate.name}",
                        file=sys.stderr,
                        flush=True,
                    )
        return rows

    def summarize(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Compare each decoder with `sort` at each budget and apply the kill criterion.

        Args:
            rows (Sequence[dict[str, Any]]): Test trial records.

        Returns:
            dict[str, Any]: Paired statistics per decoder and budget, plus the verdicts.
        """
        table: dict[str, Any] = {}
        for budget in self.budgets:
            table.update(PairedStatistics.against_sort(rows, budget, self.resamples))
        informed = {row["decoder"] for row in rows} & {"priority", "quantile"}
        killed = {
            name: all(table[f"{name}@{b:g}s"]["difference"] > 0 for b in self.budgets)
            for name in sorted(informed)
        }
        return {"paired_vs_sort": table, "killed": killed}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv (Sequence[str] | None): Arguments, or None for `sys.argv`.

    Returns:
        argparse.Namespace: Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 300])
    parser.add_argument("--budgets", type=float, nargs="+", default=[2.0, 10.0])
    parser.add_argument("--validation-seeds", type=int, default=1)
    parser.add_argument("--test-seeds", type=int, default=4)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=("threshold", "sort", "sort_seeded", "priority", "quantile"),
        default=["threshold", "sort", "sort_seeded", "priority", "quantile"],
        help="Decoders to run; sort is always included as the paired reference.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/priority-decoder-e1.json"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the study and write its JSON report.

    Args:
        argv (Sequence[str] | None): Arguments, or None for `sys.argv`.

    Returns:
        int: Process exit status.
    """
    options = parse_args(argv)
    study = PriorityDecoderStudy(
        options.sizes,
        options.budgets,
        options.validation_seeds,
        options.test_seeds,
        options.resamples,
    )
    names = list(dict.fromkeys(["sort", *options.candidates]))
    chosen: dict[str, float] = {}
    validation: dict[str, dict[float, float]] = {}
    for name, grid in (("priority", SPREADS), ("quantile", TEMPERATURES)):
        if name in names:
            chosen[name], validation[name] = study.validate(name, grid)
    candidates = [Candidate(name, chosen.get(name)) for name in names]
    rows = study.evaluate(candidates)
    report = {
        "design": vars(options) | {"output": str(options.output)},
        "validation": validation,
        "chosen": chosen,
        "summary": study.summarize(rows),
        "trials": rows,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({key: report[key] for key in ("chosen", "summary")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
