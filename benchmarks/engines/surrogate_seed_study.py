"""Test whether surrogate-built `initial_keys` help RKO across unrelated problem families.

Objective:
- Measure whether a model fitted only to (keys, cost) pairs, without domain features, can produce
  starting keys that improve the default portfolio after charging probing to the search allowance.

Technical approach:
- Five benchmark groups share one interface (a pickleable decoder and cost over random keys):
  Knapsack with two decoder variants, the traveling salesman problem with a sort decoder, a
  shifted sphere and a shifted Rastrigin function.
- The surrogate samples `4 * n_variables` uniform keys, rank-transforms the costs, fits a ridge
  regression of cost rank on key with the penalty chosen on a held-out fifth, and proposes keys
  that move against the fitted gradient at three amplitudes. A held-out correlation below
  `MIN_CORRELATION` means the model found no signal and no seed is offered.
- `--model mlp` swaps the ridge model for the NumPy network in `mlp_surrogate`; arms, rule and
  threshold are unchanged.
- Four arms use the same instances, solver seeds and nominal allowance: plain, plain repeated (the
  noise floor), best-random (the three best sampled keys, to isolate the surrogate from sampling)
  and surrogate. Probe time is subtracted from the solve allowance of the seeded arms; process
  startup and seed evaluation remain outside it, and a 0.05-second minimum can exceed the budget.
- Quality is the normalized gap `(cost - best_known) / (median_random - best_known)`, where
  `best_known` is the lowest cost any arm found on that instance.

Flow (pseudocode):
1. For each family and seed build the instance and the three seeded/unseeded arms.
2. Sample, fit and propose seeds; solve with `initial_keys` for the remaining budget.
3. Normalize costs per instance, pair each arm with plain and bootstrap the mean difference.
4. Apply the decision rule fixed before the run.

Important criteria:
- Decision rule, fixed before any result existed: the surrogate is promising only if it beats plain
  on at least three of five families with a 95 percent bootstrap interval entirely below zero, it
  does not lose significantly on any family, and it also beats best-random on the families it wins.
  Otherwise this candidate fails the acceptance rule; other learned models remain untested.
- Sampling and fitting reduce the seeded arms' search allowance. This is not a hard end-to-end
  deadline: record probe time, remaining allowance and overruns rather than claiming exact parity.
- The network run is fixed before any result: `--model mlp --size 300 --instances 12 --budget 2`,
  the same rule and the same 0.2 held-out threshold as the linear run.
- No domain knowledge reaches the surrogate; only keys and costs do.
- Losses, fallbacks and failures are reported.

Inputs/Outputs:
- Input: instance count, size, budget and output path on the command line.
- Output: a JSON file with every trial and the verdict, and a console summary.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import sys
import time
from benchmarks.engines.mlp_surrogate import MlpSurrogate
from benchmarks.engines.priority_decoder_study import InstanceFactory
from benchmarks.engines.priority_decoder_study import PairedStatistics
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from examples.priority_decoders import SortDecoder
from examples.run_kp_cpp_instances import decode_knapsack
from examples.run_kp_cpp_instances import make_cost
from pathlib import Path
from rko_solver import RKO
from typing import Any

FAMILIES = ("knapsack_sort", "knapsack_threshold", "tsp_sort", "sphere", "rastrigin")
AMPLITUDES = (0.1, 0.25, 0.5)
PENALTIES = (0.01, 0.1, 1.0, 10.0)
MIN_CORRELATION = 0.2
SAMPLES_PER_KEY = 4
BEST_RANDOM = 3
ARMS = ("plain", "plain_repeat", "best_random", "surrogate")
REQUIRED_WINS = 3


@dataclass(slots=True)
class TspDecoder:
    """Order cities by ascending key.

    Responsibilities:
    - Turn a key vector into a tour permutation.
    """

    def __call__(self, keys: np.ndarray) -> np.ndarray:
        """Decode a key vector.

        Args:
            keys (np.ndarray): One key per city.

        Returns:
            np.ndarray: City indices in visiting order.
        """
        return np.argsort(np.asarray(keys), kind="stable")


@dataclass(slots=True)
class TspCost:
    """Length of the closed tour through the cities.

    Attributes:
        coordinates (np.ndarray): City coordinates, one row per city.
    """

    coordinates: np.ndarray

    def __call__(self, order: np.ndarray) -> float:
        """Compute the tour length.

        Args:
            order (np.ndarray): Visiting order.

        Returns:
            float: Closed tour length.
        """
        path = self.coordinates[order]
        return float(np.linalg.norm(path - np.roll(path, -1, axis=0), axis=1).sum())


@dataclass(slots=True)
class SphereCost:
    """Squared distance of the keys to a hidden center.

    Attributes:
        center (np.ndarray): Minimizer inside the unit cube.
    """

    center: np.ndarray

    def __call__(self, keys: np.ndarray) -> float:
        """Compute the squared distance.

        Args:
            keys (np.ndarray): Decoded vector.

        Returns:
            float: Objective value.
        """
        return float(np.sum((np.asarray(keys) - self.center) ** 2))


@dataclass(slots=True)
class RastriginCost:
    """Rastrigin function on [-5, 5] shifted by a hidden offset.

    Attributes:
        shift (np.ndarray): Offset of the minimizer.
    """

    shift: np.ndarray

    def __call__(self, keys: np.ndarray) -> float:
        """Compute the shifted Rastrigin value.

        Args:
            keys (np.ndarray): Decoded vector in [0, 1).

        Returns:
            float: Objective value.
        """
        x = 10.0 * np.asarray(keys) - 5.0 - self.shift
        return float(10.0 * len(x) + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


@dataclass(slots=True)
class IdentityDecoder:
    """Pass the keys through unchanged."""

    def __call__(self, keys: np.ndarray) -> np.ndarray:
        """Return a copy of the keys.

        Args:
            keys (np.ndarray): Key vector.

        Returns:
            np.ndarray: Copy of the keys.
        """
        return np.array(keys, dtype=np.float64)


@dataclass(slots=True)
class Problem:
    """One instance in the shared decoder/cost form.

    Attributes:
        label (str): Instance identifier.
        decoder (Callable[..., Any]): Pickleable decoder over keys.
        cost (Callable[..., float]): Pickleable objective over decoded values.
        n_variables (int): Number of random keys.
    """

    label: str
    decoder: Callable[..., Any]
    cost: Callable[..., float]
    n_variables: int


class ProblemFamily:
    """Build instances of the five families.

    Responsibilities:
    - Produce a reproducible `Problem` from a family name, size and seed.
    """

    @staticmethod
    def make(family: str, size: int, seed: int) -> Problem:
        """Create one instance.

        Args:
            family (str): One of `FAMILIES`.
            size (int): Number of keys.
            seed (int): Instance seed.

        Returns:
            Problem: Decoder, objective and dimension.

        Raises:
            ValueError: If the family is unknown.
        """
        if family not in FAMILIES:
            raise ValueError(f"'family' must be one of {FAMILIES}.")
        label = f"{family}-{size}-{seed}"
        rng = np.random.default_rng([seed, size, FAMILIES.index(family)])
        if family in ("knapsack_sort", "knapsack_threshold"):
            instance = InstanceFactory.make("uncorrelated", size, seed)
            decoder = SortDecoder(instance) if family == "knapsack_sort" else decode_knapsack
            return Problem(label, decoder, make_cost(instance), size)
        if family == "tsp_sort":
            return Problem(label, TspDecoder(), TspCost(rng.random((size, 2))), size)
        if family == "sphere":
            return Problem(label, IdentityDecoder(), SphereCost(0.1 + 0.8 * rng.random(size)), size)
        return Problem(label, IdentityDecoder(), RastriginCost(rng.uniform(-1.0, 1.0, size)), size)


class SurrogateSeeds:
    """Fit a ridge surrogate to sampled (keys, cost) pairs and propose starting keys.

    Responsibilities:
    - Sample uniform keys and evaluate them through the decoder and objective.
    - Fit the surrogate on cost ranks, choosing the penalty on held-out keys.
    - Propose keys that move against the fitted gradient, or none when there is no signal.

    Attributes:
        problem (Problem): Problem being sampled.
        rng (np.random.Generator): Sampling generator.
        keys (np.ndarray): Sampled key matrix.
        costs (np.ndarray): Costs of the sampled keys.
        correlation (float): Held-out correlation of the selected model.
    """

    def __init__(self, problem: Problem, seed: int, model: str = "linear") -> None:
        """Sample and evaluate the probe keys.

        Args:
            problem (Problem): Problem to sample.
            seed (int): Sampling seed.
            model (str): `linear` for the ridge surrogate or `mlp` for the NumPy network.
        """
        self.problem = problem
        self.model = model
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        count = SAMPLES_PER_KEY * problem.n_variables
        self.keys = self.rng.random((count, problem.n_variables))
        self.costs = np.array([problem.cost(problem.decoder(row)) for row in self.keys])
        self.correlation = 0.0

    def fit(self) -> np.ndarray | None:
        """Fit the ridge surrogate on cost ranks.

        Returns:
            np.ndarray | None: Coefficients, or None when the held-out correlation is below
            `MIN_CORRELATION`.
        """
        count = len(self.costs)
        target = np.argsort(np.argsort(self.costs)) / (count - 1) - 0.5
        centered = self.keys - 0.5
        split = int(0.8 * count)
        best_penalty, best_correlation = None, -1.0
        for penalty in PENALTIES:
            weights = self.solve(centered[:split], target[:split], penalty * split / 12.0)
            predicted = centered[split:] @ weights
            if np.std(predicted) == 0.0:
                continue
            correlation = float(np.corrcoef(predicted, target[split:])[0, 1])
            if correlation > best_correlation:
                best_penalty, best_correlation = penalty, correlation
        self.correlation = best_correlation
        if best_penalty is None or best_correlation < MIN_CORRELATION:
            return None
        return self.solve(centered, target, best_penalty * count / 12.0)

    @staticmethod
    def solve(centered: np.ndarray, target: np.ndarray, penalty: float) -> np.ndarray:
        """Solve the ridge normal equations.

        Args:
            centered (np.ndarray): Keys minus one half.
            target (np.ndarray): Centered cost ranks.
            penalty (float): Ridge penalty.

        Returns:
            np.ndarray: Coefficients.
        """
        gram = centered.T @ centered + penalty * np.eye(centered.shape[1])
        return np.linalg.solve(gram, centered.T @ target)

    def surrogate(self) -> list[list[float]]:
        """Propose keys against the fitted gradient at every amplitude.

        Returns:
            list[list[float]]: Starting keys, empty when the surrogate found no signal.
        """
        if self.model == "mlp":
            network = MlpSurrogate(self.keys, self.costs, self.seed)
            seeds = network.propose(MIN_CORRELATION)
            self.correlation = network.correlation
            return seeds
        weights = self.fit()
        if weights is None:
            return []
        direction = weights / np.max(np.abs(weights))
        return [np.clip(0.5 - a * direction, 0.0, 0.999999).tolist() for a in AMPLITUDES]

    def best_random(self) -> list[list[float]]:
        """Return the lowest-cost sampled keys.

        Returns:
            list[list[float]]: The `BEST_RANDOM` best sampled key vectors.
        """
        return self.keys[np.argsort(self.costs)[:BEST_RANDOM]].tolist()


class SeedingStudy:
    """Run the four arms on every instance and apply the decision rule.

    Responsibilities:
    - Solve each instance under equal total wall time for every arm.
    - Normalize costs per instance and compare each arm with plain.

    Attributes:
        size (int): Number of keys per instance.
        budget (float): Total wall seconds per arm.
        instances (int): Instances per family.
        resamples (int): Bootstrap resamples.
        model (str): Surrogate used by the `surrogate` arm, `linear` or `mlp`.
    """

    def __init__(
        self, size: int, budget: float, instances: int, resamples: int, model: str = "linear"
    ) -> None:
        """Store the study design.

        Args:
            size (int): Number of keys per instance.
            budget (float): Total wall seconds per arm.
            instances (int): Instances per family.
            resamples (int): Bootstrap resamples.
            model (str): Surrogate of the `surrogate` arm, `linear` or `mlp`.
        """
        self.model = model
        self.size = size
        self.budget = budget
        self.instances = instances
        self.resamples = resamples

    def solve(self, problem: Problem, seed: int, arm: str) -> dict[str, Any]:
        """Run one arm after deducting probe work from its nominal search allowance.

        Args:
            problem (Problem): Instance to solve.
            seed (int): Solver and sampling seed.
            arm (str): One of `ARMS`.

        Returns:
            dict[str, Any]: Arm, instance, cost, seeds, correlation, end-to-end seconds, probe
            seconds, remaining search allowance and whether probing exhausted the allowance.
        """
        started = time.perf_counter()
        seeds: list[list[float]] = []
        correlation = None
        if arm in ("best_random", "surrogate"):
            sampler = SurrogateSeeds(problem, seed, self.model)
            seeds = sampler.best_random() if arm == "best_random" else sampler.surrogate()
            correlation = sampler.correlation if arm == "surrogate" else None
        probe_seconds = time.perf_counter() - started
        remaining = max(self.budget - probe_seconds, 0.05)
        result = RKO(
            problem.decoder,
            problem.cost,
            problem.n_variables,
            random_state=seed,
            initial_keys=seeds or None,
        ).solve(remaining)
        return {
            "arm": arm,
            "instance": problem.label,
            "family": problem.label.split("-")[0],
            "cost": float(result.cost),
            "seeds": len(seeds),
            "correlation": correlation,
            "probe_seconds": probe_seconds,
            "search_budget_seconds": remaining,
            "probe_over_budget": probe_seconds >= self.budget,
            "seconds": time.perf_counter() - started,
        }

    def run(self) -> list[dict[str, Any]]:
        """Run every arm on every instance.

        Returns:
            list[dict[str, Any]]: One record per trial, with the normalized gap added.
        """
        rows: list[dict[str, Any]] = []
        for family in FAMILIES:
            for seed in range(3000, 3000 + self.instances):
                problem = ProblemFamily.make(family, self.size, seed)
                for arm in ARMS:
                    rows.append(self.solve(problem, seed, arm))
                    print(f"trial {len(rows)} {problem.label} {arm}", file=sys.stderr, flush=True)
        return self.normalize(rows)

    @staticmethod
    def normalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add the normalized gap to every record.

        Args:
            rows (list[dict[str, Any]]): Trial records.

        Returns:
            list[dict[str, Any]]: The same records with a `gap` field in units of the distance
            between the median random cost and the best cost known for the instance.
        """
        for label in {row["instance"] for row in rows}:
            group = [row for row in rows if row["instance"] == label]
            family, size, seed = label.rsplit("-", 2)
            problem = ProblemFamily.make(family, int(size), int(seed))
            sample = SurrogateSeeds(problem, 0).costs
            best = min(min(row["cost"] for row in group), float(sample.min()))
            scale = max(float(np.median(sample)) - best, 1e-12)
            for row in group:
                row["gap"] = (row["cost"] - best) / scale
        return rows

    @staticmethod
    def save(report: dict[str, Any], output: Path) -> None:
        """Write the JSON report and print the verdict.

        Args:
            report (dict[str, Any]): Design, verdict and trial records.
            output (Path): Destination file; parent directories are created.
        """
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        sys.stdout.write(json.dumps(report["verdict"], indent=2) + "\n")

    @staticmethod
    def rule(table: dict[str, Any]) -> dict[str, bool]:
        """Classify one family under the pre-registered decision rule.

        Args:
            table (dict[str, Any]): Paired statistics of the family.

        Returns:
            dict[str, bool]: `beats_plain` when the surrogate interval lies below zero and it also
            beats best-random on average; `loses_to_plain` when the interval lies above zero.
        """
        surrogate = table["surrogate"]
        return {
            "beats_plain": surrogate["ci_high"] < 0
            and table["surrogate_vs_best_random"]["difference"] < 0,
            "loses_to_plain": surrogate["ci_low"] > 0,
        }

    def family_table(self, rows: Sequence[dict[str, Any]], family: str) -> dict[str, Any]:
        """Compare every arm of one family with plain and the surrogate with best-random.

        Args:
            rows (Sequence[dict[str, Any]]): Normalized trial records.
            family (str): Family to select.

        Returns:
            dict[str, Any]: Paired statistics per arm against plain, `surrogate_vs_best_random`
            and the two booleans `beats_plain` and `loses_to_plain` that feed the decision rule.
        """
        chosen = [r for r in rows if r["family"] == family]
        gaps = {arm: [r["gap"] for r in chosen if r["arm"] == arm] for arm in ARMS}
        table: dict[str, Any] = {
            arm: PairedStatistics.compare(gaps[arm], gaps["plain"], self.resamples, 0)
            for arm in ARMS[1:]
        }
        table["surrogate_vs_best_random"] = PairedStatistics.compare(
            gaps["surrogate"], gaps["best_random"], self.resamples, 0
        )
        table.update(self.rule(table))
        return table

    def verdict(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Apply the decision rule fixed before the run.

        Args:
            rows (Sequence[dict[str, Any]]): Normalized trial records.

        Returns:
            dict[str, Any]: Paired statistics per family, the families the surrogate wins, the
            families where it loses significantly, and whether it is promising.
        """
        table = {family: self.family_table(rows, family) for family in FAMILIES}
        wins = [family for family, entry in table.items() if entry["beats_plain"]]
        losses = [family for family, entry in table.items() if entry["loses_to_plain"]]
        return {
            "paired_vs_plain": table,
            "wins": wins,
            "significant_losses": losses,
            "promising": len(wins) >= REQUIRED_WINS and not losses,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv (Sequence[str] | None): Arguments, or None for `sys.argv`.

    Returns:
        argparse.Namespace: Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--instances", type=int, default=12)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--model", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--output", type=Path, default=Path("results/surrogate-seed-study.json"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the study and write its JSON report.

    Args:
        argv (Sequence[str] | None): Arguments, or None for `sys.argv`.

    Returns:
        int: Process exit status.
    """
    options = parse_args(argv)
    study = SeedingStudy(
        options.size, options.budget, options.instances, options.resamples, options.model
    )
    rows = study.run()
    report = {
        "design": vars(options) | {"output": str(options.output)},
        "verdict": study.verdict(rows),
        "trials": rows,
    }
    study.save(report, options.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
