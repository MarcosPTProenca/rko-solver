"""Tuners implemented in this repository: random search with halving and iterated racing.

Objective:
- Provide the two methods that need no extra dependency, behind the interface that the external
  tuners (Optuna, RKO itself) also follow.

Technical approach:
- `RandomHalving` draws many random configurations and keeps the best third on a growing number
  of training instances (successive halving), so cheap short evaluations discard most of the
  space before any configuration is scored on every instance.
- `IteratedRacing` follows the structure of irace: each iteration samples configurations (the
  first uniformly, later ones around the elites with a shrinking spread), races them on the
  training instances in a shared order and removes those a paired t-test shows to be worse than
  the best, then carries the survivors on as elites. It is irace-style, not irace itself.

Flow (pseudocode):
1. Receive a metered objective and a seed.
2. Spend the budget through the method's own schedule.
3. Return the configuration the method trusts most.

Important criteria:
- Every method stops when the objective raises `BudgetExhausted` and still returns its best
  configuration so far; no method reads held-out instances.
- All methods search the shared unit cube of `ParameterSpace`.

Inputs/Outputs:
- Input: a `TuningObjective`, a `ParameterSpace` and an integer seed.
- Output: the chosen configuration dictionary.
"""

from __future__ import annotations

import math
import numpy as np
from abc import ABC
from abc import abstractmethod
from benchmarks.tuning.objective import BudgetExhausted
from benchmarks.tuning.objective import TuningObjective
from benchmarks.tuning.spaces import ParameterSpace
from scipy import stats


class Tuner(ABC):
    """Common interface of every tuning method.

    Responsibilities:
    - Turn a budget of solver runs into one chosen configuration.

    Attributes:
        name (str): Short identifier used in results.
    """

    name = "tuner"

    @abstractmethod
    def tune(
        self, objective: TuningObjective, space: ParameterSpace, seed: int
    ) -> dict[str, float]:
        """Choose a configuration within the objective's budget.

        Args:
            objective (TuningObjective): Metered objective on the training instances.
            space (ParameterSpace): Parameters to tune.
            seed (int): Seed of the method's own randomness.

        Returns:
            dict[str, float]: Chosen configuration.
        """


class RandomHalving(Tuner):
    """Random search with successive halving.

    Responsibilities:
    - Sample random configurations and halve them over growing instance counts.

    Attributes:
        eta (int): Fraction kept per rung is `1 / eta`.
        first (int): Instances scored in the first rung.
    """

    name = "random_halving"

    def __init__(self, eta: int = 3, first: int = 2) -> None:
        """Create the method.

        Args:
            eta (int): Reduction factor, at least two.
            first (int): Instances in the first rung, at least one.

        Raises:
            ValueError: If a parameter is out of range.
        """
        if eta < 2 or first < 1:
            raise ValueError("eta must be at least 2 and first at least 1.")
        self.eta = eta
        self.first = first

    def tune(
        self, objective: TuningObjective, space: ParameterSpace, seed: int
    ) -> dict[str, float]:
        """Run halving rungs until one configuration is left or the budget ends.

        Args:
            objective (TuningObjective): Metered objective.
            space (ParameterSpace): Parameters to tune.
            seed (int): Seed for the random configurations.

        Returns:
            dict[str, float]: Survivor with the lowest mean score.
        """
        rng = np.random.default_rng(seed)
        rungs = self._rungs(objective.n_instances)
        count = self._population(objective.budget, rungs)
        alive: list[tuple[dict[str, float], list[float]]] = [
            (space.decode(rng.random(space.dimension)), []) for _ in range(count)
        ]
        return self._climb(objective, rungs, alive)[0]

    def _climb(
        self,
        objective: TuningObjective,
        rungs: list[int],
        alive: list[tuple[dict[str, float], list[float]]],
    ) -> tuple[dict[str, float], list[float]]:
        """Score and halve the configurations rung by rung.

        Args:
            objective (TuningObjective): Metered objective.
            rungs (list[int]): Cumulative instance counts of the rungs.
            alive (list[tuple[dict[str, float], list[float]]]): Configurations with their scores.

        Returns:
            tuple[dict[str, float], list[float]]: Best configuration and its scores; when the
                budget ends mid-rung, the best among those scored on the first rung.
        """
        best = alive[0]
        try:
            for index, size in enumerate(rungs):
                done = rungs[index - 1] if index else 0
                for config, scores in alive:
                    scores.extend(objective.run(config, i) for i in range(done, size))
                alive.sort(key=lambda item: float(np.mean(item[1])))
                best = alive[0]
                alive = alive[: max(1, len(alive) // self.eta)]
        except BudgetExhausted:
            scored = [item for item in alive if len(item[1]) >= rungs[0]]
            best = min(scored or [best], key=lambda item: float(np.mean(item[1] or [math.inf])))
        return best

    def _population(self, budget: int, rungs: list[int]) -> int:
        """Return how many configurations the first rung can afford.

        Args:
            budget (int): Total runs allowed.
            rungs (list[int]): Cumulative instance counts of the rungs.

        Returns:
            int: Number of random configurations to start with.
        """
        cost = sum((rungs[i] - (rungs[i - 1] if i else 0)) / self.eta**i for i in range(len(rungs)))
        return max(self.eta ** (len(rungs) - 1), int(budget / cost))

    def _rungs(self, n_instances: int) -> list[int]:
        """Return the cumulative instance counts of the rungs.

        Args:
            n_instances (int): Training instances available.

        Returns:
            list[int]: Increasing counts ending at `n_instances`.
        """
        sizes = [min(self.first, n_instances)]
        while sizes[-1] < n_instances:
            sizes.append(min(n_instances, sizes[-1] * self.eta))
        return sizes


class IteratedRacing(Tuner):
    """Iterated racing in the style of irace.

    Responsibilities:
    - Race configurations with statistical elimination and resample around the survivors.

    Attributes:
        iterations (int): Number of racing iterations.
        first_test (int): Instances scored before any elimination.
        keep (int): Elites carried between iterations and the race stops at this many.
        alpha (float): Significance level of the elimination test.
    """

    name = "racing"

    def __init__(
        self, iterations: int = 4, first_test: int = 3, keep: int = 4, alpha: float = 0.05
    ) -> None:
        """Create the method.

        Args:
            iterations (int): Racing iterations, at least one.
            first_test (int): Instances before the first elimination, at least two.
            keep (int): Elites kept, at least one.
            alpha (float): Elimination significance level in (0, 1).

        Raises:
            ValueError: If a parameter is out of range.
        """
        if iterations < 1 or first_test < 2 or keep < 1 or not 0 < alpha < 1:
            raise ValueError("Invalid racing parameters.")
        self.iterations = iterations
        self.first_test = first_test
        self.keep = keep
        self.alpha = alpha

    def tune(
        self, objective: TuningObjective, space: ParameterSpace, seed: int
    ) -> dict[str, float]:
        """Run the racing iterations.

        Args:
            objective (TuningObjective): Metered objective.
            space (ParameterSpace): Parameters to tune.
            seed (int): Seed for sampling and instance order.

        Returns:
            dict[str, float]: Elite with the lowest mean score over its evaluated instances.
        """
        rng = np.random.default_rng(seed)
        elites: list[tuple[np.ndarray, list[float]]] = []
        try:
            for iteration in range(self.iterations):
                share = objective.remaining // (self.iterations - iteration)
                limit = objective.runs_used + share
                candidates = self._sample(rng, space, elites, iteration, max(6, share // 4))
                elites = self._race(objective, space, rng, candidates, limit)
        except BudgetExhausted:
            pass
        if not elites:
            return space.decode(rng.random(space.dimension))
        return space.decode(min(elites, key=lambda item: float(np.mean(item[1])))[0])

    def _sample(
        self,
        rng: np.random.Generator,
        space: ParameterSpace,
        elites: list[tuple[np.ndarray, list[float]]],
        iteration: int,
        count: int,
    ) -> list[np.ndarray]:
        """Draw the candidates of one iteration.

        Args:
            rng (np.random.Generator): Random generator.
            space (ParameterSpace): Parameters to tune.
            elites (list[tuple[np.ndarray, list[float]]]): Survivors of the last race.
            iteration (int): Iteration index.
            count (int): Number of candidates including the elites.

        Returns:
            list[np.ndarray]: Unit-cube points, the elites first.
        """
        if not elites:
            return [rng.random(space.dimension) for _ in range(count)]
        spread = 0.5 * (1.0 / count) ** (iteration / (space.dimension * self.iterations))
        points = [point for point, _ in elites]
        while len(points) < count:
            parent = points[int(rng.integers(len(elites)))]
            points.append(np.clip(rng.normal(parent, spread), 0.0, 1.0))
        return points

    def _race(
        self,
        objective: TuningObjective,
        space: ParameterSpace,
        rng: np.random.Generator,
        candidates: list[np.ndarray],
        limit: int,
    ) -> list[tuple[np.ndarray, list[float]]]:
        """Race candidates on a shared instance order and return the survivors.

        Args:
            objective (TuningObjective): Metered objective.
            space (ParameterSpace): Parameters to tune.
            rng (np.random.Generator): Random generator for the instance order.
            candidates (list[np.ndarray]): Unit-cube points to race.
            limit (int): Value of `runs_used` at which this race must stop.

        Returns:
            list[tuple[np.ndarray, list[float]]]: At most `keep` survivors with their scores.
        """
        order = [int(i) for i in rng.permutation(objective.n_instances)]
        alive = [(point, []) for point in candidates]
        for step, instance in enumerate(order, start=1):
            if objective.runs_used + len(alive) > limit:
                break
            for point, scores in alive:
                scores.append(objective.run(space.decode(point), instance))
            if step >= self.first_test and len(alive) > self.keep:
                alive = self._eliminate(alive)
        alive.sort(key=lambda item: float(np.mean(item[1])))
        return alive[: self.keep]

    def _eliminate(
        self, alive: list[tuple[np.ndarray, list[float]]]
    ) -> list[tuple[np.ndarray, list[float]]]:
        """Drop candidates that a paired t-test shows to be worse than the best.

        Args:
            alive (list[tuple[np.ndarray, list[float]]]): Candidates with equal-length scores.

        Returns:
            list[tuple[np.ndarray, list[float]]]: Candidates that remain, at least `keep`.
        """
        ranked = sorted(alive, key=lambda item: float(np.mean(item[1])))
        best = np.asarray(ranked[0][1])
        kept = ranked[: self.keep]
        for item in ranked[self.keep :]:
            difference = np.asarray(item[1]) - best
            if not np.any(difference):
                kept.append(item)
            elif np.ptp(difference) == 0:
                if difference[0] < 0:
                    kept.append(item)
            elif not stats.ttest_1samp(difference, 0.0, alternative="greater").pvalue < self.alpha:
                kept.append(item)
        return kept
