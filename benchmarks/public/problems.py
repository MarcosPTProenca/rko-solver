"""Decoders, objectives and independent checkers for the public benchmark families.

Objective:
- Give each public instance one decoder and one penalized cost, in two forms (a Numba function for
  the Rust engine and a picklable Python object for the Python engine), plus a checker that
  recomputes objective and feasibility from returned keys without the solver.

Technical approach:
- TSP: keys are sorted to a visiting order; cost is the closed tour length over the rounded matrix.
- Knapsack (KPLIB and OR-Library MDKP): key >= 0.5 selects an item; cost is minus profit plus
  PENALTY times the total row excess. Feasibility means no row exceeds its capacity.
- Gap is positive when worse than the reference: (found - reference) for minimization, and
  (reference - found) for maximization, both divided by max(1, |reference|).

Flow (pseudocode):
1. Build a `PublicProblem` from an instance and its direction.
2. `numba_objective()` compiles the cost for the Rust engine; `python_objective()` returns picklable
   callables for the Python engine.
3. `check(keys)` decodes with NumPy only and returns objective, feasibility and gap.

Important criteria:
- The penalty is fixed (1000.0) for every instance, so arms are compared on the same cost.
- The checker never calls the Numba or Python cost used by the solver.
- Instances without a reference report gap None.

Inputs/Outputs:
- Input: a TSP or MDKP record and an optional reference value.
- Output: callables for solvers, and `CheckResult` records.
"""

from __future__ import annotations

import numpy as np
from .parsers import MdkpInstance
from .parsers import TspInstance
from collections.abc import Callable
from dataclasses import dataclass
from numba import njit

PENALTY = 1000.0


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Independent evaluation of one key vector.

    Attributes:
        objective (float): Tour length or profit of the decoded solution.
        feasible (bool): Whether every constraint holds.
        gap (float | None): Relative distance from the reference, positive is worse.
        cost (float): Independently recomputed penalized minimization cost.
    """

    objective: float
    feasible: bool
    gap: float | None
    cost: float


def identity(keys: np.ndarray) -> np.ndarray:
    """Return the keys unchanged; the Python engine scores keys directly.

    Args:
        keys (np.ndarray): Key vector.

    Returns:
        np.ndarray: The same key vector.
    """
    return keys


class TspObjective:
    """Tour-length objective over a rounded distance matrix.

    Responsibilities:
    - Decode keys to a city order and score the closed tour.
    """

    minimize = True

    def __init__(self, distances: np.ndarray) -> None:
        """Store the distance matrix.

        Args:
            distances (np.ndarray): Square matrix of nonnegative distances.
        """
        self.distances = distances
        self.size = distances.shape[0]

    def decode(self, keys: np.ndarray) -> np.ndarray:
        """Return the visiting order that sorts the keys ascending.

        Args:
            keys (np.ndarray): One key per city.

        Returns:
            np.ndarray: City indices in visiting order.
        """
        return np.argsort(keys, kind="stable")

    def cost(self, keys: np.ndarray) -> float:
        """Return the closed tour length of the decoded order.

        Args:
            keys (np.ndarray): One key per city.

        Returns:
            float: Tour length including the return edge.
        """
        order = self.decode(keys)
        return float(self.distances[order, np.roll(order, -1)].sum())

    def numba_function(self) -> Callable[[np.ndarray], float]:
        """Compile the tour-length objective for the native engine.

        Returns:
            Callable[[np.ndarray], float]: Lazily compiled tour objective.
        """
        distances = self.distances

        @njit(cache=False)
        def tour(keys: np.ndarray) -> float:
            """Sum closed-tour edges after stable sorting by key and city index.

            Args:
                keys (np.ndarray): One finite key per city.

            Returns:
                float: Closed-tour distance.
            """
            order = np.argsort(keys, kind="mergesort")
            total = 0.0
            for position in range(order.size):
                total += distances[order[position], order[(position + 1) % order.size]]
            return total

        return tour


class KnapsackObjective:
    """Penalized 0/1 multi-row knapsack objective with a 0.5 threshold decoder.

    Responsibilities:
    - Decode keys to a 0/1 selection and score profit minus penalized row excess.
    """

    minimize = False

    def __init__(self, profits: np.ndarray, weights: np.ndarray, capacities: np.ndarray) -> None:
        """Store the knapsack data.

        Args:
            profits (np.ndarray): Length-n profits.
            weights (np.ndarray): m-by-n weights.
            capacities (np.ndarray): Length-m capacities.
        """
        self.profits = profits
        self.weights = weights
        self.capacities = capacities

    def decode(self, keys: np.ndarray) -> np.ndarray:
        """Return the 0/1 selection with key at least one half.

        Args:
            keys (np.ndarray): One key per item.

        Returns:
            np.ndarray: Float selection vector.
        """
        return (keys >= 0.5).astype(float)

    def cost(self, keys: np.ndarray) -> float:
        """Return minus profit plus the penalized total row excess.

        Args:
            keys (np.ndarray): One key per item.

        Returns:
            float: Minimized penalized value.
        """
        selection = self.decode(keys)
        excess = np.maximum(0.0, self.weights @ selection - self.capacities).sum()
        return float(-(self.profits @ selection) + PENALTY * excess)

    def numba_function(self) -> Callable[[np.ndarray], float]:
        """Compile the penalized knapsack cost for the native engine.

        Returns:
            Callable[[np.ndarray], float]: Lazily compiled penalized objective.
        """
        profits = self.profits
        weights = self.weights
        capacities = self.capacities

        @njit(cache=False)
        def penalized(keys: np.ndarray) -> float:
            """Score selected profit and total capacity excess.

            Args:
                keys (np.ndarray): One key per item; values at least 0.5 select it.

            Returns:
                float: Negative profit plus penalized excess.
            """
            total = 0.0
            excess = 0.0
            for row in range(capacities.size):
                load = 0.0
                for item in range(keys.size):
                    if keys[item] >= 0.5:
                        load += weights[row, item]
                excess += max(0.0, load - capacities[row])
            for item in range(keys.size):
                if keys[item] >= 0.5:
                    total += profits[item]
            return -total + PENALTY * excess

        return penalized


class PublicProblem:
    """One public instance with its objective, direction and optional reference.

    Responsibilities:
    - Expose solver-facing costs and an independent checker on the same instance.

    Attributes:
        name (str): Instance name used in records.
        family (str): One of tsplib, kplib, orlib-mknap1, orlib-mknapcb1.
        objective (TspObjective | KnapsackObjective): Decoder and cost.
        reference (float | None): Optimal or best-known value, or None.
    """

    def __init__(
        self,
        name: str,
        family: str,
        objective: TspObjective | KnapsackObjective,
        reference: float | None,
    ) -> None:
        """Store the instance and its objective.

        Args:
            name (str): Instance name.
            family (str): Family label.
            objective (TspObjective | KnapsackObjective): Decoder and cost.
            reference (float | None): Reference value, or None.
        """
        self.name = name
        self.family = family
        self.objective = objective
        self.reference = reference

    @property
    def n_variables(self) -> int:
        """Return the key dimension: one per city or item.

        Returns:
            int: Number of random keys.
        """
        if isinstance(self.objective, TspObjective):
            return self.objective.size
        return self.objective.profits.size

    @staticmethod
    def from_tsp(instance: TspInstance, reference: float | None) -> PublicProblem:
        """Build a TSP problem from a parsed TSPLIB instance.

        Args:
            instance (TspInstance): Parsed distances.
            reference (float | None): Optimal tour length.

        Returns:
            PublicProblem: Minimization problem.
        """
        return PublicProblem(instance.name, "tsplib", TspObjective(instance.distances), reference)

    @staticmethod
    def from_mdkp(family: str, instance: MdkpInstance) -> PublicProblem:
        """Build a knapsack problem from a parsed OR-Library instance.

        Args:
            family (str): Family label, for example orlib-mknap1.
            instance (MdkpInstance): Parsed profits, weights and capacities.

        Returns:
            PublicProblem: Maximization problem.
        """
        objective = KnapsackObjective(instance.profits, instance.weights, instance.capacities)
        return PublicProblem(instance.name, family, objective, instance.optimum)

    def check(self, keys: np.ndarray) -> CheckResult:
        """Recompute objective, feasibility and gap from returned keys without the solver's cost.

        Args:
            keys (np.ndarray): Best key vector returned by a solve.

        Returns:
            CheckResult: Independent evaluation; infeasible solutions have no gap.

        Raises:
            ValueError: If keys have the wrong dimension or contain nonfinite values.
        """
        if keys.shape != (self.n_variables,) or not np.isfinite(keys).all():
            raise ValueError(f"{self.name}: invalid returned keys.")
        objective = self.objective
        if isinstance(objective, TspObjective):
            order = objective.decode(keys)
            found = float(objective.distances[order, np.roll(order, -1)].sum())
            feasible = bool(np.array_equal(np.sort(order), np.arange(objective.size)))
            cost = found
        else:
            selection = objective.decode(keys)
            found = float(objective.profits @ selection)
            loads = objective.weights @ selection
            excess = np.maximum(0.0, loads - objective.capacities).sum()
            feasible = bool(np.all(loads <= objective.capacities + 1e-9))
            cost = float(-found + PENALTY * excess)
        gap = self._gap(found) if feasible else None
        return CheckResult(found, feasible, gap, cost)

    def _gap(self, found: float) -> float | None:
        """Compute a direction-aware relative gap.

        Args:
            found (float): Independently recomputed objective.

        Returns:
            float | None: Relative shortfall, or None without a reference.
        """
        if self.reference is None:
            return None
        scale = max(1.0, abs(self.reference))
        if self.objective.minimize:
            return (found - self.reference) / scale
        return (self.reference - found) / scale
