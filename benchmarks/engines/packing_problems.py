"""Multi-constraint packing instances, priority decoders and hand-written priors.

Objective:
- Provide a family where a hand-written ratio rule is a plausible but beatable prior, next to
  plain Knapsack where it is nearly optimal, for testing learned item priorities.

Technical approach:
- One instance class holds profits, a constraint-by-item weight matrix and capacities; plain
  Knapsack is the single-constraint case.
- A decoder orders items by a rank-normalized priority plus a key-driven residual and fills
  greedily; without priorities it orders by key alone, the plain random-key reference.

Flow (pseudocode):
1. Generate an instance from a kind, size and seed.
2. Score items (hand-written or learned), turn scores into an order through the decoder.
3. Fill items in order while every constraint still holds; cost is the negative profit.

Important criteria:
- Every decoded selection is feasible for all constraints.
- Keys at 0.5 reproduce the score order; a very large spread reproduces the plain sort decoder.
- Decoders and costs are pickleable for spawned workers.

Inputs/Outputs:
- Input: instance kind, item count, seed, score vectors and random keys.
- Output: instances, feasible binary selections, profits and key vectors encoding an order.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from dataclasses import field


@dataclass(slots=True, eq=False)
class PackingInstance:
    """Profits, constraint weights and capacities of one packing instance.

    Responsibilities:
    - Hold validated numeric data for any number of knapsack constraints.

    Attributes:
        profits (np.ndarray): Positive profit of each item.
        weights (np.ndarray): Nonnegative weights, one row per constraint.
        capacities (np.ndarray): Positive capacity of each constraint.
        name (str): Instance identifier.
    """

    profits: np.ndarray
    weights: np.ndarray
    capacities: np.ndarray
    name: str

    @property
    def n_items(self) -> int:
        """Return the number of items.

        Returns:
            int: Item count.
        """
        return int(len(self.profits))


class PackingFamily:
    """Generate instances and compute item features, priors and greedy fills.

    Responsibilities:
    - Build the two benchmark kinds from a seed.
    - Offer the instance-only quantities that scorers and decoders consume.
    """

    KINDS = ("knapsack", "mkp")

    @classmethod
    def make(cls, kind: str, n_items: int, seed: int) -> PackingInstance:
        """Create one reproducible instance.

        Args:
            kind (str): `knapsack` (one strongly correlated constraint) or `mkp` (five
                constraints, tightness 0.25, profits correlated with the weights).
            n_items (int): Positive number of items.
            seed (int): Instance seed.

        Returns:
            PackingInstance: Generated instance.

        Raises:
            ValueError: If the kind is unknown or the item count is not positive.
        """
        if kind not in cls.KINDS:
            raise ValueError(f"'kind' must be one of {cls.KINDS}.")
        if n_items < 1:
            raise ValueError("'n_items' must be positive.")
        rng = np.random.default_rng([seed, n_items, cls.KINDS.index(kind)])
        if kind == "knapsack":
            weights = rng.integers(1, 101, (1, n_items)).astype(np.float64)
            capacity = max(float(weights.max()), float(weights.sum()) // 2)
            return PackingInstance(
                weights[0] + 10.0, weights, np.array([capacity]), f"knapsack-{n_items}-{seed}"
            )
        weights = rng.integers(0, 1001, (5, n_items)).astype(np.float64)
        profits = weights.mean(axis=0) + 500.0 * (rng.random(n_items) + 1e-3)
        capacities = np.maximum(0.25 * weights.sum(axis=1), 1.0)
        return PackingInstance(profits, weights, capacities, f"mkp-{n_items}-{seed}")

    @staticmethod
    def features(instance: PackingInstance) -> np.ndarray:
        """Return instance-only item features: relative profit, profit rank, relative weights.

        Args:
            instance (PackingInstance): Packing instance.

        Returns:
            np.ndarray: Matrix with one row per item and `2 + n_constraints` columns.
        """
        n_items = instance.n_items
        ranks = np.empty(n_items)
        ranks[np.argsort(instance.profits, kind="stable")] = np.arange(n_items)
        relative_weights = instance.weights / np.maximum(
            instance.weights.mean(axis=1, keepdims=True), 1e-12
        )
        columns = [instance.profits / instance.profits.mean(), ranks / max(n_items - 1, 1)]
        return np.column_stack([*columns, *relative_weights])

    @staticmethod
    def hand_scores(instance: PackingInstance) -> np.ndarray:
        """Return the hand-written prior: profit over capacity-normalized total weight.

        Args:
            instance (PackingInstance): Packing instance.

        Returns:
            np.ndarray: Score per item; larger means more attractive.
        """
        burden = (instance.weights / instance.capacities[:, None]).sum(axis=0)
        return np.asarray(instance.profits / np.maximum(burden, 1e-12))

    @staticmethod
    def fill(order: np.ndarray, instance: PackingInstance) -> tuple[np.ndarray, float]:
        """Take every item of the order that still fits all constraints.

        Args:
            order (np.ndarray): Item indices from most to least preferred.
            instance (PackingInstance): Packing instance.

        Returns:
            tuple[np.ndarray, float]: Binary selection and its total profit.
        """
        used = np.zeros(len(instance.capacities))
        selection = np.zeros(instance.n_items)
        for item in order.tolist():
            if np.all(used + instance.weights[:, item] <= instance.capacities):
                used += instance.weights[:, item]
                selection[item] = 1.0
        return selection, float(instance.profits @ selection)

    @staticmethod
    def order_keys(scores: np.ndarray) -> list[float]:
        """Encode a score order as one key vector whose ascending order is best-first.

        Args:
            scores (np.ndarray): Score per item.

        Returns:
            list[float]: One key in (0, 1) per item.
        """
        order = np.argsort(-np.asarray(scores), kind="stable")
        keys = np.empty(len(order))
        keys[order] = (np.arange(len(order)) + 0.5) / len(order)
        return keys.tolist()


@dataclass(slots=True, eq=False)
class PackingDecoder:
    """Decode keys into a feasible selection through an optional priority.

    Responsibilities:
    - Order items by rank-normalized score plus a key-driven residual, or by key alone.

    Attributes:
        instance (PackingInstance): Packing instance.
        scores (np.ndarray | None): Item scores, None for the plain sort decoder.
        spread (float): Residual scale; large values approach the plain sort decoder.
    """

    instance: PackingInstance
    scores: np.ndarray | None = None
    spread: float = 1.0
    base: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Precompute the rank-normalized priority once per instance."""
        if self.scores is not None:
            ranks = np.empty(self.instance.n_items)
            ranks[np.argsort(-np.asarray(self.scores), kind="stable")] = np.arange(
                self.instance.n_items
            )
            self.base = ranks / max(self.instance.n_items - 1, 1)

    def __call__(self, keys: np.ndarray) -> np.ndarray:
        """Decode one key vector.

        Args:
            keys (np.ndarray): One key per item.

        Returns:
            np.ndarray: Feasible binary selection.
        """
        keys = np.asarray(keys)
        priority = keys if self.base is None else self.base + self.spread * (keys - 0.5)
        return PackingFamily.fill(np.argsort(priority, kind="stable"), self.instance)[0]


@dataclass(slots=True)
class PackingCost:
    """Negative profit of a decoded selection.

    Attributes:
        instance (PackingInstance): Packing instance.
    """

    instance: PackingInstance

    def __call__(self, selection: np.ndarray) -> float:
        """Compute the minimization objective.

        Args:
            selection (np.ndarray): Binary selection.

        Returns:
            float: Negative total profit.

        Raises:
            ValueError: If the selection violates a constraint or has the wrong length.
        """
        selection = np.asarray(selection, dtype=np.float64)
        if selection.shape != (self.instance.n_items,):
            raise ValueError("Selection length must equal the item count.")
        if np.any(self.instance.weights @ selection > self.instance.capacities + 1e-9):
            raise ValueError("Selection violates a constraint.")
        return -float(self.instance.profits @ selection)
