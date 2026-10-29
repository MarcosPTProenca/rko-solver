"""Analytic residual-priority and quantile-codec Knapsack decoders.

Objective:
- Provide training-free decoders that inject an analytic item score into random-key decoding, so
  the value of such a prior can be measured against the plain sort decoder before any network
  is trained.

Technical approach:
- `SortDecoder` orders items by ascending key and fills the knapsack greedily: the reference form.
- `PriorityDecoder` orders items by `(1 - score) + spread * (key - 0.5)`: keys at 0.5 reproduce the
  greedy value/weight solution, and a large spread recovers the plain sort decoder.
- `QuantileDecoder` draws each position from the remaining items with softmax probabilities of the
  score, using the key as the uniform variate of an inverse-CDF draw (`QuantileCodec`).
- Every decoder returns a feasible binary vector and is a pickleable callable.

Flow (pseudocode):
1. Rank-normalize the value/weight ratios of the instance into scores in [0, 1].
2. Turn the keys into an item order with the chosen decoder.
3. Walk the order and take every item that still fits.

Important criteria:
- Decoders are deterministic functions of the keys and the instance.
- The sort and priority decoders cost one argsort per candidate; the quantile decoder costs one
  cumulative sum per position, that is O(n^2) per candidate.

Inputs/Outputs:
- Input: a `KpInstance` and a random-key vector with one key per item.
- Output: a binary list where 1 marks a selected item.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from examples.run_kp_cpp_instances import KpInstance


class GreedyFill:
    """Fill a knapsack by walking an item order.

    Responsibilities:
    - Convert an item order into a feasible binary selection.
    """

    @staticmethod
    def select(order: np.ndarray, instance: KpInstance) -> list[int]:
        """Take every item of the order that still fits.

        Args:
            order (np.ndarray): Item indices from most to least preferred.
            instance (KpInstance): Knapsack data.

        Returns:
            list[int]: Binary vector with 1 for each selected item.
        """
        chosen = [0] * instance.n_items
        remaining = instance.capacity
        weights = instance.weights
        for index in order.tolist():
            if weights[index] <= remaining:
                chosen[index] = 1
                remaining -= weights[index]
        return chosen


class ItemScores:
    """Compute analytic item scores.

    Responsibilities:
    - Rank-normalize the value/weight ratio so scores do not depend on the instance scale.
    """

    @staticmethod
    def ratio(instance: KpInstance) -> np.ndarray:
        """Return scores in [0, 1], where 1 is the best value/weight ratio.

        Args:
            instance (KpInstance): Knapsack data.

        Returns:
            np.ndarray: One score per item.
        """
        ratios = np.asarray(instance.benefits, dtype=np.float64) / np.asarray(
            instance.weights, dtype=np.float64
        )
        ranks = np.empty(instance.n_items, dtype=np.float64)
        ranks[np.argsort(ratios, kind="stable")] = np.arange(instance.n_items)
        return ranks / max(instance.n_items - 1, 1)


class QuantileCodec:
    """Map uniform keys to categorical draws through the inverse CDF.

    Responsibilities:
    - Draw one category from nonnegative weights using a key as the uniform variate.
    - Draw a whole order without replacement.
    """

    @staticmethod
    def draw(uniform: float, weights: np.ndarray) -> int:
        """Return the category whose cumulative weight interval contains the key.

        Args:
            uniform (float): Key in [0, 1).
            weights (np.ndarray): Nonnegative weights with a positive sum.

        Returns:
            int: Index of the drawn category; zero-weight categories are never drawn.
        """
        cumulative = np.cumsum(weights)
        index = int(np.searchsorted(cumulative, uniform * cumulative[-1], side="right"))
        return min(index, len(weights) - 1)

    @classmethod
    def order(cls, keys: np.ndarray, scores: np.ndarray, temperature: float) -> np.ndarray:
        """Draw every item once, position by position, with softmax probabilities.

        Args:
            keys (np.ndarray): One key per position.
            scores (np.ndarray): Item scores; higher means more likely earlier.
            temperature (float): Positive softmax temperature.

        Returns:
            np.ndarray: Item indices in drawn order.
        """
        weights = np.exp((scores - scores.max()) / temperature)
        order = np.empty(len(keys), dtype=np.int64)
        for position, key in enumerate(keys.tolist()):
            index = cls.draw(key, weights)
            order[position] = index
            weights[index] = 0.0
        return order


@dataclass(slots=True)
class SortDecoder:
    """Order items by ascending key, the plain random-key reference decoder.

    Responsibilities:
    - Decode keys into a feasible greedy selection without any prior.

    Attributes:
        instance (KpInstance): Knapsack data.
    """

    instance: KpInstance

    def __call__(self, keys: np.ndarray) -> list[int]:
        """Decode a key vector.

        Args:
            keys (np.ndarray): One key per item.

        Returns:
            list[int]: Feasible binary selection.
        """
        return GreedyFill.select(np.argsort(np.asarray(keys), kind="stable"), self.instance)


@dataclass(slots=True, eq=False)
class PriorityDecoder:
    """Order items by an analytic prior plus a key-driven residual.

    Responsibilities:
    - Keep the greedy value/weight solution reachable at the neutral key 0.5.
    - Let a single spread control how far the keys may move items away from the prior.

    Attributes:
        instance (KpInstance): Knapsack data.
        spread (float): Residual scale; large values approach the plain sort decoder.
        base (np.ndarray): Prior position of each item, 0 being preferred.
    """

    instance: KpInstance
    spread: float
    base: np.ndarray

    @classmethod
    def build(cls, instance: KpInstance, spread: float) -> PriorityDecoder:
        """Create the decoder and precompute the prior.

        Args:
            instance (KpInstance): Knapsack data.
            spread (float): Positive residual scale.

        Returns:
            PriorityDecoder: Decoder ready to be called.

        Raises:
            ValueError: If the spread is not positive.
        """
        if spread <= 0.0:
            raise ValueError("'spread' must be positive.")
        return cls(instance, spread, 1.0 - ItemScores.ratio(instance))

    def __call__(self, keys: np.ndarray) -> list[int]:
        """Decode a key vector.

        Args:
            keys (np.ndarray): One key per item.

        Returns:
            list[int]: Feasible binary selection.
        """
        priority = self.base + self.spread * (np.asarray(keys) - 0.5)
        return GreedyFill.select(np.argsort(priority, kind="stable"), self.instance)


@dataclass(slots=True, eq=False)
class QuantileDecoder:
    """Draw the item order sequentially with score-driven probabilities.

    Responsibilities:
    - Use each key as the uniform variate of one inverse-CDF draw.

    Attributes:
        instance (KpInstance): Knapsack data.
        temperature (float): Softmax temperature; small values follow the prior closely.
        scores (np.ndarray): Analytic item scores.
    """

    instance: KpInstance
    temperature: float
    scores: np.ndarray

    @classmethod
    def build(cls, instance: KpInstance, temperature: float) -> QuantileDecoder:
        """Create the decoder and precompute the scores.

        Args:
            instance (KpInstance): Knapsack data.
            temperature (float): Positive softmax temperature.

        Returns:
            QuantileDecoder: Decoder ready to be called.

        Raises:
            ValueError: If the temperature is not positive.
        """
        if temperature <= 0.0:
            raise ValueError("'temperature' must be positive.")
        return cls(instance, temperature, ItemScores.ratio(instance))

    def __call__(self, keys: np.ndarray) -> list[int]:
        """Decode a key vector.

        Args:
            keys (np.ndarray): One key per item, in [0, 1).

        Returns:
            list[int]: Feasible binary selection.
        """
        order = QuantileCodec.order(np.asarray(keys), self.scores, self.temperature)
        return GreedyFill.select(order, self.instance)
