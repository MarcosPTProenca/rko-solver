"""Positive categorical probability intervals for reversible quantile construction.

Objective:
- Represent legal-action probabilities without silently discarding numerical support.

Technical approach:
- Normalize scores over legal actions and mix with positive uniform mass.

Flow (pseudocode):
1. Validate action scores and numerical configuration.
2. Construct probabilities and their cumulative bounds.
3. Use the same intervals for selection and midpoint encoding.

Important criteria:
- Only representable, noncollapsed intervals may be used.

Inputs/Outputs:
- Input: finite legal-action scores.
- Output: categorical probability intervals.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


class NumericalFailure(ValueError):
    """Reject a categorical distribution that loses representable action support.

    Responsibilities:
    - Distinguish finite-precision failures from ordinary invalid inputs.
    """


@dataclass(frozen=True, slots=True, eq=False)
class Intervals:
    """Represent categorical actions through cumulative probability intervals.

    Responsibilities:
    - Build probability intervals from finite scores.

    Attributes:
        probabilities (np.ndarray): One probability per ordered legal action.
        bounds (np.ndarray): Cumulative endpoints, including zero and one.
    """

    probabilities: np.ndarray
    bounds: np.ndarray

    @classmethod
    def from_logits(
        cls, logits: np.ndarray, temperature: float = 1.0, mixture: float = 0.05
    ) -> Intervals:
        """Construct a categorical distribution with uniform support.

        Args:
            logits (np.ndarray): One finite score per legal action.
            temperature (float): Positive finite score temperature.
            mixture (float): Uniform probability coefficient strictly between zero and one.

        Returns:
            Intervals: Categorical probability intervals.

        Raises:
            ValueError: If scores, temperature or the uniform mixture are invalid.
            NumericalFailure: If a cumulative interval collapses in finite precision.
        """
        scores = np.asarray(logits, dtype=np.float64)
        cls.validate(scores, temperature, mixture)
        with np.errstate(over="ignore", under="ignore"):
            weights = np.exp((scores - scores.max()) / temperature)
        probabilities = (1.0 - mixture) * weights / weights.sum() + mixture / len(scores)
        bounds = np.concatenate(([0.0], np.cumsum(probabilities)))
        bounds[-1] = 1.0
        if not np.all(np.diff(bounds) > 0.0):
            raise NumericalFailure("A cumulative probability interval collapsed.")
        probabilities.setflags(write=False)
        bounds.setflags(write=False)
        return cls(probabilities, bounds)

    @staticmethod
    def validate(scores: np.ndarray, temperature: float, mixture: float) -> None:
        """Check the finite categorical input domain before normalization.

        Args:
            scores (np.ndarray): Input action scores.
            temperature (float): Positive finite temperature.
            mixture (float): Strictly interior uniform probability coefficient.

        Raises:
            ValueError: If a declared numerical input is invalid.
        """
        if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
            raise ValueError("Action scores must be a nonempty finite vector.")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("The temperature must be positive and finite.")
        if not 0.0 < mixture < 1.0:
            raise ValueError("The mixture must lie strictly between zero and one.")

    def select(self, key: float) -> int:
        """Select the action owning a half-open cumulative interval.

        Args:
            key (float): Finite key in the half-open unit interval.

        Returns:
            int: Index into the ordered legal-action list.

        Raises:
            ValueError: If the key is nonfinite or outside the declared domain.
        """
        if not np.isfinite(key) or not 0.0 <= key < 1.0:
            raise ValueError("The decision key must be finite and in [0, 1).")
        return int(np.searchsorted(self.bounds[1:], key, side="right"))

    def midpoint(self, action: int) -> float:
        """Encode one action using its representable strict interval midpoint.

        Args:
            action (int): Index into the ordered legal-action list.

        Returns:
            float: Interior decision key reproducing the action.

        Raises:
            ValueError: If the action index is invalid.
            NumericalFailure: If no strict midpoint is representable.
        """
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(self.probabilities):
            raise ValueError("The action index is outside the legal-action list.")
        lower, upper = self.bounds[action : action + 2]
        midpoint = float(lower + (upper - lower) / 2.0)
        if not lower < midpoint < upper:
            raise NumericalFailure("The interval has no representable strict interior midpoint.")
        return midpoint
