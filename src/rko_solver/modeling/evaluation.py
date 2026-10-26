"""Deterministic key mapping and original-unit linear evaluation.

Objective:
- Produce one portable completed assignment and scalar search score.

Technical approach:
- Validate keys, map domains, restore continuous coordinates and score once.

Flow (pseudocode):
1. Validate and map independent keys.
2. Restore eligible continuous coordinates with the existing strict row checker.
3. Evaluate the original objective and construct immutable diagnostics.

Important criteria:
- Preserve integer rounding, key identity, tolerances and finite-value checks.

Inputs/Outputs:
- Input: finite random-key vector and optional ephemeral owner stop hook.
- Output: immutable completed model evidence.
"""

from __future__ import annotations

import math
import numpy as np
from .records import ModelEvaluation
from .records import RowDiagnostic
from .repair import ContinuousRestoration
from collections.abc import Callable


class LinearEvaluation(ContinuousRestoration):
    """Map and score one invocation without retaining mutable candidate state.

    Responsibilities:
    - Keep original chromosome, assignment, objective and score together.
    - Apply the original finite/domain/tolerance contracts.
    """

    __slots__ = ()

    @property
    def n_keys(self) -> int:
        """Return the number of nonfixed coordinates.

        Returns:
            int: Required chromosome dimension, including zero for direct models.
        """
        return len(self._free) // np.dtype(np.intp).itemsize

    def _keys(self, keys: np.ndarray) -> np.ndarray:
        """Validate the caller vector before any mapping.

        Args:
            keys (np.ndarray): Original finite unit-interval chromosome.

        Returns:
            np.ndarray: Numeric vector without mutation.

        Raises:
            ValueError: If shape, numerical values or endpoints are invalid.
        """
        try:
            chromosome = np.asarray(keys, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Keys must be a finite numeric vector.") from exc
        finite = np.all(np.isfinite(chromosome))
        bounded = not (np.any(chromosome < 0) or np.any(chromosome > 1))
        if chromosome.shape != (self.n_keys,) or not finite or not bounded:
            raise ValueError(f"Expected {self.n_keys} finite keys in [0, 1].")
        return chromosome

    def _mapped(self, chromosome: np.ndarray) -> np.ndarray:
        """Apply the original affine/floor maps and exact physical box clipping.

        Args:
            chromosome (np.ndarray): Validated keys.

        Returns:
            np.ndarray: Invocation-owned complete original-coordinate assignment.

        Raises:
            ValueError: If mapping violates finite or exact integer domains.
        """
        lower = np.frombuffer(self._lower)
        upper = np.frombuffer(self._upper)
        integer = np.frombuffer(self._integer, dtype=bool)
        mapped = lower + chromosome * (upper - lower)
        mapped[integer] = np.minimum(
            upper[integer],
            lower[integer] + np.floor(chromosome[integer] * (upper[integer] - lower[integer] + 1)),
        )
        mapped = np.clip(mapped, lower, upper)
        finite = np.all(np.isfinite(mapped))
        integral = not np.any(mapped[integer] != np.floor(mapped[integer]))
        if not finite or not integral:
            raise ValueError("Mapping violated finite-domain/integrality invariants.")
        values = np.frombuffer(self._base).copy()
        values[np.frombuffer(self._free, dtype=np.intp)] = mapped
        return values

    @staticmethod
    def _feasible(state: tuple[np.ndarray, ...]) -> bool:
        """Check both original-unit adjusted side arrays.

        Args:
            state (tuple): Activities and raw/adjusted side residuals.

        Returns:
            bool: Whether every declared side allowance is satisfied.
        """
        return not (np.any(state[3] > 0) or np.any(state[4] > 0))

    def _score(self, objective: float, state: tuple[np.ndarray, ...]) -> float:
        """Compute the unchanged feasibility-first or explicit weighted score.

        Args:
            objective (float): Original affine objective including its constant.
            state (tuple): Original-unit activities and side violations.

        Returns:
            float: Scalar minimization score, validated by the caller.
        """
        _, lr, ur, lv, uv = state
        oriented = -objective if self.maximize else objective
        if self.scoring == "weighted_penalty":
            assert self.penalty_weight is not None
            penalty = math.fsum(np.square(lr[lv > 0]).tolist() + np.square(ur[uv > 0]).tolist())
            return oriented + self.penalty_weight * penalty
        if self._feasible(state):
            width = self._objective_upper - self._objective_lower
            fraction = (objective - self._objective_lower) / width if width > 0 else 0.0
            if self.maximize and width > 0:
                fraction = 1.0 - fraction
            return 0.5 * min(1.0, max(0.0, fraction))
        scales = np.frombuffer(self._row_scales)
        guidance = math.fsum((lv / scales + uv / scales).tolist())
        return 1.0 + guidance / max(1, len(self.rows))

    def decode(
        self, keys: np.ndarray, *, _budget_check: Callable[[], None] | None = None
    ) -> ModelEvaluation:
        """Map, restore and score one completed invocation without shared mutation.

        Args:
            keys (np.ndarray): Finite one-dimensional keys in [0,1].
            _budget_check (Callable | None): Ephemeral generated-session stop hook.

        Returns:
            ModelEvaluation: Immutable original-key and final-assignment evidence.

        Raises:
            ValueError: If keys or numerical arithmetic are invalid.
            Exception: Owner stop signals propagate without a completed evaluation.
        """
        chromosome = self._keys(keys)
        values = self._mapped(chromosome)
        state, repair = self._restore_continuous(values, self._row_state(values), _budget_check)
        mapped = values[np.frombuffer(self._free, dtype=np.intp)]
        objective = math.fsum(
            [self._objective_constant] + (np.frombuffer(self._objective) * mapped).tolist()
        )
        score = self._score(objective, state)
        if (
            not math.isfinite(objective)
            or not math.isfinite(score)
            or not np.all(np.isfinite(state[0]))
        ):
            raise ValueError("Model evaluation produced nonfinite arithmetic.")
        diagnostics = tuple(
            RowDiagnostic(row.identifier, *(float(v) for v in values))
            for row, values in zip(self.rows, zip(*state, strict=True), strict=True)
        )
        return ModelEvaluation(
            tuple(float(k) for k in chromosome),
            tuple(float(v) for v in values),
            objective,
            score,
            self._feasible(state),
            diagnostics,
            repair,
        )
