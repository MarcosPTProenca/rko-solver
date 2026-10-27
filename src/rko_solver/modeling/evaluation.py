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
from .records import RepairDiagnostic
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

    def validate_keys(self, keys: np.ndarray) -> np.ndarray:
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
        chromosome = self.validate_keys(keys)
        return self._complete(chromosome, self._mapped(chromosome), _budget_check=_budget_check)

    def evaluate_assignment(
        self,
        keys: np.ndarray,
        values: object,
        *,
        restore: bool = False,
        _budget_check: Callable[[], None] | None = None,
    ) -> ModelEvaluation:
        """Validate and evaluate supplied original coordinates without reconstructing keys.

        Args:
            keys (np.ndarray): Exact originating finite unit-interval chromosome.
            values (object): Complete one-dimensional original-order numeric assignment.
            restore (bool): Explicitly apply the frozen continuous restoration policy.
            _budget_check (Callable | None): Ephemeral owner stop hook, never candidate context.

        Returns:
            ModelEvaluation: Detached checked assignment and original-unit numerical evidence.

        Raises:
            ValueError: If keys, shape, scalars, exact domains or arithmetic are invalid.
            Exception: Owner stop signals propagate without a completed evaluation.
        """
        chromosome = self.validate_keys(keys)
        if type(restore) is not bool:
            raise ValueError("restore must be a boolean.")
        assignment = self._assignment(values)
        return self._complete(chromosome, assignment, restore=restore, _budget_check=_budget_check)

    def _assignment(self, values: object) -> np.ndarray:
        """Validate scalar identity before float conversion and detach original coordinates.

        Args:
            values (object): Untrusted complete assignment, not claimed evaluation evidence.

        Returns:
            np.ndarray: Invocation-owned finite, exact-domain numerical values.

        Raises:
            ValueError: If shape, scalar precision, bounds, integrality or fixed values fail.
        """
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise ValueError("Assignment must be a complete numeric vector.")
        if isinstance(values, np.ndarray) and values.ndim != 1:
            raise ValueError("Assignment must be one-dimensional.")
        if len(values) != len(self.variables):
            raise ValueError("Assignment must match the original variable order and dimension.")
        result = []
        for variable, domain, value in zip(self.variables, self.domains, values, strict=True):
            numeric = self._number(value, f"assignment {variable.identifier}")
            integral = variable.domain == "continuous" or numeric.is_integer()
            if not domain.lower <= numeric <= domain.upper or not integral:
                raise ValueError("Assignment violates an effective bound, fixed value or domain.")
            result.append(numeric)
        return np.asarray(result, dtype=float)

    def _complete(
        self,
        chromosome: np.ndarray,
        values: np.ndarray,
        *,
        restore: bool = True,
        _budget_check: Callable[[], None] | None = None,
    ) -> ModelEvaluation:
        """Finish a mapped or validated assignment through the single protected evaluator.

        Args:
            chromosome (np.ndarray): Validated originating keys.
            values (np.ndarray): Owned exact-domain complete assignment.
            restore (bool): Whether actual construction includes frozen continuous repair.
            _budget_check (Callable | None): Ephemeral cooperative owner stop hook.

        Returns:
            ModelEvaluation: One completed immutable original-objective and score witness.

        Raises:
            ValueError: If protected numerical arithmetic becomes nonfinite.
            Exception: Owner stop signals propagate before witness completion.
        """
        state = self._row_state(values)
        if restore:
            state, repair = self._restore_continuous(values, state, _budget_check)
        else:
            repair = RepairDiagnostic("none", "disabled", 0)
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
