"""Bounded sparse continuous restoration with invocation-owned workspaces.

Objective:
- Restore continuous coordinates while preserving discrete decisions and strict checks.

Technical approach:
- Project every affine row and the continuous box in bounded complete sweeps.

Flow (pseudocode):
1. Classify fast paths and discrete obstructions.
2. Project rows and box using owned local arrays.
3. Return checked complete evidence or propagate the owner's stop signal.

Important criteria:
- Never publish clock-dependent partial candidates or an infeasibility certificate.

Inputs/Outputs:
- Input: validated storage, original assignment and ephemeral owner stop check.
- Output: final original-unit row state and bounded repair diagnostic.
"""

from __future__ import annotations

import numpy as np
from .records import RepairDiagnostic
from .repair_plan import RepairPlan
from .storage import LinearStorage
from collections.abc import Callable


class ContinuousRestoration(LinearStorage):
    """Restore affine feasibility without storing mutable candidate state.

    Responsibilities:
    - Own sparse plan preparation and bounded per-invocation projection work.
    - Preserve original domains, tolerances and successful-completion identity.
    """

    __slots__ = ()

    def _compile_repair(self) -> None:
        """Compile an immutable sparse projection plan.

        Raises:
            ValueError: If normalized projection data are numerically unsafe.
        """
        RepairPlan(self).apply()

    def _repair_bounds(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Freeze this candidate's nonmovable effects in normalized row bounds.

        Args:
            values (np.ndarray): Invocation-owned original-coordinate assignment.

        Returns:
            tuple: Normalized lower and upper affine row sides.
        """
        offsets = np.frombuffer(self._repair_offsets).copy()
        frozen = np.frombuffer(self._repair_frozen_entries, dtype=np.intp)
        offsets += np.bincount(
            np.frombuffer(self._row_indices, dtype=np.intp)[frozen],
            weights=np.frombuffer(self._coefficients)[frozen]
            * values[np.frombuffer(self._columns, dtype=np.intp)[frozen]],
            minlength=len(self.rows),
        )
        scales = np.frombuffer(self._repair_scales)
        return (
            (np.frombuffer(self._row_lower) - offsets) / scales,
            (np.frombuffer(self._row_upper) - offsets) / scales,
        )

    def _project_rows(
        self, z: np.ndarray, lower: np.ndarray, upper: np.ndarray, check: Callable[[], None] | None
    ) -> None:
        """Visit all affine supports once with the original cyclic order.

        Args:
            z (np.ndarray): Invocation-owned normalized continuous coordinates.
            lower (np.ndarray): Candidate-normalized lower row sides.
            upper (np.ndarray): Candidate-normalized upper row sides.
            check (Callable | None): Ephemeral owner stop hook.

        Raises:
            FloatingPointError: If projection arithmetic becomes nonfinite.
            Exception: Owner stop signals abort without a completed candidate.
        """
        indptr = np.frombuffer(self._repair_indptr, dtype=np.intp)
        columns = np.frombuffer(self._repair_columns, dtype=np.intp)
        coefficients = np.frombuffer(self._repair_coefficients)
        norms = np.frombuffer(self._repair_norms)
        for row in range(len(self.rows)):
            if check is not None:
                check()
            if norms[row] == 0:
                continue
            start, end = indptr[row : row + 2]
            support = columns[start:end]
            vector = coefficients[start:end]
            activity = float(np.dot(vector, z[support]))
            target = min(upper[row], max(lower[row], activity))
            z[support] += ((target - activity) / norms[row]) * vector

    def _restore_continuous(
        self,
        values: np.ndarray,
        state: tuple[np.ndarray, ...],
        budget_check: Callable[[], None] | None,
    ) -> tuple[tuple[np.ndarray, ...], RepairDiagnostic]:
        """Restore complete sweeps while freezing integer/binary choices.

        Args:
            values (np.ndarray): Invocation-owned physical assignments.
            state (tuple): Original-unit row state before restoration.
            budget_check (Callable | None): Ephemeral owner stop hook, never stored.

        Returns:
            tuple: Final original-unit state and deterministic repair work evidence.

        Raises:
            ValueError: If projection arithmetic is unsafe.
            Exception: Owner stop signals abort without a completed invocation.
        """
        if self.repair == "none":
            return state, RepairDiagnostic("none", "disabled", 0)
        free = np.frombuffer(self._repair_free, dtype=np.intp)
        if not len(free):
            return state, RepairDiagnostic("continuous", "no_continuous_variables", 0)
        if self._row_feasible(state):
            return state, RepairDiagnostic("continuous", "already_feasible", 0)
        norms = np.frombuffer(self._repair_norms)
        if np.any((norms == 0) & ((state[3] > 0) | (state[4] > 0))):
            return state, RepairDiagnostic("continuous", "fixed_obstruction", 0)
        lower = np.frombuffer(self._repair_lower)
        upper = np.frombuffer(self._repair_upper)
        width = np.frombuffer(self._repair_width)
        z = (values[free] - lower) / width
        row_lower, row_upper = self._repair_bounds(values)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                for sweep in range(1, self.repair_max_sweeps + 1):
                    self._project_rows(z, row_lower, row_upper, budget_check)
                    np.clip(z, 0.0, 1.0, out=z)
                    values[free] = np.clip(lower + width * z, lower, upper)
                    state = self._checked_repair_state(values)
                    if self._row_feasible(state):
                        return state, RepairDiagnostic("continuous", "repaired", sweep)
        except FloatingPointError as exc:
            raise ValueError("Continuous repair produced unsafe projection arithmetic.") from exc
        return state, RepairDiagnostic("continuous", "work_limit", self.repair_max_sweeps)

    @staticmethod
    def _row_feasible(state: tuple[np.ndarray, ...]) -> bool:
        """Test adjusted row sides without a hidden numerical allowance.

        Args:
            state (tuple): Original-unit row activity and residual arrays.

        Returns:
            bool: Whether both original adjusted side arrays are zero.
        """
        return not (np.any(state[3] > 0) or np.any(state[4] > 0))

    def _checked_repair_state(self, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Reject nonfinite physical assignments before original-unit validation.

        Args:
            values (np.ndarray): Complete full-sweep assignment.

        Returns:
            tuple: Authoritative original-unit row state.

        Raises:
            ValueError: If the physical assignment is nonfinite.
        """
        if not np.all(np.isfinite(values)):
            raise ValueError("Continuous repair produced nonfinite assignments.")
        return self._row_state(values)
