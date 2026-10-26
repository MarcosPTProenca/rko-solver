"""Validated transactional native assignment export.

Objective:
- Apply completed feasible witnesses without changing model bounds or domains.

Technical approach:
- Validate snapshot provenance, save restorable values and rollback every attempted setter.

Flow (pseudocode):
1. Verify source-compatible numerical state and completed feasible evidence.
2. Apply original-coordinate values in stable order.
3. Restore every attempted value on failure and report rollback failures explicitly.

Important criteria:
- Never hide assignment or rollback errors; MathOpt has no assignment slots.

Inputs/Outputs:
- Input: source, snapshot, feasible evidence and validated native mapping.
- Output: explicit native values or an error with rollback diagnostics.
"""

from __future__ import annotations

import math
import numpy as np
from .linear import CompiledLinearModel
from .records import ModelEvaluation
from .source import ExtractedLinearModel
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

if TYPE_CHECKING:
    from pulp import LpVariable
    from pyomo.core.base.var import VarData


class AssignmentTransaction:
    """Validate and apply one explicit native assignment transaction.

    Responsibilities:
    - Reject mismatched/invalid evidence before any setter is called.
    - Restore every attempted setter, including the failing setter, on error.

    Attributes:
        source (ExtractedLinearModel): Original owned source semantics.
        snapshot (CompiledLinearModel): Completed witness's numerical policy.
        evaluation (ModelEvaluation): Exact completed feasible assignment.
        targets (tuple): Stable-order compatible native variable handles.
    """

    def __init__(
        self,
        source: ExtractedLinearModel,
        snapshot: CompiledLinearModel,
        evaluation: ModelEvaluation,
        mapping: dict[str, Any],
    ) -> None:
        """Own ephemeral export context without touching native state.

        Args:
            source (ExtractedLinearModel): Original source semantics.
            snapshot (CompiledLinearModel): Completed numerical snapshot.
            evaluation (ModelEvaluation): Completed evidence selected for export.
            mapping (dict): Already source-validated native variable mapping.
        """
        self.source = source
        self.snapshot = snapshot
        self.evaluation = evaluation
        self.targets = tuple(mapping.values())
        self.previous: tuple[Any, ...] = ()

    def _validate(self) -> None:
        """Validate original source/objective and exact completed feasible evidence.

        Raises:
            ValueError: If source provenance, objective or completed evidence disagrees.
        """
        source, snapshot = self.source, self.snapshot
        if snapshot.variables != source.variables or snapshot.rows != source.rows:
            raise ValueError("Write-back snapshot does not match the extracted source.")
        expected = source.compile(
            search_bounds={
                domain.identifier: (domain.lower, domain.upper) for domain in snapshot.domains
            },
            scoring=snapshot.scoring,
            penalty_weight=snapshot.penalty_weight,
            absolute_tolerance=snapshot.absolute_tolerance,
            relative_tolerance=snapshot.relative_tolerance,
        )
        if (
            snapshot.maximize != source.maximize
            or snapshot._objective != expected._objective
            or snapshot._objective_constant != expected._objective_constant
        ):
            raise ValueError("Write-back snapshot objective does not match the extracted source.")
        if (
            not self.evaluation.feasible
            or snapshot.decode(np.asarray(self.evaluation.keys)) != self.evaluation
        ):
            raise ValueError(
                "Native write-back requires a validated feasible completed assignment."
            )

    def _get(self, target: object) -> float | None:
        """Read the supported frontend's current assignment slot.

        Args:
            target (object): Validated local native variable.

        Returns:
            float | None: Original native value, including Pyomo's unset value.
        """
        if self.source.frontend == "pyomo":
            return cast("VarData", target).value
        return cast("LpVariable", target).varValue

    def _set(self, target: object, value: float | None) -> None:
        """Set a supported native assignment without fixing or modifying bounds.

        Args:
            target (object): Validated local native variable.
            value (float | None): New or original assignment slot value.

        Raises:
            Exception: Native setter failures are propagated to transaction handling.
        """
        if self.source.frontend == "pyomo":
            cast("VarData", target).set_value(value)
        else:
            cast("LpVariable", target).varValue = value

    def apply(self) -> None:
        """Apply a validated feasible assignment with explicit rollback diagnostics.

        Raises:
            ValueError: If evidence or restorable prior PuLP values are invalid.
            RuntimeError: If a setter fails, including all rollback failures.
        """
        self._validate()
        self.previous = tuple(self._get(target) for target in self.targets)
        if self.source.frontend == "pulp" and any(
            value is None or not math.isfinite(value) for value in self.previous
        ):
            raise ValueError(
                "PuLP 4.2 write-back requires initialized finite prior values: unset values "
                "cannot be restored by its native API. Export assignments instead."
            )
        attempted = 0
        try:
            for target, value in zip(self.targets, self.evaluation.values, strict=True):
                attempted += 1
                self._set(target, value)
        except Exception as exc:
            failures = self._rollback(attempted)
            detail = (
                "rollback failed: " + "; ".join(failures) if failures else "prior values restored"
            )
            raise RuntimeError(f"Native assignment failed; {detail}.") from exc

    def _rollback(self, attempted: int) -> list[str]:
        """Restore attempted variables in reverse order without suppressing failures.

        Args:
            attempted (int): Number of setters attempted, including the failed setter.

        Returns:
            list[str]: Stable-ID diagnostics for every failed restoration.
        """
        failures = []
        for index in range(attempted - 1, -1, -1):
            try:
                self._set(self.targets[index], self.previous[index])
                restored = self._get(self.targets[index])
                if restored is not self.previous[index] and restored != self.previous[index]:
                    raise RuntimeError("native setter did not restore the prior value")
            except Exception as rollback:
                failures.append(f"variable {self.source.variables[index].identifier}: {rollback}")
        return failures
