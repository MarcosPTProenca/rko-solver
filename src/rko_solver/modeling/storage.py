"""Owned sparse linear storage.

Objective:
- Own immutable numerical buffers and authoritative original-unit checks.

Technical approach:
- Keep validated state portable and responsibilities within cohesive classes.

Flow (pseudocode):
1. Receive owned model data.
2. Perform this module's bounded numerical responsibility.
3. Return explicit values or immutable evidence.

Important criteria:
- Preserve original domains, tolerances and finite-value checks.

Inputs/Outputs:
- Input: owned bounded linear-model data.
- Output: validated state or portable numerical evidence.
"""

from __future__ import annotations

import math
import numpy as np
from .records import EffectiveDomain
from .records import LinearRow
from .records import LinearVariable
from dataclasses import dataclass
from numbers import Integral
from numbers import Real
from typing import Literal

_ENVELOPE = 1e100
_INTEGER_LIMIT = 2**52


@dataclass(frozen=True, slots=True, init=False)
class LinearStorage:
    """Own frozen sparse buffers and scalar validation.

    Responsibilities:
    - Keep numerical model state immutable and portable.
    - Check original-unit row arithmetic without objective replay.

    Attributes:
        variables (tuple[LinearVariable, ...]): Original ordered variable identities.
        rows (tuple[LinearRow, ...]): Original affine constraint rows.
    """

    variables: tuple[LinearVariable, ...]
    rows: tuple[LinearRow, ...]
    domains: tuple[EffectiveDomain, ...]
    maximize: bool
    scoring: Literal["feasibility_first", "weighted_penalty"]
    penalty_weight: float | None
    absolute_tolerance: float
    relative_tolerance: float
    repair: Literal["none", "continuous"]
    repair_max_sweeps: int
    _repair_free: bytes
    _repair_lower: bytes
    _repair_upper: bytes
    _repair_width: bytes
    _repair_indptr: bytes
    _repair_columns: bytes
    _repair_coefficients: bytes
    _repair_offsets: bytes
    _repair_scales: bytes
    _repair_norms: bytes
    _repair_frozen_entries: bytes
    _free: bytes
    _base: bytes
    _lower: bytes
    _upper: bytes
    _integer: bytes
    _objective: bytes
    _objective_constant: float
    _objective_lower: float
    _objective_upper: float
    _row_indices: bytes
    _columns: bytes
    _coefficients: bytes
    _row_constants: bytes
    _row_lower: bytes
    _row_upper: bytes
    _lower_tolerance: bytes
    _upper_tolerance: bytes
    _row_scales: bytes

    @staticmethod
    def _number(value: object, context: str) -> float:
        """Validate finite float64 scalars without boolean or lossy integer coercion.

        Args:
            value (object): Extracted scalar to validate.
            context (str): Diagnostic field description.

        Returns:
            float: Finite float64-safe scalar.

        Raises:
            ValueError: If input is not finite real data in the supported envelope.
        """
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{context}: expected a finite real number.")
        try:
            result = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{context}: unsupported numerical range.") from exc
        LinearStorage._safe(abs(result), context)
        if isinstance(value, Integral) and result != int(value):
            raise ValueError(f"{context}: integer data are not exactly representable in float64.")
        return result

    @staticmethod
    def _safe(magnitude: float, context: str) -> float:
        """Reject conservative accumulated arithmetic outside the numerical envelope.

        Args:
            magnitude (float): Nonnegative worst-case magnitude.
            context (str): Diagnostic field description.

        Returns:
            float: Validated magnitude, at most 1e100.

        Raises:
            ValueError: If arithmetic is nonfinite or outside the supported range.
        """
        if not math.isfinite(magnitude) or magnitude > _ENVELOPE:
            raise ValueError(f"{context}: exceeds the supported float64 envelope (1e100).")
        return magnitude

    @staticmethod
    def _bounds(lower: object, upper: object, context: str) -> tuple[float, float]:
        """Validate ordered bounds while allowing only correctly oriented absent sides.

        Args:
            lower (object): Lower endpoint or negative infinity.
            upper (object): Upper endpoint or positive infinity.
            context (str): Row or variable diagnostic identity.

        Returns:
            tuple[float, float]: Ordered non-NaN endpoints.

        Raises:
            ValueError: If a bound is NaN, malformed, reversed or numerically unsafe.
        """
        lb = (
            -math.inf
            if isinstance(lower, Real) and lower == -math.inf
            else (LinearStorage._number(lower, f"{context} lower bound"))
        )
        ub = (
            math.inf
            if isinstance(upper, Real) and upper == math.inf
            else (LinearStorage._number(upper, f"{context} upper bound"))
        )
        if lb > ub:
            raise ValueError(f"{context}: lower bound exceeds upper bound.")
        return lb, ub

    @staticmethod
    def _identities(identifiers: list[str], context: str) -> None:
        """Check stable identities before constructing any export or reconstruction map.

        Args:
            identifiers (list[str]): Extracted stable row or variable identifiers.
            context (str): Identity namespace for diagnostics.

        Raises:
            ValueError: If an identity is empty, non-string or duplicated.
        """
        if any(not isinstance(identity, str) or not identity for identity in identifiers):
            raise ValueError(f"Every {context} requires a nonempty stable string identifier.")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Duplicate stable {context} identifiers are not supported.")

    def _row_state(self, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Evaluate authoritative original-unit activities and side residuals.

        Args:
            values (np.ndarray): Complete physical assignments for this invocation.

        Returns:
            tuple[np.ndarray, ...]: Activities, raw residuals and adjusted violations.

        Raises:
            ValueError: If original-unit row arithmetic is nonfinite.
        """
        activities = np.frombuffer(self._row_constants) + np.bincount(
            np.frombuffer(self._row_indices, dtype=np.intp),
            weights=np.frombuffer(self._coefficients)
            * values[np.frombuffer(self._columns, dtype=np.intp)],
            minlength=len(self.rows),
        )
        if not np.all(np.isfinite(activities)):
            raise ValueError("Model evaluation produced nonfinite row arithmetic.")
        lower = np.maximum(0.0, np.frombuffer(self._row_lower) - activities)
        upper = np.maximum(0.0, activities - np.frombuffer(self._row_upper))
        return (
            activities,
            lower,
            upper,
            np.maximum(0.0, lower - np.frombuffer(self._lower_tolerance)),
            np.maximum(0.0, upper - np.frombuffer(self._upper_tolerance)),
        )
