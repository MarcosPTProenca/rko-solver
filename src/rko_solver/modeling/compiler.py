"""Strict sparse compilation of owned bounded linear model data.

Objective:
- Validate domains and conservative arithmetic before exposing numerical callbacks.

Technical approach:
- Compile variable domains, objective bounds and sparse affine rows separately.
- Fold fixed effects and freeze every resulting array as owned bytes.

Flow (pseudocode):
1. Validate identities, policy and effective variable domains.
2. Compile objective bounds and original-unit sparse row data.
3. Freeze validated metadata and arrays on the snapshot.

Important criteria:
- No inferred coupled bounds, solver calls, relaxed domains or omitted coefficients.

Inputs/Outputs:
- Input: original records, objective coefficients and typed numerical options.
- Output: immutable validated numerical storage.
"""

from __future__ import annotations

import math
import numpy as np
from .options import LinearOptions
from .options import LinearPolicy
from .records import EffectiveDomain
from .records import LinearRow
from .records import LinearVariable
from .storage import _INTEGER_LIMIT
from .storage import LinearStorage
from typing import Any


class LinearCompiler:
    """Compile one immutable sparse model without native frontend references.

    Responsibilities:
    - Validate original identities, domains, affine data and numerical envelopes.
    - Publish only complete validated state.

    Attributes:
        model (LinearStorage): Snapshot receiving immutable state.
        policy (LinearPolicy): Normalized numerical settings.
        buffers (dict): Invocation-owned arrays until final freezing.
    """

    def __init__(
        self,
        model: LinearStorage,
        variables: tuple[LinearVariable, ...],
        objective: tuple[float, ...],
        rows: tuple[LinearRow, ...],
        options: LinearOptions,
    ) -> None:
        """Own compiler inputs and validate settings.

        Args:
            model (LinearStorage): Target snapshot under construction.
            variables (tuple): Original ordered variable records.
            objective (tuple): Original objective coefficients.
            rows (tuple): Original affine rows.
            options (LinearOptions): Accepted numerical keyword settings.

        Raises:
            ValueError: If policy settings are invalid.
            TypeError: If an unknown compiler setting is supplied.
        """
        self.model = model
        self.variables = tuple(variables)
        self.objective = tuple(objective)
        self.rows = tuple(rows)
        self.policy = LinearPolicy(**options).normalized()
        self.buffers: dict[str, list[Any]] = {}
        self.domains: list[EffectiveDomain] = []
        self.penalty_bounds: list[float] = []

    def apply(self) -> None:
        """Compile complete state and freeze it after all preflight checks pass.

        Raises:
            ValueError: If any original data, domain or accumulated arithmetic is unsafe.
        """
        self._identities()
        self._variables()
        objective = self._objective()
        self._rows()
        self._penalty(objective)
        fields = {
            "variables": self.variables,
            "rows": self.rows,
            "domains": tuple(self.domains),
            "maximize": self.policy.maximize,
            "scoring": self.policy.scoring,
            "penalty_weight": self.policy.penalty_weight,
            "absolute_tolerance": self.policy.absolute_tolerance,
            "relative_tolerance": self.policy.relative_tolerance,
            "repair": self.policy.repair,
            "repair_max_sweeps": self.policy.repair_max_sweeps,
            **objective,
        }
        for name, value in fields.items():
            object.__setattr__(self.model, name, value)
        indices = {"_free", "_row_indices", "_columns"}
        for name, values in self.buffers.items():
            dtype = np.intp if name in indices else bool if name == "_integer" else float
            object.__setattr__(self.model, name, np.asarray(values, dtype=dtype).tobytes())

    def _identities(self) -> None:
        """Validate record types, identities and original objective dimension.

        Raises:
            ValueError: If records or identities are malformed or duplicated.
        """
        if not all(isinstance(v, LinearVariable) for v in self.variables):
            raise ValueError("variables must contain LinearVariable records.")
        if not all(isinstance(row, LinearRow) for row in self.rows):
            raise ValueError("rows must contain LinearRow records.")
        self.rows = tuple(
            LinearRow(
                row.identifier,
                tuple((c, v) for c, v in row.coefficients),
                row.lower,
                row.upper,
                row.constant,
            )
            for row in self.rows
        )
        LinearStorage._identities([v.identifier for v in self.variables], "variable")
        LinearStorage._identities([row.identifier for row in self.rows], "row")
        if len(self.objective) != len(self.variables):
            raise ValueError("The objective must have one coefficient per variable.")

    def _variables(self) -> None:
        """Compile restrictive domains and key coordinates without coupled inference.

        Raises:
            ValueError: If an effective domain or override is invalid.
        """
        overrides = dict(self.policy.search_bounds or {})
        if set(overrides).difference(v.identifier for v in self.variables):
            raise ValueError("search_bounds contains unknown stable variable identifiers.")
        self.buffers.update(
            {name: [] for name in ("_free", "_base", "_lower", "_upper", "_integer")}
        )
        for index, variable in enumerate(self.variables):
            declared = self._declared(variable)
            lower, upper = self._override(variable, declared, overrides)
            restricted = (lower, upper) != declared
            lower, upper = self._fixed(variable, lower, upper)
            self._domain_range(variable, lower, upper)
            fixed = lower == upper
            self.domains.append(
                EffectiveDomain(variable.identifier, lower, upper, restricted, fixed)
            )
            self.buffers["_base"].append(lower if fixed else 0.0)
            if not fixed:
                self.buffers["_free"].append(index)
                self.buffers["_lower"].append(lower)
                self.buffers["_upper"].append(upper)
                self.buffers["_integer"].append(variable.domain != "continuous")

    @staticmethod
    def _declared(variable: LinearVariable) -> tuple[float, float]:
        """Validate declared bounds and intersect binary/integer domains.

        Args:
            variable (LinearVariable): Original domain record.

        Returns:
            tuple[float, float]: Declared effective endpoints before overrides.

        Raises:
            ValueError: If label, domain or bounds are invalid.
        """
        name = f"variable {variable.identifier!r}"
        if not isinstance(variable.label, str):
            raise ValueError(f"{name}: label must be a string.")
        if variable.domain not in ("continuous", "integer", "binary"):
            raise ValueError(f"{name}: unsupported domain {variable.domain!r}.")
        lower, upper = LinearStorage._bounds(variable.lower, variable.upper, name)
        if variable.domain == "binary":
            lower, upper = max(lower, 0.0), min(upper, 1.0)
        return LinearCompiler._integer_bounds(variable, lower, upper)

    @staticmethod
    def _integer_bounds(
        variable: LinearVariable, lower: float, upper: float
    ) -> tuple[float, float]:
        """Round only integer domain endpoints inward.

        Args:
            variable (LinearVariable): Original variable kind.
            lower (float): Candidate lower bound.
            upper (float): Candidate upper bound.

        Returns:
            tuple[float, float]: Inward integer endpoints or unchanged continuous bounds.
        """
        if variable.domain != "continuous":
            lower = float(math.ceil(lower)) if math.isfinite(lower) else lower
            upper = float(math.floor(upper)) if math.isfinite(upper) else upper
        return lower, upper

    @staticmethod
    def _override(
        variable: LinearVariable, bounds: tuple[float, float], overrides: dict
    ) -> tuple[float, float]:
        """Intersect caller endpoints without relaxing the declared domain.

        Args:
            variable (LinearVariable): Original variable identity/kind.
            bounds (tuple): Declared effective endpoints.
            overrides (dict): Restrictive stable-ID bound overrides.

        Returns:
            tuple[float, float]: Restricted endpoints.

        Raises:
            ValueError: If override shape or endpoints are invalid.
        """
        lower, upper = bounds
        if variable.identifier in overrides:
            override = overrides[variable.identifier]
            if not isinstance(override, (tuple, list)) or len(override) != 2:
                raise ValueError(
                    f"variable {variable.identifier!r}: search bounds must contain two endpoints."
                )
            a, b = override
            if a is not None:
                lower = max(
                    lower,
                    LinearStorage._number(
                        a, f"variable {variable.identifier!r} search lower bound"
                    ),
                )
            if b is not None:
                upper = min(
                    upper,
                    LinearStorage._number(
                        b, f"variable {variable.identifier!r} search upper bound"
                    ),
                )
            lower, upper = LinearCompiler._integer_bounds(variable, lower, upper)
        if lower > upper:
            raise ValueError(f"variable {variable.identifier!r}: empty effective domain.")
        return lower, upper

    @staticmethod
    def _fixed(variable: LinearVariable, lower: float, upper: float) -> tuple[float, float]:
        """Validate fixed assignments against the effective exact domain.

        Args:
            variable (LinearVariable): Original fixed-value record.
            lower (float): Restricted lower endpoint.
            upper (float): Restricted upper endpoint.

        Returns:
            tuple[float, float]: Singleton endpoints or unchanged bounds.

        Raises:
            ValueError: If a fixed assignment violates bounds or integrality.
        """
        if variable.fixed is not None:
            fixed = LinearStorage._number(
                variable.fixed, f"variable {variable.identifier!r} fixed value"
            )
            integral = variable.domain == "continuous" or fixed.is_integer()
            if not lower <= fixed <= upper or not integral:
                raise ValueError(
                    f"variable {variable.identifier!r}: fixed value violates the effective domain."
                )
            return fixed, fixed
        return lower, upper

    @staticmethod
    def _domain_range(variable: LinearVariable, lower: float, upper: float) -> None:
        """Reject unsupported free ranges before mapping or allocation.

        Args:
            variable (LinearVariable): Original variable kind/identity.
            lower (float): Effective lower endpoint.
            upper (float): Effective upper endpoint.

        Raises:
            ValueError: If a range is unbounded or exceeds the numerical envelope.
        """
        name = f"variable {variable.identifier!r}"
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError(f"{name}: finite effective bounds required; supply search_bounds.")
        if (
            variable.domain != "continuous"
            and max(abs(lower), abs(upper), upper - lower + 1) > _INTEGER_LIMIT
        ):
            raise ValueError(f"{name}: integer domain exceeds the float64-safe envelope.")
        LinearStorage._safe(max(abs(lower), abs(upper), upper - lower), name)

    def _objective(self) -> dict[str, Any]:
        """Compile original objective bounds and folded fixed effects.

        Returns:
            dict[str, Any]: Folded constant and original-box objective endpoints.

        Raises:
            ValueError: If coefficients or accumulated objective bounds are unsafe.
        """
        values = [LinearStorage._number(v, "objective coefficient") for v in self.objective]
        constant = LinearStorage._number(self.policy.objective_constant, "objective constant")
        magnitudes = [
            abs(c) * max(abs(d.lower), abs(d.upper))
            for c, d in zip(values, self.domains, strict=True)
        ]
        LinearStorage._safe(math.fsum([abs(constant), *magnitudes]), "objective accumulation")
        lower = math.fsum(
            [
                constant,
                *[min(c * d.lower, c * d.upper) for c, d in zip(values, self.domains, strict=True)],
            ]
        )
        upper = math.fsum(
            [
                constant,
                *[max(c * d.lower, c * d.upper) for c, d in zip(values, self.domains, strict=True)],
            ]
        )
        folded = math.fsum(
            [constant, *[c * v for c, v in zip(values, self.buffers["_base"], strict=True)]]
        )
        self.buffers["_objective"] = [values[index] for index in self.buffers["_free"]]
        return {"_objective_constant": folded, "_objective_lower": lower, "_objective_upper": upper}

    def _rows(self) -> None:
        """Compile sparse rows and separate finite-side tolerances.

        Raises:
            ValueError: If any row's original data or accumulated arithmetic is unsafe.
        """
        names = (
            "_row_indices",
            "_columns",
            "_coefficients",
            "_row_constants",
            "_row_lower",
            "_row_upper",
            "_lower_tolerance",
            "_upper_tolerance",
            "_row_scales",
        )
        self.buffers.update({name: [] for name in names})
        for index, row in enumerate(self.rows):
            self._row(index, row)

    def _row(self, index: int, row: LinearRow) -> None:
        """Compile one affine row after validating every coefficient.

        Args:
            index (int): Original row position.
            row (LinearRow): Original row record.

        Raises:
            ValueError: If bounds, coefficients or allowances are invalid.
        """
        name = f"row {row.identifier!r}"
        lower, upper = LinearStorage._bounds(row.lower, row.upper, name)
        constant = LinearStorage._number(row.constant, f"{name} constant")
        terms, magnitudes = self._terms(index, row, constant)
        magnitude = LinearStorage._safe(math.fsum(magnitudes), f"{name} accumulation")
        self.buffers["_row_constants"].append(math.fsum(terms))
        self.buffers["_row_lower"].append(lower)
        self.buffers["_row_upper"].append(upper)
        self.buffers["_row_scales"].append(
            max(
                1.0,
                magnitude,
                abs(lower) if math.isfinite(lower) else 0.0,
                abs(upper) if math.isfinite(upper) else 0.0,
            )
        )
        self._tolerance(lower, magnitude, "_lower_tolerance", name)
        self._tolerance(upper, magnitude, "_upper_tolerance", name)

    def _terms(
        self, index: int, row: LinearRow, constant: float
    ) -> tuple[list[float], list[float]]:
        """Validate sparse supports and fold only declared fixed terms.

        Args:
            index (int): Original row index.
            row (LinearRow): Original sparse affine record.
            constant (float): Original affine constant.

        Returns:
            tuple: Folded terms and conservative absolute magnitudes.

        Raises:
            ValueError: If an index/coefficient is duplicated, malformed or unsafe.
        """
        terms, magnitudes = [constant], [abs(constant)]
        seen: set[int] = set()
        for column, raw in row.coefficients:
            valid = type(column) is int and 0 <= column < len(self.variables)
            if not valid or column in seen:
                raise ValueError(
                    f"row {row.identifier!r}: invalid or duplicate variable index {column!r}."
                )
            seen.add(column)
            coefficient = LinearStorage._number(raw, f"row {row.identifier!r} coefficient")
            domain = self.domains[column]
            magnitudes.append(abs(coefficient) * max(abs(domain.lower), abs(domain.upper)))
            if domain.fixed:
                terms.append(coefficient * self.buffers["_base"][column])
            elif coefficient != 0:
                self.buffers["_row_indices"].append(index)
                self.buffers["_columns"].append(column)
                self.buffers["_coefficients"].append(coefficient)
        return terms, magnitudes

    def _tolerance(self, bound: float, magnitude: float, target: str, name: str) -> None:
        """Compile the unchanged absolute/relative allowance of one finite row side.

        Args:
            bound (float): Original finite or absent row side.
            magnitude (float): Conservative row accumulation magnitude.
            target (str): Lower or upper tolerance buffer.
            name (str): Original row diagnostic context.

        Raises:
            ValueError: If the allowance exceeds the supported envelope.
        """
        tolerance = 0.0
        if math.isfinite(bound):
            tolerance = LinearStorage._safe(
                self.policy.absolute_tolerance + self.policy.relative_tolerance * abs(bound),
                f"{name} tolerance",
            )
            if self.policy.scoring == "weighted_penalty":
                residual = magnitude + abs(bound)
                self.penalty_bounds.append(residual * residual if residual > tolerance else 0.0)
        self.buffers[target].append(tolerance)

    def _penalty(self, objective: dict[str, Any]) -> None:
        """Validate worst-case weighted arithmetic before publishing callbacks.

        Args:
            objective (dict): Original finite-box objective endpoints.

        Raises:
            ValueError: If the selected finite penalty can overflow its safe envelope.
        """
        if self.policy.scoring == "weighted_penalty":
            assert self.policy.penalty_weight is not None
            LinearStorage._safe(
                max(abs(objective["_objective_lower"]), abs(objective["_objective_upper"]))
                + self.policy.penalty_weight * math.fsum(self.penalty_bounds),
                "weighted-penalty worst-case score; reduce scales, search bounds or weight",
            )
