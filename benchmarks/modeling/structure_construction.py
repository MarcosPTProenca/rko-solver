"""Prototype of structure-aware construction for model decoding (issue 125, layer L1).

Objective:
- Test whether treating binary all-ones rows (assignment, cardinality, covering, packing) as
  structure raises the feasible fraction of random keys without changing the key dimension.

Technical approach:
- Detect rows whose variables are all free binaries with coefficient one.
- Read each key as a priority: variables are taken in decreasing key order (ties by index) when
  the key is at least one half and every row they touch stays within its upper bound; rows still
  below their lower bound are then topped up with the highest-priority variables that fit.
- Every other variable keeps the default box mapping; continuous restoration still runs after.

Flow (pseudocode):
1. Build the plan once per context: structured variables, their rows and the row bounds.
2. Map all variables with the default box mapping, zeroing structured binaries.
3. Select structured binaries by priority, then top up unmet lower bounds.

Important criteria:
- One key per free variable is kept, so the driver and `n_keys` are unchanged.
- Sorting is stable with an index tie-break, so decoding is deterministic for fixed keys.
- Rows with non-binary, fixed or non-unit terms are left to the default decoder and repair.

Inputs/Outputs:
- Input: keys and the frozen construction context.
- Output: a complete assignment in original variable order.
"""

from __future__ import annotations

import numpy as np
from .schema import PracticalCase
from rko_solver.modeling.construction import ConstructionInput
from rko_solver.modeling.construction import ModelConstruction
from rko_solver.modeling.linear import CompiledLinearModel
from rko_solver.modeling.records import LinearRow
from rko_solver.modeling.records import LinearVariable
from rko_solver.modeling.source import ExtractedLinearModel

THRESHOLD = 0.5


class StructurePlan:
    """Rows and variables the prototype constructs, derived once from a context.

    Responsibilities:
    - Select binary all-ones rows over free variables and index them by variable.

    Attributes:
        rows (list[tuple[list[int], float, float]]): Member indices, lower and upper bound per row.
        by_variable (dict[int, list[int]]): Row positions touching each structured variable.
    """

    def __init__(self, context: ConstructionInput) -> None:
        """Detect structured rows.

        Args:
            context (ConstructionInput): Frozen model data.
        """
        binary = [
            not d.fixed and v.domain != "continuous" and v.lower == 0 and v.upper == 1
            for v, d in zip(context.variables, context.domains, strict=True)
        ]
        self.rows: list[tuple[list[int], float, float]] = []
        self.by_variable: dict[int, list[int]] = {}
        for row in context.rows:
            if self._eligible(row, binary):
                self._add(row)

    def _add(self, row: LinearRow) -> None:
        """Index one eligible row by its member variables."""
        members = [index for index, _ in row.coefficients]
        for index in members:
            self.by_variable.setdefault(index, []).append(len(self.rows))
        self.rows.append((members, row.lower, row.upper))

    @staticmethod
    def _eligible(row: LinearRow, binary: list[bool]) -> bool:
        """Return whether every term is a free binary with coefficient one."""
        return bool(row.coefficients) and all(
            coefficient == 1 and binary[index] for index, coefficient in row.coefficients
        )


class StructureConstruction:
    """Callable construction that applies the plan on top of the default box mapping.

    Responsibilities:
    - Cache one plan per context and keep the decoder contract of `ConstructionFn`.
    """

    def __init__(self) -> None:
        """Start with no cached plan."""
        self._plan: StructurePlan | None = None
        self._context: ConstructionInput | None = None

    def __call__(self, keys: np.ndarray, context: ConstructionInput) -> list[float]:
        """Build a complete assignment from keys.

        Args:
            keys (np.ndarray): One key per free variable, in variable order.
            context (ConstructionInput): Frozen model data.

        Returns:
            list[float]: Values in original variable order.
        """
        if self._context is not context:
            self._context, self._plan = context, StructurePlan(context)
        assert self._plan is not None
        values, key_of = self._box(keys, context)
        structured = sorted(self._plan.by_variable)
        for index in structured:
            values[index] = 0.0
        order = sorted(structured, key=lambda i: (-key_of[i], i))
        activity = [0.0] * len(self._plan.rows)
        self._take(order, key_of, activity, values, threshold=THRESHOLD)
        self._top_up(order, key_of, activity, values)
        return values

    @staticmethod
    def _box(keys: np.ndarray, context: ConstructionInput) -> tuple[list[float], dict[int, float]]:
        """Apply the default box mapping and record each free variable's key."""
        values: list[float] = []
        key_of: dict[int, float] = {}
        position = 0
        for index, (variable, domain) in enumerate(
            zip(context.variables, context.domains, strict=True)
        ):
            if domain.fixed:
                values.append(variable.fixed if variable.fixed is not None else domain.lower)
                continue
            key = float(keys[position])
            position += 1
            key_of[index] = key
            width = domain.upper - domain.lower
            if variable.domain == "continuous":
                values.append(domain.lower + key * width)
            else:
                values.append(domain.lower + min(float(np.floor(key * (width + 1))), width))
        return values, key_of

    def _fits(self, index: int, activity: list[float]) -> bool:
        """Return whether adding one unit at `index` respects every touching upper bound."""
        assert self._plan is not None
        return all(
            activity[r] + 1 <= self._plan.rows[r][2] + 1e-9 for r in self._plan.by_variable[index]
        )

    def _set(self, index: int, activity: list[float], values: list[float]) -> None:
        """Select one structured variable and update row activity."""
        assert self._plan is not None
        values[index] = 1.0
        for r in self._plan.by_variable[index]:
            activity[r] += 1

    def _take(
        self,
        order: list[int],
        key_of: dict[int, float],
        activity: list[float],
        values: list[float],
        threshold: float,
    ) -> None:
        """Select variables in priority order while their rows stay within upper bounds."""
        for index in order:
            if key_of[index] >= threshold and values[index] == 0.0 and self._fits(index, activity):
                self._set(index, activity, values)

    def _top_up(
        self,
        order: list[int],
        key_of: dict[int, float],
        activity: list[float],
        values: list[float],
    ) -> None:
        """Raise rows below their lower bound using the highest-priority variables that fit."""
        assert self._plan is not None
        for position, (_, lower, _) in enumerate(self._plan.rows):
            if activity[position] < lower - 1e-9:
                self._fill(position, order, activity, values)

    def _fill(
        self,
        position: int,
        order: list[int],
        activity: list[float],
        values: list[float],
    ) -> None:
        """Add the highest-priority members that fit until one row reaches its lower bound."""
        assert self._plan is not None
        members, lower, _ = self._plan.rows[position]
        member_set = set(members)
        for index in (i for i in order if i in member_set):
            if values[index] == 0.0 and self._fits(index, activity):
                self._set(index, activity, values)
                if activity[position] >= lower - 1e-9:
                    return

    @staticmethod
    def bind(case: PracticalCase) -> ModelConstruction:
        """Bind the prototype to a fixture with continuous restoration.

        Args:
            case (PracticalCase): Fixture to decode.

        Returns:
            ModelConstruction: Decoder with `decode`, `cost` and `n_keys`.
        """
        source = ExtractedLinearModel(
            "mathopt",
            "bench",
            case.name,
            tuple(LinearVariable(v.name, v.name, v.lower, v.upper, v.kind) for v in case.variables),
            case.objective,
            0.0,
            case.maximize,
            tuple(LinearRow(r.name, r.coefficients, r.lower, r.upper) for r in case.rows),
        )
        snapshot: CompiledLinearModel = source.compile(repair="continuous")
        return ModelConstruction.bind(snapshot, source, StructureConstruction())
