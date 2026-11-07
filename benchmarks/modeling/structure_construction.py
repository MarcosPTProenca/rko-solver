"""Prototype of structure-aware construction for model decoding (issue 125, layer L1).

Objective:
- Test whether treating rows over free binary variables as structure raises the feasible fraction
  of random keys for any model, without changing the key dimension and without problem-specific
  rules (no assumption of assignment, knapsack or routing shape).

Technical approach:
- Detect rows whose variables are all free binaries, with any nonzero coefficient.
- Read each key as a priority: variables are taken in decreasing key order (ties by index) when the
  key is at least one half and the row activities stay within their upper bounds. A variable whose
  row would overflow may pull in one unselected variable with a negative coefficient in that row
  (an implication such as `x - y <= 0`); if that cannot restore the row the choice is undone.
- Rows still below their lower bound are then topped up with the highest-priority variables that
  fit. Every other variable keeps the default box mapping; continuous restoration runs afterwards.

Flow (pseudocode):
1. Build the plan once per context: structured variables, their rows and coefficients.
2. Map all variables with the default box mapping, zeroing structured binaries.
3. Select structured binaries by priority with one-step implications, then top up lower bounds.

Important criteria:
- One key per free variable is kept, so the driver and `n_keys` are unchanged.
- Sorting is stable with an index tie-break, so decoding is deterministic for fixed keys.
- Rows with non-binary, fixed or integer-range terms are left to the default decoder and repair.
- Implications are followed one level only; deeper chains are rejected, never guessed.

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
TOLERANCE = 1e-9


class StructurePlan:
    """Rows and variables the prototype constructs, derived once from a context.

    Responsibilities:
    - Select rows over free binaries and index them by variable with their coefficients.

    Attributes:
        rows (list[tuple[list[tuple[int, float]], float, float]]): Terms, lower and upper bound.
        by_variable (dict[int, list[tuple[int, float]]]): Row position and coefficient per variable.
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
        self.rows: list[tuple[list[tuple[int, float]], float, float]] = []
        self.by_variable: dict[int, list[tuple[int, float]]] = {}
        for row in context.rows:
            if self._eligible(row, binary):
                self._add(row)

    def _add(self, row: LinearRow) -> None:
        """Index one eligible row by its member variables."""
        terms = [(index, float(coefficient)) for index, coefficient in row.coefficients]
        for index, coefficient in terms:
            self.by_variable.setdefault(index, []).append((len(self.rows), coefficient))
        self.rows.append((terms, row.lower - row.constant, row.upper - row.constant))

    @staticmethod
    def _eligible(row: LinearRow, binary: list[bool]) -> bool:
        """Return whether every term is a free binary with a nonzero coefficient."""
        return bool(row.coefficients) and all(
            coefficient != 0 and binary[index] for index, coefficient in row.coefficients
        )


class _Selection:
    """One decoding in progress: selected binaries and per-row activity.

    Responsibilities:
    - Apply and undo selections, check row bounds and follow one-step implications.
    """

    def __init__(
        self, plan: StructurePlan, values: list[float], order: list[int], key_of: dict[int, float]
    ) -> None:
        """Start with nothing selected.

        Args:
            plan (StructurePlan): Detected rows.
            values (list[float]): Assignment being built; structured binaries are zero.
            order (list[int]): Structured variables by decreasing key, ties by index.
            key_of (dict[int, float]): Key of every free variable.
        """
        self.plan, self.values, self.order, self.key_of = plan, values, order, key_of
        self.activity = [0.0] * len(plan.rows)
        self.rank = {index: position for position, index in enumerate(order)}

    def _apply(self, index: int, sign: float = 1.0) -> None:
        """Select (`sign` 1) or unselect (`sign` -1) one variable and update activities."""
        self.values[index] = 1.0 if sign > 0 else 0.0
        for row, coefficient in self.plan.by_variable[index]:
            self.activity[row] += sign * coefficient

    def _allowed(self, row: int, coefficient: float) -> bool:
        """Return whether adding one coefficient keeps a row from leaving its bounds."""
        _, lower, upper = self.plan.rows[row]
        activity = self.activity[row]
        if coefficient > 0:
            return activity + coefficient <= upper + TOLERANCE
        return activity + coefficient >= lower - TOLERANCE or activity > upper + TOLERANCE

    def _fits(self, index: int) -> bool:
        """Return whether one variable can be added without leaving any touching row's bounds."""
        return all(self._allowed(row, c) for row, c in self.plan.by_variable[index])

    def _overfull(self, index: int) -> list[int]:
        """Return touching rows whose activity exceeds the upper bound."""
        return [
            row
            for row, _ in self.plan.by_variable[index]
            if self.activity[row] > self.plan.rows[row][2] + TOLERANCE
        ]

    def _helper(self, row: int) -> int | None:
        """Return the best unselected variable with a negative coefficient that fits, if any."""
        terms = self.plan.rows[row][0]
        candidates = [i for i, c in terms if c < 0 and self.values[i] == 0.0 and self._fits(i)]
        return min(candidates, key=self.rank.__getitem__, default=None)

    def try_set(self, index: int) -> bool:
        """Select a variable, pulling in one helper per overfull row; undo everything on failure.

        Args:
            index (int): Variable to select.

        Returns:
            bool: Whether the variable (and any helpers) are now selected.
        """
        if self._breaks_lower(index):
            return False
        added = [index]
        self._apply(index)
        for row in self._overfull(index):
            helper = self._helper(row)
            if helper is None:
                break
            self._apply(helper)
            added.append(helper)
        if self._overfull(index):
            self._undo(added)
            return False
        return True

    def _breaks_lower(self, index: int) -> bool:
        """Return whether a negative coefficient would push a row below its lower bound."""
        return any(c < 0 and not self._allowed(r, c) for r, c in self.plan.by_variable[index])

    def _undo(self, added: list[int]) -> None:
        """Unselect variables in reverse order of selection."""
        for variable in reversed(added):
            self._apply(variable, -1.0)

    def take(self) -> None:
        """Select variables in priority order when their key is at least the threshold."""
        for index in self.order:
            if self.key_of[index] >= THRESHOLD and self.values[index] == 0.0:
                self.try_set(index)

    def top_up(self) -> None:
        """Raise rows below their lower bound using the highest-priority variables that fit."""
        for row, (terms, lower, _) in enumerate(self.plan.rows):
            if self.activity[row] < lower - TOLERANCE:
                self._fill(row, {i for i, c in terms if c > 0}, lower)

    def _fill(self, row: int, positive: set[int], lower: float) -> None:
        """Add positive-coefficient members, highest priority first, until a row is met."""
        for index in (i for i in self.order if i in positive):
            if self.values[index] == 0.0 and self.try_set(index):
                if self.activity[row] >= lower - TOLERANCE:
                    return


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
        selection = _Selection(self._plan, values, order, key_of)
        selection.take()
        selection.top_up()
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
