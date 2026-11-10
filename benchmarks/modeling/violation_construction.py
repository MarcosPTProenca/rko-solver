"""Greedy violation-descent construction for model decoding (issue 125, follow-up).

Objective:
- Test whether unit moves on integer and binary coordinates that reduce total row violation raise
  the feasible fraction of random keys for any linear model, without naming fixtures or structure.

Technical approach:
- Map every coordinate with the default box mapping, as the shipped decoder does.
- If the assignment already violates no row, return it unchanged.
- Otherwise cache each row activity and violation, and repeat sweeps: score every +1/-1 move on a
  free non-continuous coordinate by updating only the rows it touches, apply the single move that
  most lowers total violation (ties by variable index), and stop when no move lowers it or after a
  fixed sweep cap.
- Moves stay inside the effective domains (declared bounds intersected with search bounds).
- Continuous coordinates are left to the existing continuous restoration that the bound policy runs.

Flow (pseudocode):
1. Build the default assignment from keys and cache row activities, violations and incidence.
2. While violation is positive and sweeps remain: score incident rows of each move, apply the best.
3. Return the assignment; feasibility is decided later by the original evaluator.

Important criteria:
- Row constants are included, so affine offsets are not ignored.
- The descent uses only rows and bounds; no problem-specific rules.
- Determinism: a fixed key vector always yields the same moves.
- Not a feasibility guarantee; failure to repair is reported as infeasible by the evaluator.

Inputs/Outputs:
- Input: keys and the frozen construction context.
- Output: an assignment in original variable order.
"""

from __future__ import annotations

import numpy as np
from .cases import PracticalCase
from .structure_construction import StructureConstruction
from dataclasses import dataclass
from rko_solver.modeling.construction import ConstructionInput
from rko_solver.modeling.construction import ModelConstruction
from rko_solver.modeling.records import LinearRow

MAX_SWEEPS = 50
TOLERANCE = 1e-12


@dataclass
class _RowState:
    """Cached row activities, violations and sparse variable-to-row incidence.

    Responsibilities:
    - Keep each row's activity and violation consistent with the current assignment.

    Attributes:
        activities (list[float]): Affine activity of each row, including its constant.
        violations (list[float]): Distance of each activity outside its row interval.
        incidence (list[list[tuple[int, float]]]): Row index and coefficient per variable.
    """

    activities: list[float]
    violations: list[float]
    incidence: list[list[tuple[int, float]]]


class ViolationConstruction:
    """Callable construction applying greedy unit moves that lower total row violation.

    Responsibilities:
    - Keep the decoder contract of `ConstructionFn` and map keys to original-order values.
    """

    def __call__(self, keys: np.ndarray, context: ConstructionInput) -> list[float]:
        """Build an assignment and descend on violation when it is infeasible.

        Args:
            keys (np.ndarray): One key per free variable, in variable order.
            context (ConstructionInput): Frozen model data.

        Returns:
            list[float]: Values in original variable order.
        """
        values, _ = StructureConstruction._box(keys, context)
        movable = [
            i
            for i, (v, d) in enumerate(zip(context.variables, context.domains, strict=True))
            if not d.fixed and v.domain != "continuous"
        ]
        return ViolationConstruction._descend(values, movable, context)

    @staticmethod
    def _violation(values: list[float], context: ConstructionInput) -> float:
        """Sum row violations of an assignment, including each row's affine constant."""
        return sum(
            ViolationConstruction._row_violation(
                row.constant + sum(c * values[i] for i, c in row.coefficients), row
            )
            for row in context.rows
        )

    @staticmethod
    def _state(values: list[float], context: ConstructionInput) -> _RowState:
        """Build cached activities, violations and incidence for an assignment."""
        activities = [
            row.constant + sum(c * values[i] for i, c in row.coefficients) for row in context.rows
        ]
        violations = [
            ViolationConstruction._row_violation(a, row)
            for a, row in zip(activities, context.rows, strict=True)
        ]
        incidence: list[list[tuple[int, float]]] = [[] for _ in values]
        for row_index, row in enumerate(context.rows):
            for index, coefficient in row.coefficients:
                incidence[index].append((row_index, coefficient))
        return _RowState(activities, violations, incidence)

    @staticmethod
    def _descend(
        values: list[float], movable: list[int], context: ConstructionInput
    ) -> list[float]:
        """Apply best improving moves until violation stops falling or the sweep cap is hit."""
        state = ViolationConstruction._state(values, context)
        current = sum(state.violations)
        for _ in range(MAX_SWEEPS):
            if current <= TOLERANCE:
                break
            move = ViolationConstruction._best_move(values, movable, current, context, state)
            if move is None:
                break
            current, index, candidate = move
            ViolationConstruction._apply(values, index, candidate, context, state)
        return values

    @staticmethod
    def _apply(
        values: list[float],
        index: int,
        candidate: float,
        context: ConstructionInput,
        state: _RowState,
    ) -> None:
        """Set one coordinate and refresh only the rows it touches."""
        step = candidate - values[index]
        values[index] = candidate
        for row_index, coefficient in state.incidence[index]:
            state.activities[row_index] += coefficient * step
            state.violations[row_index] = ViolationConstruction._row_violation(
                state.activities[row_index], context.rows[row_index]
            )

    @staticmethod
    def _row_violation(activity: float, row: LinearRow) -> float:
        """Return one row's distance outside its feasible activity interval."""
        return max(0.0, row.lower - activity) + max(0.0, activity - row.upper)

    @staticmethod
    def _move_score(
        index: int, step: float, current: float, context: ConstructionInput, state: _RowState
    ) -> float:
        """Return total violation after one unit move, rescoring only incident rows."""
        score = current
        for row_index, coefficient in state.incidence[index]:
            score -= state.violations[row_index]
            score += ViolationConstruction._row_violation(
                state.activities[row_index] + coefficient * step, context.rows[row_index]
            )
        return score

    @staticmethod
    def _best_move(
        values: list[float],
        movable: list[int],
        current: float,
        context: ConstructionInput,
        state: _RowState,
    ) -> tuple[float, int, float] | None:
        """Return the unit move with the lowest resulting violation, ties by variable index."""
        best: tuple[float, int, float] | None = None
        for index in movable:
            domain = context.domains[index]
            for step in (-1.0, 1.0):
                candidate = values[index] + step
                if not domain.lower <= candidate <= domain.upper:
                    continue
                score = ViolationConstruction._move_score(index, step, current, context, state)
                if score < current - TOLERANCE and (best is None or score < best[0]):
                    best = (score, index, candidate)
        return best

    @staticmethod
    def bind(case: PracticalCase) -> ModelConstruction:
        """Bind the violation descent to a fixture with continuous restoration.

        Args:
            case (PracticalCase): Fixture to decode.

        Returns:
            ModelConstruction: Decoder with `decode`, `cost` and `n_keys`.
        """
        return StructureConstruction.bind(case, ViolationConstruction())
