"""Greedy violation-descent construction for model decoding (issue 125, follow-up).

Objective:
- Test whether unit moves on integer and binary coordinates that reduce total row violation raise
  the feasible fraction of random keys for any linear model, without naming fixtures or structure.

Technical approach:
- Map every coordinate with the default box mapping, as the shipped decoder does.
- If the assignment already violates no row, return it unchanged.
- Otherwise repeat sweeps: evaluate every +1/-1 move on a free non-continuous coordinate within its
  bounds, apply the single move that most lowers total violation, ties by variable index, and stop
  when no move lowers it or after a fixed sweep cap.
- Moves stay inside the effective domains (declared bounds intersected with search bounds).
- Continuous coordinates are left to the existing continuous restoration that the bound policy runs.

Flow (pseudocode):
1. Build the default assignment from keys.
2. While violation is positive and sweeps remain: pick the best improving unit move, apply it.
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
from rko_solver.modeling.construction import ConstructionInput
from rko_solver.modeling.construction import ModelConstruction

MAX_SWEEPS = 50
TOLERANCE = 1e-12


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
        total = 0.0
        for row in context.rows:
            activity = row.constant + sum(c * values[i] for i, c in row.coefficients)
            total += max(0.0, row.lower - activity) + max(0.0, activity - row.upper)
        return total

    @staticmethod
    def _descend(
        values: list[float], movable: list[int], context: ConstructionInput
    ) -> list[float]:
        """Apply the best improving unit move until violation stops falling or the cap is hit."""
        current = ViolationConstruction._violation(values, context)
        for _ in range(MAX_SWEEPS):
            if current <= TOLERANCE:
                break
            move = ViolationConstruction._best_move(values, movable, current, context)
            if move is None:
                break
            current, index, candidate = move
            values[index] = candidate
        return values

    @staticmethod
    def _best_move(
        values: list[float], movable: list[int], current: float, context: ConstructionInput
    ) -> tuple[float, int, float] | None:
        """Return the unit move with the lowest resulting violation, ties by variable index."""
        best: tuple[float, int, float] | None = None
        for index in movable:
            domain = context.domains[index]
            for step in (-1.0, 1.0):
                candidate = values[index] + step
                if not domain.lower <= candidate <= domain.upper:
                    continue
                trial = list(values)
                trial[index] = candidate
                score = ViolationConstruction._violation(trial, context)
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
