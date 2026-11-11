"""Generated hard mixed-binary models with unknown optima, for solver-control measurements.

Objective:
- Provide instances where a plain MIP solver is slow, so completion or repair can be judged on
  models that are hard in the solver's own terms, not only on easy algebraic families.

Technical approach:
- Multi-dimensional 0/1 knapsack with strongly correlated profits: each profit is the mean row
  weight plus a constant, the classic construction that weakens the linear relaxation.
- Capacities are half the total weight of each row, so the optimum is non-trivial.
- No optimum is claimed: the control reports the solver's own dual bound beside its incumbent.

Flow (pseudocode):
1. Draw integer weights per row and item from a seeded generator.
2. Set profit to the item's mean weight plus a fixed offset, so profits track weights.
3. Set each row capacity to half its total weight and return an immutable fixture.

Important criteria:
- Deterministic for a fixed size, rows and seed.
- Generic: a sparse linear model with binary variables and one objective; no problem-specific rule.
- The expected value is None; no result may be reported as a proven optimum from this module.

Inputs/Outputs:
- Input: number of items, number of rows and generation seed.
- Output: a PracticalCase (maximize, expected None).
"""

from __future__ import annotations

import numpy as np
from .schema import CaseRow
from .schema import CaseVariable
from .schema import PracticalCase

WEIGHT_RANGE = 1000
PROFIT_OFFSET = 100


class HardCases:
    """Build correlated multi-dimensional knapsack fixtures.

    Responsibilities:
    - Keep generation deterministic and expose the fixture without any reference optimum.
    """

    @staticmethod
    def build(items: int, rows: int, seed: int) -> PracticalCase:
        """Build one correlated multi-dimensional knapsack.

        Args:
            items (int): Number of binary variables, positive.
            rows (int): Number of capacity rows, positive.
            seed (int): Generation seed.

        Returns:
            PracticalCase: Maximization fixture with expected set to None.

        Raises:
            ValueError: If items or rows is not positive.
        """
        if items <= 0 or rows <= 0:
            raise ValueError("Items and rows must be positive.")
        rng = np.random.default_rng(seed)
        weights = rng.integers(1, WEIGHT_RANGE + 1, size=(rows, items))
        profits = weights.mean(axis=0) + PROFIT_OFFSET
        variables = tuple(CaseVariable(f"x_{j}", 0, 1, "binary") for j in range(items))
        constraints = tuple(
            CaseRow(
                f"capacity_{i}",
                tuple((j, int(weights[i, j])) for j in range(items)),
                -float("inf"),
                float(weights[i].sum() // 2),
            )
            for i in range(rows)
        )
        return PracticalCase(
            f"correlated_knapsack_{items}x{rows}",
            "IP",
            f"synthetic seed {seed}",
            "Strongly correlated profits with half-capacity rows; optimum not claimed.",
            variables,
            tuple(float(p) for p in profits),
            constraints,
            True,
            None,
        )
