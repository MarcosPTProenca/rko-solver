"""Small structured MILP fixtures whose feasibility independent random keys rarely reach.

Objective:
- Give the decoder key-feasibility bench cases with the structure a generic decoder could exploit.

Technical approach:
- Hand-built immutable `PracticalCase` records with a hand-checked optimum.
- Cover a coupled equality, cardinality with big-M style linking, and set covering.

Flow (pseudocode):
1. Declare variables, sparse rows and the known optimum of each fixture.
2. Return them in a stable order for the bench and its tests.

Important criteria:
- Optima are verified by tests with a hand-built witness and, when SciPy is present, by HiGHS.
- Data are synthetic teaching cases, not industrial instances.

Inputs/Outputs:
- Input: none.
- Output: immutable fixtures.
"""

from __future__ import annotations

from .schema import CaseRow
from .schema import CaseVariable
from .schema import PracticalCase

SOURCE = "synthetic structure fixture for issue 125"


class StructureCases:
    """Build the structure fixtures in a stable order.

    Responsibilities:
    - Keep coefficients and optima next to each other so the optimum is reviewable.
    """

    @staticmethod
    def cases() -> tuple[PracticalCase, ...]:
        """Return the equality, cardinality-linking and set-cover fixtures.

        Returns:
            tuple[PracticalCase, ...]: Immutable fixtures with known optima.
        """
        return (
            StructureCases._equality(),
            StructureCases._cardinality(),
            StructureCases._set_cover(),
        )

    @staticmethod
    def _equality() -> PracticalCase:
        """`min x + 2y` with `x + y = 1` on the unit square (docs/modeling.md diagnosis)."""
        return PracticalCase(
            "coupled_equality",
            "LP",
            SOURCE,
            "Equality diagnosed in docs/modeling.md.",
            (CaseVariable("x", 0, 1), CaseVariable("y", 0, 1)),
            (1, 2),
            (CaseRow("balance", ((0, 1), (1, 1)), lower=1, upper=1),),
            False,
            1,
        )

    @staticmethod
    def _cardinality() -> PracticalCase:
        """Pick exactly three of six assets, weights sum to one, each in [0.1, 0.5] if held.

        Optimum 1.2: the three cheapest costs are 1, 1 and 3; the cost-3 asset takes the
        minimum 0.1 and the two cost-1 assets split the remaining 0.9.
        """
        costs = (3, 1, 4, 1, 5, 9)
        variables = tuple(CaseVariable(f"z_{i}", 0, 1, "binary") for i in range(6)) + tuple(
            CaseVariable(f"w_{i}", 0, 1) for i in range(6)
        )
        rows = [
            CaseRow("budget", tuple((6 + i, 1) for i in range(6)), lower=1, upper=1),
            CaseRow("cardinality", tuple((i, 1) for i in range(6)), lower=3, upper=3),
        ]
        for i in range(6):
            rows.append(CaseRow(f"upper_{i}", ((6 + i, 1), (i, -0.5)), upper=0))
            rows.append(CaseRow(f"lower_{i}", ((6 + i, 1), (i, -0.1)), lower=0))
        return PracticalCase(
            "cardinality_linking",
            "MILP",
            SOURCE,
            "Linear Markowitz-style cardinality and buy-in structure, cost objective.",
            variables,
            (0,) * 6 + costs,
            tuple(rows),
            False,
            1.2,
        )

    @staticmethod
    def _set_cover() -> PracticalCase:
        """Cover five elements with six sets; optimum 3 (sets 0, 2 and 3)."""
        sets = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (0, 1, 2, 3, 4))
        costs = (1, 1, 1, 1, 1, 4)
        rows = tuple(
            CaseRow(
                f"element_{e}",
                tuple((s, 1) for s, members in enumerate(sets) if e in members),
                lower=1,
            )
            for e in range(5)
        )
        return PracticalCase(
            "set_cover",
            "IP",
            SOURCE,
            "Five elements, six sets; the full set costs 4.",
            tuple(CaseVariable(f"s_{i}", 0, 1, "binary") for i in range(6)),
            costs,
            rows,
            False,
            3,
        )
