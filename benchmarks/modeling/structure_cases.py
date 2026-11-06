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
            StructureCases._facility_location(),
            StructureCases._project_selection(),
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

    @staticmethod
    def _facility_location() -> PracticalCase:
        """Open facilities and assign four customers; optimum 19 (open 0 and 1).

        Opening costs are 4, 5 and 6. Linking rows `x_ij - y_j <= 0` carry a negative coefficient.
        Opening 0 and 1 costs 9, and the cheapest open facility serves customers at 2, 3, 2 and 3.
        """
        fixed = (4, 5, 6)
        costs = ((2, 6, 9), (3, 4, 8), (7, 2, 5), (8, 3, 1))
        variables = tuple(CaseVariable(f"y_{j}", 0, 1, "binary") for j in range(3)) + tuple(
            CaseVariable(f"x_{i}_{j}", 0, 1, "binary") for i in range(4) for j in range(3)
        )
        rows = StructureCases._serve_rows() + StructureCases._open_rows()
        return PracticalCase(
            "facility_location",
            "IP",
            SOURCE,
            "Uncapacitated facility location with assignment and linking rows.",
            variables,
            fixed + tuple(costs[i][j] for i in range(4) for j in range(3)),
            tuple(rows),
            False,
            19,
        )

    @staticmethod
    def _serve_rows() -> list[CaseRow]:
        """Each customer is served by exactly one facility."""
        return [
            CaseRow(f"serve_{i}", tuple((3 + i * 3 + j, 1) for j in range(3)), lower=1, upper=1)
            for i in range(4)
        ]

    @staticmethod
    def _open_rows() -> list[CaseRow]:
        """An assignment needs its facility open: `x_ij - y_j <= 0`."""
        return [
            CaseRow(f"open_{i}_{j}", ((3 + i * 3 + j, 1), (j, -1)), upper=0)
            for i in range(4)
            for j in range(3)
        ]

    @staticmethod
    def _project_selection() -> PracticalCase:
        """Pick projects under a budget with precedence; optimum 14 (projects 1, 3 and 4).

        Profits 5, 4, 6, 3, 7, 2; costs 4, 3, 5, 2, 6, 1; budget 11. Project 2 needs 0; project 4
        needs 1 and 3. Projects 1, 3 and 4 cost exactly 11 and earn 4 + 3 + 7.
        """
        profits = (5, 4, 6, 3, 7, 2)
        costs = (4, 3, 5, 2, 6, 1)
        rows = [CaseRow("budget", tuple(enumerate(costs)), upper=11)] + [
            CaseRow(f"needs_{a}_{b}", ((a, 1), (b, -1)), upper=0)
            for a, b in ((2, 0), (4, 1), (4, 3))
        ]
        return PracticalCase(
            "project_selection",
            "IP",
            SOURCE,
            "Budgeted project selection with precedence rows.",
            tuple(CaseVariable(f"p_{i}", 0, 1, "binary") for i in range(6)),
            profits,
            tuple(rows),
            True,
            14,
        )
