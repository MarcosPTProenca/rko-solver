"""Generated algebraic qualification cases with proved optima and explicit witnesses.

Objective:
- Challenge general model decoding beyond small application examples without adding decoder rules.

Technical approach:
- Generate binary implication chains, ranged general-integer pairs and mixed activation rows.
- Include an explicit feasible witness and a matching analytic objective lower bound for each case.
- Seed extra signed binary rows around the witness; never pass witnesses to optimization searches.

Flow (pseudocode):
1. Validate the requested family and dimension.
2. Build original variables, sparse rows, objective and a planted witness.
3. Return immutable fixture data for independent checking and reference optimization.

Important criteria:
- Optima are algebraic bounds, separately testable with HiGHS, not heuristic best-known values.
- These are synthetic stress families, not an industrial or representative general-model corpus.

Inputs/Outputs:
- Input: family, dimension (positive multiple of six) and generation seed.
- Output: a PracticalCase and a feasible witness in original variable order.
"""

from __future__ import annotations

import numpy as np
from .schema import CaseRow
from .schema import CaseVariable
from .schema import PracticalCase

FAMILIES = ("binary_chain", "ranged_integer", "mixed")


class QualificationCases:
    """Construct generic algebraic stress families without coupling them to decoder selection.

    Responsibilities:
    - Keep generated data deterministic and witnesses independently checkable.
    - State the analytic optimum used to evaluate quality.
    """

    @staticmethod
    def build(family: str, size: int, seed: int) -> tuple[PracticalCase, tuple[float, ...]]:
        """Build one fixture and its optimal witness.

        Args:
            family (str): Binary-chain, general-integer or mixed family.
            size (int): Positive dimension divisible by six.
            seed (int): Generation seed, not an optimization seed.

        Returns:
            tuple[PracticalCase, tuple[float, ...]]: Fixture and original-coordinate witness.

        Raises:
            ValueError: If the size or family is unsupported.
        """
        builders = {
            "binary_chain": QualificationCases._binary,
            "ranged_integer": QualificationCases._integer,
            "mixed": QualificationCases._mixed,
        }
        if size <= 0 or size % 6 or family not in builders:
            raise ValueError("Expected a known family and a positive multiple-of-six dimension.")
        return builders[family](size, seed)

    @staticmethod
    def _binary(size: int, seed: int) -> tuple[PracticalCase, tuple[float, ...]]:
        """Chain plus cardinality and signed rows; sum(x) >= n/3 proves the optimum n/3."""
        witness = tuple(float(i >= 2 * size // 3) for i in range(size))
        variables = tuple(CaseVariable(f"b_{i}", 0, 1, "binary") for i in range(size))
        rows = [
            CaseRow("cardinality", tuple((i, 1) for i in range(size)), size // 3, 2 * size // 3)
        ]
        rows.extend(CaseRow(f"chain_{i}", ((i, 1), (i + 1, -1)), upper=0) for i in range(size - 1))
        rows.extend(QualificationCases._signed_rows(witness, seed))
        case = PracticalCase(
            f"binary_chain_{size}",
            "IP",
            f"synthetic qualification seed {seed}",
            "Full-length implication chain; signed rows preserve the suffix witness.",
            variables,
            (1,) * size,
            tuple(rows),
            False,
            float(size // 3),
        )
        return case, witness

    @staticmethod
    def _signed_rows(witness: tuple[float, ...], seed: int) -> tuple[CaseRow, ...]:
        """Generate sparse signed ranged rows containing the witness with two units of slack."""
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(len(witness) // 2):
            indices = sorted(int(j) for j in rng.choice(len(witness), size=4, replace=False))
            coefficients = rng.choice((-3, -2, -1, 1, 2, 3), size=4)
            terms = tuple((j, int(c)) for j, c in zip(indices, coefficients, strict=True))
            activity = sum(witness[j] * c for j, c in terms)
            rows.append(CaseRow(f"signed_{i}", terms, activity - 2, activity + 2))
        return tuple(rows)

    @staticmethod
    def _integer(size: int, seed: int) -> tuple[PracticalCase, tuple[float, ...]]:
        """Disjoint bounded integer pairs; each sum >= 1 proves the optimum n/2."""
        variables = tuple(CaseVariable(f"q_{i}", 0, 5, "integer") for i in range(size))
        rows = []
        for i in range(0, size, 2):
            rows.append(CaseRow(f"pair_{i}", ((i, 1), (i + 1, 1)), lower=1))
            rows.append(CaseRow(f"range_{i}", ((i, 1), (i + 1, -1)), -4, 5))
        witness = tuple(float(i % 2) for i in range(size))
        case = PracticalCase(
            f"ranged_integer_{size}",
            "IP",
            f"synthetic qualification seed {seed}",
            "General integer domains, signed ranged rows; no eligible binary-only row.",
            variables,
            (1,) * size,
            tuple(rows),
            False,
            float(size // 2),
        )
        return case, witness

    @staticmethod
    def _mixed(size: int, seed: int) -> tuple[PracticalCase, tuple[float, ...]]:
        """Half-active mixed blocks; per-block costs >=10 inactive and >=5.25 active prove optimum.

        Each block has z binary, q integer in [0,5], w continuous in [0,5], q+w>=5, w<=5z.
        The objective is .25z+2q+w. At most half the blocks can activate; a suffix of active
        blocks with q=0,w=5 attains the lower bound 7.625 times the number of blocks.
        """
        blocks = size // 3
        variables = tuple(CaseVariable(f"z_{i}", 0, 1, "binary") for i in range(blocks))
        variables += tuple(CaseVariable(f"q_{i}", 0, 5, "integer") for i in range(blocks))
        variables += tuple(CaseVariable(f"w_{i}", 0, 5) for i in range(blocks))
        rows = QualificationCases._mixed_rows(blocks)
        active = tuple(float(i >= blocks // 2) for i in range(blocks))
        witness = active + tuple(5 * (1 - z) for z in active) + tuple(5 * z for z in active)
        case = PracticalCase(
            f"mixed_{size}",
            "MILP",
            f"synthetic qualification seed {seed}",
            "Mixed activation and continuous/integer coupling, with a binary chain.",
            variables,
            (0.25,) * blocks + (2,) * blocks + (1,) * blocks,
            rows,
            False,
            7.625 * blocks,
        )
        return case, witness

    @staticmethod
    def _mixed_rows(blocks: int) -> tuple[CaseRow, ...]:
        """Build sparse activation, capacity, demand and implication rows for mixed blocks."""
        rows = [CaseRow("capacity", tuple((i, 1) for i in range(blocks)), upper=blocks // 2)]
        rows.extend(
            CaseRow(f"chain_{i}", ((i, 1), (i + 1, -1)), upper=0) for i in range(blocks - 1)
        )
        for i in range(blocks):
            rows.append(CaseRow(f"demand_{i}", ((blocks + i, 1), (2 * blocks + i, 1)), lower=5))
            rows.append(CaseRow(f"activation_{i}", ((2 * blocks + i, 1), (i, -5)), upper=0))
        return tuple(rows)
