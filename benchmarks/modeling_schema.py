"""Immutable original benchmark model records.

Objective:
- Preserve cited model data independently of generated numerical callbacks.

Technical approach:
- Own frozen variable, row and formulation records with original units and provenance.

Flow (pseudocode):
1. Define original finite variable domains.
2. Define sparse original affine row sides.
3. Own provenance, objective direction and independently known reference answer.

Important criteria:
- No native frontend handles or compiled arrays enter independent fixture records.

Inputs/Outputs:
- Input: original cited formulation data.
- Output: immutable independent benchmark records.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class CaseVariable:
    """Describe an original benchmark variable without native or compiled references.

    Responsibilities:
    - Keep finite physical bounds and original integrality available to an independent checker.

    Attributes:
        name (str): Unique label shared across all native frontends.
        lower (float): Declared finite lower bound.
        upper (float): Declared finite upper bound.
        kind (str): Continuous, integer or binary domain.
    """

    name: str
    lower: float
    upper: float
    kind: Literal["continuous", "integer", "binary"] = "continuous"


@dataclass(frozen=True, slots=True)
class CaseRow:
    """Describe an original benchmark row using fixture coordinates, not compiled arrays.

    Responsibilities:
    - Preserve both sides and original coefficient meanings for independent validation.

    Attributes:
        name (str): Stable row label.
        coefficients (tuple): Original variable-index/coefficient pairs.
        lower (float): Lower bound or negative infinity.
        upper (float): Upper bound or positive infinity.
    """

    name: str
    coefficients: tuple[tuple[int, float], ...]
    lower: float = -math.inf
    upper: float = math.inf


@dataclass(frozen=True, slots=True)
class PracticalCase:
    """Own a cited practical formulation independently of RKO conversion/evaluation code.

    Responsibilities:
    - Keep reproducible input coefficients, explicit adaptations and a checkable expected answer.

    Attributes:
        name (str): Unique campaign case name.
        family (str): LP, IP or MILP mathematical class.
        source (str): Primary input/model reference URL.
        adaptation (str): Explicit differences from the cited teaching/reference formulation.
        variables (tuple[CaseVariable, ...]): Original variables in shared native order.
        objective (tuple[float, ...]): Original objective coefficients.
        rows (tuple[CaseRow, ...]): Original constraints.
        maximize (bool): Original objective direction.
        expected (float | None): Known exact objective, or None for deliberate inconsistency.
    """

    name: str
    family: str
    source: str
    adaptation: str
    variables: tuple[CaseVariable, ...]
    objective: tuple[float, ...]
    rows: tuple[CaseRow, ...]
    maximize: bool
    expected: float | None
