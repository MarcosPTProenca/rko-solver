"""Immutable linear model records.

Objective:
- Represent original domains, rows and completed invocation evidence.

Technical approach:
- Keep validated state portable and responsibilities within cohesive classes.

Flow (pseudocode):
1. Receive owned model data.
2. Perform this module's bounded numerical responsibility.
3. Return explicit values or immutable evidence.

Important criteria:
- Preserve original domains, tolerances and finite-value checks.

Inputs/Outputs:
- Input: owned bounded linear-model data.
- Output: validated state or portable numerical evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class LinearVariable:
    """Describe one extracted variable independently of frontend native objects.

    Responsibilities:
    - Preserve stable identity, declared bounds, domain and any fixed value.

    Attributes:
        identifier (str): Unique frontend-qualified stable identity, not merely a label.
        label (str): Human-readable name, which need not be unique.
        lower (float): Declared lower bound; negative infinity means absent.
        upper (float): Declared upper bound; positive infinity means absent.
        domain (str): Continuous, integer or binary domain.
        fixed (float | None): Exact fixed value, or None for a free variable.
    """

    identifier: str
    label: str
    lower: float = -math.inf
    upper: float = math.inf
    domain: Literal["continuous", "integer", "binary"] = "continuous"
    fixed: float | None = None


@dataclass(frozen=True, slots=True)
class LinearRow:
    """Describe a ranged affine row in stable variable-index coordinates.

    Responsibilities:
    - Preserve both constraint sides and the original affine constant.

    Attributes:
        identifier (str): Unique stable row identity.
        coefficients (tuple): Unique variable-index/coefficient pairs; zeros are allowed.
        lower (float): Lower side or negative infinity.
        upper (float): Upper side or positive infinity.
        constant (float): Original affine expression constant.
    """

    identifier: str
    coefficients: tuple[tuple[int, float], ...] = ()
    lower: float = -math.inf
    upper: float = math.inf
    constant: float = 0.0


@dataclass(frozen=True, slots=True)
class EffectiveDomain:
    """Record the intersection used for random-key mapping.

    Responsibilities:
    - Distinguish declared domains from user-restricted search boxes.

    Attributes:
        identifier (str): Stable variable identity.
        lower (float): Effective finite lower endpoint.
        upper (float): Effective finite upper endpoint.
        restricted (bool): Whether an explicit override reduced the declared domain.
        fixed (bool): Whether mapping requires no key for this variable.
    """

    identifier: str
    lower: float
    upper: float
    restricted: bool
    fixed: bool


@dataclass(frozen=True, slots=True)
class RowDiagnostic:
    """Keep original row units and tolerance-adjusted violations distinct.

    Responsibilities:
    - Preserve the evaluated activity and independent lower/upper residuals.

    Attributes:
        identifier (str): Stable row identity.
        activity (float): Original affine row value, including its constant.
        lower_residual (float): Nonnegative raw lower-side violation.
        upper_residual (float): Nonnegative raw upper-side violation.
        lower_violation (float): Residual after the lower-side tolerance allowance.
        upper_violation (float): Residual after the upper-side tolerance allowance.
    """

    identifier: str
    activity: float
    lower_residual: float
    upper_residual: float
    lower_violation: float
    upper_violation: float


@dataclass(frozen=True, slots=True)
class RepairDiagnostic:
    """Describe bounded restoration without implying mathematical infeasibility.

    Responsibilities:
    - Retain policy, termination outcome and completed full sweeps for one invocation.

    Attributes:
        policy (str): Disabled box mapping or continuous restoration.
        outcome (str): Checked success, bounded exhaustion or fixed/discrete obstruction.
        sweeps (int): Completed row-and-box cycles.
    """

    policy: Literal["none", "continuous"]
    outcome: Literal[
        "disabled",
        "already_feasible",
        "repaired",
        "work_limit",
        "no_continuous_variables",
        "fixed_obstruction",
    ]
    sweeps: int


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """Associate one generated assignment with its numerical evaluation identity.

    Responsibilities:
    - Keep keys, assignment, original objective, score and feasibility inseparable.
    - Carry no native model references or mutable numerical arrays.

    Attributes:
        keys (tuple[float, ...]): Exact keys used for this invocation.
        values (tuple[float, ...]): Assignments in snapshot variable order, including fixed ones.
        objective (float): Original objective including constant and original sign.
        score (float): Finite minimization search score, not a model optimality claim.
        feasible (bool): Feasibility under effective bounds and declared row tolerances.
        rows (tuple[RowDiagnostic, ...]): Original-unit row evidence.
        repair (RepairDiagnostic): Bounded restoration evidence, not an infeasibility proof.
    """

    keys: tuple[float, ...]
    values: tuple[float, ...]
    objective: float
    score: float
    feasible: bool
    rows: tuple[RowDiagnostic, ...]
    repair: RepairDiagnostic = RepairDiagnostic("none", "disabled", 0)


@dataclass(frozen=True, slots=True)
class ModelEvidence:
    """Transport bounded completed witnesses independently of score-selected pools.

    Responsibilities:
    - Retain the search-score incumbent and the original-objective feasible incumbent.
    - Preserve invocation identity without replay or retaining a candidate trajectory.

    Attributes:
        incumbent (ModelEvaluation): Best completed search-score candidate for this session.
        best_feasible (ModelEvaluation | None): Best completed feasible original objective.
        first_feasible_at (float | None): First finite feasible completion on the shared
            monotonic clock; private transport data, not a wall-clock timestamp.
    """

    incumbent: ModelEvaluation
    best_feasible: ModelEvaluation | None
    first_feasible_at: float | None = None
