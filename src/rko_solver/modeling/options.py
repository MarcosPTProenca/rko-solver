"""Typed settings for generic linear compilation and public model construction.

Objective:
- Preserve public keyword names without unbounded positional interfaces.

Technical approach:
- Describe accepted keywords with TypedDict and normalize numerical policy once.

Flow (pseudocode):
1. Read explicit caller settings.
2. Validate scoring, tolerances and fixed restoration work limits.
3. Pass normalized policy to the owned compiler.

Important criteria:
- Unknown settings are rejected, never silently ignored.

Inputs/Outputs:
- Input: bounded model construction settings.
- Output: typed normalized compilation policy.
"""

from __future__ import annotations

from .storage import LinearStorage
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace
from typing import Literal
from typing import TypedDict


class SearchOptions(TypedDict, total=False):
    """Describe the accepted optional numerical compiler keywords.

    Responsibilities:
    - Keep the established keyword API statically discoverable.

    Attributes:
        repair (str): Explicit box or continuous restoration policy.
        repair_max_sweeps (int): Fixed restoration work cap.
    """

    search_bounds: Mapping[str, tuple[float | None, float | None]] | None
    scoring: Literal["feasibility_first", "weighted_penalty"]
    penalty_weight: float | None
    absolute_tolerance: float
    relative_tolerance: float
    repair: Literal["none", "continuous"]
    repair_max_sweeps: int


class LinearOptions(SearchOptions, total=False):
    """Extend search settings with original affine objective metadata.

    Responsibilities:
    - Preserve the low-level compiler's objective keywords.

    Attributes:
        objective_constant (float): Original affine offset.
        maximize (bool): Original objective direction.
    """

    objective_constant: float
    maximize: bool


@dataclass(frozen=True, slots=True)
class LinearPolicy:
    """Own validated numerical compiler settings.

    Responsibilities:
    - Reject unsupported modes and unsafe scalar settings before compilation.

    Attributes:
        search_bounds (Mapping | None): Restrictive stable-ID endpoint overrides.
        scoring (str): Feasibility-first or explicit weighted penalty.
        repair (str): Box-only or bounded continuous restoration.
    """

    objective_constant: float = 0.0
    maximize: bool = False
    search_bounds: Mapping[str, tuple[float | None, float | None]] | None = None
    scoring: Literal["feasibility_first", "weighted_penalty"] = "feasibility_first"
    penalty_weight: float | None = None
    absolute_tolerance: float = 1e-6
    relative_tolerance: float = 0.0
    repair: Literal["none", "continuous"] = "none"
    repair_max_sweeps: int = 1024

    def normalized(self) -> LinearPolicy:
        """Validate modes and normalize finite scalar allowances.

        Returns:
            LinearPolicy: Equivalent policy with validated floating-point scalars.

        Raises:
            ValueError: If a numerical setting or mode is invalid.
        """
        self._modes()
        absolute = LinearStorage._number(self.absolute_tolerance, "absolute_tolerance")
        relative = LinearStorage._number(self.relative_tolerance, "relative_tolerance")
        if min(absolute, relative) < 0:
            raise ValueError("Feasibility tolerances must be nonnegative.")
        return replace(
            self,
            absolute_tolerance=absolute,
            relative_tolerance=relative,
            penalty_weight=self._weight(),
        )

    def _modes(self) -> None:
        """Validate discrete policy choices and strict integer work limits.

        Raises:
            ValueError: If a mode, boolean or restoration limit is invalid.
        """
        if self.repair not in ("none", "continuous"):
            raise ValueError("repair must be none or continuous.")
        if type(self.repair_max_sweeps) is not int or not 1 <= self.repair_max_sweeps <= 10000:
            raise ValueError("repair_max_sweeps must be an integer from 1 to 10000.")
        if type(self.maximize) is not bool:
            raise ValueError("maximize must be a boolean.")
        if self.scoring not in ("feasibility_first", "weighted_penalty"):
            raise ValueError("scoring must be feasibility_first or weighted_penalty.")

    def _weight(self) -> float | None:
        """Validate the explicit finite-weight contract without inventing defaults.

        Returns:
            float | None: Positive penalty weight, or None in feasibility-first mode.

        Raises:
            ValueError: If the chosen scorer and weight disagree.
        """
        if self.scoring == "weighted_penalty":
            weight = LinearStorage._number(self.penalty_weight, "penalty_weight")
            if weight <= 0:
                raise ValueError("penalty_weight must be positive and finite.")
            return weight
        if self.penalty_weight is not None:
            raise ValueError("penalty_weight is only supported in weighted_penalty mode.")
        return None
