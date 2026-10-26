"""Explicit post-solve domain assessment without modifying optimization semantics.

Objective:
- Separate application constraint checks from finite numerical objective values.

Technical approach:
- Decode a detached incumbent once after search and call an optional checker once.
- Retain frozen bounded diagnostics with explicit feasibility declarations.
- Export only scalar diagnostics, never arbitrary decoded payloads or candidate keys.
- Aggregate check statuses in O(number of checks), capped at 1000 per report.

Flow (pseudocode):
1. Receive an already completed immutable RKOResult.
2. Reconstruct its domain solution using the caller's decoder.
3. Run the supplied checker or mark domain constraints not checked.
4. Return a separately timed assessment without reevaluation, repair or reranking.

Important criteria:
- A failed constraint is data; application callback exceptions propagate unchanged.
- Post-run reconstruction is not an original evaluation witness for stateful decoders.
- Passed checks do not imply global optimality or complete domain-constraint coverage.
- The result reference and decoded payload remain in memory, not in default exports.

Inputs/Outputs:
- Input: completed result, decoder and optional ValidationReport-returning checker.
- Output: decoded payload, bounded validation diagnostics and postprocessing duration.
"""

from __future__ import annotations

import math
import numpy as np
import time
from .solver import RKOResult
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Literal

type CheckStatus = Literal["passed", "failed", "not_checked"]
type DiagnosticValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """Describe one declared application constraint using bounded scalar diagnostics.

    Responsibilities:
    - Preserve a stable check identity, outcome, message and optional measured limit.
    - Reject nonfinite values and oversized text before report serialization.

    Attributes:
        name (str): Unique report-local identity, 1 through 128 characters.
        status (CheckStatus): Passed, failed or not_checked.
        message (str): Human-readable diagnostic, at most 2048 characters.
        value (DiagnosticValue): Optional finite scalar measurement or short label.
        limit (DiagnosticValue): Optional finite scalar constraint limit or short label.
        unit (str | None): Optional unit label, at most 64 characters.
    """

    name: str
    status: CheckStatus
    message: str = ""
    value: DiagnosticValue = None
    limit: DiagnosticValue = None
    unit: str | None = None

    def __post_init__(self) -> None:
        """Validate primitive fields without silently coercing diagnostic meanings.

        Raises:
            ValueError: If names/statuses/types, numeric finiteness or text bounds fail.
        """
        for label, value, bound in (
            ("name", self.name, 128),
            ("message", self.message, 2048),
            ("unit", self.unit, 64),
        ):
            if label == "unit" and value is None:
                continue
            if not isinstance(value, str) or len(value) > bound:
                raise ValueError(f"Check {label} must be a string of at most {bound} characters.")
        if not self.name.strip():
            raise ValueError("Check name must not be empty.")
        if type(self.status) is not str or self.status not in ("passed", "failed", "not_checked"):
            raise ValueError("Check status must be passed, failed or not_checked.")
        for value in (self.value, self.limit):
            if value is None:
                continue
            if type(value) not in (str, int, float, bool):
                raise ValueError("Check values and limits must be finite primitive scalars.")
            if isinstance(value, str) and len(value) > 256:
                raise ValueError("Check value/limit labels must be at most 256 characters.")
            if type(value) in (int, float):
                try:
                    finite = math.isfinite(value)
                except OverflowError:
                    finite = False
                if not finite:
                    raise ValueError("Check numeric values and limits must be finite.")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Aggregate implemented checks separately from declared domain feasibility.

    Responsibilities:
    - Snapshot at most 1000 uniquely named typed checks.
    - Keep incomplete checks and undeclared feasibility visibly unknown.

    Attributes:
        checks (Sequence[ValidationCheck]): Check sequence, snapshotted into an owned tuple.
        feasible (bool | None): Explicit checker declaration, never inferred from cost.
        checker (str | None): Optional checker identity, at most 256 characters.
        checker_version (str | None): Optional checker revision, at most 128 characters.
    """

    checks: Sequence[ValidationCheck] = ()
    feasible: bool | None = None
    checker: str | None = None
    checker_version: str | None = None

    def __post_init__(self) -> None:
        """Snapshot and validate reports without inventing missing checks.

        Raises:
            ValueError: If checks are invalid/duplicated/oversized or feasibility contradicts them.
        """
        if (
            not isinstance(self.checks, Sequence)
            or isinstance(self.checks, (str, bytes, bytearray))
            or len(self.checks) > 1000
        ):
            raise ValueError("ValidationReport requires a sequence of at most 1000 checks.")
        if any(not isinstance(check, ValidationCheck) for check in self.checks):
            raise ValueError("Every check must be a ValidationCheck.")
        object.__setattr__(self, "checks", tuple(self.checks))
        if len({check.name for check in self.checks}) != len(self.checks):
            raise ValueError("Validation check names must be unique within a report.")
        if self.feasible is not None and type(self.feasible) is not bool:
            raise ValueError("Feasibility must be True, False or None when undeclared.")
        if self.feasible is True and (
            not self.checks or any(check.status != "passed" for check in self.checks)
        ):
            raise ValueError("Declaring feasible=True requires nonempty, all-passed checks.")
        for label, value, bound in (
            ("checker", self.checker, 256),
            ("checker_version", self.checker_version, 128),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip() or len(value) > bound
            ):
                raise ValueError(
                    f"{label} must be a nonempty string of at most {bound} characters."
                )

    @property
    def status(self) -> CheckStatus:
        """Summarize diagnostics without converting passed subsets into feasibility claims.

        Returns:
            CheckStatus: Failed wins; missing/incomplete checks are not_checked; otherwise passed.
        """
        if self.feasible is False or any(check.status == "failed" for check in self.checks):
            return "failed"
        if not self.checks or any(check.status == "not_checked" for check in self.checks):
            return "not_checked"
        return "passed"

    def to_dict(self) -> dict[str, object]:
        """Export only bounded primitive diagnostic records, including aggregate status.

        Returns:
            dict[str, object]: JSON-compatible check outcomes and explicit feasibility metadata.
        """
        return {
            "status": self.status,
            "feasible": self.feasible,
            "checker": self.checker,
            "checker_version": self.checker_version,
            "checks": [asdict(check) for check in self.checks],
        }


@dataclass(frozen=True, slots=True)
class SolutionAssessment[SolutionT]:
    """Keep caller-owned reconstructed solution and post-solve validation together.

    Responsibilities:
    - Associate an assessment with its exact numerical source result.
    - Exclude arbitrary payloads/source keys from the default diagnostic export.

    Attributes:
        result (RKOResult): Immutable source result, never repaired or reranked.
        solution (SolutionT): Reconstructed caller-owned payload; not recursively frozen.
        validation (ValidationReport): Completed checker diagnostics or not_checked report.
        elapsed_seconds (float): Decoding/checking time, separate from optimization time.
        origin (str): Always post_run_decode, not an original execution witness.
    """

    result: RKOResult
    solution: SolutionT
    validation: ValidationReport
    elapsed_seconds: float
    origin: Literal["post_run_decode"] = field(default="post_run_decode", init=False)

    def __post_init__(self) -> None:
        """Require a source result, typed validation and finite postprocessing duration.

        Raises:
            ValueError: If result/validation types or postprocessing duration are invalid.
        """
        if not isinstance(self.result, RKOResult) or not isinstance(
            self.validation, ValidationReport
        ):
            raise ValueError("Assessment requires an RKOResult and ValidationReport.")
        try:
            valid_duration = (
                type(self.elapsed_seconds) in (int, float)
                and math.isfinite(self.elapsed_seconds)
                and self.elapsed_seconds >= 0
            )
        except OverflowError:
            valid_duration = False
        if not valid_duration:
            raise ValueError("Assessment duration must be finite and nonnegative.")

    def to_dict(self) -> dict[str, object]:
        """Export diagnostics without the decoded payload or original result/keys.

        Returns:
            dict[str, object]: Assessment origin, separate elapsed time and validation.
        """
        data = {
            "origin": self.origin,
            "elapsed_seconds": self.elapsed_seconds,
            "validation": self.validation.to_dict(),
        }
        if self.result.model is not None:
            data["scope"] = "search_incumbent_not_model_best_feasible"
        return data


def assess_result[SolutionT](
    result: RKOResult,
    *,
    decoder: Callable[[np.ndarray], SolutionT],
    validator: Callable[[SolutionT], ValidationReport] | None = None,
) -> SolutionAssessment[SolutionT]:
    """Decode final keys once and optionally check them outside the search lifecycle.

    Args:
        result (RKOResult): Already completed numerical optimization result.
        decoder (Callable[[np.ndarray], SolutionT]): Reconstruct a domain solution from keys.
        validator (Callable[[SolutionT], ValidationReport] | None): Trusted read-only checker;
            omit it for an explicit not_checked assessment.

    Returns:
        SolutionAssessment[SolutionT]: Reconstructed solution, diagnostics and separate duration.

    Raises:
        TypeError: If result/callbacks or the checker return type are unsupported.
        Exception: Original decoder/checker failures propagate without changing the result.
    """
    if not isinstance(result, RKOResult) or not callable(decoder):
        raise TypeError("assess_result requires an RKOResult and callable decoder.")
    if validator is not None and not callable(validator):
        raise TypeError("validator must be callable or None.")
    started = time.perf_counter()
    solution = decoder(np.array(result.keys, dtype=float, copy=True))
    validation = ValidationReport() if validator is None else validator(solution)
    if not isinstance(validation, ValidationReport):
        raise TypeError("validator must return a ValidationReport, not a boolean or dictionary.")
    return SolutionAssessment(result, solution, validation, time.perf_counter() - started)
