"""Bind bounded scheduling JSON to a self-contained example problem without rewriting RKO.

Objective:
- Show the only application-specific service wiring needed for loaded instance values.

Technical approach:
- Validate cheap bounded structure before optimizer allocation.
- Hold the self-contained scheduling problem next to its schema; replace both together.

Flow (pseudocode):
1. Validate bounded positive durations and matching nonnegative due dates.
2. Construct the scheduling problem defined below once per isolated job.
3. Expose its callbacks and an explicit JSON solution serializer.

Important criteria:
- Replace this schema/builder with trusted installed application code, never client imports.
- This JSON input is already loaded; no filesystem reader is needed per candidate/job.

Inputs/Outputs:
- Input: bounded JSON scheduling data.
- Output: trusted ProblemDefinition binding without starting optimization.
"""

import numpy as np
from dataclasses import dataclass
from itertools import accumulate
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import JsonValue
from pydantic import model_validator
from rko_solver import ValidationCheck
from rko_solver import ValidationReport
from rko_solver.service import ProblemDefinition
from rko_solver.tracing import record_decision
from typing import Annotated
from typing import Self
from typing import cast

type Schedule = tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class SchedulingInstance:
    """Hold one validated scheduling input without live files or mutable arrays.

    Responsibilities:
    - Retain the values shared by decoding and objective evaluation.

    Attributes:
        name (str): Human-readable instance label.
        processing_times (tuple[int, ...]): Positive processing duration of each job.
        due_dates (tuple[int, ...]): Nonnegative due date of each job in the same units.
    """

    name: str
    processing_times: tuple[int, ...]
    due_dates: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SchedulingProblem:
    """Bind loaded instance data to pickleable decoder and minimization callbacks.

    Responsibilities:
    - Build a priority-ordered schedule from keys and loaded processing times.
    - Score that schedule against the same instance's due dates.

    Attributes:
        instance (SchedulingInstance): Validated application-owned input snapshot.
    """

    instance: SchedulingInstance

    @property
    def n_variables(self) -> int:
        """Return the number of priority keys needed for this loaded instance.

        Returns:
            int: One key per job.
        """
        return len(self.instance.processing_times)

    def decode(self, keys: np.ndarray) -> Schedule:
        """Construct completion times in stable increasing random-key order.

        Args:
            keys (np.ndarray): One finite priority key per job; ties retain job-index order.

        Returns:
            Schedule: Ordered job indices paired with their completion times.

        Raises:
            ValueError: If keys are nonnumeric, nonfinite or have the wrong shape.
            TypeError: If keys cannot be converted into a numeric array.
        """
        values = np.asarray(keys, dtype=float)
        if values.ndim != 1 or len(values) != self.n_variables or not np.isfinite(values).all():
            raise ValueError("Decoder needs one finite key per job in a one-dimensional vector.")
        record_decision(
            "priority_order", "Use stable increasing key order; preserve job-index ties."
        )
        elapsed = 0
        schedule: list[tuple[int, int]] = []
        for index in np.argsort(values, kind="stable"):
            job = int(index)
            elapsed += self.instance.processing_times[job]
            schedule.append((job, elapsed))
            record_decision("append_job", "Append the next priority job without overlap.")
        return tuple(schedule)

    def validate(self, solution: Schedule) -> ValidationReport:
        """Check job coverage and contiguous completion times of this problem's own schedule.

        Args:
            solution (Schedule): Schedule reconstructed by this problem's decoder.

        Returns:
            ValidationReport: Explicit constraint diagnostics and declared feasibility.
                Due dates are soft objective terms, not hard feasibility deadlines.
        """
        covered = sorted(job for job, _ in solution) == list(range(self.n_variables))
        expected = list(accumulate(self.instance.processing_times[job] for job, _ in solution))
        contiguous = covered and [done for _, done in solution] == expected
        checks = (
            ValidationCheck(
                "job_coverage",
                "passed" if covered else "failed",
                "Require every instance job exactly once.",
                value=len(solution),
                limit=self.n_variables,
                unit="jobs",
            ),
            ValidationCheck(
                "completion_times",
                "passed" if contiguous else "failed",
                "Check start-at-zero, no-idle cumulative processing times.",
            ),
        )
        return ValidationReport(
            checks=checks,
            feasible=contiguous,
            checker="SchedulingProblem.validate",
            checker_version="1",
        )

    def cost(self, solution: Schedule) -> float:
        """Sum positive lateness against the loaded due dates for a decoded schedule.

        Args:
            solution (Schedule): Ordered job/completion pairs returned by this decoder.

        Returns:
            float: Total tardiness to minimize, in the input's time units.

        Raises:
            IndexError: If a supplied schedule refers to a job outside the instance.
            OverflowError: If the summed integer tardiness cannot be represented as a float.
        """
        return float(
            sum(max(0, completion - self.instance.due_dates[job]) for job, completion in solution)
        )


class InstanceInput(BaseModel):
    """Validate the reference adapter's bounded JSON data before worker allocation.

    Responsibilities:
    - Reject paths/import fields, oversized dimensions and malformed durations/dates.

    Attributes:
        name (str): Nonempty bounded application label, not a filesystem name.
        processing_times (list[int]): Positive bounded durations for at most 1024 jobs.
        due_dates (list[int]): Matching nonnegative bounded due dates.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1, max_length=128)
    processing_times: list[Annotated[int, Field(gt=0, le=1_000_000_000)]] = Field(
        min_length=1, max_length=1024
    )
    due_dates: list[Annotated[int, Field(ge=0, le=1_000_000_000)]] = Field(
        min_length=1, max_length=1024
    )

    @model_validator(mode="after")
    def matching_dimensions(self) -> Self:
        """Require a nonblank label and one due date per processing duration.

        Returns:
            Self: Validated instance data.

        Raises:
            ValueError: If label/dimensions are invalid.
        """
        if not self.name.strip() or len(self.processing_times) != len(self.due_dates):
            raise ValueError("Use a nonblank name and matching duration/due-date dimensions.")
        return self


def serialize_schedule(solution: object) -> JsonValue:
    """Convert an already reconstructed schedule without objective evaluation.

    Args:
        solution (object): Scheduling adapter's post-run Schedule payload.

    Returns:
        JsonValue: Job/completion pairs suitable for JSON, not arbitrary Python serialization.
    """
    return [list(item) for item in cast(Schedule, solution)]


def build_problem(data: dict[str, JsonValue]) -> ProblemDefinition:
    """Build one trusted application problem from already bounded loaded data.

    Args:
        data (dict[str, JsonValue]): Schema-validated scheduling values, never paths/code.

    Returns:
        ProblemDefinition: Unchanged application callbacks and explicit output conversion.

    Raises:
        ValueError: If trusted application construction/schema validation fails.
    """
    parsed = InstanceInput.model_validate(data)
    instance = SchedulingInstance(
        parsed.name, tuple(parsed.processing_times), tuple(parsed.due_dates)
    )
    problem = SchedulingProblem(instance)
    return ProblemDefinition(
        problem.decode, problem.cost, problem.n_variables, problem.validate, serialize_schedule
    )
