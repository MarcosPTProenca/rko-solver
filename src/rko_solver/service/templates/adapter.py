"""Bind bounded scheduling JSON to the existing example problem without rewriting RKO.

Objective:
- Show the only application-specific service wiring needed for loaded instance values.

Technical approach:
- Validate cheap bounded structure before optimizer allocation.
- Construct one immutable SchedulingInstance and reuse its decoder/objective/checker.

Flow (pseudocode):
1. Validate bounded positive durations and matching nonnegative due dates.
2. Construct the unchanged scheduling problem once per isolated job.
3. Expose its callbacks and an explicit JSON solution serializer.

Important criteria:
- Replace this schema/builder with trusted installed application code, never client imports.
- This JSON input is already loaded; no filesystem reader is needed per candidate/job.

Inputs/Outputs:
- Input: bounded JSON scheduling data.
- Output: trusted ProblemDefinition binding without starting optimization.
"""

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import JsonValue
from pydantic import model_validator
from rko_solver.examples.instance_pipeline import Schedule
from rko_solver.examples.instance_pipeline import SchedulingInstance
from rko_solver.examples.instance_pipeline import SchedulingProblem
from rko_solver.service import ProblemDefinition
from typing import Annotated
from typing import Self
from typing import cast


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
