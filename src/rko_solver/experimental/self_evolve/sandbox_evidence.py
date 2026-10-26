"""Independent worker evidence verification for rootless execution.

Objective:
- Reject malformed resource counters and independently recompute final witnesses.
Technical approach:
- Compare exact claim types and values against the trusted problem evaluator.
- Keep runtime allowance rejection distinct from integrity failures.
- Reconcile known cell consumption without clipping and mark reservation overruns.
Flow (pseudocode):
1. Evaluate the returned solution on the host.
2. Validate objective claims and resource measurements.
3. Validate bounded evaluation counters.
4. Retain full known cell charges and flag reservation overruns as infrastructure failures.
Important criteria:
- Integrity failures disable the calling runner; allowance rejection is separate.
Inputs/Outputs:
- Input: worker records and trusted problems.
- Output: independently computed evidence or a typed sandbox failure.
"""

from __future__ import annotations

import math
from .problems import Problem
from typing import Any
from typing import cast


class SandboxError(RuntimeError):
    """Distinguish actual resource stops from protocol or cleanup failures.

    Responsibilities:
    - Prevent malformed responses from masquerading as timeout qualification.

    Attributes:
        reason (str): Machine-readable failure category.
        phase (str | None): Actually observed timeout origin.
    """

    def __init__(self, message: str, *, reason: str = "runtime", phase: str | None = None) -> None:
        """Retain diagnostic text and explicit resource-failure identity.

        Args:
            message (str): Bounded diagnostic for the trusted controller.
            reason (str): Explicit failure category.
            phase (str | None): Host I/O or worker guardian origin.
        """
        super().__init__(message)
        self.reason = reason
        self.phase = phase


class WorkerEvidence:
    """Validate worker claims against independently computed host evidence.

    Responsibilities:
    - Enforce exact objective field types and nonnegative finite resources.
    - Enforce bounded evaluation counters without trusting worker feasibility.
    - Retain full known charges and mark resource-reservation overruns as integrity failures.

    Attributes:
        record (dict[str, Any]): Bounded worker response under verification.
    """

    def __init__(self, record: dict[str, Any]) -> None:
        """Retain a parsed worker response.

        Args:
            record (dict[str, Any]): JSON object received from the worker.
        """
        self.record = record

    def witness(self, problem: Problem) -> dict[str, Any]:
        """Recompute the witness and compare exact objective claims.

        Args:
            problem (Problem): Trusted instance and evaluator.

        Returns:
            dict[str, Any]: Independently computed solution semantics.

        Raises:
            SandboxError: If witness evaluation or objective claims fail.
        """
        record = self.record
        try:
            evidence = problem.evaluate(record["solution"])
        except (KeyError, ValueError, TypeError) as error:
            raise SandboxError("Worker witness failed independent verification.") from error
        if (
            type(record.get("objective")) is not int
            or type(record.get("feasible")) is not bool
            or type(record.get("score")) not in (int, float)
        ):
            raise SandboxError("Invalid worker objective/feasibility field types.")
        if any(record.get(key) != evidence[key] for key in ("objective", "feasible", "score")):
            raise SandboxError(
                "Worker objective/feasibility claim failed independent verification."
            )
        return evidence

    def resources(self) -> None:
        """Reject nonnumeric, negative or nonfinite resource claims.

        Raises:
            SandboxError: If a required resource measurement is invalid.
        """
        for key in ("cpu_seconds", "wall_seconds", "peak_memory_bytes", "evaluations"):
            value = self.record.get(key)
            if type(value) not in (int, float):
                raise SandboxError("Invalid worker resource evidence.")
            numeric = cast(int | float, value)
            if not math.isfinite(numeric) or numeric < 0:
                raise SandboxError("Invalid worker resource evidence.")

    def counters(self) -> None:
        """Enforce integer evaluation counts and capped nonempty diversity.

        Raises:
            SandboxError: If counts are inconsistent, empty or exceed the diversity cap.
        """
        record = self.record
        for key in ("evaluations", "infeasible_evaluations", "distinct_outputs_capped"):
            if type(record.get(key)) is not int or not 0 <= record[key] <= record["evaluations"]:
                raise SandboxError("Invalid worker evaluation counters.")
        if record["evaluations"] == 0 or record["distinct_outputs_capped"] > 256:
            raise SandboxError("Worker evidence contains no evaluations or exceeds diversity cap.")

    def charge(self, reserve: float) -> float:
        """Retain full known consumption and mark reservation overruns as integrity failures.

        Args:
            reserve (float): Conservative cell reservation committed before execution.

        Returns:
            float: Unclipped conservative charge. Overruns update the owned record's
                status, diagnostic and loss to prohibit further collection and inference.
        """
        charge = max(self.record["cpu_seconds"], self.record["outer_wall_seconds"] + 0.1)
        if charge > reserve:
            self.record.update(
                status="infrastructure_error",
                error="Known resource consumption exceeded the cell reservation.",
                error_reason="resource_overrun",
                error_phase="accounting",
                loss=1.0,
            )
        return charge
