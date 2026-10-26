"""Typed application bindings and bounded inputs for the optional single-tenant service.

Objective:
- Separate trusted installed callback construction from untrusted JSON/settings.

Technical approach:
- Use immutable server limits and problem definitions with no client-selected imports.
- Validate request options through Pydantic and the existing algorithm registry.

Flow (pseudocode):
1. Bind a developer-owned instance schema and importable factory at startup.
2. Validate bounded request settings before admitting an optimizer process.
3. Build opaque decoder/objective callbacks once inside that process.

Important criteria:
- Adapter classes are service wiring, not new requirements for ordinary optimization.
- The service deliberately uses sequential RKO inside one supervised process.

Inputs/Outputs:
- Input: trusted schema/factory and bounded JSON request options.
- Output: validated settings and callback definitions without executing optimization.
"""

from ..assessment import ValidationReport
from ..explain import DecoderDiagram
from ..orchestration.configuration import MH_SPECS
from ..orchestration.types import CostFn
from ..orchestration.types import DecoderFn
from collections.abc import Callable
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import JsonValue
from typing import Literal


@dataclass(frozen=True)
class ServiceLimits:
    """Own explicit server-side admission, retention and process resource bounds.

    Responsibilities:
    - Bound public inputs, job duration and retained output independently of RKO time.

    Attributes:
        max_body_bytes (int): Maximum complete JSON request body.
        max_variables (int): Maximum dimensions returned by the trusted factory.
        max_algorithms (int): Maximum sequential portfolio size.
        max_search_seconds (float): Maximum cooperative allowance per algorithm.
        max_wall_seconds (float): End-to-end job deadline, including postprocessing.
        startup_seconds (float): Maximum isolated-process startup handshake time.
        retention_seconds (float): Upper retention age; count pressure may expire earlier.
        max_records (int): Maximum active/completed records and artifact directories.
        max_json_bytes (int): Maximum result/IPC JSON size.
        max_artifact_bytes (int): Maximum individual artifact/file size.
        extra_address_space (int): Additional job virtual-address-space allowance, not RSS.
        allow_reports (bool): Explicit permission to request the independent reports extra.
    """

    max_body_bytes: int = 65_536
    max_variables: int = 1024
    max_algorithms: int = 3
    max_search_seconds: float = 2.0
    max_wall_seconds: float = 30.0
    startup_seconds: float = 10.0
    retention_seconds: float = 300.0
    max_records: int = 4
    max_json_bytes: int = 262_144
    max_artifact_bytes: int = 8_000_000
    extra_address_space: int = 268_435_456
    allow_reports: bool = False

    def __post_init__(self) -> None:
        """Reject invalid server configuration before any HTTP worker starts.

        Raises:
            ValueError: If limits are nonfinite, nonpositive or exceed reference bounds.
        """
        integers = (
            self.max_body_bytes,
            self.max_variables,
            self.max_algorithms,
            self.max_records,
            self.max_json_bytes,
            self.max_artifact_bytes,
            self.extra_address_space,
        )
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ValueError("Service integer limits must be positive integers.")
        if (
            self.max_body_bytes > 1_048_576
            or self.max_variables > 10000
            or self.max_algorithms > 3
            or self.max_records > 16
            or self.max_json_bytes > 1_048_576
            or self.max_artifact_bytes > 16_000_000
        ):
            raise ValueError("Service limits exceed the bounded reference envelope.")
        for value in (
            self.max_search_seconds,
            self.max_wall_seconds,
            self.startup_seconds,
            self.retention_seconds,
        ):
            if type(value) not in (int, float) or not 0 < value <= 86400:
                raise ValueError("Service durations must be finite positive bounded numbers.")
        if type(self.allow_reports) is not bool:
            raise ValueError("allow_reports must be a boolean.")


@dataclass(frozen=True)
class ProblemDefinition:
    """Bind one validated application's unchanged numerical callbacks.

    Responsibilities:
    - Construct one problem per job without callback code supplied by clients.

    Attributes:
        decoder (DecoderFn): Trusted problem decoder.
        cost (CostFn): Trusted minimization objective.
        n_variables (int): Bounded random-key dimension.
        validator (Callable | None): Optional final checker, outside solve timing.
        serialize_solution (Callable | None): Optional trusted JSON conversion of reconstruction.
        diagram (DecoderDiagram | None): Optional explicit static diagram, never auto-inspected.
    """

    decoder: DecoderFn
    cost: CostFn
    n_variables: int
    validator: Callable[[object], ValidationReport] | None = None
    serialize_solution: Callable[[object], JsonValue] | None = None
    diagram: DecoderDiagram | None = None


@dataclass(frozen=True)
class ServiceAdapter:
    """Connect a bounded instance schema to an importable trusted problem factory.

    Responsibilities:
    - Validate instance structure before allocating an optimizer process.
    - Build callbacks once in the isolated job, never per candidate.

    Attributes:
        instance_model (type[BaseModel]): Bounded, cheap developer-owned input schema.
        build (Callable): Importable factory accepting validated JSON-safe instance data.
    """

    instance_model: type[BaseModel]
    build: Callable[[dict[str, JsonValue]], ProblemDefinition]

    def __post_init__(self) -> None:
        """Reject malformed trusted wiring before startup or any HTTP request.

        Raises:
            ValueError: If the schema is not a Pydantic model or factory is not callable.
        """
        if (
            not isinstance(self.instance_model, type)
            or not issubclass(self.instance_model, BaseModel)
            or not callable(self.build)
        ):
            raise ValueError("Bind an installed Pydantic schema and callable problem factory.")


class JobOptions(BaseModel):
    """Expose only bounded optimization settings, not worker/import/filesystem control.

    Responsibilities:
    - Reject unexpected settings and use only installed algorithm parameter defaults.

    Attributes:
        algorithms (list[str]): At most three registered canonical algorithms.
        adaptation (Literal): Fixed or adaptive parameter selection.
        time_limit (float): Per-algorithm cooperative sequential allowance.
        random_state (int): Explicit reproducibility seed.
        validate_solution (bool): Request one post-run check when supplied by the adapter.
        include_solution (bool): Request trusted serialization of final reconstruction.
        report (bool): Request an optional offline report, subject to server permission.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    algorithms: list[str] = Field(default=["SA"], min_length=1, max_length=3)
    adaptation: Literal["fixed", "adaptive"] = "fixed"
    time_limit: float = Field(default=0.2, gt=0, le=30)
    random_state: int = Field(default=42, ge=0, le=2**32 - 1)
    validate_solution: bool = False
    include_solution: bool = False
    report: bool = False

    def checked(self, limits: ServiceLimits) -> JobOptions:
        """Check dynamic server caps and registered algorithm names before admission.

        Args:
            limits (ServiceLimits): Trusted server policy.

        Returns:
            JobOptions: Same validated request settings.

        Raises:
            ValueError: If duplicates, server caps or algorithm names are invalid.
        """
        if len(set(self.algorithms)) != len(self.algorithms):
            raise ValueError("Algorithms cannot be duplicated.")
        if (
            len(self.algorithms) > limits.max_algorithms
            or self.time_limit > limits.max_search_seconds
        ):
            raise ValueError("Requested search exceeds server policy.")
        if self.report and not limits.allow_reports:
            raise ValueError("Reports are not enabled.")
        if any(name not in MH_SPECS for name in self.algorithms):
            raise ValueError("Only registered canonical algorithm names are allowed.")
        return self


class JobSubmission(BaseModel):
    """Accept JSON instance values and allowlisted settings only.

    Responsibilities:
    - Keep source code, import targets, host paths and serialization protocols out of the API.

    Attributes:
        instance (dict[str, JsonValue]): Application values validated by the installed schema.
        options (JobOptions): Bounded numerical/post-run options.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    instance: dict[str, JsonValue]
    options: JobOptions = Field(default_factory=JobOptions)
