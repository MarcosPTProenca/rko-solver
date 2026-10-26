"""Shared execution and parameter mapping contracts.

Objective:
- Name callback, result and configuration types used across engines.

Technical approach:
- Use immutable configuration records and structural callback/mapping aliases.

Flow (pseudocode):
1. Describe problem callbacks and selected algorithm schemas.
2. Normalize configuration and partial overrides before engine dispatch.
3. Exchange compatible result tuples across serial, thread and process modes.

Important criteria:
- Hyperparameter mappings are read-only inputs, normalized to plain dictionaries.
- Existing tuple result and callback contracts remain unchanged.

Inputs/Outputs:
- Input: callback functions, named parameter mappings and run settings.
- Output: shared type aliases and immutable configuration/specification records.
"""

from __future__ import annotations

import numpy as np
from ..metaheuristics.base import MetaheuristicBase
from ..modeling.linear import ModelEvidence
from ..tracing import TraceConfig
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from typing import Literal
from typing import Protocol
from typing import Required
from typing import TypedDict

RunResult = tuple[list[object], list[float], float]


class ModelEvidenceOptions(TypedDict, total=False):
    """Describe optional generated-model evidence forwarding.

    Responsibilities:
    - Keep bounded collector forwarding separate from trace configuration.

    Attributes:
        model_evidence (list[ModelEvidence] | None): Completed witness collector.
    """

    model_evidence: list[ModelEvidence] | None


class WorkerCaptureOptions(TypedDict, total=False):
    """Describe optional process-worker capture and successful-completion controls.

    Responsibilities:
    - Keep bounded generated-model transport independent of positional worker arguments.

    Attributes:
        max_evaluations (int | None): Per-search successful-completion cap.
        collect_model_evidence (bool): Whether to transport bounded generated-model witnesses.
    """

    max_evaluations: int | None
    collect_model_evidence: bool


class ExecutionMetadata(ModelEvidenceOptions, total=False):
    """Describe lifecycle context forwarded to the private execution boundary.

    Responsibilities:
    - Preserve the required monotonic origin and optional generated-model archive.

    Attributes:
        started_at (float): Parent monotonic solve origin.
        model_evidence (list[ModelEvidence] | None): Bounded completed-model collector.
    """

    started_at: Required[float]


class ModelCallbackOptions(ModelEvidenceOptions, total=False):
    """Type the optional trace and bounded model-evidence forwarding keys.

    Responsibilities:
    - Preserve optional keyword forwarding without erasing callback collector types.

    Attributes:
        trace (TraceConfig): Explicit bounded trace configuration when enabled.
        model_evidence (list[ModelEvidence]): Owned collector for completed model witnesses.
    """

    trace: TraceConfig | None


DecoderFn = Callable[[np.ndarray], Any]
CostFn = Callable[[Any], float]
Hyperparameters = Mapping[str, Mapping[str, int | float]]


@dataclass(frozen=True, slots=True)
class HistoryPoint:
    """Retain one scalar incumbent observation without candidate payloads.

    Responsibilities:
    - Keep a local search timestamp, evaluation index and minimization incumbent.

    Attributes:
        elapsed_seconds (float): Seconds since this search session started.
        n_evaluations (int): Completed finite objectives when observed.
        cost (float): Best finite objective observed by this search session.
    """

    elapsed_seconds: float
    n_evaluations: int
    cost: float


@dataclass(frozen=True)
class SearchMetadata:
    """Preserve one algorithm's initial configuration and completed work.

    Responsibilities:
    - Snapshot validated initial scalar parameters and ordered adaptive axes.
    - Report finite objective completions without retaining candidate populations.

    Attributes:
        algorithm (str): Canonical portfolio name.
        control (int): Effective control; unsupported adaptive algorithms use zero.
        parameters (tuple): Named initial scalars, including Python overrides.
        parameter_grid (tuple): Adaptive axes in algorithm schema order, not a trajectory.
        random_state (int | None): Actual assigned generator seed when known.
        n_evaluations (int): Successfully completed finite objective evaluations.
        stop_reason (str): Completed, time_limit, max_evaluations, cancelled, or not_started.
        time_limit (float): Assigned search seconds excluding startup; zero if not started.
        elapsed_seconds (float): Search/session duration, excluding startup and queueing.
        history (tuple[HistoryPoint, ...]): Bounded incumbent samples in local search time.
        history_limit (int): Requested per-search cap; zero disables recording.
        history_observations (int): Samples observed before deterministic compaction.
    """

    algorithm: str
    control: int = 0
    parameters: tuple[tuple[str, int | float], ...] = ()
    parameter_grid: tuple[tuple[int | float, ...], ...] = ()
    random_state: int | None = None
    n_evaluations: int = 0
    stop_reason: Literal[
        "completed", "time_limit", "max_evaluations", "cancelled", "not_started"
    ] = "not_started"
    time_limit: float = 0.0
    elapsed_seconds: float = 0.0
    history: tuple[HistoryPoint, ...] = ()
    history_limit: int = 0
    history_observations: int = 0


@dataclass(frozen=True)
class PoolInitializationMetadata:
    """Report shared startup work independently of algorithm search reports.

    Responsibilities:
    - Account for completed startup evaluations and their owned seed exactly once.

    Attributes:
        random_state (int): Actual initialization stream seed.
        n_evaluations (int): Completed finite objective observations.
        elapsed_seconds (float): Initialization duration, including pool publication.
        pool_size (int): Admitted distinct chromosomes before searches start.
        stop_reason (str): Completed or cooperative time limit.
        time_limit (float): Independent startup allowance, not deducted from searches.
    """

    random_state: int
    n_evaluations: int
    elapsed_seconds: float
    pool_size: int
    stop_reason: Literal["completed", "time_limit"]
    time_limit: float


@dataclass(frozen=True)
class MhSpec:
    """Declarative specification for constructing a metaheuristic."""

    constructor: Callable[..., MetaheuristicBase]
    int_params: tuple[str, ...] = ()
    float_params: tuple[str, ...] = ()
    supports_control: bool = False


@dataclass(frozen=True)
class RunConfig:
    """Validated execution settings extracted from ``config.toml``.

    Responsibilities:
    - Carry shared capacity/control and optional startup policy into every engine.

    Attributes:
        control (int): Offline or adaptive search control.
        pool_size (int): Shared elite capacity, also startup slot count when opted in.
        active_metaheuristics (tuple[str, ...]): Canonical selected algorithm names.
        prefill_pool (bool): Enable a separately accounted reference-style prefix.
    """

    control: int
    pool_size: int
    active_metaheuristics: tuple[str, ...]
    prefill_pool: bool = False


class NativeProblem(Protocol):
    """Describe a built-in problem evaluated by the compiled engine.

    Responsibilities:
    - Expose the random-key dimension required to size a portfolio.

    Attributes:
        dimension (int): Positive number of random keys.
    """

    @property
    def dimension(self) -> int:
        """Return the positive random-key dimension.

        Returns:
            int: Number of keys expected by the problem.
        """
        ...
