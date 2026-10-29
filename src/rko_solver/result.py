"""Immutable portable results for ordinary and generated-model optimization.

Objective:
- Preserve actual search outcomes independently of mutable solver configuration.

Technical approach:
- Store owned result fields and derive lifecycle summaries from completed metadata.

Flow (pseudocode):
1. Own completed search-score and optional generated-model evidence.
2. Aggregate successful evaluation counts and stop reasons.
3. Expose readable adaptation metadata without replaying callbacks.

Important criteria:
- Direct fixed models report one evaluation and no search workers.

Inputs/Outputs:
- Input: completed portable search and optional model records.
- Output: immutable public optimization result.
"""

from __future__ import annotations

from .modeling.results import ModelResult
from .orchestration.adaptation import Adaptation
from .orchestration.adaptation import adaptation_name
from .orchestration.auto import AutoSelection
from .orchestration.types import PoolInitializationMetadata
from .orchestration.types import SearchMetadata
from .tracing import DecisionTrace
from dataclasses import dataclass


@dataclass(frozen=True)
class RKOResult:
    """Describe a completed optimization without mutable algorithm state.

    Responsibilities:
    - Preserve best keys and objective together with measured execution metadata.

    Attributes:
        keys (tuple[float, ...]): Best random-key vector.
        cost (float): Objective to minimize, evaluated on the decoded keys.
        elapsed_seconds (float): End-to-end solve duration including startup.
        engine (str): Executed engine: python, rust or cpp.
        algorithms (tuple[str, ...]): Canonical portfolio names.
        random_state (int | None): Requested seed; timed results may still vary.
        searches (tuple[SearchMetadata, ...]): All algorithms in portfolio order.
        n_variables (int | None): Requested dimension, when recorded.
        time_limit (float | None): Requested search seconds, excluding startup.
        control (int | None): Requested portfolio control; reports give effective control.
        pool_size (int | None): Configured elite capacity.
        n_jobs (int | None): Requested process-worker cap; for the Rust and C++ engines, the
            worker threads actually used.
        versions (tuple): RKO distribution, Python and NumPy runtime versions.
        available_cpus (int | None): Usable logical CPUs snapshotted for process execution.
        initialization (PoolInitializationMetadata | None): Separate opt-in startup work.
        trace (DecisionTrace | None): Selected original-decision evidence or unavailable status.
        auto (AutoSelection | None): How auto selected the engine, absent for generated models.
        execution (str | None): Python scheduling, or None for native engines/direct evaluation.
        max_evaluations (int | None): Requested per-algorithm evaluation limit, if any.
        model (ModelResult | None): Separate model witnesses; None for ordinary callbacks.
    """

    keys: tuple[float, ...]
    cost: float
    elapsed_seconds: float
    engine: str
    algorithms: tuple[str, ...]
    random_state: int | None
    searches: tuple[SearchMetadata, ...] = ()
    n_variables: int | None = None
    time_limit: float | None = None
    control: int | None = None
    pool_size: int | None = None
    n_jobs: int | None = None
    versions: tuple[tuple[str, str], ...] = ()
    available_cpus: int | None = None
    initialization: PoolInitializationMetadata | None = None
    trace: DecisionTrace | None = None
    auto: AutoSelection | None = None
    execution: str | None = None
    max_evaluations: int | None = None
    model: ModelResult | None = None

    @property
    def n_evaluations(self) -> int | None:
        """Return completed finite objectives, or unknown for legacy manual results.

        Returns:
            int | None: Completed startup plus all searches, or unknown for legacy results.
        """
        if self.model is not None and self.model.no_search:
            return self.model.direct_evaluations
        if not self.searches and self.initialization is None:
            return None
        return sum(report.n_evaluations for report in self.searches) + (
            self.initialization.n_evaluations if self.initialization is not None else 0
        )

    @property
    def stop_reason(self) -> str:
        """Summarize whether any search exhausted its assigned budget.

        Returns:
            str: Cancelled, time_limit, max_evaluations, completed, or unknown without reports.
                The first applicable value in that order wins, so max_evaluations means every
                search that was stopped reached its evaluation limit before the deadline.
        """
        if self.model is not None and self.model.no_search:
            return "completed"
        reasons = {report.stop_reason for report in self.searches}
        if "cancelled" in reasons:
            return "cancelled"
        if reasons.intersection(("time_limit", "not_started")):
            return "time_limit"
        if "max_evaluations" in reasons:
            return "max_evaluations"
        return "completed" if reasons else "unknown"

    @property
    def n_workers(self) -> int:
        """Return search-worker concurrency, excluding managers and native threads.

        Returns:
            int: Configured execution workers rather than CPU-core utilization.
        """
        if self.model is not None and self.model.no_search:
            return 0
        if self.execution == "sequential":
            return 1
        limit = self.n_jobs
        if limit is None and self.execution == "process":
            limit = self.available_cpus
        return min(limit or len(self.algorithms), len(self.algorithms))

    @property
    def adaptation(self) -> Adaptation | None:
        """Return the public adaptation name for the requested control mode.

        Returns:
            Adaptation | None: "fixed", "adaptive", or None when unknown.
        """
        return adaptation_name(self.control)
