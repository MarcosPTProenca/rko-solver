"""Cooperative stopping at the shared decoder/objective boundary.

Objective:
- Bound nested search work and preserve the best evaluated candidate on stopping.

Technical approach:
- Carry a monotonic deadline and an optional evaluation limit through the callbacks already
  passed to every operator.
- Pair decoded payloads with copied keys without changing user callback payloads.
- Optionally retain bounded scalar incumbent samples, never candidate histories.
- Stop between callbacks and retain completed evaluations, including initialization.

Flow (pseudocode):
1. Bind a fresh session to one algorithm's decoder and objective.
2. Check deadline, evaluation limit and cancellation before each callback after the first
   valid result.
3. Catch internal budget exhaustion and return the best completed candidate.

Important criteria:
- User callbacks remain trusted and cannot be interrupted while executing.
- A first evaluation is permitted even if initialization consumed the budget.
- The evaluation limit is exact: no objective call is made once `n_evaluations` reaches it.
- Sessions are private to one sequential search, never shared between workers.

Inputs/Outputs:
- Input: algorithm, positive duration, optional absolute deadline, evaluation limit and
  cancellation event.
- Output: incumbent-preserving algorithm result or the original callback exception.
"""

from __future__ import annotations

import hashlib
import math
import numpy as np
import threading
import time
from ..modeling.construction import ModelConstruction
from ..modeling.linear import CompiledLinearModel
from ..modeling.linear import ModelEvaluation
from ..modeling.linear import ModelEvidence
from ..modeling.records import ModelStatistics
from ..tracing import _ACTIVE_STORE
from ..tracing import _CURRENT
from ..tracing import TraceConfig
from ..tracing import _ObservedCost
from ..tracing import _TraceCell
from ..tracing import _TraceStore
from .types import CostFn
from .types import DecoderFn
from .types import HistoryPoint
from .types import ModelCallbackOptions
from .types import RunResult
from .types import SearchMetadata
from dataclasses import dataclass
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from ..components.random_streams import RandomStreams
    from ..metaheuristics.base import MetaheuristicBase


class _BudgetExpired(Exception):
    """Signal cooperative stopping without treating it as an algorithm failure."""


@dataclass
class _DecodedCandidate:
    """Keep objective payload and its originating chromosome associated.

    Responsibilities:
    - Preserve keys independently of temporary arrays reused by local searches.

    Attributes:
        keys (np.ndarray): Detached originating chromosome.
        payload (Any): Exact decoded value passed to the user's objective.
        trace (_TraceCell | None): Bounded identity/event reference for this invocation only.
    """

    keys: np.ndarray
    payload: Any
    trace: _TraceCell | None = None


class SearchSession:
    """Own one search's callback deadline and completed incumbent.

    Responsibilities:
    - Apply the same stopping criterion through all nested operators.
    - Preserve a usable result when a population or local search is interrupted.

    Attributes:
        decoder (DecoderFn): Original problem decoder.
        cost (CostFn): Original objective callback.
        deadline (float): Absolute monotonic stop time.
        cancelled (threading.Event | None): Optional portfolio failure signal.
        max_evaluations (int | None): Completed evaluations after which the search stops.
        best (tuple[np.ndarray, float] | None): Detached best completed evaluation.
        n_evaluations (int): Completed finite objective evaluations in this search.
        metadata (SearchMetadata | None): Immutable report after a successful search.
    """

    def __init__(
        self,
        decoder: DecoderFn,
        cost: CostFn,
        *,
        trace: TraceConfig | None = None,
        trace_source: str = "search",
    ) -> None:
        """Retain trusted problem callbacks without executing them.

        Args:
            decoder (DecoderFn): Original random-key decoder.
            cost (CostFn): Original minimization objective.
            trace (TraceConfig | None): Opt-in bounded original-decoder recording.
            trace_source (str): Initial phase/search label.
        """
        owner = getattr(decoder, "__self__", None)
        self._model = (
            owner
            if isinstance(owner, (CompiledLinearModel, ModelConstruction)) and cost is owner.cost
            else None
        )
        self._model_incumbent: ModelEvaluation | None = None
        self._model_feasible: ModelEvaluation | None = None
        self._model_first_feasible_at: float | None = None
        self._model_infeasible = 0
        self._model_outputs: set[str] | None = (
            set() if isinstance(self._model, ModelConstruction) else None
        )
        self._trace_config = trace
        self._trace_store = _TraceStore(trace, trace_source) if trace is not None else None
        self.decoder = (
            partial(self._model.decode, _budget_check=self._check)
            if self._model is not None
            and (isinstance(self._model, ModelConstruction) or self._model._repair_free)
            else decoder
        )
        self.cost = cost
        self.deadline = math.inf
        self.cancelled: threading.Event | None = None
        self.max_evaluations: int | None = None
        self.best: tuple[np.ndarray, float] | None = None
        self.n_evaluations = 0
        self.metadata: SearchMetadata | None = None
        self._stop_reason = "completed"
        self._history_limit = 0
        self._history: list[HistoryPoint] = []
        self._history_observations = 0
        self._started_at = 0.0

    @property
    def model_evidence(self) -> ModelEvidence | None:
        """Return only bounded successfully completed generated-model witnesses.

        Returns:
            ModelEvidence | None: Score and independent feasible incumbents, or None
                for ordinary callbacks and sessions without a completed evaluation.
        """
        if self._model_incumbent is None:
            return None
        return ModelEvidence(
            self._model_incumbent,
            self._model_feasible,
            self._model_first_feasible_at,
            ModelStatistics(
                self.n_evaluations,
                self._model_infeasible,
                tuple(sorted(self._model_outputs)) if self._model_outputs is not None else None,
            ),
        )

    def _record_history(self, elapsed: float) -> None:
        """Retain an incumbent sample and compact deterministically at the configured cap.

        Args:
            elapsed (float): Genuine elapsed search time of this observation.
        """
        if self.best is None:
            return
        self._history_observations += 1
        self._history.append(HistoryPoint(elapsed, self.n_evaluations, float(self.best[1])))
        if len(self._history) > self._history_limit:
            self._history = self._history[:1] + self._history[2:-1:2] + self._history[-1:]

    def set_deadline(self, deadline: float) -> None:
        """Bind an invocation deadline without renewing a constructed model's owner budget.

        Args:
            deadline (float): Absolute monotonic session deadline.
        """
        if isinstance(self._model, ModelConstruction) and self._model.deadline is not None:
            deadline = min(deadline, self._model.deadline)
        self.deadline = deadline

    def _check(self) -> None:
        """Stop after a valid result exists or when another worker failed.

        Raises:
            _BudgetExpired: If cancellation, the evaluation limit or the deadline is reached.
        """
        if self.cancelled is not None and self.cancelled.is_set():
            self._stop_reason = "cancelled"
            raise _BudgetExpired
        if self.max_evaluations is not None and self.n_evaluations >= self.max_evaluations:
            self._stop_reason = "max_evaluations"
            raise _BudgetExpired
        if self.best is not None and time.perf_counter() >= self.deadline:
            self._stop_reason = "time_limit"
            raise _BudgetExpired

    def decode(self, keys: np.ndarray) -> _DecodedCandidate:
        """Decode a candidate while retaining an independent key snapshot.

        Args:
            keys (np.ndarray): Candidate chromosome.

        Returns:
            _DecodedCandidate: Keys paired with the original decoded payload.

        Raises:
            _BudgetExpired: If work must stop before decoding.
        """
        self._check()
        snapshot = np.asarray(keys, dtype=float).copy()
        cell = self._trace_store.begin() if self._trace_store is not None else None
        if cell is None and _CURRENT.get() is None:
            return _DecodedCandidate(snapshot, self.decoder(snapshot.copy()))
        token = _CURRENT.set(cell)
        try:
            return _DecodedCandidate(snapshot, self.decoder(snapshot.copy()), cell)
        finally:
            _CURRENT.reset(token)

    def evaluate(self, candidate: _DecodedCandidate) -> float:
        """Evaluate the original payload and retain the best completed result.

        Args:
            candidate (_DecodedCandidate): Decoded payload and originating keys.

        Returns:
            float: Finite minimization objective.

        Raises:
            _BudgetExpired: If work must stop before the objective call.
            ValueError: If the objective returns a nonfinite value.
        """
        self._check()
        token = _CURRENT.set(None) if _CURRENT.get() is not None else None
        try:
            value = float(self.cost(candidate.payload))
        finally:
            if token is not None:
                _CURRENT.reset(token)
        if not math.isfinite(value):
            raise ValueError("Objective callbacks must return finite costs.")
        self.n_evaluations += 1
        if self._model is not None:
            self._record_model(candidate.payload, value)
        if candidate.trace is not None:
            candidate.trace.seal(candidate.keys, value)
            value = _ObservedCost(value, candidate.trace)
        if self.best is None or value < self.best[1]:
            self._record_improvement(candidate, value)
        return value

    def _record_improvement(self, candidate: _DecodedCandidate, value: float) -> None:
        """Retain a strict completed improvement and update optional bounded capture.

        Args:
            candidate (_DecodedCandidate): Completed payload and its originating keys.
            value (float): Finite strictly improving observed score.
        """
        if self.best is not None and isinstance(self.best[1], _ObservedCost):
            self.best[1]._cell.pinned = False
        if candidate.trace is not None:
            candidate.trace.pinned = True
        self.best = (candidate.keys, value)
        if self._history_limit:
            self._record_history(time.perf_counter() - self._started_at)

    def solve(
        self,
        algorithm: MetaheuristicBase,
        duration: float,
        *,
        deadline: float | None = None,
        cancelled: threading.Event | None = None,
        random_state: int | None = None,
        history_limit: int = 0,
        max_evaluations: int | None = None,
    ) -> RunResult:
        """Execute an algorithm using budget-aware callbacks and preserve its incumbent.

        Args:
            algorithm (MetaheuristicBase): Fresh algorithm to execute.
            duration (float): Positive search seconds passed to its solve method.
            deadline (float | None): Common absolute deadline, or duration from now.
            cancelled (threading.Event | None): Shared peer-failure signal.
            random_state (int | None): Actual assigned seed for this search report.
            history_limit (int): Per-search scalar sample cap, zero disables capture.
            max_evaluations (int | None): Positive completed-evaluation limit, or None.

        Returns:
            RunResult: Extras, best evaluated keys and objective; an unevaluated
                cancelled peer has empty keys and infinite cost.

        Raises:
            ValueError: If the history cap or the evaluation limit is invalid.
            Exception: Original decoder, objective or algorithm failure is propagated.
        """
        if type(history_limit) is not int or (
            history_limit != 0 and not 2 <= history_limit <= 10000
        ):
            raise ValueError("history_limit must be 0 or an integer from 2 to 10000.")
        if max_evaluations is not None and (
            type(max_evaluations) is not int or max_evaluations < 1
        ):
            raise ValueError("max_evaluations must be a positive integer or None.")
        if self._trace_config is not None:
            self._trace_store = _TraceStore(self._trace_config, algorithm.name)
        started = time.perf_counter()
        self._started_at = started
        self._history_limit = history_limit
        self._history = []
        self._history_observations = 0
        self.best = None
        self._model_incumbent = None
        self._model_feasible = None
        self._model_first_feasible_at = None
        self._reset_model_statistics()
        self.n_evaluations = 0
        self.metadata = None
        self._stop_reason = "completed"
        self.set_deadline(started + duration if deadline is None else deadline)
        self.cancelled = cancelled
        self.max_evaluations = max_evaluations
        original_decoder, original_cost = algorithm.decoder, algorithm.cost
        original_cancelled = algorithm._cancelled
        algorithm.decoder, algorithm.cost = self.decode, self.evaluate
        algorithm._cancelled = cancelled
        trace_token = _ACTIVE_STORE.set(self._trace_store)
        try:
            try:
                result = algorithm.solve(duration)
            except _BudgetExpired:
                result = ([], [], math.inf)
            result = self._selected_result(result)
            finished = time.perf_counter()
            reason = self._stop_reason
            if reason == "completed":
                if cancelled is not None and cancelled.is_set():
                    reason = "cancelled"
                elif finished >= self.deadline:
                    reason = "time_limit"
            if history_limit:
                self._record_history(finished - started)
            self.metadata = replace(
                getattr(algorithm, "_search_metadata", None)
                or SearchMetadata(algorithm=algorithm.name),
                random_state=random_state,
                n_evaluations=self.n_evaluations,
                stop_reason=reason,
                time_limit=duration,
                elapsed_seconds=finished - started,
                history=tuple(self._history),
                history_limit=history_limit,
                history_observations=self._history_observations,
            )
            return result
        finally:
            _ACTIVE_STORE.reset(trace_token)
            algorithm.decoder, algorithm.cost = original_decoder, original_cost
            algorithm._cancelled = original_cancelled

    def _record_model(self, evidence: ModelEvaluation, value: float) -> None:
        """Admit only completed generated-model invocation evidence.

        Args:
            evidence (ModelEvaluation): Original decoded payload; ignored for ordinary callbacks.
            value (float): Successfully completed finite scalar score.
        """
        if self._model is not None:
            self._record_model_statistics(evidence)
            if evidence.feasible and (
                self._model_feasible is None
                or (
                    evidence.objective > self._model_feasible.objective
                    if self._model.maximize
                    else evidence.objective < self._model_feasible.objective
                )
            ):
                if self._model_feasible is None:
                    self._model_first_feasible_at = time.perf_counter()
                self._model_feasible = evidence
            if self._model_incumbent is None or value < self._model_incumbent.score:
                self._model_incumbent = evidence

    def _reset_model_statistics(self) -> None:
        """Reset completed model counts and optional assignment identities for a new solve."""
        self._model_infeasible = 0
        if self._model_outputs is not None:
            self._model_outputs.clear()

    def _record_model_statistics(self, evidence: ModelEvaluation) -> None:
        """Retain completed counts and opt-in constructed-output hashes without trajectories.

        Args:
            evidence (ModelEvaluation): Protected successfully completed model payload.
        """
        self._model_infeasible += not evidence.feasible
        if self._model_outputs is not None and len(self._model_outputs) < 256:
            self._model_outputs.add(
                hashlib.sha256((np.asarray(evidence.values) + 0.0).tobytes()).hexdigest()
            )

    def _selected_result(self, result: RunResult) -> RunResult:
        """Keep exact completed generated-model identity even when scores tie.

        Args:
            result (RunResult): Algorithm-returned scalar search outcome.

        Returns:
            RunResult: Completed observed incumbent when required by score or model identity.
        """
        if self.best is not None and (
            self.best[1] < result[2] or (self._model is not None and self.best[1] == result[2])
        ):
            result = (result[0], self.best[0].tolist(), self.best[1])
        return result


class SearchReporting:
    """Forward optional capture and completed report data outside execution sessions.

    Responsibilities:
    - Keep ordinary callback execution independent of disabled model-evidence capture.
    - Preserve bounded generated-model witnesses and unstarted queued metadata.
    """

    @staticmethod
    def collect(session: SearchSession, collector: list[ModelEvidence] | None) -> None:
        """Append only a completed model witness when capture is explicitly enabled.

        Args:
            session (SearchSession): Completed owned search session.
            collector (list[ModelEvidence] | None): Optional bounded solve/worker archive.
        """
        if collector is None:
            return
        evidence = session.model_evidence
        if evidence is not None:
            collector.append(evidence)

    @staticmethod
    def callback_options(
        trace: TraceConfig | None, collector: list[ModelEvidence] | None
    ) -> ModelCallbackOptions:
        """Build optional generated-callback metadata without altering ordinary call signatures.

        Args:
            trace (TraceConfig | None): Explicit original-decision capture policy.
            collector (list[ModelEvidence] | None): Optional bounded completed-model archive.

        Returns:
            ModelCallbackOptions: Only explicitly enabled trace/evidence forwarding keys.
        """
        options: ModelCallbackOptions = {}
        if trace is not None:
            options["trace"] = trace
        if collector is not None:
            options["model_evidence"] = collector
        return options

    @staticmethod
    def queued_reports(
        queued: list[tuple[MetaheuristicBase, RandomStreams]], history_limit: int
    ) -> list[SearchMetadata]:
        """Describe unstarted queued searches without manufacturing evaluation evidence.

        Args:
            queued (list[tuple]): Fresh queued algorithms and their independent streams.
            history_limit (int): Requested bounded scalar history cap.

        Returns:
            list[SearchMetadata]: Zero-time unstarted records in portfolio order.
        """
        return [
            replace(
                algorithm._search_metadata or SearchMetadata(algorithm=algorithm.name),
                random_state=stream.seed,
                time_limit=0.0,
                history_limit=history_limit,
            )
            for algorithm, stream in queued
        ]
