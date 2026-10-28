"""Verify original-observation identity and bounded event storage before engine integration.

Objective:
- Reject guessed key/cost provenance and prove bounded identity transport through existing tuples.

Technical approach:
- Construct distinct equal-valued observations and exercise cache/pickle/pool selection paths.

Flow (pseudocode):
1. Record explicit bounded events for completed observations.
2. Carry their numeric references through caches and trusted pickle transport.
3. Discard a reference deliberately and require unavailable evidence instead of matching.

Important criteria:
- Equal numerical output is not a unique evaluation identity.
- No test relies on callback replay or elapsed-time equality.

Inputs/Outputs:
- Input: synthetic observations with identical coordinates/cost and distinct events.
- Output: identity, retention and resource-bound assertions.
"""

import numpy as np
import pickle
import pytest
import threading
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from rko_solver import metaheuristics
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.orchestration.execution import select_best_result
from rko_solver.orchestration.factory import build_metaheuristic
from rko_solver.orchestration.search import SearchSession
from rko_solver.reporting import OptimizationReport
from rko_solver.solver import RKO
from rko_solver.tracing import _ACTIVE_STORE
from rko_solver.tracing import _CURRENT
from rko_solver.tracing import DecisionEvent
from rko_solver.tracing import DecisionTrace
from rko_solver.tracing import TraceConfig
from rko_solver.tracing import _ObservedCost
from rko_solver.tracing import _TraceStore
from rko_solver.tracing import record_decision


def observation(store: _TraceStore, action: str, value: float = 1.0) -> _ObservedCost:
    """Construct one genuine distinct finite observation over fixed synthetic keys.

    Args:
        store (_TraceStore): Bounded private session storage.
        action (str): Distinguishing authored event.
        value (float): Finite original objective.

    Returns:
        _ObservedCost: Internal identity-preserving numerical cost.
    """
    cell = store.begin()
    cell.emit(action, "explicit decision")
    cell.seal([0.5], value)
    return _ObservedCost(value, cell)


def test_equal_values_keep_distinct_identities_through_cache_and_pickle() -> None:
    """Preserve the selected invocation, not merely another equal-cost matching trace."""
    store = _TraceStore(TraceConfig(), "SA")
    first, second = observation(store, "first"), observation(store, "second")
    assert first == second == 1.0
    assert first._cell.identity != second._cell.identity
    copied = pickle.loads(pickle.dumps([first, second]))
    keys, value = select_best_result([([], [0.5], copied[1])], SolutionPool())
    trace = DecisionTrace.selected(value, keys)
    assert trace.evaluation_id == second._cell.identity
    assert trace.events[0].action == "second"
    assert type(float(value)) is float


def test_lost_identity_is_not_recovered_from_matching_keys_and_cost() -> None:
    """Reject evidence recovery after numeric coercion, even with an exact matching observation."""
    store = _TraceStore(TraceConfig(), "SA")
    value = observation(store, "must not be borrowed")
    trace = DecisionTrace.selected(float(value), [0.5])
    assert trace.state == "unavailable"
    assert trace.reason == "identity_unavailable"
    assert not trace.events
    assert DecisionTrace.selected(value, [0.25]).reason == "binding_mismatch"


def test_population_references_do_not_retain_evicted_event_payloads() -> None:
    """Bound event storage even when a population retains arbitrarily many numeric references."""
    store = _TraceStore(TraceConfig(max_records=2), "PSO")
    population = [observation(store, str(index)) for index in range(100)]
    assert len(store.cells) == 2
    assert sum(value._cell.witness is not None for value in population) == 2
    assert DecisionTrace.selected(population[0], [0.5]).reason == "retention_limit"
    assert DecisionTrace.selected(population[-1], [0.5]).events[0].action == "99"


@pytest.mark.parametrize("transport", ["local", "pickle"])
def test_imported_elites_share_the_active_session_payload_budget(transport: str) -> None:
    """Prevent numeric population caches from retaining an archive of imported elite events.

    Args:
        transport (str): Local shared-pool read or simulated owned process unpickling.
    """
    writer = _TraceStore(TraceConfig(max_records=2), "producer")
    reader = _TraceStore(TraceConfig(max_records=2), "consumer")
    pool = SolutionPool()
    population = []
    token = _ACTIVE_STORE.set(reader)
    try:
        for index in range(100):
            value = observation(writer, str(index), -float(index))
            pool.insert((value, [0.5]))
            copied = pool.get_elite(1)[0][0]
            population.append(
                pickle.loads(pickle.dumps(copied)) if transport == "pickle" else copied
            )
    finally:
        _ACTIVE_STORE.reset(token)
    assert len(reader.cells) == 2
    assert sum(value._cell.witness is not None for value in population) <= 2
    assert DecisionTrace.selected(population[0], [0.5]).reason == "retention_limit"
    assert DecisionTrace.selected(population[-1], [0.5]).events[0].action == "99"


def test_count_and_encoded_byte_caps_are_explicit() -> None:
    """Truncate authored events rather than silently exceed count or JSON byte caps."""
    store = _TraceStore(TraceConfig(max_events=1, max_bytes=128), "SA")
    cell = store.begin()
    cell.emit("accepted", "reason")
    cell.emit("omitted", "another reason")
    cell.seal([0.5], 1.0)
    trace = cell.snapshot()
    assert trace.state == "truncated" and trace.dropped_events == 1
    assert len(trace.events) == 1
    tiny = _TraceStore(TraceConfig(max_bytes=64), "SA").begin()
    tiny.emit("x" * 100, "")
    tiny.seal([0.5], 1.0)
    assert tiny.snapshot().state == "truncated"
    assert tiny.snapshot().events == ()


@pytest.mark.parametrize(
    "field,value", [("max_events", False), ("max_records", 1), ("max_bytes", 63)]
)
def test_configuration_rejects_invalid_caps(field: str, value: object) -> None:
    """Reject malformed bounded-policy settings before any callback can execute.

    Args:
        field (str): Tracing configuration field.
        value (object): Invalid cap.
    """
    with pytest.raises(ValueError):
        TraceConfig(**{field: value})


@dataclass
class StatefulProblem:
    """Make original-evaluation provenance observable without replay or payload snapshots.

    Responsibilities:
    - Emit an explicit invocation marker and return its monotonically improving cost.

    Attributes:
        calls (int): Number of actual decoder invocations in this process.
        objectives (int): Number of actual original objective calls.
    """

    calls: int = 0
    objectives: int = 0

    def decode(self, keys: np.ndarray) -> int:
        """Emit the actual invocation counter rather than a reconstruction marker.

        Args:
            keys (np.ndarray): Original candidate; intentionally not serialized into events.

        Returns:
            int: Original stateful payload.
        """
        self.calls += 1
        invocation = self.calls
        record_decision("invocation", str(invocation))
        return invocation

    def cost(self, payload: int) -> float:
        """Score the original stateful payload without decoding again.

        Args:
            payload (int): Exact value returned by that decoder invocation.

        Returns:
            float: Finite monotonically improving objective.
        """
        self.objectives += 1
        record_decision("objective", "must not enter the decoder-only witness")
        return -float(payload)


@pytest.mark.parametrize(
    "algorithm",
    ["SA", "MultiStart", "ILS", "VNS", "LNS", "GRASP", "BRKGA", "BRKGA_CS", "PSO", "GA", "IPR"],
)
def test_all_public_algorithms_preserve_original_witness(algorithm: str) -> None:
    """Preserve actual stateful decoder evidence through every public algorithm's caches.

    Args:
        algorithm (str): Public canonical algorithm name.
    """
    problem = StatefulProblem()
    solver_type = getattr(metaheuristics, "GeneticAlgorithm" if algorithm == "GA" else algorithm)
    result = solver_type(
        problem.decode,
        problem.cost,
        4,
        execution="sequential",
        adaptation="fixed",
        trace=TraceConfig(),
        random_state=42,
    ).solve(time_limit=0.01)
    assert result.trace is not None
    assert result.trace.state == "recorded", result.trace
    assert result.trace.matches(result.keys, result.cost)
    assert int(result.trace.events[0].reason) == -result.cost
    assert problem.objectives == result.n_evaluations
    assert type(result.cost) is float


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
@pytest.mark.parametrize("prefill", [False, True])
def test_portfolio_transport_and_initialization(execution: str, prefill: bool) -> None:
    """Preserve bounded identities across real execution modes, queued searches and prefill.

    Args:
        execution (str): Actual execution mode.
        prefill (bool): Whether separate initialization can produce the final winner.
    """
    problem = StatefulProblem()
    result = RKO(
        problem.decode,
        problem.cost,
        4,
        algorithms=["SA", "PSO", "GA"],
        execution=execution,
        n_jobs=1 if execution == "process" else None,
        prefill_pool=prefill,
        adaptation="fixed",
        random_state=42,
        history_limit=4,
        trace=TraceConfig(max_records=2),
    ).solve(time_limit=0.1)
    assert result.trace is not None and result.trace.state == "recorded", result.trace
    assert result.trace.matches(result.keys, result.cost)
    assert int(result.trace.events[0].reason) == -result.cost
    assert len(result.trace.events) == 1
    assert all(type(point.cost) is float for report in result.searches for point in report.history)


def test_admitted_elite_survives_session_event_eviction() -> None:
    """Detach only admitted bounded elite evidence rather than retain every population record."""
    store = _TraceStore(TraceConfig(max_records=2), "SA")
    winner = observation(store, "original winner", 0.0)
    pool = SolutionPool()
    pool.insert((winner, [0.5]), "SA")
    for index in range(10):
        observation(store, str(index))
    assert winner._cell.witness is None
    keys, cost = select_best_result([], pool)
    assert DecisionTrace.selected(cost, keys).events[0].action == "original winner"


def test_disabled_and_failure_contexts_do_not_leak_recorders() -> None:
    """Keep disabled operation and failed/nested callbacks from contaminating other invocations."""
    problem = StatefulProblem()
    session = SearchSession(problem.decode, problem.cost)
    candidate = session.decode(np.array([0.5]))
    assert candidate.trace is None
    assert type(session.evaluate(candidate)) is float
    assert _CURRENT.get() is None
    result = RKO(
        problem.decode, problem.cost, 1, algorithms=["MultiStart"], execution="sequential"
    ).solve(time_limit=0.01)
    assert result.trace is None
    record_decision("outside evaluation", "ignored")


@dataclass
class InitialPoolWinner(StatefulProblem):
    """Make startup observations strictly better than subsequent search observations.

    Responsibilities:
    - Exercise genuine initialization-origin witness selection, not just enabled prefill.

    Attributes:
        initial (bool): Whether the last original decoder call belonged to startup.
    """

    initial: bool = False

    def decode(self, keys: np.ndarray) -> int:
        """Mark the actual recording namespace at the original decoder boundary.

        Args:
            keys (np.ndarray): Candidate chromosome.

        Returns:
            int: Original stateful payload.
        """
        self.initial = _CURRENT.get().source == "initial_pool"
        return super().decode(keys)

    def cost(self, payload: int) -> float:
        """Ensure final numeric selection genuinely originates in shared initialization.

        Args:
            payload (int): Original decoded invocation counter.

        Returns:
            float: Strictly smaller cost for initial observations.
        """
        self.objectives += 1
        return -100.0 if self.initial else 1.0


@pytest.mark.parametrize("execution", ["sequential", "process"])
def test_final_initialization_winner_keeps_its_original_namespace(execution: str) -> None:
    """Preserve initialization identity when all later search results are worse.

    Args:
        execution (str): Local or owned spawn execution.
    """
    problem = InitialPoolWinner()
    result = RKO(
        problem.decode,
        problem.cost,
        2,
        algorithms=["SA"],
        execution=execution,
        prefill_pool=True,
        trace=TraceConfig(),
        random_state=42,
    ).solve(time_limit=0.03)
    assert result.cost == -100.0
    assert result.trace.source == "initial_pool"
    assert result.trace.state == "recorded"
    assert result.trace.matches(result.keys, result.cost)


@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
def test_failed_objectives_and_decoder_exceptions_never_publish_witness(failure: str) -> None:
    """Clean contexts on errors and reject observations that never complete finitely.

    Args:
        failure (str): Original objective failure kind.
    """
    problem = StatefulProblem()

    def rejected(payload: int) -> float:
        """Fail the original objective without re-decoding its input.

        Args:
            payload (int): Actual original payload.

        Returns:
            float: Nonfinite value in the nonfinite test case.

        Raises:
            RuntimeError: For the exception case.
        """
        if failure == "exception":
            raise RuntimeError("original failure")
        return float("nan")

    session = SearchSession(problem.decode, rejected, trace=TraceConfig())
    candidate = session.decode(np.array([0.5]))
    with pytest.raises((RuntimeError, ValueError)):
        session.evaluate(candidate)
    assert session.best is None and session.n_evaluations == 0
    assert _CURRENT.get() is None

    def bad_decoder(keys: np.ndarray) -> int:
        """Raise after an explicit emission to exercise finally-based context restoration.

        Args:
            keys (np.ndarray): Candidate chromosome.

        Returns:
            int: Never returned.

        Raises:
            RuntimeError: Always for this failure fixture.
        """
        record_decision("before failure")
        raise RuntimeError("original decoder failure")

    with pytest.raises(RuntimeError, match="decoder"):
        SearchSession(bad_decoder, problem.cost, trace=TraceConfig()).decode(np.array([0.5]))
    assert _CURRENT.get() is None


def test_nested_disabled_decoder_does_not_capture_outer_events() -> None:
    """Suspend an outer recorder during nested untraced callbacks and restore it afterward."""
    problem = StatefulProblem()
    nested = SearchSession(problem.decode, problem.cost)
    store = _TraceStore(TraceConfig(), "outer")
    cell = store.begin()
    token = _CURRENT.set(cell)
    try:
        record_decision("outer_before")
        nested.evaluate(nested.decode(np.array([0.5])))
        record_decision("outer_after")
    finally:
        _CURRENT.reset(token)
    cell.seal([0.5], 1.0)
    assert [event.action for event in cell.snapshot().events] == ["outer_before", "outer_after"]


def test_pool_duplicate_ties_do_not_replace_original_identity() -> None:
    """Keep the pool's actual first-admitted observation when coordinates/cost tie exactly."""
    store = _TraceStore(TraceConfig(), "SA")
    first, second = observation(store, "first"), observation(store, "second")
    pool = SolutionPool()
    pool.insert((first, [0.5]))
    pool.insert((second, [0.5]))
    keys, value = select_best_result([], pool)
    assert DecisionTrace.selected(value, keys).evaluation_id == first._cell.identity


def test_cancelled_decode_does_not_replace_completed_original_evidence() -> None:
    """Keep original incumbent evidence when cancellation interrupts a later decode/cost pair."""
    cancelled = threading.Event()
    problem = StatefulProblem()

    def decode(keys: np.ndarray) -> int:
        """Request cancellation after emitting the second real invocation's events.

        Args:
            keys (np.ndarray): Candidate chromosome.

        Returns:
            int: Original payload, not evaluated after the cancellation boundary.
        """
        value = problem.decode(keys)
        if problem.calls == 2:
            cancelled.set()
        return value

    pool = SolutionPool()
    algorithm = build_metaheuristic("MultiStart", pool, decode, problem.cost, 1, 0, None)
    session = SearchSession(decode, problem.cost, trace=TraceConfig())
    _, keys, value = session.solve(algorithm, 0.1, cancelled=cancelled)
    trace = DecisionTrace.selected(value, keys)
    assert session.metadata.stop_reason == "cancelled"
    assert session.n_evaluations == problem.objectives == 1
    assert trace.events == (DecisionEvent("invocation", "1"),)
    assert _CURRENT.get() is None and _ACTIVE_STORE.get() is None


@dataclass
class SlowStatefulProblem(StatefulProblem):
    """Force queued searches to remain unstarted after one finite blocking objective.

    Responsibilities:
    - Exercise actual process scheduling without inventing evaluations for queued algorithms.

    Attributes:
        None: Counter state is inherited from the original stateful problem.
    """

    def cost(self, payload: int) -> float:
        """Block briefly inside the first original finite objective.

        Args:
            payload (int): Original decoder invocation payload.

        Returns:
            float: Genuine finite stateful minimization value.
        """
        time.sleep(0.2)
        return super().cost(payload)


def test_unstarted_queued_searches_do_not_fabricate_evidence() -> None:
    """Retain actual evidence when one callback exhausts the common queued-search budget."""
    problem = SlowStatefulProblem()
    result = RKO(
        problem.decode,
        problem.cost,
        1,
        algorithms=["SA", "PSO", "GA"],
        execution="process",
        n_jobs=1,
        trace=TraceConfig(),
        random_state=42,
    ).solve(time_limit=0.1)
    assert result.n_evaluations == 1
    assert result.searches[1].stop_reason == result.searches[2].stop_reason == "not_started"
    assert result.trace.state == "recorded"
    assert result.trace.events == (DecisionEvent("invocation", "1"),)


def test_reports_preserve_binding_and_never_replay_callbacks(tmp_path: Path) -> None:
    """Render original evidence offline without objective/decoder replay or HTML injection.

    Args:
        tmp_path (Path): Fixture-owned report outputs.
    """
    pytest.importorskip("plotly")
    store = _TraceStore(TraceConfig(), "SA")
    value = observation(store, "<script>alert('witness')</script>")
    problem = StatefulProblem()
    result = RKO(
        problem.decode,
        problem.cost,
        1,
        algorithms=["SA"],
        execution="sequential",
        trace=TraceConfig(),
    ).solve(time_limit=0.01)
    original_calls = (problem.calls, problem.objectives)
    report = OptimizationReport(result)
    assert report.to_dict()["decision_trace"]["evaluation_id"] == result.trace.evaluation_id
    report.write_html(tmp_path / "observed.html")
    assert (problem.calls, problem.objectives) == original_calls
    artificial = replace(
        result, keys=(0.5,), cost=float(value), trace=DecisionTrace.selected(value, [0.5])
    )
    html = (
        OptimizationReport(artificial)
        .write_html(tmp_path / "escaped.html")
        .read_text(encoding="utf-8")
    )
    assert "<script>alert('witness')</script>" not in html
    assert "&lt;script&gt;" in html and "original_evaluation" in html
    with pytest.raises(ValueError, match="different numerical"):
        OptimizationReport(replace(result, cost=result.cost - 1))
