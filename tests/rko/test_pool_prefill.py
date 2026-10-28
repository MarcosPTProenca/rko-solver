"""Verify bounded reference startup, callback ownership and public accounting.

Objective:
- Keep shared startup opt-in, separately accounted and safe across execution modes.

Technical approach:
- Exercise the real initializer on numerical, tied and failing objectives.
- Check actual public dispatch and independent initialization/search reports.

Flow (pseudocode):
1. Fill reference slots and observe callback counts, keys and RNG ownership.
2. Expire or fail callbacks and verify completed-incumbent recovery or propagation.
3. Solve through every execution mode and reconcile immutable startup/search metadata.

Important criteria:
- No uniqueness retries; flat objectives produce exactly 2P-1 observations.
- Startup never consumes an algorithm stream or runs process callbacks in the parent.
- Default-off retains existing empty-pool and per-search accounting contracts.

Inputs/Outputs:
- Input: fixed seeds, bounded capacities and importable callbacks.
- Output: reference phase, lifecycle and accounting assertions.
"""

import multiprocessing as mp
import numpy as np
import os
import pickle
import pytest
import random
import shutil
import threading
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from rko_solver import RKO
from rko_solver.components.random_streams import RandomStreams
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.orchestration import initialization as initialization_module
from rko_solver.orchestration.initialization import initialize_pool
from rko_solver.orchestration.processes import _worker
from rko_solver.orchestration.types import RunConfig
from rko_solver.parameters.loader import default_params_dir
from types import SimpleNamespace
from unittest.mock import Mock


def decode(keys: np.ndarray) -> np.ndarray:
    """Return the isolated candidate payload.

    Args:
        keys (np.ndarray): Candidate chromosome.

    Returns:
        np.ndarray: Unchanged payload.
    """
    return keys


def sphere(keys: np.ndarray) -> float:
    """Evaluate a finite numerical objective.

    Args:
        keys (np.ndarray): Candidate payload.

    Returns:
        float: Squared norm.
    """
    return float(np.dot(keys, keys))


def flat(keys: np.ndarray) -> float:
    """Give all distinct chromosomes tied objective values.

    Args:
        keys (np.ndarray): Unused finite candidate.

    Returns:
        float: Constant tied score.
    """
    return 1.0


@pytest.mark.parametrize("capacity", [1, 3, 10])
@pytest.mark.parametrize("dimension", [1, 20])
def test_flat_reference_pass_is_bounded_and_preserves_rng(capacity: int, dimension: int) -> None:
    """Count the single cost-tie pass without changing caller RNGs.

    Args:
        capacity (int): Reference slot count.
        dimension (int): One-coordinate edge case or multi-coordinate vectors.
    """
    pool = SolutionPool(capacity)
    before = pickle.dumps((random.getstate(), np.random.get_state()))
    report = initialize_pool(pool, decode, flat, dimension, capacity, 1.0, 42)
    assert report.n_evaluations == 2 * capacity - 1
    assert report.pool_size == capacity and report.stop_reason == "completed"
    assert report.time_limit == 1.0 and report.elapsed_seconds > 0
    assert pickle.dumps((random.getstate(), np.random.get_state())) == before
    with pytest.raises(FrozenInstanceError):
        report.pool_size = 0


def test_reference_selection_sort_and_slot_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Perturb the reference's tied slot, even when its replacement is worse.

    Args:
        monkeypatch (pytest.MonkeyPatch): Supply ordered initial vectors and mutation draws.
    """
    streams = RandomStreams(1)
    generator = Mock(wraps=streams.numpy)
    generator.random.side_effect = [
        np.array([0.8, 0.0]),
        np.array([0.4, 0.4]),
        np.array([0.1, 0.1]),
    ]
    monkeypatch.setattr(streams, "numpy", generator)
    monkeypatch.setattr(streams.python, "randint", lambda lower, upper: 0)
    monkeypatch.setattr(streams.python, "random", lambda: 0.9)
    monkeypatch.setattr(initialization_module, "RandomStreams", lambda seed: streams)
    observed: list[list[float]] = []

    def score(keys: np.ndarray) -> float:
        """Observe the exact candidate selected for the one tied-slot mutation.

        Args:
            keys (np.ndarray): Raw candidate.

        Returns:
            float: Sum, with initial costs 0.8, 0.8 and 0.2.
        """
        observed.append(keys.tolist())
        return float(sum(keys))

    pool = SolutionPool(3)
    report = initialize_pool(pool, decode, score, 2, 3, 1.0, 7)
    assert report.n_evaluations == 4
    assert observed[-1] == [0.9, 0.0]
    assert [keys for _, keys in pool.get_elite(3)] == [
        [0.1, 0.1],
        [0.4, 0.4],
        [0.9, 0.0],
    ]
    assert [0.8, 0.0] not in [keys for _, keys in pool.get_elite(3)]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_startup_fails_without_publishing(invalid: float) -> None:
    """Reject invalid observations rather than exposing a successful partial pool.

    Args:
        invalid (float): Nonfinite objective observation.
    """
    pool = SolutionPool(3)
    outer_rng = RandomStreams.python_rng()
    with pytest.raises(ValueError, match="finite"):
        initialize_pool(pool, decode, lambda keys: invalid, 3, 3, 1.0, 4)
    assert pool.size == 0 and RandomStreams.python_rng() is outer_rng


def test_callback_failure_after_valid_seed_is_not_a_timeout() -> None:
    """Propagate the original exception instead of publishing a successful partial phase."""
    observations: list[float] = []
    failure = ArithmeticError("later seed failed")

    def fail_second(keys: np.ndarray) -> float:
        """Fail after one completed finite objective.

        Args:
            keys (np.ndarray): Isolated candidate.

        Returns:
            float: First finite score only.

        Raises:
            ArithmeticError: On the next candidate observation.
        """
        if observations:
            raise failure
        observations.append(sphere(keys))
        return observations[0]

    pool = SolutionPool(3)
    outer_rng = RandomStreams.python_rng()
    with pytest.raises(ArithmeticError) as caught:
        initialize_pool(pool, decode, fail_second, 3, 3, 1.0, 7)
    assert caught.value is failure and pool.size == 0
    assert len(observations) == 1 and RandomStreams.python_rng() is outer_rng


def test_decoder_mutation_and_retention_cannot_change_seed_keys() -> None:
    """Keep scored key snapshots detached from arbitrary mutable decoder payloads."""
    retained: list[np.ndarray] = []

    def mutate(keys: np.ndarray) -> np.ndarray:
        """Mutate and retain the user-owned decoder payload.

        Args:
            keys (np.ndarray): Isolated payload, not the phase's source keys.

        Returns:
            np.ndarray: Mutated objective payload.
        """
        retained.append(keys)
        keys[:] = 2.0
        return keys

    pool = SolutionPool(3)
    report = initialize_pool(pool, mutate, sphere, 4, 3, 1.0, 42)
    snapshot = pool.get_elite(3)
    for keys in retained:
        keys[:] = -20.0
    assert pool.get_elite(3) == snapshot
    assert report.n_evaluations == 5
    assert all(0 <= key < 1 for _, keys in snapshot for key in keys)
    assert all(value == 16.0 for value, _ in snapshot)


def test_repeated_mutation_positions_and_duplicate_slots_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve repeated coordinate draws without retrying duplicate chromosomes.

    Args:
        monkeypatch (pytest.MonkeyPatch): Force identical seeds and no-op mutations.
    """
    streams = RandomStreams(1)
    generator = Mock(wraps=streams.numpy)
    generator.random.side_effect = lambda dimension: np.full(dimension, 0.5)
    positions = Mock(return_value=0)
    monkeypatch.setattr(streams, "numpy", generator)
    monkeypatch.setattr(streams.python, "randint", positions)
    monkeypatch.setattr(streams.python, "random", lambda: 0.5)
    monkeypatch.setattr(initialization_module, "RandomStreams", lambda seed: streams)
    pool = SolutionPool(3)
    report = initialize_pool(pool, decode, flat, 20, 3, 1.0, 7)
    assert report.n_evaluations == 5 and report.pool_size == 1
    assert generator.random.call_count == 3 and positions.call_count == 8


def test_timeout_publishes_first_completed_observation() -> None:
    """Retain the first valid observation despite startup exhausting its allowance."""
    observed: list[np.ndarray] = []

    def slow(keys: np.ndarray) -> float:
        """Complete one finite observation after the cooperative deadline.

        Args:
            keys (np.ndarray): Isolated candidate.

        Returns:
            float: Squared norm.
        """
        observed.append(keys.copy())
        time.sleep(0.01)
        return sphere(keys)

    pool = SolutionPool(4)
    report = initialize_pool(pool, decode, slow, 3, 4, 0.001, 4)
    assert report.n_evaluations == 1 and report.stop_reason == "time_limit"
    assert pool.best == (sphere(observed[0]), observed[0].tolist())


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
def test_public_prefill_accounting_and_seed_isolation(execution: str) -> None:
    """Reconcile shared startup and search observations through real dispatch.

    Args:
        execution (str): Public execution mode.
    """
    children = {child.pid for child in mp.active_children()}
    settings = dict(
        algorithms=["MultiStart"],
        control=0,
        execution=execution,
        pool_size=3,
        random_state=12,
    )
    baseline = RKO(decode, flat, 3, **settings).solve(0.02)
    result = RKO(decode, flat, 3, prefill_pool=True, **settings).solve(0.02)
    assert baseline.initialization is None
    assert result.initialization is not None
    assert result.initialization.n_evaluations == 5
    assert result.initialization.pool_size == 3
    assert result.n_evaluations == 5 + sum(row.n_evaluations for row in result.searches)
    assert baseline.n_evaluations == sum(row.n_evaluations for row in baseline.searches)
    assert [row.random_state for row in result.searches] == [
        row.random_state for row in baseline.searches
    ]
    assert all(row.time_limit > 0 for row in result.searches)
    assert result.n_workers == 1
    assert {child.pid for child in mp.active_children()} == children


@pytest.mark.parametrize("invalid", [0, 1, "true", [], 1.0])
def test_prefill_override_requires_boolean(invalid: object) -> None:
    """Validate opt-in settings before any initialization callback.

    Args:
        invalid (object): Nonboolean Python override.
    """
    with pytest.raises(ValueError, match="prefill_pool"):
        RKO(decode, sphere, 3, execution="sequential", prefill_pool=invalid).solve(0.01)


@pytest.mark.parametrize("execution", ["sequential", "thread"])
def test_partial_startup_keeps_fresh_search_allowance(execution: str) -> None:
    """Run a fresh search after an expired startup phase rather than spending its cap.

    Args:
        execution (str): Real execution mode with directly observed callbacks.
    """
    calls: list[np.ndarray] = []

    def slow_first(keys: np.ndarray) -> float:
        """Overrun startup once, then permit ordinary fast search observations.

        Args:
            keys (np.ndarray): Isolated candidate payload.

        Returns:
            float: Finite squared norm.
        """
        calls.append(keys.copy())
        if len(calls) == 1:
            time.sleep(0.01)
        return sphere(keys)

    result = RKO(
        decode,
        slow_first,
        3,
        algorithms=["MultiStart"],
        control=0,
        execution=execution,
        pool_size=3,
        prefill_pool=True,
        random_state=4,
    ).solve(0.001)
    assert result.initialization.stop_reason == "time_limit"
    assert result.initialization.n_evaluations == 1
    assert result.searches[0].n_evaluations >= 1
    assert result.searches[0].time_limit == 0.001
    assert result.n_evaluations == len(calls)
    assert result.cost <= sphere(calls[0])


@pytest.mark.parametrize("override", [None, False, True])
def test_toml_opt_in_and_python_override(tmp_path: Path, override: bool | None) -> None:
    """Preserve absent/default configuration behavior and explicit Python precedence.

    Args:
        tmp_path (Path): Isolated configuration copy.
        override (bool | None): Python setting that overrides or follows TOML.
    """
    directory = tmp_path / "params"
    shutil.copytree(
        default_params_dir(),
        directory,
        ignore=shutil.ignore_patterns("*.py", "__pycache__"),
    )
    config = directory / "config.toml"
    config.write_text(config.read_text().replace("prefill_pool = false", "prefill_pool = true"))
    result = RKO(
        decode,
        sphere,
        3,
        algorithms=["MultiStart"],
        control=0,
        execution="sequential",
        params_dir=directory,
        prefill_pool=override,
        random_state=12,
    ).solve(0.02)
    assert (result.initialization is not None) == (override is not False)


def test_startup_seed_is_independent_of_portfolio() -> None:
    """Keep the reserved startup stream stable when selected search count changes."""
    reports = []
    for names in (["MultiStart"], ["MultiStart", "IPR"]):
        result = RKO(
            decode,
            flat,
            3,
            algorithms=names,
            control=0,
            execution="sequential",
            pool_size=2,
            prefill_pool=True,
            random_state=12,
        ).solve(0.01)
        reports.append(result.initialization)
    assert reports[0].random_state == reports[1].random_state
    assert reports[0].n_evaluations == reports[1].n_evaluations == 3


class InitializationFailure:
    """Fail before searches are released so waiting workers must be torn down.

    Responsibilities:
    - Expose initializer exception propagation without a blocked callback.
    """

    def __call__(self, keys: np.ndarray) -> float:
        """Raise the original startup failure.

        Args:
            keys (np.ndarray): Unused candidate.

        Raises:
            RuntimeError: Always, to test cleanup of a failed startup phase.
        """
        raise RuntimeError("seed failure")


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
def test_startup_failure_clears_result_and_releases_workers(execution: str) -> None:
    """Keep startup failures visible without orphaning waiting searches.

    Args:
        execution (str): Public dispatcher.
    """
    children = {child.pid for child in mp.active_children()}
    optimizer = RKO(
        decode,
        InitializationFailure(),
        3,
        algorithms=["MultiStart", "IPR"],
        control=0,
        execution=execution,
        pool_size=3,
        prefill_pool=True,
        random_state=42,
        n_jobs=2 if execution == "process" else None,
    )
    with pytest.raises(RuntimeError, match="seed failure"):
        optimizer.solve(0.02)
    assert optimizer.result_ is None
    assert {child.pid for child in mp.active_children()} == children


def test_repeated_prefill_does_not_retain_a_stale_result() -> None:
    """Clear a previous successful report before a later initializer fails."""
    optimizer = RKO(
        decode,
        sphere,
        3,
        algorithms=["MultiStart"],
        control=0,
        execution="sequential",
        pool_size=3,
        prefill_pool=True,
        random_state=42,
    )
    first = optimizer.solve(0.01)
    optimizer.cost = InitializationFailure()
    with pytest.raises(RuntimeError, match="seed failure"):
        optimizer.solve(0.01)
    assert optimizer.result_ is None
    optimizer.cost = sphere
    last = optimizer.solve(0.01)
    assert last is not first and last.initialization is not first.initialization
    assert last.initialization.random_state == first.initialization.random_state


class BlockedStartup:
    """Hold the initializer past its process grace to exercise the phase watchdog.

    Responsibilities:
    - Simulate a trusted callback that does not cooperate with a deadline.
    """

    def __call__(self, keys: np.ndarray) -> float:
        """Block long enough for the parent to terminate the existing worker.

        Args:
            keys (np.ndarray): Isolated candidate.

        Returns:
            float: Finite score only if no watchdog terminates this process.
        """
        time.sleep(30.0)
        return sphere(keys)


def test_initialization_watchdog_terminates_waiting_processes() -> None:
    """Keep blocked prefill separate from constructor and search watchdog phases."""
    children = {child.pid for child in mp.active_children()}
    optimizer = RKO(
        decode,
        BlockedStartup(),
        3,
        algorithms=["MultiStart", "IPR"],
        control=0,
        execution="process",
        pool_size=3,
        prefill_pool=True,
        random_state=42,
        n_jobs=2,
    )
    with pytest.raises(TimeoutError, match="initialization timed out"):
        optimizer.solve(0.01)
    assert optimizer.result_ is None
    assert {child.pid for child in mp.active_children()} == children


@pytest.mark.parametrize("prefill", [False, True])
def test_expired_process_search_preserves_only_opted_in_startup(prefill: bool) -> None:
    """Retain startup work when no search starts without changing default error behavior.

    Args:
        prefill (bool): Explicit initialization policy supplied to the worker.
    """
    pool = SolutionPool(2)
    pool.insert((0.1, [0.2]))
    parent, child = mp.Pipe(duplex=False)
    start = threading.Event()
    start.set()

    def forbidden(keys: np.ndarray) -> float:
        """Reject any callback after an already-expired search release.

        Args:
            keys (np.ndarray): Unused candidate.

        Raises:
            AssertionError: If search work is incorrectly started.
        """
        raise AssertionError("no search callback should start")

    try:
        _worker(
            (("MultiStart", 42),),
            RunConfig(0, 2, ("MultiStart",), prefill),
            pool,
            decode,
            forbidden,
            1,
            None,
            start,
            SimpleNamespace(value=time.perf_counter() - 1.0),
            child,
            collect_metadata=True,
        )
        assert parent.recv() == ("ready", None)
        status, payload = parent.recv()
        if prefill:
            assert status == "metadata"
            assert len(payload) == 1 and payload[0].stop_reason == "not_started"
            assert payload[0].n_evaluations == 0
            assert parent.recv() == ("result", ([], [0.2], 0.1))
        else:
            assert status == "error" and isinstance(payload[0], TimeoutError)
    finally:
        parent.close()
        child.close()


class ChildOnlyCost:
    """Reject accidental parent evaluation in the spawned startup path.

    Responsibilities:
    - Distinguish child callback ownership from parent-side setup.

    Attributes:
        parent_pid (int): Process that constructs the solver.
    """

    def __init__(self) -> None:
        """Remember the parent without executing an objective."""
        self.parent_pid = os.getpid()

    def __call__(self, keys: np.ndarray) -> float:
        """Fail if a process-mode candidate is evaluated in the parent.

        Args:
            keys (np.ndarray): Isolated candidate.

        Returns:
            float: Finite squared norm.

        Raises:
            AssertionError: If callback execution moved into the parent process.
        """
        assert os.getpid() != self.parent_pid
        return sphere(keys)


def test_prefilled_process_returns_seed_without_started_searches() -> None:
    """Return completed startup context even when the fresh search deadline is immediate."""
    result = RKO(
        decode,
        ChildOnlyCost(),
        3,
        algorithms=["MultiStart"],
        control=0,
        execution="process",
        pool_size=3,
        prefill_pool=True,
        random_state=42,
        n_jobs=1,
    ).solve(1e-12)
    assert result.initialization.n_evaluations == result.n_evaluations == 1
    assert result.searches[0].stop_reason == "not_started"
    assert result.searches[0].n_evaluations == 0
    assert result.stop_reason == "time_limit" and np.isfinite(result.cost)
    assert result.cost == sphere(np.asarray(result.keys))


def test_prefilled_process_groups_keep_existing_worker_cap() -> None:
    """Release all grouped searches after startup without a dedicated extra process."""
    children = {child.pid for child in mp.active_children()}
    result = RKO(
        decode,
        ChildOnlyCost(),
        3,
        algorithms=["MultiStart", "IPR", "SA"],
        control=0,
        execution="process",
        pool_size=3,
        prefill_pool=True,
        random_state=42,
        n_jobs=2,
    ).solve(0.3)
    assert result.initialization.n_evaluations == 3
    assert result.initialization.pool_size == 3
    assert result.n_workers == 2
    assert [report.algorithm for report in result.searches] == ["MultiStart", "IPR", "SA"]
    assert result.n_evaluations == 3 + sum(report.n_evaluations for report in result.searches)
    assert {child.pid for child in mp.active_children()} == children


def test_process_prefill_keeps_callbacks_in_child() -> None:
    """Use the existing child worker rather than a parent or manager callback."""
    result = RKO(
        decode,
        ChildOnlyCost(),
        3,
        algorithms=["IPR"],
        control=0,
        execution="process",
        pool_size=3,
        prefill_pool=True,
        random_state=42,
        n_jobs=1,
    ).solve(0.02)
    assert result.initialization is not None and result.initialization.pool_size == 3
    assert result.n_workers == 1 and np.isfinite(result.cost)
