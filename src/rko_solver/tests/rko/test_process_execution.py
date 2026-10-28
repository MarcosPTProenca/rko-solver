"""Public RKO process-execution integration tests.

Objective:
- Verify real spawned workers, shared elites and failure cleanup.

Technical approach:
- Run small real portfolios with importable callbacks and temporary configs.

Flow (pseudocode):
1. Prepare a bounded portfolio and process-observable objective.
2. Execute the public solver or managed-pool contract.
3. Check objective consistency, distinct PIDs and child-process cleanup.

Important criteria:
- Tests must not depend on stochastic convergence to a specific optimum.

Inputs/Outputs:
- Input: tiny parameter directories and serializable callbacks.
- Output: assertions on search results and lifecycle behavior.
"""

from __future__ import annotations

import multiprocessing as mp
import numpy as np
import os
import pytest
import shutil
import time
from dataclasses import dataclass
from functools import partial
from multiprocessing.connection import Connection
from pathlib import Path
from rko_solver import RKO
from rko_solver import run
from rko_solver.components.random_streams import RandomStreams
from rko_solver.examples.run_kp_cpp_instances import decode_knapsack
from rko_solver.examples.run_kp_cpp_instances import default_instances_dir
from rko_solver.examples.run_kp_cpp_instances import evaluate_solution
from rko_solver.examples.run_kp_cpp_instances import make_cost
from rko_solver.examples.run_kp_cpp_instances import read_kp_instance
from rko_solver.orchestration import processes
from rko_solver.orchestration.configuration import MH_SPECS
from rko_solver.orchestration.processes import PoolManager
from rko_solver.orchestration.processes import PoolProxy
from rko_solver.parameters.loader import default_params_dir


def identity(keys: np.ndarray) -> np.ndarray:
    """Return a vector without problem-specific decoding.

    Args:
        keys (np.ndarray): Candidate vector.

    Returns:
        np.ndarray: Unmodified vector.
    """
    return keys


def sphere(keys: np.ndarray) -> float:
    """Evaluate a bounded nonnegative objective.

    Args:
        keys (np.ndarray): Candidate vector.

    Returns:
        float: Sum of squared key values.
    """
    return float(np.sum(keys**2))


@dataclass
class RecordingCost:
    """Record each worker's initial RNG sample before evaluating candidates.

    Responsibilities:
    - Make process identity and independent random streams observable.

    Attributes:
        directory (Path): Destination for one observation per child.
        recorded (bool): Whether this worker already wrote its observation.
    """

    directory: Path
    recorded: bool = False

    def __call__(self, keys: np.ndarray) -> float:
        """Record the active owned streams once and return a sphere objective.

        Args:
            keys (np.ndarray): Candidate vector.

        Returns:
            float: Sum of squared keys.
        """
        if not self.recorded:
            self.directory.joinpath(str(os.getpid())).write_text(
                f"{RandomStreams.python_rng().random()} {RandomStreams.numpy_rng().random()}",
                encoding="utf-8",
            )
            self.recorded = True
        return sphere(keys)


def failing_cost(keys: np.ndarray) -> float:
    """Raise a known objective exception inside a worker.

    Args:
        keys (np.ndarray): Unused candidate vector.

    Raises:
        ArithmeticError: Always, to verify cause propagation.
    """
    raise ArithmeticError("objective failed deliberately")


def exiting_cost(keys: np.ndarray) -> float:
    """Exit without normal Python exception handling.

    Args:
        keys (np.ndarray): Unused candidate vector.

    Returns:
        float: Never returned because the worker exits.
    """
    os._exit(17)


def blocking_cost(keys: np.ndarray, *, directory: Path) -> float:
    """Simulate a callback that ignores the solver's cooperative time budget.

    Args:
        keys (np.ndarray): Candidate vector.
        directory (Path): Destination for callback-entry observations.

    Returns:
        float: Sphere objective after blocking beyond the grace period.
    """
    directory.joinpath(str(os.getpid())).write_text("entered", encoding="utf-8")
    time.sleep(30.0)
    return sphere(keys)


@dataclass
class StallingCost:
    """Evaluate normally, then ignore the cooperative budget inside one callback.

    Attributes:
        after (int): Evaluations per process before the callback blocks.
        calls (int): Evaluations already made by this process copy.
    """

    after: int
    calls: int = 0

    def __call__(self, keys: np.ndarray) -> float:
        """Return the sphere objective until the configured call, then block.

        Args:
            keys (np.ndarray): Candidate vector.

        Returns:
            float: Sphere objective for the first `after` calls of this process.
        """
        self.calls += 1
        if self.calls > self.after:
            time.sleep(30.0)
        return sphere(keys)


@pytest.fixture
def params_dir(tmp_path: Path) -> Path:
    """Prepare a two-algorithm offline portfolio.

    Args:
        tmp_path (Path): Temporary root for configuration and observations.

    Returns:
        Path: Directory containing complete parameter TOMLs.
    """
    destination = tmp_path / "parameters"
    shutil.copytree(default_params_dir(), destination)
    destination.joinpath("config.toml").write_text(
        '[run]\ncontrol = 0\npool_size = 5\n[active_metaheuristics]\nlist = ["MultiStart", "SA"]\n',
        encoding="utf-8",
    )
    return destination


def test_process_execution_runs_distinct_children(params_dir: Path, tmp_path: Path) -> None:
    """Execute real workers and check the returned cost against its keys.

    Args:
        params_dir (Path): Complete test portfolio configuration.
        tmp_path (Path): Destination for worker identity observations.
    """
    before = {child.pid for child in mp.active_children()}
    keys, cost = run(
        identity,
        RecordingCost(tmp_path),
        4,
        0.1,
        params_dir=params_dir,
        n_jobs=2,
        random_state=42,
    )
    observations = [path for path in tmp_path.iterdir() if path.is_file()]
    assert len(observations) == 2
    assert all(int(path.name) != os.getpid() for path in observations)
    assert len({path.read_text() for path in observations}) == 2
    assert len(keys) == 4
    assert all(0 <= key < 1 for key in keys)
    assert cost == pytest.approx(sphere(np.array(keys)))
    assert {child.pid for child in mp.active_children()} == before


@pytest.mark.parametrize("filename, optimum", [("kp10.txt", 31), ("kp50.txt", 149)])
def test_knapsack_process_solution_is_feasible(
    params_dir: Path, filename: str, optimum: int
) -> None:
    """Validate actual benchmark decisions and costs through the public process solver.

    Args:
        params_dir (Path): Two-algorithm test configuration.
        filename (str): Bundled Knapsack reference instance.
        optimum (int): Known exact objective upper bound.
    """
    instance = read_kp_instance(default_instances_dir() / filename)
    keys, cost = run(
        decode_knapsack,
        make_cost(instance),
        instance.n_items,
        0.5,
        params_dir=params_dir,
        random_state=42,
    )
    profit, weight, feasible = evaluate_solution(decode_knapsack(keys), instance)
    assert feasible
    assert weight <= instance.capacity
    assert 0 <= profit <= optimum
    assert cost == -profit


@pytest.mark.parametrize("execution", ["sequential", "thread"])
def test_local_modes_remain_available(params_dir: Path, execution: str) -> None:
    """Keep local callbacks usable through explicitly selected local modes.

    Args:
        params_dir (Path): Complete test portfolio configuration.
        execution (str): Local execution mode.
    """
    keys, cost = run(lambda keys: keys, sphere, 3, 0.01, params_dir=params_dir, execution=execution)
    assert cost == pytest.approx(sphere(np.array(keys)))


def test_process_rejects_nested_callbacks(params_dir: Path) -> None:
    """Reject unpickleable callbacks before creating children.

    Args:
        params_dir (Path): Complete test portfolio configuration.
    """
    before = {child.pid for child in mp.active_children()}
    with pytest.raises(ValueError, match="pickleable"):
        run(lambda keys: keys, sphere, 3, 0.1, params_dir=params_dir)
    assert {child.pid for child in mp.active_children()} == before


@pytest.mark.parametrize("n_jobs", [0, -1, True, 1.5])
def test_process_rejects_invalid_worker_budget(params_dir: Path, n_jobs: object) -> None:
    """Reject nonpositive or noninteger worker limits before creating children.

    Args:
        params_dir (Path): Configuration selecting two algorithms.
        n_jobs (object): Invalid explicit worker budget.
    """
    before = {child.pid for child in mp.active_children()}
    with pytest.raises(ValueError, match="n_jobs"):
        run(identity, sphere, 3, 0.1, params_dir=params_dir, n_jobs=n_jobs)
    assert {child.pid for child in mp.active_children()} == before


@pytest.mark.parametrize("objective", [failing_cost, exiting_cost])
@pytest.mark.parametrize("n_jobs", [None, 1])
def test_worker_failure_reaps_children(
    params_dir: Path, objective: object, n_jobs: int | None
) -> None:
    """Propagate callback failures and abrupt exits without orphaning children.

    Args:
        params_dir (Path): Complete test portfolio configuration.
        objective (object): Importable callback that fails inside the worker.
        n_jobs (int | None): Dedicated or shared worker execution.
    """
    before = {child.pid for child in mp.active_children()}
    with pytest.raises(RuntimeError, match="MultiStart|SA") as captured:
        run(identity, objective, 3, 0.1, params_dir=params_dir, n_jobs=n_jobs)
    if objective is failing_cost:
        assert isinstance(captured.value.__cause__, ArithmeticError)
    assert {child.pid for child in mp.active_children()} == before


@pytest.mark.parametrize("n_jobs", [2, 1])
def test_search_timeout_reaps_blocked_workers(
    params_dir: Path, tmp_path: Path, n_jobs: int
) -> None:
    """Observe blocked callback entry and reap workers after the search grace period.

    Args:
        params_dir (Path): Complete test portfolio configuration.
        tmp_path (Path): Destination for callback-entry observations.
        n_jobs (int): Explicit dedicated or shared worker count.
    """
    before = {child.pid for child in mp.active_children()}
    with pytest.raises(TimeoutError, match="search timed out"):
        run(
            identity,
            partial(blocking_cost, directory=tmp_path),
            3,
            0.5,
            params_dir=params_dir,
            n_jobs=n_jobs,
        )
    observations = [path for path in tmp_path.iterdir() if path.is_file()]
    assert len(observations) == n_jobs
    assert all(path.read_text(encoding="utf-8") == "entered" for path in observations)
    assert {child.pid for child in mp.active_children()} == before


def test_interrupt_reaps_workers(params_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Clean up already-started workers when the coordinator is interrupted.

    Args:
        params_dir (Path): Complete test portfolio configuration.
        monkeypatch (pytest.MonkeyPatch): Inject a parent-side interruption.
    """

    def interrupt(*args: object, **kwargs: object) -> None:
        """Interrupt the coordinator's blocking wait.

        Args:
            args (object): Unused positional wait arguments.
            kwargs (object): Unused keyword wait arguments.

        Raises:
            KeyboardInterrupt: Always, to exercise coordinator cleanup.
        """
        raise KeyboardInterrupt

    before = {child.pid for child in mp.active_children()}
    monkeypatch.setattr(processes, "wait", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(identity, sphere, 3, 0.1, params_dir=params_dir)
    assert {child.pid for child in mp.active_children()} == before


def test_seed_allocates_stable_distinct_streams(params_dir: Path, tmp_path: Path) -> None:
    """Repeat each worker's first random sample without promising timed trajectory equality.

    Args:
        params_dir (Path): Complete test portfolio configuration.
        tmp_path (Path): Destination for independent trial observations.
    """
    samples = []
    for trial in range(2):
        directory = tmp_path / str(trial)
        directory.mkdir()
        run(
            identity,
            RecordingCost(directory),
            4,
            0.05,
            params_dir=params_dir,
            random_state=42,
            n_jobs=2,
        )
        samples.append(sorted(path.read_text() for path in directory.iterdir()))
    assert samples[0] == samples[1]
    assert len(set(samples[0])) == 2


def _exchange(pool: PoolProxy, connection: Connection) -> None:
    """Read a parent elite and publish a child improvement.

    Args:
        pool (PoolProxy): Managed elite pool shared with the parent.
        connection (Connection): Pipe for returning the observed elite.
    """
    try:
        connection.send(pool.best)
        pool.insert((1.0, [0.1]), "child")
    finally:
        connection.close()


def test_managed_pool_exchanges_elites_across_spawn() -> None:
    """Observe parent-to-child and child-to-parent elite propagation."""
    context = mp.get_context("spawn")
    with PoolManager(ctx=context) as manager:
        pool = manager.SolutionPool(2)
        pool.insert((2.0, [0.2]), "parent")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=_exchange, args=(pool, child))
        try:
            process.start()
            child.close()
            assert parent.poll(15.0)
            assert parent.recv() == (2.0, [0.2])
            process.join(timeout=15.0)
            assert process.exitcode == 0
            assert pool.best == (1.0, [0.1])
            assert pool.size == 2
        finally:
            if process.is_alive():
                process.kill()
            process.join()
            process.close()
            parent.close()
            child.close()


def test_managed_pool_count_does_not_fetch_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Query only an integer count rather than serializing all elite key vectors.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fail on accidental elite-vector retrieval.
    """
    with PoolManager(ctx=mp.get_context("spawn")) as manager:
        pool = manager.SolutionPool(2)
        pool.insert((1.0, [0.1]))
        monkeypatch.setattr(pool, "get_elite", lambda *args: pytest.fail("count fetched vectors"))
        assert pool.size == 1


@pytest.mark.parametrize("name", sorted(MH_SPECS))
def test_real_algorithm_returns_consistent_result(params_dir: Path, name: str) -> None:
    """Run every registered search through the spawned public solver.

    Args:
        params_dir (Path): Complete parameter directory.
        name (str): Registered algorithm under test.
    """
    params_dir.joinpath("config.toml").write_text(
        f'[run]\ncontrol = 0\npool_size = 5\n[active_metaheuristics]\nlist = ["{name}"]\n',
        encoding="utf-8",
    )
    keys, cost = run(identity, sphere, 4, 0.05, params_dir=params_dir, random_state=12)
    assert len(keys) == 4
    assert np.isfinite(cost)
    assert all(0 <= key < 1 for key in keys)
    assert cost == pytest.approx(sphere(np.array(keys)))


@pytest.mark.parametrize("n_jobs", [2, 1])
def test_search_overrun_returns_the_best_solution_found(params_dir: Path, n_jobs: int) -> None:
    """Return the best completed solution, with a warning, when callbacks overrun the grace.

    Args:
        params_dir (Path): Complete test portfolio configuration.
        n_jobs (int): Explicit dedicated or shared worker count.
    """
    before = {child.pid for child in mp.active_children()}
    solver = RKO(identity, StallingCost(50), 3, params_dir=params_dir, n_jobs=n_jobs)
    with pytest.warns(RuntimeWarning, match="overran the search grace"):
        result = solver.solve(0.5)
    assert len(result.keys) == 3
    assert result.cost == pytest.approx(sphere(np.array(result.keys)))
    assert result.stop_reason == "time_limit"
    assert {report.algorithm for report in result.searches} == {"MultiStart", "SA"}
    assert {child.pid for child in mp.active_children()} == before
