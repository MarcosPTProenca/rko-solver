"""Spawn-based cooperative metaheuristic execution.

Objective:
- Run CPU-bound searches in separate processes with one shared elite pool.

Technical approach:
- Host SolutionPool in a manager and exchange control/results through pipes.
- Optional startup uses one existing child before fresh search-deadline release.
- Construct searches in bounded worker groups and release ready workers together.
- Divide each worker's remaining common time among its queued searches at dispatch.

Flow (pseudocode):
1. Validate serialization and start the managed pool and workers.
2. Wait for readiness, publish a deadline, and release the search event.
3. Collect results or propagate failures, then close all child resources.

Important criteria:
- Every allocation ends no later than the original common deadline.
- Early completion leaves more time for queued searches in the same worker.
- Startup is bounded to 60 seconds; search overrun grace is five seconds. A search that overruns
  its grace after any solution exists returns the best solution found, with a warning.
- Arbitrary callbacks cannot be interrupted cooperatively.

Inputs/Outputs:
- Input: validated configuration, pickleable callbacks, duration and seed.
- Output: best keys and cost, or an algorithm-specific failure.
"""

from __future__ import annotations

import multiprocessing as mp
import numpy as np
import os
import pickle
import time
import traceback
from ..components.random_streams import RandomStreams
from ..components.solution_pool import SolutionPool
from ..modeling.linear import ModelEvidence
from ..tracing import TraceConfig
from .execution import select_best_result
from .factory import build_metaheuristic
from .factory import normalize_hyperparameters
from .initialization import PoolStartup
from .search import SearchReporting
from .search import SearchSession
from .stopped import StoppedSearches
from .types import CostFn
from .types import DecoderFn
from .types import ExecutionOptions
from .types import Hyperparameters
from .types import InitialKeys
from .types import PoolInitializationMetadata
from .types import RunConfig
from .types import RunResult
from .types import SearchMetadata
from .types import StartupJob
from .types import StartupReport
from .types import WorkerCaptureOptions
from multiprocessing.connection import Connection
from multiprocessing.connection import wait
from multiprocessing.managers import BaseManager
from multiprocessing.managers import BaseProxy
from multiprocessing.process import BaseProcess
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Unpack
from typing import cast


class PoolProxy(BaseProxy):
    """Expose the existing elite-pool contract across process boundaries.

    Responsibilities:
    - Delegate mutations and return detached elite snapshots.

    Attributes:
        _exposed_ (tuple[str, ...]): Manager methods available to workers.
    """

    _exposed_ = (
        "insert",
        "get_elite",
        "__len__",
        "start_progress",
        "stop_progress",
        "_get_elite_pair",
    )

    def insert(self, entry: tuple[float, list[float]], source: str = "") -> None:
        """Insert a candidate into the authoritative pool.

        Args:
            entry (tuple[float, list[float]]): Finite objective and key vector.
            source (str): Producing algorithm label.

        Raises:
            ValueError: If the candidate violates pool validation.
        """
        self._callmethod("insert", (entry, source))

    def start_progress(self) -> None:
        """Start periodic status reporting inside the authoritative manager."""
        self._callmethod("start_progress")

    def stop_progress(self) -> None:
        """Join the managed reporter and flush its last snapshot before shutdown."""
        self._callmethod("stop_progress")

    def get_elite(self, k: int = 1) -> list[tuple[float, list[float]]]:
        """Fetch detached elite entries from the manager.

        Args:
            k (int): Maximum number of entries to retrieve.

        Returns:
            list[tuple[float, list[float]]]: Cost-ordered elite snapshots.
        """
        return self._callmethod("get_elite", (k,))

    def _sample_elite_pair(self, pool_size: int) -> list[tuple[float, list[float]]]:
        """Sample client-side ranks and transfer only their two current entries.

        Pool occupancy never decreases, so ranks drawn from an earlier count
        remain valid. A single server lock selects both from the same snapshot.

        Args:
            pool_size (int): Previously observed occupancy, at least two.

        Returns:
            list[tuple[float, list[float]]]: Detached pair in random selection order.

        Raises:
            ValueError: If the supplied count or selected ranks are invalid.
        """
        if type(pool_size) is not int or pool_size < 2:
            raise ValueError("Pair sampling requires an integer pool size of at least two.")
        first, second = RandomStreams.python_rng().sample(list(range(pool_size)), 2)
        return self._callmethod("_get_elite_pair", ((first, second),))

    @property
    def size(self) -> int:
        """Return the number of managed entries.

        Returns:
            int: Current pool occupancy.
        """
        return self._callmethod("__len__")

    @property
    def best(self) -> tuple[float, list[float]] | None:
        """Return the best managed entry when available.

        Returns:
            tuple[float, list[float]] | None: Best detached entry or None.
        """
        entries = self.get_elite(1)
        return entries[0] if entries else None


class PoolManager(BaseManager):
    """Host the authoritative pool in a dedicated manager process.

    Responsibilities:
    - Own pool lifetime and serialize access through its existing lock.

    Attributes:
        SolutionPool (Callable): Registered constructor returning a pool proxy.
    """


PoolManager.register("SolutionPool", SolutionPool, proxytype=PoolProxy)


def _worker(
    assignments: tuple[tuple[str, int], ...],
    config: RunConfig,
    pool: PoolProxy,
    decoder: DecoderFn,
    cost: CostFn,
    dimension: int,
    params_dir: Path | None,
    start: Event,
    deadline: Synchronized,
    connection: Connection,
    hyperparameters: Hyperparameters | None = None,
    collect_metadata: bool = False,
    startup: StartupJob | None = None,
    history_limit: int = 0,
    trace: TraceConfig | None = None,
    **capture_options: Unpack[WorkerCaptureOptions],
) -> None:
    """Run assigned searches sequentially without extending the common deadline.

    Args:
        assignments (tuple[tuple[str, int], ...]): Algorithm names and independent seeds.
        config (RunConfig): Validated global settings.
        pool (PoolProxy): Shared elite-pool proxy.
        decoder (DecoderFn): Pickleable problem decoder.
        cost (CostFn): Pickleable objective callback.
        dimension (int): Number of random keys.
        params_dir (Path | None): Optional parameter directory.
        start (Event): Parent-controlled simultaneous release.
        deadline (Synchronized): Common monotonic search deadline.
        connection (Connection): Dedicated result/control pipe.
        hyperparameters (Hyperparameters | None): Normalized partial offline overrides.
        collect_metadata (bool): Send compact reports, including unstarted queued searches.
        startup (StartupJob | None): Pool startup supplied only to the designated initializer.
        history_limit (int): Per-search scalar history cap; zero disables capture.
        trace (TraceConfig | None): Optional bounded original-decoder recording.
        max_evaluations (int | None): Completed evaluations after which each search stops.
        capture_options (WorkerCaptureOptions): Evaluation cap and bounded model transport policy.
    """
    max_evaluations = capture_options.get("max_evaluations")
    model_evidence: list[ModelEvidence] | None = (
        [] if capture_options.get("collect_model_evidence", False) else None
    )
    callback_options = SearchReporting.callback_options(trace, model_evidence)
    name = "startup"
    try:
        prepared = []
        for name, seed in assignments:
            streams = RandomStreams(seed)
            with streams.activate():
                mh = build_metaheuristic(
                    name,
                    pool,
                    decoder,
                    cost,
                    dimension,
                    config.control,
                    params_dir,
                    (hyperparameters or {}).get(name),
                )
            prepared.append((mh, streams))
        connection.send(("ready", None))
        if startup is not None:
            name = "startup"
            assert startup.release is not None
            startup.release.wait()
            report = PoolStartup(decoder, cost, dimension, **callback_options).run(
                pool, config, startup
            )
            connection.send(("initialized", report))
        start.wait()
        best: RunResult | None = None
        reports: list[SearchMetadata] = []
        for index, (mh, streams) in enumerate(prepared):
            name = mh.name
            now = time.perf_counter()
            common_deadline = deadline.value
            search_deadline = min(
                common_deadline,
                now + (common_deadline - now) / (len(prepared) - index),
            )
            remaining = search_deadline - now
            if remaining <= 0:
                if collect_metadata:
                    reports.extend(SearchReporting.queued_reports(prepared[index:], history_limit))
                break
            with streams.activate():
                session = (
                    SearchSession(decoder, cost)
                    if trace is None
                    else SearchSession(decoder, cost, trace=trace)
                )
                result = session.solve(
                    mh,
                    remaining,
                    deadline=search_deadline,
                    random_state=streams.seed,
                    history_limit=history_limit,
                    max_evaluations=max_evaluations,
                )
            SearchReporting.collect(session, model_evidence)
            if collect_metadata:
                assert session.metadata is not None
                reports.append(session.metadata)
            if best is None or result[2] < best[2]:
                best = result
            del result
        if best is None and config.prefill_pool:
            keys, value = select_best_result([], pool)
            best = ([], keys, value)
        if best is None:
            raise TimeoutError("Search deadline elapsed before worker execution.")
        if collect_metadata:
            connection.send(("metadata", reports))
        if model_evidence is not None:
            connection.send(("model_evidence", model_evidence))
        connection.send(("result", best))
    except BaseException as exc:
        details = f"{name}:\n{traceback.format_exc()}"
        try:
            pickle.dumps(exc)
            cause = exc
        except Exception:
            cause = RuntimeError(f"{type(exc).__name__}: {exc}")
        connection.send(("error", (cause, details)))
    finally:
        connection.close()


class ProcessAssignments:
    """Partition independent seeded searches across available process workers.

    Responsibilities:
    - Preserve algorithm-order seed derivation and round-robin worker grouping.

    """

    @staticmethod
    def groups(
        names: tuple[str, ...], random_state: int | None, n_jobs: int | None
    ) -> list[tuple[tuple[str, int], ...]]:
        """Build original seeded assignments without changing CPU fallback behavior.

        Args:
            names (tuple[str, ...]): Validated nonempty active portfolio names.
            random_state (int | None): Existing seed source.
            n_jobs (int | None): Validated optional positive worker override.

        Returns:
            list[tuple]: Round-robin algorithm/seed assignments per worker.
        """
        seeds = np.random.SeedSequence(random_state).spawn(len(names))
        assignments = tuple(
            (name, int(sequence.generate_state(1)[0]))
            for name, sequence in zip(names, seeds, strict=True)
        )
        worker_count = min(
            max(1, os.process_cpu_count() or 1) if n_jobs is None else n_jobs, len(names)
        )
        return [assignments[index::worker_count] for index in range(worker_count)]


class WorkerShutdown:
    """Stop and reap worker processes and close their result pipes.

    Responsibilities:
    - Give finished workers a short grace, then terminate and finally kill stragglers.
    - Release every process and pipe resource exactly once.
    """

    @staticmethod
    def release(processes: list[BaseProcess], connections: list[Connection]) -> None:
        """Escalate from join to terminate to kill, then close all resources.

        Args:
            processes (list[BaseProcess]): Spawned workers, started or not.
            connections (list[Connection]): Parent ends of the worker pipes.
        """
        for process in processes:
            if process.pid is not None:
                process.join(timeout=0.2)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join()
            process.close()
        for connection in connections:
            connection.close()


def run_processes(
    config: RunConfig,
    decoder: DecoderFn,
    cost: CostFn,
    dimension: int,
    limit_time: float,
    params_dir: Path | None,
    n_jobs: int | None,
    random_state: int | None,
    *,
    verbose: bool = False,
    started_at: float | None = None,
    hyperparameters: Hyperparameters | None = None,
    search_metadata: list[SearchMetadata] | None = None,
    initialization_metadata: list[PoolInitializationMetadata] | None = None,
    history_limit: int = 0,
    max_evaluations: int | None = None,
    **callback_options: Unpack[ExecutionOptions],
) -> tuple[list[float], float]:
    """Execute a selected portfolio with a common start and managed pool.

    Args:
        config (RunConfig): Validated algorithm selection and pool settings.
        decoder (DecoderFn): Pickleable problem decoder.
        cost (CostFn): Pickleable objective callback.
        dimension (int): Number of random keys.
        limit_time (float): Search seconds excluding process startup.
        params_dir (Path | None): Optional parameter directory.
        n_jobs (int | None): Positive override; None uses process-usable logical CPUs,
            falling back to one if discovery is unavailable.
        random_state (int | None): Seed used to derive independent worker seeds.
        verbose (bool): Emit pool updates from its single authoritative manager.
        started_at (float | None): Parent solve's monotonic origin for elapsed log times.
        hyperparameters (Hyperparameters | None): Partial offline scalar mappings.
        search_metadata (list[SearchMetadata] | None): Optional collector for all algorithms.
        initialization_metadata (list | None): Separate opt-in random-startup report collector.
        history_limit (int): Per-search scalar history cap; zero disables capture.
        trace (TraceConfig | None): Optional bounded original-decoder recording.
        max_evaluations (int | None): Completed evaluations after which each search stops.
        callback_options (ExecutionOptions): Optional trace, model witness and caller-key input.

    Returns:
        tuple[list[float], float]: Best keys and minimization objective.

    Raises:
        ValueError: If callbacks are not pickleable or the worker limit is invalid.
        RuntimeError: If a worker fails or exits without returning a result.
        TimeoutError: If startup or initialization exceeds its grace, or the search overruns its
            grace before any solution was found. A search overrun with a solution returns it.
    """
    trace = callback_options.get("trace")
    model_evidence = callback_options.get("model_evidence")
    names = config.active_metaheuristics
    hyperparameters = normalize_hyperparameters(hyperparameters, names, config.control)
    if n_jobs is not None and (type(n_jobs) is not int or n_jobs < 1):
        raise ValueError("n_jobs must be a positive integer or None.")
    try:
        pickle.dumps((decoder, cost))
    except Exception as exc:
        raise ValueError(
            "Process mode requires pickleable decoder/cost callbacks; use module-level "
            "functions or callable classes, not lambdas or nested functions."
        ) from exc

    context = mp.get_context("spawn")
    groups = ProcessAssignments.groups(names, random_state, n_jobs)
    with PoolManager(ctx=context) as manager:
        pool = manager.SolutionPool(config.pool_size, verbose=verbose, started_at=started_at)
        start = context.Event()
        keys = callback_options.get("initial_keys", InitialKeys.from_values(None, dimension))
        startup_required = config.prefill_pool or keys.has_keys
        startup = (
            StartupJob(limit_time, random_state, keys, context.Event())
            if startup_required
            else None
        )
        initialization_done = not startup_required
        initialization_released, phase = False, "startup"
        deadline = context.Value("d", 0.0)
        processes: list[mp.Process] = []
        connections: list[Connection] = []
        pending: dict[Connection, str] = {}
        stopped: list[tuple[str, int]] = []
        ready: set[Connection] = set()
        results: list[RunResult] = []
        expires = time.perf_counter() + 60.0
        try:
            pool.start_progress()
            for group_index, group in enumerate(groups):
                name = "/".join(name for name, _ in group)
                parent, child = context.Pipe(duplex=False)
                connections.append(parent)
                process = context.Process(
                    target=_worker,
                    name=f"rko-{name}",
                    args=(
                        group,
                        config,
                        pool,
                        decoder,
                        cost,
                        dimension,
                        params_dir,
                        start,
                        deadline,
                        child,
                        hyperparameters,
                        search_metadata is not None,
                        startup if group_index == 0 else None,
                        history_limit,
                        trace,
                    ),
                    kwargs={
                        "max_evaluations": max_evaluations,
                        "collect_model_evidence": model_evidence is not None,
                    },
                )
                processes.append(process)
                try:
                    process.start()
                finally:
                    child.close()
                pending[parent] = name

            while pending:
                available = wait(list(pending), timeout=max(0.0, expires - time.perf_counter()))
                if not available:
                    stopped = StoppedSearches.settle(
                        phase, results, pool, pending, connections, groups
                    )
                    break
                for connection in available:
                    name = pending[connection]
                    try:
                        status, payload = connection.recv()
                    except EOFError as exc:
                        raise RuntimeError(f"{name}: worker exited without a result.") from exc
                    if status == "error":
                        cause, details = payload
                        raise RuntimeError(f"{name}: worker failed.\n{details}") from cause
                    if status == "ready":
                        ready.add(connection)
                    elif status == "initialized":
                        initialization_done = True
                        report = cast(StartupReport, payload)
                        keys.set_metadata(report.initial_keys)
                        if initialization_metadata is not None:
                            initialization_metadata.extend(report.initialization)
                    elif status == "model_evidence":
                        assert model_evidence is not None
                        model_evidence.extend(cast(list[ModelEvidence], payload))
                    elif status == "metadata":
                        assert search_metadata is not None
                        search_metadata.extend(payload)
                    else:
                        results.append(payload)
                        del pending[connection]
                if not start.is_set() and len(ready) == len(groups):
                    if initialization_done:
                        deadline.value = time.perf_counter() + limit_time
                        expires = deadline.value + 5.0
                        phase = "search"
                        start.set()
                    elif not initialization_released:
                        assert startup is not None and startup.release is not None
                        initialization_released = True
                        phase = "initialization"
                        expires = time.perf_counter() + limit_time + 5.0
                        startup.release.set()
            StoppedSearches.report(stopped, search_metadata, limit_time, history_limit)
            return select_best_result(results, pool)
        finally:
            WorkerShutdown.release(processes, connections)
            pool.stop_progress()
