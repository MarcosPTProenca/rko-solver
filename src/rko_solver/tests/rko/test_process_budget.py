"""Verify bounded portfolio workers and common-deadline search slices.

Objective:
- Execute every cooperative search without exceeding the requested worker count.

Technical approach:
- Record each real algorithm's seeded first candidate and process identity.
- Inspect worker slice deadlines with a simulated session and monotonic clock.

Flow (pseudocode):
1. Assign a five-algorithm portfolio to one, two or dedicated workers.
2. Check first-candidate streams, result accounting and child cleanup.
3. Simulate early completion, shorter groups, overruns and exhausted deadlines.

Important criteria:
- Queued search deadlines never extend the original common deadline.
- Time left at dispatch is shared among the same worker's remaining searches.
- Constructor RNG consumption must survive preparing several algorithms.

Inputs/Outputs:
- Input: importable callbacks, worker limits and controlled clock cases.
- Output: execution, deadline, RNG and ownership assertions.
"""

import json
import multiprocessing as mp
import numpy as np
import os
import pytest
import random
import threading
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from rko_solver import RKO
from rko_solver.components.random_streams import RandomStreams
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.orchestration import processes
from rko_solver.orchestration.search import _BudgetExpired
from rko_solver.orchestration.types import CostFn
from rko_solver.orchestration.types import DecoderFn
from rko_solver.orchestration.types import RunConfig
from rko_solver.orchestration.types import RunResult
from rko_solver.orchestration.types import SearchMetadata
from types import SimpleNamespace


def decode(keys: np.ndarray) -> np.ndarray:
    """Return candidate keys unchanged.

    Args:
        keys (np.ndarray): Candidate random keys.

    Returns:
        np.ndarray: Unmodified keys.
    """
    return keys


@dataclass
class InitialCost:
    """Identify algorithms through independently predicted first random keys.

    Responsibilities:
    - Record the PID and first Python random sample for each search.
    - Stop after that one completed observation rather than exhausting wall time.

    Attributes:
        directory (Path): Destination for one observation per algorithm.
        expected (tuple[tuple[float, ...], ...]): First keys for each seeded search.
        names (tuple[str, ...]): Algorithm names aligned with the first-key vectors.
        seen (set[int]): Algorithms already observed in this worker.
    """

    directory: Path
    expected: tuple[tuple[float, ...], ...]
    names: tuple[str, ...]
    seen: set[int] = field(default_factory=set)

    def __call__(self, keys: np.ndarray) -> float:
        """Record independently recognizable first keys, then evaluate their norm.

        Args:
            keys (np.ndarray): Candidate vector.

        Returns:
            float: Squared Euclidean norm.

        Raises:
            OSError: If a test observation cannot be written.
            _BudgetExpired: After the first observed candidate, or for an unknown seed.
        """
        for index, expected in enumerate(self.expected):
            if (
                index not in self.seen
                and tuple(keys) == expected
                and self.names[index] in mp.current_process().name.removeprefix("rko-").split("/")
            ):
                self.seen.add(index)
                self.directory.joinpath(f"{index}.json").write_text(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "sample": RandomStreams.python_rng().random(),
                            "at": time.perf_counter(),
                        }
                    ),
                    encoding="utf-8",
                )
                return float(np.dot(keys, keys))
        raise _BudgetExpired


@dataclass
class QueuedFailure:
    """Fail only when the second assigned algorithm reaches its first evaluation.

    Responsibilities:
    - Verify that a prior successful result cannot hide a queued failure.

    Attributes:
        target (tuple[float, ...]): Independently predicted keys of the second search.
    """

    target: tuple[float, ...]

    def __call__(self, keys: np.ndarray) -> float:
        """Evaluate normal candidates and fail at the queued search's first candidate.

        Args:
            keys (np.ndarray): Candidate vector.

        Returns:
            float: Squared norm for candidates other than the failure target.

        Raises:
            ArithmeticError: If the second algorithm begins evaluation.
        """
        if tuple(keys) == self.target:
            raise ArithmeticError("queued objective failed deliberately")
        return float(np.dot(keys, keys))


def test_queued_failure_is_not_hidden_by_completed_result() -> None:
    """Propagate a later algorithm's failure and clean up its shared worker."""
    seed = int(np.random.SeedSequence(42).spawn(2)[1].generate_state(1)[0])
    target = tuple(np.random.RandomState(seed).random_sample(2))
    solver = RKO(
        decode,
        QueuedFailure(target),
        2,
        algorithms=["MultiStart", "SA"],
        control=0,
        n_jobs=1,
        random_state=42,
    )
    before = {child.pid for child in mp.active_children()}
    with pytest.raises(RuntimeError, match="SA") as captured:
        solver.solve(0.2)
    assert isinstance(captured.value.__cause__, ArithmeticError)
    assert "\nSA:\n" in str(captured.value)
    assert solver.result_ is None
    assert {child.pid for child in mp.active_children()} == before


@pytest.mark.parametrize(
    "n_jobs,cpus,expected_workers",
    [(1, 2, 1), (2, 1, 2), (8, 1, 5), (None, 1, 1), (None, 2, 2), (None, 64, 5), (None, None, 1)],
)
def test_worker_limit_runs_every_seeded_algorithm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n_jobs: int | None,
    cpus: int | None,
    expected_workers: int,
) -> None:
    """Execute uneven groups and retain independent per-algorithm random streams.

    Args:
        tmp_path (Path): Destination for first-candidate observations.
        monkeypatch (pytest.MonkeyPatch): Parent-side CPU discovery replacement.
        n_jobs (int | None): Explicit or automatic worker limit.
        cpus (int | None): Simulated process-usable CPU discovery.
        expected_workers (int): Independently expected search process count.
    """
    monkeypatch.setattr(os, "process_cpu_count", lambda: cpus)
    names = ("MultiStart", "SA", "PSO", "GA", "BRKGA")
    seeds = [int(seed.generate_state(1)[0]) for seed in np.random.SeedSequence(42).spawn(5)]
    expected = tuple(tuple(np.random.RandomState(seed).random_sample(2)) for seed in seeds)
    before = {child.pid for child in mp.active_children()}
    result = RKO(
        decode,
        InitialCost(tmp_path, expected, names),
        2,
        algorithms=names,
        control=0,
        n_jobs=n_jobs,
        random_state=42,
    ).solve(2.0)
    observations = [
        json.loads(tmp_path.joinpath(f"{index}.json").read_text()) for index in range(5)
    ]
    assert len({row["pid"] for row in observations}) == expected_workers
    assert result.n_workers == expected_workers
    assert result.n_jobs == n_jobs
    assert result.available_cpus == max(1, cpus or 1)
    assert all(row["pid"] != os.getpid() for row in observations)
    assert [row["sample"] for row in observations] == [
        random.Random(seed).random() for seed in seeds
    ]
    assert result.algorithms == names
    assert len(result.keys) == 2
    assert all(0 <= key < 1 for key in result.keys)
    assert result.cost == pytest.approx(np.dot(result.keys, result.keys))
    assert all(report.n_evaluations == 1 for report in result.searches)
    assert {child.pid for child in mp.active_children()} == before


@pytest.mark.parametrize("cpus", [1, None])
def test_direct_process_dispatch_resolves_cpu_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpus: int | None
) -> None:
    """Keep the internal dispatcher safe when called without a pre-resolved cap.

    Args:
        tmp_path (Path): Destination for independent per-algorithm observations.
        monkeypatch (pytest.MonkeyPatch): Parent CPU discovery replacement.
        cpus (int | None): One usable CPU or unavailable discovery.
    """
    monkeypatch.setattr(os, "process_cpu_count", lambda: cpus)
    names = ("MultiStart", "SA")
    seeds = [int(seed.generate_state(1)[0]) for seed in np.random.SeedSequence(42).spawn(2)]
    expected = tuple(tuple(np.random.RandomState(seed).random_sample(2)) for seed in seeds)
    before = {child.pid for child in mp.active_children()}
    keys, value = processes.run_processes(
        RunConfig(control=0, pool_size=3, active_metaheuristics=names),
        decode,
        InitialCost(tmp_path, expected, names),
        2,
        2.0,
        None,
        None,
        42,
    )
    observations = [
        json.loads(tmp_path.joinpath(f"{index}.json").read_text()) for index in range(2)
    ]
    assert len({row["pid"] for row in observations}) == 1
    assert [row["sample"] for row in observations] == [
        random.Random(seed).random() for seed in seeds
    ]
    assert value == pytest.approx(np.dot(keys, keys))
    assert {child.pid for child in mp.active_children()} == before


@pytest.mark.parametrize(
    "group_size, overrun, duration, first_fraction, expected_slices",
    [
        (3, 0.0, 12.0, 1.0, [4.0, 4.0, 4.0]),
        (3, 1.0, 12.0, 1.0, [4.0, 3.5, 2.5]),
        (3, 7.0, 12.0, 1.0, [4.0, 0.5]),
        (3, 0.0, 0.0, 1.0, []),
        (3, 0.0, 1e-15, 1.0, []),
        (3, 0.0, 6.0, 1.0, [2.0, 2.0, 2.0]),
        (3, 0.0, 12.0, 0.0, [4.0, 6.0, 6.0]),
        (3, 0.0, 12.0, 0.25, [4.0, 5.5, 5.5]),
        (2, 0.0, 12.0, 1.0, [6.0, 6.0]),
        (2, 0.0, 12.0, 0.0, [6.0, 12.0]),
        (1, 0.0, 12.0, 1.0, [12.0]),
    ],
)
@pytest.mark.parametrize("collect_metadata", [False, True])
def test_worker_preserves_deadline_and_constructor_rng(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int,
    overrun: float,
    duration: float,
    first_fraction: float,
    expected_slices: list[float],
    collect_metadata: bool,
) -> None:
    """Inspect slices, constructor RNG states and exhausted-deadline behavior.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replace construction, session and clock.
        group_size (int): Searches assigned to this worker, including shorter groups.
        overrun (float): Simulated callback overrun after each nominal slice.
        duration (float): Remaining common budget, zero simulates late startup.
        first_fraction (float): Fraction of the first allocation consumed before returning.
        expected_slices (list[float]): Durations available to searches actually started.
        collect_metadata (bool): Exercise both compatible and enriched pipe messages.
    """
    assignments = (("MultiStart", 11), ("SA", 22), ("PSO", 33))[:group_size]
    clock = [100.0]
    calls: list[float] = []
    constructed: list[str] = []

    def now() -> float:
        """Return the test-controlled monotonic timestamp.

        Returns:
            float: Current simulated time.
        """
        return clock[0]

    def build(
        name: str,
        pool: object,
        decoder: DecoderFn,
        cost: CostFn,
        dimension: int,
        control: int,
        params_dir: Path | None,
        hyperparameters: dict[str, int | float] | None = None,
    ) -> SimpleNamespace:
        """Consume constructor RNG draws before returning a named test search.

        Args:
            name (str): Algorithm being prepared.
            pool (object): Unused local test pool.
            decoder (DecoderFn): Unused decoder.
            cost (CostFn): Unused objective.
            dimension (int): Unused key dimension.
            control (int): Unused control mode.
            params_dir (Path | None): Unused parameter location.
            hyperparameters (dict[str, int | float] | None): Absent optional overrides.

        Returns:
            SimpleNamespace: Algorithm name used by the simulated session.
        """
        assert hyperparameters is None
        constructed.append(name)
        RandomStreams.python_rng().random()
        RandomStreams.numpy_rng().random()
        return SimpleNamespace(name=name, _search_metadata=None)

    class Session:
        """Simulate completed or overrunning searches at their supplied deadlines.

        Responsibilities:
        - Assert preparation and owned RNG activation before advancing the test clock.
        """

        def __init__(self, decoder: DecoderFn, cost: CostFn) -> None:
            """Accept worker callbacks without evaluating them.

            Args:
                decoder (DecoderFn): Unused decoder.
                cost (CostFn): Unused objective.
            """

        def solve(
            self,
            algorithm: SimpleNamespace,
            limit_time: float,
            *,
            deadline: float,
            random_state: int | None = None,
            history_limit: int = 0,
            max_evaluations: int | None = None,
        ) -> RunResult:
            """Check the reactivated owned stream and simulate one search slice.

            Args:
                algorithm (SimpleNamespace): Named prepared search.
                limit_time (float): Supplied remaining slice duration.
                deadline (float): Absolute deadline bounded by the portfolio.
                random_state (int | None): Actual assigned search seed.
                history_limit (int): Disabled capture for this scheduling-only test.
                max_evaluations (int | None): Unused evaluation limit, always None here.

            Returns:
                RunResult: Consistent synthetic incumbent.
            """
            assert history_limit == 0 and max_evaluations is None
            assert constructed == [name for name, _ in assignments]
            seed = dict(assignments)[algorithm.name]
            assert random_state == seed
            self.metadata = SearchMetadata(
                algorithm=algorithm.name,
                random_state=seed,
                time_limit=limit_time,
                stop_reason="time_limit",
                n_evaluations=2,
            )
            python_rng = random.Random(seed)
            numpy_rng = np.random.RandomState(seed)
            python_rng.random()
            numpy_rng.random()
            assert RandomStreams.python_rng().random() == python_rng.random()
            assert RandomStreams.numpy_rng().random() == numpy_rng.random()
            assert deadline <= 100.0 + duration
            assert deadline == pytest.approx(clock[0] + limit_time)
            calls.append(limit_time)
            clock[0] += limit_time * (first_fraction if len(calls) == 1 else 1.0) + overrun
            return [], [seed / 100], 0.25 if seed >= 22 else 1.0

    python_state, numpy_state = random.getstate(), np.random.get_state()
    parent, child = mp.Pipe(duplex=False)
    start = threading.Event()
    start.set()
    monkeypatch.setattr(processes.time, "perf_counter", now)
    monkeypatch.setattr(processes, "build_metaheuristic", build)
    monkeypatch.setattr(processes, "SearchSession", Session)
    try:
        processes._worker(
            assignments,
            RunConfig(0, 2, tuple(name for name, _ in assignments)),
            SolutionPool(2),
            decode,
            float,
            1,
            None,
            start,
            SimpleNamespace(value=100.0 + duration),
            child,
            collect_metadata=collect_metadata,
        )
        assert random.getstate() == python_state
        assert np.array_equal(np.random.get_state()[1], numpy_state[1])
        assert np.random.get_state()[2:] == numpy_state[2:]
        assert parent.recv() == ("ready", None)
        status, payload = parent.recv()
        if collect_metadata and expected_slices:
            assert status == "metadata"
            assert [report.algorithm for report in payload] == [name for name, _ in assignments]
            assert [report.random_state for report in payload] == [seed for _, seed in assignments]
            assert sum(report.n_evaluations for report in payload) == 2 * len(expected_slices)
            assert all(report.stop_reason == "not_started" for report in payload[len(calls) :])
            assert all(report.elapsed_seconds == 0 for report in payload[len(calls) :])
            assert all(report.time_limit == 0 for report in payload[len(calls) :])
            assert [report.time_limit for report in payload[: len(calls)]] == pytest.approx(
                expected_slices
            )
            status, payload = parent.recv()
        assert calls == pytest.approx(expected_slices)
        if expected_slices:
            assert status == "result", payload
            assert payload == (([], [0.22], 0.25) if len(calls) > 1 else ([], [0.11], 1.0))
        else:
            assert status == "error"
            assert isinstance(payload[0], TimeoutError)
    finally:
        parent.close()
        child.close()
        random.setstate(python_state)
        np.random.set_state(numpy_state)
