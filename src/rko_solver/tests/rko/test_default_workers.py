"""Protect affinity-aware process defaults and immutable resource provenance.

Objective:
- Bound automatic workers without weakening explicit overrides or legacy results.

Technical approach:
- Replace process dispatch with a recording boundary and simulate CPU discovery.
- Exercise actual spawned groups separately in test_process_budget.

Flow (pseudocode):
1. Supply known, unavailable or changing process CPU counts.
2. Solve with default/explicit caps and inspect the dispatched integer.
3. Change discovery after execution and verify historical metadata stays fixed.

Important criteria:
- Discover capacity once per process solve, never during result property access.
- Explicit positive limits can exceed available CPUs; threads/sequential stay unchanged.
- Missing CPU discovery conservatively means one worker.

Inputs/Outputs:
- Input: controlled CPU counts and domain-neutral callback/factory configuration.
- Output: dispatch, snapshot, fallback and compatibility assertions.
"""

import numpy as np
import os
import pytest
from rko_solver import RKO
from rko_solver import RKOResult
from rko_solver import solver
from rko_solver.orchestration.types import RunConfig


@pytest.mark.parametrize("cpus", [None, 0, 1, 2, 64])
@pytest.mark.parametrize("requested", [None, 1, 8])
def test_process_default_snapshots_available_cpus(
    monkeypatch: pytest.MonkeyPatch, cpus: int | None, requested: int | None
) -> None:
    """Freeze one usable-CPU observation while preserving the requested cap.

    Args:
        monkeypatch (pytest.MonkeyPatch): Isolated CPU/dispatched execution replacements.
        cpus (int | None): Known or unavailable logical process capacity.
        requested (int | None): Explicit cap or automatic default.
    """
    counts = iter((cpus,))
    forwarded: list[object] = []
    monkeypatch.setattr(os, "process_cpu_count", lambda: next(counts))

    def dispatch(*args: object, **kwargs: object) -> tuple[list[float], float]:
        """Observe dispatch without starting an operating-system process.

        Args:
            args (object): Original process dispatcher positional arguments.
            kwargs (object): Original process dispatcher keyword arguments.

        Returns:
            tuple[list[float], float]: Finite synthetic search result.
        """
        forwarded.append(args[6])
        return [0.1, 0.2], 0.05

    monkeypatch.setattr(solver, "run_processes", dispatch)
    optimizer = RKO(
        lambda keys: keys,
        lambda keys: float(np.dot(keys, keys)),
        2,
        algorithms=["MultiStart", "SA", "PSO"],
        control=0,
        n_jobs=requested,
    )
    result = optimizer.solve(0.1)
    available = max(1, cpus or 1)
    assert forwarded == [available if requested is None else requested]
    assert result.available_cpus == available
    assert result.n_jobs == requested
    assert result.n_workers == min(3, available if requested is None else requested)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 99)
    assert result.available_cpus == available
    assert result.n_workers == min(3, available if requested is None else requested)


def test_each_solve_refreshes_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refresh affinity between solves without rewriting an earlier result.

    Args:
        monkeypatch (pytest.MonkeyPatch): Controlled per-solve resource discovery.
    """
    counts = iter((1, 2))
    forwarded: list[object] = []
    monkeypatch.setattr(os, "process_cpu_count", lambda: next(counts))

    def dispatch(*args: object, **kwargs: object) -> tuple[list[float], float]:
        """Capture the resolved per-solve limit.

        Args:
            args (object): Original positional dispatch arguments.
            kwargs (object): Original keyword dispatch arguments.

        Returns:
            tuple[list[float], float]: Finite synthetic result.
        """
        forwarded.append(args[6])
        return [0.1], 0.01

    monkeypatch.setattr(solver, "run_processes", dispatch)
    optimizer = RKO(
        lambda keys: keys,
        lambda keys: float(keys[0]),
        1,
        algorithms=["MultiStart", "SA"],
        control=0,
    )
    first = optimizer.solve(0.1)
    second = optimizer.solve(0.1)
    assert forwarded == [1, 2]
    assert (first.n_workers, second.n_workers) == (1, 2)
    assert (first.available_cpus, second.available_cpus) == (1, 2)


@pytest.mark.parametrize("execution,expected", [("sequential", 1), ("thread", 2)])
def test_nonprocess_modes_do_not_query_capacity(
    monkeypatch: pytest.MonkeyPatch, execution: str, expected: int
) -> None:
    """Leave serial/thread execution and worker accounting unchanged.

    Args:
        monkeypatch (pytest.MonkeyPatch): CPU query that must not be consumed.
        execution (str): Existing nonprocess execution mode.
        expected (int): Existing worker cardinality.
    """
    monkeypatch.setattr(os, "process_cpu_count", lambda: next(iter(())))
    result = RKO(
        lambda keys: keys,
        lambda keys: float(keys[0]),
        1,
        algorithms=["MultiStart", "SA"],
        execution=execution,
        control=0,
    ).solve(0.01)
    assert result.available_cpus is None
    assert result.n_workers == expected


def test_direct_execution_keeps_explicit_worker_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve the configured cap when invoking execution without a solve snapshot.

    Args:
        monkeypatch (pytest.MonkeyPatch): Recording process-dispatch boundary.
    """
    forwarded: list[object] = []

    def dispatch(*args: object, **kwargs: object) -> tuple[list[float], float]:
        """Observe the private execution boundary without spawning.

        Args:
            args (object): Original positional process-dispatch arguments.
            kwargs (object): Original keyword process-dispatch arguments.

        Returns:
            tuple[list[float], float]: Finite synthetic result.
        """
        forwarded.append(args[6])
        return [0.1], 0.01

    monkeypatch.setattr(solver, "run_processes", dispatch)
    optimizer = RKO(lambda keys: keys, lambda keys: float(keys[0]), 1, n_jobs=2)
    optimizer._execute(
        RunConfig(control=0, pool_size=2, active_metaheuristics=("MultiStart", "SA")),
        0.1,
        started_at=0.0,
    )
    assert forwarded == [2]


def test_legacy_manual_process_result_keeps_historical_count() -> None:
    """Preserve the worker estimate for six-argument manually constructed results."""
    result = RKOResult((0.1,), 0.01, 0.1, "python", ("SA", "PSO"), 42, execution="process")
    assert result.available_cpus is None
    assert result.n_workers == 2
