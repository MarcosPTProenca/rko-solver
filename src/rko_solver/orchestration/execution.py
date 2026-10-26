"""Threaded portfolio execution and incumbent selection.

Objective:
- Preserve worker failures and select a valid completed optimization result.

Technical approach:
- Run each search with a private callback budget and shared failure cancellation.
- Collect outcomes by portfolio position and retain the original failure cause.

Flow (pseudocode):
1. Start budget-aware worker threads.
2. Cancel peer work on failure and join returning workers.
3. Propagate failures or choose the best search/pool incumbent.

Important criteria:
- Arbitrarily blocked Python threads cannot be safely force-terminated.
- An absent or nonfinite incumbent is not a successful optimization result.

Inputs/Outputs:
- Input: configured algorithms, duration and shared elite pool.
- Output: completed worker results, best keys/cost or a caller-visible failure.
"""

from __future__ import annotations

import math
import threading
from ..components.random_streams import RandomStreams
from ..components.solution_pool import ElitePool
from ..metaheuristics.base import MetaheuristicBase
from ..tracing import TraceConfig
from .search import SearchSession
from .types import RunResult
from .types import SearchMetadata


def run_metaheuristics_parallel(
    metaheuristics: list[MetaheuristicBase],
    limit_time: float,
    streams: list[RandomStreams] | None = None,
    *,
    search_metadata: list[SearchMetadata] | None = None,
    history_limit: int = 0,
    trace: TraceConfig | None = None,
) -> list[RunResult]:
    """Run thread searches with callback-budget and failure cancellation checks.

    Args:
        metaheuristics (list[MetaheuristicBase]): Fresh configured portfolio.
        limit_time (float): Search seconds per concurrently started algorithm.
        streams (list[RandomStreams] | None): Owned streams in portfolio order.
        search_metadata (list[SearchMetadata] | None): Optional completed report collector.
        history_limit (int): Per-search scalar history cap; zero disables capture.
        trace (TraceConfig | None): Optional bounded original-decision recorder.

    Returns:
        list[RunResult]: Search outcomes in portfolio order.

    Raises:
        ValueError: If stream count does not match the portfolio.
        RuntimeError: If any worker fails or returns no outcome.
    """
    streams = streams if streams is not None else [RandomStreams(None) for _ in metaheuristics]
    if len(streams) != len(metaheuristics):
        raise ValueError("One random stream pair is required per search.")
    results: list[RunResult | None] = [None] * len(metaheuristics)
    errors: list[BaseException | None] = [None] * len(metaheuristics)
    reports: list[SearchMetadata | None] = [None] * len(metaheuristics)
    cancelled = threading.Event()

    def _worker(index: int, mh: MetaheuristicBase) -> None:
        """Collect one search outcome and cancel peers after any failure.

        Args:
            index (int): Portfolio position for the outcome.
            mh (MetaheuristicBase): Fresh search instance.
        """
        try:
            with streams[index].activate():
                session = (
                    SearchSession(mh.decoder, mh.cost)
                    if trace is None
                    else SearchSession(mh.decoder, mh.cost, trace=trace)
                )
                results[index] = session.solve(
                    mh,
                    limit_time,
                    cancelled=cancelled,
                    random_state=streams[index].seed,
                    history_limit=history_limit,
                )
                reports[index] = session.metadata
        except BaseException as exc:
            errors[index] = exc
            cancelled.set()

    threads = [
        threading.Thread(target=_worker, args=(index, mh), daemon=True, name=mh.name)
        for index, mh in enumerate(metaheuristics)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    except BaseException:
        cancelled.set()
        raise
    failures = [
        f"{mh.name}: {error!r}"
        for mh, error in zip(metaheuristics, errors, strict=True)
        if error is not None
    ]
    if failures:
        cause = next(error for error in errors if error is not None)
        raise RuntimeError("One or more metaheuristics failed: " + "; ".join(failures)) from cause
    if any(result is None for result in results):
        raise RuntimeError("A thread worker returned no outcome.")
    if search_metadata is not None:
        assert all(report is not None for report in reports)
        search_metadata.extend(report for report in reports if report is not None)
    return [result for result in results if result is not None]


def select_best_result(
    results: list[RunResult],
    pool: ElitePool,
) -> tuple[list[float], float]:
    """Choose the best completed candidate across search results and pool.

    Args:
        results (list[RunResult]): Completed search outcomes.
        pool (ElitePool): Shared elite pool.

    Returns:
        tuple[list[float], float]: Detached best keys and finite objective.

    Raises:
        RuntimeError: If no finite nonempty solution exists.
    """
    best_keys: list[float] = []
    best_cost = math.inf
    for _, keys, cost in results:
        if keys and math.isfinite(cost) and cost < best_cost:
            best_cost = cost
            best_keys = list(keys)
    best = pool.best
    if best is not None:
        pool_cost, pool_keys = best
        if pool_cost < best_cost:
            best_cost = pool_cost
            best_keys = list(pool_keys)
    if not best_keys or not math.isfinite(best_cost):
        raise RuntimeError("RKO completed without a finite nonempty solution.")
    return best_keys, best_cost
