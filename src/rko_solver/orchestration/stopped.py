"""Accounting for process searches stopped after overrunning the search grace.

Objective:
- Decide whether an expired process watchdog is fatal or leaves a usable solution.

Technical approach:
- Treat only a search-phase expiry with a finished result or pooled solution as recoverable.
- Describe each stopped search with a zero-evaluation `time_limit` report and one warning.

Flow (pseudocode):
1. On watchdog expiry, classify the phase and look for any solution.
2. Raise `TimeoutError` unless the search phase holds a solution.
3. Otherwise list the unfinished algorithms, warn once and append their reports.

Important criteria:
- Evaluation counts of stopped workers are lost with the process and reported as zero.

Inputs/Outputs:
- Input: phase name, finished results, shared pool and worker assignments.
- Output: stopped algorithm/seed pairs, a warning and appended search reports.
"""

from __future__ import annotations

import warnings
from .types import RunResult
from .types import SearchMetadata
from multiprocessing.connection import Connection
from typing import Protocol


class BestPool(Protocol):
    """Expose the best pooled solution of a local or managed elite pool.

    Attributes:
        best (tuple[float, list[float]] | None): Lowest-cost entry, or None when empty.
    """

    @property
    def best(self) -> tuple[float, list[float]] | None:
        """Return the lowest-cost pooled entry.

        Returns:
            tuple[float, list[float]] | None: Cost and keys, or None when the pool is empty.
        """
        ...


class StoppedSearches:
    """Account for searches whose workers were stopped after overrunning the search grace.

    Responsibilities:
    - Warn once and describe each stopped search without inventing evaluation evidence.
    """

    @staticmethod
    def settle(
        phase: str,
        results: list[RunResult],
        pool: BestPool,
        pending: dict[Connection, str],
        connections: list[Connection],
        groups: list[tuple[tuple[str, int], ...]],
    ) -> list[tuple[str, int]]:
        """Decide what an expired watchdog means for the unfinished workers.

        Args:
            phase (str): Expired phase: startup, initialization or search.
            results (list[RunResult]): Outcomes of the searches that finished.
            pool (BestPool): Shared elite pool; its best counts as a solution.
            pending (dict[Connection, str]): Unfinished worker pipes and their group names.
            connections (list[Connection]): Every worker pipe, in group order.
            groups (list[tuple]): Algorithm names and seeds assigned to each worker.

        Returns:
            list[tuple[str, int]]: Algorithms and seeds of the workers to stop.

        Raises:
            TimeoutError: If the phase is not the search or no solution exists yet.
        """
        if phase != "search" or not (results or pool.best is not None):
            raise TimeoutError(f"RKO {phase} timed out; unfinished: {list(pending.values())}")
        return [
            entry
            for connection, group in zip(connections, groups, strict=True)
            if connection in pending
            for entry in group
        ]

    @staticmethod
    def report(
        stopped: list[tuple[str, int]],
        search_metadata: list[SearchMetadata] | None,
        limit_time: float,
        history_limit: int,
    ) -> None:
        """Warn about stopped searches and append their zero-evaluation reports.

        Args:
            stopped (list[tuple[str, int]]): Algorithm names and seeds of unfinished workers.
            search_metadata (list[SearchMetadata] | None): Optional collector of reports.
            limit_time (float): Search seconds assigned to each stopped search.
            history_limit (int): Requested per-search scalar history cap.
        """
        if not stopped:
            return
        warnings.warn(
            f"RKO stopped {[name for name, _ in stopped]} after their callbacks overran the "
            "search grace; their evaluation counts are unavailable and reported as zero.",
            RuntimeWarning,
            stacklevel=4,
        )
        if search_metadata is None:
            return
        reported = {report.algorithm for report in search_metadata}
        search_metadata.extend(
            SearchMetadata(
                algorithm=name,
                random_state=seed,
                stop_reason="time_limit",
                time_limit=limit_time,
                history_limit=history_limit,
            )
            for name, seed in stopped
            if name not in reported
        )
