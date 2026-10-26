"""Thread-safe shared elite solution pool for cooperative metaheuristics.

Objective:
- Maintain a fixed-capacity sorted pool of the best solutions found across all
  running metaheuristics.

Technical approach:
- Use a threading lock to protect concurrent reads and writes.
- Keep one best observation per coordinate-equal chromosome, sorted by cost.
- Expose an insert method compatible with the duck-typed pool contract already
  used by SA, and a get_elite query required by IPR and pool-aware Nelder-Mead.
- Copy two requested ranks atomically for sparse process-client snapshots.

Flow (pseudocode):
1. Initialise an empty pool with a fixed capacity.
2. On insert: validate; acquire lock; reject equal/worse duplicate observations,
   replace better duplicates or admit competitive novel vectors; re-sort; release.
3. On get_elite: acquire lock; return detached copies of the top-k entries.

Important criteria:
- Pool capacity is fixed at construction time and never exceeded.
- All public methods are thread-safe.
- Public entries remain (cost, keys_list) pairs; internal entries retain source labels.
- Flush compact new-best events and one-second status snapshots under the pool lock.
- Explicit start/stop methods bound the progress thread to its owning solve.

Inputs/Outputs:
- Input: capacity at construction; (cost, keys_list) tuples on insert.
- Output: sorted list of top-k entries on get_elite; best entry on best property.
"""

import math
import threading
import time
from ..tracing import _ObservedCost
from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class SolutionPoolWriter(Protocol):
    """Describe the insertion contract shared by metaheuristic pool clients.

    Responsibilities:
    - Keep algorithms independent of the concrete pool transport.
    - Accept objective/key pairs produced by searches.
    """

    def insert(self, entry: tuple[float, list[float]], source: str = "") -> None:
        """Insert an objective/key pair.

        Args:
            entry (tuple[float, list[float]]): Candidate objective and vector.
            source (str): Producing algorithm label.

        Raises:
            ValueError: If the candidate is invalid.
        """
        ...


@runtime_checkable
class ElitePool(SolutionPoolWriter, Protocol):
    """Describe a pool that also exposes elite entries for search operators.

    Responsibilities:
    - Extend the insertion contract with elite solution access.
    - Support pool-aware local search and path relinking.

    Attributes:
        size (int): Current number of entries.
    """

    def get_elite(self, k: int = 1) -> list[tuple[float, list[float]]]:
        """Read detached elite entries.

        Args:
            k (int): Maximum requested count.

        Returns:
            list[tuple[float, list[float]]]: Sorted entries.
        """
        ...

    @property
    def size(self) -> int:
        """Return current occupancy.

        Returns:
            int: Stored entry count.
        """
        ...


class SolutionPool:
    """Thread-safe fixed-capacity pool of elite solutions.

    Responsibilities:
    - Accept solution insertions from any metaheuristic thread.
    - Provide read access to elite solutions for IPR and pool-aware Nelder-Mead.

    Attributes:
      capacity (int): Maximum number of solutions retained in the pool.
      verbose (bool): Emit new bests and periodic status with clock/elapsed timestamps.
      _pool (list[tuple[float, list[float], str]]): Sorted cost/key/source entries.
      _lock (threading.Lock): Mutual-exclusion guard for mutation and progress output.
      _started_at (float): Monotonic origin shared with the owning solve.
      _progress_guard (threading.Lock): Serialize reporter start/stop operations.
      _progress_stop (threading.Event): Wake and stop the periodic reporter.
      _progress_thread (threading.Thread | None): Owner-managed status reporter.
    """

    def __init__(
        self, capacity: int = 10, *, verbose: bool = False, started_at: float | None = None
    ) -> None:
        """Initialises an empty solution pool.

        Args:
          capacity (int): Maximum number of solutions to keep. Must be >= 1.
          verbose (bool): Print new bests and periodic status; default is silent.
          started_at (float | None): Solve's perf_counter origin, or pool creation time.

        Returns:
          None: Initialises internal state.

        Raises:
          ValueError: If capacity is less than 1.
        """
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity: int = capacity
        self.verbose = verbose
        self._started_at = time.perf_counter() if started_at is None else started_at
        self._pool: list[tuple[float, list[float], str]] = []
        self._lock: threading.Lock = threading.Lock()
        self._progress_guard = threading.Lock()
        self._progress_stop = threading.Event()
        self._progress_thread: threading.Thread | None = None

    def _timestamp(self) -> str:
        """Format local wall-clock time and monotonic elapsed solve seconds.

        Returns:
            str: Timestamp prefix for an immediately emitted progress record.
        """
        return (
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')} +{time.perf_counter() - self._started_at:.3f}s]"
        )

    def _report_status(self) -> None:
        """Flush an atomic pool snapshot even when no candidate improved.

        Raises:
            OSError: If stdout cannot accept progress output.
            ValueError: If the output stream has been closed.
        """
        with self._lock:
            if self.verbose:
                best = f"{self._pool[0][0]:.8g}" if self._pool else "pending"
                costs = ", ".join(
                    f"{value:.8g} ({owner or 'unknown'})" for value, _, owner in self._pool
                )
                print(
                    f"{self._timestamp()} STATUS best={best} "
                    f"pool={len(self._pool)}/{self.capacity} [{costs}]",
                    flush=True,
                )

    def _report_periodically(self) -> None:
        """Emit one snapshot per second until stopped or stdout becomes unavailable."""
        while not self._progress_stop.wait(1.0):
            try:
                self._report_status()
            except OSError, ValueError:
                return

    def start_progress(self) -> None:
        """Start one owner-managed status thread when verbose output is enabled.

        Raises:
            RuntimeError: If the operating system cannot start the reporter thread.
            OSError: If the initial status cannot be written.
            ValueError: If the output stream has been closed.
        """
        with self._progress_guard:
            if not self.verbose or self._progress_thread is not None:
                return
            self._progress_stop.clear()
            self._report_status()
            reporter = threading.Thread(
                target=self._report_periodically, name="rko-progress", daemon=True
            )
            reporter.start()
            self._progress_thread = reporter

    def stop_progress(self) -> None:
        """Join the reporter and flush the final snapshot before the owner returns.

        Raises:
            OSError: If the final status cannot be written.
            ValueError: If the output stream has been closed.
        """
        with self._progress_guard:
            if self._progress_thread is None:
                return
            self._progress_stop.set()
            self._progress_thread.join()
            self._progress_thread = None
            self._report_status()

    def insert(self, entry: tuple[float, list[float]], source: str = "") -> None:
        """Inserts a candidate solution if it improves the pool.

        The pool keeps at most *capacity* coordinate-distinct solutions, sorted
        by cost. Equal/worse observations of the same chromosome are ignored;
        a strictly better observation replaces that chromosome's retained entry.
        Different chromosomes with tied costs remain eligible. A novel entry is
        accepted if capacity remains or its cost strictly beats the worst elite.
        Admission is atomic, not a fitness cache: callbacks have already run.

        Args:
          entry: A ``(cost, keys_list)`` pair describing the candidate.
          source: Human-readable label of the metaheuristic that produced the
            solution, retained internally for attributed progress output.

        Returns:
          None: Mutates the pool in-place when the candidate is accepted.

        Raises:
          ValueError: If the cost or keys are nonfinite, or the vector is empty.
        """
        cost, keys = entry
        keys = list(keys)
        if not math.isfinite(cost) or not keys or not all(math.isfinite(key) for key in keys):
            raise ValueError("Pool entries require a finite cost and a nonempty finite key vector.")
        with self._lock:
            for index, (old_cost, old_keys, _) in enumerate(self._pool):
                if old_keys == keys:
                    if cost >= old_cost:
                        return
                    del self._pool[index]
                    break
            if len(self._pool) < self.capacity or cost < self._pool[-1][0]:
                improved = not self._pool or cost < self._pool[0][0]
                self._pool.append((_ObservedCost.retain(cost), keys, source))
                self._pool.sort(key=lambda t: t[0])
                if len(self._pool) > self.capacity:
                    self._pool.pop()
                if self.verbose and improved and not self._progress_stop.is_set():
                    print(
                        f"{self._timestamp()} NEW BEST "
                        f"algorithm={source or 'unknown'} cost={cost:.8g} best={cost:.8g}",
                        flush=True,
                    )

    def get_elite(self, k: int = 1) -> list[tuple[float, list[float]]]:
        """Returns up to *k* elite solutions sorted by cost (ascending).

        Args:
          k: Number of top solutions to return. Clamped to the current pool
            size if larger.

        Returns:
          list[tuple[float, list[float]]]: Detached copies of the top-k entries
            as ``(cost, keys_list)`` pairs.

        Raises:
          ValueError: When required by external domain validation.
        """
        with self._lock:
            limit = min(k, len(self._pool))
            return [
                (_ObservedCost.for_reader(cost), keys.copy())
                for cost, keys, _ in self._pool[:limit]
            ]

    def _get_elite_pair(self, indices: tuple[int, int]) -> list[tuple[float, list[float]]]:
        """Copy two requested ranks under one lock for process-client sampling.

        Args:
            indices (tuple[int, int]): Distinct nonnegative ranks in request order.

        Returns:
            list[tuple[float, list[float]]]: Two detached entries in request order.

        Raises:
            ValueError: If ranks are malformed or outside current occupancy.
        """
        if (
            not isinstance(indices, tuple)
            or len(indices) != 2
            or any(type(index) is not int or index < 0 for index in indices)
            or indices[0] == indices[1]
        ):
            raise ValueError("Elite ranks must be two distinct nonnegative integers.")
        with self._lock:
            if any(index >= len(self._pool) for index in indices):
                raise ValueError("Elite rank is outside current pool occupancy.")
            return [
                (_ObservedCost.for_reader(self._pool[index][0]), self._pool[index][1].copy())
                for index in indices
            ]

    def __len__(self) -> int:
        """Read current occupancy without copying key vectors.

        Returns:
            int: Stored entry count under the pool lock.
        """
        with self._lock:
            return len(self._pool)

    @property
    def size(self) -> int:
        """Number of solutions currently stored in the pool.

        Returns:
          int: Current pool occupancy.
        """
        return len(self)

    @property
    def best(self) -> tuple[float, list[float]] | None:
        """Best (lowest-cost) solution in the pool, or None when empty.

        Returns:
          tuple[float, list[float]] | None: The entry with the smallest cost,
            or None if the pool has not been populated yet.
        """
        with self._lock:
            if not self._pool:
                return None
            cost, keys, _ = self._pool[0]
            return cost, keys.copy()

    def __repr__(self) -> str:
        """Return a short debug representation of the pool.

        Returns:
          str: String summarising capacity and current occupancy.
        """
        return f"SolutionPool(capacity={self.capacity}, size={self.size})"
