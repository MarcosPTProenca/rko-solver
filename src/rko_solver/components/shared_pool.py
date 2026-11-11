"""Shared-memory elite pool for process workers.

Objective:
- Give spawned workers direct access to the elite pool without a manager round trip, so that
  each insert or read costs a lock and a memory copy instead of a pickled call.

Technical approach:
- One `multiprocessing.shared_memory` block holds an int64 occupancy counter, the float64 costs
  sorted ascending, and the float64 key matrix. A spawn-context lock serializes every access.
- Admission matches `SolutionPool`: exact key equality removes the older entry only when the new
  cost is strictly better; a novel entry enters at the stable position after equal costs when
  capacity remains or it beats the worst retained entry.
- Children attach by block name on unpickling, without registering with the resource tracker,
  so only the creating process unlinks the block.

Flow (pseudocode):
1. Create: allocate counter, costs and keys; create the lock; set occupancy to zero.
2. Insert: validate; under the lock, delete an equal-key entry if the new cost is better, then
   place the entry at its sorted position and drop the worst one if over capacity.
3. Read: under the lock, copy the requested entries as plain floats and lists.
4. Close: release the views, then the block handle; unlink only in the creating process.

Important criteria:
- Same observable order, deduplication and occupancy as `SolutionPool` (differential-tested).
- Progress output is not supported; the caller uses this pool only when verbose output and
  decision tracing are off.

Inputs/Outputs:
- Input: capacity, key dimension, a spawn context for the lock, and candidate entries.
- Output: detached entries as `(float, list[float])` pairs, and occupancy counts.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import numpy as np
from ..components.random_streams import RandomStreams
from ..components.solution_pool import validate_rank_pair
from multiprocessing import shared_memory
from multiprocessing.context import BaseContext
from multiprocessing.synchronize import Lock
from typing import Any

_COUNTER_BYTES = 8


class SharedElitePool:
    """Fixed-capacity elite pool stored in shared memory and shared by spawned workers.

    Responsibilities:
    - Admit, order and deduplicate candidates exactly as `SolutionPool`.
    - Serve detached copies to metaheuristics in any attached process.

    Attributes:
        capacity (int): Maximum number of retained entries.
        dimension (int): Length of every key vector.
    """

    def __init__(
        self,
        capacity: int,
        dimension: int,
        block: shared_memory.SharedMemory,
        lock: Lock,
        owner: bool,
    ) -> None:
        """Bind views onto a shared block.

        Args:
            capacity (int): Maximum number of retained entries.
            dimension (int): Length of every key vector.
            block (shared_memory.SharedMemory): Attached or created shared block.
            lock (Lock): Lock shared by every process using this pool.
            owner (bool): True only in the process that created the block.
        """
        self.capacity = capacity
        self.dimension = dimension
        self._block = block
        self._lock = lock
        self._owner = owner
        self._bind()

    def _bind(self) -> None:
        """Create the counter, cost and key views over the shared buffer."""
        buffer = self._block.buf
        costs_offset = _COUNTER_BYTES
        keys_offset = costs_offset + 8 * self.capacity
        self._count = np.ndarray((1,), dtype=np.int64, buffer=buffer, offset=0)
        self._costs = np.ndarray(
            (self.capacity,), dtype=np.float64, buffer=buffer, offset=costs_offset
        )
        self._keys = np.ndarray(
            (self.capacity, self.dimension), dtype=np.float64, buffer=buffer, offset=keys_offset
        )

    @classmethod
    def create(
        cls, capacity: int, dimension: int, context: BaseContext | None = None
    ) -> SharedElitePool:
        """Allocate an empty pool and the lock shared with spawned children.

        Args:
            capacity (int): Maximum number of retained entries, at least one.
            dimension (int): Length of every key vector, at least one.
            context (BaseContext | None): Multiprocessing context for the lock; spawn by default.

        Returns:
            SharedElitePool: Owner of the new shared block.

        Raises:
            ValueError: If capacity or dimension is smaller than one.
        """
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer.")
        if type(dimension) is not int or dimension < 1:
            raise ValueError("dimension must be a positive integer.")
        size = _COUNTER_BYTES + 8 * capacity + 8 * capacity * dimension
        block = shared_memory.SharedMemory(create=True, size=size)
        lock = (context or mp.get_context("spawn")).Lock()
        pool = cls(capacity, dimension, block, lock, owner=True)
        pool._count[0] = 0
        return pool

    def __getstate__(self) -> dict[str, Any]:
        """Describe the block by name and share the lock with the child.

        Returns:
            dict[str, Any]: Pickle state without views or ownership.
        """
        return {
            "capacity": self.capacity,
            "dimension": self.dimension,
            "name": self._block.name,
            "lock": self._lock,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Attach to the existing block in a child process without owning it.

        Args:
            state (dict[str, Any]): Values produced by `__getstate__`.
        """
        self.capacity = state["capacity"]
        self.dimension = state["dimension"]
        self._block = shared_memory.SharedMemory(name=state["name"], track=False)
        self._lock = state["lock"]
        self._owner = False
        self._bind()

    def close(self) -> None:
        """Release the views and this process's handle to the block."""
        self._count = self._costs = self._keys = None  # type: ignore[assignment]
        self._block.close()

    def unlink(self) -> None:
        """Remove the block when called by its creator; other processes do nothing.

        Raises:
            FileNotFoundError: If the block was already removed by someone else.
        """
        if self._owner:
            self._block.unlink()

    def start_progress(self) -> None:
        """Accept the progress hook of `SolutionPool`; this pool never reports progress."""

    def stop_progress(self) -> None:
        """Accept the progress hook of `SolutionPool`; this pool never reports progress."""

    def insert(self, entry: tuple[float, list[float]], source: str = "") -> None:
        """Insert a candidate if it improves the pool, as `SolutionPool.insert` does.

        Args:
            entry (tuple[float, list[float]]): Finite objective and key vector.
            source (str): Producing algorithm label; not retained, as progress is off.

        Raises:
            ValueError: If the cost or keys are nonfinite, empty or of the wrong length.
        """
        cost, vector = self._checked_entry(entry)
        with self._lock:
            count = int(self._count[0])
            for index in range(count):
                if np.array_equal(self._keys[index], vector):
                    if cost >= self._costs[index]:
                        return
                    count = self._delete(index, count)
                    break
            if count < self.capacity or cost < self._costs[count - 1]:
                position = int(np.searchsorted(self._costs[:count], cost, side="right"))
                self._place(position, count, cost, vector)
                self._count[0] = min(count + 1, self.capacity)

    def _checked_entry(self, entry: tuple[float, list[float]]) -> tuple[float, np.ndarray]:
        """Validate a candidate and convert its keys to a float array.

        Args:
            entry (tuple[float, list[float]]): Finite objective and key vector.

        Returns:
            tuple[float, np.ndarray]: Cost and key vector of the pool's dimension.

        Raises:
            ValueError: If the cost or keys are nonfinite, empty or of the wrong length.
        """
        cost, keys = entry
        values = list(keys)
        if not math.isfinite(cost) or not values or not all(math.isfinite(v) for v in values):
            raise ValueError("Pool entries require a finite cost and a nonempty finite key vector.")
        if len(values) != self.dimension:
            raise ValueError("Pool entries must have the pool's key dimension.")
        return cost, np.asarray(values, dtype=np.float64)

    def _delete(self, index: int, count: int) -> int:
        """Remove one entry by shifting the later entries left.

        Args:
            index (int): Position of the entry to remove.
            count (int): Current occupancy.

        Returns:
            int: Occupancy after the removal.
        """
        self._costs[index : count - 1] = self._costs[index + 1 : count].copy()
        self._keys[index : count - 1] = self._keys[index + 1 : count].copy()
        self._count[0] = count - 1
        return count - 1

    def _place(self, position: int, count: int, cost: float, vector: np.ndarray) -> None:
        """Insert an entry at its sorted position, dropping the worst one when full.

        Args:
            position (int): Sorted insertion index, at most the current occupancy.
            count (int): Current occupancy.
            cost (float): Cost of the new entry.
            vector (np.ndarray): Key vector of the new entry.
        """
        end = count if count < self.capacity else self.capacity - 1
        self._costs[position + 1 : end + 1] = self._costs[position:end].copy()
        self._keys[position + 1 : end + 1] = self._keys[position:end].copy()
        self._costs[position] = cost
        self._keys[position] = vector

    def get_elite(self, k: int = 1) -> list[tuple[float, list[float]]]:
        """Return up to *k* best entries as detached copies, cost ascending.

        Args:
            k (int): Maximum number of entries.

        Returns:
            list[tuple[float, list[float]]]: Copied `(cost, keys)` pairs.
        """
        with self._lock:
            limit = min(k, int(self._count[0]))
            return [(float(self._costs[i]), self._keys[i].tolist()) for i in range(limit)]

    def _get_elite_pair(self, indices: tuple[int, int]) -> list[tuple[float, list[float]]]:
        """Copy two requested ranks under one lock, as the process proxy does.

        Args:
            indices (tuple[int, int]): Distinct nonnegative ranks in request order.

        Returns:
            list[tuple[float, list[float]]]: Two detached entries in request order.

        Raises:
            ValueError: If ranks are malformed or outside current occupancy.
        """
        validate_rank_pair(indices)
        with self._lock:
            count = int(self._count[0])
            if any(index >= count for index in indices):
                raise ValueError("Elite rank is outside current pool occupancy.")
            return [(float(self._costs[i]), self._keys[i].tolist()) for i in indices]

    def _sample_elite_pair(self, pool_size: int) -> list[tuple[float, list[float]]]:
        """Draw two ranks client-side from a previously observed occupancy.

        Args:
            pool_size (int): Occupancy observed by the caller, at least two.

        Returns:
            list[tuple[float, list[float]]]: Detached pair in random selection order.

        Raises:
            ValueError: If the supplied count is invalid.
        """
        if type(pool_size) is not int or pool_size < 2:
            raise ValueError("Pair sampling requires an integer pool size of at least two.")
        first, second = RandomStreams.python_rng().sample(list(range(pool_size)), 2)
        return self._get_elite_pair((first, second))

    def __len__(self) -> int:
        """Return the current occupancy under the lock.

        Returns:
            int: Stored entry count.
        """
        with self._lock:
            return int(self._count[0])

    @property
    def size(self) -> int:
        """Number of stored entries.

        Returns:
            int: Current occupancy.
        """
        return len(self)

    @property
    def best(self) -> tuple[float, list[float]] | None:
        """Best entry as a detached copy, or None when empty.

        Returns:
            tuple[float, list[float]] | None: Lowest-cost entry.
        """
        with self._lock:
            if int(self._count[0]) == 0:
                return None
            return float(self._costs[0]), self._keys[0].tolist()
