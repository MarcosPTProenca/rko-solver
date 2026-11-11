"""Manager-hosted elite pool proxy for process workers.

Objective:
- Give spawned workers the same pool API as `SolutionPool`, served by one manager process.

Technical approach:
- `PoolManager` registers `SolutionPool` for remote construction; `PoolProxy` forwards each
  exposed method through the manager connection and returns detached copies.
- Pair sampling draws the two ranks in the client and transfers only those two entries.

Flow (pseudocode):
1. Register `SolutionPool` with `PoolProxy` as its proxy type.
2. A worker calls a method; the proxy sends it to the manager and returns the detached result.
3. Pair sampling draws ranks from the observed occupancy, then fetches just those entries.

Important criteria:
- Used only when progress output or decision tracing needs the manager-hosted pool.

Inputs/Outputs:
- Input: pool method names and arguments from worker processes.
- Output: manager results as detached Python values.
"""

from __future__ import annotations

from ..components.random_streams import RandomStreams
from ..components.solution_pool import SolutionPool
from multiprocessing.managers import BaseManager
from multiprocessing.managers import BaseProxy
from typing import Any


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

    def _call(self, name: str, args: tuple[object, ...] = ()) -> Any:  # noqa: ANN401
        """Forward one exposed method to the manager and return its value.

        Args:
            name (str): Exposed method name.
            args (tuple[object, ...]): Positional arguments for the method.

        Returns:
            Any: Detached value returned by the manager.
        """
        return self._callmethod(name, args)  # type: ignore[func-returns-value]

    def insert(self, entry: tuple[float, list[float]], source: str = "") -> None:
        """Insert a candidate into the authoritative pool.

        Args:
            entry (tuple[float, list[float]]): Finite objective and key vector.
            source (str): Producing algorithm label.

        Raises:
            ValueError: If the candidate violates pool validation.
        """
        self._call("insert", (entry, source))

    def start_progress(self) -> None:
        """Start periodic status reporting inside the authoritative manager."""
        self._call("start_progress")

    def stop_progress(self) -> None:
        """Join the managed reporter and flush its last snapshot before shutdown."""
        self._call("stop_progress")

    def get_elite(self, k: int = 1) -> list[tuple[float, list[float]]]:
        """Fetch detached elite entries from the manager.

        Args:
            k (int): Maximum number of entries to retrieve.

        Returns:
            list[tuple[float, list[float]]]: Cost-ordered elite snapshots.
        """
        return self._call("get_elite", (k,))

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
        return self._call("_get_elite_pair", ((first, second),))

    @property
    def size(self) -> int:
        """Return the number of managed entries.

        Returns:
            int: Current pool occupancy.
        """
        return self._call("__len__")

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
