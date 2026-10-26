"""Shared contracts and runtime support for RKO metaheuristics.

Objective:
- Define the base lifecycle and composable operator interfaces.

Technical approach:
- Use structural protocols for local-search and shaking callables.
- Store optional operator overrides while preserving algorithm defaults.
- Track elapsed time and cooperative cancellation through one shared base.

Flow (pseudocode):
1. Retain the shared pool and any supplied operator overrides.
2. Let each algorithm run its solve loop using the inherited timer helpers.
3. Return algorithm-specific results through the abstract solve contract.

Important criteria:
- Existing callers may omit optional operators and retain package defaults.
- Time limits are cooperative and checked by concrete algorithm loops.

Inputs/Outputs:
- Input: a pool writer, optional name, and optional operator callables.
- Output: an abstract algorithm base with timing and operator state.
"""

from __future__ import annotations

import numpy as np
import threading
import time
from ..components.solution_pool import ElitePool
from ..components.solution_pool import SolutionPoolWriter
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol

if TYPE_CHECKING:
    from ..orchestration.types import SearchMetadata


class LocalSearchOperator(Protocol):
    """Describe a replaceable local-search strategy.

    Responsibilities:
    - Refine a random-key vector using the supplied problem callbacks.
    - Optionally use the shared elite pool as search context.
    """

    def __call__(
        self,
        keys: np.ndarray,
        decoder: Callable[[np.ndarray], Any],
        cost: Callable[[Any], float],
        *,
        pool: ElitePool | None,
    ) -> tuple[np.ndarray, float]:
        """Apply local search to a random-key vector.

        Args:
            keys (np.ndarray): Initial random-key vector.
            decoder (Callable[[np.ndarray], Any]): Problem decoder.
            cost (Callable[[Any], float]): Objective function.
            pool (ElitePool | None): Optional elite solutions.

        Returns:
            tuple[np.ndarray, float]: Refined keys and their objective value.
        """
        ...


class ShakingOperator(Protocol):
    """Describe a replaceable perturbation strategy.

    Responsibilities:
    - Perturb a random-key vector within an intensity interval.
    """

    def __call__(
        self,
        keys: np.ndarray,
        beta_min: float,
        beta_max: float,
    ) -> np.ndarray:
        """Perturb a random-key vector.

        Args:
            keys (np.ndarray): Input random-key vector.
            beta_min (float): Minimum perturbation intensity.
            beta_max (float): Maximum perturbation intensity.

        Returns:
            np.ndarray: Perturbed random-key vector.
        """
        ...


class MetaheuristicBase(ABC):
    """Abstract base class for all metaheuristic algorithms in the RKO framework.

    Every concrete metaheuristic must inherit from this class and implement
    the :meth:`solve` method, which is the main entry point for the search loop.

    Example::

      class MySA(MetaheuristicBase):
        def solve(self, limit_time: float) -> tuple[list[object], list[float], float]:
          self._start_timer()
          return [], [], 0.0
    """

    def __init__(
        self,
        solution_pool: SolutionPoolWriter,
        name: str = "",
        local_search_operator: LocalSearchOperator | None = None,
        shaking_operator: ShakingOperator | None = None,
    ) -> None:
        """Initialise the metaheuristic.

        Args:
          solution_pool (SolutionPoolWriter): Pool used to exchange solutions.
          name (str): Log label; defaults to the class name.
          local_search_operator (LocalSearchOperator | None): Optional local-search implementation.
          shaking_operator (ShakingOperator | None): Optional perturbation implementation.

        Returns:
          None: Initializes shared pool, operator, and lifecycle state.
        """
        self.solution_pool = solution_pool
        self.name = name or self.__class__.__name__
        self.local_search_operator = local_search_operator
        self.shaking_operator = shaking_operator
        self._start_time: float | None = None
        self._cancelled: threading.Event | None = None
        self._search_metadata: SearchMetadata | None = None

    @abstractmethod
    def solve(self, limit_time: float) -> tuple[list[object], list[float], float]:
        """Run the metaheuristic for at most *limit_time* seconds.

        Subclasses must override this method with their search logic.
        Call :meth:`_start_timer` at the beginning and use
        :meth:`time_limit_reached` to check the stopping criterion inside
        the main loop. Whenever a new best solution is found, insert it into
        the shared pool via ``self.solution_pool.insert((cost, keys), self.name)``.

        Args:
          limit_time: Maximum wall-clock time (seconds) allowed for this run.

        Returns:
          A three-element tuple ``(extra, best_keys, best_cost)`` where:

          - ``extra``   – list of supplementary data produced during the run
                   (use an empty list when not needed),
          - ``best_keys`` – random-key vector (``list[float]``) of the best
                   solution found,
          - ``best_cost`` – objective value (``float``) of that solution.
        """

    def elapsed(self) -> float:
        """Return the number of seconds since :meth:`_start_timer` was called.

        Returns 0.0 if the timer has not been started yet.
        """
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    def time_limit_reached(self, limit_time: float) -> bool:
        """Return ``True`` when :meth:`elapsed` exceeds *limit_time*."""
        return (
            self._cancelled is not None and self._cancelled.is_set()
        ) or self.elapsed() >= limit_time

    def _start_timer(self) -> None:
        """Record the wall-clock start time.

        Call this once at the very beginning of :meth:`solve` so that
        :meth:`elapsed` and :meth:`time_limit_reached` work correctly.
        """
        self._start_time = time.perf_counter()

    def __repr__(self) -> str:
        """Return a concise debug representation for the metaheuristic."""
        return f"{self.__class__.__name__}(name={self.name!r})"
