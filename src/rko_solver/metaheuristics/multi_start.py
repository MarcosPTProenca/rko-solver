"""Metaheuristic MultiStart for random-key optimisation.

Objective:
- Search for good solutions by repeatedly generating and evaluating independent
  random starting points without any local refinement.

Technical approach:
- Generate a new uniformly random key vector on every iteration.
- Decode and evaluate each vector immediately.
- Keep track of the global best and update the shared pool on improvement.

Flow (pseudocode):
1. Initialise a random solution and decode it.
2. While time budget not exhausted:
   a. Generate a new random solution and decode it.
   b. If better than current best: update best and insert into pool.

Important criteria:
- No local search or perturbation operators are applied.
- Serves as a baseline and warm-up mechanism for the shared solution pool.
- Reference: docs/references.md, section multi-start

Inputs/Outputs:
- Input: solution pool, decoder, cost, solution size, and time limit.
- Output: tuple with empty extras, best key vector, and best cost.
"""

from __future__ import annotations

import numpy as np
from ..components.random_streams import RandomStreams
from ..components.solution_pool import SolutionPoolWriter
from ..components.validation import ParameterValidator
from .base import MetaheuristicBase
from collections.abc import Callable
from typing import Any
from typing import override


class MultiStart(MetaheuristicBase):
    """Pure random multi-start search over a random-key vector.

    Responsibilities:
    - Repeatedly sample the search space from scratch.
    - Update the shared solution pool whenever a new global best is found.

    Attributes:
      decoder (Callable[[np.ndarray], Any]): Solution decoding function.
      cost (Callable[[Any], float]): Objective function over decoded solutions.
      tam_solution (int): Length of the random-key vector.
    """

    def __init__(
        self,
        solution_pool: SolutionPoolWriter,
        decoder: Callable[[np.ndarray], Any],
        cost: Callable[[Any], float],
        tam_solution: int,
        *,
        name: str = "MultiStart",
    ) -> None:
        """Initialises the MultiStart metaheuristic.

        Args:
          solution_pool: Shared pool for exchanging elite solutions.
          decoder: Function that converts a key vector into a problem solution.
          cost: Objective function applied to a decoded solution.
          tam_solution: Length of the random-key vector.
          name: Human-readable identifier for logging.

        Returns:
          None: Initialises instance state.

        Raises:
          ValueError: When required by external domain validation.
        """
        super().__init__(solution_pool=solution_pool, name=name)
        self.decoder = decoder
        self.cost = cost
        self.tam_solution = tam_solution
        self._state_filter = ParameterValidator.check(self, "MultiStart")

    @override
    def solve(self, limit_time: float) -> tuple[list[object], list[float], float]:
        """Runs MultiStart for at most *limit_time* seconds.

        Args:
          limit_time (float): Maximum wall-clock time in seconds.

        Returns:
          tuple[list[object], list[float], float]: Empty extras, best key
            vector, and best objective value found.

        Raises:
          ValueError: When required by external domain validation.
        """
        numpy_rng = RandomStreams.numpy_rng()
        self._start_timer()
        n = self.tam_solution

        keys: np.ndarray = numpy_rng.random(n)
        best_cost: float = self.cost(self.decoder(keys))
        best_keys: np.ndarray = keys.copy()
        self.solution_pool.insert((best_cost, best_keys.tolist()), self.name)

        while not self.time_limit_reached(limit_time):
            keys = numpy_rng.random(n)
            current_cost: float = self.cost(self.decoder(keys))

            if current_cost < best_cost:
                best_cost = current_cost
                best_keys = keys.copy()
                self.solution_pool.insert((best_cost, best_keys.tolist()), self.name)

        return [], best_keys.tolist(), best_cost
