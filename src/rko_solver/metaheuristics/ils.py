"""Metaheuristic Iterated Local Search (ILS) for random-key optimisation.

Objective:
- Escape local optima through repeated perturbation and local refinement,
  retaining only improvements over the incumbent best.

Technical approach:
- Apply RVND after the initial solution construction.
- Each iteration perturbs the best solution with a random shake, decodes the
  result, and refines it with RVND.
- Q-Learning (control=1) adapts beta_min/beta_max online via an epsilon-greedy
  policy over a discrete state space.

Flow (pseudocode):
1. Build initial solution, decode, apply RVND, insert into pool.
2. While time budget not exhausted:
   a. [Q-Learning] choose action, update parameters.
   b. shake(best, beta_min, beta_max) -> sLine
   c. decode sLine; apply RVND.
   d. If improved: update best, insert into pool.
   e. [Q-Learning] compute reward, update Q-table.

Important criteria:
- Acceptance criterion is improvement-only (no Metropolis).
- RVND cycles randomly through swap, mirror, Nelder-Mead, and Farey LS.

Inputs/Outputs:
- Input: solution pool, decoder, cost, solution size, perturbation bounds,
  control mode, and optional Q-Learning parameter grid.
- Output: tuple with empty extras, best key vector, and best cost.
"""

from __future__ import annotations

import numpy as np
import random as random
from ..components.adaptive_control.q_learning import QLearningController
from ..components.local_search.rvnd import rvnd
from ..components.random_streams import RandomStreams
from ..components.shaking.random import shake
from ..components.solution_pool import ElitePool
from ..components.validation import ParameterValidator
from .base import LocalSearchOperator
from .base import MetaheuristicBase
from .base import ShakingOperator
from collections.abc import Callable
from typing import Any
from typing import override


class ILS(MetaheuristicBase):
    """Iterated Local Search over a random-key vector.

    Responsibilities:
    - Perform repeated shake-and-RVND iterations starting from the best
      known solution.
    - Optionally adapt perturbation parameters via Q-Learning.

    Attributes:
      decoder (Callable[[np.ndarray], Any]): Solution decoding function.
      cost (Callable[[Any], float]): Objective function over decoded solutions.
      tam_solution (int): Length of the random-key vector.
      beta_min (float): Minimum relative perturbation intensity.
      beta_max (float): Maximum relative perturbation intensity.
      control (int): 0 = offline (fixed params); 1 = online (Q-Learning).
      parameters (list[list[float]]): Discrete Q-Learning parameter grid
        ``[[beta_min values], [beta_max values]]``.
    """

    def __init__(
        self,
        solution_pool: ElitePool,
        decoder: Callable[[np.ndarray], Any],
        cost: Callable[[Any], float],
        tam_solution: int,
        *,
        beta_min: float = 0.10,
        beta_max: float = 0.40,
        control: int = 0,
        parameters: list[list[float]] | None = None,
        local_search_operator: LocalSearchOperator | None = None,
        shaking_operator: ShakingOperator | None = None,
        name: str = "ILS",
    ) -> None:
        """Initialises the ILS metaheuristic.

        Args:
          solution_pool: Shared pool for exchanging elite solutions.
          decoder: Function that converts a key vector into a problem solution.
          cost: Objective function applied to a decoded solution.
          tam_solution: Length of the random-key vector.
          beta_min: Minimum perturbation intensity (offline mode).
          beta_max: Maximum perturbation intensity (offline mode).
          control: 0 for offline fixed parameters, 1 for Q-Learning.
          parameters: Discrete grid for Q-Learning ``[[bmin...], [bmax...]]``.
          local_search_operator (LocalSearchOperator | None): Optional RVND replacement.
          shaking_operator (ShakingOperator | None): Optional shaking replacement.
          name: Human-readable identifier for logging.

        Returns:
          None: Initialises instance state.

        Raises:
          ValueError: When required by external domain validation.
        """
        super().__init__(
            solution_pool=solution_pool,
            name=name,
            local_search_operator=local_search_operator,
            shaking_operator=shaking_operator,
        )
        self.decoder = decoder
        self.cost = cost
        self.tam_solution = tam_solution
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.control = control
        self.parameters: list[list[float]] = (
            parameters
            if parameters is not None
            else [
                [0.05, 0.10, 0.20, 0.30],
                [0.10, 0.20, 0.30, 0.40],
            ]
        )
        self._state_filter = ParameterValidator.check(self, "ILS")

    @override
    def solve(self, limit_time: float) -> tuple[list[object], list[float], float]:
        """Runs ILS for at most *limit_time* seconds.

        Args:
          limit_time (float): Maximum wall-clock time in seconds.

        Returns:
          tuple[list[object], list[float], float]: Empty extras, best key
            vector, and best objective value found.

        Raises:
          ValueError: When required by external domain validation.
        """
        random = RandomStreams.python_rng()
        numpy_rng = RandomStreams.numpy_rng()
        self._start_timer()
        n = self.tam_solution
        beta_min = self.beta_min
        beta_max = self.beta_max

        ql: QLearningController | None = None
        st = 0
        at = 0
        lf = 1.0
        df = 0.8
        if self.control == 1:
            ql = QLearningController(
                self.parameters,
                epsilon_max=1.0,
                epsilon_min=0.1,
                restart_epsilon=1,
                max_time=limit_time,
                state_filter=self._state_filter,
            )
            st = random.randint(0, max(0, len(ql.states) - 1))
            beta_min = ql.states[st].par[0] if ql.states else beta_min
            beta_max = ql.states[st].par[1] if ql.states else beta_max

        keys: np.ndarray = numpy_rng.random(n)
        keys, current_cost = (self.local_search_operator or rvnd)(
            keys, self.decoder, self.cost, pool=self.solution_pool
        )

        best_keys: np.ndarray = keys.copy()
        best_cost: float = current_cost
        self.solution_pool.insert((best_cost, best_keys.tolist()), self.name)

        while not self.time_limit_reached(limit_time):
            if ql is not None:
                epsilon, lf, df, _, _, _ = ql.set_parameters(self.elapsed())
                at = ql.choose_action(st, epsilon)
                i_curr = ql.states[st].Ai[at]
                beta_min = ql.states[i_curr].par[0]
                beta_max = ql.states[i_curr].par[1]

            s_line = (self.shaking_operator or shake)(best_keys.copy(), beta_min, beta_max)
            s_line, s_line_cost = (self.local_search_operator or rvnd)(
                s_line, self.decoder, self.cost, pool=self.solution_pool
            )

            improv = 0
            if s_line_cost < best_cost:
                best_cost = s_line_cost
                best_keys = s_line.copy()
                improv = 1
                self.solution_pool.insert((best_cost, best_keys.tolist()), self.name)

            if ql is not None:
                reward = 1.0 if improv else ql.relative_reward(best_cost, s_line_cost)
                st_next = ql.states[st].Ai[at]
                ql.update_q_value(st, at, reward, lf, df)
                st = st_next

        return [], best_keys.tolist(), best_cost
