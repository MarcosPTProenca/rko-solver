"""Metaheuristic Variable Neighborhood Search (VNS) for random-key optimisation.

Objective:
- Escape local optima by systematically increasing perturbation intensity
  across a sequence of neighborhood structures.

Technical approach:
- Shake intensity scales with the current neighborhood index k, ranging from
  k*beta_min to (k+1)*beta_min.
- RVND is applied after each shake; on improvement k resets to 1, otherwise k
  increments until kMax is reached.
- Q-Learning (control=1) adapts kMax and beta_min online.

Flow (pseudocode):
1. Build initial solution and set s = sBest.
2. While time budget not exhausted:
   a. [Q-Learning] choose action, update parameters.
   b. k = 1; while k <= kMax:
      i.  beta = random(k*beta_min, (k+1)*beta_min)
      ii. shake s by beta; decode; apply RVND.
      iii. If improved over s: s = result; if better than sBest: sBest, k=1, insert pool.
      iv.  Else: k++.
   c. [Q-Learning] compute reward, update Q-table.

Important criteria:
- Perturbation intensity grows with neighborhood depth.
- Acceptance criterion for moving s is improvement over s (not sBest).

Inputs/Outputs:
- Input: solution pool, decoder, cost, solution size, kMax, beta_min, control.
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


class VNS(MetaheuristicBase):
    """Variable Neighborhood Search over a random-key vector.

    Responsibilities:
    - Apply a sequence of increasingly intense perturbations interleaved with
      RVND local search.
    - Optionally adapt kMax and beta_min via Q-Learning.

    Attributes:
      decoder (Callable[[np.ndarray], Any]): Solution decoding function.
      cost (Callable[[Any], float]): Objective function over decoded solutions.
      tam_solution (int): Length of the random-key vector.
      k_max (int): Maximum number of neighborhood structures.
      beta_min (float): Base perturbation intensity per neighborhood level.
      control (int): 0 = offline; 1 = online Q-Learning.
      parameters (list[list[float]]): Discrete Q-Learning grid
        ``[[kMax values], [beta_min values]]``.
    """

    def __init__(
        self,
        solution_pool: ElitePool,
        decoder: Callable[[np.ndarray], Any],
        cost: Callable[[Any], float],
        tam_solution: int,
        *,
        k_max: int = 5,
        beta_min: float = 0.10,
        control: int = 0,
        parameters: list[list[float]] | None = None,
        local_search_operator: LocalSearchOperator | None = None,
        shaking_operator: ShakingOperator | None = None,
        name: str = "VNS",
    ) -> None:
        """Initialises the VNS metaheuristic.

        Args:
          solution_pool: Shared pool for exchanging elite solutions.
          decoder: Function that converts a key vector into a problem solution.
          cost: Objective function applied to a decoded solution.
          tam_solution: Length of the random-key vector.
          k_max: Maximum number of neighborhood structures (offline mode).
          beta_min: Base perturbation intensity (offline mode).
          control: 0 for offline fixed parameters, 1 for Q-Learning.
          parameters: Discrete grid ``[[kMax values], [beta_min values]]``.
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
        self.k_max = k_max
        self.beta_min = beta_min
        self.control = control
        self.parameters: list[list[float]] = (
            parameters
            if parameters is not None
            else [
                [3.0, 5.0, 7.0],
                [0.05, 0.10, 0.20],
            ]
        )
        self._state_filter = ParameterValidator.check(self, "VNS")

    @override
    def solve(self, limit_time: float) -> tuple[list[object], list[float], float]:
        """Runs VNS for at most *limit_time* seconds.

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
        k_max = self.k_max
        beta_min = self.beta_min

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
            k_max = int(ql.states[st].par[0]) if ql.states else k_max
            beta_min = ql.states[st].par[1] if ql.states else beta_min

        keys: np.ndarray = numpy_rng.random(n)
        best_cost: float = self.cost(self.decoder(keys))
        best_keys: np.ndarray = keys.copy()
        s = best_keys.copy()
        s_cost = best_cost
        self.solution_pool.insert((best_cost, best_keys.tolist()), self.name)

        while not self.time_limit_reached(limit_time):
            if ql is not None:
                epsilon, lf, df, _, _, _ = ql.set_parameters(self.elapsed())
                at = ql.choose_action(st, epsilon)
                i_curr = ql.states[st].Ai[at]
                k_max = int(ql.states[i_curr].par[0])
                beta_min = ql.states[i_curr].par[1]

            improv = 0
            k = 1
            while k <= k_max and not self.time_limit_reached(limit_time):
                beta = random.uniform(k * beta_min, (k + 1) * beta_min)
                s_line = (self.shaking_operator or shake)(s.copy(), beta, beta)
                s_line, s_line_cost = (self.local_search_operator or rvnd)(
                    s_line, self.decoder, self.cost, pool=self.solution_pool
                )

                if s_line_cost < s_cost:
                    s = s_line.copy()
                    s_cost = s_line_cost
                    if s_cost < best_cost:
                        best_cost = s_cost
                        best_keys = s.copy()
                        k = 1
                        improv = 1
                        self.solution_pool.insert((best_cost, best_keys.tolist()), self.name)
                        continue
                else:
                    k += 1

            if ql is not None:
                reward = 1.0 if improv else ql.relative_reward(best_cost, s_cost)
                st_next = ql.states[st].Ai[at]
                ql.update_q_value(st, at, reward, lf, df)
                st = st_next

        return [], best_keys.tolist(), best_cost
