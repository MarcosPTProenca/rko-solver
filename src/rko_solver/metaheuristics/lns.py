"""Large Neighborhood Search (LNS) for random-key optimisation.

Objective:
- Escape local optima by repeatedly destroying and repairing portions of the
  incumbent solution, accepting improvements and occasional deteriorations via
  a Metropolis criterion with reannealing.

Technical approach:
- Destroy phase: randomise a fraction of genes in the key vector.
- Repair phase: restore each destroyed gene by testing every Farey-sequence
  candidate value and keeping the one with the lowest cost.
- Nelder-Mead local search is applied to the repaired solution.
- Temperature is cooled geometrically; reannealing restarts at T0*0.3 when T
  falls below the threshold.
- Q-Learning (control=1) adapts beta_min, beta_max, t0, and alpha_lns
  online via an epsilon-greedy policy over a discrete state space.

Flow (pseudocode):
1. Build initial random solution, apply Nelder-Mead, insert into pool.
2. Outer loop (while time budget not exhausted):
   a. Reannealing: T = T0 if first pass, else T0 * 0.3.
   b. Inner loop (while T > 0.01 and time not exhausted):
      i.   [Q-Learning] choose action, update parameters.
      ii.  Destroy beta fraction of genes.
      iii. Repair each gene with best Farey candidate.
      iv.  Apply Nelder-Mead.
      v.   Metropolis acceptance.
      vi.  Update best, insert pool on improvement.
      vii. Cool T; [Q-Learning] reward, update Q-table.

Important criteria:
- Farey-sequence candidates come from the FAREY_INTERVALS constant in
  ``components.local_search.farey_ls``.
- Reannealing limits wasteful cooling beyond the search horizon.

Inputs/Outputs:
- Input: pool, decoder, cost function, solution size, LNS parameters,
  control mode, optional Q-Learning grid.
- Output: tuple with empty extras, best key vector, and best cost.
"""

from __future__ import annotations

import math
import numpy as np
import random as random
from ..components.adaptive_control.q_learning import QLearningController
from ..components.local_search.farey_ls import FAREY_INTERVALS
from ..components.local_search.nelder_mead_ls import nelder_mead_search
from ..components.random_streams import RandomStreams
from ..components.solution_pool import ElitePool
from ..components.validation import ParameterValidator
from .base import LocalSearchOperator
from .base import MetaheuristicBase
from collections.abc import Callable
from typing import Any
from typing import override

_FAREY_CANDIDATES: list[float] = [v for v in FAREY_INTERVALS if v < 1.0]


class LNS(MetaheuristicBase):
    """Large Neighborhood Search over a random-key vector.

    Responsibilities:
    - Alternately destroy and repair portions of the incumbent key vector.
    - Accept the repaired solution via a Metropolis criterion with reannealing.
    - Optionally adapt algorithm parameters via Q-Learning.

    Attributes:
      decoder (Callable[[np.ndarray], Any]): Solution decoding function.
      cost (Callable[[Any], float]): Objective function over decoded solutions.
      tam_solution (int): Length of the random-key vector.
      beta_min (float): Minimum destroy fraction.
      beta_max (float): Maximum destroy fraction.
      t0 (float): Initial temperature for Metropolis criterion.
      alpha_lns (float): Geometric cooling factor.
      control (int): 0 = offline (fixed params); 1 = online (Q-Learning).
      parameters (list[list[float]]): Discrete Q-Learning parameter grid
        ``[[beta_min], [beta_max], [t0], [alpha]]``.
    """

    def __init__(
        self,
        solution_pool: ElitePool,
        decoder: Callable[[np.ndarray], Any],
        cost: Callable[[Any], float],
        tam_solution: int,
        *,
        beta_min: float = 0.10,
        beta_max: float = 0.30,
        t0: float = 500.0,
        alpha_lns: float = 0.95,
        control: int = 0,
        parameters: list[list[float]] | None = None,
        local_search_operator: LocalSearchOperator | None = None,
        name: str = "LNS",
    ) -> None:
        """Initialises the LNS metaheuristic.

        Args:
          solution_pool: Shared pool for exchanging elite solutions.
          decoder: Function that converts a key vector into a problem solution.
          cost: Objective function applied to a decoded solution.
          tam_solution: Length of the random-key vector.
          beta_min: Minimum destroy fraction (offline mode).
          beta_max: Maximum destroy fraction (offline mode).
          t0: Initial temperature for Metropolis acceptance (offline mode).
          alpha_lns: Geometric cooling factor (offline mode).
          control: 0 for offline fixed parameters, 1 for Q-Learning.
          parameters: Discrete grid for Q-Learning
            ``[[bmin...], [bmax...], [t0...], [alpha...]]``.
          local_search_operator (LocalSearchOperator | None): Optional local-search replacement.
          name: Human-readable identifier for logging.

        Returns:
          None: Initialises instance state.

        Raises:
          ValueError: When required by external domain validation.
        """
        super().__init__(solution_pool, name, local_search_operator=local_search_operator)
        self.decoder = decoder
        self.cost = cost
        self.tam_solution = tam_solution
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.t0 = t0
        self.alpha_lns = alpha_lns
        self.control = control
        self.parameters: list[list[float]] = (
            parameters
            if parameters is not None
            else [
                [0.10, 0.20, 0.30],
                [0.20, 0.30, 0.40],
                [100.0, 500.0, 1000.0],
                [0.90, 0.95, 0.99],
            ]
        )
        self._state_filter = ParameterValidator.check(self, "LNS")

    def _destroy(
        self,
        keys: np.ndarray,
        beta_min: float,
        beta_max: float,
    ) -> tuple[np.ndarray, list[int]]:
        """Destroy a random fraction of the key vector.

        Args:
          keys: Current random-key vector.
          beta_min: Minimum destroy fraction.
          beta_max: Maximum destroy fraction.

        Returns:
          Tuple of (modified key vector, list of destroyed positions).
        """
        random = RandomStreams.python_rng()
        n = len(keys)
        intensity = random.randint(
            max(1, int(beta_min * n)),
            max(1, int(beta_max * n)),
        )
        positions: list[int] = list(range(n))
        random.shuffle(positions)
        destroyed = positions[:intensity]
        result = keys.copy()
        for p in destroyed:
            result[p] = 0.0
        return result, destroyed

    def _repair(
        self,
        keys: np.ndarray,
        destroyed: list[int],
    ) -> tuple[np.ndarray, float]:
        """Repair destroyed positions by testing all Farey candidates.

        Args:
          keys: Partially modified key vector.
          destroyed: Indices of positions to restore.

        Returns:
          Tuple of (repaired key vector, cost of the repaired solution).
        """
        repaired = keys.copy()
        for pos in destroyed:
            best_val = repaired[pos]
            candidate = repaired.copy()
            best_c = self.cost(self.decoder(candidate))
            for fval in _FAREY_CANDIDATES:
                candidate[pos] = fval
                c = self.cost(self.decoder(candidate))
                if c < best_c:
                    best_c = c
                    best_val = fval
                candidate[pos] = best_val
            repaired[pos] = best_val
        final_cost: float = self.cost(self.decoder(repaired))
        return repaired, final_cost

    @override
    def solve(self, limit_time: float) -> tuple[list[object], list[float], float]:
        """Runs LNS for at most *limit_time* seconds.

        Args:
          limit_time: Maximum wall-clock time in seconds.

        Returns:
          Tuple of (empty extras, best key vector as list, best cost).
        """
        random = RandomStreams.python_rng()
        numpy_rng = RandomStreams.numpy_rng()
        self._start_timer()
        n = self.tam_solution

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

        best_keys: np.ndarray = numpy_rng.random(n)
        best_cost: float = self.cost(self.decoder(best_keys))
        best_keys, best_cost = (self.local_search_operator or nelder_mead_search)(
            best_keys, self.decoder, self.cost, pool=self.solution_pool
        )
        self.solution_pool.insert((best_cost, best_keys.tolist()), self.name)

        s = best_keys.copy()
        s_cost = best_cost

        beta_min = self.beta_min
        beta_max = self.beta_max
        t0 = self.t0
        alpha = self.alpha_lns
        if ql is not None and ql.states:
            beta_min = ql.states[st].par[0]
            beta_max = ql.states[st].par[1]
            t0 = ql.states[st].par[2]
            alpha = ql.states[st].par[3]

        first_pass = True
        while not self.time_limit_reached(limit_time):
            temp = t0 if first_pass else t0 * 0.3
            first_pass = False
            while temp > 0.01 and not self.time_limit_reached(limit_time):
                if ql is not None:
                    epsilon, lf, df, _, _, _ = ql.set_parameters(self.elapsed())
                    at = ql.choose_action(st, epsilon)
                    i_curr = ql.states[st].Ai[at]
                    beta_min = ql.states[i_curr].par[0]
                    beta_max = ql.states[i_curr].par[1]
                    t0 = ql.states[i_curr].par[2]
                    alpha = ql.states[i_curr].par[3]

                previous_best = best_cost
                s_line, destroyed = self._destroy(s, beta_min, beta_max)
                s_line, s_line_cost = self._repair(s_line, destroyed)
                s_line, s_line_cost = (self.local_search_operator or nelder_mead_search)(
                    s_line, self.decoder, self.cost, pool=self.solution_pool
                )

                delta = s_line_cost - s_cost
                accepted = False
                if delta <= 0.0:
                    accepted = True
                elif temp > 0.0:
                    accepted = random.random() < math.exp(-delta / temp)

                if accepted:
                    s = s_line.copy()
                    s_cost = s_line_cost
                    if s_cost < best_cost:
                        best_cost = s_cost
                        best_keys = s.copy()
                        self.solution_pool.insert((best_cost, best_keys.tolist()), self.name)

                temp *= alpha

                if ql is not None:
                    reward = ql.relative_reward(previous_best, s_line_cost)
                    st_next = ql.states[st].Ai[at]
                    ql.update_q_value(st, at, reward, lf, df)
                    st = st_next

        return [], best_keys.tolist(), best_cost
