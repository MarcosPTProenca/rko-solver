"""Simulated annealing with optional per-temperature Q-learning control.

Objective:
- Minimize arbitrary decoded objectives using thermal acceptance and local refinement.

Technical approach:
- Shake the current keys, accept improvements or Metropolis-selected deteriorations.
- In online mode choose feasible named parameter states at each temperature level.
- Credit completed levels, including refinement, using sign-safe bounded rewards.

Flow (pseudocode):
1. Select initial settings and evaluate one owned-stream random candidate.
2. Choose online parameters, evaluate the level's neighbors and cool geometrically.
3. Refine locally, update the incumbent/Q state and repeat within the search budget.

Important criteria:
- Offline control preserves the original fixed-parameter transition/draw sequence.
- A selected t0 governs the next cooling schedule, never resets an active temperature.
- Completed schedules reanneal at 0.3 times the latest selected t0.
- Empty interrupted levels receive no Q update; callbacks retain shared stopping guards.

Inputs/Outputs:
- Input: elite pool, decoder/objective, fixed settings and an optional feasible Q grid.
- Output: extras, detached incumbent keys and its minimization objective.
"""

from __future__ import annotations

import math
import numpy as np
import random as random
from ..components.adaptive_control.q_learning import QLearningController
from ..components.local_search.nelder_mead_ls import nelder_mead_search
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


class SA(MetaheuristicBase):
    """Search random-key chromosomes with fixed or adaptive annealing schedules.

    Responsibilities:
    - Apply Metropolis acceptance and preserve the best evaluated solution.
    - Adapt feasible parameters per completed temperature level when requested.
    - Preserve operator overrides and privately scoped random streams.

    Attributes:
        decoder (Callable): Decode a random-key chromosome.
        cost (Callable): Score a decoded problem solution.
        tam_solution (int): Positive chromosome dimension.
        control (int): Zero for fixed settings, one for Q-learning.
        parameters (list[list[float]]): Axes sa_max, t0, alpha, beta_min, beta_max.
        sa_max (int): Offline proposals per temperature level.
        t0 (float): Offline initial/reannealing temperature scale.
        alpha (float): Offline geometric cooling factor.
        beta_min (float): Offline minimum shaking intensity.
        beta_max (float): Offline maximum shaking intensity.
    """

    def __init__(
        self,
        solution_pool: ElitePool,
        decoder: Callable[[np.ndarray], Any],
        cost: Callable[[Any], float],
        tam_solution: int,
        *,
        sa_max: int = 50,
        t0: float = 10_000.0,
        alpha: float = 0.99,
        beta_min: float = 0.05,
        beta_max: float = 0.25,
        control: int = 0,
        parameters: list[list[float]] | None = None,
        local_search_operator: LocalSearchOperator | None = None,
        shaking_operator: ShakingOperator | None = None,
        name: str = "SA",
    ) -> None:
        """Validate fixed settings and optional named-axis adaptive parameters.

        Args:
          solution_pool (ElitePool): Shared elite context.
          decoder (Callable[[np.ndarray], Any]): Decode a chromosome.
          cost (Callable[[Any], float]): Evaluate the decoded solution.
          tam_solution (int): Positive random-key dimension.
          sa_max (int): Fixed proposal count per temperature level.
          t0 (float): Fixed initial temperature scale.
          alpha (float): Fixed geometric cooling rate.
          beta_min (float): Fixed minimum shaking intensity.
          beta_max (float): Fixed maximum shaking intensity.
          control (int): Zero for offline or one for adaptive control.
          parameters (list[list[float]] | None): Axes sa_max, t0, alpha, beta_min, beta_max.
          local_search_operator (LocalSearchOperator | None): Optional Nelder-Mead replacement.
          shaking_operator (ShakingOperator | None): Optional shaking replacement.
          name (str): Search identifier used by results and progress.

        Returns:
          None: Initializes validated search settings.

        Raises:
          ValueError: If the fixed settings, control flag or online states are invalid.
        """
        super().__init__(
            solution_pool=solution_pool,
            name=name,
            local_search_operator=local_search_operator,
            shaking_operator=shaking_operator,
        )
        self.tam_solution = tam_solution
        self.decoder = decoder
        self.cost = cost
        self.sa_max = sa_max
        self.t0 = t0
        self.alpha = alpha
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.control = control
        self.parameters: list[list[float]] = (
            parameters
            if parameters is not None
            else [
                [250, 500, 750, 1000],
                [100000.0, 500000.0, 1000000.0],
                [0.97, 0.98, 0.99],
                [0.01, 0.03, 0.05],
                [0.05, 0.08, 0.10],
            ]
        )
        self._state_filter = ParameterValidator.check(self, "SA")

    @override
    def solve(self, limit_time: float) -> tuple[list[object], list[float], float]:
        """Run Simulated Annealing for at most *limit_time* seconds.

        Args:
          limit_time (float): Maximum search duration in seconds.

        Returns:
          tuple[list[object], list[float], float]: Extras, best keys and best cost.

        Raises:
          Exception: Original callback, operator or controller failures are propagated.
        """
        random = RandomStreams.python_rng()
        numpy_rng = RandomStreams.numpy_rng()
        self._start_timer()
        n = self.tam_solution
        sa_max, t0, alpha = self.sa_max, self.t0, self.alpha
        beta_min, beta_max = self.beta_min, self.beta_max
        ql: QLearningController | None = None
        st = 0
        if self.control == 1:
            ql = QLearningController(
                self.parameters,
                epsilon_max=1.0,
                epsilon_min=0.1,
                restart_epsilon=1,
                max_time=limit_time,
                state_filter=self._state_filter,
            )
            st = random.randint(0, len(ql.states) - 1)
            sa_max, t0, alpha, beta_min, beta_max = ql.states[st].par
            sa_max = int(sa_max)
        keys: np.ndarray = numpy_rng.random(n)
        current_cost: float = self.cost(self.decoder(keys))

        best_keys: np.ndarray = keys.copy()
        best_cost: float = current_cost
        self.solution_pool.insert((best_cost, best_keys.tolist()), self.name)

        reannealing: bool = False
        while not self.time_limit_reached(limit_time):
            T: float = t0 if not reannealing else t0 * 0.3
            while T > 0.0001 and not self.time_limit_reached(limit_time):
                previous_best = best_cost
                if ql is not None:
                    epsilon, lf, df, _, _, _ = ql.set_parameters(self.elapsed())
                    at = ql.choose_action(st, epsilon)
                    next_state = ql.states[st].Ai[at]
                    sa_max, t0, alpha, beta_min, beta_max = ql.states[next_state].par
                    sa_max = int(sa_max)
                best_ofv_level: float = math.inf
                iter_t: int = 0
                while iter_t < sa_max and not self.time_limit_reached(limit_time):
                    iter_t += 1
                    viz_keys: np.ndarray = (self.shaking_operator or shake)(
                        keys.copy(), beta_min, beta_max
                    )
                    viz_cost: float = self.cost(self.decoder(viz_keys))
                    if viz_cost < best_ofv_level:
                        best_ofv_level = viz_cost

                    delta: float = viz_cost - current_cost

                    if delta < 0:
                        keys = viz_keys
                        current_cost = viz_cost

                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_keys = keys.copy()
                            self.solution_pool.insert((best_cost, best_keys.tolist()), self.name)
                    else:
                        if random.random() < math.exp(-delta / T):
                            keys = viz_keys
                            current_cost = viz_cost
                T *= alpha
                ls_keys, ls_cost = (self.local_search_operator or nelder_mead_search)(
                    keys.copy(), self.decoder, self.cost, pool=self.solution_pool
                )
                if ls_cost < best_cost:
                    best_cost = ls_cost
                    best_keys = ls_keys.copy()
                    keys = ls_keys
                    current_cost = ls_cost
                    self.solution_pool.insert((best_cost, best_keys.tolist()), self.name)
                if ql is not None and iter_t > 0:
                    reward = (
                        1.0
                        if best_cost < previous_best
                        else ql.relative_reward(best_cost, best_ofv_level)
                    )
                    ql.update_q_value(st, at, reward, lf, df)
                    st = next_state
            reannealing = True

        return [], best_keys.tolist(), best_cost
