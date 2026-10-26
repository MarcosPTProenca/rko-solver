"""Metaheuristic Genetic Algorithm (GA) for random-key optimisation.

Objective:
- Evolve a population of random-key chromosomes via tournament selection,
  uniform crossover, and per-gene mutation.

Technical approach:
- 3-way tournament selection fills an intermediate population.
- Uniform crossover swaps alleles gene-by-gene when applied (prob_cros gate).
- Per-gene mutation replaces a gene with a new random value (prob_mut gate).
- Nelder-Mead local search is applied to a randomly selected offspring.
- Q-Learning (control=1) adapts size_pop, prob_cros, and prob_mut online.

Flow (pseudocode):
1. Initialise population of size_pop, decode all.
2. While time budget not exhausted:
   a. [Q-Learning] choose action, update parameters.
   b. Tournament selection -> PopInter.
   c. Paired crossover + mutation -> PopNew; decode all changed pairs.
   d. Apply NelderMead to random individual in PopNew.
   e. Update best; [Q-Learning] reward & Q-table update.

Important criteria:
- Only evaluates fitness after crossover is applied (gate by prob_cros).
- Local search is applied exactly once per generation on a random individual.

Inputs/Outputs:
- Input: pool, decoder, cost, solution size, GA params, control mode.
- Output: tuple with empty extras, best key vector, and best cost.
"""

from __future__ import annotations

import numpy as np
import random as random
from ..components.adaptive_control.q_learning import QLearningController
from ..components.local_search.nelder_mead_ls import nelder_mead_search
from ..components.random_streams import RandomStreams
from ..components.solution_pool import ElitePool
from ..components.validation import ParameterValidator
from .base import LocalSearchOperator
from .base import MetaheuristicBase
from collections.abc import Callable
from typing import Any
from typing import override


class GeneticAlgorithm(MetaheuristicBase):
    """Genetic Algorithm over a random-key vector.

    Responsibilities:
    - Evolve a population using tournament selection, uniform crossover, and
      per-gene mutation.
    - Optionally adapt GA parameters via Q-Learning.

    Attributes:
      decoder (Callable[[np.ndarray], Any]): Solution decoding function.
      cost (Callable[[Any], float]): Objective function over decoded solutions.
      tam_solution (int): Length of the random-key vector.
      size_pop (int): Population size.
      prob_cros (float): Gate probability for applying crossover to a pair.
      prob_mut (float): Per-gene mutation probability.
      control (int): 0 = offline; 1 = online Q-Learning.
      parameters (list[list[float]]): Discrete Q-Learning grid
        ``[[size_pop], [prob_cros], [prob_mut]]``.
    """

    def __init__(
        self,
        solution_pool: ElitePool,
        decoder: Callable[[np.ndarray], Any],
        cost: Callable[[Any], float],
        tam_solution: int,
        *,
        size_pop: int = 50,
        prob_cros: float = 0.80,
        prob_mut: float = 0.02,
        control: int = 0,
        parameters: list[list[float]] | None = None,
        local_search_operator: LocalSearchOperator | None = None,
        name: str = "GA",
    ) -> None:
        """Initialises the Genetic Algorithm metaheuristic.

        Args:
          solution_pool: Shared pool for exchanging elite solutions.
          decoder: Function that converts a key vector into a problem solution.
          cost: Objective function applied to a decoded solution.
          tam_solution: Length of the random-key vector.
          size_pop: Population size (offline mode).
          prob_cros: Crossover gate probability per pair (offline mode).
          prob_mut: Per-gene mutation probability (offline mode).
          control: 0 for offline fixed parameters, 1 for Q-Learning.
          parameters: Discrete grid ``[[size_pop], [prob_cros], [prob_mut]]``.
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
        self.size_pop = size_pop
        self.prob_cros = prob_cros
        self.prob_mut = prob_mut
        self.control = control
        self.parameters: list[list[float]] = (
            parameters
            if parameters is not None
            else [
                [30.0, 50.0, 100.0],
                [0.60, 0.80, 0.90],
                [0.01, 0.02, 0.05],
            ]
        )
        self._state_filter = ParameterValidator.check(self, "GA")

    @override
    def solve(self, limit_time: float) -> tuple[list[object], list[float], float]:
        """Runs the Genetic Algorithm for at most *limit_time* seconds.

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
        size_pop = self.size_pop
        prob_cros = self.prob_cros
        prob_mut = self.prob_mut

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
            if ql.states:
                size_pop = int(ql.states[st].par[0])
                prob_cros = ql.states[st].par[1]
                prob_mut = ql.states[st].par[2]

        pop: list[np.ndarray] = [numpy_rng.random(n) for _ in range(size_pop)]
        pop_costs: list[float] = [self.cost(self.decoder(c)) for c in pop]

        best_idx = min(range(size_pop), key=lambda i: pop_costs[i])
        best_keys: np.ndarray = pop[best_idx].copy()
        best_cost: float = pop_costs[best_idx]
        self.solution_pool.insert((best_cost, best_keys.tolist()), self.name)

        while not self.time_limit_reached(limit_time):
            if ql is not None:
                epsilon, lf, df, _, _, _ = ql.set_parameters(self.elapsed())
                at = ql.choose_action(st, epsilon)
                i_curr = ql.states[st].Ai[at]
                size_pop = int(ql.states[i_curr].par[0])
                prob_cros = ql.states[i_curr].par[1]
                prob_mut = ql.states[i_curr].par[2]
                while len(pop) < size_pop:
                    new_c = numpy_rng.random(n)
                    pop.append(new_c)
                    pop_costs.append(self.cost(self.decoder(new_c)))
                if len(pop) > size_pop:
                    pop = pop[:size_pop]
                    pop_costs = pop_costs[:size_pop]

            pop_inter: list[np.ndarray] = []
            selected_costs: list[float] = []
            for _ in range(size_pop):
                p1, p2, p3 = random.sample(range(size_pop), 3)
                best_t = min(p1, p2, p3, key=lambda i: pop_costs[i])
                pop_inter.append(pop[best_t].copy())
                selected_costs.append(pop_costs[best_t])

            del pop, pop_costs
            pop_new: list[np.ndarray] = [c.copy() for c in pop_inter]
            pop_new_costs: list[float] = list(selected_costs)

            best_off_cost = float("inf")
            i = 0
            while i < size_pop - 1:
                if random.random() < prob_cros:
                    for j in range(n):
                        if random.random() < 0.5:
                            pop_new[i][j] = pop_inter[i + 1][j]
                        if random.random() < 0.5:
                            pop_new[i + 1][j] = pop_inter[i][j]
                        if random.random() <= prob_mut:
                            pop_new[i][j] = random.random()
                        if random.random() <= prob_mut:
                            pop_new[i + 1][j] = random.random()
                    pop_new_costs[i] = self.cost(self.decoder(pop_new[i]))
                    pop_new_costs[i + 1] = self.cost(self.decoder(pop_new[i + 1]))

                if pop_new_costs[i] < best_off_cost:
                    best_off_cost = pop_new_costs[i]
                if pop_new_costs[i + 1] < best_off_cost:
                    best_off_cost = pop_new_costs[i + 1]
                i += 2

            ls_pos = random.randint(0, size_pop - 1)
            ls_keys, ls_cost = (self.local_search_operator or nelder_mead_search)(
                pop_new[ls_pos].copy(), self.decoder, self.cost, pool=self.solution_pool
            )
            pop_new[ls_pos] = ls_keys
            pop_new_costs[ls_pos] = ls_cost
            if ls_cost < best_off_cost:
                best_off_cost = ls_cost

            pop = pop_new
            pop_costs = pop_new_costs

            improv = 0
            current_best_idx = min(range(size_pop), key=lambda i: pop_costs[i])
            if pop_costs[current_best_idx] < best_cost:
                best_cost = pop_costs[current_best_idx]
                best_keys = pop[current_best_idx].copy()
                improv = 1
                self.solution_pool.insert((best_cost, best_keys.tolist()), self.name)

            if ql is not None:
                reward = 1.0 if improv else ql.relative_reward(best_cost, best_off_cost)
                st_next = ql.states[st].Ai[at]
                ql.update_q_value(st, at, reward, lf, df)
                st = st_next

        return [], best_keys.tolist(), best_cost
