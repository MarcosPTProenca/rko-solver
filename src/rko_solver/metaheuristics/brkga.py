"""Metaheuristic Biased Random-Key Genetic Algorithm (BRKGA).

Objective:
- Evolve a population of random-key chromosomes toward high-quality solutions
  using biased crossover that preserves elite genetic material.

Technical approach:
- Divide each generation into elite, crossover offspring, and random mutants.
- Parametric Uniform Crossover (PUX): for each gene, inherit from the elite
  parent with probability rhoe, otherwise from the non-elite parent.
- Apply Nelder-Mead local search to a randomly chosen elite chromosome every
  generation.
- Q-Learning (control=1) adapts p, pe, pm, and rhoe online.

Flow (pseudocode):
1. Initialise population of size p, decode, sort ascending.
2. While time budget not exhausted:
   a. [Q-Learning] choose action, update parameters.
   b. Copy top p*pe elites to PopInter.
   c. Fill p*(1-pe-pm) slots with PUX offspring; decode.
   d. Fill p*pm slots with random mutants; decode.
   e. Apply NelderMead to random elite slot.
   f. Sort PopInter; update best; [Q-Learning] reward & update Q-table.

Important criteria:
- Constructor validation requires elite plus mutant counts below population size,
  leaving at least one crossover offspring slot.
- Pool update happens only when the global best improves.

Inputs/Outputs:
- Input: pool, decoder, cost, solution size, population params, control mode.
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


class BRKGA(MetaheuristicBase):
    """Biased Random-Key Genetic Algorithm over a random-key vector.

    Responsibilities:
    - Evolve a population using elite-biased crossover and random mutants.
    - Optionally adapt population parameters via Q-Learning.

    Attributes:
      decoder (Callable[[np.ndarray], Any]): Solution decoding function.
      cost (Callable[[Any], float]): Objective function over decoded solutions.
      tam_solution (int): Length of the random-key vector.
      p (int): Population size.
      pe (float): Elite fraction (0, 1).
      pm (float): Mutant fraction (0, 1).
      rhoe (float): Elite inheritance probability per gene.
      control (int): 0 = offline; 1 = online Q-Learning.
      parameters (list[list[float]]): Discrete Q-Learning grid
        ``[[p], [pe], [pm], [rhoe]]``.
    """

    def __init__(
        self,
        solution_pool: ElitePool,
        decoder: Callable[[np.ndarray], Any],
        cost: Callable[[Any], float],
        tam_solution: int,
        *,
        p: int = 50,
        pe: float = 0.15,
        pm: float = 0.10,
        rhoe: float = 0.70,
        control: int = 0,
        parameters: list[list[float]] | None = None,
        local_search_operator: LocalSearchOperator | None = None,
        name: str = "BRKGA",
    ) -> None:
        """Initialises the BRKGA metaheuristic.

        Args:
          solution_pool: Shared pool for exchanging elite solutions.
          decoder: Function that converts a key vector into a problem solution.
          cost: Objective function applied to a decoded solution.
          tam_solution: Length of the random-key vector.
          p: Population size (offline mode).
          pe: Elite fraction (offline mode).
          pm: Mutant fraction (offline mode).
          rhoe: Elite gene inheritance probability (offline mode).
          control: 0 for offline fixed parameters, 1 for Q-Learning.
          parameters: Discrete grid ``[[p values], [pe], [pm], [rhoe]]``.
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
        self.p = p
        self.pe = pe
        self.pm = pm
        self.rhoe = rhoe
        self.control = control
        self.parameters: list[list[float]] = (
            parameters
            if parameters is not None
            else [
                [30.0, 50.0, 100.0],
                [0.10, 0.15, 0.20],
                [0.05, 0.10, 0.15],
                [0.60, 0.70, 0.80],
            ]
        )
        self._state_filter = ParameterValidator.check(self, "BRKGA")

    def _pux(
        self,
        pop: list[np.ndarray],
        elite_size: int,
        rhoe: float,
    ) -> np.ndarray:
        """Parametric Uniform Crossover between one elite and one non-elite parent.

        Args:
          pop (list[np.ndarray]): Current population sorted by cost ascending.
          elite_size (int): Number of elite chromosomes at the front of pop.
          rhoe (float): Probability of inheriting gene from the elite parent.

        Returns:
          np.ndarray: New offspring chromosome.

        Raises:
          ValueError: When required by external domain validation.
        """
        random = RandomStreams.python_rng()
        n = self.tam_solution
        elite_idx = random.randint(0, elite_size - 1)
        non_elite_idx = random.randint(elite_size, len(pop) - 1)
        child = np.empty(n, dtype=float)
        for j in range(n):
            child[j] = pop[elite_idx][j] if random.random() < rhoe else pop[non_elite_idx][j]
        return child

    @override
    def solve(self, limit_time: float) -> tuple[list[object], list[float], float]:
        """Runs BRKGA for at most *limit_time* seconds.

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
        p = self.p
        pe = self.pe
        pm = self.pm
        rhoe = self.rhoe

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
                p = int(ql.states[st].par[0])
                pe = ql.states[st].par[1]
                pm = ql.states[st].par[2]
                rhoe = ql.states[st].par[3]

        pop: list[np.ndarray] = [numpy_rng.random(n) for _ in range(p)]
        pop_costs: list[float] = [self.cost(self.decoder(c)) for c in pop]
        order = sorted(range(p), key=lambda i: pop_costs[i])
        pop = [pop[i] for i in order]
        pop_costs = [pop_costs[i] for i in order]

        best_keys: np.ndarray = pop[0].copy()
        best_cost: float = pop_costs[0]
        self.solution_pool.insert((best_cost, best_keys.tolist()), self.name)

        while not self.time_limit_reached(limit_time):
            if ql is not None:
                epsilon, lf, df, _, _, _ = ql.set_parameters(self.elapsed())
                at = ql.choose_action(st, epsilon)
                i_curr = ql.states[st].Ai[at]
                p = int(ql.states[i_curr].par[0])
                pe = ql.states[i_curr].par[1]
                pm = ql.states[i_curr].par[2]
                rhoe = ql.states[i_curr].par[3]
                while len(pop) < p:
                    new_c = numpy_rng.random(n)
                    pop.append(new_c)
                    pop_costs.append(self.cost(self.decoder(new_c)))
                if len(pop) > p:
                    pop = pop[:p]
                    pop_costs = pop_costs[:p]

            elite_size = max(1, int(p * pe))
            mutant_count = max(0, int(p * pm))
            cross_count = p - elite_size - mutant_count

            pop_inter: list[np.ndarray] = list(pop[:elite_size])
            costs_inter: list[float] = list(pop_costs[:elite_size])

            best_off_cost = float("inf")
            for _ in range(cross_count):
                child = self._pux(pop, elite_size, rhoe)
                child_cost = self.cost(self.decoder(child))
                pop_inter.append(child)
                costs_inter.append(child_cost)
                if child_cost < best_off_cost:
                    best_off_cost = child_cost

            for _ in range(mutant_count):
                mutant = numpy_rng.random(n)
                mutant_cost = self.cost(self.decoder(mutant))
                pop_inter.append(mutant)
                costs_inter.append(mutant_cost)

            ls_pos = random.randint(0, elite_size - 1)
            ls_keys, ls_cost = (self.local_search_operator or nelder_mead_search)(
                pop_inter[ls_pos].copy(), self.decoder, self.cost, pool=self.solution_pool
            )
            pop_inter[ls_pos] = ls_keys
            costs_inter[ls_pos] = ls_cost

            order = sorted(range(len(pop_inter)), key=lambda i: costs_inter[i])
            pop = [pop_inter[i] for i in order]
            pop_costs = [costs_inter[i] for i in order]

            improv = 0
            if pop_costs[0] < best_cost:
                best_cost = pop_costs[0]
                best_keys = pop[0].copy()
                improv = 1
                self.solution_pool.insert((best_cost, best_keys.tolist()), self.name)

            if ql is not None:
                reward = 1.0 if improv else ql.relative_reward(best_cost, best_off_cost)
                st_next = ql.states[st].Ai[at]
                ql.update_q_value(st, at, reward, lf, df)
                st = st_next

        return [], best_keys.tolist(), best_cost
