"""Metaheuristic Particle Swarm Optimization (PSO) for random-key optimisation.

Objective:
- Explore the random-key space by simulating a swarm of particles whose
  velocities are influenced by personal and global bests.

Technical approach:
- Each particle has a position X in [0,1)^n and a velocity V.
- Velocity update: V = w*(V + c1*r1*(Pbest-X) + c2*r2*(Gbest-X)).
- Position update: X += V; if X falls outside [0,1) revert to old X and set V=0.
- Nelder-Mead local search is applied to a randomly selected Pbest each
  generation.
- Q-Learning (control=1) adapts p_size, c1, c2, and w online.
- Batch independent genes with NumPy at 32+ dimensions, retaining RNG draw order.

Flow (pseudocode):
1. Initialise X, V (random), Pbest = X, Gbest = argmin(X).
2. While time budget not exhausted:
   a. [Q-Learning] choose action, update parameters.
   b. For each particle: update V and X; clamp; decode; update Pbest/Gbest.
   c. Apply NelderMead to random Pbest.
   d. If Gbest improved: insert into pool.
   e. [Q-Learning] reward & Q-table update.

Important criteria:
- Out-of-bounds positions are reverted individually per gene.
- Pool update only when Gbest strictly improves.

Inputs/Outputs:
- Input: pool, decoder, cost, solution size, swarm params, control mode.
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


class PSO(MetaheuristicBase):
    """Particle Swarm Optimization over a random-key vector.

    Responsibilities:
    - Update particle positions using inertia, cognitive, and social forces.
    - Optionally adapt swarm parameters via Q-Learning.

    Attributes:
      decoder (Callable[[np.ndarray], Any]): Solution decoding function.
      cost (Callable[[Any], float]): Objective function over decoded solutions.
      tam_solution (int): Length of the random-key vector.
      p_size (int): Number of particles.
      c1 (float): Cognitive acceleration coefficient.
      c2 (float): Social acceleration coefficient.
      w (float): Inertia weight.
      control (int): 0 = offline; 1 = online Q-Learning.
      parameters (list[list[float]]): Discrete Q-Learning grid
        ``[[p_size], [c1], [c2], [w]]``.
    """

    def __init__(
        self,
        solution_pool: ElitePool,
        decoder: Callable[[np.ndarray], Any],
        cost: Callable[[Any], float],
        tam_solution: int,
        *,
        p_size: int = 30,
        c1: float = 2.0,
        c2: float = 2.0,
        w: float = 0.7,
        control: int = 0,
        parameters: list[list[float]] | None = None,
        local_search_operator: LocalSearchOperator | None = None,
        name: str = "PSO",
    ) -> None:
        """Initialises the PSO metaheuristic.

        Args:
          solution_pool: Shared pool for exchanging elite solutions.
          decoder: Function that converts a key vector into a problem solution.
          cost: Objective function applied to a decoded solution.
          tam_solution: Length of the random-key vector.
          p_size: Number of particles (offline mode).
          c1: Cognitive coefficient (offline mode).
          c2: Social coefficient (offline mode).
          w: Inertia weight (offline mode).
          control: 0 for offline fixed parameters, 1 for Q-Learning.
          parameters: Discrete grid ``[[p_size], [c1], [c2], [w]]``.
          local_search_operator: Optional replacement for the default local search.
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
        self.p_size = p_size
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.control = control
        self.parameters: list[list[float]] = (
            parameters
            if parameters is not None
            else [
                [20.0, 30.0, 50.0],
                [1.5, 2.0, 2.5],
                [1.5, 2.0, 2.5],
                [0.4, 0.7, 0.9],
            ]
        )
        self._state_filter = ParameterValidator.check(self, "PSO")

    @staticmethod
    def _move_particle(
        position: np.ndarray,
        velocity: np.ndarray,
        personal_best: np.ndarray,
        global_best: np.ndarray,
        c1: float,
        c2: float,
        inertia: float,
    ) -> None:
        """Update one particle in place, preserving scalar arithmetic and RNG order.

        Args:
            position (np.ndarray): Current random keys, mutated in place.
            velocity (np.ndarray): Current velocities, mutated in place.
            personal_best (np.ndarray): Detached best keys for this particle.
            global_best (np.ndarray): Detached current swarm-best keys.
            c1 (float): Cognitive acceleration coefficient.
            c2 (float): Social acceleration coefficient.
            inertia (float): Multiplier applied to the complete velocity update.
        """
        random = RandomStreams.python_rng()
        n = len(position)
        if n < 32:
            for j in range(n):
                r1 = random.random()
                r2 = random.random()
                velocity[j] = inertia * (
                    velocity[j]
                    + c1 * r1 * (personal_best[j] - position[j])
                    + c2 * r2 * (global_best[j] - position[j])
                )
                old_key = position[j]
                position[j] = position[j] + velocity[j]
                if position[j] < 0.0 or position[j] >= 1.0:
                    position[j] = old_key
                    velocity[j] = 0.0
            return
        draws = np.fromiter(
            (random.random() for _ in range(2 * n)), dtype=float, count=2 * n
        ).reshape(n, 2)
        velocity += c1 * draws[:, 0] * (personal_best - position)
        velocity += c2 * draws[:, 1] * (global_best - position)
        velocity *= inertia
        proposed = position + velocity
        invalid = (proposed < 0.0) | (proposed >= 1.0)
        velocity[invalid] = 0.0
        np.copyto(position, proposed, where=~invalid)

    @override
    def solve(self, limit_time: float) -> tuple[list[object], list[float], float]:
        """Runs PSO for at most *limit_time* seconds.

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
        p_size = self.p_size
        c1 = self.c1
        c2 = self.c2
        w = self.w

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
                p_size = int(ql.states[st].par[0])
                c1 = ql.states[st].par[1]
                c2 = ql.states[st].par[2]
                w = ql.states[st].par[3]

        x: list[np.ndarray] = [numpy_rng.random(n) for _ in range(p_size)]
        v: list[np.ndarray] = [numpy_rng.random(n) for _ in range(p_size)]
        x_costs: list[float] = [self.cost(self.decoder(xi)) for xi in x]
        pbest: list[np.ndarray] = [xi.copy() for xi in x]
        pbest_costs: list[float] = list(x_costs)

        gbest_idx = min(range(p_size), key=lambda i: x_costs[i])
        gbest: np.ndarray = x[gbest_idx].copy()
        gbest_cost: float = x_costs[gbest_idx]
        self.solution_pool.insert((gbest_cost, gbest.tolist()), self.name)

        while not self.time_limit_reached(limit_time):
            if ql is not None:
                epsilon, lf, df, _, _, _ = ql.set_parameters(self.elapsed())
                at = ql.choose_action(st, epsilon)
                i_curr = ql.states[st].Ai[at]
                p_size = int(ql.states[i_curr].par[0])
                c1 = ql.states[i_curr].par[1]
                c2 = ql.states[i_curr].par[2]
                w = ql.states[i_curr].par[3]
                while len(x) < p_size:
                    new_xi = numpy_rng.random(n)
                    new_vi = numpy_rng.random(n)
                    new_cost = self.cost(self.decoder(new_xi))
                    x.append(new_xi)
                    v.append(new_vi)
                    x_costs.append(new_cost)
                    pbest.append(new_xi.copy())
                    pbest_costs.append(new_cost)
                if len(x) > p_size:
                    x = x[:p_size]
                    v = v[:p_size]
                    x_costs = x_costs[:p_size]
                    pbest = pbest[:p_size]
                    pbest_costs = pbest_costs[:p_size]

            best_of_current = float("inf")
            improv = 0
            old_gbest_cost = gbest_cost

            for i in range(len(x)):
                self._move_particle(x[i], v[i], pbest[i], gbest, c1, c2, w)

                x_costs[i] = self.cost(self.decoder(x[i]))
                if x_costs[i] < best_of_current:
                    best_of_current = x_costs[i]

                if x_costs[i] < pbest_costs[i]:
                    pbest[i] = x[i].copy()
                    pbest_costs[i] = x_costs[i]

                if x_costs[i] < gbest_cost:
                    gbest = x[i].copy()
                    gbest_cost = x_costs[i]
                    improv = 1

            chosen = random.randint(0, len(pbest) - 1)
            ls_keys, ls_cost = (self.local_search_operator or nelder_mead_search)(
                pbest[chosen].copy(), self.decoder, self.cost, pool=self.solution_pool
            )
            if ls_cost < pbest_costs[chosen]:
                pbest[chosen] = ls_keys
                pbest_costs[chosen] = ls_cost
                if ls_cost < gbest_cost:
                    gbest = ls_keys.copy()
                    gbest_cost = ls_cost
                    improv = 1

            if gbest_cost < old_gbest_cost:
                self.solution_pool.insert((gbest_cost, gbest.tolist()), self.name)

            if ql is not None:
                reward = 1.0 if improv else ql.relative_reward(gbest_cost, best_of_current)
                st_next = ql.states[st].Ai[at]
                ql.update_q_value(st, at, reward, lf, df)
                st = st_next

        return [], gbest.tolist(), gbest_cost
