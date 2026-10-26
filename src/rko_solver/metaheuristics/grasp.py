"""Greedy Randomised Adaptive Search Procedure (GRASP) for random-key optimisation.

Objective:
- Iteratively construct and improve solutions by combining a greedy-randomised
  constructive phase with random variable neighbourhood descent (RVND).

Technical approach:
- Private instance methods perform line search and greedy-randomised construction,
  adding the best gene value to the partial solution at each construction step.
  The subset size and order are randomised to yield diverse starting points.
- Improvement phase: RVND refines the constructed solution.
- Step h is halved only when no global improvement is found; it stays at the
  current value when a new best is discovered and restarts from hs at each new
  Q-Learning state, matching the reference grid progression.  A Metropolis
  criterion based on elapsed time controls acceptance of non-improving moves.
- Construction shuffles a fresh copy of unfixed coordinates, retaining their order.
- Q-Learning selects initial/final step settings online; the reference alpha axis
  is retained in the state graph but does not influence construction directly.
- Sample a bounded subset of the wider reference lattice without materializing it.
- Round the construction fraction upward and scan all small remaining coordinate sets.

Flow (pseudocode):
1. Build and score an initial random solution; insert into pool without refinement.
2. While time budget not exhausted:
   a. [Q-Learning] choose action, retrieve (alpha_grasp, hs, he).
   b. h = hs; while h >= he and time not exhausted:
      i.   Greedy-randomised construction.
      ii.  RVND.
      iii. If improved globally: update best, insert pool, keep h.
      iv.  Else: halve h.
      v.   Metropolis acceptance for local incumbent.
   c. [Q-Learning] reward, update Q-table.

Important criteria:
- Line-search samples at most ceil(log2(max(1,floor(1/h))))+1 eligible positions.
- The incumbent may be omitted, allowing exploratory non-improving construction.
- Exact-ratio bounds avoid reciprocal overflow; tiny steps need only O(log(1/h)) slots.
- Selected shifted values stay in [0,1); an existing endpoint-one incumbent is retained.
- Temperature for Metropolis decreases linearly from 100 to 0 over limit_time.

Inputs/Outputs:
- Input: pool, decoder, cost function, solution size, GRASP parameters,
  control mode, optional Q-Learning grid.
- Output: tuple with empty extras, best key vector, and best cost.
"""

from __future__ import annotations

import math
import numpy as np
import random as random
from ..components.adaptive_control.q_learning import QLearningController
from ..components.local_search.rvnd import rvnd
from ..components.random_streams import RandomStreams
from ..components.solution_pool import ElitePool
from ..components.validation import ParameterValidator
from .base import LocalSearchOperator
from .base import MetaheuristicBase
from collections.abc import Callable
from numbers import Real
from typing import Any
from typing import override


class GRASP(MetaheuristicBase):
    """Greedy Randomised Adaptive Search Procedure over a random-key vector.

    Responsibilities:
    - Alternate construction and RVND improvement phases.
    - Progressively refine the line-search step h to intensify the search.
    - Optionally adapt algorithm parameters via Q-Learning.

    Attributes:
      decoder (Callable[[np.ndarray], Any]): Solution decoding function.
      cost (Callable[[Any], float]): Objective function over decoded solutions.
      tam_solution (int): Length of the random-key vector.
      hs (float): Initial line-search step size.
      he (float): Minimum line-search step size.
      control (int): Zero for fixed hs/he, one for adaptive step settings.
      parameters (list[list[float]]): Discrete Q-Learning parameter grid
        ``[[alpha_grasp values], [hs values], [he values]]``; alpha is retained for
        reference state-graph compatibility, not constructive candidate selection.
      _replay_contract (str): Internal RNG-replay compatibility token, not an option.
    """

    _replay_contract: str = "grasp-construction-v4"

    def __init__(
        self,
        solution_pool: ElitePool,
        decoder: Callable[[np.ndarray], Any],
        cost: Callable[[Any], float],
        tam_solution: int,
        *,
        hs: float = 0.25,
        he: float = 0.001,
        control: int = 0,
        parameters: list[list[float]] | None = None,
        local_search_operator: LocalSearchOperator | None = None,
        name: str = "GRASP",
    ) -> None:
        """Initialises the GRASP metaheuristic.

        Args:
          solution_pool: Shared pool for exchanging elite solutions.
          decoder: Function that converts a key vector into a problem solution.
          cost: Objective function applied to a decoded solution.
          tam_solution: Length of the random-key vector.
          hs: Initial line-search step size (offline mode).
          he: Minimum line-search step size (offline mode).
          control: 0 for offline fixed parameters, 1 for Q-Learning.
          parameters: Discrete grid for Q-Learning
            ``[[alpha_grasp...], [hs...], [he...]]``. Alpha labels affect adaptive
            states/actions, not the constructive kernel.
          local_search_operator (LocalSearchOperator | None): Optional RVND replacement.
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
        self.hs = hs
        self.he = he
        self.control = control
        self.parameters: list[list[float]] = (
            parameters
            if parameters is not None
            else [
                [0.1, 0.3, 0.5, 0.7, 0.9],
                [0.50, 0.25, 0.10],
                [0.01, 0.001, 0.0001],
            ]
        )
        self._state_filter = ParameterValidator.check(self, "GRASP")

    def _line_search(self, pos: int, h: float, keys: np.ndarray) -> tuple[float, float]:
        """Sample the reference lattice uniformly without replacement using sparse slots.

        Exact float ratios define eligible shifts through floor(1/h)/2+1 steps
        per sign. Rank zero is the incumbent; subsequent ranks interleave signs
        until one side ends. Sparse swap-removal samples only the logarithmic
        callback budget, including when reciprocal counts exceed machine integers.

        Args:
            pos (int): Coordinate to vary; other coordinates remain fixed.
            h (float): Positive finite representable step.
            keys (np.ndarray): Caller-owned random keys, never mutated.

        Returns:
            tuple[float, float]: First sampled minimum and its objective, possibly
                worse than an incumbent omitted from the subset.

        Raises:
            ValueError: If the step is invalid or the coordinate is outside [0,1].
            IndexError: If the coordinate is out of range.
            Exception: Original callback or cooperative stopping failures propagate.
        """
        message = "GRASP line-search step must be a positive finite number."
        if isinstance(h, bool) or not isinstance(h, Real):
            raise ValueError(message)
        try:
            step = float(h)
        except OverflowError as exc:
            raise ValueError(message) from exc
        if not math.isfinite(step) or step <= 0:
            raise ValueError(message)
        z = float(keys[pos])
        if not 0 <= z <= 1:
            raise ValueError("GRASP line-search coordinate must be finite and in [0,1].")
        step_num, step_den = step.as_integer_ratio()
        z_num, z_den = z.as_integer_ratio()
        divisions = max(1, step_den // step_num)
        max_shift = divisions // 2 + 1
        origin = z_num * step_den
        stride = step_num * z_den
        denominator = z_den * step_den
        positive = min(max_shift, max(0, (denominator - origin - 1) // stride))
        negative = min(max_shift, origin // stride)
        paired = min(positive, negative)
        remaining = 1 + positive + negative
        samples = min(remaining, (divisions - 1).bit_length() + 1)
        random = RandomStreams.python_rng()
        swaps: dict[int, int] = {}
        best_val, best_cost = z, math.inf
        for _ in range(samples):
            index = random.randrange(remaining)
            rank = swaps.get(index, index)
            remaining -= 1
            swaps[index] = swaps.get(remaining, remaining)
            value = z
            if rank:
                if rank <= 2 * paired:
                    tau = (rank + 1) // 2
                    direction = 1 if rank % 2 else -1
                else:
                    tau = rank - paired
                    direction = 1 if positive > negative else -1
                value = (origin + direction * tau * stride) / denominator
                value = min(value, math.nextafter(1.0, 0.0))
            candidate = np.asarray(keys, dtype=float).copy()
            candidate[pos] = value
            candidate_cost = self.cost(self.decoder(candidate))
            if candidate_cost < best_cost:
                best_cost, best_val = candidate_cost, value
        return best_val, best_cost

    def _constructive(self, h: float, keys: np.ndarray) -> tuple[np.ndarray, float]:
        """Commit ceil(n*intensity) coordinates using the reference subset cardinality.

        Each pass considers floor(0.1*remaining) coordinates, or every remaining
        coordinate if that count is below two. A fresh shuffled copy selects the
        subset, leaving unfixed order intact like the reference; each candidate
        sees the same already committed partial state.

        Args:
            h (float): Line-search step size.
            keys (np.ndarray): Starting random-key vector.

        Returns:
            tuple[np.ndarray, float]: Constructed keys and their objective value.

        Raises:
            ValueError: If a line search rejects the step or varying coordinate.
            Exception: Propagates callback failures and cooperative stop signals.
        """
        random = RandomStreams.python_rng()
        intensity = math.ceil(random.uniform(0.3, 0.7) * self.tam_solution)
        unfixed: list[int] = list(range(self.tam_solution))
        solution = keys.copy()
        for _ in range(min(intensity, len(unfixed))):
            k_max = int(len(unfixed) * 0.1)
            if k_max < 2:
                k_max = len(unfixed)
            chosen = unfixed.copy()
            random.shuffle(chosen)
            chosen = chosen[:k_max]
            best_pos = chosen[0]
            best_cost = math.inf
            best_value = solution[best_pos]
            for pos in chosen:
                value, candidate_cost = self._line_search(pos, h, solution)
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_pos = pos
                    best_value = value
            solution[best_pos] = best_value
            unfixed.remove(best_pos)
        return solution, self.cost(self.decoder(solution))

    @override
    def solve(self, limit_time: float) -> tuple[list[object], list[float], float]:
        """Runs GRASP for at most *limit_time* seconds.

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
        self.solution_pool.insert((best_cost, best_keys.tolist()), self.name)

        s = best_keys.copy()
        s_cost = best_cost

        hs_cur = self.hs
        he_cur = self.he

        if ql is not None and ql.states:
            hs_cur = ql.states[st].par[1]
            he_cur = ql.states[st].par[2]

        while not self.time_limit_reached(limit_time):
            previous_best = best_cost
            if ql is not None:
                epsilon, lf, df, _, _, _ = ql.set_parameters(self.elapsed())
                at = ql.choose_action(st, epsilon)
                i_curr = ql.states[st].Ai[at]
                hs_cur = ql.states[i_curr].par[1]
                he_cur = ql.states[i_curr].par[2]

            h = hs_cur
            while h >= he_cur and not self.time_limit_reached(limit_time):
                improved = False
                s_line, _ = self._constructive(h, s.copy())
                s_line, s_line_cost = (self.local_search_operator or rvnd)(
                    s_line, self.decoder, self.cost, pool=self.solution_pool
                )

                if s_line_cost < best_cost:
                    best_cost = s_line_cost
                    best_keys = s_line.copy()
                    improved = True
                    self.solution_pool.insert((best_cost, best_keys.tolist()), self.name)

                if s_line_cost < s_cost:
                    s = s_line.copy()
                    s_cost = s_line_cost
                else:
                    elapsed = self.elapsed()
                    t_metropolis = max(1e-9, 100.0 - 100.0 * (elapsed / limit_time))
                    delta = s_line_cost - s_cost
                    if random.random() < math.exp(-delta / t_metropolis):
                        s = s_line.copy()
                        s_cost = s_line_cost

                if not improved:
                    h /= 2.0

            if ql is not None:
                reward = 1.0 if best_cost < previous_best else ql.relative_reward(best_cost, s_cost)
                st_next = ql.states[st].Ai[at]
                ql.update_q_value(st, at, reward, lf, df)
                st = st_next

        return [], best_keys.tolist(), best_cost
