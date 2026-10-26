"""Implicit Path Relinking (IPR) for random-key optimisation.

Objective:
- Explore the solution space between pairs of elite pool solutions by
  iteratively moving blocks of genes from one solution toward another.

Technical approach:
- At each iteration, two solutions are drawn at random from the shared pool.
- Managed pools with more than two entries sample ranks client-side and transfer
  only the chosen pair; other pools retain the existing full-snapshot protocol.
- The key vector is divided into fixed-size blocks (10 % of chromosme length).
- A greedy block-relinking procedure selects, one block at a time, the most
  improving swap from the current position toward the guide solution.
- Multi-gene blocks use NumPy slice assignment; single-gene blocks use direct
  scalar assignment without constructing short-lived array views.
- The best solution encountered along the explored path is inserted back into
  the pool.
- No Q-Learning or auxiliary parameters are required; all decisions are
  deterministic once the pair of solutions is chosen.

Flow (pseudocode):
1. If fewer than two elites exist, insert the initial candidate and at most one fresh seed.
2. Draw ``atual`` and ``guia`` randomly from the pool.
3. Divide the chromosome into ceil(n / block_size) blocks.
4. Repeat max(1, num_blocks-1) times:
   a. For every unfixed block i, evaluate swapping those genes to guia.
   b. Fix the block with the best improvement; update s_current.
5. Insert the best solution seen along the path into the pool.

Important criteria:
- Block size is at least 1 gene; include any partial final block.
- For multiple blocks, keep num_blocks-1 virtual steps; shared blocks can reach the guide earlier.
- Representation-identical moves reuse the known current cost, not a callback.
- Pure deterministic path results/ties are retained; stochastic/stateful callback counts change.
- A one-block chromosome evaluates the guide rather than an empty block.
- Identical effective parents from custom pools use bounded backoff, not relinking.
- Initialization uses the existing callbacks and search deadline, with at most two seed evaluations.
- An already usable elite pool keeps its original initialization RNG draw order.
- Pools unable to retain a pair still use bounded waiting rather than repeated seed retries.
- The routine repeats continuously until the time budget is exhausted,
  drawing a fresh pair from the pool at each IPR call.

Inputs/Outputs:
- Input: pool, decoder, cost function, solution size.
- Output: tuple with empty extras, best key vector found during IPR calls,
  and its cost.
"""

from __future__ import annotations

import math
import numpy as np
import random as random
import time
from ..components.random_streams import RandomStreams
from ..components.solution_pool import ElitePool
from ..components.validation import ParameterValidator
from .base import MetaheuristicBase
from collections.abc import Callable
from typing import Any
from typing import override


class IPR(MetaheuristicBase):
    """Implicit Path Relinking between pool elite solutions.

    Responsibilities:
    - Draw pairs of solutions from the shared pool.
    - Greedily relink them block by block toward the guide solution.
    - Insert the best path solution back into the pool.

    Attributes:
      decoder (Callable[[np.ndarray], Any]): Solution decoding function.
      cost (Callable[[Any], float]): Objective function over decoded solutions.
      tam_solution (int): Length of the random-key vector.
      _replay_contract (str): Diagnostic full-search compatibility token, not an option.
    """

    _replay_contract: str = "ipr-unchanged-block-pruning-v1"

    def __init__(
        self,
        solution_pool: ElitePool,
        decoder: Callable[[np.ndarray], Any],
        cost: Callable[[Any], float],
        tam_solution: int,
        *,
        name: str = "IPR",
    ) -> None:
        """Initialises the IPR metaheuristic.

        Args:
          solution_pool: Shared pool for exchanging elite solutions (needs >= 2).
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
        self._state_filter = ParameterValidator.check(self, "IPR")

    def _relink(
        self,
        atual_keys: np.ndarray,
        guia_keys: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Relink with original virtual steps, omitting representation-identical evaluations.

        A no-op candidate uses the last completed current cost but still participates
        in first-tied selection and consumes its original virtual step. Changed
        candidates remain ordinary callback evaluations. Deterministic pure scoring
        retains the original path/minimum; stateful or stochastic callbacks receive
        fewer evaluations, so their trajectories can change.

        Args:
          atual_keys (np.ndarray): Starting solution chromosome.
          guia_keys (np.ndarray): Guide solution chromosome.

        Returns:
          tuple[np.ndarray, float]: Best chromosome found along the path and its cost.

        Raises:
          Exception: Propagates decoder/objective errors and cooperative stop signals.
        """
        n = self.tam_solution
        block_size = max(1, int(n * 0.10))
        num_blocks = (n + block_size - 1) // block_size
        fixed_block = [False] * num_blocks

        s_current = atual_keys.copy()
        best_path_keys = s_current.copy()
        best_path_cost: float = self.cost(self.decoder(s_current))
        current_cost = best_path_cost

        for _ in range(max(1, num_blocks - 1)):
            best_iter_cost = math.inf
            best_iter_keys: np.ndarray | None = None
            best_block = -1

            for b in range(num_blocks):
                if fixed_block[b]:
                    continue
                start = b * block_size
                end = min((b + 1) * block_size, n)
                if (
                    s_current.dtype == guia_keys.dtype
                    and s_current[start:end].tobytes() == guia_keys[start:end].tobytes()
                ):
                    s_viz = s_current
                    viz_cost = current_cost
                else:
                    s_viz = s_current.copy()
                    if block_size == 1:
                        s_viz[start] = guia_keys[start]
                    else:
                        s_viz[start:end] = guia_keys[start:end]
                    viz_cost = self.cost(self.decoder(s_viz))
                if viz_cost < best_iter_cost:
                    best_iter_cost = viz_cost
                    best_iter_keys = s_viz.copy()
                    best_block = b

            if best_iter_cost < best_path_cost and best_iter_keys is not None:
                best_path_cost = best_iter_cost
                best_path_keys = best_iter_keys.copy()

            if best_block >= 0 and best_iter_keys is not None:
                s_current = best_iter_keys.copy()
                current_cost = best_iter_cost
                fixed_block[best_block] = True

        return best_path_keys, best_path_cost

    @override
    def solve(self, limit_time: float) -> tuple[list[object], list[float], float]:
        """Runs IPR continuously until the time budget is exhausted.

        Args:
          limit_time: Maximum wall-clock time in seconds.

        Returns:
          Tuple of (empty extras, best key vector as list, best cost).
        """
        random = RandomStreams.python_rng()
        numpy_rng = RandomStreams.numpy_rng()
        self._start_timer()

        best_keys: np.ndarray = numpy_rng.random(self.tam_solution)
        best_cost: float = self.cost(self.decoder(best_keys))

        if getattr(self.solution_pool, "size", 0) < 2:
            self.solution_pool.insert((best_cost, best_keys.tolist()), self.name)
            if not self.time_limit_reached(limit_time):
                seed_keys = numpy_rng.random(self.tam_solution)
                seed_cost = self.cost(self.decoder(seed_keys))
                self.solution_pool.insert((seed_cost, seed_keys.tolist()), self.name)
                if seed_cost < best_cost:
                    best_keys, best_cost = seed_keys, seed_cost

        while not self.time_limit_reached(limit_time):
            pool_size: int = getattr(self.solution_pool, "size", 0)
            if pool_size < 2:
                time.sleep(min(0.01, max(0.0, limit_time - self.elapsed())))
                continue

            sample_pair = getattr(self.solution_pool, "_sample_elite_pair", None)
            if sample_pair is None or pool_size == 2:
                elite = self.solution_pool.get_elite(pool_size)
                indices = list(range(len(elite)))
                k1, k2 = random.sample(indices, 2)
            else:
                elite = sample_pair(pool_size)
                k1, k2 = 0, 1
            atual_keys = np.array(elite[k1][1], dtype=float)
            guia_keys = np.array(elite[k2][1], dtype=float)
            if np.array_equal(atual_keys, guia_keys):
                time.sleep(min(0.01, max(0.0, limit_time - self.elapsed())))
                continue

            path_keys, path_cost = self._relink(atual_keys, guia_keys)
            self.solution_pool.insert((path_cost, path_keys.tolist()), self.name)

            if path_cost < best_cost:
                best_cost = path_cost
                best_keys = path_keys.copy()

        return [], best_keys.tolist(), best_cost
