"""Random Variable Neighborhood Descent (RVND) local search.

Objective:
- Improve a random-key solution by iterating randomly through multiple
  neighborhood structures until none of them produces further improvement.

Technical approach:
- Maintain an active set of four neighborhoods: swap, mirror, Nelder-Mead,
  and Farey interval local searches.
- At each step, pick a neighborhood at random from the active set, apply it,
  and reset the active set on improvement or remove the non-improving
  neighborhood otherwise.

Flow (pseudocode):
1. Build the active neighborhood list with all four operators.
2. Pick a random neighborhood k from the active list.
3. Apply k to the current solution.
4. If improved: update current, reset active list to all four operators.
5. Else: remove k from the active list.
6. Repeat until the active list is empty.
7. Return the best solution found.

Important criteria:
- Pool is forwarded to Nelder-Mead when provided so that elite solutions are
  used as simplex vertices (matching C++ behaviour).
- Swap, mirror and Farey build their own shuffled exploration orders; this
  function has no rk_order parameter.

Inputs/Outputs:
- Input: key vector, decoder, cost, optional strategy, optional pool.
- Output: improved key vector and associated cost.
"""

import numpy as np
import random as random
from ..random_streams import RandomStreams
from ..solution_pool import ElitePool
from .farey_ls import farey_ls
from .mirror_ls import mirror_ls
from .nelder_mead_ls import nelder_mead_search
from .swap_ls import swap_ls
from collections.abc import Callable
from typing import Any


def rvnd(
    keys: np.ndarray,
    decoder: Callable[[np.ndarray], Any],
    cost: Callable[[Any], float],
    strategy: int = 2,
    pool: ElitePool | None = None,
) -> tuple[np.ndarray, float]:
    """Applies Random Variable Neighborhood Descent to a random-key solution.

    Args:
      keys (np.ndarray): Key vector of the starting solution.
      decoder (Callable[[np.ndarray], Any]): Solution decoding function.
      cost (Callable[[Any], float]): Objective function over the decoded solution.
      strategy (int): Improvement strategy passed to each neighborhood operator
        (1 = sequential acceptance, 2 = fixed-base best improvement).
      pool (object | None): Optional elite solution pool forwarded to
        nelder_mead_search for pool-aware simplex initialisation.

    Returns:
      tuple[np.ndarray, float]: Best key vector and corresponding cost found
        after exhausting all neighborhoods.

    Raises:
      ValueError: When required by external domain validation.
    """
    random = RandomStreams.python_rng()
    current = keys.copy()
    current_cost = cost(decoder(current))

    neighborhoods: list[int] = [0, 1, 2, 3]

    while neighborhoods:
        idx = random.randint(0, len(neighborhoods) - 1)
        k = neighborhoods[idx]

        if k == 0:
            candidate, candidate_cost = swap_ls(current.copy(), decoder, cost, strategy)
        elif k == 1:
            candidate, candidate_cost = mirror_ls(current.copy(), decoder, cost, strategy)
        elif k == 2:
            candidate, candidate_cost = nelder_mead_search(current.copy(), decoder, cost, pool=pool)
        else:
            candidate, candidate_cost = farey_ls(current.copy(), decoder, cost, strategy)

        if candidate_cost < current_cost:
            current = candidate
            current_cost = candidate_cost
            neighborhoods = [0, 1, 2, 3]
        else:
            neighborhoods.pop(idx)

    return current, current_cost
