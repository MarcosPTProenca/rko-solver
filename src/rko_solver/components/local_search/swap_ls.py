"""Pair-swap local search.

Objective:
- Intensify a solution by permuting two positions in the random-key vector.

Technical approach:
- Use sequential acceptance (strategy 1) or fixed-base best improvement (strategy 2).

Flow (pseudocode):
1. Define exploration index order.
2. Generate candidates by pair swaps.
3. Update the best solution according to strategy.

Important criteria:
- Return the original solution when no improvement exists.
- Reference: docs/references.md, section swap-local-search

Inputs/Outputs:
- Input: key vector, decoder/cost callbacks, and strategy.
- Output: improved vector and associated cost.
"""

import numpy as np
import random as random
from ..random_streams import RandomStreams
from collections.abc import Callable
from typing import Any


def swap_ls(
    keys: np.ndarray,
    decoder: Callable[[np.ndarray], Any],
    cost: Callable[[Any], float],
    strategy: int = 2,
    rk_order: list[int] | None = None,
) -> tuple[np.ndarray, float]:
    """Runs local search by pairwise swaps.

    Args:
      keys (np.ndarray): Key vector of the current solution.
      decoder (Callable[[np.ndarray], Any]): Solution decoding function.
      cost (Callable[[Any], float]): Objective function over the decoded solution.
      strategy (int): Sweep policy (1: sequential acceptance; 2: fixed-base best).
      rk_order (list[int] | None): Optional base indices, copied and shuffled for the sweep.

    Returns:
      tuple[np.ndarray, float]: Pair with best key vector and corresponding cost.

    Raises:
      ValueError: When required by external domain validation.
    """
    random = RandomStreams.python_rng()
    n = len(keys)
    if n < 2:
        return keys, cost(decoder(keys))

    order = list(range(n)) if rk_order is None else list(rk_order)
    random.shuffle(order)

    current = keys.copy()
    current_cost = cost(decoder(current))

    best = current.copy()
    best_cost = current_cost

    if strategy == 1:
        for i in range(n - 1):
            for j in range(i + 1, n):
                candidate = best.copy()
                a = order[i]
                b = order[j]
                candidate[a], candidate[b] = candidate[b], candidate[a]

                candidate_cost = cost(decoder(candidate))
                if candidate_cost < best_cost:
                    best = candidate
                    best_cost = candidate_cost
        return best, best_cost

    improved = False
    for i in range(n - 1):
        for j in range(i + 1, n):
            candidate = current.copy()
            a = order[i]
            b = order[j]
            candidate[a], candidate[b] = candidate[b], candidate[a]

            candidate_cost = cost(decoder(candidate))
            if candidate_cost < best_cost:
                best = candidate
                best_cost = candidate_cost
                improved = True

    if improved:
        return best, best_cost
    return current, current_cost
