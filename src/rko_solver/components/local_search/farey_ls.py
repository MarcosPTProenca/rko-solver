"""Farey-interval based local search.

Objective:
- Intensify a solution by exploring substitutions across discretized Farey ranges.

Technical approach:
- Select genes and test candidates by sampling predefined intervals.
- Support sequential-acceptance and fixed-base best-improvement sweeps.

Flow (pseudocode):
1. Define the evaluation order of genes.
2. For each gene, generate candidates from Farey intervals.
3. Update the best solution according to the selected strategy.

Important criteria:
- Preserve the base vector when no improvement is found.
- Reference: docs/references.md, section farey-local-search

Inputs/Outputs:
- Input: key vector, decoder/cost callbacks, and strategy.
- Output: best found vector and associated cost.
"""

import numpy as np
import random as random
from ..random_streams import RandomStreams
from collections.abc import Callable
from typing import Any

FAREY_INTERVALS = [
    0.00,
    0.142857,
    0.166667,
    0.20,
    0.25,
    0.285714,
    0.333333,
    0.40,
    0.428571,
    0.50,
    0.571429,
    0.60,
    0.666667,
    0.714286,
    0.75,
    0.80,
    0.833333,
    0.857143,
    1.0,
]


def farey_ls(
    keys: np.ndarray,
    decoder: Callable[[np.ndarray], Any],
    cost: Callable[[Any], float],
    strategy: int = 2,
    rk_order: list[int] | None = None,
) -> tuple[np.ndarray, float]:
    """Runs local search with discretization through Farey intervals.

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
    if n == 0:
        return keys, cost(decoder(keys))

    order = list(range(n)) if rk_order is None else list(rk_order)
    random.shuffle(order)

    current = keys.copy()
    current_cost = cost(decoder(current))

    best = current.copy()
    best_cost = current_cost

    if strategy == 1:
        for i in range(n):
            idx = order[i]
            for j in range(len(FAREY_INTERVALS) - 1):
                candidate = best.copy()
                candidate[idx] = random.uniform(FAREY_INTERVALS[j], FAREY_INTERVALS[j + 1])

                candidate_cost = cost(decoder(candidate))
                if candidate_cost < best_cost:
                    best = candidate
                    best_cost = candidate_cost
        return best, best_cost

    improved = False
    for i in range(n):
        idx = order[i]
        for j in range(len(FAREY_INTERVALS) - 1):
            candidate = current.copy()
            candidate[idx] = random.uniform(FAREY_INTERVALS[j], FAREY_INTERVALS[j + 1])

            candidate_cost = cost(decoder(candidate))
            if candidate_cost < best_cost:
                best = candidate
                best_cost = candidate_cost
                improved = True

    if improved:
        return best, best_cost
    return current, current_cost
