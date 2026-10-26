"""Mirror-based local search.

Objective:
- Intensify a solution by applying complementary transformations to selected genes.

Technical approach:
- Evaluate candidates by mirroring positions according to scan order.
- Support sequential-acceptance and fixed-base best-improvement sweeps.

Flow (pseudocode):
1. Define the order of indices to explore.
2. Generate candidates by mirroring each current gene.
3. Update the best solution according to the strategy.

Important criteria:
- Handle minimum threshold to avoid near-zero values.

Inputs/Outputs:
- Input: key vector, decoder/cost callbacks, and strategy.
- Output: improved vector and resulting cost.
"""

import numpy as np
import random as random
from ..random_streams import RandomStreams
from collections.abc import Callable
from typing import Any


def mirror_ls(
    keys: np.ndarray,
    decoder: Callable[[np.ndarray], Any],
    cost: Callable[[Any], float],
    strategy: int = 2,
    rk_order: list[int] | None = None,
) -> tuple[np.ndarray, float]:
    """Runs local search by mirroring genes.

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
            candidate = best.copy()
            idx = order[i]
            if candidate[idx] > 0.00001:
                candidate[idx] = 1.0 - candidate[idx]
            else:
                candidate[idx] = 0.99999

            candidate_cost = cost(decoder(candidate))
            if candidate_cost < best_cost:
                best = candidate
                best_cost = candidate_cost
        return best, best_cost

    improved = False
    for i in range(n):
        candidate = current.copy()
        idx = order[i]
        if candidate[idx] > 0.00001:
            candidate[idx] = 1.0 - candidate[idx]
        else:
            candidate[idx] = 0.99999

        candidate_cost = cost(decoder(candidate))
        if candidate_cost < best_cost:
            best = candidate
            best_cost = candidate_cost
            improved = True

    if improved:
        return best, best_cost
    return current, current_cost
