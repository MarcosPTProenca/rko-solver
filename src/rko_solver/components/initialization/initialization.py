"""Initialization and maintenance of the solution pool.

Objective:
- Build initial random-key solutions and keep an elite pool sorted by cost.

Technical approach:
- Generate random vectors and evaluate them with decoder/cost callbacks.
- Remove duplicate costs with controlled perturbation and reordering.

Flow (pseudocode):
1. Generate an initial random population.
2. Evaluate and sort by objective function value.
3. Fix duplicates and update the pool with better candidates.

Important criteria:
- Preserve pool size.
- Avoid duplicate entries by objective value.

Inputs/Outputs:
- Input: pool size, solution dimension, and evaluation callbacks.
- Output: sorted list of (cost, key vector) pairs.
"""

import numpy as np
from ...tracing import _ObservedCost
from ..random_streams import RandomStreams
from collections.abc import Callable
from typing import Any


def create_initial_solution(tam_solution: int) -> np.ndarray:
    """Creates an initial solution in random-key format.

    Args:
        tam_solution (int): Key vector dimension.

    Returns:
        np.ndarray: Randomly initialized key vector.

    Raises:
        ValueError: When required by external domain validation.
    """
    numpy_rng = RandomStreams.numpy_rng()
    if tam_solution <= 0:
        return np.array([], dtype=float)
    return numpy_rng.random(tam_solution)


def create_pool_solutions(
    size_pool: int,
    tam_solution: int,
    decoder: Callable[[np.ndarray], Any],
    cost: Callable[[Any], float],
) -> list[tuple[float, np.ndarray]]:
    """Creates and sorts a random-key solution pool.

    Args:
        size_pool (int): Number of solutions in the pool.
        tam_solution (int): Dimension of each key vector.
        decoder (Callable[[np.ndarray], Any]): Solution decoding function.
        cost (Callable[[Any], float]): Objective function over the decoded solution.

    Returns:
        list[tuple[float, np.ndarray]]: Pool sorted by increasing cost.

    Raises:
        ValueError: When required by external domain validation.
    """
    numpy_rng = RandomStreams.numpy_rng()
    pool: list[tuple[float, np.ndarray]] = []

    for _ in range(max(0, size_pool)):
        keys = create_initial_solution(tam_solution)
        ofv = _ObservedCost.cast(cost(decoder(keys)))
        pool.append((ofv, keys))

    pool.sort(key=lambda item: item[0])

    clone_fixed = False
    for i in range(len(pool) - 1, 0, -1):
        if pool[i][0] == pool[i - 1][0]:
            keys = pool[i][1].copy()
            num_changes = int(0.2 * tam_solution)

            if tam_solution > 0 and num_changes > 0:
                positions = numpy_rng.randint(0, tam_solution, size=num_changes)
                keys[positions] = numpy_rng.random(num_changes)

            ofv = _ObservedCost.cast(cost(decoder(keys)))
            pool[i] = (ofv, keys)
            clone_fixed = True

    if clone_fixed:
        pool.sort(key=lambda item: item[0])

    return pool


def update_pool_solutions(
    pool: list[tuple[float, np.ndarray]],
    candidate: tuple[float, np.ndarray],
) -> list[tuple[float, np.ndarray]]:
    """Updates the pool with a candidate while preserving order and cardinality.

    Args:
        pool (list[tuple[float, np.ndarray]]): Current pool sorted by cost.
        candidate (tuple[float, np.ndarray]): Candidate in the format (cost, solution).

    Returns:
        list[tuple[float, np.ndarray]]: Updated sorted pool limited to the original size.

    Raises:
        ValueError: When required by external domain validation.
    """
    if not pool:
        return [candidate]

    candidate_ofv, candidate_keys = candidate

    for ofv, _ in pool:
        if ofv == candidate_ofv:
            return pool

    new_pool = list(pool)
    new_pool.append((_ObservedCost.cast(candidate_ofv), candidate_keys.copy()))
    new_pool.sort(key=lambda item: item[0])
    return new_pool[: len(pool)]
