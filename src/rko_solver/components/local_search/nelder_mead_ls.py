"""Nelder-Mead local search for random-key vectors.

Objective:
- Refine continuous random-key solutions using simplex operations in key space.

Technical approach:
- Maintain a three-point simplex with objective evaluation over decoded solutions.
- Apply reflection, expansion, contraction, and shrink steps until stabilization.
- When an elite pool is available, use its top-2 solutions as the other two
  simplex vertices. The pinned original C++ instead samples distinct pool members
  and reuses their scores; the matched benchmark normalizes those policies explicitly.

Flow (pseudocode):
1. Initialize the simplex: x1 from the provided keys; x2 and x3 from the pool
   elite (if available) or random otherwise.
2. Iterate Nelder-Mead operators according to cost ordering.
3. Return the best point found or the original solution.

Important criteria:
- Reset the stagnation counter when improvement occurs.
- Use a minimum iteration limit based on problem dimension.
- Pool is optional; behaviour degrades gracefully to random simplex when absent.
- Reference: docs/references.md, section nelder-mead-simplex

Inputs/Outputs:
- Input: key vector, decoder, objective function, and optional elite pool.
- Output: best random-key vector and associated cost.
"""

import math
import numpy as np
from ..blending.blending import blending as _blend
from ..random_streams import RandomStreams
from ..solution_pool import ElitePool
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class _Point:
    """Represents a simplex point with key vector and cost.

    Attributes:
        keys (np.ndarray): Point key vector.
        ofv (float): Objective function value at the point.
    """

    keys: np.ndarray
    ofv: float


def _evaluate(
    keys: np.ndarray, decoder: Callable[[np.ndarray], Any], cost: Callable[[Any], float]
) -> float:
    """Evaluates the cost of a decoded random-key vector.

    Args:
        keys (np.ndarray): Key vector to evaluate.
        decoder (Callable[[np.ndarray], Any]): Solution decoding function.
        cost (Callable[[Any], float]): Objective function over the decoded solution.

    Returns:
        float: Computed objective function value.

    Raises:
        ValueError: When required by external domain validation.
    """
    return cost(decoder(keys))


def nelder_mead_search(
    keys: np.ndarray,
    decoder: Callable[[np.ndarray], Any],
    cost: Callable[[Any], float],
    pool: ElitePool | None = None,
) -> tuple[np.ndarray, float]:
    """Runs Nelder-Mead local search on a random-key vector.

    When *pool* is provided and contains at least two elite solutions, its
    top-2 entries are used as the second and third simplex vertices. This is
    the Python policy, not the pinned original C++ random-pool policy.
    Otherwise two random vectors are generated instead.

    Args:
        keys (np.ndarray): Key vector of the initial solution.
        decoder (Callable[[np.ndarray], Any]): Solution decoding function.
        cost (Callable[[Any], float]): Objective function over the decoded solution.
        pool (object | None): Optional elite solution pool with a
          ``get_elite(k)`` method returning ``(cost, keys_list)`` pairs.

    Returns:
        tuple[np.ndarray, float]: Best found vector and corresponding cost.

    Raises:
        ValueError: When required by external domain validation.
    """
    numpy_rng = RandomStreams.numpy_rng()
    n = len(keys)
    if n == 0:
        return keys, _evaluate(keys, decoder, cost)

    x1_origin = keys.copy()
    x1 = _Point(keys.copy(), _evaluate(keys, decoder, cost))
    origin_cost = x1.ofv

    elite = pool.get_elite(2) if pool is not None and hasattr(pool, "get_elite") else []
    if len(elite) >= 2:
        x2_keys = np.array(elite[0][1], dtype=float)
        x3_keys = np.array(elite[1][1], dtype=float)
        if len(x2_keys) != n:
            x2_keys = numpy_rng.random(n)
        if len(x3_keys) != n:
            x3_keys = numpy_rng.random(n)
    elif len(elite) == 1:
        x2_keys = np.array(elite[0][1], dtype=float)
        if len(x2_keys) != n:
            x2_keys = numpy_rng.random(n)
        x3_keys = numpy_rng.random(n)
    else:
        x2_keys = numpy_rng.random(n)
        x3_keys = numpy_rng.random(n)

    x2 = _Point(x2_keys, _evaluate(x2_keys, decoder, cost))
    x3 = _Point(x3_keys, _evaluate(x3_keys, decoder, cost))

    simplex = sorted([x1, x2, x3], key=lambda p: p.ofv)
    x1, x2, x3 = simplex[0], simplex[1], simplex[2]

    x0_keys = _blend(x1.keys, x2.keys, 1)
    x0 = _Point(x0_keys, _evaluate(x0_keys, decoder, cost))

    x_best = min([x1, x2, x3, x0], key=lambda p: p.ofv)
    improved = x0.ofv < x1.ofv
    improved_x1 = 1 if x_best.ofv < origin_cost else 0

    iter_count = 1
    max_iter = int(n * math.exp(-2.0))
    if max_iter < 1:
        max_iter = 1

    while iter_count <= max_iter:
        shrink = False

        xr_keys = _blend(x0.keys, x3.keys, -1)
        xr = _Point(xr_keys, _evaluate(xr_keys, decoder, cost))
        if xr.ofv < x_best.ofv:
            x_best = xr
            improved = True
            improved_x1 = 1

        if xr.ofv < x1.ofv:
            xe_keys = _blend(xr.keys, x0.keys, -1)
            xe = _Point(xe_keys, _evaluate(xe_keys, decoder, cost))
            if xe.ofv < x_best.ofv:
                x_best = xe
                improved = True
                improved_x1 = 1

            if xe.ofv < xr.ofv:
                x3 = xe
            else:
                x3 = xr

        else:
            if xr.ofv < x2.ofv:
                x3 = xr
            else:
                if xr.ofv < x3.ofv:
                    xc_keys = _blend(xr.keys, x0.keys, 1)
                    xc = _Point(xc_keys, _evaluate(xc_keys, decoder, cost))
                    if xc.ofv < x_best.ofv:
                        x_best = xc
                        improved = True
                        improved_x1 = 1

                    if xc.ofv < xr.ofv:
                        x3 = xc
                    else:
                        shrink = True
                else:
                    xc_keys = _blend(x0.keys, x3.keys, 1)
                    xc = _Point(xc_keys, _evaluate(xc_keys, decoder, cost))
                    if xc.ofv < x_best.ofv:
                        x_best = xc
                        improved = True
                        improved_x1 = 1

                    if xc.ofv < x3.ofv:
                        x3 = xc
                    else:
                        shrink = True

        if shrink:
            x2_keys = _blend(x1.keys, x2.keys, 1)
            x2 = _Point(x2_keys, _evaluate(x2_keys, decoder, cost))
            if x2.ofv < x_best.ofv:
                x_best = x2
                improved = True
                improved_x1 = 1

            x3_keys = _blend(x1.keys, x3.keys, 1)
            x3 = _Point(x3_keys, _evaluate(x3_keys, decoder, cost))
            if x3.ofv < x_best.ofv:
                x_best = x3
                improved = True
                improved_x1 = 1

        simplex = sorted([x1, x2, x3], key=lambda p: p.ofv)
        x1, x2, x3 = simplex[0], simplex[1], simplex[2]

        x0_keys = _blend(x1.keys, x2.keys, 1)
        x0 = _Point(x0_keys, _evaluate(x0_keys, decoder, cost))
        if x0.ofv < x_best.ofv:
            x_best = x0
            improved = True
            improved_x1 = 1

        if improved:
            improved = False
            iter_count = 0
        else:
            iter_count += 1

    if improved_x1 == 1:
        return x_best.keys.copy(), x_best.ofv
    return x1_origin, origin_cost
