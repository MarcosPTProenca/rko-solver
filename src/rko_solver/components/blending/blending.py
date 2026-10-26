"""Blending recombination operator for random keys.

Objective:
- Generate a child vector from two parent vectors in population-based metaheuristics.

Technical approach:
- Apply uniform gene-by-gene recombination with a small mutation rate.
- Support complementary mode for inversion of the second parent.

Flow (pseudocode):
1. Validate dimensional compatibility of both parents.
2. For each gene, apply mutation or inheritance by probability.
3. Return the generated child vector.

Important criteria:
- Point mutation rate of 2%.
- Preserve valid gene range in [0, 1).

Inputs/Outputs:
- Input: parent_a, parent_b, and blending factor.
- Output: recombined child vector.
"""

import numpy as np
import random as random
from ..random_streams import RandomStreams


def blending(parent_a: np.ndarray, parent_b: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Runs uniform recombination between two parent vectors.

    Args:
      parent_a (np.ndarray): First parent vector.
      parent_b (np.ndarray): Second parent vector.
      factor (float): Complementary mode factor when equal to -1.

    Returns:
      np.ndarray: Recombined child vector.

    Raises:
      ValueError: If parent vectors do not have the same shape.
    """
    if parent_a.shape != parent_b.shape:
        raise ValueError("parent_a and parent_b must have the same shape")
    if parent_a.ndim != 1:
        raise ValueError("parent vectors must be one-dimensional")

    n = len(parent_a)
    mutation = np.zeros(n, dtype=bool)
    use_parent_a = np.zeros(n, dtype=bool)
    mutation_values = np.empty(n, dtype=float)
    draw = RandomStreams.python_rng().random

    for j in range(n):
        if draw() < 0.02:
            mutation[j] = True
            mutation_values[j] = draw()
        else:
            use_parent_a[j] = draw() < 0.5

    child = np.where(use_parent_a, parent_a, parent_b).astype(float, copy=False)
    if factor == -1:
        complemented = np.clip(1.0 - child, 0.0, 0.9999999)
        child[~use_parent_a] = complemented[~use_parent_a]
    child[mutation] = mutation_values[mutation]
    return child
