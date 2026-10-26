"""Stochastic shaking for random-key vectors.

Objective:
- Perturb solutions to expand exploration in the search space.

Technical approach:
- Sample perturbation intensity from a parameterized range.
- Apply random operators for direct alteration, mirroring, and swaps.

Flow (pseudocode):
1. Compute intensity from beta_min and beta_max.
2. Repeatedly apply a randomly selected shaking operator.
3. Return the perturbed vector.

Important criteria:
- Minimum intensity of one perturbation per call.
- Preserve vector dimension.

Inputs/Outputs:
- Input: key vector and intensity range.
- Output: perturbed key vector.
"""

import numpy as np
import random as random
from ..random_streams import RandomStreams
from .mirror import mirror_shake
from .neighbor_swap import neighbor_swap_shake
from .swap import swap_shake


def shake(keys: np.ndarray, beta_min: float, beta_max: float) -> np.ndarray:
    """Applies stochastic perturbations to a random-key solution.

    Args:
      keys (np.ndarray): Key vector to perturb.
      beta_min (float): Lower bound of relative intensity.
      beta_max (float): Upper bound of relative intensity.

    Returns:
      np.ndarray: Perturbed vector after applying operators.

    Raises:
      ValueError: When required by external domain validation.
    """
    random = RandomStreams.python_rng()
    n = len(keys)
    if n == 0:
        return keys

    intensity = int(n * random.uniform(beta_min, beta_max)) + 1
    if intensity < 1:
        intensity = 1

    for _ in range(intensity):
        shaking_type = random.randint(1, 4)

        if shaking_type == 1:
            idx = random.randint(0, n - 1)
            keys[idx] = random.random()
        elif shaking_type == 2:
            keys = mirror_shake(keys)
        elif shaking_type == 3:
            keys = swap_shake(keys)
        else:
            keys = neighbor_swap_shake(keys)

    return keys
