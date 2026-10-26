"""Adjacent-neighbor swap shaking operator.

Objective:
- Produce a mild local perturbation by swapping adjacent elements.

Technical approach:
- Select a random starting index and swap with the immediate neighbor.

Flow (pseudocode):
1. Validate that at least two genes exist.
2. Sample an index between 0 and n-2.
3. Swap positions i and i+1.

Important criteria:
- Low-amplitude perturbation per call.

Inputs/Outputs:
- Input: random-key vector.
- Output: random-key vector with adjacent swap.
"""

import numpy as np
import random as random
from ..random_streams import RandomStreams


def neighbor_swap_shake(keys: np.ndarray) -> np.ndarray:
    """Swaps one gene with its immediate neighbor.

    Args:
      keys (np.ndarray): Solution key vector.

    Returns:
      np.ndarray: Vector with adjacent swap applied.

    Raises:
      ValueError: When required by external domain validation.
    """
    random = RandomStreams.python_rng()
    n = len(keys)
    if n < 2:
        return keys

    i = random.randint(0, n - 2)
    keys[i], keys[i + 1] = keys[i + 1], keys[i]
    return keys
