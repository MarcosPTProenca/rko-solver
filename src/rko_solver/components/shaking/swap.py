"""Random-position swap shaking operator.

Objective:
- Promote fast structural variation by permuting genes in the solution.

Technical approach:
- Sample two independent positions and swap their values.

Flow (pseudocode):
1. Validate minimum vector dimension.
2. Sample two positions.
3. Swap position contents and return.

Important criteria:
- In-place operation with constant cost per call.

Inputs/Outputs:
- Input: random-key vector.
- Output: random-key vector with two swapped positions.
"""

import numpy as np
import random as random
from ..random_streams import RandomStreams


def swap_shake(keys: np.ndarray) -> np.ndarray:
    """Swaps two random positions in the solution.

    Args:
      keys (np.ndarray): Solution key vector.

    Returns:
      np.ndarray: Vector with two swapped positions.

    Raises:
      ValueError: When required by external domain validation.
    """
    random = RandomStreams.python_rng()
    n = len(keys)
    if n < 2:
        return keys

    i = random.randint(0, n - 1)
    j = random.randint(0, n - 1)
    keys[i], keys[j] = keys[j], keys[i]
    return keys
