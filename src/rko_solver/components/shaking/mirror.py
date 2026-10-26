"""Gene-mirroring shaking operator.

Objective:
- Locally modify one solution position using a mirror transformation.

Technical approach:
- Select a random index and apply a complementary value transformation.

Flow (pseudocode):
1. Sample a valid index.
2. Apply mirroring to the selected position value.
3. Return the updated vector.

Important criteria:
- Avoid degenerate values very close to zero.

Inputs/Outputs:
- Input: random-key vector.
- Output: updated random-key vector.
"""

import numpy as np
import random as random
from ..random_streams import RandomStreams


def mirror_shake(keys: np.ndarray) -> np.ndarray:
    """Applies mirroring to one sampled gene.

    Args:
      keys (np.ndarray): Solution key vector.

    Returns:
      np.ndarray: Vector after mirroring at a random position.

    Raises:
      ValueError: When required by external domain validation.
    """
    random = RandomStreams.python_rng()
    n = len(keys)
    if n == 0:
        return keys

    idx = random.randint(0, n - 1)
    if keys[idx] > 0.0001:
        keys[idx] = 1.0 - keys[idx]
    else:
        keys[idx] = 0.9999
    return keys
