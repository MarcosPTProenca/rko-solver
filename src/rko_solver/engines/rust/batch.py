"""Batch-scored objectives for solvers that evaluate many candidates per call.

Objective:
- Let users write one vectorized objective over a block of random-key vectors and run
  it on any engine, with real block scoring on the Rust and C++ engines.

Technical approach:
- `BatchObjective` validates the block contract (a `(rows, n)` float array in, `rows`
  finite costs out) and doubles as a picklable scalar decoder/objective adapter for the
  Python engine, which scores one row per call.
- The compiled engines group independent candidates and speculates on sequential local
  searches only when the objective declares a block size above one.

Flow (pseudocode):
1. Wrap the user's vectorized function in `BatchObjective`.
2. Rust and C++ engines: pass the function and block size to the compiled engine.
3. Python engine: decode keys unchanged and score each row as a block of one.

Important criteria:
- The array handed to the function is a fresh copy; mutating it never affects the search.
- The function must return exactly one finite cost per row, in row order.
- Block size bounds both speculative waste and the time one uninterruptible call can take.

Inputs/Outputs:
- Input: a callable mapping a `(rows, n)` array to `rows` costs, and a block size.
- Output: scalar decoder/objective adapters and validated costs.
"""

from __future__ import annotations

import math
import numpy as np
from collections.abc import Callable
from collections.abc import Sequence

BatchFn = Callable[[np.ndarray], Sequence[float] | np.ndarray]
MAX_BATCH_SIZE = 4096


class BatchObjective:
    """Validate and adapt a vectorized objective.

    Responsibilities:
    - Hold the user's function and its preferred block size.
    - Provide scalar decoder/objective callables for engines without block scoring.

    Attributes:
        function (BatchFn): User function scoring a `(rows, n)` array.
        batch_size (int): Largest block the compiled engines pass in one call.
    """

    def __init__(self, function: BatchFn, batch_size: int = 64) -> None:
        """Store the function and validate the block size.

        Args:
            function (BatchFn): Callable mapping a `(rows, n)` array to `rows` costs.
            batch_size (int): Largest block per call, from 1 to 4096.

        Raises:
            ValueError: If the function is not callable or the block size is invalid.
        """
        if not callable(function):
            raise ValueError("'batch_objective' must be callable.")
        if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"'batch_size' must be an integer from 1 to {MAX_BATCH_SIZE}.")
        self.function = function
        self.batch_size = batch_size

    def scores(self, keys: np.ndarray) -> np.ndarray:
        """Score a `(rows, n)` block and validate the reply.

        Args:
            keys (np.ndarray): Candidate matrix, one random-key vector per row.

        Returns:
            np.ndarray: One float cost per row.

        Raises:
            ValueError: If the reply has the wrong length or contains a nonfinite cost.
        """
        costs = np.asarray(self.function(keys), dtype=float)
        if costs.shape != (len(keys),):
            raise ValueError(
                f"batch objective returned shape {costs.shape} for {len(keys)} rows; "
                "expected one cost per row."
            )
        if not np.all(np.isfinite(costs)):
            raise ValueError("Objective callbacks must return finite costs.")
        return costs

    def __call__(self, keys: np.ndarray) -> float:
        """Score one vector as a block of one, for engines that score scalars.

        Args:
            keys (np.ndarray): One random-key vector.

        Returns:
            float: Its finite cost.

        Raises:
            ValueError: If the underlying function violates the block contract.
        """
        value = float(self.scores(np.asarray(keys, dtype=float).reshape(1, -1))[0])
        if not math.isfinite(value):
            raise ValueError("Objective callbacks must return finite costs.")
        return value


def identity_decoder(keys: np.ndarray) -> np.ndarray:
    """Return keys unchanged, so the batch objective decodes internally.

    Args:
        keys (np.ndarray): Candidate random-key vector.

    Returns:
        np.ndarray: The same vector.
    """
    return keys
