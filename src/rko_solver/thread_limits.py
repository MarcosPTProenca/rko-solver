"""Native thread limit for process workers spawned inside a solve.

Objective:
- Let a caller give each spawned process worker one BLAS/OpenMP thread without changing
  the parent's native libraries.

Technical approach:
- A spawned child copies the parent's environment when it starts. The parent's BLAS
  libraries read theirs once at import, so setting the variables in the parent later has
  no effect on the parent and does reach children started afterwards.
- The context manager exports the four thread variables, yields, and restores the saved
  values on exit (removing those that were unset).

Flow (pseudocode):
1. Validate the thread count: a positive int, not a bool.
2. Save the current values of the four variables.
3. Export the count to all four.
4. Yield; the caller solves while children are started.
5. Restore every saved value, popping the ones that were unset.

Important criteria:
- Children started before entry or after exit are not affected.
- The restore runs even when the block raises.

Inputs/Outputs:
- Input: positive integer thread count.
- Output: a context manager yielding None.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


@contextlib.contextmanager
def worker_thread_limit(threads: int) -> Iterator[None]:
    """Give process workers started inside the block a fixed native thread count.

    Args:
        threads (int): Positive number of BLAS/OpenMP threads for each new worker.

    Yields:
        None: Control returns to the caller while the limit is exported.

    Raises:
        ValueError: If `threads` is not a positive integer.
    """
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer.")
    saved = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    os.environ.update({name: str(threads) for name in THREAD_VARIABLES})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
