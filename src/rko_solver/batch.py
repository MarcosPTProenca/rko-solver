"""Deprecated location of the block-objective adapter; use `rko_solver.Blocks`.

Objective:
- Keep `rko_solver.batch` imports working until 0.4.

Technical approach:
- Re-export the names that moved to `rko_solver.engines.rust.batch` and warn on import.

Flow (pseudocode):
1. Warn that the module moved.
2. Re-export the moved names.

Important criteria:
- Removed in 0.4; nothing inside the package imports this module.

Inputs/Outputs:
- Input: none.
- Output: `BatchFn`, `BatchObjective`, `MAX_BATCH_SIZE` and `identity_decoder`.
"""

from __future__ import annotations

import warnings
from .engines.rust.batch import MAX_BATCH_SIZE
from .engines.rust.batch import BatchFn
from .engines.rust.batch import BatchObjective
from .engines.rust.batch import identity_decoder

warnings.warn(
    "'rko_solver.batch' is deprecated and will be removed in 0.4; use 'rko_solver.Blocks'.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["MAX_BATCH_SIZE", "BatchFn", "BatchObjective", "identity_decoder"]
