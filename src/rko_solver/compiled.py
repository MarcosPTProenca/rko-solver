"""Deprecated location of the Numba objective preparation; use `rko_solver.Numba`.

Objective:
- Keep `rko_solver.compiled` imports working until 0.4.

Technical approach:
- Re-export the names that moved to `rko_solver.engines.rust.compiled` and warn on import.

Flow (pseudocode):
1. Warn that the module moved.
2. Re-export the moved names.

Important criteria:
- Removed in 0.4; nothing inside the package imports this module.

Inputs/Outputs:
- Input: none.
- Output: `CompiledObjective` and `compile_objective`.
"""

from __future__ import annotations

import warnings
from .engines.rust.compiled import CompiledObjective
from .engines.rust.compiled import compile_objective

warnings.warn(
    "'rko_solver.compiled' is deprecated and will be removed in 0.4; use 'rko_solver.Numba'.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["CompiledObjective", "compile_objective"]
