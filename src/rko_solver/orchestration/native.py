"""Deprecated location of the compiled-engine runner.

Objective:
- Keep `rko_solver.orchestration.native` imports working until 0.4.

Technical approach:
- Re-export `NativeRunner`, which moved to `rko_solver.engines.rust.runner`, and warn on import.

Flow (pseudocode):
1. Warn that the module moved.
2. Re-export the moved class.

Important criteria:
- Removed in 0.4; nothing inside the package imports this module.

Inputs/Outputs:
- Input: none.
- Output: `NativeRunner`.
"""

from __future__ import annotations

import warnings
from ..engines.rust.runner import NativeRunner

warnings.warn(
    "'rko_solver.orchestration.native' is deprecated and will be removed in 0.4; "
    "use 'rko_solver.engines.rust.runner'.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["NativeRunner"]
