"""Deprecated location of the built-in problems; use `rko_solver.problems`.

Objective:
- Keep `from rko_solver.native import Knapsack` working until 0.4.

Technical approach:
- Re-export `rko_solver.problems` names and warn on import.

Flow (pseudocode):
1. Warn that the module moved.
2. Resolve names from `rko_solver.problems` on access.

Important criteria:
- Removed in 0.4.

Inputs/Outputs:
- Input: an attribute name.
- Output: the same object `rko_solver.problems` returns.
"""

from __future__ import annotations

import warnings
from . import problems
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._native import KnapsackProblem as Knapsack

warnings.warn(
    "'rko_solver.native' is deprecated and will be removed in 0.4; use 'rko_solver.problems'.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Knapsack"]


def __getattr__(name: str) -> object:
    """Delegate to `rko_solver.problems`.

    Args:
        name (str): Public problem name.

    Returns:
        object: The problem class.

    Raises:
        AttributeError: If the name is not a built-in problem.
        ImportError: If the compiled extension is not installed.
    """
    return getattr(problems, name)
