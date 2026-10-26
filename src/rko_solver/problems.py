"""Built-in problems evaluated by the Rust and C++ engines without the interpreter.

Objective:
- Offer problem definitions whose decoder and objective run entirely in compiled code, so
  portfolios scale across threads instead of serializing on the interpreter lock.

Technical approach:
- Resolve the extension's problem classes lazily under stable public names.

Flow (pseudocode):
1. Import this module; no extension code loads yet.
2. Access a problem class, which imports the compiled extension.
3. Build a problem and pass it to `RKO(problem, engine="rust")`.

Important criteria:
- Accessing a problem fails clearly when the compiled extension is unavailable.
- Built-in problems run on the `rust` and `cpp` engines only.

Inputs/Outputs:
- Input: problem data such as profits, weights and capacity.
- Output: problem objects accepted by `RKO`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._native import KnapsackProblem as Knapsack

__all__ = ["Knapsack"]


def __getattr__(name: str) -> object:
    """Load a built-in problem class from the compiled extension on first access.

    Args:
        name (str): Public problem name.

    Returns:
        object: The problem class.

    Raises:
        AttributeError: If the name is not a built-in problem.
        ImportError: If the compiled extension is not installed.
    """
    if name != "Knapsack":
        raise AttributeError(f"module 'rko_solver.problems' has no attribute {name!r}")
    try:
        from ._native import KnapsackProblem
    except ImportError as error:
        raise ImportError(
            "rko_solver.problems requires the compiled rko_solver._native extension; "
            "install a platform wheel or build with maturin."
        ) from error
    return KnapsackProblem
