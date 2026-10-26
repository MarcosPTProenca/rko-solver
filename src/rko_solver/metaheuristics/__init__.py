"""Public single-algorithm solvers and advanced metaheuristic contracts.

Objective:
- Expose consistent configure-then-solve objects for every built-in algorithm.

Technical approach:
- Resolve public solver classes lazily to avoid the solver/registry import cycle.
- Keep low-level algorithm kernels in their implementation modules for orchestration.
- Expose operator protocols and the kernel base for explicitly advanced integration.

Flow (pseudocode):
1. Import shared kernel/operator contracts without loading public solver subclasses.
2. On a named public class import, load and cache its RKO-based solver class.
3. Construct the solver and use its inherited solve method for managed execution.

Important criteria:
- Public algorithm solvers share RKO constructor settings and RKOResult output.
- Kernel constructors used by the registry remain unchanged.

Inputs/Outputs:
- Input: public solver names or advanced contract imports.
- Output: consistent solver classes and typed operator/base contracts.
"""

from __future__ import annotations

from .base import LocalSearchOperator
from .base import MetaheuristicBase
from .base import ShakingOperator
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..solver import RKO
    from ._solvers import BRKGA
    from ._solvers import BRKGA_CS
    from ._solvers import GRASP
    from ._solvers import ILS
    from ._solvers import IPR
    from ._solvers import LNS
    from ._solvers import PSO
    from ._solvers import SA
    from ._solvers import VNS
    from ._solvers import GeneticAlgorithm
    from ._solvers import MultiStart

__all__ = [
    "LocalSearchOperator",
    "MetaheuristicBase",
    "ShakingOperator",
    "BRKGA",
    "BRKGA_CS",
    "GeneticAlgorithm",
    "GRASP",
    "ILS",
    "IPR",
    "LNS",
    "MultiStart",
    "PSO",
    "SA",
    "VNS",
]


def __getattr__(name: str) -> type[RKO]:
    """Load a public algorithm solver after the portfolio solver is importable.

    Args:
        name (str): Public single-algorithm solver class name.

    Returns:
        type[RKO]: Cached RKO subclass with one fixed algorithm.

    Raises:
        AttributeError: If the requested attribute is not a public solver class.
    """
    if name not in __all__[3:]:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    solver = getattr(import_module("._solvers", __name__), name)
    globals()[name] = solver
    return solver


def __dir__() -> list[str]:
    """Keep lazy public solver names visible to interactive discovery and inspection.

    Returns:
        list[str]: Sorted module attributes and declared public exports.
    """
    return sorted(set(globals()) | set(__all__))
