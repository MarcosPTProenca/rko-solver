"""Public package interface for the RKO optimization library.

Objective:
- Expose stable package-level imports for the RKO orchestration entrypoints.

Technical approach:
- Mark `src/rko_solver` as a regular Python package for tooling compatibility.
- Export RKO and RKOResult as the primary class-based API.
- Export the problem definitions `Callbacks`, `Numba` and `Blocks`; built-in problems live
  in the lazily loaded `problems` module.
- Expose optional post-solve assessment and bounded domain-check records.
- Preserve run as a compatibility adapter.

Flow (pseudocode):
1. Import the solver, result and compatibility adapter.
2. Expose stable symbols through __all__.

Important criteria:
- Keep package API minimal and explicit.

Inputs/Outputs:
- Input: internal module symbols.
- Output: package-level exports.
"""

from . import problems
from .assessment import SolutionAssessment
from .assessment import ValidationCheck
from .assessment import ValidationReport
from .assessment import assess_result
from .orchestration.types import PoolInitializationMetadata
from .orchestration.types import SearchMetadata
from .problem import Blocks
from .problem import Callbacks
from .problem import Numba
from .rko_main import run
from .solver import RKO
from .solver import RKOResult

__all__ = [
    "RKO",
    "Callbacks",
    "Numba",
    "Blocks",
    "problems",
    "RKOResult",
    "SearchMetadata",
    "PoolInitializationMetadata",
    "run",
    "ValidationCheck",
    "ValidationReport",
    "SolutionAssessment",
    "assess_result",
]
