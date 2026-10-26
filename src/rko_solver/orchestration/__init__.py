"""Validated search construction and engine execution exports.

Objective:
- Expose shared orchestration contracts without duplicating engine logic.

Technical approach:
- Re-export configuration parsing, construction, execution and callback types.

Flow (pseudocode):
1. Parse and validate the run configuration.
2. Construct and execute selected searches.
3. Select the best completed result.

Important criteria:
- Search engines share the same callback and result contracts.

Inputs/Outputs:
- Input: configuration, decoder and objective callbacks.
- Output: configured algorithms and measured run results.
"""

from .configuration import parse_run_config
from .execution import run_metaheuristics_parallel
from .execution import select_best_result
from .factory import build_metaheuristic
from .types import CostFn
from .types import DecoderFn
from .types import RunConfig
from .types import RunResult

__all__ = [
    "build_metaheuristic",
    "parse_run_config",
    "run_metaheuristics_parallel",
    "select_best_result",
    "CostFn",
    "DecoderFn",
    "RunConfig",
    "RunResult",
]
