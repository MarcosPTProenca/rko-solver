"""Compatibility adapter for the class-based RKO solver.

Objective:
- Preserve the original run signature and tuple result for existing callers.

Technical approach:
- Delegate all validation, configuration and execution to RKO.

Flow (pseudocode):
1. Construct an RKO solver using the legacy arguments.
2. Execute solve with the requested search budget.
3. Return detached keys and objective in the original tuple format.

Important criteria:
- RKO is the primary API; no orchestration implementation is duplicated here.

Inputs/Outputs:
- Input: legacy problem definition and execution settings.
- Output: best key list and minimization objective.
"""

from __future__ import annotations

from .orchestration.types import CostFn
from .orchestration.types import DecoderFn
from .solver import RKO
from .solver import Engine
from .solver import Execution
from .solver import LegacyBackend
from pathlib import Path


def run(
    decoder: DecoderFn,
    cost: CostFn,
    tam_solution: int,
    limit_time: float,
    *,
    params_dir: Path | None = None,
    engine: Engine | None = None,
    execution: Execution | None = None,
    backend: LegacyBackend | None = None,
    n_jobs: int | None = None,
    random_state: int | None = None,
    verbose: bool = False,
    prefill_pool: bool | None = None,
) -> tuple[list[float], float]:
    """Delegate a legacy call to RKO and retain its tuple result contract.

    Args:
        decoder (DecoderFn): Problem-specific random-key decoder.
        cost (CostFn): Objective to minimize after decoding.
        tam_solution (int): Positive key dimension, named n_variables in RKO.
        limit_time (float): Positive search seconds, named time_limit in solve.
        params_dir (Path | None): Optional parameter TOML directory.
        engine (Engine | None): Python (default), rust, cpp or auto, exactly as for RKO.
        execution (Execution | None): Python-engine process, thread or sequential scheduling.
        backend (LegacyBackend | None): Deprecated combined name, removed in 0.4.
        n_jobs (int | None): Optional process or Rust worker budget.
        random_state (int | None): Nonnegative per-search seed in every engine.
        verbose (bool): Flush lifecycle and pool updates; quiet by default.
        prefill_pool (bool | None): Opt-in reference startup override; None uses TOML.
            Startup has a separate allowance and does not reduce search durations.

    Returns:
        tuple[list[float], float]: Best keys and objective value.

    Raises:
        ValueError: If problem inputs or solver settings are invalid.
        RuntimeError: If an algorithm fails or a worker exits unexpectedly.
        TimeoutError: If process startup exceeds its watchdog, or a process search overruns its
            grace before any solution was found.
        OSError: If parameter files cannot be read.
    """
    result = RKO(
        decoder,
        cost,
        tam_solution,
        params_dir=params_dir,
        engine=engine,
        execution=execution,
        backend=backend,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        prefill_pool=prefill_pool,
    ).solve(limit_time)
    return list(result.keys), result.cost
