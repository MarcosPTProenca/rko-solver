"""Automatic-engine solve: probe the candidates, then run the winner with the full budget.

Objective:
- Run `engine="auto"` end to end for one solver: select an engine by short probes and finish the
  solve on the winner, keeping the best solution seen by any probe.

Technical approach:
- Probe copies never receive the caller's initial keys, history, trace or progress output, so
  selection stays cheap and the seed evaluations are charged once, to the final run only.
- The final run is a fresh copy of the solver on the chosen candidate; its result is merged
  with the best probe solution and the selection record.

Flow (pseudocode):
1. Reject `n_jobs` and ask `AutoEngine.select` for the winner and the best probe solution.
2. Solve with the chosen candidate over the full time limit.
3. Keep the better of the final and probe solutions and attach the selection record.

Important criteria:
- The selection budget is separate from the search `time_limit` and is reported.
- Initial keys are evaluated only by the final run.

Inputs/Outputs:
- Input: a configured solver, the search time limit and the solve start time.
- Output: an `RKOResult` carrying the `AutoSelection` record.
"""

from __future__ import annotations

import os
import time
from ..result import RKOResult
from .auto import AutoCandidate
from .auto import AutoEngine
from collections.abc import Sequence
from dataclasses import replace
from typing import Any
from typing import Protocol


class AutoSolver(Protocol):
    """Describe the solver surface the automatic solve needs.

    Attributes:
        random_state (int | None): Seed the probe and final seeds derive from.
        n_jobs (int | None): Worker limit, which `auto` rejects.
        verbose (bool): Whether to print the decision.
        auto_time (float | None): Selection budget in seconds.
        auto_candidates (Sequence[str] | None): Candidate subset.
        result_ (RKOResult | None): Last successful solve.
    """

    random_state: int | None
    n_jobs: int | None
    verbose: bool
    auto_time: float | None
    auto_candidates: Sequence[str] | None
    result_: RKOResult | None

    def _kind(self) -> str:
        """Return `callbacks`, `batch` or `native`."""
        ...

    def _variant(
        self, candidate: AutoCandidate, seed: int | None, probe: bool = True
    ) -> AutoSolver:
        """Return a copy running the candidate engine."""
        ...

    def solve(self, time_limit: float) -> RKOResult:
        """Run the copy and return its result."""
        ...


class AutoRun:
    """Choose an engine by short probes, then run the full budget on the winner.

    Responsibilities:
    - Drive selection and the final run for one solver.
    - Merge the best probe solution and the selection record into the final result.
    """

    @staticmethod
    def solve(solver: AutoSolver, time_limit: float, started: float) -> RKOResult:
        """Run the automatic selection and the final solve.

        Args:
            solver (AutoSolver): Configured solver whose engine is `auto`.
            time_limit (float): Search seconds of the final run.
            started (float): Monotonic origin of this solve.

        Returns:
            RKOResult: Best solution over probes and final run, with the selection record.

        Raises:
            ValueError: If the candidate list is invalid or `n_jobs` was given.
            RuntimeError: If every candidate fails.
        """
        if solver.n_jobs is not None:
            raise ValueError("'n_jobs' cannot be combined with engine='auto'.")
        from .. import _native

        selection, probe_keys, probe_cost = AutoEngine.select(
            solver,
            time_limit,
            solver.auto_time,
            solver.auto_candidates,
            _native.cpp_available(),
            time.perf_counter,
        )
        cpus = max(1, os.process_cpu_count() or 1)
        candidate = AutoEngine.build(selection.chosen, cpus)
        if solver.verbose:
            print(
                f"RKO AUTO chosen={selection.chosen} reason='{selection.reason}' "
                f"scores={dict(selection.scores)}",
                flush=True,
            )
        final = solver._variant(candidate, solver.random_state, probe=False).solve(time_limit)
        better = probe_keys is not None and probe_cost < final.cost
        selection = replace(selection, improved_by_probe=better)
        found: dict[str, Any] = (
            {"keys": tuple(probe_keys or ()), "cost": float(probe_cost)} if better else {}
        )
        result = replace(
            final,
            **found,
            elapsed_seconds=time.perf_counter() - started,
            random_state=solver.random_state,
            auto=selection,
        )
        solver.result_ = result
        return result
