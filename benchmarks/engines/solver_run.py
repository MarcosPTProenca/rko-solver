"""Run the public solver once and verify its answer independently of the solver.

Objective:
- Share one solver call between benchmark studies so each study reports only profits or lengths
  that were re-decoded outside the solver.

Technical approach:
- Build `RKO` from a decoder, a cost and optional `initial_keys`, solve for a time slice, decode
  the returned keys again and compare the independently computed cost with the reported one.

Flow (pseudocode):
1. Construct the solver with the given seed, execution mode and portfolio.
2. Solve for the requested seconds.
3. Re-decode the returned keys and evaluate the cost.
4. Return that cost and whether it equals the solver's reported cost.

Important criteria:
- Verification uses a relative tolerance of 1e-10 and an absolute tolerance of 1e-8.

Inputs/Outputs:
- Input: decoder, cost, key count, optional seed keys, timing and engine settings.
- Output: the independently computed cost and a verification flag.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Callable
from rko_solver import RKO
from typing import Any


class SolverRun:
    """Solve once through the public interface and re-verify the returned keys.

    Responsibilities:
    - Hold the single solver call shared by the benchmark studies.
    """

    @staticmethod
    def solve(
        decoder: Callable[[np.ndarray], Any],
        cost: Callable[[Any], float],
        n_variables: int,
        engine: tuple[str, tuple[str, ...] | None],
        timing: tuple[int, float],
        keys: list[list[float]] | None,
    ) -> tuple[float, bool]:
        """Solve once and re-decode the returned keys outside the solver.

        Args:
            decoder (Callable[[np.ndarray], Any]): Decoder handed to the solver.
            cost (Callable[[Any], float]): Objective on a decoded solution, minimized.
            n_variables (int): Keys per vector.
            engine (tuple[str, tuple[str, ...] | None]): Python-engine execution mode and an
                optional portfolio override.
            timing (tuple[int, float]): Solver seed and search seconds.
            keys (list[list[float]] | None): Optional `initial_keys` rows.

        Returns:
            tuple[float, bool]: Independently decoded cost and whether it matches the solver's
            reported cost.
        """
        result = RKO(
            decoder,
            cost,
            n_variables,
            random_state=timing[0],
            initial_keys=keys,
            execution=engine[0],
            algorithms=engine[1],
        ).solve(timing[1])
        observed = float(cost(decoder(np.asarray(result.keys))))
        return observed, bool(np.isclose(observed, result.cost, rtol=1e-10, atol=1e-8))
