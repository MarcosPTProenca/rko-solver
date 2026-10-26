"""Characterize empty-pool startup through the real serial and thread drivers.

Objective:
- Distinguish shared-pool capacity from an evaluated initial population.

Technical approach:
- Observe the real factory and replace only one search body with a bounded probe.
- Keep public solve, owned streams, callback sessions and result accounting intact.

Flow (pseudocode):
1. Construct a public solver with a selected execution mode and pool capacity.
2. Observe pool occupancy and callback count before construction and search entry.
3. Complete one finite observation and verify reported evaluation accounting.

Important criteria:
- A larger configured capacity must not imply driver-level seed evaluations.
- The probe isolates driver policy, not MultiStart's own initialization algorithm.
- Spawn startup is source-audited separately; no process monkeypatch is implied.

Inputs/Outputs:
- Input: positive capacities and one registered search selected through RKO.
- Output: empty startup-pool and exact callback/report assertions.
"""

import numpy as np
import pytest
from collections.abc import Mapping
from pathlib import Path
from rko_solver import RKO
from rko_solver import solver as solver_module
from rko_solver.components.solution_pool import ElitePool
from rko_solver.metaheuristics.base import MetaheuristicBase
from rko_solver.orchestration.types import CostFn
from rko_solver.orchestration.types import DecoderFn
from rko_solver.orchestration.types import RunResult


@pytest.mark.parametrize("execution", ["sequential", "thread"])
@pytest.mark.parametrize("capacity", [1, 10, 100])
def test_driver_does_not_prefill_shared_pool(
    monkeypatch: pytest.MonkeyPatch,
    execution: str,
    capacity: int,
) -> None:
    """Observe startup without conflating a search's own candidates with pool seeding.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replace the factory's returned search body.
        execution (str): Real serial or thread dispatcher.
        capacity (int): Shared-pool capacity, not a requested initial seed count.
    """
    calls: list[float] = []
    startup: list[tuple[int, int]] = []
    factory = solver_module.build_metaheuristic

    def objective(keys: np.ndarray) -> float:
        """Record each completed user objective observation.

        Args:
            keys (np.ndarray): Isolated candidate payload.

        Returns:
            float: Finite squared norm.
        """
        value = float(np.dot(keys, keys))
        calls.append(value)
        return value

    def build_probe(
        name: str,
        pool: ElitePool,
        decoder: DecoderFn,
        cost: CostFn,
        dimension: int,
        control: int,
        params_dir: Path | None,
        hyperparameters: Mapping[str, int | float] | None = None,
    ) -> MetaheuristicBase:
        """Delegate real construction and observe driver state at both boundaries.

        Args:
            name (str): Registered search name.
            pool (ElitePool): Actual shared pool.
            decoder (DecoderFn): Original user decoder.
            cost (CostFn): Original user objective.
            dimension (int): Chromosome length.
            control (int): Offline control used by this probe.
            params_dir (Path | None): Normal parameter resources.
            hyperparameters (Mapping | None): Offline overrides.

        Returns:
            MetaheuristicBase: Real configured instance with a one-observation body.
        """
        startup.append((pool.size, len(calls)))
        algorithm = factory(
            name,
            pool,
            decoder,
            cost,
            dimension,
            control,
            params_dir,
            hyperparameters,
        )

        def observe_once(limit_time: float) -> RunResult:
            """Run one candidate through the session-installed callback boundary.

            Args:
                limit_time (float): Real driver-provided search duration.

            Returns:
                RunResult: Finite evaluated candidate without an optimization loop.
            """
            startup.append((pool.size, len(calls)))
            keys = np.full(dimension, 0.5)
            value = algorithm.cost(algorithm.decoder(keys))
            pool.insert((value, keys.tolist()), name)
            return [], keys.tolist(), value

        algorithm.solve = observe_once
        return algorithm

    monkeypatch.setattr(solver_module, "build_metaheuristic", build_probe)
    result = RKO(
        lambda keys: keys,
        objective,
        3,
        algorithms=["MultiStart"],
        control=0,
        execution=execution,
        pool_size=capacity,
        random_state=7,
    ).solve(1.0)
    assert startup == [(0, 0), (0, 0)]
    assert calls == [0.75]
    assert result.cost == 0.75 and result.pool_size == capacity
    assert len(result.searches) == 1 and result.searches[0].n_evaluations == 1
