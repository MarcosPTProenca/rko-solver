"""Tests for metaheuristic base behavior and factory construction.

Objective:
- Validate common base contracts and factory creation for all registered specs.

Technical approach:
- Use a lightweight `DummyMH` implementation for base timing checks.
- Parametrize factory tests using the live orchestration registry.

Flow (pseudocode):
1. Instantiate dummy metaheuristic and verify timer/repr behavior.
2. Build all registered metaheuristics in offline mode.
3. Build control-enabled metaheuristics in online mode.

Important criteria:
- Every spec in the registry must construct successfully.
- Online-capable specs must expose control flag set to `1`.

Inputs/Outputs:
- Input: orchestration spec names and synthetic decoder/cost callbacks.
- Output: assertions over constructed object contracts.
"""

from __future__ import annotations

import numpy as np
import pytest
import random
from collections.abc import Callable
from rko_solver.components.solution_pool import ElitePool
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.components.solution_pool import SolutionPoolWriter
from rko_solver.metaheuristics.base import MetaheuristicBase
from rko_solver.metaheuristics.brkga_cs import BRKGA_CS
from rko_solver.metaheuristics.grasp import GRASP
from rko_solver.metaheuristics.sa import SA
from rko_solver.orchestration.configuration import MH_SPECS
from rko_solver.orchestration.factory import build_metaheuristic
from typing import Any


class DummyMH(MetaheuristicBase):
    """Minimal concrete implementation for testing the base contract."""

    def solve(self, limit_time: float) -> tuple[list[object], list[float], float]:
        """Run a trivial solve routine and return a fixed feasible result."""
        self._start_timer()
        return [], [0.0], 0.0


def test_solution_pool_implements_search_protocols() -> None:
    """Verify the built-in pool satisfies insertion and elite-access protocols."""
    pool = SolutionPool(capacity=4)
    assert isinstance(pool, SolutionPoolWriter)
    assert isinstance(pool, ElitePool)


def test_grasp_uses_injected_local_search_operator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify GRASP delegates construction refinement to a supplied operator."""
    pool = SolutionPool(capacity=4)
    calls: list[object] = []

    def local_search(
        keys: np.ndarray,
        decoder: Callable[[np.ndarray], Any],
        cost: Callable[[Any], float],
        *,
        pool: ElitePool,
    ) -> tuple[np.ndarray, float]:
        """Return the unchanged vector and record the injected pool."""
        calls.append(pool)
        return keys, cost(decoder(keys))

    grasp = GRASP(
        pool,
        lambda keys: keys,
        lambda decoded: float(np.sum(decoded)),
        3,
        local_search_operator=local_search,
    )
    monkeypatch.setattr(grasp, "_constructive", lambda step, keys: (keys, float(np.sum(keys))))
    monkeypatch.setattr(grasp, "time_limit_reached", lambda duration: len(calls) >= 1)
    grasp.solve(60.0)

    assert calls == [pool]


def test_grasp_rebuilds_candidates_from_committed_partial_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure each construction pass evaluates candidates against current keys."""
    evaluated: list[tuple[int, np.ndarray]] = []

    def line_search(pos: int, h: float, keys: np.ndarray) -> tuple[float, float]:
        """Record the candidate context and return a tied improvement."""
        evaluated.append((pos, keys.copy()))
        return 0.5, 0.0

    grasp = GRASP(
        SolutionPool(capacity=4),
        lambda keys: keys,
        lambda decoded: 0.0,
        4,
    )
    monkeypatch.setattr(grasp, "_line_search", line_search)
    monkeypatch.setattr(random, "uniform", lambda lower, upper: 0.5)
    monkeypatch.setattr(random, "shuffle", lambda values: None)

    grasp._constructive(0.1, np.zeros(4))

    assert len(evaluated) == 7
    assert all(np.array_equal(keys, np.zeros(4)) for _, keys in evaluated[:4])
    assert all(np.array_equal(keys, np.array([0.5, 0.0, 0.0, 0.0])) for _, keys in evaluated[4:])


def test_brkga_cs_helpers_are_instance_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify BRKGA-CS owns and invokes its graph and perturbation helpers.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patches random choices for determinism.
    """
    algorithm = BRKGA_CS(
        SolutionPool(capacity=4),
        lambda keys: keys,
        lambda decoded: float(np.sum(decoded)),
        3,
        sigma=0.9,
    )
    monkeypatch.setattr(random, "shuffle", lambda values: None)
    monkeypatch.setattr(random, "choice", lambda values: values[0])

    labels = algorithm._label_propagation([[(1, 1.0)], [(0, 1.0)], []], 3)
    representatives = algorithm._identify_promising(
        [np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0, 5.0])],
        2,
    )
    keys = np.array([0.2, 0.4])
    monkeypatch.setattr(random, "random", lambda: 0.9)

    assert labels[0] == labels[1]
    assert labels[2] == 2
    assert len(representatives) == 1
    assert representatives[0] in (0, 1)
    assert algorithm._chaotic_ind(keys, 0.7) is keys
    np.testing.assert_array_equal(keys, np.array([0.9, 0.9]))


def test_sa_uses_injected_shaking_operator() -> None:
    """Verify SA invokes a supplied perturbation without requiring defaults."""
    pool = SolutionPool(capacity=4)
    calls: list[tuple[float, float]] = []

    def local_search(
        keys: np.ndarray,
        decoder: Callable[[np.ndarray], Any],
        cost: Callable[[Any], float],
        *,
        pool: ElitePool,
    ) -> tuple[np.ndarray, float]:
        """Return the unchanged vector and its objective value."""
        return keys, cost(decoder(keys))

    def shaking(keys: np.ndarray, beta_min: float, beta_max: float) -> np.ndarray:
        """Return the unchanged vector and record the requested interval."""
        calls.append((beta_min, beta_max))
        return keys

    sa = SA(
        pool,
        lambda keys: keys,
        lambda decoded: float(np.sum(decoded)),
        3,
        sa_max=1,
        local_search_operator=local_search,
        shaking_operator=shaking,
    )
    checks = iter([False, False, False, True, True])
    sa.time_limit_reached = lambda limit_time: next(checks)
    sa.solve(1.0)

    assert len(calls) == 1
    assert calls[0] == (sa.beta_min, sa.beta_max)


def test_metaheuristic_base_timer_and_repr() -> None:
    """Ensure base timer helpers and representation are functional."""
    mh = DummyMH(solution_pool=object(), name="Dummy")
    assert mh.elapsed() == 0.0
    _, keys, cost = mh.solve(limit_time=0.01)
    assert keys == [0.0]
    assert cost == 0.0
    assert mh.elapsed() >= 0.0
    assert "Dummy" in repr(mh)


@pytest.mark.parametrize("mh_name", sorted(MH_SPECS.keys()))
def test_build_metaheuristic_offline_for_all_specs(mh_name: str) -> None:
    """Ensure every registered spec can be built in offline mode."""
    mh = build_metaheuristic(
        mh_name=mh_name,
        pool=SolutionPool(capacity=4),
        decoder=lambda keys: keys,
        cost=lambda decoded: float(np.sum(decoded)),
        tam_solution=6,
        control=0,
        params_dir=None,
    )

    assert isinstance(mh, MetaheuristicBase)
    assert mh.name


@pytest.mark.parametrize(
    "mh_name",
    sorted(name for name, spec in MH_SPECS.items() if spec.supports_control),
)
def test_build_metaheuristic_online_sets_control_flag(mh_name: str) -> None:
    """Ensure control-enabled specs are built with online control flag."""
    mh = build_metaheuristic(
        mh_name=mh_name,
        pool=SolutionPool(capacity=4),
        decoder=lambda keys: keys,
        cost=lambda decoded: float(np.sum(decoded)),
        tam_solution=6,
        control=1,
        params_dir=None,
    )

    assert mh.control == 1
