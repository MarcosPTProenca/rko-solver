"""Contract tests for the optional C++ reference engine.

Objective:
- Verify that `engine="cpp"` honors the public RKO API when the extension was built with the
  C++ engine, and fails clearly when it was not.

Technical approach:
- Probe `cpp_available()`; run short solves on a small Knapsack through built-in problems,
  Python callbacks and every algorithm, recomputing each objective independently.

Flow (pseudocode):
1. Build a deterministic Knapsack instance.
2. Solve with the C++ engine using native problems, callbacks and the full portfolio.
3. Assert objectives, validation errors and the missing-engine message.

Important criteria:
- Budgets below one second are shortened last seconds of the reference's integer clock.
- Tests never assert timing-dependent quality beyond a loose sanity bound.

Inputs/Outputs:
- Input: small synthetic instances and one-second limits.
- Output: pass/fail assertions.
"""

import numpy as np
import pytest
from rko_solver import RKO
from rko_solver.orchestration.configuration import MH_SPECS

native = pytest.importorskip("rko_solver._native")
needs_cpp = pytest.mark.skipif(not native.cpp_available(), reason="built without the C++ engine")

PROFITS = [10 + (index * 37) % 91 for index in range(60)]
WEIGHTS = [5 + (index * 53) % 47 for index in range(60)]
CAPACITY = 700


def python_cost(keys: list[float] | np.ndarray) -> float:
    """Score keys with the reference Knapsack objective.

    Args:
        keys (list[float] | np.ndarray): Random keys.

    Returns:
        float: Penalized negative profit.
    """
    chosen = np.asarray(keys) > 0.5
    profit = int(np.dot(chosen, PROFITS))
    weight = int(np.dot(chosen, WEIGHTS))
    return float(100000 * max(0, weight - CAPACITY) - profit)


@needs_cpp
def test_builtin_problem_reports_an_independently_verified_objective() -> None:
    """The C++ engine returns keys whose Python score equals the reported cost."""
    problem = native.KnapsackProblem(PROFITS, WEIGHTS, CAPACITY)
    result = RKO(problem, engine="cpp", random_state=3).solve(1)
    assert result.engine == "cpp"
    assert python_cost(result.keys) == result.cost
    assert result.cost < -400


@needs_cpp
def test_python_callbacks_and_every_algorithm_run_on_the_cpp_engine() -> None:
    """Callbacks work with the whole portfolio, fixed and adaptive."""
    for adaptation in ("fixed", "adaptive"):
        solver = RKO(
            lambda keys: keys,
            python_cost,
            len(PROFITS),
            algorithms=list(MH_SPECS),
            adaptation=adaptation,
            engine="cpp",
            random_state=5,
        )
        result = solver.solve(1)
        assert python_cost(result.keys) == result.cost
        assert len(result.searches) == len(MH_SPECS)


@needs_cpp
def test_invalid_budgets_and_worker_caps_are_rejected() -> None:
    """The reference engine needs a budget of at least 0.05 s and one thread per algorithm."""
    problem = native.KnapsackProblem(PROFITS, WEIGHTS, CAPACITY)
    with pytest.raises(ValueError, match="0.05"):
        RKO(problem, engine="cpp").solve(0.01)
    with pytest.raises(ValueError, match="n_jobs"):
        RKO(problem, engine="cpp", n_jobs=2).solve(1)


@needs_cpp
def test_ipr_on_tiny_problems_is_rejected_instead_of_crashing_python() -> None:
    """The reference IPR divides by zero below ten keys; the wrapper must refuse first."""
    problem = native.KnapsackProblem([1, 2, 3], [1, 1, 1], 2)
    with pytest.raises(ValueError, match="at least 10 keys"):
        RKO(problem, engine="cpp", algorithms=["IPR"]).solve(1)


@needs_cpp
def test_nonfinite_callback_costs_fail_the_run() -> None:
    """A callback returning NaN surfaces as an error instead of a result."""
    solver = RKO(lambda keys: keys, lambda keys: float("nan"), 10, engine="cpp", algorithms=["SA"])
    with pytest.raises((ValueError, RuntimeError)):
        solver.solve(1)


@pytest.mark.skipif(native.cpp_available(), reason="C++ engine is present")
def test_missing_cpp_engine_is_reported_clearly() -> None:
    """Default builds explain how to enable the engine."""
    problem = native.KnapsackProblem(PROFITS, WEIGHTS, CAPACITY)
    with pytest.raises(ValueError, match="RKO_BUILD_CPP"):
        RKO(problem, engine="cpp").solve(1)
