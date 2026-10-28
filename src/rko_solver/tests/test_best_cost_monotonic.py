"""Best-so-far cost invariants across engines, portfolios and adaptation modes.

Objective:
- Pin that the reported best cost never loses an improvement: it equals the lowest objective
  value any evaluation produced, and every recorded history is non-increasing.

Technical approach:
- A recording objective stores every value the engine asks for. Hypothesis draws portfolios of
  one to three algorithms, adaptation modes, dimensions, seeds and evaluation limits.
- The assertions compare the result and the bounded histories with that independent record, so
  a pool, restart or migration step that drops an improvement shows up as a mismatch.

Flow (pseudocode):
1. Generate a configuration and solve it with a recording objective and history enabled.
2. Assert the result cost equals the minimum recorded objective value.
3. Assert every history is non-increasing and ends at or above the result cost, and that the
   best final history cost over the searches equals the result cost.

Important criteria:
- This is the archive-monotonicity assumption of the best-so-far convergence argument for
  random-key search; it does not claim a convergence rate or finite-time quality.
- Runs are sequential or single-worker so the recorded stream is complete; the C++ engine takes
  part when it is compiled in.
- Hypothesis runs derandomized with no deadline, like the other property tests.

Inputs/Outputs:
- Input: generated portfolios, dimensions, seeds and evaluation limits.
- Output: assertions tying the result and histories to every evaluated objective value.
"""

import numpy as np
import pytest
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from rko_solver import RKO
from rko_solver import Callbacks
from rko_solver.orchestration.configuration import MH_SPECS

native = pytest.importorskip("rko_solver._native")
SETTINGS = {"python": {"execution": "sequential"}, "rust": {"n_jobs": 1}, "cpp": {}}
needs_cpp = pytest.mark.skipif(not native.cpp_available(), reason="built without the C++ engine")
ENGINES = ["python", "rust", pytest.param("cpp", marks=needs_cpp)]
PORTFOLIOS = st.lists(st.sampled_from(sorted(MH_SPECS)), min_size=1, max_size=3, unique=True)
FUZZ = {"deadline": None, "derandomize": True, "suppress_health_check": list(HealthCheck)}


def solve_recorded(
    engine: str, algorithms: list[str], dimension: int, **options: object
) -> tuple[object, list[float]]:
    """Solve with an objective that records every value it returns.

    Args:
        engine (str): Key of `SETTINGS`.
        algorithms (list[str]): Portfolio to run.
        dimension (int): Number of random keys.
        **options (object): Additional `RKO` settings.

    Returns:
        tuple[object, list[float]]: The `RKOResult` and every objective value, in call order.
    """
    seen: list[float] = []

    def cost(point: np.ndarray) -> float:
        """Return the squared distance to 0.3 and record it.

        Args:
            point (np.ndarray): Decoded keys.

        Returns:
            float: Objective to minimize.
        """
        value = float(np.sum((np.asarray(point) - 0.3) ** 2))
        seen.append(value)
        return value

    result = RKO(
        Callbacks(lambda keys: np.asarray(keys), cost, dimension),
        engine=engine,
        algorithms=algorithms,
        history_limit=16,
        **SETTINGS[engine],
        **options,
    ).solve(60.0)
    return result, seen


@pytest.mark.parametrize("engine", ENGINES)
@settings(max_examples=60, **FUZZ)
@given(
    algorithms=PORTFOLIOS,
    adaptation=st.sampled_from(["fixed", "adaptive"]),
    dimension=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    limit=st.integers(min_value=20, max_value=200),
)
def test_the_reported_best_is_the_lowest_objective_ever_evaluated(
    engine: str, algorithms: list[str], adaptation: str, dimension: int, seed: int, limit: int
) -> None:
    """No improvement is lost between the searches, the shared pool and the final result.

    Args:
        engine (str): Engine under test.
        algorithms (list[str]): Generated portfolio of one to three algorithms.
        adaptation (str): `fixed` or `adaptive`.
        dimension (int): Number of keys.
        seed (int): Random seed.
        limit (int): Evaluation limit per search.
    """
    result, seen = solve_recorded(
        engine,
        algorithms,
        dimension,
        adaptation=adaptation,
        max_evaluations=limit,
        random_state=seed,
    )
    assert seen
    assert result.cost == min(seen)
    finals = []
    for search in result.searches:
        costs = [point.cost for point in search.history]
        assert costs == sorted(costs, reverse=True), search.algorithm
        assert all(cost >= result.cost for cost in costs), search.algorithm
        finals.append(costs[-1])
    assert min(finals) == result.cost
