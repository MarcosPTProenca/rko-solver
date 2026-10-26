"""Contract tests for objectives compiled with Numba and called without the interpreter.

Objective:
- Verify that compiled objectives run in the native engine with correct scoring, parallel
  workers, validation and failure handling.

Technical approach:
- Skip when Numba or the extension is unavailable; compile a Knapsack objective once.
- Recompute every returned objective independently in Python.

Flow (pseudocode):
1. Compile a Knapsack objective as an `@njit` function and as a raw `cfunc`.
2. Solve with several algorithms and compare with independent scoring.
3. Check validation errors and nonfinite handling.

Important criteria:
- Tests use short budgets and never assert timing-dependent quality.

Inputs/Outputs:
- Input: a small synthetic instance and short time limits.
- Output: pass/fail assertions.
"""

import numpy as np
import pytest
from rko_solver import RKO
from rko_solver import Numba

numba = pytest.importorskip("numba")
pytest.importorskip("rko_solver._native")

PROFITS = np.array([10.0 + (index * 37) % 91 for index in range(60)])
WEIGHTS = np.array([5.0 + (index * 53) % 47 for index in range(60)])
CAPACITY = 700.0


@numba.njit
def knapsack(keys: np.ndarray) -> float:
    """Score a key vector with the reference Knapsack objective.

    Args:
        keys (np.ndarray): Random keys.

    Returns:
        float: Penalized negative profit.
    """
    profit = 0.0
    weight = 0.0
    for index in range(keys.shape[0]):
        if keys[index] > 0.5:
            profit += PROFITS[index]
            weight += WEIGHTS[index]
    return 100000.0 * max(0.0, weight - CAPACITY) - profit


def python_cost(keys: tuple[float, ...]) -> float:
    """Recompute the objective without Numba.

    Args:
        keys (tuple[float, ...]): Returned random keys.

    Returns:
        float: Penalized negative profit.
    """
    chosen = np.asarray(keys) > 0.5
    return float(100000.0 * max(0.0, WEIGHTS[chosen].sum() - CAPACITY) - PROFITS[chosen].sum())


@pytest.mark.parametrize("name", ["SA", "BRKGA", "ILS", "GA", "GRASP", "IPR", "PSO"])
def test_njit_objective_runs_and_matches_independent_scoring(name: str) -> None:
    """An njit function is compiled into an entry point and scored natively."""
    result = RKO(
        Numba(knapsack, 60), engine="rust", algorithms=[name], adaptation="adaptive", random_state=4
    ).solve(0.3)
    assert result.engine == "rust"
    assert python_cost(result.keys) == result.cost
    assert result.searches[0].n_evaluations > 100


def test_workers_run_in_parallel_without_the_interpreter_lock() -> None:
    """Several compiled searches overlap in time instead of serializing on the GIL."""
    result = RKO(
        Numba(knapsack, 60), engine="rust", algorithms=["SA", "BRKGA", "ILS", "GA"], random_state=1
    ).solve(0.5)
    assert result.n_workers == 4
    assert all(search.elapsed_seconds >= 0.45 for search in result.searches)
    assert result.elapsed_seconds < 0.5 * 2
    assert all(search.n_evaluations > 1000 for search in result.searches)


def test_cfunc_is_accepted_directly() -> None:
    """A cfunc with the documented signature is used as is."""
    signature = numba.types.float64(numba.types.CPointer(numba.types.float64), numba.types.intp)

    @numba.cfunc(signature)
    def entry(keys: object, size: object) -> float:
        """Score the first key, as a minimal objective.

        Args:
            keys (object): Pointer to the key buffer.
            size (object): Number of keys.

        Returns:
            float: Sum of the keys.
        """
        total = 0.0
        for index in range(size):
            total += keys[index]
        return total

    result = RKO(Numba(entry, 8), engine="rust", algorithms=["SA"], random_state=2).solve(0.2)
    assert result.cost == pytest.approx(sum(result.keys))
    assert result.cost < 1.0


def test_invalid_inputs_are_rejected() -> None:
    """Plain Python functions and bad dimensions fail before any search starts."""
    with pytest.raises(ValueError, match="numba"):
        RKO(Numba(lambda keys: 0.0, 8), engine="rust")
    with pytest.raises(ValueError, match="n_variables"):
        RKO(Numba(knapsack, 0), engine="rust")


def test_nonfinite_objective_fails_the_solve() -> None:
    """A compiled objective returning NaN is rejected like any other objective."""

    @numba.njit
    def broken(keys: np.ndarray) -> float:
        """Return a nonfinite value.

        Args:
            keys (np.ndarray): Random keys.

        Returns:
            float: NaN.
        """
        return np.nan

    with pytest.raises(RuntimeError, match="finite"):
        RKO(Numba(broken, 8), engine="rust", algorithms=["SA"]).solve(0.2)
