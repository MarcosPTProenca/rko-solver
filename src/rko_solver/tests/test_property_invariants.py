"""Property-based tests for reusable RKO behavior contracts.

Objective:
- Exercise pool, random-key, validation, Q-learning, and solver invariants
  across generated inputs.

Technical approach:
- Use bounded Hypothesis strategies and deterministic CI settings.
- Assert public API invariants rather than implementation details.

Flow (pseudocode):
1. Generate valid and invalid dimensions, keys, entries, and parameter grids.
2. Run the public component or solver API for each generated case.
3. Assert ordering, bounds, rejection, transition, and result consistency.

Important criteria:
- Generated numeric values exclude nonfinite values unless the test targets
  validation rejection.
- Solver cases use a small serial MultiStart portfolio and a short budget.

Inputs/Outputs:
- Input: Hypothesis-generated pool entries, keys, parameter grids, and dimensions.
- Output: Assertions over public component and solver contracts.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from math import prod
from rko_solver import RKO
from rko_solver.components.adaptive_control.q_learning import QLearningController
from rko_solver.components.initialization.initialization import create_initial_solution
from rko_solver.components.shaking.mirror import mirror_shake
from rko_solver.components.shaking.neighbor_swap import neighbor_swap_shake
from rko_solver.components.shaking.random import shake
from rko_solver.components.shaking.swap import swap_shake
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.ga import GeneticAlgorithm

FINITE_FLOAT = st.floats(allow_nan=False, allow_infinity=False, width=32)
UNIT_FLOAT = st.floats(
    min_value=0.0,
    max_value=1.0,
    exclude_max=True,
    allow_nan=False,
    allow_infinity=False,
)


def test_solution_pool_default_source_is_reported_as_unknown(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep default source attribution stable in verbose pool output.

    Args:
        capsys (pytest.CaptureFixture[str]): Captured standard output fixture.
    """
    pool = SolutionPool(verbose=True)
    pool.insert((1.0, [0.5]))

    assert "algorithm=unknown" in capsys.readouterr().out


@settings(max_examples=60, derandomize=True, deadline=None)
@given(
    capacity=st.integers(min_value=1, max_value=8),
    entries=st.lists(
        st.tuples(
            FINITE_FLOAT,
            st.lists(FINITE_FLOAT, min_size=1, max_size=8),
        ),
        max_size=24,
    ),
)
@example(capacity=2, entries=[(0.0, [0.0]), (0.0, [0.0]), (1.0, [1.0])])
@example(capacity=2, entries=[(10.0, [0.1]), (5.0, [0.2]), (2.0, [0.1])])
def test_solution_pool_retains_sorted_best_entries(
    capacity: int,
    entries: list[tuple[float, list[float]]],
) -> None:
    """Keep the lowest per-chromosome observations within the configured capacity.

    Args:
        capacity (int): Maximum pool occupancy.
        entries (list[tuple[float, list[float]]]): Finite objective/key pairs.
    """
    pool = SolutionPool(capacity=capacity)
    for entry in entries:
        pool.insert(entry)

    elite = pool.get_elite(capacity + 1)
    best_observations: dict[tuple[float, ...], float] = {}
    for cost, keys in entries:
        vector = tuple(keys)
        best_observations[vector] = min(cost, best_observations.get(vector, cost))
    expected_costs = sorted(best_observations.values())[:capacity]

    assert pool.size == min(capacity, len(best_observations))
    assert len({tuple(keys) for _, keys in elite}) == pool.size
    assert all(cost == best_observations[tuple(keys)] for cost, keys in elite)
    assert [cost for cost, _ in elite] == expected_costs
    costs = [cost for cost, _ in elite]
    assert costs == sorted(costs)


@settings(max_examples=60, derandomize=True, deadline=None)
@given(
    keys=st.lists(
        st.floats(
            min_value=-100.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=1,
        max_size=12,
    )
)
def test_solution_pool_queries_return_detached_key_vectors(keys: list[float]) -> None:
    """Prevent mutations to query results from changing stored entries.

    Args:
        keys (list[float]): Finite generated key vector.
    """
    original = keys.copy()
    pool = SolutionPool(capacity=1)
    pool.insert((1.0, keys))

    best = pool.best
    assert best is not None
    best[1][0] = 999.0
    pool.get_elite(1)[0][1][0] = 999.0

    assert pool.best == (1.0, original)


@settings(max_examples=60, derandomize=True, deadline=None)
@given(size=st.integers(min_value=1, max_value=128))
def test_random_key_initialization_respects_dimension_and_bounds(size: int) -> None:
    """Generate one finite random key in the unit interval per dimension.

    Args:
        size (int): Positive requested vector dimension.
    """
    keys = create_initial_solution(size)

    assert keys.shape == (size,)
    assert np.isfinite(keys).all()
    assert np.all((keys >= 0.0) & (keys < 1.0))


@settings(max_examples=60, derandomize=True, deadline=None)
@given(keys=st.lists(UNIT_FLOAT, min_size=1, max_size=64))
def test_stochastic_shaking_preserves_key_shape_and_bounds(keys: list[float]) -> None:
    """Keep every packaged shaking operator finite, bounded, and same-length.

    Args:
        keys (list[float]): Valid random-key vector.
    """
    vector = np.asarray(keys, dtype=float)
    results = [
        operator(vector.copy()) for operator in (mirror_shake, neighbor_swap_shake, swap_shake)
    ]
    results.append(shake(vector.copy(), beta_min=0.0, beta_max=1.0))

    for result in results:
        assert result.shape == vector.shape
        assert np.isfinite(result).all()
        assert np.all((result >= 0.0) & (result < 1.0))


@settings(max_examples=60, derandomize=True, deadline=None)
@given(
    dimension=st.one_of(
        st.integers(max_value=0),
        st.booleans(),
        st.floats(allow_nan=True, allow_infinity=True),
    )
)
def test_solver_rejects_invalid_variable_dimensions(dimension: object) -> None:
    """Reject nonpositive or noninteger dimensions before starting a search.

    Args:
        dimension (object): Generated invalid variable dimension.
    """
    solver = RKO(lambda keys: keys, lambda decoded: float(np.sum(decoded)), dimension)

    with pytest.raises(ValueError, match="n_variables"):
        solver.solve(time_limit=0.01)


@settings(max_examples=60, derandomize=True, deadline=None)
@given(
    probability=st.one_of(
        st.floats(
            max_value=0.0,
            exclude_max=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        st.floats(
            min_value=1.0,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        st.booleans(),
        st.just(float("nan")),
        st.just(float("inf")),
        st.just(float("-inf")),
    )
)
def test_genetic_algorithm_rejects_generated_invalid_mutation_probabilities(
    probability: object,
) -> None:
    """Reject mutation probabilities outside the finite numeric unit interval.

    Args:
        probability (object): Generated invalid probability value.
    """
    with pytest.raises(ValueError, match="prob_mut"):
        GeneticAlgorithm(
            SolutionPool(),
            lambda keys: keys,
            lambda decoded: float(np.sum(decoded)),
            3,
            prob_mut=probability,
        )


@settings(max_examples=50, derandomize=True, deadline=None)
@given(
    axes=st.lists(
        st.lists(st.integers(min_value=-3, max_value=3), min_size=1, max_size=3, unique=True),
        min_size=1,
        max_size=3,
    ),
    epsilon=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    reward=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    learning_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    discount=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_q_learning_states_and_updates_preserve_graph_invariants(
    axes: list[list[int]],
    epsilon: float,
    reward: float,
    learning_rate: float,
    discount: float,
) -> None:
    """Keep generated Q-state edges valid and each update's best-action cache consistent.

    Args:
        axes (list[list[int]]): Nonempty axes defining a Cartesian state space.
        epsilon (float): Exploration probability in the unit interval.
        reward (float): Finite generated transition reward.
        learning_rate (float): Bellman update learning rate.
        discount (float): Bellman update discount factor.
    """
    parameters = [[float(value) for value in axis] for axis in axes]
    controller = QLearningController(parameters)

    assert len(controller.states) == prod(map(len, parameters))
    for state_index, state in enumerate(controller.states):
        assert state.label == state_index
        assert state_index in state.Ai
        assert len(state.Ai) == len(set(state.Ai))
        assert len(state.par) == len(parameters)
        assert len(state.Ai) == len(state.Qa)
        assert all(0 <= target < len(controller.states) for target in state.Ai)
        assert all(
            sum(
                left != right
                for left, right in zip(state.par, controller.states[target].par, strict=True)
            )
            <= 1
            for target in state.Ai
        )

        action = controller.choose_action(state_index, epsilon)
        assert action in range(len(state.Ai))
        controller.update_q_value(state_index, action, reward, learning_rate, discount)
        assert state.maxA in range(len(state.Qa))
        assert state.maxQ == max(state.Qa)
        assert state.Qa[state.maxA] == state.maxQ


@settings(max_examples=8, derandomize=True, deadline=None)
@given(
    dimension=st.integers(min_value=1, max_value=8),
    random_state=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_serial_solver_results_match_their_returned_keys(
    dimension: int,
    random_state: int,
) -> None:
    """Return bounded keys whose cost matches the public objective callback.

    Args:
        dimension (int): Positive random-key dimension.
        random_state (int): Valid seed for serial execution.
    """
    solver = RKO(
        lambda keys: keys,
        lambda decoded: float(np.sum(np.square(decoded))),
        dimension,
        algorithms=["MultiStart"],
        pool_size=1,
        execution="sequential",
        random_state=random_state,
    )

    result = solver.solve(time_limit=0.001)

    assert len(result.keys) == dimension
    assert all(0.0 <= key < 1.0 for key in result.keys)
    assert result.cost == pytest.approx(sum(key**2 for key in result.keys))
    assert result.random_state == random_state
