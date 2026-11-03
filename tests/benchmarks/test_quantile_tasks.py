"""Independent construction and feasibility contracts for two combinatorial tasks.

Objective:
- Validate exact legal actions and objectives independently of learned preferences.

Technical approach:
- Use literal tiny tasks with manually established feasible solutions and costs.

Flow (pseudocode):
1. Create a complete tour instance or a fully coverable incidence matrix.
2. Query legal actions and finish an admissible trace.
3. Check independent costs and malformed-solution rejection.

Important criteria:
- A valid local action must not destroy the stated finite-horizon completion guarantee.

Inputs/Outputs:
- Input: small task data and explicit action traces.
- Output: assertions on task semantics and exact objectives.
"""

import numpy as np
import pytest
from benchmarks.quantile import tasks


def test_square_tour_actions_and_independent_objective() -> None:
    """A unit-square boundary tour visits every city once and costs exactly four."""
    task = tasks.Tour(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    assert task.horizon == 3
    assert task.legal(()) == (1, 2, 3)
    assert task.legal((1,)) == (2, 3)
    assert task.terminal((1, 2, 3))
    solution = task.finish((1, 2, 3))
    assert solution == (0, 1, 2, 3, 0)
    assert task.check(solution) == 4.0
    with pytest.raises(ValueError, match="tour"):
        task.check((0, 1, 0, 2, 0))


def test_cover_progress_and_independent_objective() -> None:
    """Two unit-cost singleton columns beat a three-cost column covering both rows."""
    task = tasks.Cover(np.array([[1, 0, 1], [0, 1, 1]]), np.array([1.0, 1.0, 3.0]))
    assert task.horizon == 3
    assert task.legal(()) == (0, 1, 2)
    assert task.legal((0,)) == (1, 2)
    assert task.terminal((0, 1))
    solution = task.finish((0, 1))
    assert solution == (0, 1)
    assert task.check(solution) == 2.0
    with pytest.raises(ValueError, match="cover"):
        task.check((0,))


@pytest.mark.parametrize(
    "coordinates", [[], [[0.0, 0.0]], [[0.0, float("nan")], [1.0, 1.0]], [[0.0], [1.0]]]
)
def test_invalid_tour_data_is_rejected(coordinates: list[object]) -> None:
    """Reject malformed geometry before any construction.

    Args:
        coordinates (list[object]): Invalid coordinates.
    """
    with pytest.raises(ValueError, match="coordinates"):
        tasks.Tour(np.asarray(coordinates))


@pytest.mark.parametrize(
    "matrix,costs",
    [
        ([[0, 0]], [1.0, 2.0]),
        ([[1, 2]], [1.0, 2.0]),
        ([[1]], [0.0]),
        ([[1]], [float("nan")]),
        ([[1, 1]], [1.0]),
    ],
)
def test_invalid_cover_data_is_rejected(matrix: list[list[int]], costs: list[float]) -> None:
    """Reject un-coverable rows, nonbinary incidence and invalid costs.

    Args:
        matrix (list[list[int]]): Invalid incidence or cost-associated geometry.
        costs (list[float]): Candidate costs.
    """
    with pytest.raises(ValueError):
        tasks.Cover(np.asarray(matrix), np.asarray(costs))


def test_task_data_is_detached_and_read_only() -> None:
    """Mutating caller arrays cannot change task semantics after construction."""
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0]])
    tour = tasks.Tour(coordinates)
    coordinates[1] = 99.0
    assert tour.check((0, 1, 0)) == 2.0
    with pytest.raises(ValueError):
        tour.coordinates[0] = 1.0
    incidence, costs = np.ones((1, 1)), np.ones(1)
    cover = tasks.Cover(incidence, costs)
    incidence[:] = 0.0
    costs[:] = 99.0
    assert cover.check((0,)) == 1.0


def test_invalid_construction_traces_are_rejected() -> None:
    """Repeated cities and a redundant action after coverage are not valid construction states."""
    tour = tasks.Tour(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    cover = tasks.Cover(np.ones((1, 2)), np.ones(2))
    for trace in ((0,), (1, 1), (3,)):
        with pytest.raises(ValueError, match="trace"):
            tour.legal(trace)
    with pytest.raises(ValueError, match="trace"):
        cover.legal((0, 1))


def test_contextual_features_and_analytic_preferences() -> None:
    """Features expose task state while analytic scores prefer shorter or cheaper actions."""
    tour = tasks.Tour(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    features = tour.features((1,), (2, 3))
    assert features.shape == (2, 4)
    np.testing.assert_allclose(features[:, 0], [1.0 / np.sqrt(2.0), 1.0])
    assert tour.scores((1,), (2, 3))[0] > tour.scores((1,), (2, 3))[1]
    cover = tasks.Cover(np.array([[1, 0, 1], [0, 1, 1]]), np.array([1.0, 1.0, 3.0]))
    assert cover.features((), (0, 1, 2)).shape == (3, 4)
    assert cover.scores((), (0, 1, 2))[0] > cover.scores((), (0, 1, 2))[2]
