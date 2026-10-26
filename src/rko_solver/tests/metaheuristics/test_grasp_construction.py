"""Check GRASP construction counts, subset fallback and committed partial states.

Objective:
- Match the reference real-bound loop and small-unfixed-set search cardinality.

Technical approach:
- Drive the actual constructive method with controlled intensity and shuffle order.
- Compare every line-search context with a separate scalar/list reference model.
- Exercise actual one-variable progress, callback stopping and caller ownership.

Flow (pseudocode):
1. Run a literal real-bound construction model using deterministic coordinate costs.
2. Observe actual evaluated coordinates and partial vectors after each commit.
3. Check rounded intensity, threshold crossings and finite returned objective values.

Important criteria:
- Intensity is rounded upward, but no coordinate is committed more than once.
- A 10-percent subset below two means every remaining coordinate, not two.
- Subsets come from a shuffled copy; the unfixed order survives every pass.

Inputs/Outputs:
- Input: positive dimensions, matched intensities, numerical objectives and owned RNGs.
- Output: complete construction/count/ownership assertions or the original callback stop.
"""

import math
import numpy as np
import pickle
import pytest
import random
from rko_solver.components.random_streams import RandomStreams
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.grasp import GRASP
from rko_solver.orchestration.search import SearchSession
from rko_solver.orchestration.search import _BudgetExpired


def decode(keys: np.ndarray) -> np.ndarray:
    """Preserve continuous random keys for these domain-neutral objectives.

    Args:
        keys (np.ndarray): Candidate chromosome.

    Returns:
        np.ndarray: Unmodified payload.
    """
    return keys


def objective(keys: np.ndarray) -> float:
    """Prefer changing larger-index coordinates under a deterministic linear objective.

    Args:
        keys (np.ndarray): Numerical chromosome.

    Returns:
        float: Negative weighted sum, minimized by larger committed coordinates.
    """
    return -float(np.dot(keys, np.arange(1, len(keys) + 1)))


def construction_reference(
    dimension: int, intensity: float, reverse_order: bool
) -> tuple[list[tuple[int, tuple[float, ...]]], list[float]]:
    """Model the reference loop count and fallback with explicit list-based states.

    Args:
        dimension (int): Positive chromosome dimension.
        intensity (float): Matched real construction fraction.
        reverse_order (bool): Reverse a fresh copy of remaining coordinates at each
            shuffle point, leaving the stored order itself untouched.

    Returns:
        tuple[list[tuple[int, tuple[float, ...]]], list[float]]: Expected callbacks and final keys.
    """
    state = [0.0] * dimension
    remaining = list(range(dimension))
    visits: list[tuple[int, tuple[float, ...]]] = []
    completed = 0
    while completed < dimension * intensity:
        limit = int(len(remaining) * 0.1)
        if limit < 2:
            limit = len(remaining)
        subset = list(reversed(remaining)) if reverse_order else list(remaining)
        best_score = math.inf
        best_pos = subset[0]
        for pos in subset[:limit]:
            visits.append((pos, tuple(state)))
            trial = state.copy()
            trial[pos] = 0.5
            score = -sum((index + 1) * value for index, value in enumerate(trial))
            if score < best_score:
                best_pos, best_score = pos, score
        state[best_pos] = 0.5
        remaining.remove(best_pos)
        completed += 1
    return visits, state


def test_shuffled_copy_preserves_remaining_order_and_first_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep each remaining list ordered while choosing first-tied shuffled candidates.

    Args:
        monkeypatch (pytest.MonkeyPatch): Reverse each selection and fix equal scores.
    """
    shuffles: list[list[int]] = []
    before_shuffle: list[list[int]] = []
    visits: list[int] = []
    streams = RandomStreams(42)

    def reverse_copy(values: list[int]) -> None:
        """Record each independent input, then reverse it for deterministic selection.

        Args:
            values (list[int]): Fresh candidate-order list.
        """
        before_shuffle.append(values.copy())
        shuffles.append(values)
        values.reverse()

    def tied_search(pos: int, h: float, keys: np.ndarray) -> tuple[float, float]:
        """Return equal costs so the first selected coordinate wins each commit.

        Args:
            pos (int): Evaluated coordinate.
            h (float): Forwarded grid step.
            keys (np.ndarray): Committed partial state.

        Returns:
            tuple[float, float]: Candidate value and tied cost.
        """
        visits.append(pos)
        return 0.5, 0.0

    algorithm = GRASP(SolutionPool(), decode, lambda keys: 0.0, 4)
    monkeypatch.setattr(algorithm, "_line_search", tied_search)
    monkeypatch.setattr(streams.python, "uniform", lambda lower, upper: 0.7)
    monkeypatch.setattr(streams.python, "shuffle", reverse_copy)
    with streams.activate():
        result, cost = algorithm._constructive(0.125, np.zeros(4))
    assert before_shuffle == [[0, 1, 2, 3], [0, 1, 2], [0, 1]]
    assert len({id(values) for values in shuffles}) == 3
    assert visits == [3, 2, 1, 0, 2, 1, 0, 1, 0]
    assert result.tolist() == [0.0, 0.5, 0.5, 0.5] and cost == 0.0


@pytest.mark.parametrize(
    "dimension, intensity",
    [
        (1, 0.3),
        (1, 0.7),
        (2, 0.3),
        (3, 0.7),
        (4, 0.5),
        (4, math.nextafter(0.5, 0.0)),
        (4, math.nextafter(0.5, 1.0)),
        (10, 0.31),
        (19, 0.3),
        (20, 0.3),
        (21, 0.7),
        (39, 0.3),
        (40, 0.7),
    ],
)
@pytest.mark.parametrize("reverse_order", [False, True])
def test_constructive_matches_counts_and_partial_contexts(
    monkeypatch: pytest.MonkeyPatch, dimension: int, intensity: float, reverse_order: bool
) -> None:
    """Compare complete coordinate traces, commits, results and RNG ownership.

    Args:
        monkeypatch (pytest.MonkeyPatch): Control intensity, order and coordinate scoring.
        dimension (int): Small dimension or boundary near the 10-percent cutoff.
        intensity (float): Fixed fraction, including integer and adjacent floating bounds.
        reverse_order (bool): Controlled shuffled-copy selection.
    """
    expected, expected_keys = construction_reference(dimension, intensity, reverse_order)
    observed: list[tuple[int, tuple[float, ...]]] = []
    streams = RandomStreams(42)
    algorithm = GRASP(SolutionPool(), decode, objective, dimension)

    def line_search(pos: int, h: float, keys: np.ndarray) -> tuple[float, float]:
        """Score one possible commit without modifying the shared partial solution.

        Args:
            pos (int): Candidate coordinate.
            h (float): Forwarded fixed step.
            keys (np.ndarray): Current committed partial solution.

        Returns:
            tuple[float, float]: A0.5 coordinate and its independently evaluated cost.
        """
        assert h == 0.125
        observed.append((pos, tuple(keys)))
        trial = keys.copy()
        trial[pos] = 0.5
        return 0.5, objective(trial)

    monkeypatch.setattr(algorithm, "_line_search", line_search)
    monkeypatch.setattr(streams.python, "uniform", lambda lower, upper: intensity)
    monkeypatch.setattr(
        streams.python, "shuffle", lambda values: values.reverse() if reverse_order else None
    )
    keys = np.zeros(dimension)
    keys.flags.writeable = False
    before = pickle.dumps((random.getstate(), np.random.get_state()))
    with streams.activate():
        result, score = algorithm._constructive(0.125, keys)
    assert observed == expected
    assert result.tolist() == expected_keys
    assert np.count_nonzero(result) == math.ceil(dimension * intensity)
    assert score == objective(result)
    assert np.array_equal(keys, np.zeros(dimension))
    assert not np.shares_memory(result, keys)
    assert pickle.dumps((random.getstate(), np.random.get_state())) == before
    assert streams.python.getstate() == random.Random(42).getstate()


def test_subset_recomputed_when_remaining_count_crosses_twenty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Switch from two sampled coordinates at20 to all nineteen in the next pass.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fix the construction fraction and shuffle order.
    """
    counts: dict[int, int] = {}

    def line_search(pos: int, h: float, keys: np.ndarray) -> tuple[float, float]:
        """Count the candidates evaluated at each independently observable partial state.

        Args:
            pos (int): Candidate coordinate.
            h (float): Unused construction step.
            keys (np.ndarray): Partial solution with one nonzero value per prior commit.

        Returns:
            tuple[float, float]: A deterministic coordinate improvement.
        """
        stage = int(np.count_nonzero(keys))
        counts[stage] = counts.get(stage, 0) + 1
        trial = keys.copy()
        trial[pos] = 0.5
        return 0.5, objective(trial)

    streams = RandomStreams(42)
    monkeypatch.setattr(streams.python, "uniform", lambda lower, upper: 0.3)
    monkeypatch.setattr(streams.python, "shuffle", lambda values: None)
    algorithm = GRASP(SolutionPool(), decode, objective, 21)
    monkeypatch.setattr(algorithm, "_line_search", line_search)
    with streams.activate():
        algorithm._constructive(0.125, np.zeros(21))
    assert [counts[index] for index in range(7)] == [2, 2, 19, 18, 17, 16, 15]


def test_one_variable_performs_a_real_line_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Construct a real improved point instead of truncating the single step to zero.

    Args:
        monkeypatch (pytest.MonkeyPatch): Select known eligible ranks deterministically.
    """
    observations: list[float] = []

    def target(keys: np.ndarray) -> float:
        """Score a numerical target available in the corrected coordinate lattice.

        Args:
            keys (np.ndarray): Single-coordinate candidate.

        Returns:
            float: Squared target distance.
        """
        observations.append(float(keys[0]))
        return float((keys[0] - 0.375) ** 2)

    streams = RandomStreams(42)
    monkeypatch.setattr(streams.python, "uniform", lambda lower, upper: 0.3)
    monkeypatch.setattr(streams.python, "randrange", lambda stop: 0)
    algorithm = GRASP(SolutionPool(), decode, target, 1)
    with streams.activate():
        keys, score = algorithm._constructive(0.25, np.array([0.875]))
    assert observations == [0.875, 0.125, 0.375, 0.375]
    assert keys.tolist() == [0.375] and score == 0


def test_construction_honors_callback_deadline() -> None:
    """Keep expanded coordinate scans inside the existing SearchSession stop boundary."""
    session = SearchSession(decode, objective)
    session.deadline = -1.0
    algorithm = GRASP(SolutionPool(), session.decode, session.evaluate, 3)
    keys = np.full(3, 0.375)
    original = keys.copy()
    with RandomStreams(42).activate(), pytest.raises(_BudgetExpired):
        algorithm._constructive(0.125, keys)
    assert session.n_evaluations == 1 and session.best is not None
    assert np.array_equal(keys, original)


def test_construction_propagates_callback_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not swallow a failed coordinate evaluation or mutate caller keys.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fail the first requested line search.
    """

    def fail(pos: int, h: float, keys: np.ndarray) -> tuple[float, float]:
        """Mutate only the private partial vector before propagating an original error.

        Args:
            pos (int): Candidate coordinate.
            h (float): Requested step.
            keys (np.ndarray): Private partial vector.

        Returns:
            tuple[float, float]: Never returns.

        Raises:
            ArithmeticError: Always, to exercise failure propagation.
        """
        keys[:] = 99
        raise ArithmeticError("constructive line failed")

    algorithm = GRASP(SolutionPool(), decode, objective, 1)
    monkeypatch.setattr(algorithm, "_line_search", fail)
    keys = np.zeros(1)
    with RandomStreams(42).activate(), pytest.raises(ArithmeticError, match="constructive line"):
        algorithm._constructive(0.125, keys)
    assert np.array_equal(keys, np.zeros(1))
