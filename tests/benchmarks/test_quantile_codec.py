"""Trace-level inverse and construction checks for the reference quantile mapping.

Objective:
- Verify deterministic feasible construction and an exact right inverse on admissible traces.

Technical approach:
- Enumerate every fixed-start tour on seven cities and use tiny independent covering cases.

Flow (pseudocode):
1. Decode valid key vectors using exact task semantics.
2. Encode admissible traces using interior cumulative-probability midpoints.
3. Replay and independently compare solutions and costs.

Important criteria:
- Trace equality precedes optional symmetry normalization or cover-column deletion.

Inputs/Outputs:
- Input: tour/cover tasks, keys and explicit admissible traces.
- Output: exact replay, independent costs and invalid-input rejection.
"""

import itertools
import numpy as np
import pytest
from benchmarks.quantile import codec
from benchmarks.quantile.tasks import Cover
from benchmarks.quantile.tasks import Tour
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st


def test_tour_decoding_and_midpoint_round_trip() -> None:
    """A three-city trace can be encoded without a learned inverse network."""
    task = Tour(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    mapping = codec.TrajectoryCodec(task)
    candidate = mapping.decode(np.zeros(task.horizon))
    assert candidate.trace == (1, 2)
    assert candidate.solution == (0, 1, 2, 0)
    assert candidate.cost == pytest.approx(2.0 + np.sqrt(2.0))
    encoded = mapping.encode((2, 1))
    replay = mapping.decode(encoded)
    assert replay.trace == (2, 1)
    assert replay.cost == task.check(replay.solution)
    assert np.all(encoded > 0.0) and np.all(encoded < 1.0)


def test_greedy_reference_is_checked_and_reconstructible() -> None:
    """The analytic baseline supplies a checked feasible reference without sampled keys."""
    tour = Tour(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    cover = Cover(np.array([[1, 0, 1], [0, 1, 1]]), np.array([1.0, 1.0, 3.0]))
    for task, expected in ((tour, 4.0), (cover, 2.0)):
        mapping = codec.TrajectoryCodec(task)
        candidate = mapping.greedy()
        assert candidate.cost == expected
        assert mapping.decode(mapping.encode(candidate.trace)) == candidate


@pytest.mark.parametrize(
    "keys", [[0.0], [0.0, 0.0, 0.0], [[0.0, 0.0]], [float("nan"), 0.0], [1.0, 0.0], [-0.1, 0.0]]
)
def test_entire_key_vector_is_validated(keys: list[object]) -> None:
    """Reject invalid dimensions or values before any construction.

    Args:
        keys (list[object]): Malformed decision vector.
    """
    task = Tour(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="keys"):
        codec.TrajectoryCodec(task).decode(np.asarray(keys))


def test_unused_padding_is_neutral_but_still_validated() -> None:
    """Early termination does not allow an invalid value to hide in unused padding."""
    mapping = codec.TrajectoryCodec(Cover(np.ones((1, 3)), np.ones(3)))
    assert mapping.decode(np.zeros(3)) == mapping.decode(np.array([0.0, 0.9, 0.7]))
    with pytest.raises(ValueError, match="keys"):
        mapping.decode(np.array([0.0, float("nan"), 0.0]))


@pytest.mark.parametrize("trace", [(), (1,), (1, 1), (1, 2, 1), (1.0, 2), (0, 1)])
def test_encoding_rejects_nonadmissible_or_incomplete_traces(trace: tuple[object, ...]) -> None:
    """Reject repeated, truncated, overlong or noninteger trajectories.

    Args:
        trace (tuple[object, ...]): Invalid action trajectory.
    """
    task = Tour(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        codec.TrajectoryCodec(task).encode(trace)


def test_cover_trace_cannot_continue_after_full_coverage() -> None:
    """Feasible sets are not automatically admissible action traces after coverage."""
    task = Cover(np.ones((1, 2)), np.ones(2))
    with pytest.raises(ValueError, match="trace"):
        codec.TrajectoryCodec(task).encode((0, 1))


def test_all_720_fixed_start_seven_city_tours_round_trip() -> None:
    """Every permutation of six noninitial cities retains its original ordered trace."""
    task = Tour(np.random.default_rng(20261027).random((7, 2)))
    mapping = codec.TrajectoryCodec(task)
    count = 0
    for trace in itertools.permutations(range(1, 7)):
        result = mapping.decode(mapping.encode(trace))
        assert result.trace == trace
        assert result.cost == task.check((0, *trace, 0))
        count += 1
    assert count == 720


def test_all_tiny_admissible_cover_traces_round_trip() -> None:
    """Enumerated binary matrices and independent set coverage define the admissible oracle."""
    count = 0
    for cells in itertools.product((0, 1), repeat=6):
        matrix = np.asarray(cells).reshape(2, 3)
        columns = [{row for row in range(2) if matrix[row, column]} for column in range(3)]
        if set.union(*columns) != {0, 1}:
            continue
        mapping = codec.TrajectoryCodec(Cover(matrix, np.array([1.0, 2.0, 3.0])))
        for length in range(1, 4):
            for trace in itertools.permutations(range(3), length):
                covered = set()
                admissible = True
                for column in trace:
                    if not columns[column] - covered:
                        admissible = False
                    covered |= columns[column]
                if not admissible or covered != {0, 1}:
                    continue
                result = mapping.decode(mapping.encode(trace))
                assert result.trace == trace
                assert result.cost == sum(column + 1.0 for column in trace)
                count += 1
    assert count > 100


class DeadEndTour(Tour):
    """Model an invalid semantic adapter that has a nonterminal empty action set.

    Responsibilities:
    - Exercise the construction-failure boundary without mocking numerical internals.
    """

    def terminal(self, trace: tuple[int, ...]) -> bool:
        """Never terminate this deliberately broken adapter.

        Args:
            trace (tuple[int, ...]): Ignored construction trace.

        Returns:
            bool: False, exposing an invalid semantic contract.
        """
        return False

    def legal(self, trace: tuple[int, ...]) -> tuple[int, ...]:
        """Expose a nonterminal state with no legal actions.

        Args:
            trace (tuple[int, ...]): Ignored construction trace.

        Returns:
            tuple[int, ...]: Empty action set.
        """
        return ()


class ShortHorizonTour(Tour):
    """Model a task whose declared horizon is insufficient for its own construction.

    Responsibilities:
    - Exercise horizon exhaustion using the real decoding loop.
    """

    @property
    def horizon(self) -> int:
        """Return a deliberately insufficient decision count.

        Returns:
            int: One decision for a task requiring two.
        """
        return 1


def test_semantic_dead_end_has_an_explicit_failure_type() -> None:
    """An empty legal mask before termination is not an invalid-key or successful outcome."""
    task = DeadEndTour(np.array([[0.0, 0.0], [1.0, 0.0]]))
    with pytest.raises(codec.ConstructionFailure, match="legal"):
        codec.TrajectoryCodec(task).decode(np.zeros(1))


def test_insufficient_semantic_horizon_has_an_explicit_failure_type() -> None:
    """A completed key vector is not a completed solution when the horizon is wrong."""
    task = ShortHorizonTour(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(codec.HorizonExhausted):
        codec.TrajectoryCodec(task).decode(np.zeros(1))


@settings(max_examples=50, derandomize=True, deadline=None)
@given(st.integers(min_value=2, max_value=9), st.integers(min_value=0, max_value=100000))
def test_random_instances_replay_both_task_families(size: int, seed: int) -> None:
    """Generated tasks retain the exact decoded trace and independent cost after reconstruction.

    Args:
        size (int): City and column count.
        seed (int): Deterministic instance and key initialization.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.random((3, size)) < 0.3
    for row in matrix:
        if not row.any():
            row[int(rng.integers(size))] = True
    for task in (Tour(rng.random((size, 2))), Cover(matrix, rng.uniform(1.0, 10.0, size))):
        mapping = codec.TrajectoryCodec(task)
        candidate = mapping.decode(rng.random(task.horizon))
        replay = mapping.decode(mapping.encode(candidate.trace))
        assert replay == candidate
        assert replay.cost == task.check(candidate.solution)
