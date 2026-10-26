"""Independent list-model oracles for IPR block-copy optimization.

Objective:
- Protect every evaluated chromosome, final incumbent and array ownership.

Technical approach:
- Model greedy block relinking with Python lists, independently of NumPy copying.
- Compare input/output bits and complete decoder traces at small/large block sizes.

Flow (pseudocode):
1. Build contiguous or strided numerical chromosomes with boundary values.
2. Run the list model and actual IPR with the same deterministic objective.
3. Compare every candidate and verify that neither input was modified or retained.

Important criteria:
- Include partial blocks, single-gene/vectorized boundaries and stable-order ties.
- Preserve negative-zero bits and the first candidate when objective costs tie.

Inputs/Outputs:
- Input: seeded keys for numerical and stable-permutation problems.
- Output: exact candidate-trace, fitness and ownership assertions.
"""

import math
import numpy as np
import pytest
import struct
from collections.abc import Callable
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.ipr import IPR


def _reference_path(
    current: list[float],
    guide: list[float],
    evaluate: Callable[[np.ndarray], float],
    *,
    skip_unchanged: bool = False,
) -> tuple[list[float], float]:
    """Model the greedy block path without NumPy indexing or assignment.

    Args:
        current (list[float]): Starting chromosome.
        guide (list[float]): Destination chromosome.
        evaluate (Callable): Deterministic candidate evaluator.
        skip_unchanged (bool): Reuse known current cost for representation-identical moves.

    Returns:
        tuple[list[float], float]: First best chromosome and its objective.
    """
    dimension = len(current)
    width = max(1, int(dimension * 0.10))
    count = (dimension + width - 1) // width
    remaining = list(range(count))
    current = current[:]
    best = current[:]
    best_cost = evaluate(np.asarray(current))
    current_cost = best_cost
    for _ in range(max(1, count - 1)):
        winner: list[float] | None = None
        winner_cost = math.inf
        winner_block = -1
        for block in remaining:
            start = block * width
            end = min(start + width, dimension)
            candidate = current[:]
            candidate[start:end] = guide[start:end]
            identical = all(
                struct.pack("!d", old) == struct.pack("!d", new)
                for old, new in zip(current, candidate, strict=True)
            )
            cost = current_cost if skip_unchanged and identical else evaluate(np.asarray(candidate))
            if cost < winner_cost:
                winner, winner_cost, winner_block = candidate, cost, block
        if winner is not None:
            if winner_cost < best_cost:
                best, best_cost = winner[:], winner_cost
            current = winner[:]
            current_cost = winner_cost
            remaining.remove(winner_block)
    return best, best_cost


@pytest.mark.parametrize("dimension", [1, 8, 20, 21])
@pytest.mark.parametrize("direction", [-1, 0, 1])
@pytest.mark.parametrize("changed", ["none", "first", "last", "both"])
def test_shared_blocks_preserve_legacy_path_and_pruned_trace(
    dimension: int, direction: int, changed: str
) -> None:
    """Preserve virtual steps and deterministic first-tied choices while pruning no-ops.

    Args:
        dimension (int): One-gene, full-block or partial-tail dimension.
        direction (int): Make guide changes worse, tied or better.
        changed (str): Differing block locations, including an entirely identical pair.
    """
    current = np.full(dimension, 0.8)
    guide = current.copy()
    width = max(1, int(dimension * 0.1))
    if changed in ("first", "both"):
        guide[:width] = 0.2
    if changed in ("last", "both"):
        guide[((dimension - 1) // width) * width :] = 0.2
    current.flags.writeable = guide.flags.writeable = False
    expected_trace: list[bytes] = []
    observed: list[bytes] = []

    def score(keys: np.ndarray) -> float:
        """Score a deterministic numerical/tied objective without callback side effects.

        Args:
            keys (np.ndarray): Path chromosome.

        Returns:
            float: Signed sum of squares.
        """
        return direction * sum(float(key) * float(key) for key in keys)

    def reference_score(keys: np.ndarray) -> float:
        """Record independent list-oracle callbacks.

        Args:
            keys (np.ndarray): Oracle candidate.

        Returns:
            float: Deterministic score.
        """
        expected_trace.append(keys.tobytes())
        return score(keys)

    def actual_score(keys: np.ndarray) -> float:
        """Record production callback representations.

        Args:
            keys (np.ndarray): Production candidate.

        Returns:
            float: Deterministic score.
        """
        observed.append(keys.tobytes())
        return score(keys)

    expected, expected_cost = _reference_path(
        current.tolist(), guide.tolist(), reference_score, skip_unchanged=True
    )
    legacy, legacy_cost = _reference_path(current.tolist(), guide.tolist(), score)
    algorithm = IPR(SolutionPool(), lambda keys: keys, actual_score, dimension)
    actual, cost = algorithm._relink(current, guide)
    assert observed == expected_trace
    assert actual.tobytes() == np.asarray(expected).tobytes() == np.asarray(legacy).tobytes()
    assert cost == expected_cost == legacy_cost
    assert not np.shares_memory(actual, current) and not np.shares_memory(actual, guide)


def test_signed_zero_change_is_not_pruned() -> None:
    """Keep callbacks for representation changes a deterministic decoder can distinguish."""
    observed: list[bool] = []

    def score(keys: np.ndarray) -> float:
        """Detect the sign of a finite zero-valued coordinate.

        Args:
            keys (np.ndarray): Single-gene path chromosome.

        Returns:
            float: One for negative zero, zero for positive zero.
        """
        negative = bool(np.signbit(keys[0]))
        observed.append(negative)
        return float(negative)

    algorithm = IPR(SolutionPool(), lambda keys: keys, score, 1)
    algorithm._relink(np.array([0.0]), np.array([-0.0]))
    assert observed == [False, True]


def test_mixed_dtype_no_op_is_conservatively_evaluated() -> None:
    """Avoid assuming identical representations when guide and current dtypes differ."""
    calls: list[bytes] = []

    def score(keys: np.ndarray) -> float:
        """Record effective candidates at a mixed-dtype private kernel boundary.

        Args:
            keys (np.ndarray): Candidate retaining current dtype.

        Returns:
            float: Constant finite cost.
        """
        calls.append(keys.tobytes())
        return 0.0

    algorithm = IPR(SolutionPool(), lambda keys: keys, score, 1)
    algorithm._relink(np.array([0.5], dtype=np.float32), np.array([0.5], dtype=np.float64))
    assert len(calls) == 2


def test_changed_block_propagates_callback_failure() -> None:
    """Keep real evaluation errors intact and never mutate caller parents."""
    calls = 0
    failure = ArithmeticError("changed block failed")

    def score(keys: np.ndarray) -> float:
        """Fail the first changed candidate after a valid initial observation.

        Args:
            keys (np.ndarray): Actual path chromosome.

        Returns:
            float: Finite initial score.

        Raises:
            ArithmeticError: On the second actual evaluation.
        """
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        return float(np.dot(keys, keys))

    start = np.full(20, 0.8)
    guide = start.copy()
    guide[:2] = 0.2
    original_start, original_guide = start.copy(), guide.copy()
    algorithm = IPR(SolutionPool(), lambda keys: keys, score, 20)
    with pytest.raises(ArithmeticError) as caught:
        algorithm._relink(start, guide)
    assert caught.value is failure and calls == 2
    assert np.array_equal(start, original_start) and np.array_equal(guide, original_guide)


def test_identical_parent_cost_is_evaluated_once() -> None:
    """Make the intentional callback-count change explicit for stateful objectives."""
    observations: list[np.ndarray] = []

    def score(keys: np.ndarray) -> float:
        """Return a different finite observation at each actual callback.

        Args:
            keys (np.ndarray): Evaluated chromosome.

        Returns:
            float: Callback ordinal, deliberately stateful.
        """
        observations.append(keys.copy())
        return float(len(observations))

    keys = np.full(20, 0.5)
    algorithm = IPR(SolutionPool(), lambda keys: keys, score, 20)
    result, value = algorithm._relink(keys, keys)
    assert len(observations) == 1 and value == 1.0
    assert np.array_equal(result, keys)


@pytest.mark.parametrize("dimension", [1, 8, 19, 20, 32, 79, 80, 81, 101, 1000])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("ordering", [False, True])
def test_relink_matches_independent_candidate_trace(
    dimension: int, strided: bool, ordering: bool
) -> None:
    """Match all candidates and ownership contracts, not only the winning cost.

    Args:
        dimension (int): Positive chromosome length, including partial-block boundaries.
        strided (bool): Exercise noncontiguous and negative-stride source arrays.
        ordering (bool): Decode into a stable permutation instead of numerical values.
    """
    generator = np.random.RandomState(42)
    current = generator.random_sample(2 * dimension)[::2]
    guide = generator.random_sample(2 * dimension)[::-2]
    if not strided:
        current, guide = current.copy(), guide.copy()
    current[0] = -0.0
    guide[0] = np.nextafter(1.0, 0.0)
    if dimension > 1:
        current[-1], guide[-1] = 0.0, -0.0
    saved_current, saved_guide = current.copy(), guide.copy()
    weights = np.arange(1, dimension + 1)
    reference_trace: list[np.ndarray] = []
    actual_trace: list[np.ndarray] = []

    def decode(keys: np.ndarray) -> np.ndarray:
        """Apply the selected representation.

        Args:
            keys (np.ndarray): Numerical candidate keys.

        Returns:
            np.ndarray: Stable permutation or unchanged keys.
        """
        return np.argsort(keys, kind="stable") if ordering else keys

    def objective(values: np.ndarray) -> float:
        """Compute the numerical or permutation objective.

        Args:
            values (np.ndarray): Decoded candidate.

        Returns:
            float: Finite deterministic minimization value.
        """
        return float(np.dot(values, weights if ordering else values))

    def reference_evaluate(keys: np.ndarray) -> float:
        """Record model candidate bits before evaluating them.

        Args:
            keys (np.ndarray): Independently assembled model chromosome.

        Returns:
            float: Decoded objective value.
        """
        reference_trace.append(keys.view(np.uint64).copy())
        return objective(decode(keys))

    def actual_decode(keys: np.ndarray) -> np.ndarray:
        """Record production candidate bits before applying the same representation.

        Args:
            keys (np.ndarray): IPR candidate chromosome.

        Returns:
            np.ndarray: Decoded candidate.
        """
        actual_trace.append(keys.view(np.uint64).copy())
        return decode(keys)

    expected_keys, expected_cost = _reference_path(
        current.tolist(), guide.tolist(), reference_evaluate, skip_unchanged=True
    )
    algorithm = IPR(SolutionPool(), actual_decode, objective, dimension)
    keys, cost = algorithm._relink(current, guide)
    assert cost == expected_cost
    np.testing.assert_array_equal(keys.view(np.uint64), np.asarray(expected_keys).view(np.uint64))
    assert len(actual_trace) == len(reference_trace)
    for actual, expected in zip(actual_trace, reference_trace, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(current.view(np.uint64), saved_current.view(np.uint64))
    np.testing.assert_array_equal(guide.view(np.uint64), saved_guide.view(np.uint64))
    assert not np.shares_memory(keys, current)
    assert not np.shares_memory(keys, guide)
