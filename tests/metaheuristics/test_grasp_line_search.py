"""Check GRASP's sampled lattice against independent bounded reference models.

Objective:
- Restore broad candidate coverage without materializing a dense reciprocal-step grid.

Technical approach:
- Reconstruct the pinned C++ lattice literally for small dyadic examples.
- Compare lazy sampling against a materialized swap-removal oracle using matched draws.
- Exercise tiny steps, caller ownership and the existing callback stopping boundary.

Flow (pseudocode):
1. Enumerate a small reference lattice and independently sample its slots.
2. Observe actual callback candidates, count, result and owned generator state.
3. Probe endpoint, precision, allocation, deadline and failure cases separately.

Important criteria:
- Sampling is without replacement over lattice positions, not decoded outcomes.
- The incumbent may be sampled out; tied scores retain the first sampled minimum.
- Small-step checks must finish with bounded callbacks/memory, including subnormals.

Inputs/Outputs:
- Input: finite random keys, positive finite steps and deterministic owned RNGs.
- Output: candidate-coverage, callback-budget, ownership and failure assertions.
"""

import importlib
import itertools
import math
import numpy as np
import pickle
import pytest
import random
import tracemalloc
from collections import Counter
from fractions import Fraction
from pathlib import Path
from rko_solver.components.random_streams import RandomStreams
from rko_solver.components.solution_pool import ElitePool
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.grasp import GRASP
from rko_solver.metaheuristics.multi_start import MultiStart
from rko_solver.orchestration.search import SearchSession
from rko_solver.orchestration.search import _BudgetExpired
from rko_solver.orchestration.types import CostFn
from rko_solver.orchestration.types import DecoderFn
from unittest.mock import Mock


def decode(keys: np.ndarray) -> np.ndarray:
    """Keep numerical chromosomes unchanged.

    Args:
        keys (np.ndarray): Detached candidate.

    Returns:
        np.ndarray: Original payload.
    """
    return keys


def quadratic(keys: np.ndarray) -> float:
    """Score the variable coordinate relative to a fixed numerical target.

    Args:
        keys (np.ndarray): Three-coordinate candidate.

    Returns:
        float: Finite squared distance of the middle coordinate from0.375.
    """
    return float((keys[1] - 0.375) ** 2)


def reference_lattice(z: float, h: float) -> list[float]:
    """Enumerate the literal pinned reference loop for bounded dyadic steps.

    Args:
        z (float): Incumbent coordinate, including the compatible endpoint1.
        h (float): Positive dyadic step with a small finite reciprocal.

    Returns:
        list[float]: Incumbent followed by eligible interleaved positive/negative shifts.
    """
    values = [z]
    tau = 0
    for _ in range(0, int(1.0 / h) + 1, 2):
        tau += 1
        for sign in (1, -1):
            value = z + sign * tau * h
            if 0 <= value < 1:
                values.append(value)
    return values


@pytest.mark.parametrize("h", [1.0, 0.5, 0.25, 0.125, 1 / 1024])
@pytest.mark.parametrize("z", [0.0, 0.125, 0.375, 0.875, math.nextafter(1.0, 0.0), 1.0])
@pytest.mark.parametrize("seed", [7, 42])
def test_sampled_candidates_match_reference_slots(h: float, z: float, seed: int) -> None:
    """Compare callback order, minimum and complete private RNG consumption.

    Args:
        h (float): Small dyadic grid step.
        z (float): Incumbent coordinate.
        seed (int): Matched generator seed for production and materialized oracle.
    """
    eligible = reference_lattice(z, h)
    count = min(len(eligible), math.ceil(math.log2(int(1 / h))) + 1)
    generator = random.Random(seed)
    remaining = eligible.copy()
    expected = []
    for _ in range(count):
        index = generator.randrange(len(remaining))
        expected.append(remaining[index])
        remaining[index] = remaining[-1]
        remaining.pop()
    observed: list[float] = []

    def observe(keys: np.ndarray) -> float:
        """Record the actual selected coordinate before evaluating its norm.

        Args:
            keys (np.ndarray): Sampled candidate.

        Returns:
            float: Quadratic objective.
        """
        observed.append(float(keys[1]))
        assert keys[0] == 0.25 and keys[2] == 0.75
        return quadratic(keys)

    keys = np.array([0.25, z, 0.75])
    original = keys.copy()
    algorithm = GRASP(SolutionPool(), decode, observe, 3)
    globals_before = pickle.dumps((random.getstate(), np.random.get_state()))
    streams = RandomStreams(seed)
    with streams.activate():
        value, score = algorithm._line_search(1, h, keys)
    assert observed == expected
    best = min(expected, key=lambda value: (value - 0.375) ** 2)
    assert (value, score) == (best, (best - 0.375) ** 2)
    assert streams.python.getstate() == generator.getstate()
    assert pickle.dumps((random.getstate(), np.random.get_state())) == globals_before
    assert np.array_equal(keys, original)


def test_fine_grid_can_reach_a_distant_reference_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reach0.375 from0.875 where the old logarithmic-radius neighborhood cannot.

    Args:
        monkeypatch (pytest.MonkeyPatch): Choose a known eligible rank deterministically.
    """
    observed: list[float] = []

    def observe(keys: np.ndarray) -> float:
        """Retain sampled coordinates and score distance from the distant target.

        Args:
            keys (np.ndarray): Sampled candidate.

        Returns:
            float: Squared target distance.
        """
        observed.append(float(keys[1]))
        return quadratic(keys)

    algorithm = GRASP(SolutionPool(), decode, observe, 3)
    streams = RandomStreams(42)
    monkeypatch.setattr(streams.python, "randrange", lambda stop: 0)
    with streams.activate():
        value, score = algorithm._line_search(1, 1 / 1024, np.array([0.25, 0.875, 0.75]))
    assert len(observed) == 11
    assert 0.375 in observed
    assert (value, score) == (0.375, 0.0)


def test_incumbent_can_be_omitted_and_construction_can_worsen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not silently force a greedy incumbent into the randomized subset.

    Args:
        monkeypatch (pytest.MonkeyPatch): Select the three non-incumbent lattice positions.
    """
    observed: list[float] = []

    def observe(keys: np.ndarray) -> float:
        """Score sampled coordinates while recording incumbent exclusion.

        Args:
            keys (np.ndarray): Sampled candidate.

        Returns:
            float: Squared target distance.
        """
        observed.append(float(keys[1]))
        return quadratic(keys)

    algorithm = GRASP(SolutionPool(), decode, observe, 3)
    streams = RandomStreams(42)
    monkeypatch.setattr(streams.python, "randrange", lambda stop: 1)
    with streams.activate():
        value, score = algorithm._line_search(1, 0.25, np.array([0.25, 0.375, 0.75]))
    assert observed == [0.625, 0.875, 0.125]
    assert (value, score) == (0.625, 0.0625)


def test_solver_retains_global_best_after_worse_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow exploratory construction without losing the already known incumbent.

    Args:
        monkeypatch (pytest.MonkeyPatch): Force non-incumbent samples and one construction.
    """
    refinements: list[float] = []

    def objective(keys: np.ndarray) -> float:
        """Score the controlled initial optimum and deliberately worse constructions.

        Args:
            keys (np.ndarray): Numerical candidate.

        Returns:
            float: Squared distance from the all0.375 optimum.
        """
        return float(np.sum((keys - 0.375) ** 2))

    def refine(
        keys: np.ndarray, decoder: DecoderFn, cost: CostFn, *, pool: ElitePool | None = None
    ) -> tuple[np.ndarray, float]:
        """Return a deliberately worse point and mark completed refinement calls.

        Args:
            keys (np.ndarray): Constructed point.
            decoder (DecoderFn): Ordinary decoder.
            cost (CostFn): Numerical objective.
            pool (ElitePool | None): Shared context.

        Returns:
            tuple[np.ndarray, float]: Shifted candidate and its worse cost.
        """
        worse = np.clip(keys + 0.25, 0.0, 1.0)
        value = cost(decoder(worse))
        refinements.append(value)
        return worse, value

    streams = RandomStreams(42)
    generator = Mock(wraps=streams.numpy)
    generator.random.side_effect = lambda size: np.full(size, 0.375)
    monkeypatch.setattr(streams, "numpy", generator)
    monkeypatch.setattr(streams.python, "randrange", lambda stop: 1)
    monkeypatch.setattr(streams.python, "uniform", lambda lower, upper: 0.5)
    monkeypatch.setattr(streams.python, "shuffle", lambda values: None)
    algorithm = GRASP(SolutionPool(), decode, objective, 4, local_search_operator=refine)
    monkeypatch.setattr(algorithm, "time_limit_reached", lambda duration: len(refinements) >= 2)
    with streams.activate():
        result = algorithm.solve(60.0)
    assert len(refinements) == 2 and all(value > 0 for value in refinements)
    assert result[1:] == ([0.375] * 4, 0.0)


def test_all_ordered_subsets_have_one_draw_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exhaust the small sampler rather than rely on a flaky statistical uniformity test.

    Args:
        monkeypatch (pytest.MonkeyPatch): Enumerate every admissible sequence of draws.
    """
    frequencies: Counter[tuple[float, ...]] = Counter()
    for choices in itertools.product(range(4), range(3), range(2)):
        observed: list[float] = []

        def observe(keys: np.ndarray, trace: list[float] = observed) -> float:
            """Retain one sampled position and return a tied cost.

            Args:
                keys (np.ndarray): Sampled candidate.
                trace (list[float]): Observation list bound to this draw sequence.

            Returns:
                float: Constant objective to test first-minimum retention.
            """
            trace.append(float(keys[1]))
            return 0.0

        draws = iter(choices)
        streams = RandomStreams(42)
        monkeypatch.setattr(streams.python, "randrange", lambda stop, draws=draws: next(draws))
        with streams.activate():
            value, score = GRASP(SolutionPool(), decode, observe, 3)._line_search(
                1, 0.25, np.array([0.25, 0.5, 0.75])
            )
        assert len(observed) == len(set(observed)) == 3
        assert (value, score) == (observed[0], 0.0)
        frequencies[tuple(observed)] += 1
    assert len(frequencies) == 24
    assert set(frequencies.values()) == {1}
    assert set(frequencies) == set(itertools.permutations(reference_lattice(0.5, 0.25), 3))


@pytest.mark.parametrize("h", [0.1, 1 / 7, 1e-100, 1e-300, math.ulp(0.0), 2.0])
def test_extreme_and_decimal_steps_stay_bounded(h: float) -> None:
    """Use exact-ratio counts without reciprocal overflow or dense lattice storage.

    Args:
        h (float): Decimal, tiny/subnormal or larger-than-unit step.
    """
    count = 0

    def observe(keys: np.ndarray) -> float:
        """Check finite half-open shifted values without retaining large traces.

        Args:
            keys (np.ndarray): Sampled candidate.

        Returns:
            float: Finite numerical cost.
        """
        nonlocal count
        count += 1
        assert 0 <= keys[1] < 1
        return quadratic(keys)

    algorithm = GRASP(SolutionPool(), decode, observe, 3)
    step = Fraction(h)
    divisions = max(1, step.denominator // step.numerator)
    q = (divisions - 1).bit_length() + 1
    radius = divisions // 2 + 1
    positive = min(radius, max(0, math.ceil((Fraction(1) - Fraction(0.5)) / step) - 1))
    negative = min(radius, math.floor(Fraction(0.5) / step))
    expected_count = min(q, 1 + positive + negative)
    with RandomStreams(42).activate():
        tracemalloc.start()
        try:
            value, score = algorithm._line_search(1, h, np.array([0.25, 0.5, 0.75]))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    assert count == expected_count
    assert 1 <= count <= q <= 1075
    assert peak < 2_000_000
    assert math.isfinite(value) and score == (value - 0.375) ** 2


@pytest.mark.parametrize("h", [0.0, -0.1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_steps_fail_before_callbacks(h: float) -> None:
    """Reject invalid private-kernel steps before evaluating user code.

    Args:
        h (float): Deliberately invalid step, including nonnumeric/coercible values.
    """
    observe = Mock(side_effect=quadratic)
    streams = RandomStreams(42)
    with streams.activate(), pytest.raises(ValueError):
        GRASP(SolutionPool(), decode, observe, 3)._line_search(1, h, np.array([0.25, 0.5, 0.75]))
    observe.assert_not_called()
    assert streams.python.getstate() == random.Random(42).getstate()


@pytest.mark.parametrize("z", [-0.1, 1.01, float("nan"), float("inf")])
def test_invalid_coordinates_fail_before_sampling(z: float) -> None:
    """Reject an invalid varying coordinate before RNG or callback side effects.

    Args:
        z (float): Nonfinite or out-of-domain coordinate.
    """
    observe = Mock(side_effect=quadratic)
    streams = RandomStreams(42)
    with streams.activate(), pytest.raises(ValueError):
        GRASP(SolutionPool(), decode, observe, 3)._line_search(1, 0.125, np.array([0.25, z, 0.75]))
    observe.assert_not_called()
    assert streams.python.getstate() == random.Random(42).getstate()


def test_tiny_shift_rounding_does_not_produce_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round an exact valid shift inward if binary64 conversion reaches the upper bound.

    Args:
        monkeypatch (pytest.MonkeyPatch): Select the last positive subnormal-grid shift.
    """
    h = math.ulp(0.0)
    z = math.nextafter(1.0, 0.0)
    positive = math.ceil((Fraction(1) - Fraction(z)) / Fraction(h)) - 1
    streams = RandomStreams(42)
    monkeypatch.setattr(streams.python, "randrange", lambda stop: 2 * positive - 1)
    observed: list[float] = []

    def stop(keys: np.ndarray) -> float:
        """Check the selected coordinate and stop after the first callback.

        Args:
            keys (np.ndarray): Candidate from the final positive lattice position.

        Returns:
            float: Never returns.

        Raises:
            ArithmeticError: Always, bounding this precision-only probe.
        """
        observed.append(float(keys[1]))
        raise ArithmeticError("precision checked")

    with streams.activate(), pytest.raises(ArithmeticError, match="precision checked"):
        GRASP(SolutionPool(), decode, stop, 3)._line_search(1, h, np.array([0.25, z, 0.75]))
    assert observed == [math.nextafter(1.0, 0.0)]


def test_rng_replay_detects_intentional_grid_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid diagnosing changed search semantics as a random-stream regression.

    Args:
        monkeypatch (pytest.MonkeyPatch): Expose the maintained replay helper.
    """

    class Historical:
        """Represent an old adaptive constructor without the corrected lattice contract.

        Responsibilities:
        - Supply the legacy control signature without the new internal comparison token.
        """

        def __init__(self, control: int = 0) -> None:
            """Accept an unused flag without running any algorithm.

            Args:
                control (int): Simulated legacy adaptive flag.
            """

    class LatticeOnly(GRASP):
        """Advertise the earlier lattice correction without corrected construction counts.

        Responsibilities:
        - Model the preceding full-search replay contract, not its execution body.

        Attributes:
            _replay_contract (str): Earlier diagnostic revision marker.
        """

        _replay_contract: str = "grasp-sampled-lattice-v1"

    class InPlace(GRASP):
        """Represent the prior grid-aligned constructor with in-place subset shuffling.

        Responsibilities:
        - Supply the preceding replay identity without executing historical code.

        Attributes:
            _replay_contract (str): Previous diagnostic revision marker.
        """

        _replay_contract: str = "grasp-construction-v3"

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    replay = importlib.import_module("benchmarks.algorithms.random_streams")
    assert replay.replay_modes(GRASP, Historical) == ()
    assert replay.replay_modes(GRASP, LatticeOnly) == ()
    assert replay.replay_modes(GRASP, InPlace) == ()
    assert replay.replay_modes(GRASP, GRASP) == (0, 1)
    assert replay.replay_modes(MultiStart, MultiStart) == (0,)


def test_callback_arrays_are_detached_from_input_and_each_other() -> None:
    """Isolate readonly strided float32 input from retained or mutating callbacks."""
    backing = np.array([0.25, 9.0, 0.5, 9.0, 0.75, 9.0], dtype=np.float32)
    keys = backing[::2]
    original = backing.copy()
    keys.flags.writeable = False
    retained: list[np.ndarray] = []

    def observe(candidate: np.ndarray) -> float:
        """Mutate a non-target coordinate only after checking the original context.

        Args:
            candidate (np.ndarray): Private sampled candidate.

        Returns:
            float: Objective independent of the deliberately mutated coordinate.
        """
        assert candidate[0] == 0.25 and candidate[2] == 0.75
        retained.append(candidate)
        candidate[0] = 99.0
        return quadratic(candidate)

    with RandomStreams(42).activate():
        value, score = GRASP(SolutionPool(), decode, observe, 3)._line_search(1, 0.125, keys)
    assert len({id(candidate) for candidate in retained}) == len(retained)
    assert not any(np.shares_memory(candidate, backing) for candidate in retained)
    assert np.array_equal(backing, original)
    assert score == (value - 0.375) ** 2


def test_callback_deadline_stops_between_samples() -> None:
    """Retain one finite evaluation rather than sampling past the session deadline."""
    session = SearchSession(decode, quadratic)
    session.deadline = -1.0
    algorithm = GRASP(SolutionPool(), session.decode, session.evaluate, 3)
    keys = np.array([0.25, 0.5, 0.75])
    original = keys.copy()
    with RandomStreams(42).activate(), pytest.raises(_BudgetExpired):
        algorithm._line_search(1, 1 / 1024, keys)
    assert session.n_evaluations == 1 and session.best is not None
    assert np.array_equal(keys, original)


def test_callback_failure_preserves_caller_keys() -> None:
    """Propagate objective errors without leaking candidate mutation to the input."""
    keys = np.array([0.25, 0.5, 0.75])
    original = keys.copy()

    def fail(candidate: np.ndarray) -> float:
        """Mutate the private sample and fail deliberately.

        Args:
            candidate (np.ndarray): Private sampled vector.

        Returns:
            float: Never returns.

        Raises:
            ArithmeticError: Always, to exercise the original failure boundary.
        """
        candidate[:] = 99
        raise ArithmeticError("line objective failed")

    with RandomStreams(42).activate(), pytest.raises(ArithmeticError, match="line objective"):
        GRASP(SolutionPool(), decode, fail, 3)._line_search(1, 0.125, keys)
    assert np.array_equal(keys, original)
