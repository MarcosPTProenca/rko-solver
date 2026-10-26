"""Verify library random draws do not reseed caller-owned global generators.

Objective:
- Detect transient global reseeding and contamination across concurrent solves.

Technical approach:
- Observe global states inside deterministic callbacks, not only after return.
- Run independent seeded solvers concurrently with overlapping callbacks.

Flow (pseudocode):
1. Snapshot caller Python and NumPy global generator states.
2. Execute seeded serial and thread portfolios with observable callbacks.
3. Require unchanged global streams throughout execution and on return.

Important criteria:
- Timed cooperative results need not reproduce complete search trajectories.
- Callback-owned randomness is not implicitly seeded by the solver.

Inputs/Outputs:
- Input: neutral objectives and seeded portfolios.
- Output: assertions on global state ownership and concurrent execution.
"""

import numpy as np
import pickle
import pytest
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from rko_solver import RKO
from rko_solver.orchestration.configuration import MH_SPECS


def global_state() -> bytes:
    """Serialize caller generator states without advancing either stream.

    Returns:
        bytes: Python and NumPy state fingerprint.
    """
    return pickle.dumps((random.getstate(), np.random.get_state()))


@pytest.mark.parametrize("execution", ["sequential", "thread"])
@pytest.mark.parametrize("algorithm", list(MH_SPECS))
@pytest.mark.parametrize("control", [0, 1])
def test_callbacks_never_observe_library_global_reseeding(
    execution: str, algorithm: str, control: int
) -> None:
    """Observe global isolation during every callback and after a seeded solve.

    Args:
        execution (str): In-process execution mode.
        algorithm (str): Registered search whose random draws must be isolated.
        control (int): Offline or adaptive control.
    """
    before = global_state()
    calls = 0

    def objective(keys: np.ndarray) -> float:
        """Check global ownership and evaluate a deterministic objective.

        Args:
            keys (np.ndarray): Candidate vector.

        Returns:
            float: Squared key norm.
        """
        nonlocal calls
        assert global_state() == before
        calls += 1
        return float(np.dot(keys, keys))

    result = RKO(
        lambda keys: keys,
        objective,
        64,
        algorithms=[algorithm],
        control=control,
        execution=execution,
        random_state=42,
    ).solve(0.02)
    assert calls > 0
    assert np.isfinite(result.cost)
    assert global_state() == before


def test_callback_global_draws_are_not_seeded_or_rolled_back() -> None:
    """Leave callback-owned global draws visible rather than resetting them on return."""
    saved_python, saved_numpy = random.getstate(), np.random.get_state()
    python = random.Random()
    python.setstate(saved_python)
    numpy = np.random.RandomState()
    numpy.set_state(saved_numpy)

    def objective(keys: np.ndarray) -> float:
        """Compare explicit global draws with independent caller-state clones.

        Args:
            keys (np.ndarray): Candidate vector.

        Returns:
            float: Squared key norm.
        """
        assert random.random() == python.random()
        assert np.random.random() == numpy.random()
        return float(np.dot(keys, keys))

    try:
        RKO(
            lambda keys: keys,
            objective,
            8,
            algorithms=["SA"],
            control=0,
            execution="sequential",
            random_state=42,
        ).solve(0.02)
        assert random.getstate() == python.getstate()
        assert pickle.dumps(np.random.get_state()) == pickle.dumps(numpy.get_state())
    finally:
        random.setstate(saved_python)
        np.random.set_state(saved_numpy)


def test_concurrent_serial_solves_preserve_global_state() -> None:
    """Force two differently seeded solves to overlap inside their first callback.

    Raises:
        AssertionError: If either solve contaminates caller generator states.
    """
    before = global_state()
    barrier = threading.Barrier(2, timeout=10)

    def solve(seed: int) -> tuple[float, ...]:
        """Run one independent search with a synchronized first evaluation.

        Args:
            seed (int): Independent portfolio seed.

        Returns:
            tuple[float, ...]: First random-key candidate.
        """
        initial: list[tuple[float, ...]] = []

        def objective(keys: np.ndarray) -> float:
            """Synchronize once and check generator isolation throughout execution.

            Args:
                keys (np.ndarray): Candidate vector.

            Returns:
                float: Squared key norm.
            """
            if not initial:
                initial.append(tuple(keys))
                barrier.wait()
            assert global_state() == before
            return float(np.dot(keys, keys))

        RKO(
            lambda keys: keys,
            objective,
            8,
            algorithms=["MultiStart"],
            control=0,
            execution="sequential",
            random_state=seed,
        ).solve(0.02)
        return initial[0]

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(executor.map(solve, [42, 43]))
    for seed, actual in zip([42, 43], [first, second], strict=True):
        child_seed = int(np.random.SeedSequence(seed).spawn(1)[0].generate_state(1)[0])
        assert actual == tuple(np.random.RandomState(child_seed).random_sample(8))
    assert global_state() == before
