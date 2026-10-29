"""Verify initial-key validation, pool seeding and evaluation accounting.

Objective:
- Prove that caller-supplied keys reach the shared pool before search work across execution engines.

Technical approach:
- Exercise the public `RKO` API with deterministic callbacks and one-evaluation searches.
- Check copied inputs, early validation, startup ordering and evaluation accounting.

Flow (pseudocode):
1. Construct a solver with known seed keys and mutate the caller-owned input.
2. Run Python, process, Rust and C++ routes with bounded search work.
3. Compare the returned incumbent, callback order and reported evaluation counts.

Important criteria:
- Every supplied row is evaluated exactly once, including duplicate rows.
- Invalid dimensions, non-finite values and out-of-range keys fail before any callback.
- Seeded-key evaluations are reported separately and included in `RKOResult.n_evaluations`.

Inputs/Outputs:
- Input: finite key matrices, deterministic callbacks and optional native engines.
- Output: public result and callback assertions for seeded search behavior.
"""

import multiprocessing as mp
import numpy as np
import pytest
import time
from rko_solver import RKO

native = pytest.importorskip("rko_solver._native")


def decode(keys: np.ndarray) -> np.ndarray:
    """Return the candidate key vector as its decoded solution.

    Args:
        keys (np.ndarray): Candidate keys supplied by the solver.

    Returns:
        np.ndarray: The unchanged key vector.
    """
    return keys


def seeded_cost(solution: np.ndarray) -> float:
    """Give the designated warm-start vector the unique best objective.

    Args:
        solution (np.ndarray): Candidate vector to score.

    Returns:
        float: Zero for the designated vector and one otherwise.
    """
    return 0.0 if np.array_equal(solution, np.array([0.25, 0.75])) else 1.0


def process_seeded_cost(solution: np.ndarray) -> float:
    """Reject parent-process evaluation and score the designated warm-start vector.

    Args:
        solution (np.ndarray): Decoded candidate vector.

    Returns:
        float: Zero for the designated vector and one otherwise.

    Raises:
        RuntimeError: If the callback runs in the coordinating process.
    """
    if mp.current_process().name == "MainProcess":
        raise RuntimeError("process callbacks must not run in the parent")
    return seeded_cost(solution)


def test_python_solver_copies_keys_seeds_before_search_and_reports_evaluations() -> None:
    """Use the original key snapshot before search and account for its evaluation.

    Returns:
        None: Asserts public result and callback behavior.
    """
    source = [[0.25, 0.75]]
    seen: list[tuple[float, ...]] = []

    def objective(solution: np.ndarray) -> float:
        """Record and score each objective candidate.

        Args:
            solution (np.ndarray): Decoded candidate.

        Returns:
            float: Objective value.
        """
        seen.append(tuple(float(value) for value in solution))
        return seeded_cost(solution)

    solver = RKO(
        decode,
        objective,
        2,
        algorithms=["SA"],
        adaptation="fixed",
        execution="sequential",
        max_evaluations=1,
        random_state=7,
        initial_keys=source,
    )
    source[0][0] = 0.9
    result = solver.solve(0.2)

    assert result.cost == 0.0
    assert result.keys == (0.25, 0.75)
    assert seen[0] == (0.25, 0.75)
    assert result.initial_keys is not None
    assert result.initial_keys.n_keys == result.initial_keys.n_evaluations == 1
    assert result.n_evaluations == result.searches[0].n_evaluations + 1


@pytest.mark.parametrize(
    "keys",
    [
        [[0.1]],
        [[0.1, 0.2, 0.3]],
        [[float("nan"), 0.2]],
        [[float("inf"), 0.2]],
        [[-0.01, 0.2]],
        [[0.1, 1.0]],
        [[], [0.1, 0.2]],
    ],
)
def test_invalid_initial_keys_fail_before_callbacks(keys: list[list[float]]) -> None:
    """Reject malformed key matrices before evaluating the problem.

    Args:
        keys (list[list[float]]): Invalid matrix to validate.

    Returns:
        None: Asserts validation failure and zero callback invocations.
    """
    calls = 0

    def objective(solution: np.ndarray) -> float:
        """Count objective calls during validation.

        Args:
            solution (np.ndarray): Candidate solution.

        Returns:
            float: Constant objective value.
        """
        nonlocal calls
        calls += 1
        return 1.0

    solver = RKO(
        decode,
        objective,
        2,
        algorithms=["SA"],
        execution="sequential",
        initial_keys=keys,
    )

    with pytest.raises(ValueError, match="initial_keys"):
        solver.solve(0.2)
    assert calls == 0


@pytest.mark.parametrize("engine", ["rust", "cpp"])
def test_native_solver_keeps_initial_candidate_and_accounts_for_it(engine: str) -> None:
    """Return the seeded incumbent from each compiled engine with exact accounting.

    Args:
        engine (str): Compiled engine to exercise.

    Returns:
        None: Asserts incumbent preservation and evaluation metadata.
    """
    if engine == "cpp" and not native.cpp_available():
        pytest.skip("C++ engine is not built")
    result = RKO(
        decode,
        seeded_cost,
        2,
        algorithms=["SA"],
        adaptation="fixed",
        engine=engine,
        max_evaluations=1,
        random_state=7,
        initial_keys=[[0.25, 0.75]],
    ).solve(0.2)

    assert result.cost == 0.0
    assert result.keys == (0.25, 0.75)
    assert result.initial_keys is not None
    assert result.initial_keys.n_keys == result.initial_keys.n_evaluations == 1
    initialization_evaluations = (
        result.initialization.n_evaluations if result.initialization is not None else 0
    )
    assert result.n_evaluations == result.searches[0].n_evaluations + initialization_evaluations + 1


def test_process_solver_seeds_before_workers_and_reports_evaluations() -> None:
    """Make the seeded incumbent available to spawned workers without parent callbacks.

    Returns:
        None: Asserts incumbent preservation and total evaluation metadata.
    """
    result = RKO(
        decode,
        process_seeded_cost,
        2,
        algorithms=["SA"],
        adaptation="fixed",
        execution="process",
        max_evaluations=1,
        random_state=7,
        initial_keys=[[0.25, 0.75]],
    ).solve(0.2)

    assert result.cost == 0.0
    assert result.keys == (0.25, 0.75)
    assert result.initial_keys is not None
    assert result.initial_keys.n_evaluations == 1
    assert result.n_evaluations == result.searches[0].n_evaluations + 1


def test_initial_keys_and_random_prefill_have_separate_counts() -> None:
    """Count duplicate supplied rows once each without folding them into prefill metadata.

    Returns:
        None: Asserts separate and aggregate startup accounting.
    """
    result = RKO(
        decode,
        seeded_cost,
        2,
        algorithms=["SA"],
        adaptation="fixed",
        execution="sequential",
        pool_size=3,
        prefill_pool=True,
        max_evaluations=1,
        random_state=7,
        initial_keys=[[0.25, 0.75], [0.25, 0.75]],
    ).solve(0.2)

    assert result.initial_keys is not None
    assert result.initial_keys.n_keys == result.initial_keys.n_evaluations == 2
    assert result.initialization is not None
    assert result.initialization.n_evaluations == 5
    assert result.n_evaluations == result.searches[0].n_evaluations + 7


def test_auto_calibration_does_not_repeat_seed_key_evaluations() -> None:
    """Evaluate supplied keys only in the final auto-selected run, not calibration probes.

    Returns:
        None: Asserts the exact seed vector is evaluated once across the complete solve.
    """
    target = (0.25, 0.75)
    seen: list[tuple[float, ...]] = []

    def objective(solution: np.ndarray) -> float:
        """Record objective candidates and score the exact warm-start vector.

        Args:
            solution (np.ndarray): Decoded candidate.

        Returns:
            float: Zero for the warm-start vector and one otherwise.
        """
        seen.append(tuple(float(value) for value in solution))
        return 0.0 if tuple(solution) == target else 1.0

    result = RKO(
        decode,
        objective,
        2,
        algorithms=["SA"],
        adaptation="fixed",
        engine="auto",
        auto_candidates=["rust"],
        auto_time=0.2,
        max_evaluations=1,
        random_state=11,
        initial_keys=[target],
    ).solve(0.2)

    assert result.initial_keys is not None
    assert result.initial_keys.n_evaluations == 1
    assert seen.count(target) == 1


def test_native_seed_startup_does_not_consume_search_time_limit() -> None:
    """Give the Rust search its full budget after a slow caller-supplied seed.

    Returns:
        None: Asserts both allotted search evaluations complete after startup delay.
    """
    target = np.array([0.25, 0.75])

    def objective(solution: np.ndarray) -> float:
        """Delay startup and search evaluations to distinguish their budgets.

        Args:
            solution (np.ndarray): Candidate vector.

        Returns:
            float: Constant objective value.
        """
        time.sleep(0.25 if np.array_equal(solution, target) else 0.04)
        return 0.0

    result = RKO(
        decode,
        objective,
        2,
        algorithms=["SA"],
        adaptation="fixed",
        engine="rust",
        max_evaluations=2,
        random_state=7,
        initial_keys=[target],
    ).solve(0.2)

    assert result.initial_keys is not None
    assert result.searches[0].n_evaluations == 2
    assert result.n_evaluations == 3


def test_empty_initial_keys_are_a_no_op() -> None:
    """Treat an empty matrix like no warm start and omit its metadata.

    Returns:
        None: Asserts absence of seed metadata.
    """
    result = RKO(
        decode,
        seeded_cost,
        2,
        algorithms=["SA"],
        execution="sequential",
        max_evaluations=1,
        initial_keys=[],
    ).solve(0.2)

    assert result.initial_keys is None
    assert result.n_evaluations == result.searches[0].n_evaluations
