"""Public deadline and worker-failure regression contracts.

Objective:
- Stop returning-callback local searches without losing their completed incumbent.

Technical approach:
- Run the previously overrunning ILS shape through every public execution mode.
- Observe bounded callback counts independently of spawn/import latency.
- Inject a BaseException in a thread callback and observe caller-visible failure.

Flow (pseudocode):
1. Define importable returning and failing callbacks.
2. Execute a bounded single-algorithm portfolio.
3. Check valid results, elapsed time or preserved failure causes.

Important criteria:
- Process startup is not part of the requested search budget.
- No claim of safely killing arbitrarily blocked Python threads is made.

Inputs/Outputs:
- Input: short search budget and dimension100 constant-cost problem.
- Output: completion, cost consistency and exception propagation assertions.
"""

import numpy as np
import pytest
import time
from dataclasses import dataclass
from pathlib import Path
from rko_solver import RKO
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.multi_start import MultiStart
from rko_solver.orchestration.search import SearchSession


def decode(keys: np.ndarray) -> np.ndarray:
    """Return the random-key vector.

    Args:
        keys (np.ndarray): Candidate chromosome.

    Returns:
        np.ndarray: Unmodified chromosome.
    """
    return keys


@dataclass
class RecordingCost:
    """Observe a returning objective without sharing mutable process state.

    Responsibilities:
    - Persist completed evaluation counts and fail quickly on budget regressions.

    Attributes:
        path (Path): Trial-owned byte-per-evaluation observation file.
        calls (int): Evaluations attempted by this callback instance.
    """

    path: Path
    calls: int = 0

    def __call__(self, keys: np.ndarray) -> float:
        """Record a finite objective requiring at least one millisecond per call.

        Args:
            keys (np.ndarray): Unused candidate vector.

        Returns:
            float: Constant finite objective.

        Raises:
            AssertionError: If a 10 ms search attempts more than twelve evaluations.
            OSError: If the observation cannot be written.
        """
        self.calls += 1
        assert self.calls <= 12, "Callback evaluation budget exceeded."
        with self.path.open("ab") as stream:
            stream.write(b"1")
        time.sleep(0.001)
        return 1.0


def exit_cost(keys: np.ndarray) -> float:
    """Raise SystemExit inside a callback rather than ending its thread silently.

    Args:
        keys (np.ndarray): Unused candidate.

    Raises:
        SystemExit: Always, to verify worker failure propagation.
    """
    raise SystemExit("callback exit")


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
def test_returning_callback_local_search_obeys_budget(execution: str, tmp_path: Path) -> None:
    """Preserve an evaluated result without waiting for the full local neighborhood.

    Args:
        execution (str): Execution mode under test.
        tmp_path (Path): Trial-specific callback observation directory.
    """
    observation = tmp_path / "evaluations.bin"
    result = RKO(
        decode,
        RecordingCost(observation),
        100,
        algorithms=["ILS"],
        control=0,
        execution=execution,
    ).solve(0.01)
    assert result.cost == 1.0
    assert len(result.keys) == 100
    assert 1 <= len(observation.read_bytes()) <= 12


def test_thread_base_exception_reaches_caller() -> None:
    """Do not return an empty successful-looking result after a thread exits."""
    with pytest.raises(RuntimeError, match="MultiStart") as captured:
        RKO(decode, exit_cost, 3, algorithms=["MultiStart"], control=0, execution="thread").solve(1)
    assert isinstance(captured.value.__cause__, SystemExit)


def test_decoder_cannot_mutate_the_search_chromosome() -> None:
    """Keep returned keys associated with their decoded objective after mutation."""

    def mutating_decoder(keys: np.ndarray) -> np.ndarray:
        """Mutate callback-owned input without changing the algorithm's chromosome.

        Args:
            keys (np.ndarray): Detached candidate input.

        Returns:
            np.ndarray: Shifted decoded solution.
        """
        keys += 0.1
        return keys

    result = RKO(
        mutating_decoder,
        lambda values: float(values.sum()),
        3,
        algorithms=["MultiStart"],
        control=0,
        execution="sequential",
    ).solve(0.01)
    assert all(0 <= key < 1 for key in result.keys)
    assert result.cost == pytest.approx(sum(key + 0.1 for key in result.keys))


def test_incumbent_keeps_snapshot_despite_callback_mutation() -> None:
    """Retain detached keys and materialize public lists after cooperative stopping."""
    decoded_inputs: list[np.ndarray] = []

    def mutate_decode(keys: np.ndarray) -> np.ndarray:
        """Record original keys before mutating callback-owned storage.

        Args:
            keys (np.ndarray): Detached input belonging to the decoder.

        Returns:
            np.ndarray: Mutated payload for the objective.
        """
        decoded_inputs.append(keys.copy())
        keys[:] = 9.0
        return keys

    def mutate_cost(payload: np.ndarray) -> float:
        """Mutate the decoded payload and expire the next callback boundary.

        Args:
            payload (np.ndarray): Decoder-owned array.

        Returns:
            float: Finite completed objective.
        """
        payload[:] = -9.0
        session.deadline = 0.0
        return 1.0

    session = SearchSession(mutate_decode, mutate_cost)
    algorithm = MultiStart(SolutionPool(), mutate_decode, mutate_cost, 100)
    result = session.solve(algorithm, 60.0)
    assert isinstance(result[1], list)
    assert result[2] == 1.0
    np.testing.assert_array_equal(result[1], decoded_inputs[0])
    assert algorithm.decoder is mutate_decode
    assert algorithm.cost is mutate_cost
    assert session.best is not None
    assert isinstance(session.best[0], np.ndarray)
    assert not np.shares_memory(session.best[0], decoded_inputs[0])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_objective_is_not_a_completed_result(value: float) -> None:
    """Reject invalid objective values at the shared evaluation boundary.

    Args:
        value (float): Nonfinite callback return value.
    """
    with pytest.raises(ValueError, match="finite"):
        RKO(decode, lambda keys: value, 3, algorithms=["MultiStart"], execution="sequential").solve(
            0.01
        )
