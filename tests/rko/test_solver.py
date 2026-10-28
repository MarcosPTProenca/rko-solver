"""Public solver class and result contracts.

Objective:
- Verify configuration ownership, real execution and repeated-solve state.

Technical approach:
- Exercise a small real portfolio through each supported execution mode.

Flow (pseudocode):
1. Construct RKO with Python settings and importable problem callbacks.
2. Solve and independently verify returned keys, cost and metadata.
3. Repeat execution and inject failure to check result ownership.

Important criteria:
- No parameter directory editing is required for portfolio selection.
- A failed attempt must not leave an earlier result marked as current.

Inputs/Outputs:
- Input: bounded sphere problem and valid/invalid solver settings.
- Output: assertions on API behavior and immutable result snapshots.
"""

import numpy as np
import pytest
from dataclasses import FrozenInstanceError
from pathlib import Path
from rko_solver import RKO
from rko_solver import RKOResult


def decode(keys: np.ndarray) -> np.ndarray:
    """Return candidate keys unchanged.

    Args:
        keys (np.ndarray): Candidate vector.

    Returns:
        np.ndarray: Unmodified vector.
    """
    return keys


def objective(keys: np.ndarray) -> float:
    """Evaluate a sphere objective.

    Args:
        keys (np.ndarray): Candidate vector.

    Returns:
        float: Sum of squared coordinates.
    """
    return float(np.sum(keys**2))


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
def test_solver_owns_real_execution(execution: str) -> None:
    """Check actual execution, Python selection and immutable results.

    Args:
        execution (str): Backend under test.
    """
    optimizer = RKO(decode, objective, 4, algorithms=["MultiStart"], control=0, execution=execution)
    assert optimizer.result_ is None
    result = optimizer.solve(0.02)
    assert isinstance(result, RKOResult)
    assert result is optimizer.result_
    assert len(result.keys) == 4
    assert result.cost == pytest.approx(objective(np.asarray(result.keys)))
    assert result.algorithms == ("MultiStart",)
    assert result.engine == "python" and result.execution == execution
    assert result.elapsed_seconds > 0
    with pytest.raises(FrozenInstanceError):
        result.cost = 99.0


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
@pytest.mark.parametrize("control", [0, 1])
def test_solver_runs_batched_pso(execution: str, control: int) -> None:
    """Exercise batched particle movement through every execution mode and control mode.

    Args:
        execution (str): Execution mode under test.
        control (int): Offline or adaptive parameter mode.
    """
    result = RKO(
        decode, objective, 128, algorithms=["PSO"], control=control, execution=execution
    ).solve(0.1)
    keys = np.asarray(result.keys)
    assert len(keys) == 128
    assert np.all((keys >= 0.0) & (keys < 1.0))
    assert np.isfinite(result.cost)
    assert result.cost == pytest.approx(objective(keys))


def test_repeated_solve_and_failure_clear_stale_result() -> None:
    """Create independent result snapshots and clear the last result on failure."""
    optimizer = RKO(
        decode, objective, 3, algorithms=["MultiStart"], control=0, execution="sequential"
    )
    first = optimizer.solve(0.005)
    second = optimizer.solve(0.005)
    assert first is not second
    assert optimizer.result_ is second
    with pytest.raises(ValueError, match="time_limit"):
        optimizer.solve(0)
    assert optimizer.result_ is None


def test_constructor_performs_no_io_or_execution(tmp_path: Path) -> None:
    """Delay configuration I/O until solve.

    Args:
        tmp_path (Path): Root without parameter files.
    """
    optimizer = RKO(decode, objective, 3, params_dir=tmp_path)
    assert optimizer.result_ is None
    with pytest.raises(FileNotFoundError):
        optimizer.solve(0.01)


@pytest.mark.parametrize(
    "settings",
    [
        {"algorithms": "GA"},
        {"algorithms": []},
        {"algorithms": ["GA", "GeneticAlgorithm"]},
        {"algorithms": [42]},
        {"control": 0.9},
        {"control": True},
        {"pool_size": 2.9},
        {"pool_size": False},
    ],
)
def test_solver_rejects_ambiguous_python_settings(settings: dict[str, object]) -> None:
    """Reject ambiguous settings before running a worker.

    Args:
        settings (dict[str, object]): Invalid Python overrides.
    """
    with pytest.raises(ValueError):
        RKO(decode, objective, 3, **settings).solve(0.01)
