"""Process execution with the shared-memory elite pool.

Objective:
- Verify that the shared pool gives the same search result as the manager pool, and that it
  starts no manager process.

Technical approach:
- Run one worker with a fixed seed and an evaluation cap, once per pool mode, and compare the
  returned solution and per-search evaluation counts. Patch the mode selector and the manager.

Flow (pseudocode):
1. Build a two-algorithm offline portfolio with one worker and an evaluation cap.
2. Solve with the manager pool and with the shared pool; compare results.
3. Solve with the shared pool while the manager class raises; the solve must still succeed.

Important criteria:
- Single worker, fixed seed and evaluation cap make both modes deterministic.

Inputs/Outputs:
- Input: tiny parameter directories and a module-level objective.
- Output: equal keys, cost and evaluation counts; no manager in shared mode.
"""

from __future__ import annotations

import numpy as np
import pytest
import shutil
from pathlib import Path
from rko_solver import RKO
from rko_solver.orchestration import processes
from rko_solver.parameters.loader import default_params_dir


def identity(keys: np.ndarray) -> np.ndarray:
    """Return the key vector unchanged.

    Args:
        keys (np.ndarray): Candidate vector.

    Returns:
        np.ndarray: Unmodified vector.
    """
    return keys


def sphere(keys: np.ndarray) -> float:
    """Return the sum of squares of the keys.

    Args:
        keys (np.ndarray): Candidate vector.

    Returns:
        float: Finite objective value.
    """
    return float(np.dot(keys, keys))


@pytest.fixture
def params_dir(tmp_path: Path) -> Path:
    """Prepare a two-algorithm offline portfolio.

    Args:
        tmp_path (Path): Temporary root for configuration.

    Returns:
        Path: Directory containing complete parameter TOMLs.
    """
    destination = tmp_path / "parameters"
    shutil.copytree(default_params_dir(), destination)
    destination.joinpath("config.toml").write_text(
        '[run]\ncontrol = 0\npool_size = 5\n[active_metaheuristics]\nlist = ["IPR", "GA"]\n',
        encoding="utf-8",
    )
    return destination


def solve(params_dir: Path) -> tuple[list[float], float, list[int]]:
    """Run one deterministic single-worker process solve.

    Args:
        params_dir (Path): Two-algorithm offline portfolio.

    Returns:
        tuple[list[float], float, list[int]]: Keys, cost and per-search evaluation counts.
    """
    result = RKO(
        identity,
        sphere,
        4,
        params_dir=params_dir,
        n_jobs=1,
        random_state=3,
        max_evaluations=200,
    ).solve(20.0)
    return list(result.keys), result.cost, [search.n_evaluations for search in result.searches]


def test_shared_and_manager_pools_give_the_same_result(
    params_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two pool modes return the same solution and evaluation counts.

    Args:
        params_dir (Path): Two-algorithm offline portfolio.
        monkeypatch (pytest.MonkeyPatch): Fixture used to select the pool mode.
    """
    monkeypatch.setattr(processes, "_use_shared_pool", lambda verbose, trace: False)
    manager_outcome = solve(params_dir)
    monkeypatch.setattr(processes, "_use_shared_pool", lambda verbose, trace: True)
    shared_outcome = solve(params_dir)
    assert shared_outcome == manager_outcome


def test_shared_mode_starts_no_manager(params_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the shared pool selected, the solve never constructs a manager.

    Args:
        params_dir (Path): Two-algorithm offline portfolio.
        monkeypatch (pytest.MonkeyPatch): Fixture used to replace the manager class.
    """

    def forbidden(*args: object, **kwargs: object) -> None:
        """Fail if a manager is constructed in shared mode.

        Args:
            *args (object): Ignored.
            **kwargs (object): Ignored.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("manager constructed in shared mode")

    monkeypatch.setattr(processes, "_use_shared_pool", lambda verbose, trace: True)
    monkeypatch.setattr(processes, "PoolManager", forbidden)
    keys, cost, counts = solve(params_dir)
    assert len(keys) == 4
    assert cost >= 0.0
    assert sum(counts) > 0
