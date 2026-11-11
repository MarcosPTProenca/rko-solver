"""Public thread-limit context manager for spawned process workers.

Objective:
- Verify that `worker_thread_limit` reaches every spawned worker of a solve inside its block
  and restores the parent environment on exit.

Technical approach:
- Run a real two-worker process portfolio with a module-level objective that records the
  BLAS/OpenMP variables it sees, so the check uses the same spawn path users get.

Flow (pseudocode):
1. Build a tiny offline portfolio and a picklable objective that writes its environment.
2. Solve inside `worker_thread_limit(1)` and snapshot the parent environment before and after.
3. Check each worker saw the limit, the parent is restored, and invalid values are rejected.

Important criteria:
- Each search stops after one evaluation, so the tests do not depend on convergence.

Inputs/Outputs:
- Input: tiny parameter directories and serializable callbacks.
- Output: assertions on worker environment values and parent environment equality.
"""

from __future__ import annotations

import numpy as np
import os
import pytest
import shutil
from dataclasses import dataclass
from pathlib import Path
from rko_solver import RKO
from rko_solver import worker_thread_limit
from rko_solver.parameters.loader import default_params_dir

THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def identity(keys: np.ndarray) -> np.ndarray:
    """Return the key vector unchanged.

    Args:
        keys (np.ndarray): Candidate vector.

    Returns:
        np.ndarray: Unmodified vector.
    """
    return keys


@dataclass
class ThreadEnvironmentRecorder:
    """Write each worker's thread-limit variables to one file per process.

    Responsibilities:
    - Make the environment seen inside a spawned worker observable from the test.

    Attributes:
        directory (Path): Destination for one observation per worker process.
    """

    directory: Path

    def __call__(self, keys: np.ndarray) -> float:
        """Record the thread variables of the current process once and return zero.

        Args:
            keys (np.ndarray): Candidate vector.

        Returns:
            float: Constant objective.
        """
        path = self.directory / str(os.getpid())
        if not path.exists():
            values = [os.environ.get(name, "unset") for name in THREAD_VARIABLES]
            path.write_text("|".join(values), encoding="utf-8")
        return 0.0


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
        '[run]\ncontrol = 0\npool_size = 5\n[active_metaheuristics]\nlist = ["MultiStart", "SA"]\n',
        encoding="utf-8",
    )
    return destination


def solve_observed(params_dir: Path, directory: Path) -> list[str]:
    """Run one limited process solve and return what each worker observed.

    Args:
        params_dir (Path): Two-algorithm offline portfolio.
        directory (Path): Empty destination for worker observations.

    Returns:
        list[str]: One pipe-joined value string per worker process.
    """
    with worker_thread_limit(1):
        RKO(
            identity,
            ThreadEnvironmentRecorder(directory),
            3,
            params_dir=params_dir,
            n_jobs=2,
            random_state=7,
            max_evaluations=1,
        ).solve(5.0)
    return [path.read_text(encoding="utf-8") for path in directory.iterdir()]


def test_limit_reaches_every_worker(params_dir: Path, tmp_path: Path) -> None:
    """Every spawned worker starts with the requested thread limit in its environment.

    Args:
        params_dir (Path): Two-algorithm offline portfolio.
        tmp_path (Path): Destination for worker observations.
    """
    observations = tmp_path / "observations"
    observations.mkdir()
    seen = solve_observed(params_dir, observations)
    assert len(seen) == 2
    assert all(value == "|".join(["1"] * len(THREAD_VARIABLES)) for value in seen)


def test_limit_restores_the_parent_environment(params_dir: Path, tmp_path: Path) -> None:
    """Values set before the block come back, and variables that were unset are removed.

    Args:
        params_dir (Path): Two-algorithm offline portfolio.
        tmp_path (Path): Destination for worker observations.
    """
    observations = tmp_path / "observations"
    observations.mkdir()
    os.environ["OMP_NUM_THREADS"] = "3"
    os.environ.pop("MKL_NUM_THREADS", None)
    try:
        before = {name: os.environ.get(name) for name in THREAD_VARIABLES}
        solve_observed(params_dir, observations)
        after = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    finally:
        os.environ.pop("OMP_NUM_THREADS", None)
    assert before["OMP_NUM_THREADS"] == "3"
    assert after == before


@pytest.mark.parametrize("value", [0, -1, 1.5, "1", True])
def test_limit_rejects_invalid_values(value: object) -> None:
    """Only positive integers are accepted, and the parent environment is not touched.

    Args:
        value (object): Invalid thread count.
    """
    before = dict(os.environ)
    with pytest.raises(ValueError, match="threads must be a positive integer"):
        with worker_thread_limit(value):  # type: ignore[arg-type]
            pass
    assert dict(os.environ) == before
