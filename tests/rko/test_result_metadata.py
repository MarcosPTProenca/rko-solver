"""Result accounting and configuration provenance regression contracts.

Objective:
- Independently verify all completed objective calls and immutable search settings.

Technical approach:
- Delay process release and bound objective work independently of scheduler timing.
- Persist one byte per finite objective in separate process/thread observation files.
- Compare public totals with actual callback work across every execution mode.

Flow (pseudocode):
1. Execute a seeded two-algorithm numerical portfolio.
2. Count independent objective observations and inspect per-search reports.
3. Verify owned configuration snapshots and compatibility tuple results.

Important criteria:
- Process startup is not a search stop reason or an objective evaluation.
- Metadata includes all algorithms, not only the worker or portfolio winner.

Inputs/Outputs:
- Input: temporary observations and evaluation-bounded domain-neutral searches.
- Output: accounting, configuration, seed and ownership assertions.
"""

import json
import numpy as np
import os
import pytest
import threading
import time
from dataclasses import dataclass
from multiprocessing.synchronize import Event
from pathlib import Path
from rko_solver import RKO
from rko_solver.parameters.loader import load_offline


def decode(keys: np.ndarray) -> np.ndarray:
    """Return candidate keys without adding stochastic or mutable state.

    Args:
        keys (np.ndarray): Detached candidate.

    Returns:
        np.ndarray: Objective payload.
    """
    return keys


@dataclass
class ObservedCost:
    """Count finite objectives independently of solver instrumentation.

    Responsibilities:
    - Keep concurrent writers in distinct process/thread-owned files.

    Attributes:
        directory (Path): Existing trial-owned observation directory.
    """

    directory: Path

    def __call__(self, keys: np.ndarray) -> float:
        """Record one successful numerical objective.

        Args:
            keys (np.ndarray): Decoded numerical vector.

        Returns:
            float: Finite squared distance from the origin.

        Raises:
            OSError: If the independent observation cannot be persisted.
        """
        value = float(keys @ keys)
        path = self.directory / f"{os.getpid()}-{threading.get_ident()}.bin"
        with path.open("ab") as stream:
            stream.write(b"1")
        return value


class FirstKeysCost(ObservedCost):
    """Retain the first candidate independently of an unknown requested seed.

    Responsibilities:
    - Observe the actual initialization stream inside the executing worker.
    """

    def __call__(self, keys: np.ndarray) -> float:
        """Persist the first candidate and count the finite objective normally.

        Args:
            keys (np.ndarray): First or subsequent numerical candidate.

        Returns:
            float: Finite squared distance.

        Raises:
            OSError: If observations cannot be persisted.
        """
        first = self.directory / "first.json"
        if not first.exists():
            first.write_text(json.dumps(keys.tolist()), encoding="utf-8")
        return super().__call__(keys)


@pytest.mark.parametrize(
    "execution,n_jobs", [("sequential", None), ("thread", None), ("process", 1), ("process", 2)]
)
@pytest.mark.parametrize("control", [0, 1])
def test_metadata_counts_all_searches(
    execution: str,
    n_jobs: int | None,
    control: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match successful callback work, actual seeds and initial settings.

    Args:
        execution (str): Execution mode.
        n_jobs (int | None): Dedicated or grouped process budget.
        control (int): Offline or supported adaptive control.
        tmp_path (Path): Independent trial observations.
        monkeypatch (pytest.MonkeyPatch): Inject delayed process release to exercise scheduling lag.
    """
    release = Event.set

    def delayed_release(event: Event) -> None:
        """Delay the parent's event release to simulate a descheduled search worker.

        Args:
            event (Event): Process synchronization event awaiting release.
        """
        time.sleep(0.05)
        release(event)

    if execution == "process":
        monkeypatch.setattr(Event, "set", delayed_release)
    result = RKO(
        decode,
        ObservedCost(tmp_path),
        8,
        algorithms=["MultiStart", "PSO"],
        control=control,
        execution=execution,
        n_jobs=n_jobs,
        random_state=42,
        max_evaluations=40,
        hyperparameters={"PSO": {"p_size": 4}} if control == 0 else None,
    ).solve(60.0)
    observed = sum(path.stat().st_size for path in tmp_path.iterdir())
    assert result.n_evaluations == observed
    assert result.n_evaluations == sum(item.n_evaluations for item in result.searches)
    assert [item.algorithm for item in result.searches] == ["MultiStart", "PSO"]
    seeds = np.random.SeedSequence(42).spawn(2)
    assert [item.random_state for item in result.searches] == [
        int(seed.generate_state(1)[0]) for seed in seeds
    ]
    assert result.stop_reason == "max_evaluations"
    assert result.max_evaluations == 40
    assert result.n_variables == 8
    assert result.time_limit == 60.0
    assert result.control == control
    assert result.pool_size > 0
    assert result.n_jobs == n_jobs
    assert result.n_workers == (2 if execution == "thread" or n_jobs == 2 else 1)
    assert result.searches[0].control == 0
    pso = result.searches[1]
    assert pso.control == control
    assert dict(pso.parameters)["p_size"] == (4 if control == 0 else load_offline("PSO")["p_size"])
    assert bool(pso.parameter_grid) == bool(control)
    for report in result.searches:
        assert report.elapsed_seconds >= 0
        assert 0 <= report.time_limit <= result.time_limit
        assert report.time_limit > 0
        assert report.n_evaluations == 40
        assert report.stop_reason == "max_evaluations"


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
def test_unseeded_result_records_actual_execution_seed(execution: str, tmp_path: Path) -> None:
    """Record the worker seed, not a different seed used by parent-side validation.

    Args:
        execution (str): Execution mode.
        tmp_path (Path): Initial-vector observation directory.
    """
    result = RKO(
        decode,
        FirstKeysCost(tmp_path),
        4,
        algorithms=["MultiStart"],
        control=0,
        execution=execution,
        max_evaluations=1,
    ).solve(60.0)
    report = result.searches[0]
    assert result.random_state is None
    assert type(report.random_state) is int
    expected = np.random.RandomState(report.random_state).random(4)
    np.testing.assert_array_equal(json.loads((tmp_path / "first.json").read_text()), expected)
