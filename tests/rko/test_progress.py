"""Live progress contracts for shared-pool and solver output.

Objective:
- Make cooperative improvements visible without changing elite selection.

Technical approach:
- Capture deterministic pool updates and actual serial/thread/spawn execution.
- Hold a callback open to prove output appears before solve returns.

Flow (pseudocode):
1. Insert accepted/rejected candidates and inspect attributed pool summaries.
2. Compare quiet and verbose results and exercise each execution mode.
3. Pause an active search, read progress, release it and verify completion.

Important criteria:
- Quiet is the library default; rejected candidates produce no update lines.
- Logs contain objective values and sources, not full chromosomes.

Inputs/Outputs:
- Input: finite candidates, solver settings and captured output streams.
- Output: assertions on live reporting, ordering and unchanged pool contracts.
"""

import numpy as np
import pytest
import re
import subprocess
import sys
import threading
import time
from rko_solver import RKO
from rko_solver.components.solution_pool import SolutionPool
from tests.rko.test_solver import decode
from tests.rko.test_solver import objective


def test_pool_reports_attributed_updates_and_keeps_public_pairs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report sorted accepted costs with their producing algorithm labels.

    Args:
        capsys (pytest.CaptureFixture[str]): Capture local pool output.
    """
    pool = SolutionPool(2, verbose=True)
    pool.insert((10.0, [0.1]), "SA")
    pool.insert((5.0, [0.2]), "GA")
    pool.insert((8.0, [0.3]), "PSO")
    pool.insert((12.0, [0.4]), "rejected")
    pool._report_status()
    output = capsys.readouterr().out
    lines = output.splitlines()
    assert len(lines) == 3
    assert "NEW BEST" in lines[0] and "SA" in lines[0]
    assert "NEW BEST" in lines[1] and "GA" in lines[1]
    assert "STATUS" in lines[2] and "PSO" in lines[2]
    assert all("pool=" not in line for line in lines[:2])
    assert "best=5" in lines[2]
    assert "pool=2/2 [5 (GA), 8 (PSO)]" in lines[2]
    assert "rejected" not in output
    assert "0.3" not in lines[2].split("pool=")[1]
    assert pool.best == (5.0, [0.2])
    assert pool.get_elite(2) == [(5.0, [0.2]), (8.0, [0.3])]


@pytest.mark.parametrize("verbose", [False, True])
def test_reporting_does_not_change_elite_selection(
    capsys: pytest.CaptureFixture[str],
    verbose: bool,
) -> None:
    """Keep insertion/selection identical regardless of output settings.

    Args:
        capsys (pytest.CaptureFixture[str]): Capture local output.
        verbose (bool): Enable or disable progress.
    """
    pool = SolutionPool(2, verbose=verbose)
    for value in [9.0, 4.0, 6.0, 4.0, 7.0]:
        pool.insert((value, [0.25]), "worker")
    assert pool.get_elite(2) == [(4.0, [0.25])]
    assert bool(capsys.readouterr().out) == verbose


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
@pytest.mark.parametrize("verbose", [False, True])
def test_solver_progress_on_each_execution(
    capfd: pytest.CaptureFixture[str],
    execution: str,
    verbose: bool,
) -> None:
    """Capture real manager output and preserve quiet defaults in all execution modes.

    Args:
        capfd (pytest.CaptureFixture[str]): Capture parent and spawned stdout.
        execution (str): Execution mode to exercise.
        verbose (bool): Whether progress is requested.
    """
    result = RKO(
        decode,
        objective,
        3,
        algorithms=["MultiStart"],
        control=0,
        execution=execution,
        verbose=verbose,
    ).solve(0.02)
    output = capfd.readouterr().out
    assert np.isfinite(result.cost)
    if verbose:
        assert "RKO START" in output and f"execution={execution}" in output
        assert "NEW BEST" in output and "algorithm=MultiStart" in output
        assert "pool=" in output and "(MultiStart)" in output
        assert "RKO DONE" in output
        assert output.index("RKO START") < output.index("NEW BEST") < output.index("RKO DONE")
    else:
        assert output == ""


def test_default_solver_is_quiet(capsys: pytest.CaptureFixture[str]) -> None:
    """Avoid unsolicited output when the new setting is omitted.

    Args:
        capsys (pytest.CaptureFixture[str]): Capture default library output.
    """
    RKO(decode, objective, 2, algorithms=["MultiStart"], control=0, execution="sequential").solve(
        0.01
    )
    assert capsys.readouterr().out == ""


def test_pool_output_is_live_before_solve_finishes(capsys: pytest.CaptureFixture[str]) -> None:
    """Observe a flushed update while a second objective evaluation is blocked.

    Args:
        capsys (pytest.CaptureFixture[str]): Capture worker-thread progress.
    """
    entered = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []
    calls = 0

    def pausing_objective(keys: np.ndarray) -> float:
        """Hold the second evaluation until the test has inspected live output.

        Args:
            keys (np.ndarray): Unused candidate keys.

        Returns:
            float: Initial cost one, then improved cost zero.
        """
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            release.wait(10)
        return 1.0 if calls == 1 else 0.0

    optimizer = RKO(
        decode,
        pausing_objective,
        2,
        algorithms=["MultiStart"],
        control=0,
        execution="thread",
        verbose=True,
    )

    def solve() -> None:
        """Capture search failures so the owning test can report them."""
        try:
            optimizer.solve(2.0)
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=solve, daemon=True)
    try:
        worker.start()
        assert entered.wait(10)
        assert worker.is_alive()
        output = capsys.readouterr().out
        assert "NEW BEST" in output and "best=1" in output
        assert "RKO DONE" not in output
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    assert not failures
    assert optimizer.result_ is not None
    assert "RKO DONE" in capsys.readouterr().out


def constant_objective(keys: np.ndarray) -> float:
    """Hold the objective constant to reproduce long stretches without improvements.

    Args:
        keys (np.ndarray): Unused candidate vector.

    Returns:
        float: Fixed objective, producing only one shared-pool improvement.
    """
    return 1.0


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
def test_status_streams_during_plateaus_and_stops_with_search(execution: str) -> None:
    """Read periodic stdout while a real plateauing search is still running.

    Args:
        execution (str): Execution mode of the isolated search.
    """
    program = (
        "import threading; from rko_solver import RKO; "
        "from tests.rko.test_progress import decode, constant_objective; "
        "RKO(decode, constant_objective, 2, algorithms=['MultiStart'], control=0, "
        f"execution={execution!r}, verbose=True).solve(2.2); "
        "print('PROGRESS_THREADS', sum(t.name == 'rko-progress' for t in threading.enumerate()))"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    watchdog = threading.Timer(45, process.kill)
    watchdog.daemon = True
    received: list[tuple[float, str, bool]] = []
    try:
        watchdog.start()
        for line in process.stdout:
            received.append((time.perf_counter(), line.strip(), process.poll() is None))
        assert process.wait(timeout=10) == 0, process.stderr.read()
    finally:
        watchdog.cancel()
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)
    improvements = [stamp for stamp, line, _ in received if "NEW BEST" in line]
    assert len(improvements) == 1
    statuses = [(stamp, line, alive) for stamp, line, alive in received if "STATUS" in line]
    plateau = [stamp for stamp, _, _ in statuses if stamp > improvements[0]]
    assert len(plateau) >= 2
    assert plateau[-1] - plateau[0] >= 0.5
    assert any(alive for _, _, alive in statuses)
    assert all(
        re.match(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \+\d+\.\d{3}s\]", line)
        for _, line, _ in statuses
    )
    assert any(line == "PROGRESS_THREADS 0" for _, line, _ in received)


def failing_objective(keys: np.ndarray) -> float:
    """Raise an importable callback failure for verbose lifecycle checks.

    Args:
        keys (np.ndarray): Unused candidate keys.

    Raises:
        ValueError: Always, to emulate a failed problem evaluation.
    """
    raise ValueError("deliberate objective failure")


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
def test_failed_search_does_not_announce_completion(
    capfd: pytest.CaptureFixture[str],
    execution: str,
) -> None:
    """Preserve worker errors and avoid a misleading successful finish message.

    Args:
        capfd (pytest.CaptureFixture[str]): Capture parent/manager output.
        execution (str): Execution mode running the failing callback.
    """
    optimizer = RKO(
        decode,
        failing_objective,
        2,
        algorithms=["MultiStart"],
        control=0,
        execution=execution,
        verbose=True,
    )
    with pytest.raises((ValueError, RuntimeError), match="deliberate objective failure"):
        optimizer.solve(0.02)
    assert optimizer.result_ is None
    output = capfd.readouterr().out
    assert "RKO START" in output
    assert "RKO DONE" not in output


@pytest.mark.parametrize("verbose", [1, "yes", None])
def test_solver_rejects_invalid_verbosity(verbose: object) -> None:
    """Reject ambiguous verbosity before constructing or starting a portfolio.

    Args:
        verbose (object): Nonboolean verbosity value.
    """
    with pytest.raises(ValueError, match="verbose"):
        RKO(decode, objective, 2, execution="sequential", verbose=verbose).solve(0.01)
