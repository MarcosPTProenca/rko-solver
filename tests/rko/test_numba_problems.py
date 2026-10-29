"""Contract tests for objectives compiled with Numba and called without the interpreter.

Objective:
- Verify that compiled objectives run in the native engine with correct scoring, parallel
  workers, validation and failure handling.

Technical approach:
- Gate first native evaluations in an owned subprocess to observe worker overlap.
- Skip when Numba or the extension is unavailable; compile a Knapsack objective once.
- Recompute every returned objective independently in Python.

Flow (pseudocode):
1. Compile a Knapsack objective as an `@njit` function and as a raw `cfunc`.
2. Solve with several algorithms and compare with independent scoring.
3. Check validation errors and nonfinite handling.

Important criteria:
- Functional tests never infer parallel execution from elapsed-time thresholds.
- A bounded subprocess owns the native-worker synchronization probe.

Inputs/Outputs:
- Input: a small synthetic instance and short time limits.
- Output: pass/fail assertions.
"""

import ctypes
import multiprocessing as mp
import numpy as np
import pytest
import threading
from multiprocessing.connection import Connection
from rko_solver import RKO
from rko_solver import Numba

numba = pytest.importorskip("numba")
pytest.importorskip("rko_solver._native")

PROFITS = np.array([10.0 + (index * 37) % 91 for index in range(60)])
WEIGHTS = np.array([5.0 + (index * 53) % 47 for index in range(60)])
CAPACITY = 700.0


@numba.njit
def knapsack(keys: np.ndarray) -> float:
    """Score a key vector with the reference Knapsack objective.

    Args:
        keys (np.ndarray): Random keys.

    Returns:
        float: Penalized negative profit.
    """
    profit = 0.0
    weight = 0.0
    for index in range(keys.shape[0]):
        if keys[index] > 0.5:
            profit += PROFITS[index]
            weight += WEIGHTS[index]
    return 100000.0 * max(0.0, weight - CAPACITY) - profit


def python_cost(keys: tuple[float, ...]) -> float:
    """Recompute the objective without Numba.

    Args:
        keys (tuple[float, ...]): Returned random keys.

    Returns:
        float: Penalized negative profit.
    """
    chosen = np.asarray(keys) > 0.5
    return float(100000.0 * max(0.0, WEIGHTS[chosen].sum() - CAPACITY) - PROFITS[chosen].sum())


@pytest.mark.parametrize("name", ["SA", "BRKGA", "ILS", "GA", "GRASP", "IPR", "PSO"])
def test_njit_objective_runs_and_matches_independent_scoring(name: str) -> None:
    """An njit function is compiled into an entry point and scored natively."""
    result = RKO(
        Numba(knapsack, 60), engine="rust", algorithms=[name], adaptation="adaptive", random_state=4
    ).solve(0.3)
    assert result.engine == "rust"
    assert python_cost(result.keys) == result.cost
    assert result.searches[0].n_evaluations > 100


class CompiledOverlapProbe:
    """Observe native-worker overlap through a test-only compiled synchronization entry point.

    Responsibilities:
    - Require four distinct workers to enter their first objective before any can finish it.
    - Isolate a potential interpreter-lock deadlock in a killable child process.

    Attributes:
        None: Each invocation owns its process, connection and synchronization objects.
    """

    @staticmethod
    def worker(connection: Connection) -> None:
        """Run a real compiled portfolio whose first objective waits for all workers.

        Args:
            connection (Connection): Owned pipe used to return observations or a bounded error.

        Raises:
            OSError: If the parent closes the observation pipe before the worker responds.
        """
        barrier = threading.Barrier(4, timeout=10)
        arrivals: set[int] = set()
        guard = threading.Lock()

        @ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_ssize_t)
        def synchronized(keys: object, size: int) -> float:
            """Gate a test-only native callback until four distinct worker threads arrive.

            Args:
                keys (object): Unmodified engine-owned key buffer; unused by this probe.
                size (int): Key-buffer length; unused by this probe.

            Returns:
                float: Zero after concurrent arrival, or NaN when synchronization fails.
            """
            with guard:
                arrivals.add(threading.get_ident())
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                return float("nan")
            return 0.0

        signature = numba.types.float64(numba.types.CPointer(numba.types.float64), numba.types.intp)

        @numba.cfunc(signature)
        def entry(keys: object, size: int) -> float:
            """Forward the raw compiled call to the test-only synchronization observer.

            Args:
                keys (object): Unmodified engine-owned key buffer.
                size (int): Number of keys in the buffer.

            Returns:
                float: The synchronization observer's objective value.
            """
            return synchronized(keys, size)

        try:
            result = RKO(
                Numba(entry, 8),
                engine="rust",
                algorithms=["SA", "BRKGA", "ILS", "GA"],
                max_evaluations=1,
                random_state=1,
            ).solve(60.0)
            connection.send(
                {
                    "workers": result.n_workers,
                    "arrivals": len(arrivals),
                    "counts": tuple(search.n_evaluations for search in result.searches),
                    "stops": tuple(search.stop_reason for search in result.searches),
                }
            )
        except Exception as error:
            connection.send({"error": str(error)})
        finally:
            connection.close()

    @staticmethod
    def run() -> dict[str, object]:
        """Collect worker-overlap evidence with a bounded native-call lifetime.

        Returns:
            dict[str, object]: Child observations or a reported engine error.

        Raises:
            AssertionError: If the child hangs or exits unsuccessfully.
            OSError: If the process or observation pipe cannot be created.
            EOFError: If the child exits without sending its observations.
        """
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=CompiledOverlapProbe.worker, args=(child,))
        try:
            process.start()
            child.close()
            assert parent.poll(30), "Native overlap probe stalled before reporting worker arrivals"
            observations = parent.recv()
            process.join(timeout=5)
            assert process.exitcode == 0, f"Native overlap probe exited with {process.exitcode}"
            return observations
        finally:
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
            parent.close()
            child.close()


def test_workers_run_in_parallel_without_the_interpreter_lock() -> None:
    """Require simultaneous native-worker arrivals without a throughput or elapsed-time claim."""
    observations = CompiledOverlapProbe.run()
    assert "error" not in observations, observations
    assert observations["workers"] == 4
    assert observations["arrivals"] == 4
    assert observations["counts"] == (1, 1, 1, 1)
    assert observations["stops"] == ("max_evaluations",) * 4


def test_cfunc_is_accepted_directly() -> None:
    """A cfunc with the documented signature is used as is."""
    signature = numba.types.float64(numba.types.CPointer(numba.types.float64), numba.types.intp)

    @numba.cfunc(signature)
    def entry(keys: object, size: object) -> float:
        """Score the first key, as a minimal objective.

        Args:
            keys (object): Pointer to the key buffer.
            size (object): Number of keys.

        Returns:
            float: Sum of the keys.
        """
        total = 0.0
        for index in range(size):
            total += keys[index]
        return total

    result = RKO(Numba(entry, 8), engine="rust", algorithms=["SA"], random_state=2).solve(0.2)
    assert result.cost == pytest.approx(sum(result.keys))
    assert result.cost < 1.0


def test_invalid_inputs_are_rejected() -> None:
    """Plain Python functions and bad dimensions fail before any search starts."""
    with pytest.raises(ValueError, match="numba"):
        RKO(Numba(lambda keys: 0.0, 8), engine="rust")
    with pytest.raises(ValueError, match="n_variables"):
        RKO(Numba(knapsack, 0), engine="rust")


def test_nonfinite_objective_fails_the_solve() -> None:
    """A compiled objective returning NaN is rejected like any other objective."""

    @numba.njit
    def broken(keys: np.ndarray) -> float:
        """Return a nonfinite value.

        Args:
            keys (np.ndarray): Random keys.

        Returns:
            float: NaN.
        """
        return np.nan

    with pytest.raises(ValueError, match="finite"):
        RKO(Numba(broken, 8), engine="rust", algorithms=["SA"]).solve(0.2)
