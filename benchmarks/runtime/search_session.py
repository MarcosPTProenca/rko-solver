"""Measure shared callback-boundary incumbent storage at fixed work.

Objective:
- Quantify allocations avoided by retaining detached NumPy incumbent snapshots.
- Measure accepted elite-pool insertion allocations without tracing source data.

Technical approach:
- Compare production SearchSession against its previous list-materializing evaluator.
- Use identical input vectors and always-improving, periodic and plateau cost streams.
- Measure untraced runtime separately from traced current/peak allocation.

Flow (pseudocode):
1. Replay 1,000 decoder/objective calls through each session variant.
2. Compare exact incumbent costs/keys and assert isolation from input mutation.
3. Report paired timing medians and separately measured allocation footprints.

Important criteria:
- Both candidate copies and finite-objective validation remain enabled.
- This isolates the shared callback boundary, not whole-solver speed or process RSS.
- Traced current allocation includes the retained incumbent; source data predates tracing.

Inputs/Outputs:
- Input: synthetic vectors of 1,000 and 10,000 keys; no external data.
- Output: median elapsed time and current/peak tracemalloc bytes for both variants.
"""

import math
import numpy as np
import statistics
import time
import tracemalloc
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.orchestration.search import SearchSession
from rko_solver.orchestration.search import _DecodedCandidate


class _ReferenceSession(SearchSession):
    """Reproduce the previous incumbent conversion at each improvement.

    Responsibilities:
    - Retain the former evaluate behavior solely for benchmark comparison.
    """

    def evaluate(self, candidate: _DecodedCandidate) -> float:
        """Evaluate a payload and eagerly materialize improved keys as Python floats.

        Args:
            candidate (_DecodedCandidate): Detached keys and original decoded payload.

        Returns:
            float: Finite objective value.

        Raises:
            ValueError: If the objective is nonfinite.
        """
        self._check()
        value = float(self.cost(candidate.payload))
        if not math.isfinite(value):
            raise ValueError("Objective callbacks must return finite costs.")
        if self.best is None or value < self.best[1]:
            self.best = (candidate.keys.tolist(), value)
        return value


def _sample(
    keys: np.ndarray, interval: int, reference: bool, traced: bool = False
) -> tuple[float, int, int]:
    """Run one fixed-work callback stream and verify the retained incumbent.

    Args:
        keys (np.ndarray): Candidate data allocated before tracing.
        interval (int): Objective calls between strict improvements.
        reference (bool): Whether to use the previous evaluator.
        traced (bool): Whether to collect allocations instead of clean timing.

    Returns:
        tuple[float, int, int]: Elapsed seconds, traced current bytes and peak bytes.
    """
    calls = 0

    def objective(payload: np.ndarray) -> float:
        """Emit a deterministic finite cost stream.

        Args:
            payload (np.ndarray): Unchanged decoded candidate.

        Returns:
            float: Cost improving at the requested interval.
        """
        nonlocal calls
        calls += 1
        return -float(calls // interval)

    if traced:
        tracemalloc.start()
    try:
        session = (_ReferenceSession if reference else SearchSession)(lambda keys: keys, objective)
        started = time.perf_counter()
        for _ in range(1000):
            session.evaluate(session.decode(keys))
        elapsed = time.perf_counter() - started
        current, peak = tracemalloc.get_traced_memory() if traced else (0, 0)
    finally:
        if traced:
            tracemalloc.stop()
    assert session.best is not None
    np.testing.assert_array_equal(session.best[0], keys)
    assert session.best[1] == -float(1000 // interval)
    assert calls == 1000
    original = keys[0]
    keys[0] = -1
    assert session.best[0][0] == original
    keys[0] = original
    return elapsed, current, peak


def _pool_allocations(dimension: int) -> tuple[int, int]:
    """Measure a capacity-one pool replacing its incumbent 100 times.

    Args:
        dimension (int): Number of keys in each inserted entry.

    Returns:
        tuple[int, int]: Current and peak traced allocation in bytes.
    """
    keys = np.linspace(0, 1, dimension).tolist()
    pool = SolutionPool(capacity=1)
    tracemalloc.start()
    try:
        for index in range(100):
            pool.insert((-float(index), keys))
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert pool.best == (-99.0, keys)
    keys[0] = -1.0
    assert pool.best[1][0] == 0.0
    return current, peak


def main() -> None:
    """Print fixed-work medians and separately traced shared-state allocations."""
    for dimension in (1000, 10000):
        current, peak = _pool_allocations(dimension)
        print(f"pool n={dimension} current_bytes={current} peak_bytes={peak}", flush=True)
        keys = np.linspace(0, 1, dimension)
        for workload, interval in (("improving", 1), ("periodic", 50), ("plateau", 1001)):
            samples: list[list[float]] = [[], []]
            for repeat in range(7):
                for index in range(2) if repeat % 2 == 0 else reversed(range(2)):
                    samples[index].append(_sample(keys, interval, reference=index == 0)[0])
            scalar, array = map(statistics.median, samples)
            print(
                f"n={dimension} workload={workload} reference_s={scalar:.6f} "
                f"production_s={array:.6f} speedup={scalar / array:.3f}x",
                flush=True,
            )
            for reference in (True, False):
                _, current, peak = _sample(keys, interval, reference, traced=True)
                print(
                    f"  reference={reference} current_bytes={current} peak_bytes={peak}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
