"""Measure fixed-work decision-recording overhead without timed-search quality claims.

Objective:
- Compare disabled/enabled generic callback boundaries at equal completed evaluation counts.

Technical approach:
- Run identical cheap/heavier vector callbacks through private SearchSession boundaries.
- Emit three constant authored steps, retain bounded records and sample Python allocation peaks.
- Report raw repeated elapsed time and traced allocation observations with runtime versions.

Flow (pseudocode):
1. Validate small explicit work/dimension/repetition settings.
2. Warm each disabled/enabled callback boundary without changing the timed workload.
3. Execute exactly the requested finite objectives and measure elapsed/allocation peak.
4. Print source/runtime/configuration-labelled JSON without general speedup claims.

Important criteria:
- This is recording overhead, not a quality benchmark or uninstrumented-engine baseline.
- Allocation sampling is a separate equal-work pass and excludes NumPy native allocations.
- No report replay, process startup, domain tuning or unbounded event archive is measured.

Inputs/Outputs:
- Input: positive bounded evaluation/dimension/repetition counts.
- Output: reproducible raw disabled/enabled measurements for two generic workload sizes.
"""

import argparse
import hashlib
import json
import numpy as np
import pickle
import platform
import time
import tracemalloc
from .performance_profile import PerformanceProbe
from dataclasses import asdict
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from rko_solver.orchestration.search import SearchSession
from rko_solver.tracing import TraceConfig
from rko_solver.tracing import record_decision


@dataclass
class CallbackWork:
    """Provide fixed generic vector work with explicit nonprivate event labels.

    Responsibilities:
    - Separate recording costs from cheap/heavier objective computation at equal work.

    Attributes:
        repetitions (int): Dot products per original objective, not extra evaluations.
        calls (int): Completed original objective calls.
    """

    repetitions: int
    calls: int = 0

    def decode(self, keys: np.ndarray) -> np.ndarray:
        """Emit three authored constant steps without exposing input coordinates.

        Args:
            keys (np.ndarray): Candidate vector copied by the shared boundary.

        Returns:
            np.ndarray: Original vector payload, not a recorder snapshot.
        """
        record_decision("begin", "Read the supplied in-memory vector.")
        record_decision("choose", "Use the application-defined vector construction.")
        record_decision("return", "Return the numerical payload without logging its values.")
        return keys

    def cost(self, payload: np.ndarray) -> float:
        """Complete one finite objective with a configurable amount of generic vector work.

        Args:
            payload (np.ndarray): Exact original decoder payload.

        Returns:
            float: Finite generic objective.
        """
        value = 0.0
        for _ in range(self.repetitions):
            value += float(np.dot(payload, payload))
        self.calls += 1
        return value


class TraceOverhead:
    """Collect equal-work raw timing/Python-allocation observations, not performance gates.

    Responsibilities:
    - Keep disabled/enabled runs bounded and count original evaluations explicitly.

    Attributes:
        None: Each repetition owns fresh callbacks and private recording storage.
    """

    @staticmethod
    def measure(evaluations: int, dimension: int, work: int, enabled: bool) -> dict[str, object]:
        """Measure timing and Python allocation peaks in separate equal-work passes.

        Args:
            evaluations (int): Exact completed objective count.
            dimension (int): Fixed candidate dimension.
            work (int): Vector dot products per objective.
            enabled (bool): Whether to allocate bounded original-decision recording.

        Returns:
            dict[str, object]: Raw elapsed/peak/count/cap observations.

        Raises:
            AssertionError: If the workload count differs from the requested fixed work.
        """
        problem = CallbackWork(work)
        config = TraceConfig() if enabled else None
        session = SearchSession(problem.decode, problem.cost, trace=config)
        keys = np.linspace(0.1, 0.9, dimension)
        for _ in range(8):
            session.evaluate(session.decode(keys))
        before = problem.calls
        started = time.perf_counter()
        for _ in range(evaluations):
            session.evaluate(session.decode(keys))
        elapsed = time.perf_counter() - started
        assert problem.calls - before == evaluations
        before = problem.calls
        tracemalloc.start()
        for _ in range(evaluations):
            session.evaluate(session.decode(keys))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        completed = problem.calls - before
        assert completed == evaluations
        return {
            "enabled": enabled,
            "evaluations": completed,
            "dimension": dimension,
            "dot_products_per_evaluation": work,
            "elapsed_seconds": elapsed,
            "python_peak_bytes": peak,
            "retained_incumbent_pickle_bytes": len(pickle.dumps(session.best[1])),
            "trace_caps": asdict(config) if config else None,
        }

    @staticmethod
    def main() -> None:
        """Print bounded repeated measurements and their interpretation limits.

        Raises:
            SystemExit: If requested work exceeds the diagnostic envelope.
        """
        parser = argparse.ArgumentParser(
            description="Measure bounded decoder-decision recording overhead."
        )
        parser.add_argument("--evaluations", type=int, default=1000)
        parser.add_argument("--dimension", type=int, default=64)
        parser.add_argument("--repeats", type=int, default=3)
        args = parser.parse_args()
        if not (
            1 <= args.evaluations <= 100000
            and 1 <= args.dimension <= 4096
            and 1 <= args.repeats <= 10
        ):
            parser.error("Use bounded positive evaluations/dimension/repeats.")
        measurements = [
            TraceOverhead.measure(args.evaluations, args.dimension, work, enabled)
            for _ in range(args.repeats)
            for work in (1, 64)
            for enabled in (False, True)
        ]
        print(
            json.dumps(
                {
                    "schema": 1,
                    "purpose": "fixed-work recording overhead, not search quality",
                    "provenance": PerformanceProbe.provenance(),
                    "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "rko-solver": version("rko-solver"),
                    "clock": "perf_counter",
                    "allocation_scope": "Separate equal-work tracemalloc pass; excludes native RSS",
                    "measurements": measurements,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    TraceOverhead.main()
