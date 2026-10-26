"""Measure bounded portfolio startup, worker counts and sampled resident memory.

Objective:
- Quantify resource use without confusing a search budget with end-to-end time.

Technical approach:
- Run the unchanged ten-algorithm selection with automatic or dedicated workers.
- Identify seeded first decodes and sample parent/child RSS through Linux procfs.

Flow (pseudocode):
1. Alternate worker limits over three seeds and two neutral problem adapters.
2. Record first-decode times, worker PIDs, final accounting and sampled memory.
3. Emit JSON measurements without asserting timed convergence or speedups.

Important criteria:
- Workers divide remaining common-deadline time among their own remaining searches.
- Historical references used fixed longest-group slices, not this allocation policy.
- RSS sums count shared pages multiple times; sampling is not an exact high-water mark.
- Procfs memory measurements are unavailable on non-Linux hosts.

Inputs/Outputs:
- Input: packaged ten-algorithm portfolio, 128 keys, one-second search budget.
- Output: JSON records for constructor-plus-solve latency, internal solve latency,
  worker counts and sampled RSS, or failed assertions.
"""

import json
import multiprocessing as mp
import numpy as np
import os
import platform
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from rko_solver import RKO
from rko_solver.orchestration.configuration import parse_run_config
from rko_solver.parameters.loader import load_config


@dataclass
class Decoder:
    """Record first seeded candidates before applying a numerical or ordering adapter.

    Responsibilities:
    - Identify the search process and first-decode timestamp for every algorithm.

    Attributes:
        directory (Path): Trial-specific observation directory.
        expected (dict): First-key lookup of algorithm indices, names and initial vectors.
        ordering (bool): Decode keys into a stable permutation instead of numerical values.
        seen (set[int]): Algorithms already recorded by this worker.
        assigned (set[str]): This worker's assigned names, resolved on first use.
    """

    directory: Path
    expected: dict[float, tuple[int, str, tuple[float, ...]]]
    ordering: bool
    seen: set[int] = field(default_factory=set)
    assigned: set[str] = field(default_factory=set)

    def __call__(self, keys: np.ndarray) -> np.ndarray:
        """Record first decodes and return the problem representation.

        Args:
            keys (np.ndarray): Candidate random keys.

        Returns:
            np.ndarray: Stable permutation or unchanged numerical keys.

        Raises:
            OSError: If an observation cannot be written.
        """
        if not self.assigned:
            self.assigned = set(mp.current_process().name.removeprefix("rko-").split("/"))
        candidate = self.expected.get(float(keys[0]))
        if candidate is not None:
            index, name, expected = candidate
            if index not in self.seen and name in self.assigned and tuple(keys) == expected:
                self.seen.add(index)
                self.directory.joinpath(f"{index}.json").write_text(
                    json.dumps({"pid": os.getpid(), "at": time.perf_counter()}), encoding="utf-8"
                )
        return np.argsort(keys, kind="stable") if self.ordering else keys


@dataclass
class Objective:
    """Evaluate the numerical sphere or weighted-permutation problem.

    Responsibilities:
    - Keep independent result accounting available outside worker callbacks.

    Attributes:
        ordering (bool): Use positional weights on decoded permutations.
    """

    ordering: bool

    def __call__(self, values: np.ndarray) -> float:
        """Compute a finite minimization objective.

        Args:
            values (np.ndarray): Numerical vector or decoded permutation.

        Returns:
            float: Squared norm or weighted index sum.
        """
        weights = np.arange(1, len(values) + 1) if self.ordering else values
        return float(np.dot(values, weights))


class ResidentMonitor:
    """Sample Linux parent/direct-child RSS without a runtime dependency.

    Responsibilities:
    - Retain time-aligned approximate RSS snapshots until the trial completes.
    - Stop and join the sampling thread even if the solver fails.

    Attributes:
        samples (list[dict[int, int]]): Per-process resident bytes at each sample.
        available (bool): Whether procfs supplies this process's RSS.
        parent_pid (int): Solver coordinator PID.
        _children (Path): Main thread's direct-child PID list.
        _stop (threading.Event): Sampling cancellation signal.
        _thread (threading.Thread): Trial-owned sampler.
    """

    def __init__(self) -> None:
        """Prepare a stopped sampler and probe whether procfs memory data is available."""
        self.parent_pid = os.getpid()
        self._children = Path(f"/proc/{self.parent_pid}/task/{self.parent_pid}/children")
        self.available = self._read(self.parent_pid) > 0 and self._children.is_file()
        self.samples: list[dict[int, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, name="rss-probe", daemon=True)

    @staticmethod
    def _read(pid: int) -> int:
        """Read a live process's RSS, or zero when unavailable or already exited.

        Args:
            pid (int): Process identifier.

        Returns:
            int: Resident bytes, or zero for an unavailable observation.
        """
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            pass
        return 0

    def _sample(self) -> None:
        """Collect parent and child observations approximately every 20 milliseconds."""
        while not self._stop.is_set():
            try:
                pids = [self.parent_pid, *map(int, self._children.read_text().split())]
                self.samples.append({pid: self._read(pid) for pid in pids})
            except OSError:
                pass
            self._stop.wait(0.02)

    def start(self) -> None:
        """Start sampling when supported.

        Raises:
            RuntimeError: If the operating system cannot start the sampler.
        """
        if self.available:
            self._thread.start()

    def stop(self) -> None:
        """Cancel and join the trial's sampler when started."""
        if self.available:
            self._stop.set()
            self._thread.join()


def main() -> None:
    """Run alternating worker-limit trials and independently validate every result.

    Raises:
        AssertionError: If searches are omitted, worker limits fail or costs disagree.
        OSError: If trial observations cannot be read or written.
        RuntimeError: If a solver worker fails.
        TimeoutError: If process startup or search exceeds its grace period.
    """
    names = parse_run_config(load_config()).active_metaheuristics
    assert len(names) == 10
    dimension, budget = 128, 1.0
    print(
        json.dumps(
            {
                "python": sys.version,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "cpus": os.process_cpu_count(),
                "algorithms": names,
                "dimension": dimension,
                "search_seconds": budget,
                "rss_sample_seconds": 0.02,
            }
        ),
        flush=True,
    )
    for workload in ("sphere", "ordering"):
        for trial, seed in enumerate((42, 43, 44)):
            limits = (None, len(names)) if trial % 2 == 0 else (len(names), None)
            initial_keys = [
                np.random.RandomState(int(stream.generate_state(1)[0])).random_sample(dimension)
                for stream in np.random.SeedSequence(seed).spawn(len(names))
            ]
            expected = {
                float(keys[0]): (index, name, tuple(keys.tolist()))
                for index, (name, keys) in enumerate(zip(names, initial_keys, strict=True))
            }
            assert len(expected) == len(names)
            for limit in limits:
                with tempfile.TemporaryDirectory(prefix="rko-process-budget-") as directory:
                    monitor = ResidentMonitor()
                    decoder = Decoder(Path(directory), expected, workload == "ordering")
                    objective = Objective(workload == "ordering")
                    monitor.start()
                    started = time.perf_counter()
                    try:
                        result = RKO(
                            decoder,
                            objective,
                            dimension,
                            algorithms=names,
                            control=0,
                            n_jobs=limit,
                            random_state=seed,
                        ).solve(budget)
                        invocation_seconds = time.perf_counter() - started
                    finally:
                        monitor.stop()
                    assert invocation_seconds >= result.elapsed_seconds
                    assert (
                        sum(report.n_evaluations for report in result.searches)
                        == result.n_evaluations
                    )
                    records = [
                        json.loads(Path(directory, f"{index}.json").read_text())
                        for index in range(len(names))
                    ]
                    pids = {row["pid"] for row in records}
                    expected_workers = min(limit or max(1, os.process_cpu_count() or 1), len(names))
                    assert len(pids) == expected_workers == result.n_workers
                    assert result.n_jobs == limit
                    assert result.available_cpus == max(1, os.process_cpu_count() or 1)
                    assert all(0.0 <= key < 1.0 for key in result.keys)
                    keys = np.asarray(result.keys)
                    decoded = np.argsort(keys, kind="stable") if objective.ordering else keys
                    assert np.isfinite(result.cost)
                    assert result.cost == objective(decoded)
                    first = min(row["at"] for row in records)
                    last = max(row["at"] for row in records)
                    peak_workers = max(
                        (sum(sample.get(pid, 0) for pid in pids) for sample in monitor.samples),
                        default=None,
                    )
                    peak_all = max(
                        (sum(sample.values()) for sample in monitor.samples), default=None
                    )
                    print(
                        json.dumps(
                            {
                                "workload": workload,
                                "seed": seed,
                                "n_jobs": limit,
                                "search_workers": len(pids),
                                "available_cpus": result.available_cpus,
                                "n_evaluations": result.n_evaluations,
                                "invocation_seconds": invocation_seconds,
                                "elapsed_seconds": result.elapsed_seconds,
                                "searches": [
                                    {
                                        "algorithm": report.algorithm,
                                        "random_state": report.random_state,
                                        "n_evaluations": report.n_evaluations,
                                        "elapsed_seconds": report.elapsed_seconds,
                                        "time_limit": report.time_limit,
                                        "stop_reason": report.stop_reason,
                                    }
                                    for report in result.searches
                                ],
                                "first_decode_seconds": first - started,
                                "last_first_decode_seconds": last - started,
                                "sampled_peak_worker_rss_bytes": peak_workers,
                                "sampled_peak_parent_children_rss_bytes": peak_all,
                                "rss_samples": len(monitor.samples),
                                "cost": result.cost,
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
