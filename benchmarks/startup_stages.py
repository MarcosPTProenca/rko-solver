"""Real-spawn startup stages for a domain-neutral cooperative portfolio.

Objective:
- Separate parent validation, manager startup and worker preparation latency.
- Identify environment costs without confusing them with solver optimizations.

Technical approach:
- Wrap the existing factory, manager start and worker entry without replacing them.
- Record the first candidate and construction timings in per-process JSON files.
- Report interpreter, dependency and source paths for controlled environment comparisons.

Flow (pseudocode):
1. Capture environment provenance and install temporary timing wrappers.
2. Alternate one/four-worker full-portfolio solves over the requested repetitions.
3. Verify seeded initial vectors, results and complete worker observations.
4. Emit JSONL measurements and restore all patched entry points.

Important criteria:
- Startup and callbacks use real clocks; there are no latency pass/fail thresholds.
- Both environments must use the same source, interpreter version and lockfile.
- Worker entry includes spawn/bootstrap imports, not just Python process creation.
- Unequal slices and real clocks preclude equal-work or solution-quality claims.
- Filesystem, dependency placement and host load are not library-code speedups.

Inputs/Outputs:
- Input: optional positive repetition count; fixed neutral 128-key objective.
- Output: environment record and one timing/accounting record per complete solve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import numpy as np
import os
import rko_solver
import rko_solver.solver as solver_module
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from rko_solver import RKO
from rko_solver.metaheuristics.base import MetaheuristicBase
from rko_solver.orchestration import processes
from rko_solver.orchestration.configuration import parse_run_config
from rko_solver.orchestration.processes import PoolManager
from rko_solver.parameters.loader import load_config

_WORKER = processes._worker


@dataclass
class FirstDecode:
    """Observe the first candidate per worker without altering its payload.

    Responsibilities:
    - Retain one initial-vector fingerprint and its real timestamp.

    Attributes:
        directory (Path): Private per-trial observation directory.
        seen (bool): Whether this worker has already recorded its first vector.
    """

    directory: Path
    seen: bool = False

    def __call__(self, keys: np.ndarray) -> np.ndarray:
        """Write one worker-local observation and return the same candidate.

        Args:
            keys (np.ndarray): Detached candidate vector.

        Returns:
            np.ndarray: Unchanged candidate.

        Raises:
            OSError: If the observation cannot be written.
        """
        if not self.seen:
            self.seen = True
            (self.directory / f"first-{os.getpid()}.json").write_text(
                json.dumps(
                    {
                        "at": time.perf_counter(),
                        "keys_sha256": hashlib.sha256(keys.astype("<f8").tobytes()).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
        return keys


class StartupProbe:
    """Time production spawn stages while preserving searches and their RNGs.

    Responsibilities:
    - Observe parent construction and actual manager startup boundaries.
    - Observe child entry/preparation and validate the first executed stream.
    """

    @staticmethod
    def objective(keys: np.ndarray) -> float:
        """Compute an independently verifiable numerical sphere cost.

        Args:
            keys (np.ndarray): Decoded candidate vector.

        Returns:
            float: Finite squared Euclidean norm.
        """
        return float(np.dot(keys, keys))

    @staticmethod
    def worker(*args: object, **kwargs: object) -> None:
        """Record entry and constructor time around the original spawn worker.

        Args:
            args (object): Original positional worker arguments.
            kwargs (object): Original keyword worker arguments, forwarded unchanged.

        Raises:
            AssertionError: If the expected observation callback is missing.
            OSError: If a worker observation cannot be persisted.
        """
        entered = time.perf_counter()
        decoder = args[3]
        assert isinstance(decoder, FirstDecode)
        calls: list[float] = []
        original = processes.build_metaheuristic

        def build(*values: object, **kwargs: object) -> MetaheuristicBase:
            """Measure an unchanged child-side constructor.

            Args:
                values (object): Original factory positional arguments.
                kwargs (object): Original factory keyword arguments.

            Returns:
                MetaheuristicBase: Original validated algorithm instance.
            """
            before = time.perf_counter()
            algorithm = original(*values, **kwargs)
            calls.append(time.perf_counter() - before)
            return algorithm

        processes.build_metaheuristic = build
        try:
            _WORKER(*args, **kwargs)
        finally:
            processes.build_metaheuristic = original
            (decoder.directory / f"prepare-{os.getpid()}.json").write_text(
                json.dumps(
                    {
                        "entered": entered,
                        "construction_seconds": sum(calls),
                        "first_algorithm": args[0][0][0],
                    }
                ),
                encoding="utf-8",
            )

    @staticmethod
    def run(repeats: int) -> None:
        """Measure alternating worker caps with unchanged production execution.

        Args:
            repeats (int): Positive number of one/four-worker trial pairs.

        Raises:
            AssertionError: If observations, first seeds or final results disagree.
            OSError: If observation files cannot be accessed.
            RuntimeError: If the production portfolio fails.
        """
        parent_calls: list[float] = []
        managers: list[tuple[float, float]] = []
        original_build = solver_module.build_metaheuristic
        original_start = PoolManager.start
        original_worker = processes._worker

        def build(*values: object, **kwargs: object) -> MetaheuristicBase:
            """Measure the existing parent validation constructor.

            Args:
                values (object): Original factory positional arguments.
                kwargs (object): Original factory keyword arguments.

            Returns:
                MetaheuristicBase: Original validated algorithm instance.
            """
            before = time.perf_counter()
            algorithm = original_build(*values, **kwargs)
            parent_calls.append(time.perf_counter() - before)
            return algorithm

        def start(manager: PoolManager, *values: object, **kwargs: object) -> None:
            """Measure the existing manager's synchronous startup.

            Args:
                manager (PoolManager): Actual production manager.
                values (object): Original startup positional arguments.
                kwargs (object): Original startup keyword arguments.
            """
            before = time.perf_counter()
            original_start(manager, *values, **kwargs)
            managers.append((before, time.perf_counter()))

        solver_module.build_metaheuristic = build
        PoolManager.start = start
        processes._worker = StartupProbe.worker
        try:
            names = parse_run_config(load_config()).active_metaheuristics
            print(
                json.dumps(
                    {
                        "kind": "environment",
                        "python": sys.version,
                        "prefix": sys.prefix,
                        "numpy": np.__version__,
                        "numpy_source": str(Path(np.__file__).resolve()),
                        "rko_source": str(Path(rko_solver.__file__).resolve()),
                        "cwd": str(Path.cwd()),
                    }
                ),
                flush=True,
            )
            for repeat in range(repeats):
                for workers in (1, 4) if repeat % 2 == 0 else (4, 1):
                    parent_calls.clear()
                    managers.clear()
                    with tempfile.TemporaryDirectory(prefix="rko-startup-stages-") as temporary:
                        directory = Path(temporary)
                        before = time.perf_counter()
                        result = RKO(
                            FirstDecode(directory),
                            StartupProbe.objective,
                            128,
                            algorithms=names,
                            control=1,
                            n_jobs=workers,
                            random_state=42,
                        ).solve(0.2)
                        assert len(managers) == 1
                        reports = {report.algorithm: report for report in result.searches}
                        first = []
                        prepared = []
                        for path in directory.glob("prepare-*.json"):
                            observation = json.loads(path.read_text(encoding="utf-8"))
                            pid = path.stem.split("-")[1]
                            candidate = json.loads(
                                (directory / f"first-{pid}.json").read_text(encoding="utf-8")
                            )
                            seed = reports[observation["first_algorithm"]].random_state
                            expected = np.random.RandomState(seed).random_sample(128)
                            assert (
                                candidate["keys_sha256"]
                                == hashlib.sha256(expected.astype("<f8").tobytes()).hexdigest()
                            )
                            first.append(candidate["at"] - before)
                            prepared.append(observation)
                        assert len(first) == len(prepared) == workers
                        assert tuple(reports) == names
                        assert result.n_workers == workers and result.n_evaluations > 0
                        keys = np.asarray(result.keys)
                        assert len(keys) == 128 and np.all((keys >= 0) & (keys < 1))
                        assert np.isclose(result.cost, StartupProbe.objective(keys))
                        print(
                            json.dumps(
                                {
                                    "kind": "solve",
                                    "workers": workers,
                                    "repeat": repeat,
                                    "budget_seconds": 0.2,
                                    "elapsed_seconds": result.elapsed_seconds,
                                    "parent_construction_seconds": sum(parent_calls),
                                    "manager_seconds": managers[0][1] - managers[0][0],
                                    "worker_entry_after_manager_seconds": [
                                        item["entered"] - managers[0][1] for item in prepared
                                    ],
                                    "worker_construction_seconds": [
                                        item["construction_seconds"] for item in prepared
                                    ],
                                    "first_callback_seconds": first,
                                    "seeded_workers_checked": len(first),
                                    "n_evaluations": result.n_evaluations,
                                }
                            ),
                            flush=True,
                        )
        finally:
            solver_module.build_metaheuristic = original_build
            PoolManager.start = original_start
            processes._worker = original_worker


def main() -> None:
    """Parse a bounded repetition count and run the guarded spawn benchmark.

    Raises:
        SystemExit: If the CLI repetition count is not positive.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    StartupProbe.run(args.repeats)


if __name__ == "__main__":
    main()
