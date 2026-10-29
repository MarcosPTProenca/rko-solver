"""Measure inherited versus launch-limited BLAS teams in real spawned searches.

Objective:
- Diagnose nested native parallelism without changing RKO or the caller environment.

Technical approach:
- Launch fresh interpreters with copied inherited or one-thread native environments.
- Stop population initialization at a fixed objective count using the private search
  boundary; preserve identical candidate streams without asynchronous elite feedback.
- Inspect loaded teams with the benchmark-only threadpoolctl dependency.

Flow (pseudocode):
1. Alternate launch policies over cheap, matrix-vector and matrix-matrix objectives.
2. Run one or four actual dedicated RKO population workers with oversized populations.
3. Check candidate fingerprints, seeds, counts, costs and native-team configuration.

Important criteria:
- Run with: uv run --locked --with threadpoolctl python -m benchmarks.runtime.native_threads
- Fixed initialization prefixes are not complete generations or quality benchmarks.
- Callback CPU includes native threads but excludes manager/coordinator CPU.
- Warm-up and imports are excluded from callback timing, not search/end-to-end timing.
- BLAS reduction order can change floating-point results; compare costs tolerantly.
- The private stopping signal is benchmark instrumentation, not a supported callback API.

Inputs/Outputs:
- Input: positive dimension, 1..127 evaluations, repeats and 1..4 workers.
- Output: JSONL environment provenance, validated trials and a parent-only negative control.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import numpy as np
import os
import platform
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import field
from importlib.metadata import version
from pathlib import Path
from rko_solver import RKO
from rko_solver.orchestration.search import _BudgetExpired
from threadpoolctl import threadpool_info
from threadpoolctl import threadpool_limits
from typing import Any

THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
ALGORITHMS = ("GA", "BRKGA", "BRKGA_CS", "PSO")
POPULATION_KEYS = ("size_pop", "p", "p", "p_size")


@dataclass
class PopulationCost:
    """Observe a fixed population prefix and time deterministic numerical callbacks.

    Responsibilities:
    - Keep setup/warm-up outside callback measurements and record detached observations.
    - Terminate only this worker's search after exactly the requested finite evaluations.

    Attributes:
        directory (str): Trial-specific destination for per-algorithm observations.
        workload (str): Sphere, matrix-vector or matrix-matrix objective.
        limit (int): Completed objective evaluations before cooperative instrumentation.
        matrix (np.ndarray | None): Worker-owned deterministic dense matrix.
        calls (int): Completed finite callbacks.
        wall (float): Sum of unprofiled objective wall seconds.
        cpu (float): Sum of objective process CPU seconds, including native threads.
        trace (Any): Incremental candidate fingerprint.
        first (list[float]): First candidate for independent assigned-seed validation.
    """

    directory: str
    workload: str
    limit: int
    matrix: np.ndarray | None = None
    calls: int = 0
    wall: float = 0.0
    cpu: float = 0.0
    trace: Any = None
    first: list[float] = field(default_factory=list)

    def value(self, keys: np.ndarray) -> float:
        """Calculate the objective using a private matrix with a fixed independent seed.

        Args:
            keys (np.ndarray): Candidate continuous random keys.

        Returns:
            float: Squared norm of keys, a projection or a dense matrix product.
        """
        if self.workload == "sphere":
            return float(np.dot(keys, keys))
        if self.matrix is None:
            self.matrix = np.random.RandomState(100).random_sample((len(keys), len(keys)))
        if self.workload == "gemv":
            projected = self.matrix @ keys
            return float(np.dot(projected, projected))
        product = (self.matrix * keys) @ self.matrix
        return float(np.sum(product * product))

    def __call__(self, keys: np.ndarray) -> float:
        """Warm up, measure a finite callback, or publish the completed prefix.

        Args:
            keys (np.ndarray): Candidate chromosome owned by the search boundary.

        Returns:
            float: Finite objective value until the prefix is complete.

        Raises:
            _BudgetExpired: Benchmark-only termination after exactly limit evaluations.
            OSError: If a completed observation cannot be written.
        """
        if not self.first:
            self.first = keys.tolist()
            self.trace = hashlib.sha256()
            for _ in range(3):
                self.value(keys)
        if self.calls == self.limit:
            name = mp.current_process().name.removeprefix("rko-")
            Path(self.directory, f"{name}.json").write_text(
                json.dumps(
                    {
                        "algorithm": name,
                        "evaluations": self.calls,
                        "candidates_sha256": self.trace.hexdigest(),
                        "first_keys": self.first,
                        "callback_wall_seconds": self.wall,
                        "callback_cpu_seconds": self.cpu,
                        "native_teams": threadpool_info(),
                    }
                ),
                encoding="utf-8",
            )
            raise _BudgetExpired
        started, cpu = time.perf_counter(), time.process_time()
        value = self.value(keys)
        self.wall += time.perf_counter() - started
        self.cpu += time.process_time() - cpu
        self.trace.update(np.asarray(keys, dtype="<f8").tobytes())
        self.calls += 1
        return value


def decode(keys: np.ndarray) -> np.ndarray:
    """Preserve continuous keys for these numerical objective adapters.

    Args:
        keys (np.ndarray): Detached continuous candidate.

    Returns:
        np.ndarray: Unmodified candidate.
    """
    return keys


def main() -> None:
    """Run paired launch experiments or a private single-trial subprocess.

    Raises:
        AssertionError: If counts, streams, native caps or independent costs disagree.
        OSError: If trial observations cannot be created or read.
        subprocess.SubprocessError: If a fresh interpreter fails or times out.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--evaluations", type=int, default=64)
    parser.add_argument("--workers", type=int, nargs="+", choices=range(1, 5), default=[1, 4])
    parser.add_argument("--trial", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--parent-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--workload", choices=("sphere", "gemv", "gemm"), default="sphere")
    args = parser.parse_args()
    if args.repeats < 1 or args.dimension < 1 or not 1 <= args.evaluations < 128:
        parser.error("repeats/dimension must be positive; evaluations must be in 1..127")
    if args.trial:
        assert len(args.workers) == 1
        workers = args.workers[0]
        names = ALGORITHMS[:workers]
        with tempfile.TemporaryDirectory(prefix="rko-native-") as directory:
            cost = PopulationCost(directory, args.workload, args.evaluations)
            before = threadpool_info()
            with threadpool_limits(limits=1) if args.parent_only else nullcontext():
                parent_teams = threadpool_info()
                result = RKO(
                    decode,
                    cost,
                    args.dimension,
                    algorithms=names,
                    control=0,
                    n_jobs=workers,
                    random_state=42,
                    hyperparameters={
                        name: {parameter: 128}
                        for name, parameter in zip(names, POPULATION_KEYS[:workers], strict=True)
                    },
                ).solve(120.0)
            assert threadpool_info() == before
            rows = [json.loads(Path(directory, f"{name}.json").read_text()) for name in names]
            assert result.n_evaluations == workers * args.evaluations
            assert result.n_workers == workers
            for row, report in zip(rows, result.searches, strict=True):
                assert row["evaluations"] == report.n_evaluations == args.evaluations
                assert (
                    row["first_keys"]
                    == np.random.RandomState(report.random_state)
                    .random_sample(args.dimension)
                    .tolist()
                )
                del row["first_keys"]
            assert np.isclose(result.cost, cost.value(np.asarray(result.keys)), rtol=1e-12)
            print(
                json.dumps(
                    {
                        "workload": args.workload,
                        "workers": workers,
                        "dimension": args.dimension,
                        "evaluations_per_worker": args.evaluations,
                        "elapsed_seconds": result.elapsed_seconds,
                        "search_seconds": [report.elapsed_seconds for report in result.searches],
                        "cost": result.cost,
                        "parent_teams": parent_teams,
                        "searches": rows,
                    }
                ),
                flush=True,
            )
        return
    inherited = dict(os.environ)
    parent_teams = threadpool_info()
    print(
        json.dumps(
            {
                "kind": "environment",
                "python": sys.version,
                "numpy": np.__version__,
                "threadpoolctl": version("threadpoolctl"),
                "platform": platform.platform(),
                "available_cpus": os.process_cpu_count(),
                "rko-solver": version("rko-solver"),
                "source": str(Path(__file__).resolve()),
                "dimension": args.dimension,
                "evaluations_per_worker": args.evaluations,
                "repeats": args.repeats,
                "workers": args.workers,
                "parent_teams": parent_teams,
                "inherited_thread_variables": {
                    name: inherited.get(name) for name in THREAD_VARIABLES
                },
            }
        ),
        flush=True,
    )
    worker = [sys.executable, "-m", "benchmarks.runtime.native_threads"]
    for workers in args.workers:
        for workload in ("sphere", "gemv", "gemm"):
            command = [
                *worker,
                "--trial",
                "--workload",
                workload,
                "--workers",
                str(workers),
                "--dimension",
                str(args.dimension),
                "--evaluations",
                str(args.evaluations),
            ]
            for repeat in range(args.repeats):
                modes = ("inherited", "one") if repeat % 2 == 0 else ("one", "inherited")
                pair = []
                for mode in modes:
                    environment = dict(inherited)
                    if mode == "one":
                        environment.update(dict.fromkeys(THREAD_VARIABLES, "1"))
                    process = subprocess.run(
                        command,
                        env=environment,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=180,
                    )
                    row = json.loads(process.stdout)
                    if mode == "one":
                        for report in row["searches"]:
                            teams = report["native_teams"]
                            assert teams, "No recognized native runtime; cannot verify the cap."
                            assert all(team["num_threads"] == 1 for team in teams)
                    pair.append(row)
                    print(json.dumps({"mode": mode, "repeat": repeat, **row}), flush=True)
                assert [r["candidates_sha256"] for r in pair[0]["searches"]] == [
                    r["candidates_sha256"] for r in pair[1]["searches"]
                ]
                assert np.isclose(pair[0]["cost"], pair[1]["cost"], rtol=1e-12)
                assert dict(os.environ) == inherited and threadpool_info() == parent_teams
        control = subprocess.run(
            [*command, "--parent-only"],
            env=inherited,
            capture_output=True,
            text=True,
            check=True,
            timeout=180,
        )
        row = json.loads(control.stdout)
        print(json.dumps({"mode": "parent_only_negative_control", **row}), flush=True)
        for report in row["searches"]:
            assert report["native_teams"] == parent_teams
        assert all(team["num_threads"] == 1 for team in row["parent_teams"])
    assert dict(os.environ) == inherited and threadpool_info() == parent_teams


if __name__ == "__main__":
    main()
