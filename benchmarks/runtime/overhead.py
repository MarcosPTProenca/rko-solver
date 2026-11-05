"""Framework overhead profile of the Rust engine with a constant-time objective.

Objective:
- Measure how many objective evaluations per second each algorithm sustains when the objective
  costs O(1), so the figure is dominated by framework work (pool, operators, local search).

Technical approach:
- Compile `f(keys) = keys[0]` with Numba, which does not read the other keys, and run one
  algorithm at a time on the Rust engine for a fixed wall-clock budget at each dimension.
- Rate = completed evaluations divided by the search's own elapsed seconds, from the result.
- When `perf` exists, profile the `rko_bench` binary on a shifted Rastrigin file of the same
  dimension and print the five heaviest symbols; when `cargo flamegraph` exists, also write an
  SVG. Otherwise print the commands that produce them and continue without failing.

Flow (pseudocode):
1. For each dimension and algorithm, build the problem and solve for the budget.
2. Record evaluations, search seconds and evaluations per second as one JSON line.
3. Print the table, then run or describe the profilers.

Important criteria:
- The objective is O(1) in n, so evaluations per second falling with n is framework cost.
- Missing profilers never fail the run.
- A single rate is one sample; the table states the machine's CPU count and is not a benchmark.

Inputs/Outputs:
- Input: dimensions, algorithms, seconds per cell and an optional output JSONL path.
- Output: a table on standard output, optional JSONL, optional profiler output files.
"""

from __future__ import annotations

import argparse
import json
import numba
import numpy as np
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from rko_solver import RKO
from rko_solver import Numba

DIMENSIONS = (10, 100, 1000)
ALGORITHMS = ("ILS", "BRKGA", "SA", "GA", "PSO")
ROOT = Path(__file__).resolve().parents[2]


@numba.njit
def constant_time_objective(keys: np.ndarray) -> float:
    """Return the first key, an objective whose cost does not depend on the dimension.

    Args:
        keys (np.ndarray): One-dimensional float64 key array.

    Returns:
        float: The first key.
    """
    return keys[0]


class OverheadProfile:
    """Measure evaluations per second per algorithm and dimension, and profile the engine.

    Responsibilities:
    - Run each cell on a single native thread with a wall-clock budget.
    - Run `perf` or `cargo flamegraph` when present, and document them otherwise.

    Attributes:
        dimensions (tuple[int, ...]): Key counts to measure.
        algorithms (tuple[str, ...]): Algorithm names.
        seconds (float): Search seconds per cell.
        seed (int): Solver seed.
    """

    def __init__(
        self,
        dimensions: Sequence[int],
        algorithms: Sequence[str],
        seconds: float,
        seed: int = 1,
    ) -> None:
        """Store the options.

        Args:
            dimensions (Sequence[int]): Positive key counts.
            algorithms (Sequence[str]): Algorithm names.
            seconds (float): Positive seconds per cell.
            seed (int): Nonnegative solver seed.

        Raises:
            ValueError: If a dimension or the budget is not positive.
        """
        if seconds <= 0 or any(size < 1 for size in dimensions):
            raise ValueError("seconds and dimensions must be positive.")
        self.dimensions = tuple(dimensions)
        self.algorithms = tuple(algorithms)
        self.seconds = seconds
        self.seed = seed

    def measure(self) -> list[dict[str, object]]:
        """Run every cell.

        Returns:
            list[dict[str, object]]: One record per dimension and algorithm.

        Raises:
            ValueError: If a solver rejects an algorithm name.
        """
        records: list[dict[str, object]] = []
        for dimension in self.dimensions:
            problem = Numba(constant_time_objective, dimension)
            for algorithm in self.algorithms:
                result = RKO(
                    problem,
                    algorithms=[algorithm],
                    engine="rust",
                    random_state=self.seed,
                    n_jobs=1,
                ).solve(self.seconds)
                search = result.searches[0]
                evaluations = search.n_evaluations
                records.append(
                    {
                        "dimension": dimension,
                        "algorithm": algorithm,
                        "evaluations": evaluations,
                        "search_seconds": search.elapsed_seconds,
                        "evaluations_per_second": evaluations / search.elapsed_seconds
                        if search.elapsed_seconds > 0
                        else 0.0,
                        "cpus": os.cpu_count(),
                    }
                )
        return records

    @staticmethod
    def _row_cells(records: Sequence[dict[str, object]], algorithm: str) -> dict[int, float]:
        """Map each dimension to the evaluation rate of one algorithm.

        Args:
            records (Sequence[dict[str, object]]): Output of `measure`.
            algorithm (str): Algorithm whose rows are collected.

        Returns:
            dict[int, float]: Evaluations per second keyed by dimension.
        """
        return {
            int(str(record["dimension"])): float(str(record["evaluations_per_second"]))
            for record in records
            if record["algorithm"] == algorithm
        }

    @staticmethod
    def render(records: Sequence[dict[str, object]]) -> str:
        """Format records as a table of evaluations per second.

        Args:
            records (Sequence[dict[str, object]]): Output of `measure`.

        Returns:
            str: Rows by algorithm and columns by dimension.
        """
        dimensions = sorted({int(str(record["dimension"])) for record in records})
        lines = [f"{'algorithm':10}" + "".join(f"{'n=' + str(size):>16}" for size in dimensions)]
        for algorithm in dict.fromkeys(str(record["algorithm"]) for record in records):
            cells = OverheadProfile._row_cells(records, algorithm)
            lines.append(
                f"{algorithm:10}" + "".join(f"{cells[size]:16,.0f}" for size in dimensions)
            )
        return "\n".join([*lines, "evaluations per second, O(1) objective, one native thread"])

    @staticmethod
    def _manual_instructions(algorithm: str) -> str:
        """Describe the manual perf commands when no profiler is installed.

        Args:
            algorithm (str): Algorithm named in the printed commands.

        Returns:
            str: Human-readable instructions for a manual perf run.
        """
        command = (
            f"cargo build --release --locked --bin rko_bench && "
            f"cargo flamegraph --release --bin rko_bench -o flamegraph.svg -- "
            f"INSTANCE 10 --problem rastrigin --algo {algorithm} --threads 1"
        )
        return (
            "perf not found; install linux-perf, then (INSTANCE = file with n then n shifts):\n"
            f"  {command}\n  perf record -g target/release/rko_bench INSTANCE 10 "
            f"--problem rastrigin --algo {algorithm} --threads 1 && perf report --stdio"
        )

    @staticmethod
    def _write_instance(directory: Path, dimension: int) -> Path:
        """Write a shifted Rastrigin instance file and return its path.

        Args:
            directory (Path): Destination directory.
            dimension (int): Key count written to the file.

        Returns:
            Path: The written instance file.

        Raises:
            OSError: If the file cannot be written.
        """
        generator = random.Random(1)
        instance = directory / f"rastrigin-{dimension}.txt"
        shifts = (repr(generator.uniform(-2, 2)) for _ in range(dimension))
        instance.write_text("\n".join([str(dimension), *shifts]) + "\n", encoding="utf-8")
        return instance

    @staticmethod
    def _perf_symbols(directory: Path, arguments: list[str]) -> str:
        """Run perf record then report and return the five heaviest symbols or a failure note.

        Args:
            directory (Path): Directory that holds the perf data file.
            arguments (list[str]): Native binary invocation, binary path first.

        Returns:
            str: The five heaviest symbols, or a message when perf record fails.
        """
        data = directory / "perf.data"
        record = subprocess.run(
            ["perf", "record", "-o", str(data), "-g", "--", *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if record.returncode != 0:
            return f"perf record failed (permissions?): {record.stderr[-300:]}"
        report = subprocess.run(
            ["perf", "report", "-i", str(data), "--stdio", "--no-children", "-g", "none"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        top = [line.strip() for line in report.splitlines() if re.match(r"\s*\d+\.\d+%", line)][:5]
        return "top five symbols:\n  " + "\n  ".join(top)

    @staticmethod
    def _flamegraph_note(directory: Path, arguments: list[str]) -> str:
        """Render a flamegraph when `cargo-flamegraph` is installed and report its path.

        Args:
            directory (Path): Directory that receives `flamegraph.svg`.
            arguments (list[str]): Native binary invocation, binary path first.

        Returns:
            str: Flamegraph path, a failure note, or install guidance when absent.
        """
        if shutil.which("cargo-flamegraph") is None:
            return "\ncargo-flamegraph not found: cargo install flamegraph"
        svg = directory / "flamegraph.svg"
        subprocess.run(
            ["cargo", "flamegraph", "--release", "--bin", "rko_bench", "-o", str(svg), "--",
             *arguments[1:]],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )  # fmt: skip
        return f"\nflamegraph: {svg}" if svg.exists() else "\ncargo flamegraph failed"

    def profile(self, dimension: int, algorithm: str, directory: Path) -> str:
        """Profile the native runner, or describe how to when no profiler exists.

        Args:
            dimension (int): Key count of the generated Rastrigin file.
            algorithm (str): Algorithm to profile.
            directory (Path): Where the instance file and the SVG are written.

        Returns:
            str: The five heaviest symbols, the flamegraph path, or the manual commands.
        """
        if shutil.which("perf") is None:
            return self._manual_instructions(algorithm)
        instance = self._write_instance(directory, dimension)
        build = subprocess.run(
            ["cargo", "build", "--release", "--locked", "--bin", "rko_bench"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if build.returncode != 0:
            return f"native build failed, no profile: {build.stderr[-500:]}"
        binary = ROOT / "target" / "release" / "rko_bench"
        arguments = [
            str(binary), str(instance), "5", "--problem", "rastrigin",
            "--algo", algorithm, "--threads", "1",
        ]  # fmt: skip
        text = self._perf_symbols(directory, arguments)
        if text.startswith("perf record failed"):
            return text
        return text + self._flamegraph_note(directory, arguments)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the overhead profile from the command line.

    Args:
        argv (Sequence[str] | None): Optional argument vector.

    Returns:
        int: Zero on completion, including when no profiler exists.

    Raises:
        OSError: If the output file cannot be written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", default=",".join(map(str, DIMENSIONS)))
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS))
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSONL path.")
    parser.add_argument("--profile-dimension", type=int, default=100)
    parser.add_argument("--profile-algorithm", default="ILS")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument(
        "--quick", action="store_true", help="Dimensions 10 and 100, 1 s, no profile."
    )
    args = parser.parse_args(argv)
    if args.quick:
        args.dimensions, args.seconds, args.no_profile = "10,100", 1.0, True
    study = OverheadProfile(
        [int(item) for item in args.dimensions.split(",")],
        args.algorithms.split(","),
        args.seconds,
    )
    records = study.measure()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(record, allow_nan=False) + "\n" for record in records),
            encoding="utf-8",
        )
    sys.stdout.write(study.render(records) + "\n")
    if not args.no_profile:
        with tempfile.TemporaryDirectory(prefix="rko-overhead-") as directory:
            sys.stdout.write(
                study.profile(args.profile_dimension, args.profile_algorithm, Path(directory))
                + "\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
