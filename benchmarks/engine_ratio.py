"""Throughput ratio between the Rust and Python engines, checked against a reference.

Objective:
- Detect a large loss of native speed, or an accidental slowdown of the engine boundary, by
  comparing how many objective evaluations per second the Rust engine completes relative to the
  Python engine on the same problem, inside one process on one machine.

Technical approach:
- The same 60-item Knapsack is solved by one algorithm at a time for a fixed wall-clock budget:
  the Python engine runs it as NumPy callbacks (sequentially, one search), the Rust engine as
  the built-in problem with one worker. Throughput is completed evaluations divided by the
  search's own elapsed time, so process start-up is excluded.
- Each engine is measured several times after a short warm-up and the median rate is kept.
  The ratio of the two medians is machine-independent to first order, which is why a ratio and
  not an absolute rate is compared with a versioned reference.
- A check fails only when a ratio falls more than the allowed fraction below the reference; a
  higher ratio is reported but never fails.

Flow (pseudocode):
1. Warm both engines up, then measure `repeats` runs per engine and algorithm.
2. Reduce each run set to its median rate and each algorithm to the Rust/Python ratio.
3. Print and optionally write the JSON report, or compare it with a reference file.

Important criteria:
- The reference records the machine that produced it; it is produced on the CI runner, not on a
  development machine, because absolute throughput and cache effects differ between them.
- The default allowed drop is 30 percent, set from the measured run-to-run spread, which is far
  smaller.
- This is a regression gate, not a speed claim; native performance claims need the studies in
  `benchmarks/native_study.py`.

Inputs/Outputs:
- Input: seconds per run, repeats, algorithms, and optionally a reference path and allowed drop.
- Output: a JSON report on standard output and exit status 0 (pass) or 1 (regression).
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import os
import platform
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from rko_solver import RKO
from rko_solver import Callbacks
from rko_solver import problems

DEFAULT_ALGORITHMS = ("SA", "BRKGA", "ILS")


class EngineRatio:
    """Measure Rust and Python throughput on one problem and compare ratios with a reference.

    Responsibilities:
    - Build the shared Knapsack and run one algorithm on either engine for a fixed budget.
    - Reduce repeated runs to median rates and Rust/Python ratios.
    - Decide whether current ratios regressed against a reference by more than a fraction.

    Attributes:
        items (int): Number of Knapsack items.
        profits (np.ndarray): Item profits.
        weights (np.ndarray): Item weights.
        capacity (int): Knapsack capacity.
    """

    def __init__(self, items: int = 60) -> None:
        """Create the deterministic Knapsack instance.

        Args:
            items (int): Number of items, which is also the key dimension.
        """
        self.items = items
        self.profits = np.array([10 + (index * 37) % 91 for index in range(items)], dtype=np.int64)
        self.weights = np.array([5 + (index * 53) % 47 for index in range(items)], dtype=np.int64)
        self.capacity = 700

    def decode(self, keys: np.ndarray) -> np.ndarray:
        """Select the items whose key exceeds one half.

        Args:
            keys (np.ndarray): Random keys.

        Returns:
            np.ndarray: Boolean selection.
        """
        return np.asarray(keys) > 0.5

    def cost(self, selection: np.ndarray) -> float:
        """Return the penalized negative profit of a selection.

        Args:
            selection (np.ndarray): Boolean selection.

        Returns:
            float: Cost to minimize.
        """
        chosen = np.asarray(selection)
        overweight = max(0, int(chosen @ self.weights) - self.capacity)
        return float(100000 * overweight - int(chosen @ self.profits))

    def rate(self, engine: str, algorithm: str, seconds: float) -> float:
        """Run one algorithm for a fixed budget and return its evaluations per second.

        Args:
            engine (str): `python` or `rust`.
            algorithm (str): Registered algorithm name.
            seconds (float): Search budget in seconds.

        Returns:
            float: Completed evaluations divided by the search's elapsed seconds.
        """
        options = {"algorithms": [algorithm], "adaptation": "fixed", "random_state": 1}
        if engine == "python":
            solver = RKO(
                Callbacks(self.decode, self.cost, self.items),
                engine="python",
                execution="sequential",
                **options,
            )
        else:
            native = problems.Knapsack(self.profits.tolist(), self.weights.tolist(), self.capacity)
            solver = RKO(native, engine="rust", n_jobs=1, **options)
        search = solver.solve(seconds).searches[0]
        return search.n_evaluations / search.elapsed_seconds

    def measure(
        self, algorithms: Sequence[str], seconds: float, repeats: int
    ) -> dict[str, dict[str, float]]:
        """Measure median rates and the Rust/Python ratio for every algorithm.

        Args:
            algorithms (Sequence[str]): Algorithm names.
            seconds (float): Search budget of each run.
            repeats (int): Runs per engine and algorithm after the warm-up.

        Returns:
            dict[str, dict[str, float]]: Per algorithm `python_per_second`, `rust_per_second`,
            `ratio` and the two relative spreads (population standard deviation over mean).
        """
        report: dict[str, dict[str, float]] = {}
        for algorithm in algorithms:
            runs = {}
            for engine in ("python", "rust"):
                self.rate(engine, algorithm, min(0.2, seconds))
                runs[engine] = [self.rate(engine, algorithm, seconds) for _ in range(repeats)]
            python, rust = (statistics.median(runs[name]) for name in ("python", "rust"))
            report[algorithm] = {
                "python_per_second": python,
                "rust_per_second": rust,
                "ratio": rust / python,
                "python_spread": statistics.pstdev(runs["python"])
                / statistics.mean(runs["python"]),
                "rust_spread": statistics.pstdev(runs["rust"]) / statistics.mean(runs["rust"]),
            }
        return report

    @staticmethod
    def compare(
        current: dict[str, dict[str, float]],
        reference: dict[str, dict[str, float]],
        max_drop: float,
    ) -> list[str]:
        """List the algorithms whose ratio fell too far below the reference.

        Args:
            current (dict[str, dict[str, float]]): Report from `measure`.
            reference (dict[str, dict[str, float]]): Reference `algorithms` mapping.
            max_drop (float): Largest tolerated fractional loss, such as 0.30.

        Returns:
            list[str]: One message per regression, or per algorithm missing from the report.
        """
        problems_found = []
        for algorithm, expected in reference.items():
            if algorithm not in current:
                problems_found.append(f"{algorithm}: missing from the current measurement")
                continue
            floor = expected["ratio"] * (1.0 - max_drop)
            if current[algorithm]["ratio"] < floor:
                problems_found.append(
                    f"{algorithm}: ratio {current[algorithm]['ratio']:.1f} is below "
                    f"{floor:.1f} ({max_drop:.0%} under the reference {expected['ratio']:.1f})"
                )
        return problems_found

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        """Measure, print the report and optionally compare it with a reference.

        Args:
            argv (Sequence[str] | None): Command-line arguments, or the process arguments.

        Returns:
            int: 0 when no ratio regressed, 1 otherwise.

        Raises:
            SystemExit: If the arguments are invalid.
            OSError: If the reference file cannot be read.
        """
        parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        parser.add_argument("--seconds", type=float, default=1.0)
        parser.add_argument("--repeats", type=int, default=5)
        parser.add_argument("--algorithms", nargs="+", default=list(DEFAULT_ALGORITHMS))
        parser.add_argument("--reference", type=Path, default=None)
        parser.add_argument("--max-drop", type=float, default=0.30)
        args = parser.parse_args(argv)
        report = {
            "schema_version": 1,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "cpus": os.process_cpu_count(),
                "seconds": args.seconds,
                "repeats": args.repeats,
            },
            "algorithms": cls().measure(args.algorithms, args.seconds, args.repeats),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.reference is None:
            return 0
        reference = json.loads(args.reference.read_text(encoding="utf-8"))["algorithms"]
        failures = cls.compare(report["algorithms"], reference, args.max_drop)
        for message in failures:
            print(f"REGRESSION {message}", file=sys.stderr)
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(EngineRatio.main())
