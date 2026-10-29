"""Measure whether `engine="auto"` picks an engine as good as the best fixed one.

Objective:
- Compare automatic selection with every fixed candidate on problems defined as Python
  callbacks and as Numba objectives, at equal total compute time, and report where the
  choice was right, where it cost quality and what it chose.

Technical approach:
- Generate deterministic instances of four problem types: scalar-callback Knapsack and
  TSP (the interpreter lock serializes them), a heavy NumPy objective whose matrix products
  release the lock, and a Numba TSP objective.
- For each instance and replicate run auto first, then give every fixed candidate the same
  total seconds (search plus the measured selection time), in alternating order.
- Score by the gap to the best cost seen on the instance and compare auto with the default
  candidate and with the a-posteriori best fixed candidate (an optimistic reference).

Flow (pseudocode):
1. Build the instance and its decoder/cost (or Numba function) from a seed.
2. Run auto, record its choice and selection seconds, then run each fixed candidate.
3. Write one row per run and print per-case medians, ranks and choice frequencies.

Important criteria:
- Instance seeds are separate from every seed used to tune the engine.
- Auto is never given more total seconds than the fixed candidates it is compared with.
- A failed run stays in the output as an error row; nothing is dropped silently.

Inputs/Outputs:
- Input: search seconds, selection seconds, instances, replicates and an output path.
- Output: a raw CSV and a printed summary table.
"""

import argparse
import csv
import numpy as np
import os
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from rko_solver import RKO
from rko_solver import Numba

INSTANCE_SEED = 20261120
CALIBRATION_SEED = 20261120
FIXED = {
    "process": {"engine": "python", "execution": "process"},
    "rust": {"engine": "rust"},
    "rust-threads": {"engine": "rust", "n_jobs": max(2, os.process_cpu_count() or 2)},
    "cpp": {"engine": "cpp"},
}
CASES = ("knapsack-callbacks", "tsp-callbacks", "heavy-numpy", "heavy-python", "tsp-numba")


class KnapsackCallbacks:
    """Scalar-callback 0/1 Knapsack with a penalized negative profit.

    Responsibilities:
    - Hold one deterministic instance and expose a decoder and an objective.

    Attributes:
        profits (np.ndarray): Item profits.
        weights (np.ndarray): Item weights.
        capacity (int): Knapsack capacity.
    """

    def __init__(self, seed: int, size: int = 300) -> None:
        """Generate an instance.

        Args:
            seed (int): Instance seed.
            size (int): Number of items.
        """
        rng = np.random.default_rng(seed)
        self.profits = rng.integers(10, 1000, size)
        self.weights = rng.integers(5, 500, size)
        self.capacity = int(self.weights.sum() * 0.3)

    def decode(self, keys: np.ndarray) -> np.ndarray:
        """Select the items whose key exceeds one half.

        Args:
            keys (np.ndarray): Random keys.

        Returns:
            np.ndarray: Boolean selection.
        """
        return np.asarray(keys) > 0.5

    def evaluate(self, selection: np.ndarray) -> float:
        """Score a selection.

        Args:
            selection (np.ndarray): Boolean selection.

        Returns:
            float: Penalized negative profit.
        """
        chosen = np.asarray(selection)
        weight = int(self.weights[chosen].sum())
        return float(100000 * max(0, weight - self.capacity) - self.profits[chosen].sum())


class TspCallbacks:
    """Scalar-callback Euclidean TSP decoded by sorting the keys.

    Responsibilities:
    - Hold one deterministic instance and expose a decoder and an objective.

    Attributes:
        distances (np.ndarray): Rounded pairwise distances.
    """

    def __init__(self, seed: int, size: int = 100) -> None:
        """Generate an instance.

        Args:
            seed (int): Instance seed.
            size (int): Number of cities.
        """
        rng = np.random.default_rng(seed)
        points = rng.uniform(0, 1000, (size, 2))
        delta = points[:, None, :] - points[None, :, :]
        self.distances = np.floor(np.sqrt((delta**2).sum(axis=2)) + 0.5)

    def decode(self, keys: np.ndarray) -> np.ndarray:
        """Sort the keys into a tour.

        Args:
            keys (np.ndarray): Random keys.

        Returns:
            np.ndarray: City order.
        """
        return np.argsort(np.asarray(keys), kind="stable")

    def evaluate(self, order: np.ndarray) -> float:
        """Measure the closed tour.

        Args:
            order (np.ndarray): City order.

        Returns:
            float: Tour length.
        """
        return float(self.distances[order, np.roll(order, -1)].sum())


class HeavyNumpy:
    """Quadratic objective dominated by a matrix product that releases the interpreter lock.

    Responsibilities:
    - Hold one deterministic matrix and expose a decoder and an objective.

    Attributes:
        matrix (np.ndarray): Dense matrix applied to the decoded vector.
    """

    def __init__(self, seed: int, size: int = 400, rows: int = 1500) -> None:
        """Generate an instance.

        Args:
            seed (int): Instance seed.
            size (int): Key dimension.
            rows (int): Matrix rows.
        """
        rng = np.random.default_rng(seed)
        self.matrix = rng.normal(0, 1, (rows, size))
        self.target = rng.normal(0, 1, rows)

    def decode(self, keys: np.ndarray) -> np.ndarray:
        """Map keys to the interval [-2, 2].

        Args:
            keys (np.ndarray): Random keys.

        Returns:
            np.ndarray: Decoded vector.
        """
        return np.asarray(keys) * 4.0 - 2.0

    def evaluate(self, vector: np.ndarray) -> float:
        """Measure the squared residual.

        Args:
            vector (np.ndarray): Decoded vector.

        Returns:
            float: Residual sum of squares.
        """
        residual = self.matrix @ vector - self.target
        return float(residual @ residual)


class HeavyPython:
    """Pure-Python objective whose bytecode loop holds the interpreter lock for about a millisecond.

    Responsibilities:
    - Hold one deterministic instance and expose a decoder and an objective.

    Attributes:
        weights (list[float]): Per-position coefficients.
    """

    def __init__(self, seed: int, size: int = 150) -> None:
        """Generate an instance.

        Args:
            seed (int): Instance seed.
            size (int): Key dimension.
        """
        rng = np.random.default_rng(seed)
        self.weights = [float(value) for value in rng.uniform(0.5, 2.0, size)]

    def decode(self, keys: np.ndarray) -> list[float]:
        """Convert the keys to plain Python floats.

        Args:
            keys (np.ndarray): Random keys.

        Returns:
            list[float]: Decoded values.
        """
        return [float(key) for key in keys]

    def evaluate(self, values: list[float]) -> float:
        """Run a nested pure-Python loop over the values.

        Args:
            values (list[float]): Decoded values.

        Returns:
            float: Objective to minimize.
        """
        total = 0.0
        for index, value in enumerate(values):
            inner = 0.0
            for step in range(1, 40):
                inner += abs(value - step / 40.0) * self.weights[(index + step) % len(values)]
            total += inner
        return total


def numba_tsp(seed: int, size: int = 100) -> object:
    """Build a Numba objective for a TSP instance.

    Args:
        seed (int): Instance seed.
        size (int): Number of cities.

    Returns:
        object: `@njit` function mapping a key array to the tour length.
    """
    import numba

    distances = TspCallbacks(seed, size).distances

    @numba.njit(cache=False)
    def objective(keys: np.ndarray) -> float:
        order = np.argsort(keys)
        total = 0.0
        for index in range(order.shape[0]):
            total += distances[order[index], order[(index + 1) % order.shape[0]]]
        return total

    return objective


class AutoEngineStudy:
    """Run the comparison and summarize it.

    Responsibilities:
    - Build solvers for auto and for each fixed candidate with equal total time.
    - Write raw rows and print per-case medians, ranks and choice frequencies.

    Attributes:
        seconds (float): Search seconds given to auto's final run.
        auto_time (float | None): Selection budget requested from auto, or its default.
        instances (int): Instances per case.
        repetitions (int): Replicates per instance.
        output (Path): Raw CSV destination.
        cases (tuple[str, ...]): Cases to run.
    """

    def __init__(
        self,
        output: Path,
        seconds: float,
        auto_time: float | None,
        instances: int,
        repetitions: int,
        cases: Sequence[str] = CASES,
        instance_seed: int = INSTANCE_SEED,
    ) -> None:
        """Store options.

        Args:
            output (Path): Raw CSV destination.
            seconds (float): Search seconds of auto's final run.
            auto_time (float | None): Selection budget requested from auto, or its default.
            instances (int): Instances per case.
            repetitions (int): Replicates per instance.
            cases (Sequence[str]): Case names to run.
            instance_seed (int): First instance seed; use a new one for held-out evaluation.

        Raises:
            ValueError: If a case is unknown or a count is not positive.
        """
        if any(case not in CASES for case in cases) or min(instances, repetitions) < 1:
            raise ValueError(f"cases must be among {CASES} and counts positive")
        self.output = output
        self.seconds = seconds
        self.auto_time = auto_time
        self.instances = instances
        self.repetitions = repetitions
        self.cases = tuple(cases)
        self.instance_seed = instance_seed

    @staticmethod
    def solver(case: str, seed: int, **options: object) -> RKO:
        """Create a solver for one instance of a case.

        Args:
            case (str): Case name.
            seed (int): Instance seed.
            **options (object): Engine options passed to the constructor.

        Returns:
            RKO: Configured solver.
        """
        if case == "tsp-numba":
            return RKO(Numba(numba_tsp(seed), 100), **options)
        problem = {
            "knapsack-callbacks": (KnapsackCallbacks, 300),
            "tsp-callbacks": (TspCallbacks, 100),
            "heavy-numpy": (HeavyNumpy, 400),
            "heavy-python": (HeavyPython, 150),
        }[case]
        instance = problem[0](seed)
        return RKO(instance.decode, instance.evaluate, problem[1], **options)

    def strategies(self, case: str) -> list[str]:
        """List the fixed candidates that can run a case.

        Args:
            case (str): Case name.

        Returns:
            list[str]: Fixed strategy names.
        """
        from rko_solver import _native

        names = ["rust", "cpp"] if case == "tsp-numba" else list(FIXED)
        return [name for name in names if name != "cpp" or _native.cpp_available()]

    def run(self) -> list[dict[str, object]]:
        """Execute every run and write the raw rows.

        Returns:
            list[dict[str, object]]: Raw rows.

        Raises:
            OSError: If the output cannot be written.
        """
        rows: list[dict[str, object]] = []
        for case in self.cases:
            for index in range(self.instances):
                instance_seed = self.instance_seed + index
                for replicate in range(self.repetitions):
                    seed = 1000 * index + replicate
                    started = time.perf_counter()
                    outcome = self.attempt(case, instance_seed, "auto", seed, self.seconds, True)
                    rows.append(
                        {**outcome, "case": case, "instance": index, "replicate": replicate}
                    )
                    spent = outcome["selection_seconds"] or 0.0
                    names = self.strategies(case)
                    shift = replicate % len(names)
                    for name in names[shift:] + names[:shift]:
                        result = self.attempt(
                            case, instance_seed, name, seed, self.seconds + float(spent), False
                        )
                        rows.append(
                            {**result, "case": case, "instance": index, "replicate": replicate}
                        )
                    print(
                        f"{case} i={index} r={replicate} "
                        f"auto={outcome['chosen']} {time.perf_counter() - started:.0f}s",
                        flush=True,
                    )
        self.write(rows)
        return rows

    def attempt(
        self, case: str, instance_seed: int, name: str, seed: int, seconds: float, auto: bool
    ) -> dict[str, object]:
        """Run one strategy once.

        Args:
            case (str): Case name.
            instance_seed (int): Instance seed.
            name (str): `auto` or a fixed strategy name.
            seed (int): Solver seed.
            seconds (float): Search seconds.
            auto (bool): Whether this is the automatic run.

        Returns:
            dict[str, object]: Strategy, cost, chosen candidate, selection seconds and error.
        """
        options: dict[str, object] = {"engine": "auto"} if auto else dict(FIXED[name])
        if auto and self.auto_time is not None:
            options["auto_time"] = self.auto_time
        began = time.perf_counter()
        try:
            result = self.solver(case, instance_seed, random_state=seed, **options).solve(seconds)
            return {
                "strategy": name,
                "cost": result.cost,
                "chosen": result.auto.chosen if result.auto else name,
                "selection_seconds": result.auto.selection_seconds if result.auto else 0.0,
                "elapsed": time.perf_counter() - began,
                "error": "",
            }
        except Exception as failure:
            return {
                "strategy": name,
                "cost": "",
                "chosen": "",
                "selection_seconds": 0.0,
                "elapsed": time.perf_counter() - began,
                "error": f"{type(failure).__name__}: {failure}",
            }

    def write(self, rows: Sequence[dict[str, object]]) -> None:
        """Write the raw CSV.

        Args:
            rows (Sequence[dict[str, object]]): Rows to persist.

        Raises:
            OSError: If the file cannot be written.
        """
        fields = [
            "case",
            "instance",
            "replicate",
            "strategy",
            "cost",
            "chosen",
            "selection_seconds",
            "elapsed",
            "error",
        ]
        with self.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def summarize(rows: Sequence[dict[str, object]]) -> str:
        """Print per-case medians of the gap to the best cost, ranks and choices.

        Args:
            rows (Sequence[dict[str, object]]): Raw rows with finite costs for successes.

        Returns:
            str: Plain-text summary.
        """
        lines: list[str] = []
        for case in dict.fromkeys(str(row["case"]) for row in rows):
            subset = [row for row in rows if row["case"] == case and row["cost"] != ""]
            errors = [row for row in rows if row["case"] == case and row["cost"] == ""]
            best: dict[int, float] = {}
            for row in subset:
                key = int(row["instance"])
                best[key] = min(best.get(key, float("inf")), float(row["cost"]))
            strategies = list(dict.fromkeys(str(row["strategy"]) for row in subset))
            lines.append(f"\n[{case}] median gap to the best cost on the instance, percent")
            medians: dict[str, float] = {}
            for name in strategies:
                gaps = [
                    100
                    * (float(row["cost"]) - best[int(row["instance"])])
                    / abs(best[int(row["instance"])])
                    for row in subset
                    if row["strategy"] == name
                ]
                medians[name] = statistics.median(gaps)
            fixed = {name: value for name, value in medians.items() if name != "auto"}
            ranking = sorted(medians, key=lambda name: medians[name])
            lines.append(
                "  "
                + "   ".join(f"{name} {medians[name]:.3f}" for name in ranking)
                + f"   | best fixed: {min(fixed, key=fixed.get)}"
            )
            choices: dict[str, int] = {}
            for row in subset:
                if row["strategy"] == "auto":
                    choices[str(row["chosen"])] = choices.get(str(row["chosen"]), 0) + 1
            lines.append(f"  auto chose: {choices}; failed runs: {len(errors)}")
        return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv (Sequence[str] | None): Optional argument vector.

    Returns:
        int: Zero on success.

    Raises:
        OSError: If the output cannot be written.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--auto-time", type=float, default=None)
    parser.add_argument("--instance-seed", type=int, default=INSTANCE_SEED)
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--summary-only", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.summary_only:
        with arguments.output.open(encoding="utf-8", newline="") as stream:
            print(AutoEngineStudy.summarize(list(csv.DictReader(stream))))
        return 0
    study = AutoEngineStudy(
        arguments.output,
        arguments.seconds,
        arguments.auto_time,
        arguments.instances,
        arguments.repetitions,
        arguments.cases.split(","),
        arguments.instance_seed,
    )
    print(study.summarize(study.run()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
