"""Run RKO on C++ KP benchmark instances using a Python reader/decoder.

Objective:
- Execute the Python RKO implementation on Knapsack Problem instances from
  `RKO_Cpp_v2.0/Instances/KP`.

Technical approach:
- Parse the C++ KP text format (`n_items`, `capacity`, then `benefit weight`).
- Decode keys with the C++ threshold rule (`rk > 0.5`).
- Evaluate cost exactly like `Problem.h` and persist execution summaries to CSV.

Flow (pseudocode):
1. Resolve instance files from CLI path or default C++ folder.
2. Read each instance and build pickleable decoder/cost callbacks.
3. Construct RKO with the instance dimension and call solve.
4. Decode best keys, print summary, and save rows to CSV.

Important criteria:
- Cost mirrors C++ sign convention and penalty magnitude.
- Decoder and reader are standalone and reusable.

Inputs/Outputs:
- Input: KP `.txt` files, time limit, and optional params directory.
- Output: console summary plus CSV with objective metrics per instance.
"""

from __future__ import annotations

import argparse
import csv
import numpy as np
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from rko_solver import RKO
from rko_solver import Callbacks
from rko_solver import problems
from rko_solver.orchestration.engine import EngineSelection
from rko_solver.solver import Engine
from rko_solver.solver import Execution


@dataclass(slots=True)
class KpInstance:
    """Stores a single knapsack benchmark instance.

    Responsibilities:
    - Keep all numeric data required by decoder and objective evaluation.

    Attributes:
      n_items (int): Number of items.
      capacity (int): Maximum knapsack capacity.
      benefits (list[int]): Item profit values.
      weights (list[int]): Item weight values.
      name (str): Human-readable instance name.
    """

    n_items: int
    capacity: int
    benefits: list[int]
    weights: list[int]
    name: str


@dataclass(slots=True)
class KpRunResult:
    """Stores one execution summary row for reporting/export.

    Responsibilities:
    - Keep normalized values that are printed and persisted to CSV.

    Attributes:
      instance (str): Instance file name.
      n_items (int): Number of items.
      capacity (int): Knapsack capacity.
      cost (float): Best minimization cost returned by RKO.
      profit (int): Decoded total profit of the best solution.
      weight (int): Decoded total weight of the best solution.
      feasible (bool): Feasibility flag relative to capacity.
      engine (str): Selected engine: python, rust, cpp or auto.
      execution (str | None): Python-engine scheduling, or None for the other engines.
      n_jobs (int | None): Requested process budget, or automatic usable-CPU cap.
      random_state (int | None): Requested worker-seed source.
      elapsed_seconds (float): End-to-end solver time including process startup.
    """

    instance: str
    n_items: int
    capacity: int
    cost: float
    profit: int
    weight: int
    feasible: bool
    engine: str
    execution: str | None
    n_jobs: int | None
    random_state: int | None
    elapsed_seconds: float


def read_kp_instance(path: Path) -> KpInstance:
    """Parse a KP instance from the C++ benchmark text format.

    Args:
      path (Path): Path to an instance file like `kp10.txt`.

    Returns:
      KpInstance: Parsed data object.

    Raises:
      ValueError: If the file has insufficient or inconsistent numeric fields.
    """
    tokens = path.read_text(encoding="utf-8").split()
    if len(tokens) < 2:
        raise ValueError(f"Invalid instance file '{path}': missing header values.")

    n_items = int(tokens[0])
    capacity = int(tokens[1])
    if n_items <= 0 or capacity < 0:
        raise ValueError("Item count must be positive and capacity must be nonnegative.")
    expected = 2 + 2 * n_items
    if len(tokens) != expected:
        raise ValueError(
            "Invalid instance file "
            f"'{path}': expected {expected} numeric values, got {len(tokens)}."
        )

    benefits: list[int] = []
    weights: list[int] = []
    cursor = 2
    for _ in range(n_items):
        benefits.append(int(tokens[cursor]))
        weights.append(int(tokens[cursor + 1]))
        cursor += 2

    if any(benefit < 0 for benefit in benefits) or any(weight <= 0 for weight in weights):
        raise ValueError("Benefits must be nonnegative and weights must be positive.")

    return KpInstance(
        n_items=n_items,
        capacity=capacity,
        benefits=benefits,
        weights=weights,
        name=path.name,
    )


def decode_knapsack(keys: list[float] | np.ndarray) -> list[int]:
    """Decode random keys with the C++ reference threshold rule.

    Args:
      keys (list[float] | np.ndarray): Random-key chromosome.

    Returns:
      list[int]: Binary vector where 1 means selected item.
    """
    return [1 if key > 0.5 else 0 for key in keys]


@dataclass(slots=True)
class KnapsackCost:
    """Evaluate the C++ Knapsack objective with pickleable instance state.

    Responsibilities:
    - Apply the reference penalty without a nested function closure.

    Attributes:
        instance (KpInstance): Parsed problem data.
    """

    instance: KpInstance

    def __call__(self, solution: list[int]) -> float:
        """Compute penalized negative profit for a binary decision vector.

        Args:
            solution (list[int]): One binary decision per item.

        Returns:
            float: Minimization objective including excess-weight penalty.

        Raises:
            ValueError: If the decisions have invalid dimensions or values.
        """
        profit, weight, _ = evaluate_solution(solution, self.instance)
        return float(100000 * max(0, weight - self.instance.capacity) - profit)


def make_cost(instance: KpInstance) -> Callable[[list[int]], float]:
    """Build a cost callback equivalent to the C++ `Problem.h` objective.

    Args:
      instance (KpInstance): Parsed benchmark data.

    Returns:
      callable: Function that maps a decoded binary solution to minimization cost.

    Raises:
      ValueError: When required by external domain validation.
    """
    return KnapsackCost(instance)


def evaluate_solution(sol: list[int], instance: KpInstance) -> tuple[int, int, bool]:
    """Compute total profit, weight, and feasibility of a decoded solution.

    Args:
      sol (list[int]): Binary decision vector.
      instance (KpInstance): Parsed benchmark data.

    Returns:
      tuple[int, int, bool]: Total profit, total weight, and feasibility flag.

    Raises:
      ValueError: When required by external domain validation.
    """
    if len(sol) != instance.n_items or any(chosen not in (0, 1) for chosen in sol):
        raise ValueError("Solution must contain exactly one binary decision per item.")
    total_profit = 0
    total_weight = 0
    for i, chosen in enumerate(sol):
        if chosen == 1:
            total_profit += instance.benefits[i]
            total_weight += instance.weights[i]
    return total_profit, total_weight, total_weight <= instance.capacity


def default_instances_dir() -> Path:
    """Return the bundled reference KP instances directory.

    Args:
      None

    Returns:
      Path: Absolute path to the packaged `examples/instances` directory.

    Raises:
      ValueError: When required by external domain validation.
    """
    return Path(__file__).resolve().parent / "instances"


def default_output_csv() -> Path:
    """Return default CSV output path for benchmark summaries.

    Args:
      None

    Returns:
      Path: Absolute results path under the caller's working directory.

    Raises:
      ValueError: When required by external domain validation.
    """
    return Path.cwd() / "results" / "kp_results.csv"


def run_single_instance(
    instance_path: Path,
    limit_time: float,
    params_dir: Path | None,
    *,
    engine: Engine = "python",
    execution: Execution | None = None,
    n_jobs: int | None = None,
    random_state: int | None = None,
    verbose: bool = False,
    prefill_pool: bool | None = None,
) -> KpRunResult:
    """Execute one KP instance and print a concise summary line.

    Args:
      instance_path (Path): Path to a `.txt` benchmark file.
      limit_time (float): Time limit in seconds for one RKO run.
      params_dir (Path | None): Optional override for RKO parameter TOMLs.
      engine (Engine): Python, rust, cpp or auto; every engine but python evaluates the
        built-in Knapsack inside compiled code without Python callbacks.
      execution (Execution | None): Python-engine process, thread or sequential scheduling.
      n_jobs (int | None): Optional process or Rust worker budget.
      random_state (int | None): Optional worker-seed source.
      verbose (bool): Emit lifecycle and attributed shared-pool updates.
      prefill_pool (bool | None): Optional separately budgeted shared startup phase.

    Returns:
      KpRunResult: Normalized result row including end-to-end elapsed time.

    Raises:
      ValueError: When required by external domain validation.
    """
    instance = read_kp_instance(instance_path)
    started = time.perf_counter()
    if engine != "python":
        solver = RKO(
            problems.Knapsack(list(instance.benefits), list(instance.weights), instance.capacity),
            params_dir=params_dir,
            engine=engine,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            prefill_pool=prefill_pool,
        )
    else:
        solver = RKO(
            Callbacks(decode_knapsack, make_cost(instance), instance.n_items),
            params_dir=params_dir,
            engine=engine,
            execution=execution,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            prefill_pool=prefill_pool,
        )
    outcome = solver.solve(limit_time)
    elapsed = time.perf_counter() - started
    best_sol = decode_knapsack(outcome.keys)
    profit, weight, feasible = evaluate_solution(best_sol, instance)

    result = KpRunResult(
        instance=instance.name,
        n_items=instance.n_items,
        capacity=instance.capacity,
        cost=outcome.cost,
        profit=profit,
        weight=weight,
        feasible=feasible,
        engine=outcome.engine,
        execution=outcome.execution,
        n_jobs=n_jobs,
        random_state=random_state,
        elapsed_seconds=elapsed,
    )
    print(
        f"{result.instance:10s} | n={result.n_items:3d} | cap={result.capacity:4d} | "
        f"cost={result.cost:12.1f} | profit={result.profit:6d} | "
        f"weight={result.weight:6d} | feasible={result.feasible}"
    )
    return result


def write_results_csv(output_path: Path, rows: list[KpRunResult]) -> None:
    """Write benchmark execution rows to a CSV file.

    Args:
      output_path (Path): Destination CSV path.
      rows (list[KpRunResult]): Execution rows to persist.

    Returns:
      None: Writes file contents to disk.

    Raises:
      ValueError: When required by external domain validation.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "instance",
                "n_items",
                "capacity",
                "cost",
                "profit",
                "weight",
                "feasible",
                "engine",
                "execution",
                "worker_budget",
                "seed",
                "elapsed_seconds",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.instance,
                    row.n_items,
                    row.capacity,
                    row.cost,
                    row.profit,
                    row.weight,
                    row.feasible,
                    row.engine,
                    row.execution,
                    row.n_jobs,
                    row.random_state,
                    row.elapsed_seconds,
                ]
            )


def parse_args() -> argparse.Namespace:
    """Parse one instance or an unambiguous legacy batch selection.

    Returns:
      argparse.Namespace: Instance selection, parameter directory and execution options.

    Raises:
      SystemExit: If help is requested or arguments/input paths are invalid.
    """
    parser = argparse.ArgumentParser(
        description="Run RKO on Knapsack instances with a TOML parameter directory.",
    )
    parser.add_argument(
        "instance",
        type=Path,
        nargs="?",
        help="Single Knapsack instance file; omit to use directory/glob batch selection.",
    )
    parser.add_argument(
        "--instances-dir",
        type=Path,
        default=None,
        help="Directory containing KP instance files (default: bundled reference instances).",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Glob pattern used to select instances within --instances-dir.",
    )
    parser.add_argument(
        "--limit-time",
        type=float,
        default=3.0,
        help="Time limit in seconds per instance.",
    )
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=None,
        help="Optional custom directory containing RKO TOML parameter files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=default_output_csv(),
        help="Destination CSV path for execution results.",
    )
    parser.add_argument("--engine", choices=EngineSelection.ENGINES, default=None)
    parser.add_argument(
        "--execution",
        choices=EngineSelection.EXECUTIONS,
        default=None,
        help="Python-engine scheduling (default: process).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Explicit search-worker limit (default: available logical CPUs, capped by portfolio).",
    )
    parser.add_argument("--random-state", type=int, default=None, help="Worker seed source.")
    parser.add_argument(
        "--prefill-pool",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reference-style shared startup (separate allowance); default follows TOML.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide live algorithm/pool updates; retain the final result and CSV.",
    )
    args = parser.parse_args()
    if args.instance is not None:
        if args.instances_dir is not None or args.pattern is not None:
            parser.error("A single instance cannot be combined with --instances-dir or --pattern.")
        if not args.instance.is_file():
            parser.error(f"Instance is not a file: {args.instance}")
    args.engine = args.engine or "python"
    try:
        EngineSelection.validate(args.engine, args.execution)
    except ValueError as error:
        parser.error(str(error))
    if args.instances_dir is None:
        args.instances_dir = default_instances_dir()
    if args.pattern is None:
        args.pattern = "kp*.txt"
    return args


def main() -> None:
    """Run selected Knapsack instances through the class API and export CSV.

    Raises:
      FileNotFoundError: If no instance matches or a required file is missing.
      ValueError: If instance contents or solver settings are invalid.
      OSError: If configuration, instance or result files cannot be accessed.
    """
    args = parse_args()
    instances = (
        [args.instance]
        if args.instance is not None
        else sorted(args.instances_dir.glob(args.pattern))
    )
    if not instances:
        raise FileNotFoundError(
            f"No instances found in '{args.instances_dir}' with pattern '{args.pattern}'."
        )

    print("instance    | n   | cap  | cost         | profit | weight | feasible")
    print("-" * 74)
    results: list[KpRunResult] = []
    for path in instances:
        results.append(
            run_single_instance(
                path,
                limit_time=args.limit_time,
                params_dir=args.params_dir,
                engine=args.engine,
                execution=args.execution,
                n_jobs=args.n_jobs,
                random_state=args.random_state,
                verbose=not args.quiet,
                prefill_pool=args.prefill_pool,
            )
        )

    write_results_csv(args.output_csv, results)
    print(f"CSV saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
