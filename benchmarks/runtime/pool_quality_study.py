"""Compare final solution quality of the manager and shared-memory elite pools at equal wall time.

Objective:
- Measure whether the shared-memory pool changes the best cost reached in a fixed time budget,
  on instances that separate the two arms, with paired seeds.

Technical approach:
- Run the default process-mode portfolio through the public `RKO` API for each seed, arm and
  instance, interleaving arms seed by seed.
- The arm is selected by replacing `_use_shared_pool` in the parent process only; workers receive
  whichever pool object the parent builds.

Flow (pseudocode):
1. Build module-level Knapsack and Rastrigin callbacks, which spawned workers can import.
2. For each seed, run the manager arm and the shared arm on each instance.
3. Append one JSON row per run with cost, evaluations, elapsed time and stop reasons.

Important criteria:
- Both arms use the same time budget, worker cap, seed and portfolio.
- Rows are appended as they finish, so an interrupted campaign keeps its completed evidence.

Inputs/Outputs:
- Input: seed list, time budget and output path.
- Output: a JSON Lines file with one object per run.
"""

from __future__ import annotations

import argparse
import json
import math
import numpy as np
import rko_solver.orchestration.processes as processes
import statistics
from benchmarks.measurement.paired import sign_test_p
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from rko_solver import RKO
from rko_solver import Callbacks
from typing import TypedDict

KNAPSACK_ITEMS = 300
RASTRIGIN_DIMENSION = 300


def _knapsack_profits_weights() -> tuple[np.ndarray, np.ndarray, float]:
    """Build the deterministic Knapsack instance shared by every run.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: Profits, weights and capacity.
    """
    index = np.arange(KNAPSACK_ITEMS)
    profits = 10.0 + (index * 37) % 91
    weights = 5.0 + (index * 53) % 47
    return profits, weights, 0.4 * float(weights.sum())


PROFITS, WEIGHTS, CAPACITY = _knapsack_profits_weights()


def identity_decode(keys: np.ndarray) -> np.ndarray:
    """Return the key vector unchanged, so the cost receives raw keys.

    Args:
        keys (np.ndarray): Random keys in [0, 1).

    Returns:
        np.ndarray: The same keys.
    """
    return keys


def knapsack_cost(keys: np.ndarray) -> float:
    """Score a Knapsack key vector with a capacity penalty.

    Args:
        keys (np.ndarray): Random keys; items with key above one half are selected.

    Returns:
        float: Penalized negative profit to minimize.
    """
    chosen = np.asarray(keys) > 0.5
    weight = float(WEIGHTS[chosen].sum())
    return float(100000.0 * max(0.0, weight - CAPACITY) - PROFITS[chosen].sum())


def rastrigin_cost(keys: np.ndarray) -> float:
    """Score a key vector with a Rastrigin function shifted off the decoder's grid.

    Args:
        keys (np.ndarray): Random keys in [0, 1).

    Returns:
        float: Rastrigin value, zero at the global optimum.
    """
    values = np.asarray(keys) * 10.24 - 5.49
    return float(10.0 * values.size + np.sum(values**2 - 10.0 * np.cos(2.0 * math.pi * values)))


class Instance(TypedDict):
    """Describe one benchmark instance and its callbacks.

    Attributes:
        name (str): Label written to every row.
        dimension (int): Number of random keys.
        cost (Callable[[np.ndarray], float]): Module-level objective.
    """

    name: str
    dimension: int
    cost: Callable[[np.ndarray], float]


INSTANCES: tuple[Instance, ...] = (
    {"name": f"knapsack-{KNAPSACK_ITEMS}", "dimension": KNAPSACK_ITEMS, "cost": knapsack_cost},
    {
        "name": f"rastrigin-{RASTRIGIN_DIMENSION}",
        "dimension": RASTRIGIN_DIMENSION,
        "cost": rastrigin_cost,
    },
)


def run_once(
    instance: Instance, arm: str, seed: int, seconds: float, workers: int
) -> dict[str, object]:
    """Solve one instance with one pool arm and return a summary row.

    Args:
        instance (Instance): Problem to solve.
        arm (str): `manager` forces the manager-hosted pool; `shared` uses the default.
        seed (int): Random seed shared by both arms of the pair.
        seconds (float): Wall-time budget passed to `solve`.
        workers (int): Process worker cap.

    Returns:
        dict[str, object]: Instance, arm, seed, cost, evaluations, elapsed and stop reasons.
    """
    original = processes._use_shared_pool
    if arm == "manager":
        processes._use_shared_pool = lambda verbose, trace: False
    try:
        solver = RKO(
            Callbacks(identity_decode, instance["cost"], instance["dimension"]),
            execution="process",
            n_jobs=workers,
            random_state=seed,
        )
        result = solver.solve(seconds)
    finally:
        processes._use_shared_pool = original
    return {
        "instance": instance["name"],
        "arm": arm,
        "seed": seed,
        "cost": result.cost,
        "n_evaluations": result.n_evaluations,
        "elapsed": result.elapsed_seconds,
        "stop_reason": result.stop_reason,
        "stops": [report.stop_reason for report in result.searches],
    }


def paired_runs(
    rows: list[dict[str, object]],
) -> dict[str, list[tuple[dict[str, object], dict[str, object]]]]:
    """Group complete manager and shared pairs by instance name.

    Args:
        rows (list[dict[str, object]]): Parsed JSON Lines records.

    Returns:
        dict[str, list[tuple[dict[str, object], dict[str, object]]]]: Matched (manager, shared)
            records per instance, in seed order.
    """
    by_key: dict[tuple[str, int], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_key[(str(row["instance"]), int(row["seed"]))][str(row["arm"])] = row
    grouped: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = defaultdict(list)
    for (instance, _), arms in sorted(by_key.items(), key=lambda item: item[0]):
        if "manager" in arms and "shared" in arms:
            grouped[instance].append((arms["manager"], arms["shared"]))
    return dict(grouped)


def instance_report(instance: str, pairs: list[tuple[dict[str, object], dict[str, object]]]) -> str:
    """Describe the paired cost difference and its exact sign test for one instance.

    Args:
        instance (str): Instance label.
        pairs (list[tuple[dict[str, object], dict[str, object]]]): Matched manager and
            shared records.

    Returns:
        str: One line with the pair count, mean, 95% interval and sign-test p-value.
    """
    diffs = [float(shared["cost"]) - float(manager["cost"]) for manager, shared in pairs]
    wins = sum(1 for value in diffs if value < 0)
    losses = sum(1 for value in diffs if value > 0)
    spread = statistics.stdev(diffs) / math.sqrt(len(diffs))
    mean = statistics.mean(diffs)
    return (
        f"{instance} pairs={len(pairs)} paired_mean={mean:.3f} "
        f"ci95=[{mean - 2.045 * spread:.3f},{mean + 2.045 * spread:.3f}] "
        f"shared_better={wins} shared_worse={losses} sign_p={sign_test_p(wins, losses):.4f}"
    )


def summarize(path: Path) -> list[str]:
    """Summarize paired manager and shared runs per instance.

    Args:
        path (Path): JSON Lines file written by `main`.

    Returns:
        list[str]: One report line per instance with at least one complete pair.
    """
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [instance_report(name, pairs) for name, pairs in sorted(paired_runs(rows).items())]


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the interleaved campaign and append rows to the output file.

    Args:
        argv (list[str] | None): Optional command-line arguments; None reads `sys.argv`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args(argv)
    if options.summarize is not None:
        print("\n".join(summarize(options.summarize)))
        return
    if options.seeds is None or options.output is None:
        parser.error("--seeds and --output are required unless --summarize is given")
    with options.output.open("a", encoding="utf-8") as sink:
        for seed in options.seeds:
            for instance in INSTANCES:
                for arm in ("manager", "shared"):
                    row = run_once(instance, arm, seed, options.seconds, options.workers)
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()


if __name__ == "__main__":
    main()
