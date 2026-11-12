"""Measure whole public `RKO.solve` invocations against their own search time.

Objective:
- Quantify the time a user pays for a complete `RKO(...).solve()` call that the search-kernel
  diagnostic does not cover: construction, portfolio start-up, process spawn and result assembly.

Technical approach:
- Fix the work with `max_evaluations`, so every repeat of a configuration performs the same number
  of objective evaluations, and give the time limit enough room that it never ends a search.
- For each repeat, record the solver's `elapsed_seconds` for the whole call and the sum of the
  searches' own `elapsed_seconds`. For sequential runs the searches run back to back, so the
  difference is the time spent outside the searches: construction, portfolio setup and assembly.

Flow (pseudocode):
1. Build a module-level sphere objective, which spawned workers can import.
2. For each configuration (sequential, thread, process) run the repeats and time each complete call.
3. Summarize the per-configuration median overhead fraction and evaluation count.

Important criteria:
- Each configuration uses the same portfolio, seed, dimension and evaluation cap.
- Only one solve runs at a time, so configurations never compete inside this process.

Inputs/Outputs:
- Input: repeats, dimension and evaluation cap from the command line.
- Output: one JSON line per run on stdout and a summary JSON document.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import statistics
import sys
from rko_solver import RKO
from typing import Any

CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "sequential": {"execution": "sequential"},
    "thread": {"execution": "thread"},
    "process": {"execution": "process", "n_jobs": 2},
}


def identity_decode(keys: np.ndarray) -> np.ndarray:
    """Return the key vector unchanged.

    Args:
        keys (np.ndarray): Random keys.

    Returns:
        np.ndarray: The same keys.
    """
    return keys


def sphere_cost(solution: np.ndarray) -> float:
    """Score a key vector with the sphere function.

    Args:
        solution (np.ndarray): Decoded key vector.

    Returns:
        float: Sum of squares.
    """
    return float(np.sum(np.asarray(solution) ** 2))


def overhead_fraction(elapsed: float, search: float) -> float:
    """Return the share of a sequential call's elapsed time spent outside its searches.

    Args:
        elapsed (float): The solver's `elapsed_seconds` for the call; must be positive.
        search (float): Sum of the searches' own `elapsed_seconds`, which run back to back.

    Returns:
        float: `(elapsed - search) / elapsed`.

    Raises:
        ValueError: If the elapsed time is not positive.
    """
    if not elapsed > 0:
        raise ValueError("elapsed time must be positive")
    return (elapsed - search) / elapsed


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Summarize the median overhead fraction and run count for sequential configurations.

    Parallel configurations are skipped: their search times overlap, so the sum does not
    isolate overhead.

    Args:
        rows (list[dict[str, Any]]): Measured rows with `config`, `elapsed` and `search` keys.

    Returns:
        dict[str, dict[str, float]]: Per sequential configuration, run count and median overhead.
    """
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if str(row["config"]) != "sequential":
            continue
        grouped.setdefault(str(row["config"]), []).append(
            overhead_fraction(float(row["elapsed"]), float(row["search"]))
        )
    return {
        name: {"n": len(values), "median_overhead_fraction": statistics.median(values)}
        for name, values in grouped.items()
    }


def run_once(name: str, dimension: int, evaluations: int, seed: int) -> dict[str, Any]:
    """Time one complete public `solve` call for a configuration.

    Args:
        name (str): Key into `CONFIGURATIONS`.
        dimension (int): Number of random keys.
        evaluations (int): Per-algorithm evaluation cap that fixes the work.
        seed (int): Random seed.

    Returns:
        dict[str, Any]: Configuration, wall time, solver elapsed time, evaluations and cost.
    """
    solver = RKO(
        identity_decode,
        sphere_cost,
        dimension,
        algorithms=["SA", "GA"],
        random_state=seed,
        max_evaluations=evaluations,
        **CONFIGURATIONS[name],
    )
    result = solver.solve(60.0)
    return {
        "config": name,
        "elapsed": result.elapsed_seconds,
        "search": sum(report.elapsed_seconds for report in result.searches),
        "n_evaluations": result.n_evaluations,
        "cost": result.cost,
    }


def main(argv: list[str] | None = None) -> int:
    """Run every configuration, print each row and the summary.

    Args:
        argv (list[str] | None): Optional command-line arguments; None reads `sys.argv`.

    Returns:
        int: Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--evaluations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    options = parser.parse_args(argv)
    rows: list[dict[str, Any]] = []
    for repeat in range(options.repeats):
        for name in CONFIGURATIONS:
            row = run_once(name, options.dimension, options.evaluations, options.seed + repeat)
            rows.append(row)
            print(json.dumps(row), flush=True)
    print(json.dumps({"summary": summarize(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
