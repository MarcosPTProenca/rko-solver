"""Python-defined benchmark problems for the native engine studies.

Objective:
- Provide Knapsack, TSP and Rastrigin objectives written in Python (Numba-compiled and
  NumPy block-scored) so the studies can measure whether a problem defined in Python
  reaches the speed and quality of the pinned C++ reference and of Rust-native problems.

Technical approach:
- Each family has one objective per execution style: an `@njit` function over one key
  vector (called without the interpreter lock) and a NumPy function over a block of
  vectors (one interpreter call per block). Both reproduce the C++/Rust decoders exactly:
  Knapsack selects keys above 0.5 with a 100000 excess penalty, TSP sorts keys with ties
  broken by index and rounds distances like TSPLIB EUC_2D, Rastrigin maps keys linearly to
  `[-5.12, 5.12)` and subtracts a shift.
- `cpp_parameter_directory` writes TOML files carrying the C++ reference's online grids, so
  the Python API can run with the same parameter grids as the C++ comparison.
- A worker entry point runs one solve in a fresh interpreter and prints the result line
  the study runners already parse.

Flow (pseudocode):
1. Load the instance and build the requested objective.
2. Create a solver with the C++ parameter grids and run it for the search budget.
3. Print one `RKO_BENCH_RESULT` line with full-precision keys and the objective.

Important criteria:
- Compilation happens before the timed search; the study reports it separately and the
  search budget excludes it.
- Objectives must stay numerically identical to the independent scoring of the studies.

Inputs/Outputs:
- Input: problem kind, instance path, algorithm names, style, budget and seed.
- Output: an `RKO_BENCH_RESULT {"keys": [...], "objective": x}` line on standard output.
"""

from __future__ import annotations

import json
import math
import numba
import numpy as np
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from examples.run_kp_cpp_instances import read_kp_instance
from pathlib import Path
from rko_solver import RKO
from rko_solver import Blocks
from rko_solver import Numba
from rko_solver.parameters.loader import default_params_dir
from typing import Any

PORTFOLIO = ("SA", "ILS", "VNS", "BRKGA", "BRKGA_CS", "PSO", "GA", "LNS", "GRASP", "IPR")

CPP_GRIDS: dict[str, dict[str, dict[str, list[float]]]] = {
    "brkga": {
        "offline": {"p": [987], "pe": [0.20], "pm": [0.15], "rhoe": [0.65]},
        "online": {
            "p": [300, 400, 500],
            "pe": [0.10, 0.15, 0.20],
            "pm": [0.10, 0.15, 0.20],
            "rhoe": [0.60, 0.65, 0.70],
        },
    },
    "brkga_cs": {
        "offline": {"p": [987], "pe": [0.25], "pm": [0.03], "rhoe": [0.65]},
        "online": {
            "p": [377, 610, 987],
            "pe": [0.15, 0.20, 0.25],
            "pm": [0.01, 0.02, 0.03],
            "rhoe": [0.60, 0.65, 0.70],
        },
    },
    "sa": {
        "offline": {
            "sa_max": [1000],
            "t0": [1000000.0],
            "alpha": [0.99],
            "beta_min": [0.01],
            "beta_max": [0.05],
        },
        "online": {
            "sa_max": [250, 500, 750],
            "t0": [100000.0],
            "alpha": [0.97, 0.99, 0.99],
            "beta_min": [0.03, 0.04, 0.05],
            "beta_max": [0.08, 0.09, 0.10],
        },
    },
    "ils": {
        "offline": {"beta_min": [0.10], "beta_max": [0.20]},
        "online": {"beta_min": [0.05, 0.10, 0.15], "beta_max": [0.15, 0.20, 0.25]},
    },
    "vns": {
        "offline": {"k_max": [5], "beta_min": [0.05]},
        "online": {"k_max": [5, 8, 10], "beta_min": [0.02, 0.04, 0.05]},
    },
    "lns": {
        "offline": {
            "beta_min": [0.10],
            "beta_max": [0.30],
            "t0": [10000.0],
            "alpha_lns": [0.95],
        },
        "online": {
            "beta_min": [0.05, 0.10, 0.15],
            "beta_max": [0.15, 0.20, 0.25],
            "t0": [1000.0, 10000.0],
            "alpha_lns": [0.95, 0.96],
        },
    },
    "pso": {
        "offline": {"p_size": [200], "c1": [2.05], "c2": [2.05], "w": [0.73]},
        "online": {
            "p_size": [100, 200, 300],
            "c1": [1.95, 2.0, 2.05],
            "c2": [1.95, 2.0, 2.05],
            "w": [0.73, 0.74, 0.75],
        },
    },
    "ga": {
        "offline": {"size_pop": [400], "prob_cros": [0.98], "prob_mut": [0.005]},
        "online": {
            "size_pop": [200, 400, 600],
            "prob_cros": [0.90, 0.95, 0.98],
            "prob_mut": [0.02, 0.03, 0.05],
        },
    },
    "grasp": {
        "offline": {"hs": [0.125], "he": [0.00098]},
        "online": {
            "alpha_grasp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "hs": [0.125, 0.0625],
            "he": [0.00098, 0.00049],
        },
    },
}


def cpp_parameter_directory(target: Path) -> Path:
    """Copy the packaged parameter files and replace the grids by the C++ reference's.

    Args:
        target (Path): Existing empty directory that receives the files.

    Returns:
        Path: The same directory, usable as `params_dir`.

    Raises:
        OSError: If files cannot be copied or written.
    """
    for source in default_params_dir().glob("*.toml"):
        shutil.copy(source, target / source.name)
    for name, grids in CPP_GRIDS.items():
        lines = [f"# C++ reference grids for {name}.", "", "[offline]"]
        for key, values in grids["offline"].items():
            lines.append(f"{key} = {values[0]!r}")
        lines += ["", "[online]"]
        for key, values in grids["online"].items():
            lines.append(f"{key} = {list(values)!r}")
        (target / f"{name}.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def knapsack_objectives(
    path: Path,
) -> tuple[int, Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """Build the Knapsack objectives for one instance file.

    Args:
        path (Path): Instance in the RKO text format.

    Returns:
        tuple: Dimension, an njit objective over one vector and a NumPy block objective.

    Raises:
        OSError: If the instance cannot be read.
        ValueError: If the instance is malformed.
    """
    instance = read_kp_instance(path)
    profits = np.asarray(instance.benefits, dtype=np.float64)
    weights = np.asarray(instance.weights, dtype=np.float64)
    capacity = float(instance.capacity)

    @numba.njit
    def compiled(keys: np.ndarray) -> float:
        """Score one key vector.

        Args:
            keys (np.ndarray): Random keys.

        Returns:
            float: Penalized negative profit.
        """
        profit = 0.0
        weight = 0.0
        for index in range(keys.shape[0]):
            if keys[index] > 0.5:
                profit += profits[index]
                weight += weights[index]
        return 100000.0 * max(0.0, weight - capacity) - profit

    def blocked(keys: np.ndarray) -> np.ndarray:
        """Score a block of key vectors.

        Args:
            keys (np.ndarray): Candidate matrix.

        Returns:
            np.ndarray: Penalized negative profit per row.
        """
        chosen = keys > 0.5
        return 100000.0 * np.maximum(0.0, chosen @ weights - capacity) - chosen @ profits

    return len(profits), compiled, blocked


def tsp_objectives(
    path: Path,
) -> tuple[int, Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """Build the TSP objectives for one instance file.

    Args:
        path (Path): City count followed by one `x y` pair per line.

    Returns:
        tuple: Dimension, an njit objective over one vector and a NumPy block objective.

    Raises:
        OSError: If the instance cannot be read.
        ValueError: If the instance is malformed.
    """
    tokens = path.read_text(encoding="utf-8").split()
    size = int(tokens[0])
    coordinates = np.array(tokens[1 : 1 + 2 * size], dtype=np.float64).reshape(size, 2)
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.floor(np.hypot(delta[..., 0], delta[..., 1]) + 0.5)

    @numba.njit
    def compiled(keys: np.ndarray) -> float:
        """Score one key vector as a closed tour.

        Args:
            keys (np.ndarray): Random keys, sorted ascending with ties by index.

        Returns:
            float: Closed tour length.
        """
        order = np.argsort(keys, kind="mergesort")
        total = 0.0
        previous = order[order.shape[0] - 1]
        for position in range(order.shape[0]):
            city = order[position]
            total += distances[previous, city]
            previous = city
        return total

    def blocked(keys: np.ndarray) -> np.ndarray:
        """Score a block of key vectors as closed tours.

        Args:
            keys (np.ndarray): Candidate matrix.

        Returns:
            np.ndarray: Closed tour length per row.
        """
        order = np.argsort(keys, axis=1, kind="stable")
        return distances[np.roll(order, 1, axis=1), order].sum(axis=1)

    return size, compiled, blocked


def rastrigin_objectives(
    path: Path,
) -> tuple[int, Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """Build the shifted Rastrigin objectives for one instance file.

    Args:
        path (Path): Dimension followed by one shift per coordinate.

    Returns:
        tuple: Dimension, an njit objective over one vector and a NumPy block objective.

    Raises:
        OSError: If the instance cannot be read.
        ValueError: If the instance is malformed.
    """
    tokens = path.read_text(encoding="utf-8").split()
    size = int(tokens[0])
    shifts = np.array(tokens[1 : 1 + size], dtype=np.float64)
    tau = 2.0 * math.pi

    @numba.njit
    def compiled(keys: np.ndarray) -> float:
        """Score one key vector.

        Args:
            keys (np.ndarray): Random keys.

        Returns:
            float: Shifted Rastrigin value.
        """
        total = 10.0 * keys.shape[0]
        for index in range(keys.shape[0]):
            z = -5.12 + 10.24 * keys[index] - shifts[index]
            total += z * z - 10.0 * np.cos(tau * z)
        return total

    def blocked(keys: np.ndarray) -> np.ndarray:
        """Score a block of key vectors.

        Args:
            keys (np.ndarray): Candidate matrix.

        Returns:
            np.ndarray: Shifted Rastrigin value per row.
        """
        z = -5.12 + 10.24 * keys - shifts
        return 10.0 * size + np.sum(z * z - 10.0 * np.cos(tau * z), axis=1)

    return size, compiled, blocked


OBJECTIVES = {
    "kp": knapsack_objectives,
    "tsp": tsp_objectives,
    "rastrigin": rastrigin_objectives,
}


def run_worker(payload: Mapping[str, Any]) -> None:
    """Solve one instance with a Python-defined objective and print the result line.

    Args:
        payload (Mapping[str, Any]): Keys `problem`, `instance_path`, `algorithms`
            (list of factory names), `style` (`numba`, `batch` or `cpp-numba`), `budget_seconds`,
            `seed` and optional `n_jobs`.

    Raises:
        KeyError: If a required payload field is missing.
        ValueError: If the style or problem kind is unknown.
        OSError: If the instance cannot be read.
    """
    kind = str(payload["problem"])
    style = str(payload["style"])
    if kind not in OBJECTIVES or style not in ("numba", "batch", "cpp-numba"):
        raise ValueError(f"Unknown problem {kind!r} or style {style!r}.")
    dimension, compiled, blocked = OBJECTIVES[kind](Path(str(payload["instance_path"])))
    algorithms = [str(name) for name in payload["algorithms"]]
    with tempfile.TemporaryDirectory(prefix="rko-cpp-grids-") as directory:
        params = cpp_parameter_directory(Path(directory))
        options = {
            "algorithms": algorithms,
            "adaptation": "adaptive",
            "pool_size": 10,
            "params_dir": params,
            "random_state": int(payload["seed"]),
            "n_jobs": payload.get("n_jobs"),
        }
        if style in ("numba", "cpp-numba"):
            if style == "cpp-numba":
                options["engine"] = "cpp"
            solver = RKO(Numba(compiled, dimension), **{"engine": "rust"} | options)
            compiled(np.zeros(dimension))
        else:
            solver = RKO(Blocks(blocked, dimension, 64), **{"engine": "rust"} | options)
            blocked(np.zeros((2, dimension)))
        started = time.perf_counter()
        result = solver.solve(float(payload["budget_seconds"]))
        wall = time.perf_counter() - started
    print(
        "RKO_BENCH_RESULT "
        + json.dumps(
            {"keys": list(result.keys), "objective": result.cost, "solve_seconds": wall},
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the worker entry point.

    Args:
        argv (Sequence[str] | None): Optional argument vector: `--worker <json>`.

    Returns:
        int: Zero on success.

    Raises:
        ValueError: If the arguments are not a worker request.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2 or arguments[0] != "--worker":
        raise ValueError("Usage: python -m benchmarks.python_problems --worker <json payload>")
    run_worker(json.loads(arguments[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
