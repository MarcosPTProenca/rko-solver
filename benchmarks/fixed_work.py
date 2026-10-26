"""Deterministic fixed-evaluation search workloads for cross-revision comparison.

Objective:
- Compare complete serial search paths without conflating timed search trajectories.

Technical approach:
- Run SA, GA and BRKGA on numerical, rugged and permutation workloads.
- Support seeded elite pools for IPR and configurable fixed evaluation counts.
- Expire the shared callback boundary after the requested completed evaluations.
- Record result and RNG fingerprints alongside unprofiled elapsed search time.
- Optionally alternate a trusted baseline SearchSession module in the same process.

Flow (pseudocode):
1. Seed Python and NumPy for each workload and instantiate a fresh algorithm.
2. Execute actual population, local-search, callback and elite-pool operations.
3. Emit JSON lines for timing and deterministic cross-checkout replay validation.

Important criteria:
- Real local search and both callback-protecting copies remain enabled.
- Copy this runner into the baseline checkout and compare matching JSON records.
- Prefer --baseline-search for paired callback-boundary timing without filesystem drift.
- Synthetic serial measurements exclude configuration I/O and process startup/IPC.
- A fingerprint mismatch invalidates equal-trajectory speed comparisons.

Inputs/Outputs:
- Input: no external data; 1,000 keys and seeds 200 through 202.
- Output: JSON lines with elapsed time, cost, evaluation count and replay fingerprints.
"""

import argparse
import hashlib
import importlib.util
import json
import numpy as np
import pickle
import random
import sys
import time
from pathlib import Path
from rko_solver.components.solution_pool import ElitePool
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics import MetaheuristicBase
from rko_solver.metaheuristics.brkga import BRKGA
from rko_solver.metaheuristics.ga import GeneticAlgorithm
from rko_solver.metaheuristics.ipr import IPR
from rko_solver.metaheuristics.sa import SA
from rko_solver.orchestration.search import SearchSession


def _decode(keys: np.ndarray) -> np.ndarray:
    """Pass continuous random keys to a numerical objective unchanged.

    Args:
        keys (np.ndarray): Candidate chromosome.

    Returns:
        np.ndarray: Unchanged payload.
    """
    return keys


def _order(keys: np.ndarray) -> np.ndarray:
    """Decode random keys as a stable permutation.

    Args:
        keys (np.ndarray): Candidate chromosome.

    Returns:
        np.ndarray: Indices ordered by key.
    """
    return np.argsort(keys, kind="stable")


def _run(
    algorithm_type: type[MetaheuristicBase],
    workload: str,
    seed: int,
    session_type: type[SearchSession] = SearchSession,
    dimension: int = 1000,
    evaluation_limit: int = 2000,
    pool: ElitePool | None = None,
) -> dict[str, object]:
    """Record one deterministic full-search result after a fixed evaluation count.

    Args:
        algorithm_type (type[MetaheuristicBase]): Production search implementation.
        workload (str): Sphere, rugged or ordering workload.
        seed (int): Seed used for both RNG streams.
        session_type (type[SearchSession]): Callback-boundary implementation to compare.
        dimension (int): Positive number of random keys.
        evaluation_limit (int): Completed evaluations before cooperative stopping.
        pool (ElitePool | None): Optional empty precreated pool; setup is not timed.

    Returns:
        dict[str, object]: Elapsed seconds and deterministic search fingerprints.
    """
    random.seed(seed)
    np.random.seed(seed)
    evaluations = 0
    weights = np.arange(1, dimension + 1)

    def score(payload: np.ndarray) -> float:
        """Compute consistent fitness for initial elites and search candidates.

        Args:
            payload (np.ndarray): Continuous keys or a decoded permutation.

        Returns:
            float: Deterministic finite minimization objective.
        """
        if workload == "rugged":
            values = 10 * payload - 5
            return float(10 * dimension + np.sum(values**2 - 10 * np.cos(2 * np.pi * values)))
        if workload == "ordering":
            return float(np.dot(payload, weights))
        return float(np.dot(payload, payload))

    def objective(payload: np.ndarray) -> float:
        """Evaluate search candidates and expire the boundary at the requested count.

        Args:
            payload (np.ndarray): Continuous keys or a decoded permutation.

        Returns:
            float: Deterministic finite minimization objective.
        """
        nonlocal evaluations
        evaluations += 1
        if evaluations == evaluation_limit:
            session.deadline = 0.0
        return score(payload)

    decoder = _order if workload == "ordering" else _decode
    pool = SolutionPool() if pool is None else pool
    if issubclass(algorithm_type, IPR):
        generator = np.random.RandomState(seed + 1000)
        for _ in range(10):
            keys = generator.random_sample(dimension)
            pool.insert((score(decoder(keys)), keys.tolist()), "seed")
    session = session_type(decoder, objective)
    algorithm = algorithm_type(pool, decoder, objective, dimension)
    started = time.perf_counter()
    started_cpu = time.process_time()
    result = session.solve(algorithm, 3600.0)
    cpu_seconds = time.process_time() - started_cpu
    elapsed = time.perf_counter() - started
    assert evaluations == evaluation_limit
    assert np.isfinite(result[2])
    assert len(result[1]) == dimension
    return {
        "algorithm": algorithm_type.__name__,
        "dimension": dimension,
        "workload": workload,
        "seed": seed,
        "evaluations": evaluations,
        "seconds": elapsed,
        "cpu_seconds": cpu_seconds,
        "cost": result[2],
        "keys_sha256": hashlib.sha256(np.asarray(result[1], dtype="<f8").tobytes()).hexdigest(),
        "pool_sha256": hashlib.sha256(pickle.dumps(pool.get_elite(10))).hexdigest(),
        "rng_sha256": hashlib.sha256(
            pickle.dumps((random.getstate(), np.random.get_state()))
        ).hexdigest(),
    }


def main() -> None:
    """Run fixed work, optionally pairing a trusted baseline module in the same process.

    Raises:
        ValueError: If the supplied baseline cannot be loaded as a Python module.
        AssertionError: If replayed baseline and production results differ.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-search", type=Path, help="Trusted baseline orchestration/search.py to execute."
    )
    args = parser.parse_args()
    sessions = {"production": SearchSession}
    if args.baseline_search is not None:
        name = "rko_solver.orchestration._benchmark_reference"
        spec = importlib.util.spec_from_file_location(name, args.baseline_search)
        if spec is None or spec.loader is None:
            raise ValueError("Baseline must be a loadable Python source module.")
        reference = importlib.util.module_from_spec(spec)
        sys.modules[name] = reference
        spec.loader.exec_module(reference)
        sessions["reference"] = reference.SearchSession
    saved_python, saved_numpy = random.getstate(), np.random.get_state()
    try:
        np.dot(np.ones(1000), np.ones(1000))
        np.argsort(np.ones(1000), kind="stable")
        np.cos(np.ones(1000))
        print(json.dumps({"python": sys.version, "numpy": np.__version__}), flush=True)
        for repeat in range(3 if len(sessions) == 2 else 1):
            for algorithm_type in (SA, GeneticAlgorithm, BRKGA):
                for workload in ("sphere", "rugged", "ordering"):
                    for seed in range(200, 203):
                        labels = list(sessions)
                        if (repeat + seed) % 2:
                            labels.reverse()
                        runs = {
                            label: _run(algorithm_type, workload, seed, sessions[label])
                            for label in labels
                        }
                        expected = {
                            k: v
                            for k, v in runs["production"].items()
                            if k not in ("seconds", "cpu_seconds")
                        }
                        for label, row in runs.items():
                            assert {
                                k: v for k, v in row.items() if k not in ("seconds", "cpu_seconds")
                            } == expected
                            print(
                                json.dumps({"variant": label, "repeat": repeat, **row}), flush=True
                            )
    finally:
        random.setstate(saved_python)
        np.random.set_state(saved_numpy)


if __name__ == "__main__":
    main()
