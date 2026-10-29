"""Paired feasible-state construction and adaptive-search allocation measurements.

Objective:
- Isolate constructing only accepted QState records from indexed graph changes.

Technical approach:
- Load a trusted previous q_learning.py source module as the baseline.
- Alternate builders with identical seeds and compare all state fields and RNGs.
- Trace graph allocations separately from unprofiled process CPU and wall time.
- Replay complete adaptive PSO searches with the existing fixed-work runner.

Flow (pseudocode):
1. Warm graph construction and measure all/slice/singleton feasible grids.
2. Assert identical graph fingerprints and report retained/peak traced allocations.
3. Freeze adaptive clocks and alternate full-search comparisons on neutral workloads.

Important criteria:
- Execute only trusted reference source supplied explicitly by the caller.
- Both variants retain the same indexed adjacency and Q-update implementation.
- Singleton-heavy synthetic grids illustrate filtering, not typical solver usage.
- Frozen clocks do not validate real adaptive timing or cold-spawn improvements.
- Tracemalloc measures traced allocation, not RSS or user-retained predicate inputs.

Inputs/Outputs:
- Input: --reference path to the pre-change q_learning.py source file.
- Output: graph/state/RNG equivalence, allocation and paired full-search JSON lines.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import numpy as np
import pickle
import random
import statistics
import sys
import time
import tracemalloc
from benchmarks.algorithms.q_learning import OnlinePSO
from benchmarks.algorithms.q_learning import _frozen_elapsed
from benchmarks.runtime.fixed_work import _run
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from rko_solver.components.adaptive_control.q_learning import QLearningController
from rko_solver.components.adaptive_control.q_learning import QState
from rko_solver.metaheuristics import MetaheuristicBase


def _fingerprint(states: list[QState]) -> tuple[str, object]:
    """Hash every state field independently of the baseline module's class identity.

    Args:
        states (list[QState]): Ordered states from one builder.

    Returns:
        tuple[str, object]: Exact serialized field hash and current Python RNG state.
    """
    fields = [asdict(state) for state in states]
    return hashlib.sha256(pickle.dumps(fields)).hexdigest(), random.getstate()


def _slice(values: list[float]) -> bool:
    """Accept the first coordinate slice without changing the predicate payload.

    Args:
        values (list[float]): Cartesian coordinate list.

    Returns:
        bool: Whether the first coordinate equals zero.
    """
    return values[0] == 0.0


def _singleton(values: list[float]) -> bool:
    """Keep only the all-zero combination in a nonnegative synthetic grid.

    Args:
        values (list[float]): Cartesian coordinate list.

    Returns:
        bool: Whether all coordinates sum to zero.
    """
    return sum(values) == 0.0


def main() -> None:
    """Compare filtered graph construction and fixed-work adaptive search behavior.

    Raises:
        ValueError: If the trusted reference source cannot be loaded.
        AssertionError: If state fields, RNGs or complete search fingerprints differ.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    name = "_rko_materialized_states_reference"
    spec = importlib.util.spec_from_file_location(name, args.reference)
    if spec is None or spec.loader is None:
        raise ValueError("Reference must be a trusted Python source file.")
    reference = importlib.util.module_from_spec(spec)
    sys.modules[name] = reference
    spec.loader.exec_module(reference)
    production = QLearningController._create_states
    methods = {"materialized": reference.QLearningController._create_states, "filtered": production}
    clock = MetaheuristicBase.elapsed
    python_state, numpy_state = random.getstate(), np.random.get_state()
    predicates: dict[str, Callable[[list[float]], bool] | None] = {
        "all": None,
        "slice": _slice,
        "singleton": _singleton,
    }
    try:
        print(json.dumps({"python": sys.version, "numpy": np.__version__}), flush=True)
        for width, axes in ((1, 4), (3, 4), (5, 4), (7, 4), (5, 6)):
            grid = [[float(index) for index in range(width)] for _ in range(axes)]
            for mode, predicate in predicates.items():
                if axes == 6 and mode == "all":
                    continue
                for method in methods.values():
                    method(grid, predicate)
                wall: dict[str, list[float]] = {label: [] for label in methods}
                cpu: dict[str, list[float]] = {label: [] for label in methods}
                for repeat in range(7):
                    rows = {}
                    labels = list(methods)
                    if repeat % 2:
                        labels.reverse()
                    for label in labels:
                        gc.collect()
                        random.seed(42 + repeat)
                        started = time.perf_counter()
                        started_cpu = time.process_time()
                        states = methods[label](grid, predicate)
                        cpu[label].append(time.process_time() - started_cpu)
                        wall[label].append(time.perf_counter() - started)
                        rows[label] = _fingerprint(states)
                        count = len(states)
                        del states
                    assert rows["materialized"] == rows["filtered"]
                memory = {}
                for label, method in methods.items():
                    gc.collect()
                    random.seed(42)
                    tracemalloc.start()
                    try:
                        states = method(grid, predicate)
                        memory[label] = tracemalloc.get_traced_memory()
                    finally:
                        tracemalloc.stop()
                    del states
                print(
                    json.dumps(
                        {
                            "kind": "graph",
                            "combinations": width**axes,
                            "accepted_states": count,
                            "filter": mode,
                            "median_wall_speedup": statistics.median(
                                a / b
                                for a, b in zip(wall["materialized"], wall["filtered"], strict=True)
                            ),
                            "median_cpu_speedup": statistics.median(
                                a / b
                                for a, b in zip(cpu["materialized"], cpu["filtered"], strict=True)
                            ),
                            "retained_peak_traced_bytes": memory,
                        }
                    ),
                    flush=True,
                )
        MetaheuristicBase.elapsed = _frozen_elapsed
        for dimension in (32, 1000):
            for workload in ("sphere", "rugged", "ordering"):
                for method in methods.values():
                    QLearningController._create_states = staticmethod(method)
                    _run(OnlinePSO, workload, 41, dimension=dimension, evaluation_limit=200)
                wall_ratios, cpu_ratios = [], []
                for repeat in range(3):
                    for seed in (42, 43, 44):
                        rows = {}
                        labels = list(methods)
                        if (repeat + seed) % 2:
                            labels.reverse()
                        for label in labels:
                            QLearningController._create_states = staticmethod(methods[label])
                            rows[label] = _run(OnlinePSO, workload, seed, dimension=dimension)
                        old, new = rows["materialized"], rows["filtered"]
                        assert {
                            key: value
                            for key, value in old.items()
                            if key not in ("seconds", "cpu_seconds")
                        } == {
                            key: value
                            for key, value in new.items()
                            if key not in ("seconds", "cpu_seconds")
                        }
                        wall_ratios.append(old["seconds"] / new["seconds"])
                        cpu_ratios.append(old["cpu_seconds"] / new["cpu_seconds"])
                print(
                    json.dumps(
                        {
                            "kind": "search",
                            "dimension": dimension,
                            "workload": workload,
                            "pairs": len(wall_ratios),
                            "median_wall_speedup": statistics.median(wall_ratios),
                            "median_cpu_speedup": statistics.median(cpu_ratios),
                        }
                    ),
                    flush=True,
                )
    finally:
        QLearningController._create_states = staticmethod(production)
        MetaheuristicBase.elapsed = clock
        random.setstate(python_state)
        np.random.set_state(numpy_state)


if __name__ == "__main__":
    main()
