"""Q-learning graph initialization and fixed-work adaptive search comparisons.

Objective:
- Measure indexed state adjacency without changing states, edge order or RNGs.

Technical approach:
- Keep the original graph builder as an executable pairwise reference.
- Alternate warmed builders and compare serialized graphs and RNG states.
- Swap builders on one controller class for complete fixed-work PSO searches.

Flow (pseudocode):
1. Measure increasing four-axis grids with exact graph replay assertions.
2. Measure traced constructor memory separately from unprofiled timings.
3. Compare adaptive searches with a frozen control clock and restore all state.

Important criteria:
- Frozen elapsed time tests search semantics, not real adaptive time behavior.
- Graph timing excludes process startup; search timing includes graph creation.
- Additional signature memory and dense output limits must be reported.

Inputs/Outputs:
- Input: synthetic finite grids, numerical/permutation workloads and fixed seeds.
- Output: JSONL measurements and exact graph/search fingerprints.
"""

from __future__ import annotations

import gc
import hashlib
import json
import numpy as np
import pickle
import random
import statistics
import time
import tracemalloc
from benchmarks.runtime.fixed_work import _run
from collections.abc import Callable
from rko_solver.components.adaptive_control.q_learning import QLearningController
from rko_solver.components.adaptive_control.q_learning import QState
from rko_solver.components.solution_pool import ElitePool
from rko_solver.metaheuristics.base import MetaheuristicBase
from rko_solver.metaheuristics.pso import PSO
from rko_solver.orchestration.types import CostFn
from rko_solver.orchestration.types import DecoderFn


class PairwiseController(QLearningController):
    """Retain the former graph initialization body as the timing reference.

    Responsibilities:
    - Enumerate every state pair before drawing its initial edge weight.
    """

    @staticmethod
    def _create_states(
        parameters: list[list[float]],
        state_filter: Callable[[list[float]], bool] | None = None,
    ) -> list[QState]:
        """Build the original Cartesian graph with exhaustive distance comparisons.

        Args:
            parameters (list[list[float]]): Nonempty parameter axes.
            state_filter (Callable | None): Optional joint feasibility predicate.

        Returns:
            list[QState]: Ordered states with ordered actions and initial weights.
        """
        if not parameters:
            return []

        num_par = len(parameters)
        num_states = 1
        for values in parameters:
            num_states *= max(1, len(values))

        states: list[QState] = []
        for i in range(num_states):
            states.append(QState(label=i, par=[0.0] * num_par))

        n_c = 1
        for j in range(num_par):
            n_c *= len(parameters[j])
            for i in range(num_states):
                column_index = i // (num_states // n_c)
                column_index = column_index % len(parameters[j])
                states[i].par[j] = parameters[j][column_index]

        if state_filter is not None:
            states = [state for state in states if state_filter(state.par)]
            for label, state in enumerate(states):
                state.label = label
            num_states = len(states)

        for i in range(num_states):
            for j in range(num_states):
                distance = 0
                for k in range(num_par):
                    if states[i].par[k] != states[j].par[k]:
                        distance += 1

                if distance <= 1:
                    states[i].Ai.append(j)
                    q0 = random.uniform(0.005, 0.01)
                    states[i].Qa.append(q0)
                    if q0 > states[i].maxQ:
                        states[i].maxQ = q0
                        states[i].maxA = len(states[i].Ai) - 1

        return states


class OnlinePSO(PSO):
    """Supply the same moderate adaptive grid to every fixed-work search.

    Responsibilities:
    - Exercise real controller construction inside the unchanged PSO solve loop.
    """

    def __init__(self, pool: ElitePool, decoder: DecoderFn, cost: CostFn, dimension: int) -> None:
        """Construct adaptive PSO with five values on each of four axes.

        Args:
            pool (ElitePool): Shared local elite pool.
            decoder (DecoderFn): Problem decoder.
            cost (CostFn): Minimization objective.
            dimension (int): Random-key vector length.
        """
        super().__init__(
            pool,
            decoder,
            cost,
            dimension,
            control=1,
            parameters=[
                [1, 2, 3, 4, 5],
                [0.1, 0.3, 0.5, 0.7, 0.9],
                [0.1, 0.3, 0.5, 0.7, 0.9],
                [0.1, 0.2, 0.3, 0.4, 0.5],
            ],
        )


def _frozen_elapsed(self: MetaheuristicBase) -> float:
    """Freeze adaptive scheduling without disabling callback evaluation stopping.

    Args:
        self (MetaheuristicBase): Search whose adaptive clock is isolated.

    Returns:
        float: Constant elapsed value for repeatable adaptive decisions.
    """
    return 0.0


def main() -> None:
    """Compare graph builders and complete seeded adaptive search workloads.

    Raises:
        AssertionError: If graphs, RNG states or search fingerprints differ.
    """
    production = QLearningController._create_states
    elapsed = MetaheuristicBase.elapsed
    python_state, numpy_state = random.getstate(), np.random.get_state()
    methods = {"pairwise": PairwiseController._create_states, "indexed": production}
    try:
        for width in (1, 3, 5, 7):
            grid = [[float(index) for index in range(width)] for _ in range(4)]
            for method in methods.values():
                method(grid)
            timings: dict[str, list[float]] = {label: [] for label in methods}
            for repeat in range(3):
                results = {}
                for label in (
                    ("pairwise", "indexed") if repeat % 2 == 0 else ("indexed", "pairwise")
                ):
                    gc.collect()
                    random.seed(42 + repeat)
                    started = time.perf_counter()
                    states = methods[label](grid)
                    timings[label].append(time.perf_counter() - started)
                    results[label] = (
                        hashlib.sha256(pickle.dumps(states)).hexdigest(),
                        random.getstate(),
                    )
                    del states
                assert results["pairwise"] == results["indexed"]
            memory = {}
            if width in (3, 5):
                for label, method in methods.items():
                    gc.collect()
                    random.seed(42)
                    tracemalloc.start()
                    try:
                        states = method(grid)
                        memory[label] = tracemalloc.get_traced_memory()
                    finally:
                        tracemalloc.stop()
                    del states
            print(
                json.dumps(
                    {
                        "kind": "graph",
                        "states": width**4,
                        "seconds": timings,
                        "median_speedup": statistics.median(
                            a / b
                            for a, b in zip(timings["pairwise"], timings["indexed"], strict=True)
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
                wall, cpu = [], []
                for repeat in range(3):
                    for seed in (42, 43, 44):
                        results = {}
                        for label in (
                            ("pairwise", "indexed") if repeat % 2 == 0 else ("indexed", "pairwise")
                        ):
                            gc.collect()
                            QLearningController._create_states = staticmethod(methods[label])
                            results[label] = _run(OnlinePSO, workload, seed, dimension=dimension)
                        old, new = results["pairwise"], results["indexed"]
                        assert {
                            key: value
                            for key, value in old.items()
                            if key not in ("seconds", "cpu_seconds")
                        } == {
                            key: value
                            for key, value in new.items()
                            if key not in ("seconds", "cpu_seconds")
                        }
                        wall.append(old["seconds"] / new["seconds"])
                        cpu.append(old["cpu_seconds"] / new["cpu_seconds"])
                        print(
                            json.dumps(
                                {
                                    "kind": "search",
                                    "dimension": dimension,
                                    "workload": workload,
                                    "seed": seed,
                                    "repeat": repeat,
                                    "runs": results,
                                }
                            ),
                            flush=True,
                        )
                print(
                    json.dumps(
                        {
                            "kind": "summary",
                            "dimension": dimension,
                            "workload": workload,
                            "pairs": len(wall),
                            "median_wall_speedup": statistics.median(wall),
                            "median_cpu_speedup": statistics.median(cpu),
                        }
                    ),
                    flush=True,
                )
    finally:
        QLearningController._create_states = staticmethod(production)
        MetaheuristicBase.elapsed = elapsed
        random.setstate(python_state)
        np.random.set_state(numpy_state)


if __name__ == "__main__":
    main()
