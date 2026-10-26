"""Paired fixed-work IPR comparison against former per-gene block copies.

Objective:
- Measure full-search effects of moving sufficiently large blocks through NumPy.

Technical approach:
- Swap only block-copy kernels under the CURRENT no-op-pruned callback policy.
- Use the same IPR class to avoid subclass-dispatch timing bias.
- Seed elites with each workload's real objective through the fixed-work runner.
- Warm both variants, alternate order and compare results, pools and RNG states.

Flow (pseudocode):
1. Measure relinking allocation peaks separately from timing.
2. Run three seeds and three paired repeats per dimension/problem family.
3. Assert exact fingerprints and report median paired wall/CPU ratios.

Important criteria:
- Use 20,000 evaluations through 80 keys and 2,000 at larger dimensions.
- Shared callback-protecting copies and elite-pool operations remain enabled.
- Synthetic serial timing excludes process startup, manager IPC and costly callbacks.
- Allocation peaks are traced Python/NumPy allocations, not process RSS.

Inputs/Outputs:
- Input: sphere, rugged and stable-permutation problems; 1–10,000 keys.
- Output: JSON allocation, exact replay and paired full-search timing records.
"""

import json
import math
import numpy as np
import random
import statistics
import sys
import tracemalloc
from fixed_work import _run
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.ipr import IPR
from typing import override


class ScalarIPR(IPR):
    """Retain per-gene copying under current path/callback policy for kernel comparison.

    Responsibilities:
    - Share current no-op pruning and virtual steps, varying only guide assignment.
    - Share all other production search and callback behavior.

    Attributes:
        tam_solution (int): Positive chromosome dimension inherited from IPR.
    """

    @override
    def _relink(self, atual_keys: np.ndarray, guia_keys: np.ndarray) -> tuple[np.ndarray, float]:
        """Replay the former greedy block path without slice assignment.

        Args:
            atual_keys (np.ndarray): Starting chromosome.
            guia_keys (np.ndarray): Guide chromosome.

        Returns:
            tuple[np.ndarray, float]: First best path chromosome and its objective.
        """
        n = self.tam_solution
        block_size = max(1, int(n * 0.10))
        num_blocks = (n + block_size - 1) // block_size
        fixed_block = [False] * num_blocks
        s_current = atual_keys.copy()
        best_path_keys = s_current.copy()
        best_path_cost: float = self.cost(self.decoder(s_current))
        current_cost = best_path_cost
        for _ in range(max(1, num_blocks - 1)):
            best_iter_cost = math.inf
            best_iter_keys: np.ndarray | None = None
            best_block = -1
            for b in range(num_blocks):
                if fixed_block[b]:
                    continue
                start = b * block_size
                end = min((b + 1) * block_size, n)
                if (
                    s_current.dtype == guia_keys.dtype
                    and s_current[start:end].tobytes() == guia_keys[start:end].tobytes()
                ):
                    s_viz = s_current
                    viz_cost = current_cost
                else:
                    s_viz = s_current.copy()
                    for k in range(start, end):
                        s_viz[k] = guia_keys[k]
                    viz_cost = self.cost(self.decoder(s_viz))
                if viz_cost < best_iter_cost:
                    best_iter_cost = viz_cost
                    best_iter_keys = s_viz.copy()
                    best_block = b
            if best_iter_cost < best_path_cost and best_iter_keys is not None:
                best_path_cost = best_iter_cost
                best_path_keys = best_iter_keys.copy()
            if best_block >= 0 and best_iter_keys is not None:
                s_current = best_iter_keys.copy()
                current_cost = best_iter_cost
                fixed_block[best_block] = True
        return best_path_keys, best_path_cost


def main() -> None:
    """Measure paired IPR searches and allocation peaks without retaining caller RNG changes.

    Raises:
        AssertionError: If search results, final elites or RNG fingerprints differ.
    """
    saved_python, saved_numpy = random.getstate(), np.random.get_state()
    production_relink = IPR._relink
    try:
        print(json.dumps({"python": sys.version, "numpy": np.__version__}), flush=True)
        for dimension in (1, 8, 32, 79, 80, 1000, 10000):
            IPR._relink = production_relink
            generator = np.random.RandomState(199)
            arrays = [generator.random_sample(dimension) for _ in range(2)]
            allocation_peaks = {}
            for algorithm_type in (ScalarIPR, IPR):
                algorithm = algorithm_type(
                    SolutionPool(),
                    lambda keys: keys,
                    lambda keys: float(np.dot(keys, keys)),
                    dimension,
                )
                algorithm._relink(*arrays)
                tracemalloc.start()
                try:
                    algorithm._relink(*arrays)
                    allocation_peaks[algorithm_type.__name__] = tracemalloc.get_traced_memory()[1]
                finally:
                    tracemalloc.stop()
            print(
                json.dumps({"dimension": dimension, "relink_peak_bytes": allocation_peaks}),
                flush=True,
            )
            evaluation_limit = 20000 if dimension <= 80 else 2000
            for workload in ("sphere", "rugged", "ordering"):
                for method in (ScalarIPR._relink, production_relink):
                    IPR._relink = method
                    _run(
                        IPR,
                        workload,
                        199,
                        dimension=dimension,
                        evaluation_limit=evaluation_limit,
                    )
                speedups = []
                cpu_speedups = []
                for seed in range(200, 203):
                    for repeat in range(3):
                        labels = ["scalar", "production"]
                        if (seed + repeat) % 2:
                            labels.reverse()
                        rows = {}
                        for label in labels:
                            IPR._relink = (
                                ScalarIPR._relink if label == "scalar" else production_relink
                            )
                            rows[label] = _run(
                                IPR,
                                workload,
                                seed,
                                dimension=dimension,
                                evaluation_limit=evaluation_limit,
                            )
                        expected = {
                            k: v
                            for k, v in rows["scalar"].items()
                            if k not in ("algorithm", "seconds", "cpu_seconds")
                        }
                        for row in rows.values():
                            assert {
                                k: v
                                for k, v in row.items()
                                if k not in ("algorithm", "seconds", "cpu_seconds")
                            } == expected
                        speedups.append(rows["scalar"]["seconds"] / rows["production"]["seconds"])
                        cpu_speedups.append(
                            rows["scalar"]["cpu_seconds"] / rows["production"]["cpu_seconds"]
                        )
                        print(json.dumps({"repeat": repeat, **rows}), flush=True)
                print(
                    json.dumps(
                        {
                            "dimension": dimension,
                            "workload": workload,
                            "median_paired_speedup": statistics.median(speedups),
                            "median_paired_cpu_speedup": statistics.median(cpu_speedups),
                            "pairs": len(speedups),
                        }
                    ),
                    flush=True,
                )
    finally:
        IPR._relink = production_relink
        random.setstate(saved_python)
        np.random.set_state(saved_numpy)


if __name__ == "__main__":
    main()
