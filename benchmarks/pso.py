"""Paired fixed-work PSO comparison with the former scalar movement formula.

Objective:
- Measure full-search effects of vectorizing independent genes within each particle.

Technical approach:
- Reuse the fixed-evaluation runner with real local search and callback isolation.
- Override movement only for the scalar reference; all other search code is shared.
- Alternate execution order and assert exact keys, cost, counts and both RNG states.
- Report wall/CPU time and separately trace movement's temporary allocations.

Flow (pseudocode):
1. Warm NumPy numerical/permutation operations outside timing.
2. Run three seeds and three paired repeats for each dimension and workload.
3. Print JSON records and per-case median paired speed ratios.

Important criteria:
- Particle order and sequential global-best updates are unchanged.
- All runs use 2,000 evaluations and offline default PSO coefficients/population.
- Timing is serial; it excludes process startup and expensive user callbacks.
- Scalar-reference method calls make tiny-dimensional results approximate legacy overhead.

Inputs/Outputs:
- Input: synthetic sphere, rugged and permutation problems with 8/32/100/1,000 keys.
- Output: JSON lines containing exact replay fingerprints and paired search times.
"""

import json
import numpy as np
import random
import statistics
import sys
import tracemalloc
from fixed_work import _run
from rko_solver.metaheuristics.pso import PSO
from typing import override


class ScalarPSO(PSO):
    """Replay the pre-vectorization movement formula within actual PSO searches.

    Responsibilities:
    - Retain the former two-draw-per-gene arithmetic and boundary behavior.
    - Share the production population, callback, local-search and pool paths.

    Attributes:
        tam_solution (int): Chromosome length inherited from PSO.
    """

    @staticmethod
    @override
    def _move_particle(
        position: np.ndarray,
        velocity: np.ndarray,
        personal_best: np.ndarray,
        global_best: np.ndarray,
        c1: float,
        c2: float,
        inertia: float,
    ) -> None:
        """Replay the former scalar movement, mutating position and velocity in place.

        Args:
            position (np.ndarray): Current particle keys.
            velocity (np.ndarray): Current particle velocities.
            personal_best (np.ndarray): Detached personal-best keys.
            global_best (np.ndarray): Detached current global-best keys.
            c1 (float): Cognitive acceleration coefficient.
            c2 (float): Social acceleration coefficient.
            inertia (float): Complete velocity-update multiplier.
        """
        for j in range(len(position)):
            r1 = random.random()
            r2 = random.random()
            velocity[j] = inertia * (
                velocity[j]
                + c1 * r1 * (personal_best[j] - position[j])
                + c2 * r2 * (global_best[j] - position[j])
            )
            old_key = position[j]
            position[j] = position[j] + velocity[j]
            if position[j] < 0.0 or position[j] >= 1.0:
                position[j] = old_key
                velocity[j] = 0.0


def main() -> None:
    """Print paired search measurements while preserving caller RNG state.

    Raises:
        AssertionError: If scalar and production search trajectories differ.
    """
    saved_python, saved_numpy = random.getstate(), np.random.get_state()
    try:
        np.dot(np.ones(1000), np.ones(1000))
        np.cos(np.ones(1000))
        np.argsort(np.ones(1000), kind="stable")
        print(json.dumps({"python": sys.version, "numpy": np.__version__}), flush=True)
        for dimension in (8, 32, 100, 1000):
            allocation_peaks = {}
            for algorithm_type in (ScalarPSO, PSO):
                arrays = [np.linspace(0, 1, dimension) for _ in range(4)]
                tracemalloc.start()
                try:
                    algorithm_type._move_particle(*arrays, 2.0, 2.0, 0.7)
                    allocation_peaks[algorithm_type.__name__] = tracemalloc.get_traced_memory()[1]
                finally:
                    tracemalloc.stop()
            print(
                json.dumps({"dimension": dimension, "movement_peak_bytes": allocation_peaks}),
                flush=True,
            )
            for workload in ("sphere", "rugged", "ordering"):
                speedups = []
                cpu_speedups = []
                for seed in range(200, 203):
                    for repeat in range(3):
                        labels = ["scalar", "production"]
                        if (seed + repeat) % 2:
                            labels.reverse()
                        rows = {
                            label: _run(
                                ScalarPSO if label == "scalar" else PSO,
                                workload,
                                seed,
                                dimension=dimension,
                            )
                            for label in labels
                        }
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
        random.setstate(saved_python)
        np.random.set_state(saved_numpy)


if __name__ == "__main__":
    main()
