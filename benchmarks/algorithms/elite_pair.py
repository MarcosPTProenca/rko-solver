"""Managed IPR pair transfer against the former full-snapshot search.

Objective:
- Measure selected elite transfer with exact full-search replay checks.

Technical approach:
- Swap original/production solve methods on the same IPR class.
- Use a real spawned pool server with fresh workload-consistent elites.
- Warm both paths, alternate paired order and report client CPU separately.

Flow (pseudocode):
1. Start one server and measure full/pair retrieval and serialized entry sizes.
2. Replay numerical/permutation searches with identical seeds and work limits.
3. Assert result, final-pool and RNG fingerprints; restore methods and streams.

Important criteria:
- No concurrent portfolio writers are simulated by this timing benchmark.
- Startup, setup and final fingerprint queries are outside search timing.
- Client CPU excludes manager CPU; wall time includes RPC waits.

Inputs/Outputs:
- Input: packaged defaults and synthetic workloads; no external datasets.
- Output: JSONL timing records and exact-equivalence assertions.
"""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import numpy as np
import pickle
import random
import statistics
import time
from benchmarks.runtime.fixed_work import _run
from rko_solver.metaheuristics.ipr import IPR
from rko_solver.orchestration.processes import PoolManager
from typing import override


class SnapshotIPR(IPR):
    """Retain the former search body as an executable baseline.

    Responsibilities:
    - Fetch all elite vectors before selecting two client-side ranks.
    """

    @override
    def solve(self, limit_time: float) -> tuple[list[object], list[float], float]:
        """Run original full-snapshot relinking with the production numeric kernel.

        Args:
            limit_time (float): Cooperative wall-clock duration.

        Returns:
            tuple[list[object], list[float], float]: Extras, best keys and cost.
        """
        self._start_timer()

        best_keys: np.ndarray = np.random.random(self.tam_solution)
        best_cost: float = self.cost(self.decoder(best_keys))

        while not self.time_limit_reached(limit_time):
            pool_size: int = getattr(self.solution_pool, "size", 0)
            if pool_size < 2:
                time.sleep(min(0.01, max(0.0, limit_time - self.elapsed())))
                continue

            elite = self.solution_pool.get_elite(pool_size)
            indices = list(range(len(elite)))
            k1, k2 = random.sample(indices, 2)
            atual_keys = np.array(elite[k1][1], dtype=float)
            guia_keys = np.array(elite[k2][1], dtype=float)

            path_keys, path_cost = self._relink(atual_keys, guia_keys)
            self.solution_pool.insert((path_cost, path_keys.tolist()), self.name)

            if path_cost < best_cost:
                best_cost = path_cost
                best_keys = path_keys.copy()

        return [], best_keys.tolist(), best_cost


def main() -> None:
    """Measure sparse RPC transfer and paired complete managed-pool searches.

    Raises:
        AssertionError: If a replay fingerprint or expected pair differs.
        RuntimeError: If the spawned manager cannot complete a request.
    """
    production = IPR.solve
    methods = {"snapshot": SnapshotIPR.solve, "pair": production}
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with PoolManager(ctx=mp.get_context("spawn")) as manager:
            for dimension in (32, 1000, 10000):
                pool = manager.SolutionPool(10)
                for index in range(10):
                    pool.insert((float(index), [index / 10] * dimension))
                full = pool.get_elite(10)
                pair = pool._sample_elite_pair(10)
                sizes = {"snapshot": len(pickle.dumps(full)), "pair": len(pickle.dumps(pair))}
                rpc: dict[str, list[float]] = {label: [] for label in methods}
                for repeat in range(7):
                    for label in ("snapshot", "pair") if repeat % 2 == 0 else ("pair", "snapshot"):
                        fetch = pool.get_elite if label == "snapshot" else pool._sample_elite_pair
                        fetch(10)
                        started = time.perf_counter()
                        for _ in range(200):
                            fetch(10)
                        rpc[label].append((time.perf_counter() - started) / 200)
                print(
                    json.dumps(
                        {"dimension": dimension, "entry_pickle_bytes": sizes, "rpc_seconds": rpc}
                    ),
                    flush=True,
                )
                for workload in ("sphere", "rugged", "ordering"):
                    for method in methods.values():
                        IPR.solve = method
                        _run(
                            IPR,
                            workload,
                            41,
                            dimension=dimension,
                            evaluation_limit=500,
                            pool=manager.SolutionPool(10),
                        )
                    ratios, cpu_ratios = [], []
                    for repeat in range(3):
                        for seed in (42, 43, 44):
                            runs = {}
                            for label in (
                                ("snapshot", "pair") if repeat % 2 == 0 else ("pair", "snapshot")
                            ):
                                gc.collect()
                                IPR.solve = methods[label]
                                runs[label] = _run(
                                    IPR,
                                    workload,
                                    seed,
                                    dimension=dimension,
                                    evaluation_limit=20000 if dimension == 32 else 2000,
                                    pool=manager.SolutionPool(10),
                                )
                            old, new = runs["snapshot"], runs["pair"]
                            assert {
                                key: value
                                for key, value in old.items()
                                if key not in ("seconds", "cpu_seconds")
                            } == {
                                key: value
                                for key, value in new.items()
                                if key not in ("seconds", "cpu_seconds")
                            }
                            ratios.append(old["seconds"] / new["seconds"])
                            cpu_ratios.append(old["cpu_seconds"] / new["cpu_seconds"])
                            print(
                                json.dumps(
                                    {
                                        "dimension": dimension,
                                        "workload": workload,
                                        "seed": seed,
                                        "repeat": repeat,
                                        "runs": runs,
                                    }
                                ),
                                flush=True,
                            )
                    print(
                        json.dumps(
                            {
                                "dimension": dimension,
                                "workload": workload,
                                "pairs": len(ratios),
                                "median_wall_speedup": statistics.median(ratios),
                                "median_client_cpu_speedup": statistics.median(cpu_ratios),
                            }
                        ),
                        flush=True,
                    )
    finally:
        IPR.solve = production
        random.setstate(python_state)
        np.random.set_state(numpy_state)


if __name__ == "__main__":
    main()
