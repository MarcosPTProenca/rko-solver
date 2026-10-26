"""Measure the existing RKO on one pinned 1,000-item public Knapsack instance.

Objective:
- Check solution quality, feasibility and end-to-end CPU use beyond smoke scale.

Technical approach:
- Verify the downloaded KPLIB file fingerprint and compute exact 0/1 DP with NumPy.
- Run the same four-algorithm portfolio with three seeds under both execution modes.
- Measure parent plus reaped-child CPU seconds using Linux resource accounting.

Flow (pseudocode):
1. Validate data and self-check the independent oracle on bundled instances.
2. Compute exact and greedy references; copy configuration into a temporary folder.
3. Run searches, independently check returned decisions, and write CSV results.

Important criteria:
- Search budgets are equal; startup is included only in end-to-end measurements.
- Thread seeds initialize shared RNGs and do not reproduce process RNG streams.
- Linux/WSL only: resource accounting is not a Windows portability test.

Inputs/Outputs:
- Input: pinned s000.kp path and output CSV path as positional arguments.
- Output: reference objectives, console results, and per-trial CSV measurements.
"""

import argparse
import csv
import hashlib
import numpy as np
import random
import resource
import shutil
import tempfile
import time
from benchmarks.knapsack_utils import exact_profit
from pathlib import Path
from rko_solver import RKO
from rko_solver.examples.run_kp_cpp_instances import KpInstance
from rko_solver.examples.run_kp_cpp_instances import decode_knapsack
from rko_solver.examples.run_kp_cpp_instances import default_instances_dir
from rko_solver.examples.run_kp_cpp_instances import make_cost
from rko_solver.examples.run_kp_cpp_instances import read_kp_instance
from rko_solver.parameters.loader import default_params_dir


def main() -> None:
    """Run a fixed, auditable experiment without modifying production configuration.

    Raises:
        AssertionError: If data, oracle self-checks, or returned solution accounting fail.
        OSError: If benchmark files cannot be read or written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256(args.instance.read_bytes()).hexdigest()
    assert digest == "8f555b2b40dff1298d841a8e5abc327a8f1135ce8188e84249cd1352f959ffed"
    for filename, optimum in (("kp10.txt", 31), ("kp50.txt", 149)):
        assert exact_profit(read_kp_instance(default_instances_dir() / filename)) == optimum
    assert exact_profit(KpInstance(1, 4, [3], [2], "single")) == 3
    instance = read_kp_instance(args.instance)
    assert (instance.n_items, instance.capacity) == (1000, 252480)
    started = time.perf_counter()
    optimum = exact_profit(instance)
    oracle_seconds = time.perf_counter() - started
    greedy_profit = 0
    greedy_weight = 0
    for profit, weight in sorted(
        zip(instance.benefits, instance.weights, strict=True),
        key=lambda item: item[0] / item[1],
        reverse=True,
    ):
        if greedy_weight + weight <= instance.capacity:
            greedy_weight += weight
            greedy_profit += profit
    print(f"Exact={optimum}, DP seconds={oracle_seconds:.3f}, greedy={greedy_profit}", flush=True)
    fields = [
        "backend",
        "seed",
        "budget_seconds",
        "status",
        "profit",
        "weight",
        "feasible",
        "cost",
        "optimum",
        "gap_percent",
        "elapsed_seconds",
        "cpu_seconds",
        "average_cpu_cores",
        "error",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rko-large-") as temporary:
        params = Path(temporary)
        for source in default_params_dir().glob("*.toml"):
            shutil.copy2(source, params / source.name)
        (params / "config.toml").write_text(
            "[run]\ncontrol = 1\npool_size = 10\n"
            '[active_metaheuristics]\nlist = ["SA", "BRKGA", "GA", "PSO"]\n',
            encoding="utf-8",
        )
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for seed in (42, 43, 44):
                for execution in ("process", "thread"):
                    random.seed(seed)
                    np.random.seed(seed)
                    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
                    parent_before = time.process_time()
                    started = time.perf_counter()
                    row = dict.fromkeys(fields, "")
                    row.update(backend=execution, seed=seed, budget_seconds=15, optimum=optimum)
                    try:
                        result = RKO(
                            decode_knapsack,
                            make_cost(instance),
                            instance.n_items,
                            params_dir=params,
                            execution=execution,
                            n_jobs=4 if execution == "process" else None,
                            random_state=seed if execution == "process" else None,
                        ).solve(15.0)
                        keys, cost = result.keys, result.cost
                        assert len(keys) == instance.n_items
                        selected = [key > 0.5 for key in keys]
                        profit = sum(
                            p
                            for p, chosen in zip(instance.benefits, selected, strict=True)
                            if chosen
                        )
                        weight = sum(
                            w
                            for w, chosen in zip(instance.weights, selected, strict=True)
                            if chosen
                        )
                        feasible = weight <= instance.capacity
                        assert cost == 100000 * max(0, weight - instance.capacity) - profit
                        assert not feasible or profit <= optimum
                        row.update(
                            status="ok",
                            profit=profit,
                            weight=weight,
                            feasible=feasible,
                            cost=cost,
                            gap_percent=100 * (optimum - profit) / optimum if feasible else "",
                        )
                    except (RuntimeError, TimeoutError) as error:
                        row.update(status="error", error=f"{type(error).__name__}: {error}")
                    elapsed = time.perf_counter() - started
                    children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
                    cpu = (
                        time.process_time()
                        - parent_before
                        + children_after.ru_utime
                        - children_before.ru_utime
                        + children_after.ru_stime
                        - children_before.ru_stime
                    )
                    row.update(
                        elapsed_seconds=elapsed, cpu_seconds=cpu, average_cpu_cores=cpu / elapsed
                    )
                    writer.writerow(row)
                    stream.flush()
                    print(row, flush=True)


if __name__ == "__main__":
    main()
