"""Process-mode elite-pool cost for one configuration (issue 136 measurement).

Objective:
- Measure the share of worker wall time spent in elite-pool manager calls during real process-mode
  solves, for a chosen algorithm set and fixture.

Technical approach:
- Import the timing probe before the solver so spawned workers are instrumented too.
- Run one `RKO.from_model` solve with `execution="process"` and aggregate every process's totals
  file written by the probe; the probe's last partial second per process is not counted.
- Report pool seconds, alive seconds, their ratio, calls per method and calls per 1000 evaluations.

Flow (pseudocode):
1. Require RKO_POOL_PROBE_DIR to name an existing empty directory.
2. Build the fixture and the native MathOpt model; import the probe first.
3. Solve in process mode for the budget and read every per-process totals file.
4. Print one JSON record; the caller runs one configuration per process invocation.

Important criteria:
- Process-mode only; thread and sequential runs are not represented.
- One host per run; hosts must be compared separately.
- Probe totals exclude each worker's unflushed tail, so the share is a slight underestimate.

Inputs/Outputs:
- Input: fixture spec (`hard:ITEMSxROWS` or `practical:NAME`), algorithms, seconds, seed.
- Output: one JSON record on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from ..modeling.cases import PracticalModels
from ..modeling.hard_cases import HardCases
from . import pool_call_probe  # noqa: F401  (must import before the solver: instruments workers)
from pathlib import Path
from rko_solver import RKO


def _case(spec: str) -> object:
    """Return the fixture named by `hard:ITEMSxROWS` or `practical:NAME`."""
    kind, _, value = spec.partition(":")
    if kind == "hard":
        items, rows = (int(x) for x in value.split("x"))
        return HardCases.build(items, rows, 1)
    if kind == "practical":
        return next(c for c in PracticalModels.cases() if c.name == value)
    raise ValueError("Fixture must be 'hard:ITEMSxROWS' or 'practical:NAME'.")


def _read_processes(directory: Path) -> list[dict[str, object]]:
    """Load one summary per process from the probe's totals files."""
    summaries = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        methods = data["methods"]
        summaries.append(
            {
                "pid": data["pid"],
                "worker": bool(set(methods) & {"insert", "_get_elite_pair"}),
                "alive_seconds": data["alive_seconds"],
                "pool_seconds": sum(item["seconds"] for item in methods.values()),
                "methods": methods,
            }
        )
    return summaries


def _merge_methods(processes: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """Add calls and seconds of every method across the given processes."""
    merged: dict[str, dict[str, float]] = {}
    for process in processes:
        for name, item in process["methods"].items():
            bucket = merged.setdefault(name, {"calls": 0.0, "seconds": 0.0})
            bucket["calls"] += item["calls"]
            bucket["seconds"] += item["seconds"]
    return merged


def _aggregate(directory: Path) -> dict[str, object]:
    """Sum every process's totals file, separating search workers from the parent."""
    processes = _read_processes(directory)
    workers = [p for p in processes if p["worker"]]
    alive = sum(p["alive_seconds"] for p in workers)
    pool = sum(p["pool_seconds"] for p in workers)
    return {
        "processes": len(processes),
        "workers": len(workers),
        "worker_alive_seconds": alive,
        "worker_pool_seconds": pool,
        "worker_pool_share": _share(pool, alive),
        "methods": _merge_methods(processes),
        "per_process": processes,
    }


def _share(part: float, whole: float) -> float | None:
    """Return part over whole, or None when there is no time to divide by."""
    return part / whole if whole else None


def main() -> None:
    """Run one process-mode solve and print the aggregated pool record.

    Raises:
        RuntimeError: If the probe directory is missing or not empty.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--algorithms", nargs="+", required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()
    directory = os.environ.get("RKO_POOL_PROBE_DIR")
    if directory is None or not Path(directory).is_dir() or any(Path(directory).iterdir()):
        raise RuntimeError("Set RKO_POOL_PROBE_DIR to an existing empty directory.")
    case = _case(arguments.fixture)
    native = PracticalModels.native(case, "mathopt")
    solver = RKO.from_model(
        native,
        algorithms=arguments.algorithms,
        adaptation="fixed",
        engine="python",
        execution="process",
        random_state=arguments.seed,
    )
    result = solver.solve(arguments.seconds)
    evaluations = sum(search.n_evaluations for search in result.searches)
    record = {
        "fixture": arguments.fixture,
        "algorithms": arguments.algorithms,
        "seconds": arguments.seconds,
        "seed": arguments.seed,
        "evaluations": evaluations,
        **_aggregate(Path(directory)),
    }
    if evaluations:
        calls = sum(m["calls"] for m in record["methods"].values())
        record["calls_per_1000_evaluations"] = 1000.0 * calls / evaluations
    print(json.dumps(record), file=sys.stdout, flush=True)


if __name__ == "__main__":
    main()
