"""Compare fixed and remaining-time process scheduling on neutral problems.

Objective:
- Measure unused worker search time, evaluations and solution costs before adoption.

Technical approach:
- Alternate trusted baseline/current source trees using the same interpreter and dependencies.
- Reuse the process-resource benchmark's seeded decoder and numerical/ordering objectives.
- Measure real spawned searches, not mocked clocks or compilation microbenchmarks.

Flow (pseudocode):
1. Launch balanced, uneven and full-portfolio cases under each source tree.
2. Validate all assigned seeds, first candidates, reports and independent result costs.
3. Emit per-search durations/counts and per-worker unused session time as JSONL.

Important criteria:
- Baseline source is executable trusted code; use a local archived checkout only.
- Both source trees should reside on the same filesystem, with identical dependencies.
- Session wall time includes waiting and callback work; it is not CPU utilization.
- Changed timed budgets/cooperation preclude exact trajectories or quality guarantees.
- Native launch settings are copied into child environments; the caller is unchanged.

Inputs/Outputs:
- Input: baseline src directory, repeats and positive finite common budget.
- Output: provenance and alternating trial records, or failed accounting assertions.
"""

import argparse
import json
import math
import multiprocessing as mp
import numpy as np
import os
import platform
import rko_solver
import subprocess
import sys
import tempfile
from pathlib import Path
from process_budget import Decoder
from process_budget import Objective
from rko_solver import RKO
from rko_solver.orchestration.configuration import parse_run_config
from rko_solver.parameters.loader import load_config


def main() -> None:
    """Compare actual source trees or execute one internal spawned-portfolio trial.

    Raises:
        AssertionError: If workers, assigned streams, reports or independent costs disagree.
        OSError: If source trees or first-candidate observations cannot be read.
        subprocess.SubprocessError: If a trial fails or exceeds startup/search allowance.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--budget", type=float, default=1.2)
    parser.add_argument("--trial", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=("balanced", "uneven", "portfolio"), default="uneven")
    parser.add_argument("--workload", choices=("sphere", "ordering"), default="sphere")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repeats < 1 or not math.isfinite(args.budget) or args.budget <= 0:
        parser.error("repeats and the finite common budget must be positive")
    if args.trial:
        dimension = 64
        if args.case == "portfolio":
            names = parse_run_config(load_config()).active_metaheuristics
            workers = 4
        else:
            names = ("SA", "GA", "PSO", "BRKGA", "ILS")[: 4 if args.case == "balanced" else 5]
            workers = 2
        seeds = [
            int(stream.generate_state(1)[0])
            for stream in np.random.SeedSequence(args.seed).spawn(len(names))
        ]
        expected = {}
        for index, (name, seed) in enumerate(zip(names, seeds, strict=True)):
            keys = tuple(np.random.RandomState(seed).random_sample(dimension))
            expected[keys[0]] = (index, name, keys)
        before = {child.pid for child in mp.active_children()}
        with tempfile.TemporaryDirectory(prefix="rko-schedule-") as directory:
            ordering = args.workload == "ordering"
            objective = Objective(ordering)
            result = RKO(
                Decoder(Path(directory), expected, ordering),
                objective,
                dimension,
                algorithms=names,
                control=0,
                n_jobs=workers,
                random_state=args.seed,
            ).solve(args.budget)
            observations = [
                json.loads(Path(directory, f"{index}.json").read_text())
                for index in range(len(names))
            ]
            assert len({row["pid"] for row in observations}) == result.n_workers == workers
            assert len(result.searches) == len(names)
            assert [report.algorithm for report in result.searches] == list(names)
            assert [report.random_state for report in result.searches] == seeds
            assert all(report.n_evaluations > 0 for report in result.searches)
            assert result.n_evaluations == sum(r.n_evaluations for r in result.searches)
            assert all(0 <= key < 1 for key in result.keys) and len(result.keys) == dimension
            values = np.argsort(result.keys, kind="stable") if ordering else np.asarray(result.keys)
            assert math.isclose(result.cost, objective(values), rel_tol=1e-12, abs_tol=1e-12)
            sessions = [
                sum(report.elapsed_seconds for report in result.searches[index::workers])
                for index in range(workers)
            ]
            print(
                json.dumps(
                    {
                        "source": str(Path(rko_solver.__file__).resolve().parent.parent),
                        "case": args.case,
                        "workload": args.workload,
                        "seed": args.seed,
                        "budget_seconds": args.budget,
                        "workers": workers,
                        "dimension": dimension,
                        "elapsed_seconds": result.elapsed_seconds,
                        "cost": result.cost,
                        "known_minimum": dimension * (dimension - 1) * (dimension + 1) / 6
                        if ordering
                        else 0,
                        "n_evaluations": result.n_evaluations,
                        "not_started": sum(r.stop_reason == "not_started" for r in result.searches),
                        "worker_session_seconds": sessions,
                        "unused_worker_session_seconds": sum(
                            max(0.0, args.budget - s) for s in sessions
                        ),
                        "searches": [
                            {
                                "algorithm": r.algorithm,
                                "seed": r.random_state,
                                "n_evaluations": r.n_evaluations,
                                "stop_reason": r.stop_reason,
                                "time_limit": r.time_limit,
                                "elapsed_seconds": r.elapsed_seconds,
                            }
                            for r in result.searches
                        ],
                    }
                ),
                flush=True,
            )
        assert {child.pid for child in mp.active_children()} == before
        return
    current = Path(rko_solver.__file__).resolve().parent.parent
    if args.baseline_source is None or not (args.baseline_source / "rko" / "__init__.py").is_file():
        parser.error("--baseline-source must identify a trusted archived checkout's src directory")
    baseline = args.baseline_source.resolve()
    if baseline == current:
        parser.error("baseline and current source trees must differ")
    inherited = dict(os.environ)
    environment = inherited | dict.fromkeys(
        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"), "1"
    )
    print(
        json.dumps(
            {
                "kind": "environment",
                "python": sys.version,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "available_cpus": os.process_cpu_count(),
                "baseline_source": str(baseline),
                "current_source": str(current),
                "budget_seconds": args.budget,
                "repeats": args.repeats,
                "native_launch_threads": 1,
                "baseline_is_trusted_executable_code": True,
            }
        ),
        flush=True,
    )
    for case in ("balanced", "uneven", "portfolio"):
        for workload in ("sphere", "ordering"):
            for repeat in range(args.repeats):
                for index, seed in enumerate((42, 43, 44)):
                    modes = (
                        ("baseline", "current")
                        if (index + repeat) % 2 == 0
                        else ("current", "baseline")
                    )
                    for mode in modes:
                        source = baseline if mode == "baseline" else current
                        process = subprocess.run(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--trial",
                                "--case",
                                case,
                                "--workload",
                                workload,
                                "--seed",
                                str(seed),
                                "--budget",
                                str(args.budget),
                            ],
                            env=environment | {"PYTHONPATH": str(source)},
                            stdout=subprocess.PIPE,
                            text=True,
                            check=True,
                            timeout=max(90.0, args.budget + 90.0),
                        )
                        row = json.loads(process.stdout)
                        assert row["source"] == str(source)
                        print(json.dumps({"mode": mode, "repeat": repeat, **row}), flush=True)
                    assert dict(os.environ) == inherited


if __name__ == "__main__":
    main()
