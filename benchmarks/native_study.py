"""Compare the native Rust engine with Python RKO and the C++ reference on Knapsack.

Objective:
- Decide, with paired statistics, whether the native engine is at least as good as
  the pinned C++ reference and the Python engine under equal wall-clock budgets.

Technical approach:
- Reuse the hash-pinned KPLIB instances, exact optima, patched 64-bit C++ build,
  Python worker and independent solution re-scoring from `cpp_python_knapsack`.
- Add a `rust` implementation that runs the `rko_bench` binary built in release mode,
  with the same target telemetry and full-precision returned keys.
- Summarize per-solver instance-level medians with exact sign tests, Holm correction
  and bootstrap intervals for rust-versus-C++ and rust-versus-Python differences.

Flow (pseudocode):
1. Build the C++ reference and the native binary; fetch and verify the instances.
2. For every instance, solver and replicate run all implementations in alternating
   order and independently score each returned key vector.
3. Write raw rows and print the paired summary with explicit wins, ties and losses.

Important criteria:
- Quality is the normalized objective difference to the exact optimum (lower wins);
  infeasible results are penalized by the shared objective, never dropped.
- Every failed trial is kept as a row; no comparison silently discards a failure.
- Implementations are run sequentially, never concurrently, on the same machine.

Inputs/Outputs:
- Input: budget, repetitions, solver names, implementations and cache/output paths.
- Output: raw trial CSV and a printed (and CSV) paired summary.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from benchmarks.cpp_python_knapsack import KPLIB_RAW
from benchmarks.cpp_python_knapsack import KPLIB_REVISION
from benchmarks.cpp_python_knapsack import SAMPLE_SEED
from benchmarks.cpp_python_knapsack import SOLVER_CONFIGS
from benchmarks.cpp_python_knapsack import InstanceCase
from benchmarks.cpp_python_knapsack import KnapsackBenchmark
from benchmarks.cpp_python_knapsack import SolverOutput
from benchmarks.cpp_python_knapsack import TrialRecord
from benchmarks.knapsack_utils import exact_profit
from collections.abc import Sequence
from pathlib import Path
from rko_solver.examples.run_kp_cpp_instances import read_kp_instance

IMPLEMENTATIONS = (
    "python",
    "cpp64",
    "rust",
    "rust-callback",
    "rust-numba",
    "rust-batch",
    "cpp-numba",
    "cpp-engine",
)
PORTFOLIO_NAMES = ["SA", "ILS", "VNS", "BRKGA", "BRKGA_CS", "PSO", "GA", "LNS", "GRASP", "IPR"]
RUST_PORTFOLIO_THREADS = 10


class NativeStudy:
    """Run and summarize the three-way implementation comparison.

    Responsibilities:
    - Build and invoke the native binary under the shared trial protocol.
    - Reuse the historical benchmark for instances, C++ and independent scoring.
    - Produce paired summaries of the native engine against both references.

    Attributes:
        output_path (Path): Raw trial CSV destination.
        budget_seconds (int): Search budget per trial.
        repetitions (int): Independent repetitions per cell.
        solvers (tuple[str, ...]): Display names of the solver configurations.
        implementations (tuple[str, ...]): Implementations to execute.
        rust_grid (str): Native parameter grid flavor, `toml` or `cpp`.
        rust_threads (int | None): Native worker cap for portfolios.
        rust_prefill (bool): Whether the native engine prefills its pool.
        match_reference_time (bool): Whether native budgets follow the C++ wall time.
        instance_seed (int): Seed selecting the KPLIB sample.
        instance_limit (int | None): Optional number of instances per run.
        benchmark (KnapsackBenchmark): Shared protocol helper.
        repo_root (Path): Repository root.
    """

    def __init__(
        self,
        output_path: Path,
        cache_dir: Path,
        *,
        budget_seconds: int,
        repetitions: int,
        solvers: Sequence[str],
        implementations: Sequence[str],
        rust_grid: str = "toml",
        rust_threads: int | None = None,
        instance_limit: int | None = None,
        rust_prefill: bool = False,
        match_reference_time: bool = False,
        instance_seed: int = SAMPLE_SEED,
    ) -> None:
        """Store study options without building or executing anything.

        Args:
            output_path (Path): Raw trial CSV destination.
            cache_dir (Path): External cache for KPLIB data and C++ source.
            budget_seconds (int): Positive search budget per trial.
            repetitions (int): Positive repetition count.
            solvers (Sequence[str]): Display names to run.
            implementations (Sequence[str]): Subset of python, cpp64 and rust.
            rust_grid (str): Native grid flavor, `toml` or `cpp`.
            rust_threads (int | None): Native worker cap for portfolios.
            instance_limit (int | None): Positive number of instances to use, or all.
            rust_prefill (bool): Fill the native pool with random vectors before searching.
            match_reference_time (bool): Run C++ first and give the native engine the wall
                time that trial actually used when it exceeded the budget.
            instance_seed (int): KPLIB selection seed; the default is the pinned sample and
                any other value draws a new sample, hashed on download but not pinned.

        Raises:
            ValueError: If a solver, implementation or grid name is unknown.
        """
        known = {name for name, _ in SOLVER_CONFIGS}
        if any(solver not in known for solver in solvers):
            raise ValueError(f"Unknown solver; choose from {sorted(known)}.")
        if any(name not in IMPLEMENTATIONS for name in implementations):
            raise ValueError(f"Unknown implementation; choose from {IMPLEMENTATIONS}.")
        if rust_grid not in ("toml", "cpp"):
            raise ValueError("rust_grid must be 'toml' or 'cpp'.")
        self.output_path = output_path
        self.budget_seconds = budget_seconds
        self.repetitions = repetitions
        self.solvers = tuple(solvers)
        self.implementations = tuple(implementations)
        self.rust_grid = rust_grid
        self.rust_threads = rust_threads
        self.instance_limit = instance_limit
        self.rust_prefill = rust_prefill
        self.match_reference_time = match_reference_time
        self.instance_seed = instance_seed
        self.benchmark = KnapsackBenchmark(
            output_path,
            cache_dir,
            budget_seconds=budget_seconds,
            repetitions=repetitions,
            resume=False,
        )
        self.repo_root = self.benchmark.repo_root

    def _cases(self) -> list[InstanceCase]:
        """Return the pinned sample, or a freshly drawn one for another seed.

        Returns:
            list[InstanceCase]: Parsed instances with exact optima.

        Raises:
            OSError: If an input cannot be downloaded.
            ValueError: If an instance is malformed or too large.
        """
        if self.instance_seed == SAMPLE_SEED:
            return self.benchmark._fetch_instances()
        cases: list[InstanceCase] = []
        selected = self.benchmark.select_instance_ids(self.instance_seed)
        for family, item_ids in selected.items():
            for item_id in item_ids:
                filename = f"s{item_id:03d}.kp"
                path = self.benchmark.cache_dir / "kplib" / family / filename
                if not path.exists():
                    url = f"{KPLIB_RAW}/{KPLIB_REVISION}/{family}/n00500/R01000/{filename}"
                    with urllib.request.urlopen(url, timeout=30) as response:
                        data = response.read(65537)
                    if len(data) > 65536:
                        raise ValueError(f"Unexpectedly large KPLIB input: {family}/{filename}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                instance = read_kp_instance(path)
                cases.append(
                    InstanceCase(
                        family=family,
                        relative_path=f"{family}/{filename}",
                        path=path,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        instance=instance,
                        optimum=exact_profit(instance),
                    )
                )
        return cases

    def _build_rust(self, cpp: bool = False) -> tuple[Path, str]:
        """Compile a native benchmark binary in release mode.

        Args:
            cpp (bool): Build `rko_bench_cpp` with the vendored C++ engine instead of the Rust one.

        Returns:
            tuple[Path, str]: Binary path and a SHA-256 digest of its bytes.

        Raises:
            RuntimeError: If compilation fails.
        """
        name = "rko_bench_cpp" if cpp else "rko_bench"
        command = ["cargo", "build", "--release", "--locked", "--bin", name]
        if cpp:
            command += ["-p", "rko-cpp"]
        build = subprocess.run(
            command,
            cwd=self.repo_root,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "RKO_BUILD_CPP": "1"} if cpp else None,
        )
        if build.returncode != 0:
            raise RuntimeError(f"Native build failed: {build.stderr[-3000:]}")
        binary = self.repo_root / "target" / "release" / name
        return binary, hashlib.sha256(binary.read_bytes()).hexdigest()

    def _run_rust_trial(
        self,
        binary: Path,
        case: InstanceCase,
        solver: str,
        seed: int,
        budget: float,
        cpp: bool = False,
    ) -> SolverOutput:
        """Run one native algorithm or portfolio on an instance.

        Args:
            binary (Path): Native benchmark executable.
            case (InstanceCase): Verified instance and exact optimum.
            solver (str): Display name of the algorithm or portfolio.
            seed (int): Native engine seed.
            budget (float): Native search seconds.
            cpp (bool): Whether `binary` is the C++ engine runner, which fixes grid and threads.

        Returns:
            SolverOutput: Parsed keys, objective, target events and status.
        """
        names = (
            "SA,ILS,VNS,BRKGA,BRKGA_CS,PSO,GA,LNS,GRASP,IPR"
            if solver == "RKO-default"
            else solver.replace("-", "_")
        )
        targets = ",".join(
            f"{gap}:{profit}" for gap, profit in self.benchmark._target_profits(case.optimum)
        )
        command = [
            str(binary),
            str(case.path),
            f"{budget:.3f}",
            "--algo",
            names,
            "--grid",
            "cpp" if cpp else self.rust_grid,
            "--seed",
            str(seed),
            "--targets",
            targets,
        ]
        if self.rust_prefill:
            command.append("--prefill")
        if solver == "RKO-default" and self.rust_threads is not None and not cpp:
            command += ["--threads", str(self.rust_threads)]
        started_ns = time.perf_counter_ns()
        return self.benchmark._execute_process(command, self.repo_root, {}, started_ns)

    def _run_rust_callback_trial(
        self,
        case: InstanceCase,
        solver: str,
        python_algorithm: str | None,
        seed: int,
    ) -> SolverOutput:
        """Run the native engine with the same Python callbacks as the Python engine.

        Args:
            case (InstanceCase): Verified instance and exact optimum.
            solver (str): Display name of the algorithm or portfolio.
            python_algorithm (str | None): Factory name, or None for the default portfolio.
            seed (int): Engine seed.

        Returns:
            SolverOutput: Parsed keys, objective, target events and status.
        """
        payload = {
            "instance_path": str(case.path),
            "algorithm": python_algorithm,
            "budget_seconds": self.budget_seconds,
            "seed": seed,
            "target_profits": self.benchmark._target_profits(case.optimum),
            "engine": "rust",
        }
        command = [
            sys.executable,
            "-m",
            "benchmarks.cpp_python_knapsack",
            "--worker",
            json.dumps(payload, separators=(",", ":")),
        ]
        started_ns = time.perf_counter_ns()
        return self.benchmark._execute_process(
            command, self.repo_root, os.environ.copy(), started_ns
        )

    def _run_python_style_trial(
        self, case: InstanceCase, solver: str, style: str, seed: int, budget: float
    ) -> SolverOutput:
        """Run a Python-defined objective (Numba or NumPy blocks) in a fresh interpreter.

        Args:
            case (InstanceCase): Verified instance and exact optimum.
            solver (str): Display name of the algorithm or portfolio.
            style (str): `numba` or `batch`.
            seed (int): Engine seed.
            budget (float): Search seconds, excluding compilation and start-up.

        Returns:
            SolverOutput: Parsed keys and objective; no target telemetry is produced.
        """
        names = PORTFOLIO_NAMES if solver == "RKO-default" else [solver.replace("-", "_")]
        payload = {
            "problem": "kp",
            "instance_path": str(case.path),
            "algorithms": names,
            "style": style,
            "budget_seconds": budget,
            "seed": seed,
        }
        command = [
            sys.executable,
            "-m",
            "benchmarks.python_problems",
            "--worker",
            json.dumps(payload, separators=(",", ":")),
        ]
        started_ns = time.perf_counter_ns()
        return self.benchmark._execute_process(
            command, self.repo_root, os.environ.copy(), started_ns
        )

    def run(self) -> list[dict[str, str]]:
        """Execute every trial, persist raw rows and return them.

        Returns:
            list[dict[str, str]]: Raw CSV rows as read back from disk.

        Raises:
            OSError: If inputs, outputs or tools are unavailable.
            RuntimeError: If a build or verification step fails.
        """
        self.benchmark.cache_dir.mkdir(parents=True, exist_ok=True)
        cases = self._cases()
        if self.instance_limit is not None:
            cases = cases[: self.instance_limit]
        binary, binary_hash = self._build_rust() if "rust" in self.implementations else (None, "")
        cpp_binary, cpp_hash = (
            self._build_rust(cpp=True) if "cpp-engine" in self.implementations else (None, "")
        )
        binary_hash += cpp_hash
        with tempfile.TemporaryDirectory(prefix="native-study-") as temporary:
            program_dir = None
            patch_hash, compiler = "", ""
            if "cpp64" in self.implementations:
                program_dir, patch_hash, compiler = self.benchmark._prepare_cpp(Path(temporary))
            metadata = self.benchmark._collect_metadata(patch_hash, compiler)
            base = self.benchmark._run_signature(
                cases, metadata, self.budget_seconds, self.repetitions
            )
            signature = hashlib.sha256(
                json.dumps(
                    [
                        base,
                        binary_hash,
                        self.rust_grid,
                        self.rust_threads,
                        self.rust_prefill,
                        self.implementations,
                    ]
                ).encode()
            ).hexdigest()
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=[item.name for item in dataclasses.fields(TrialRecord)]
                )
                writer.writeheader()
                solver_index = {name: index for index, (name, _) in enumerate(SOLVER_CONFIGS)}
                algorithm = dict(SOLVER_CONFIGS)
                for case_index, case in enumerate(cases):
                    for solver in self.solvers:
                        for replicate in range(self.repetitions):
                            seed = (
                                SAMPLE_SEED
                                + solver_index[solver] * 1_000_000
                                + case_index * 100
                                + replicate
                            )
                            shift = replicate % len(self.implementations)
                            order = self.implementations[shift:] + self.implementations[:shift]
                            if self.match_reference_time and "cpp64" in order:
                                order = ("cpp64", *(name for name in order if name != "cpp64"))
                            rust_budget = float(self.budget_seconds)
                            for implementation in order:
                                trial_id = self.benchmark._trial_id(
                                    implementation, solver, case, replicate
                                )
                                if implementation == "python":
                                    output = self.benchmark._run_python_trial(
                                        case,
                                        solver,
                                        algorithm[solver],
                                        self.budget_seconds,
                                        seed,
                                        trial_id,
                                    )
                                elif implementation == "rust-callback":
                                    output = self._run_rust_callback_trial(
                                        case, solver, algorithm[solver], seed
                                    )
                                elif implementation in ("rust-numba", "rust-batch", "cpp-numba"):
                                    output = self._run_python_style_trial(
                                        case,
                                        solver,
                                        implementation.removeprefix("rust-"),
                                        seed,
                                        rust_budget,
                                    )
                                elif implementation == "cpp-engine":
                                    output = self._run_rust_trial(
                                        cpp_binary, case, solver, seed, rust_budget, cpp=True
                                    )
                                elif implementation == "cpp64":
                                    output = self.benchmark._run_cpp_trial(
                                        case, solver, self.budget_seconds, trial_id, program_dir
                                    )
                                else:
                                    output = self._run_rust_trial(
                                        binary, case, solver, seed, rust_budget
                                    )
                                if implementation == "cpp64" and self.match_reference_time:
                                    rust_budget = max(rust_budget, round(output.elapsed_seconds, 3))
                                record = self.benchmark._record_trial(
                                    output,
                                    case,
                                    solver,
                                    implementation,
                                    replicate,
                                    seed if implementation != "cpp64" else None,
                                    rust_budget
                                    if implementation
                                    in (
                                        "rust",
                                        "rust-numba",
                                        "rust-batch",
                                        "cpp-numba",
                                        "cpp-engine",
                                    )
                                    else self.budget_seconds,
                                    trial_id,
                                    signature,
                                    metadata,
                                    require_telemetry=implementation
                                    not in ("rust-numba", "rust-batch", "cpp-numba"),
                                )
                                if implementation in (
                                    "rust",
                                    "rust-callback",
                                    "rust-numba",
                                    "rust-batch",
                                ):
                                    threads = (
                                        self.rust_threads or RUST_PORTFOLIO_THREADS
                                        if solver == "RKO-default"
                                        else 1
                                    )
                                    record = dataclasses.replace(
                                        record,
                                        execution_mode=f"{implementation}-threads-{threads}",
                                    )
                                writer.writerow(dataclasses.asdict(record))
                                stream.flush()
                                print(
                                    f"{record.implementation:6s} {record.solver:12s} "
                                    f"{record.instance:9s} r={record.replicate:02d} "
                                    f"status={record.status} profit={record.profit}",
                                    flush=True,
                                )
        with self.output_path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    @staticmethod
    def summarize(
        rows: Sequence[dict[str, str]], subject: str = "rust"
    ) -> list[dict[str, str | float | int]]:
        """Pair implementations per instance and test subject-versus-reference differences.

        Args:
            rows (Sequence[dict[str, str]]): Raw trial rows.
            subject (str): Implementation under evaluation.

        Returns:
            list[dict[str, str | float | int]]: One row per solver, reference and metric.

        Raises:
            ValueError: If a metric value is nonfinite.
        """
        results: list[dict[str, str | float | int]] = []
        solvers = sorted({row["solver"] for row in rows})
        references = sorted({row["implementation"] for row in rows} - {subject})
        metrics = (
            ("gap", "normalized_objective_difference"),
            ("elapsed_seconds", "elapsed_seconds"),
        )
        for metric, column in metrics:
            for reference in references:
                comparisons: dict[str, list[float]] = {}
                cells: dict[str, tuple[float, float, int, int, int]] = {}
                for solver in solvers:
                    per_instance: dict[str, dict[str, list[float]]] = {}
                    failures = {subject: 0, reference: 0}
                    for row in rows:
                        if row["solver"] != solver or row["implementation"] not in (
                            subject,
                            reference,
                        ):
                            continue
                        if row["status"] != "ok":
                            failures[row["implementation"]] += 1
                            if metric == "gap":
                                continue
                        if row[column] == "":
                            continue
                        per_instance.setdefault(row["instance_sha256"], {}).setdefault(
                            row["implementation"], []
                        ).append(float(row[column]))
                    differences: list[float] = []
                    subject_values: list[float] = []
                    reference_values: list[float] = []
                    for sides in per_instance.values():
                        if subject in sides and reference in sides:
                            native = statistics.median(sides[subject])
                            other = statistics.median(sides[reference])
                            differences.append(native - other)
                            subject_values.append(native)
                            reference_values.append(other)
                    if differences:
                        comparisons[solver] = differences
                        cells[solver] = (
                            statistics.median(subject_values),
                            statistics.median(reference_values),
                            failures[subject],
                            failures[reference],
                            len(differences),
                        )
                raw = {
                    solver: KnapsackBenchmark.exact_sign_test(values)[0]
                    for solver, values in comparisons.items()
                }
                adjusted = KnapsackBenchmark.holm_adjust(raw) if raw else {}
                for solver, values in comparisons.items():
                    p_value, non_tied, positive, negative = KnapsackBenchmark.exact_sign_test(
                        values
                    )
                    low, high = KnapsackBenchmark.bootstrap_median_interval(
                        values, seed=SAMPLE_SEED + sum(map(ord, solver + metric + reference))
                    )
                    native_median, reference_median, own_failed, ref_failed, paired = cells[solver]
                    results.append(
                        {
                            "metric": metric,
                            "subject": subject,
                            "reference": reference,
                            "solver": solver,
                            "subject_median": native_median,
                            "reference_median": reference_median,
                            "median_difference_subject_minus_reference": statistics.median(values),
                            "bootstrap_95_low": low,
                            "bootstrap_95_high": high,
                            "subject_better_instances": negative,
                            "subject_worse_instances": positive,
                            "tied_instances": paired - non_tied,
                            "p_exact_sign": p_value,
                            "p_holm": adjusted[solver],
                            "paired_instances": paired,
                            "subject_failed_trials": own_failed,
                            "reference_failed_trials": ref_failed,
                        }
                    )
        return results

    @staticmethod
    def render(summary: Sequence[dict[str, str | float | int]]) -> str:
        """Format the summary as a plain-text table.

        Args:
            summary (Sequence[dict[str, str | float | int]]): Rows from `summarize`.

        Returns:
            str: Human-readable fixed-width report.
        """
        lines = []
        pairs = sorted({(row["metric"], row["reference"]) for row in summary}, reverse=True)
        for metric, reference in pairs:
            subset = [
                row for row in summary if row["metric"] == metric and row["reference"] == reference
            ]
            subject = subset[0]["subject"]
            lines.append(
                f"\n[{metric}] {subject} vs {reference} (lower is better; "
                "B/W/T = subject better/worse/tied instances)"
            )
            lines.append(
                f"{'solver':12} {'subject':>9} {'ref':>9} {'diff':>9} "
                f"{'95% CI':>21} {'B/W/T':>9} {'p_holm':>8} {'fail s/ref':>10}"
            )
            for row in subset:
                interval = f"[{row['bootstrap_95_low']:.3f},{row['bootstrap_95_high']:.3f}]"
                record = (
                    f"{row['subject_better_instances']}/{row['subject_worse_instances']}/"
                    f"{row['tied_instances']}"
                )
                failed = f"{row['subject_failed_trials']}/{row['reference_failed_trials']}"
                lines.append(
                    f"{row['solver']:12} {row['subject_median']:9.3f} "
                    f"{row['reference_median']:9.3f} "
                    f"{row['median_difference_subject_minus_reference']:9.3f} "
                    f"{interval:>21} {record:>9} {row['p_holm']:8.4f} {failed:>10}"
                )
        return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the study from the command line and print the paired summary.

    Args:
        argv (Sequence[str] | None): Optional argument vector.

    Returns:
        int: Zero on completion.

    Raises:
        OSError: If files or tools are unavailable.
        RuntimeError: If a build fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Raw trial CSV path.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "rko-cpp-python-knapsack",
    )
    parser.add_argument("--budget-seconds", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--solvers", default="BRKGA,SA")
    parser.add_argument("--implementations", default="cpp64,rust")
    parser.add_argument("--rust-grid", choices=("toml", "cpp"), default="toml")
    parser.add_argument("--rust-threads", type=int, default=None)
    parser.add_argument("--rust-prefill", action="store_true")
    parser.add_argument(
        "--match-reference-time",
        action="store_true",
        help="Give the native engine the wall time each C++ trial actually used.",
    )
    parser.add_argument("--instances", type=int, default=None, help="Use the first N instances.")
    parser.add_argument(
        "--instance-seed",
        type=int,
        default=SAMPLE_SEED,
        help="KPLIB selection seed; keep calibration and held-out samples on different seeds.",
    )
    parser.add_argument("--summary-only", action="store_true", help="Summarize an existing CSV.")
    parser.add_argument("--subject", default="rust", help="Implementation under evaluation.")
    args = parser.parse_args(argv)
    study = NativeStudy(
        args.output,
        args.cache_dir,
        budget_seconds=args.budget_seconds,
        repetitions=args.repetitions,
        solvers=args.solvers.split(","),
        implementations=args.implementations.split(","),
        rust_grid=args.rust_grid,
        rust_threads=args.rust_threads,
        instance_limit=args.instances,
        rust_prefill=args.rust_prefill,
        match_reference_time=args.match_reference_time,
        instance_seed=args.instance_seed,
    )
    if args.summary_only:
        with args.output.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = study.run()
    summary = study.summarize(rows, args.subject)
    summary_path = args.output.with_name(f"{args.output.stem}-native-summary.csv")
    if summary:
        with summary_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    sys.stdout.write(study.render(summary) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
