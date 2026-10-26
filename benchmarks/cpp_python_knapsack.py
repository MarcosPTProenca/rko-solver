"""Compare Python RKO and an arithmetic-corrected C++ reference on Knapsack.

Objective:
- Measure runtime and solution quality for all individual metaheuristics and the default portfolio.

Technical approach:
- Download hash-pinned KPLIB inputs, compute exact optima, and build pinned C++ source.
- Change only the temporary C++ objective arithmetic and add benchmark telemetry/output precision.
- Run independent, budget-limited trials and analyze paired per-instance summaries.

Flow (pseudocode):
1. Select and verify 15 500-item instances from five fixed KPLIB families.
2. Build isolated Python/C++ trial commands and record every result or failure.
3. Analyze quality, feasibility, and time-to-target without a composite score.

Important criteria:
- The historical C++ checkout and KPLIB data are never modified or committed.
- C++ uses 64-bit objective arithmetic in the benchmark copy to avoid signed overflow.
- Search-time targets are censored at the fixed budget; invocation latency includes startup.

Inputs/Outputs:
- Input: pinned source/data revisions, run budget, repeat count, and output path.
- Output: raw trial CSV and per-instance/significance summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import numpy as np
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from benchmarks.knapsack_utils import exact_profit
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from io import BytesIO
from pathlib import Path
from rko_solver import RKO
from rko_solver import Callbacks
from rko_solver.examples.run_kp_cpp_instances import KpInstance
from rko_solver.examples.run_kp_cpp_instances import decode_knapsack
from rko_solver.examples.run_kp_cpp_instances import evaluate_solution
from rko_solver.examples.run_kp_cpp_instances import read_kp_instance
from typing import Any

KPLIB_REVISION = "c5bea0df8169749caaba5ce7dfcf21437aa1ca5c"
CPP_REVISION = "126e35873e50dc308795b58b8c75d37f8de4c857"
CPP_REPOSITORY = "https://github.com/RKO-solver/RKO_Cpp_v2.0.git"
KPLIB_RAW = "https://raw.githubusercontent.com/likr/kplib"
SAMPLE_SEED = 20261019
DEFAULT_BUDGET_SECONDS = 120
DEFAULT_REPETITIONS = 5
TARGET_GAPS = (10, 1, 0)
CPP_FLAGS = "-fopenmp -O3 -Wall -std=c++2a -Wno-unused-but-set-variable"
FAMILIES = (
    "00Uncorrelated",
    "01WeaklyCorrelated",
    "02StronglyCorrelated",
    "05SubsetSum",
    "07SpannerUncorrelated",
)
PYTHON_ALGORITHMS = (
    ("BRKGA", "BRKGA"),
    ("BRKGA-CS", "BRKGA_CS"),
    ("GA", "GA"),
    ("GRASP", "GRASP"),
    ("ILS", "ILS"),
    ("IPR", "IPR"),
    ("LNS", "LNS"),
    ("MultiStart", "MultiStart"),
    ("PSO", "PSO"),
    ("SA", "SA"),
    ("VNS", "VNS"),
)
CPP_PORTFOLIO = (
    "SA",
    "ILS",
    "VNS",
    "BRKGA",
    "BRKGA-CS",
    "PSO",
    "GA",
    "LNS",
    "GRASP",
    "IPR",
)
SOLVER_CONFIGS = (*PYTHON_ALGORITHMS, ("RKO-default", None))
INSTANCE_SHA256 = {
    path: digest
    for path, digest in (
        line.split()
        for line in """
00Uncorrelated/s041.kp b764f09ca075cb3b05c9704a7ff57a23a33d73816d0fe7758d3ea34f91a20655
00Uncorrelated/s075.kp 6512ca8db5837337ae58a867e88d41be74d61689bdded3b9db54dc7829ea11fc
00Uncorrelated/s092.kp 28c37692613a1e3af6b0979f2e27d19b3c0e5d32cec4eb7866c0b87f8cc59f45
01WeaklyCorrelated/s015.kp a5b3f5c82db8f0f6ae3d5790a2ec1aa39573be6e435b44762a84961c45a727c2
01WeaklyCorrelated/s027.kp 5009309ca808e864b6c9cc604281e06934cd55f79cf681e41c57a5cbc0eb480f
01WeaklyCorrelated/s060.kp bd44328fd09454961870c69a3b9e7b8d975491584f20245d0287bb3767e4fd65
02StronglyCorrelated/s034.kp d840a1121e0d29e99554ed95267558247180397877c1ea73e4e02cb9da01c6f9
02StronglyCorrelated/s037.kp 26e58386ea06bb4eeb0a70dd189dc51c04261c652fe4ae8de49b3e3181519d20
02StronglyCorrelated/s069.kp bf043da2b0ef25e44ade43c55cfcb00365dbe9dfe5db73d59f706a110d0deb96
05SubsetSum/s003.kp 1b0783a6ffa08b616b2ff6cf8ebbecce4e10905060abd1b4702f449663aaec97
05SubsetSum/s044.kp 951803299442fab73f11adf98f6a06f25599433ea3f0f06baba6ecc104ec8ba4
05SubsetSum/s091.kp 04a7bdc7d87ad555d8eb7d776fd521eca6f02c247f7397e485045deb8f635495
07SpannerUncorrelated/s000.kp 3dd36adec09e23cbf24b71d1663c315fb5cd7c0456f846b23cd95abd1a3b85dd
07SpannerUncorrelated/s017.kp ee673cd708aba7046f9cbacbf60aaa1ba70d20e0811746d8c0999d9a5aee2ab4
07SpannerUncorrelated/s092.kp f574a32e6b87a832c9c70bc70bb7e2b73567caa3615c7ebb173fb7a075d45652
""".splitlines()
        if line.strip()
    )
}


@dataclass(frozen=True, slots=True)
class InstanceCase:
    """Store one verified benchmark instance and its exact optimum.

    Responsibilities:
    - Bind input provenance to parsed problem data and an exact reference value.
    - Provide immutable metadata for repeated solver trials.

    Attributes:
        family (str): Pinned KPLIB generator family.
        relative_path (str): Repository-relative family and filename.
        path (Path): Cached local input path.
        sha256 (str): Expected content digest.
        instance (KpInstance): Parsed Knapsack data.
        optimum (int): Exact feasible maximum profit.
    """

    family: str
    relative_path: str
    path: Path
    sha256: str
    instance: KpInstance
    optimum: int


@dataclass(slots=True)
class TargetTrackingCost:
    """Evaluate Knapsack candidates and record first feasible target hits.

    Responsibilities:
    - Apply the public Python Knapsack objective.
    - Emit one low-volume telemetry record per target reached in each process.

    Attributes:
        instance (KpInstance): Parsed Knapsack data.
        target_profits (tuple[tuple[int, int], ...]): Gap percentage and required profit.
        started_ns (int): Monotonic origin captured immediately before search starts.
        reported (set[int]): Targets already reported by this process-local callback.
    """

    instance: KpInstance
    target_profits: tuple[tuple[int, int], ...]
    started_ns: int
    reported: set[int] = field(default_factory=set)

    def __call__(self, solution: list[int]) -> float:
        """Score a decoded candidate and report newly reached feasible targets.

        Args:
            solution (list[int]): One binary decision per Knapsack item.

        Returns:
            float: Penalized negative profit to minimize.

        Raises:
            ValueError: If the candidate dimension or binary values are invalid.
        """
        profit, weight, feasible = evaluate_solution(solution, self.instance)
        reached = (
            [
                gap
                for gap, target_profit in self.target_profits
                if profit >= target_profit and gap not in self.reported
            ]
            if feasible
            else []
        )
        if reached:
            elapsed = (time.perf_counter_ns() - self.started_ns) / 1_000_000_000
            for gap in reached:
                self.reported.add(gap)
                message = f"RKO_BENCH_TARGET gap={gap} elapsed_seconds={elapsed:.9f}\n"
                try:
                    os.write(2, message.encode("ascii"))
                except OSError:
                    pass
        return float(100000 * max(0, weight - self.instance.capacity) - profit)


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """Store one independently checked run and its provenance.

    Responsibilities:
    - Preserve raw solution metrics, target latency, and implementation metadata.
    - Represent failed executions explicitly without inventing solution values.

    Attributes:
        trial_id (str): Stable implementation/configuration/input/repetition key.
        implementation (str): Python or corrected C++ reference label.
        solver (str): Single algorithm name or the default portfolio label.
        family (str): KPLIB instance family.
        instance (str): KPLIB filename.
        instance_sha256 (str): Verified input digest.
        capacity (int): Knapsack capacity.
        exact_optimum (int): Independently computed maximum feasible profit.
        replicate (int): Independent repetition number.
        python_seed (int | None): Python random-state input when applicable.
        cpp_seed (int | None): C++ clock-derived engine seed when emitted.
        budget_seconds (int): Configured search budget.
        status (str): `ok` or `error`.
        profit (int | None): Independently decoded profit.
        weight (int | None): Independently decoded weight.
        feasible (bool | None): Independent capacity check.
        objective (float | None): Independently recomputed reference objective.
        normalized_objective_difference (float | None): Scale-normalized cost difference.
        feasible_gap_percent (float | None): Exact-optimum gap for feasible solutions.
        success_10_percent (bool): Whether a feasible 10% target was reached.
        time_10_percent_seconds (float | None): Search-time latency capped at the budget.
        success_1_percent (bool): Whether a feasible 1% target was reached.
        time_1_percent_seconds (float | None): Search-time latency capped at the budget.
        success_optimum (bool): Whether the exact optimum was reached.
        time_optimum_seconds (float | None): Search-time latency capped at the budget.
        elapsed_seconds (float): End-to-end invocation time.
        execution_mode (str): Serial, process-portfolio, or OpenMP execution mode.
        solution_sha256 (str | None): Digest of the returned random-key vector.
        return_code (int | None): Child process exit code.
        error (str | None): Bounded failure detail when the run did not complete.
        run_signature (str): Hash of workload, source, and runtime configuration.
        python_revision (str): Python repository base revision.
        python_source_sha256 (str): Hash of relevant Python sources.
        python_version (str): Interpreter version.
        numpy_version (str): NumPy version.
        platform (str): Operating-system and architecture description.
        cpu_model (str): Host processor model when available.
        cpu_count (int): Available logical CPU count.
        memory_total_gib (str): Host physical memory when available.
        memory_available_gib (str): Host available memory at benchmark startup.
        compiler (str): C++ compiler version.
        cpp_flags (str): Exact C++ compiler flags.
        cpp_revision (str): Pinned upstream C++ revision.
        cpp_patch_sha256 (str): Hash of temporary benchmark-only C++ source changes.
    """

    trial_id: str
    implementation: str
    solver: str
    family: str
    instance: str
    instance_sha256: str
    capacity: int
    exact_optimum: int
    replicate: int
    python_seed: int | None
    cpp_seed: int | None
    budget_seconds: int
    status: str
    profit: int | None
    weight: int | None
    feasible: bool | None
    objective: float | None
    normalized_objective_difference: float | None
    feasible_gap_percent: float | None
    success_10_percent: bool
    time_10_percent_seconds: float | None
    success_1_percent: bool
    time_1_percent_seconds: float | None
    success_optimum: bool
    time_optimum_seconds: float | None
    elapsed_seconds: float
    execution_mode: str
    solution_sha256: str | None
    return_code: int | None
    error: str | None
    run_signature: str
    python_revision: str
    python_source_sha256: str
    python_version: str
    numpy_version: str
    platform: str
    cpu_model: str
    cpu_count: int
    memory_total_gib: str
    memory_available_gib: str
    compiler: str
    cpp_flags: str
    cpp_revision: str
    cpp_patch_sha256: str


@dataclass(frozen=True, slots=True)
class SolverOutput:
    """Carry parsed child output before independent solution verification.

    Responsibilities:
    - Preserve returned keys, reported objective, target events and process status.

    Attributes:
        keys (tuple[float, ...] | None): Full-precision returned random keys.
        objective (float | None): Objective reported by the solver.
        events (dict[int, float]): Earliest observed time for each target gap.
        cpp_seed (int | None): C++ engine seed emitted by the temporary runner.
        elapsed_seconds (float): End-to-end process duration.
        return_code (int | None): Child exit status, or None on timeout.
        error (str | None): Process or output parsing failure.
    """

    keys: tuple[float, ...] | None
    objective: float | None
    events: dict[int, float]
    cpp_seed: int | None
    elapsed_seconds: float
    return_code: int | None
    error: str | None


class KnapsackBenchmark:
    """Prepare, execute, and analyze the approved Python/C++ comparison.

    Responsibilities:
    - Verify pinned KPLIB data and isolated C++ source before timing trials.
    - Record resumable independent runs and paired per-instance statistics.

    Attributes:
        output_path (Path): Raw trial CSV destination.
        cache_dir (Path): External cache for source and problem inputs.
        budget_seconds (int): Configured solver time limit per trial.
        repetitions (int): Number of trials per implementation/configuration/instance.
        resume (bool): Whether a matching existing raw CSV may be continued.
        repo_root (Path): Python repository root containing this module.
    """

    def __init__(
        self,
        output_path: Path,
        cache_dir: Path,
        *,
        budget_seconds: int = DEFAULT_BUDGET_SECONDS,
        repetitions: int = DEFAULT_REPETITIONS,
        resume: bool = False,
    ) -> None:
        """Store benchmark options without network access or compilation.

        Args:
            output_path (Path): Raw per-trial CSV destination.
            cache_dir (Path): External cache root for inputs and source.
            budget_seconds (int): Positive integer search budget.
            repetitions (int): Positive independent repetition count.
            resume (bool): Continue only a matching existing output file.

        Raises:
            ValueError: If the budget or repetition count is not positive.
        """
        if type(budget_seconds) is not int or budget_seconds < 1:
            raise ValueError("budget_seconds must be a positive integer.")
        if type(repetitions) is not int or repetitions < 1:
            raise ValueError("repetitions must be a positive integer.")
        self.output_path = output_path
        self.cache_dir = cache_dir
        self.budget_seconds = budget_seconds
        self.repetitions = repetitions
        self.resume = resume
        self.repo_root = Path(__file__).resolve().parents[1]

    @staticmethod
    def select_instance_ids(seed: int) -> dict[str, tuple[int, int, int]]:
        """Select three stable KPLIB file IDs from each 100-instance family.

        Args:
            seed (int): Seed for deterministic family-stratified selection.

        Returns:
            dict[str, tuple[int, int, int]]: Sorted selected IDs by family.

        Raises:
            ValueError: If the seed is not a nonnegative integer.
        """
        if type(seed) is not int or seed < 0:
            raise ValueError("The sample seed must be a nonnegative integer.")
        generator = random.Random(seed)
        return {family: tuple(sorted(generator.sample(range(100), 3))) for family in FAMILIES}

    @staticmethod
    def normalized_objective_difference(objective: float, optimum: int) -> float:
        """Normalize the reference penalized objective relative to exact profit.

        Args:
            objective (float): Penalized minimization objective.
            optimum (int): Positive exact feasible profit.

        Returns:
            float: `100 * (objective + optimum) / optimum`.

        Raises:
            ValueError: If the optimum is not positive or an input is nonfinite.
        """
        if optimum <= 0 or not math.isfinite(objective):
            raise ValueError("The optimum must be positive and the objective finite.")
        return 100.0 * (objective + optimum) / optimum

    @staticmethod
    def exact_sign_test(values: Sequence[float]) -> tuple[float, int, int, int]:
        """Compute an exact two-sided sign-test p-value, excluding ties.

        Args:
            values (Sequence[float]): Paired per-instance differences.

        Returns:
            tuple[float, int, int, int]: P-value, non-ties, positive, negative.

        Raises:
            ValueError: If any difference is nonfinite.
        """
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Sign-test differences must be finite.")
        positive = sum(value > 0 for value in values)
        negative = sum(value < 0 for value in values)
        count = positive + negative
        if count == 0:
            return 1.0, 0, 0, 0
        lower_tail = sum(math.comb(count, index) for index in range(min(positive, negative) + 1))
        p_value = min(1.0, 2.0 * lower_tail / 2**count)
        return p_value, count, positive, negative

    @staticmethod
    def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
        """Apply Holm step-down family-wise correction to named p-values.

        Args:
            p_values (Mapping[str, float]): Raw test probabilities by solver.

        Returns:
            dict[str, float]: Adjusted probabilities in input-name order.

        Raises:
            ValueError: If any p-value is outside the finite interval [0, 1].
        """
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in p_values.values()):
            raise ValueError("P-values must be finite values in [0, 1].")
        ordered = sorted(p_values.items(), key=lambda item: item[1])
        adjusted: dict[str, float] = {}
        running = 0.0
        count = len(ordered)
        for index, (name, value) in enumerate(ordered):
            running = max(running, min(1.0, (count - index) * value))
            adjusted[name] = running
        return {name: adjusted[name] for name in p_values}

    @staticmethod
    def bootstrap_median_interval(
        values: Sequence[float], *, seed: int, samples: int = 10000
    ) -> tuple[float, float]:
        """Estimate a deterministic percentile interval for a sample median.

        Args:
            values (Sequence[float]): Paired instance-level differences.
            seed (int): Nonnegative bootstrap random seed.
            samples (int): Positive number of bootstrap resamples.

        Returns:
            tuple[float, float]: 95% percentile interval for the median.

        Raises:
            ValueError: If the sample, seed, or resample count is invalid.
        """
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("Bootstrap input must contain finite values.")
        if type(seed) is not int or seed < 0 or type(samples) is not int or samples < 1:
            raise ValueError("The bootstrap seed and sample count must be valid integers.")
        generator = random.Random(seed)
        count = len(values)
        medians = sorted(
            statistics.median(generator.choices(values, k=count)) for _ in range(samples)
        )
        lower = medians[int(0.025 * (samples - 1))]
        upper = medians[int(0.975 * (samples - 1))]
        return float(lower), float(upper)

    @staticmethod
    def parse_target_events(output: str) -> dict[int, float]:
        """Parse target events and retain each target's earliest observed time.

        Args:
            output (str): Combined standard-error text from one solver process tree.

        Returns:
            dict[int, float]: Earliest elapsed time indexed by gap percentage.

        Raises:
            ValueError: If a target record is malformed, unknown, or nonfinite.
        """
        events: dict[int, float] = {}
        prefix = "RKO_BENCH_TARGET "
        for line in output.splitlines():
            if not line.startswith(prefix):
                continue
            match = re.fullmatch(r"RKO_BENCH_TARGET gap=(\d+) elapsed_seconds=([0-9.eE+-]+)", line)
            if match is None:
                raise ValueError(f"Malformed target event: {line!r}")
            gap = int(match.group(1))
            elapsed = float(match.group(2))
            if gap not in TARGET_GAPS or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError(f"Invalid target event: {line!r}")
            events[gap] = min(events.get(gap, math.inf), elapsed)
        return events

    def run(self, *, pilot: bool = False) -> None:
        """Fetch inputs, build C++, execute trials, and write raw/summary CSVs.

        Args:
            pilot (bool): Run one solo algorithm and the full portfolio on one input.

        Returns:
            None: Persists trial and analysis rows to the selected output paths.

        Raises:
            OSError: If source, data, cache, or output files cannot be accessed.
            RuntimeError: If pinned source preparation or result validation fails.
            ValueError: If existing results do not match the requested plan.
        """
        if not sys.platform.startswith("linux"):
            raise RuntimeError("The C++ reference benchmark requires Linux or WSL.")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cases = self._fetch_instances()
        if pilot:
            cases = cases[:1]
            solver_configs = [*PYTHON_ALGORITHMS[:2], ("RKO-default", None)]
            repetitions = 1
            budget_seconds = 1
        else:
            solver_configs = list(SOLVER_CONFIGS)
            repetitions = self.repetitions
            budget_seconds = self.budget_seconds
        with tempfile.TemporaryDirectory(prefix="cpp-python-knapsack-") as temporary:
            program_dir, patch_hash, compiler = self._prepare_cpp(Path(temporary))
            metadata = self._collect_metadata(patch_hash, compiler)
            signature = self._run_signature(cases, metadata, budget_seconds, repetitions)
            completed = self._load_existing(signature)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self.output_path.exists() or self.output_path.stat().st_size == 0
            with self.output_path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[record_field.name for record_field in fields(TrialRecord)],
                )
                if write_header:
                    writer.writeheader()
                    stream.flush()
                for case_index, case in enumerate(cases):
                    for solver_index, (solver, python_algorithm) in enumerate(solver_configs):
                        for replicate in range(repetitions):
                            python_seed = (
                                SAMPLE_SEED
                                + solver_index * 1_000_000
                                + case_index * 100
                                + replicate
                            )
                            trial_ids = {
                                implementation: self._trial_id(
                                    implementation, solver, case, replicate
                                )
                                for implementation in ("python", "cpp64")
                            }
                            order = (
                                ("python", "cpp64") if replicate % 2 == 0 else ("cpp64", "python")
                            )
                            for implementation in order:
                                trial_id = trial_ids[implementation]
                                if trial_id in completed:
                                    continue
                                if implementation == "python":
                                    output = self._run_python_trial(
                                        case,
                                        solver,
                                        python_algorithm,
                                        budget_seconds,
                                        python_seed,
                                        trial_id,
                                    )
                                else:
                                    output = self._run_cpp_trial(
                                        case,
                                        solver,
                                        budget_seconds,
                                        trial_id,
                                        program_dir,
                                    )
                                record = self._record_trial(
                                    output,
                                    case,
                                    solver,
                                    implementation,
                                    replicate,
                                    python_seed if implementation == "python" else None,
                                    budget_seconds,
                                    trial_id,
                                    signature,
                                    metadata,
                                )
                                writer.writerow(asdict(record))
                                stream.flush()
                                completed.add(trial_id)
                                print(
                                    f"{record.implementation:5s} {record.solver:12s} "
                                    f"{record.instance:9s} r={record.replicate:02d} "
                                    f"status={record.status} profit={record.profit}",
                                    flush=True,
                                )
        self._write_summary(signature)

    def _fetch_instances(self) -> list[InstanceCase]:
        """Download, hash-check, parse, and exactly solve the selected instances.

        Returns:
            list[InstanceCase]: Fifteen verified 500-item cases in family order.

        Raises:
            OSError: If a pinned KPLIB file cannot be downloaded or cached.
            ValueError: If an input hash, dimensions, or optimum is invalid.
        """
        selected = self.select_instance_ids(SAMPLE_SEED)
        cases: list[InstanceCase] = []
        for family, item_ids in selected.items():
            for item_id in item_ids:
                filename = f"s{item_id:03d}.kp"
                relative_path = f"{family}/{filename}"
                expected_hash = INSTANCE_SHA256[relative_path]
                path = self.cache_dir / "kplib" / family / filename
                if not path.exists():
                    url = f"{KPLIB_RAW}/{KPLIB_REVISION}/{family}/n00500/R01000/{filename}"
                    with urllib.request.urlopen(url, timeout=30) as response:
                        data = response.read(65537)
                    if len(data) > 65536:
                        raise ValueError(f"Unexpectedly large KPLIB input: {relative_path}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(path.suffix + ".tmp")
                    temporary.write_bytes(data)
                    temporary.replace(path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != expected_hash:
                    raise ValueError(f"KPLIB SHA-256 mismatch for {relative_path}.")
                instance = read_kp_instance(path)
                if instance.n_items != 500:
                    raise ValueError(f"KPLIB item-count mismatch for {relative_path}.")
                optimum = exact_profit(instance)
                if optimum <= 0:
                    raise ValueError(f"Invalid exact optimum for {relative_path}.")
                cases.append(
                    InstanceCase(
                        family=family,
                        relative_path=relative_path,
                        path=path,
                        sha256=digest,
                        instance=instance,
                        optimum=optimum,
                    )
                )
        return cases

    def _prepare_cpp(self, temporary: Path) -> tuple[Path, str, str]:
        """Extract pinned C++ sources, apply the approved isolated patch, and build.

        Args:
            temporary (Path): Temporary build root removed after all trials.

        Returns:
            tuple[Path, str, str]: Program directory, patch hash, compiler version.

        Raises:
            OSError: If source files or the compiler are unavailable.
            RuntimeError: If source patching, pinned checkout, or compilation fails.
        """
        repository = self._clone_cpp_source()
        archive = subprocess.run(
            ["git", "-C", str(repository), "archive", "--format=tar", CPP_REVISION, "Program"],
            check=True,
            capture_output=True,
        ).stdout
        with tarfile.open(fileobj=BytesIO(archive), mode="r:") as source_archive:
            source_archive.extractall(temporary, filter="data")
        program_dir = temporary / "Program"
        patch_hash = self._patch_cpp_source(program_dir)
        compiler = subprocess.run(
            ["g++", "--version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0]
        build = subprocess.run(
            ["make", "runRKO", f"CCFLAGS={CPP_FLAGS}"],
            cwd=program_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        if build.returncode != 0:
            raise RuntimeError(f"C++ build failed: {build.stderr[-3000:]}")
        (temporary / "Results").mkdir(parents=True, exist_ok=True)
        return program_dir, patch_hash, compiler

    def _clone_cpp_source(self) -> Path:
        """Ensure the exact upstream C++ revision exists in the external cache.

        Returns:
            Path: Git checkout pinned to the configured C++ commit.

        Raises:
            OSError: If Git cannot create or inspect the cache checkout.
            RuntimeError: If the checkout resolves to a different revision.
        """
        repository = self.cache_dir / "cpp-source"
        if not repository.exists():
            temporary = self.cache_dir / "cpp-source.tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "--filter=blob:none",
                    "--no-checkout",
                    CPP_REPOSITORY,
                    str(temporary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "-C", str(temporary), "checkout", "--quiet", "--detach", CPP_REVISION],
                check=True,
                capture_output=True,
                text=True,
            )
            temporary.replace(repository)
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != CPP_REVISION:
            raise RuntimeError(f"C++ checkout is {revision}, expected {CPP_REVISION}.")
        return repository

    def _patch_cpp_source(self, program_dir: Path) -> str:
        """Patch only evaluator-width and trial-observation code in the temporary copy.

        Args:
            program_dir (Path): Extracted C++ Program source directory.

        Returns:
            str: SHA-256 digest over all modified source files.

        Raises:
            OSError: If source files cannot be read or written.
            RuntimeError: If an expected unique source fragment is missing or repeated.
        """
        problem_path = program_dir / "src" / "Problem" / "Problem.h"
        problem = problem_path.read_text(encoding="utf-8")
        replacements = (
            ("#define _PROBLEM_H\n", "#define _PROBLEM_H\n\n#include <cstdint>\n"),
            ("    int cost = 0;", "    std::int64_t cost = 0;"),
            ("cost = cost - (100000 * infeasible);", "cost = cost - (100000LL * infeasible);"),
        )
        for old, new in replacements:
            if problem.count(old) != 1:
                raise RuntimeError(f"Expected one C++ objective fragment: {old!r}")
            problem = problem.replace(old, new)
        telemetry = (
            "    if (rko_benchmark_enabled && totalW <= data.cap) {\n"
            "        for (int target = 0; target < 3; ++target) {\n"
            "            if (cost >= rko_benchmark_targets[target] &&\n"
            "                !rko_benchmark_target_reported[target].exchange(true)) {\n"
            "                const auto now_ns = "
            "std::chrono::duration_cast<std::chrono::nanoseconds>(\n"
            "                    std::chrono::steady_clock::now().time_since_epoch()).count();\n"
            "                const double elapsed = "
            "(now_ns - rko_benchmark_start_ns) / 1e9;\n"
            "                const int gap = target == 0 ? 10 : (target == 1 ? 1 : 0);\n"
            '                std::fprintf(stderr, "RKO_BENCH_TARGET '
            'gap=%d elapsed_seconds=%.9f\\n", gap, elapsed);\n'
            "                std::fflush(stderr);\n"
            "            }\n"
            "        }\n"
            "    }\n"
        )
        marker = "    #define MAX(x,y) ((x)<(y) ? (y) : (x))\n"
        if problem.count(marker) != 1:
            raise RuntimeError("Could not locate the pinned C++ decoder telemetry point.")
        problem = problem.replace(marker, telemetry + marker)
        problem_path.write_text(problem, encoding="utf-8")

        main_path = program_dir / "src" / "Main" / "main.cpp"
        main = main_path.read_text(encoding="utf-8")
        include = "#include <atomic>\n"
        if main.count(include) != 1:
            raise RuntimeError("Could not locate the pinned C++ atomic include.")
        main = main.replace(
            include, include + "#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n"
        )
        global_marker = "std::atomic<bool> stop_execution(false);"
        global_telemetry = (
            global_marker
            + "\nbool rko_benchmark_enabled = false;"
            + "\nlong long rko_benchmark_targets[3] = {};"
            + "\nstd::atomic<bool> rko_benchmark_target_reported[3] = {false, false, false};"
            + "\nunsigned long long rko_benchmark_start_ns = 0;"
        )
        if main.count(global_marker) != 1:
            raise RuntimeError("Could not locate the pinned C++ global insertion point.")
        main = main.replace(global_marker, global_telemetry)
        function_pattern = r"(int main\(int argc, char \*argv\[ \]\)\n\{[ \t]*\n)"
        function_match = re.search(function_pattern, main)
        if function_match is None:
            raise RuntimeError("Could not locate the pinned C++ main telemetry point.")
        function_telemetry = (
            '    const char *benchmark_targets = std::getenv("RKO_BENCH_TARGETS");\n'
            "    if (benchmark_targets &&\n"
            '        std::sscanf(benchmark_targets, "%lld,%lld,%lld",\n'
            "                    &rko_benchmark_targets[0], &rko_benchmark_targets[1],\n"
            "                    &rko_benchmark_targets[2]) == 3) {\n"
            "        rko_benchmark_enabled = true;\n"
            "    }\n"
        )
        main = main[: function_match.end()] + function_telemetry + main[function_match.end() :]
        start_marker = "        start_time = get_time_in_seconds();\n"
        if main.count(start_marker) != 1:
            raise RuntimeError("Could not locate the pinned C++ search start marker.")
        start_telemetry = (
            start_marker
            + "        if (rko_benchmark_enabled) {\n"
            + "            rko_benchmark_start_ns = "
            + "std::chrono::duration_cast<std::chrono::nanoseconds>(\n"
            + "                std::chrono::steady_clock::now().time_since_epoch()).count();\n"
            + "        }\n"
        )
        main = main.replace(start_marker, start_telemetry)
        seed_marker = "        if (runData.debug) rng.seed (1234);\n"
        seed_log = (
            seed_marker
            + "        if (rko_benchmark_enabled) {\n"
            + '            std::fprintf(stderr, "RKO_BENCH_SEED=%d\\n", RSEED);\n'
            + "            std::fflush(stderr);\n"
            + "        }\n"
        )
        if main.count(seed_marker) != 1:
            raise RuntimeError("Could not locate the pinned C++ seed reporting point.")
        main = main.replace(seed_marker, seed_log)
        main_path.write_text(main, encoding="utf-8")

        output_path = program_dir / "src" / "Output.h"
        output = output_path.read_text(encoding="utf-8")
        key_format = 'fprintf(solFile,"%.3lf ", s.rk[i]);'
        if output.count(key_format) != 1:
            raise RuntimeError("Could not locate the pinned C++ key serialization format.")
        output_path.write_text(
            output.replace(key_format, 'fprintf(solFile,"%.17g ", s.rk[i]);'),
            encoding="utf-8",
        )
        source_files = (problem_path, main_path, output_path)
        digest = hashlib.sha256()
        for path in source_files:
            digest.update(path.relative_to(program_dir).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _collect_metadata(self, patch_hash: str, compiler: str) -> dict[str, str]:
        """Capture source and host details needed to identify a reference run.

        Args:
            patch_hash (str): Digest of the temporary C++ benchmark source changes.
            compiler (str): First line returned by `g++ --version`.

        Returns:
            dict[str, str]: Stable source/runtime values repeated in raw rows.

        Raises:
            OSError: If Git metadata or source files cannot be read.
            RuntimeError: If the Python checkout revision cannot be determined.
        """
        revision = subprocess.run(
            ["git", "-C", str(self.repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        source_hash = self._python_source_hash()
        cpu_model = "unknown"
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
        memory = {"MemTotal": "unknown", "MemAvailable": "unknown"}
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition(":")
                if separator and key in memory:
                    memory[key] = f"{int(value.split()[0]) / 1_048_576:.2f}"
        cpu_count = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 0
        )
        return {
            "python_revision": revision,
            "python_source_sha256": source_hash,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "cpu_model": cpu_model,
            "cpu_count": str(cpu_count),
            "memory_total_gib": memory["MemTotal"],
            "memory_available_gib": memory["MemAvailable"],
            "compiler": compiler,
            "cpp_flags": CPP_FLAGS,
            "cpp_revision": CPP_REVISION,
            "cpp_patch_sha256": patch_hash,
        }

    def _python_source_hash(self) -> str:
        """Hash package code, packaged TOMLs and benchmark dependencies.

        Returns:
            str: SHA-256 digest over relevant source paths and contents.

        Raises:
            OSError: If a source file cannot be read.
        """
        digest = hashlib.sha256()
        source_paths = sorted((self.repo_root / "src" / "rko_solver").rglob("*.py"))
        source_paths.extend(sorted((self.repo_root / "src" / "rko_solver").rglob("*.toml")))
        source_paths.extend(
            (
                self.repo_root / "benchmarks" / "__init__.py",
                self.repo_root / "benchmarks" / "knapsack.py",
                self.repo_root / "benchmarks" / "knapsack_utils.py",
                self.repo_root / "pyproject.toml",
                self.repo_root / "uv.lock",
                Path(__file__).resolve(),
            )
        )
        for path in source_paths:
            digest.update(path.relative_to(self.repo_root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _run_signature(
        self,
        cases: Sequence[InstanceCase],
        metadata: Mapping[str, str],
        budget_seconds: int,
        repetitions: int,
    ) -> str:
        """Hash all settings that must match before an existing CSV can resume.

        Args:
            cases (Sequence[InstanceCase]): Verified ordered workload.
            metadata (Mapping[str, str]): Source and host identity.
            budget_seconds (int): Configured solver time limit.
            repetitions (int): Planned trial count per cell.

        Returns:
            str: Stable SHA-256 plan signature.
        """
        stable_metadata = {
            key: value
            for key, value in metadata.items()
            if key not in {"python_revision", "memory_available_gib"}
        }
        plan = {
            "sample_seed": SAMPLE_SEED,
            "families": FAMILIES,
            "solvers": SOLVER_CONFIGS,
            "budget_seconds": budget_seconds,
            "repetitions": repetitions,
            "kplib_revision": KPLIB_REVISION,
            "inputs": [(case.relative_path, case.sha256, case.optimum) for case in cases],
            "metadata": stable_metadata,
        }
        serialized = json.dumps(plan, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _load_existing(self, signature: str) -> set[str]:
        """Validate a prior raw CSV and return completed trial identifiers.

        Args:
            signature (str): Expected run-plan fingerprint.

        Returns:
            set[str]: Trial IDs already present in the matching output.

        Raises:
            FileExistsError: If output exists without explicit resume permission.
            ValueError: If an existing CSV belongs to another plan or schema.
            OSError: If the output file cannot be read.
        """
        if not self.output_path.exists() or self.output_path.stat().st_size == 0:
            return set()
        if not self.resume:
            raise FileExistsError(
                f"Output already exists; pass --resume or choose another path: {self.output_path}"
            )
        with self.output_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {record_field.name for record_field in fields(TrialRecord)}
            if reader.fieldnames is None or set(reader.fieldnames) != required:
                raise ValueError("Existing output has an incompatible CSV schema.")
            rows = list(reader)
        if any(row["run_signature"] != signature for row in rows):
            raise ValueError("Existing output belongs to a different benchmark plan.")
        return {row["trial_id"] for row in rows}

    def _run_python_trial(
        self,
        case: InstanceCase,
        solver: str,
        python_algorithm: str | None,
        budget_seconds: int,
        seed: int,
        trial_id: str,
    ) -> SolverOutput:
        """Run one Python algorithm or default portfolio in an isolated process.

        Args:
            case (InstanceCase): Verified Knapsack instance and optimum.
            solver (str): Display name for one algorithm or the default portfolio.
            python_algorithm (str | None): Python factory name, or None for defaults.
            budget_seconds (int): Configured search limit.
            seed (int): Python random-state input.
            trial_id (str): Stable run identity.

        Returns:
            SolverOutput: Parsed keys, metrics, target events, and child status.
        """
        started_ns = time.perf_counter_ns()
        target_profits = self._target_profits(case.optimum)
        payload = {
            "instance_path": str(case.path),
            "algorithm": python_algorithm,
            "budget_seconds": budget_seconds,
            "seed": seed,
            "target_profits": target_profits,
        }
        command = [
            sys.executable,
            "-m",
            "benchmarks.cpp_python_knapsack",
            "--worker",
            json.dumps(payload, separators=(",", ":")),
        ]
        return self._execute_process(command, self.repo_root, os.environ.copy(), started_ns)

    def _run_cpp_trial(
        self,
        case: InstanceCase,
        solver: str,
        budget_seconds: int,
        trial_id: str,
        program_dir: Path,
    ) -> SolverOutput:
        """Run one C++ algorithm or the default portfolio in the temp build.

        Args:
            case (InstanceCase): Verified Knapsack instance and optimum.
            solver (str): Display name for one algorithm or the default portfolio.
            budget_seconds (int): Configured integer search limit.
            trial_id (str): Stable run identity.
            program_dir (Path): Isolated temporary C++ program directory.

        Returns:
            SolverOutput: Parsed keys, metrics, target events, and child status.

        Raises:
            OSError: If temporary config or C++ result files cannot be accessed.
        """
        self._write_cpp_config(program_dir, solver)
        results_dir = program_dir.parent / "Results"
        for filename in ("Solutions_RKO.txt", "Results_RKO.csv"):
            path = results_dir / filename
            if path.exists():
                path.unlink()
        started_ns = time.perf_counter_ns()
        environment = os.environ.copy()
        environment["RKO_BENCH_TARGETS"] = ",".join(
            str(profit) for _, profit in self._target_profits(case.optimum)
        )
        command = [str(program_dir / "runRKO"), str(case.path), str(budget_seconds)]
        process = self._execute_process(command, program_dir, environment, started_ns)
        if process.error is not None or process.return_code != 0:
            return process
        solution_path = results_dir / "Solutions_RKO.txt"
        if not solution_path.exists():
            return SolverOutput(
                None,
                None,
                process.events,
                process.cpp_seed,
                process.elapsed_seconds,
                process.return_code,
                "C++ process completed without writing Solutions_RKO.txt.",
            )
        try:
            keys, objective = self._parse_cpp_solution(solution_path.read_text(encoding="utf-8"))
        except ValueError as error:
            return SolverOutput(
                None,
                None,
                process.events,
                process.cpp_seed,
                process.elapsed_seconds,
                process.return_code,
                str(error),
            )
        return SolverOutput(
            keys,
            objective,
            process.events,
            process.cpp_seed,
            process.elapsed_seconds,
            process.return_code,
            None,
        )

    def _execute_process(
        self,
        command: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
        started_ns: int,
    ) -> SolverOutput:
        """Execute a child with bounded time and parse target telemetry.

        Args:
            command (Sequence[str]): Executable and argument vector.
            cwd (Path): Child working directory.
            environment (Mapping[str, str]): Process environment.
            started_ns (int): Shared monotonic invocation origin.

        Returns:
            SolverOutput: Captured child result or a bounded failure record.
        """
        try:
            process = subprocess.run(
                command,
                cwd=cwd,
                env=dict(environment),
                check=False,
                capture_output=True,
                text=True,
                timeout=max(90, self.budget_seconds * 2),
            )
        except subprocess.TimeoutExpired as error:
            elapsed = (time.perf_counter_ns() - started_ns) / 1_000_000_000
            stderr = self._process_text(error.stderr)
            return SolverOutput(
                None,
                None,
                {},
                None,
                elapsed,
                None,
                f"Child process timed out: {stderr[-1200:]}",
            )
        elapsed = (time.perf_counter_ns() - started_ns) / 1_000_000_000
        try:
            events = self.parse_target_events(process.stderr)
        except ValueError as error:
            return SolverOutput(
                None,
                None,
                {},
                None,
                elapsed,
                process.returncode,
                str(error),
            )
        seed_match = re.search(r"^RKO_BENCH_SEED=(-?\d+)$", process.stderr, re.MULTILINE)
        cpp_seed = int(seed_match.group(1)) if seed_match else None
        if process.returncode != 0:
            return SolverOutput(
                None,
                None,
                events,
                cpp_seed,
                elapsed,
                process.returncode,
                f"Child exited {process.returncode}: {process.stderr[-1200:]}",
            )
        if command[0] == sys.executable or "RKO_BENCH_RESULT " in process.stdout:
            try:
                result_line = next(
                    line.removeprefix("RKO_BENCH_RESULT ")
                    for line in reversed(process.stdout.splitlines())
                    if line.startswith("RKO_BENCH_RESULT ")
                )
                result = json.loads(result_line)
                keys = tuple(float(value) for value in result["keys"])
                objective = float(result["objective"])
            except (StopIteration, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                return SolverOutput(
                    None,
                    None,
                    events,
                    cpp_seed,
                    elapsed,
                    process.returncode,
                    f"Invalid Python result record: {error}; {process.stderr[-1200:]}",
                )
        else:
            keys = None
            objective = None
        return SolverOutput(keys, objective, events, cpp_seed, elapsed, process.returncode, None)

    @staticmethod
    def _process_text(value: str | bytes | None) -> str:
        """Normalize subprocess output returned by completed or timed-out calls.

        Args:
            value (str | bytes | None): Captured child output.

        Returns:
            str: UTF-8 text suitable for a bounded error record.
        """
        if value is None:
            return ""
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value

    @staticmethod
    def _parse_cpp_solution(output: str) -> tuple[tuple[float, ...], float]:
        """Extract full-precision C++ keys and the reported corrected objective.

        Args:
            output (str): Contents of the per-run C++ solution file.

        Returns:
            tuple[tuple[float, ...], float]: Random keys and objective value.

        Raises:
            ValueError: If the C++ output is missing or malformed.
        """
        solution_match = re.search(r"\nSol:\s*(.*?)\nofv:\s*([+\-0-9.eE]+)", output, re.DOTALL)
        if solution_match is None:
            raise ValueError("C++ solution file has no full-precision solution record.")
        try:
            keys = tuple(float(value) for value in solution_match.group(1).split())
            objective = float(solution_match.group(2))
        except ValueError as error:
            raise ValueError("C++ solution record contains invalid numbers.") from error
        if not keys or not math.isfinite(objective) or any(not math.isfinite(key) for key in keys):
            raise ValueError("C++ solution record contains nonfinite or empty values.")
        return keys, objective

    def _write_cpp_config(self, program_dir: Path, solver: str) -> None:
        """Select one C++ algorithm or the pinned ten-algorithm portfolio.

        Args:
            program_dir (Path): Isolated temporary C++ Program directory.
            solver (str): Solo solver name or `RKO-default`.

        Returns:
            None: Replaces only the temporary run configuration.

        Raises:
            OSError: If the temporary configuration cannot be written.
            ValueError: If a solver name is unknown.
        """
        cpp_solver = next(
            (display_name for display_name, _ in PYTHON_ALGORITHMS if display_name == solver),
            None,
        )
        algorithms = CPP_PORTFOLIO if solver == "RKO-default" else (cpp_solver,)
        if any(algorithm is None for algorithm in algorithms):
            raise ValueError(f"Unknown solver configuration: {solver}")
        lines = [
            *algorithms,
            "MAXRUNS 1",
            "debug 0",
            "control 1",
            "strategy 1",
            "restart 1",
            "sizePool 10",
            "",
        ]
        config = program_dir / "config" / "config_tests.conf"
        config.write_text("\n".join(lines), encoding="utf-8")

    def _record_trial(
        self,
        output: SolverOutput,
        case: InstanceCase,
        solver: str,
        implementation: str,
        replicate: int,
        python_seed: int | None,
        budget_seconds: int,
        trial_id: str,
        signature: str,
        metadata: Mapping[str, str],
        require_telemetry: bool = True,
    ) -> TrialRecord:
        """Independently score a returned solution and create one raw row.

        Args:
            output (SolverOutput): Child result and parsed progress events.
            case (InstanceCase): Verified problem instance and exact optimum.
            solver (str): Display name for the algorithm or portfolio.
            implementation (str): Python or corrected C++ implementation label.
            replicate (int): Repetition number.
            python_seed (int | None): Python seed, if applicable.
            budget_seconds (int): Configured search limit.
            trial_id (str): Stable cell/repetition identifier.
            signature (str): Benchmark plan hash.
            metadata (Mapping[str, str]): Source and host identity.
            require_telemetry (bool): Demand target-time events for every reached target;
                implementations without telemetry report targets as not reached.

        Returns:
            TrialRecord: Independently checked raw trial record.
        """
        if output.error is not None or output.keys is None or output.objective is None:
            return self._failure_record(
                output,
                case,
                solver,
                implementation,
                replicate,
                python_seed,
                budget_seconds,
                trial_id,
                signature,
                metadata,
            )
        try:
            if len(output.keys) != case.instance.n_items:
                raise ValueError("Returned key dimension does not match the instance.")
            decisions = decode_knapsack(list(output.keys))
            profit, weight, feasible = evaluate_solution(decisions, case.instance)
            if feasible and profit > case.optimum:
                raise ValueError("Feasible solution exceeds the exact optimum.")
            objective = float(100000 * max(0, weight - case.instance.capacity) - profit)
            if not math.isclose(output.objective, objective, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(
                    f"Reported objective {output.objective} differs from independently "
                    f"recomputed objective {objective}."
                )
            targets = dict(self._target_profits(case.optimum))
            if require_telemetry and any(
                feasible and profit >= target_profit and gap not in output.events
                for gap, target_profit in targets.items()
            ):
                raise ValueError("Final solution reached a target missing from telemetry.")
            hits = {
                gap: gap in output.events and output.events[gap] <= budget_seconds
                for gap in targets
            }
            if any(
                elapsed < 0 or elapsed > output.elapsed_seconds + 0.05
                for elapsed in output.events.values()
            ):
                raise ValueError("Target telemetry falls outside the measured invocation.")
            times = {
                gap: min(output.events.get(gap, budget_seconds), budget_seconds)
                for gap in TARGET_GAPS
            }
            keys_text = ",".join(format(key, ".17g") for key in output.keys)
            solution_hash = hashlib.sha256(keys_text.encode("ascii")).hexdigest()
            return TrialRecord(
                trial_id=trial_id,
                implementation=implementation,
                solver=solver,
                family=case.family,
                instance=Path(case.relative_path).name,
                instance_sha256=case.sha256,
                capacity=case.instance.capacity,
                exact_optimum=case.optimum,
                replicate=replicate,
                python_seed=python_seed,
                cpp_seed=output.cpp_seed,
                budget_seconds=budget_seconds,
                status="ok",
                profit=profit,
                weight=weight,
                feasible=feasible,
                objective=objective,
                normalized_objective_difference=self.normalized_objective_difference(
                    objective, case.optimum
                ),
                feasible_gap_percent=(
                    100.0 * (case.optimum - profit) / case.optimum if feasible else None
                ),
                success_10_percent=hits[10],
                time_10_percent_seconds=times[10],
                success_1_percent=hits[1],
                time_1_percent_seconds=times[1],
                success_optimum=hits[0],
                time_optimum_seconds=times[0],
                elapsed_seconds=output.elapsed_seconds,
                execution_mode=(
                    "python-process-portfolio"
                    if implementation == "python" and solver == "RKO-default"
                    else "python-serial"
                    if implementation == "python"
                    else "cpp-openmp-10"
                    if solver == "RKO-default"
                    else "cpp-openmp-1"
                ),
                solution_sha256=solution_hash,
                return_code=output.return_code,
                error=None,
                run_signature=signature,
                **metadata,
            )
        except (ValueError, OverflowError) as error:
            failed = SolverOutput(
                output.keys,
                output.objective,
                output.events,
                output.cpp_seed,
                output.elapsed_seconds,
                output.return_code,
                str(error),
            )
            return self._failure_record(
                failed,
                case,
                solver,
                implementation,
                replicate,
                python_seed,
                budget_seconds,
                trial_id,
                signature,
                metadata,
            )

    def _failure_record(
        self,
        output: SolverOutput,
        case: InstanceCase,
        solver: str,
        implementation: str,
        replicate: int,
        python_seed: int | None,
        budget_seconds: int,
        trial_id: str,
        signature: str,
        metadata: Mapping[str, str],
    ) -> TrialRecord:
        """Preserve one failed or unverifiable trial without fabricated metrics.

        Args:
            output (SolverOutput): Failure details from execution or verification.
            case (InstanceCase): Input provenance.
            solver (str): Algorithm or portfolio label.
            implementation (str): Python or corrected C++ implementation label.
            replicate (int): Repetition number.
            python_seed (int | None): Python random seed, if applicable.
            budget_seconds (int): Configured search limit.
            trial_id (str): Stable cell/repetition identifier.
            signature (str): Benchmark plan hash.
            metadata (Mapping[str, str]): Source and host identity.

        Returns:
            TrialRecord: Explicit error row with solution metrics left empty.
        """
        return TrialRecord(
            trial_id=trial_id,
            implementation=implementation,
            solver=solver,
            family=case.family,
            instance=Path(case.relative_path).name,
            instance_sha256=case.sha256,
            capacity=case.instance.capacity,
            exact_optimum=case.optimum,
            replicate=replicate,
            python_seed=python_seed,
            cpp_seed=output.cpp_seed,
            budget_seconds=budget_seconds,
            status="error",
            profit=None,
            weight=None,
            feasible=None,
            objective=None,
            normalized_objective_difference=None,
            feasible_gap_percent=None,
            success_10_percent=False,
            time_10_percent_seconds=None,
            success_1_percent=False,
            time_1_percent_seconds=None,
            success_optimum=False,
            time_optimum_seconds=None,
            elapsed_seconds=output.elapsed_seconds,
            execution_mode=(
                "python-process-portfolio"
                if implementation == "python" and solver == "RKO-default"
                else "python-serial"
                if implementation == "python"
                else "cpp-openmp-10"
                if solver == "RKO-default"
                else "cpp-openmp-1"
            ),
            solution_sha256=None,
            return_code=output.return_code,
            error=(output.error or "Unspecified solver failure")[:1500],
            run_signature=signature,
            **metadata,
        )

    def _write_summary(self, signature: str) -> None:
        """Compute per-instance summaries and corrected version-comparison tests.

        Args:
            signature (str): Run-plan fingerprint to validate in raw data.

        Returns:
            None: Writes a summary CSV next to the raw trial file.

        Raises:
            OSError: If raw or summary files cannot be read or written.
            ValueError: If raw rows belong to a different run plan.
        """
        with self.output_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if any(row["run_signature"] != signature for row in rows):
            raise ValueError("Raw results contain a different run signature.")
        trial_counts = {
            solver: {
                f"{implementation}_{label}_trials": sum(
                    row["solver"] == solver
                    and row["implementation"] == implementation
                    and row["status"] == status
                    for row in rows
                )
                for implementation, label, status in (
                    ("python", "successful", "ok"),
                    ("python", "failed", "error"),
                    ("cpp64", "successful", "ok"),
                    ("cpp64", "failed", "error"),
                )
            }
            for solver, _ in SOLVER_CONFIGS
        }
        metrics = (
            ("normalized_objective_difference", "lower"),
            ("feasible_rate", "higher"),
            ("success_1_percent_rate", "higher"),
            ("time_1_percent_seconds", "lower"),
            ("elapsed_seconds", "lower"),
        )
        summaries: list[dict[str, str | int | float]] = []
        for metric, direction in metrics:
            comparisons: dict[str, tuple[list[float], list[float], list[float]]] = {}
            for solver, _ in SOLVER_CONFIGS:
                python_by_instance: dict[str, list[float]] = {}
                cpp_by_instance: dict[str, list[float]] = {}
                for row in rows:
                    if row["solver"] != solver:
                        continue
                    target = (
                        python_by_instance if row["implementation"] == "python" else cpp_by_instance
                    )
                    if metric in ("feasible_rate", "success_1_percent_rate"):
                        field_name = (
                            "feasible" if metric == "feasible_rate" else "success_1_percent"
                        )
                        target.setdefault(row["instance_sha256"], []).append(
                            float(row["status"] == "ok" and row[field_name] == "True")
                        )
                    elif metric == "time_1_percent_seconds" and row["status"] != "ok":
                        target.setdefault(row["instance_sha256"], []).append(
                            float(row["budget_seconds"])
                        )
                    elif row[metric] != "" and (
                        row["status"] == "ok" or metric == "elapsed_seconds"
                    ):
                        target.setdefault(row["instance_sha256"], []).append(float(row[metric]))
                python_values: list[float] = []
                cpp_values: list[float] = []
                differences: list[float] = []
                for instance_hash in sorted(python_by_instance.keys() & cpp_by_instance.keys()):
                    python_summary = (
                        statistics.mean(python_by_instance[instance_hash])
                        if metric in ("feasible_rate", "success_1_percent_rate")
                        else statistics.median(python_by_instance[instance_hash])
                    )
                    cpp_summary = (
                        statistics.mean(cpp_by_instance[instance_hash])
                        if metric in ("feasible_rate", "success_1_percent_rate")
                        else statistics.median(cpp_by_instance[instance_hash])
                    )
                    python_values.append(python_summary)
                    cpp_values.append(cpp_summary)
                    differences.append(python_summary - cpp_summary)
                comparisons[solver] = (python_values, cpp_values, differences)
            raw_p_values = {
                solver: self.exact_sign_test(differences)[0]
                for solver, (_, _, differences) in comparisons.items()
                if differences
            }
            adjusted = self.holm_adjust(raw_p_values)
            for solver, (python_values, cpp_values, differences) in comparisons.items():
                if not differences:
                    continue
                interval = self.bootstrap_median_interval(
                    differences,
                    seed=SAMPLE_SEED + sum(ord(character) for character in solver + metric),
                )
                p_value, non_tied, positive, negative = self.exact_sign_test(differences)
                summaries.append(
                    {
                        "solver": solver,
                        "metric": metric,
                        "direction": direction,
                        "python_median_across_instances": statistics.median(python_values),
                        "cpp64_median_across_instances": statistics.median(cpp_values),
                        "median_paired_difference_python_minus_cpp": statistics.median(differences),
                        "bootstrap_95_percent_low": interval[0],
                        "bootstrap_95_percent_high": interval[1],
                        "p_value_exact_sign": p_value,
                        "p_value_holm": adjusted[solver],
                        "paired_instances": len(differences),
                        "non_tied_instances": non_tied,
                        "positive_python_minus_cpp": positive,
                        "negative_python_minus_cpp": negative,
                        "tied_instances": len(differences) - non_tied,
                        **trial_counts[solver],
                    }
                )
        summary_path = self.output_path.with_name(f"{self.output_path.stem}-summary.csv")
        fields = list(summaries[0]) if summaries else ["solver", "metric"]
        with summary_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summaries)

    def _target_profits(self, optimum: int) -> tuple[tuple[int, int], ...]:
        """Convert declared relative-gap targets to integer profit thresholds.

        Args:
            optimum (int): Exact maximum feasible profit.

        Returns:
            tuple[tuple[int, int], ...]: Gap percentage and minimum profit pairs.

        Raises:
            ValueError: If the exact optimum is not positive.
        """
        if optimum <= 0:
            raise ValueError("Exact optimum must be positive.")
        return tuple((gap, math.ceil(optimum * (1.0 - gap / 100.0))) for gap in TARGET_GAPS)

    @staticmethod
    def _trial_id(implementation: str, solver: str, case: InstanceCase, replicate: int) -> str:
        """Construct a stable key for one implementation and repeat.

        Args:
            implementation (str): Python or corrected C++ implementation.
            solver (str): Algorithm or portfolio label.
            case (InstanceCase): Input identity.
            replicate (int): Repetition number.

        Returns:
            str: Stable slash-separated trial identity.
        """
        return f"{implementation}/{solver}/{case.relative_path}/r{replicate:02d}"

    @staticmethod
    def _run_worker(payload: Mapping[str, Any]) -> None:
        """Execute one Python solver inside a process suitable for spawn pickling.

        Args:
            payload (Mapping[str, Any]): Instance, algorithm, seed and telemetry data.

        Returns:
            None: Writes one machine-readable result to standard output.

        Raises:
            KeyError: If a required worker field is absent.
            OSError: If standard output cannot be written.
            ValueError: If the input instance or solver settings are invalid.
        """
        instance = read_kp_instance(Path(str(payload["instance_path"])))
        cost = TargetTrackingCost(
            instance=instance,
            target_profits=tuple(
                (int(gap), int(profit)) for gap, profit in payload["target_profits"]
            ),
            started_ns=0,
        )
        algorithm = payload["algorithm"]
        engine = payload.get("engine") or "python"
        execution = (
            payload.get("execution") or ("sequential" if algorithm is not None else "process")
            if engine == "python"
            else None
        )
        solver = RKO(
            Callbacks(decode_knapsack, cost, instance.n_items),
            algorithms=[str(algorithm)] if algorithm is not None else None,
            control=1,
            pool_size=10,
            engine=engine,
            execution=execution,
            random_state=int(payload["seed"]),
        )
        cost.started_ns = time.perf_counter_ns()
        result = solver.solve(float(payload["budget_seconds"]))
        print(
            "RKO_BENCH_RESULT "
            + json.dumps(
                {"keys": result.keys, "objective": result.cost},
                separators=(",", ":"),
                allow_nan=False,
            ),
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full comparison, a small pilot, or an internal Python trial.

    Args:
        argv (Sequence[str] | None): Optional command-line argument vector.

    Returns:
        int: Process exit code; zero indicates successful completion.

    Raises:
        OSError: If benchmark files cannot be read or written.
        RuntimeError: If C++ compilation or verification fails.
        ValueError: If a resume file has incompatible provenance.
    """
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "--worker":
        if len(argv) != 2:
            raise ValueError("The internal worker requires one JSON payload.")
        KnapsackBenchmark._run_worker(json.loads(argv[1]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "cpp-python-knapsack-reference.csv",
        help="Raw per-trial CSV path.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "rko-cpp-python-knapsack",
        help="External source and KPLIB cache directory.",
    )
    parser.add_argument(
        "--budget-seconds",
        type=int,
        default=DEFAULT_BUDGET_SECONDS,
        help="Integer search budget per trial; default 120 seconds.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="Independent repetitions per implementation/configuration/instance; default 5.",
    )
    parser.add_argument("--resume", action="store_true", help="Continue a matching partial CSV.")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run BRKGA, BRKGA-CS, and the portfolio on one instance for one second.",
    )
    args = parser.parse_args(argv)
    output = args.output
    if (
        args.pilot
        and output == Path(__file__).resolve().parent / "cpp-python-knapsack-reference.csv"
    ):
        output = Path(tempfile.gettempdir()) / "rko-cpp-python-knapsack-pilot.csv"
    benchmark = KnapsackBenchmark(
        output,
        args.cache_dir,
        budget_seconds=args.budget_seconds,
        repetitions=args.repetitions,
        resume=args.resume,
    )
    benchmark.run(pilot=args.pilot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
