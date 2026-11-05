"""Non-inferiority gate comparing a candidate engine configuration with a baseline.

Objective:
- Decide, per problem family and budget, whether a candidate is no worse than a baseline by
  more than a margin fixed beforehand, with the evidence tied to one recorded native binary.

Technical approach:
- Run exact-optimum Knapsack (KPLIB, 500 items) and shifted Rastrigin (100 and 300 dimensions,
  optimum 0) through the public solver on the Rust or C++ engine, at wall-time and equal
  `max_evaluations` budgets, with fixed per-pair seeds and alternating run order.
- Calibration and test instances come from different seeds and never overlap.
- Write one JSONL row per trial carrying the SHA-256 of the native extension, the git commit,
  the host, the budget and the absolute gap to the reference value.
- Per family and budget, collapse replicates to per-instance medians, pair the
  candidate-minus-baseline difference, and reuse `NativeReport` for the exact signed-rank test
  and Holm correction and `KnapsackBenchmark` for the seeded 10,000-resample bootstrap.

Flow (pseudocode):
1. Fingerprint the native extension; refuse to run when it cannot be hashed.
2. Build the split's instances and, for each instance, budget and replicate, run both sides.
3. Append every trial (failures included) to the JSONL file.
4. Compute the margin (larger of a fraction of the median baseline gap and a per-family floor),
   the bootstrap upper bound of the median difference and a one-sided signed-rank p.
5. Apply Holm across the families of each budget and print one verdict per family and budget.

Important criteria:
- A cell passes only when the 95% bootstrap upper bound is at most the margin and the
  Holm-adjusted p is below 0.05; "not significant" is never read as "equal".
- A cell with fewer than 15 paired instances is `inconclusive` and is not tested.
- Normalized gap is percent of the optimum for Knapsack and the absolute value for Rastrigin.
- The margin is max(fraction * median baseline gap, family floor). The floor stops the margin
  from collapsing to zero when the baseline gap is near zero, which made a same-binary A/A
  comparison fail every cell. Floors are in the unit of `normalized_gap` (`KNAPSACK_FLOOR`,
  `RASTRIGIN_FLOOR`) and `--floor` overrides them; the report prints the effective margin and
  the term (`fraction` or `floor`) that set it.
- Rastrigin uses 100 and 300 dimensions. 30 dimensions was removed because every budget solves
  it to a gap of about 0, so it cannot discriminate between engines.
- The `evals` budget runs the Rust engine with `n_jobs=1` (algorithms in sequence, each capped
  by `max_evaluations`) and a high time cap (`--evals-time-cap`, 600 s) so the evaluation cap is
  the active limit. The C++ engine rejects `n_jobs`. Rows record `n_jobs` and `time_cap`.
- `--candidate-option KEY=VALUE` and `--baseline-option KEY=VALUE` (repeatable) override scalar
  keys of the solver `config.toml` for one side. `ConfigOverrides` copies the default parameter
  directory to a temporary directory, rewrites the keys there and passes it as `params_dir`, so
  two configurations run with the same native binary, interleaved in the same instant. A bare
  key resolves to the top level and then the unique table that defines it; `table.key` is
  explicit. The key must exist in the default file and the value is parsed with its type (bool,
  int, float or str); anything else is refused before any trial. Rows record `spec` and
  `options`.
- Time-budget results are noisy on a machine with fewer CPUs than engine threads (the study
  machine has 6 CPUs and the portfolio runs 10 threads); the report prints this caveat.

Inputs/Outputs:
- Input: candidate and baseline `engine[/ALGO+ALGO]` specs (the baseline may be a JSONL file
  from an earlier gate run), per-side `config.toml` overrides, families, budgets, replicates and
  instance count.
- Output: a JSONL trial file and a printed per-family, per-budget non-inferiority table.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
from benchmarks.engines.gate_instances import EVALUATION_TIME_CAP
from benchmarks.engines.gate_instances import GateConfig
from benchmarks.engines.gate_instances import GateInstance
from benchmarks.engines.gate_instances import GateInstances
from benchmarks.engines.gate_instances import SideSpec
from benchmarks.engines.gate_instances import TrialSpec
from benchmarks.engines.gate_instances import expand_families
from benchmarks.engines.gate_instances import parse_budgets
from benchmarks.engines.gate_support import MARGIN_FRACTION
from benchmarks.engines.gate_support import MIN_INSTANCES
from benchmarks.engines.gate_support import ConfigOverrides
from benchmarks.engines.gate_support import GateReport
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from rko_solver import RKO
from typing import Any
from typing import TextIO

SCHEMA = "rko-gate-1"
GATE_SEED = 20261201
ENGINES = ("rust", "cpp")
DEFAULT_CACHE = Path.home() / ".cache" / "rko-cpp-python-knapsack"
__all__ = [
    "ConfigOverrides",
    "Gate",
    "GateConfig",
    "GateInstance",
    "GateInstances",
    "GateReport",
    "SideSpec",
    "TrialSpec",
    "main",
    "parse_budgets",
]


class Gate:
    """Run the trials of one gate invocation and write the JSONL evidence.

    Responsibilities:
    - Refuse to run without a hashable native binary.
    - Execute candidate and baseline in alternating order, never concurrently.
    - Load a recorded baseline from an earlier JSONL file.
    - Give each side its own temporary parameter directory when it has `config.toml` overrides.

    Attributes:
        config (GateConfig): The validated run options.
        instances (GateInstances): Instance factory.
        candidate (str): Candidate `engine[/ALGO+ALGO]` spec.
        baseline (str | None): Baseline spec, a JSONL path, or None to only record the candidate.
        families (tuple[str, ...]): Analysis families to run.
        budgets (tuple[Budget, ...]): Budgets to run.
        replicates (int): Replicates per instance and budget.
        output (Path): JSONL destination.
    """

    def __init__(self, config: GateConfig, instances: GateInstances) -> None:
        """Validate and store the run options.

        Args:
            config (GateConfig): Candidate and baseline specs, families, budgets, replicates,
                output path and per-side `config.toml` overrides.
            instances (GateInstances): Instance factory.

        Raises:
            ValueError: If a spec, family, replicate count or option is invalid, or baseline
                options are given with a recorded JSONL baseline or without a baseline.
        """
        if config.replicates < 1:
            raise ValueError("replicates must be positive.")
        recorded = config.baseline is not None and Path(config.baseline).is_file()
        if config.baseline_options and (config.baseline is None or recorded):
            raise ValueError("--baseline-option needs a live --baseline spec, not a JSONL file.")
        self.config = config
        self.instances = instances
        self.candidate = config.candidate
        self.baseline = config.baseline
        self.families = expand_families(config.families)
        self.budgets = tuple(config.budgets)
        self.replicates = config.replicates
        self.output = config.output
        self._sides = self._build_sides(config, recorded)

    def _build_sides(self, config: GateConfig, recorded: bool) -> list[SideSpec]:
        """Build the candidate and, unless the baseline is recorded, the baseline side.

        Args:
            config (GateConfig): Validated run options with per-side specs and overrides.
            recorded (bool): Whether the baseline is a JSONL file rather than a live spec.

        Returns:
            list[SideSpec]: One or two sides with parsed engines and overrides.

        Raises:
            ValueError: If a spec or option is invalid.
        """
        candidate_engine, candidate_algorithms = self.parse_spec(config.candidate)
        sides = [
            SideSpec(
                "candidate",
                candidate_engine,
                candidate_algorithms,
                ConfigOverrides.parse(config.candidate_options),
            )
        ]
        if config.baseline is not None and not recorded:
            baseline_engine, baseline_algorithms = self.parse_spec(config.baseline)
            sides.append(
                SideSpec(
                    "baseline",
                    baseline_engine,
                    baseline_algorithms,
                    ConfigOverrides.parse(config.baseline_options),
                )
            )
        return sides

    @staticmethod
    def parse_spec(spec: str) -> tuple[str, tuple[str, ...] | None]:
        """Split `engine[/ALGO+ALGO]` into an engine and an optional portfolio.

        Args:
            spec (str): Engine spec; `default` or no suffix means the configured portfolio.

        Returns:
            tuple[str, tuple[str, ...] | None]: Engine and algorithm names.

        Raises:
            ValueError: If the engine is not `rust` or `cpp`.
        """
        engine, _, config = spec.partition("/")
        if engine not in ENGINES:
            raise ValueError(f"Engine must be one of {ENGINES}, got {engine!r}.")
        return engine, (tuple(config.split("+")) if config not in ("", "default") else None)

    @staticmethod
    def native_fingerprint() -> tuple[str, str]:
        """Locate and hash the compiled extension.

        Returns:
            tuple[str, str]: Path and SHA-256 of `rko_solver._native`.

        Raises:
            RuntimeError: If the extension is missing or unreadable, so no hash can be recorded.
        """
        spec = importlib.util.find_spec("rko_solver._native")
        if spec is None or spec.origin is None or not Path(spec.origin).is_file():
            raise RuntimeError("Refusing to run: the native extension has no recordable hash.")
        origin = Path(spec.origin)
        return str(origin), hashlib.sha256(origin.read_bytes()).hexdigest()

    @staticmethod
    def git_state() -> tuple[str, bool]:
        """Return the checked-out commit and whether the tree has uncommitted changes.

        Returns:
            tuple[str, bool]: Commit hash (or `unknown`) and the dirty flag.
        """
        root = Path(__file__).resolve().parents[2]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return commit or "unknown", bool(dirty)

    @staticmethod
    def load_baseline(path: Path) -> list[dict[str, Any]]:
        """Read the candidate-side rows of an earlier gate file as the baseline.

        Args:
            path (Path): JSONL file written by this module.

        Returns:
            list[dict[str, Any]]: Rows relabeled with side `baseline`.

        Raises:
            ValueError: If the file has no candidate rows of this schema.
        """
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        selected = [
            {**row, "side": "baseline"}
            for row in rows
            if row.get("schema") == SCHEMA and row.get("side") == "candidate"
        ]
        if not selected:
            raise ValueError(f"{path} has no {SCHEMA} candidate rows.")
        return selected

    def _context(self) -> dict[str, Any]:
        """Build the shared JSONL fields that fingerprint the binary, commit and host.

        Returns:
            dict[str, Any]: Constant fields merged into every trial row.

        Raises:
            RuntimeError: If the native extension cannot be hashed.
        """
        binary_path, binary_hash = self.native_fingerprint()
        commit, dirty = self.git_state()
        return {
            "schema": SCHEMA,
            "native_path": binary_path,
            "native_sha256": binary_hash,
            "git_commit": commit,
            "git_dirty": dirty,
            "host": {
                "node": platform.node(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "cpus": os.cpu_count(),
            },
            "split": self.instances.split,
        }

    def _resolved_sides(self, scratch: Path) -> list[SideSpec]:
        """Return the sides with a per-side parameter directory for every override.

        Args:
            scratch (Path): Temporary root that holds one subdirectory per overridden side.

        Returns:
            list[SideSpec]: Sides whose `params_dir` is the override directory or None.

        Raises:
            ValueError: If an override cannot be applied to `config.toml`.
            OSError: If a parameter file cannot be copied or written.
        """
        resolved: list[SideSpec] = []
        for side in self._sides:
            params_dir: Path | None = None
            if side.options.values:
                directory = scratch / side.name
                directory.mkdir()
                params_dir = side.options.write(directory)
            resolved.append(
                SideSpec(side.name, side.engine, side.algorithms, side.options, params_dir)
            )
        return resolved

    def run(self) -> list[dict[str, Any]]:
        """Execute every trial, stream rows to the JSONL file and return them.

        Returns:
            list[dict[str, Any]]: Candidate and baseline rows, loaded baseline included.

        Raises:
            RuntimeError: If the native extension cannot be hashed.
            OSError: If an input or the output cannot be read or written.
            ValueError: If a baseline file is invalid.
        """
        context = self._context()
        rows: list[dict[str, Any]] = []
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with (
            tempfile.TemporaryDirectory(prefix="rko-gate-") as scratch,
            self.output.open("w", encoding="utf-8") as stream,
        ):
            sides = self._resolved_sides(Path(scratch))
            for indexed in enumerate(self.families):
                rows.extend(self._run_family(indexed, sides, context, stream))
        if self.baseline is not None and Path(self.baseline).is_file():
            rows.extend(self.load_baseline(Path(self.baseline)))
        return rows

    def _run_family(
        self,
        indexed: tuple[int, str],
        sides: Sequence[SideSpec],
        context: Mapping[str, Any],
        stream: TextIO,
    ) -> list[dict[str, Any]]:
        """Run every instance, budget and replicate of one family and stream the rows.

        Args:
            indexed (tuple[int, str]): Family position (used to seed trials) and family name.
            sides (Sequence[SideSpec]): Candidate and, when present, baseline side.
            context (Mapping[str, Any]): Constant JSONL fields merged into every row.
            stream (TextIO): Open text stream that receives one JSON line per trial.

        Returns:
            list[dict[str, Any]]: Every trial row written for this family.

        Raises:
            OSError: If an input or the output cannot be read or written.
        """
        family_index, family = indexed
        rows: list[dict[str, Any]] = []
        for instance_index, instance in enumerate(self.instances.build(family)):
            for budget in self.budgets:
                for replicate in range(self.replicates):
                    seed = GATE_SEED + family_index * 1_000_000 + instance_index * 100 + replicate
                    spec = TrialSpec(instance, budget, replicate, seed)
                    for row in self._pair(spec, sides, context):
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
                        rows.append(row)
        return rows

    def _pair(
        self,
        spec: TrialSpec,
        sides: Sequence[SideSpec],
        context: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Run both sides of one pair in an order that alternates with the replicate.

        Args:
            spec (TrialSpec): Instance, budget and seeding of the pair.
            sides (Sequence[SideSpec]): Candidate and, when present, baseline side.
            context (Mapping[str, Any]): Constant JSONL fields merged into every row.

        Returns:
            list[dict[str, Any]]: One trial row per side, in run order.
        """
        shift = spec.replicate % len(sides)
        rows: list[dict[str, Any]] = []
        for side in list(sides[shift:]) + list(sides[:shift]):
            row = self._trial(spec, side)
            row.update(
                context,
                side=side.name,
                spec=self.candidate if side.name == "candidate" else self.baseline,
                options=side.options.as_record(),
            )
            rows.append(row)
        return rows

    @staticmethod
    def _trial(spec: TrialSpec, side: SideSpec) -> dict[str, Any]:
        """Solve once and score the returned keys independently of the solver.

        Args:
            spec (TrialSpec): Instance, budget and seeding of the trial.
            side (SideSpec): Engine, portfolio and parameter directory of this side.

        Returns:
            dict[str, Any]: Trial row; failures keep `status` `error` and a null gap.
        """
        instance, budget, engine = spec.instance, spec.budget, side.engine
        row: dict[str, Any] = {
            "family": instance.family,
            "instance": instance.name,
            "instance_sha256": instance.sha256,
            "budget": budget.label,
            "time_limit": budget.time_limit,
            "max_evaluations": budget.max_evaluations,
            "n_jobs": budget.solver_options(engine).get("n_jobs"),
            "time_cap": budget.time_limit,
            "replicate": spec.replicate,
            "seed": spec.seed,
            "engine": engine,
            "reference_value": instance.reference,
        }
        try:
            result = RKO(
                instance.problem,
                algorithms=side.algorithms,
                engine=engine,
                random_state=spec.seed,
                params_dir=side.params_dir,
                **budget.solver_options(engine),
            ).solve(budget.time_limit)
            gap = instance.score(result.keys)
        except Exception as error:
            row.update(status="error", error=f"{type(error).__name__}: {error}", gap=None)
            row.update(normalized_gap=None, evaluations=None, elapsed_seconds=None)
            return row
        row.update(
            status="ok",
            gap=gap,
            normalized_gap=100.0 * gap / instance.reference if instance.reference > 0 else gap,
            evaluations=result.n_evaluations,
            elapsed_seconds=result.elapsed_seconds,
            stop_reason=result.stop_reason,
        )
        return row


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser of the gate.

    Returns:
        argparse.ArgumentParser: Parser with every gate option registered.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", help="Candidate engine[/ALGO+ALGO] spec.")
    parser.add_argument("--baseline", help="Baseline spec or a JSONL file from an earlier run.")
    parser.add_argument("--families", default="knapsack,rastrigin")
    parser.add_argument("--budgets", default="2,10,evals")
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--instances", type=int, default=MIN_INSTANCES)
    parser.add_argument("--evaluations", type=int, default=100_000)
    parser.add_argument(
        "--evals-time-cap",
        type=float,
        default=EVALUATION_TIME_CAP,
        help="Seconds of wall-clock cap for the sequential n_jobs=1 evals budget.",
    )
    parser.add_argument(
        "--candidate-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config.toml key for the candidate (repeatable).",
    )
    parser.add_argument(
        "--baseline-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config.toml key for a live baseline spec (repeatable).",
    )
    parser.add_argument("--margin", type=float, default=MARGIN_FRACTION)
    parser.add_argument(
        "--floor",
        default="",
        help="Absolute margin floors, e.g. knapsack=0.05,rastrigin=0.05.",
    )
    parser.add_argument("--split", choices=("test", "calibration"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--quick", action="store_true", help="3 instances, 1 s, 1 replicate.")
    parser.add_argument("--summary-only", action="store_true", help="Summarize --output only.")
    return parser


def _summary_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load the `--output` rows of the requested families for `--summary-only`.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        list[dict[str, Any]]: Rows whose family is in `--families`.

    Raises:
        OSError: If the output file cannot be read.
    """
    wanted = set(expand_families(args.families.split(",")))
    loaded = [json.loads(line) for line in args.output.read_text().splitlines() if line]
    rows = [row for row in loaded if row["family"] in wanted]
    ignored = sorted({str(row["family"]) for row in loaded} - wanted)
    if ignored:
        sys.stdout.write(f"Ignored families outside --families: {', '.join(ignored)}.\n")
    return rows


def _run_gate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    """Build and run the gate from parsed arguments, exiting the parser on bad input.

    Args:
        args (argparse.Namespace): Parsed arguments; `--quick` is already applied.
        parser (argparse.ArgumentParser): Parser used to report option errors.

    Returns:
        list[dict[str, Any]]: Trial rows, candidate and baseline.

    Raises:
        SystemExit: On an invalid option (through `parser.error`) or a refused run (code 2).
        OSError: If an input or the output cannot be read or written.
    """
    replicates = args.replicates or (1 if args.quick else 3)
    config = GateConfig(
        args.candidate,
        args.baseline,
        args.families.split(","),
        parse_budgets(args.budgets, args.evaluations, args.evals_time_cap),
        replicates,
        args.output,
        args.candidate_option,
        args.baseline_option,
    )
    try:
        gate = Gate(config, GateInstances(args.cache_dir, args.split, args.instances))
    except ValueError as error:
        parser.error(str(error))
    try:
        return gate.run()
    except RuntimeError as error:
        sys.stderr.write(f"{error}\n")
        raise SystemExit(2) from error


def main(argv: Sequence[str] | None = None) -> int:
    """Run or summarize the gate from the command line.

    Args:
        argv (Sequence[str] | None): Optional argument vector.

    Returns:
        int: Zero when no cell fails, one when any cell fails, two on a refused run.

    Raises:
        OSError: If files or inputs are unavailable.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = GateReport(args.margin, floors=GateReport.parse_floors(args.floor))
    if args.summary_only:
        rows = _summary_rows(args)
    else:
        if args.candidate is None:
            parser.error("--candidate is required unless --summary-only is given.")
        if args.quick:
            args.instances, args.budgets, args.evaluations = 3, "1,evals", 20_000
        rows = _run_gate(args, parser)
    if args.baseline is None and not args.summary_only:
        sys.stdout.write(f"Recorded {len(rows)} candidate rows to {args.output}.\n")
        return 0
    cells = report.analyze(rows)
    sys.stdout.write(report.render(cells) + "\n")
    return int(any(cell["verdict"] == "fail" for cell in cells))


if __name__ == "__main__":
    raise SystemExit(main())
