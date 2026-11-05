"""Command-line entry points of the expanded utility study.

Objective:
- Freeze the manifest, run one mechanism with provenance and print the six class decisions.

Technical approach:
- `manifest` draws and fingerprints fresh inputs; `run` builds the mechanism's executor,
  runs the seeded plan sequentially and writes an exclusive sidecar; `analyze` combines
  both mechanisms' rows with the registered win rules and Holm correction.

Flow (pseudocode):
1. `manifest` writes entries and their digest.
2. `run` verifies the digest, runs every trial or stops at the first error or the ceiling.
3. `analyze` prints a status per registered claim and a JSON line.

Important criteria:
- Output files are never overwritten and a tampered manifest is refused.
- An incomplete campaign is reported as such, never as evidence.

Inputs/Outputs:
- Input: argument vectors, a manifest file and result rows.
- Output: JSON files, a printed table and exit codes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from benchmarks.algorithms.utility_analysis import UtilityAnalysis
from benchmarks.algorithms.utility_executors import AdaptiveExecutor
from benchmarks.algorithms.utility_executors import TemperatureExecutor
from benchmarks.algorithms.utility_executors import UtilityScoring
from benchmarks.algorithms.utility_inputs import UtilityInputs
from benchmarks.algorithms.utility_study import CLASSES
from benchmarks.algorithms.utility_study import UtilityManifest
from benchmarks.algorithms.utility_study import UtilityRunner
from benchmarks.algorithms.utility_study import UtilityStudy
from benchmarks.engines.gate import DEFAULT_CACHE
from benchmarks.engines.gate import Gate
from benchmarks.engines.gate_support import ConfigOverrides
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import Any

BUDGET = (10.0, 100_000)
CLAIMS = tuple(
    f"{mechanism}:{kind}" for mechanism in ("adaptive", "temperature") for kind in CLASSES
)
BINARY = Path(__file__).resolve().parents[2] / "target" / "release" / "rko_bench"


class UtilityCli:
    """Command-line entry points: build the manifest, run one mechanism, analyze rows.

    Responsibilities:
    - Freeze a manifest and write exclusive, provenance-tagged result files.
    - Combine both mechanisms' rows into the six registered class decisions.
    """

    @staticmethod
    def main(argv: Sequence[str] | None = None) -> int:
        """Dispatch a subcommand.

        Args:
            argv (Sequence[str] | None): Optional argument vector.

        Returns:
            int: Zero on success, two for refused input, three for an incomplete campaign.

        Raises:
            SystemExit: On invalid arguments.
        """
        args = UtilityCli._parser().parse_args(argv)
        handlers = {
            "manifest": UtilityCli._manifest,
            "run": UtilityCli._run,
            "analyze": UtilityCli._analyze,
        }
        return handlers[args.command](args)

    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        """Build the argument parser with its three subcommands.

        Returns:
            argparse.ArgumentParser: Configured parser.
        """
        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="command", required=True)
        manifest = commands.add_parser("manifest")
        manifest.add_argument("--role", choices=("pilot", "confirmation"), required=True)
        manifest.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
        manifest.add_argument("--output", type=Path, required=True)
        run = commands.add_parser("run")
        run.add_argument("--manifest", type=Path, required=True)
        run.add_argument("--output", type=Path, required=True)
        run.add_argument("--order-seed", type=int, required=True)
        run.add_argument("--mechanism", choices=("adaptive", "temperature"), required=True)
        run.add_argument("--profiles", default="evaluations,time")
        run.add_argument("--max-hours", type=float, default=24.0)
        run.add_argument("--binary", type=Path, default=BINARY)
        analyze = commands.add_parser("analyze")
        analyze.add_argument("--manifest", type=Path, required=True)
        analyze.add_argument("--rows", action="append", default=[], metavar="MECHANISM=PATH")
        analyze.add_argument("--profiles", default="evaluations,time")
        return parser

    @staticmethod
    def _manifest(args: argparse.Namespace) -> int:
        """Build and write a manifest, refusing to overwrite one.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            int: Zero on success, two if the output exists.

        Raises:
            ValueError: If an input is invalid or a pool is exhausted.
            OSError: If an input cannot be fetched or the output written.
        """
        if args.output.exists():
            sys.stderr.write(f"{args.output} exists; refusing to overwrite.\n")
            return 2
        entries = UtilityManifest(UtilityInputs(args.cache_dir), args.role).build()
        document = {
            "role": args.role,
            "digest": UtilityManifest.digest(entries),
            "entries": entries,
        }
        with args.output.open("x") as handle:
            json.dump(document, handle, indent=1)
        return 0

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        """Read a manifest and require its digest to match its entries.

        Args:
            path (Path): Manifest file.

        Returns:
            dict[str, Any]: Manifest document.

        Raises:
            ValueError: If the digest does not match the entries.
            OSError: If the file cannot be read.
        """
        document = json.loads(path.read_text())
        if document["digest"] != UtilityManifest.digest(document["entries"]):
            raise ValueError(f"{path} does not match its recorded digest")
        return document

    @staticmethod
    def _executor(
        args: argparse.Namespace, entries: Sequence[Mapping[str, Any]], scratch: Path
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Build the executor of the chosen mechanism.

        Args:
            args (argparse.Namespace): Parsed arguments.
            entries (Sequence[Mapping[str, Any]]): Manifest entries.
            scratch (Path): Directory for the copied parameter sets.

        Returns:
            Callable[[dict[str, Any]], dict[str, Any]]: Trial executor.

        Raises:
            ValueError: If a configuration override cannot be applied.
            OSError: If a parameter directory cannot be written.
        """
        scoring = UtilityScoring(UtilityInputs(DEFAULT_CACHE))
        if args.mechanism == "temperature":
            return TemperatureExecutor(args.binary, entries, scoring, BUDGET)
        params = {}
        for arm, value in (("baseline", "false"), ("candidate", "true")):
            (scratch / arm).mkdir()
            overrides = ConfigOverrides.parse([f"progress_schedule={value}"])
            params[arm] = overrides.write(scratch / arm)
        return AdaptiveExecutor(entries, scoring, params, BUDGET)

    @staticmethod
    def _run(args: argparse.Namespace) -> int:
        """Run one mechanism over the manifest and write rows plus provenance.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            int: Zero if complete, two if refused, three if the ceiling ended the campaign.

        Raises:
            RuntimeError: If a trial violates its plan.
            OSError: If files cannot be written.
        """
        meta = Path(f"{args.output}.meta.json")
        if args.output.exists() or meta.exists():
            sys.stderr.write("output or sidecar exists; refusing to overwrite.\n")
            return 2
        document = UtilityCli._load(args.manifest)
        entries = document["entries"]
        profiles = tuple(args.profiles.split(","))
        runner = UtilityRunner(entries, profiles, args.order_seed)
        status, error = "complete", ""
        with tempfile.TemporaryDirectory() as scratch:
            try:
                runner.run(
                    UtilityCli._executor(args, entries, Path(scratch)),
                    args.output,
                    args.max_hours * 3600.0,
                )
            except TimeoutError as problem:
                status, error = "incomplete", str(problem)
            except Exception as problem:
                status, error = "failed", f"{type(problem).__name__}: {problem}"
        completed = len(args.output.read_text().splitlines()) if args.output.exists() else 0
        provenance = UtilityCli._provenance(args, document, len(runner.plan()), completed)
        with meta.open("x") as handle:
            json.dump({**provenance, "status": status, "error": error}, handle, indent=1)
        sys.stderr.write(f"{status}: {completed} trials. {error}\n")
        return {"complete": 0, "incomplete": 3}.get(status, 1)

    @staticmethod
    def _provenance(
        args: argparse.Namespace, document: Mapping[str, Any], planned: int, completed: int
    ) -> dict[str, Any]:
        """Collect what is needed to reproduce and audit a run.

        Args:
            args (argparse.Namespace): Parsed arguments.
            document (Mapping[str, Any]): Manifest document.
            planned (int): Planned trial count.
            completed (int): Completed trial count.

        Returns:
            dict[str, Any]: Provenance record.

        Raises:
            OSError: If the executable cannot be read.
        """
        commit, dirty = Gate.git_state()
        binary = args.binary if args.mechanism == "temperature" else None
        return {
            "mechanism": args.mechanism,
            "manifest_digest": document["digest"],
            "role": document["role"],
            "order_seed": args.order_seed,
            "profiles": args.profiles.split(","),
            "budget": {"seconds": BUDGET[0], "evaluations": BUDGET[1]},
            "max_hours": args.max_hours,
            "planned": planned,
            "completed": completed,
            "git_commit": commit,
            "git_dirty": dirty,
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest() if binary else None,
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
            "python": sys.version.split()[0],
        }

    @staticmethod
    def _analyze(args: argparse.Namespace) -> int:
        """Print the six registered class decisions.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            int: Zero after printing.

        Raises:
            ValueError: If a mechanism label is unknown or an instance has wrong seed counts.
            OSError: If a file cannot be read.
        """
        entries = UtilityCli._load(args.manifest)["entries"]
        profiles = tuple(args.profiles.split(","))
        claims: dict[str, list[bool]] = {name: [] for name in CLAIMS}
        for item in args.rows:
            mechanism, _, path = item.partition("=")
            if mechanism not in ("adaptive", "temperature"):
                raise ValueError(f"unknown mechanism {mechanism!r}")
            rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
            for kind, wins in UtilityStudy.class_wins(rows, entries, profiles).items():
                claims[f"{mechanism}:{kind}"] = wins
        status = UtilityAnalysis.decide(claims)
        for name in CLAIMS:
            sys.stdout.write(
                f"{name:26s} {status[name]:13s} wins {sum(claims[name])}/{len(claims[name])}\n"
            )
        sys.stdout.write(f"JSON {json.dumps(status)}\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(UtilityCli.main())
