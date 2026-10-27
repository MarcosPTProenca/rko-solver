"""Explicit command-line lifecycle for experimental decoder campaigns and offline replay.

Objective:
- Support the Pi artifact-exchange pilot without provider credentials or core changes.
Technical approach:
- Require explicit seed import and qualified rootless runtime for execution commands.
- Separate argument construction, offline operations and qualified execution phases.
Flow (pseudocode):
1. Parse the explicit operation and its required paths or approval.
2. Create or load frozen campaign/study inputs before qualifying execution.
3. Emit JSON evidence or exclusively create the requested artifact.
Important criteria:
- No automated API/harness provider or production deployment is exposed.
- Missing seeds are never silently created; study approval precedes runtime setup.
Inputs/Outputs:
- Input: explicit local paths and supported subcommands.
- Output: bounded JSON evidence and non-overwriting artifacts.
"""

from __future__ import annotations

import argparse
import json
from .artifacts import FAMILIES
from .artifacts import DecoderArtifact
from .campaign import Campaign
from .pi_commands import PI_COMMANDS
from .pi_commands import PiCommands
from .problems import Problem
from .proposals import PiCampaign
from .sandbox import PodmanRunner
from .study import ConfirmationStudy
from pathlib import Path
from typing import Any


class _CommandLine:
    """Coordinate explicit offline and qualified execution command phases.

    Responsibilities:
    - Preserve command arguments, file-size bounds and exclusive artifact writes.
    - Require qualification and exact study approval before execution.

    Attributes:
        args (argparse.Namespace): Parsed operation and command-specific settings.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Retain parsed operation settings.

        Args:
            args (argparse.Namespace): Validated command-line syntax.
        """
        self.args = args

    @staticmethod
    def parser() -> argparse.ArgumentParser:
        """Construct campaign, replay and study commands with unchanged defaults.

        Returns:
            argparse.ArgumentParser: Complete command parser.
        """
        parser = argparse.ArgumentParser(
            description="Experimental decoder evolution; no statistical claim"
        )
        commands = parser.add_subparsers(dest="command", required=True)
        PiCommands.add(commands)
        seed = commands.add_parser(
            "example-seed", help="Explicitly export a bundled reference seed"
        )
        seed.add_argument("family", choices=FAMILIES)
        seed.add_argument("output", type=Path)
        create = commands.add_parser("create")
        create.add_argument("seed_artifact", type=Path)
        create.add_argument("output", type=Path)
        create.add_argument("--instance-seed", type=int, default=1701)
        request = commands.add_parser("request")
        request.add_argument("campaign", type=Path)
        request.add_argument("output", type=Path)
        request.add_argument("--parent")
        submit = commands.add_parser("submit")
        submit.add_argument("campaign", type=Path)
        submit.add_argument("source", type=Path)
        submit.add_argument("--parent", required=True)
        submit.add_argument("--hypothesis", required=True)
        submit.add_argument("--reviewed-local-work", action="store_true")
        export = commands.add_parser("export")
        export.add_argument("campaign", type=Path)
        export.add_argument("candidate")
        export.add_argument("output", type=Path)
        commands.add_parser("report").add_argument("campaign", type=Path)
        commands.add_parser("calibrate").add_argument("campaign", type=Path)
        replay = commands.add_parser("replay")
        replay.add_argument("artifact", type=Path)
        replay.add_argument("instance", type=Path, help="JSON object with family and data fields")
        replay.add_argument("--seconds", type=float, default=0.1)
        replay.add_argument("--seed", type=int, default=0)
        commands.add_parser("qualify")
        study_create = commands.add_parser("study-create")
        study_create.add_argument("design", type=Path)
        study_create.add_argument("output", type=Path)
        study_create.add_argument("--rebind-runtime", action="store_true")
        study_run = commands.add_parser("study-run")
        study_run.add_argument("study", type=Path)
        study_run.add_argument("--approve-spec", required=True)
        study_run.add_argument("--max-jobs", type=int)
        for name in ("study-progress", "study-report"):
            commands.add_parser(name).add_argument("study", type=Path)
        for name in ("create", "calibrate", "submit", "replay", "qualify"):
            commands.choices[name].add_argument("--image", default="localhost/rko-self-evolve:dev")
        return parser

    def study(self) -> dict[str, Any]:
        """Create, run or report a confirmation study with approval before runtime setup.

        Returns:
            dict[str, Any]: Study progress or report.

        Raises:
            ValueError: If the design exceeds its bound or approval does not match.
            OSError: If study inputs cannot be read or output cannot be created.
            RuntimeError: If study creation or execution fails.
        """
        args = self.args
        if args.command == "study-create":
            return self._create_study().progress()
        study = ConfirmationStudy(args.study)
        if args.command == "study-run":
            if args.approve_spec != study.digest:
                raise ValueError("Approve the exact frozen specification before runtime setup.")
            with study.runtime(approved_spec=args.approve_spec) as runner:
                study.run(runner, approved_spec=args.approve_spec, max_jobs=args.max_jobs)
            return study.progress()
        return study.report() if args.command == "study-report" else study.progress()

    def _create_study(self) -> ConfirmationStudy:
        """Load bounded design JSON and resolve artifacts relative to its directory.

        Returns:
            ConfirmationStudy: Newly created frozen study.

        Raises:
            ValueError: If the design exceeds 16 MiB or its contents are invalid.
            OSError: If design/artifact reads or study creation fail.
        """
        args = self.args
        if args.design.stat().st_size > 16777216:
            raise ValueError("Study design exceeds 16 MiB.")
        design = json.loads(args.design.read_text(encoding="utf-8"))
        design["artifacts"] = {
            family: {
                role: DecoderArtifact.load(args.design.parent / path)
                for role, path in roles.items()
            }
            for family, roles in design["artifacts"].items()
        }
        design["instances"] = {
            family: [Problem(family, data) for data in instances]
            for family, instances in design["instances"].items()
        }
        design["allow_runtime_rebind"] = args.rebind_runtime
        return ConfirmationStudy.create(args.output, **design)

    def offline(self) -> bool:
        """Perform artifact-only operations without creating a runtime.

        Returns:
            bool: Whether an artifact-only command was handled without JSON output.

        Raises:
            ValueError: If artifacts or campaign inputs are invalid.
            OSError: If reads fail or an exclusive output already exists.
        """
        args = self.args
        if args.command == "example-seed":
            DecoderArtifact.create(args.family, Problem.seed_source(args.family)).export(
                args.output
            )
        elif args.command == "request":
            packet = Campaign(args.campaign).request(args.parent)
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(packet, handle, indent=2)
        elif args.command == "export":
            Campaign(args.campaign).export(args.candidate, args.output)
        else:
            return False
        return True

    def report(self) -> dict[str, Any]:
        """Summarize campaign events without creating an execution runtime.

        Returns:
            dict[str, Any]: Experimental campaign ledger summary.

        Raises:
            ValueError: If campaign metadata or events are invalid.
            OSError: If campaign files cannot be read.
        """
        campaign = PiCampaign(self.args.campaign)
        events = campaign.events()
        return {
            "family": campaign.spec["family"],
            "status": "experimental_development_only",
            "statistical_claim": None,
            "submissions": sum(e["kind"] == "submission" for e in events),
            "decisions": [e for e in events if e["kind"] == "decision"],
            "charged_cpu_upper_bounds": {
                p: sum(e.get("charge", 0) for e in events if e.get("partition") == p)
                for p in ("baseline", "candidate")
            },
            "evaluations": [e for e in events if e["kind"] == "evaluation"],
            "proposals": campaign.proposals(),
        }

    def _create_campaign(self, runner: PodmanRunner) -> dict[str, Any]:
        """Freeze generated campaign instances against the inspected runtime image.

        Args:
            runner (PodmanRunner): Inspected runtime, not yet live-qualified.

        Returns:
            dict[str, Any]: Newly frozen, not-started campaign identity.

        Raises:
            ValueError: If seed or generated campaign inputs are invalid.
            OSError: If seed reads or campaign creation fail.
        """
        args = self.args
        artifact = DecoderArtifact.load(args.seed_artifact)
        sizes = {
            "knapsack": [24, 40, 60],
            "scheduling": [8, 12, 14],
            "set_cover": [18, 24, 30],
        }[artifact.family]
        problems = [
            Problem.generate(artifact.family, args.instance_seed + i, size)
            for i, size in enumerate(sizes)
        ]
        campaign = Campaign.create(args.output, artifact, problems, runner.image)
        return {
            "campaign": str(campaign.path),
            "seed": artifact.digest,
            "image": runner.image,
            "status": "not_started",
        }

    def runtime(self) -> dict[str, Any]:
        """Inspect the runtime and qualify it before executing any requested job.

        Returns:
            dict[str, Any]: Qualification, calibration, submission or replay evidence.

        Raises:
            ValueError: If command-specific inputs are invalid or oversized.
            OSError: If source or instance reads fail.
            RuntimeError: If runtime qualification or execution fails.
        """
        args = self.args
        if args.command == "replay":
            return self._replay()
        runner = PodmanRunner(args.image)
        if args.command == "create":
            return self._create_campaign(runner)
        qualification = runner.qualify()
        if args.command == "qualify":
            return qualification
        if args.command == "calibrate":
            records = Campaign(args.campaign).calibrate(runner)
            return {"status": "calibrated", "records": len(records)}
        if args.command == "submit":
            if args.source.stat().st_size > 32768:
                raise ValueError("Candidate source exceeds 32 KiB.")
            return Campaign(args.campaign).submit(
                args.source.read_text(encoding="utf-8"),
                args.hypothesis,
                args.parent,
                runner,
                reviewed=args.reviewed_local_work,
            )
        raise ValueError("Unsupported runtime command.")

    def _replay(self) -> dict[str, Any]:
        """Reject pending/unusable bundles and inputs before constructing a replay runtime.

        Returns:
            dict[str, Any]: Qualified independently verified replay evidence.

        Raises:
            ValueError: If source status, instance size or contract is invalid.
            OSError: If files are missing or unreadable.
            RuntimeError: If qualification or isolated execution fails.
        """
        args = self.args
        artifact = DecoderArtifact.load(args.artifact)
        if args.instance.stat().st_size > 65536:
            raise ValueError("Instance JSON exceeds 64 KiB.")
        data = json.loads(args.instance.read_text(encoding="utf-8"))
        problem = Problem.restore(data["family"], data["data"])
        runner = PodmanRunner(args.image)
        runner.qualify()
        return runner.run(artifact, problem, seconds=args.seconds, seed=args.seed)

    def dispatch(self) -> None:
        """Dispatch the parsed operation and print JSON only for evidence commands.

        Raises:
            ValueError: If operation inputs or approval are invalid.
            OSError: If input reads or exclusive writes fail.
            RuntimeError: If qualification, execution or campaign budgets fail.
        """
        if self.args.command in PI_COMMANDS:
            result = PiCommands(self.args).dispatch()
        elif self.args.command.startswith("study-"):
            result = self.study()
        elif self.offline():
            return
        elif self.args.command == "report":
            result = self.report()
        else:
            result = self.runtime()
        print(json.dumps(result, indent=2, allow_nan=False))


def main() -> None:
    """Parse and dispatch an explicit pilot operation without creating implicit seeds.

    Raises:
        SystemExit: If command-line syntax is invalid.
        ValueError: If user inputs or source-review preconditions are invalid.
        OSError: If files cannot be read or output paths already exist.
        RuntimeError: If qualification or campaign budgets prevent execution.
    """
    _CommandLine(_CommandLine.parser().parse_args()).dispatch()


if __name__ == "__main__":
    main()
