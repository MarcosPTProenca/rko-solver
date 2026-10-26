"""Explicit command-line lifecycle for experimental decoder campaigns and offline replay.

Objective:
- Support the Pi artifact-exchange pilot without provider credentials or core changes.
Technical approach:
- Require explicit seed import and qualified rootless runtime for every execution command.
Flow (pseudocode):
1. Create an explicit example seed or load an existing artifact.
2. Freeze/calibrate a campaign, export a request and submit a reviewed modification.
3. Export experimental artifacts and replay them on new instance data without an LLM.
Important criteria:
- No automated API/harness provider or production deployment is exposed in this milestone.
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
from .problems import Problem
from .sandbox import PodmanRunner
from .study import ConfirmationStudy
from pathlib import Path


def main() -> None:
    """Dispatch an explicit pilot operation; never silently create a missing seed.

    Raises:
        ValueError: If user inputs or source-review preconditions are invalid.
        OSError: If supplied files cannot be read or output paths already exist.
        RuntimeError: If qualification or campaign budgets prevent execution.
    """
    parser = argparse.ArgumentParser(
        description="Experimental decoder evolution; no statistical claim"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    seed = commands.add_parser("example-seed", help="Explicitly export a bundled reference seed")
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
    report = commands.add_parser("report")
    report.add_argument("campaign", type=Path)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("campaign", type=Path)
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
    args = parser.parse_args()
    if args.command.startswith("study-"):
        if args.command == "study-create":
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
            study = ConfirmationStudy.create(args.output, **design)
            result = study.progress()
        else:
            study = ConfirmationStudy(args.study)
            if args.command == "study-run":
                if args.approve_spec != study.digest:
                    raise ValueError("Approve the exact frozen specification before runtime setup.")
                with study.runtime(approved_spec=args.approve_spec) as runner:
                    study.run(runner, approved_spec=args.approve_spec, max_jobs=args.max_jobs)
                result = study.progress()
            else:
                result = study.report() if args.command == "study-report" else study.progress()
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    if args.command == "example-seed":
        DecoderArtifact.create(args.family, Problem.seed_source(args.family)).export(args.output)
        return
    if args.command == "request":
        packet = Campaign(args.campaign).request(args.parent)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(packet, handle, indent=2)
        return
    if args.command == "export":
        Campaign(args.campaign).export(args.candidate, args.output)
        return
    if args.command == "report":
        campaign = Campaign(args.campaign)
        events = campaign.events()
        result = {
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
        }
    else:
        runner = PodmanRunner(args.image)
        if args.command == "create":
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
            result = {
                "campaign": str(campaign.path),
                "seed": artifact.digest,
                "image": runner.image,
                "status": "not_started",
            }
        else:
            qualification = runner.qualify()
            if args.command == "qualify":
                result = qualification
            elif args.command == "calibrate":
                campaign = Campaign(args.campaign)
                records = campaign.calibrate(runner)
                result = {"status": "calibrated", "records": len(records)}
            elif args.command == "submit":
                if args.source.stat().st_size > 32768:
                    raise ValueError("Candidate source exceeds 32 KiB.")
                result = Campaign(args.campaign).submit(
                    args.source.read_text(encoding="utf-8"),
                    args.hypothesis,
                    args.parent,
                    runner,
                    reviewed=args.reviewed_local_work,
                )
            else:
                artifact = DecoderArtifact.load(args.artifact)
                if args.instance.stat().st_size > 65536:
                    raise ValueError("Instance JSON exceeds 64 KiB.")
                data = json.loads(args.instance.read_text(encoding="utf-8"))
                result = runner.run(
                    artifact,
                    Problem(data["family"], data["data"]),
                    seconds=args.seconds,
                    seed=args.seed,
                )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
