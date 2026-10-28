"""Offline and qualified command-line lifecycle contracts.

Objective:
- Exercise the public CLI without importing decoder source or requiring a container runtime.
Technical approach:
- Use real parsers, artifact files and journals with a trusted external-runner fixture.
Flow (pseudocode):
1. Create source-only fixtures through the ordinary offline commands.
2. Drive campaign and study commands against deterministic runner evidence.
3. Reject invalid approval and oversized inputs before candidate execution.
Important criteria:
- Runtime seams provide trusted records only; no generated source or research is executed.
Inputs/Outputs:
- Input: temporary files, explicit argv and deterministic runtime fixtures.
- Output: asserted JSON responses, exact source identities and admission failures.
"""

import contextlib
import json
import pytest
import sys
from collections.abc import Iterator
from pathlib import Path
from rko_solver.experimental.self_evolve import DecoderArtifact
from rko_solver.experimental.self_evolve import Problem
from rko_solver.experimental.self_evolve import __main__ as cli
from rko_solver.experimental.self_evolve.study import ConfirmationStudy
from tests.experimental.test_confirmation_study import create_study
from tests.experimental.test_self_evolve_campaign import FixtureRunner
from tests.experimental.test_self_evolve_campaign import make_campaign


def invoke(monkeypatch: pytest.MonkeyPatch, *arguments: str | Path) -> None:
    """Run the actual CLI entry point with explicit isolated arguments.

    Args:
        monkeypatch (pytest.MonkeyPatch): Scoped argv replacement.
        arguments (str | Path): Command and its explicit arguments.
    """
    monkeypatch.setattr(sys, "argv", ["self-evolve", *map(str, arguments)])
    cli.main()


class CommandRunner(FixtureRunner):
    """Expose trusted qualification evidence without starting Podman.

    Responsibilities:
    - Keep CLI lifecycle coverage independent of live isolation availability.

    Attributes:
        qualifications (int): Number of explicit qualification calls.
    """

    qualifications = 0

    def qualify(self) -> dict[str, str]:
        """Count qualification calls and return a clearly synthetic capability record.

        Returns:
            dict[str, str]: Fixture qualification identity.
        """
        self.qualifications += 1
        return {"image": self.image, "qualification": "trusted_test_fixture"}


@pytest.mark.parametrize("family", ["knapsack", "scheduling", "set_cover"])
def test_seed_command_is_offline_and_non_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    """Explicit seed export needs no runtime and never replaces existing artifacts.

    Args:
        tmp_path (Path): Private bundle destination.
        monkeypatch (pytest.MonkeyPatch): Runtime prohibition and explicit argv.
        family (str): Supported source-only seed contract.
    """
    monkeypatch.setattr(cli, "PodmanRunner", lambda *args: pytest.fail("Offline command."))
    output = tmp_path / "seed"
    invoke(monkeypatch, "example-seed", family, output)
    artifact = DecoderArtifact.load(output)
    assert artifact.source == Problem.seed_source(family)
    with pytest.raises(FileExistsError):
        invoke(monkeypatch, "example-seed", family, output)


def test_campaign_commands_preserve_lineage_and_failure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Calibrate, request, submit, report and export use the same durable campaign.

    Args:
        tmp_path (Path): Private campaign and exchange files.
        monkeypatch (pytest.MonkeyPatch): Trusted runtime constructor and argv.
        capsys (pytest.CaptureFixture[str]): Public JSON output capture.
    """
    campaign = make_campaign(tmp_path / "campaign")
    runner = CommandRunner()
    monkeypatch.setattr(cli, "PodmanRunner", lambda image: runner)
    invoke(monkeypatch, "calibrate", campaign.path)
    assert json.loads(capsys.readouterr().out)["status"] == "calibrated"
    request = tmp_path / "request.json"
    invoke(monkeypatch, "request", campaign.path, request)
    assert json.loads(request.read_text())["parent"] == campaign.spec["seed"]
    with pytest.raises(FileExistsError):
        invoke(monkeypatch, "request", campaign.path, request)
    source = tmp_path / "candidate.py"
    source.write_text(Problem.seed_source("knapsack") + "\n", encoding="utf-8")
    invoke(
        monkeypatch,
        "submit",
        campaign.path,
        source,
        "--parent",
        campaign.spec["seed"],
        "--hypothesis",
        "trusted fixture mutation",
        "--reviewed-local-work",
    )
    decision = json.loads(capsys.readouterr().out)
    assert decision["status"] == "accepted"
    invoke(monkeypatch, "report", campaign.path)
    report = json.loads(capsys.readouterr().out)
    assert report["submissions"] == 1
    assert report["statistical_claim"] is None
    assert report["charged_cpu_upper_bounds"]["candidate"] > 0
    output = tmp_path / "export"
    invoke(monkeypatch, "export", campaign.path, decision["candidate"], output)
    assert DecoderArtifact.load(output).parent == campaign.spec["seed"]
    assert runner.qualifications == 2


def test_create_inspects_but_does_not_qualify_or_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Campaign creation freezes data without spending a qualification or evaluation.

    Args:
        tmp_path (Path): Seed and new campaign destinations.
        monkeypatch (pytest.MonkeyPatch): Trusted inspected runtime and argv.
        capsys (pytest.CaptureFixture[str]): Public JSON output capture.
    """
    seed = tmp_path / "seed"
    DecoderArtifact.create("scheduling", Problem.seed_source("scheduling")).export(seed)
    runner = CommandRunner()
    monkeypatch.setattr(cli, "PodmanRunner", lambda image: runner)
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: pytest.fail("Creation executed."))
    invoke(monkeypatch, "create", seed, tmp_path / "campaign", "--instance-seed", "42")
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "not_started"
    assert result["image"] == runner.image
    assert runner.qualifications == 0


@pytest.mark.parametrize("command", ["qualify", "replay"])
def test_qualified_commands_return_explicit_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    """Replay loads source as data and returns the external runner's verified evidence.

    Args:
        tmp_path (Path): Artifact and instance files.
        monkeypatch (pytest.MonkeyPatch): Trusted runner and explicit argv.
        capsys (pytest.CaptureFixture[str]): Public JSON output capture.
        command (str): Qualification-only or replay command.
    """
    runner = CommandRunner()
    monkeypatch.setattr(cli, "PodmanRunner", lambda image: runner)
    if command == "qualify":
        invoke(monkeypatch, command)
        assert json.loads(capsys.readouterr().out)["qualification"] == "trusted_test_fixture"
    else:
        artifact = tmp_path / "artifact"
        DecoderArtifact.create("knapsack", Problem.seed_source("knapsack")).export(artifact)
        instance = tmp_path / "instance.json"
        problem = Problem.generate("knapsack", 12, 4)
        instance.write_text(json.dumps({"family": problem.family, "data": problem.data}))
        monkeypatch.setattr(
            runner,
            "run",
            lambda artifact, problem, **kwargs: FixtureRunner.run(
                runner, artifact, problem, mode="rko", **kwargs
            ),
        )
        invoke(monkeypatch, command, artifact, instance, "--seconds", "0.2", "--seed", "3")
        assert json.loads(capsys.readouterr().out)["solution"] == problem.constructive()
    assert runner.qualifications == 1


@pytest.mark.parametrize("command", ["submit", "replay", "study-create"])
def test_oversized_cli_inputs_never_reach_runner_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    """Per-command file bounds reject oversized source, instances and study designs.

    Args:
        tmp_path (Path): Deliberately oversized input file.
        monkeypatch (pytest.MonkeyPatch): Trusted inspection seam and prohibited execution.
        command (str): Command owning the input size boundary.
    """
    oversized = tmp_path / "oversized"
    oversized.write_bytes(b" " * (16777217 if command == "study-create" else 65537))
    runner = CommandRunner()
    monkeypatch.setattr(cli, "PodmanRunner", lambda image: runner)
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: pytest.fail("Oversized execution."))
    if command == "submit":
        arguments = [
            str(tmp_path / "campaign"),
            str(oversized),
            "--parent",
            "a" * 64,
            "--hypothesis",
            "oversized",
        ]
    elif command == "replay":
        artifact = tmp_path / "artifact"
        DecoderArtifact.create("knapsack", Problem.seed_source("knapsack")).export(artifact)
        arguments = [str(artifact), str(oversized)]
    else:
        arguments = [str(oversized), str(tmp_path / "study")]
    with pytest.raises(ValueError, match="exceeds"):
        invoke(monkeypatch, command, *arguments)


def test_study_design_creation_and_offline_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Design paths resolve relative to their file and reporting never starts a runtime.

    Args:
        tmp_path (Path): Reference source bundles, declaration and new study.
        monkeypatch (pytest.MonkeyPatch): Runtime prohibition and explicit argv.
        capsys (pytest.CaptureFixture[str]): Progress and incomplete report output.
    """
    original = create_study(tmp_path / "original")
    design = {
        key: original.spec[key]
        for key in (
            "image",
            "seeds",
            "seconds",
            "cpu_budget",
            "elapsed_budget",
            "minimum_effect",
            "prior_instance_hashes",
        )
    }
    design["artifacts"] = {
        family: {role: f"original/artifacts/{family}/{role}" for role in roles}
        for family, roles in original.spec["artifacts"].items()
    }
    design["instances"] = {
        family: [entry["data"] for entry in entries]
        for family, entries in original.spec["instances"].items()
    }
    path = tmp_path / "design.json"
    path.write_text(json.dumps(design))
    monkeypatch.setattr(cli, "PodmanRunner", lambda *args: pytest.fail("Offline study command."))
    output = tmp_path / "study"
    invoke(monkeypatch, "study-create", path, output, "--rebind-runtime")
    progress = json.loads(capsys.readouterr().out)
    assert progress["pending"] == 72
    invoke(monkeypatch, "study-progress", output)
    assert json.loads(capsys.readouterr().out) == progress
    invoke(monkeypatch, "study-report", output)
    assert json.loads(capsys.readouterr().out)["status"] == "incomplete_no_inference"


def test_study_run_requires_exact_approval_before_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An approved study chunk uses the lifecycle, while wrong approval does not enter it.

    Args:
        tmp_path (Path): Frozen study and durable measurements.
        monkeypatch (pytest.MonkeyPatch): Trusted lifecycle context and argv.
        capsys (pytest.CaptureFixture[str]): Chunk progress output.
    """
    study = create_study(tmp_path / "study")
    approvals = []

    @contextlib.contextmanager
    def runtime(controller: ConfirmationStudy, *, approved_spec: str) -> Iterator[FixtureRunner]:
        """Record exact approval and supply a trusted runner without runtime construction.

        Args:
            controller (ConfirmationStudy): Reopened frozen controller.
            approved_spec (str): Owner-approved digest forwarded by the CLI.

        Yields:
            FixtureRunner: Deterministic external-runner records.
        """
        assert controller.digest == approved_spec
        approvals.append(approved_spec)
        yield FixtureRunner()

    monkeypatch.setattr(ConfirmationStudy, "runtime", runtime)
    with pytest.raises(ValueError, match="exact"):
        invoke(monkeypatch, "study-run", study.path, "--approve-spec", "wrong")
    assert approvals == []
    invoke(monkeypatch, "study-run", study.path, "--approve-spec", study.digest, "--max-jobs", "1")
    assert json.loads(capsys.readouterr().out)["complete"] == 1
    assert approvals == [study.digest]
