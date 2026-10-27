"""Public CLI proposal and exact human approval contracts.

Objective:
- Keep Pi proposal generation separate from qualified decoder execution.
Technical approach:
- Drive real argv parsing with a trusted fake Pi and external evaluation fixture.
Flow (pseudocode):
1. Generate and inspect pending source through the public commands.
2. Reject missing attestation or a changed hash before constructing an evaluation runtime.
3. Approve once and retain the same counted campaign submission.
Important criteria:
- No provider request or candidate-source execution occurs in this suite.
Inputs/Outputs:
- Input: temporary campaigns, explicit non-secret configuration and CLI arguments.
- Output: JSON proposal states and honest review-gated experimental decisions.
"""

import json
import pytest
import sys
from pathlib import Path
from rko_solver.experimental.self_evolve.campaign import BudgetExpired
from rko_solver.experimental.self_evolve.pi_config import PiConfig
from rko_solver.tests.experimental.test_pi_transport import configured_pi as configured_pi
from rko_solver.tests.experimental.test_self_evolve_campaign import FixtureRunner
from rko_solver.tests.experimental.test_self_evolve_campaign import make_campaign
from rko_solver.tests.experimental.test_self_evolve_cli import CommandRunner
from rko_solver.tests.experimental.test_self_evolve_cli import invoke


@pytest.mark.skipif(sys.platform != "linux", reason="Initial Pi process supervision is Linux/WSL.")
def test_public_generation_inspection_and_review_are_separate(
    tmp_path: Path,
    configured_pi: PiConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No evaluation runtime is constructed until exact source approval and attestation.

    Args:
        tmp_path (Path): Private campaign/configuration files.
        configured_pi (PiConfig): Trusted fake executable and isolated profile.
        monkeypatch (pytest.MonkeyPatch): External evaluation-runtime constructor seam.
        capsys (pytest.CaptureFixture[str]): Public JSON output capture.
    """
    campaign = make_campaign(tmp_path / "campaign")
    campaign.calibrate(FixtureRunner())
    config_path = tmp_path / "pi.json"
    config_path.write_text(
        json.dumps(
            {
                "executable": str(configured_pi.executable),
                "profile": str(configured_pi.profile),
                "provider": configured_pi.provider,
                "model": configured_pi.model,
                "seconds": configured_pi.seconds,
            }
        )
    )
    invoke(monkeypatch, "propose", campaign.path, config_path, "--consent-development-data")
    proposal = json.loads(capsys.readouterr().out)
    assert proposal["status"] == "pending_review"
    invoke(monkeypatch, "proposal", campaign.path, "1")
    assert json.loads(capsys.readouterr().out)["candidate"] == proposal["candidate"]
    monkeypatch.setattr(
        "rko_solver.experimental.self_evolve.pi_commands.PodmanRunner",
        lambda *args: pytest.fail("Unreviewed evaluation runtime."),
    )
    monkeypatch.setattr(
        "rko_solver.experimental.self_evolve.__main__.PodmanRunner",
        lambda *args: pytest.fail("Pending source must fail before replay runtime."),
    )
    with pytest.raises(ValueError, match="schema/status"):
        invoke(
            monkeypatch, "replay", Path(proposal["source_path"]).parent, tmp_path / "unused.json"
        )
    with pytest.raises(ValueError, match="review"):
        invoke(
            monkeypatch, "approve-proposal", campaign.path, "1", "--sha256", proposal["candidate"]
        )
    with pytest.raises(ValueError, match="hash"):
        invoke(
            monkeypatch,
            "approve-proposal",
            campaign.path,
            "1",
            "--sha256",
            "0" * 64,
            "--reviewed-local-work",
        )
    started = next(e["at"] for e in campaign.events() if e["kind"] == "start")
    with monkeypatch.context() as clock:
        clock.setattr(
            "rko_solver.experimental.self_evolve.campaign.time.time", lambda: started + 5401
        )
        with pytest.raises(BudgetExpired, match="elapsed"):
            invoke(
                monkeypatch,
                "approve-proposal",
                campaign.path,
                "1",
                "--sha256",
                proposal["candidate"],
                "--reviewed-local-work",
            )
    monkeypatch.setattr(
        "rko_solver.experimental.self_evolve.pi_commands.PodmanRunner",
        lambda image: CommandRunner(),
    )
    invoke(
        monkeypatch,
        "approve-proposal",
        campaign.path,
        "1",
        "--sha256",
        proposal["candidate"],
        "--reviewed-local-work",
    )
    assert json.loads(capsys.readouterr().out)["status"] == "accepted"
    invoke(monkeypatch, "report", campaign.path)
    report = json.loads(capsys.readouterr().out)
    assert report["submissions"] == 1
    assert report["proposals"][0]["status"] == "accepted"


def test_profile_command_creates_settings_without_starting_or_authenticating_pi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Profile setup works without an installed executable or any supplied credential.

    Args:
        tmp_path (Path): Non-secret declaration and brand-new private profile.
        monkeypatch (pytest.MonkeyPatch): Public CLI invocation boundary.
        capsys (pytest.CaptureFixture[str]): Capture only safe setup metadata.
    """
    configuration = tmp_path / "pi.json"
    profile = tmp_path / "new-profile"
    configuration.write_text(
        json.dumps(
            {
                "executable": str(tmp_path / "not-installed"),
                "profile": str(profile),
                "provider": "openai-codex",
                "model": "gpt-fixture",
                "seconds": 10,
            }
        )
    )
    invoke(monkeypatch, "pi-profile", configuration)
    result = json.loads(capsys.readouterr().out)
    assert result["authentication"] == "login_with_pi_required"
    assert result["profile"] == str(profile)
    assert (profile / "settings.json").is_file()
    assert not (profile / "auth.json").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Initial Pi subprocess support is Linux/WSL.")
def test_public_rejection_keeps_the_reserved_slot_without_asserting_review(
    tmp_path: Path,
    configured_pi: PiConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The documented positional declaration and stopped-attestation flag parse correctly.

    Args:
        tmp_path (Path): Private campaign and declaration.
        configured_pi (PiConfig): Trusted fixture executable; no real provider is called.
        monkeypatch (pytest.MonkeyPatch): Public CLI argument boundary.
        capsys (pytest.CaptureFixture[str]): Safe public command output.
    """
    campaign = make_campaign(tmp_path / "campaign")
    campaign.calibrate(FixtureRunner())
    configuration = tmp_path / "pi.json"
    configuration.write_text(
        json.dumps(
            {
                "executable": str(configured_pi.executable),
                "profile": str(configured_pi.profile),
                "provider": configured_pi.provider,
                "model": configured_pi.model,
                "seconds": configured_pi.seconds,
            }
        )
    )
    invoke(monkeypatch, "propose", campaign.path, configuration, "--consent-development-data")
    assert json.loads(capsys.readouterr().out)["status"] == "pending_review"
    invoke(
        monkeypatch,
        "reject-proposal",
        campaign.path,
        "1",
        "--reason",
        "declined",
        "--confirmed-stopped",
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "rejected"
    assert result["reviewed"] is False
    invoke(monkeypatch, "report", campaign.path)
    report = json.loads(capsys.readouterr().out)
    assert report["submissions"] == 1
    assert report["proposals"][0]["status"] == "rejected"
