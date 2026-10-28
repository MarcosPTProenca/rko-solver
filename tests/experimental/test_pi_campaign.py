"""Review-gated campaign generation through the real fake-Pi process seam.

Objective:
- Preserve exact human approval and one counted attempt across durable proposal states.
Technical approach:
- Use real campaign files and Pi transport with trusted external evaluation records.
Flow (pseudocode):
1. Calibrate a tiny campaign and reserve/generate one source-only proposal.
2. Reopen pending review and reject new submissions or wrong source identities.
3. Explicitly approve once and verify that evaluation does not consume a second slot.
Important criteria:
- No candidate source, live authentication or Podman execution is performed.
Inputs/Outputs:
- Input: temporary campaigns, fake Pi and trusted runner evidence.
- Output: durable proposal states, exact hashes and failure-inclusive submission counts.
"""

import json
import pytest
import sys
from dataclasses import replace
from pathlib import Path
from rko_solver.experimental.self_evolve import DecoderArtifact
from rko_solver.experimental.self_evolve.pi_config import PiConfig
from rko_solver.experimental.self_evolve.pi_config import PiError
from rko_solver.experimental.self_evolve.proposals import PiCampaign
from tests.experimental.test_pi_transport import configured_pi as configured_pi
from tests.experimental.test_pi_transport import set_scenario
from tests.experimental.test_self_evolve_campaign import FixtureRunner
from tests.experimental.test_self_evolve_campaign import make_campaign


@pytest.mark.skipif(sys.platform != "linux", reason="Initial Pi subprocess support is Linux/WSL.")
def test_proposal_requires_exact_human_review_and_uses_one_slot(
    tmp_path: Path, configured_pi: PiConfig
) -> None:
    """Generation and approval share one durable attempt without trusting modified bytes.

    Args:
        tmp_path (Path): Private campaign root separate from its authentication profile.
        configured_pi (PiConfig): Trusted fake Pi executable and test-owned profile.
    """
    original = make_campaign(tmp_path / "campaign")
    original.calibrate(FixtureRunner())
    campaign = PiCampaign(original.path)
    result = campaign.generate(configured_pi, consent=True)
    assert result["status"] == "pending_review"
    assert result["attempt"] == 1
    with pytest.raises(ValueError, match="schema/status"):
        DecoderArtifact.load(Path(result["source_path"]).parent)
    reopened = PiCampaign(campaign.path)
    assert reopened.proposal(1)["candidate"] == result["candidate"]
    assert sum(e["kind"] == "submission" for e in reopened.events()) == 1
    with pytest.raises(ValueError, match="Unresolved"):
        original.request()
    with pytest.raises(ValueError, match="review"):
        reopened.approve(1, result["candidate"], FixtureRunner(), reviewed=False)
    with pytest.raises(ValueError, match="hash"):
        reopened.approve(1, "0" * 64, FixtureRunner(), reviewed=True)
    decision = reopened.approve(1, result["candidate"], FixtureRunner(), reviewed=True)
    assert decision["status"] == "accepted"
    assert PiCampaign(campaign.path).proposal(1)["status"] == "accepted"
    assert PiCampaign(campaign.path).proposal(1)["reviewed"] is True
    assert sum(e["kind"] == "submission" for e in reopened.events()) == 1
    with pytest.raises(ValueError, match="pending"):
        reopened.approve(1, result["candidate"], FixtureRunner(), reviewed=True)


@pytest.mark.skipif(sys.platform != "linux", reason="Initial Pi subprocess support is Linux/WSL.")
def test_failed_generations_count_and_tenth_attempt_can_still_be_reviewed(
    tmp_path: Path, configured_pi: PiConfig
) -> None:
    """Known auth failures are retained; approval does not miscount the final reserved slot.

    Args:
        tmp_path (Path): Private campaign root.
        configured_pi (PiConfig): Trusted external fake Pi response seam.
    """
    original = make_campaign(tmp_path / "campaign")
    original.calibrate(FixtureRunner())
    campaign = PiCampaign(original.path)
    set_scenario(configured_pi, {"auth": {"status": "not_ready", "provider": "openai-codex"}})
    for attempt in range(1, 10):
        with pytest.raises(PiError):
            campaign.generate(configured_pi, consent=True)
        assert PiCampaign(campaign.path).proposal(attempt)["status"] == "generation_failed"
    set_scenario(
        configured_pi,
        {"auth": {"status": "ready", "provider": "openai-codex", "authType": "oauth"}},
    )
    result = campaign.generate(configured_pi, consent=True)
    assert result["attempt"] == 10
    assert (
        campaign.approve(10, result["candidate"], FixtureRunner(), reviewed=True)["status"]
        == "accepted"
    )
    assert sum(e["kind"] == "submission" for e in campaign.events()) == 10
    with pytest.raises(RuntimeError, match="count"):
        campaign.generate(configured_pi, consent=True)


@pytest.mark.skipif(sys.platform != "linux", reason="Initial Pi subprocess support is Linux/WSL.")
def test_interrupted_generation_requires_explicit_inspected_abandonment(
    tmp_path: Path, configured_pi: PiConfig
) -> None:
    """A signal interruption keeps its slot and cannot be silently resumed as success.

    Args:
        tmp_path (Path): Private durable journal.
        configured_pi (PiConfig): Fake Pi signaling only its owned test-controller parent.
    """
    original = make_campaign(tmp_path / "campaign")
    original.calibrate(FixtureRunner())
    campaign = PiCampaign(original.path)
    set_scenario(configured_pi, {"interrupt_parent": True})
    with pytest.raises(KeyboardInterrupt):
        campaign.generate(configured_pi, consent=True)
    reopened = PiCampaign(campaign.path)
    assert reopened.proposal(1)["status"] == "generation_unknown"
    with pytest.raises(ValueError, match="Unresolved"):
        reopened.generate(configured_pi, consent=True)
    with pytest.raises(ValueError, match="Inspect"):
        reopened.reject(1, "Inspected interruption")
    reopened.reject(1, "Inspected stopped owned processes", confirmed_stopped=True)
    assert reopened.proposal(1)["status"] == "abandoned_unknown"
    set_scenario(configured_pi, {"interrupt_parent": False})
    assert reopened.generate(configured_pi, consent=True)["attempt"] == 2
    assert sum(e["kind"] == "submission" for e in reopened.events()) == 2


@pytest.mark.skipif(sys.platform != "linux", reason="Initial Pi subprocess support is Linux/WSL.")
def test_consent_frozen_settings_and_changed_source_fail_before_evaluation(
    tmp_path: Path, configured_pi: PiConfig
) -> None:
    """Consent, immutable settings and the reviewed source hash remain separate gates.

    Args:
        tmp_path (Path): Private campaign and modified review bundle.
        configured_pi (PiConfig): Trusted fake Pi settings before any proposal.
    """
    original = make_campaign(tmp_path / "campaign")
    original.calibrate(FixtureRunner())
    campaign = PiCampaign(original.path)
    with pytest.raises(ValueError, match="consent"):
        campaign.generate(configured_pi)
    assert campaign.proposals() == []
    result = campaign.generate(configured_pi, consent=True)
    source = Path(result["source_path"])
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash"):
        campaign.approve(1, result["candidate"], FixtureRunner(), reviewed=True)
    campaign.reject(1, "Modified source is not approved")
    with pytest.raises(ValueError, match="settings changed"):
        campaign.generate(replace(configured_pi, seconds=20), consent=True)
    assert len(campaign.proposals()) == 1
    records = json.dumps(campaign.events())
    assert '"monetary_cost": null' in records


@pytest.mark.skipif(sys.platform != "linux", reason="Initial Pi subprocess support is Linux/WSL.")
@pytest.mark.parametrize("change", ["missing", "substituted"])
def test_frozen_proposer_cannot_be_recreated_or_retargeted_after_a_failed_attempt(
    tmp_path: Path, configured_pi: PiConfig, change: str
) -> None:
    """Past submission identities stay authoritative even when the settings file is damaged.

    Args:
        tmp_path (Path): Private campaign with one recorded failed generation.
        configured_pi (PiConfig): Original qualified declaration for the trusted fake process.
        change (str): Delete the declaration or replace it with a different valid declaration.
    """
    original = make_campaign(tmp_path / "campaign")
    original.calibrate(FixtureRunner())
    campaign = PiCampaign(original.path)
    set_scenario(configured_pi, {"version": "unsupported"})
    with pytest.raises(PiError):
        campaign.generate(configured_pi, consent=True)
    changed = replace(configured_pi, seconds=configured_pi.seconds + 1)
    path = campaign.path / "proposer.json"
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(json.dumps(changed.descriptor(), sort_keys=True, allow_nan=False).encode())
    set_scenario(configured_pi, {"version": "1.3.0"})
    with pytest.raises(ValueError, match="proposer"):
        campaign.generate(changed, consent=True)
    assert len(campaign.proposals()) == 1
