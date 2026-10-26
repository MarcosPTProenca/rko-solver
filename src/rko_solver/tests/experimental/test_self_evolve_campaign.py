"""Public campaign lifecycle tests with a deterministic external-runner seam.

Objective:
- Validate seed-backed campaign lifecycle, proposal limits and portable lineage.
Technical approach:
- Supply deterministic trusted runner records without executing candidate source.
Flow (pseudocode):
1. Create small seed-backed campaigns and require baseline calibration.
2. Submit invalid or reviewed modifications and reopen persisted campaigns.
3. Export artifacts and reject tampered specifications or unknown parents.
Important criteria:
- Rejected proposals consume slots; exports preserve lineage without provider access.
Inputs/Outputs:
- Input: temporary campaign paths, synthetic problems and reviewed source descriptors.
- Output: deterministic evidence and assertions on lifecycle guards and artifact integrity.
"""

import pytest
from pathlib import Path
from rko_solver.experimental.self_evolve import DecoderArtifact
from rko_solver.experimental.self_evolve import Problem
from rko_solver.experimental.self_evolve.campaign import BudgetExpired
from rko_solver.experimental.self_evolve.campaign import Campaign
from typing import Any


class FixtureRunner:
    """Supply deterministic trusted worker records without executing candidate source.

    Responsibilities:
    - Exercise real collectors with controlled independently valid witness records.

    Attributes:
        image (str): Immutable synthetic runtime identity.
        qualified (bool): Trusted fixture capability state.
    """

    image = "sha256:" + "a" * 64
    qualified = True

    def run(
        self, artifact: DecoderArtifact, problem: Problem, *, mode: str, seed: int, seconds: float
    ) -> dict[str, Any]:
        """Return a known feasible witness independently checked by the problem contract.

        Args:
            artifact (DecoderArtifact): Unexecuted source descriptor.
            problem (Problem): Validated instance.
            mode (str): Frozen evaluation mode.
            seed (int): Frozen solver seed.
            seconds (float): Invocation allowance.

        Returns:
            dict[str, Any]: Deterministic evidence at the runner boundary.
        """
        solution = problem.constructive()
        return {
            "status": "ok",
            **problem.evaluate(solution),
            "solution": solution,
            "cpu_seconds": 0.1,
            "wall_seconds": seconds,
            "outer_wall_seconds": 0.2,
            "peak_memory_bytes": 1000,
            "evaluations": 1,
            "infeasible_evaluations": 0,
        }


def make_campaign(path: Path) -> Campaign:
    """Create a seed-backed campaign using deterministic small development instances.

    Args:
        path (Path): New campaign directory.

    Returns:
        Campaign: Not-yet-calibrated test campaign.
    """
    seed = DecoderArtifact.create("knapsack", Problem.seed_source("knapsack"))
    problems = [Problem.generate("knapsack", i, 6) for i in range(3)]
    return Campaign.create(path, seed, problems, FixtureRunner.image)


def test_campaign_requires_seed_and_calibration(tmp_path: Path) -> None:
    """Missing seed or missing baseline measurements cannot trigger from-scratch generation.

    Args:
        tmp_path (Path): Temporary parent for missing-seed and calibration test campaigns.
    """
    with pytest.raises(ValueError, match="root seed"):
        Campaign.create(tmp_path / "missing", None, [], FixtureRunner.image)
    campaign = make_campaign(tmp_path / "campaign")
    with pytest.raises(ValueError, match="Calibrate"):
        campaign.request()
    campaign.calibrate(FixtureRunner())
    request = Campaign(campaign.path).request()
    assert request["parent"] == campaign.spec["seed"]
    assert request["source"] == Problem.seed_source("knapsack")
    assert "development_instances" in request


def test_rejected_modifications_consume_slots_and_survive_resume(tmp_path: Path) -> None:
    """Invalid proposals are preserved and count toward the hard ten-attempt ceiling.

    Args:
        tmp_path (Path): Temporary parent for the campaign reopened after rejected submissions.
    """
    campaign = make_campaign(tmp_path / "campaign")
    runner = FixtureRunner()
    campaign.calibrate(runner)
    for _ in range(10):
        result = campaign.submit(
            "import os", "invalid import", campaign.spec["seed"], runner, reviewed=True
        )
        assert result["status"] == "rejected"
    reopened = Campaign(campaign.path)
    assert sum(e["kind"] == "submission" for e in reopened.events()) == 10
    assert (campaign.path / "rejected-1.py").read_text() == "import os"
    with pytest.raises(BudgetExpired, match="count"):
        reopened.request()


def test_export_preserves_lineage_and_requires_no_provider(tmp_path: Path) -> None:
    """A shortlisted modification exports and loads independently of campaign/provider access.

    Args:
        tmp_path (Path): Temporary parent for the campaign and portable export directory.
    """
    campaign = make_campaign(tmp_path / "campaign")
    runner = FixtureRunner()
    campaign.calibrate(runner)
    source = 'def decode(keys, instance):\n    """A reviewed modification."""\n    return [0]\n'
    decision = campaign.submit(
        source, "simple mutation", campaign.spec["seed"], runner, reviewed=True
    )
    assert decision["status"] == "accepted"
    output = tmp_path / "export"
    campaign.export(decision["candidate"], output)
    loaded = DecoderArtifact.load(output)
    assert loaded.source == source
    assert loaded.parent == campaign.spec["seed"]
    assert (output / "evidence.json").exists()
    with pytest.raises(FileExistsError):
        campaign.export(decision["candidate"], output)


def test_tampered_spec_and_unknown_parents_are_rejected(tmp_path: Path) -> None:
    """Specification changes and external source hashes do not enter the lineage silently.

    Args:
        tmp_path (Path): Temporary parent for the campaign whose specification is tampered with.
    """
    campaign = make_campaign(tmp_path / "campaign")
    campaign.calibrate(FixtureRunner())
    with pytest.raises(ValueError, match="Parent"):
        campaign.request("b" * 64)
    spec = campaign.path / "campaign.json"
    spec.write_text(spec.read_text().replace('"max_submissions": 10', '"max_submissions": 100'))
    with pytest.raises(ValueError, match="hash"):
        Campaign(campaign.path)
