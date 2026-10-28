"""Opt-in complete model proposal/review/isolation/export/unseen-replay control.

Objective:
- Exercise the actual shared lifecycle without a live proposer or scientific campaign.
Technical approach:
- A clearly synthetic equality-only fixture has a direct bounded algebraic constructor.
- Freeze repair='none' uniformly as a numerical-policy control, not per-case tuning.
Flow (pseudocode):
1. Calibrate the explicitly exported existing mapper on original owned equality records.
2. Receive static fixture source through credential-free scripted Pi and exact review.
3. Evaluate only in qualified isolation, export a strict complete winner and replay unseen size.
Important criteria:
- This is a software integration fixture, not optimizer-value or provider qualification.
Inputs/Outputs:
- Input: explicit test worker image and scripted fake Pi profile.
- Output: real independently checked model job evidence through the shared lifecycle.
"""

import os
import pytest
from pathlib import Path
from rko_solver.experimental.self_evolve.artifacts import DecoderArtifact
from rko_solver.experimental.self_evolve.campaign import Campaign
from rko_solver.experimental.self_evolve.model_data import ModelProblem
from rko_solver.experimental.self_evolve.model_data import ModelReference
from rko_solver.experimental.self_evolve.pi_config import PiConfig
from rko_solver.experimental.self_evolve.proposals import PiCampaign
from rko_solver.experimental.self_evolve.sandbox import PodmanRunner
from rko_solver.modeling.records import LinearRow
from rko_solver.modeling.records import LinearVariable
from rko_solver.modeling.source import ExtractedLinearModel
from rko_solver.tests.experimental.test_pi_transport import configured_pi as configured_pi
from rko_solver.tests.experimental.test_pi_transport import set_scenario

SOURCE = """def decode(keys, instance):
    values = [domain.lower for domain in instance.domains]
    for row in instance.rows:
        if row.lower == row.upper and len(row.coefficients) == 1:
            index, coefficient = row.coefficients[0]
            values[index] = (row.lower - row.constant) / coefficient
    return values
"""


def _problem(size: int, offset: int) -> ModelProblem:
    """Build a transparent controlled equality system with an independent closed-form reference.

    Args:
        size (int): Original variable/key dimension, including unseen replay sizes.
        offset (int): Distinct original affine objective constant.

    Returns:
        ModelProblem: Bounded original source with uniform explicit no-restoration policy.
    """
    source = ExtractedLinearModel(
        "pyomo",
        "owned-lifecycle-control",
        f"private-control-{size}-{offset}",
        tuple(LinearVariable(f"x{i}", f"x{i}", 0.0, 1.0, "continuous") for i in range(size)),
        (1.0,) * size,
        float(offset),
        False,
        tuple(LinearRow(f"equality{i}", ((i, float(7 + i)),), 1.0, 1.0) for i in range(size)),
    )
    reference = sum(1.0 / (7 + i) for i in range(size)) + offset
    return ModelProblem.from_source(
        source,
        policy={"repair": "none"},
        reference=ModelReference(
            reference,
            "independent closed form of original one-coordinate equalities; no native solver",
        ),
    )


def test_model_review_to_rootless_export_and_unseen_offline_replay(
    tmp_path: Path, configured_pi: PiConfig
) -> None:
    """A static synthetic proposal needs exact review before real isolated evaluation/export.

    Args:
        tmp_path (Path): Temporary software-test campaign and artifact root.
        configured_pi (PiConfig): Credential-free scripted process, never a live model/account.
    """
    image = os.environ.get("RKO_EVOLVE_IMAGE")
    if not image:
        pytest.skip("Set RKO_EVOLVE_IMAGE for the actual isolated model lifecycle test.")
    runner = PodmanRunner(image)
    runner.qualify()
    original = Campaign.create(
        tmp_path / "campaign",
        ModelProblem.seed_artifact(),
        [_problem(6, i) for i in range(3)],
        runner.image,
    )
    original.calibrate(runner)
    set_scenario(
        configured_pi,
        {
            "candidate": {
                "source": SOURCE,
                "hypothesis": "fixture: bounded direct algebra for equality rows",
            }
        },
    )
    campaign = PiCampaign(original.path)
    proposal = campaign.generate(configured_pi, consent=True)
    assert proposal["status"] == "pending_review"
    with pytest.raises(ValueError, match="review"):
        campaign.approve(1, proposal["candidate"], runner, reviewed=False)
    decision = campaign.approve(1, proposal["candidate"], runner, reviewed=True)
    assert decision["status"] == "accepted", decision
    assert campaign.spec["seconds"] == 1.0
    measured = [event for event in campaign.events() if event["kind"] == "evaluation"]
    assert all(event["finalization_allowance_seconds"] == 0.05 for event in measured)
    assert all(event["wall_seconds"] <= 1.02 for event in measured)
    assert sum(event["kind"] == "submission" for event in campaign.events()) == 1
    campaign.export(proposal["candidate"], tmp_path / "export")
    artifact = DecoderArtifact.load(tmp_path / "export")
    assert artifact.source == SOURCE
    unseen = _problem(8, 5)
    record = runner.run(artifact, unseen, mode="check", seed=29, seconds=0.2)
    assert unseen.dimension == 8 and record["feasible"] is True
    assert record["objective"] == pytest.approx(unseen.optimum(), rel=1e-12, abs=1e-12)
    assert record["model_status"] == "feasible"
