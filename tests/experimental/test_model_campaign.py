"""Generic model campaign policy checks without providers or candidate host execution.

Objective:
- Reuse the proposal lifecycle while keeping model feasibility/references explicit.
Technical approach:
- Use original GAP-derived mathematical records and a clearly synthetic runner.
Flow (pseudocode):
1. Freeze three distinct owned models and trusted tiny-oracle references.
2. Exercise development-only requests, calibration and reviewed manual shortlist logic.
3. Check no hidden human baseline, optimum leakage or row-infeasibility reinterpretation.
Important criteria:
- These controller fakes do not qualify runtime isolation or optimizer value.
Inputs/Outputs:
- Input: static trusted source and original owned records.
- Output: shared lifecycle/policy assertions with no live generation.
"""

import dataclasses
import numpy as np
import pytest
import sys
from benchmarks.modeling_cases import PracticalModels
from benchmarks.self_evolve.gap import GeneralizedAssignment
from pathlib import Path
from rko_solver.experimental.self_evolve.artifacts import DecoderArtifact
from rko_solver.experimental.self_evolve.campaign import Campaign
from rko_solver.experimental.self_evolve.model_data import ModelProblem
from rko_solver.experimental.self_evolve.model_data import ModelReference
from rko_solver.experimental.self_evolve.pi_config import PiConfig
from rko_solver.experimental.self_evolve.proposals import PiCampaign
from rko_solver.modeling.adapters import ModelAdapter
from tests.experimental.test_pi_transport import configured_pi as configured_pi


class _ModelRunner:
    """Emulate bounded successful records solely for controller-policy unit tests.

    Responsibilities:
    - Report row-infeasible completed evaluations alongside a feasible retained witness.

    Attributes:
        image (str): Synthetic pinned image identity, not actual isolation evidence.
        qualified (bool): Synthetic admission state.
        modes (list[str]): Requested matched-control modes.
    """

    image = "sha256:" + "d" * 64
    qualified = True

    def __init__(self, *, improve: bool = True) -> None:
        """Start a fresh explicit synthetic comparison without executing source.

        Args:
            improve (bool): Whether fake child observations improve original reference loss.
        """
        self.modes: list[str] = []
        self.seconds: list[float] = []
        self.improve = improve

    def run(self, artifact: DecoderArtifact, problem: ModelProblem, **options: object) -> dict:
        """Return mathematical fake evidence, never interpreting artifact source.

        Args:
            artifact (object): Unused source identity.
            problem (ModelProblem): Protected original numerical model.
            options (object): Frozen cell coordinates.

        Returns:
            dict: Explicitly synthetic completed policy-test record.
        """
        self.modes.append(str(options["mode"]))
        self.seconds.append(float(options["seconds"]))
        values = (0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0)
        if artifact.parent is not None and self.improve:
            _, machines = GeneralizedAssignment.fixture().reference()
            values = tuple(int(machine == j) for machine in machines for j in range(3))
        point = problem.snapshot.evaluate_assignment(np.full(problem.dimension, 0.5), values)
        return {
            "status": "ok",
            "objective": point.objective,
            "score": point.score,
            "feasible": True,
            "evaluations": 5,
            "infeasible_evaluations": 1,
            "cpu_seconds": 0.01,
            "outer_wall_seconds": 0.01,
        }


def _models(*, references: bool = True) -> list[ModelProblem]:
    """Build distinct trusted mathematical records with independently enumerated references.

    Args:
        references (bool): Whether explicit parent-only trusted references are supplied.

    Returns:
        list[ModelProblem]: Three different objective-constant models.
    """
    pytest.importorskip("pyomo")
    fixture = GeneralizedAssignment.fixture()
    source = ModelAdapter.extract(PracticalModels.native(fixture.formulation(), "pyomo"))
    objective, _ = fixture.reference()
    return [
        ModelProblem.from_source(
            dataclasses.replace(source, name=f"private-fixture-{i}", objective_constant=float(i)),
            reference=ModelReference(objective + i, "private-tiny-oracle") if references else None,
        )
        for i in range(3)
    ]


def test_model_campaign_requires_explicit_references_before_creating_state(tmp_path: Path) -> None:
    """Unknown optima cannot silently become a fictitious automatic comparator.

    Args:
        tmp_path (Path): Isolated campaign directory parent.
    """
    with pytest.raises(ValueError, match="reference"):
        Campaign.create(
            tmp_path / "campaign",
            ModelProblem.seed_artifact(),
            _models(references=False),
            _ModelRunner.image,
        )
    assert not (tmp_path / "campaign").exists()


def test_model_campaign_request_excludes_reference_and_case_identity(tmp_path: Path) -> None:
    """The proposer receives original mathematics, not known references or source labels.

    Args:
        tmp_path (Path): Isolated campaign directory parent.
    """
    campaign = Campaign.create(
        tmp_path / "campaign", ModelProblem.seed_artifact(), _models(), _ModelRunner.image
    )
    runner = _ModelRunner()
    campaign.calibrate(runner)
    assert set(runner.seconds) == {1.0}
    request = campaign.request(campaign.spec["seed"])
    assert "original-order" in request["semantics"]
    assert "private-fixture" not in str(request["development_instances"])
    assert "private-tiny-oracle" not in str(request)
    assert "optimum" not in str(request["development_instances"])
    assert campaign.spec["feasibility_policy"] == "feasibility_first"
    assert campaign.spec["model_timing_contract"] == 2
    assert campaign.spec["seconds"] == 1.0
    assert request["timing_policy"] == "end_to_end" and request["job_seconds"] == 1.0


def test_model_campaign_explores_row_infeasible_outputs_without_fake_human_baseline(
    tmp_path: Path,
) -> None:
    """Only model campaign policy permits explored infeasibility; primary search stays feasible.

    Args:
        tmp_path (Path): Isolated campaign directory parent.
    """
    runner = _ModelRunner()
    seed = ModelProblem.seed_artifact()
    campaign = Campaign.create(tmp_path / "campaign", seed, _models(), runner.image)
    campaign.calibrate(runner)
    assert set(runner.modes) == {"check", "rko", "random", "fixed"}
    decision = campaign.submit(
        seed.source + "\n# reviewed synthetic controller test\n",
        "policy-only fixture",
        seed.digest,
        runner,
        reviewed=True,
    )
    assert decision["status"] == "accepted"
    assert sum(e["kind"] == "submission" for e in campaign.events()) == 1
    campaign.export(decision["candidate"], tmp_path / "export")
    exported = DecoderArtifact.load(tmp_path / "export")
    assert exported.family == "linear_model" and exported.metadata()["model_contract"] == 1


@pytest.mark.skipif(sys.platform != "linux", reason="Initial Pi subprocess support is Linux/WSL.")
def test_model_proposal_uses_shared_pending_review_and_one_slot(
    tmp_path: Path, configured_pi: PiConfig
) -> None:
    """A model proposal remains non-executable pending review and uses the existing journal.

    Args:
        tmp_path (Path): Private fake campaign/profile root.
        configured_pi (PiConfig): Credential-free scripted Pi process, never a live provider.
    """
    original = Campaign.create(
        tmp_path / "campaign", ModelProblem.seed_artifact(), _models(), _ModelRunner.image
    )
    original.calibrate(_ModelRunner())
    campaign = PiCampaign(original.path)
    proposal = campaign.generate(configured_pi, consent=True)
    assert proposal["status"] == "pending_review" and proposal["attempt"] == 1
    assert sum(e["kind"] == "submission" for e in campaign.events()) == 1
    with pytest.raises(ValueError, match="review"):
        campaign.approve(1, proposal["candidate"], _ModelRunner(), reviewed=False)
    campaign.reject(1, "Inspected fixture-only source; no numerical construction approval.")
    assert PiCampaign(campaign.path).proposal(1)["status"] == "rejected"
    assert sum(e["kind"] == "submission" for e in campaign.events()) == 1


def test_strict_model_policy_still_rejects_row_infeasible_candidate_outputs(tmp_path: Path) -> None:
    """A frozen strict campaign cannot silently adopt feasibility-first exploration.

    Args:
        tmp_path (Path): Isolated strict campaign root.
    """
    runner = _ModelRunner()
    seed = ModelProblem.seed_artifact()
    campaign = Campaign.create(
        tmp_path / "strict", seed, _models(), runner.image, feasibility_policy="strict_feasible"
    )
    campaign.calibrate(runner)
    decision = campaign.submit(
        seed.source + "\n# strict policy test\n",
        "policy fixture",
        seed.digest,
        runner,
        reviewed=True,
    )
    assert decision["status"] == "rejected"
    assert Campaign(campaign.path).spec["feasibility_policy"] == "strict_feasible"


def test_model_tie_is_not_an_evolved_win_or_export(tmp_path: Path) -> None:
    """An unchanged primary-loss panel must not become a completed evolved shortlist.

    Args:
        tmp_path (Path): Synthetic complete tied-panel campaign.
    """
    runner = _ModelRunner(improve=False)
    seed = ModelProblem.seed_artifact()
    campaign = Campaign.create(tmp_path / "tied", seed, _models(), runner.image)
    campaign.calibrate(runner)
    decision = campaign.submit(
        seed.source + "\n# tied synthetic test\n",
        "no improvement",
        seed.digest,
        runner,
        reviewed=True,
    )
    assert decision["status"] == "rejected"
    with pytest.raises(ValueError):
        campaign.export(
            DecoderArtifact.create(
                "linear_model", seed.source + "\n# tied synthetic test\n", seed.digest
            ).digest,
            tmp_path / "not-a-win",
        )


def test_model_evolution_requires_actual_construction_freedom(tmp_path: Path) -> None:
    """Fixed-model inference is valid, but an entirely zero-key evolution panel is not.

    Args:
        tmp_path (Path): Directory that must remain absent after invalid evolution setup.
    """
    models = []
    values = (0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0)
    for problem in _models():
        fixed = dataclasses.replace(
            problem.source,
            variables=tuple(
                dataclasses.replace(variable, fixed=float(value))
                for variable, value in zip(problem.source.variables, values, strict=True)
            ),
        )
        models.append(
            ModelProblem.from_source(
                fixed,
                reference=ModelReference(23 + fixed.objective_constant, "fixed original witness"),
            )
        )
    with pytest.raises(ValueError, match="construction freedom"):
        Campaign.create(
            tmp_path / "fixed", ModelProblem.seed_artifact(), models, _ModelRunner.image
        )
    assert not (tmp_path / "fixed").exists()
