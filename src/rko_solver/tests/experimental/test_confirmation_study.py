"""Frozen-panel collection tests at the external-runner and analysis-report seams.

Objective:
- Verify approved frozen-panel collection, conservative accounting and inference gates.
Technical approach:
- Create synthetic three-family studies and inject deterministic or failing runner records.
Flow (pseudocode):
1. Freeze artifacts, instances, arms, seeds and resource allowances.
2. Collect or interrupt jobs, reopen progress and exercise resource admission.
3. Check full-panel reports and reject altered specifications or unsafe continuation.
Important criteria:
- Incomplete panels yield no inference; failures retain cells and reserved resource charges.
- Infrastructure failures stop collection rather than fabricate algorithmic evidence.
Inputs/Outputs:
- Input: temporary study paths, frozen synthetic artifacts and external-runner fixtures.
- Output: study fixtures, deterministic failures and assertions on progress and reports.
"""

import pytest
from pathlib import Path
from rko_solver.experimental.self_evolve import DecoderArtifact
from rko_solver.experimental.self_evolve import Problem
from rko_solver.experimental.self_evolve.sandbox import SandboxError
from rko_solver.experimental.self_evolve.study import ConfirmationStudy
from rko_solver.tests.experimental.test_self_evolve_campaign import FixtureRunner
from typing import Any


def create_study(
    path: Path,
    *,
    cpu_budget: float = 5000,
    prior_instance_hashes: list[str] | None = None,
    claim: str = "exploratory",
    transport: str = "per_job",
    image: str = FixtureRunner.image,
) -> ConfirmationStudy:
    """Freeze tiny synthetic fixtures without executing candidate source.

    Args:
        path (Path): New study directory.
        cpu_budget (float): Explicit synthetic accounting allowance.
        prior_instance_hashes (list[str] | None): Known already-exposed instances.
        claim (str): Desired evidence label, without a fabricated power plan.
        transport (str): Explicit measurement transport identity.
        image (str): Immutable fixture or explicitly qualified live image identity.

    Returns:
        ConfirmationStudy: Frozen exploratory study with 72 jobs.
    """
    artifacts = {}
    instances = {}
    for family in ("knapsack", "scheduling", "set_cover"):
        seed = DecoderArtifact.create(family, Problem.seed_source(family))
        first = DecoderArtifact.create(family, seed.source + "\n", seed.digest)
        candidate = DecoderArtifact.create(family, seed.source + "\n\n", first.digest)
        artifacts[family] = {"seed": seed, "first_shot": first, "candidate": candidate}
        instances[family] = [Problem.generate(family, seed, 4) for seed in (3001, 3002)]
    return ConfirmationStudy.create(
        path,
        artifacts=artifacts,
        instances=instances,
        prior_instance_hashes=prior_instance_hashes or [],
        claim=claim,
        transport=transport,
        image=image,
        seeds=[0, 1],
        seconds=0.1,
        cpu_budget=cpu_budget,
        elapsed_budget=3600,
        minimum_effect=0.01,
        bootstrap_samples=200,
        permutation_samples=200,
    )


def test_exact_approval_and_complete_panel_are_required(tmp_path: Path) -> None:
    """A partial panel exposes progress only, never interim p-values or promotion.

    Args:
        tmp_path (Path): Temporary parent for the partially collected study.
    """
    study = create_study(tmp_path / "study")
    with pytest.raises(ValueError, match="approval"):
        study.run(FixtureRunner(), approved_spec="wrong")
    progress = study.run(FixtureRunner(), approved_spec=study.digest, max_jobs=1)
    assert progress["complete"] == 1
    report = ConfirmationStudy(study.path).report()
    assert report["status"] == "incomplete_no_inference"
    assert "primary_comparisons" not in report


def test_resume_runs_full_first_shot_controls_and_six_primary_comparisons(tmp_path: Path) -> None:
    """Resume preserves all cells and controls; solver seeds remain nested within instances.

    Args:
        tmp_path (Path): Temporary parent for the study resumed to a complete panel.
    """
    study = create_study(tmp_path / "study")
    study.run(FixtureRunner(), approved_spec=study.digest, max_jobs=2)
    reopened = ConfirmationStudy(study.path)
    result = reopened.run(FixtureRunner(), approved_spec=study.digest)
    assert result["complete"] == 72
    report = reopened.report()
    assert len(report["primary_comparisons"]) == 6
    assert all(effect["n_instances"] == 2 for effect in report["primary_comparisons"])
    assert all(effect["primary_family_size"] == 6 for effect in report["primary_comparisons"])
    assert all("first_shot" in losses for losses in report["mean_losses"].values())
    assert report["passes_predeclared_artifact_checks"] is False
    assert report["production_promotion"] is False
    assert report["method_reliability_claim"] is False


class InterruptingRunner(FixtureRunner):
    """Model controller interruption after reservation without fabricated evidence.

    Responsibilities:
    - Interrupt a real collector after durable admission.

    Attributes:
        image (str): Inherited immutable fixture runtime identity.
    """

    def run(
        self, artifact: DecoderArtifact, problem: Problem, *, mode: str, seed: int, seconds: float
    ) -> dict[str, Any]:
        """Interrupt the runner seam before a result exists.

        Args:
            artifact (DecoderArtifact): Unexecuted fixture source.
            problem (Problem): Validated fixture instance.
            mode (str): Declared arm mode.
            seed (int): Declared solver seed.
            seconds (float): Declared inner allowance.

        Raises:
            KeyboardInterrupt: Always, to represent an interrupted measurement.
        """
        raise KeyboardInterrupt


def test_interrupted_jobs_keep_reservation_and_block_inference(tmp_path: Path) -> None:
    """A crash is neither a free retry nor a silently dropped statistical observation.

    Args:
        tmp_path (Path): Temporary parent for the study with an interrupted job reservation.
    """
    study = create_study(tmp_path / "study")
    with pytest.raises(KeyboardInterrupt):
        study.run(InterruptingRunner(), approved_spec=study.digest)
    progress = ConfirmationStudy(study.path).progress()
    assert progress["interrupted"] == 1
    assert progress["charged_cpu_upper_bound"] > 45
    with pytest.raises(ValueError, match="Unknown reserved"):
        study.run(FixtureRunner(), approved_spec=study.digest)
    assert study.report()["status"] == "incomplete_no_inference"


class DisabledRunner(InterruptingRunner):
    """Represent failed runtime cleanup rather than an algorithmic timeout.

    Responsibilities:
    - Invalidate runtime integrity at the collector boundary.

    Attributes:
        qualified (bool): False when cleanup cannot be confirmed.
    """

    def run(
        self, artifact: DecoderArtifact, problem: Problem, *, mode: str, seed: int, seconds: float
    ) -> dict[str, Any]:
        """Disable the runner and require an infrastructure stop.

        Args:
            artifact (DecoderArtifact): Unexecuted source.
            problem (Problem): Validated input.
            mode (str): Frozen arm.
            seed (int): Frozen solver seed.
            seconds (float): Frozen allowance.

        Raises:
            SandboxError: Always, indicating cleanup could not be confirmed.
        """
        self.qualified = False
        raise SandboxError("Cleanup could not be confirmed.")


def test_disabled_runtime_stops_instead_of_fabricating_algorithm_failures(tmp_path: Path) -> None:
    """Infrastructure unavailability must not make every subsequent source look unsuccessful.

    Args:
        tmp_path (Path): Temporary parent for the study stopped by failed runtime cleanup.
    """
    study = create_study(tmp_path / "study")
    with pytest.raises(ValueError, match="integrity"):
        study.run(DisabledRunner(), approved_spec=study.digest)
    assert study.progress()["complete"] == 1
    assert study.report()["status"] == "incomplete_no_inference"


class FailingRunner(FixtureRunner):
    """Return genuine failure categories instead of deleting unsuccessful cells.

    Responsibilities:
    - Preserve failed-cell utility and conservative resource charges.

    Attributes:
        image (str): Inherited immutable fixture runtime identity.
    """

    def run(
        self, artifact: DecoderArtifact, problem: Problem, *, mode: str, seed: int, seconds: float
    ) -> dict[str, Any]:
        """Fail at the external runner boundary for all fixtures.

        Args:
            artifact (DecoderArtifact): Unexecuted fixture source.
            problem (Problem): Validated fixture data.
            mode (str): Frozen arm.
            seed (int): Frozen solver seed.
            seconds (float): Frozen allowance.

        Raises:
            SandboxError: Always, representing callback resource failure.
        """
        raise SandboxError("External worker deadline exceeded.", reason="deadline")


def test_failed_jobs_remain_in_loss_and_budget_accounting(tmp_path: Path) -> None:
    """Complete failure panels retain loss one and cannot pass constructive/candidate guards.

    Args:
        tmp_path (Path): Temporary parent for the study collecting only failed jobs.
    """
    study = create_study(tmp_path / "study")
    study.run(FailingRunner(), approved_spec=study.digest)
    report = study.report()
    assert all(value == 1 for arms in report["mean_losses"].values() for value in arms.values())
    assert report["strict_candidate_gate"] is False
    assert report["constructive_control_gate"] is False
    assert report["progress"]["charged_cpu_upper_bound"] == pytest.approx(72 * 60.1)


def test_budget_admission_stops_without_inventing_missing_results(tmp_path: Path) -> None:
    """A small explicit budget yields an incomplete study, not a favorable partial report.

    Args:
        tmp_path (Path): Temporary parent for the resource-limited study.
    """
    study = create_study(tmp_path / "study", cpu_budget=100)
    result = study.run(FailingRunner(), approved_spec=study.digest)
    assert result["complete"] == 1
    assert result["charged_cpu_upper_bound"] <= 100
    assert study.report()["status"] == "incomplete_no_inference"


def test_mutating_frozen_settings_blocks_collection(tmp_path: Path) -> None:
    """An in-memory time-limit change cannot bypass the approved specification hash.

    Args:
        tmp_path (Path): Temporary parent for the study whose in-memory settings are altered.
    """
    study = create_study(tmp_path / "study")
    study.spec["seconds"] = 10
    with pytest.raises(ValueError, match="modified"):
        study.run(FixtureRunner(), approved_spec=study.digest)
