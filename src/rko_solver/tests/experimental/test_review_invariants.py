"""Durable admission and verification regressions found by independent review.

Objective:
- Block unknown work, tainted evidence and unreserved runtime setup.
Technical approach:
- Exercise actual campaign/study journals with controlled trusted runtime doubles.
Flow (pseudocode):
1. Interrupt work or inject invalid evidence at a real collector boundary.
2. Reopen the ledger or attempt runtime construction.
3. Require a durable stop before further spending or inference.
Important criteria:
- No candidate code is executed by these unit tests.
Inputs/Outputs:
- Input: controlled collector interruptions and runtime/reference failures.
- Output: explicit durable rejection and retained conservative reservations.
"""

import importlib
import json
import pytest
import sqlite3
import sys
from pathlib import Path
from rko_solver.experimental.self_evolve import DecoderArtifact
from rko_solver.experimental.self_evolve import Problem
from rko_solver.experimental.self_evolve import study as study_module
from rko_solver.experimental.self_evolve.campaign import Campaign
from rko_solver.experimental.self_evolve.sandbox import PodmanRunner
from rko_solver.experimental.self_evolve.sandbox import SandboxError
from rko_solver.tests.experimental.test_confirmation_study import create_study
from rko_solver.tests.experimental.test_self_evolve_campaign import FixtureRunner
from rko_solver.tests.experimental.test_self_evolve_campaign import make_campaign
from rko_solver.tests.experimental.test_timeout_attribution import HostTimeoutRunner
from typing import Any


@pytest.mark.skipif(sys.platform != "linux", reason="Trusted Linux container-worker contract.")
def test_worker_uses_supported_explicit_execution() -> None:
    """Exercise the real worker configuration using only a hard-coded trusted decoder fixture."""
    evaluate_job = importlib.import_module(
        "rko_solver.experimental.self_evolve.worker"
    ).evaluate_job
    evidence = evaluate_job(
        {
            "family": "knapsack",
            "source": (
                "def decode(keys, instance):\n"
                '    """Trusted empty-packing fixture."""\n    return []\n'
            ),
            "data": {"weights": [1], "profits": [3], "capacity": 1},
            "mode": "rko",
            "seconds": 0.01,
            "seed": 0,
        }
    )
    assert evidence["status"] == "ok"
    assert evidence["feasible"] is True
    assert evidence["evaluations"] > 0


def test_replay_identity_binds_extracted_helpers() -> None:
    """Moving trusted logic into helpers must not remove it from frozen replay identity."""
    bound = study_module.ConfirmationStudy._implementation()["code"]
    assert {
        "sandbox_evidence.py",
        "campaign_evaluation.py",
        "study_collection.py",
        "study_definition.py",
        "study_runtime.py",
        "study_report.py",
    } <= set(bound)


def test_campaign_unknown_reservation_is_not_retried(tmp_path: Path) -> None:
    """Unknown campaign cells retain both their charge and an irreversible collection stop.

    Args:
        tmp_path (Path): Isolated append-only campaign journal.
    """
    campaign = make_campaign(tmp_path / "campaign")
    campaign._append(
        {
            "kind": "reservation",
            "partition": "development",
            "charge": 60.1,
            "cell": [campaign.spec["seed"], 0, 0, "rko", "baseline"],
        }
    )
    before = campaign.events()
    with pytest.raises(ValueError, match="Unknown reserved"):
        campaign.calibrate(FixtureRunner())
    assert campaign.events() == before


def test_invalid_witness_disables_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """A claimed successful witness contradicting the evaluator is not ordinary candidate loss.

    Args:
        monkeypatch (pytest.MonkeyPatch): Controlled worker response boundary.
    """
    runner = object.__new__(PodmanRunner)
    runner.qualified = True
    runner.image = FixtureRunner.image
    problem = Problem("knapsack", {"weights": [1], "profits": [3], "capacity": 1})
    artifact = DecoderArtifact.create("knapsack", Problem.seed_source("knapsack"))
    monkeypatch.setattr(
        runner,
        "_execute",
        lambda *args: {
            "status": "ok",
            "solution": [0],
            "objective": 4,
            "feasible": True,
            "score": 0.0,
        },
    )
    with pytest.raises(SandboxError, match="independent verification"):
        runner.run(artifact, problem)
    assert runner.qualified is False


def test_integrity_stop_survives_new_study_object(tmp_path: Path) -> None:
    """A new qualified runner cannot resume a study invalidated by infrastructure failure.

    Args:
        tmp_path (Path): Isolated frozen measurement panel.
    """
    study = create_study(tmp_path / "study", transport="persistent_exec")
    with pytest.raises(ValueError, match="integrity"):
        study.run(HostTimeoutRunner(), approved_spec=study.digest)
    reopened = type(study)(study.path)
    before = reopened.progress()
    runner = FixtureRunner()
    runner.transport = "persistent_exec"
    with pytest.raises(ValueError, match="integrity"):
        reopened.run(runner, approved_spec=reopened.digest)
    assert reopened.progress() == before


@pytest.mark.parametrize("fail_setup", [False, True])
def test_runtime_setup_is_reserved_before_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_setup: bool
) -> None:
    """Startup is prepaid durably; interruption keeps its full reservation and blocks reuse.

    Args:
        tmp_path (Path): Isolated setup and measurement journal.
        monkeypatch (pytest.MonkeyPatch): Trusted administrative runtime seam.
        fail_setup (bool): Whether qualification is interrupted before evidence is returned.
    """
    study = create_study(tmp_path / "study")
    constructed = []

    class Runtime(FixtureRunner):
        """Represent qualified metadata while asserting durable prepayment before construction.

        Responsibilities:
        - Observe setup prepayment and optionally interrupt qualification.

        Attributes:
            image (str): Inherited immutable fixture runtime identity.
        """

        def __init__(self, image: str) -> None:
            """Validate the prior durable setup reservation.

            Args:
                image (str): Pinned test image, not actually launched.
            """
            with sqlite3.connect(study.path / "measurements.sqlite") as db:
                assert db.execute("SELECT value FROM state WHERE key='setup_pending'").fetchone()
                assert (
                    float(
                        db.execute("SELECT value FROM state WHERE key='setup_charge'").fetchone()[0]
                    )
                    >= 900
                )
            constructed.append(image)

        def qualify(self) -> dict[str, Any]:
            """Return controlled evidence or interrupt the startup transaction.

            Returns:
                dict[str, Any]: Trusted empty qualification record.

            Raises:
                KeyboardInterrupt: If the test injects unknown setup completion.
            """
            if fail_setup:
                raise KeyboardInterrupt
            return {}

    monkeypatch.setattr(study_module, "PodmanRunner", Runtime)
    if fail_setup:
        with pytest.raises(KeyboardInterrupt), study.runtime(approved_spec=study.digest):
            pytest.fail("Interrupted setup must not release a runtime.")
        assert study.progress()["session_startup_upper_bound"] == 900
        with pytest.raises(ValueError, match="setup"), study.runtime(approved_spec=study.digest):
            pytest.fail("Unknown setup must not be repeated.")
    else:
        with study.runtime(approved_spec=study.digest) as runner:
            before = study.progress()["session_startup_upper_bound"]
            with pytest.raises(ValueError, match="Another runtime"):
                with study.runtime(approved_spec=study.digest):
                    pytest.fail("Nested setup must not construct another runtime.")
            assert study.progress()["session_startup_upper_bound"] == before
            study.run(runner, approved_spec=study.digest, max_jobs=1)
        assert 0 < study.progress()["session_startup_upper_bound"] < 900
    assert len(constructed) == 1


def test_exhausted_study_does_not_construct_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Startup cannot consume additional budget after measurement allocation is exhausted.

    Args:
        tmp_path (Path): Isolated budget ledger.
        monkeypatch (pytest.MonkeyPatch): Constructor must remain unreachable.
    """
    study = create_study(tmp_path / "study")
    with sqlite3.connect(study.path / "measurements.sqlite") as db:
        db.execute("INSERT INTO state VALUES ('setup_charge',?)", (str(study.spec["cpu_budget"]),))
    monkeypatch.setattr(
        study_module, "PodmanRunner", lambda *args: pytest.fail("Unreserved startup.")
    )
    with pytest.raises(ValueError, match="budget"), study.runtime(approved_spec=study.digest):
        pytest.fail("Exhausted study must not start a runtime.")


def test_cleanup_overrun_is_charged_and_blocks_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Known cleanup consumption cannot be clipped to a smaller reservation.

    Args:
        tmp_path (Path): Isolated measurement/setup ledger.
        monkeypatch (pytest.MonkeyPatch): Trusted runtime and deterministic measured-clock seam.
    """
    study = create_study(tmp_path / "study")

    class Runtime(FixtureRunner):
        """Supply trusted unit-test qualification without a live subprocess.

        Responsibilities:
        - Preserve collector lifecycle while measuring a controlled cleanup overrun.

        Attributes:
            image (str): Existing immutable fixture runtime identity.
        """

        def __init__(self, image: str) -> None:
            """Retain the study's image.

            Args:
                image (str): Immutable runtime identity.
            """
            self.image = image

        def qualify(self) -> dict[str, Any]:
            """Return the trusted fixture qualification.

            Returns:
                dict[str, Any]: Empty unit-test policy record.
            """
            return {}

    monkeypatch.setattr(study_module, "PodmanRunner", Runtime)
    clock = iter([0.0, 1.0, 2.0, 80.0])
    monkeypatch.setattr(study_module.time, "monotonic", lambda: next(clock))
    with study.runtime(approved_spec=study.digest):
        pass
    assert study.progress()["session_startup_upper_bound"] == pytest.approx(79.2)
    with pytest.raises(ValueError, match="setup"), study.runtime(approved_spec=study.digest):
        pytest.fail("Cleanup overrun must durably invalidate collection.")


@pytest.mark.parametrize("field", ["cpu_seconds", "outer_wall_seconds"])
@pytest.mark.parametrize("usage", [75.0, 6000.0])
@pytest.mark.parametrize("completed_before", [0, 71])
def test_study_overrun_retains_full_charge_and_blocks_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    usage: float,
    completed_before: int,
) -> None:
    """Known cell overruns retain full charges and durably block collection and inference.

    Args:
        tmp_path (Path): Isolated study journal.
        monkeypatch (pytest.MonkeyPatch): Controlled trusted runner evidence seam.
        field (str): Resource field exceeding the per-cell reservation.
        usage (float): Known consumption, possibly exceeding the entire allocation.
        completed_before (int): Whether the overrun is the first or last panel cell.
    """
    study = create_study(tmp_path / "study")
    if completed_before:
        study.run(FixtureRunner(), approved_spec=study.digest, max_jobs=completed_before)
    previous_charge = study.progress()["charged_cpu_upper_bound"]
    runner = FixtureRunner()
    run = runner.run
    monkeypatch.setattr(
        runner, "run", lambda *args, **kwargs: {**run(*args, **kwargs), field: usage}
    )
    with pytest.raises(ValueError, match="integrity"):
        study.run(runner, approved_spec=study.digest)
    assert runner.qualified is False
    reopened = type(study)(study.path)
    progress = reopened.progress()
    expected = previous_charge + usage + (0.1 if field == "outer_wall_seconds" else 0)
    assert progress["charged_cpu_upper_bound"] == pytest.approx(expected)
    assert progress["complete"] == completed_before + 1
    assert progress["pending"] == 71 - completed_before
    assert progress["inference_available"] is False
    with sqlite3.connect(study.path / "measurements.sqlite") as db:
        result = db.execute(
            "SELECT result FROM jobs WHERE json_extract(result, '$.status')='infrastructure_error'"
        ).fetchone()[0]
    record = json.loads(result)
    assert record["status"] == "infrastructure_error"
    assert record["error_reason"] == "resource_overrun"
    assert reopened.report()["status"] == "incomplete_no_inference"
    with pytest.raises(ValueError, match="integrity"):
        reopened.run(FixtureRunner(), approved_spec=reopened.digest)
    assert reopened.progress() == progress


@pytest.mark.parametrize("field", ["cpu_seconds", "outer_wall_seconds"])
@pytest.mark.parametrize("usage", [75.0, 6000.0])
def test_campaign_overrun_retains_full_charge_and_blocks_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, usage: float
) -> None:
    """Known overruns cannot become successful baseline cells or free resumed work.

    Args:
        tmp_path (Path): Isolated campaign ledger.
        monkeypatch (pytest.MonkeyPatch): Controlled trusted runner evidence seam.
        field (str): Resource field exceeding the per-cell reservation.
        usage (float): Known consumption, possibly exceeding the entire allocation.
    """
    campaign = make_campaign(tmp_path / "campaign")
    runner = FixtureRunner()
    run = runner.run
    monkeypatch.setattr(
        runner, "run", lambda *args, **kwargs: {**run(*args, **kwargs), field: usage}
    )
    with pytest.raises(ValueError, match="integrity"):
        campaign.calibrate(runner)
    assert runner.qualified is False
    reopened = Campaign(campaign.path)
    events = reopened.events()
    expected = usage + (0.1 if field == "outer_wall_seconds" else 0)
    assert sum(event.get("charge", 0) for event in events) == pytest.approx(expected)
    records = [event for event in events if event["kind"] == "evaluation"]
    assert len(records) == 1
    assert records[0]["status"] == "infrastructure_error"
    assert records[0]["error_reason"] == "resource_overrun"
    assert not any(event["kind"] == "calibrated" for event in events)
    with pytest.raises(ValueError, match="integrity"):
        reopened.calibrate(FixtureRunner())
    with pytest.raises(ValueError, match="integrity"):
        reopened.request()
    assert reopened.events() == events
