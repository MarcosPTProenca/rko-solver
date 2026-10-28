"""Timeout attribution regressions for the real bounded-I/O collector.

Objective:
- Prevent delayed OCI exec startup from becoming a decoder-quality observation.
Technical approach:
- Drive the real selector/process cleanup with a controlled host clock.
Flow (pseudocode):
1. Launch a harmless local sleeping subprocess through the collector seam.
2. Expire its host I/O deadline before any worker evidence exists.
3. Require infrastructure invalidation, stage evidence and confirmed retirement.
Important criteria:
- No generated source or live Podman runtime is used in unit tests.
Inputs/Outputs:
- Input: deterministic host-clock expiration.
- Output: explicit infrastructure stop rather than a candidate failure.
"""

import json
import pytest
import sqlite3
import sys
from pathlib import Path
from rko_solver.experimental.self_evolve import DecoderArtifact
from rko_solver.experimental.self_evolve import Problem
from rko_solver.experimental.self_evolve import sandbox
from rko_solver.experimental.self_evolve.sandbox import PodmanRunner
from rko_solver.experimental.self_evolve.sandbox import SandboxError
from rko_solver.experimental.self_evolve.session import PersistentPodmanRunner
from tests.experimental.test_confirmation_study import create_study
from tests.experimental.test_self_evolve_campaign import FixtureRunner
from tests.experimental.test_self_evolve_campaign import make_campaign
from typing import Any


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX bounded-process cleanup contract.")
@pytest.mark.parametrize("job,reason", [({}, "deadline"), ({"probe": "hang"}, "runtime")])
def test_host_deadline_disables_leased_runner(
    monkeypatch: pytest.MonkeyPatch, job: dict[str, str], reason: str
) -> None:
    """An outer deadline must not be attributed to decoder runtime inside GNU timeout.

    Args:
        monkeypatch (pytest.MonkeyPatch): Controlled collector clock and runtime launch seam.
        job (dict[str, str]): Ordinary request or qualification probe.
        reason (str): Expected non-spoofable failure category.
    """
    runner = object.__new__(PersistentPodmanRunner)
    runner.qualified = True
    runner.needs_restart = False
    runner.environment = {}
    retired = []
    monkeypatch.setattr(runner, "_stop", lambda: retired.append(True))
    monkeypatch.setattr(
        runner, "_invocation", lambda *args: [sys.executable, "-c", "import time; time.sleep(1)"]
    )
    clock = iter([0.0, 30.0])
    monkeypatch.setattr(sandbox.time, "monotonic", lambda: next(clock))
    with pytest.raises(SandboxError) as caught:
        runner._execute(job, 0.01)
    assert runner.qualified is False, "Host startup/transport timeout is not candidate failure."
    assert caught.value.reason == reason
    assert caught.value.phase == "host_io"
    assert runner.needs_restart is True
    assert retired == [True]


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX bounded-process cleanup contract.")
def test_default_transport_host_timeout_is_infrastructure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default transport must not accept an outer timeout as candidate evidence either.

    Args:
        monkeypatch (pytest.MonkeyPatch): Harmless process seam and deterministic host clock.
    """
    runner = object.__new__(PodmanRunner)
    runner.qualified = True
    runner.environment = {}
    monkeypatch.setattr(
        runner, "_invocation", lambda *args: [sys.executable, "-c", "import time; time.sleep(1)"]
    )
    monkeypatch.setattr(runner, "_cleanup_invocation", lambda *args: None)
    clock = iter([0.0, 30.0])
    monkeypatch.setattr(sandbox.time, "monotonic", lambda: next(clock))
    with pytest.raises(SandboxError) as caught:
        runner._execute({}, 0.01)
    assert caught.value.phase == "host_io"
    assert runner.qualified is False


class HostTimeoutRunner(FixtureRunner):
    """Represent the collector-visible outcome of an unqualified leased host timeout.

    Responsibilities:
    - Invalidate qualification without producing decoder observations.

    Attributes:
        transport (str): Frozen leased execution identity.
        qualified (bool): False after injected infrastructure failure.
    """

    transport = "persistent_exec"

    def run(
        self, artifact: DecoderArtifact, problem: Problem, *, mode: str, seed: int, seconds: float
    ) -> dict[str, Any]:
        """Expose infrastructure failure without producing decoder observations.

        Args:
            artifact (DecoderArtifact): Unexecuted artifact.
            problem (Problem): Unexecuted case.
            mode (str): Frozen control arm.
            seed (int): Frozen solver seed.
            seconds (float): Inner allowance, never consumed here.

        Raises:
            SandboxError: Actual host I/O origin with invalidated runtime qualification.
        """
        self.qualified = False
        raise SandboxError("External worker deadline exceeded.", reason="deadline", phase="host_io")


def test_study_preserves_timeout_origin_and_blocks_inference(tmp_path: Path) -> None:
    """Infrastructure failure cannot become a paired candidate utility observation.

    Args:
        tmp_path (Path): Isolated frozen-study storage.
    """
    study = create_study(tmp_path / "study", transport="persistent_exec")
    with pytest.raises(ValueError, match="integrity"):
        study.run(HostTimeoutRunner(), approved_spec=study.digest)
    with sqlite3.connect(study.path / "measurements.sqlite") as db:
        result = json.loads(
            db.execute("SELECT result FROM jobs WHERE status='complete'").fetchone()[0]
        )
    assert result["status"] == "infrastructure_error"
    assert result["error_reason"] == "deadline"
    assert result["error_phase"] == "host_io"
    report = study.report()
    assert report["status"] == "incomplete_no_inference"
    assert "primary_comparisons" not in report


def test_campaign_stops_on_host_timeout(tmp_path: Path) -> None:
    """The development campaign must not select from infrastructure-tainted evidence.

    Args:
        tmp_path (Path): Isolated campaign storage.
    """
    campaign = make_campaign(tmp_path / "campaign")
    with pytest.raises(ValueError, match="integrity"):
        campaign.calibrate(HostTimeoutRunner())
    failures = [e for e in campaign.events() if e.get("status") == "infrastructure_error"]
    assert len(failures) == 1
    assert failures[0]["error_phase"] == "host_io"
