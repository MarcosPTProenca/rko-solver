"""Opt-in lease/exec transport tests for process freshness and hard failure retirement.

Objective:
- Verify isolation and failure retirement for persistent container leases.
Technical approach:
- Use an explicitly selected live image and fresh decoder processes within a lease.
Flow (pseudocode):
1. Qualify the opted-in persistent runner.
2. Exercise state mutation, job directories, OOM bookkeeping and timeout handling.
3. Assert isolation and require requalification after lease retirement.
Important criteria:
- Skip live checks without RKO_EVOLVE_IMAGE; never reuse a retired lease unqualified.
Inputs/Outputs:
- Input: live image environment setting, temporary paths and pytest fixtures.
- Output: assertions on process freshness, private paths and hard failure handling.
"""

import os
import pytest
from collections.abc import Iterator
from pathlib import Path
from rko_solver.experimental.self_evolve import DecoderArtifact
from rko_solver.experimental.self_evolve import Problem
from rko_solver.experimental.self_evolve.sandbox import PodmanRunner
from rko_solver.experimental.self_evolve.sandbox import SandboxError
from rko_solver.experimental.self_evolve.session import PersistentPodmanRunner
from tests.experimental.test_confirmation_study import create_study


def test_oom_marker_stays_outside_caller_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conmon OOM bookkeeping belongs in a private host directory, not the project checkout.

    Args:
        tmp_path (Path): Temporary directory used as the caller's working directory.
        monkeypatch (pytest.MonkeyPatch): Fixture that temporarily changes the working directory.
    """
    image = os.environ.get("RKO_EVOLVE_IMAGE")
    if not image:
        pytest.skip("Set RKO_EVOLVE_IMAGE for live OOM bookkeeping checks.")
    monkeypatch.chdir(tmp_path)
    with PersistentPodmanRunner(image, lease_seconds=120) as runner:
        with pytest.raises(SandboxError) as error:
            runner._execute({"probe": "memory"}, 20)
        assert error.value.reason == "memory"
    assert not (tmp_path / "oom").exists()


def test_study_owns_live_setup_and_cleanup(tmp_path: Path) -> None:
    """Run a real measured cell only after the study has durably prepaid qualification.

    Args:
        tmp_path (Path): Isolated synthetic study, never a research-panel replacement.
    """
    image = os.environ.get("RKO_EVOLVE_IMAGE")
    if not image:
        pytest.skip("Set RKO_EVOLVE_IMAGE for the real study runtime lifecycle.")
    pinned = PodmanRunner(image).image
    study = create_study(tmp_path / "study", transport="persistent_exec", image=pinned)
    with study.runtime(approved_spec=study.digest) as runner:
        study.run(runner, approved_spec=study.digest, max_jobs=1)
        assert study.progress()["session_startup_upper_bound"] > 0
    result = study.progress()
    assert result["complete"] == 1
    assert result["interrupted"] == 0
    assert result["charged_cpu_upper_bound"] < result["cpu_budget"]
    assert result["inference_available"] is False


@pytest.fixture(scope="module")
def session() -> Iterator[PersistentPodmanRunner]:
    """Qualify a bounded leased runtime only when an explicit live image is provided.

    Yields:
        PersistentPodmanRunner: Actual resource-qualified fresh-process transport.
    """
    image = os.environ.get("RKO_EVOLVE_IMAGE")
    if not image:
        pytest.skip("Set RKO_EVOLVE_IMAGE to exercise live leased isolation.")
    with PersistentPodmanRunner(image, lease_seconds=600) as runner:
        runner.qualify()
        yield runner


def test_each_job_gets_a_fresh_interpreter(session: PersistentPodmanRunner) -> None:
    """A decoder's NumPy global setting cannot contaminate the next decoder process.

    Args:
        session (PersistentPodmanRunner): Qualified live lease for sequential decoder jobs.
    """
    problem = Problem("knapsack", {"weights": [2, 3], "profits": [3, 5], "capacity": 5})
    mutate = DecoderArtifact.create(
        "knapsack",
        "import numpy as np\n"
        'def decode(keys, instance):\n    """Controlled state-mutation fixture."""\n'
        '    np.seterr(divide="raise")\n    return [0]\n',
    )
    observe = DecoderArtifact.create(
        "knapsack",
        "import numpy as np\n"
        'def decode(keys, instance):\n    """Observe fresh process defaults."""\n'
        '    return [0, 1] if np.geterr()["divide"] != "raise" else []\n',
    )
    assert session.run(mutate, problem, mode="fixed", seconds=0.03)["objective"] == 3
    assert session.run(observe, problem, mode="fixed", seconds=0.03)["objective"] == 8


def test_each_job_gets_a_private_working_directory(session: PersistentPodmanRunner) -> None:
    """The fixed guard creates distinct per-job temporary paths inside the bounded tmpfs.

    Args:
        session (PersistentPodmanRunner): Qualified live lease for repeated policy probes.
    """
    first = session._execute({"probe": "policy"}, 20)
    second = session._execute({"probe": "policy"}, 20)
    assert first["working_directory"] != second["working_directory"]
    assert first["working_directory"].startswith("/tmp/rko-job.")
    assert second["tmpfs_restricted"] is True


def test_nonreturning_job_retires_lease_before_reuse(session: PersistentPodmanRunner) -> None:
    """External timeout destroys the lease, and no source executes until requalification.

    Args:
        session (PersistentPodmanRunner): Qualified live lease to retire and requalify.
    """
    problem = Problem("knapsack", {"weights": [2], "profits": [3], "capacity": 5})
    bad = DecoderArtifact.create(
        "knapsack",
        "def decode(keys, instance):\n"
        '    """Adversarial timeout fixture."""\n    while True:\n        pass\n',
    )
    with pytest.raises(SandboxError) as error:
        session.run(bad, problem, seconds=0.01)
    assert error.value.reason == "deadline"
    assert error.value.phase == "worker_guardian"
    assert session.needs_restart is True
    good = DecoderArtifact.create("knapsack", Problem.seed_source("knapsack"))
    with pytest.raises(SandboxError, match="requalification"):
        session.run(good, problem, seconds=0.03)
    session.qualify()
    assert session.run(good, problem, mode="fixed", seconds=0.03)["feasible"] is True
