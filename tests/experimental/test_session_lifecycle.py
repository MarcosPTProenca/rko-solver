"""Leased transport lifecycle and immutable invocation policy contracts.

Objective:
- Verify startup, admission and cleanup failures without requiring live Podman.
Technical approach:
- Replace only administrative calls and policy-probe responses with explicit fixtures.
Flow (pseudocode):
1. Inspect lease limits and construct a real transport object.
2. Observe the exact immutable launch/exec policy and fresh working-directory guard.
3. Inject setup or cleanup failures and assert owned-container retirement and invalidation.
Important criteria:
- Synthetic policy records test controller decisions, not live isolation effectiveness.
Inputs/Outputs:
- Input: trusted administrative responses and controlled lifecycle errors.
- Output: command vectors, bounded admission and fail-closed transport state.
"""

import pytest
import subprocess
import sys
import typing
from rko_solver.experimental.self_evolve.sandbox import PodmanRunner
from rko_solver.experimental.self_evolve.sandbox import SandboxError
from rko_solver.experimental.self_evolve.session import PersistentPodmanRunner


@pytest.fixture
def policy() -> dict[str, typing.Any]:
    """Return a clearly synthetic exact policy response for decision tests.

    Returns:
        dict[str, Any]: Required immutable policy fields for the default limits.
    """
    return {
        "uid": 65534,
        "memory": 536870912,
        "pids": 32,
        "cpu": "100000 100000",
        "interfaces": ["lo"],
        "read_only": True,
        "capabilities_dropped": True,
        "tmpfs_restricted": True,
        "no_new_privileges": True,
    }


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> PersistentPodmanRunner:
    """Construct an inspected transport without performing any administrative command.

    Args:
        monkeypatch (pytest.MonkeyPatch): Platform and immutable-image inspection seam.

    Returns:
        PersistentPodmanRunner: Unstarted and unqualified real transport object.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(PodmanRunner, "_inspect_image", lambda *args: "sha256:" + "a" * 64)
    return PersistentPodmanRunner("trusted-fixture", lease_seconds=60)


@pytest.mark.parametrize("lease", [59, 86401, float("nan"), float("inf")])
def test_invalid_leases_never_inspect_or_start_runtime(
    monkeypatch: pytest.MonkeyPatch, lease: float
) -> None:
    """Out-of-envelope leases are rejected before any external resource is created.

    Args:
        monkeypatch (pytest.MonkeyPatch): Prohibition of runtime inspection.
        lease (float): Invalid lease duration.
    """
    monkeypatch.setattr(PodmanRunner, "_inspect_image", lambda *args: pytest.fail("Inspection."))
    with pytest.raises(ValueError, match="lease"):
        PersistentPodmanRunner("fixture", lease_seconds=lease)


def test_session_policy_exec_and_lease_admission(
    runner: PersistentPodmanRunner, policy: dict[str, typing.Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An owned lease uses fixed restrictions, fresh process guards and confirmed cleanup.

    Args:
        runner (PersistentPodmanRunner): Inspected but unstarted transport.
        policy (dict[str, Any]): Synthetic exact effective restriction response.
        monkeypatch (pytest.MonkeyPatch): Administrative/probe seams only.
    """
    commands = []
    monkeypatch.setattr(
        runner, "_command", lambda args, **kwargs: commands.append(args) or "GNU coreutils 9"
    )
    monkeypatch.setattr(runner, "_execute", lambda *args: policy)
    with runner:
        name = runner._container
        assert name is not None
        launch = commands[0]
        for restriction in (
            "--network=none",
            "--read-only",
            "--cap-drop=all",
            "--security-opt=no-new-privileges",
            "--cpus=1",
            "--pids-limit=32",
            "--detach",
            "--init",
        ):
            assert restriction in launch
        assert launch[-2:] == ["sleep", "60"]
        before = len(commands)
        runner._ensure_started()
        assert len(commands) == before
        command = runner._invocation({}, "unused", 0.1)
        assert command[:4] == ["podman", "exec", "--interactive", name]
        assert "mktemp -d /tmp/rko-job.XXXXXX" in command[6]
        assert command[-3:] == ["python", "-m", "rko_solver.experimental.self_evolve.worker"]
        probe = runner._invocation({"probe": "policy"}, "unused", 1)
        assert probe[-3:-1] == ["python", "-c"]
        runner.qualified = True
        runner._expires = 0
        with pytest.raises(SandboxError, match="lease"):
            runner._invocation({}, "unused", 1)
        assert runner.qualified is False
    assert runner._container is None
    assert commands[-1] == ["rm", "--force", "--ignore", "--time=1", name]
    assert runner.startup_cpu_charge > 0


@pytest.mark.parametrize("failure", ["policy", "version", "launch"])
def test_failed_startup_removes_exact_owned_lease(
    runner: PersistentPodmanRunner,
    policy: dict[str, typing.Any],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Incomplete startup cannot leave a qualified transport or an unowned live lease.

    Args:
        runner (PersistentPodmanRunner): Inspected real transport object.
        policy (dict[str, Any]): Mutable synthetic policy fixture.
        monkeypatch (pytest.MonkeyPatch): Bounded administrative response seam.
        failure (str): Stage whose result or execution is invalid.
    """
    commands = []

    def administrative(args: list[str], *, cwd: str | None = None) -> str:
        """Inject one setup-stage failure while permitting exact owned cleanup.

        Args:
            args (list[str]): Transport-owned administrative vector.
            cwd (str | None): Optional administrative working-directory argument.

        Returns:
            str: Deliberately invalid GNU-version response when requested.

        Raises:
            subprocess.CalledProcessError: If launch failure is selected.
        """
        commands.append(args)
        if args[0] == "run" and failure == "launch":
            raise subprocess.CalledProcessError(125, args)
        return "not GNU timeout"

    if failure == "policy":
        policy["uid"] = 0
    monkeypatch.setattr(runner, "_command", administrative)
    monkeypatch.setattr(runner, "_execute", lambda *args: policy)
    try:
        with pytest.raises(SandboxError, match="start/qualify"):
            runner.__enter__()
        assert runner.qualified is False
        assert runner._container is None
        assert commands[-1][0] == "rm"
        assert commands[-1][-1].startswith("rko-evolve-session-")
        assert runner.startup_cpu_charge > 0
    finally:
        runner._host_directory.cleanup()


def test_unconfirmed_cleanup_disables_transport_and_keeps_owned_identity(
    runner: PersistentPodmanRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed removal retains the exact identity for recovery instead of claiming cleanup.

    Args:
        runner (PersistentPodmanRunner): Transport with a synthetic owned container.
        monkeypatch (pytest.MonkeyPatch): Administrative removal failure seam.
    """
    runner._container = "owned-lease"
    runner.qualified = True

    def unavailable(args: list[str], *, cwd: str | None = None) -> str:
        """Represent an unavailable administrative executable.

        Args:
            args (list[str]): Ignored owned administrative command.
            cwd (str | None): Ignored private working-directory argument.

        Raises:
            OSError: Always, to make cleanup unconfirmed.
        """
        raise OSError("fixture runtime unavailable")

    monkeypatch.setattr(runner, "_command", unavailable)
    try:
        with pytest.raises(SandboxError, match="cleanup unconfirmed"):
            runner._stop()
        assert runner.qualified is False
        assert runner._container == "owned-lease"
    finally:
        runner._host_directory.cleanup()
