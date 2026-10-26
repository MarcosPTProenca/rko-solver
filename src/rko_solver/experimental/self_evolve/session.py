"""Leased-container transport with a fresh supervised Python process for every job.

Objective:
- Amortize OCI startup without sharing a Python interpreter between decoder evaluations.
Technical approach:
- Reuse the base rootless policy, with init reaping and a bounded container lease.
- Execute each job through GNU timeout in a private temporary directory.
- Destroy the entire container after failed execution; no unsafe fallback is provided.
Flow (pseudocode):
1. Start a leased idle container and validate its actual restrictions.
2. Qualify timeout/output/memory/PID behavior through the real exec transport.
3. Launch a fresh Python worker per request, preserving bounded host I/O collection.
4. Recycle failures and remove the lease on context exit.
Important criteria:
- Filesystem state is shared only within the bounded container; decoder file I/O remains forbidden.
- Same-interpreter worker/evaluator integrity against hostile Python remains out of scope.
Inputs/Outputs:
- Input: pinned image, explicit lease and ordinary source-only worker jobs.
- Output: unchanged independently verified job evidence with reduced administrative overhead.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
import time
import uuid
from .sandbox import _POLICY_PROBE
from .sandbox import PodmanRunner
from .sandbox import SandboxError
from types import TracebackType
from typing import Any

_EXEC_GUARD = """
directory=$(mktemp -d /tmp/rko-job.XXXXXX) || exit 125
cd "$directory" || exit 125
limit=$1
shift
LC_ALL=C timeout --verbose --signal=TERM --kill-after=1s "$limit" "$@"
status=$?
cd /tmp || exit 125
rm -rf -- "$directory" || exit 125
exit "$status"
"""


class PersistentPodmanRunner(PodmanRunner):
    """Reuse a bounded container but never a decoder interpreter or working directory.

    Responsibilities:
    - Preserve existing rootless restrictions and hard per-job process supervision.
    - Retire every failed invocation's container before permitting another request.

    Attributes:
        transport (str): Explicit persistent_exec measurement transport identity.
        lease_seconds (float): Hard maximum container lifetime, even after controller death.
        startup_cpu_charge (float): Conservative measured creation/policy-check wall-time charges.
        needs_restart (bool): Whether a failed invocation requires a new qualified lease.
    """

    transport = "persistent_exec"

    def __init__(self, image: str, *, lease_seconds: float = 600, memory: int = 536870912) -> None:
        """Pin runtime and validate an explicit bounded lease without starting decoder code.

        Args:
            image (str): Existing immutable runtime image/name.
            lease_seconds (float): Between sixty seconds and one day.
            memory (int): Existing rootless memory limit.

        Raises:
            ValueError: If lease or base settings are invalid.
            SandboxError: If rootless execution cannot be inspected.
        """
        if not math.isfinite(lease_seconds) or not 60 <= lease_seconds <= 86400:
            raise ValueError("Session lease must be between sixty seconds and one day.")
        super().__init__(image, memory)
        self.lease_seconds = lease_seconds
        self.startup_cpu_charge = 0.0
        self._host_directory = tempfile.TemporaryDirectory(prefix="rko-evolve-session-")
        self._container: str | None = None
        self._expires = 0.0
        self.needs_restart = False

    def __enter__(self) -> PersistentPodmanRunner:
        """Start the leased runtime; callers still must pass live qualification.

        Returns:
            PersistentPodmanRunner: Active but not automatically qualified transport.

        Raises:
            SandboxError: If creation or effective policy verification fails.
        """
        self._ensure_started()
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Remove the exact owned session, never unrelated containers.

        Args:
            _exception_type (type | None): Propagating exception type, intentionally unused.
            exception (BaseException | None): Propagating exception instance.
            traceback (TracebackType | None): Original traceback.

        Raises:
            SandboxError: If cleanup cannot be confirmed.
        """
        self.qualified = False
        self._stop()
        self._host_directory.cleanup()

    def qualify(self) -> dict[str, Any]:
        """Qualify the actual exec path and clear restart state only after every probe passes.

        Returns:
            dict[str, Any]: Live policy/resource enforcement evidence.

        Raises:
            SandboxError: If the exec supervisor or isolation controls fail.
        """
        report = super().qualify()
        self.needs_restart = False
        return {
            **report,
            "transport": self.transport,
            "startup_cpu_upper_bound": self.startup_cpu_charge,
        }

    def _ensure_started(self) -> None:
        """Create an idle leased container and validate real flags before any source job.

        Raises:
            SandboxError: If startup, GNU timeout or namespace checks fail.
        """
        if self._container is not None:
            return
        started = time.monotonic()
        name = "rko-evolve-session-" + uuid.uuid4().hex
        command = self._container_command(name)
        prefix = command[: command.index(self.image)]
        launch = prefix + [
            "--detach",
            "--init",
            self.image,
            "timeout",
            "--signal=TERM",
            "--kill-after=1s",
            str(self.lease_seconds),
            "sleep",
            str(self.lease_seconds),
        ]
        self._container = name
        self._expires = started + self.lease_seconds
        try:
            self._command(launch[1:], cwd=self._host_directory.name)
            report = self._execute({"probe": "policy"}, 20)
            self._validate_policy(report)
            version = self._command(["exec", name, "timeout", "--version"])
            if "GNU coreutils" not in version:
                raise SandboxError("Persistent execution requires qualified GNU timeout semantics.")
        except (OSError, subprocess.SubprocessError, SandboxError) as error:
            self.qualified = False
            self._stop()
            raise SandboxError("Cannot start/qualify leased isolated runtime.") from error
        finally:
            self.startup_cpu_charge += time.monotonic() - started

    def _stop(self) -> None:
        """Remove the leased container and disable the runner on unconfirmed cleanup.

        Raises:
            SandboxError: If Podman cannot confirm removal.
        """
        if self._container is None:
            return
        try:
            self._command(["rm", "--force", "--ignore", "--time=1", self._container])
        except (OSError, subprocess.SubprocessError) as error:
            self.qualified = False
            raise SandboxError("Session cleanup unconfirmed; runner disabled.") from error
        self._container = None

    def _invocation(self, job: dict[str, Any], name: str, timeout: float) -> list[str]:
        """Execute a fixed guard and a fresh worker, passing source exclusively through stdin.

        Args:
            job (dict[str, Any]): Trusted job envelope, never shell source.
            name (str): Base transport invocation identity.
            timeout (float): Worker guardian deadline, excluding host I/O grace.

        Returns:
            list[str]: Explicit command vector with safely separated guard arguments.
        """
        self._ensure_started()
        if self._container is None:
            raise SandboxError("Leased runtime did not start.")
        if self._expires - time.monotonic() < max(1.0, timeout) + 2:
            self.qualified = False
            raise SandboxError(
                "Session lease cannot admit this job; create a fresh qualified lease."
            )
        worker = ["python", "-m", "rko_solver.experimental.self_evolve.worker"]
        if job.get("probe") == "policy":
            worker = ["python", "-c", _POLICY_PROBE]
        return [
            "podman",
            "exec",
            "--interactive",
            self._container,
            "sh",
            "-c",
            _EXEC_GUARD,
            "rko-job",
            str(max(1.0, timeout)),
            *worker,
        ]

    def _cleanup_invocation(self, name: str) -> None:
        """Retain a healthy leased container; failed requests are retired by _execute.

        Args:
            name (str): Base invocation identity; no per-job container exists.
        """
        return None

    def _exit_error(self, job: dict[str, Any], code: int, output: bytes) -> SandboxError:
        """Distinguish GNU guardian expiration from actual memory/process failures.

        Args:
            job (dict[str, Any]): Trusted envelope.
            code (int): Actual exec exit status.
            output (bytes): Bounded guardian/worker output.

        Returns:
            SandboxError: Explicitly classified stop, never a guessed generic-message deadline.
        """
        if code == 124 or b"timeout: sending signal" in output:
            return SandboxError(
                "External worker deadline exceeded.", reason="deadline", phase="worker_guardian"
            )
        return super()._exit_error(job, code, output)

    def _execute(self, job: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Apply bounded I/O and retire the container on every non-normal execution path.

        Args:
            job (dict[str, Any]): Ordinary source-only job or qualification probe.
            timeout (float): Guardian deadline, excluding ten seconds of exec launch grace.

        Returns:
            dict[str, Any]: Unchanged worker/probe record.

        Raises:
            SandboxError: If execution or container cleanup fails.
        """
        if self.needs_restart and self.qualified:
            raise SandboxError("Retired session requires explicit requalification before reuse.")
        try:
            return super()._execute(job, timeout)
        except BaseException as error:
            host_timeout = isinstance(error, SandboxError) and error.phase == "host_io"
            if host_timeout:
                self.qualified = False
            self.needs_restart = True
            self._stop()
            if host_timeout and "probe" in job:
                raise SandboxError(
                    "Host transport timeout cannot qualify worker enforcement.", phase="host_io"
                ) from error
            raise
