"""Fail-closed rootless Podman execution for experimental decoder jobs.

Objective:
- Bound generated-code execution outside the trusted Python controller.
Technical approach:
- Pin a local image ID, strip host environment, impose cgroup/filesystem/network limits.
- Drain bounded output and enforce external deadlines with container/process-group cleanup.
Flow (pseudocode):
1. Inspect rootless capability and pin the explicitly supplied local image.
2. Qualify effective restrictions and adversarial timeout/output/memory behavior.
3. Execute bounded JSON jobs and independently verify returned witnesses on the host.
Important criteria:
- No raw decoder import or unsafe subprocess fallback is provided.
- Same-interpreter worker integrity is not guaranteed against hostile Python.
Inputs/Outputs:
- Input: a prebuilt local image, artifact and validated instance.
- Output: bounded evidence plus independently recomputed final solution semantics.
"""

from __future__ import annotations

import json
import math
import os
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from .artifacts import DecoderArtifact
from .model_data import ModelProblem
from .problems import Problem
from .sandbox_evidence import SandboxError
from .sandbox_evidence import WorkerEvidence
from typing import Any

JOB_GRACE_SECONDS = 60
_POLICY_PROBE = """
import json
import os
from pathlib import Path
mounts = [line.split() for line in Path('/proc/mounts').read_text().splitlines()]
root = next(row for row in mounts if row[1] == '/')
tmp = next(row for row in mounts if row[1] == '/tmp')
status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
print(json.dumps({
    'uid': os.getuid(),
    'working_directory': os.getcwd(),
    'memory': int(Path('/sys/fs/cgroup/memory.max').read_text()),
    'pids': int(Path('/sys/fs/cgroup/pids.max').read_text()),
    'cpu': Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
    'interfaces': sorted(p.name for p in Path('/sys/class/net').iterdir()),
    'read_only': 'ro' in root[3].split(','),
    'tmpfs_restricted': (
        tmp[2] == 'tmpfs' and {'noexec', 'nosuid', 'nodev'}.issubset(tmp[3].split(','))
    ),
    'capabilities_dropped': all(
        int(status[key].strip(), 16) == 0 for key in ('CapEff', 'CapPrm', 'CapBnd')
    ),
    'no_new_privileges': status['NoNewPrivs'].strip() == '1',
}))
"""


class PodmanRunner:
    """Own one pinned rootless runtime and require qualification before candidate execution.

    Responsibilities:
    - Enforce external limits and independently check returned witnesses.

    Attributes:
        image (str): Immutable local image ID.
        memory (int): Job memory/swap ceiling in bytes.
        qualified (bool): Whether this runner passed live qualification.
    """

    def __init__(self, image: str, memory: int = 536870912) -> None:
        """Pin an existing image and inspect rootless cgroup capability without pulling.

        Args:
            image (str): Explicit local Podman image name or ID.
            memory (int): Memory ceiling between 128 MiB and 2 GiB.

        Raises:
            SandboxError: If rootless cgroup isolation or the image is unavailable.
            ValueError: If limits are outside the experimental envelope.
        """
        if sys.platform != "linux":
            raise SandboxError(
                "Live decoder execution currently requires Linux/WSL rootless Podman."
            )
        if not isinstance(image, str) or not image or image.startswith("-"):
            raise ValueError("An explicit local image name or ID is required.")
        if type(memory) is not int or not 134217728 <= memory <= 2147483648:
            raise ValueError("Unsupported sandbox memory limit.")
        self.environment = self._environment()
        self.image = self._inspect_image(image)
        self.memory, self.qualified = memory, False

    @staticmethod
    def _environment() -> dict[str, str]:
        """Build the minimal host environment, excluding WSL-mounted executable paths.

        Returns:
            dict[str, str]: Administrative and invocation environment.
        """
        environment = {
            key: os.environ[key]
            for key in ("PATH", "HOME", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")
            if key in os.environ
        }
        if any(key in os.environ for key in ("WSL_INTEROP", "WSL_DISTRO_NAME")):
            environment["PATH"] = (
                os.pathsep.join(
                    entry
                    for entry in environment.get("PATH", os.defpath).split(os.pathsep)
                    if not re.match(r"^/mnt/[A-Za-z](?:/|$)", entry)
                )
                or os.defpath
            )
        return environment

    def _inspect_image(self, image: str) -> str:
        """Require rootless resource controllers and resolve an immutable local image ID.

        Args:
            image (str): Explicit local image name or ID.

        Returns:
            str: Canonical SHA-256 image identity.

        Raises:
            SandboxError: If runtime inspection or immutable image resolution fails.
        """
        try:
            info = json.loads(self._command(["info", "--format=json"]))
            host = info["host"]
            if not host["security"]["rootless"] or host["cgroupVersion"] != "v2":
                raise SandboxError("Rootless Podman with cgroup v2 is required.")
            if not {"cpu", "memory", "pids"}.issubset(host["cgroupControllers"]):
                raise SandboxError("CPU, memory and PID controllers are required.")
            resolved = self._command(["image", "inspect", "--format={{.Id}}", image]).strip()
        except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError) as error:
            raise SandboxError("Cannot inspect rootless Podman/local runtime image.") from error
        digest = resolved.removeprefix("sha256:")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise SandboxError("Runtime image did not resolve to an immutable image ID.")
        return "sha256:" + digest

    def _command(self, args: list[str], *, cwd: str | None = None) -> str:
        """Run a bounded administrative Podman command with a minimal environment.

        Args:
            args (list[str]): Trusted command arguments, never shell text.
            cwd (str | None): Optional private administrative working directory.

        Returns:
            str: Standard output on success.

        Raises:
            OSError: If Podman cannot launch.
            subprocess.SubprocessError: If the command fails or times out.
        """
        return subprocess.run(
            ["podman", *args],
            env=self.environment,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout

    def _container_command(self, name: str) -> list[str]:
        """Build the shared immutable isolation policy for a named container.

        Args:
            name (str): Controller-generated container identity.

        Returns:
            list[str]: Explicit argument vector, never shell-interpolated source.
        """
        return [
            "podman",
            "run",
            "--pull=never",
            "--rm",
            "--interactive",
            "--name",
            name,
            "--network=none",
            "--read-only",
            "--cap-drop=all",
            "--security-opt=no-new-privileges",
            "--user=65534:65534",
            f"--memory={self.memory}",
            f"--memory-swap={self.memory}",
            "--cpus=1",
            "--pids-limit=32",
            "--log-driver=none",
            "--ulimit=nofile=64:64",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16777216",
            "--env=OPENBLAS_NUM_THREADS=1",
            "--env=OMP_NUM_THREADS=1",
            "--env=PYTHONDONTWRITEBYTECODE=1",
            self.image,
            "python",
            "-m",
            "rko_solver.experimental.self_evolve.worker",
        ]

    def _invocation(self, job: dict[str, Any], name: str, timeout: float) -> list[str]:
        """Select a worker or controller-owned policy probe under the same isolation flags.

        Args:
            job (dict[str, Any]): Trusted job envelope.
            name (str): Unique container identity.
            timeout (float): External deadline, used by specialized transports.

        Returns:
            list[str]: Complete launch argument vector.
        """
        command = self._container_command(name)
        if job.get("probe") == "policy":
            command[-3:] = ["python", "-c", _POLICY_PROBE]
        position = command.index(self.image) + 1
        command[position:position] = [
            "timeout",
            "--verbose",
            "--signal=TERM",
            "--kill-after=1s",
            str(timeout),
        ]
        return command

    def _cleanup_invocation(self, name: str) -> None:
        """Remove this invocation's named container and confirm administrative success.

        Args:
            name (str): Exact controller-generated container name.

        Raises:
            subprocess.SubprocessError: If cleanup fails or exceeds its deadline.
            OSError: If Podman is unavailable.
        """
        self._command(["rm", "--force", "--ignore", "--time=1", name])

    def _exit_error(self, job: dict[str, Any], code: int, output: bytes) -> SandboxError:
        """Classify controlled process exits without interpreting message substrings as deadlines.

        Args:
            job (dict[str, Any]): Trusted probe/job envelope.
            code (int): Actual process exit code.
            output (bytes): Bounded captured output.

        Returns:
            SandboxError: Typed resource/protocol diagnostic.
        """
        if code == 124 or b"timeout: sending signal" in output:
            return SandboxError(
                "External worker deadline exceeded.", reason="deadline", phase="worker_guardian"
            )
        memory_stop = job.get("probe") == "memory" and (
            code in (137, -9) or (code == 1 and b"MemoryError" in output)
        )
        return SandboxError(
            f"Worker exited with status {code}.", reason="memory" if memory_stop else "exit"
        )

    def _execute(self, job: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Execute a bounded input record with hard timeout and bounded output collection.

        Args:
            job (dict[str, Any]): JSON-compatible trusted job envelope.
            timeout (float): External wall-time ceiling in seconds.

        Returns:
            dict[str, Any]: Parsed worker record.

        Raises:
            SandboxError: If deadline, output limit, exit status or JSON contract fails.
        """
        payload = json.dumps(job, allow_nan=False).encode("utf-8")
        if len(payload) > 65536:
            raise SandboxError("Worker input exceeds 64 KiB.")
        name = "rko-evolve-" + uuid.uuid4().hex
        command = self._invocation(job, name, timeout)
        timeout += 10
        started = time.monotonic()
        with (
            tempfile.TemporaryDirectory(prefix="rko-evolve-") as directory,
            tempfile.TemporaryFile() as input_file,
        ):
            input_file.write(payload)
            input_file.seek(0)
            try:
                process = subprocess.Popen(
                    command,
                    stdin=input_file,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=self.environment,
                    cwd=directory,
                    start_new_session=True,
                )
            except OSError as error:
                self.qualified = False
                raise SandboxError("Cannot launch isolated runtime; runner disabled.") from error
            try:
                output = self._drain_output(process, started, timeout)
                code = process.wait(timeout=max(0.1, timeout - (time.monotonic() - started)))
                if code != 0:
                    raise self._exit_error(job, code, bytes(output))
                record = json.loads(output)
                if not isinstance(record, dict):
                    raise SandboxError("Worker record is not a JSON object.")
                return record
            except subprocess.TimeoutExpired as error:
                self.qualified = False
                raise SandboxError(
                    "External worker deadline exceeded.", reason="deadline", phase="host_io"
                ) from error
            except (ValueError, RecursionError) as error:
                raise SandboxError(
                    "Invalid or truncated worker output.", reason="protocol"
                ) from error
            finally:
                self._finish_process(process, name)

    def _drain_output(
        self, process: subprocess.Popen[bytes], started: float, timeout: float
    ) -> bytearray:
        """Drain merged output within the byte cap and host I/O deadline.

        Args:
            process (subprocess.Popen[bytes]): Launched isolated invocation.
            started (float): Monotonic launch start.
            timeout (float): Host deadline including transport grace.

        Returns:
            bytearray: Bounded merged output.

        Raises:
            SandboxError: If host time or output exceeds its limit.
            OSError: If reading the process pipe fails.
        """
        output = bytearray()
        if process.stdout is None:
            raise SandboxError("Worker output pipe is unavailable.")
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                if time.monotonic() - started > timeout:
                    self.qualified = False
                    raise SandboxError(
                        "External worker deadline exceeded.", reason="deadline", phase="host_io"
                    )
                for key, _ in selector.select(timeout=0.05):
                    chunk = os.read(key.fd, 16384)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    output.extend(chunk)
                    if len(output) > 1048576:
                        raise SandboxError("Worker output exceeds 1 MiB.", reason="output")
        return output

    def _finish_process(self, process: subprocess.Popen[bytes], name: str) -> None:
        """Reap the host process and confirm container cleanup on every exit path.

        Args:
            process (subprocess.Popen[bytes]): Owned invocation process.
            name (str): Exact controller-generated container identity.

        Raises:
            SandboxError: If process or container cleanup cannot be confirmed.
            OSError: If process-group termination fails unexpectedly.
        """
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        cleanup_error: OSError | subprocess.SubprocessError | None = None
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired as error:
            cleanup_error = error
        if process.stdout is not None:
            process.stdout.close()
        try:
            self._cleanup_invocation(name)
        except (OSError, subprocess.SubprocessError) as error:
            cleanup_error = error
        if cleanup_error is not None:
            self.qualified = False
            raise SandboxError(
                "Cannot confirm container cleanup; runner disabled."
            ) from cleanup_error

    def _validate_policy(self, report: dict[str, Any]) -> None:
        """Reject ineffective runtime restrictions using actual namespace/cgroup evidence.

        Args:
            report (dict[str, Any]): Controller-owned policy probe response.

        Raises:
            SandboxError: If any required control is absent or mismatched.
        """
        if (
            report.get("uid") != 65534
            or report.get("memory") != self.memory
            or report.get("pids") != 32
            or report.get("cpu") != "100000 100000"
            or report.get("interfaces") != ["lo"]
            or not report.get("read_only")
            or not report.get("capabilities_dropped")
            or not report.get("tmpfs_restricted")
            or not report.get("no_new_privileges")
        ):
            raise SandboxError("Effective container restrictions do not match the policy.")

    def qualify(self) -> dict[str, Any]:
        """Test actual restrictions, timeout, output flooding and memory termination.

        Returns:
            dict[str, Any]: Effective runtime policy evidence.

        Raises:
            SandboxError: If any restriction/probe is unavailable or ineffective.
        """
        self.qualified = False
        report = self._execute({"probe": "policy"}, 20)
        self._validate_policy(report)
        for probe, expected in (("hang", "deadline"), ("output", "output"), ("memory", "memory")):
            try:
                self._execute({"probe": probe}, 3 if probe == "hang" else 20)
            except SandboxError as error:
                if error.reason != expected or (
                    probe == "hang" and error.phase != "worker_guardian"
                ):
                    raise SandboxError(
                        f"Unexpected {probe} qualification failure: {error}"
                    ) from error
            else:
                raise SandboxError(f"The {probe} probe was not externally stopped.")
        pids = self._execute({"probe": "pids"}, 20)
        if not pids.get("pid_limit_enforced") or not 0 < pids.get("spawned", 0) < 32:
            raise SandboxError("PID exhaustion was not enforced.")
        self._execute({"probe": "policy"}, 20)
        self.qualified = True
        return {"image": self.image, "policy": report, "adversarial_probes": "passed"}

    def run(
        self,
        artifact: DecoderArtifact,
        problem: Problem | ModelProblem,
        *,
        mode: str = "rko",
        seconds: float = 0.1,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Run an artifact or control in isolation and independently verify its final witness.

        Args:
            artifact (DecoderArtifact): Existing validated seed/candidate.
            problem (Problem): Compatible trusted instance.
            mode (str): rko, random, fixed or constructive control.
            seconds (float): Entire inner invocation allowance, including decoder work.
            seed (int): Nonnegative solver/control seed.

        Returns:
            dict[str, Any]: Evidence with parent-recomputed final objective/feasibility.

        Raises:
            SandboxError: If isolation, execution or returned evidence is invalid.
            ValueError: If mode, seed, duration or family is incompatible.
        """
        if not self.qualified:
            raise SandboxError("Runner must pass live isolation qualification before execution.")
        if artifact.qualified_image is not None and artifact.qualified_image != self.image:
            raise ValueError("Artifact requires its qualified pinned runtime image.")
        self._validate_job(artifact, problem, mode, seconds, seed)
        outer_started = time.monotonic()
        record = self._execute(
            {
                "source": artifact.source,
                "family": problem.family,
                "data": problem.data,
                "mode": mode,
                "seconds": seconds,
                "seed": seed,
            },
            seconds + 15,
        )
        if record.get("status") != "ok":
            raise SandboxError(
                "Worker rejected candidate: " + str(record.get("error", "unknown"))[:500]
            )
        evidence = self._verify_record(record, problem, seconds)
        return {
            **record,
            **evidence,
            "outer_wall_seconds": time.monotonic() - outer_started,
            "candidate": artifact.digest,
            "image": self.image,
            "seed": seed,
            "mode": mode,
            "family": problem.family,
        }

    def _validate_job(
        self,
        artifact: DecoderArtifact,
        problem: Problem | ModelProblem,
        mode: str,
        seconds: float,
        seed: int,
    ) -> None:
        """Validate compatible settings and source identity before launching a job.

        Args:
            artifact (DecoderArtifact): Validated source artifact.
            problem (Problem): Trusted compatible instance.
            mode (str): Requested solver or control mode.
            seconds (float): Inner wall-time allowance.
            seed (int): Solver seed.

        Raises:
            ValueError: If settings or artifact identity are invalid.
        """
        if (
            mode not in {"rko", "random", "fixed", "constructive", "check"}
            or not math.isfinite(seconds)
            or not 0 < seconds <= 30
            or type(seed) is not int
            or not 0 <= seed < 2**32
            or artifact.family != problem.family
        ):
            raise ValueError("Incompatible family or invalid worker settings.")
        if isinstance(problem, ModelProblem) and mode == "constructive":
            raise ValueError("No generic independent human model baseline was supplied.")
        checked = DecoderArtifact.create(artifact.family, artifact.source, artifact.parent)
        if checked.digest != artifact.digest:
            raise ValueError("Artifact source hash mismatch.")

    def _verify_record(
        self, record: dict[str, Any], problem: Problem | ModelProblem, seconds: float
    ) -> dict[str, Any]:
        """Verify claims in order, disabling on integrity but not allowance failures.

        Args:
            record (dict[str, Any]): Parsed worker response.
            problem (Problem): Trusted instance evaluator.
            seconds (float): Inner invocation allowance.

        Returns:
            dict[str, Any]: Independently recomputed witness evidence.

        Raises:
            SandboxError: If witness, resources, runtime or counters are invalid.
        """
        verifier = WorkerEvidence(record)
        try:
            evidence = verifier.witness(problem)
            verifier.resources()
        except SandboxError:
            self.qualified = False
            raise
        if record["wall_seconds"] > seconds + 0.02:
            raise SandboxError("Decoder exceeded the inner runtime allowance.")
        try:
            verifier.counters()
        except SandboxError:
            self.qualified = False
            raise
        return evidence
