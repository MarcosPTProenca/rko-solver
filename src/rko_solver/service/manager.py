"""Bounded single-job supervision, process-group cleanup and ephemeral retention.

Objective:
- Admit at most one optimizer independently of HTTP request lifetime and responsiveness.

Technical approach:
- One supervisor thread owns spawn/handshake/deadline/whole-group termination.
- Retain bounded private JSON/artifacts with TTL/count eviction and restart interruption records.
- Lock a dedicated storage directory so extra HTTP workers cannot multiply job admission.

Flow (pseudocode):
1. Acquire private storage ownership and mark unfinished old jobs interrupted.
2. Admit one schema-validated request or reject while busy.
3. Spawn an owned process group, authorize work and monitor bounded IPC/deadlines.
4. Terminate the entire group before reaping its reserved leader PID.
5. Preserve numerical snapshots, remove scratch work and expire retained records/artifacts.

Important criteria:
- No queue, unbounded all-results dictionary, durable resume or exactly-once claim.
- PID/group ownership is checked before work and never inferred from a client value.
- The reference lifecycle is Linux-only; container replacement can erase tmpfs records.

Inputs/Outputs:
- Input: installed adapter, server-owned storage/limits and validated JSON submissions.
- Output: bounded job views, safe artifact paths and explicit terminal states.
"""

import json
import logging
import multiprocessing as mp
import os
import pickle
import shutil
import signal
import sys
import threading
import time
from .models import JobSubmission
from .models import ServiceAdapter
from .models import ServiceLimits
from .worker import JobWorker
from collections import deque
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID
from uuid import uuid4

if sys.platform == "linux":
    import fcntl


@dataclass
class JobRecord:
    """Retain small primitive lifecycle metadata without request values or callback objects.

    Responsibilities:
    - Persist interruption/retention state separately from bounded result files.

    Attributes:
        id (str): Server-generated UUID.
        state (str): Explicit queued/running/terminal lifecycle state.
        created (float): UTC epoch for restart-aware age tracking.
        finished (float | None): Terminal epoch, or None before completion.
        error (str | None): Safe phase/category, never callback exception text.
        cancel (bool): Supervisor cancellation request.
    """

    id: str
    state: str
    created: float
    finished: float | None = None
    error: str | None = None
    cancel: bool = False


class AdmissionError(ValueError):
    """Represent bounded capacity refusal rather than starting another optimizer.

    Responsibilities:
    - Allow HTTP mapping to backpressure without leaking application details.

    Attributes:
        None: The exception contains no instance data.
    """


class JobManager:
    """Own one supervised job and bounded private single-tenant metadata/artifacts.

    Responsibilities:
    - Keep spawn and callback CPU work outside HTTP handlers/event loops.
    - Enforce deadlines, cancellation, shutdown and bounded retention honestly.

    Attributes:
        adapter (ServiceAdapter): Trusted installed schema/factory.
        limits (ServiceLimits): Server policy, independent of client settings.
        storage (Path): Dedicated private server-owned directory.
        instance_id (str): New server incarnation, not a durable recovery identifier.
        records (dict[str, JobRecord]): Bounded lifecycle records, not retained input payloads.
    """

    terminal = frozenset(("completed", "failed", "cancelled", "timed_out", "interrupted"))

    def __init__(self, adapter: ServiceAdapter, limits: ServiceLimits, storage: Path) -> None:
        """Initialize inert supervision state without starting HTTP or optimizer work.

        Args:
            adapter (ServiceAdapter): Trusted startup binding.
            limits (ServiceLimits): Explicit server resource/retention policy.
            storage (Path): Dedicated owner-controlled path, never a request field.

        Raises:
            ValueError: If the trusted factory cannot be serialized for spawn.
        """
        try:
            encoded = pickle.dumps(adapter)
        except Exception as exc:
            raise ValueError(
                "Service adapter/schema must be importable and spawn-pickleable."
            ) from exc
        if len(encoded) > 65_536:
            raise ValueError("Adapter should bind configuration, not large instance data.")
        if storage.is_symlink():
            raise ValueError("Service storage cannot be a symlink.")
        self.adapter, self.limits, self.storage = adapter, limits, storage.resolve()
        self.failed = False
        self.instance_id = str(uuid4())
        self.records: dict[str, JobRecord] = {}
        self._expired: deque[str] = deque(maxlen=limits.max_records)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._pending: tuple[str, JobSubmission] | None = None
        self._active: str | None = None
        self._thread: threading.Thread | None = None
        self._storage_lock = None

    def start(self) -> None:
        """Acquire one-worker storage ownership and recover only bounded lifecycle metadata.

        Raises:
            RuntimeError: If Linux ownership/single-worker requirements are not satisfied.
            OSError: If private storage cannot be created or inspected.
        """
        if sys.platform != "linux":
            raise RuntimeError(
                "Supervised service requires Linux; use the Linux container on Windows/WSL."
            )
        self.storage.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.storage.is_symlink() or self.storage.stat().st_uid != os.getuid():
            raise RuntimeError("Service storage must be an owned private directory.")
        marker = self.storage / ".rko-service"
        if marker.is_symlink():
            raise RuntimeError("Invalid private storage marker.")
        if marker.exists():
            if marker.read_text(encoding="utf-8") != "rko-service-v1":
                raise RuntimeError("Unknown storage format; refusing to alter existing files.")
        else:
            if any(self.storage.iterdir()):
                raise RuntimeError("Refusing to reuse an unrelated/nonempty storage directory.")
            marker.write_text("rko-service-v1", encoding="utf-8")
        if (self.storage / "instance.lock").is_symlink():
            raise RuntimeError("Invalid private storage lock.")
        self.storage.chmod(0o700)
        self._storage_lock = (self.storage / "instance.lock").open("a")
        try:
            fcntl.flock(self._storage_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._storage_lock.close()
            raise RuntimeError("Use one HTTP worker per service storage/container.") from exc
        for directory in sorted(self.storage.iterdir()):
            if not directory.is_dir() or directory.is_symlink():
                continue
            try:
                if str(UUID(directory.name)) != directory.name:
                    continue
            except ValueError:
                continue
            try:
                path = directory / "state.json"
                if path.stat().st_size > 4096:
                    raise ValueError("Oversized state")
                data = json.loads(path.read_text())
                record = JobRecord(**data)
                if record.id != directory.name:
                    raise ValueError("State identity mismatch")
                if record.state not in self.terminal:
                    record.state, record.error, record.finished = (
                        "interrupted",
                        "server_restart",
                        time.time(),
                    )
                record.cancel = False
                self.records[record.id] = record
                shutil.rmtree(directory / "work", ignore_errors=True)
                self._persist(record)
            except ValueError, OSError, TypeError:
                shutil.rmtree(directory)
        self._prune()
        while len(self.records) > self.limits.max_records:
            self._evict(min(self.records.values(), key=lambda item: item.created).id)
        self._thread = threading.Thread(
            target=self._loop, name="rko-service-supervisor", daemon=True
        )
        self._thread.start()

    def _persist(self, record: JobRecord) -> None:
        """Atomically store bounded lifecycle data, never request values.

        Args:
            record (JobRecord): Owned job state.

        Raises:
            OSError: If private state storage fails.
        """
        directory = self.storage / record.id
        temporary = directory / "state.tmp"
        temporary.write_text(json.dumps(asdict(record)), encoding="utf-8")
        temporary.replace(directory / "state.json")

    def _evict(self, identifier: str) -> None:
        """Remove one owned terminal record/artifact directory and bound its tombstone.

        Args:
            identifier (str): Server-generated stored job ID.
        """
        self.records.pop(identifier, None)
        shutil.rmtree(self.storage / identifier, ignore_errors=True)
        self._expired.append(identifier)

    def _prune(self) -> None:
        """Expire terminal records using a bounded scan and the explicit upper TTL."""
        now = time.time()
        for record in list(self.records.values()):
            if (
                record.finished is not None
                and now - record.finished >= self.limits.retention_seconds
            ):
                self._evict(record.id)

    def submit(self, submission: JobSubmission) -> str:
        """Admit one validated instance or reject busy service without a waiting queue.

        Args:
            submission (JobSubmission): Schema-validated data/options, not Python code.

        Returns:
            str: Server-generated job UUID.

        Raises:
            AdmissionError: If active work or shutdown prevents admission.
            OSError: If private lifecycle storage cannot be created.
        """
        with self._lock:
            self._prune()
            if (
                self._active is not None
                or self._stop.is_set()
                or self._thread is None
                or self.failed
            ):
                raise AdmissionError("Service busy or unavailable.")
            while len(self.records) >= self.limits.max_records:
                self._evict(min(self.records.values(), key=lambda item: item.created).id)
            identifier = str(uuid4())
            (self.storage / identifier).mkdir(mode=0o700)
            record = JobRecord(identifier, "queued", time.time())
            self.records[identifier] = record
            self._persist(record)
            self._active = identifier
            self._pending = (identifier, submission)
            self._wake.set()
            return identifier

    def result(self, identifier: str) -> dict[str, object] | None:
        """Load a bounded primitive snapshot, rejecting unsafe or partial artifact paths.

        Args:
            identifier (str): Known server job ID.

        Returns:
            dict[str, object] | None: Numeric snapshot when present and within policy.
        """
        path = self.storage / identifier / "result.json"
        try:
            if path.is_symlink() or path.stat().st_size > self.limits.max_json_bytes:
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) and "numerical" in data else None
        except OSError, ValueError:
            return None

    def view(self, identifier: str) -> dict[str, object]:
        """Return safe lifecycle/numerical data without stored input values or private paths.

        Args:
            identifier (str): Server-generated UUID.

        Returns:
            dict[str, object]: Bounded job state, or an explicit expired state.

        Raises:
            KeyError: If the ID is unknown in this server/store incarnation.
        """
        with self._lock:
            self._prune()
            if identifier in self._expired:
                return {"id": identifier, "state": "expired", "instance_id": self.instance_id}
            record = self.records[identifier]
            return {
                "id": identifier,
                "state": record.state,
                "created": record.created,
                "finished": record.finished,
                "error": record.error,
                "instance_id": self.instance_id,
                "result": self.result(identifier) if record.state == "completed" else None,
            }

    def cancel(self, identifier: str) -> dict[str, object]:
        """Request actual supervisor termination, not merely a successful HTTP response.

        Args:
            identifier (str): Known job UUID.

        Returns:
            dict[str, object]: Current state; clients poll until terminal.

        Raises:
            KeyError: If the ID is unknown.
        """
        with self._lock:
            record = self.records[identifier]
            if record.state not in self.terminal:
                record.cancel = True
                self._wake.set()
            return self.view(identifier)

    def artifact(self, identifier: str, name: str) -> bytes:
        """Read fixed bounded completed artifacts atomically with respect to TTL eviction.

        Args:
            identifier (str): Known completed job UUID.
            name (str): Fixed result.json or report.html name, not a user filesystem path.

        Returns:
            bytes: Bounded attachment content without any exposed filesystem path.

        Raises:
            KeyError: If the job/artifact is absent, expired, unsafe or incomplete.
        """
        with self._lock:
            self._prune()
            if (
                name not in ("result.json", "report.html")
                or self.records[identifier].state != "completed"
            ):
                raise KeyError(identifier)
            path = self.storage / identifier / name
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size > self.limits.max_artifact_bytes
            ):
                raise KeyError(name)
            with path.open("rb") as handle:
                content = handle.read(self.limits.max_artifact_bytes + 1)
            if len(content) > self.limits.max_artifact_bytes:
                raise KeyError(name)
            return content

    @staticmethod
    def _terminate(process: mp.Process, grouped: bool) -> None:
        """Terminate the owned group before reaping its leader, including lingering descendants.

        Args:
            process (mp.Process): Only the process created for this job; not yet polled/reaped.
            grouped (bool): Whether the readiness handshake proved session/group ownership.
        """
        pid = process.pid
        if pid is None:
            return
        try:
            grouped = grouped or os.getpgid(pid) == pid
            if grouped and pid > 1 and pid != os.getpgrp():
                os.killpg(pid, signal.SIGTERM)
                time.sleep(0.1)
                os.killpg(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.join(timeout=2)
        process.close()

    def _execute(self, identifier: str, submission: JobSubmission) -> None:
        """Spawn/monitor one job without event-loop CPU work or unbounded IPC reads.

        Args:
            identifier (str): Owned job ID.
            submission (JobSubmission): Validated private data discarded after this method.
        """
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        directory = self.storage / identifier
        process = context.Process(
            target=JobWorker.run,
            args=(
                self.adapter,
                submission.instance,
                submission.options.model_dump(mode="json"),
                child,
                directory,
                self.limits,
            ),
        )
        grouped = False
        state, error = "failed", "worker_failed"
        started = time.monotonic()
        try:
            process.start()
            child.close()
            ready = False
            while True:
                with self._lock:
                    self._prune()
                    cancelled = self.records[identifier].cancel
                if self._stop.is_set():
                    state, error = "interrupted", "server_shutdown"
                    break
                if cancelled:
                    state, error = "cancelled", "cancelled"
                    break
                allowance = (
                    self.limits.max_wall_seconds
                    if ready
                    else min(self.limits.startup_seconds, self.limits.max_wall_seconds)
                )
                if time.monotonic() - started > allowance:
                    state, error = "timed_out", "job_deadline"
                    break
                if not parent.poll(0.05):
                    continue
                message = parent.recv_bytes(self.limits.max_json_bytes)
                if message == b"ready" and not ready:
                    if os.getpgid(process.pid) != process.pid or process.pid == os.getpgrp():
                        raise RuntimeError("Job process ownership handshake failed.")
                    grouped, ready = True, True
                    with self._lock:
                        self.records[identifier].state = "running"
                        self._persist(self.records[identifier])
                    parent.send_bytes(b"run")
                elif message == b"done":
                    state, error = "completed", None
                    break
                elif message != b"numerical":
                    data = json.loads(message)
                    error = data.get("error", "worker_failed")
                    if error not in (
                        "startup_failed",
                        "instance_failed",
                        "optimization_failed",
                        "postprocessing_failed",
                    ):
                        error = "worker_failed"
                    break
        except OSError, EOFError, ValueError, RuntimeError:
            state, error = "failed", "worker_failed"
        finally:
            self._terminate(process, grouped)
            parent.close()
            child.close()
            shutil.rmtree(directory / "work", ignore_errors=True)
        with self._lock:
            snapshot = self.result(identifier)
            if snapshot is not None:
                if state != "completed":
                    snapshot["postprocessing"] = {"state": state, "code": error}
                    snapshot["report"] = {"state": "failed", "code": "postprocessing_interrupted"}
                    JobWorker.write_result(
                        directory / "result.json", snapshot, self.limits.max_json_bytes
                    )
                state = "completed"
            record = self.records[identifier]
            record.state, record.error, record.finished = state, error, time.time()
            self._persist(record)
            self._active = None
            logging.getLogger("rko_solver.service").info(
                "job %s terminal=%s code=%s", identifier, state, error
            )

    def _loop(self) -> None:
        """Supervise admitted work and expire private records even without incoming traffic."""
        while not self._stop.is_set():
            with self._lock:
                self._prune()
                pending, self._pending = self._pending, None
            if pending is not None:
                try:
                    self._execute(*pending)
                except Exception:
                    with self._lock:
                        self.failed = True
                        record = self.records[pending[0]]
                        record.state, record.error, record.finished = (
                            "failed",
                            "supervisor_failed",
                            time.time(),
                        )
                        self._active = None
                    self._stop.set()
            self._wake.wait(0.1)
            self._wake.clear()

    def close(self) -> None:
        """Stop active work, join supervision and release single-worker ownership."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=self.limits.startup_seconds + 5)
            if self._thread.is_alive():
                raise RuntimeError("Supervisor failed to confirm job shutdown.")
        with self._lock:
            if self._active is not None:
                record = self.records[self._active]
                record.state, record.error, record.finished = (
                    "interrupted",
                    "server_shutdown",
                    time.time(),
                )
                self._persist(record)
                self._pending, self._active = None, None
        if self._storage_lock is not None:
            self._storage_lock.close()
