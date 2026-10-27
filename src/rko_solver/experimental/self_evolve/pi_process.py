"""Bounded Linux process supervision for trusted Pi metadata and RPC commands.

Objective:
- Drain bounded output without deadlocking and reap the exact owned process group.
Technical approach:
- Use nonblocking pipes, a single monotonic deadline and Linux non-reaping exit inspection.
Flow (pseudocode):
1. Start a trusted executable in a new session with an explicit minimal environment.
2. Concurrently drain bounded stdout/stderr while writing bounded JSON commands.
3. Confirm normal exit or kill the owned group before reaping its leader.
Important criteria:
- No shell, candidate execution or raw error logging is provided.
- Interrupted or unconfirmed cleanup is distinct from a known failed attempt.
Inputs/Outputs:
- Input: trusted argument vector, private cwd, environment and absolute deadline.
- Output: bounded raw stdout or parsed JSONL records, never unrestricted diagnostics.
"""

from __future__ import annotations

import collections
import json
import os
import selectors
import signal
import subprocess
import time
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from rko_solver.experimental.self_evolve.pi_config import PiError


@dataclass
class PiBudget:
    """Share one deadline and output ceilings across version, auth and RPC processes.

    Responsibilities:
    - Prevent metadata subprocesses from each receiving a fresh output/time allocation.

    Attributes:
        deadline (float): Absolute monotonic generation deadline.
        stdout (int): Total observed stdout bytes across all subprocesses.
        stderr (int): Total discarded diagnostic bytes across all subprocesses.
    """

    deadline: float
    stdout: int = 0
    stderr: int = 0


class PiProcess:
    """Own one trusted child and its bounded nonblocking pipes.

    Responsibilities:
    - Enforce output/deadline limits and confirm whole-group cleanup before reaping.

    Attributes:
        process (subprocess.Popen): Exact owned child process.
        deadline (float): Absolute monotonic time including metadata/RPC work.
        output (bytearray): Bounded raw stdout, not persisted by the transport.
        records (deque): Parsed bounded JSONL records awaiting consumption.
        cwd (Path): Physical private working directory in the fixed system declaration.
        cleanup_deadline (float | None): One non-renewable five-second cleanup allowance.
    """

    def __init__(
        self,
        command: list[str],
        environment: dict[str, str],
        cwd: Path,
        budget: PiBudget,
        observer: Callable[[int], None] | None = None,
    ) -> None:
        """Launch the explicit trusted process and configure nonblocking pipes.

        Args:
            command (list[str]): Trusted explicit argument vector.
            environment (dict[str, str]): Minimal profile-specific environment.
            cwd (Path): Private controller-created working directory.
            budget (PiBudget): Shared monotonic deadline and cumulative output counters.
            observer (Callable | None): Durable ownership observer, before any prompt.

        Raises:
            PiError: If launch, process registration or setup fails.
        """
        self.budget = budget
        self.deadline = budget.deadline
        self.output = bytearray()
        self.pending = bytearray()
        self.outgoing = bytearray()
        self.records: collections.deque[dict[str, typing.Any]] = collections.deque()
        self.reaped = False
        self.cleanup_deadline: float | None = None
        self.rpc = False
        self.selector = selectors.DefaultSelector()
        try:
            self.cwd = cwd.resolve()
            self.process = subprocess.Popen(
                command,
                env=environment,
                cwd=cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as error:
            self.selector.close()
            raise PiError("executable") from error
        try:
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                assert stream is not None
                os.set_blocking(stream.fileno(), False)
            assert self.process.stdout is not None and self.process.stderr is not None
            self.selector.register(self.process.stdout, selectors.EVENT_READ, "stdout")
            self.selector.register(self.process.stderr, selectors.EVENT_READ, "stderr")
            if observer is not None:
                observer(self.process.pid)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> PiProcess:
        """Expose the owned process until confirmed cleanup.

        Returns:
            PiProcess: This bounded process supervisor.
        """
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """Confirm cleanup on every success, error or interruption path.

        Args:
            kind (type | None): Propagating exception type.
            error (BaseException | None): Propagating exception instance.
            traceback (TracebackType | None): Original traceback.

        Raises:
            PiError: If whole-group cleanup cannot be confirmed.
        """
        try:
            self.close()
        except PiError as cleanup:
            if isinstance(error, PiError):
                cleanup.usage = error.usage
            raise

    def _tick(self) -> None:
        """Move bounded pipe data within the shared deadline.

        Raises:
            PiError: If deadline, pipe or output boundaries fail.
        """
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise PiError("deadline")
        try:
            for key, _ in self.selector.select(min(0.05, remaining)):
                if key.data == "stdin":
                    self._write(key.fd)
                else:
                    self._read(key.fd, key.data)
        except (OSError, ValueError) as error:
            if isinstance(error, PiError):
                raise
            raise PiError("pipe") from error

    def _write(self, descriptor: int) -> None:
        """Write pending command bytes without a blocking stdin operation.

        Args:
            descriptor (int): Owned child stdin file descriptor.
        """
        try:
            count = os.write(descriptor, self.outgoing[:16384])
        except BlockingIOError:
            return
        del self.outgoing[:count]
        if not self.outgoing:
            self.selector.unregister(descriptor)

    def _read(self, descriptor: int, channel: str) -> None:
        """Drain one bounded chunk and discard diagnostics without interpreting them.

        Args:
            descriptor (int): Owned output descriptor.
            channel (str): stdout or stderr.

        Raises:
            PiError: If output caps or JSONL framing are violated.
        """
        try:
            chunk = os.read(descriptor, 16384)
        except BlockingIOError:
            return
        if not chunk:
            self.selector.unregister(descriptor)
            return
        if channel == "stderr":
            self.budget.stderr += len(chunk)
            if self.budget.stderr > 65536:
                raise PiError("output_limit")
            return
        self.budget.stdout += len(chunk)
        if self.budget.stdout > 4194304:
            raise PiError("output_limit")
        self.output.extend(chunk)
        if self.rpc:
            self.pending.extend(chunk)
            self._decode()

    def _decode(self) -> None:
        """Decode strict LF-framed JSON records without Unicode-separator splitting.

        Raises:
            PiError: If an individual record is oversized or malformed.
        """
        while b"\n" in self.pending:
            line, _, rest = self.pending.partition(b"\n")
            self.pending = bytearray(rest)
            if not line or len(line) > 1048576:
                raise PiError("protocol")
            self.records.append(self.json_value(line))
        if len(self.pending) > 1048576:
            raise PiError("output_limit")

    @staticmethod
    def json_value(payload: bytes | str) -> dict[str, typing.Any]:
        """Reject malformed or nonfinite JSON without exposing parser/provider text.

        Args:
            payload (bytes | str): One bounded JSON object.

        Returns:
            dict[str, Any]: Strict parsed object.

        Raises:
            PiError: If JSON is invalid, nonfinite or not an object.
        """
        try:
            value = json.loads(payload)
            json.dumps(value, allow_nan=False)
            if not isinstance(value, dict):
                raise ValueError
            return value
        except (ValueError, RecursionError) as error:
            raise PiError("protocol") from error

    def send(self, command: dict[str, typing.Any]) -> None:
        """Send a bounded JSON command while continuing to drain both output pipes.

        Args:
            command (dict[str, Any]): Controller-owned command; never model-selected RPC.

        Raises:
            PiError: If the command exceeds the bound or the pipe is closed.
        """
        payload = json.dumps(command, allow_nan=False).encode("utf-8") + b"\n"
        if len(payload) > 524288 or self.process.stdin is None:
            raise PiError("request_limit")
        self.outgoing.extend(payload)
        self.selector.register(self.process.stdin, selectors.EVENT_WRITE, "stdin")
        while self.outgoing:
            self._tick()

    def receive(self) -> dict[str, typing.Any]:
        """Wait for one complete bounded RPC/event record.

        Returns:
            dict[str, Any]: Next record in capture order.

        Raises:
            PiError: If the stream truncates or expires before a complete record.
        """
        while not self.records:
            if not self.selector.get_map():
                raise PiError("truncated")
            self._tick()
        return self.records.popleft()

    def finish(self) -> bytes:
        """Close input and require bounded normal exit without premature reaping.

        Returns:
            bytes: Bounded captured stdout for metadata commands.

        Raises:
            PiError: If framing, exit status or cleanup is invalid.
        """
        if self.process.stdin is not None:
            self.process.stdin.close()
        while (
            self.selector.get_map()
            or os.waitid(os.P_PID, self.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None
        ):
            self._tick()
        if self.rpc and self.pending:
            raise PiError("truncated")
        self._kill_group()
        code = self.process.wait(timeout=5)
        self.reaped = True
        if code != 0:
            raise PiError("process_exit")
        return bytes(self.output)

    def _kill_group(self) -> None:
        """Terminate the owned group while the unreaped leader still reserves its identity.

        Raises:
            PiError: If group termination cannot be confirmed.
        """
        if self.cleanup_deadline is None:
            self.cleanup_deadline = time.monotonic() + 5
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as error:
            raise PiError("cleanup", unknown=True) from error
        while self.groups_active([self.process.pid]):
            if time.monotonic() >= self.cleanup_deadline:
                raise PiError("cleanup", unknown=True)
            time.sleep(0.01)

    @staticmethod
    def groups_active(groups: list[int]) -> bool:
        """Conservatively inspect owned groups without treating signal delivery as exit.

        Args:
            groups (list[int]): Owned group identities, never kill targets in this probe.

        Returns:
            bool: True if a live member exists or inactivity cannot be established.
        """
        if not groups:
            return False
        if not Path("/proc").is_dir():
            return True
        unseen = set(groups)
        for path in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = path.read_text().rsplit(")", 1)[1].split()
                group = int(fields[2])
                if group in groups:
                    unseen.discard(group)
                    if fields[0] != "Z":
                        return True
            except FileNotFoundError:
                continue
            except OSError, ValueError, IndexError:
                return True
        return any(PiProcess._group_exists(group) for group in unseen)

    @staticmethod
    def _group_exists(group: int) -> bool:
        """Probe existence when procfs exposed no member, without sending a real signal.

        Args:
            group (int): Owned group identity not observed through procfs.

        Returns:
            bool: False only when the kernel confirms that the group does not exist.
        """
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True
        return True

    def close(self) -> None:
        """Kill before reaping on failure and close all controller-owned descriptors.

        Raises:
            PiError: If cleanup cannot be confirmed within five seconds.
        """
        try:
            if not self.reaped:
                try:
                    self._kill_group()
                finally:
                    remaining = (
                        max(0.0, self.cleanup_deadline - time.monotonic())
                        if self.cleanup_deadline is not None
                        else 0.0
                    )
                    self.process.wait(timeout=remaining)
                    self.reaped = True
        except (OSError, subprocess.SubprocessError) as error:
            raise PiError("cleanup", unknown=True) from error
        finally:
            self.selector.close()
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                if stream is not None:
                    stream.close()
