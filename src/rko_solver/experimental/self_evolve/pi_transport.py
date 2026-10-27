"""Review-gated Pi proposal generation without direct API credentials.

Objective:
- Exchange one bounded development packet with an explicitly selected Pi RPC process.
Technical approach:
- Require a private policy-controlled profile and safe OAuth metadata before prompting.
Flow (pseudocode):
1. Validate platform/profile and inspect the pinned CLI without reading credentials.
2. Start one no-tools RPC session and verify exact model/endpoint before generation.
3. Validate terminal source data and return safe provenance for human review.
Important criteria:
- This transport never evaluates source or attests that human review occurred.
- Metadata/protocol compatibility is not live-provider or subscription certification.
Inputs/Outputs:
- Input: frozen non-secret Pi settings and authorized development context.
- Output: source/hypothesis and reported usage, or a fixed safe failure category.
"""

from __future__ import annotations

import sys
import tempfile
import time
import typing
from collections.abc import Callable
from pathlib import Path
from rko_solver.experimental.self_evolve.artifacts import ARTIFACT_FAMILIES
from rko_solver.experimental.self_evolve.artifacts import DecoderArtifact
from rko_solver.experimental.self_evolve.pi_config import PI_VERSION
from rko_solver.experimental.self_evolve.pi_config import PiConfig
from rko_solver.experimental.self_evolve.pi_config import PiError
from rko_solver.experimental.self_evolve.pi_process import PiBudget
from rko_solver.experimental.self_evolve.pi_process import PiProcess
from rko_solver.experimental.self_evolve.pi_protocol import SYSTEM_PROMPT
from rko_solver.experimental.self_evolve.pi_protocol import PiProtocol


class PiTransport:
    """Own one bounded proposal exchange with no candidate execution capability.

    Responsibilities:
    - Inspect the exact supported access mode and constrain inherited Pi resources.

    Attributes:
        config (PiConfig): Explicit provider/model, profile and finite deadline.
    """

    def __init__(self, config: PiConfig) -> None:
        """Retain the non-secret generation declaration without launching anything.

        Args:
            config (PiConfig): Immutable exact proposer settings.
        """
        self.config = config

    @staticmethod
    def _flags() -> list[str]:
        """Disable model tools, inherited resources and ordinary developer context.

        Returns:
            list[str]: Fixed documented Pi flags, not operator-supplied shell options.
        """
        return [
            "--no-tools",
            "--no-extensions",
            "--no-mcp",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-context-files",
            "--no-session",
            "--no-approve",
            "--offline",
        ]

    def generate(
        self, packet: dict[str, typing.Any], observer: Callable[[int], None] | None = None
    ) -> dict[str, typing.Any]:
        """Generate source data only after bounded exact-identity and access preflight.

        Args:
            packet (dict[str, Any]): Authorized bounded development-only context.
            observer (Callable | None): Durable ownership recorder called before any prompt.

        Returns:
            dict[str, Any]: Proposed source and safe generation provenance for human review.

        Raises:
            PiError: If profile, executable, access, protocol, deadline or source policy fails.
            BaseException: Interruptions propagate after owned process cleanup.
        """
        if sys.platform != "linux":
            raise PiError("platform")
        if not isinstance(packet, dict) or packet.get("family") not in ARTIFACT_FAMILIES:
            raise PiError("request")
        self.config.verify_profile()
        started = time.monotonic()
        budget = PiBudget(started + self.config.seconds)
        protocol: PiProtocol | None = None
        try:
            with tempfile.TemporaryDirectory(prefix="rko-pi-") as directory:
                cwd = Path(directory)
                self._metadata(cwd, budget, observer)
                self.config.verify_profile()
                command = [
                    str(self.config.executable),
                    *self._flags(),
                    "--mode",
                    "rpc",
                    "--provider",
                    self.config.provider,
                    "--model",
                    self.config.model,
                    "--system-prompt",
                    SYSTEM_PROMPT,
                ]
                with PiProcess(
                    command, self.config.environment(), cwd, budget, observer
                ) as process:
                    protocol = PiProtocol(process, self.config)
                    result = protocol.generate(packet)
                DecoderArtifact.create(packet["family"], result["source"])
                return {**result, "elapsed_seconds": time.monotonic() - started}
        except PiError as error:
            error.usage = protocol.usage if protocol else error.usage
            raise
        except OSError, ValueError, TypeError, KeyError, RecursionError:
            raise PiError("candidate", usage=protocol.usage if protocol else None) from None

    def _metadata(
        self, cwd: Path, budget: PiBudget, observer: Callable[[int], None] | None
    ) -> None:
        """Inspect version and auth TYPE only, with no refresh or credential-print option.

        Args:
            cwd (Path): Private working directory.
            budget (PiBudget): Shared deadline/output limits, never reset for metadata.
            observer (Callable | None): Durable child identity recorder.

        Raises:
            PiError: If the CLI version or supported OAuth access cannot be established.
        """
        command = [str(self.config.executable), *self._flags(), "--version"]
        with PiProcess(command, self.config.environment(), cwd, budget, observer) as process:
            version = process.finish().strip()
        if version != PI_VERSION.encode():
            raise PiError("version")
        command = [
            str(self.config.executable),
            "auth",
            "check",
            "--provider",
            self.config.provider,
            "--json",
            "--no-refresh",
            *self._flags(),
        ]
        with PiProcess(command, self.config.environment(), cwd, budget, observer) as process:
            auth = PiProcess.json_value(process.finish())
        if auth != {"status": "ready", "provider": self.config.provider, "authType": "oauth"}:
            raise PiError("authentication")
