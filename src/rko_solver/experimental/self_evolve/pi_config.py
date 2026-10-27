"""Explicit non-secret Pi proposer configuration and profile policy.

Objective:
- Admit only the pinned OAuth-backed Pi protocol without inheriting operator automation.
Technical approach:
- Freeze exact executable/profile/model settings and require a minimal private Pi profile.
Flow (pseudocode):
1. Validate bounded non-secret configuration and the initially supported provider.
2. Exclusively prepare settings without copying or reading authentication credentials.
3. Check profile policy before launch and supply a minimal child-only environment.
Important criteria:
- Pi owns login and refresh; RKO never reads token values or changes billing modes.
- Profile policy is not an operating-system sandbox or live-provider qualification.
Inputs/Outputs:
- Input: explicit paths, model identity and finite generation deadline.
- Output: immutable settings, a private profile or a sanitized admission error.
"""

from __future__ import annotations

import json
import math
import os
import re
import typing
from dataclasses import dataclass
from pathlib import Path
from rko_solver.experimental.self_evolve.study_state import StudyState

PI_VERSION = "1.3.0"
PI_PROVIDER = "openai-codex"
PI_ENDPOINT = "https://chatgpt.com/backend-api"
PI_API = "openai-codex-responses"
_HINTS = {
    "authentication": "Use Pi's supported login in this profile; API fallback is disabled.",
    "profile": "Use a separate minimal pi-profile; do not edit your ordinary Pi settings.",
    "version": "This experimental transport requires Pi 1.3.0.",
    "model": "Select an exact available built-in Codex model, not a custom endpoint or fuzzy name.",
    "terminal": "Inspect login/quota in Pi before starting another explicitly counted attempt.",
    "deadline": "The attempt stopped; provider token/billing consumption may remain unknown.",
    "cleanup": "Do not retry; inspect the owned processes and durable proposal state.",
}


class PiError(ValueError):
    """Expose a fixed diagnostic category without provider text or credentials.

    Responsibilities:
    - Separate known failed attempts from unconfirmed process cleanup.

    Attributes:
        reason (str): Controller-owned failure category.
        unknown (bool): Whether cleanup or generation outcome remains unconfirmed.
        usage (dict[str, int] | None): Available allowlisted reported usage, never actual billing.
    """

    def __init__(
        self, reason: str, *, unknown: bool = False, usage: dict[str, int] | None = None
    ) -> None:
        """Retain a safe category, never a subprocess error payload.

        Args:
            reason (str): Internal fixed category identifying the rejected boundary.
            unknown (bool): Preserve a durable unknown rather than allow a blind retry.
            usage (dict[str, int] | None): Safe available counters from a failed exchange.
        """
        hint = _HINTS.get(reason, "Inspect the proposal record; do not retry automatically.")
        super().__init__(f"Pi generation failed: {reason}. {hint}")
        self.reason, self.unknown, self.usage = reason, unknown, usage


@dataclass(frozen=True)
class PiConfig:
    """Own one explicit pinned proposer configuration without secret fields.

    Responsibilities:
    - Validate exact access/model settings and the minimal private profile policy.

    Attributes:
        executable (Path): Trusted installed Pi executable, never shell text.
        profile (Path): Separate Pi-managed authentication/profile directory.
        model (str): Exact built-in model ID, not a fuzzy model expression.
        seconds (float): Entire generation deadline including metadata preflight.
        provider (str): Initially supported built-in OAuth provider.
    """

    executable: Path
    profile: Path
    model: str
    seconds: float = 120.0
    provider: str = PI_PROVIDER

    def __post_init__(self) -> None:
        """Validate the supported identity and normalize explicit filesystem paths.

        Raises:
            ValueError: If settings are malformed, unsupported or unbounded.
        """
        if (
            not isinstance(self.executable, Path)
            or not isinstance(self.profile, Path)
            or self.provider != PI_PROVIDER
            or not isinstance(self.model, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", self.model) is None
            or type(self.seconds) not in (int, float)
            or not math.isfinite(self.seconds)
            or not 1 <= self.seconds <= 600
        ):
            raise ValueError("Explicit supported Pi settings and a 1–600 second deadline required.")
        object.__setattr__(self, "executable", self.executable.resolve())
        object.__setattr__(self, "profile", self.profile.resolve())

    @classmethod
    def load(cls, path: Path) -> PiConfig:
        """Read bounded configuration with paths relative to its declaration file.

        Args:
            path (Path): Non-secret JSON declaration.

        Returns:
            PiConfig: Validated exact proposer settings.

        Raises:
            ValueError: If declaration fields, size or values are invalid.
            OSError: If the declaration is unavailable.
        """
        if path.stat().st_size > 16384:
            raise ValueError("Pi configuration exceeds 16 KiB.")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != {
            "executable",
            "profile",
            "provider",
            "model",
            "seconds",
        }:
            raise ValueError("Pi configuration requires exact non-secret fields.")
        if any(not isinstance(data[key], str) for key in ("executable", "profile")):
            raise ValueError("Pi paths must be strings.")
        return cls(
            executable=path.parent / data["executable"],
            profile=path.parent / data["profile"],
            model=data["model"],
            seconds=data["seconds"],
            provider=data["provider"],
        )

    def descriptor(self) -> dict[str, typing.Any]:
        """Return the bounded frozen identity without resolving credentials.

        Returns:
            dict[str, Any]: Canonical proposer declaration for a campaign journal.
        """
        return {
            "schema": 1,
            "transport": "pi_rpc",
            "pi_version": PI_VERSION,
            "access_mode": "oauth",
            "api": PI_API,
            "endpoint": PI_ENDPOINT,
            "implementation": StudyState._implementation(),
            "executable": str(self.executable),
            "profile": str(self.profile),
            "provider": self.provider,
            "model": self.model,
            "seconds": self.seconds,
        }

    def settings(self) -> dict[str, typing.Any]:
        """Declare the complete supported profile policy before any model request.

        Returns:
            dict[str, Any]: No retries, warming, compaction, telemetry or inherited resources.
        """
        return {
            "defaultProvider": self.provider,
            "defaultModel": self.model,
            "defaultThinkingLevel": "medium",
            "defaultProjectTrust": "never",
            "cacheWarming": "off",
            "transport": "sse",
            "lastChangelogVersion": PI_VERSION,
            "retry": {"enabled": False, "provider": {"maxRetries": 0}},
            "compaction": {"enabled": False},
            "enableInstallTelemetry": False,
            "enableAnalytics": False,
        }

    def prepare_profile(self) -> None:
        """Exclusively create non-secret settings; login remains an explicit Pi operation.

        Raises:
            FileExistsError: If the directory already exists, including an operator profile.
            OSError: If creation fails.
        """
        self.profile.mkdir(parents=True, exist_ok=False, mode=0o700)
        path = self.profile / "settings.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(self.settings(), handle, allow_nan=False)

    def verify_profile(self) -> None:
        """Reject unsafe profile resources without reading authentication values.

        Raises:
            PiError: If profile privacy, settings or resource policy cannot be verified.
        """
        try:
            self._private(self.profile, 0o700)
            for name in ("settings.json", "auth.json"):
                self._private(self.profile / name, 0o600)
            path = self.profile / "settings.json"
            if path.stat().st_size > 16384:
                raise PiError("profile")
            actual = json.dumps(
                json.loads(path.read_text(encoding="utf-8")), sort_keys=True, allow_nan=False
            )
            if actual != json.dumps(self.settings(), sort_keys=True, allow_nan=False):
                raise PiError("profile")
            forbidden = (
                "models.json",
                "mcp.json",
                "extensions",
                "skills",
                "prompts",
                "themes",
                "SYSTEM.md",
                "APPEND_SYSTEM.md",
            )
            if any((self.profile / name).exists() for name in forbidden):
                raise PiError("profile")
        except (OSError, ValueError, RecursionError) as error:
            raise PiError("profile") from error

    @staticmethod
    def _private(path: Path, mode: int) -> None:
        """Check owner-only profile metadata without opening any credential file.

        Args:
            path (Path): Profile directory or controller/Pi-owned regular file.
            mode (int): Required private POSIX access mode.

        Raises:
            PiError: If ownership, type or permissions do not match the policy.
        """
        stat = path.stat()
        if (
            path.is_symlink()
            or stat.st_uid != os.getuid()
            or stat.st_mode & 0o777 != mode
            or (not path.is_dir() if mode == 0o700 else not path.is_file())
        ):
            raise PiError("profile")

    def environment(self) -> dict[str, str]:
        """Exclude ambient credentials, Node injection and inherited session settings.

        Returns:
            dict[str, str]: Minimal environment for the separately managed profile.
        """
        return {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": str(self.profile),
            "PI_CODING_AGENT_DIR": str(self.profile),
            "PI_OFFLINE": "1",
            "PI_SKIP_VERSION_CHECK": "1",
            "PI_TELEMETRY": "0",
        }
