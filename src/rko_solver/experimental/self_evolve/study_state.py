"""Frozen study identity and durable state admission.

Objective:
- Verify declarations before any collection or inference.
Technical approach:
- Hash replay-critical source and enforce durable integrity stops.
- Verify canonical specification and nine artifact identities before journal interpretation.
- Report only counts and resource charges, never interim statistical results.
Flow (pseudocode):
1. Reject missing databases, oversized declarations and changed implementation identities.
2. Exclude concurrent controllers and reject unknown work or prior integrity failures.
3. Summarize journal counts and permit inference only for a complete, valid panel.
Important criteria:
- Preserve specification schema, operation order, reservations and inference gates.
Inputs/Outputs:
- Input: a private frozen study directory and its durable SQLite journal.
- Output: verified declaration identities, exclusive access or evidence-free progress.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import numpy as np
import os
import platform
import sqlite3
from .artifacts import FAMILIES
from .artifacts import DecoderArtifact
from .problems import Problem
from .sandbox import JOB_GRACE_SECONDS
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ARMS = ("seed", "constructive", "first_shot", "candidate", "random", "fixed")


class StudyState:
    """Verify immutable identities and guard durable state.

    Responsibilities:
    - Verify immutable identities and guard durable state.

    Attributes:
        path (Path): Private controller directory.
        spec (dict[str, Any]): Hash-verified declaration.
        digest (str): Approved specification identity.
        _runtime_token (str | None): Active object-local runtime authorization.
    """

    _runtime_token: str | None

    def __init__(self, path: Path) -> None:
        """Verify a frozen study and its current collector/analysis implementation.

        Args:
            path (Path): Existing study directory.

        Raises:
            ValueError: If source/runtime/specification identity has changed.
            OSError: If required files are unavailable.
        """
        self.path = path
        if not (path / "measurements.sqlite").is_file():
            raise ValueError(
                "Study measurement database is missing; no empty replacement is created."
            )
        if (path / "study.json").stat().st_size > 16777216:
            raise ValueError("Frozen study exceeds 16 MiB.")
        self.spec = json.loads((path / "study.json").read_text(encoding="utf-8"))
        self.digest = self._hash(self.spec)
        if self.digest != (path / "study.sha256").read_text().strip():
            raise ValueError("Frozen study specification hash mismatch.")
        if self.spec["implementation"] != self._implementation():
            raise ValueError(
                "Collector/analysis implementation changed; audited reanalysis is required."
            )
        for family in FAMILIES:
            for role in ("seed", "first_shot", "candidate"):
                artifact = DecoderArtifact.load(path / "artifacts" / family / role)
                if artifact.digest != self.spec["artifacts"][family][role]["sha256"]:
                    raise ValueError("Frozen study decoder hash mismatch.")

    def _verify(self) -> None:
        """Reject in-memory or on-disk changes before collecting or interpreting evidence.

        Raises:
            ValueError: If the frozen specification or source identities changed.
        """
        current = type(self)(self.path)
        if current.digest != self.digest or self._hash(self.spec) != self.digest:
            raise ValueError("Frozen study declaration was modified.")

    @staticmethod
    def _hash(value: object) -> str:
        """Hash canonical JSON without executing or interpreting its text.

        Args:
            value (object): Finite JSON-compatible declaration.

        Returns:
            str: SHA-256 of its sorted JSON encoding.
        """
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    @staticmethod
    def instance_hash(problem: Problem) -> str:
        """Identify exact instance semantics across development/validation/test manifests.

        Args:
            problem (Problem): Trusted validated instance.

        Returns:
            str: Family-sensitive semantic hash.
        """
        return StudyState._hash({"family": problem.family, "data": problem.data})

    @staticmethod
    def _implementation() -> dict[str, Any]:
        """Fingerprint the measurement/analysis code and numerical runtime.

        Returns:
            dict[str, Any]: Replay-critical controller implementation identity.
        """
        root = Path(__file__).parent
        return {
            "code": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(root.glob("*.py"))
            },
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rko": importlib.metadata.version("rko-solver"),
        }

    @contextlib.contextmanager
    def _lock(self) -> Iterator[None]:
        """Exclude concurrent measurement controllers without hiding stale crash locks.

        Yields:
            None: Exclusive write access for one collection chunk.

        Raises:
            FileExistsError: If an active/stale lock requires operator inspection.
        """
        path = self.path / "controller.lock"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, str(os.getpid()).encode())
            yield
        finally:
            os.close(descriptor)
            path.unlink()

    def _admit_collection(self, db: sqlite3.Connection, *, active: bool = False) -> None:
        """Reject unknown work or a prior integrity stop before spending more resources.

        Args:
            db (sqlite3.Connection): Locked measurement database.
            active (bool): Allow collection, but never another setup, under the current token.

        Raises:
            ValueError: If a cell/setup is unknown or runtime/reference integrity failed.
        """
        if db.execute("SELECT 1 FROM jobs WHERE status='running' LIMIT 1").fetchone():
            raise ValueError("Unknown reserved cell; collection remains stopped.")
        if db.execute(
            "SELECT 1 FROM jobs WHERE json_extract(result, '$.status') IN "
            "('reference_error','infrastructure_error') LIMIT 1"
        ).fetchone():
            raise ValueError("Study integrity previously failed; collection remains stopped.")
        if db.execute(
            "SELECT 1 FROM state WHERE key IN ('setup_pending','setup_failed')"
        ).fetchone():
            raise ValueError("Unknown or failed runtime setup; collection remains stopped.")
        active_runtime = db.execute("SELECT value FROM state WHERE key='setup_ready'").fetchone()
        if active_runtime and not (
            active and active_runtime[0] == getattr(self, "_runtime_token", None)
        ):
            raise ValueError("Another runtime setup is active or interrupted; collection stopped.")

    def progress(self) -> dict[str, Any]:
        """Return collection counts/charges without enabling interim significance peeking.

        Returns:
            dict[str, Any]: Completion and conservative budget totals only.
        """
        self._verify()
        with contextlib.closing(sqlite3.connect(self.path / "measurements.sqlite")) as db:
            if (
                db.execute("SELECT value FROM state WHERE key='specification'").fetchone()[0]
                != self.digest
            ):
                raise ValueError("Measurement database belongs to a different specification.")
            counts = dict(db.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status"))
            charge = db.execute("SELECT COALESCE(SUM(charge), 0) FROM jobs").fetchone()[0]
            setup = db.execute("SELECT value FROM state WHERE key='setup_charge'").fetchone()
            setup_charge = float(setup[0]) if setup else 0
            invalid_setup = (
                db.execute(
                    "SELECT 1 FROM state WHERE key IN "
                    "('setup_pending','setup_ready','setup_failed') LIMIT 1"
                ).fetchone()
                is not None
            )
            invalid_cells = (
                db.execute(
                    "SELECT 1 FROM jobs WHERE json_extract(result, '$.status') IN "
                    "('reference_error','infrastructure_error') LIMIT 1"
                ).fetchone()
                is not None
            )
        return {
            "specification": self.digest,
            "jobs": len(self.spec["jobs"]),
            "complete": counts.get("complete", 0),
            "pending": counts.get("pending", 0),
            "interrupted": counts.get("running", 0),
            "charged_cpu_upper_bound": charge + setup_charge,
            "session_startup_upper_bound": setup_charge,
            "cpu_budget": self.spec["cpu_budget"],
            "inner_allowances_total": len(self.spec["jobs"]) * self.spec["seconds"],
            "all_jobs_worst_case_cpu_reservation": len(self.spec["jobs"])
            * (self.spec["seconds"] + JOB_GRACE_SECONDS),
            "inference_available": counts.get("complete", 0) == len(self.spec["jobs"])
            and not invalid_setup
            and not invalid_cells,
        }
