"""One-at-a-time Pi proposals with durable attempts and exact human approval.

Objective:
- Automate generation without bypassing source review or changing evaluation budgets.
Technical approach:
- Extend the existing campaign journal with distinct generation/review transitions.
Flow (pseudocode):
1. Freeze non-secret Pi settings and reserve one submission before any external process.
2. Store source-only output as pending review, retaining all failures and unknown outcomes.
3. Require exact source identity and explicit review before reusing the counted attempt.
Important criteria:
- Unknown attempts never retry automatically; cell integrity stops remain authoritative.
- Existing manual campaign and artifact semantics remain unchanged.
Inputs/Outputs:
- Input: calibrated campaign, explicit Pi settings/consent and reviewed source identity.
- Output: pending proposal, explicit rejection/abandonment or bounded experimental decision.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import typing
from functools import partial
from pathlib import Path
from rko_solver.experimental.self_evolve.artifacts import MAX_SOURCE
from rko_solver.experimental.self_evolve.artifacts import DecoderArtifact
from rko_solver.experimental.self_evolve.campaign import BudgetExpired
from rko_solver.experimental.self_evolve.campaign import Campaign
from rko_solver.experimental.self_evolve.pi_config import PiConfig
from rko_solver.experimental.self_evolve.pi_config import PiError
from rko_solver.experimental.self_evolve.pi_process import PiProcess
from rko_solver.experimental.self_evolve.pi_transport import PiTransport
from rko_solver.experimental.self_evolve.proposal_journal import ProposalJournal
from rko_solver.experimental.self_evolve.sandbox import JOB_GRACE_SECONDS
from rko_solver.experimental.self_evolve.sandbox import PodmanRunner
from rko_solver.experimental.self_evolve.study_state import StudyState


class PiCampaign(Campaign):
    """Add explicit generation and human review to an ordinary bounded campaign.

    Responsibilities:
    - Preserve submission accounting and immutable source/proposer identities.
    - Reuse the existing reviewed evaluation ladder without charging a second slot.

    Attributes:
        path (Path): Existing private campaign directory.
        spec (dict[str, Any]): Original unchanged evaluation specification.
    """

    def _verify(self) -> None:
        """Reject an in-memory or on-disk campaign change before spending resources.

        Raises:
            ValueError: If the original declaration no longer matches this controller.
        """
        if type(self)(self.path).spec != self.spec:
            raise ValueError("Frozen campaign changed.")

    def _freeze(self, config: PiConfig) -> str:
        """Freeze non-secret proposer settings before the first submitted attempt.

        Args:
            config (PiConfig): Explicit supported proposer declaration.

        Returns:
            str: Exact canonical configuration digest.

        Raises:
            ValueError: If profiles overlap artifacts or a prior declaration differs.
            OSError: If persistence fails.
        """
        root = self.path.resolve()
        if config.profile.is_relative_to(root) or root.is_relative_to(config.profile):
            raise ValueError("Pi authentication profile must be separate from campaign artifacts.")
        encoded = json.dumps(config.descriptor(), sort_keys=True, allow_nan=False).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        ProposalJournal(self.events()).verify_proposer(digest)
        path = self.path / "proposer.json"
        if path.exists():
            if self._declaration() != encoded:
                raise ValueError("Frozen Pi proposer settings changed.")
        else:
            if any(e["kind"] == "submission" for e in self.events()):
                raise ValueError("Freeze the proposer before any submitted attempt.")
            self._write(path, encoded)
        return digest

    def _declaration(self) -> bytes:
        """Read only bounded non-secret proposer bytes, never an authentication file.

        Returns:
            bytes: Existing frozen declaration of at most sixteen KiB.

        Raises:
            ValueError: If the frozen declaration grew beyond its admitted bound.
            OSError: If the declaration is missing or unreadable.
        """
        with (self.path / "proposer.json").open("rb") as handle:
            payload = handle.read(16385)
        if len(payload) > 16384:
            raise ValueError("Frozen proposer declaration exceeds 16 KiB.")
        return payload

    @staticmethod
    def _write(path: Path, payload: bytes) -> None:
        """Exclusively persist private bounded bytes before their journal transition.

        Args:
            path (Path): Controller-owned new file.
            payload (bytes): Bounded non-secret declaration or development request.

        Raises:
            OSError: If creation or durable flush fails.
        """
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        PiCampaign._sync_directory(path.parent)

    @staticmethod
    def _sync_directory(path: Path) -> None:
        """Flush newly created directory entries before recording their durable state.

        Args:
            path (Path): Controller-owned Linux directory.

        Raises:
            OSError: If directory durability cannot be established.
        """
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _generation_budget(self, config: PiConfig) -> None:
        """Admit generation plus bounded process cleanup without spending evaluation CPU.

        Args:
            config (PiConfig): Finite generation allowance.

        Raises:
            BudgetExpired: If the remaining original elapsed allowance cannot admit the call.
        """
        self._check_budget("candidate", self.spec["seconds"] + JOB_GRACE_SECONDS)
        starts = [e["at"] for e in self.events() if e["kind"] == "start"]
        elapsed = time.time() - starts[0]
        if elapsed < 0 or elapsed + config.seconds + 15 >= self.spec["elapsed_seconds"]:
            raise BudgetExpired("Insufficient campaign elapsed allowance for Pi generation.")

    def generate(
        self, config: PiConfig, *, parent: str | None = None, consent: bool = False
    ) -> dict[str, typing.Any]:
        """Reserve and generate one source-only proposal, then stop for human review.

        Args:
            config (PiConfig): Explicit frozen provider/profile/model/deadline.
            parent (str | None): Eligible parent; defaults to the original seed.
            consent (bool): Explicit consent to send development data to this provider.

        Returns:
            dict[str, Any]: Pending proposal identity and private review source path.

        Raises:
            ValueError: If consent, settings, parent or existing work forbids generation.
            BudgetExpired: If original attempt/time/evaluation ceilings prevent admission.
            PiError: If the counted generation attempt fails; its failure remains durable.
            BaseException: Interruptions preserve an unknown counted attempt.
        """
        if consent is not True:
            raise ValueError("Explicit development-data consent is required before Pi generation.")
        if sys.platform != "linux":
            raise PiError("platform")
        with self._lock():
            self._verify()
            packet = self.request(parent)
            self._generation_budget(config)
            payload = json.dumps(packet, ensure_ascii=True, allow_nan=False).encode()
            if len(payload) > 262144:
                raise ValueError("Pi development request exceeds 256 KiB.")
            digest = self._freeze(config)
            attempt = sum(e["kind"] == "submission" for e in self.events()) + 1
            self._append(
                {
                    "kind": "submission",
                    "transport": "pi_rpc",
                    "attempt": attempt,
                    "parent": packet["parent"],
                    "proposer_sha256": digest,
                    "reviewed": False,
                }
            )
            started = time.monotonic()
            result: dict[str, typing.Any] | None = None
            try:
                directory = self.path / "proposals" / str(attempt)
                directory.mkdir(parents=True, exist_ok=False, mode=0o700)
                self._write(directory / "request.json", payload)
                result = PiTransport(config).generate(packet, partial(self._process, attempt))
                artifact = DecoderArtifact.create(
                    self.spec["family"], result["source"], packet["parent"]
                )
                self._store_pending(attempt, artifact, result["usage"])
                self._finish_generation(attempt, artifact, result, started, config.seconds + 15)
            except (PiError, OSError, ValueError) as error:
                failure = (
                    error
                    if isinstance(error, PiError)
                    else PiError(
                        "storage_or_source", usage=result["usage"] if result is not None else None
                    )
                )
                self._failed_generation(attempt, failure, started, config.seconds + 15)
                raise failure from None
            return self.proposal(attempt)

    def _failed_generation(
        self, attempt: int, failure: PiError, started: float, limit: float
    ) -> None:
        """Retain counted failed/unknown work, full elapsed time and available reported usage.

        Args:
            attempt (int): Already reserved submission slot.
            failure (PiError): Safe internal error category and optional reported counters.
            started (float): Original monotonic generation start.
            limit (float): Admitted generation plus process-cleanup allowance.
        """
        elapsed = time.monotonic() - started
        status = "generation_unknown" if failure.unknown else "generation_failed"
        if elapsed > limit or failure.reason == "generation_overrun":
            status = "infrastructure_error"
        self._append(
            {
                "kind": "generation_finished",
                "attempt": attempt,
                "status": status,
                "error_reason": failure.reason,
                "elapsed_seconds": elapsed,
                "usage": failure.usage,
                "usage_complete": False,
                "monetary_cost": None,
            }
        )

    def _store_pending(
        self, attempt: int, artifact: DecoderArtifact, usage: dict[str, int] | None
    ) -> None:
        """Persist source in a review-only bundle rejected by the ordinary execution loader.

        Args:
            attempt (int): Counted proposal identity.
            artifact (DecoderArtifact): Validated proposed bytes, not reviewed/accepted source.
            usage (dict[str, int] | None): Available reported counters if persistence fails.

        Raises:
            PiError: If review-only source persistence fails after generation.
        """
        path = self.path / "proposals" / str(attempt) / "artifact"
        try:
            path.mkdir(mode=0o700, exist_ok=False)
            self._write(path / "decoder.py", artifact.source.encode("utf-8"))
            metadata = {**artifact.metadata(), "status": "pending_review"}
            self._write(path / "manifest.json", json.dumps(metadata, sort_keys=True).encode())
        except OSError:
            raise PiError("storage", usage=usage) from None

    def _process(self, attempt: int, pid: int) -> None:
        """Persist owned child identities before sending any model prompt.

        Args:
            attempt (int): Already counted submission identity.
            pid (int): New session leader and owned process-group identity.
        """
        self._append({"kind": "generation_process", "attempt": attempt, "pid": pid})

    def _finish_generation(
        self,
        attempt: int,
        artifact: DecoderArtifact,
        result: dict[str, typing.Any],
        started: float,
        limit: float,
    ) -> None:
        """Persist safe completed source metadata only after the private bundle is durable.

        Args:
            attempt (int): Already counted submission identity.
            artifact (DecoderArtifact): Validated source-only proposal, not an accepted artifact.
            result (dict[str, Any]): Safe transport result with no raw provider diagnostics.
            started (float): Original generation start timestamp.
            limit (float): Admitted generation and cleanup allowance, never enlarged.
        """
        directory = self.path / "proposals" / str(attempt) / "artifact"
        for name in ("decoder.py", "manifest.json"):
            with (directory / name).open("rb") as handle:
                os.fsync(handle.fileno())
        for path in (directory, directory.parent, directory.parent.parent, self.path):
            self._sync_directory(path)
        elapsed = time.monotonic() - started
        if elapsed > limit:
            raise PiError("generation_overrun", usage=result["usage"])
        metadata = {
            key: result[key]
            for key in (
                "provider",
                "model",
                "pi_version",
                "access_mode",
                "api",
                "endpoint",
                "responseModel",
                "usage",
                "monetary_cost",
            )
        }
        self._append(
            {
                "kind": "generation_finished",
                "attempt": attempt,
                "status": "pending_review",
                "candidate": artifact.digest,
                "hypothesis": result["hypothesis"],
                "usage_complete": result["usage"] is not None,
                "elapsed_seconds": elapsed,
                **metadata,
            }
        )

    def proposal(self, attempt: int) -> dict[str, typing.Any]:
        """Return one proposal's durable status without evaluating or trusting its source.

        Args:
            attempt (int): Existing Pi submission identity.

        Returns:
            dict[str, Any]: Review status, source hash and private source path when available.

        Raises:
            ValueError: If the proposal identity does not exist.
        """
        entry = ProposalJournal(self.events()).entry(attempt)
        if "candidate" in entry:
            entry = {
                **entry,
                "source_path": str(
                    self.path / "proposals" / str(attempt) / "artifact" / "decoder.py"
                ),
            }
        return entry

    def proposals(self) -> list[dict[str, typing.Any]]:
        """List bounded durable proposal states without launching or evaluating anything.

        Returns:
            list[dict[str, Any]]: Submission-ordered generation/review summaries.
        """
        return [self.proposal(attempt) for attempt in ProposalJournal(self.events()).entries]

    def review_source(self, attempt: int, digest: str) -> DecoderArtifact:
        """Verify exact pending source and its frozen proposer identity before runtime setup.

        Args:
            attempt (int): Pending submission identity.
            digest (str): Exact source hash explicitly inspected by the human.

        Returns:
            DecoderArtifact: Still-unaccepted source-only artifact.

        Raises:
            ValueError: If state, bytes, parent or frozen proposer identity changed.
            OSError: If required source files are unavailable.
        """
        self._verify()
        entry = self.proposal(attempt)
        if entry["status"] != "pending_review":
            raise ValueError("Pi proposal is not pending review.")
        frozen = self._declaration()
        actual = hashlib.sha256(frozen).hexdigest()
        if digest != entry["candidate"] or actual != entry["proposer_sha256"]:
            raise ValueError("Reviewed source or proposer hash mismatch.")
        if json.loads(frozen).get("implementation") != StudyState._implementation():
            raise ValueError("Frozen Pi controller implementation changed; use a new campaign.")
        artifact = self._pending_artifact(attempt, entry["parent"])
        if (
            artifact.digest != digest
            or artifact.parent != entry["parent"]
            or artifact.family != self.spec["family"]
        ):
            raise ValueError("Reviewed source/parent hash mismatch.")
        self._check_budget("candidate", self.spec["seconds"] + JOB_GRACE_SECONDS, active=attempt)
        return artifact

    def _pending_artifact(self, attempt: int, parent: str) -> DecoderArtifact:
        """Validate a bounded review-only bundle without treating it as executable/exported.

        Args:
            attempt (int): Exact existing pending identity.
            parent (str): Frozen source-lineage parent from the controller journal.

        Returns:
            DecoderArtifact: Validated in-memory source for explicit review, never executed here.

        Raises:
            ValueError: If source, status, contract or size does not match the frozen record.
            OSError: If the pending files are unavailable.
        """
        path = self.path / "proposals" / str(attempt) / "artifact"
        with (path / "decoder.py").open("rb") as handle:
            source = handle.read(MAX_SOURCE + 1)
        with (path / "manifest.json").open("rb") as handle:
            manifest = handle.read(16385)
        if len(source) > MAX_SOURCE or len(manifest) > 16384:
            raise ValueError("Pending review source/manifest exceeds its bound.")
        artifact = DecoderArtifact.create(self.spec["family"], source.decode("utf-8"), parent)
        expected = {**artifact.metadata(), "status": "pending_review"}
        actual = json.dumps(json.loads(manifest), sort_keys=True, allow_nan=False)
        if actual != json.dumps(expected, sort_keys=True, allow_nan=False):
            raise ValueError("Pending review source/contract hash mismatch.")
        return artifact

    def approve(
        self, attempt: int, digest: str, runner: PodmanRunner, *, reviewed: bool = False
    ) -> dict[str, typing.Any]:
        """Evaluate exactly the reviewed proposal using its already consumed submission slot.

        Args:
            attempt (int): Existing pending proposal identity.
            digest (str): Exact human-reviewed source hash.
            runner (PodmanRunner): Qualified pinned evaluation runtime.
            reviewed (bool): Explicit local-work source-review attestation.

        Returns:
            dict[str, Any]: Ordinary experimental shortlist decision, not production promotion.

        Raises:
            ValueError: If attestation, source identity, state or runtime is invalid.
            BudgetExpired: If the unchanged evaluation budget cannot admit work.
        """
        if reviewed is not True:
            raise ValueError("Explicit local-work review is required before evaluation.")
        with self._lock():
            artifact = self.review_source(attempt, digest)
            if (
                artifact.parent is None
                or runner.image != self.spec["image"]
                or not runner.qualified
            ):
                raise ValueError("Qualified frozen evaluation runtime required.")
            self._active_proposal = attempt
            try:
                self._check_budget("candidate", self.spec["seconds"] + JOB_GRACE_SECONDS)
                self._append(
                    {
                        "kind": "generation_reviewed",
                        "attempt": attempt,
                        "candidate": digest,
                        "reviewed": True,
                    }
                )
                return self._evaluate_submission(artifact.source, artifact.parent, runner, attempt)
            finally:
                self._active_proposal = None

    def reject(self, attempt: int, reason: str, *, confirmed_stopped: bool = False) -> None:
        """Reject pending source or explicitly abandon an inspected unknown generation.

        Args:
            attempt (int): Unresolved proposal identity.
            reason (str): Bounded operator reason, never a pasted credential/error payload.
            confirmed_stopped (bool): Explicit operator inspection of unknown owned processes.

        Raises:
            ValueError: If the state/reason is invalid or unknown work is still active.
        """
        if not isinstance(reason, str) or not 0 < len(reason) <= 500:
            raise ValueError("A bounded non-secret rejection/abandonment reason is required.")
        with self._lock():
            entry = self.proposal(attempt)
            unknown = entry["status"] in {"generation_unknown", "evaluating"}
            if entry["status"] != "pending_review" and not unknown:
                raise ValueError("Only unresolved Pi proposals can be rejected/abandoned.")
            if unknown and (
                confirmed_stopped is not True or PiProcess.groups_active(entry["processes"])
            ):
                raise ValueError(
                    "Inspect and stop every owned generation process before abandonment."
                )
            self._append(
                {
                    "kind": "generation_abandoned" if unknown else "generation_rejected",
                    "attempt": attempt,
                    "status": "abandoned_unknown" if unknown else "rejected",
                    "reason": reason,
                    "monetary_cost": None,
                }
            )
