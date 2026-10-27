"""Durable Pi proposal state interpretation independent of evaluation-cell reservations.

Objective:
- Preserve consumed attempts across generation, human review, evaluation and interruptions.
Technical approach:
- Derive proposal state from append-only campaign events without changing old journal schemas.
Flow (pseudocode):
1. Index Pi-tagged submission identities and their bounded lifecycle records.
2. Apply known generation/review/decision transitions in capture order.
3. Reject unresolved work except under the object's exact active approval identity.
Important criteria:
- Generation reservations never masquerade as evaluation-cell reservation/evaluation pairs.
- Missing terminal events remain unknown, never successful or automatically retryable.
Inputs/Outputs:
- Input: bounded trusted-controller campaign events.
- Output: proposal summaries and fail-closed admission decisions.
"""

from __future__ import annotations

from typing import Any


class ProposalJournal:
    """Interpret proposal states without mutating the append-only evidence.

    Responsibilities:
    - Keep unknown/pending attempts visible and prevent implicit review or free retries.

    Attributes:
        entries (dict[int, dict]): Indexed Pi proposal states and owned process identities.
    """

    def __init__(self, events: list[dict[str, Any]]) -> None:
        """Derive all known proposal states from bounded controller events.

        Args:
            events (list[dict[str, Any]]): Existing failure-inclusive capture order.

        Raises:
            ValueError: If a Pi submission has an invalid identity.
        """
        self.entries: dict[int, dict[str, Any]] = {}
        for event in events:
            attempt = event.get("attempt")
            if event["kind"] == "submission" and event.get("transport") == "pi_rpc":
                if type(attempt) is not int:
                    raise ValueError("Invalid Pi submission identity in journal.")
                self.entries[attempt] = {
                    **event,
                    "status": "generation_unknown",
                    "processes": [],
                    "usage": None,
                    "usage_complete": False,
                    "monetary_cost": None,
                    "elapsed_seconds": None,
                }
            elif attempt in self.entries:
                self._apply(self.entries[attempt], event)

    @staticmethod
    def _apply(entry: dict[str, Any], event: dict[str, Any]) -> None:
        """Apply one bounded proposal-specific transition while retaining submission identity.

        Args:
            entry (dict[str, Any]): Existing indexed attempt, updated in place.
            event (dict[str, Any]): Later durable process/generation/review/decision event.
        """
        kind = event["kind"]
        if kind == "generation_process":
            entry["processes"].append(event["pid"])
        elif kind == "generation_finished":
            entry.update({k: v for k, v in event.items() if k not in {"kind", "at"}})
        elif kind == "generation_reviewed":
            entry["status"] = "evaluating"
            entry["reviewed"] = event.get("reviewed") is True
        elif kind in {"generation_rejected", "generation_abandoned", "decision"}:
            entry["status"] = event.get("status", "rejected")

    def verify_proposer(self, digest: str) -> None:
        """Keep recorded submission identities authoritative over a substituted settings file.

        Args:
            digest (str): Canonical non-secret declaration identity proposed for the next call.

        Raises:
            ValueError: If any prior Pi submission binds another proposer declaration.
        """
        if any(entry.get("proposer_sha256") != digest for entry in self.entries.values()):
            raise ValueError("Frozen Pi proposer settings changed from recorded submissions.")

    def admit(self, active: int | None = None) -> None:
        """Block unresolved generation/review except the exact object's explicit approval.

        Args:
            active (int | None): Object-local reviewed attempt currently being evaluated.

        Raises:
            ValueError: If unknown, pending or interrupted proposal work forbids new submissions.
        """
        for attempt, entry in self.entries.items():
            status = entry["status"]
            if active == attempt and status in {"pending_review", "evaluating"}:
                continue
            if status in {"generation_unknown", "pending_review", "evaluating"}:
                raise ValueError("Unresolved Pi proposal; inspect and explicitly review/reject it.")

    def entry(self, attempt: int) -> dict[str, Any]:
        """Find an explicit existing proposal without accepting boolean or unknown identities.

        Args:
            attempt (int): One reserved submission identity.

        Returns:
            dict[str, Any]: Derived private controller metadata, not an accepted artifact.

        Raises:
            ValueError: If identity does not name an existing Pi proposal.
        """
        if type(attempt) is not int or attempt not in self.entries:
            raise ValueError("Unknown Pi proposal identity.")
        return self.entries[attempt]
