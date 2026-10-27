"""Explicit CLI operations for Pi proposals and human-reviewed evaluation.

Objective:
- Keep generation, inspection and source approval separate user actions.
Technical approach:
- Extend the existing parser without changing manual or study command defaults.
Flow (pseudocode):
1. Load explicit non-secret Pi settings or inspect an existing proposal.
2. Generate at most one pending source or record explicit rejection/abandonment.
3. Verify source hash and human attestation before constructing an evaluation runtime.
Important criteria:
- No automatic evaluation, credential extraction or direct API fallback exists.
Inputs/Outputs:
- Input: explicit campaign/configuration paths and review/consent flags.
- Output: bounded JSON lifecycle records or fail-closed precondition errors.
"""

from __future__ import annotations

import argparse
import typing
from pathlib import Path
from rko_solver.experimental.self_evolve.pi_config import PI_VERSION
from rko_solver.experimental.self_evolve.pi_config import PiConfig
from rko_solver.experimental.self_evolve.proposals import PiCampaign
from rko_solver.experimental.self_evolve.sandbox import PodmanRunner

PI_COMMANDS = ("pi-profile", "propose", "proposal", "approve-proposal", "reject-proposal")


class PiCommands:
    """Expose one bounded proposal at a time without attesting review implicitly.

    Responsibilities:
    - Separate source-generation and evaluation capabilities at the public CLI boundary.

    Attributes:
        args (argparse.Namespace): Explicit parsed operation and its bounded settings.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Retain parsed explicit operation arguments.

        Args:
            args (argparse.Namespace): Parsed Pi lifecycle command.
        """
        self.args = args

    @staticmethod
    def add(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
        """Register Pi operations without expanding ordinary campaign defaults.

        Args:
            commands (argparse._SubParsersAction): Existing command parser registry.
        """
        profile = commands.add_parser(
            "pi-profile", help="Create a NEW minimal Pi profile; no login"
        )
        profile.add_argument("config", type=Path)
        propose = commands.add_parser("propose", help="Generate one proposal and stop for review")
        propose.add_argument("campaign", type=Path)
        propose.add_argument("config", type=Path)
        propose.add_argument("--parent")
        propose.add_argument("--consent-development-data", action="store_true")
        for name in ("proposal", "approve-proposal", "reject-proposal"):
            parser = commands.add_parser(name)
            parser.add_argument("campaign", type=Path)
            parser.add_argument("attempt", type=int)
        approve = commands.choices["approve-proposal"]
        approve.add_argument("--sha256", required=True)
        approve.add_argument("--reviewed-local-work", action="store_true")
        reject = commands.choices["reject-proposal"]
        reject.add_argument("--reason", required=True)
        reject.add_argument("--confirmed-stopped", action="store_true")

    def dispatch(self) -> dict[str, typing.Any]:
        """Perform exactly one explicitly selected operation.

        Returns:
            dict[str, Any]: Safe operation result for the existing JSON output path.

        Raises:
            ValueError: If settings, consent, review or durable state are invalid.
            OSError: If required files are unavailable or output already exists.
        """
        args = self.args
        if args.command == "pi-profile":
            config = PiConfig.load(args.config)
            config.prepare_profile()
            return {
                "profile": str(config.profile),
                "pi_version": PI_VERSION,
                "provider": config.provider,
                "authentication": "login_with_pi_required",
            }
        campaign = PiCampaign(args.campaign)
        if args.command == "propose":
            return campaign.generate(
                PiConfig.load(args.config),
                parent=args.parent,
                consent=args.consent_development_data,
            )
        if args.command == "approve-proposal":
            return self._approve(campaign)
        if args.command == "reject-proposal":
            campaign.reject(args.attempt, args.reason, confirmed_stopped=args.confirmed_stopped)
        return campaign.proposal(args.attempt)

    def _approve(self, campaign: PiCampaign) -> dict[str, typing.Any]:
        """Verify review/source identity BEFORE constructing the frozen evaluation runtime.

        Args:
            campaign (PiCampaign): Existing review-gated campaign.

        Returns:
            dict[str, Any]: Ordinary bounded experimental evaluation decision.

        Raises:
            ValueError: If attestation, source, runtime or remaining budgets reject evaluation.
            OSError: If source files are unavailable.
        """
        args = self.args
        if not args.reviewed_local_work:
            raise ValueError("Explicit local-work source review is required before evaluation.")
        campaign.review_source(args.attempt, args.sha256)
        runner = PodmanRunner(campaign.spec["image"])
        runner.qualify()
        return campaign.approve(args.attempt, args.sha256, runner, reviewed=True)
