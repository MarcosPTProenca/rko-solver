"""Locked artifact-comparison panels with durable reservations.

Objective:
- Collect fresh paired evidence without adaptive selection.
Technical approach:
- Compose typed definition, durable collection and complete-panel reporting.
- Authorize runtime setup only after durable reservation and exact digest approval.
Flow (pseudocode):
1. Freeze artifacts, fresh instances, seeds, budgets and analysis settings through typed keywords.
2. Require exact approval and reserve runtime startup before construction and qualification.
3. Collect bounded paired jobs, retain lifecycle stops and analyze only complete panels.
Important criteria:
- Preserve specification schema, operation order, reservations and inference gates.
- This is a frozen-artifact case study, not replicated evidence about evolution reliability.
- External power declarations are not collector-issued power certificates.
Inputs/Outputs:
- Input: explicit typed experiment keywords, exact approval and a qualified pinned runtime.
- Output: resumable measurements and evidence-labelled reports, never production deployment.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
import uuid
from .sandbox import PodmanRunner
from .session import PersistentPodmanRunner
from .study_collection import StudyCollection
from .study_definition import StudyDeclaration
from .study_options import StudyOptions
from .study_options import StudySettings
from .study_report import StudyReport
from .study_runtime import StudyRuntimeSession
from .study_state import ARMS as ARMS
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from typing import Self
from typing import Unpack


class ConfirmationStudy(StudyCollection, StudyReport):
    """Own one non-adaptive frozen-artifact comparison study.

    Responsibilities:
    - Freeze cohesive typed declarations and enforce exact runtime approval.
    - Compose durable collection and complete-panel reporting without adaptive selection.

    Attributes:
        path (Path): Private controller directory.
        spec (dict[str, Any]): Immutable hash-verified declaration.
        digest (str): Identity required for execution approval.
    """

    @classmethod
    def create(
        cls,
        path: Path,
        **options: Unpack[StudyOptions],
    ) -> Self:
        """Freeze all inputs before exposing test outcomes or requesting execution approval.

        Args:
            path (Path): New study directory.
            options (Unpack[StudyOptions]): Typed declaration with the following keywords.
            artifacts (dict): Each family maps seed/first_shot/candidate to existing artifacts.
            instances (dict): Two to one hundred fresh test instances per family.
            prior_instance_hashes (list[str]): Every previously exposed instance identity.
            image (str): Explicit immutable local runtime ID.
            seeds (list[int]): Two to one hundred distinct solver seeds.
            seconds (float): Common inner allowance, at most thirty seconds.
            cpu_budget (float): Explicit container CPU upper-bound budget in seconds.
            elapsed_budget (float): Explicit campaign wall-time budget in seconds.
            minimum_effect (float): Required absolute reduction in the bounded mean loss.
            execution_seed (int): Seed for paired-block and within-block randomization.
            claim (str): exploratory or confirmatory; never a method-level claim.
            power_plan (dict | None): Frozen external power justification for confirmation.
            allow_runtime_rebind (bool): Explicitly move unchanged source to a new runtime.
            bootstrap_samples (int): Frozen bounded instance-bootstrap count.
            permutation_samples (int): Frozen bounded Monte Carlo sign-flip count.
            strata (dict | None): Fixed sampling labels with at least two cases per stratum.
            transport (str): Explicit per_job or fresh-process persistent_exec transport.

        Returns:
            ConfirmationStudy: Unexecuted, frozen panel requiring digest approval.

        Raises:
            ValueError: If inputs, freshness, resource limits or power declaration are invalid.
            FileExistsError: If output exists; no source/data are overwritten.
        """
        StudyDeclaration(StudySettings(**options), cls).freeze(path)
        return cls(path)

    def run(
        self,
        runner: PodmanRunner,
        *,
        approved_spec: str,
        max_jobs: int | None = None,
        startup_charge: float = 0,
    ) -> dict[str, Any]:
        """Collect approved bounded jobs without exposing partial statistical inference.

        Args:
            runner (PodmanRunner): Qualified runner for the exact frozen image.
            approved_spec (str): Exact approved specification digest.
            max_jobs (int | None): Optional safe pause after this many additional jobs.
            startup_charge (float): Conservative startup charge for the qualified session.

        Returns:
            dict[str, Any]: Counts and resource charges, without adaptive quality feedback.

        Raises:
            ValueError: If approval, runtime, chunk settings or exact references are invalid.
        """
        self._verify()
        if self._invalid_runtime(runner, approved_spec, startup_charge) or self._invalid_chunk(
            startup_charge, max_jobs
        ):
            raise ValueError("Exact specification approval and its qualified runtime are required.")
        return self._collect(runner, max_jobs, startup_charge)

    def _invalid_runtime(
        self, runner: PodmanRunner, approved_spec: str, startup_charge: float
    ) -> bool:
        """Check exact approval, image, transport and object-local runtime authorization.

        Args:
            runner (PodmanRunner): Proposed qualified runtime.
            approved_spec (str): Caller approval identity.
            startup_charge (float): Proposed external startup accounting charge.

        Returns:
            bool: Whether any runtime admission constraint rejects collection.
        """
        return (
            approved_spec != self.digest
            or runner.image != self.spec["image"]
            or not runner.qualified
            or (
                isinstance(runner, PodmanRunner)
                and (not getattr(self, "_runtime_token", None) or startup_charge != 0)
            )
            or getattr(runner, "transport", "per_job") != self.spec["transport"]
        )

    @staticmethod
    def _invalid_chunk(startup_charge: float, max_jobs: int | None) -> bool:
        """Validate conservative startup accounting and the optional chunk boundary.

        Args:
            startup_charge (float): Proposed session startup charge.
            max_jobs (int | None): Proposed additional job count.

        Returns:
            bool: Whether the charge or chunk count violates the original domain.
        """
        return (
            type(startup_charge) not in (int, float)
            or not math.isfinite(startup_charge)
            or startup_charge < 0
            or (max_jobs is not None and (type(max_jobs) is not int or max_jobs < 1))
        )

    @contextlib.contextmanager
    def runtime(self, *, approved_spec: str) -> Iterator[PodmanRunner]:
        """Reserve startup before construction/qualification and retain unknown setup charges.

        Args:
            approved_spec (str): Exact owner-approved specification digest.

        Yields:
            PodmanRunner: Qualified runtime authorized for this study object only.

        Raises:
            ValueError: If prior state or remaining budgets cannot admit bounded setup and a job.
            SandboxError: If runtime startup/qualification/cleanup fails.
        """
        self._verify()
        if approved_spec != self.digest:
            raise ValueError("Approve the exact specification before runtime setup.")
        token = uuid.uuid4().hex
        session = StudyRuntimeSession(self, token)
        session.reserve()
        began = time.monotonic()
        stack = contextlib.ExitStack()
        try:
            context = (
                PersistentPodmanRunner(
                    self.spec["image"], lease_seconds=min(86400, self.spec["elapsed_budget"] + 120)
                )
                if self.spec["transport"] == "persistent_exec"
                else contextlib.nullcontext(PodmanRunner(self.spec["image"]))
            )
            runner = stack.enter_context(context)
            qualification = runner.qualify()
            session.qualified(time.monotonic() - began + 0.1)
            with (self.path / "qualifications.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(qualification, allow_nan=False) + "\n")
            self._runtime_token = token
            session.ready = True
            yield runner
        finally:
            closing = time.monotonic()
            cleaned = False
            try:
                stack.close()
                cleaned = True
            finally:
                session.finish(cleaned, closing)
                self._runtime_token = None
