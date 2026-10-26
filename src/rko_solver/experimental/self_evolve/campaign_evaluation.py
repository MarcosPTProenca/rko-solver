"""Durable campaign cell execution and frozen evaluation phases.

Objective:
- Execute baseline and candidate cells without changing accounting or screening rules.

Technical approach:
- Bundle cell coordinates and use controller callbacks for durable reservations and budgets.
- Reuse baseline evidence and execute the frozen candidate validation ladder sequentially.

Flow (pseudocode):
1. Verify runtime and budget, then append a reservation before worker execution.
2. Record worker failures, exact-reference checks and the net reservation adjustment.
3. Resume baseline cells or run candidate checks, screen and development controls.

Important criteria:
- Unknown reservations remain durable; infrastructure/reference failures stop collection.
- Screen cells remain distinct from development cells and never replace them.

Inputs/Outputs:
- Input: frozen specification, event callbacks, qualified runner and source artifact.
- Output: failure-inclusive evaluation records or explicit budget/integrity exceptions.
"""

from __future__ import annotations

import itertools
import typing
from collections.abc import Callable
from dataclasses import dataclass
from rko_solver.experimental.self_evolve.artifacts import DecoderArtifact
from rko_solver.experimental.self_evolve.problems import Problem
from rko_solver.experimental.self_evolve.sandbox import JOB_GRACE_SECONDS
from rko_solver.experimental.self_evolve.sandbox import PodmanRunner
from rko_solver.experimental.self_evolve.sandbox import SandboxError
from rko_solver.experimental.self_evolve.sandbox_evidence import WorkerEvidence


@dataclass(frozen=True)
class CampaignJob:
    """Identify one frozen evaluation cell and its accounting partition.

    Responsibilities:
    - Carry source identity and solver coordinates without long execution signatures.

    Attributes:
        artifact (DecoderArtifact): Source-addressed decoder.
        index (int): Development instance index.
        seed (int): Solver seed.
        mode (str): Worker evaluation mode.
        partition (str): Baseline or candidate allocation.
        stage (str): Screen or development cell label.
    """

    artifact: DecoderArtifact
    index: int
    seed: int
    mode: str
    partition: str
    stage: str = "development"


class CampaignEvaluation:
    """Execute frozen evaluation phases through trusted controller callbacks.

    Responsibilities:
    - Reserve and reconcile each cell and enforce exact-reference integrity.
    - Resume baseline measurements and apply candidate screening in fixed order.

    Attributes:
        spec (dict[str, typing.Any]): Frozen campaign specification.
        events (Callable): Read durable controller evidence.
        append (Callable): Append and synchronize one event.
        check_budget (Callable): Enforce controller accounting and integrity stops.
    """

    def __init__(
        self,
        spec: dict[str, typing.Any],
        events: Callable[[], list[dict[str, typing.Any]]],
        append: Callable[[dict[str, typing.Any]], None],
        check_budget: Callable[[str, float], None],
    ) -> None:
        """Bind evaluation to the controller's frozen state and durable operations.

        Args:
            spec (dict[str, typing.Any]): Frozen work and runtime limits.
            events (Callable): Read current events.
            append (Callable): Durably append an event.
            check_budget (Callable): Check a partition and proposed reservation.
        """
        self.spec = spec
        self.events = events
        self.append = append
        self.check_budget = check_budget

    def job(self, runner: PodmanRunner, job: CampaignJob) -> dict[str, typing.Any]:
        """Reserve, execute and durably reconcile one evaluation cell.

        Args:
            runner (PodmanRunner): Qualified pinned worker runtime.
            job (CampaignJob): Source identity and frozen cell coordinates.

        Returns:
            dict[str, typing.Any]: Worker evidence with campaign accounting and loss.

        Raises:
            ValueError: If runtime qualification or reference integrity fails.
            RuntimeError: If the budget callback refuses new work.
        """
        if runner.image != self.spec["image"] or not runner.qualified:
            raise ValueError("Campaign requires its qualified pinned runtime image.")
        reserve = self.spec["seconds"] + JOB_GRACE_SECONDS
        self.check_budget(job.partition, reserve)
        cell = [job.artifact.digest, job.index, job.seed, job.mode, job.stage]
        self.append(
            {"kind": "reservation", "partition": job.partition, "charge": reserve, "cell": cell}
        )
        entry = self.spec["instances"][job.index]
        problem = Problem(entry["family"], entry["data"])
        record, charge = self._execute(runner, job, problem, reserve)
        record = {
            **record,
            "kind": "evaluation",
            "cell": cell,
            "partition": job.partition,
            "candidate": job.artifact.digest,
            "instance": job.index,
            "seed": job.seed,
            "mode": job.mode,
            "charge": charge - reserve,
            "stage": job.stage,
        }
        self._reference_loss(record, problem.family, entry["optimum"])
        self.append(record)
        if record["status"] in {"infrastructure_error", "reference_error"}:
            runner.qualified = False
            raise ValueError("Runtime/reference integrity failed; stop the entire campaign.")
        return record

    def _execute(
        self, runner: PodmanRunner, job: CampaignJob, problem: Problem, reserve: float
    ) -> tuple[dict[str, typing.Any], float]:
        """Run a reserved cell and convert sandbox failures into charged evidence.

        Args:
            runner (PodmanRunner): Qualified worker runtime.
            job (CampaignJob): Evaluation coordinates and artifact.
            problem (Problem): Frozen development instance.
            reserve (float): Maximum cell charge.

        Returns:
            tuple[dict[str, typing.Any], float]: Worker record and bounded charge.
        """
        try:
            record = runner.run(
                job.artifact, problem, mode=job.mode, seed=job.seed, seconds=self.spec["seconds"]
            )
            charge = WorkerEvidence(record).charge(reserve)
        except SandboxError as error:
            record = {
                "status": "error" if runner.qualified else "infrastructure_error",
                "error": str(error)[:500],
                "error_reason": error.reason,
                "error_phase": error.phase,
            }
            charge = reserve
        return record, charge

    @staticmethod
    def _reference_loss(record: dict[str, typing.Any], family: str, optimum: int) -> None:
        """Check feasible witnesses against the exact reference and assign bounded loss.

        Args:
            record (dict[str, typing.Any]): Evaluation record updated in place.
            family (str): Determines objective direction.
            optimum (int): Frozen exact objective value.
        """
        if record["status"] == "ok":
            if record["feasible"] and (
                record["objective"] > optimum
                if family == "knapsack"
                else record["objective"] < optimum
            ):
                record.update(
                    status="reference_error", error="Witness contradicts the exact reference."
                )
        if record["status"] == "ok":
            regret = abs(record["objective"] - optimum) / max(1, abs(optimum))
            record["loss"] = 0.5 * regret / (1 + regret) if record["feasible"] else 1.0
        else:
            record["loss"] = 1.0

    def baseline(
        self, runner: PodmanRunner, artifact: DecoderArtifact
    ) -> list[dict[str, typing.Any]]:
        """Resume missing baseline cells in the original instance/seed/mode order.

        Args:
            runner (PodmanRunner): Qualified worker runtime.
            artifact (DecoderArtifact): Campaign root seed.

        Returns:
            list[dict[str, typing.Any]]: Existing or newly measured baseline cells.

        Raises:
            ValueError: If cell execution detects an integrity failure.
            RuntimeError: If a budget callback refuses new work.
        """
        records = []
        modes = ("check", "rko", "random", "fixed", "constructive")
        for index, seed, mode in itertools.product(range(3), self.spec["solver_seeds"], modes):
            job = CampaignJob(artifact, index, seed, mode, "baseline")
            previous = self._previous(job)
            records.append(previous if previous is not None else self.job(runner, job))
        return records

    def _previous(self, job: CampaignJob) -> dict[str, typing.Any] | None:
        """Find the latest matching baseline cell using the original resume keys.

        Args:
            job (CampaignJob): Requested baseline cell.

        Returns:
            dict[str, typing.Any] | None: Last matching evaluation, if any.
        """
        previous = [
            e
            for e in self.events()
            if e["kind"] == "evaluation"
            and e.get("partition") == "baseline"
            and e.get("instance") == job.index
            and e.get("seed") == job.seed
            and e.get("mode") == job.mode
        ]
        return previous[-1] if previous else None

    def candidate(self, runner: PodmanRunner, artifact: DecoderArtifact) -> None:
        """Apply source-output checks, the seed screen and the full control ladder.

        Args:
            runner (PodmanRunner): Qualified worker runtime.
            artifact (DecoderArtifact): Reviewed candidate artifact.

        Raises:
            ValueError: If any candidate gate or runtime/reference integrity check fails.
            RuntimeError: If a budget callback refuses new work.
        """
        for index in range(3):
            check = self.job(runner, CampaignJob(artifact, index, 0, "check", "candidate"))
            if check["status"] != "ok" or check.get("infeasible_evaluations", 1):
                raise ValueError("Candidate failed endpoint/shape/feasibility checks.")
        self._screen(runner, artifact)
        for index, seed, mode in itertools.product(
            range(3), self.spec["solver_seeds"], ("rko", "random", "fixed")
        ):
            record = self.job(runner, CampaignJob(artifact, index, seed, mode, "candidate"))
            if record["status"] != "ok" or record.get("infeasible_evaluations", 1):
                raise ValueError("Candidate failed a strict returned-solution gate.")

    def _screen(self, runner: PodmanRunner, artifact: DecoderArtifact) -> None:
        """Compare a separately labelled first-instance screen to the seed RKO score.

        Args:
            runner (PodmanRunner): Qualified worker runtime.
            artifact (DecoderArtifact): Reviewed candidate artifact.

        Raises:
            ValueError: If the screen is unsuccessful, infeasible or worse than baseline.
            StopIteration: If required baseline evidence is absent.
            RuntimeError: If a budget callback refuses new work.
        """
        screen = self.job(runner, CampaignJob(artifact, 0, 0, "rko", "candidate", "screen"))
        baseline = next(
            e
            for e in self.events()
            if e["kind"] == "evaluation"
            and e.get("partition") == "baseline"
            and e.get("instance") == 0
            and e.get("seed") == 0
            and e.get("mode") == "rko"
        )
        if (
            screen["status"] != "ok"
            or screen.get("infeasible_evaluations", 1)
            or screen["score"] > baseline["score"]
        ):
            raise ValueError("Candidate failed the frozen feasibility/baseline screen.")
