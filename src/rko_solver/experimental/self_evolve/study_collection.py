"""Durable paired-panel collection.

Objective:
- Reserve every job before execution and never retry unknown work.
Technical approach:
- Lock the journal, admit bounded cells and retain conservative charges on interruption.
- Score verified evidence against exact frozen optima and persist integrity failures.
Flow (pseudocode):
1. Admit the collection chunk and charge session startup.
2. Load and verify each job, then commit its reservation before calling the runner.
3. Persist the result before checking integrity stops and runtime restart requests.
Important criteria:
- Unknown cells are never retried; exact-reference and strict-candidate gates are unchanged.
- Resource exhaustion pauses collection without inventing missing measurements.
Inputs/Outputs:
- Input: approved frozen study, qualified runtime and optional chunk boundary.
- Output: durable measurements and counts, never partial statistical inference.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from .artifacts import DecoderArtifact
from .problems import Problem
from .sandbox import JOB_GRACE_SECONDS
from .sandbox import PodmanRunner
from .sandbox import SandboxError
from .sandbox_evidence import WorkerEvidence
from .study_state import StudyState
from typing import Any


class StudyCollection(StudyState):
    """Collect locked jobs with conservative durable accounting.

    Responsibilities:
    - Admit chunks and reserve cells before external execution.
    - Persist scores and stop on reference or runtime integrity failures.

    Attributes:
        spec (dict[str, Any]): Frozen study declaration.
    """

    def _start_chunk(self, db: sqlite3.Connection, startup_charge: float) -> tuple[float, float]:
        """Admit collection and commit session accounting before scanning pending jobs.

        Args:
            db (sqlite3.Connection): Exclusively locked measurement journal.
            startup_charge (float): Approved conservative session startup charge.

        Returns:
            tuple[float, float]: Accumulated setup charge and original wall-clock start.

        Raises:
            ValueError: If database identity or prior durable state forbids collection.
        """
        if (
            db.execute("SELECT value FROM state WHERE key='specification'").fetchone()[0]
            != self.digest
        ):
            raise ValueError("Measurement database belongs to a different specification.")
        self._admit_collection(db, active=True)
        previous_setup = db.execute("SELECT value FROM state WHERE key='setup_charge'").fetchone()
        setup_charge = (float(previous_setup[0]) if previous_setup else 0) + startup_charge
        db.execute("INSERT OR REPLACE INTO state VALUES ('setup_charge', ?)", (str(setup_charge),))
        db.execute("INSERT OR IGNORE INTO state VALUES ('started', ?)", (str(time.time()),))
        db.commit()
        started = float(db.execute("SELECT value FROM state WHERE key='started'").fetchone()[0])
        return setup_charge, started

    def _within_budget(
        self, db: sqlite3.Connection, setup_charge: float, started: float, reserve: float
    ) -> bool:
        """Check remaining job admission using original inclusive/exclusive boundaries.

        Args:
            db (sqlite3.Connection): Locked measurement journal.
            setup_charge (float): Current accumulated startup charge.
            started (float): Original study wall-clock start.
            reserve (float): Conservative next-cell reservation.

        Returns:
            bool: Whether the job fits both budgets without a backwards clock.
        """
        charged = (
            db.execute("SELECT COALESCE(SUM(charge), 0) FROM jobs").fetchone()[0] + setup_charge
        )
        elapsed = time.time() - started
        return not (
            elapsed < 0
            or elapsed + reserve >= self.spec["elapsed_budget"]
            or charged + reserve > self.spec["cpu_budget"]
        )

    def _collect(
        self, runner: PodmanRunner, max_jobs: int | None, startup_charge: float
    ) -> dict[str, Any]:
        """Execute an already approved chunk in the immutable pending-job order.

        Args:
            runner (PodmanRunner): Qualified runtime matching the frozen declaration.
            max_jobs (int | None): Optional number of additional cells before pausing.
            startup_charge (float): Validated conservative session charge.

        Returns:
            dict[str, Any]: Verified progress without interim quality feedback.

        Raises:
            ValueError: If durable state or measured reference/runtime integrity fails.
        """
        completed = 0
        with (
            self._lock(),
            contextlib.closing(sqlite3.connect(self.path / "measurements.sqlite")) as db,
        ):
            setup_charge, started = self._start_chunk(db, startup_charge)
            for (job_id,) in db.execute(
                "SELECT id FROM jobs WHERE status='pending' ORDER BY id"
            ).fetchall():
                reserve = self.spec["seconds"] + JOB_GRACE_SECONDS
                if not self._within_budget(db, setup_charge, started, reserve) or (
                    max_jobs is not None and completed >= max_jobs
                ):
                    break
                result = self._measure(db, runner, job_id, reserve)
                completed += 1
                if result["status"] in {"reference_error", "infrastructure_error"}:
                    runner.qualified = False
                    raise ValueError("Reference/runtime integrity failed; stop the entire study.")
                if getattr(runner, "needs_restart", False):
                    break
        return self.progress()

    def _measure(
        self, db: sqlite3.Connection, runner: PodmanRunner, job_id: int, reserve: float
    ) -> dict[str, Any]:
        """Verify inputs, reserve a cell durably and persist its bounded execution result.

        Args:
            db (sqlite3.Connection): Locked measurement journal.
            runner (PodmanRunner): Approved qualified runtime.
            job_id (int): Index in the frozen randomized job schedule.
            reserve (float): Full conservative charge retained on unknown execution.

        Returns:
            dict[str, Any]: Persisted scored evidence or a categorized sandbox failure.

        Raises:
            ValueError: If frozen source identity changed.
            OSError: If frozen inputs cannot be read.
            BaseException: Unhandled execution interruptions retain the running reservation.
        """
        family, index, seed, arm = self.spec["jobs"][job_id]
        role = (
            "candidate"
            if arm in {"candidate", "random", "fixed"}
            else ("seed" if arm == "constructive" else arm)
        )
        artifact = DecoderArtifact.load(self.path / "artifacts" / family / role)
        if artifact.digest != self.spec["artifacts"][family][role]["sha256"]:
            raise ValueError("Frozen study decoder hash mismatch.")
        entry = self.spec["instances"][family][index]
        problem = Problem(family, entry["data"])
        db.execute("UPDATE jobs SET status='running', charge=? WHERE id=?", (reserve, job_id))
        db.commit()
        try:
            mode = arm if arm in {"random", "fixed", "constructive"} else "rko"
            evidence = runner.run(
                artifact, problem, mode=mode, seconds=self.spec["seconds"], seed=seed
            )
            result = self._score(evidence, family, role, entry["optimum"])
            charge = WorkerEvidence(result).charge(reserve)
        except SandboxError as error:
            charge = reserve
            result = {
                "status": "failed" if runner.qualified else "infrastructure_error",
                "error": str(error)[:500],
                "error_reason": error.reason,
                "error_phase": error.phase,
                "loss": 1.0,
                "strict_candidate_gate": role != "candidate",
            }
        db.execute(
            "UPDATE jobs SET status='complete', charge=?, result=? WHERE id=?",
            (charge, json.dumps(result, allow_nan=False), job_id),
        )
        db.commit()
        return result

    @staticmethod
    def _score(evidence: dict[str, Any], family: str, role: str, optimum: float) -> dict[str, Any]:
        """Apply the exact reference check and bounded loss to independently verified evidence.

        Args:
            evidence (dict[str, Any]): Runner evidence with feasibility and objective fields.
            family (str): Problem family determining objective direction.
            role (str): Frozen source role, including candidate-derived controls.
            optimum (float): Exact optimum frozen in the declaration.

        Returns:
            dict[str, Any]: Evidence augmented with loss, strict gate and integrity status.
        """
        invalid = role == "candidate" and evidence["infeasible_evaluations"] > 0
        contradiction = evidence["feasible"] and (
            evidence["objective"] > optimum
            if family == "knapsack"
            else evidence["objective"] < optimum
        )
        regret = abs(evidence["objective"] - optimum) / max(1, abs(optimum))
        loss = 0.5 * regret / (1 + regret) if evidence["feasible"] and not invalid else 1.0
        return {
            **evidence,
            "loss": loss,
            "strict_candidate_gate": not invalid,
            "status": "reference_error" if contradiction else "ok",
        }
