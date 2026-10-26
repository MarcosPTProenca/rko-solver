"""Locked artifact-comparison panels with durable budget reservations and no adaptive selection.

Objective:
- Collect fresh paired test evidence without reusing development trajectories as confirmation.
Technical approach:
- Freeze source/data/runtime/analysis identities and randomized job order in a private manifest.
- Persist each reservation/result in SQLite; missing/interrupted cells block inference.
Flow (pseudocode):
1. Freeze three-family artifacts, fresh instances, seeds, budgets and effect threshold.
2. Require approval of the exact specification digest and a qualified pinned runner.
3. Execute independently verified jobs in paired randomized blocks without retries or tuning.
4. Analyze only the complete panel using instance-level estimates and six-way Holm correction.
Important criteria:
- This is an artifact case study, not replicated evidence about the evolution procedure.
- A declared power plan is researcher-supplied evidence, not a certificate from this collector.
Inputs/Outputs:
- Input: explicit frozen artifacts/test instances and approved resource envelope.
- Output: resumable measurements and evidence-labelled JSON reports, never production deployment.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import numpy as np
import os
import platform
import sqlite3
import time
import uuid
from ...components.random_streams import RandomStreams
from .analysis import PairedAnalysis
from .artifacts import FAMILIES
from .artifacts import DecoderArtifact
from .problems import Problem
from .sandbox import JOB_GRACE_SECONDS
from .sandbox import PodmanRunner
from .sandbox import SandboxError
from .session import PersistentPodmanRunner
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

ARMS = ("seed", "constructive", "first_shot", "candidate", "random", "fixed")


class ConfirmationStudy:
    """Own one non-adaptive, budgeted, frozen-artifact comparison study.

    Responsibilities:
    - Prevent incomplete panels, seed pseudoreplication and post-result source replacement.

    Attributes:
        path (Path): Private trusted controller directory, never mounted into jobs.
        spec (dict[str, Any]): Hash-verified immutable study declaration.
        digest (str): Specification identity required for explicit execution approval.
    """

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
        current = ConfirmationStudy(self.path)
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
        return ConfirmationStudy._hash({"family": problem.family, "data": problem.data})

    @staticmethod
    def _implementation() -> dict[str, Any]:
        """Fingerprint the measurement/analysis code and numerical runtime.

        Returns:
            dict[str, Any]: Replay-critical controller implementation identity.
        """
        root = Path(__file__).parent
        return {
            "code": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ("study.py", "analysis.py", "problems.py", "sandbox.py", "session.py")
            },
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rko": importlib.metadata.version("rko-solver"),
        }

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        artifacts: dict[str, dict[str, DecoderArtifact]],
        instances: dict[str, list[Problem]],
        prior_instance_hashes: list[str],
        image: str,
        seeds: list[int],
        seconds: float,
        cpu_budget: float,
        elapsed_budget: float,
        minimum_effect: float,
        execution_seed: int = 42,
        claim: str = "exploratory",
        power_plan: dict[str, Any] | None = None,
        allow_runtime_rebind: bool = False,
        bootstrap_samples: int = 10000,
        permutation_samples: int = 9999,
        strata: dict[str, list[str]] | None = None,
        transport: str = "per_job",
    ) -> ConfirmationStudy:
        """Freeze all inputs before exposing test outcomes or requesting execution approval.

        Args:
            path (Path): New study directory.
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
        analysis = PairedAnalysis(
            bootstrap_samples=bootstrap_samples, permutation_samples=permutation_samples
        )
        if (
            set(artifacts) != set(FAMILIES)
            or set(instances) != set(FAMILIES)
            or not isinstance(image, str)
            or len(image) != 71
            or not image.startswith("sha256:")
            or any(c not in "0123456789abcdef" for c in image[7:])
            or not 2 <= len(seeds) <= 100
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)
            or type(execution_seed) is not int
            or not 0 <= execution_seed < 2**32
            or any(
                type(value) not in (int, float)
                for value in (seconds, cpu_budget, elapsed_budget, minimum_effect)
            )
            or not math.isfinite(seconds)
            or not 0 < seconds <= 30
            or not math.isfinite(minimum_effect)
            or not 0 < minimum_effect <= 1
            or any(
                not math.isfinite(b) or not seconds + JOB_GRACE_SECONDS < b <= 604800
                for b in (cpu_budget, elapsed_budget)
            )
            or claim not in {"exploratory", "confirmatory"}
            or transport not in {"per_job", "persistent_exec"}
        ):
            raise ValueError(
                "Invalid frozen study dimensions, runtime, budgets or effect threshold."
            )
        if claim == "confirmatory" and (
            not isinstance(power_plan, dict)
            or not isinstance(power_plan.get("method"), str)
            or not power_plan["method"]
            or len(power_plan["method"]) > 2000
            or type(power_plan.get("planned_power")) not in (int, float)
            or not 0.8 <= power_plan["planned_power"] <= 1
            or type(power_plan.get("calibration_instances")) is not int
            or power_plan["calibration_instances"] < 10
        ):
            raise ValueError("Confirmation requires an explicit externally justified power plan.")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
            for value in prior_instance_hashes
        ):
            raise ValueError("Prior instance identities must be lowercase SHA-256 hashes.")
        if strata is not None and set(strata) != set(FAMILIES):
            raise ValueError("Strata must cover all three declared families.")
        strata = strata or {family: ["all"] * len(instances[family]) for family in FAMILIES}
        for family in FAMILIES:
            analysis.strata_indices(len(instances[family]), strata[family])
        seen = set(prior_instance_hashes)
        snapshots: dict[str, Any] = {}
        source_records: dict[str, Any] = {}
        rebound: dict[str, dict[str, DecoderArtifact]] = {}
        for family in FAMILIES:
            if set(artifacts[family]) != {"seed", "first_shot", "candidate"}:
                raise ValueError(
                    "Every family needs frozen seed, first-shot and candidate artifacts."
                )
            if not 2 <= len(instances[family]) <= 100:
                raise ValueError("Each family requires two to one hundred independent instances.")
            snapshots[family], source_records[family], rebound[family] = [], {}, {}
            for problem in instances[family]:
                if problem.family != family:
                    raise ValueError("Instance family mismatch.")
                digest = cls.instance_hash(problem)
                if digest in seen:
                    raise ValueError(
                        "Duplicate or previously exposed instance; use fresh test data."
                    )
                seen.add(digest)
                snapshots[family].append(
                    {"data": problem.data, "sha256": digest, "optimum": problem.optimum()}
                )
            if artifacts[family]["first_shot"].parent != artifacts[family]["seed"].digest:
                raise ValueError("First-shot source must identify the seed as its lineage parent.")
            for role, artifact in artifacts[family].items():
                if artifact.family != family:
                    raise ValueError("Artifact family mismatch.")
                if artifact.qualified_image not in (None, image) and not allow_runtime_rebind:
                    raise ValueError(
                        "Explicit runtime rebinding is required; source bytes stay unchanged."
                    )
                source_records[family][role] = {
                    "sha256": artifact.digest,
                    "parent": artifact.parent,
                    "original_image": artifact.qualified_image,
                }
                rebound[family][role] = replace(artifact, qualified_image=None)
        blocks = [
            (family, i, seed)
            for family in FAMILIES
            for i in range(len(snapshots[family]))
            for seed in seeds
        ]
        if len(blocks) * len(ARMS) > 20000:
            raise ValueError("Study exceeds the twenty-thousand-job work envelope.")
        rng = RandomStreams(execution_seed).python
        rng.shuffle(blocks)
        jobs = []
        for family, index, seed in blocks:
            arms = list(ARMS)
            rng.shuffle(arms)
            jobs.extend([[family, index, seed, arm] for arm in arms])
        spec = {
            "schema": 1,
            "transport": transport,
            "claim": claim,
            "claim_scope": "frozen_artifact_case_study_only",
            "instances": snapshots,
            "strata": strata,
            "artifacts": source_records,
            "image": image,
            "prior_instance_hashes": sorted(set(prior_instance_hashes)),
            "seeds": seeds,
            "seconds": seconds,
            "cpu_budget": cpu_budget,
            "elapsed_budget": elapsed_budget,
            "minimum_effect": minimum_effect,
            "effect_units": "absolute_mean_loss_reduction",
            "execution_seed": execution_seed,
            "jobs": jobs,
            "power_plan_declared": power_plan,
            "analysis": {
                "bootstrap_samples": analysis.bootstrap_samples,
                "permutation_samples": analysis.permutation_samples,
                "random_state": 42,
                "family_size": 6,
            },
            "primary_baselines": ["seed", "constructive"],
            "alpha": 0.05,
            "implementation": cls._implementation(),
            "rko": {
                "algorithms": ["SA"],
                "adaptation": "fixed",
                "engine": "python",
                "execution": "sequential",
                "pool_size": 4,
                "prefill_pool": False,
            },
        }
        encoded = json.dumps(spec, sort_keys=True, allow_nan=False)
        if len(encoded.encode()) > 16777216:
            raise ValueError("Frozen study exceeds 16 MiB.")
        path.mkdir(parents=True, exist_ok=False, mode=0o700)
        for family in FAMILIES:
            for role, artifact in rebound[family].items():
                artifact.export(path / "artifacts" / family / role)
        (path / "study.json").write_text(encoded, encoding="utf-8")
        (path / "study.sha256").write_text(cls._hash(spec))
        with contextlib.closing(sqlite3.connect(path / "measurements.sqlite")) as database:
            database.execute("CREATE TABLE state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            database.execute(
                "CREATE TABLE jobs (id INTEGER PRIMARY KEY, status TEXT NOT NULL, "
                "charge REAL NOT NULL, result TEXT)"
            )
            database.executemany(
                "INSERT INTO jobs VALUES (?, 'pending', 0, NULL)", [(i,) for i in range(len(jobs))]
            )
            database.execute("INSERT INTO state VALUES ('specification', ?)", (cls._hash(spec),))
            database.commit()
        return cls(path)

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
        reservation, cleanup_reserve = 900.0, 60.0
        token = uuid.uuid4().hex
        with (
            self._lock(),
            contextlib.closing(sqlite3.connect(self.path / "measurements.sqlite")) as db,
        ):
            self._admit_collection(db)
            if not db.execute("SELECT 1 FROM jobs WHERE status='pending' LIMIT 1").fetchone():
                raise ValueError("No pending jobs require runtime setup.")
            row = db.execute("SELECT value FROM state WHERE key='setup_charge'").fetchone()
            setup = float(row[0]) if row else 0.0
            charged = db.execute("SELECT COALESCE(SUM(charge),0) FROM jobs").fetchone()[0] + setup
            started = db.execute("SELECT value FROM state WHERE key='started'").fetchone()
            elapsed = time.time() - float(started[0]) if started else 0
            required = reservation + self.spec["seconds"] + JOB_GRACE_SECONDS
            if (
                elapsed < 0
                or charged + required > self.spec["cpu_budget"]
                or (elapsed + required >= self.spec["elapsed_budget"])
            ):
                raise ValueError("Insufficient budget for runtime setup and another job.")
            db.execute(
                "INSERT OR REPLACE INTO state VALUES ('setup_charge',?)",
                (str(setup + reservation),),
            )
            db.execute("INSERT INTO state VALUES ('setup_pending',?)", (token,))
            db.execute("INSERT OR IGNORE INTO state VALUES ('started',?)", (str(time.time()),))
            db.commit()
        began = time.monotonic()
        stack = contextlib.ExitStack()
        ready = False
        setup_elapsed = reservation
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
            setup_elapsed = time.monotonic() - began + 0.1
            if setup_elapsed + cleanup_reserve > reservation:
                raise ValueError("Runtime setup exhausted its conservative reservation.")
            with (
                self._lock(),
                contextlib.closing(sqlite3.connect(self.path / "measurements.sqlite")) as db,
            ):
                db.execute(
                    "UPDATE state SET value=? WHERE key='setup_charge'",
                    (str(setup + setup_elapsed + cleanup_reserve),),
                )
                db.execute("DELETE FROM state WHERE key='setup_pending'")
                db.execute("INSERT INTO state VALUES ('setup_ready',?)", (token,))
                db.commit()
            with (self.path / "qualifications.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(qualification, allow_nan=False) + "\n")
            self._runtime_token = token
            ready = True
            yield runner
        finally:
            closing = time.monotonic()
            cleaned = False
            try:
                stack.close()
                cleaned = True
            finally:
                with (
                    self._lock(),
                    contextlib.closing(sqlite3.connect(self.path / "measurements.sqlite")) as db,
                ):
                    if ready and cleaned:
                        cleanup = time.monotonic() - closing + 0.1
                        if cleanup > cleanup_reserve:
                            db.execute(
                                "INSERT OR REPLACE INTO state VALUES ('setup_failed',?)", (token,)
                            )
                        db.execute(
                            "UPDATE state SET value=? WHERE key='setup_charge'",
                            (str(setup + setup_elapsed + cleanup),),
                        )
                        db.execute("DELETE FROM state WHERE key='setup_ready'")
                    else:
                        if not ready:
                            db.execute(
                                "UPDATE state SET value=? WHERE key='setup_charge'",
                                (str(setup + reservation),),
                            )
                        db.execute(
                            "INSERT OR REPLACE INTO state VALUES ('setup_failed',?)", (token,)
                        )
                    db.commit()
                self._runtime_token = None

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
            approved_spec (str): Explicit approved manifest digest, not a generic yes flag.
            max_jobs (int | None): Optional safe pause after this many additional jobs.
            startup_charge (float): Measured conservative startup charge for this qualified session.

        Returns:
            dict[str, Any]: Counts and resource charges only; no adaptive quality feedback.

        Raises:
            ValueError: If approval/runtime/chunk settings or exact references are incompatible.
        """
        self._verify()
        if (
            approved_spec != self.digest
            or runner.image != self.spec["image"]
            or not runner.qualified
            or (
                isinstance(runner, PodmanRunner)
                and (not getattr(self, "_runtime_token", None) or startup_charge != 0)
            )
            or getattr(runner, "transport", "per_job") != self.spec["transport"]
            or type(startup_charge) not in (int, float)
            or not math.isfinite(startup_charge)
            or startup_charge < 0
            or (max_jobs is not None and (type(max_jobs) is not int or max_jobs < 1))
        ):
            raise ValueError("Exact specification approval and its qualified runtime are required.")
        completed = 0
        with (
            self._lock(),
            contextlib.closing(sqlite3.connect(self.path / "measurements.sqlite")) as db,
        ):
            if (
                db.execute("SELECT value FROM state WHERE key='specification'").fetchone()[0]
                != self.digest
            ):
                raise ValueError("Measurement database belongs to a different specification.")
            self._admit_collection(db, active=True)
            previous_setup = db.execute(
                "SELECT value FROM state WHERE key='setup_charge'"
            ).fetchone()
            setup_charge = (float(previous_setup[0]) if previous_setup else 0) + startup_charge
            db.execute(
                "INSERT OR REPLACE INTO state VALUES ('setup_charge', ?)", (str(setup_charge),)
            )
            db.execute("INSERT OR IGNORE INTO state VALUES ('started', ?)", (str(time.time()),))
            db.commit()
            started = float(db.execute("SELECT value FROM state WHERE key='started'").fetchone()[0])
            for (job_id,) in db.execute(
                "SELECT id FROM jobs WHERE status='pending' ORDER BY id"
            ).fetchall():
                reserve = self.spec["seconds"] + JOB_GRACE_SECONDS
                charged = (
                    db.execute("SELECT COALESCE(SUM(charge), 0) FROM jobs").fetchone()[0]
                    + setup_charge
                )
                elapsed = time.time() - started
                if (
                    elapsed < 0
                    or elapsed + reserve >= self.spec["elapsed_budget"]
                    or charged + reserve > self.spec["cpu_budget"]
                    or (max_jobs is not None and completed >= max_jobs)
                ):
                    break
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
                db.execute(
                    "UPDATE jobs SET status='running', charge=? WHERE id=?", (reserve, job_id)
                )
                db.commit()
                try:
                    mode = arm if arm in {"random", "fixed", "constructive"} else "rko"
                    evidence = runner.run(
                        artifact, problem, mode=mode, seconds=self.spec["seconds"], seed=seed
                    )
                    charge = min(
                        reserve, max(evidence["cpu_seconds"], evidence["outer_wall_seconds"] + 0.1)
                    )
                    invalid = role == "candidate" and evidence["infeasible_evaluations"] > 0
                    optimum = entry["optimum"]
                    contradiction = evidence["feasible"] and (
                        evidence["objective"] > optimum
                        if family == "knapsack"
                        else evidence["objective"] < optimum
                    )
                    regret = abs(evidence["objective"] - optimum) / max(1, abs(optimum))
                    loss = (
                        0.5 * regret / (1 + regret) if evidence["feasible"] and not invalid else 1.0
                    )
                    result = {
                        **evidence,
                        "loss": loss,
                        "strict_candidate_gate": not invalid,
                        "status": "reference_error" if contradiction else "ok",
                    }
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
                completed += 1
                if result["status"] in {"reference_error", "infrastructure_error"}:
                    raise ValueError("Reference/runtime integrity failed; stop the entire study.")
                if getattr(runner, "needs_restart", False):
                    break
        return self.progress()

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

    def report(self) -> dict[str, Any]:
        """Analyze only complete panels; correct all six primary hypotheses together.

        Returns:
            dict[str, Any]: Labelled complete evidence, or incomplete counts without p-values.

        Raises:
            ValueError: If completed measurements contradict frozen references or panel identity.
        """
        progress = self.progress()
        if not progress["inference_available"]:
            return {"status": "incomplete_no_inference", **progress}
        with contextlib.closing(sqlite3.connect(self.path / "measurements.sqlite")) as db:
            rows = {
                job: json.loads(result) for job, result in db.execute("SELECT id, result FROM jobs")
            }
        panels: dict[str, Any] = {family: {arm: [] for arm in ARMS} for family in FAMILIES}
        cells = {}
        for job_id, job in enumerate(self.spec["jobs"]):
            family, index, seed, arm = job
            result = rows[job_id]
            if result["status"] not in {"ok", "failed"}:
                raise ValueError(
                    "Invalid measurement/reference state; no statistical report is permitted."
                )
            cells[(family, index, seed, arm)] = result
        summaries = {}
        comparisons = []
        analysis = PairedAnalysis(**self.spec["analysis"])
        for family in FAMILIES:
            for arm in ARMS:
                panels[family][arm] = [
                    [cells[(family, i, seed, arm)]["loss"] for seed in self.spec["seeds"]]
                    for i in range(len(self.spec["instances"][family]))
                ]
            summaries[family] = {arm: float(np.mean(panels[family][arm])) for arm in ARMS}
            for baseline in self.spec["primary_baselines"]:
                effect = analysis.compare(
                    panels[family][baseline],
                    panels[family]["candidate"],
                    strata=self.spec["strata"][family],
                )
                comparisons.append({"family": family, "baseline": baseline, **effect})
        adjusted = PairedAnalysis.holm([effect["p_value"] for effect in comparisons])
        for effect, corrected in zip(comparisons, adjusted, strict=True):
            effect["holm_p_value"] = corrected
            effect["passes_numerical_checks"] = (
                corrected <= self.spec["alpha"]
                and effect["familywise_lower_bound_95"] > self.spec["minimum_effect"]
            )
        strict = all(
            result.get("strict_candidate_gate", False)
            for job, result in rows.items()
            if self.spec["jobs"][job][3] in {"candidate", "random", "fixed"}
        )
        constructive = all(
            result.get("status") == "ok" and result.get("feasible") is True
            for job, result in rows.items()
            if self.spec["jobs"][job][3] == "constructive"
        )
        return {
            "status": "complete_artifact_report",
            "claim": self.spec["claim"],
            "claim_scope": self.spec["claim_scope"],
            "specification": self.digest,
            "minimum_effect": self.spec["minimum_effect"],
            "effect_units": self.spec["effect_units"],
            "mean_losses": summaries,
            "primary_comparisons": comparisons,
            "strict_candidate_gate": strict,
            "constructive_control_gate": constructive,
            "passes_predeclared_artifact_checks": self.spec["claim"] == "confirmatory"
            and strict
            and constructive
            and all(effect["passes_numerical_checks"] for effect in comparisons),
            "production_promotion": False,
            "method_reliability_claim": False,
            "power_plan_is_external_declaration": True,
            "seed_target": "conditional_on_the_fixed_declared_solver_seed_panel",
            "secondary_controls": "First-shot/random/fixed means are descriptive only.",
            "limitations": "Bootstrap bounds are approximate. No universal correctness, "
            "hostile-code safety, provider reproducibility or method reliability is established.",
            "progress": progress,
        }
