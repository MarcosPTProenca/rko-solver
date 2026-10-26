"""Non-adaptive study definition and persistence.

Objective:
- Freeze fresh data, nine source artifacts and a balanced randomized schedule.
Technical approach:
- Build exact schema-one declarations before exposing any measurements.
- Snapshot exact optima and original lineage, then rebind unchanged artifact bytes.
- Construct and persist O(J) jobs for J paired instance/seed/arm combinations.
Flow (pseudocode):
1. Validate analysis settings, dimensions, prior exposure and fixed strata.
2. Snapshot instances and sources, shuffle paired blocks, then shuffle arms within each block.
3. Persist bounded canonical JSON, nine artifact exports and a pending-job SQLite journal.
Important criteria:
- Preserve specification schema, operation order, reservations and inference gates.
Inputs/Outputs:
- Input: a typed experiment declaration and a new private controller directory.
- Output: a canonical immutable manifest, lineage-preserving source exports and pending jobs.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from ...components.random_streams import RandomStreams
from .analysis import PairedAnalysis
from .artifacts import FAMILIES
from .artifacts import DecoderArtifact
from .study_options import StudySettings
from .study_state import ARMS
from .study_state import StudyState
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrozenStudyInputs:
    """Carry validated snapshots and rebound sources into manifest persistence.

    Responsibilities:
    - Keep instance snapshots, original source lineage and rebound exports together.

    Attributes:
        instances (dict[str, Any]): Fresh data, identity and exact-optimum records.
        sources (dict[str, Any]): Original artifact lineage and runtime identities.
        artifacts (dict[str, dict[str, DecoderArtifact]]): Unchanged rebound exports.
        strata (dict[str, list[str]]): Validated effective sampling labels.
    """

    instances: dict[str, Any]
    sources: dict[str, Any]
    artifacts: dict[str, dict[str, DecoderArtifact]]
    strata: dict[str, list[str]]


class StudyDeclaration:
    """Build and persist the non-adaptive frozen experiment.

    Responsibilities:
    - Validate fresh instance identities, source lineage and explicit runtime rebinding.
    - Freeze paired execution order and persist the original immutable specification schema.

    Attributes:
        settings (StudySettings): Typed creation settings.
        analysis (PairedAnalysis): Validated inference configuration.
        controller_type (type[StudyState]): Caller class providing declaration identity hooks.
    """

    def __init__(self, settings: StudySettings, controller_type: type[StudyState]) -> None:
        """Validate analysis settings before the experiment declaration.

        Args:
            settings (StudySettings): Explicit normalized experiment settings.
            controller_type (type[StudyState]): Caller class retaining identity hook dispatch.

        Raises:
            ValueError: If analysis counts or declaration settings are invalid.
        """
        self.settings = settings
        self.controller_type = controller_type
        self.analysis = PairedAnalysis(
            bootstrap_samples=settings.bootstrap_samples,
            permutation_samples=settings.permutation_samples,
        )
        settings.validate()

    def _instances(self, family: str, seen: set[str]) -> list[dict[str, Any]]:
        """Snapshot fresh independent instances in declaration order.

        Args:
            family (str): Current declared family.
            seen (set[str]): Prior and current identities updated in place.

        Returns:
            list[dict[str, Any]]: Data, exact optimum and identity snapshots.

        Raises:
            ValueError: If families mismatch or an instance was previously exposed.
        """
        snapshots: list[dict[str, Any]] = []
        for problem in self.settings.instances[family]:
            if problem.family != family:
                raise ValueError("Instance family mismatch.")
            digest = self.controller_type.instance_hash(problem)
            if digest in seen:
                raise ValueError("Duplicate or previously exposed instance; use fresh test data.")
            seen.add(digest)
            snapshots.append({"data": problem.data, "sha256": digest, "optimum": problem.optimum()})
        return snapshots

    def _artifacts(self, family: str) -> tuple[dict[str, Any], dict[str, DecoderArtifact]]:
        """Validate lineage and freeze all source identities before rebinding.

        Args:
            family (str): Current declared family.

        Returns:
            tuple: Original source records and unchanged rebound artifacts.

        Raises:
            ValueError: If lineage, family or explicit runtime authorization fails.
        """
        records: dict[str, Any] = {}
        rebound: dict[str, DecoderArtifact] = {}
        if (
            self.settings.artifacts[family]["first_shot"].parent
            != self.settings.artifacts[family]["seed"].digest
        ):
            raise ValueError("First-shot source must identify the seed as its lineage parent.")
        for role, artifact in self.settings.artifacts[family].items():
            if artifact.family != family:
                raise ValueError("Artifact family mismatch.")
            if artifact.qualified_image not in (None, self.settings.image) and (
                not self.settings.allow_runtime_rebind
            ):
                raise ValueError(
                    "Explicit runtime rebinding is required; source bytes stay unchanged."
                )
            records[role] = {
                "sha256": artifact.digest,
                "parent": artifact.parent,
                "original_image": artifact.qualified_image,
            }
            rebound[role] = replace(artifact, qualified_image=None)
        return records, rebound

    def _snapshots(self) -> FrozenStudyInputs:
        """Validate strata then snapshot all families in their original order.

        Returns:
            FrozenStudyInputs: Instance snapshots, source records, rebound sources and strata.

        Raises:
            ValueError: If strata, dimensions, freshness or source lineage is invalid.
        """
        if self.settings.strata is not None and set(self.settings.strata) != set(FAMILIES):
            raise ValueError("Strata must cover all three declared families.")
        strata = self.settings.strata or {
            family: ["all"] * len(self.settings.instances[family]) for family in FAMILIES
        }
        for family in FAMILIES:
            self.analysis.strata_indices(len(self.settings.instances[family]), strata[family])
        seen = set(self.settings.prior_instance_hashes)
        snapshots: dict[str, Any] = {}
        source_records: dict[str, Any] = {}
        rebound: dict[str, dict[str, DecoderArtifact]] = {}
        for family in FAMILIES:
            if set(self.settings.artifacts[family]) != {"seed", "first_shot", "candidate"}:
                raise ValueError(
                    "Every family needs frozen seed, first-shot and candidate artifacts."
                )
            if not 2 <= len(self.settings.instances[family]) <= 100:
                raise ValueError("Each family requires two to one hundred independent instances.")
            snapshots[family] = self._instances(family, seen)
            source_records[family], rebound[family] = self._artifacts(family)
        return FrozenStudyInputs(snapshots, source_records, rebound, strata)

    def _jobs(self, snapshots: dict[str, Any]) -> list[list[Any]]:
        """Shuffle paired blocks then arms using the original RNG operation order.

        Args:
            snapshots (dict[str, Any]): Ordered family instance snapshots.

        Returns:
            list[list[Any]]: Frozen balanced job schedule.

        Raises:
            ValueError: If the twenty-thousand-job envelope is exceeded.
        """
        blocks = [
            (family, i, seed)
            for family in FAMILIES
            for i in range(len(snapshots[family]))
            for seed in self.settings.seeds
        ]
        if len(blocks) * len(ARMS) > 20000:
            raise ValueError("Study exceeds the twenty-thousand-job work envelope.")
        rng = RandomStreams(self.settings.execution_seed).python
        rng.shuffle(blocks)
        jobs = []
        for family, index, seed in blocks:
            arms = list(ARMS)
            rng.shuffle(arms)
            jobs.extend([[family, index, seed, arm] for arm in arms])
        return jobs

    def freeze(self, path: Path) -> None:
        """Freeze the immutable schema and initialize all durable pending reservations.

        Args:
            path (Path): New private controller directory.

        Raises:
            ValueError: If the work envelope or serialized declaration is oversized.
            FileExistsError: If the output directory already exists.
            OSError: If source or database persistence fails.
        """
        inputs = self._snapshots()
        jobs = self._jobs(inputs.instances)
        spec = {
            "schema": 1,
            "transport": self.settings.transport,
            "claim": self.settings.claim,
            "claim_scope": "frozen_artifact_case_study_only",
            "instances": inputs.instances,
            "strata": inputs.strata,
            "artifacts": inputs.sources,
            "image": self.settings.image,
            "prior_instance_hashes": sorted(set(self.settings.prior_instance_hashes)),
            "seeds": self.settings.seeds,
            "seconds": self.settings.seconds,
            "cpu_budget": self.settings.cpu_budget,
            "elapsed_budget": self.settings.elapsed_budget,
            "minimum_effect": self.settings.minimum_effect,
            "effect_units": "absolute_mean_loss_reduction",
            "execution_seed": self.settings.execution_seed,
            "jobs": jobs,
            "power_plan_declared": self.settings.power_plan,
            "analysis": {
                "bootstrap_samples": self.analysis.bootstrap_samples,
                "permutation_samples": self.analysis.permutation_samples,
                "random_state": 42,
                "family_size": 6,
            },
            "primary_baselines": ["seed", "constructive"],
            "alpha": 0.05,
            "implementation": self.controller_type._implementation(),
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
            for role, artifact in inputs.artifacts[family].items():
                artifact.export(path / "artifacts" / family / role)
        (path / "study.json").write_text(encoded, encoding="utf-8")
        (path / "study.sha256").write_text(self.controller_type._hash(spec))
        with contextlib.closing(sqlite3.connect(path / "measurements.sqlite")) as database:
            database.execute("CREATE TABLE state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            database.execute(
                "CREATE TABLE jobs (id INTEGER PRIMARY KEY, status TEXT NOT NULL, "
                "charge REAL NOT NULL, result TEXT)"
            )
            database.executemany(
                "INSERT INTO jobs VALUES (?, 'pending', 0, NULL)", [(i,) for i in range(len(jobs))]
            )
            database.execute(
                "INSERT INTO state VALUES ('specification', ?)", (self.controller_type._hash(spec),)
            )
            database.commit()
