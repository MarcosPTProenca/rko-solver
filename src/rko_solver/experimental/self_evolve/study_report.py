"""Complete-panel artifact inference.

Objective:
- Prevent partial inference and preserve six-way statistical gates.
Technical approach:
- Build instance-nested panels and use the unchanged paired analysis.
- Compare candidate against seed and constructive baselines in fixed family order.
- Apply six-way Holm correction, effect bounds and strict candidate/control gates.
Flow (pseudocode):
1. Return progress without inference until all jobs and runtime state are valid.
2. Index durable cells and build seed-nested instance panels in frozen declaration order.
3. Analyze primary effects and emit evidence-labelled results without production promotion.
Important criteria:
- Preserve specification schema, operation order, reservations and inference gates.
Inputs/Outputs:
- Input: immutable study settings and complete durable observations.
- Output: descriptive losses, corrected primary comparisons and predeclared artifact gates.
"""

from __future__ import annotations

import contextlib
import json
import numpy as np
import sqlite3
from .analysis import PairedAnalysis
from .artifacts import FAMILIES
from .study_state import ARMS
from .study_state import StudyState
from typing import Any


class StudyReport(StudyState):
    """Interpret only complete valid frozen panels.

    Responsibilities:
    - Interpret only complete valid frozen panels.

    Attributes:
        spec (dict[str, Any]): Frozen study declaration.
    """

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
        cells = self._cells(rows)
        summaries, comparisons = self._analyze(cells)
        self._correct(comparisons)
        strict = self._strict_gate(rows)
        constructive = self._constructive_gate(rows)
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

    def _cells(
        self, rows: dict[int, dict[str, Any]]
    ) -> dict[tuple[str, int, int, str], dict[str, Any]]:
        """Index complete measurements by their frozen family, instance, seed and arm.

        Args:
            rows (dict[int, dict[str, Any]]): Durable results indexed by frozen job ID.

        Returns:
            dict: Paired cells retaining failures as loss-bearing observations.

        Raises:
            ValueError: If any result has an invalid measurement or reference state.
        """
        cells = {}
        for job_id, job in enumerate(self.spec["jobs"]):
            family, index, seed, arm = job
            result = rows[job_id]
            if result["status"] not in {"ok", "failed"}:
                raise ValueError(
                    "Invalid measurement/reference state; no statistical report is permitted."
                )
            cells[(family, index, seed, arm)] = result
        return cells

    def _analyze(
        self, cells: dict[tuple[str, int, int, str], dict[str, Any]]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Build seed-nested instance panels and analyze primary baselines in fixed order.

        Args:
            cells (dict): Validated observations keyed by family, instance, seed and arm.

        Returns:
            tuple: Descriptive arm means and unadjusted primary paired comparisons.

        Raises:
            ValueError: If the frozen paired analysis design is invalid.
        """
        panels: dict[str, Any] = {family: {arm: [] for arm in ARMS} for family in FAMILIES}
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
        return summaries, comparisons

    def _correct(self, comparisons: list[dict[str, Any]]) -> None:
        """Apply six-way Holm adjustment and the strict predeclared effect threshold.

        Args:
            comparisons (list[dict[str, Any]]): Primary comparisons updated in place.

        Raises:
            ValueError: If multiplicity adjustment receives invalid p-values.
        """
        adjusted = PairedAnalysis.holm([effect["p_value"] for effect in comparisons])
        for effect, corrected in zip(comparisons, adjusted, strict=True):
            effect["holm_p_value"] = corrected
            effect["passes_numerical_checks"] = (
                corrected <= self.spec["alpha"]
                and effect["familywise_lower_bound_95"] > self.spec["minimum_effect"]
            )

    def _strict_gate(self, rows: dict[int, dict[str, Any]]) -> bool:
        """Require the strict gate for every candidate-derived arm, including controls.

        Args:
            rows (dict[int, dict[str, Any]]): Complete durable measurement panel.

        Returns:
            bool: Whether every candidate, random and fixed cell passes the strict gate.
        """
        return all(
            result.get("strict_candidate_gate", False)
            for job, result in rows.items()
            if self.spec["jobs"][job][3] in {"candidate", "random", "fixed"}
        )

    def _constructive_gate(self, rows: dict[int, dict[str, Any]]) -> bool:
        """Require successful feasible constructive-control evidence in every cell.

        Args:
            rows (dict[int, dict[str, Any]]): Complete durable measurement panel.

        Returns:
            bool: Whether all constructive cells succeeded with explicit feasibility.
        """
        return all(
            result.get("status") == "ok" and result.get("feasible") is True
            for job, result in rows.items()
            if self.spec["jobs"][job][3] == "constructive"
        )
