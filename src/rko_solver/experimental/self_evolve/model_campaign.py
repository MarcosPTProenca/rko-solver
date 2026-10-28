"""Generic mathematical campaign policy behind the unchanged shared proposal lifecycle.

Objective:
- Freeze explicit original references and feasibility-first versus strict admission.
Technical approach:
- Keep model-only schema, safe development input and policy gates in one boundary.
Flow (pseudocode):
1. Validate owned models and explicit trusted original-reference declarations.
2. Freeze the policy before any source generation or measured candidate work.
3. Admit only protected numerical records and never invent an independent human baseline.
Important criteria:
- Legacy pilot families retain their original strict feasibility policy.
- A trusted optimum declaration is provenance, not a solver-generated certificate.
Inputs/Outputs:
- Input: original models, frozen campaign specification and completed evidence.
- Output: safe proposer mathematics and explicit shortlist admission decisions.
"""

import math
import typing
from rko_solver.experimental.self_evolve.model_data import MODEL_FINALIZATION_ALLOWANCE
from rko_solver.experimental.self_evolve.model_data import ModelProblem
from rko_solver.experimental.self_evolve.model_data import ModelReference
from rko_solver.experimental.self_evolve.problems import Problem


class ModelCampaignPolicy:
    """Own model-only campaign differences without replacing shared accounting or lineage.

    Responsibilities:
    - Keep original-objective references parent-only and reject unknown comparisons.
    - Separate strict validity from permitted exploration of row-infeasible assignments.

    Attributes:
        MODES (tuple[str, ...]): Actual declared model controls, with no fabricated human arm.
    """

    MODES = ("check", "rko", "random", "fixed")

    @staticmethod
    def instance(problem: Problem | ModelProblem) -> dict[str, typing.Any]:
        """Freeze one validated mathematical input and explicit reference declaration.

        Args:
            problem (Problem | ModelProblem): Original development input.

        Returns:
            dict: Original evaluator payload and parent-only reference metadata.

        Raises:
            ValueError: If the original model lacks a valid reference or has drifted.
        """
        entry = {"family": problem.family, "data": problem.data, "optimum": problem.optimum()}
        if isinstance(problem, ModelProblem):
            ModelProblem(problem.data, problem.reference)
            if problem.reference is None:
                raise ValueError("Missing original model reference.")
            entry.update(
                maximize=problem.source.maximize, reference_provenance=problem.reference.provenance
            )
        return entry

    @staticmethod
    def freeze(spec: dict[str, typing.Any], mode: str) -> None:
        """Add explicit model semantics without changing legacy campaign byte contracts.

        Args:
            spec (dict): Campaign specification before hashing and persistence.
            mode (str): Strict-feasible or feasibility-first model admission.

        Raises:
            ValueError: If a model policy is not explicitly supported.
        """
        if spec["family"] != "linear_model":
            return
        if not any(ModelProblem(entry["data"]).dimension for entry in spec["instances"]):
            raise ValueError(
                "Evolution requires original model construction freedom; all keys are fixed."
            )
        if mode not in ("strict_feasible", "feasibility_first"):
            raise ValueError("Unsupported frozen model feasibility policy.")
        spec.update(feasibility_policy=mode, **ModelCampaignPolicy._settings())
        spec["advancement"] = (
            "screen primary loss nonworse; complete strict mean-loss improvement, "
            "nonworse per-case matched-job feasibility and a retained feasible witness"
        )

    @staticmethod
    def _settings() -> dict[str, typing.Any]:
        """Declare one versioned model promotion/reference/verification profile.

        Returns:
            dict: Fresh canonical settings shared by creation and integrity checks.
        """
        return {
            "model_contract": 1,
            "model_timing_contract": 2,
            "seconds": 1.0,
            "timing_policy": "end_to_end",
            "finalization_allowance_seconds": MODEL_FINALIZATION_ALLOWANCE,
            "reference_policy": "trusted_optimum",
            "human_baseline": None,
            "promotion_policy": {
                "primary": "strict_mean_loss",
                "feasibility": "nonworse_per_case_matched_jobs",
                "require_feasible_witness": True,
            },
            "verification_allowance": {"relative": 1e-12, "absolute": 1e-12},
        }

    @staticmethod
    def restore(spec: dict[str, typing.Any]) -> None:
        """Reject protected model/policy drift before generation or runtime creation.

        Args:
            spec (dict): Integrity-checked frozen campaign specification.

        Raises:
            ValueError: If model schema, reference semantics or runtime identity changed.
        """
        if spec["family"] != "linear_model":
            return
        if any(
            type(spec.get(key)) is not type(value) or spec.get(key) != value
            for key, value in ModelCampaignPolicy._settings().items()
        ):
            raise ValueError("Incompatible original model campaign policy/verification contract.")
        if spec.get("feasibility_policy") not in ("strict_feasible", "feasibility_first"):
            raise ValueError("Unknown model campaign feasibility policy.")
        for entry in spec["instances"]:
            problem = ModelProblem(
                entry["data"], ModelReference(entry["optimum"], entry["reference_provenance"])
            )
            if (
                type(entry.get("maximize")) is not bool
                or entry["maximize"] != problem.source.maximize
            ):
                raise ValueError("Original model objective direction changed.")

    @staticmethod
    def local_work() -> dict[str, typing.Any]:
        """Describe a bounded generic source-review allowance separately from protected repair.

        Returns:
            dict: Frozen construction work shape, not a claim of mechanically enforced complexity.
        """
        return {
            "passes": 2,
            "moves_per_pass": "n*n+n",
            "repair_moves": "n",
            "review": "required before submission; no nested search/restarts",
            "neighborhood": "bounded original-coordinate assignments/swaps; n original variables",
            "repair": "candidate at most n moves; protected continuous policy unchanged",
        }

    @staticmethod
    def proposer(spec: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Override model prompt fields without leaking references, source names or case identity.

        Args:
            spec (dict): Frozen shared campaign specification.

        Returns:
            dict: Model-only safe prompt fields, or no changes for a legacy pilot.
        """
        if spec["family"] != "linear_model":
            return {}
        return {
            "semantics": "Keys lie in [0,1], including endpoints. "
            "decode(keys, instance) returns complete original-order numerical values. "
            "The immutable context has variables, finite effective "
            "domains, original sparse rows, objective, objective_constant, maximize and n_keys. "
            "Shape/domains/fixed values are strict; rows/objective/tolerances, scoring and "
            "continuous repair remain protected. All construction/restoration/checking is charged.",
            "feasibility_policy": spec["feasibility_policy"],
            "promotion_policy": spec["promotion_policy"],
            "timing_policy": spec["timing_policy"],
            "job_seconds": spec["seconds"],
            "finalization_allowance_seconds": spec["finalization_allowance_seconds"],
            "development_instances": [
                {"family": "linear_model", "data": ModelProblem(entry["data"]).proposer_data()}
                for entry in spec["instances"]
            ],
        }

    @staticmethod
    def reference_loss(record: dict[str, typing.Any], optimum: int | float, maximize: bool) -> None:
        """Compare only feasible original objectives to explicitly trusted parent references.

        Args:
            record (dict): Independently verified primary model evidence.
            optimum (int | float): Explicit trusted original reference.
            maximize (bool): Original model objective orientation.
        """
        record["loss"] = 1.0
        if record["status"] != "ok" or not record.get("feasible"):
            return
        objective = record["objective"]
        better = objective > optimum if maximize else objective < optimum
        if better and not math.isclose(objective, optimum, rel_tol=1e-12, abs_tol=1e-12):
            record.update(
                status="reference_error", error="Witness contradicts the original model reference."
            )
            return
        regret = abs(objective - optimum) / max(1.0, abs(optimum))
        record["loss"] = 0.5 * regret / (1.0 + regret)

    @staticmethod
    def admissible(spec: dict[str, typing.Any], record: dict[str, typing.Any]) -> bool:
        """Apply strict original validity while allowing explicitly frozen row exploration.

        Args:
            spec (dict): Frozen feasibility policy.
            record (dict): Completed independently verified numerical evidence.

        Returns:
            bool: Whether the cell meets its shape/domain/feasibility policy.
        """
        if record["status"] != "ok":
            return False
        if spec.get("feasibility_policy", "strict_feasible") == "strict_feasible":
            return not record.get("infeasible_evaluations", 1)
        return True

    @staticmethod
    def promote(
        spec: dict[str, typing.Any], events: list[dict[str, typing.Any]], candidate: str
    ) -> None:
        """Require a complete strict original-loss win and nonworse per-case job feasibility.

        Args:
            spec (dict): Frozen development cases/seeds and declared model policy.
            events (list[dict]): Durable completed numerical evidence, including failures.
            candidate (str): Exact reviewed child source digest.

        Raises:
            ValueError: If the panel is incomplete, tied, witness-free or worse per case.
        """
        if spec["family"] != "linear_model":
            return
        baseline = ModelCampaignPolicy._panel(spec, events, spec["seed"], "baseline")
        measured = ModelCampaignPolicy._panel(spec, events, candidate, "candidate")
        if not any(row["feasible"] for row in measured.values()) or math.fsum(
            row["loss"] for row in measured.values()
        ) >= math.fsum(row["loss"] for row in baseline.values()):
            raise ValueError("Model candidate needs strict complete primary-loss improvement.")
        for index in range(len(spec["instances"])):
            if ModelCampaignPolicy._success(measured, index) < ModelCampaignPolicy._success(
                baseline, index
            ):
                raise ValueError("Model candidate worsened matched-job feasibility for a case.")

    @staticmethod
    def _panel(
        spec: dict[str, typing.Any],
        events: list[dict[str, typing.Any]],
        candidate: str,
        partition: str,
    ) -> dict[tuple[int, int], dict[str, typing.Any]]:
        """Select exactly one completed ordinary-RKO development record per frozen job.

        Args:
            spec (dict): Frozen cases and solver seeds.
            events (list[dict]): Durable measured records.
            candidate (str): Original seed or reviewed child digest.
            partition (str): Baseline or candidate allocation.

        Returns:
            dict: Complete records keyed by original instance and solver seed.

        Raises:
            ValueError: If records are missing, duplicated or outside the frozen panel.
        """
        wanted = {
            "kind": "evaluation",
            "mode": "rko",
            "stage": "development",
            "partition": partition,
            "candidate": candidate,
            "status": "ok",
        }
        rows = [
            row for row in events if all(row.get(key) == value for key, value in wanted.items())
        ]
        panel = {(row["instance"], row["seed"]): row for row in rows}
        expected = {
            (index, seed)
            for index in range(len(spec["instances"]))
            for seed in spec["solver_seeds"]
        }
        if len(rows) != len(panel) or set(panel) != expected:
            raise ValueError("Model comparison requires a complete nonduplicated matched panel.")
        return panel

    @staticmethod
    def _success(panel: dict[tuple[int, int], dict[str, typing.Any]], index: int) -> int:
        """Count actual feasible matched jobs, never throughput-dependent feasible callbacks.

        Args:
            panel (dict): Complete original-objective RKO records.
            index (int): Frozen development case coordinate.

        Returns:
            int: Feasible completed job count for the requested case.
        """
        return sum(bool(row["feasible"]) for (case, _), row in panel.items() if case == index)
