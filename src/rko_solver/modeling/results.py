"""Portable model solutions separate from scalar search-score incumbents.

Objective:
- Expose completed feasible model witnesses without changing legacy search-result meanings.

Technical approach:
- Bind immutable stable-ID assignments to their original completed evaluation.
- Retain owned source/snapshot provenance for explicit compatible-model reconstruction.
- Export assignment-free summaries separately from explicit portable assignment exports.

Flow (pseudocode):
1. Merge bounded completed startup/search/worker witnesses.
2. Build independent search-incumbent diagnostics and best-feasible model solution.
3. Export portable assignments or explicitly validate native write-back on request.

Important criteria:
- No native model references, callback replay or feasibility/optimality certificates.
- A missing feasible witness has no primary model solution or objective.
- Search-score traces belong only to the root search incumbent, not another witness.

Inputs/Outputs:
- Input: owned source/snapshot and completed evaluation evidence.
- Output: immutable model result, portable assignments and explicit native mappings.
"""

from __future__ import annotations

import hashlib
from .adapters import ExtractedLinearModel
from .adapters import ModelAdapter
from .linear import CompiledLinearModel
from .linear import ModelEvaluation
from .linear import ModelEvidence
from .linear import RepairDiagnostic
from .linear import RowDiagnostic
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Literal
from typing import TypedDict
from typing import Unpack


class ResultTiming(TypedDict, total=False):
    """Describe optional direct-evaluation and monotonic timing metadata.

    Responsibilities:
    - Preserve explicit result-construction keywords without positional expansion.

    Attributes:
        no_search (bool): Whether the model required only one direct evaluation.
        solve_started_at (float | None): Parent monotonic solve origin.
    """

    no_search: bool
    solve_started_at: float | None


@dataclass(frozen=True, slots=True)
class ModelAssignment:
    """Associate a portable scalar assignment with a stable frontend-qualified identity.

    Responsibilities:
    - Keep duplicate display labels unambiguous without native object references.

    Attributes:
        identifier (str): Stable original frontend variable ID.
        label (str): Original display name, not necessarily unique.
        value (float): Completed assigned value, including fixed variables.
    """

    identifier: str
    label: str
    value: float


@dataclass(frozen=True, slots=True)
class ModelSolution:
    """Preserve one original model evaluation and its explicit immutable assignments.

    Responsibilities:
    - Keep keys, raw objective, search score and row evidence from one invocation together.

    Attributes:
        evaluation (ModelEvaluation): Original completed numerical witness.
        direction (str): Original objective direction, minimize or maximize.
        assignments (tuple[ModelAssignment, ...]): Stable-ID original variable values.
    """

    evaluation: ModelEvaluation
    direction: Literal["minimize", "maximize"]
    assignments: tuple[ModelAssignment, ...]

    @property
    def keys(self) -> tuple[float, ...]:
        """Return this witness's keys, not necessarily the root search incumbent keys.

        Returns:
            tuple[float, ...]: Original completed invocation keys.
        """
        return self.evaluation.keys

    @property
    def objective(self) -> float:
        """Return the original objective sign and constant without search normalization.

        Returns:
            float: Original objective value, not an optimality claim.
        """
        return self.evaluation.objective

    @property
    def search_score(self) -> float:
        """Return the scalar minimization score used by existing RKO algorithms.

        Returns:
            float: This witness's actual completed search score.
        """
        return self.evaluation.score

    @property
    def feasible(self) -> bool:
        """Return feasibility under the snapshot's declared numerical rules.

        Returns:
            bool: Checked model feasibility, not an external solver certificate.
        """
        return self.evaluation.feasible

    @property
    def rows(self) -> tuple[RowDiagnostic, ...]:
        """Return original-unit row residuals and tolerance-adjusted violations.

        Returns:
            tuple[RowDiagnostic, ...]: Completed row evidence for this assignment.
        """
        return self.evaluation.rows

    @property
    def repair(self) -> RepairDiagnostic:
        """Return deterministic restoration work for this exact completed invocation.

        Returns:
            RepairDiagnostic: Policy, termination outcome and completed sweep count.
        """
        return self.evaluation.repair

    def to_dict(self, *, include_assignments: bool = True) -> dict[str, Any]:
        """Export original numerical diagnostics and optionally explicit assignment coordinates.

        Args:
            include_assignments (bool): Include stable-ID assignments and keys when explicit.

        Returns:
            dict[str, Any]: Portable evidence; false omits assignment/keys and row activities.
        """
        data: dict[str, Any] = {
            "objective": self.objective,
            "direction": self.direction,
            "search_score": self.search_score,
            "feasible": self.feasible,
            "repair": asdict(self.repair),
            "violated_rows": sum(
                row.lower_violation > 0 or row.upper_violation > 0 for row in self.rows
            ),
            "summed_adjusted_violation": sum(
                row.lower_violation + row.upper_violation for row in self.rows
            ),
        }
        if include_assignments:
            data.update(
                keys=list(self.keys),
                assignments=[asdict(a) for a in self.assignments],
                rows=[asdict(row) for row in self.rows],
            )
        return data


@dataclass(frozen=True, slots=True)
class ModelResult:
    """Separate validated feasible solutions from the actual score-selected search incumbent.

    Responsibilities:
    - Preserve original source and effective numerical provenance without native references.
    - Expose honest no-feasible-found status and explicit compatible-model assignment actions.

    Attributes:
        source (ExtractedLinearModel): Owned original frontend/version/structure evidence.
        snapshot (CompiledLinearModel): Effective bounds, tolerances and scoring data.
        search_incumbent (ModelSolution): Actual root keys/cost invocation diagnostics.
        best_feasible (ModelSolution | None): Independent feasible original-objective witness.
        compilation_seconds (float): Conversion/compilation duration outside measured solve work.
        no_search (bool): True only for direct evaluation of a zero-key snapshot.
        direct_evaluations (int): One actual completed direct evaluation, otherwise zero.
        time_to_first_feasible_seconds (float | None): First feasible completion since solve
            began, including startup/execution work but excluding snapshot compilation.
    """

    source: ExtractedLinearModel
    snapshot: CompiledLinearModel
    search_incumbent: ModelSolution
    best_feasible: ModelSolution | None
    compilation_seconds: float
    no_search: bool = False
    direct_evaluations: int = 0
    time_to_first_feasible_seconds: float | None = None

    @classmethod
    def from_evidence(
        cls,
        source: ExtractedLinearModel,
        snapshot: CompiledLinearModel,
        evidence: ModelEvidence,
        compilation_seconds: float,
        **timing: Unpack[ResultTiming],
    ) -> ModelResult:
        """Build portable solution records only from retained completed witnesses.

        Args:
            source (ExtractedLinearModel): Owned source provenance.
            snapshot (CompiledLinearModel): Effective numerical compilation settings.
            evidence (ModelEvidence): Identity-matched search and independent feasible witnesses.
            compilation_seconds (float): Separately measured initial conversion duration.
            timing (ResultTiming): Optional direct-evaluation flag and parent monotonic
                solve origin; omission keeps timing unknown for manual evidence.

        Returns:
            ModelResult: Immutable model solution/diagnostic section with no callback replay.
        """
        no_search = timing.get("no_search", False)
        solve_started_at = timing.get("solve_started_at")
        solutions = []
        for evaluation in (evidence.incumbent, evidence.best_feasible):
            solutions.append(
                None
                if evaluation is None
                else ModelSolution(
                    evaluation,
                    "maximize" if source.maximize else "minimize",
                    tuple(
                        ModelAssignment(variable.identifier, variable.label, value)
                        for variable, value in zip(source.variables, evaluation.values, strict=True)
                    ),
                )
            )
        assert solutions[0] is not None
        return cls(
            source,
            snapshot,
            solutions[0],
            solutions[1],
            compilation_seconds,
            no_search,
            int(no_search),
            max(0.0, evidence.first_feasible_at - solve_started_at)
            if evidence.first_feasible_at is not None and solve_started_at is not None
            else None,
        )

    @property
    def status(self) -> Literal["feasible", "no_feasible_found"]:
        """Distinguish a validated witness from failure to find one, never certify infeasibility.

        Returns:
            str: Feasible or no_feasible_found under the declared numerical rules.
        """
        return "feasible" if self.best_feasible is not None else "no_feasible_found"

    def native_mapping(self, model: object) -> dict[str, Any]:
        """Reconstruct native variable identities against an explicit unchanged compatible model.

        Args:
            model (object): Native source/compatible reconstructed model to validate.

        Returns:
            dict[str, Any]: Stable IDs mapped to local native variable handles.

        Raises:
            ValueError: If source structure, values of fixed variables or identities changed.
            ImportError: If the optional native runtime cannot be loaded.
        """
        return ModelAdapter.variable_mapping(model, self.source)

    def write_back(self, model: object) -> None:
        """Explicitly apply only the retained feasible witness with transactional validation.

        Args:
            model (object): Compatible native Pyomo/PuLP model receiving stored values.

        Raises:
            ValueError: If no feasible witness, source drift or unrestorable prior values exist.
            NotImplementedError: If MathOpt native write-back is requested.
            RuntimeError: If assignment fails, including explicit rollback failure diagnostics.
            ImportError: If the optional native runtime cannot be loaded.
        """
        if self.best_feasible is None:
            raise ValueError("No feasible model witness is available for native write-back.")
        ModelAdapter.write_back(model, self.source, self.snapshot, self.best_feasible.evaluation)

    def to_dict(self, *, include_assignments: bool = True) -> dict[str, Any]:
        """Export model semantics independently of root search-score histories and traces.

        Args:
            include_assignments (bool): Explicitly include assignment/keys/row activities;
                reports pass false to preserve their coordinate-exclusion contract.

        Returns:
            dict[str, Any]: Portable model status, witnesses and effective numerical provenance.
        """
        return {
            "status": self.status,
            "frontend": self.source.frontend,
            "frontend_version": self.source.version,
            "source_name": self.source.name,
            "source_fingerprint": self.source.fingerprint,
            "snapshot_fingerprint": hashlib.sha256(repr(self.snapshot).encode("utf-8")).hexdigest(),
            "scoring": self.snapshot.scoring,
            "penalty_weight": self.snapshot.penalty_weight,
            "absolute_tolerance": self.snapshot.absolute_tolerance,
            "relative_tolerance": self.snapshot.relative_tolerance,
            "repair": self.snapshot.repair,
            "repair_max_sweeps": self.snapshot.repair_max_sweeps,
            "effective_domains": [
                asdict(domain)
                if include_assignments
                else {
                    "identifier": domain.identifier,
                    "fixed": domain.fixed,
                    "restricted": domain.restricted,
                }
                for domain in self.snapshot.domains
            ],
            "compilation_seconds": self.compilation_seconds,
            "no_search": self.no_search,
            "direct_evaluations": self.direct_evaluations,
            "time_to_first_feasible_seconds": self.time_to_first_feasible_seconds,
            "best_feasible": self.best_feasible.to_dict(include_assignments=include_assignments)
            if self.best_feasible is not None
            else None,
            "search_incumbent": self.search_incumbent.to_dict(
                include_assignments=include_assignments
            ),
            "certificate": None,
        }
