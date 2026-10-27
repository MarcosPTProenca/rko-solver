"""Owned generic linear-model callback facade.

Objective:
- Preserve the public numerical compiler and completed-evidence contracts.

Technical approach:
- Delegate validated sparse storage, restoration and original-unit evaluation.
- Merge only bounded completed invocations, never replay objectives.

Flow (pseudocode):
1. Compile owned finite model data.
2. Decode through deterministic domain mapping and optional restoration.
3. Merge search-score and independent feasible witnesses by invocation identity.

Important criteria:
- Keep original key, objective and score evidence together.

Inputs/Outputs:
- Input: original linear records and explicit numerical settings.
- Output: immutable model callbacks and bounded evidence.
"""

from __future__ import annotations

from .compiler import LinearCompiler
from .evaluation import LinearEvaluation
from .options import LinearOptions
from .records import EffectiveDomain
from .records import LinearRow
from .records import LinearVariable
from .records import ModelEvaluation
from .records import ModelEvidence
from .records import ModelStatistics
from .records import RepairDiagnostic
from .records import RowDiagnostic
from dataclasses import dataclass
from typing import Unpack

__all__ = [
    "CompiledLinearModel",
    "EffectiveDomain",
    "LinearRow",
    "LinearVariable",
    "ModelEvaluation",
    "ModelEvidence",
    "RepairDiagnostic",
    "RowDiagnostic",
]


@dataclass(frozen=True, slots=True, init=False)
class CompiledLinearModel(LinearEvaluation):
    """Expose owned mapping, scoring and bounded evidence merging.

    Responsibilities:
    - Delegate strict compilation, restoration and original-unit evaluation.
    - Keep completed assignment, objective, score and originating keys consistent.

    Attributes:
        n_keys (int): Nonfixed chromosome dimension.
    """

    def __init__(
        self,
        variables: tuple[LinearVariable, ...],
        objective: tuple[float, ...],
        rows: tuple[LinearRow, ...] = (),
        **options: Unpack[LinearOptions],
    ) -> None:
        """Compile owned numerical state with the established keyword contract.

        Args:
            variables (tuple[LinearVariable, ...]): Ordered original variable records.
            objective (tuple[float, ...]): Original objective coefficients.
            rows (tuple[LinearRow, ...]): Original affine constraints.
            options (LinearOptions): Explicit bounds, scorer, tolerance and repair policy.

        Raises:
            ValueError: If model data or numerical settings are invalid or unsafe.
            TypeError: If an unknown keyword is supplied.
        """
        LinearCompiler(self, variables, objective, rows, options).apply()
        self._compile_repair()

    def merge_evidence(
        self, evidence: tuple[ModelEvidence, ...], keys: list[float], score: float
    ) -> ModelEvidence:
        """Merge only completed invocations and identify the actual returned incumbent.

        Args:
            evidence (tuple[ModelEvidence, ...]): Bounded per-search completed witnesses.
            keys (list[float]): Actual returned search chromosome.
            score (float): Actual returned finite search score.

        Returns:
            ModelEvidence: Matching search incumbent and best original feasible objective.

        Raises:
            ValueError: If the returned incumbent has no matching completed evidence.
        """
        identity = tuple(keys)
        incumbent = None
        feasible = None
        times = [a.first_feasible_at for a in evidence if a.first_feasible_at is not None]
        for archive in evidence:
            for candidate in (archive.incumbent, archive.best_feasible):
                if self._matches(candidate, identity, score):
                    incumbent = candidate
            if self._better(archive.best_feasible, feasible):
                feasible = archive.best_feasible
        if incumbent is None:
            raise ValueError("Search incumbent has no matching completed model evidence.")
        statistics = ModelStatistics.merge(tuple(archive.statistics for archive in evidence))
        return ModelEvidence(incumbent, feasible, min(times) if times else None, statistics)

    @staticmethod
    def _matches(candidate: ModelEvaluation | None, keys: tuple[float, ...], score: float) -> bool:
        """Compare completed invocation identity without reconstructing assignments.

        Args:
            candidate (ModelEvaluation | None): Retained invocation.
            keys (tuple[float, ...]): Actual returned keys.
            score (float): Actual returned score.

        Returns:
            bool: Whether both chromosome and scalar score match exactly.
        """
        return candidate is not None and candidate.keys == keys and candidate.score == score

    def _better(self, candidate: ModelEvaluation | None, previous: ModelEvaluation | None) -> bool:
        """Compare feasible original objectives in their native direction.

        Args:
            candidate (ModelEvaluation | None): Completed feasible candidate.
            previous (ModelEvaluation | None): Existing feasible witness.

        Returns:
            bool: Whether a real candidate improves the original objective.
        """
        if candidate is None:
            return False
        if previous is None:
            return True
        return (
            candidate.objective > previous.objective
            if self.maximize
            else candidate.objective < previous.objective
        )

    @staticmethod
    def cost(evaluation: ModelEvaluation) -> float:
        """Expose the generated finite minimization score.

        Args:
            evaluation (ModelEvaluation): Exact completed decoder payload.

        Returns:
            float: Score computed on that invocation's final assignment.
        """
        return evaluation.score
