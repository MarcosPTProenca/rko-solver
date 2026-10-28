"""Explicit trusted construction around immutable original-model evaluation.

Objective:
- Replace only assignment construction without replacing mathematical semantics.
Technical approach:
- Pass frozen numerical records and a detached immutable-backed key vector to a callable.
- Validate its values and apply the existing protected restoration/evaluation policy.
Flow (pseudocode):
1. Bind an owned source and compiled evaluator to explicit trusted application code.
2. Construct original-coordinate values from detached keys and immutable data.
3. Retain one protected completed evaluation with the original keys.
Important criteria:
- This trusted callback interface never loads source artifacts or provides OS isolation.
- Default compiled mapping and native source extraction remain separate and unchanged.
Inputs/Outputs:
- Input: immutable source/snapshot and an explicit callable.
- Output: completed original-model evidence, never candidate-claimed objective values.
"""

from __future__ import annotations

import math
import numpy as np
from collections.abc import Callable
from dataclasses import dataclass
from rko_solver.modeling.linear import CompiledLinearModel
from rko_solver.modeling.records import EffectiveDomain
from rko_solver.modeling.records import LinearRow
from rko_solver.modeling.records import LinearVariable
from rko_solver.modeling.records import ModelEvaluation
from rko_solver.modeling.records import ModelEvidence
from rko_solver.modeling.source import ExtractedLinearModel


@dataclass(frozen=True, slots=True)
class ConstructionInput:
    """Expose only immutable original mathematics and effective domain coordinates.

    Responsibilities:
    - Exclude evaluator functions, references, native objects and mutable buffers.

    Attributes:
        variables (tuple): Original ordered immutable variable records.
        domains (tuple): Effective finite domains in the same original order.
        rows (tuple): Original sparse affine row semantics.
        objective (tuple[float, ...]): Original ordered objective coefficients.
        objective_constant (float): Original affine objective constant.
        maximize (bool): Original direction, never a search-score reinterpretation.
    """

    variables: tuple[LinearVariable, ...]
    domains: tuple[EffectiveDomain, ...]
    rows: tuple[LinearRow, ...]
    objective: tuple[float, ...]
    objective_constant: float
    maximize: bool

    @property
    def n_keys(self) -> int:
        """Derive chromosome dimension solely from nonfixed effective domains.

        Returns:
            int: Number of free coordinates, including zero for fixed models.
        """
        return sum(not domain.fixed for domain in self.domains)


ConstructionFn = Callable[[np.ndarray, ConstructionInput], object]


@dataclass(frozen=True, slots=True)
class ModelConstruction:
    """Own explicit trusted construction without granting it evaluator authority.

    Responsibilities:
    - Preserve actual key/value identity and bounded owner cancellation.
    - Delegate immutable mathematical checking and evidence merging to one snapshot.

    Attributes:
        snapshot (CompiledLinearModel): Protected original numerical evaluator.
        context (ConstructionInput): Frozen data visible to the constructor.
        construction (ConstructionFn): Explicit trusted callable, not a source descriptor.
        deadline (float | None): Optional absolute monotonic ceiling for model search sessions.
    """

    snapshot: CompiledLinearModel
    context: ConstructionInput
    construction: ConstructionFn
    deadline: float | None = None

    @staticmethod
    def validate_deadline(deadline: float | None) -> None:
        """Validate an explicit owner ceiling without changing ordinary search budgets.

        Args:
            deadline (float | None): Absolute monotonic owner deadline, or no ceiling.

        Raises:
            ValueError: If the supplied deadline is nonnumeric, boolean or nonfinite.
        """
        if deadline is not None and (
            type(deadline) not in (int, float)
            or abs(deadline) > 1e100
            or not math.isfinite(deadline)
        ):
            raise ValueError("construction_deadline must be a finite monotonic time or None.")

    @classmethod
    def bind(
        cls,
        snapshot: CompiledLinearModel,
        source: ExtractedLinearModel,
        construction: ConstructionFn,
        *,
        deadline: float | None = None,
    ) -> ModelConstruction:
        """Bind already compiled original semantics without importing executable source.

        Args:
            snapshot (CompiledLinearModel): Validated numerical model.
            source (ExtractedLinearModel): Matching original source records.
            construction (ConstructionFn): Trusted application callable.
            deadline (float | None): Optional absolute owner ceiling for search sessions.

        Returns:
            ModelConstruction: Immutable construction/evaluation adapter.

        Raises:
            ValueError: If the explicit construction value is not callable.
        """
        cls.validate_deadline(deadline)
        if not callable(construction):
            raise ValueError("construction must be a trusted callable, not an artifact or path.")
        context = ConstructionInput(
            snapshot.variables,
            snapshot.domains,
            snapshot.rows,
            tuple(float(value) for value in source.objective),
            float(source.objective_constant),
            source.maximize,
        )
        return cls(snapshot, context, construction, deadline)

    @property
    def n_keys(self) -> int:
        """Return the unchanged numerical chromosome dimension.

        Returns:
            int: Number of nonfixed model coordinates.
        """
        return self.snapshot.n_keys

    @property
    def maximize(self) -> bool:
        """Return original-objective orientation for independent feasible retention.

        Returns:
            bool: Whether larger original objectives are better.
        """
        return self.snapshot.maximize

    def decode(
        self, keys: np.ndarray, *, _budget_check: Callable[[], None] | None = None
    ) -> ModelEvaluation:
        """Construct once and complete protected evaluation without retaining caller buffers.

        Args:
            keys (np.ndarray): Original finite unit-interval chromosome.
            _budget_check (Callable | None): Ephemeral owner hook, not constructor input.

        Returns:
            ModelEvaluation: Actual completed values and their original key identity.

        Raises:
            ValueError: If keys or constructor values violate the model contract.
            Exception: Trusted constructor and owner cancellation failures propagate.
        """
        original = self.snapshot.validate_keys(keys).copy()
        if self.n_keys == 0:
            return self.snapshot.decode(original, _budget_check=_budget_check)
        if _budget_check is not None:
            _budget_check()
        detached = np.frombuffer(original.tobytes(), dtype=float)
        values = self.construction(detached, self.context)
        if _budget_check is not None:
            _budget_check()
        return self.snapshot.evaluate_assignment(
            original, values, restore=True, _budget_check=_budget_check
        )

    @staticmethod
    def cost(evaluation: ModelEvaluation) -> float:
        """Return the protected already-computed minimization score.

        Args:
            evaluation (ModelEvaluation): Completed numerical evidence.

        Returns:
            float: Actual search score without a second objective computation.
        """
        return evaluation.score

    def merge_evidence(
        self, evidence: tuple[ModelEvidence, ...], keys: list[float], score: float
    ) -> ModelEvidence:
        """Identify completed score and original-objective witnesses without replay.

        Args:
            evidence (tuple[ModelEvidence, ...]): Bounded completed search archives.
            keys (list[float]): Actual returned incumbent chromosome.
            score (float): Actual returned minimization score.

        Returns:
            ModelEvidence: Identity-matched incumbent and independent feasible witness.

        Raises:
            ValueError: If no completed invocation matches the returned incumbent.
        """
        return self.snapshot.merge_evidence(evidence, keys, score)
