"""Generated-model integration at the ordinary solve lifecycle boundary.

Objective:
- Keep model-specific validation and bounded result evidence outside generic search code.

Technical approach:
- Capture immutable source/snapshot state and a bounded per-solve evidence collector.

Flow (pseudocode):
1. Reject backend or callback identity mutation before execution.
2. Directly evaluate fixed models or collect completed ordinary search witnesses.
3. Build portable model results without replaying decoders or objectives.

Important criteria:
- Preserve actual search-score identity independently of the best feasible original objective.

Inputs/Outputs:
- Input: configured solver and completed per-search evidence.
- Output: optional model result attached to the ordinary result.
"""

from __future__ import annotations

import numpy as np
import time
from .construction import ModelConstruction
from .linear import CompiledLinearModel
from .records import ModelEvidence
from .results import ModelResult
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..orchestration.types import ModelEvidenceOptions
    from ..solver import RKO


class ModelRunState:
    """Own model-specific integration state for one fresh solve.

    Responsibilities:
    - Validate immutable generated callback identity before generic execution.
    - Keep fixed-model and completed-search evidence consistent without replay.

    Attributes:
        solver (RKO): Ordinary configured solver, never a native source model.
        snapshot (CompiledLinearModel | None): Owned immutable numerical snapshot.
        evidence (list[ModelEvidence] | None): Bounded completed per-search witnesses.
    """

    def __init__(self, solver: RKO) -> None:
        """Capture numerical model context and initialize an optional evidence collector.

        Args:
            solver (RKO): Ordinary public solver under execution.
        """
        self.solver = solver
        self.snapshot = solver._model_snapshot
        self.evidence: list[ModelEvidence] | None = [] if self.snapshot is not None else None

    @property
    def no_search(self) -> bool:
        """Identify generated models with no nonfixed random-key coordinates.

        Returns:
            bool: Whether one direct completed evaluation replaces portfolio execution.
        """
        return self.snapshot is not None and self.snapshot.n_keys == 0

    def validate(self) -> None:
        """Reject engine, execution or generated-callback identity mutation.

        Raises:
            ValueError: If the configured backend or callback/dimension changed.
        """
        snapshot, solver = self.snapshot, self.solver
        if snapshot is None:
            return
        if solver.engine != "python" or solver.execution not in ("process", "thread", "sequential"):
            raise ValueError("Model decoders require engine='python' and Python execution.")
        expected = (
            ModelConstruction.decode
            if isinstance(snapshot, ModelConstruction)
            else CompiledLinearModel.decode
        )
        if (
            solver.n_variables != snapshot.n_keys
            or getattr(solver.decoder, "__self__", None) is not snapshot
            or getattr(solver.decoder, "__func__", None) is not expected
            or solver.cost is not snapshot.cost
        ):
            raise ValueError("Model callbacks/dimension changed; reconvert with RKO.from_model.")

    def arguments(self) -> ModelEvidenceOptions:
        """Forward evidence collection only for generated-model execution.

        Returns:
            ModelEvidenceOptions: Empty for ordinary callbacks, otherwise the owned collector.
        """
        return {"model_evidence": self.evidence} if self.evidence is not None else {}

    def direct(self) -> tuple[list[float], float]:
        """Complete the single zero-key evaluation with original objective evidence.

        Returns:
            tuple[list[float], float]: Empty incumbent keys and its exact scalar score.
        """
        assert self.snapshot is not None and self.evidence is not None
        evaluation = self.snapshot.decode(np.empty(0))
        self.evidence.append(
            ModelEvidence(
                evaluation,
                evaluation if evaluation.feasible else None,
                time.perf_counter() if evaluation.feasible else None,
            )
        )
        return [], self.snapshot.cost(evaluation)

    def finish(self, keys: list[float], value: float, started: float) -> ModelResult | None:
        """Build portable model diagnostics from retained completed witnesses only.

        Args:
            keys (list[float]): Actual returned search-score incumbent keys.
            value (float): Actual returned search score.
            started (float): Parent monotonic solve origin.

        Returns:
            ModelResult | None: Completed generated-model section, or None for ordinary callbacks.

        Raises:
            ValueError: If the actual search incumbent lacks completed model evidence.
        """
        if self.snapshot is None:
            return None
        assert self.solver._model_source is not None and self.evidence is not None
        return ModelResult.from_evidence(
            self.solver._model_source,
            self.snapshot.snapshot
            if isinstance(self.snapshot, ModelConstruction)
            else self.snapshot,
            self.snapshot.merge_evidence(tuple(self.evidence), keys, value),
            self.solver._model_compilation_seconds,
            no_search=self.no_search,
            solve_started_at=started,
            construction_used=isinstance(self.snapshot, ModelConstruction),
        )
