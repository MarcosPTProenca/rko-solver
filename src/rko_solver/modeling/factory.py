"""Typed public model-construction policy.

Objective:
- Build generated-model solvers without enlarging the general callback driver.

Technical approach:
- Preserve explicit keyword defaults in typed options and an immutable settings record.

Flow (pseudocode):
1. Reject unsupported generated-model engines before native extraction.
2. Own source semantics and compile numerical callbacks once.
3. Construct the ordinary solver and attach portable model provenance.

Important criteria:
- Generated models use only Python callback engines; no native model enters workers.

Inputs/Outputs:
- Input: trusted native linear model and explicitly typed public settings.
- Output: ordinary solver holding only owned source/numerical data.
"""

from __future__ import annotations

import time
from ..orchestration.adaptation import Adaptation
from ..orchestration.types import Hyperparameters
from ..problem import Callbacks
from ..tracing import TraceConfig
from .adapters import ModelAdapter
from .options import SearchOptions
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal

if TYPE_CHECKING:
    from ..solver import RKO

Execution = Literal["process", "thread", "sequential"]


class ModelOptions(SearchOptions, total=False):
    """Describe public generated-model construction keywords.

    Responsibilities:
    - Preserve discoverable static keyword contracts for model and solver policy.

    Attributes:
        engine (str): Python only, rejected before extraction otherwise.
        execution (str): Process, thread or sequential Python scheduling.
        repair (str): Continuous restoration by default at the public factory.
    """

    algorithms: Sequence[str] | None
    adaptation: Adaptation | None
    control: int | None
    pool_size: int | None
    prefill_pool: bool | None
    params_dir: Path | None
    engine: Literal["python"]
    execution: Execution
    max_evaluations: int | None
    n_jobs: int | None
    random_state: int | None
    verbose: bool
    hyperparameters: Hyperparameters | None
    history_limit: int
    trace: TraceConfig | None


@dataclass(frozen=True, slots=True)
class ModelFactorySettings:
    """Own established public defaults without an expanding positional interface.

    Responsibilities:
    - Preserve all existing factory keyword defaults exactly.

    Attributes:
        engine (str): Python-only generated-model engine.
        execution (str): Existing Python execution mode.
        repair (str): Bounded continuous restoration or explicit box-only mapping.
    """

    search_bounds: Mapping[str, tuple[float | None, float | None]] | None = None
    scoring: Literal["feasibility_first", "weighted_penalty"] = "feasibility_first"
    penalty_weight: float | None = None
    absolute_tolerance: float = 1e-06
    relative_tolerance: float = 0.0
    repair: Literal["none", "continuous"] = "continuous"
    repair_max_sweeps: int = 1024
    algorithms: Sequence[str] | None = None
    adaptation: Adaptation | None = None
    control: int | None = None
    pool_size: int | None = None
    prefill_pool: bool | None = None
    params_dir: Path | None = None
    engine: Literal["python"] = "python"
    execution: Execution = "process"
    max_evaluations: int | None = None
    n_jobs: int | None = None
    random_state: int | None = None
    verbose: bool = False
    hyperparameters: Hyperparameters | None = None
    history_limit: int = 0
    trace: TraceConfig | None = None


class ModelFactory:
    """Construct one ordinary solver from owned generated-model callbacks.

    Responsibilities:
    - Validate model-specific backend policy before extraction.
    - Keep native frontend references out of the returned solver.
    """

    @staticmethod
    def create(solver_type: type[RKO], model: object, options: ModelOptions) -> RKO:
        """Compile portable model callbacks and attach their owned provenance.

        Args:
            solver_type (type[RKO]): Public solver class, including compatible subclasses.
            model (object): Trusted qualified native linear source model.
            options (ModelOptions): Explicit public compilation and portfolio settings.

        Returns:
            RKO: Ordinary solver owning only numerical/source data.

        Raises:
            ValueError: If model semantics, settings or backend are unsupported.
            TypeError: If an unknown construction keyword is supplied.
            ImportError: If the selected optional frontend cannot load.
        """
        settings = ModelFactorySettings(**options)
        if settings.engine != "python" or settings.execution not in (
            "process",
            "thread",
            "sequential",
        ):
            raise ValueError("Model decoders require engine='python' and Python execution.")
        started = time.perf_counter()
        source = ModelAdapter.extract(model)
        snapshot = source.compile(
            search_bounds=settings.search_bounds,
            scoring=settings.scoring,
            penalty_weight=settings.penalty_weight,
            absolute_tolerance=settings.absolute_tolerance,
            relative_tolerance=settings.relative_tolerance,
            repair=settings.repair,
            repair_max_sweeps=settings.repair_max_sweeps,
        )
        compilation_seconds = time.perf_counter() - started
        solver = solver_type(
            Callbacks(snapshot.decode, snapshot.cost, snapshot.n_keys),
            algorithms=settings.algorithms,
            adaptation=settings.adaptation,
            control=settings.control,
            pool_size=settings.pool_size,
            prefill_pool=settings.prefill_pool,
            params_dir=settings.params_dir,
            engine=settings.engine,
            execution=settings.execution,
            max_evaluations=settings.max_evaluations,
            n_jobs=settings.n_jobs,
            random_state=settings.random_state,
            verbose=settings.verbose,
            hyperparameters=settings.hyperparameters,
            history_limit=settings.history_limit,
            trace=settings.trace,
        )
        solver._model_source = source
        solver._model_snapshot = snapshot
        solver._model_compilation_seconds = compilation_seconds
        return solver
