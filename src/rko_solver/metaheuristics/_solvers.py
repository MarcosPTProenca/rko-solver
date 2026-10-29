"""Single-algorithm solver objects sharing the public RKO lifecycle.

Objective:
- Give each built-in algorithm the same configure-then-solve interface as RKO.

Technical approach:
- Inherit RKO's validated constructor state, solve method and immutable result contract.
- Fix algorithm selection per public class without duplicating search execution.
- Load this module lazily from the package to avoid the solver/registry import cycle.

Flow (pseudocode):
1. Construct an algorithm-specific solver with the same problem callbacks/settings as RKO.
2. Store its one-element algorithm selection without reading files or starting work.
3. Inherited solve creates fresh search state and dispatches the selected engine.
4. Return RKOResult and retain result_ exactly as the portfolio solver does.

Important criteria:
- Defaults, nested hyperparameters, seeds, budgets and errors match RKO.
- A dedicated solver cannot be reconfigured into another algorithm/portfolio.
- Numerical kernels remain in their implementation modules and registry unchanged.

Inputs/Outputs:
- Input: a problem (or decoder, objective and dimension) and shared RKO configuration options.
- Output: stateful solver objects returning RKOResult through solve(time_limit).
"""

from __future__ import annotations

from ..orchestration.adaptation import Adaptation
from ..orchestration.types import CostFn
from ..orchestration.types import DecoderFn
from ..orchestration.types import Hyperparameters
from ..solver import RKO
from ..solver import Engine
from ..solver import Execution
from ..solver import Problem
from ..tracing import TraceConfig
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar


class _SingleAlgorithmSolver(RKO):
    """Specialize RKO configuration while inheriting its complete execution lifecycle.

    Responsibilities:
    - Bind one canonical algorithm to shared problem/execution settings.
    - Preserve RKO solve, reporting, RNG ownership and failure behavior.

    Attributes:
        _algorithm (ClassVar[str]): Canonical fixed algorithm selected by the subclass.
        algorithms (tuple[str, ...]): Read-only one-element selection.
        result_ (RKOResult | None): Latest successful solve, inherited from RKO.
    """

    _algorithm: ClassVar[str]

    def __init__(
        self,
        problem: Problem | DecoderFn | None = None,
        cost: CostFn | None = None,
        n_variables: int | None = None,
        *,
        decoder: DecoderFn | None = None,
        adaptation: Adaptation | None = None,
        control: int | None = None,
        pool_size: int | None = None,
        prefill_pool: bool | None = None,
        initial_keys: Sequence[Sequence[float]] | None = None,
        params_dir: Path | None = None,
        engine: Engine | None = None,
        execution: Execution | None = None,
        auto_time: float | None = None,
        auto_candidates: Sequence[str] | None = None,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
        hyperparameters: Hyperparameters | None = None,
        history_limit: int = 0,
        trace: TraceConfig | None = None,
        max_evaluations: int | None = None,
    ) -> None:
        """Store the same settings as RKO with a fixed single-algorithm selection.

        Args:
            problem (Problem | DecoderFn | None): Problem object or, in the short form, the decoder.
            cost (CostFn | None): Finite minimization objective; short form only.
            n_variables (int | None): Positive random-key dimension; short form only.
            decoder (DecoderFn | None): Keyword spelling of the short form's decoder.
            adaptation (Adaptation | None): Fixed/adaptive mode, or configured default.
            control (int | None): Legacy mode code; must agree with adaptation if supplied.
            pool_size (int | None): Elite capacity, or configured default.
            prefill_pool (bool | None): Separate shared-startup policy, or configured default.
            initial_keys (Sequence[Sequence[float]] | None): Caller vectors offered before search.
            params_dir (Path | None): Optional custom TOML directory.
            engine (Engine | None): Python, rust, cpp or auto, exactly as for RKO.
            execution (Execution | None): Python-engine scheduling, exactly as for RKO.
            auto_time (float | None): Selection budget for auto, as for RKO.
            auto_candidates (Sequence[str] | None): Candidates auto may choose from, as for RKO.
            n_jobs (int | None): Optional process or Rust worker cap; not supported by
                thread and sequential execution.
            random_state (int | None): Nonnegative seed for library-owned random streams.
            verbose (bool): Print lifecycle/shared-pool progress when enabled.
            hyperparameters (Hyperparameters | None): Same nested per-algorithm scalar
                mapping as RKO; fixed adaptation is required for scalar overrides.
            history_limit (int): Per-search scalar history cap; zero disables recording.
            trace (TraceConfig | None): Opt-in bounded original-decoder decisions.
            max_evaluations (int | None): Per-algorithm evaluation limit, as for RKO.
        """
        super().__init__(
            problem,
            cost,
            n_variables,
            decoder=decoder,
            algorithms=(self._algorithm,),
            adaptation=adaptation,
            control=control,
            pool_size=pool_size,
            prefill_pool=prefill_pool,
            initial_keys=initial_keys,
            params_dir=params_dir,
            engine=engine,
            execution=execution,
            auto_time=auto_time,
            auto_candidates=auto_candidates,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            hyperparameters=hyperparameters,
            history_limit=history_limit,
            trace=trace,
            max_evaluations=max_evaluations,
        )

    @property
    def algorithms(self) -> tuple[str, ...]:
        """Return the canonical algorithm fixed by this public solver class.

        Returns:
            tuple[str, ...]: One canonical algorithm name.
        """
        return (self._algorithm,)

    @algorithms.setter
    def algorithms(self, value: Sequence[str] | None) -> None:
        """Allow parent initialization while rejecting a different algorithm selection.

        Args:
            value (Sequence[str] | None): Selection supplied by the parent constructor.

        Raises:
            ValueError: If selection is not exactly this class's canonical algorithm.
        """
        if value is None or isinstance(value, (str, bytes)) or tuple(value) != (self._algorithm,):
            raise ValueError("Algorithm selection is fixed; use RKO to configure a portfolio.")


class SA(_SingleAlgorithmSolver):
    """Configure a simulated-annealing solver using the shared RKO lifecycle.

    Responsibilities:
    - Select SA while retaining RKO's callbacks, execution settings and solve result.

    Attributes:
        _algorithm (ClassVar[str]): Canonical SA selection.
        hyperparameters (Hyperparameters | None): SA map: sa_max, t0, alpha, beta_min, beta_max.
    """

    _algorithm = "SA"


class ILS(_SingleAlgorithmSolver):
    """Configure an iterated-local-search solver using the shared RKO lifecycle.

    Responsibilities:
    - Select ILS while retaining RKO's callbacks, execution settings and solve result.

    Attributes:
        _algorithm (ClassVar[str]): Canonical ILS selection.
        hyperparameters (Hyperparameters | None): ILS map: beta_min, beta_max.
    """

    _algorithm = "ILS"


class VNS(_SingleAlgorithmSolver):
    """Configure a variable-neighborhood-search solver using the shared RKO lifecycle.

    Responsibilities:
    - Select VNS while retaining RKO's callbacks, execution settings and solve result.

    Attributes:
        _algorithm (ClassVar[str]): Canonical VNS selection.
        hyperparameters (Hyperparameters | None): VNS map: k_max, beta_min.
    """

    _algorithm = "VNS"


class LNS(_SingleAlgorithmSolver):
    """Configure a large-neighborhood-search solver using the shared RKO lifecycle.

    Responsibilities:
    - Select LNS while retaining RKO's callbacks, execution settings and solve result.

    Attributes:
        _algorithm (ClassVar[str]): Canonical LNS selection.
        hyperparameters (Hyperparameters | None): LNS map: beta_min, beta_max, t0, alpha_lns.
    """

    _algorithm = "LNS"


class BRKGA(_SingleAlgorithmSolver):
    """Configure a biased-random-key genetic solver using the shared RKO lifecycle.

    Responsibilities:
    - Select BRKGA while retaining RKO's callbacks, execution settings and solve result.

    Attributes:
        _algorithm (ClassVar[str]): Canonical BRKGA selection.
        hyperparameters (Hyperparameters | None): BRKGA map: p, pe, pm, rhoe.
    """

    _algorithm = "BRKGA"


class BRKGA_CS(_SingleAlgorithmSolver):
    """Configure a community-structured BRKGA solver using the shared RKO lifecycle.

    Responsibilities:
    - Select BRKGA-CS while retaining RKO's callbacks, execution settings and solve result.

    Attributes:
        _algorithm (ClassVar[str]): Canonical BRKGA_CS selection.
        hyperparameters (Hyperparameters | None): BRKGA_CS map: p, pe, pm, rhoe.
    """

    _algorithm = "BRKGA_CS"


class GeneticAlgorithm(_SingleAlgorithmSolver):
    """Configure a genetic-algorithm solver using the shared RKO lifecycle.

    Responsibilities:
    - Select GA while retaining RKO's callbacks, execution settings and solve result.

    Attributes:
        _algorithm (ClassVar[str]): Canonical GA selection.
        hyperparameters (Hyperparameters | None): GA map: size_pop, prob_cros, prob_mut.
    """

    _algorithm = "GA"


class PSO(_SingleAlgorithmSolver):
    """Configure a particle-swarm solver using the shared RKO lifecycle.

    Responsibilities:
    - Select PSO while retaining RKO's callbacks, execution settings and solve result.

    Attributes:
        _algorithm (ClassVar[str]): Canonical PSO selection.
        hyperparameters (Hyperparameters | None): PSO map: p_size, c1, c2, w.
    """

    _algorithm = "PSO"


class GRASP(_SingleAlgorithmSolver):
    """Configure a randomized-construction solver using the shared RKO lifecycle.

    Responsibilities:
    - Select GRASP while retaining RKO's callbacks, execution settings and solve result.

    Attributes:
        _algorithm (ClassVar[str]): Canonical GRASP selection.
        hyperparameters (Hyperparameters | None): GRASP map: hs, he.
    """

    _algorithm = "GRASP"


class IPR(_SingleAlgorithmSolver):
    """Configure an implicit-path-relinking solver using the shared RKO lifecycle.

    Responsibilities:
    - Select IPR while retaining RKO's callbacks, execution settings and solve result.

    Attributes:
        _algorithm (ClassVar[str]): Canonical IPR selection.
        hyperparameters (Hyperparameters | None): No configurable search scalars.
    """

    _algorithm = "IPR"


class MultiStart(_SingleAlgorithmSolver):
    """Configure a multi-start solver using the shared RKO lifecycle.

    Responsibilities:
    - Select MultiStart while retaining RKO's callbacks, execution settings and solve result.

    Attributes:
        _algorithm (ClassVar[str]): Canonical MultiStart selection.
        hyperparameters (Hyperparameters | None): No configurable search scalars.
    """

    _algorithm = "MultiStart"
