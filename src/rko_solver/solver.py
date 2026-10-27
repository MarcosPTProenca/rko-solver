"""Class-based cooperative random-key optimization.

Objective:
- Own problem callbacks, portfolio configuration, execution and result state.

Technical approach:
- Validate settings at solve time and build a fresh cooperating portfolio.
- Dispatch to the Python, Rust or C++ engine; the Python engine runs its algorithms as
  processes, threads or one after another.

Flow (pseudocode):
1. Construct RKO without starting work or reading configuration files.
2. Validate inputs and combine Python settings with packaged TOML defaults.
3. Execute searches and publish an immutable, measured result.

Important criteria:
- Repeated solves do not reuse algorithm state or a previous successful result.
- Process callbacks must be importable and pickleable under spawn.
- Timed cooperation is scheduling-dependent even with a fixed seed.

Inputs/Outputs:
- Input: callbacks/Numba/block/built-in problem, execution settings and time budget.
- Output: RKOResult with best keys, objective and execution metadata.
"""

from __future__ import annotations

import copy
import numpy as np
import os
import platform
import time
import warnings
from .components.random_streams import RandomStreams
from .components.solution_pool import SolutionPool
from .engines.rust.batch import BatchFn
from .engines.rust.batch import BatchObjective
from .engines.rust.batch import identity_decoder
from .engines.rust.runner import NativeRunner
from .metaheuristics.base import MetaheuristicBase
from .modeling.adapters import ExtractedLinearModel
from .modeling.construction import ModelConstruction
from .modeling.factory import ModelFactory
from .modeling.factory import ModelOptions
from .modeling.linear import CompiledLinearModel
from .modeling.linear import ModelEvidence
from .modeling.runtime import ModelRunState
from .orchestration.adaptation import Adaptation
from .orchestration.auto import AutoCandidate
from .orchestration.auto import AutoEngine
from .orchestration.configuration import parse_run_config
from .orchestration.engine import EngineSelection
from .orchestration.execution import run_metaheuristics_parallel
from .orchestration.execution import select_best_result
from .orchestration.factory import build_metaheuristic
from .orchestration.factory import normalize_hyperparameters
from .orchestration.initialization import initialize_pool
from .orchestration.limits import RunLimits
from .orchestration.preparation import SolverPreparation
from .orchestration.processes import run_processes
from .orchestration.search import SearchReporting
from .orchestration.search import SearchSession
from .orchestration.types import CostFn
from .orchestration.types import DecoderFn
from .orchestration.types import ExecutionMetadata
from .orchestration.types import Hyperparameters
from .orchestration.types import NativeProblem
from .orchestration.types import PoolInitializationMetadata
from .orchestration.types import RunConfig
from .orchestration.types import RunResult
from .orchestration.types import SearchMetadata
from .parameters.loader import load_config
from .problem import Blocks
from .problem import Callbacks
from .problem import Numba
from .result import RKOResult as RKOResult
from .tracing import DecisionTrace
from .tracing import TraceConfig
from collections.abc import Sequence
from dataclasses import replace
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path
from typing import Literal
from typing import Unpack

Engine = Literal["python", "rust", "cpp", "auto"]
Execution = Literal["process", "thread", "sequential"]
LegacyBackend = Literal["process", "thread", "serial", "native", "cpp", "auto"]
Problem = Callbacks | Numba | Blocks | NativeProblem


class RKO:
    """Configure and execute a cooperating portfolio of random-key searches.

    Responsibilities:
    - Own problem definition and explicit engine/execution/portfolio settings.
    - Validate each solve, create fresh search state and expose its result.

    Attributes:
        problem (Problem): Problem definition: `Callbacks`, `Numba`, `Blocks` or a built-in.
        decoder (DecoderFn): Random-key decoder (identity for compiled problems).
        cost (CostFn): Objective applied to decoded candidates.
        n_variables (int): Positive key dimension.
        algorithms (Sequence[str] | None): Selection overriding TOML defaults.
        control (int | None): Legacy mode override; None leaves configuration in charge.
        adaptation (Adaptation | None): Readable fixed/adaptive mode override.
        pool_size (int | None): Shared elite capacity; None uses TOML defaults.
        prefill_pool (bool | None): Opt-in reference-style startup; None uses TOML.
        params_dir (Path | None): Optional parameter directory.
        engine (str): `python`, `rust`, `cpp` or `auto`; auto chooses among the others by a
            short simulation on this problem.
        execution (str | None): For the python engine, `process` (default), `thread` or
            `sequential`; None for the other engines.
        auto_time (float | None): Selection simulation seconds for auto, separate from the limit.
        auto_candidates (Sequence[str] | None): Candidates auto may choose from.
        n_jobs (int | None): Positive process or Rust worker limit, excluding the manager.
        random_state (int | None): Nonnegative per-search seed in every engine.
        verbose (bool): Print lifecycle and attributed shared-pool updates; default False.
        hyperparameters (Hyperparameters | None): Partial named offline scalar overrides.
        history_limit (int): Per-search scalar history cap; zero disables recording.
        max_evaluations (int | None): Completed evaluations after which each algorithm stops;
            None leaves only the time limit.
        result_ (RKOResult | None): Last successful solve; cleared before each attempt.
    """

    def __init__(
        self,
        problem: Problem | DecoderFn | None = None,
        cost: CostFn | None = None,
        n_variables: int | None = None,
        *,
        decoder: DecoderFn | None = None,
        algorithms: Sequence[str] | None = None,
        adaptation: Adaptation | None = None,
        control: int | None = None,
        pool_size: int | None = None,
        prefill_pool: bool | None = None,
        params_dir: Path | None = None,
        engine: Engine | None = None,
        execution: Execution | None = None,
        backend: LegacyBackend | None = None,
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
        """Store problem and solver settings without I/O or search execution.

        Args:
            problem (Problem | DecoderFn | None): A problem object (`rko_solver.Callbacks`,
                `rko_solver.Numba`, `rko_solver.Blocks` or a built-in from
                `rko_solver.problems`), or, in the short form `RKO(decoder, cost, n_variables)`,
                the random-key decoder.
            cost (CostFn | None): Objective to minimize after decoding; short form only.
            n_variables (int | None): Positive key dimension; short form only.
            decoder (DecoderFn | None): Keyword spelling of the short form's decoder, so
                `RKO(decoder=..., cost=..., n_variables=...)` keeps working.
            algorithms (Sequence[str] | None): Portfolio overriding configuration.
            adaptation (Adaptation | None): "fixed" parameters chosen before the search
                or "adaptive" Q-learning choices during it; None uses the configured
                default. Prefer this readable name over the legacy integer code.
            control (int | None): Legacy 0 (fixed) or 1 (adaptive); accepted on its own
                for compatibility and rejected when it contradicts `adaptation`.
            pool_size (int | None): Elite capacity, or configured default.
            prefill_pool (bool | None): Reference startup override; adds a separately
                accounted cooperative allowance equal to time_limit before fresh searches.
            params_dir (Path | None): Optional parameter TOML directory.
            engine (Engine | None): "python" (default), "rust" (compiled engine), "cpp" (C++
                reference, available only in builds made with `RKO_BUILD_CPP=1`) or "auto".
                Python callbacks serialize on the interpreter lock in the compiled engines;
                `Numba` and built-in problems run in parallel and need rust, cpp or auto.
            execution (Execution | None): Python engine only: "process" (default),
                "thread" (legacy, GIL-bound) or "sequential" (algorithms one after another in
                the calling thread, each with the full time limit).
            backend (LegacyBackend | None): Deprecated combined name, removed in 0.4;
                cannot be combined with `engine` or `execution`.
            auto_time (float | None): For `engine="auto"`, total seconds of the selection
                simulation, separate from the search `time_limit`; None uses 10 percent of it.
                Each candidate runs twice; a probe gets `auto_time / (2 * candidates)`
                seconds, at least 0.05.
            auto_candidates (Sequence[str] | None): For `engine="auto"`, a subset of
                "process", "rust", "rust-threads" and "cpp" to choose from.
            n_jobs (int | None): Positive worker limit; None uses available logical CPUs
                (process) or one Rust thread per algorithm.
                Missing CPU discovery falls back to one worker. Smaller limits share
                workers and split the common search budget; explicit limits override CPUs.
            random_state (int | None): Nonnegative seed for owned library RNG streams.
            verbose (bool): Flush lifecycle and shared-pool progress to stdout.
            hyperparameters (Hyperparameters | None): Per-algorithm offline overrides;
                require control=0 and merge with TOML defaults without writing files.
            history_limit (int): Zero disables history; 2 through 10000 bounds retained
                scalar observations per search. Compacted curves are approximate.
            trace (TraceConfig | None): Opt-in bounded decoder decisions; None disables capture.
            max_evaluations (int | None): Positive number of completed objective evaluations
                after which EACH algorithm stops, in every engine. The time limit still applies
                and the first of the two to be reached ends the search; opt-in pool prefill is
                not counted. It makes runs independent of machine speed, and with one worker
                and a fixed seed reproducible.

        Raises:
            TypeError: If a problem object is combined with `cost` or `n_variables`, the
                short form lacks them, or `problem` and `decoder` are both given.
            ValueError: If `backend` is unknown or combined with `engine` or `execution`.
        """
        self._native_problem: Problem | None = None
        self._batch: BatchObjective | None = None
        if decoder is not None:
            if problem is not None:
                raise TypeError("Pass either 'problem' or 'decoder', not both.")
            problem = decoder
        if isinstance(problem, (Callbacks, Numba, Blocks)) or (
            cost is None and n_variables is None and hasattr(problem, "dimension")
        ):
            if cost is not None or n_variables is not None:
                raise TypeError("'cost' and 'n_variables' belong to the problem object.")
            self.problem = problem
        else:
            if cost is None and n_variables is None:
                raise TypeError("RKO(decoder, cost, n_variables) needs a cost and a dimension.")
            self.problem = Callbacks(problem, cost, n_variables)
        if isinstance(self.problem, Callbacks):
            self.decoder, self.cost = self.problem.decoder, self.problem.cost
            self.n_variables = self.problem.n_variables
        elif isinstance(self.problem, Blocks):
            self.decoder, self.cost = identity_decoder, self.problem.objective
            self.n_variables = self.problem.n_variables
            self._batch = self.problem.objective
        else:
            self.decoder, self.cost = identity_decoder, lambda value: 0.0
            self.n_variables = self.problem.dimension
            self._native_problem = self.problem
        self.algorithms = algorithms
        self.adaptation = adaptation
        self.control = control
        self.pool_size = pool_size
        self.prefill_pool = prefill_pool
        self.params_dir = params_dir
        self.engine, self.execution = EngineSelection.request(engine, execution, backend)
        self.auto_time = auto_time
        self.auto_candidates = auto_candidates
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.hyperparameters = hyperparameters
        self.history_limit = history_limit
        self.trace = trace
        self.max_evaluations = max_evaluations
        self.result_: RKOResult | None = None
        self._native_workers: int | None = None
        self._model_source: ExtractedLinearModel | None = None
        self._model_snapshot: CompiledLinearModel | ModelConstruction | None = None
        self._model_compilation_seconds = 0.0

    @property
    def backend(self) -> str:
        """Return the deprecated combined backend name for this solver.

        Returns:
            str: `process`, `thread`, `serial`, `native`, `cpp` or `auto`.
        """
        warnings.warn(
            "'RKO.backend' is deprecated and will be removed in 0.4; use 'engine' and 'execution'.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.engine == "auto":
            return "auto"
        return EngineSelection.legacy_name(self.engine, self.execution)

    @classmethod
    def native(cls, problem: NativeProblem, **options: object) -> RKO:
        """Build a solver around a built-in problem; deprecated, use `RKO(problem, engine=...)`.

        Args:
            problem (NativeProblem): Built-in problem from `rko_solver.problems`.
            options (object): Solver settings as for `RKO`; `engine` defaults to "rust".

        Returns:
            RKO: Solver that runs the compiled engine on `problem`.
        """
        warnings.warn(
            "'RKO.native' is deprecated and will be removed in 0.4; "
            "use RKO(problem, engine='rust').",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls._legacy_constructor(problem, options)

    @classmethod
    def compiled(cls, objective: object, n_variables: int, **options: object) -> RKO:
        """Build a solver around a Numba objective; deprecated, use `rko_solver.Numba`.

        Args:
            objective (object): An `@njit` function or `cfunc`, as for `rko_solver.Numba`.
            n_variables (int): Positive key dimension.
            options (object): Solver settings as for `RKO`; `engine` defaults to "rust".

        Returns:
            RKO: Solver running the compiled engine on `objective`.

        Raises:
            ImportError: If Numba is not installed.
            ValueError: If `objective` is not a Numba function or the dimension is invalid.
        """
        warnings.warn(
            "'RKO.compiled' is deprecated and will be removed in 0.4; "
            "use RKO(rko_solver.Numba(function, n_variables), engine='rust').",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls._legacy_constructor(Numba(objective, n_variables), options)

    @classmethod
    def batched(
        cls, batch_objective: BatchFn, n_variables: int, batch_size: int = 64, **options: object
    ) -> RKO:
        """Build a solver around a block objective; deprecated, use `rko_solver.Blocks`.

        Args:
            batch_objective (BatchFn): Decodes and scores a block of random-key vectors.
            n_variables (int): Positive key dimension.
            batch_size (int): Largest block per call, from 1 to 4096.
            options (object): Solver settings as for `RKO`; `engine` defaults to "rust".

        Returns:
            RKO: Solver using the block objective.

        Raises:
            ValueError: If the function or block size is invalid.
        """
        warnings.warn(
            "'RKO.batched' is deprecated and will be removed in 0.4; "
            "use RKO(rko_solver.Blocks(function, n_variables, batch_size), engine='rust').",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls._legacy_constructor(Blocks(batch_objective, n_variables, batch_size), options)

    @classmethod
    def _legacy_constructor(cls, problem: Problem, options: dict[str, object]) -> RKO:
        """Apply the historical Rust default of the deprecated alternative constructors.

        Args:
            problem (Problem): Problem object to solve.
            options (dict[str, object]): Caller keyword settings.

        Returns:
            RKO: Solver on the Rust engine unless `engine` or `backend` is given.
        """
        if "engine" not in options and "backend" not in options:
            options = {**options, "engine": "rust"}
        return cls(problem, **options)

    @classmethod
    def from_model(cls, model: object, **options: Unpack[ModelOptions]) -> RKO:
        """Own a supported linear model for the Python callback backends.

        Args:
            model (object): Trusted qualified Pyomo/PuLP/MathOpt model.
            options (ModelOptions): Explicit compilation and existing portfolio settings.

        Returns:
            RKO: Solver holding only owned source/numerical data, never the native model.

        Raises:
            ValueError: If model features, bounds, numerical settings or backend are unsupported.
            TypeError: If an unknown construction keyword is supplied.
            ImportError: If the selected optional frontend cannot load.
        """
        return ModelFactory.create(cls, model, options)

    def solve(self, time_limit: float) -> RKOResult:
        """Execute a fresh portfolio and retain its measured result.

        Args:
            time_limit (float): Positive finite search seconds, excluding startup;
                sequential execution gives each algorithm this duration. Opt-in
                shared initialization has a separate equal cooperative allowance.

        Returns:
            RKOResult: Best candidate and execution metadata.

        Raises:
            ValueError: If callbacks, dimensions, settings or serialization are invalid.
            RuntimeError: If an algorithm fails or a process exits unexpectedly.
            TimeoutError: If process startup, initialization or search exceeds its grace.
            OSError: If parameter files cannot be read.
        """
        self.result_ = None
        started = time.perf_counter()
        model_state = ModelRunState(self)
        model_state.validate()
        preparation = SolverPreparation(self)
        execution = preparation.validate(time_limit)
        if self.engine == "auto":
            return self._solve_auto(time_limit, started)
        config = parse_run_config(preparation.overrides(load_config(self.params_dir)))
        RunLimits.allocation("pool_size", config.pool_size, self.n_variables)
        hyperparameters = normalize_hyperparameters(
            self.hyperparameters, config.active_metaheuristics, config.control
        )
        if self.verbose:
            print(
                f"RKO START engine={self.engine}"
                + (f" execution={execution}" if execution else "")
                + " "
                f"algorithms=[{', '.join(config.active_metaheuristics)}] "
                f"search_budget={time_limit:g}s",
                flush=True,
            )
        metadata: list[SearchMetadata] = []
        initialization: list[PoolInitializationMetadata] = []
        engine, n_jobs, random_state, n_variables = (
            self.engine,
            self.n_jobs,
            self.random_state,
            self.n_variables,
        )
        available_cpus = max(1, os.process_cpu_count() or 1) if execution == "process" else None
        no_search = model_state.no_search
        if no_search:
            keys, value = model_state.direct()
        else:
            keys, value = self._execute(
                config,
                time_limit,
                started_at=started,
                hyperparameters=hyperparameters,
                search_metadata=metadata,
                initialization_metadata=initialization,
                process_worker_limit=n_jobs if n_jobs is not None else available_cpus,
                **model_state.arguments(),
            )
        model_result = model_state.finish(keys, value, started)
        positions = {name: index for index, name in enumerate(config.active_metaheuristics)}
        try:
            package_version = version("rko-solver")
        except PackageNotFoundError:
            package_version = "unknown"
        result = RKOResult(
            keys=tuple(float(key) for key in keys),
            cost=float(value),
            elapsed_seconds=time.perf_counter() - started,
            engine=engine,
            execution=None if no_search else execution,
            algorithms=() if no_search else config.active_metaheuristics,
            random_state=random_state,
            searches=tuple(sorted(metadata, key=lambda report: positions[report.algorithm])),
            n_variables=n_variables,
            time_limit=float(time_limit),
            control=None if no_search else config.control,
            pool_size=None if no_search else config.pool_size,
            n_jobs=(
                None if no_search else self._native_workers if engine in ("rust", "cpp") else n_jobs
            ),
            available_cpus=None if no_search else available_cpus,
            initialization=initialization[0] if initialization else None,
            trace=DecisionTrace.selected(value, keys) if self.trace is not None else None,
            model=model_result,
            max_evaluations=self.max_evaluations,
            versions=(
                ("rko-solver", package_version),
                ("python", platform.python_version()),
                ("numpy", np.__version__),
            ),
        )
        if self.verbose:
            print(
                f"RKO DONE best={result.cost:.8g} elapsed={result.elapsed_seconds:.3f}s",
                flush=True,
            )
        self.result_ = result
        return result

    def _native_object(self) -> object:
        """Return the extension problem for a compiled or built-in problem.

        Returns:
            object: The object passed to the compiled engines.

        Raises:
            ImportError: If a `Numba` problem needs the missing compiled extension.
        """
        problem = self._native_problem
        return problem.native() if isinstance(problem, Numba) else problem

    def _kind(self) -> str:
        """Classify how the problem is defined, which decides the candidate engines.

        Returns:
            str: `native` for built-in or compiled problems, `batch` or `callbacks`.
        """
        if self._native_problem is not None:
            return "native"
        return "batch" if self._batch is not None else "callbacks"

    def _variant(self, candidate: AutoCandidate, seed: int | None, probe: bool = True) -> RKO:
        """Copy this solver with another engine, sharing the problem definition.

        Args:
            candidate (AutoCandidate): Engine, execution and worker limit to run.
            seed (int | None): Seed for the copy.
            probe (bool): Disable history, tracing and progress output for short probes.

        Returns:
            RKO: Independent solver object that can be solved separately.
        """
        variant = copy.copy(self)
        variant.engine = candidate.engine
        variant.execution = candidate.execution
        variant.n_jobs = candidate.n_jobs
        variant.random_state = seed
        variant.result_ = None
        variant._native_workers = None
        if probe:
            variant.verbose = False
            variant.history_limit = 0
            variant.trace = None
        return variant

    def _solve_auto(self, time_limit: float, started: float) -> RKOResult:
        """Choose an engine by short probes, then run the full budget on the winner.

        Args:
            time_limit (float): Search seconds of the final run.
            started (float): Monotonic origin of this solve.

        Returns:
            RKOResult: Best solution over probes and final run, with the selection record.

        Raises:
            ValueError: If the candidate list is invalid or `n_jobs` was given.
            RuntimeError: If every candidate fails.
        """
        if self.n_jobs is not None:
            raise ValueError("'n_jobs' cannot be combined with engine='auto'.")
        from . import _native

        cpp_available = _native.cpp_available()
        selection, probe_keys, probe_cost = AutoEngine.select(
            self, time_limit, self.auto_time, self.auto_candidates, cpp_available, time.perf_counter
        )
        cpus = max(1, os.process_cpu_count() or 1)
        candidate = AutoEngine.build(selection.chosen, cpus)
        if self.verbose:
            print(
                f"RKO AUTO chosen={selection.chosen} reason='{selection.reason}' "
                f"scores={dict(selection.scores)}",
                flush=True,
            )
        final = self._variant(candidate, self.random_state, probe=False).solve(time_limit)
        improved = probe_keys is not None and probe_cost < final.cost
        selection = replace(selection, improved_by_probe=improved)
        result = replace(
            final,
            keys=tuple(probe_keys) if improved else final.keys,
            cost=float(probe_cost) if improved else final.cost,
            elapsed_seconds=time.perf_counter() - started,
            random_state=self.random_state,
            auto=selection,
        )
        self.result_ = result
        return result

    def _execute(
        self,
        config: RunConfig,
        time_limit: float,
        *,
        hyperparameters: Hyperparameters | None = None,
        search_metadata: list[SearchMetadata] | None = None,
        process_worker_limit: int | None = None,
        initialization_metadata: list[PoolInitializationMetadata] | None = None,
        **context: Unpack[ExecutionMetadata],
    ) -> tuple[list[float], float]:
        """Dispatch a validated portfolio without retaining worker state.

        Args:
            config (RunConfig): Validated selection and elite settings.
            time_limit (float): Positive search duration.
            hyperparameters (Hyperparameters | None): Owned normalized offline overrides.
            search_metadata (list[SearchMetadata] | None): Optional immutable report collector.
            process_worker_limit (int | None): Resolved per-solve process cap.
            initialization_metadata (list | None): Optional shared startup report collector.
            context (ExecutionMetadata): Required solve origin and optional bounded model archive.

        Returns:
            tuple[list[float], float]: Best keys and objective.

        Raises:
            ValueError: If process callbacks or worker budget are invalid.
            RuntimeError: If a worker fails.
            TimeoutError: If process execution exceeds its watchdog.
        """
        started_at = context["started_at"]
        model_evidence = context.get("model_evidence")
        execution = EngineSelection.validate(self.engine, self.execution)
        callback_options = SearchReporting.callback_options(self.trace, model_evidence)
        if self.engine in ("rust", "cpp"):
            runner = NativeRunner(
                config,
                self.n_variables,
                self.params_dir,
                hyperparameters,
                engine=self.engine,
            )
            outcome = runner.run(
                self._native_object()
                if self._native_problem is not None
                else self._batch
                if self._batch is not None
                else (self.decoder, self.cost),
                time_limit,
                self.random_state,
                self.n_jobs,
                self.history_limit,
                search_metadata,
                initialization_metadata,
                self.max_evaluations,
            )
            self._native_workers = runner.workers
            return outcome
        pool = SolutionPool(capacity=config.pool_size, verbose=self.verbose, started_at=started_at)
        seeds = np.random.SeedSequence(self.random_state).spawn(len(config.active_metaheuristics))
        streams = [RandomStreams(int(sequence.generate_state(1)[0])) for sequence in seeds]
        metaheuristics = []
        for name, stream in zip(config.active_metaheuristics, streams, strict=True):
            with stream.activate():
                metaheuristics.append(
                    build_metaheuristic(
                        name,
                        pool,
                        self.decoder,
                        self.cost,
                        self.n_variables,
                        config.control,
                        self.params_dir,
                        (hyperparameters or {}).get(name),
                    )
                )
        if execution == "process":
            return run_processes(
                config,
                self.decoder,
                self.cost,
                self.n_variables,
                time_limit,
                self.params_dir,
                process_worker_limit if process_worker_limit is not None else self.n_jobs,
                self.random_state,
                verbose=self.verbose,
                started_at=started_at,
                hyperparameters=hyperparameters,
                search_metadata=search_metadata,
                initialization_metadata=initialization_metadata,
                history_limit=self.history_limit,
                max_evaluations=self.max_evaluations,
                **callback_options,
            )
        pool.start_progress()
        try:
            if config.prefill_pool:
                report = initialize_pool(
                    pool,
                    self.decoder,
                    self.cost,
                    self.n_variables,
                    config.pool_size,
                    time_limit,
                    self.random_state,
                    **callback_options,
                )
                if initialization_metadata is not None:
                    initialization_metadata.append(report)
            if execution == "thread":
                return select_best_result(
                    run_metaheuristics_parallel(
                        metaheuristics,
                        time_limit,
                        streams,
                        search_metadata=search_metadata,
                        history_limit=self.history_limit,
                        max_evaluations=self.max_evaluations,
                        **callback_options,
                    ),
                    pool,
                )
            results = self._run_sequential(
                metaheuristics, streams, time_limit, search_metadata, model_evidence
            )
            return select_best_result(results, pool)
        finally:
            pool.stop_progress()

    def _run_sequential(
        self,
        metaheuristics: list[MetaheuristicBase],
        streams: list[RandomStreams],
        time_limit: float,
        search_metadata: list[SearchMetadata] | None,
        model_evidence: list[ModelEvidence] | None,
    ) -> list[RunResult]:
        """Run fresh sequential searches with independent streams and bounded evidence.

        Args:
            metaheuristics (list[MetaheuristicBase]): Fresh configured algorithms.
            streams (list[RandomStreams]): Independent streams in portfolio order.
            time_limit (float): Positive duration available to each algorithm.
            search_metadata (list[SearchMetadata] | None): Completed report collector.
            model_evidence (list[ModelEvidence] | None): Completed model witness collector.

        Returns:
            list[RunResult]: Completed scalar search outcomes in portfolio order.

        Raises:
            Exception: Original callback, algorithm or validation failures propagate.
        """
        results = []
        for mh, stream in zip(metaheuristics, streams, strict=True):
            with stream.activate():
                session = (
                    SearchSession(self.decoder, self.cost)
                    if self.trace is None
                    else SearchSession(self.decoder, self.cost, trace=self.trace)
                )
                results.append(
                    session.solve(
                        mh,
                        time_limit,
                        random_state=stream.seed,
                        history_limit=self.history_limit,
                        max_evaluations=self.max_evaluations,
                    )
                )
                SearchReporting.collect(session, model_evidence)
                if search_metadata is not None:
                    assert session.metadata is not None
                    search_metadata.append(session.metadata)
        return results
