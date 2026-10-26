"""General solver preflight and per-solve configuration preparation.

Objective:
- Validate ordinary solver controls before allocating or launching search workers.

Technical approach:
- Separate dimension, duration, callback, engine and portfolio validation.

Flow (pseudocode):
1. Validate scalar controls and existing callback/backend contracts.
2. Apply explicit overrides to an invocation-owned configuration.
3. Validate portfolio settings and bounded allocation dimensions.

Important criteria:
- Preserve error ordering, duration/allocation limits and existing callback behavior.

Inputs/Outputs:
- Input: configured ordinary solver, positive duration and loaded configuration.
- Output: validated execution mode and portfolio configuration.
"""

from __future__ import annotations

from ..tracing import TraceConfig
from .adaptation import normalize_adaptation
from .engine import EngineSelection
from .limits import RunLimits
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from ..solver import RKO


class SolverPreparation:
    """Validate one ordinary solver invocation before search startup.

    Responsibilities:
    - Preserve existing public validation and override semantics.
    - Apply central run limits before allocations and worker creation.

    Attributes:
        solver (RKO): Configured solver whose public settings are being validated.
    """

    def __init__(self, solver: RKO) -> None:
        """Retain the current solver configuration without copying callbacks.

        Args:
            solver (RKO): Public solver under validation.
        """
        self.solver = solver

    def validate(self, time_limit: float) -> str | None:
        """Validate ordinary controls in their established public error order.

        Args:
            time_limit (float): Requested positive finite search duration.

        Returns:
            str | None: Validated Python execution mode or native-engine None.

        Raises:
            ValueError: If dimensions, settings, callbacks or backend controls are invalid.
        """
        solver = self.solver
        if type(solver.n_variables) is not int or solver.n_variables < (
            0 if solver._model_snapshot is not None else 1
        ):
            raise ValueError("'n_variables' must be a positive integer.")
        self._budgets(time_limit)
        execution = EngineSelection.validate(solver.engine, solver.execution)
        if solver._native_problem is not None and solver.engine == "python":
            raise ValueError(
                f"{type(solver._native_problem).__name__} runs only on the compiled engines; "
                "use engine='rust', 'cpp' or 'auto'."
            )
        if solver.auto_time is not None:
            RunLimits.seconds("auto_time", solver.auto_time)
        self._callbacks()
        self._jobs(execution)
        return execution

    def _budgets(self, time_limit: float) -> None:
        """Apply shared duration/evaluation limits and bounded capture controls.

        Args:
            time_limit (float): Requested positive finite search duration.

        Raises:
            ValueError: If a duration, evaluation, history, trace or verbosity setting is invalid.
        """
        solver = self.solver
        RunLimits.seconds("time_limit", time_limit)
        if type(solver.history_limit) is not int or (
            solver.history_limit != 0 and not 2 <= solver.history_limit <= 10000
        ):
            raise ValueError("'history_limit' must be 0 (disabled) or an integer from 2 to 10000.")
        if solver.trace is not None and not isinstance(solver.trace, TraceConfig):
            raise ValueError("'trace' must be TraceConfig or None.")
        RunLimits.evaluations(solver.max_evaluations)
        if type(solver.verbose) is not bool:
            raise ValueError("'verbose' must be a boolean.")

    def _callbacks(self) -> None:
        """Validate opaque callback callability and the existing seed source contract.

        Raises:
            ValueError: If callbacks or random-state settings are invalid.
        """
        solver = self.solver
        if not callable(solver.decoder) or not callable(solver.cost):
            raise ValueError("'decoder' and 'cost' must be callable.")
        if solver.random_state is not None and (
            type(solver.random_state) is not int or solver.random_state < 0
        ):
            raise ValueError("'random_state' must be a nonnegative integer or None.")

    def _jobs(self, execution: str | None) -> None:
        """Reject worker controls that do not apply to the selected backend.

        Args:
            execution (str | None): Validated backend execution mode.

        Raises:
            ValueError: If n_jobs is supplied to an unsupported backend.
        """
        solver = self.solver
        if solver.n_jobs is not None and not (
            solver.engine == "rust" or (solver.engine == "python" and execution == "process")
        ):
            raise ValueError(
                "'n_jobs' is only supported by engine='rust' and by engine='python' "
                "with execution='process'."
            )

    def overrides(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Apply invocation-owned portfolio overrides before driver-side parsing.

        Args:
            raw (dict[str, Any]): Fresh configuration loaded through the public driver boundary.

        Returns:
            dict[str, Any]: Invocation-owned configuration with explicit public overrides.

        Raises:
            ValueError: If configuration, overrides or bounded allocations are invalid.
        """
        solver = self.solver
        run_section = raw.get("run")
        if not isinstance(run_section, dict):
            raise ValueError("Configuration section 'run' must be a table.")
        if solver.algorithms is not None:
            if isinstance(solver.algorithms, (str, bytes)) or not isinstance(
                solver.algorithms, Sequence
            ):
                raise ValueError("'algorithms' must be a nonempty sequence of names.")
            raw["active_metaheuristics"] = {"list": list(solver.algorithms)}
        control = normalize_adaptation(solver.adaptation, solver.control)
        if control is not None:
            run_section.pop("adaptation", None)
        for key, setting in (
            ("control", control),
            ("pool_size", solver.pool_size),
            ("prefill_pool", solver.prefill_pool),
        ):
            if setting is not None:
                run_section[key] = setting
        return raw
