"""Orthogonal engine and execution selection.

Objective:
- Give every entry point one validated vocabulary for which engine runs the search
  and, for the Python engine, how its algorithms are scheduled.

Technical approach:
- Treat the engine (`python`, `rust`, `cpp`, `auto`) and the execution (`process`,
  `thread`, `sequential`) as orthogonal choices; only the Python engine has an execution.

Flow (pseudocode):
1. Receive the requested engine and optional execution.
2. Reject unknown names and execution settings for non-Python engines.
3. Resolve the default execution for the Python engine.

Important criteria:
- `execution` is rejected for every engine except `python`.
- Defaults reproduce the previous behavior: engine `python`, execution `process`.

Inputs/Outputs:
- Input: engine and optional execution names.
- Output: validated effective execution, or None for non-Python engines.
"""

from __future__ import annotations

from .configuration import canonical_mh_name
from collections.abc import Sequence
from pathlib import Path
from typing import Final

MIN_KEYS_CPP_IPR: Final = 10


class EngineSelection:
    """Validate independent engine and execution names.

    Responsibilities:
    - Hold the accepted engine and execution vocabulary.
    - Resolve Python scheduling defaults without side effects.

    Attributes:
        ENGINES (tuple[str, ...]): Accepted engine names.
        EXECUTIONS (tuple[str, ...]): Accepted Python-engine execution names.
        DEFAULT_EXECUTION (str): Execution used by the Python engine when none is given.
    """

    ENGINES: Final = ("python", "rust", "cpp", "auto")
    EXECUTIONS: Final = ("process", "thread", "sequential")
    DEFAULT_EXECUTION: Final = "process"

    @staticmethod
    def request(engine: str | None, execution: str | None) -> tuple[str, str | None]:
        """Resolve the constructor default without starting solve-time validation.

        Args:
            engine (str | None): Requested engine, or None for Python.
            execution (str | None): Requested Python scheduling, or None.

        Returns:
            tuple[str, str | None]: Stored engine and requested execution.
        """
        return ("python" if engine is None else engine), execution

    @staticmethod
    def check_cpp_keys(
        engine: str,
        n_variables: int,
        algorithms: Sequence[str] | str | None,
        params_dir: Path | None,
    ) -> None:
        """Reject, before `solve`, a C++ run whose IPR cannot work on so few keys.

        The reference computes IPR's block size as `dimension / 10`. Selections the constructor
        cannot see (custom `params_dir` without explicit `algorithms`) stay checked at solve time.

        Args:
            engine (str): Requested engine.
            n_variables (int): Number of random keys.
            algorithms (Sequence[str] | str | None): Explicit portfolio, or None for the default.
            params_dir (Path | None): Optional parameter directory that may change the default.

        Raises:
            ValueError: If `engine` is `cpp`, IPR is selected and `n_variables` is below
                `MIN_KEYS_CPP_IPR`.
        """
        if engine != "cpp" or n_variables >= MIN_KEYS_CPP_IPR:
            return
        if not EngineSelection._selects_ipr(algorithms, params_dir):
            return
        raise ValueError(
            f"engine='cpp' with IPR needs at least {MIN_KEYS_CPP_IPR} keys, got {n_variables}: "
            "the C++ reference's IPR block size is dimension / 10 and divides by zero below "
            "that. Use more keys, drop IPR from `algorithms`, or choose another engine."
        )

    @staticmethod
    def _selects_ipr(algorithms: Sequence[str] | str | None, params_dir: Path | None) -> bool:
        """Report whether IPR is in the resolved portfolio the constructor can see.

        Args:
            algorithms (Sequence[str] | str | None): Explicit portfolio, or None for the default.
            params_dir (Path | None): Optional parameter directory that may change the default.

        Returns:
            bool: True when IPR is selected. A custom `params_dir` without explicit `algorithms`
                hides the portfolio from the constructor, so this returns False and defers to
                solve-time validation.
        """
        if algorithms is None:
            return params_dir is None
        if isinstance(algorithms, str):
            return False
        return any(
            isinstance(name, str) and canonical_mh_name(name) == "IPR" for name in algorithms
        )

    @classmethod
    def validate(cls, engine: str, execution: str | None) -> str | None:
        """Check a pair and resolve the execution default.

        Args:
            engine (str): Engine name.
            execution (str | None): Python-engine execution, or None.

        Returns:
            str | None: The effective execution for `python`, otherwise None.

        Raises:
            ValueError: If a name is unknown or `execution` is used with another engine.
        """
        if engine not in cls.ENGINES:
            raise ValueError("'engine' must be 'python', 'rust', 'cpp', or 'auto'.")
        if execution is not None and execution not in cls.EXECUTIONS:
            raise ValueError("'execution' must be 'process', 'thread', or 'sequential'.")
        if engine != "python":
            if execution is not None:
                raise ValueError(
                    f"'execution' applies only to engine='python', not engine={engine!r}."
                )
            return None
        return execution or cls.DEFAULT_EXECUTION
