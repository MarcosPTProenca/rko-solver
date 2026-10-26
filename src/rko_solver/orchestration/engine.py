"""Engine and execution selection, including the deprecated `backend` spelling.

Objective:
- Give every entry point one validated vocabulary for which engine runs the search
  and, for the Python engine, how its algorithms are scheduled.

Technical approach:
- Treat the engine (`python`, `rust`, `cpp`, `auto`) and the execution (`process`,
  `thread`, `sequential`) as orthogonal choices; only the Python engine has an execution.
- Translate the legacy single `backend` string into that pair and warn once per call.
- Expose the reverse mapping so read-only legacy aliases keep their old values.

Flow (pseudocode):
1. Receive the requested engine, execution and optional legacy backend.
2. If a backend is given, reject mixing it with the new names, warn and translate it.
3. Validate the pair and resolve the default execution for the Python engine.

Important criteria:
- `execution` is rejected for every engine except `python`.
- Defaults reproduce the previous behavior: engine `python`, execution `process`.
- The legacy spelling is removed in 0.4.

Inputs/Outputs:
- Input: engine and execution names and an optional deprecated backend name.
- Output: validated `(engine, execution)` pairs and legacy backend names.
"""

from __future__ import annotations

import warnings
from typing import Final


class EngineSelection:
    """Validate and translate engine and execution names.

    Responsibilities:
    - Hold the accepted vocabulary and the legacy translation table.
    - Resolve a constructor request into a pair without side effects beyond a warning.

    Attributes:
        ENGINES (tuple[str, ...]): Accepted engine names.
        EXECUTIONS (tuple[str, ...]): Accepted Python-engine execution names.
        DEFAULT_EXECUTION (str): Execution used by the Python engine when none is given.
        LEGACY (dict[str, tuple[str, str | None]]): Deprecated backend to engine and execution.
    """

    ENGINES: Final = ("python", "rust", "cpp", "auto")
    EXECUTIONS: Final = ("process", "thread", "sequential")
    DEFAULT_EXECUTION: Final = "process"
    LEGACY: Final = {
        "process": ("python", "process"),
        "thread": ("python", "thread"),
        "serial": ("python", "sequential"),
        "native": ("rust", None),
        "cpp": ("cpp", None),
        "auto": ("auto", None),
    }

    @classmethod
    def from_legacy(cls, backend: str) -> tuple[str, str | None]:
        """Translate a deprecated backend name.

        Args:
            backend (str): One of the keys of `LEGACY`.

        Returns:
            tuple[str, str | None]: Engine and execution (None outside the Python engine).

        Raises:
            ValueError: If the name is not a known legacy backend.
        """
        if backend not in cls.LEGACY:
            raise ValueError(
                "'backend' must be 'process', 'thread', 'serial', 'native', 'cpp', or 'auto'; "
                "it is deprecated, use 'engine' and 'execution'."
            )
        return cls.LEGACY[backend]

    @classmethod
    def request(
        cls,
        engine: str | None,
        execution: str | None,
        backend: str | None,
        stacklevel: int = 3,
    ) -> tuple[str, str | None]:
        """Resolve constructor arguments into the stored engine and execution.

        Args:
            engine (str | None): Requested engine, or None for the default.
            execution (str | None): Requested Python-engine execution, or None.
            backend (str | None): Deprecated combined name, or None.
            stacklevel (int): Frames to skip so the warning points at the caller.

        Returns:
            tuple[str, str | None]: Engine and execution exactly as requested; validation of
            the values is deferred to `validate`.

        Raises:
            ValueError: If `backend` is combined with `engine` or `execution`, or is unknown.
        """
        if backend is None:
            return ("python" if engine is None else engine), execution
        if engine is not None or execution is not None:
            raise ValueError("'backend' cannot be combined with 'engine' or 'execution'.")
        resolved = cls.from_legacy(backend)
        warnings.warn(
            f"'backend={backend!r}' is deprecated and will be removed in 0.4; use "
            f"engine={resolved[0]!r}"
            + (f", execution={resolved[1]!r}" if resolved[1] is not None else "")
            + ".",
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        return resolved

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

    @classmethod
    def legacy_name(cls, engine: str, execution: str | None) -> str:
        """Return the deprecated backend name equivalent to a resolved pair.

        Args:
            engine (str): Executed engine.
            execution (str | None): Python-engine execution, or None.

        Returns:
            str: `process`, `thread`, `serial`, `native` or `cpp`.
        """
        if engine == "python":
            return {"thread": "thread", "sequential": "serial"}.get(execution or "", "process")
        return "native" if engine == "rust" else engine
