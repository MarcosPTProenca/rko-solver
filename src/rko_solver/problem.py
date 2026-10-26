"""Problem definitions accepted by the single `RKO` constructor.

Objective:
- Name how a problem is expressed, so the engine stays an independent parameter.

Technical approach:
- `Callbacks` wraps a Python decoder and objective, `Numba` a compiled objective evaluated
  without the interpreter, and `Blocks` a vectorized objective scoring many candidates.
- Built-in problems live in `rko_solver.problems` and expose a `dimension`.
- Each definition only stores and validates data; engines decide how to run it.

Flow (pseudocode):
1. Build a problem object from callbacks, a Numba function or a block function.
2. Pass it to `RKO(problem, engine=..., execution=...)`.
3. `RKO` reads `kind` to decide which engines can run it.

Important criteria:
- `Numba` validates its function and dimension when constructed and loads the compiled
  extension only when a native engine needs it.
- `Callbacks` defers validation to `solve`, like the positional `RKO(decoder, cost, n)` form.

Inputs/Outputs:
- Input: callbacks, a Numba function or a block function plus the key dimension.
- Output: immutable-by-convention problem objects consumed by `RKO`.
"""

from __future__ import annotations

from .engines.rust.batch import BatchFn
from .engines.rust.batch import BatchObjective
from .engines.rust.compiled import compile_objective
from .orchestration.types import CostFn
from .orchestration.types import DecoderFn
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Callbacks:
    """Describe a problem by a Python decoder and objective.

    Responsibilities:
    - Carry the callbacks and key dimension unchanged to any engine.

    Attributes:
        decoder (DecoderFn): Random-key decoder.
        cost (CostFn): Objective applied to decoded candidates.
        n_variables (int): Positive key dimension, validated at solve time.
        kind (ClassVar[str]): Always `callbacks`.
    """

    decoder: DecoderFn
    cost: CostFn
    n_variables: int
    kind: ClassVar[str] = "callbacks"


class Numba:
    """Describe a problem by a Numba-compiled objective that runs without the interpreter.

    Responsibilities:
    - Validate and compile the function once, keeping its machine code alive.
    - Build the extension problem lazily for the Rust and C++ engines.

    Attributes:
        n_variables (int): Positive key dimension.
        entry (CompiledObjective): Native entry point and its owner.
        kind (str): Always `native`.
    """

    kind = "native"

    def __init__(self, function: object, n_variables: int) -> None:
        """Compile the function and validate the dimension.

        Args:
            function (object): An `@njit` function mapping a 1-D float64 key array to a finite
                float, or a `cfunc` with signature `float64(CPointer(float64), intp)`. It is
                called from several threads, so it must be pure and must not raise.
            n_variables (int): Positive key dimension.

        Raises:
            ImportError: If Numba is not installed.
            ValueError: If the function is not a Numba function or the dimension is invalid.
        """
        if type(n_variables) is not int or n_variables <= 0:
            raise ValueError("'n_variables' must be a positive integer.")
        self.n_variables = n_variables
        self.entry = compile_objective(function)
        self._problem: object | None = None

    @property
    def dimension(self) -> int:
        """Return the key dimension.

        Returns:
            int: Positive number of random keys.
        """
        return self.n_variables

    def native(self) -> object:
        """Return the extension problem object, creating it on first use.

        Returns:
            object: A `rko_solver._native.CompiledProblem`.

        Raises:
            ImportError: If the compiled extension is not installed.
        """
        if self._problem is None:
            from . import _native

            self._problem = _native.CompiledProblem(
                self.entry.address, self.n_variables, self.entry.owner
            )
        return self._problem


class Blocks:
    """Describe a problem by one vectorized objective that scores blocks of candidates.

    Responsibilities:
    - Validate the function and block size and expose the scalar adapter.

    Attributes:
        objective (BatchObjective): Validated block objective.
        n_variables (int): Positive key dimension, validated at solve time.
        kind (str): Always `batch`.
    """

    kind = "batch"

    def __init__(self, function: BatchFn, n_variables: int, batch_size: int = 64) -> None:
        """Validate the function and block size.

        The function receives a fresh `(rows, n_variables)` float array and returns one
        finite cost per row. The native engines group independent candidates and speculate on
        sequential local-search sweeps to fill blocks of up to `batch_size` rows; the Python
        engine calls it with one row.

        Args:
            function (BatchFn): Decodes and scores a block of random-key vectors.
            n_variables (int): Positive key dimension.
            batch_size (int): Largest block per call, from 1 to 4096; bounds the time of one
                call, which cannot be interrupted.

        Raises:
            ValueError: If the function or block size is invalid.
        """
        self.objective = BatchObjective(function, batch_size)
        self.n_variables = n_variables
