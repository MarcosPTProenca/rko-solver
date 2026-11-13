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
from .engines.rust.compiled import CompiledObjective
from .engines.rust.compiled import compile_objective
from .engines.rust.compiled import compile_swap
from .orchestration.types import CostFn
from .orchestration.types import DecoderFn
from dataclasses import dataclass
from pathlib import Path
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
    swap_entry = None

    def __init__(
        self,
        function: object,
        n_variables: int,
        *,
        cost: object | None = None,
        swap_delta: object | None = None,
    ) -> None:
        """Compile the function and validate the dimension.

        Args:
            function (object): An `@njit` function mapping a 1-D float64 key array to a finite
                float, or a `cfunc` with signature `float64(CPointer(float64), intp)`. It is
                called from several threads, so it must be pure and must not raise.
            n_variables (int): Positive key dimension.
            cost (object | None): Optional `@njit` objective applied to the output of `function`,
                which then acts as a decoder. Compiled as `cost(function(keys))` in one entry
                point; the decoder's output must be accepted by `cost` inside nopython mode.
            swap_delta (object | None): Optional exact `@njit` function `(keys, score, i, j)`
                returning the objective after exchanging key values `i` and `j`. Only the Rust
                engine uses it; it is incompatible with `cost`, because the delta is defined on
                key vectors.

        Raises:
            ImportError: If Numba is not installed.
            ValueError: If the function is not a Numba function or the dimension is invalid.
        """
        if type(n_variables) is not int or n_variables <= 0:
            raise ValueError("'n_variables' must be a positive integer.")
        if swap_delta is not None and cost is not None:
            raise ValueError("'swap_delta' acts on key vectors; it cannot be combined with 'cost'.")
        self.n_variables = n_variables
        self.entry = compile_objective(function, cost)
        self.swap_entry = None if swap_delta is None else compile_swap(swap_delta)
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

            if self.swap_entry is None:
                self._problem = _native.CompiledProblem(
                    self.entry.address, self.n_variables, self.entry.owner
                )
            else:
                self._problem = _native.CompiledProblem(
                    self.entry.address,
                    self.n_variables,
                    (self.entry.owner, self.swap_entry.owner),
                    self.swap_entry.address,
                )
        return self._problem


class NativeObjective(Numba):
    """Describe a problem by a C function that a user compiled into a shared library.

    The function must have the C signature `double f(const double *keys, long n)`, be reentrant,
    never raise, and return a finite value. It is called from every worker of the native engines.

    Responsibilities:
    - Load the library and resolve the symbol once, keeping both alive for the solve.
    - Reuse the compiled-objective path that `Numba` uses, so the engines call it directly.

    Attributes:
        n_variables (int): Positive key dimension.
        entry (CompiledObjective): Native entry point and its owners.
        kind (str): Always `native`.
    """

    def __init__(self, library: str | Path, symbol: str, n_variables: int) -> None:
        """Load the shared library and bind the objective symbol.

        Args:
            library (str | Path): Path of the shared library exporting the objective.
            symbol (str): Exported C function name.
            n_variables (int): Positive key dimension.

        Raises:
            ValueError: If the dimension is invalid, the library cannot be loaded, or the
                symbol is not exported.
        """
        import ctypes

        if type(n_variables) is not int or n_variables <= 0:
            raise ValueError("'n_variables' must be a positive integer.")
        path = Path(library)
        try:
            handle = ctypes.CDLL(str(path))
        except OSError as error:
            raise ValueError(f"Cannot load native objective library {path}: {error}") from error
        try:
            function = getattr(handle, symbol)
        except AttributeError as error:
            raise ValueError(f"The library does not export the symbol {symbol!r}.") from error
        function.restype = ctypes.c_double
        function.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_ssize_t]
        address = ctypes.cast(function, ctypes.c_void_p).value
        if address is None:
            raise ValueError(f"The symbol {symbol!r} has no address.")
        self.n_variables = n_variables
        self.entry = CompiledObjective(int(address), (handle, function))
        self._problem: object | None = None


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
