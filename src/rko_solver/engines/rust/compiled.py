"""Compiled objectives that the native engine calls without the interpreter.

Objective:
- Let a user define the decoder and objective in Python once, compile it with Numba, and
  have every native worker thread evaluate it in parallel at compiled speed.

Technical approach:
- `compile_objective` accepts a Numba `@njit` function of one float64 array, or an
  existing `cfunc` with signature `float64(CPointer(float64), intp)`, and returns its
  native entry point together with the objects that must stay alive.
- A dispatcher is wrapped in a `cfunc` that builds an array view over the engine's key
  buffer with `numba.carray`, so the user function receives a plain 1-D array.

Flow (pseudocode):
1. Import Numba lazily and validate the object passed by the user.
2. Wrap a dispatcher in a `cfunc` with the engine's calling convention.
3. Hand the native address and the owner to the compiled engine.

Important criteria:
- The function must be pure: reentrant, free of Python objects and of exceptions, and
  must not keep or write through the key array; it is called from several threads.
- The array is a view over engine memory for the duration of one call.
- Numba is an optional dependency; this module imports it only when used.

Inputs/Outputs:
- Input: a Numba-compiled objective over one random-key vector.
- Output: a `CompiledObjective` carrying the function address and owner.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from typing import cast


@dataclass(frozen=True)
class CompiledObjective:
    """Describe a native objective entry point and what keeps it alive.

    Responsibilities:
    - Carry the function address and the compiled objects that own its machine code.

    Attributes:
        address (int): Address of `double f(const double *keys, intptr_t n)`.
        owner (object): Compiled objects that must outlive every call.
    """

    address: int
    owner: object


def compile_objective(function: object, cost: object | None = None) -> CompiledObjective:
    """Prepare a Numba function for the native engine.

    Args:
        function (object): An `@njit` function taking a 1-D float64 array and returning a
            finite float, or a `cfunc` with signature `float64(CPointer(float64), intp)`.
            With `cost`, the decoder: an `@njit` function returning whatever `cost` accepts.
        cost (object | None): Optional `@njit` objective applied to the decoder's output,
            compiled into the same native entry point as `cost(function(keys))`.

    Returns:
        CompiledObjective: Native entry point and owner.

    Raises:
        ImportError: If Numba is not installed.
        ValueError: If the object is neither a Numba dispatcher nor a cfunc, or if `cost`
            is given with a cfunc decoder or a non-Numba function.
    """
    carray, cfunc, types = _numba_modules()
    if cost is not None:
        return _compile_pair(function, cost)
    if hasattr(function, "address") and hasattr(function, "ctypes"):
        return CompiledObjective(int(function.address), function)
    if not hasattr(function, "py_func"):
        raise ValueError(
            "Pass a numba @njit function of one float64 array or a cfunc with signature "
            "float64(CPointer(float64), intp)."
        )
    inner = function
    signature = types.float64(types.CPointer(types.float64), types.intp)

    @cfunc(signature, nopython=True)
    def entry(keys: object, size: object) -> float:
        """Adapt the engine's pointer calling convention to an array argument.

        Args:
            keys (object): Pointer to the key buffer.
            size (object): Number of keys.

        Returns:
            float: Objective of the key vector.
        """
        return inner(carray(keys, size))

    return CompiledObjective(int(entry.address), (entry, inner))


def compile_swap(function: object) -> CompiledObjective:
    """Prepare an exact swap delta for the native engine.

    Args:
        function (object): An `@njit` function `(keys, score, i, j) -> float` returning the
            objective after exchanging the key values of indices `i` and `j`, given the current
            key vector and its `score`. It must be exact, pure and must not raise.

    Returns:
        CompiledObjective: Native entry point `double f(const double *keys, intptr_t n,
        double score, intptr_t i, intptr_t j)` and its owner.

    Raises:
        ImportError: If Numba is not installed.
        ValueError: If the object is not a Numba dispatcher.
    """
    carray, cfunc, types = _numba_modules()
    if not hasattr(function, "py_func"):
        raise ValueError("swap_delta must be an @njit function of (keys, score, i, j).")
    inner = cast(Callable[..., float], function)
    signature = types.float64(
        types.CPointer(types.float64), types.intp, types.float64, types.intp, types.intp
    )

    @cfunc(signature, nopython=True)
    def entry(keys: object, size: object, score: object, i: object, j: object) -> float:
        """Adapt the engine's pointer calling convention to an array argument.

        Args:
            keys (object): Pointer to the key buffer.
            size (object): Number of keys.
            score (object): Objective of the current keys.
            i (object): First index.
            j (object): Second index.

        Returns:
            float: Objective after the swap.
        """
        return inner(carray(keys, size), score, i, j)

    return CompiledObjective(int(entry.address), (entry, inner))


def _numba_modules() -> tuple[Any, Any, Any]:
    """Import the Numba helpers used to build a native entry point.

    Returns:
        tuple[Any, Any, Any]: The `carray` adapter, the `cfunc` decorator and the type namespace.

    Raises:
        ImportError: If Numba is not installed.
    """
    try:
        from numba import carray
        from numba import cfunc
        from numba import types
    except ImportError as error:
        raise ImportError(
            "Compiled objectives need Numba; install it with `pip install numba`."
        ) from error
    return carray, cfunc, types


def _compile_pair(decoder: Callable[..., object], cost: Callable[..., object]) -> CompiledObjective:
    """Compile a Numba decoder and cost into one native entry point.

    Args:
        decoder (Callable[..., object]): Numba dispatcher mapping keys to the cost's input.
        cost (Callable[..., object]): Numba dispatcher mapping the decoder's output to a float.

    Returns:
        CompiledObjective: Native entry point and the owned dispatchers.

    Raises:
        ImportError: If Numba is not installed.
        ValueError: If either half is not a Numba dispatcher, or the decoder is a cfunc.
    """
    carray, cfunc, types = _numba_modules()
    if (
        hasattr(decoder, "address")
        or not hasattr(decoder, "py_func")
        or not hasattr(cost, "py_func")
    ):
        raise ValueError("With cost=, both decoder and cost must be numba @njit functions.")
    signature = types.float64(types.CPointer(types.float64), types.intp)

    @cfunc(signature, nopython=True)
    def entry(keys: object, size: object) -> float:
        """Adapt the pointer convention and apply the decoder before the cost.

        Args:
            keys (object): Pointer to the key buffer.
            size (object): Number of keys.

        Returns:
            float: Cost of the decoded key vector.
        """
        return cost(decoder(carray(keys, size)))

    return CompiledObjective(int(entry.address), (entry, decoder, cost))
