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

from dataclasses import dataclass


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


def compile_objective(function: object) -> CompiledObjective:
    """Prepare a Numba function for the native engine.

    Args:
        function (object): An `@njit` function taking a 1-D float64 array and returning a
            finite float, or a `cfunc` with signature `float64(CPointer(float64), intp)`.

    Returns:
        CompiledObjective: Native entry point and owner.

    Raises:
        ImportError: If Numba is not installed.
        ValueError: If the object is neither a Numba dispatcher nor a cfunc.
    """
    try:
        from numba import carray
        from numba import cfunc
        from numba import types
    except ImportError as error:
        raise ImportError(
            "Compiled objectives need Numba; install it with `pip install numba`."
        ) from error
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
