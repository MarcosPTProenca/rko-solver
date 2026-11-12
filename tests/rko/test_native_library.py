"""Contract tests for user-supplied native objectives loaded from a shared library.

Objective:
- Verify that a C function `double objective(const double*, long)` compiled by the user can be
  run by the Rust and C++ engines through `NativeObjective`, with the same result as a Python
  recomputation of its cost.

Technical approach:
- Compile a tiny C library inside the test with the system compiler, then solve through the public
  API. Skip when no compiler or compiled extension is available.

Flow (pseudocode):
1. Compile the shared library into a temporary directory.
2. Solve with the Rust engine and recompute the cost from the returned keys.
3. Repeat on the C++ engine when it is built, and check that bad library, symbol or dimension
   arguments fail before any search starts.

Important criteria:
- The C function receives the key vector without copying semantics beyond the call.
- Nothing about the library is cached between tests; each test compiles its own copy.

Inputs/Outputs:
- Input: C source text, library path, symbol name and dimension.
- Output: solver results and ValueError messages.
"""

from __future__ import annotations

import numpy as np
import pytest
import shutil
import subprocess
from pathlib import Path

native = pytest.importorskip("rko_solver._native")

C_SOURCE = """
double objective(const double *keys, long n) {
    double total = 0.0;
    for (long i = 0; i < n; i++) {
        double d = keys[i] - 0.3;
        total += d * d;
    }
    return total;
}
"""

needs_cc = pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler available")
needs_cpp = pytest.mark.skipif(not native.cpp_available(), reason="built without the C++ engine")


def build_library(directory: Path, source: str = C_SOURCE) -> Path:
    """Compile a shared library exposing `objective`.

    Args:
        directory (Path): Destination directory for the source and the library.
        source (str): C source text to compile.

    Returns:
        Path: Path of the built shared library.
    """
    source_path = directory / "objective.c"
    source_path.write_text(source, encoding="utf-8")
    library = directory / "libobjective.so"
    subprocess.run(
        ["cc", "-shared", "-fPIC", "-O2", str(source_path), "-o", str(library)],
        check=True,
        capture_output=True,
    )
    return library


def python_cost(keys: object) -> float:
    """Recompute the C objective in Python for a returned key vector.

    Args:
        keys (object): Key vector returned by the solver.

    Returns:
        float: Sum of squared distances from 0.3.
    """
    values = np.asarray(keys, dtype=float)
    return float(np.sum((values - 0.3) ** 2))


@needs_cc
def test_rust_engine_runs_the_user_library_and_matches_python(tmp_path: Path) -> None:
    """A user C objective runs natively and its cost equals the Python recomputation."""
    from rko_solver import RKO
    from rko_solver import NativeObjective

    library = build_library(tmp_path)
    result = RKO(
        NativeObjective(library, "objective", 6),
        engine="rust",
        n_jobs=1,
        algorithms=["SA"],
        max_evaluations=300,
        random_state=3,
    ).solve(30.0)
    assert result.engine == "rust"
    assert result.cost == pytest.approx(python_cost(result.keys))
    assert result.searches[0].n_evaluations == 300


@needs_cc
@needs_cpp
def test_cpp_engine_runs_the_user_library_and_matches_python(tmp_path: Path) -> None:
    """The same library runs on the C++ reference engine with the same cost check."""
    from rko_solver import RKO
    from rko_solver import NativeObjective

    library = build_library(tmp_path)
    result = RKO(
        NativeObjective(library, "objective", 6),
        engine="cpp",
        algorithms=["SA"],
        max_evaluations=300,
        random_state=3,
    ).solve(30.0)
    assert result.engine == "cpp"
    assert result.cost == pytest.approx(python_cost(result.keys))


@needs_cc
def test_missing_library_is_rejected_before_any_search(tmp_path: Path) -> None:
    """A path that is not a shared library fails at construction with the library in the message."""
    from rko_solver import NativeObjective

    with pytest.raises(ValueError, match="library"):
        NativeObjective(tmp_path / "absent.so", "objective", 6)


@needs_cc
def test_missing_symbol_is_rejected_before_any_search(tmp_path: Path) -> None:
    """A symbol the library does not export fails with the symbol in the message."""
    from rko_solver import NativeObjective

    library = build_library(tmp_path)
    with pytest.raises(ValueError, match="symbol"):
        NativeObjective(library, "not_there", 6)


@needs_cc
@pytest.mark.parametrize("dimension", [0, -1, 2.5, True])
def test_invalid_dimension_is_rejected(tmp_path: Path, dimension: object) -> None:
    """Only a positive integer dimension is accepted, as for the other problem types.

    Args:
        tmp_path (Path): Directory holding the compiled library.
        dimension (object): Rejected key dimension.
    """
    from rko_solver import NativeObjective

    library = build_library(tmp_path)
    with pytest.raises(ValueError, match="n_variables"):
        NativeObjective(library, "objective", dimension)
