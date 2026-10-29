"""Removed compatibility surface contract.

Objective:
- Reject deprecated solver names without a compatibility translation layer.

Technical approach:
- Exercise public constructors, result attributes and imports independently of native engines.

Flow (pseudocode):
1. Supply the removed keyword to each public construction entry point.
2. Check removed attributes on ordinary solver and result objects.
3. Import each removed module and verify the exact missing module name.

Important criteria:
- Canonical problem objects and engine/execution settings remain available.
- Missing optional dependencies must not masquerade as removed-module failures.

Inputs/Outputs:
- Input: deprecated keywords, attributes and import paths.
- Output: exact rejection assertions without timed optimization.
"""

import importlib
import numpy as np
import pytest
import sys
from rko_solver import RKO
from rko_solver import Callbacks
from rko_solver import RKOResult
from rko_solver import run
from rko_solver.metaheuristics import SA


def decode(keys: np.ndarray) -> np.ndarray:
    """Return unchanged keys for a minimal public callback problem.

    Args:
        keys (np.ndarray): Candidate vector.

    Returns:
        np.ndarray: Unchanged candidate vector.
    """
    return keys


def cost(keys: np.ndarray) -> float:
    """Sum the candidate coordinates.

    Args:
        keys (np.ndarray): Candidate vector.

    Returns:
        float: Coordinate sum to minimize.
    """
    return float(np.sum(keys))


@pytest.mark.parametrize("entry", ["RKO", "SA", "run"])
@pytest.mark.parametrize("backend", [None, "serial", "native", "cpp", "automatic"])
def test_removed_backend_keyword_is_rejected(entry: str, backend: str | None) -> None:
    """Reject the removed keyword at every public callback entry point.

    Args:
        entry (str): Public construction entry point.
        backend (str | None): Historical combined engine/execution setting.
    """
    with pytest.raises(TypeError, match="unexpected keyword argument 'backend'"):
        if entry == "run":
            run(decode, cost, 3, 0.1, backend=backend)
        else:
            constructor = RKO if entry == "RKO" else SA
            constructor(Callbacks(decode, cost, 3), backend=backend)


@pytest.mark.parametrize("name", ["native", "compiled", "batched", "backend"])
def test_removed_solver_attributes_are_absent(name: str) -> None:
    """Keep only canonical constructors and engine/execution metadata.

    Args:
        name (str): Removed solver method or property.
    """
    solver = RKO(Callbacks(decode, cost, 3))
    with pytest.raises(AttributeError):
        getattr(solver, name)
    with pytest.raises(AttributeError):
        getattr(SA(Callbacks(decode, cost, 3)), name)


def test_result_has_only_engine_and_execution_metadata() -> None:
    """Reject the removed combined property on portable results."""
    result = RKOResult((0.0,), 0.0, 0.1, "python", ("SA",), 1, execution="sequential")
    assert (result.engine, result.execution) == ("python", "sequential")
    with pytest.raises(AttributeError):
        _ = result.backend


@pytest.mark.parametrize(
    "name",
    [
        "rko_solver.batch",
        "rko_solver.compiled",
        "rko_solver.native",
        "rko_solver.orchestration.native",
    ],
)
def test_removed_import_paths_are_unavailable(name: str) -> None:
    """Reject obsolete imports rather than silently loading compatibility shims.

    Args:
        name (str): Removed package module.
    """
    sys.modules.pop(name, None)
    with pytest.raises(ModuleNotFoundError) as error:
        importlib.import_module(name)
    assert error.value.name == name
