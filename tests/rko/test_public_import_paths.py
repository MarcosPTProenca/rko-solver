"""Canonical imports and solver execution in an isolated interpreter.

Objective:
- Verify maintained engine imports and reject interpreter-lock fallback during installation.

Technical approach:
- Import the public package and Rust adapters with warnings promoted to errors.

Flow (pseudocode):
1. Start an isolated interpreter using the installed editable distribution.
2. Import canonical engine modules and run the public solver.
3. Check successful execution and Python engine metadata.
4. Check that free-threaded builds reject an enabled global interpreter lock.

Important criteria:
- Import failures and unexpected warnings fail the subprocess.

Inputs/Outputs:
- Input: installed package and canonical engine module paths.
- Output: successful subprocess exit with verified result metadata.
"""

import pytest
import subprocess
import sys
import sysconfig
from tools import smoke_installed
from unittest.mock import Mock


@pytest.mark.parametrize("free_threaded", [False, True])
@pytest.mark.parametrize("lock_enabled", [False, True])
def test_installed_smoke_rejects_threading_fallback(
    monkeypatch: pytest.MonkeyPatch, free_threaded: bool, lock_enabled: bool
) -> None:
    """Fail installation verification if a free-threaded build enables the interpreter lock.

    Args:
        monkeypatch (pytest.MonkeyPatch): Restore interpreter probes after this unit test.
        free_threaded (bool): Whether this interpreter was built for free threading.
        lock_enabled (bool): Simulated current interpreter-lock state.
    """
    monkeypatch.setattr(sysconfig, "get_config_var", Mock(return_value=int(free_threaded)))
    monkeypatch.setattr(sys, "_is_gil_enabled", Mock(return_value=lock_enabled))
    if free_threaded and lock_enabled:
        with pytest.raises(RuntimeError, match="Free-threaded installation enabled"):
            smoke_installed._check_free_threading()
    else:
        smoke_installed._check_free_threading()


def test_canonical_imports_and_solve_are_warning_free() -> None:
    """Import canonical engine modules and solve without deprecation warnings."""
    code = (
        "import warnings; warnings.simplefilter('error', DeprecationWarning); "
        "import rko_solver, rko_solver.engines.rust.batch, rko_solver.engines.rust.compiled; "
        "import rko_solver.engines.rust.runner; "
        "r = rko_solver.RKO(rko_solver.Callbacks(lambda k: k, lambda v: float(sum(v)), 3), "
        "algorithms=['SA'], execution='sequential', random_state=1).solve(0.05); "
        "assert r.engine == 'python'"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
