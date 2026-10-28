"""Contract tests for the import paths that moved under `rko_solver.engines.rust`.

Objective:
- Verify that the pre-0.3.0a1 module paths still resolve to the same objects with a
  deprecation warning until 0.4, and that the package itself no longer imports them.

Technical approach:
- Import each deprecated module freshly under `pytest.warns` and compare identities with the
  new locations; import the package in a subprocess with warnings turned into errors.

Flow (pseudocode):
1. Drop the deprecated module from `sys.modules` and import it again.
2. Compare each re-exported name with the object at its new path.
3. Import `rko` and the engine modules with deprecation warnings as errors.

Important criteria:
- A fresh interpreter importing `rko` never triggers a deprecated path.

Inputs/Outputs:
- Input: module names and their new locations.
- Output: identity and warning assertions.
"""

import importlib
import pytest
import subprocess
import sys
from rko_solver.engines.rust import batch
from rko_solver.engines.rust import compiled
from rko_solver.engines.rust import runner


@pytest.mark.parametrize(
    "legacy,current,names",
    [
        (
            "rko_solver.batch",
            batch,
            ["BatchFn", "BatchObjective", "MAX_BATCH_SIZE", "identity_decoder"],
        ),
        ("rko_solver.compiled", compiled, ["CompiledObjective", "compile_objective"]),
        ("rko_solver.orchestration.native", runner, ["NativeRunner"]),
    ],
)
def test_legacy_module_reexports_the_moved_objects(
    legacy: str, current: object, names: list[str]
) -> None:
    """Each deprecated module warns once on import and re-exports the moved names.

    Args:
        legacy (str): Deprecated module path.
        current (object): Module that now owns the names.
        names (list[str]): Names the deprecated module must still export.
    """
    sys.modules.pop(legacy, None)
    with pytest.warns(DeprecationWarning, match="deprecated and will be removed in 0.4"):
        module = importlib.import_module(legacy)
    for name in names:
        assert getattr(module, name) is getattr(current, name)
    assert sorted(module.__all__) == sorted(names)


def test_package_does_not_import_deprecated_paths() -> None:
    """Importing `rko`, the engine adapters and a solve in a fresh interpreter is warning free."""
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
