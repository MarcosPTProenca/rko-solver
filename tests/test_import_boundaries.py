"""Executable regression checks for the library's static import boundaries.

Objective:
- Reject forbidden dependency directions without importing optional features or native engines.

Technical approach:
- Run the installed Import Linter against small source trees using the repository's real policy.
- Inject representative direct, indirect and type-checking-only imports and inspect its exit code.

Flow (pseudocode):
1. Build a minimal package with the same architectural areas as the production package.
2. Insert an allowed or forbidden import and run the configured contract in a fresh process.
3. Require a kept contract for allowed dependencies and a broken contract for forbidden ones.

Important criteria:
- Use the actual project configuration, not a separately maintained test policy.
- Disable caches and select the temporary source tree through the child process import path.

Inputs/Outputs:
- Input: temporary source modules, imported modules and contract identifiers.
- Output: assertions on Import Linter's reports and process exit codes.
"""

import os
import pytest
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ImportBoundaryProbe:
    """Exercise the real dependency policy against a small isolated source tree.

    Responsibilities:
    - Create representative package areas and execute the installed architectural checker.

    Attributes:
        root (Path): Temporary repository root containing a source-layout package.
    """

    def __init__(self, root: Path) -> None:
        """Create the package areas referenced by the production contracts.

        Args:
            root (Path): Empty temporary repository directory.

        Raises:
            OSError: If the temporary source files cannot be created.
        """
        self.root = root
        for module in (
            "__init__",
            "solver",
            "new_pipeline",
            "components/__init__",
            "components/operators",
            "parameters/__init__",
            "engines/__init__",
            "metaheuristics/__init__",
            "modeling/__init__",
            "modeling/linear",
            "service/__init__",
            "service/api",
            "experimental/__init__",
            "experimental/self_evolve",
        ):
            self.write(module, "")

    def write(self, module: str, source: str) -> None:
        """Write one module beneath the temporary library package.

        Args:
            module (str): Slash-separated module path without its Python suffix.
            source (str): Python source parsed by the checker, never executed by this helper.

        Raises:
            OSError: If the source directory or module cannot be written.
        """
        path = self.root / "src" / "rko_solver" / f"{module}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    def run(self, contract: str) -> subprocess.CompletedProcess[str]:
        """Check a contract against the temporary package with the project's configuration.

        Args:
            contract (str): Identifier from the repository's Import Linter configuration.

        Returns:
            subprocess.CompletedProcess[str]: Exit code and captured checker output.

        Raises:
            OSError: If the interpreter cannot be launched.
            subprocess.TimeoutExpired: If the isolated checker exceeds thirty seconds.
        """
        return subprocess.run(
            [
                sys.executable,
                "-c",
                "from importlinter.cli import lint_imports_command; lint_imports_command()",
                "--config",
                str(ROOT / "pyproject.toml"),
                "--contract",
                contract,
                "--no-cache",
                "--no-logo",
            ],
            cwd=self.root,
            env={**os.environ, "PYTHONPATH": str(self.root / "src"), "NO_COLOR": "1"},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )


@pytest.fixture
def probe(tmp_path: Path) -> ImportBoundaryProbe:
    """Prepare a source tree isolated from the real installed package.

    Args:
        tmp_path (Path): Test-owned temporary repository root.

    Returns:
        ImportBoundaryProbe: Helper for injecting imports and checking contracts.
    """
    return ImportBoundaryProbe(tmp_path)


@pytest.mark.parametrize("target", ["benchmarks", "examples", "tests", "tools"])
def test_library_cannot_import_repository_support(probe: ImportBoundaryProbe, target: str) -> None:
    """Reject runtime imports of repository-only packages from any library area.

    Args:
        probe (ImportBoundaryProbe): Isolated library source tree.
        target (str): Repository-only package to inject into the service module.
    """
    probe.write("service/api", f"import {target}\n")
    result = probe.run("repository-isolation")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "BROKEN" in result.stdout, result.stdout + result.stderr
    assert target in result.stdout


@pytest.mark.parametrize(
    "module", ["components/operators", "parameters/__init__", "engines/__init__", "new_pipeline"]
)
@pytest.mark.parametrize("target", ["service.api", "experimental.self_evolve"])
def test_runtime_cannot_import_deployment_or_research(
    probe: ImportBoundaryProbe, module: str, target: str
) -> None:
    """Protect existing descendants and newly added runtime modules from optional dependencies.

    Args:
        probe (ImportBoundaryProbe): Isolated library source tree.
        module (str): Source module under an existing or newly introduced runtime area.
        target (str): Deployment or research module forbidden to the runtime.
    """
    probe.write(module, f"import rko_solver.{target}\n")
    result = probe.run("runtime-isolation")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "BROKEN" in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize("guarded", [False, True])
def test_runtime_rejects_indirect_and_type_only_dependencies(
    probe: ImportBoundaryProbe, guarded: bool
) -> None:
    """Reject an indirect path and a forbidden import hidden behind a type-checking guard.

    Args:
        probe (ImportBoundaryProbe): Isolated library source tree.
        guarded (bool): Put the forbidden import inside a TYPE_CHECKING block when true.
    """
    source = "import rko_solver.experimental.self_evolve\n"
    if guarded:
        source = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    " + source
    probe.write("new_pipeline", source)
    probe.write("components/operators", "import rko_solver.new_pipeline\n")
    result = probe.run("runtime-isolation")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "BROKEN" in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize("target", ["service.api", "experimental.self_evolve"])
def test_public_exports_cannot_load_deployment_or_research(
    probe: ImportBoundaryProbe, target: str
) -> None:
    """Protect the public package initializer, which is not matched by child-module patterns.

    Args:
        probe (ImportBoundaryProbe): Isolated library source tree.
        target (str): Deployment or research module forbidden to the public facade.
    """
    probe.write("__init__", f"import rko_solver.{target}\n")
    result = probe.run("public-facade-isolation")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "BROKEN" in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize(
    "module",
    ["components/operators", "parameters/__init__", "engines/__init__", "metaheuristics/__init__"],
)
def test_numerical_implementation_cannot_import_modeling(
    probe: ImportBoundaryProbe, module: str
) -> None:
    """Keep model-specific imports at the existing solve and orchestration integration points.

    Args:
        probe (ImportBoundaryProbe): Isolated library source tree.
        module (str): Numerical implementation package receiving a forbidden model import.
    """
    probe.write(module, "import rko_solver.modeling.linear\n")
    result = probe.run("numerical-modeling-boundary")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "BROKEN" in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize(
    "contract",
    [
        "repository-isolation",
        "runtime-isolation",
        "public-facade-isolation",
        "numerical-modeling-boundary",
    ],
)
def test_supported_integration_directions_remain_allowed(
    probe: ImportBoundaryProbe, contract: str
) -> None:
    """Allow model conversion, single-algorithm facades and deployment/research consumers.

    Args:
        probe (ImportBoundaryProbe): Isolated library source tree.
        contract (str): Boundary that must permit the currently supported integrations.
    """
    probe.write("__init__", "from rko_solver.solver import Solver\n")
    probe.write("solver", "import rko_solver.modeling.linear\n")
    probe.write("metaheuristics/__init__", "import rko_solver.solver\n")
    probe.write("service/__init__", "import rko_solver.service.api\n")
    probe.write("service/api", "import rko_solver.solver\n")
    probe.write("experimental/__init__", "import rko_solver.experimental.self_evolve\n")
    probe.write("experimental/self_evolve", "import rko_solver.modeling.linear\n")
    result = probe.run(contract)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 kept, 0 broken" in result.stdout, result.stdout + result.stderr
