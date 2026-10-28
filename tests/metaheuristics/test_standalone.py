"""Standalone built-in algorithm entry-point and readable adaptation contracts.

Objective:
- Verify that every built-in algorithm runs from the same problem callbacks.

Technical approach:
- Exercise real bounded serial searches through the inherited class method.
- Check mode resolution, conflicting inputs and legacy keyword compatibility.

Flow (pseudocode):
1. Define deterministic identity decoding and a finite sphere objective.
2. Run every registered class and inspect immutable result metadata.
3. Reject unsupported modes and conflicting legacy inputs.

Important criteria:
- Short time budgets test API contracts, not solution quality or speed.
- Existing constructors and their tuple-returning solve methods remain unchanged.

Inputs/Outputs:
- Input: built-in classes, callbacks and bounded settings.
- Output: assertions covering results, validation and configuration overrides.
"""

import ast
import inspect
import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path
from rko_solver import RKO
from rko_solver import metaheuristics
from rko_solver.orchestration.configuration import MH_SPECS
from rko_solver.orchestration.configuration import parse_run_config
from unittest.mock import Mock


def decode(keys: np.ndarray) -> np.ndarray:
    """Return keys as the decoded solution.

    Args:
        keys (np.ndarray): Candidate vector.

    Returns:
        np.ndarray: Unchanged candidate.
    """
    return keys


def objective(solution: np.ndarray) -> float:
    """Compute the finite squared Euclidean norm.

    Args:
        solution (np.ndarray): Numerical candidate.

    Returns:
        float: Squared norm to minimize.
    """
    return float(np.dot(solution, solution))


@pytest.mark.parametrize("name", tuple(MH_SPECS))
@pytest.mark.parametrize("adaptation", ["fixed", "adaptive"])
def test_each_algorithm_runs_alone(name: str, adaptation: str) -> None:
    """Run a registered class without requiring caller-managed pools.

    Args:
        name (str): Canonical algorithm name.
        adaptation (str): Requested readable parameter mode.
    """
    solver_type = getattr(metaheuristics, "GeneticAlgorithm" if name == "GA" else name)
    solver = solver_type(
        decode,
        objective,
        3,
        random_state=42,
        adaptation=adaptation,
        execution="sequential",
    )
    assert isinstance(solver, RKO)
    assert solver_type.solve is RKO.solve
    assert solver.result_ is None
    result = solver.solve(time_limit=0.01)
    assert solver.result_ is result
    assert result.algorithms == (name,)
    assert result.execution == "sequential"
    assert result.adaptation == adaptation
    expected_control = int(adaptation == "adaptive" and MH_SPECS[name].supports_control)
    assert result.searches[0].control == expected_control
    assert result.n_evaluations is not None and result.n_evaluations > 0
    assert len(result.keys) == 3
    assert result.cost == pytest.approx(objective(np.asarray(result.keys)))


@pytest.mark.parametrize("adaptation,control", [("fixed", 0), ("adaptive", 1)])
def test_readable_and_legacy_modes(adaptation: str, control: int) -> None:
    """Keep readable modes and legacy mode codes equivalent.

    Args:
        adaptation (str): Public mode name.
        control (int): Equivalent legacy code.
    """
    result = RKO(
        decode,
        objective,
        3,
        algorithms=["SA"],
        execution="sequential",
        adaptation=adaptation,
        control=control,
    ).solve(0.01)
    assert result.control == control
    assert result.adaptation == adaptation


@pytest.mark.parametrize("adaptation", ["online", "offline", "", 0, [], True])
def test_invalid_adaptation_is_rejected(adaptation: object) -> None:
    """Reject unknown names and non-string adaptation values.

    Args:
        adaptation (object): Invalid public input.
    """
    with pytest.raises(ValueError, match="adaptation"):
        RKO(decode, objective, 3, adaptation=adaptation, execution="sequential").solve(0.01)


def test_conflicting_modes_are_rejected() -> None:
    """Reject contradictory readable and legacy settings without running callbacks."""
    with pytest.raises(ValueError, match="contradicts"):
        RKO(decode, objective, 3, adaptation="fixed", control=1).solve(0.01)


def test_standalone_scalar_overrides() -> None:
    """Merge single-algorithm scalar overrides with packaged fixed defaults."""
    solver = metaheuristics.GeneticAlgorithm(
        decode,
        objective,
        3,
        adaptation="fixed",
        execution="sequential",
        hyperparameters={"GA": {"size_pop": 8, "prob_mut": 0.1}},
    )
    result = solver.solve(time_limit=0.01)
    assert dict(result.searches[0].parameters)["size_pop"] == 8
    assert dict(result.searches[0].parameters)["prob_mut"] == 0.1


def test_public_solver_signatures_and_defaults_match_rko() -> None:
    """Keep every dedicated solver's shared constructor and solve contract aligned with RKO."""
    expected = dict(inspect.signature(RKO).parameters)
    expected.pop("algorithms")
    for name in MH_SPECS:
        solver_type = getattr(metaheuristics, "GeneticAlgorithm" if name == "GA" else name)
        actual = dict(inspect.signature(solver_type).parameters)
        assert tuple(actual) == tuple(expected)
        for parameter, spec in expected.items():
            assert actual[parameter].default == spec.default
            assert actual[parameter].kind == spec.kind
        assert solver_type.solve is RKO.solve
        for key in MH_SPECS[name].int_params + MH_SPECS[name].float_params:
            assert key in solver_type.__doc__


def test_shared_constructor_settings_are_forwarded_without_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Store every shared RKO option without reading files or constructing a search.

    Args:
        monkeypatch (pytest.MonkeyPatch): Prevents configuration I/O during construction.
        tmp_path (Path): Example custom parameter directory.
    """
    reader = Mock(side_effect=AssertionError("constructor must not read configuration"))
    monkeypatch.setattr("rko_solver.solver.load_config", reader)
    options = {
        "adaptation": "fixed",
        "control": 0,
        "pool_size": 5,
        "prefill_pool": False,
        "params_dir": tmp_path,
        "execution": "process",
        "n_jobs": 2,
        "random_state": 42,
        "verbose": True,
        "history_limit": 10,
        "hyperparameters": {"SA": {"sa_max": 100}},
    }
    reference = RKO(decode, objective, 3, algorithms=["SA"], **options)
    solver = metaheuristics.SA(decode, objective, 3, **options)
    for key, value in vars(reference).items():
        if key == "algorithms":
            assert tuple(value) == solver.algorithms
        else:
            assert getattr(solver, key) == value
    reader.assert_not_called()


def test_dedicated_selection_cannot_become_a_portfolio() -> None:
    """Reject changing a dedicated solver's fixed identity; portfolio selection belongs to RKO."""
    solver = metaheuristics.SA(decode, objective, 3, execution="sequential")
    with pytest.raises(ValueError, match="fixed"):
        solver.algorithms = ["PSO"]
    assert solver.algorithms == ("SA",)
    with pytest.raises(TypeError, match="algorithms"):
        metaheuristics.SA(decode, objective, 3, algorithms=["SA"])


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
def test_public_solver_reuses_execution_history_and_failure_contracts(execution: str) -> None:
    """Preserve managed results, history and clearing on failed repeated solves.

    Args:
        execution (str): Shared RKO execution mode.
    """
    solver = metaheuristics.SA(
        decode,
        objective,
        3,
        adaptation="fixed",
        execution=execution,
        history_limit=5,
        random_state=42,
        hyperparameters={"SA": {"sa_max": 100, "t0": 1000.0}},
    )
    result = solver.solve(time_limit=0.1)
    assert result.algorithms == ("SA",)
    assert result.execution == execution
    assert solver.result_ is result
    assert result.searches[0].history
    assert dict(result.searches[0].parameters)["sa_max"] == 100
    assert dict(result.searches[0].parameters)["t0"] == 1000.0
    solver.n_variables = 0
    with pytest.raises(ValueError, match="n_variables"):
        solver.solve(time_limit=0.1)
    assert solver.result_ is None


def test_public_exports_import_in_a_fresh_interpreter() -> None:
    """Exercise public lazy exports and kernel submodule imports without import cycles."""
    command = (
        "from rko_solver import RKO; from rko_solver.metaheuristics import SA, PSO; "
        "from rko_solver.metaheuristics import sa; "
        "assert issubclass(SA, RKO); assert issubclass(PSO, RKO); "
        "assert SA is not sa.SA"
    )
    subprocess.run([sys.executable, "-c", command], check=True, timeout=30)


def test_readable_toml_configuration_and_python_override(tmp_path: Path) -> None:
    """Resolve readable TOML settings and allow explicit Python mode overrides.

    Args:
        tmp_path (Path): Isolated configuration directory.
    """
    raw = {
        "run": {"adaptation": "adaptive", "pool_size": 2},
        "active_metaheuristics": {"list": ["MultiStart"]},
    }
    assert parse_run_config(raw).control == 1
    (tmp_path / "config.toml").write_text(
        '[run]\nadaptation="adaptive"\npool_size=2\n[active_metaheuristics]\nlist=["MultiStart"]\n',
        encoding="utf-8",
    )
    (tmp_path / "multi_start.toml").write_text("[offline]\n[online]\n", encoding="utf-8")
    result = RKO(
        decode, objective, 3, params_dir=tmp_path, execution="sequential", adaptation="fixed"
    ).solve(0.01)
    assert result.adaptation == "fixed"


def test_runtime_modules_have_technical_headers() -> None:
    """Keep algorithm and runtime module explanations complete and discoverable."""
    root = Path(__file__).resolve().parents[2] / "src" / "rko_solver"
    required = (
        "Objective:",
        "Technical approach:",
        "Flow (pseudocode):",
        "Important criteria:",
        "Inputs/Outputs:",
    )
    for path in root.rglob("*.py"):
        if "tests" in path.relative_to(root).parts:
            continue
        header = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
        assert all(section in header for section in required), str(path)


def test_toml_conflicting_adaptation_is_rejected() -> None:
    """Reject mutually contradictory names in a custom configuration table."""
    with pytest.raises(ValueError, match="contradicts"):
        parse_run_config(
            {
                "run": {"adaptation": "adaptive", "control": 0, "pool_size": 2},
                "active_metaheuristics": {"list": ["SA"]},
            }
        )


def test_legacy_control_without_readable_alias_still_works() -> None:
    """Preserve the existing integer-only Python interface and result metadata."""
    result = RKO(decode, objective, 3, algorithms=["SA"], control=0, execution="sequential").solve(
        0.01
    )
    assert result.control == 0
    assert result.adaptation == "fixed"


def test_getting_started_example_executes() -> None:
    """Execute the introductory Python snippet and verify its reported incumbent."""
    root = Path(__file__).resolve().parents[2]
    page = (root / "docs" / "getting-started.md").read_text(encoding="utf-8")
    code = page.split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {"__name__": "__main__"}
    exec(compile(code, "docs/getting-started.md", "exec"), namespace)
    result = namespace["result"]
    assert len(result.keys) == 10
    assert result.cost == pytest.approx(
        namespace["objective"](namespace["decode"](np.asarray(result.keys)))
    )
