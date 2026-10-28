"""Contract tests for engine and execution selection and the deprecated backend names.

Objective:
- Verify that `engine` and `execution` are validated as orthogonal choices, that problem
  objects and the short constructor form are equivalent, and that every deprecated spelling
  still works with a warning until 0.4.

Technical approach:
- Test the pure translation table, then run short real solves on a small Knapsack through
  the Python and Rust engines and through each deprecated alias.

Flow (pseudocode):
1. Check name validation and the legacy translation.
2. Check constructor forms, problem objects and the errors they raise.
3. Check each deprecated alias warns and reproduces the historical behavior.

Important criteria:
- Tests use sub-second budgets and never assert timing-dependent quality.

Inputs/Outputs:
- Input: small synthetic problems and short time limits.
- Output: exact assertions on names, warnings and result metadata.
"""

import importlib
import numpy as np
import pytest
import sys
from rko_solver import RKO
from rko_solver import Blocks
from rko_solver import Callbacks
from rko_solver import Numba
from rko_solver import problems
from rko_solver import run
from rko_solver.metaheuristics import SA
from rko_solver.orchestration.engine import EngineSelection

native = pytest.importorskip("rko_solver._native")
PROFITS = [10 + (index * 37) % 91 for index in range(30)]
WEIGHTS = [5 + (index * 53) % 47 for index in range(30)]
CAPACITY = 400


def decode(keys: np.ndarray) -> np.ndarray:
    """Select items whose key exceeds one half.

    Args:
        keys (np.ndarray): Random keys.

    Returns:
        np.ndarray: Boolean selection.
    """
    return np.asarray(keys) > 0.5


def cost(selection: np.ndarray) -> float:
    """Score a selection with a penalized negative profit.

    Args:
        selection (np.ndarray): Boolean selection.

    Returns:
        float: Cost to minimize.
    """
    chosen = np.asarray(selection)
    profit = int(np.dot(chosen, PROFITS))
    weight = int(np.dot(chosen, WEIGHTS))
    return float(100000 * max(0, weight - CAPACITY) - profit)


def block_cost(keys: np.ndarray) -> np.ndarray:
    """Score a block of key vectors with the same objective as `cost`.

    Args:
        keys (np.ndarray): Matrix with one key vector per row.

    Returns:
        np.ndarray: One cost per row.
    """
    return np.array([cost(decode(row)) for row in keys])


@pytest.mark.parametrize(
    "backend,expected",
    [
        ("process", ("python", "process")),
        ("thread", ("python", "thread")),
        ("serial", ("python", "sequential")),
        ("native", ("rust", None)),
        ("cpp", ("cpp", None)),
        ("auto", ("auto", None)),
    ],
)
def test_legacy_backend_names_translate_to_engine_and_execution(
    backend: str, expected: tuple[str, str | None]
) -> None:
    """Every deprecated backend maps to one pair and back to its old name.

    Args:
        backend (str): Deprecated name.
        expected (tuple[str, str | None]): Engine and execution it stands for.
    """
    assert EngineSelection.from_legacy(backend) == expected
    if backend != "auto":
        assert EngineSelection.legacy_name(*expected) == backend


def test_validation_resolves_the_default_execution_and_rejects_mixtures() -> None:
    """Only the Python engine has an execution and it defaults to process."""
    assert EngineSelection.validate("python", None) == "process"
    assert EngineSelection.validate("python", "sequential") == "sequential"
    assert EngineSelection.validate("rust", None) is None
    for engine in ("rust", "cpp", "auto"):
        with pytest.raises(ValueError, match="applies only to engine='python'"):
            EngineSelection.validate(engine, "thread")
    with pytest.raises(ValueError, match="'engine'"):
        EngineSelection.validate("native", None)
    with pytest.raises(ValueError, match="'execution'"):
        EngineSelection.validate("python", "serial")
    with pytest.raises(ValueError, match="'backend' must be"):
        EngineSelection.from_legacy("automatic")


def test_solve_rejects_invalid_engine_settings_without_running() -> None:
    """Invalid names, mixtures and worker limits fail at solve time with clear text."""
    with pytest.raises(ValueError, match="'engine'"):
        RKO(decode, cost, 30, engine="native").solve(0.1)
    with pytest.raises(ValueError, match="applies only to engine='python'"):
        RKO(decode, cost, 30, engine="rust", execution="process").solve(0.1)
    with pytest.raises(ValueError, match="n_jobs"):
        RKO(decode, cost, 30, execution="sequential", n_jobs=2).solve(0.1)
    with pytest.raises(ValueError, match="n_jobs"):
        RKO(decode, cost, 30, engine="cpp", n_jobs=2).solve(0.1)


def test_python_engine_defaults_to_process_and_reports_both_names() -> None:
    """The default solve reports engine python with execution process."""
    result = RKO(decode, cost, 30, algorithms=["SA"], n_jobs=1, random_state=1).solve(0.3)
    assert (result.engine, result.execution) == ("python", "process")
    sequential = RKO(
        decode, cost, 30, algorithms=["SA"], execution="sequential", random_state=1
    ).solve(0.1)
    assert (sequential.engine, sequential.execution) == ("python", "sequential")
    assert sequential.n_workers == 1


def test_rust_engine_reports_no_execution() -> None:
    """Compiled engines have no Python scheduling to report."""
    result = RKO(decode, cost, 30, algorithms=["SA"], engine="rust", random_state=1).solve(0.1)
    assert (result.engine, result.execution) == ("rust", None)


def test_short_form_and_problem_objects_are_equivalent() -> None:
    """`RKO(decoder, cost, n)`, the keyword form and `Callbacks` build the same problem."""
    positional = RKO(decode, cost, 30)
    keywords = RKO(decoder=decode, cost=cost, n_variables=30)
    problem = RKO(Callbacks(decode, cost, 30))
    for solver in (positional, keywords, problem):
        assert (solver.decoder, solver.cost, solver.n_variables) == (decode, cost, 30)
        assert solver.problem == Callbacks(decode, cost, 30)
    standalone = SA(Callbacks(decode, cost, 30), execution="sequential", random_state=1)
    assert standalone.solve(0.1).algorithms == ("SA",)
    assert SA(decoder=decode, cost=cost, n_variables=30).n_variables == 30


def test_constructor_rejects_ambiguous_problem_arguments() -> None:
    """A problem object owns its callbacks and the short form needs all three values."""
    with pytest.raises(TypeError, match="belong to the problem"):
        RKO(Callbacks(decode, cost, 30), cost)
    with pytest.raises(TypeError, match="belong to the problem"):
        RKO(Blocks(block_cost, 30), n_variables=30)
    with pytest.raises(TypeError, match="either 'problem' or 'decoder'"):
        RKO(Callbacks(decode, cost, 30), decoder=decode)
    with pytest.raises(TypeError, match="needs a cost"):
        RKO(decode)


def test_builtin_and_numba_problems_need_a_compiled_engine() -> None:
    """The Python engine explains which engines can run a compiled problem."""
    problem = problems.Knapsack(PROFITS, WEIGHTS, CAPACITY)
    with pytest.raises(ValueError, match="engine='rust', 'cpp' or 'auto'"):
        RKO(problem).solve(0.1)
    result = RKO(problem, engine="rust", algorithms=["SA"], random_state=1).solve(0.1)
    assert result.engine == "rust"
    assert problem.evaluate(list(result.keys)) == result.cost
    numba = pytest.importorskip("numba")
    function = numba.njit(lambda keys: float(np.sum(keys * keys)))
    with pytest.raises(ValueError, match="compiled engines"):
        RKO(Numba(function, 8), algorithms=["SA"]).solve(0.1)
    compiled = RKO(Numba(function, 8), engine="rust", algorithms=["SA"], random_state=1)
    assert compiled.solve(0.1).engine == "rust"


def test_blocks_run_on_python_and_rust_engines() -> None:
    """A block objective works everywhere; the Python engine scores one row per call."""
    problem = Blocks(block_cost, 30, batch_size=16)
    sequential = RKO(problem, algorithms=["SA"], execution="sequential", random_state=1).solve(0.1)
    assert (sequential.engine, sequential.execution) == ("python", "sequential")
    assert cost(decode(np.asarray(sequential.keys))) == sequential.cost
    rust = RKO(problem, engine="rust", algorithms=["SA"], random_state=1).solve(0.1)
    assert rust.engine == "rust" and cost(decode(np.asarray(rust.keys))) == rust.cost


def test_backend_argument_is_deprecated_but_equivalent() -> None:
    """`backend=` warns, translates and cannot be mixed with the new names."""
    with pytest.warns(DeprecationWarning, match="engine='python', execution='sequential'"):
        solver = RKO(decode, cost, 30, backend="serial")
    assert (solver.engine, solver.execution) == ("python", "sequential")
    with pytest.warns(DeprecationWarning, match="engine='rust'"):
        assert RKO(decode, cost, 30, backend="native").engine == "rust"
    with pytest.raises(ValueError, match="cannot be combined"):
        RKO(decode, cost, 30, backend="serial", engine="python")
    with pytest.raises(ValueError, match="cannot be combined"):
        RKO(decode, cost, 30, backend="serial", execution="sequential")
    with pytest.raises(ValueError, match="'backend' must be"):
        RKO(decode, cost, 30, backend="automatic")
    with pytest.warns(DeprecationWarning):
        standalone = SA(decode, cost, 30, backend="thread")
    assert (standalone.engine, standalone.execution) == ("python", "thread")


def test_deprecated_backend_properties_report_the_historical_names() -> None:
    """`RKO.backend` and `RKOResult.backend` keep their old values with a warning."""
    solver = RKO(decode, cost, 30, algorithms=["SA"], execution="sequential", random_state=1)
    with pytest.warns(DeprecationWarning, match="RKO.backend"):
        assert solver.backend == "serial"
    result = solver.solve(0.1)
    with pytest.warns(DeprecationWarning, match="RKOResult.backend"):
        assert result.backend == "serial"
    rust = RKO(decode, cost, 30, algorithms=["SA"], engine="rust", random_state=1).solve(0.1)
    with pytest.warns(DeprecationWarning):
        assert rust.backend == "native"
    with pytest.warns(DeprecationWarning):
        assert RKO(decode, cost, 30, engine="auto").backend == "auto"


def test_alternative_constructors_are_deprecated_and_default_to_rust() -> None:
    """`RKO.native`, `RKO.compiled` and `RKO.batched` warn and still run on Rust."""
    problem = problems.Knapsack(PROFITS, WEIGHTS, CAPACITY)
    with pytest.warns(DeprecationWarning, match="RKO.native"):
        native_solver = RKO.native(problem, algorithms=["SA"], random_state=1)
    assert native_solver.solve(0.1).engine == "rust"
    with pytest.warns(DeprecationWarning, match="RKO.batched"):
        batched = RKO.batched(block_cost, 30, batch_size=16, algorithms=["SA"], random_state=1)
    assert isinstance(batched.problem, Blocks) and batched.problem.objective.batch_size == 16
    assert batched.solve(0.1).engine == "rust"
    with pytest.warns(DeprecationWarning):
        sequential = RKO.batched(
            block_cost, 30, algorithms=["SA"], backend="serial", random_state=1
        )
    assert (sequential.engine, sequential.execution) == ("python", "sequential")
    numba = pytest.importorskip("numba")
    with pytest.warns(DeprecationWarning, match="RKO.compiled"):
        compiled = RKO.compiled(
            numba.njit(lambda keys: float(np.sum(keys))), 8, algorithms=["SA"], engine="cpp"
        )
    assert compiled.engine == "cpp"


def test_native_module_is_a_deprecated_alias_of_problems() -> None:
    """`rko_solver.native.Knapsack` is the same class as `rko_solver.problems.Knapsack`."""
    sys.modules.pop("rko_solver.native", None)
    with pytest.warns(DeprecationWarning, match="rko_solver.problems"):
        module = importlib.import_module("rko_solver.native")
    assert module.Knapsack is problems.Knapsack
    with pytest.raises(AttributeError):
        _ = module.Missing
    with pytest.raises(AttributeError):
        _ = problems.Missing


def test_legacy_run_adapter_accepts_both_spellings() -> None:
    """`rko_solver.run` forwards `execution` and still honors the deprecated `backend`."""
    keys, value = run(decode, cost, 30, 0.1, execution="sequential", random_state=1)
    assert len(keys) == 30 and value == cost(decode(np.asarray(keys)))
    with pytest.warns(DeprecationWarning, match="engine='python', execution='sequential'"):
        keys, value = run(decode, cost, 30, 0.1, backend="serial", random_state=1)
    assert value == cost(decode(np.asarray(keys)))
