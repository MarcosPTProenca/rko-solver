"""Contract tests for canonical engine and execution selection.

Objective:
- Verify orthogonal engine/execution choices and equivalent canonical problem constructors.

Technical approach:
- Validate names, then run short real solves through the Python and Rust engines.

Flow (pseudocode):
1. Check independent engine and execution validation.
2. Check canonical constructor forms and invalid combinations.
3. Check public results through real solves and the tuple-returning adapter.

Important criteria:
- Tests use sub-second budgets and never assert timing-dependent quality.

Inputs/Outputs:
- Input: small synthetic problems and short time limits.
- Output: exact assertions on names, errors and result metadata.
"""

import numpy as np
import pytest
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


def test_run_adapter_preserves_canonical_execution() -> None:
    """Forward canonical execution and retain independently verified tuple results."""
    keys, value = run(decode, cost, 30, 0.1, execution="sequential", random_state=1)
    assert len(keys) == 30 and value == cost(decode(np.asarray(keys)))
