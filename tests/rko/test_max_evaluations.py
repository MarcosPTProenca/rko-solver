"""Contract tests for the per-algorithm evaluation limit in every engine.

Objective:
- Verify that `max_evaluations` stops each algorithm after exactly that many completed
  evaluations on the Python, Rust and C++ engines, reports it uniformly and is reproducible
  where no concurrent pool exchange exists.

Technical approach:
- Run the full default portfolio on a small Knapsack with a generous time limit, so only the
  evaluation limit can end a search, and compare the per-search counters and stop reasons.
- Repeat seeded runs to separate the reproducible configurations from the concurrent ones.

Flow (pseudocode):
1. Validate the argument and its interaction with the time limit and the pool prefill.
2. Solve with every engine and execution mode and inspect each search report.
3. Solve twice with the same seed and compare keys, costs and counters.

Important criteria:
- The count is exact for scalar callbacks, block objectives and built-in problems.
- Concurrent threads and the C++ reference are not claimed to be reproducible.

Inputs/Outputs:
- Input: a 60-item Knapsack and limits of a few hundred evaluations.
- Output: exact assertions on counters, stop reasons and reproducibility.
"""

import numpy as np
import pytest
from rko_solver import RKO
from rko_solver import Blocks
from rko_solver import problems
from rko_solver.metaheuristics import SA
from rko_solver.reporting import OptimizationReport

native = pytest.importorskip("rko_solver._native")
PROFITS = [10 + (index * 37) % 91 for index in range(60)]
WEIGHTS = [5 + (index * 53) % 47 for index in range(60)]
CAPACITY = 700
needs_cpp = pytest.mark.skipif(not native.cpp_available(), reason="built without the C++ engine")
CONFIGURATIONS = [
    pytest.param({"engine": "python", "execution": "sequential"}, id="python-sequential"),
    pytest.param({"engine": "python", "execution": "thread"}, id="python-thread"),
    pytest.param({"engine": "python", "execution": "process", "n_jobs": 2}, id="python-process"),
    pytest.param({"engine": "rust"}, id="rust"),
    pytest.param({"engine": "cpp"}, id="cpp", marks=needs_cpp),
]


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


def fingerprint(result: object) -> tuple[object, ...]:
    """Reduce a result to the values that must repeat in a reproducible run.

    Args:
        result (object): An `RKOResult`.

    Returns:
        tuple[object, ...]: Objective, keys and per-search counters.
    """
    return (
        result.cost,
        result.keys,
        tuple((item.algorithm, item.n_evaluations, item.stop_reason) for item in result.searches),
    )


@pytest.mark.parametrize("settings", CONFIGURATIONS)
def test_every_algorithm_stops_after_exactly_the_limit(settings: dict[str, object]) -> None:
    """The default portfolio ends each search at the limit long before the time limit.

    Args:
        settings (dict[str, object]): Engine, execution and worker options under test.
    """
    result = RKO(decode, cost, 60, max_evaluations=300, random_state=3, **settings).solve(60.0)
    assert len(result.searches) == 10
    assert {item.n_evaluations for item in result.searches} == {300}
    assert {item.stop_reason for item in result.searches} == {"max_evaluations"}
    assert result.stop_reason == "max_evaluations" and result.max_evaluations == 300
    startup = result.initialization.n_evaluations if result.initialization else 0
    assert result.n_evaluations == 3000 + startup
    assert startup == (10 if settings["engine"] == "cpp" else 0)
    assert result.elapsed_seconds < 30
    assert cost(decode(np.asarray(result.keys))) == result.cost


@pytest.mark.parametrize("engine", ["rust", pytest.param("cpp", marks=needs_cpp)])
def test_builtin_problem_counts_exactly_on_the_compiled_engines(engine: str) -> None:
    """A problem evaluated without the interpreter obeys the same limit.

    Args:
        engine (str): Compiled engine under test.
    """
    problem = problems.Knapsack(PROFITS, WEIGHTS, CAPACITY)
    result = RKO(problem, engine=engine, max_evaluations=500, random_state=2).solve(60.0)
    assert {item.n_evaluations for item in result.searches} == {500}
    assert problem.evaluate(list(result.keys)) == result.cost


def test_block_objective_is_truncated_at_the_limit_instead_of_overshooting() -> None:
    """Blocks larger than the remaining budget score only the rows that fit."""
    problem = Blocks(block_cost, 60, batch_size=16)
    rust = RKO(problem, engine="rust", max_evaluations=250, random_state=2).solve(60.0)
    assert {item.n_evaluations for item in rust.searches} == {250}
    python = RKO(
        problem, engine="python", execution="sequential", max_evaluations=250, random_state=2
    ).solve(60.0)
    assert {item.n_evaluations for item in python.searches} == {250}


def test_the_time_limit_still_wins_when_it_comes_first() -> None:
    """A huge evaluation limit never extends the search beyond its time budget."""
    result = RKO(
        decode, cost, 60, algorithms=["SA"], execution="sequential", max_evaluations=10**9
    ).solve(0.2)
    assert result.searches[0].stop_reason == "time_limit"
    assert result.stop_reason == "time_limit"
    assert result.searches[0].n_evaluations < 10**9


@pytest.mark.parametrize("value", [0, -1, 1.5, "10", True])
def test_invalid_limits_are_rejected(value: object) -> None:
    """Only positive integers (or None) are accepted.

    Args:
        value (object): Rejected limit.
    """
    with pytest.raises(ValueError, match="max_evaluations"):
        RKO(decode, cost, 60, max_evaluations=value).solve(0.1)


def test_prefill_is_not_counted_against_the_limit() -> None:
    """Startup evaluations are accounted separately from the per-algorithm limit."""
    result = RKO(
        decode,
        cost,
        60,
        algorithms=["SA"],
        execution="sequential",
        prefill_pool=True,
        pool_size=20,
        max_evaluations=30,
        random_state=1,
    ).solve(30.0)
    assert result.initialization is not None and result.initialization.n_evaluations == 20
    assert result.searches[0].n_evaluations == 30
    assert result.n_evaluations == 50


def test_standalone_solvers_and_reports_carry_the_limit() -> None:
    """The single-algorithm classes accept it and the report summary records it."""
    solver = SA(decode, cost, 60, execution="sequential", max_evaluations=120, random_state=1)
    result = solver.solve(30.0)
    assert result.searches[0].n_evaluations == 120
    assert OptimizationReport(result).to_dict()["summary"]["max_evaluations"] == 120
    assert RKO(decode, cost, 60).max_evaluations is None


def test_auto_engine_applies_the_limit_to_probes_and_the_final_run() -> None:
    """The automatic engine forwards the limit to every copy it runs."""
    problem = problems.Knapsack(PROFITS, WEIGHTS, CAPACITY)
    result = RKO(
        problem,
        engine="auto",
        auto_candidates=["rust"],
        auto_time=0.2,
        max_evaluations=200,
        random_state=1,
    ).solve(30.0)
    assert {item.n_evaluations for item in result.searches} == {200}
    assert result.max_evaluations == 200


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({"engine": "python", "execution": "sequential"}, id="python-sequential"),
        pytest.param(
            {"engine": "python", "execution": "process", "n_jobs": 1}, id="python-process"
        ),
        pytest.param({"engine": "rust", "n_jobs": 1}, id="rust-one-worker"),
        pytest.param({"engine": "rust"}, id="rust-callbacks"),
    ],
)
def test_runs_without_concurrent_pool_exchange_repeat_exactly(
    settings: dict[str, object],
) -> None:
    """With a seed and a limit, sequential configurations return identical results.

    Args:
        settings (dict[str, object]): Engine and worker options with one search at a time.
    """
    first = RKO(decode, cost, 60, max_evaluations=400, random_state=7, **settings).solve(60.0)
    second = RKO(decode, cost, 60, max_evaluations=400, random_state=7, **settings).solve(60.0)
    assert fingerprint(first) == fingerprint(second)
