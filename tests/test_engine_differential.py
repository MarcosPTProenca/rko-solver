"""Differential tests across engines, execution modes and problem definitions.

Objective:
- Catch an engine or a problem-definition layer that computes something different from the
  others, using checks that cannot be satisfied by two engines that are both wrong the same way.

Technical approach:
- Exact differential: the same problem written as Python callbacks, a Numba function, a
  built-in problem and a block objective must give bit-identical seeded runs wherever the
  evaluation schedule is the same (Rust scalar definitions, Python callbacks versus blocks).
- Exact invariants on every engine and execution mode: the reported cost is recomputed from
  the reported keys with independent code, keys stay in [0, 1), and every search ends at
  exactly the evaluation limit.
- Sanity floors against an exact optimum (Knapsack dynamic program) and against the best of
  2,000 random key vectors. These are not performance claims: they only prove each engine
  searches far better than chance, with margins several times wider than the measured spread.

Flow (pseudocode):
1. Build Knapsack, TSP and Rastrigin instances from fixed seeds.
2. Compare problem definitions on the Rust and Python engines, then check invariants on all
   engines, then compare quality with the exact and random references.

Important criteria:
- Seeded equality is asserted only where it is expected: block objectives on Rust batch
  candidates, so their evaluation schedule differs and only invariants are asserted there.
- The C++ engine is not repeatable even with a seed, so it takes part in invariants and floors
  but never in equality.

Inputs/Outputs:
- Input: a 60-item Knapsack, a 40-city TSP and a 10-variable Rastrigin function.
- Output: exact equality or inequality assertions, with the floors stated in each test.
"""

import hashlib
import numpy as np
import pytest
from rko_solver import RKO
from rko_solver import Blocks
from rko_solver import Callbacks
from rko_solver import Numba
from rko_solver import problems
from rko_solver.orchestration.configuration import MH_SPECS

native = pytest.importorskip("rko_solver._native")
N_ITEMS = 60
PROFITS = np.array([10 + (index * 37) % 91 for index in range(N_ITEMS)], dtype=np.int64)
WEIGHTS = np.array([5 + (index * 53) % 47 for index in range(N_ITEMS)], dtype=np.int64)
CAPACITY = 700
CITIES = np.random.default_rng(1).random((40, 2))
DISTANCES = np.linalg.norm(CITIES[:, None] - CITIES[None], axis=2)
DIMENSIONS = 10
ENGINES = {
    "python-sequential": {"engine": "python", "execution": "sequential"},
    "python-thread": {"engine": "python", "execution": "thread"},
    "python-process": {"engine": "python", "execution": "process", "n_jobs": 2},
    "rust": {"engine": "rust"},
    "cpp": {"engine": "cpp"},
}
needs_cpp = pytest.mark.skipif(not native.cpp_available(), reason="built without the C++ engine")
ENGINE_PARAMS = [pytest.param(name, marks=needs_cpp) if name == "cpp" else name for name in ENGINES]


def decode_items(keys: np.ndarray) -> np.ndarray:
    """Select the Knapsack items whose key exceeds one half.

    Args:
        keys (np.ndarray): Random keys.

    Returns:
        np.ndarray: Boolean selection.
    """
    return np.asarray(keys) > 0.5


def cost_items(selection: np.ndarray) -> float:
    """Return the penalized negative profit of a selection.

    Args:
        selection (np.ndarray): Boolean selection.

    Returns:
        float: Cost to minimize.
    """
    chosen = np.asarray(selection)
    return float(100000 * max(0, int(chosen @ WEIGHTS) - CAPACITY) - int(chosen @ PROFITS))


def block_items(keys: np.ndarray) -> np.ndarray:
    """Score a block of key vectors with the Knapsack objective.

    Args:
        keys (np.ndarray): Matrix with one key vector per row.

    Returns:
        np.ndarray: One cost per row.
    """
    chosen = np.asarray(keys) > 0.5
    return 100000.0 * np.maximum(0.0, chosen @ WEIGHTS - CAPACITY) - chosen @ PROFITS


def decode_tour(keys: np.ndarray) -> np.ndarray:
    """Order the cities by increasing key.

    Args:
        keys (np.ndarray): Random keys.

    Returns:
        np.ndarray: City permutation.
    """
    return np.argsort(np.asarray(keys), kind="stable")


def cost_tour(tour: np.ndarray) -> float:
    """Return the closed tour length.

    Args:
        tour (np.ndarray): City permutation.

    Returns:
        float: Tour length.
    """
    return float(DISTANCES[tour, np.roll(tour, -1)].sum())


def decode_point(keys: np.ndarray) -> np.ndarray:
    """Map keys to the Rastrigin domain [-5.12, 5.12].

    Args:
        keys (np.ndarray): Random keys.

    Returns:
        np.ndarray: Point in the domain.
    """
    return 10.24 * np.asarray(keys) - 5.12


def cost_point(point: np.ndarray) -> float:
    """Return the Rastrigin function.

    Args:
        point (np.ndarray): Point in the domain.

    Returns:
        float: Function value, zero at the origin.
    """
    return float(10 * DIMENSIONS + np.sum(point * point - 10 * np.cos(2 * np.pi * point)))


PROBLEMS = {
    "knapsack": (decode_items, cost_items, N_ITEMS),
    "tsp": (decode_tour, cost_tour, len(CITIES)),
    "rastrigin": (decode_point, cost_point, DIMENSIONS),
}


def optimum() -> int:
    """Compute the exact Knapsack optimum by dynamic programming over capacities.

    Returns:
        int: Largest total profit within the capacity.
    """
    best = np.zeros(CAPACITY + 1, dtype=np.int64)
    for profit, weight in zip(PROFITS, WEIGHTS, strict=True):
        best[weight:] = np.maximum(best[weight:], best[:-weight] + profit)
    return int(best[CAPACITY])


def best_random(problem: str, draws: int = 2000) -> float:
    """Return the lowest cost among random key vectors from a fixed generator.

    Args:
        problem (str): Key of `PROBLEMS`.
        draws (int): Number of random vectors.

    Returns:
        float: Best cost found by chance.
    """
    decode, cost, size = PROBLEMS[problem]
    rng = np.random.default_rng(0)
    return min(cost(decode(rng.random(size))) for _ in range(draws))


def fingerprint(result: object) -> tuple[object, ...]:
    """Reduce a result to what must repeat in an identical seeded run.

    Args:
        result (object): An `RKOResult`.

    Returns:
        tuple[object, ...]: Cost, digest of the keys and per-search evaluation counts.
    """
    digest = hashlib.sha256(repr(result.keys).encode()).hexdigest()[:12]
    return result.cost, digest, tuple(item.n_evaluations for item in result.searches)


def one(problem: object, algorithm: str, **settings: object) -> tuple[object, ...]:
    """Run one algorithm alone for 400 evaluations with a fixed seed.

    Args:
        problem (object): Problem definition passed to `RKO`.
        algorithm (str): Registered algorithm name.
        **settings (object): Engine and execution settings.

    Returns:
        tuple[object, ...]: Fingerprint of the result.
    """
    return fingerprint(
        RKO(
            problem,
            algorithms=[algorithm],
            adaptation="fixed",
            max_evaluations=400,
            random_state=5,
            **settings,
        ).solve(60.0)
    )


@pytest.mark.parametrize("algorithm", list(MH_SPECS))
def test_rust_runs_are_identical_for_callbacks_numba_and_builtin(algorithm: str) -> None:
    """Three definitions of the same Knapsack drive the Rust engine to the same run.

    Args:
        algorithm (str): Algorithm under test.
    """
    numba = pytest.importorskip("numba")

    @numba.njit
    def compiled(keys: np.ndarray) -> float:
        """Score one key vector exactly like `cost_items`.

        Args:
            keys (np.ndarray): Random keys.

        Returns:
            float: Penalized negative profit.
        """
        profit, weight = 0.0, 0.0
        for index in range(keys.shape[0]):
            if keys[index] > 0.5:
                profit += PROFITS[index]
                weight += WEIGHTS[index]
        return 100000.0 * max(0.0, weight - CAPACITY) - profit

    settings = {"engine": "rust", "n_jobs": 1}
    builtin = problems.Knapsack(PROFITS.tolist(), WEIGHTS.tolist(), CAPACITY)
    callbacks = one(Callbacks(decode_items, cost_items, N_ITEMS), algorithm, **settings)
    assert one(Numba(compiled, N_ITEMS), algorithm, **settings) == callbacks
    assert one(builtin, algorithm, **settings) == callbacks


@pytest.mark.parametrize("algorithm", list(MH_SPECS))
def test_python_runs_are_identical_for_callbacks_and_block_objective(algorithm: str) -> None:
    """The Python engine scores a block objective one row at a time, so runs coincide.

    Args:
        algorithm (str): Algorithm under test.
    """
    settings = {"engine": "python", "execution": "sequential"}
    callbacks = one(Callbacks(decode_items, cost_items, N_ITEMS), algorithm, **settings)
    assert one(Blocks(block_items, N_ITEMS, 16), algorithm, **settings) == callbacks


def test_rust_block_objective_batches_but_stays_valid_and_exact_in_count() -> None:
    """Blocks change the Rust evaluation schedule, so only invariants are asserted."""
    result = RKO(
        Blocks(block_items, N_ITEMS, 16), engine="rust", max_evaluations=400, random_state=5
    ).solve(60.0)
    assert {item.n_evaluations for item in result.searches} == {400}
    assert cost_items(decode_items(np.asarray(result.keys))) == result.cost


@pytest.mark.parametrize("problem", list(PROBLEMS))
@pytest.mark.parametrize("engine", ENGINE_PARAMS)
def test_invariants_hold_on_every_engine_execution_and_problem(engine: str, problem: str) -> None:
    """The reported cost is recomputable from the keys and every search ends at the limit.

    Args:
        engine (str): Key of `ENGINES`.
        problem (str): Key of `PROBLEMS`.
    """
    decode, cost, size = PROBLEMS[problem]
    result = RKO(
        Callbacks(decode, cost, size), max_evaluations=300, random_state=2, **ENGINES[engine]
    ).solve(120.0)
    keys = np.asarray(result.keys)
    assert keys.shape == (size,) and np.all((keys >= 0.0) & (keys < 1.0))
    assert np.isfinite(result.cost)
    assert cost(decode(keys)) == pytest.approx(result.cost, rel=1e-9, abs=1e-9)
    assert len(result.searches) == 10
    assert {item.n_evaluations for item in result.searches} == {300}
    assert {item.stop_reason for item in result.searches} == {"max_evaluations"}
    assert result.stop_reason == "max_evaluations"
    assert all(item.random_state is not None for item in result.searches)
    startup = result.initialization.n_evaluations if result.initialization else 0
    assert result.n_evaluations == 3000 + startup


@pytest.mark.parametrize(
    "engine", ["python-sequential", "rust", pytest.param("cpp", marks=needs_cpp)]
)
def test_every_engine_searches_far_better_than_chance(engine: str) -> None:
    """Sanity floors, not performance claims: each engine beats random sampling by a wide margin.

    Measured with the default portfolio at 2,000 evaluations per algorithm, the Knapsack profit
    was 0.97 to 0.99 of the exact optimum (random sampling: 0.80), the TSP tour 0.35 to 0.55 of
    the best random tour and Rastrigin under 0.4 percent of the best random value. The floors
    below are several times looser than those measurements.

    Args:
        engine (str): Engine under test.
    """
    results = {}
    for problem, (decode, cost, size) in PROBLEMS.items():
        results[problem] = RKO(
            Callbacks(decode, cost, size), max_evaluations=2000, random_state=1, **ENGINES[engine]
        ).solve(120.0)
    assert -results["knapsack"].cost >= 0.92 * optimum()
    assert results["knapsack"].cost < best_random("knapsack")
    assert results["tsp"].cost <= 0.65 * best_random("tsp")
    assert results["rastrigin"].cost <= 0.1 * best_random("rastrigin")
