"""Undefined-correlation community contracts for BRKGA-CS.

Objective:
- Keep valid constant chromosomes from causing numerical runtime warnings.

Technical approach:
- Exercise real elite graph construction and label propagation with warnings as errors.

Flow (pseudocode):
1. Construct a normal BRKGA-CS instance with two finite constant elites.
2. Identify representatives without introducing an undefined-correlation edge.

Important criteria:
- Undefined Pearson correlation must not create a community edge.
- No timing assertions or replacement graph implementation is used.

Inputs/Outputs:
- Input: two bounded two-gene chromosomes.
- Output: two independent elite representatives, without numerical warnings.
"""

import numpy as np
import pytest
import random
import warnings
from benchmarks.runtime.fixed_work import _run
from collections.abc import Iterator
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.brkga_cs import BRKGA_CS


@pytest.fixture(autouse=True)
def restore_random_state() -> Iterator[None]:
    """Restore caller Python/NumPy random streams after functional graph fixtures.

    Yields:
        None: Tests may exercise the library's legacy direct-call RNG boundary.
    """
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def test_constant_elites_remain_isolated_without_runtime_warnings() -> None:
    """Preserve the existing no-edge decision for undefined constant correlations.

    Raises:
        AssertionError: If constant elites are merged or representatives are lost.
        RuntimeWarning: If graph construction divides by a zero standard deviation.
    """
    algorithm = BRKGA_CS(SolutionPool(), lambda keys: keys, lambda decoded: 0.0, 2)
    population = [np.array([0.25, 0.25]), np.array([0.75, 0.75])]

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        representatives = algorithm._identify_promising(population, 2)

    assert representatives == [0, 1]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("length,value", [(1, 0.25), (2, 0.5), (128, 0.1)])
def test_constant_and_single_gene_elites_have_no_edges(
    dtype: type, length: int, value: float
) -> None:
    """Exclude mathematically undefined pairs even when rounding fabricates a coefficient.

    Args:
        dtype (type): Input floating dtype.
        length (int): Valid chromosome size, including a single observation.
        value (float): Constant allele value, including a nonbinary fraction.

    Raises:
        AssertionError: If constant elites are joined or warnings escape.
    """
    algorithm = BRKGA_CS(SolutionPool(), lambda keys: keys, lambda decoded: 0.0, length)
    population = [np.full(length, value, dtype=dtype), np.full(length, value, dtype=dtype)]
    before = [keys.copy() for keys in population]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        representatives = algorithm._identify_promising(population, 2)

    assert representatives == [0, 1]
    for original, actual in zip(before, population, strict=True):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("scale,expected_count", [(1e-250, 2), (1e-150, 1), (1e-160, 1)])
def test_numerical_normalization_boundary_has_no_epsilon_cutoff(
    scale: float, expected_count: int
) -> None:
    """Skip failed normalization but retain finite small-variance correlations.

    Args:
        scale (float): Bounded values spanning underflow and nonzero covariance.
        expected_count (int): Separate undefined elites or one correlated community.

    Raises:
        AssertionError: If normalization warnings escape or a finite edge is removed.
    """
    algorithm = BRKGA_CS(SolutionPool(), lambda keys: keys, lambda decoded: 0.0, 2)
    population = [np.array([0.0, scale]), np.array([0.0, 2 * scale])]
    before = np.geterr()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        representatives = algorithm._identify_promising(population, 2)

    assert len(representatives) == expected_count
    assert np.geterr() == before


def test_nonfinite_coefficient_cannot_create_a_graph_edge() -> None:
    """Keep an unusable coefficient out of community weights.

    Raises:
        AssertionError: If a nonfinite coefficient creates a community edge.
    """
    algorithm = BRKGA_CS(SolutionPool(), lambda keys: keys, lambda decoded: 0.0, 2)
    population = [np.array([0.25, np.nan]), np.array([0.0, 0.75])]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert algorithm._identify_promising(population, 2) == [0, 1]


def test_failed_normalization_restores_custom_caller_error_policy() -> None:
    """Handle a numerical degeneracy without leaking a NumPy error-policy change.

    Raises:
        AssertionError: If the error branch changes its caller's floating policy.
    """
    algorithm = BRKGA_CS(SolutionPool(), lambda keys: keys, lambda decoded: 0.0, 2)
    population = [np.array([0.0, 1e-250]), np.array([0.0, 2e-250])]
    with np.errstate(invalid="warn", divide="ignore", over="raise"):
        before = np.geterr()
        assert algorithm._identify_promising(population, 2) == [0, 1]
        assert np.geterr() == before


def test_one_gene_search_completes_fixed_work_without_runtime_warnings() -> None:
    """Reach real community construction during a complete direct search path.

    Raises:
        AssertionError: If bounded search work or finite incumbent recovery fails.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        row = _run(BRKGA_CS, "sphere", 7, dimension=1, evaluation_limit=128)
    assert row["evaluations"] == 128
    assert np.isfinite(row["cost"])


@pytest.mark.parametrize("constant_first", [True, False])
def test_mixed_constant_elites_do_not_block_defined_communities(constant_first: bool) -> None:
    """Keep positive and negative defined correlations while isolating constants.

    Args:
        constant_first (bool): Exercise either side of graph pair enumeration.

    Raises:
        AssertionError: If an undefined edge appears or a defined component is lost.
    """
    a = np.array([0.0, 0.25, 0.5, 0.75])
    population = [a, a[::-1].copy()]
    constant = np.full(4, 0.25)
    population.insert(0 if constant_first else 2, constant)
    algorithm = BRKGA_CS(SolutionPool(), lambda keys: keys, lambda decoded: 0.0, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        representatives = algorithm._identify_promising(population, 3)

    assert len(representatives) == 2
    assert (0 if constant_first else 2) in representatives


@pytest.mark.parametrize("seed", [7, 42, 200])
def test_defined_graph_preserves_representatives_and_random_draws(seed: int) -> None:
    """Compare real graph construction against a hand-worked two-component oracle.

    Args:
        seed (int): Deterministic label-propagation and representative draws.

    Raises:
        AssertionError: If defined graph decisions or random consumption change.
    """
    a = np.array([0.0, 0.25, 0.5, 0.75])
    b = np.array([0.25, 0.75, 0.0, 0.5])
    population = [a, a[::-1].copy(), b, b[::-1].copy()]
    graph = [[(1, 1.0)], [(0, 1.0)], [(3, 1.0)], [(2, 1.0)]]
    algorithm = BRKGA_CS(SolutionPool(), lambda keys: keys, lambda decoded: 0.0, 4)
    saved = random.getstate()
    try:
        random.seed(seed)
        labels = algorithm._label_propagation(graph, 4)
        communities: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            communities.setdefault(label, []).append(index)
        expected = [random.choice(members) for members in communities.values()]
        expected_state = random.getstate()
        random.seed(seed)
        actual = algorithm._identify_promising(population, 4)
        assert actual == expected
        assert random.getstate() == expected_state
        assert len(actual) == 2
        assert sum(index in (0, 1) for index in actual) == 1
        assert sum(index in (2, 3) for index in actual) == 1
    finally:
        random.setstate(saved)
