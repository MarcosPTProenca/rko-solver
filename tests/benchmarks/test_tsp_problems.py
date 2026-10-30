"""Tests for the traveling-salesman instances, priors and residual sort decoder."""

from __future__ import annotations

import itertools
import numpy as np
import pytest
from benchmarks.engines.solver_run import SolverRun
from benchmarks.engines.tsp_problems import KINDS
from benchmarks.engines.tsp_problems import TspCost
from benchmarks.engines.tsp_problems import TspDecoder
from benchmarks.engines.tsp_problems import TspFamily
from benchmarks.engines.tsp_problems import TspInstance


def optimum(instance: TspInstance) -> float:
    """Brute-force the optimal tour of a tiny instance.

    Args:
        instance (TspInstance): Instance with at most nine cities.

    Returns:
        float: Optimal closed tour length.
    """
    count = instance.n_cities
    return min(
        TspFamily.length(instance, np.array((0, *rest)))
        for rest in itertools.permutations(range(1, count))
    )


@pytest.mark.parametrize("kind", KINDS)
def test_instances_are_deterministic_and_seed_dependent(kind: str) -> None:
    """The same seed reproduces an instance and another seed changes it."""
    first = TspFamily.make(kind, 40, 3)
    again = TspFamily.make(kind, 40, 3)
    other = TspFamily.make(kind, 40, 4)
    assert first.name == f"{kind}-40-3" and first.n_cities == 40
    assert np.array_equal(first.coordinates, again.coordinates)
    assert not np.array_equal(first.coordinates, other.coordinates)
    assert first.coordinates.shape == (40, 2)


def test_clustered_cities_sit_closer_to_a_neighbour_than_uniform_ones() -> None:
    """Clustering is real: the mean nearest-neighbour distance is smaller."""

    def mean_nearest(kind: str) -> float:
        values = []
        for seed in range(5):
            points = TspFamily.make(kind, 200, seed).coordinates
            gaps = np.linalg.norm(points[:, None] - points[None], axis=2)
            np.fill_diagonal(gaps, np.inf)
            values.append(gaps.min(axis=1).mean())
        return float(np.mean(values))

    assert mean_nearest("clustered") < 0.8 * mean_nearest("uniform")


def test_generator_rejects_unknown_kind_and_tiny_size() -> None:
    """Invalid arguments raise ValueError."""
    with pytest.raises(ValueError, match="kind"):
        TspFamily.make("ring", 20, 0)
    with pytest.raises(ValueError, match="cities"):
        TspFamily.make("uniform", 3, 0)


def test_length_of_a_unit_square_tour_and_wrong_size_order() -> None:
    """A square visited in order has perimeter four; a bad order raises."""
    square = TspInstance("sq", np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    assert TspFamily.length(square, np.arange(4)) == pytest.approx(4.0)
    assert TspFamily.length(square, np.array([0, 2, 1, 3])) == pytest.approx(2 + 2 * 2**0.5)
    assert TspCost(square)(np.arange(4)) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="one entry per city"):
        TspFamily.length(square, np.arange(3))


@pytest.mark.parametrize("kind", KINDS)
def test_nearest_neighbour_is_a_permutation_from_city_zero_and_beats_random(kind: str) -> None:
    """The greedy tour is valid, starts at zero and is far shorter than a random order."""
    instance = TspFamily.make(kind, 120, 1)
    tour = TspFamily.nearest_neighbour(instance)
    assert tour[0] == 0 and sorted(tour.tolist()) == list(range(120))
    random_order = np.random.default_rng(0).permutation(120)
    assert TspFamily.length(instance, tour) < 0.4 * TspFamily.length(instance, random_order)


@pytest.mark.parametrize("kind", KINDS)
def test_two_opt_never_lengthens_and_stays_a_permutation(kind: str) -> None:
    """The improved tour is valid and no longer than the nearest-neighbour start."""
    instance = TspFamily.make(kind, 150, 2)
    start = TspFamily.nearest_neighbour(instance)
    tour = TspFamily.two_opt(instance, start)
    assert sorted(tour.tolist()) == list(range(150))
    assert TspFamily.length(instance, tour) < TspFamily.length(instance, start)


@pytest.mark.parametrize("kind", KINDS)
def test_two_opt_result_has_no_improving_exchange(kind: str) -> None:
    """The returned tour is a genuine 2-opt local optimum, not a truncated search."""
    instance = TspFamily.make(kind, 60, 4)
    tour = TspFamily.two_opt(instance, np.random.default_rng(0).permutation(60))
    best = TspFamily.length(instance, tour)
    for first in range(59):
        for second in range(first + 1, 60):
            candidate = tour.copy()
            candidate[first : second + 1] = candidate[first : second + 1][::-1]
            assert TspFamily.length(instance, candidate) >= best - 1e-9


def test_two_opt_solves_points_in_convex_position() -> None:
    """With no crossing possible at the optimum, 2-opt reaches the perimeter from a shuffle."""
    angles = np.sort(np.random.default_rng(0).uniform(0, 2 * np.pi, 30))
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    shuffle = np.random.default_rng(1).permutation(30)
    instance = TspInstance("circle", points[shuffle])
    tour = TspFamily.two_opt(instance, np.arange(30))
    perimeter = TspFamily.length(TspInstance("hull", points), np.arange(30))
    assert TspFamily.length(instance, tour) == pytest.approx(perimeter)


@pytest.mark.parametrize("seed", range(4))
def test_lower_bound_is_valid_and_close_on_tiny_instances(seed: int) -> None:
    """The 1-tree bound does not exceed the optimum and is within 25 percent of it."""
    instance = TspFamily.make("uniform", 8, seed)
    best = optimum(instance)
    bound = TspFamily.lower_bound(instance)
    assert bound <= best + 1e-9
    assert bound >= 0.75 * best


@pytest.mark.parametrize("kind", KINDS)
def test_lower_bound_is_below_every_tour_and_near_the_two_opt_length(kind: str) -> None:
    """On a mid-size instance the bound sits below the 2-opt tour by a modest margin."""
    instance = TspFamily.make(kind, 100, 5)
    tour = TspFamily.two_opt(instance, TspFamily.nearest_neighbour(instance))
    length = TspFamily.length(instance, tour)
    bound = TspFamily.lower_bound(instance)
    assert bound <= length
    assert bound >= 0.85 * length


def test_position_keys_reproduce_the_tour_and_stay_inside_the_unit_interval() -> None:
    """Sorting the keys gives the tour back."""
    tour = np.random.default_rng(3).permutation(25)
    keys = np.asarray(TspFamily.position_keys(tour))
    assert np.array_equal(np.argsort(keys, kind="stable"), tour)
    assert keys.min() > 0.0 and keys.max() < 1.0


def test_perturbed_keys_start_exact_and_validate_arguments() -> None:
    """The first row is the exact tour, noise zero repeats it, noise moves later rows."""
    tour = np.random.default_rng(4).permutation(60)
    rows = TspFamily.perturbed_keys(tour, 5, 0.05, 9)
    assert len(rows) == 5 and rows[0] == TspFamily.position_keys(tour)
    assert any(row != rows[0] for row in rows[1:])
    assert all(len(row) == 60 and 0 < min(row) and max(row) < 1 for row in rows)
    assert TspFamily.perturbed_keys(tour, 3, 0.0, 9) == [rows[0]] * 3
    assert TspFamily.perturbed_keys(tour, 5, 0.05, 9) == rows
    with pytest.raises(ValueError, match="count"):
        TspFamily.perturbed_keys(tour, 0, 0.05, 9)
    with pytest.raises(ValueError, match="noise"):
        TspFamily.perturbed_keys(tour, 2, -0.1, 9)


def test_plain_decoder_sorts_by_key_and_prior_decoder_interpolates() -> None:
    """No prior sorts by key; a tiny spread returns the prior; a large one follows the keys."""
    instance = TspFamily.make("uniform", 50, 6)
    tour = np.random.default_rng(5).permutation(50)
    keys = np.random.default_rng(6).random(50)
    assert np.array_equal(TspDecoder(instance)(keys), np.argsort(keys, kind="stable"))
    assert np.array_equal(TspDecoder(instance, tour, 1e-6)(keys), tour)
    assert np.array_equal(TspDecoder(instance, tour, 1e6)(keys), np.argsort(keys, kind="stable"))


def test_decoder_always_returns_a_permutation() -> None:
    """Across spreads and random keys the decoded order is a permutation."""
    instance = TspFamily.make("clustered", 64, 7)
    tour = TspFamily.nearest_neighbour(instance)
    rng = np.random.default_rng(7)
    for spread in (0.01, 0.1, 1.0, 5.0):
        decoder = TspDecoder(instance, tour, spread)
        for _ in range(20):
            assert sorted(decoder(rng.random(64)).tolist()) == list(range(64))


def test_solver_run_reports_a_verified_length_and_honours_seed_keys() -> None:
    """A short real solve through the shared runner verifies and starts from the seed tour."""
    instance = TspFamily.make("uniform", 30, 1)
    tour = TspFamily.two_opt(instance, TspFamily.nearest_neighbour(instance))
    keys = [TspFamily.position_keys(tour)]
    length, verified = SolverRun.solve(
        TspDecoder(instance), TspCost(instance), 30, ("sequential", ("SA",)), (3, 0.3), keys
    )
    assert verified and length <= TspFamily.length(instance, tour) + 1e-9
    assert SolverRun.solve(
        TspDecoder(instance), TspCost(instance), 30, ("sequential", ("SA",)), (3, 0.3), None
    )[1]
