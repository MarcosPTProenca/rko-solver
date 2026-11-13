"""Contract tests for the optional swap delta of a Numba objective (issue 157).

Objective:
- Verify that a user-supplied swap delta is exact, leaves the search trajectory unchanged, and is
  counted separately as delta evaluations.

Technical approach:
- Use a small symmetric TSP objective with integer distances, so every sum is exact in float64.
- Compare solver runs with and without the delta under a fixed evaluation budget and seed.
- Recompute the swapped objective directly in Python for random and tied key vectors.

Flow (pseudocode):
1. Compile the tour objective and an exact swap delta with Numba.
2. Check the delta against full evaluation on random and tied key vectors.
3. Solve with and without the delta; compare keys, costs and evaluation counts.
4. Check that a cost decoder and a non-Numba delta are rejected.

Important criteria:
- Identical trajectories are required: an exact delta must not change the result.
- Delta counts are reported separately and are zero without a delta.

Inputs/Outputs:
- Input: a 20-city instance, short solver runs with fixed seeds.
- Output: pass/fail assertions.
"""

import numpy as np
import pytest
from rko_solver import RKO
from rko_solver import Numba

numba = pytest.importorskip("numba")
pytest.importorskip("rko_solver._native")

CITIES = 20
_RNG = np.random.default_rng(157)
POINTS = _RNG.integers(0, 100, size=(CITIES, 2)).astype(np.float64)
DISTANCES = np.floor(np.sqrt(((POINTS[:, None, :] - POINTS[None, :, :]) ** 2).sum(axis=2)) + 0.5)


@numba.njit
def tour_length(keys: np.ndarray) -> float:
    """Return the closed tour length of the order that sorts the keys.

    Args:
        keys (np.ndarray): One key per city.

    Returns:
        float: Tour length over the rounded distance matrix.
    """
    order = np.argsort(keys, kind="mergesort")
    total = 0.0
    for position in range(order.size):
        total += DISTANCES[order[position], order[(position + 1) % order.size]]
    return total


@numba.njit
def tour_swap(keys: np.ndarray, score: float, i: int, j: int) -> float:
    """Return the tour length after exchanging the key values of cities i and j.

    Args:
        keys (np.ndarray): Current key vector, not modified.
        score (float): Tour length of `keys`.
        i (int): First city.
        j (int): Second city.

    Returns:
        float: Tour length of the swapped key vector.
    """
    swapped = keys.copy()
    swapped[i], swapped[j] = keys[j], keys[i]
    return tour_length(swapped)


def test_swap_delta_matches_full_evaluation_including_ties() -> None:
    """The delta equals a full re-evaluation for random and tied key vectors."""
    rng = np.random.default_rng(3)
    for trial in range(200):
        keys = rng.random(CITIES)
        if trial % 2:
            keys = np.round(keys * 3) / 3
        i, j = rng.choice(CITIES, size=2, replace=False)
        swapped = keys.copy()
        swapped[i], swapped[j] = keys[j], keys[i]
        assert tour_swap(keys, tour_length(keys), int(i), int(j)) == tour_length(swapped)


def test_swap_delta_leaves_the_trajectory_unchanged_and_counts_deltas() -> None:
    """A run with an exact delta reproduces the run without one and reports its delta count."""
    options = {"engine": "rust", "n_jobs": 1, "max_evaluations": 4000, "random_state": 11}
    plain = RKO(Numba(tour_length, CITIES), algorithms=["ILS"], **options).solve(60.0)
    scored = RKO(
        Numba(tour_length, CITIES, swap_delta=tour_swap), algorithms=["ILS"], **options
    ).solve(60.0)
    assert scored.keys == plain.keys
    assert scored.cost == plain.cost
    assert scored.searches[0].n_evaluations == plain.searches[0].n_evaluations
    assert plain.searches[0].delta_evaluations == 0
    assert scored.searches[0].delta_evaluations > 0


def test_cost_decoder_with_swap_delta_is_rejected() -> None:
    """A delta is defined on key vectors, so it cannot be combined with a decoder."""

    @numba.njit
    def first_key(keys: np.ndarray) -> float:
        """Return the first key.

        Args:
            keys (np.ndarray): Random keys.

        Returns:
            float: The first key.
        """
        return keys[0]

    with pytest.raises(ValueError, match="swap_delta"):
        Numba(tour_length, CITIES, cost=first_key, swap_delta=tour_swap)


def test_non_numba_swap_delta_is_rejected() -> None:
    """The delta must be a Numba function, like the objective."""

    def plain_swap(keys: np.ndarray, score: float, i: int, j: int) -> float:
        """Return the score unchanged.

        Args:
            keys (np.ndarray): Key vector.
            score (float): Current score.
            i (int): First index.
            j (int): Second index.

        Returns:
            float: The score.
        """
        return score

    with pytest.raises(ValueError, match="swap_delta"):
        Numba(tour_length, CITIES, swap_delta=plain_swap)
