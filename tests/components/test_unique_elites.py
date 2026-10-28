"""Check atomic chromosome-unique elite admission without caching objective work.

Objective:
- Prevent repeated chromosomes from replacing genuine elite diversity.

Technical approach:
- Exercise tied costs, repeated observations, key ownership and concurrent writers.
- Keep validation before duplicate rejection and preserve scalar key representation.

Flow (pseudocode):
1. Insert unique and repeated numerical vectors into bounded pools.
2. Replace a duplicate only on a strictly better observed cost.
3. Verify sorted unique snapshots, intact callers and concurrent minimum retention.

Important criteria:
- Uniqueness is coordinate-value equality, not objective equality or quantization.
- Invalid duplicates still fail validation; re-evaluated better costs remain usable.

Inputs/Outputs:
- Input: finite entries, duplicate observations and concurrent insertions.
- Output: unique sorted pools or unchanged contents after invalid insertion.
"""

import math
import pytest
from concurrent.futures import ThreadPoolExecutor
from rko_solver.components.solution_pool import SolutionPool


@pytest.mark.parametrize("cost", [2.0, 3.0, 10.0])
def test_duplicate_does_not_occupy_another_slot(cost: float) -> None:
    """Retain one coordinate vector despite equal or worse repeat observations.

    Args:
        cost (float): Duplicate cost equal to or worse than the stored observation.
    """
    pool = SolutionPool(3)
    pool.insert((2.0, [0.1, 0.2]), "first")
    pool.insert((cost, [0.1, 0.2]), "repeat")
    assert pool.get_elite(3) == [(2.0, [0.1, 0.2])]
    pool.insert((2.0, [0.2, 0.1]), "tie")
    assert pool.size == 2


def test_better_duplicate_updates_cost_order_and_ownership() -> None:
    """Retain the best repeated observation without evicting an unrelated elite."""
    pool = SolutionPool(2)
    pool.insert((10.0, [0.1]), "old")
    pool.insert((5.0, [0.2]), "other")
    keys = [0.1]
    pool.insert((2.0, keys), "better")
    keys[0] = 99
    assert pool.get_elite(2) == [(2.0, [0.1]), (5.0, [0.2])]
    pool.insert((3.0, [0.1]), "worse")
    assert pool.get_elite(2) == [(2.0, [0.1]), (5.0, [0.2])]


def test_exact_coordinates_preserve_ties_lengths_and_integer_precision() -> None:
    """Avoid float packing, rounded comparison and deduplication by objective value."""
    pool = SolutionPool(10)
    vectors = [[2**60], [2**60 + 1], [0.1], [math.nextafter(0.1, 1.0)], [0.1, 0.2]]
    for keys in vectors:
        pool.insert((0.0, keys))
    assert [keys for _, keys in pool.get_elite(10)] == vectors
    pool.insert((0.0, [float(2**60)]))
    assert pool.size == len(vectors)
    assert type(pool.get_elite(1)[0][1][0]) is int


@pytest.mark.parametrize("cost", [math.nan, math.inf, -math.inf])
def test_invalid_duplicate_is_not_silently_ignored(cost: float) -> None:
    """Keep objective validation ahead of duplicate admission checks.

    Args:
        cost (float): Nonfinite repeated observation.
    """
    pool = SolutionPool(2)
    pool.insert((0.0, [0.1]))
    with pytest.raises(ValueError, match="finite"):
        pool.insert((cost, [0.1]))
    assert pool.get_elite(2) == [(0.0, [0.1])]


def test_concurrent_duplicate_inserts_are_atomic() -> None:
    """Keep one best observation per vector across interleaved insertion threads."""
    pool = SolutionPool(10)

    def publish(index: int) -> None:
        """Insert one repeat observation of either of two tied-coordinate families.

        Args:
            index (int): Independent observation identity.
        """
        pool.insert((float(-index), [index % 2, 0.25]), str(index))

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(publish, range(64)))
    assert pool.get_elite(10) == [(-63.0, [1, 0.25]), (-62.0, [0, 0.25])]
