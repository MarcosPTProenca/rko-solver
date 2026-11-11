"""Shared-memory elite pool: equivalence with the reference pool and cross-process visibility.

Objective:
- Verify that `SharedElitePool` admits, orders and reports entries exactly as `SolutionPool`,
  and that a spawned child's insert is visible to the parent.

Technical approach:
- Differential property test: random insert sequences with a small key domain (many duplicates)
  applied to both pools, comparing every snapshot. One spawned child inserts into the shared pool.

Flow (pseudocode):
1. Generate capacity, dimension and insert sequences with hypothesis.
2. Apply each insert to the reference and to the shared pool; compare snapshots and best.
3. Insert from a spawned child and read the entry from the parent.

Important criteria:
- The reference is `SolutionPool`; any difference in order, deduplication or occupancy is a bug.
- Shared memory is always closed and unlinked, even when an assertion fails.

Inputs/Outputs:
- Input: capacities, key vectors over a small domain, finite costs.
- Output: equal snapshots and visible cross-process entries.
"""

from __future__ import annotations

import multiprocessing as mp
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from rko_solver.components.shared_pool import SharedElitePool
from rko_solver.components.solution_pool import SolutionPool

COSTS = st.sampled_from([0.0, 0.25, 0.5, 1.0, 2.0])
KEYS = st.lists(st.sampled_from([0.0, 0.5, 1.0]), min_size=2, max_size=2)
OPERATIONS = st.lists(st.tuples(COSTS, KEYS), min_size=1, max_size=60)


def child_insert(pool: SharedElitePool, cost: float) -> None:
    """Insert one entry from a spawned process.

    Args:
        pool (SharedElitePool): Pool attached in the child.
        cost (float): Cost of the inserted entry.
    """
    pool.insert((cost, [0.5, 0.5]), "child")


@settings(max_examples=200, deadline=None)
@given(capacity=st.integers(min_value=1, max_value=5), operations=OPERATIONS)
def test_shared_pool_matches_the_reference_pool_after_every_insert(
    capacity: int, operations: list[tuple[float, list[float]]]
) -> None:
    """Snapshots, occupancy and best entry agree with SolutionPool at every step.

    Args:
        capacity (int): Pool capacity.
        operations (list[tuple[float, list[float]]]): Sequence of candidate inserts.
    """
    reference = SolutionPool(capacity)
    shared = SharedElitePool.create(capacity, 2)
    try:
        for cost, keys in operations:
            reference.insert((cost, keys), "algorithm")
            shared.insert((cost, keys), "algorithm")
            expected = [(value, list(vector)) for value, vector in reference.get_elite(capacity)]
            assert shared.get_elite(capacity) == expected
            assert shared.size == reference.size
        assert shared.best == reference.best
    finally:
        shared.close()
        shared.unlink()


def test_shared_pool_rejects_invalid_entries() -> None:
    """Nonfinite costs, empty vectors and wrong dimensions are refused before any write."""
    shared = SharedElitePool.create(3, 2)
    try:
        with pytest.raises(ValueError):
            shared.insert((float("nan"), [0.1, 0.2]), "algorithm")
        with pytest.raises(ValueError):
            shared.insert((0.1, []), "algorithm")
        with pytest.raises(ValueError):
            shared.insert((0.1, [0.1, 0.2, 0.3]), "algorithm")
        assert shared.size == 0
    finally:
        shared.close()
        shared.unlink()


def test_insert_from_a_spawned_child_is_visible_to_the_parent() -> None:
    """A real spawned process writes into the shared pool and the parent reads it."""
    context = mp.get_context("spawn")
    shared = SharedElitePool.create(4, 2, context)
    try:
        child = context.Process(target=child_insert, args=(shared, 0.125))
        child.start()
        child.join()
        assert child.exitcode == 0
        assert shared.best == (0.125, [0.5, 0.5])
    finally:
        shared.close()
        shared.unlink()
