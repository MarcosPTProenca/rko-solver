"""Selected elite transfer and caller-owned random sampling contracts.

Objective:
- Verify sparse pool snapshots without changing pair selection or ownership.

Technical approach:
- Compare managed pairs to full snapshots under identical Python RNG states.
- Exercise malformed ranks locally and controlled spawned publication remotely.

Flow (pseudocode):
1. Populate known finite entries and record the full-snapshot reference.
2. Sample managed ranks and compare entries and final random states.
3. Publish concurrently and verify detached, coherent pair snapshots.

Important criteria:
- Sampling remains client-side; NumPy RNG state is untouched.
- Old occupancy bounds remain valid because the pool never shrinks.
- Existing duck-typed pools need no new method.

Inputs/Outputs:
- Input: local/managed pools, seeded selections and a spawned publisher.
- Output: assertions for exact data, stream ownership and child cleanup.
"""

from __future__ import annotations

import multiprocessing as mp
import numpy as np
import pickle
import pytest
import random
from collections.abc import Iterator
from multiprocessing.synchronize import Event
from rko_solver.components.random_streams import RandomStreams
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.ipr import IPR
from rko_solver.orchestration.processes import PoolManager
from rko_solver.orchestration.processes import PoolProxy


@pytest.fixture(autouse=True)
def preserve_rng() -> Iterator[None]:
    """Restore both caller RNG states after each sampling regression.

    Returns:
        Iterator[None]: Single test scope with caller states restored on exit.
    """
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@pytest.fixture(scope="module")
def manager() -> Iterator[PoolManager]:
    """Own one spawned server for the managed sampling tests.

    Returns:
        Iterator[PoolManager]: Live server closed after all dependent tests.
    """
    with PoolManager(ctx=mp.get_context("spawn")) as server:
        yield server


@pytest.mark.parametrize(
    "indices",
    [None, [], (), (0,), (0, 1, 2), (0, 0), (-1, 0), (True, 0), (0, 1.0), (0, 2)],
)
def test_pair_rejects_invalid_ranks(indices: object) -> None:
    """Reject malformed requests without modifying pool contents.

    Args:
        indices (object): Invalid private transport request.
    """
    pool = SolutionPool()
    pool.insert((1.0, [0.1]))
    pool.insert((2.0, [0.2]))
    before = pool.get_elite(10)
    with pytest.raises(ValueError, match="Elite rank"):
        pool._get_elite_pair(indices)
    assert pool.get_elite(10) == before


def test_managed_pool_keeps_unique_vectors_and_best_repeat_observations(
    manager: PoolManager,
) -> None:
    """Apply the same authoritative admission rule through actual manager RPCs.

    Args:
        manager (PoolManager): Spawned pool server.
    """
    pool = manager.SolutionPool(10)
    for index in range(32):
        pool.insert((float(-index), [index % 2, 0.25]), str(index))
    assert pool.size == 2
    assert pool.get_elite(10) == [(-31.0, [1, 0.25]), (-30.0, [0, 0.25])]
    with RandomStreams(42).activate():
        pair = pool._sample_elite_pair(2)
    assert len(pair) == 2 and pair[0][1] != pair[1][1]
    pool.insert((-31.0, [0.5, 0.25]), "tied novel")
    assert pool.size == 3


def test_pair_preserves_requested_order_bits_types_and_ownership() -> None:
    """Keep tied ranks stable and retain key types without numeric repacking."""
    pool = SolutionPool()
    for index in range(3):
        pool.insert((1.0, [2**60 + index, -0.0]))
    pair = pool._get_elite_pair((2, 0))
    assert pair == [pool.get_elite(3)[2], pool.get_elite(3)[0]]
    assert type(pair[0][1][0]) is int
    assert pair[0][1][0] == 2**60 + 2
    assert np.signbit(pair[0][1][1])
    pair[0][1][0] = 99.0
    assert pool.get_elite(3)[2][1][0] == 2**60 + 2


def test_managed_pair_matches_full_snapshot_and_rng(
    manager: PoolManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sample the same ordered pair without transferring all vectors.

    Args:
        manager (PoolManager): Spawned authoritative server.
        monkeypatch (pytest.MonkeyPatch): Guard against full-snapshot proxy calls.
    """
    pool = manager.SolutionPool(10)
    for index in range(10):
        pool.insert((float(index), [index / 10] * 128))
    elite = pool.get_elite(10)
    monkeypatch.setattr(pool, "get_elite", lambda *_: pytest.fail("full snapshot fetched"))
    numpy_state = pickle.dumps(np.random.get_state())
    for seed in range(32):
        random.seed(seed)
        first, second = random.sample(list(range(len(elite))), 2)
        expected_state = random.getstate()
        random.seed(seed)
        pair = pool._sample_elite_pair(10)
        assert pair == [elite[first], elite[second]]
        assert random.getstate() == expected_state
        pair[0][1][0] = 999.0
        random.seed(seed)
        assert pool._sample_elite_pair(10) == [elite[first], elite[second]]
    assert pickle.dumps(np.random.get_state()) == numpy_state
    for count in (None, True, 1, -1, 2.0):
        with pytest.raises(ValueError, match="pool size"):
            pool._sample_elite_pair(count)


def _publish(pool: PoolProxy, ready: Event, release: Event) -> None:
    """Publish one synchronized improvement and then a concurrent burst.

    Args:
        pool (PoolProxy): Shared managed pool.
        ready (Event): Signal the first insertion has completed.
        release (Event): Parent authorization to publish the remaining entries.
    """
    pool.insert((-1.0, [-1.0] * 64))
    ready.set()
    if not release.wait(30):
        raise TimeoutError("Publisher was not released.")
    for value in range(2, 32):
        pool.insert((-float(value), [-float(value)] * 64))
        for cost, keys in pool._sample_elite_pair(pool.size):
            assert keys == [cost] * 64


def test_pair_uses_current_snapshot_after_growth_and_concurrent_writes(
    manager: PoolManager,
) -> None:
    """Keep old count bounds valid while selecting coherent new entries.

    Args:
        manager (PoolManager): Spawned authoritative server.
    """
    pool = manager.SolutionPool(10)
    for value in (0.0, 1.0):
        pool.insert((value, [value] * 64))
    old_count = pool.size
    context = mp.get_context("spawn")
    ready, release = context.Event(), context.Event()
    publisher = context.Process(target=_publish, args=(pool, ready, release))
    publisher.start()
    try:
        assert ready.wait(30)
        current = pool.get_elite(old_count)
        random.seed(42)
        indices = random.sample(list(range(old_count)), 2)
        expected_state = random.getstate()
        random.seed(42)
        assert pool._sample_elite_pair(old_count) == [current[index] for index in indices]
        assert random.getstate() == expected_state
        release.set()
        for _ in range(40):
            pair = pool._sample_elite_pair(pool.size)
            assert len(pair) == 2
            assert pair[0][0] != pair[1][0]
            for cost, keys in pair:
                assert keys == [cost] * 64
        publisher.join(timeout=30)
        assert publisher.exitcode == 0
    finally:
        release.set()
        if publisher.is_alive():
            publisher.terminate()
            publisher.join()
        publisher.close()


@pytest.mark.parametrize("managed,count", [(False, 3), (True, 2)])
def test_pool_without_overfetch_retains_snapshot_selection(
    manager: PoolManager, monkeypatch: pytest.MonkeyPatch, managed: bool, count: int
) -> None:
    """Keep local/custom and two-entry pools on the original sampling path.

    Args:
        manager (PoolManager): Spawned authoritative server.
        monkeypatch (pytest.MonkeyPatch): Replace elapsed-time and relinking work.
        managed (bool): Whether to exercise the built-in managed proxy.
        count (int): Initial pool occupancy.
    """
    pool = manager.SolutionPool() if managed else SolutionPool()
    for index in range(count):
        pool.insert((float(index), [index / count] * 8))
    if managed:
        monkeypatch.setattr(
            pool, "_sample_elite_pair", lambda *_: pytest.fail("nothing to avoid fetching")
        )
    else:
        assert not hasattr(pool, "_sample_elite_pair")
    elite = pool.get_elite(count)
    random.seed(17)
    indices = random.sample(list(range(count)), 2)
    expected_state = random.getstate()
    calls = []
    algorithm = IPR(pool, lambda keys: keys, lambda keys: float(np.sum(keys)), 8)
    checks = iter((False, True))
    monkeypatch.setattr(algorithm, "time_limit_reached", lambda _: next(checks))

    def relink(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, float]:
        """Record selected keys without additional random draws.

        Args:
            first (np.ndarray): Starting elite keys.
            second (np.ndarray): Guide elite keys.

        Returns:
            tuple[np.ndarray, float]: Starting vector and its known pool cost.
        """
        calls.append((first.tolist(), second.tolist()))
        return first, elite[indices[0]][0]

    monkeypatch.setattr(algorithm, "_relink", relink)
    random.seed(17)
    algorithm.solve(1.0)
    assert calls == [(elite[indices[0]][1], elite[indices[1]][1])]
    assert random.getstate() == expected_state
