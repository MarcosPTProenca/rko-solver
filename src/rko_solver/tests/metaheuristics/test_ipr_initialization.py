"""Check bounded, domain-neutral initialization of missing IPR elite pairs.

Objective:
- Prevent standalone IPR from returning its only random evaluation without relinking.

Technical approach:
- Bound iterations deterministically rather than relying on host wall-clock speed.
- Exercise actual relinking, callback accounting, owned RNGs and callback restoration.

Flow (pseudocode):
1. Prepare empty, partial, full or one-entry-capacity pools.
2. Execute a bounded search through SearchSession with independent numerical costs.
3. Check pair creation, relinking, exact seed consumption and stop/error boundaries.

Important criteria:
- Existing usable pools must not consume additional initialization draws.
- A pool unable to retain two entries must not trigger unbounded seed retries.
- Initialization evaluations obey the existing callback deadline and failure contract.

Inputs/Outputs:
- Input: deterministic streams, pool occupancies and simulated stopping predicates.
- Output: finite accounted results or correctly propagated callback failures.
"""

import numpy as np
import pytest
import random
from rko_solver.components.random_streams import RandomStreams
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.ipr import IPR
from rko_solver.orchestration.search import SearchSession
from rko_solver.orchestration.types import CostFn
from unittest.mock import Mock
from unittest.mock import patch


def decode(keys: np.ndarray) -> np.ndarray:
    """Return continuous keys without imposing a domain-specific repair.

    Args:
        keys (np.ndarray): Detached search candidate.

    Returns:
        np.ndarray: Unmodified numerical candidate.
    """
    return keys


def sphere(keys: np.ndarray) -> float:
    """Compute a deterministic numerical minimization objective.

    Args:
        keys (np.ndarray): Continuous candidate.

    Returns:
        float: Squared Euclidean norm.
    """
    return float(np.dot(keys, keys))


def plateau(keys: np.ndarray) -> float:
    """Keep every candidate tied to check non-improving pool initialization.

    Args:
        keys (np.ndarray): Unused candidate with a valid dimension.

    Returns:
        float: Constant finite objective.
    """
    return 0.0


@pytest.mark.parametrize("occupancy", [0, 1, 2])
@pytest.mark.parametrize("cost", [sphere, plateau])
def test_missing_pairs_bootstrap_and_relink(
    monkeypatch: pytest.MonkeyPatch, occupancy: int, cost: CostFn
) -> None:
    """Use actual paths even with no producing peer, preserving prefilled draw order.

    Args:
        monkeypatch (pytest.MonkeyPatch): Install bounded iteration/backoff controls.
        occupancy (int): Initially available elites.
        cost (CostFn): Smooth or tied numerical objective.
    """
    pool = SolutionPool(4)
    for value in (0.1, 0.9)[:occupancy]:
        keys = np.full(5, value)
        pool.insert((cost(keys), keys.tolist()))
    algorithm = IPR(pool, decode, cost, 5)
    checks = iter([False] * 4 + [True])
    monkeypatch.setattr(algorithm, "time_limit_reached", lambda duration: next(checks))
    monkeypatch.setattr("rko_solver.metaheuristics.ipr.time.sleep", lambda duration: None)
    streams = RandomStreams(42)
    with streams.activate(), patch.object(algorithm, "_relink", wraps=algorithm._relink) as relink:
        session = SearchSession(decode, cost)
        result = session.solve(algorithm, 60.0)
    assert pool.size >= 2
    assert relink.call_count > 0
    assert session.n_evaluations > 1
    assert result[2] == cost(np.asarray(result[1]))
    assert session.metadata is not None
    assert session.metadata.n_evaluations == session.n_evaluations
    expected = np.random.RandomState(42)
    expected.random_sample(5)
    if occupancy < 2:
        expected.random_sample(5)
    assert np.array_equal(streams.numpy.get_state()[1], expected.get_state()[1])
    assert streams.numpy.get_state()[2:] == expected.get_state()[2:]
    if occupancy == 2:
        expected_python = random.Random(42)
        for size in (2, 2, 3, 3) if cost is sphere else (2, 2, 2, 2):
            expected_python.sample(list(range(size)), 2)
        assert streams.python.getstate() == expected_python.getstate()


@pytest.mark.parametrize("parents", [([0.2], [0.2]), ([2**60], [2**60 + 1])])
def test_custom_pool_identical_effective_parents_back_off(
    monkeypatch: pytest.MonkeyPatch, parents: tuple[list[float], list[float]]
) -> None:
    """Skip custom-pool clones, including distinct integers collapsing to binary64.

    Args:
        monkeypatch (pytest.MonkeyPatch): Simulate custom snapshots and bounded waits.
        parents (tuple[list[float],list[float]]): Equal effective numerical chromosomes.
    """
    pool = SolutionPool(2)
    pool.insert((0.0, [0.1]))
    pool.insert((1.0, [0.9]))
    monkeypatch.setattr(pool, "get_elite", lambda count: [(0.0, keys.copy()) for keys in parents])
    algorithm = IPR(pool, decode, sphere, 1)
    checks = iter([False, False, True])
    monkeypatch.setattr(algorithm, "time_limit_reached", lambda duration: next(checks))
    monkeypatch.setattr(algorithm, "_relink", lambda *_: pytest.fail("identical pair relinked"))
    with patch("rko_solver.metaheuristics.ipr.time.sleep") as sleep, RandomStreams(42).activate():
        session = SearchSession(decode, sphere)
        result = session.solve(algorithm, 60.0)
    assert session.n_evaluations == 1
    assert sleep.call_count == 2
    assert result[2] == sphere(np.asarray(result[1]))


def test_repeated_seed_keys_do_not_retry_until_unique(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attempt only one extra seed when duplicate draws cannot establish diversity.

    Args:
        monkeypatch (pytest.MonkeyPatch): Install a repeated-key RNG and bounded timer.
    """
    pool = SolutionPool(4)
    algorithm = IPR(pool, decode, sphere, 5)
    checks = iter([False, False, False, True])
    monkeypatch.setattr(algorithm, "time_limit_reached", lambda duration: next(checks))
    monkeypatch.setattr(algorithm, "_relink", lambda *_: pytest.fail("no distinct pair"))
    streams = RandomStreams(42)
    generator = Mock(wraps=streams.numpy)
    generator.random.side_effect = lambda size: np.full(size, 0.5)
    monkeypatch.setattr(streams, "numpy", generator)
    with patch("rko_solver.metaheuristics.ipr.time.sleep") as sleep, streams.activate():
        session = SearchSession(decode, sphere)
        session.solve(algorithm, 60.0)
    assert session.n_evaluations == 2 and generator.random.call_count == 2
    assert pool.size == 1 and sleep.call_count == 2


def test_pair_bootstrap_is_bounded_when_capacity_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain bounded backoff after at most two seed evaluations when no pair can fit.

    Args:
        monkeypatch (pytest.MonkeyPatch): Install deterministic iteration bounds.
    """
    pool = SolutionPool(1)
    algorithm = IPR(pool, decode, sphere, 5)
    checks = iter([False] * 4 + [True])
    monkeypatch.setattr(algorithm, "time_limit_reached", lambda duration: next(checks))
    with patch("rko_solver.metaheuristics.ipr.time.sleep") as sleep, RandomStreams(42).activate():
        session = SearchSession(decode, sphere)
        result = session.solve(algorithm, 60.0)
    assert session.n_evaluations == 2
    assert pool.size == 1
    assert sleep.call_count > 0
    assert result[2] == sphere(np.asarray(result[1]))


def test_expired_session_does_not_evaluate_another_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve the first result if its callback exhausts the session's deadline.

    Args:
        monkeypatch (pytest.MonkeyPatch): Bound iterations even if no seed is attempted.
    """
    session: SearchSession

    def cost(keys: np.ndarray) -> float:
        """Expire the active session after its first finite objective completes.

        Args:
            keys (np.ndarray): Continuous candidate.

        Returns:
            float: Finite squared norm.
        """
        session.deadline = -1.0
        return sphere(keys)

    pool = SolutionPool(4)
    algorithm = IPR(pool, decode, cost, 5)
    checks = iter([False] * 4 + [True])
    monkeypatch.setattr(algorithm, "time_limit_reached", lambda duration: next(checks))
    session = SearchSession(decode, cost)
    with RandomStreams(42).activate():
        result = session.solve(algorithm, 60.0)
    assert session.n_evaluations == 1
    assert result[2] == sphere(np.asarray(result[1]))
    assert session.metadata is not None
    assert session.metadata.stop_reason == "time_limit"
    assert algorithm.decoder is decode and algorithm.cost is cost


def test_seed_failure_propagates_and_restores_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose initialization callback errors instead of silently waiting for peers.

    Args:
        monkeypatch (pytest.MonkeyPatch): Bound the empty-pool wait in a regressed build.
    """
    calls = 0

    def cost(keys: np.ndarray) -> float:
        """Fail on the additional seed, after one valid incumbent.

        Args:
            keys (np.ndarray): Continuous candidate.

        Returns:
            float: Initial finite norm.

        Raises:
            ArithmeticError: On the second evaluated candidate.
        """
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ArithmeticError("seed objective failed")
        return sphere(keys)

    algorithm = IPR(SolutionPool(4), decode, cost, 5)
    checks = iter([False] * 4 + [True])
    monkeypatch.setattr(algorithm, "time_limit_reached", lambda duration: next(checks))
    with RandomStreams(42).activate(), pytest.raises(ArithmeticError, match="seed objective"):
        SearchSession(decode, cost).solve(algorithm, 0.03)
    assert algorithm.decoder is decode and algorithm.cost is cost
