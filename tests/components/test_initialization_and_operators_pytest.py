"""Tests for initialization, blending, and shaking operators.

Objective:
- Validate invariants and contract behavior of low-level component operators.

Technical approach:
- Use deterministic monkeypatching where stochastic behavior exists.
- Check shape/range guarantees and update semantics.

Flow (pseudocode):
1. Create synthetic vectors/pools.
2. Execute operators under controlled random behavior.
3. Assert ordering, dimensionality, and transformation outputs.

Important criteria:
- Operators preserve expected dimensions.
- Invalid input shape for blending raises `ValueError`.

Inputs/Outputs:
- Input: NumPy vectors and synthetic callbacks.
- Output: assertions for deterministic component behavior.
"""

from __future__ import annotations

import numpy as np
import pytest
from rko_solver.components.blending.blending import blending
from rko_solver.components.initialization.initialization import create_initial_solution
from rko_solver.components.initialization.initialization import create_pool_solutions
from rko_solver.components.initialization.initialization import update_pool_solutions
from rko_solver.components.shaking.mirror import mirror_shake
from rko_solver.components.shaking.neighbor_swap import neighbor_swap_shake
from rko_solver.components.shaking.random import shake
from rko_solver.components.shaking.swap import swap_shake


def test_create_initial_solution_shape_and_empty() -> None:
    """Ensure random-key initialization handles empty and fixed sizes."""
    assert create_initial_solution(0).size == 0
    keys = create_initial_solution(6)
    assert keys.shape == (6,)
    assert np.all((keys >= 0.0) & (keys < 1.0))


def test_create_pool_solutions_returns_sorted_pool() -> None:
    """Ensure generated pool is sorted by objective value."""
    pool = create_pool_solutions(
        size_pool=5,
        tam_solution=4,
        decoder=lambda x: x,
        cost=lambda x: float(np.sum(x)),
    )
    costs = [entry[0] for entry in pool]
    assert len(pool) == 5
    assert costs == sorted(costs)


def test_update_pool_solutions_rejects_duplicate_cost() -> None:
    """Ensure candidate with duplicate cost does not change the pool."""
    base = [
        (1.0, np.array([0.1, 0.2])),
        (2.0, np.array([0.2, 0.3])),
    ]
    updated = update_pool_solutions(base, (2.0, np.array([0.9, 0.9])))
    assert updated == base


def test_update_pool_solutions_keeps_original_size() -> None:
    """Ensure pool update preserves the original cardinality."""
    base = [
        (1.0, np.array([0.1, 0.2])),
        (3.0, np.array([0.2, 0.3])),
    ]
    updated = update_pool_solutions(base, (2.0, np.array([0.9, 0.9])))
    assert len(updated) == 2
    assert [c for c, _ in updated] == [1.0, 2.0]


def test_blending_raises_for_invalid_parent_dimensions() -> None:
    """Reject parents that cannot represent one-dimensional key vectors."""
    with pytest.raises(ValueError, match="must have the same shape"):
        blending(np.array([0.1, 0.2]), np.array([0.1]))
    with pytest.raises(ValueError, match="one-dimensional"):
        blending(np.array([[0.1]]), np.array([[0.2]]))


def test_blending_preserves_random_draw_order_and_parent_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the reference draw sequence, offspring values and parent arrays."""
    draws = iter([0.01, 0.77, 0.5, 0.2, 0.01, 0.33, 0.5, 0.8])
    parent_a = np.array([0.1, 0.2, 0.3, 0.4])
    parent_b = np.array([0.6, 0.7, 0.8, 0.9])
    original_a = parent_a.copy()
    original_b = parent_b.copy()
    monkeypatch.setattr("random.random", lambda: next(draws))

    child = blending(parent_a, parent_b)

    assert child == pytest.approx([0.77, 0.2, 0.33, 0.9])
    assert parent_a == pytest.approx(original_a)
    assert parent_b == pytest.approx(original_b)
    with pytest.raises(StopIteration):
        next(draws)


def test_blending_preserves_nonfinite_complement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the reference NaN result for a NaN parent key."""
    monkeypatch.setattr("random.random", lambda: 0.6)
    child = blending(np.array([0.2]), np.array([np.nan]), factor=-1)
    assert np.isnan(child[0])


def test_blending_with_factor_minus_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure complementary blending mode mirrors genes from parent B."""
    draws = iter([0.6, 0.6, 0.6, 0.6])
    parent_a = np.array([0.2, 0.4])
    parent_b = np.array([0.1, 0.9])
    monkeypatch.setattr("random.random", lambda: next(draws))

    child = blending(parent_a, parent_b, factor=-1)

    assert np.allclose(child, np.array([0.9, 0.1]))
    assert np.all((child >= 0.0) & (child < 1.0))


def test_swap_shake_swaps_two_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure swap shake exchanges two sampled positions."""
    calls = iter([0, 2])
    monkeypatch.setattr(
        "rko_solver.components.shaking.swap.random.randint", lambda a, b: next(calls)
    )
    keys = np.array([0.1, 0.2, 0.3])

    out = swap_shake(keys.copy())
    assert np.allclose(out, np.array([0.3, 0.2, 0.1]))


def test_neighbor_swap_shake_swaps_adjacent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure neighbor swap shake exchanges adjacent positions."""
    monkeypatch.setattr(
        "rko_solver.components.shaking.neighbor_swap.random.randint", lambda a, b: 1
    )
    keys = np.array([0.1, 0.2, 0.3])

    out = neighbor_swap_shake(keys.copy())
    assert np.allclose(out, np.array([0.1, 0.3, 0.2]))


def test_mirror_shake_uses_complement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure mirror shake applies complementary transformation."""
    monkeypatch.setattr("rko_solver.components.shaking.mirror.random.randint", lambda a, b: 0)
    keys = np.array([0.3, 0.7])

    out = mirror_shake(keys.copy())
    assert np.allclose(out, np.array([0.7, 0.7]))


def test_shake_empty_vector_returns_empty() -> None:
    """Ensure random shaking is a no-op for empty vectors."""
    keys = np.array([])
    out = shake(keys, beta_min=0.1, beta_max=0.2)
    assert out.size == 0
