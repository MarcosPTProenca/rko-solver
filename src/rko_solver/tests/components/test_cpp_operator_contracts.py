"""Deterministic local-search contracts checked against the C++ reference.

Objective:
- Preserve actual reference sweep behavior instead of interpreting misleading labels.

Technical approach:
- Expected vectors/costs were checked with compiled SwapLS/InvertLS/FareyLS kernels
  from RKO_Cpp_v2.0 revision 126e35873e50dc308795b58b8c75d37f8de4c857.
- Fix index order and Farey midpoint draws to isolate transition semantics.

Flow (pseudocode):
1. Reproduce shuffled exploration order and a linear objective.
2. Execute both strategies for all three neighborhoods.
3. Compare reference outputs and verify the complete sweep was evaluated.

Important criteria:
- Reference strategy 1 continues after improvements; it does not return immediately.
- These checks do not establish whole-solver or random-stream equivalence.

Inputs/Outputs:
- Input: reference vectors, objective, deterministic order/draws and strategy.
- Output: reference vector/cost and evaluation-count assertions.
"""

import numpy as np
import pytest
import random
from collections.abc import Callable
from rko_solver.components.local_search.farey_ls import farey_ls
from rko_solver.components.local_search.mirror_ls import mirror_ls
from rko_solver.components.local_search.swap_ls import swap_ls


@pytest.mark.parametrize(
    "operator, strategy, initial, expected, expected_cost, evaluations",
    [
        (swap_ls, 1, [0.9, 0.1, 0.8], [0.1, 0.8, 0.9], 2.8, 4),
        (swap_ls, 2, [0.9, 0.1, 0.8], [0.1, 0.9, 0.8], 2.9, 4),
        (mirror_ls, 1, [0.9, 0.8, 0.1], [0.1, 0.2, 0.1], 0.8, 4),
        (mirror_ls, 2, [0.9, 0.8, 0.1], [0.1, 0.8, 0.1], 2.0, 4),
        (farey_ls, 1, [0.9, 0.8, 0.1], [0.0714285] * 3, 0.428571, 55),
        (farey_ls, 2, [0.9, 0.8, 0.1], [0.0714285, 0.8, 0.1], 1.9142855, 55),
    ],
)
def test_reference_local_search_sweeps(
    monkeypatch: pytest.MonkeyPatch,
    operator: Callable[..., tuple[np.ndarray, float]],
    strategy: int,
    initial: list[float],
    expected: list[float],
    expected_cost: float,
    evaluations: int,
) -> None:
    """Match compiled reference examples while keeping caller-owned inputs intact.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fix scan order and Farey draws.
        operator (Callable): Neighborhood function under test.
        strategy (int): Sequential-acceptance or fixed-base best strategy.
        initial (list[float]): Initial key vector.
        expected (list[float]): Reference final key vector.
        expected_cost (float): Reference objective value.
        evaluations (int): Initial evaluation plus full reference candidate sweep.
    """
    calls: list[float] = []
    shuffles: list[list[int]] = []

    def objective(keys: np.ndarray) -> float:
        """Record the same descending-weight objective used by the C++ probe.

        Args:
            keys (np.ndarray): Candidate vector.

        Returns:
            float: Linear objective to minimize.
        """
        value = float(np.dot(keys, [3.0, 2.0, 1.0]))
        calls.append(value)
        return value

    def shuffle(order: list[int]) -> None:
        """Supply the exploration order verified by the compiled C++ probe.

        Args:
            order (list[int]): Mutable internal copy of the base indices.
        """
        shuffles.append(order.copy())
        order[:] = [0, 1, 2]

    monkeypatch.setattr(random, "shuffle", shuffle)
    monkeypatch.setattr(random, "uniform", lambda low, high: (low + high) / 2)
    keys = np.asarray(initial)
    base_order = [2, 1, 0]
    result, cost = operator(keys, lambda vector: vector, objective, strategy, base_order)
    assert result == pytest.approx(expected)
    assert cost == pytest.approx(expected_cost)
    assert len(calls) == evaluations
    assert shuffles == [[2, 1, 0]]
    assert keys == pytest.approx(initial)
    assert base_order == [2, 1, 0]
