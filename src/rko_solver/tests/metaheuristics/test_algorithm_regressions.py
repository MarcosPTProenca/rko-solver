"""Deterministic regression oracles for audited algorithm-state defects.

Objective:
- Verify chromosome fitness ownership, current Q maxima and relinking coverage.

Technical approach:
- Control one GA generation and compare stored fitness with selected chromosomes.
- Apply a negative Q update and a guide differing only in the final gene.

Flow (pseudocode):
1. Create small controlled states without stochastic convergence assumptions.
2. Execute the actual algorithm transition.
3. Compare with independently computed expected values.

Important criteria:
- Fitness must belong to its chromosome even without crossover and for odd sizes.
- Relinking must consider every gene, including a partial final block.

Inputs/Outputs:
- Input: fixed populations, Q values and guide vectors.
- Output: exact state/fitness contract assertions.
"""

import builtins
import numpy as np
import pytest
from collections.abc import Callable
from rko_solver.components.adaptive_control.q_learning import QLearningController
from rko_solver.components.blending.blending import blending
from rko_solver.components.local_search import nelder_mead_ls as nelder_mead_module
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics import ga as ga_module
from rko_solver.metaheuristics.ga import GeneticAlgorithm
from rko_solver.metaheuristics.ipr import IPR


@pytest.mark.parametrize("size", [3, 4])
def test_ga_keeps_selected_parent_fitness(size: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the actual generation's selection costs without inspecting stack frames.

    Args:
        size (int): Odd/even population size.
        monkeypatch (pytest.MonkeyPatch): Control selection, draws and local search.
    """
    algorithm = GeneticAlgorithm(
        SolutionPool(),
        lambda keys: keys,
        lambda keys: float(keys.sum()),
        1,
        size_pop=size,
        prob_cros=0.0,
        prob_mut=0.0,
    )
    candidates = iter(np.array([value / 10]) for value in range(1, size + 1))
    ticks = iter([False, True])
    observed = []

    def record_min(*values: object, key: Callable[[int], float]) -> int:
        """Observe costs used to rank the full population.

        Args:
            values (object): Indices supplied to min.
            key (Callable): Fitness lookup.

        Returns:
            int: Index with the lowest stored fitness.
        """
        if len(values) == 1 and isinstance(values[0], range):
            observed.append([key(index) for index in values[0]])
        return builtins.min(*values, key=key)

    monkeypatch.setattr(algorithm, "time_limit_reached", lambda _: next(ticks))
    monkeypatch.setattr(ga_module.np.random, "random", lambda _: next(candidates))
    monkeypatch.setattr(ga_module.random, "sample", lambda population, k: [0, 1, 2])
    monkeypatch.setattr(ga_module.random, "random", lambda: 0.9)
    monkeypatch.setattr(ga_module.random, "randint", lambda a, b: 0)
    monkeypatch.setattr(ga_module, "min", record_min, raising=False)
    monkeypatch.setattr(
        ga_module,
        "nelder_mead_search",
        lambda keys, decoder, cost, pool: (keys, cost(decoder(keys))),
    )
    algorithm.solve(1.0)
    assert observed[-1] == pytest.approx([0.1] * size)


def test_q_update_recomputes_decreased_best_action() -> None:
    """Choose the current best action after the previous best receives a penalty."""
    controller = QLearningController([[0.0, 1.0]])
    state = controller.states[0]
    state.Qa = [1.0, 0.5]
    state.maxA = 0
    state.maxQ = 1.0
    controller.update_q_value(0, 0, reward=-1.0, lf=1.0, df=0.0)
    assert state.Qa == [-1.0, 0.5]
    assert state.maxQ == 0.5
    assert controller.choose_action(0, epsilon=0.0) == 1


@pytest.mark.parametrize("dimension", [1, 25, 101])
def test_ipr_visits_partial_final_block(dimension: int) -> None:
    """Find a profitable move at the final chromosome position.

    Args:
        dimension (int): Chromosome length, including non-divisible sizes.
    """
    algorithm = IPR(SolutionPool(), lambda keys: keys, lambda keys: -float(keys[-1]), dimension)
    guide = np.zeros(dimension)
    guide[-1] = 0.9
    keys, cost = algorithm._relink(np.zeros(dimension), guide)
    assert keys[-1] == 0.9
    assert cost == -0.9


def test_nelder_mead_retains_better_initial_pool_vertex(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not discard a superior simplex seed when generated points are worse.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fix all generated simplex candidates.
    """
    pool = SolutionPool()
    pool.insert((0.1, [0.1]))
    pool.insert((0.2, [0.2]))
    monkeypatch.setattr(nelder_mead_module, "_blend", lambda a, b, factor: np.array([0.5]))
    keys, cost = nelder_mead_module.nelder_mead_search(
        np.array([0.9]),
        lambda keys: keys,
        lambda keys: float(keys.sum()),
        pool,
    )
    assert keys.tolist() == [0.1]
    assert cost == 0.1


def test_nelder_mead_reuses_shared_blending() -> None:
    """Keep one mutation/inheritance implementation for both public and local use."""
    assert nelder_mead_module._blend is blending


def test_ipr_waits_without_busy_polling() -> None:
    """Bound manager count requests while waiting for another elite producer."""

    class CountingPool(SolutionPool):
        """Count empty-pool observations during a bounded relinking attempt.

        Responsibilities:
        - Make repeated occupancy requests observable.

        Attributes:
            polls (int): Number of size queries.
        """

        polls: int = 0

        @property
        def size(self) -> int:
            """Record an occupancy query.

            Returns:
                int: Zero because no producer inserts entries in this test.
            """
            self.polls += 1
            return 0

    pool = CountingPool()
    IPR(pool, lambda keys: keys, lambda keys: 0.0, 2).solve(0.02)
    assert 1 <= pool.polls < 10
