"""Objective-sign-independent adaptive reward contracts.

Objective:
- Reward strict improvements and never reward deterioration of a minimized cost.

Technical approach:
- Check positive, zero, negative and extreme finite objectives.
- Observe rewards emitted by a controlled real ILS transition.

Flow (pseudocode):
1. Compare candidate costs against a previous incumbent.
2. Compute the controller reward or execute one adaptive ILS transition.
3. Assert bounded finite values and the correct reward sign.

Important criteria:
- Objective translation across zero must not reverse reward sign.

Inputs/Outputs:
- Input: known objective pairs and controlled local-search results.
- Output: reward-policy and actual caller assertions.
"""

import math
import numpy as np
import pytest
from rko_solver.components.adaptive_control.q_learning import QLearningController
from rko_solver.components.random_streams import RandomStreams
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics import grasp as grasp_module
from rko_solver.metaheuristics import ils as ils_module
from rko_solver.metaheuristics import lns as lns_module
from rko_solver.metaheuristics.grasp import GRASP
from rko_solver.metaheuristics.ils import ILS
from rko_solver.metaheuristics.lns import LNS
from unittest.mock import Mock


@pytest.mark.parametrize("previous", [-100.0, 0.0, 100.0])
@pytest.mark.parametrize("change", [-50.0, 0.0, 50.0])
def test_reward_has_correct_sign(previous: float, change: float) -> None:
    """Compare reward sign with independent minimization ordering.

    Args:
        previous (float): Previous incumbent cost.
        change (float): Candidate minus previous cost.
    """
    reward = QLearningController.relative_reward(previous, previous + change)
    assert math.isfinite(reward) and -1 <= reward <= 1
    assert (reward > 0) == (change < 0)
    assert (reward < 0) == (change > 0)


def test_extreme_finite_rewards_do_not_overflow() -> None:
    """Normalize before subtracting to avoid overflowing extreme finite costs."""
    assert QLearningController.relative_reward(1e308, -1e308) == 1
    assert QLearningController.relative_reward(-1e308, 1e308) == -1


@pytest.mark.parametrize("candidate", [-150.0, -100.0, -50.0])
def test_ils_emits_sign_safe_rewards(candidate: float, monkeypatch: pytest.MonkeyPatch) -> None:
    """Observe a real adaptive transition after a negative incumbent.

    Args:
        candidate (float): Controlled perturbed solution objective.
        monkeypatch (pytest.MonkeyPatch): Control the controller and local search.
    """
    controller = QLearningController([[0.1], [0.2]])
    algorithm = ILS(
        SolutionPool(),
        lambda keys: keys,
        lambda keys: -100.0 if keys[0] < 0.15 else candidate,
        1,
        control=1,
        parameters=[[0.1], [0.2]],
    )
    ticks = iter([False, True])
    solutions = iter([(np.array([0.1]), -100.0), (np.array([0.2]), candidate)])
    rewards = []
    monkeypatch.setattr(algorithm, "time_limit_reached", lambda _: next(ticks))
    monkeypatch.setattr(ils_module, "QLearningController", lambda *args, **kwargs: controller)
    monkeypatch.setattr(ils_module, "rvnd", lambda *args, **kwargs: next(solutions))
    monkeypatch.setattr(
        controller, "update_q_value", lambda st, at, reward, lf, df: rewards.append(reward)
    )
    algorithm.solve(1.0)
    assert len(rewards) == 1
    assert (rewards[0] > 0) == (candidate < -100)
    assert (rewards[0] < 0) == (candidate > -100)


@pytest.mark.parametrize("name", ["LNS", "GRASP"])
def test_reward_survives_incumbent_update(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reward a real improvement even after the algorithm replaces its incumbent.

    Args:
        name (str): Algorithm that previously compared against an updated incumbent.
        monkeypatch (pytest.MonkeyPatch): Control one adaptive transition.
    """
    grid = [[0.1], [0.2], [0.1], [0.1]] if name == "LNS" else [[0.3], [0.1], [0.1]]
    controller = QLearningController(grid)
    module = lns_module if name == "LNS" else grasp_module
    constructor = LNS if name == "LNS" else GRASP
    algorithm = constructor(
        SolutionPool(),
        lambda keys: keys,
        lambda keys: -100.0 if keys[0] < 0.15 else -150.0,
        1,
        control=1,
        parameters=grid,
    )
    ticks = iter([False, False, True, True])
    solutions = iter(
        [(np.array([0.2]), -150.0)]
        if name == "GRASP"
        else [(np.array([0.1]), -100.0), (np.array([0.2]), -150.0)]
    )
    rewards = []
    monkeypatch.setattr(algorithm, "time_limit_reached", lambda _: next(ticks))
    monkeypatch.setattr(module, "QLearningController", lambda *args, **kwargs: controller)
    if name == "GRASP":
        # GRASP scores its initial vector without refinement and then constructs,
        # so the first scripted result must arrive during that construction.
        streams = RandomStreams(42)
        generator = Mock(wraps=streams.numpy)
        generator.random.side_effect = lambda size: np.full(np.atleast_1d(size).astype(int), 0.1)
        monkeypatch.setattr(streams, "numpy", generator)
        monkeypatch.setattr(streams.python, "uniform", lambda lower, upper: 0.5)
        monkeypatch.setattr(streams.python, "shuffle", lambda values: None)
        monkeypatch.setattr(streams.python, "randrange", lambda stop: 0)
        monkeypatch.setattr(
            algorithm, "local_search_operator", lambda *args, **kwargs: next(solutions)
        )
        rewards = []
        monkeypatch.setattr(
            controller,
            "update_q_value",
            lambda st, at, reward, lf, df: rewards.append(reward),
        )
        with streams.activate():
            _, _, best = algorithm.solve(1.0)
        assert best == -150.0
        assert rewards == [1.0]
        return
    monkeypatch.setattr(
        module,
        "nelder_mead_search",
        lambda *args, **kwargs: next(solutions),
    )
    monkeypatch.setattr(
        controller,
        "update_q_value",
        lambda st, at, reward, lf, df: rewards.append(reward),
    )
    _, _, best = algorithm.solve(1.0)
    assert best == -150.0
    assert rewards == [1.0]
