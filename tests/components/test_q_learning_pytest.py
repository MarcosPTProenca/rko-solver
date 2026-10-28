"""Tests for the adaptive Q-learning controller.

Objective:
- Verify state creation, action policy, parameter scheduling, and Q-updates.

Technical approach:
- Build minimal parameter grids and assert controller invariants.
- Use deterministic random monkeypatching for policy branches.

Flow (pseudocode):
1. Create controller/states from small grids.
2. Force exploitation branch and validate selected action.
3. Update Q-values and validate changed estimates.

Important criteria:
- Cartesian state space is generated from the discrete grid.
- Exploration floor is respected by dynamic parameter updates.

Inputs/Outputs:
- Input: small parameter grids and synthetic rewards.
- Output: assertions over controller transitions and state values.
"""

from __future__ import annotations

import pytest
from rko_solver.components.adaptive_control.q_learning import QLearningController
from rko_solver.components.adaptive_control.q_learning import create_states


def test_create_states_builds_cartesian_space() -> None:
    """Ensure state construction follows the cartesian product of parameters."""
    states = create_states([[10.0, 20.0], [0.1, 0.2]])
    assert len(states) == 4
    assert all(len(st.Ai) > 0 for st in states)


def test_choose_action_exploitation_prefers_best(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure epsilon-greedy exploitation selects the state's best action."""
    controller = QLearningController(parameters=[[1.0], [2.0]])
    controller.states[0].maxA = 0
    monkeypatch.setattr(
        "rko_solver.components.adaptive_control.q_learning.random.random", lambda: 0.0
    )

    action = controller.choose_action(st=0, epsilon=0.2)
    assert action == 0


def test_update_q_value_changes_state_when_reward_is_positive() -> None:
    """Ensure Bellman update modifies Q-value with positive reward."""
    controller = QLearningController(parameters=[[1.0], [2.0]])
    state = controller.states[0]
    initial_q = state.Qa[0]

    controller.update_q_value(st=0, at=0, reward=1.0, lf=0.8, df=0.8)

    assert state.Qa[0] != initial_q
    assert state.maxQ >= state.Qa[0]


def test_set_parameters_respects_epsilon_floor() -> None:
    """Ensure epsilon never drops below the configured minimum."""
    controller = QLearningController(
        parameters=[[1.0]],
        epsilon_max=0.2,
        epsilon_min=0.1,
        restart_epsilon=1,
        max_time=10.0,
    )

    epsilon, *_ = controller.set_parameters(current_time=10.0)
    assert epsilon >= 0.1
