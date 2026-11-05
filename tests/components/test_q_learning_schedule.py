"""Tests for the progress-driven Q-learning schedule and its scale-free reward.

Objective:
- Pin the epsilon trace, learning factor and reward to the values of the Rust engine.

Technical approach:
- Sample epsilon every 0.05 of progress for horizons of 2, 10 and 100 seconds.
- Drive the schedule from evaluation counts alone with no clock.
- Compare the default (legacy) mode against the earlier whole-second windows and fixed-scale reward.

Flow (pseudocode):
1. Build a controller and sample the schedule at 21 progress points.
2. Compare each sample with the trace recorded in the Rust unit tests.
3. Compare rewards at very different cost scales.

Important criteria:
- The progress trace is identical for every horizon and equals the Rust trace.
- The penalty depends only on the relative worsening, never on the cost scale.
- `progress_schedule=False` (the default) reproduces the earlier trace and reward exactly.

Inputs/Outputs:
- Input: horizons, progress points and cost pairs.
- Output: assertions over epsilon, learning factor and reward.
"""

from __future__ import annotations

import pytest
from rko_solver.components.adaptive_control.q_learning import QLearningController

PROGRESS_TRACE = [
    1.0,
    0.55,
    0.9,
    0.5,
    0.8,
    0.45,
    0.7,
    0.4,
    0.6,
    0.35,
    0.5,
    0.3,
    0.4,
    0.25,
    0.3,
    0.2,
    0.2,
    0.15,
    0.1,
    0.1,
    0.1,
]
LEGACY_TRACE_10 = [
    1.0,
    1.0,
    0.9,
    0.9,
    0.8,
    0.8,
    0.7,
    0.7,
    0.6,
    0.6,
    0.5,
    0.5,
    0.4,
    0.4,
    0.3,
    0.3,
    0.2,
    0.2,
    0.1,
    0.1,
    0.1,
]


def trace(horizon: float, progress: bool) -> list[float]:
    """Sample epsilon every 0.05 of the horizon.

    Args:
        horizon (float): Time horizon in seconds.
        progress (bool): Whether to use the progress schedule instead of the legacy windows.

    Returns:
        list[float]: Epsilon at 21 equally spaced points of progress.
    """
    controller = QLearningController([[1.0]], max_time=horizon)
    controller.progress_schedule = progress
    samples = []
    for step in range(21):
        share = step / 20
        samples.append(controller.set_parameters(horizon * share, share)[0])
    return samples


@pytest.mark.parametrize("horizon", [2.0, 10.0, 100.0])
def test_progress_schedule_is_the_same_for_every_horizon(horizon: float) -> None:
    """Match the Rust epsilon trace regardless of the time horizon.

    Args:
        horizon (float): Time horizon in seconds.
    """
    assert trace(horizon, progress=True) == pytest.approx(PROGRESS_TRACE, abs=1e-9)


def test_progress_schedule_follows_evaluation_counts_without_a_clock() -> None:
    """Reproduce the trace from progress alone, with the elapsed time held at zero."""
    controller = QLearningController([[1.0]], max_time=1000.0)
    controller.progress_schedule = True
    counted = [controller.set_parameters(0.0, (step * 20) / 400)[0] for step in range(21)]
    assert counted == pytest.approx(PROGRESS_TRACE, abs=1e-9)
    assert controller.set_parameters(0.0, 1.0)[1] == pytest.approx(0.1)
    assert controller.set_parameters(0.0, 7.0)[1] == pytest.approx(0.1)


def test_progress_defaults_to_the_time_share() -> None:
    """Derive progress from elapsed time and horizon when none is supplied."""
    controller = QLearningController([[1.0]], max_time=10.0)
    controller.progress_schedule = True
    assert controller.set_parameters(5.0)[1] == pytest.approx(0.55)


def test_legacy_schedule_keeps_the_whole_second_window() -> None:
    """Reproduce the earlier trace: ten flat samples per second-wide window at 2 s."""
    two = trace(2.0, progress=False)
    assert two[:10] == pytest.approx([1.0] * 10)
    assert two[10:20] == pytest.approx([0.9] * 10)
    assert trace(10.0, progress=False) == pytest.approx(LEGACY_TRACE_10, abs=1e-9)
    assert trace(100.0, progress=False) == pytest.approx(PROGRESS_TRACE, abs=1e-9)


def test_legacy_schedule_ignores_progress() -> None:
    """Keep the earlier learning factor, which depends on elapsed time only."""
    controller = QLearningController([[1.0]], max_time=10.0)
    controller.progress_schedule = False
    assert controller.set_parameters(5.0, 1.0)[1] == pytest.approx(0.55)


def test_penalty_does_not_depend_on_cost_scale() -> None:
    """Give the same penalty for the same relative worsening at 1e-6 and 1e6."""
    small = QLearningController.progress_reward(1e-6, 1.5e-6)
    large = QLearningController.progress_reward(1e6, 1.5e6)
    assert small == pytest.approx(large, abs=1e-12)
    assert small == pytest.approx(-1 / 3, abs=1e-12)
    assert QLearningController.relative_reward(1e-6, 1.5e-6) == pytest.approx(-5e-7, abs=1e-12)


def test_reward_rules() -> None:
    """Cover improvement, equality, zero costs and the lower bound."""
    assert QLearningController.progress_reward(10.0, 9.0) == 1.0
    assert QLearningController.progress_reward(10.0, 10.0) == 0.0
    assert QLearningController.progress_reward(0.0, 0.0) == 0.0
    assert QLearningController.progress_reward(0.0, 5.0) == -1.0
    assert QLearningController.progress_reward(-100.0, -50.0) < 0.0
    assert QLearningController.progress_reward(1e308, -1e308) == 1.0
    assert QLearningController.progress_reward(-1e308, 1e308) == -1.0


def test_legacy_reward_keeps_the_fixed_scale() -> None:
    """Keep the earlier reward as the default and select the scale-free one through the flag."""
    assert QLearningController.relative_reward(0.0, 0.5) == -0.5
    assert QLearningController.relative_reward(1.0, 3.0) == pytest.approx(-2 / 3)
    legacy = QLearningController([[1.0]])
    progress = QLearningController([[1.0]])
    progress.progress_schedule = True
    assert legacy.reward(0.0, 0.5) == -0.5
    assert progress.reward(0.0, 0.5) == -1.0
