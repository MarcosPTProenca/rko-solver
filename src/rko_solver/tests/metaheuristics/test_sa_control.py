"""Check actual SA adaptive actions, thermal schedules and offline compatibility.

Objective:
- Prevent configured online SA from silently executing its offline-only path.

Technical approach:
- Inspect real factory metadata and the existing Q controller under controlled actions.
- Bound temperature levels with local-search counts instead of timing assertions.
- Check an independent scalar trajectory for the unchanged offline implementation.

Flow (pseudocode):
1. Load configured online settings and validate named parameter axes.
2. Force two real controller transitions and observe proposals, temperatures and rewards.
3. Compare offline candidates and RNG consumption with a fixed scalar oracle.

Important criteria:
- A chosen t0 applies to the following cooling/reannealing schedule, not a mid-level reset.
- All chosen scalar parameters must reach the actual search operators.
- Invalid grids are rejected before a stochastic action is selected.

Inputs/Outputs:
- Input: small numerical objectives, owned streams and deterministic iteration bounds.
- Output: configuration, transition, reward and legacy-trace assertions.
"""

import importlib
import math
import numpy as np
import pytest
import random
from pathlib import Path
from rko_solver.components.adaptive_control.q_learning import QLearningController
from rko_solver.components.random_streams import RandomStreams
from rko_solver.components.solution_pool import ElitePool
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics import sa as sa_module
from rko_solver.metaheuristics.multi_start import MultiStart
from rko_solver.metaheuristics.sa import SA
from rko_solver.orchestration.factory import build_metaheuristic
from rko_solver.orchestration.types import CostFn
from rko_solver.orchestration.types import DecoderFn
from rko_solver.parameters.loader import load_online_grid
from unittest.mock import patch


def decode(keys: np.ndarray) -> np.ndarray:
    """Preserve continuous candidates for the numerical contract tests.

    Args:
        keys (np.ndarray): Candidate chromosome.

    Returns:
        np.ndarray: Unmodified candidate.
    """
    return keys


def cost(keys: np.ndarray) -> float:
    """Compute a deterministic finite minimization objective.

    Args:
        keys (np.ndarray): Candidate chromosome.

    Returns:
        float: Squared Euclidean norm.
    """
    return float(np.dot(keys, keys))


def test_rng_replay_rejects_offline_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not label an offline-only historical constructor as an online replay.

    Args:
        monkeypatch (pytest.MonkeyPatch): Expose the maintained benchmark's script directory.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    replay = importlib.import_module("random_streams")
    with pytest.raises(ValueError, match="adaptive control"):
        replay.replay_type(MultiStart, 1)
    offline = replay.replay_type(MultiStart, 0)(SolutionPool(), decode, cost, 2)
    online = replay.replay_type(SA, 1)(SolutionPool(), decode, cost, 2)
    assert offline.elapsed() == online.elapsed() == 0
    assert online.control == 1


def test_factory_enables_sa_control_and_records_grid() -> None:
    """Pass requested online control into SA rather than documenting a fallback."""
    algorithm = build_metaheuristic("SA", SolutionPool(), decode, cost, 2, 1, None)
    assert getattr(algorithm, "control", 0) == 1
    assert algorithm.parameters == load_online_grid("SA")
    assert algorithm._search_metadata is not None
    assert algorithm._search_metadata.control == 1
    assert algorithm._search_metadata.parameter_grid == tuple(
        tuple(axis) for axis in algorithm.parameters
    )


@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("local_improvement", [False, True])
def test_online_actions_reannealing_and_rewards(
    monkeypatch: pytest.MonkeyPatch, seed: int, local_improvement: bool
) -> None:
    """Exercise selected parameters, next-cycle t0 and signed-safe Q rewards.

    Args:
        monkeypatch (pytest.MonkeyPatch): Install action and stopping observations.
        seed (int): Owned stream with independently predicted initial state.
        local_improvement (bool): Whether refinement improves the global incumbent.
    """
    chosen: list[int] = []
    updates: list[tuple[int, int, float]] = []
    proposals: list[tuple[float, float]] = []
    exponents: list[float] = []
    refinements: list[np.ndarray] = []
    controllers: list[QLearningController] = []

    class Controller(QLearningController):
        """Observe real Q updates while forcing an admissible temperature-axis action.

        Responsibilities:
        - Keep the real state graph, schedule and reward update implementation.
        - Select the other t0 state first, then remain there for the second cycle.

        Attributes:
            target (int | None): Selected state label after the first action.
        """

        target: int | None = None

        def choose_action(self, st: int, epsilon: float) -> int:
            """Select a real distance-one action without consuming acceptance draws.

            Args:
                st (int): Current state label.
                epsilon (float): Actual scheduled exploration value.

            Returns:
                int: Action index reaching the controlled target.
            """
            assert 0 <= epsilon <= 1
            if self.target is None:
                self.target = 1 - st
            controllers.append(self)
            chosen.append(st)
            return self.states[st].Ai.index(self.target)

        def update_q_value(self, st: int, at: int, reward: float, lf: float, df: float) -> None:
            """Record finite transition credit before applying the real Bellman update.

            Args:
                st (int): Source state.
                at (int): Chosen action index.
                reward (float): Actual scalar search reward.
                lf (float): Scheduled learning factor.
                df (float): Scheduled discount factor.
            """
            assert math.isfinite(reward) and -1 <= reward <= 1
            updates.append((st, self.states[st].Ai[at], reward))
            super().update_q_value(st, at, reward, lf, df)

    def shake(keys: np.ndarray, beta_min: float, beta_max: float) -> np.ndarray:
        """Generate a worse proposal and record the selected perturbation interval.

        Args:
            keys (np.ndarray): Current vector.
            beta_min (float): Selected minimum perturbation.
            beta_max (float): Selected maximum perturbation.

        Returns:
            np.ndarray: Deliberately non-improving candidate.
        """
        proposals.append((beta_min, beta_max))
        return np.ones_like(keys)

    def refine(
        keys: np.ndarray, decoder: DecoderFn, objective: CostFn, *, pool: ElitePool | None = None
    ) -> tuple[np.ndarray, float]:
        """Bound execution and optionally produce a global local-search improvement.

        Args:
            keys (np.ndarray): Current candidate.
            decoder (DecoderFn): Active decoder.
            objective (CostFn): Active objective.
            pool (ElitePool | None): Shared context forwarded by SA.

        Returns:
            tuple[np.ndarray, float]: Independently scored refinement.
        """
        result = np.zeros_like(keys) if local_improvement else keys.copy()
        refinements.append(result)
        return result, objective(decoder(result))

    original_exp = math.exp

    def exponential(value: float) -> float:
        """Record actual Metropolis temperatures without changing acceptance math.

        Args:
            value (float): Negative objective delta divided by temperature.

        Returns:
            float: Ordinary exponential probability.
        """
        exponents.append(value)
        return original_exp(value)

    grid = [[2], [1.0, 8.0], [0.00001], [0.1], [0.2]]
    algorithm = SA(
        SolutionPool(),
        decode,
        cost,
        1,
        sa_max=1,
        t0=100.0,
        alpha=0.8,
        beta_min=0.01,
        beta_max=0.03,
        control=1,
        parameters=grid,
        local_search_operator=refine,
        shaking_operator=shake,
    )
    monkeypatch.setattr(sa_module, "QLearningController", Controller)
    monkeypatch.setattr(algorithm, "time_limit_reached", lambda duration: len(refinements) >= 2)
    monkeypatch.setattr(sa_module.math, "exp", exponential)
    streams = RandomStreams(seed)
    monkeypatch.setattr(streams.python, "random", lambda: 0.999)
    with streams.activate():
        result = algorithm.solve(60.0)
    initial = random.Random(seed).randint(0, 1)
    target = 1 - initial
    assert chosen == [initial, target]
    assert [item[:2] for item in updates] == [(initial, target), (target, target)]
    assert proposals == [(0.1, 0.2)] * 4
    initial_cost = cost(np.random.RandomState(seed).random_sample(1))
    first_t0 = grid[1][initial]
    next_t0 = grid[1][target] * 0.3
    second_cost = 0.0 if local_improvement else initial_cost
    assert exponents == pytest.approx(
        [-(1 - initial_cost) / first_t0] * 2 + [-(1 - second_cost) / next_t0] * 2
    )
    assert updates[0][2] == (
        1.0 if local_improvement else QLearningController.relative_reward(initial_cost, 1.0)
    )
    assert updates[1][2] == QLearningController.relative_reward(second_cost, 1.0)
    assert controllers[0].max_time == 60.0
    assert controllers[0].parameters == grid
    assert result[2] == cost(np.asarray(result[1]))


def test_offline_preserves_legacy_candidate_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the prior fixed-parameter trajectory and generator consumption exactly.

    Args:
        monkeypatch (pytest.MonkeyPatch): Bound temperature levels and forbid Q construction.
    """
    trace: list[np.ndarray] = []
    levels: list[float] = []

    def observe(keys: np.ndarray) -> float:
        """Record independently detached callback candidates.

        Args:
            keys (np.ndarray): Candidate chromosome.

        Returns:
            float: Its squared norm.
        """
        trace.append(keys.copy())
        return cost(keys)

    def shake(keys: np.ndarray, beta_min: float, beta_max: float) -> np.ndarray:
        """Apply a deterministic improvement with the original fixed interval.

        Args:
            keys (np.ndarray): Current chromosome.
            beta_min (float): Offline minimum.
            beta_max (float): Offline maximum.

        Returns:
            np.ndarray: Reduced chromosome.
        """
        assert (beta_min, beta_max) == (0.1, 0.2)
        return keys * 0.8

    def refine(
        keys: np.ndarray, decoder: DecoderFn, objective: CostFn, *, pool: ElitePool | None = None
    ) -> tuple[np.ndarray, float]:
        """Apply one deterministic refinement per completed temperature level.

        Args:
            keys (np.ndarray): Current chromosome.
            decoder (DecoderFn): Original decoder.
            objective (CostFn): Observed objective.
            pool (ElitePool | None): Forwarded shared context.

        Returns:
            tuple[np.ndarray, float]: Refined keys and measured norm.
        """
        levels.append(1.0)
        result = keys * 0.5
        return result, objective(decoder(result))

    def forbidden(*args: object, **kwargs: object) -> None:
        """Reject any controller construction in the legacy offline path.

        Args:
            args (object): Unexpected positional settings.
            kwargs (object): Unexpected controller settings.

        Raises:
            AssertionError: Always, because offline SA must not construct Q state.
        """
        raise AssertionError("offline SA constructed Q-learning")

    algorithm = SA(
        SolutionPool(),
        decode,
        observe,
        2,
        sa_max=2,
        t0=1.0,
        alpha=0.5,
        beta_min=0.1,
        beta_max=0.2,
        local_search_operator=refine,
        shaking_operator=shake,
    )
    monkeypatch.setattr(algorithm, "time_limit_reached", lambda duration: len(levels) >= 2)
    monkeypatch.setattr(sa_module, "QLearningController", forbidden, raising=False)
    streams = RandomStreams(42)
    with streams.activate():
        result = algorithm.solve(60.0)
    expected_numpy = np.random.RandomState(42)
    original = expected_numpy.random_sample(2)
    expected = [original]
    current = original.copy()
    for _ in range(2):
        for _ in range(2):
            current = current * 0.8
            expected.append(current.copy())
        current = current * 0.5
        expected.append(current.copy())
    assert np.array_equal(np.asarray(trace), np.asarray(expected))
    assert np.array_equal(result[1], current)
    assert result[2] == cost(current)
    assert streams.python.getstate() == random.Random(42).getstate()
    assert np.array_equal(streams.numpy.get_state()[1], expected_numpy.get_state()[1])
    assert streams.numpy.get_state()[2:] == expected_numpy.get_state()[2:]


def test_offline_metropolis_trace_and_cooling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept two worse moves, then reject the last one at the cooled temperature.

    Args:
        monkeypatch (pytest.MonkeyPatch): Bound levels and observe acceptance exponents.
    """
    observed: list[float] = []
    starts: list[float] = []
    levels: list[float] = []
    exponents: list[float] = []
    proposals = iter([0.5, 0.9, 0.1, 0.8])

    def objective(keys: np.ndarray) -> float:
        """Score a single scalar and retain the complete legacy callback trace.

        Args:
            keys (np.ndarray): One random key.

        Returns:
            float: Linear minimization value.
        """
        observed.append(float(keys[0]))
        return float(keys[0])

    def shake(keys: np.ndarray, beta_min: float, beta_max: float) -> np.ndarray:
        """Return a predetermined neighbor after recording the actual current state.

        Args:
            keys (np.ndarray): Current state, including accepted deteriorations.
            beta_min (float): Fixed lower intensity.
            beta_max (float): Fixed upper intensity.

        Returns:
            np.ndarray: Next scalar proposal.
        """
        assert (beta_min, beta_max) == (0.1, 0.2)
        starts.append(float(keys[0]))
        return np.array([next(proposals)])

    def refine(
        keys: np.ndarray, decoder: DecoderFn, objective: CostFn, *, pool: ElitePool | None = None
    ) -> tuple[np.ndarray, float]:
        """Rescore the current state without perturbing legacy acceptance.

        Args:
            keys (np.ndarray): Current state.
            decoder (DecoderFn): Active decoder.
            objective (CostFn): Observed scalar cost.
            pool (ElitePool | None): Shared context.

        Returns:
            tuple[np.ndarray, float]: Current state and its measured cost.
        """
        levels.append(float(keys[0]))
        return keys.copy(), objective(decoder(keys))

    exponential = math.exp

    def observe_exp(value: float) -> float:
        """Record the exact delta/temperature input without changing its probability.

        Args:
            value (float): Negative scaled deterioration.

        Returns:
            float: Ordinary Metropolis probability.
        """
        exponents.append(value)
        return exponential(value)

    algorithm = SA(
        SolutionPool(),
        decode,
        objective,
        1,
        sa_max=2,
        t0=1.0,
        alpha=0.5,
        beta_min=0.1,
        beta_max=0.2,
        local_search_operator=refine,
        shaking_operator=shake,
    )
    monkeypatch.setattr(algorithm, "time_limit_reached", lambda duration: len(levels) >= 2)
    monkeypatch.setattr(sa_module.math, "exp", observe_exp)
    streams = RandomStreams(42)
    with streams.activate():
        result = algorithm.solve(60.0)
    expected_numpy = np.random.RandomState(42)
    initial = float(expected_numpy.random_sample(1)[0])
    assert observed == [initial, 0.5, 0.9, 0.9, 0.1, 0.8, 0.1]
    assert starts == [initial, 0.5, 0.9, 0.1]
    assert exponents == pytest.approx([-(0.5 - initial), -0.4, -0.7 / 0.5])
    expected_python = random.Random(42)
    for _ in range(3):
        expected_python.random()
    assert streams.python.getstate() == expected_python.getstate()
    assert np.array_equal(streams.numpy.get_state()[1], expected_numpy.get_state()[1])
    assert streams.numpy.get_state()[2:] == expected_numpy.get_state()[2:]
    assert result[1:] == ([0.1], 0.1)


def test_mixed_beta_grid_filters_infeasible_states() -> None:
    """Preserve named-axis meanings while excluding reversed online shaking bounds."""
    algorithm = SA(
        SolutionPool(),
        decode,
        cost,
        2,
        control=1,
        parameters=[[1], [1.0], [0.9], [0.1, 0.4], [0.2, 0.5]],
    )
    with RandomStreams(42).activate():
        controller = QLearningController(algorithm.parameters, state_filter=algorithm._state_filter)
    assert [state.par[-2:] for state in controller.states] == [[0.1, 0.2], [0.1, 0.5], [0.4, 0.5]]


def test_empty_temperature_level_does_not_update_q(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid nonfinite level rewards if stopping occurs before any proposal.

    Args:
        monkeypatch (pytest.MonkeyPatch): Stop the first Metropolis loop before work.
    """

    def refine(
        keys: np.ndarray, decoder: DecoderFn, objective: CostFn, *, pool: ElitePool | None = None
    ) -> tuple[np.ndarray, float]:
        """Keep a finite incumbent without generating a Metropolis candidate.

        Args:
            keys (np.ndarray): Current chromosome.
            decoder (DecoderFn): Active decoder.
            objective (CostFn): Active objective.
            pool (ElitePool | None): Shared context.

        Returns:
            tuple[np.ndarray, float]: Unchanged keys and their norm.
        """
        return keys, objective(decoder(keys))

    algorithm = SA(
        SolutionPool(),
        decode,
        cost,
        2,
        control=1,
        parameters=[[1], [1.0], [0.9], [0.1], [0.2]],
        local_search_operator=refine,
    )
    checks = iter([False, False, True, True, True])
    monkeypatch.setattr(algorithm, "time_limit_reached", lambda duration: next(checks))
    with (
        RandomStreams(42).activate(),
        patch.object(QLearningController, "update_q_value") as update,
    ):
        result = algorithm.solve(60.0)
    update.assert_not_called()
    assert result[2] == cost(np.asarray(result[1]))


@pytest.mark.parametrize(
    "control, grid",
    [
        (True, [[1], [1.0], [0.5], [0.1], [0.2]]),
        (2, [[1], [1.0], [0.5], [0.1], [0.2]]),
        (1, [[1], [1.0], [0.5], [0.1]]),
        (1, [[1], [], [0.5], [0.1], [0.2]]),
        (1, [[1.5], [1.0], [0.5], [0.1], [0.2]]),
        (1, [[1], [float("inf")], [0.5], [0.1], [0.2]]),
        (1, [[1], [1.0], [1.0], [0.1], [0.2]]),
        (1, [[1], [1.0], [0.5], [0.3], [0.2]]),
    ],
)
def test_invalid_online_settings_fail_before_search(control: int, grid: list[list[float]]) -> None:
    """Reject malformed scalar or joint states rather than truncating chosen values.

    Args:
        control (int): Invalid flag or adaptive mode.
        grid (list[list[float]]): Candidate named-axis parameter values.
    """
    with pytest.raises(ValueError):
        SA(SolutionPool(), decode, cost, 2, control=control, parameters=grid)
