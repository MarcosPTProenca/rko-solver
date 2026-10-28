"""Check GRASP step decay and initial-vector handling against the reference loop.

Objective:
- Keep the current grid step when a construction improves the global best.
- Score the initial random vector without spending a refinement on it.

Technical approach:
- Drive ``solve`` with a scripted construction and a deterministic objective.
- Record the step sequence, refinement inputs and applied Q-table updates.
- Replace local search with an identity operator so only the loop is observed.

Flow (pseudocode):
1. Patch owned RNG draws to a known initial vector and stop by request count.
2. Script construction outcomes that improve, then tie or worsen.
3. Assert the step sequence, initial pool entry and Q-table progression.

Important criteria:
- Improvement must not reset h; non-improvement halves it.
- h restarts from hs only when a new Q state or outer iteration begins.
- No refinement receives the initial random vector.
- Bounded improvements still reach a Q-table update.

Inputs/Outputs:
- Input: controlled owned RNGs, scripted candidates, offline or adaptive control.
- Output: step/refinement/update assertions or the original callback error.
"""

import numpy as np
import pytest
from rko_solver.components.adaptive_control.q_learning import QLearningController
from rko_solver.components.random_streams import RandomStreams
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.grasp import GRASP
from unittest.mock import Mock


def decode(keys: np.ndarray) -> np.ndarray:
    """Preserve continuous random keys for these domain-neutral objectives.

    Args:
        keys (np.ndarray): Candidate chromosome.

    Returns:
        np.ndarray: Unmodified payload.
    """
    return keys


def objective(keys: np.ndarray) -> float:
    """Prefer larger first coordinates so scripted candidates rank predictably.

    Args:
        keys (np.ndarray): Numerical chromosome.

    Returns:
        float: Negative first coordinate.
    """
    return -float(keys[0])


def identity(
    keys: np.ndarray, decoder: object, cost: object, *, pool: object | None = None
) -> tuple[np.ndarray, float]:
    """Return the candidate unchanged so only the outer loop is exercised.

    Args:
        keys (np.ndarray): Candidate chromosome.
        decoder (object): Ordinary decoder.
        cost (object): Numerical objective.
        pool (object | None): Shared elite context.

    Returns:
        tuple[np.ndarray, float]: Unchanged candidate and its objective value.
    """
    return keys, cost(decoder(keys))


def _controlled(
    monkeypatch: pytest.MonkeyPatch, streams: RandomStreams, start: float = 0.5
) -> None:
    """Force one initial vector and one construction subset under owned streams.

    Args:
        monkeypatch (pytest.MonkeyPatch): Applies the stream patches.
        streams (RandomStreams): Owned streams used by the solver.
        start (float): Coordinate value of the initial random vector.
    """
    generator = Mock(wraps=streams.numpy)
    generator.random.side_effect = lambda size: np.full(size, start)
    monkeypatch.setattr(streams, "numpy", generator)
    monkeypatch.setattr(streams.python, "uniform", lambda lower, upper: 0.5)
    monkeypatch.setattr(streams.python, "shuffle", lambda values: None)
    monkeypatch.setattr(streams.python, "randrange", lambda stop: 0)


def _script(algorithm: GRASP, monkeypatch: pytest.MonkeyPatch, values: list[float]) -> list[float]:
    """Replace construction with a recorded sequence of first-coordinate values.

    Args:
        algorithm (GRASP): Solver whose construction is replaced.
        monkeypatch (pytest.MonkeyPatch): Applies the replacement.
        values (list[float]): Construction outcomes consumed in call order.

    Returns:
        list[float]: Recorded grid steps, appended on each construction call.
    """
    steps: list[float] = []
    queue = list(values)

    def constructive(step: float, keys: np.ndarray) -> tuple[np.ndarray, float]:
        """Return the scripted candidate and record the requested grid step.

        Args:
            step (float): Current grid step h.
            keys (np.ndarray): Incumbent chromosome used as the template.

        Returns:
            tuple[np.ndarray, float]: Scripted candidate and objective value.
        """
        steps.append(step)
        value = queue.pop(0) if queue else 0.0
        candidate = keys.copy()
        candidate[0] = value
        return candidate, objective(candidate)

    monkeypatch.setattr(algorithm, "_constructive", constructive)
    return steps


def test_improving_construction_keeps_current_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Halve the grid only after a construction that fails to improve the best.

    Args:
        monkeypatch (pytest.MonkeyPatch): Controls RNG draws, construction and stopping.
    """
    streams = RandomStreams(1234)
    _controlled(monkeypatch, streams)
    algorithm = GRASP(
        SolutionPool(), decode, objective, 1, hs=0.5, he=0.125, local_search_operator=identity
    )
    steps = _script(algorithm, monkeypatch, [0.9, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(algorithm, "time_limit_reached", lambda duration: len(steps) >= 5)
    with streams.activate():
        result = algorithm.solve(60.0)
    assert steps == [0.5, 0.5, 0.25, 0.125, 0.5]
    assert result[1:] == ([0.9], -0.9)


def test_initial_vector_is_scored_without_refinement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Publish the raw initial observation instead of spending RVND on it.

    Args:
        monkeypatch (pytest.MonkeyPatch): Controls RNG draws, refinement and stopping.
    """
    refinements: list[float] = []

    def refine(
        keys: np.ndarray,
        decoder: object,
        cost: object,
        *,
        pool: object | None = None,
    ) -> tuple[np.ndarray, float]:
        """Record refined first coordinates without changing the candidate.

        Args:
            keys (np.ndarray): Candidate chromosome.
            decoder (object): Ordinary decoder.
            cost (object): Numerical objective.
            pool (object | None): Shared elite context.

        Returns:
            tuple[np.ndarray, float]: Unchanged candidate and its objective value.
        """
        refinements.append(float(keys[0]))
        return keys, cost(decoder(keys))

    streams = RandomStreams(99)
    _controlled(monkeypatch, streams)
    pool = SolutionPool()
    algorithm = GRASP(pool, decode, objective, 1, hs=0.5, he=0.25, local_search_operator=refine)
    steps = _script(algorithm, monkeypatch, [0.75, 0.75, 0.75])
    monkeypatch.setattr(algorithm, "time_limit_reached", lambda duration: len(steps) >= 3)
    with streams.activate():
        algorithm.solve(60.0)
    assert 0.5 not in refinements
    assert refinements == [0.75, 0.75, 0.75]
    assert pool.best == (-0.75, [0.75])


def test_fine_step_improvement_does_not_restart_the_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Continue from the improving fine step until the grid is exhausted.

    Args:
        monkeypatch (pytest.MonkeyPatch): Controls RNG draws and construction outcome.
    """
    updates: list[tuple[int, int]] = []
    original = QLearningController.update_q_value
    improved = False

    def update_q_value(
        controller: QLearningController,
        state: int,
        action: int,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Record one applied Q-table update and delegate to the real method.

        Args:
            controller (QLearningController): Owning adaptive controller.
            state (int): Current state index.
            action (int): Chosen action index.
            *args: Reward, learning factor and discount factor.
            **kwargs: Forwarded keyword arguments.
        """
        updates.append((state, action))
        original(controller, state, action, *args, **kwargs)

    monkeypatch.setattr(QLearningController, "update_q_value", update_q_value)
    streams = RandomStreams(7)
    _controlled(monkeypatch, streams)
    algorithm = GRASP(
        SolutionPool(),
        decode,
        objective,
        1,
        control=1,
        parameters=[[0.5], [0.5], [0.125]],
        local_search_operator=identity,
    )
    steps: list[float] = []

    def constructive(step: float, keys: np.ndarray) -> tuple[np.ndarray, float]:
        """Improve exactly once at the second grid level, then return a worse point.

        Args:
            step (float): Current grid step h.
            keys (np.ndarray): Incumbent chromosome.

        Returns:
            tuple[np.ndarray, float]: Scripted candidate and objective value.
        """
        nonlocal improved
        steps.append(step)
        candidate = keys.copy()
        if step == 0.25 and not improved:
            improved = True
            candidate[0] = 0.9
        else:
            candidate[0] = 0.0
        return candidate, objective(candidate)

    monkeypatch.setattr(algorithm, "_constructive", constructive)
    monkeypatch.setattr(algorithm, "time_limit_reached", lambda duration: len(steps) >= 4)
    with streams.activate():
        algorithm.solve(60.0)
    assert steps == [0.5, 0.25, 0.25, 0.125]
    assert len(updates) == 1
