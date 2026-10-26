"""Preserve GRASP's reference alpha axis without inventing construction semantics.

Objective:
- Retain the three-axis reference grid while distinguishing construction from policy.

Technical approach:
- Compare real construction traces under different singleton alpha labels.
- Build actual controllers from packaged and collapsed alpha grids.

Flow (pseudocode):
1. Repeat construction with identical owned streams and different alpha labels.
2. Compare callbacks, returned chromosomes, costs and final stream states.
3. Check controller states/actions and initialization draws with a smaller alpha axis.

Important criteria:
- Alpha does not influence the constructive kernel directly.
- Removing alpha values changes the adaptive state graph and cannot be silent.
- No external C++ checkout or timing measurement is required for these checks.

Inputs/Outputs:
- Input: valid three-axis GRASP grids and deterministic numerical callbacks.
- Output: construction equivalence and adaptive-graph compatibility assertions.
"""

import numpy as np
import pickle
from rko_solver.components.adaptive_control.q_learning import QLearningController
from rko_solver.components.random_streams import RandomStreams
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.grasp import GRASP
from rko_solver.parameters.loader import load_online_grid
from rko_solver.parameters.loader import load_online_named


def test_alpha_labels_do_not_change_construction() -> None:
    """Match real line-search callbacks and streams when only an alpha label changes."""
    results: list[tuple[bytes, float, list[bytes], bytes]] = []
    for alpha in (0.1, 0.9):
        trace: list[bytes] = []

        def objective(keys: np.ndarray, observations: list[bytes] = trace) -> float:
            """Record the evaluated chromosome and return its squared norm.

            Args:
                keys (np.ndarray): Actual construction candidate.
                observations (list[bytes]): Trace bound to this construction run.

            Returns:
                float: Deterministic squared norm.
            """
            observations.append(keys.tobytes())
            return float(np.dot(keys, keys))

        algorithm = GRASP(
            SolutionPool(),
            lambda keys: keys,
            objective,
            4,
            control=1,
            parameters=[[alpha], [0.25], [0.125]],
        )
        streams = RandomStreams(17)
        with streams.activate():
            keys, cost = algorithm._constructive(0.25, np.full(4, 0.75))
        state = pickle.dumps((streams.python.getstate(), streams.numpy.get_state()))
        results.append((keys.tobytes(), cost, trace, state))
    assert results[0] == results[1]
    assert len(results[0][2]) > 1


def test_reference_alpha_axis_changes_controller_graph() -> None:
    """Retain real alpha-only actions and their initialization draws in packaged grids."""
    grid = load_online_grid("GRASP")
    assert tuple(load_online_named("GRASP")) == ("alpha_grasp", "hs", "he")
    assert len(grid) == 3 and len(grid[0]) > 1
    collapsed = [[grid[0][0]], grid[1].copy(), grid[2].copy()]
    controllers: list[QLearningController] = []
    states: list[bytes] = []
    for parameters in (grid, collapsed):
        algorithm = GRASP(
            SolutionPool(),
            lambda keys: keys,
            lambda keys: 0.0,
            4,
            control=1,
            parameters=parameters,
        )
        streams = RandomStreams(23)
        with streams.activate():
            controller = QLearningController(
                parameters,
                state_filter=algorithm._state_filter,
            )
        expected_count = len(parameters[0]) * len(parameters[1]) * len(parameters[2])
        expected_actions = 1 + sum(len(axis) - 1 for axis in parameters)
        assert len(controller.states) == expected_count
        assert all(len(state.Ai) == expected_actions for state in controller.states)
        controllers.append(controller)
        states.append(pickle.dumps(streams.python.getstate()))
    reference, singleton = controllers
    assert len(reference.states) > len(singleton.states)
    assert states[0] != states[1]
    assert any(
        state.par[0] != reference.states[target].par[0]
        and state.par[1:] == reference.states[target].par[1:]
        for state in reference.states
        for target in state.Ai
    )
