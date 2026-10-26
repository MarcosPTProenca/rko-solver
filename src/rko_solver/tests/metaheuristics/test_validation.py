"""Algorithm parameter domains and feasible adaptive-state contracts.

Objective:
- Reject invalid settings before execution and exclude impossible joint Q states.

Technical approach:
- Exercise direct constructors, factory casts and the installed default grids.

Flow (pseudocode):
1. Supply invalid scalar or coupled settings.
2. Build controllers from real named parameter grids.
3. Verify early errors, feasible states and consistent real online execution.

Important criteria:
- Do not silently truncate fractional population settings or rename axes.
- Filtering infeasible combinations preserves valid axis values and labels.

Inputs/Outputs:
- Input: constructor settings, temporary TOMLs and packaged defaults.
- Output: domain and adaptive-state assertions.
"""

import multiprocessing as mp
import numpy as np
import pytest
from math import prod
from pathlib import Path
from rko_solver import RKO
from rko_solver.components.adaptive_control.q_learning import QLearningController
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.ga import GeneticAlgorithm
from rko_solver.orchestration.configuration import MH_SPECS
from rko_solver.orchestration.configuration import parse_run_config
from rko_solver.orchestration.factory import build_metaheuristic


@pytest.mark.parametrize(
    "name, settings",
    [
        ("GA", {"size_pop": 2}),
        ("GA", {"size_pop": 3.5}),
        ("GA", {"prob_cros": True}),
        ("GA", {"prob_mut": -0.1}),
        ("SA", {"alpha": 1.0}),
        ("SA", {"t0": 0.0}),
        ("ILS", {"beta_min": 0.5, "beta_max": 0.1}),
        ("LNS", {"alpha_lns": float("nan")}),
        ("GRASP", {"he": 0.5, "hs": 0.1}),
        ("PSO", {"p_size": 0}),
        ("PSO", {"w": -1}),
        ("BRKGA", {"p": 10, "pe": 0.8, "pm": 0.3}),
        ("BRKGA_CS", {"pe": 1.0}),
        ("BRKGA_CS", {"sigma": 0.0}),
        ("GA", {"control": 1, "parameters": []}),
        ("GA", {"control": 1, "parameters": [[2.0], [0.8], [0.02]]}),
        ("ILS", {"control": 1, "parameters": [[0.5], [0.1]]}),
    ],
)
def test_constructor_rejects_invalid_domains(name: str, settings: dict[str, object]) -> None:
    """Reject invalid settings rather than crashing during a random transition.

    Args:
        name (str): Canonical algorithm name.
        settings (dict[str, object]): Invalid constructor arguments.
    """
    with pytest.raises(ValueError):
        MH_SPECS[name].constructor(
            SolutionPool(),
            lambda keys: keys,
            lambda keys: float(np.sum(keys)),
            3,
            **settings,
        )


@pytest.mark.parametrize("value", [3.9, True, "30", float("inf")])
def test_factory_does_not_coerce_invalid_integers(tmp_path: Path, value: object) -> None:
    """Reject fractional, boolean, string and nonfinite population values.

    Args:
        tmp_path (Path): Temporary parameter root.
        value (object): Invalid parsed population value.
    """
    literal = repr(value).lower() if not isinstance(value, str) else '"30"'
    (tmp_path / "ga.toml").write_text(
        f"[offline]\nsize_pop={literal}\nprob_cros=0.8\nprob_mut=0.02\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="size_pop"):
        build_metaheuristic(
            "GA", SolutionPool(), lambda keys: keys, lambda keys: 0.0, 3, 0, tmp_path
        )


@pytest.mark.parametrize("setting", ["strategy", "max_runs", "debug", "poolsize"])
def test_unsupported_run_keys_are_not_silently_ignored(setting: str) -> None:
    """Reject both formerly ignored settings and misspelled keys.

    Args:
        setting (str): Unsupported configuration key.
    """
    with pytest.raises(ValueError, match=setting):
        parse_run_config(
            {
                "run": {"control": 0, "pool_size": 2, setting: 1},
                "active_metaheuristics": {"list": ["SA"]},
            }
        )


def test_feasible_online_states_keep_correct_labels_and_edges() -> None:
    """Exclude reversed perturbation bounds without changing valid axis values."""
    algorithm = build_metaheuristic(
        "ILS",
        SolutionPool(),
        lambda keys: keys,
        lambda keys: 0.0,
        3,
        1,
        None,
    )
    controller = QLearningController(algorithm.parameters, state_filter=algorithm._state_filter)
    assert 0 < len(controller.states) < prod(map(len, algorithm.parameters))
    for index, state in enumerate(controller.states):
        assert state.label == index
        assert state.par[0] <= state.par[1]
        assert all(0 <= destination < len(controller.states) for destination in state.Ai)
        assert len(state.Ai) == len(state.Qa)


@pytest.mark.parametrize("name", [name for name, spec in MH_SPECS.items() if spec.supports_control])
def test_actual_online_portfolio_returns_consistent_result(name: str) -> None:
    """Execute each adaptive algorithm through its budget-aware class API.

    Args:
        name (str): Adaptive algorithm to execute with packaged defaults.
    """
    result = RKO(
        lambda keys: keys,
        lambda keys: float(np.sum(keys**2)),
        4,
        algorithms=[name],
        control=1,
        execution="sequential",
        random_state=42,
    ).solve(0.02)
    assert result.cost == pytest.approx(float(np.sum(np.asarray(result.keys) ** 2)))
    assert np.isfinite(result.cost)


def test_invalid_process_configuration_fails_before_spawning(tmp_path: Path) -> None:
    """Validate parameters in the parent before starting a manager or search worker.

    Args:
        tmp_path (Path): Invalid GA parameter directory.
    """
    (tmp_path / "config.toml").write_text(
        '[run]\ncontrol=0\npool_size=2\n[active_metaheuristics]\nlist=["GA"]\n',
        encoding="utf-8",
    )
    (tmp_path / "ga.toml").write_text(
        "[offline]\nsize_pop=2\nprob_cros=0.8\nprob_mut=0.02\n",
        encoding="utf-8",
    )
    before = {child.pid for child in mp.active_children()}
    with pytest.raises(ValueError, match="size_pop"):
        RKO(lambda keys: keys, lambda keys: 0.0, 3, params_dir=tmp_path).solve(0.01)
    assert {child.pid for child in mp.active_children()} == before


def test_direct_ga_population_error_is_immediate() -> None:
    """Reject the audited two-member tournament before any solve starts."""
    with pytest.raises(ValueError, match="size_pop"):
        GeneticAlgorithm(SolutionPool(), lambda keys: keys, lambda keys: 0.0, 3, size_pop=2)
