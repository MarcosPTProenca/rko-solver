"""Programmatic offline parameter ownership and execution propagation.

Objective:
- Tune named algorithms without mutating installed parameter resources.

Technical approach:
- Compare factory attributes with validated Python overrides.
- Repair deliberately invalid temporary defaults through every execution mode.

Flow (pseudocode):
1. Supply partial named overrides and retain unspecified TOML defaults.
2. Reject unknown, ambiguous or invalid settings before execution.
3. Verify fresh snapshots and spawned/grouped worker propagation.

Important criteria:
- Adaptive grids must not silently replace explicitly requested scalar settings.
- Mapping proxies must become ordinary owned dictionaries before spawn.

Inputs/Outputs:
- Input: partial scalar mappings, aliases and temporary TOML fixtures.
- Output: validated construction, execution and input/resource ownership assertions.
"""

import numpy as np
import pickle
import pytest
import rko_solver.solver as solver_module
from collections.abc import Mapping
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from pathlib import Path
from rko_solver import RKO
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.orchestration.configuration import MH_SPECS
from rko_solver.orchestration.factory import build_metaheuristic
from rko_solver.orchestration.factory import normalize_hyperparameters
from rko_solver.parameters.loader import load_offline
from types import MappingProxyType


def decode(keys: np.ndarray) -> np.ndarray:
    """Return candidate coordinates unchanged.

    Args:
        keys (np.ndarray): Random-key vector.

    Returns:
        np.ndarray: Original callback-owned vector.
    """
    return keys


def objective(keys: np.ndarray) -> float:
    """Evaluate a finite domain-neutral minimization objective.

    Args:
        keys (np.ndarray): Decoded vector.

    Returns:
        float: Sum of squared coordinates.
    """
    return float(np.dot(keys, keys))


@pytest.mark.parametrize("algorithm", tuple(MH_SPECS))
def test_explicit_defaults_match_existing_construction(algorithm: str) -> None:
    """Keep every registered algorithm unchanged when overrides equal its defaults.

    Args:
        algorithm (str): Registered canonical algorithm name.
    """
    defaults = load_offline(algorithm)
    old = build_metaheuristic(algorithm, SolutionPool(), decode, objective, 8, 0, None)
    new = build_metaheuristic(algorithm, SolutionPool(), decode, objective, 8, 0, None, defaults)
    old_state = {key: value for key, value in vars(old).items() if key != "solution_pool"}
    new_state = {key: value for key, value in vars(new).items() if key != "solution_pool"}
    assert pickle.dumps(old_state) == pickle.dumps(new_state)


@settings(max_examples=20, deadline=None, derandomize=True)
@given(size=st.integers(min_value=3, max_value=30), probability=st.floats(0, 1))
def test_normalization_owns_numeric_mapping_snapshot(size: int, probability: float) -> None:
    """Detach accepted scalars while preserving exact integral and finite values.

    Args:
        size (int): Valid genetic population size.
        probability (float): Finite probability in the closed unit interval.
    """
    values = {"size_pop": float(size), "prob_mut": probability}
    mapping = {"GeneticAlgorithm": MappingProxyType(values)}
    normalized = normalize_hyperparameters(mapping, ("GA",), 0)
    assert normalized == {"GA": {"size_pop": size, "prob_mut": probability}}
    assert type(normalized["GA"]["size_pop"]) is int
    assert values == {"size_pop": float(size), "prob_mut": probability}
    values["size_pop"] = 100.0
    assert normalized["GA"]["size_pop"] == size


@pytest.mark.parametrize(
    ("algorithm", "overrides"),
    [
        ("ILS", {"beta_min": 0.8, "beta_max": 0.2}),
        ("BRKGA", {"pe": 0.8, "pm": 0.8}),
        ("GA", {"size_pop": 2}),
        ("PSO", {"c1": -1.0}),
    ],
)
def test_overrides_retain_constructor_domain_checks(
    algorithm: str, overrides: dict[str, int | float]
) -> None:
    """Reject invalid joint domains after merging with fresh defaults.

    Args:
        algorithm (str): Algorithm whose constructor defines the domain.
        overrides (dict[str, int | float]): Invalid scalar or joint values.
    """
    with pytest.raises(ValueError):
        build_metaheuristic(algorithm, SolutionPool(), decode, objective, 8, 0, None, overrides)


def test_factory_merges_partial_overrides_without_changing_defaults() -> None:
    """Keep unspecified defaults and leave the supplied dictionary unchanged."""
    overrides = {"p_size": 4, "w": 0.5}
    default = build_metaheuristic("PSO", SolutionPool(), decode, objective, 32, 0, None)
    configured = build_metaheuristic(
        "PSO", SolutionPool(), decode, objective, 32, 0, None, overrides
    )
    assert configured.p_size == 4
    assert configured.w == 0.5
    assert configured.c1 == default.c1
    assert configured.c2 == default.c2
    assert overrides == {"p_size": 4, "w": 0.5}
    fresh = build_metaheuristic("PSO", SolutionPool(), decode, objective, 32, 0, None)
    assert fresh.p_size == default.p_size
    assert fresh.w == default.w


@pytest.mark.parametrize(
    "overrides",
    [
        [],
        {"PSO": []},
        {1: {}},
        {"unknown": {}},
        {"SA": {"t0": 2.0}},
        {"PSO": {"population_typo": 4}},
        {"PSO": {1: 4}},
        {"PSO": {"p_size": True}},
        {"PSO": {"p_size": "4"}},
        {"PSO": {"p_size": 4.5}},
        {"PSO": {"p_size": 0}},
        {"PSO": {"w": float("nan")}},
        {"PSO": {"w": float("inf")}},
        {"PSO": {"w": 10**400}},
    ],
)
def test_invalid_overrides_fail_before_callbacks(overrides: object) -> None:
    """Reject malformed or unused settings without executing a problem callback.

    Args:
        overrides (object): Invalid user-supplied parameter mapping.
    """
    called = []
    optimizer = RKO(
        lambda keys: called.append(True),
        objective,
        8,
        algorithms=["PSO"],
        control=0,
        execution="sequential",
        hyperparameters=overrides,
    )
    assert optimizer.result_ is None
    with pytest.raises(ValueError):
        optimizer.solve(0.01)
    assert called == []
    assert optimizer.result_ is None


def test_adaptive_mode_rejects_scalar_overrides() -> None:
    """Require explicit offline mode rather than silently ignoring tuned values."""
    with pytest.raises(ValueError, match="control=0"):
        RKO(
            decode,
            objective,
            8,
            algorithms=["PSO"],
            control=1,
            hyperparameters={"PSO": {"p_size": 4}},
        ).solve(0.01)


def test_aliases_and_repeated_solve_read_current_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Canonicalize alias keys and take a fresh owned snapshot on each solve.

    Args:
        monkeypatch (pytest.MonkeyPatch): Factory observation fixture.
    """
    seen = []
    original = solver_module.build_metaheuristic

    def observe(*args: object, **kwargs: object) -> object:
        """Record configured population size without changing construction.

        Args:
            args (object): Original positional factory arguments.
            kwargs (object): Original keyword factory arguments.

        Returns:
            object: Constructed algorithm.
        """
        algorithm = original(*args, **kwargs)
        seen.append(algorithm.size_pop)
        return algorithm

    monkeypatch.setattr(solver_module, "build_metaheuristic", observe)
    values = {"size_pop": 4}
    optimizer = RKO(
        decode,
        objective,
        8,
        algorithms=["GeneticAlgorithm"],
        control=0,
        execution="sequential",
        hyperparameters={" GA ": values},
    )
    first = optimizer.solve(0.005)
    values["size_pop"] = 6
    second = optimizer.solve(0.005)
    assert seen == [4, 6]
    assert first.algorithms == second.algorithms == ("GA",)
    assert values == {"size_pop": 6}
    optimizer.hyperparameters = {"GA": {"size_pop": 1}}
    with pytest.raises(ValueError):
        optimizer.solve(0.005)
    assert optimizer.result_ is None


def test_duplicate_alias_overrides_are_rejected() -> None:
    """Reject competing values for the same canonical algorithm."""
    with pytest.raises(ValueError, match="Duplicate"):
        RKO(
            decode,
            objective,
            8,
            algorithms=["GA"],
            control=0,
            hyperparameters={"GA": {"size_pop": 4}, "GeneticAlgorithm": {"size_pop": 6}},
        ).solve(0.01)


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
def test_overrides_reach_workers_without_writing_toml(execution: str, tmp_path: Path) -> None:
    """Repair invalid fixture defaults only through immutable Python mappings.

    Args:
        execution (str): Execution mode under test.
        tmp_path (Path): Temporary configuration directory.
    """
    (tmp_path / "config.toml").write_text(
        '[run]\ncontrol=0\npool_size=3\n[active_metaheuristics]\nlist=["PSO", "GA"]\n',
        encoding="utf-8",
    )
    fixtures = {
        "pso.toml": "[offline]\np_size=0\nc1=1.5\nc2=1.5\nw=0.5\n",
        "ga.toml": "[offline]\nsize_pop=1\nprob_cros=0.8\nprob_mut=0.1\n",
    }
    for name, text in fixtures.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    overrides: Mapping[str, Mapping[str, int | float]] = MappingProxyType(
        {
            "PSO": MappingProxyType({"p_size": 4}),
            "GeneticAlgorithm": MappingProxyType({"size_pop": 4}),
        }
    )
    result = RKO(
        decode,
        objective,
        32,
        params_dir=tmp_path,
        execution=execution,
        n_jobs=1 if execution == "process" else None,
        hyperparameters=overrides,
    ).solve(0.1)
    assert result.algorithms == ("PSO", "GA")
    assert result.cost == pytest.approx(objective(np.asarray(result.keys)))
    assert np.all((np.asarray(result.keys) >= 0) & (np.asarray(result.keys) < 1))
    assert overrides["PSO"]["p_size"] == 4
    for name, text in fixtures.items():
        assert (tmp_path / name).read_text(encoding="utf-8") == text
