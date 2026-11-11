"""Tests for the `progress_schedule` option and the progress seen by metaheuristics.

Objective:
- Verify that the option is parsed, validated and reaches both engines, and that search
  progress combines the time share with the evaluation share.

Technical approach:
- Parse run configurations with and without the key.
- Run a bounded adaptive search per engine and compare modes.
- Read `progress` from inside a session bounded only by evaluations.

Flow (pseudocode):
1. Parse the key and reject non-boolean values.
2. Run each engine in both modes with one seed and compare the outcomes.
3. Observe progress during a solve capped by `max_evaluations`.

Important criteria:
- The default is the legacy whole-second schedule (`progress_schedule` false).
- Both modes are deterministic for a fixed seed and evaluation cap.
- Progress follows evaluations when no clock limit is near.

Inputs/Outputs:
- Input: run configuration dictionaries and a small knapsack objective.
- Output: assertions over parsed flags, results and observed progress.
"""

from __future__ import annotations

import numpy as np
import pytest
import rko_solver.solver as solver_module
from pathlib import Path
from rko_solver import RKO
from rko_solver import Callbacks
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.base import MetaheuristicBase
from rko_solver.orchestration.configuration import parse_run_config
from rko_solver.orchestration.limits import SearchLimits
from rko_solver.orchestration.search import SearchSession
from rko_solver.parameters.loader import load_config
from typing import Literal

PROFITS = np.array([10 + (index * 37) % 91 for index in range(30)])
WEIGHTS = np.array([5 + (index * 53) % 47 for index in range(30)])


def decode(keys: np.ndarray) -> np.ndarray:
    """Select items whose key exceeds one half.

    Args:
        keys (np.ndarray): Random keys.

    Returns:
        np.ndarray: Boolean selection.
    """
    return np.asarray(keys) > 0.5


def cost(selection: np.ndarray) -> float:
    """Score a selection with a penalized negative profit.

    Args:
        selection (np.ndarray): Boolean selection.

    Returns:
        float: Cost to minimize.
    """
    chosen = np.asarray(selection)
    return float(1000 * max(0, int(chosen @ WEIGHTS) - 300) - int(chosen @ PROFITS))


def parse(value: object) -> bool:
    """Parse a run configuration carrying `progress_schedule`.

    Args:
        value (object): Raw TOML value, or the string `absent` to omit the key.

    Returns:
        bool: Parsed flag.
    """
    run: dict[str, object] = {"control": 1, "pool_size": 5}
    if value != "absent":
        run["progress_schedule"] = value
    cfg = {"run": run, "active_metaheuristics": {"list": ["SA"]}}
    return parse_run_config(cfg).progress_schedule


def test_progress_schedule_defaults_to_false_and_parses_booleans() -> None:
    """Accept the key as a boolean and default it to the legacy schedule."""
    assert parse("absent") is False
    assert parse(True) is True
    assert parse(False) is False


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_progress_schedule_rejects_non_booleans(value: object) -> None:
    """Reject non-boolean values with a configuration error.

    Args:
        value (object): Invalid raw TOML value.
    """
    with pytest.raises(ValueError, match="progress_schedule"):
        parse(value)


def test_packaged_config_keeps_the_legacy_schedule() -> None:
    """Ship the packaged configuration with the legacy schedule as the default."""
    assert parse_run_config(load_config(None)).progress_schedule is False


def run_case(
    engine: Literal["python", "rust"], progress: bool, monkeypatch: pytest.MonkeyPatch
) -> tuple[float, list[float]]:
    """Run one adaptive GA search bounded by evaluations.

    Args:
        engine (Literal["python", "rust"]): Engine under test.
        progress (bool): Whether to inject `progress_schedule = true` into the loaded configuration.
        monkeypatch (pytest.MonkeyPatch): Scope the configuration override to this call.

    Returns:
        tuple[float, list[float]]: Best cost and best keys.
    """
    if progress:
        load = solver_module.load_config

        def forced(params_dir: Path | None = None) -> dict[str, object]:
            """Load the packaged configuration with `progress_schedule` enabled.

            Args:
                params_dir (Path | None): Parameter directory.

            Returns:
                dict[str, object]: Raw configuration.
            """
            raw = load(params_dir)
            run = raw["run"]
            assert isinstance(run, dict)
            run["progress_schedule"] = True
            return raw

        monkeypatch.setattr(solver_module, "load_config", forced)
    solver = RKO(
        Callbacks(decode, cost, len(PROFITS)),
        algorithms=["GA"],
        adaptation="adaptive",
        max_evaluations=3000,
        random_state=3,
        engine=engine,
        execution="sequential" if engine == "python" else None,
        n_jobs=1 if engine == "rust" else None,
    )
    result = solver.solve(120.0)
    return result.cost, list(result.keys)


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_each_mode_is_deterministic_and_the_modes_differ(
    engine: Literal["python", "rust"], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeat each mode exactly and confirm the option reaches the engine.

    Args:
        engine (Literal["python", "rust"]): Engine under test.
        monkeypatch (pytest.MonkeyPatch): Scope the configuration override.
    """
    pytest.importorskip("rko_solver._native")
    legacy = run_case(engine, False, monkeypatch)
    assert run_case(engine, False, monkeypatch) == legacy
    progress = run_case(engine, True, monkeypatch)
    assert run_case(engine, True, monkeypatch) == progress
    assert progress != legacy


def test_progress_follows_evaluations_under_a_long_time_budget() -> None:
    """Report evaluation share as progress while the clock share stays near zero."""
    seen: list[float] = []

    class Probe(MetaheuristicBase):
        """Record progress after each completed evaluation.

        Responsibilities:
        - Evaluate fixed vectors and sample `progress` between evaluations.
        """

        def __init__(self, pool: SolutionPool, name: str) -> None:
            """Install placeholder callbacks that the session replaces during solve.

            Args:
                pool (SolutionPool): Shared pool.
                name (str): Log label.
            """
            super().__init__(pool, name)
            self.decoder = decode
            self.cost = lambda selection: 1.0

        def solve(self, limit_time: float) -> tuple[list[object], list[float], float]:
            """Evaluate ten vectors and record progress after each one.

            Args:
                limit_time (float): Time horizon in seconds.

            Returns:
                tuple[list[object], list[float], float]: Empty extras, last keys and cost.
            """
            self._start_timer()
            value = 0.0
            for _ in range(10):
                value = self.cost(self.decoder(np.full(4, 0.3)))
                seen.append(self.progress(limit_time))
            return [], [0.3] * 4, value

    session = SearchSession(decode, lambda selection: 1.0)
    session.solve(Probe(SolutionPool(), "probe"), 1000.0, limits=SearchLimits(max_evaluations=20))
    assert seen == pytest.approx([(index + 1) / 20 for index in range(10)], abs=1e-3)
    unbounded = Probe(SolutionPool(), "probe")
    unbounded._start_timer()
    assert unbounded.progress(1000.0) == pytest.approx(0.0, abs=1e-3)
