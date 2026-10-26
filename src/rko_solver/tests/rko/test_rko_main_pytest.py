"""Tests for the public RKO entrypoint orchestration flow.

Objective:
- Validate input guards and end-to-end orchestration wiring in `rko_main.run`.

Technical approach:
- Assert explicit validation errors for invalid scalar inputs.
- Monkeypatch orchestration collaborators to isolate control flow.

Flow (pseudocode):
1. Call `run` with invalid inputs and assert raised errors.
2. Stub configuration/factory/execution collaborators.
3. Execute `run` and assert forwarded values and final selection.

Important criteria:
- Input validation must reject non-positive `tam_solution` and `limit_time`.
- Orchestration must use active metaheuristics from parsed config.

Inputs/Outputs:
- Input: mocked orchestration dependencies and synthetic callbacks.
- Output: assertions over selected best result and forwarded arguments.
"""

from __future__ import annotations

import pytest
from rko_solver import rko_main
from rko_solver import solver
from rko_solver.orchestration.types import RunConfig


def test_run_rejects_invalid_tam_solution() -> None:
    """Ensure `run` rejects non-positive solution size."""
    with pytest.raises(ValueError, match="n_variables"):
        rko_main.run(lambda x: x, lambda x: 0.0, tam_solution=0, limit_time=1.0)


def test_run_rejects_invalid_limit_time() -> None:
    """Ensure `run` rejects non-positive execution time limits."""
    with pytest.raises(ValueError, match="time_limit"):
        rko_main.run(lambda x: x, lambda x: 0.0, tam_solution=3, limit_time=0.0)


@pytest.mark.parametrize("limit_time", [float("nan"), float("inf"), True, "1", -1])
def test_run_rejects_nonfinite_or_nonnumeric_time(limit_time: object) -> None:
    """Reject invalid time budgets before configuration or child startup.

    Args:
        limit_time (object): Invalid public time-budget input.
    """
    with pytest.raises(ValueError, match="time_limit"):
        rko_main.run(lambda keys: keys, lambda keys: 0.0, 3, limit_time)


def test_run_executes_orchestration_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure orchestration collaborators are called and best result is returned."""
    captured: dict[str, object] = {}

    monkeypatch.setattr(solver, "load_config", lambda params_dir: {"run": {}})
    monkeypatch.setattr(
        solver,
        "parse_run_config",
        lambda cfg: RunConfig(control=0, pool_size=3, active_metaheuristics=("SA", "GA")),
    )
    monkeypatch.setattr(
        solver,
        "build_metaheuristic",
        lambda name, pool, decoder, cost, tam_solution, control, params_dir, hyperparameters=None: (
            f"mh:{name}"
        ),
    )

    def _fake_parallel(
        metaheuristics: list[object],
        limit_time: float,
        streams: list[object],
        *,
        search_metadata: list[object] | None = None,
        history_limit: int = 0,
        max_evaluations: int | None = None,
    ) -> list[tuple[list[object], list[float], float]]:
        """Record dispatch settings and verify stream allocation cardinality.

        Args:
            metaheuristics (list[object]): Constructed test search labels.
            limit_time (float): Search duration supplied by the solver.
            streams (list[object]): One owned stream pair per search.
            search_metadata (list[object] | None): Optional unused report collector.
            history_limit (int): Disabled history capture in the compatibility API.
            max_evaluations (int | None): Unused evaluation limit, always None here.

        Returns:
            list[tuple]: One synthetic successful search outcome.
        """
        assert len(streams) == len(metaheuristics)
        assert history_limit == 0 and max_evaluations is None
        captured["metaheuristics"] = metaheuristics
        captured["limit_time"] = limit_time
        return [([], [0.2, 0.3], 12.0)]

    monkeypatch.setattr(solver, "run_metaheuristics_parallel", _fake_parallel)
    monkeypatch.setattr(solver, "select_best_result", lambda results, pool: ([0.2, 0.3], 12.0))

    best_keys, best_cost = rko_main.run(
        decoder=lambda keys: keys,
        cost=lambda decoded: float(sum(decoded)),
        tam_solution=2,
        limit_time=5.0,
        execution="thread",
    )

    assert best_keys == [0.2, 0.3]
    assert best_cost == 12.0
    assert captured["limit_time"] == 5.0
    assert captured["metaheuristics"] == ["mh:SA", "mh:GA"]
