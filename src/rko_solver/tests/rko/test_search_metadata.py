"""Deterministic callback accounting and immutable provenance boundaries.

Objective:
- Distinguish normal completion, budget exhaustion, cancellation and failure.

Technical approach:
- Use a three-evaluation algorithm and a frozen monotonic clock.
- Exercise tuple compatibility, report ownership and portable serialization.

Flow (pseudocode):
1. Execute controlled candidate evaluations through SearchSession.
2. Compare finite completions and explicit stopping classifications.
3. Reconfigure a solver and check that previous reports remain detached.

Important criteria:
- Invalid objectives and exceptions do not count as finite completions.
- Reports never retain callbacks, RNG handles or candidate populations.

Inputs/Outputs:
- Input: controlled callback outcomes and private session deadlines.
- Output: deterministic accounting, lifecycle and ownership assertions.
"""

import json
import math
import numpy as np
import pickle
import pytest
import rko_solver.orchestration.search as search_module
import rko_solver.solver as solver_module
import threading
from dataclasses import FrozenInstanceError
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError
from rko_solver import RKO
from rko_solver import RKOResult
from rko_solver import SearchMetadata
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.base import MetaheuristicBase
from rko_solver.orchestration.configuration import MH_SPECS
from rko_solver.orchestration.factory import build_metaheuristic
from rko_solver.orchestration.search import SearchSession
from rko_solver.orchestration.types import CostFn
from rko_solver.orchestration.types import DecoderFn


def decode(keys: np.ndarray) -> np.ndarray:
    """Return candidate values for a deterministic numerical objective.

    Args:
        keys (np.ndarray): Detached candidate.

    Returns:
        np.ndarray: Unchanged payload.
    """
    return keys


def objective(keys: np.ndarray) -> float:
    """Evaluate squared distance from the origin.

    Args:
        keys (np.ndarray): Numerical payload.

    Returns:
        float: Finite objective.
    """
    return float(keys @ keys)


class ThreeEvaluations(MetaheuristicBase):
    """Perform three known evaluations without publishing a separate incumbent.

    Responsibilities:
    - Exercise improvement, worsening and session incumbent recovery.
    """

    def __init__(self, decoder: DecoderFn, cost: CostFn) -> None:
        """Retain callbacks for the controlled search.

        Args:
            decoder (DecoderFn): Session-replaced decoder.
            cost (CostFn): Session-replaced objective.
        """
        super().__init__(SolutionPool())
        self.decoder = decoder
        self.cost = cost

    def solve(self, limit_time: float) -> tuple[list[object], list[float], float]:
        """Evaluate three candidates and leave incumbent recovery to the session.

        Args:
            limit_time (float): Duration enforced at the callback boundary.

        Returns:
            tuple: Empty incumbent to exercise the session's completed best.
        """
        for value in (0.8, 0.2, 0.4):
            self.cost(self.decoder(np.array([value])))
        return [], [], math.inf


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze only tests that have a finite controlled iteration count.

    Args:
        monkeypatch (pytest.MonkeyPatch): Restore the clock after each test.
    """
    monkeypatch.setattr(search_module.time, "perf_counter", lambda: 10.0)


@pytest.mark.parametrize("expired", [False, True])
def test_completed_and_deadline_counts(expired: bool, frozen_clock: None) -> None:
    """Count every finite objective, not only incumbent improvements.

    Args:
        expired (bool): Use a past deadline to stop after the permitted first result.
        frozen_clock (None): Fixed clock fixture.
    """
    algorithm = ThreeEvaluations(decode, objective)
    session = SearchSession(decode, objective)
    result = session.solve(algorithm, 1.0, deadline=9.0 if expired else 11.0, random_state=42)
    report = session.metadata
    assert report is not None
    assert report.n_evaluations == (1 if expired else 3)
    assert report.stop_reason == ("time_limit" if expired else "completed")
    assert report.random_state == 42
    assert report.elapsed_seconds == 0.0
    assert isinstance(result, tuple) and len(result) == 3
    assert result[2] == pytest.approx(0.64 if expired else 0.04)
    assert algorithm.decoder is decode and algorithm.cost is objective
    if not expired:
        session.solve(algorithm, 1.0)
        assert session.metadata.n_evaluations == 3


def test_cancelled_unevaluated_session(frozen_clock: None) -> None:
    """Report peer cancellation without inventing an evaluation or finite result.

    Args:
        frozen_clock (None): Fixed clock fixture.
    """
    event = threading.Event()
    event.set()
    session = SearchSession(decode, objective)
    result = session.solve(ThreeEvaluations(decode, objective), 1, cancelled=event)
    assert result[1] == [] and math.isinf(result[2])
    assert session.metadata.n_evaluations == 0
    assert session.metadata.stop_reason == "cancelled"


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_invalid_objective_has_no_completed_report(value: float, frozen_clock: None) -> None:
    """Do not count invalid values or publish a successful report after failure.

    Args:
        value (float): Nonfinite objective.
        frozen_clock (None): Fixed clock fixture.
    """
    session = SearchSession(decode, lambda keys: value)
    algorithm = ThreeEvaluations(decode, objective)
    with pytest.raises(ValueError, match="finite"):
        session.solve(algorithm, 1)
    assert session.metadata is None
    assert session.n_evaluations == 0
    assert algorithm.decoder is decode and algorithm.cost is objective


def test_result_configuration_snapshots_and_serialization() -> None:
    """Keep prior parameters detached across solves without mutable mapping proxies."""
    overrides = {"PSO": {"p_size": 4}}
    solver = RKO(
        decode,
        objective,
        4,
        algorithms=["PSO"],
        execution="sequential",
        control=0,
        hyperparameters=overrides,
        random_state=42,
    )
    first = solver.solve(0.003)
    overrides["PSO"]["p_size"] = 6
    second = solver.solve(0.003)
    assert dict(first.searches[0].parameters)["p_size"] == 4
    assert dict(second.searches[0].parameters)["p_size"] == 6
    assert dict(first.versions)["numpy"] == np.__version__
    assert pickle.loads(pickle.dumps(first)) == first
    assert json.loads(json.dumps(asdict(first)))["searches"][0]["algorithm"] == "PSO"
    with pytest.raises(FrozenInstanceError):
        first.searches[0].n_evaluations = 0
    with pytest.raises(TypeError):
        first.searches[0].parameters[0] = ("p_size", 100)


def test_legacy_manual_result_and_stop_summary() -> None:
    """Keep the six-argument constructor and mark missing accounting as unknown."""
    legacy = RKOResult((0.1,), 1.0, 0.1, "serial", ("PSO",), 42)
    assert legacy.n_evaluations is None
    assert legacy.stop_reason == "unknown"
    assert legacy.n_workers == 1
    for reason, expected in (
        ("not_started", "time_limit"),
        ("completed", "completed"),
        ("cancelled", "cancelled"),
    ):
        result = RKOResult(
            (0.1,),
            1.0,
            0.1,
            "process",
            ("PSO",),
            42,
            searches=(SearchMetadata("PSO", stop_reason=reason),),
        )
        assert result.stop_reason == expected
        assert result.n_evaluations == 0


@pytest.mark.parametrize("name", list(MH_SPECS))
@pytest.mark.parametrize("control", [0, 1])
def test_factory_snapshots_initial_constructor_settings(name: str, control: int) -> None:
    """Capture supported control and detached actual constructor parameters.

    Args:
        name (str): Registered canonical algorithm.
        control (int): Requested offline or adaptive mode.
    """
    algorithm = build_metaheuristic(name, SolutionPool(), decode, objective, 4, control, None)
    report = algorithm._search_metadata
    spec = MH_SPECS[name]
    assert report.algorithm == name
    assert report.control == (control if spec.supports_control else 0)
    assert dict(report.parameters) == {
        key: getattr(algorithm, key) for key in spec.int_params + spec.float_params
    }
    assert report.stop_reason == "not_started"
    if spec.supports_control and control == 1:
        grid = tuple(tuple(axis) for axis in algorithm.parameters)
        assert report.parameter_grid == grid
        algorithm.parameters[0][0] += 1
        assert report.parameter_grid == grid
    else:
        assert report.parameter_grid == ()


def test_source_only_installation_has_unknown_distribution_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not fail a source-only solve when distribution metadata is unavailable.

    Args:
        monkeypatch (pytest.MonkeyPatch): Simulate absent installed distribution metadata.
    """

    def missing(name: str) -> str:
        """Simulate an importable source package without installation metadata.

        Args:
            name (str): Requested distribution.

        Raises:
            PackageNotFoundError: Always, to exercise source-only provenance.
        """
        raise PackageNotFoundError(name)

    monkeypatch.setattr(solver_module, "version", missing)
    result = RKO(decode, objective, 2, algorithms=["MultiStart"], execution="sequential").solve(
        0.001
    )
    assert dict(result.versions)["rko-solver"] == "unknown"
