"""Externally owned construction deadlines without renewed search allowances.

Objective:
- Keep model preparation/startup from renewing an external job's absolute deadline.
Technical approach:
- Use an already-expired owner deadline and the existing first-completion grace.
Flow (pseudocode):
1. Bind trusted construction to owned original model records.
2. Start an ordinary solve after its external allowance is exhausted.
3. Retain exactly one actual completion; default solves keep their evaluation limit.
Important criteria:
- No sleeping, wall-time races, generated host execution or relaxed guard.
Inputs/Outputs:
- Input: explicit monotonic deadline and ordinary public owned-model factory.
- Output: actual completed result and evidence counts.
"""

import numpy as np
import pytest
import time
from rko_solver import RKO
from rko_solver.modeling.construction import ConstructionInput
from rko_solver.modeling.factory import ModelFactory
from rko_solver.modeling.records import LinearVariable
from rko_solver.modeling.source import ExtractedLinearModel


def _construct(keys: np.ndarray, instance: ConstructionInput) -> tuple[float, ...]:
    """Return complete original lower-bound values without replacing numerical scoring.

    Args:
        keys (np.ndarray): Actual original keys, intentionally irrelevant to this control.
        instance (ConstructionInput): Protected immutable original variable/domain data.

    Returns:
        tuple[float, ...]: Actual complete original-coordinate values.
    """
    return tuple(domain.lower for domain in instance.domains)


def test_expired_construction_deadline_does_not_renew_after_startup() -> None:
    """A spent external budget permits one completion, not a fresh per-search budget."""
    source = ExtractedLinearModel(
        "pyomo",
        "owned-deadline-control",
        "deadline-control",
        (LinearVariable("x", "x", 0.0, 1.0, "continuous"),),
        (1.0,),
        0.0,
        False,
        (),
    )
    options = {
        "construction": _construct,
        "algorithms": ["SA"],
        "adaptation": "fixed",
        "engine": "python",
        "execution": "sequential",
        "prefill_pool": False,
        "pool_size": 4,
        "random_state": 17,
        "max_evaluations": 3,
    }
    bounded = ModelFactory.create_owned(
        RKO,
        source,
        {
            **options,
            "construction_deadline": time.perf_counter() - 1.0,
        },
    )
    result = bounded.solve(time_limit=1.0)
    assert result.n_evaluations == 1
    assert result.model.statistics.evaluations == 1
    assert result.model.best_feasible.objective == 0.0
    assert result.searches[0].stop_reason == "time_limit"
    ordinary = ModelFactory.create_owned(RKO, source, options).solve(time_limit=1.0)
    assert ordinary.n_evaluations == 3


def test_construction_deadline_retains_only_completed_invocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crossing the owner ceiling during construction cannot admit a partial third point.

    Args:
        monkeypatch (pytest.MonkeyPatch): Deterministic monotonic clock, no wall-time sleeping.
    """
    clock = [100.0]
    calls = []

    def construct(keys: np.ndarray, instance: ConstructionInput) -> tuple[float, ...]:
        """Advance one simulated second while constructing a complete candidate vector.

        Args:
            keys (np.ndarray): Original keys.
            instance (ConstructionInput): Immutable domains.

        Returns:
            tuple[float, ...]: Original lower-bound values.
        """
        calls.append(1)
        clock[0] += 1.0
        return tuple(domain.lower for domain in instance.domains)

    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    source = ExtractedLinearModel(
        "pyomo",
        "owned-deadline-control",
        "deadline-control",
        (LinearVariable("x", "x", 0.0, 1.0, "continuous"),),
        (1.0,),
        0.0,
        False,
        (),
    )
    solver = ModelFactory.create_owned(
        RKO,
        source,
        {
            "construction": construct,
            "construction_deadline": 102.5,
            "algorithms": ["SA"],
            "adaptation": "fixed",
            "engine": "python",
            "execution": "sequential",
            "prefill_pool": False,
            "pool_size": 4,
            "random_state": 17,
            "max_evaluations": 10,
        },
    )
    result = solver.solve(time_limit=1000.0)
    assert len(calls) == 3 and result.n_evaluations == 2
    assert result.model.statistics.evaluations == 2
    assert result.model.best_feasible.objective == 0.0


@pytest.mark.parametrize("deadline", [True, float("nan"), float("inf"), "later", 10**1000])
def test_construction_deadline_requires_finite_monotonic_scalar(deadline: object) -> None:
    """A caller cannot inject a malformed stopping policy into the trusted hook.

    Args:
        deadline (object): Invalid explicit owner deadline.
    """
    source = ExtractedLinearModel(
        "pyomo",
        "owned-deadline-control",
        "deadline-control",
        (LinearVariable("x", "x", 0.0, 1.0, "continuous"),),
        (1.0,),
        0.0,
        False,
        (),
    )
    with pytest.raises(ValueError, match="deadline"):
        ModelFactory.create_owned(
            RKO, source, {"construction": _construct, "construction_deadline": deadline}
        )
