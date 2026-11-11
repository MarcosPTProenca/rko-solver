"""Contract tests for immutable per-search stop limits.

Objective:
- Verify that search stopping criteria are validated and represented as one value.

Technical approach:
- Construct valid limits and reject invalid evaluation caps, targets and patience values.

Flow (pseudocode):
1. Construct a limit value with no optional criteria.
2. Construct a value with each supported criterion.
3. Reject invalid values without changing the caller's inputs.

Important criteria:
- Evaluation caps retain the existing `RunLimits` validation contract.
- Patience is a positive integer and target costs are finite real numbers.

Inputs/Outputs:
- Input: evaluation cap, optional target cost and optional patience.
- Output: immutable validated `SearchLimits` values or a `ValueError`.
"""

import numpy as np
import pytest
from dataclasses import FrozenInstanceError
from rko_solver import RKO
from rko_solver.orchestration.limits import SearchLimits

EXECUTIONS = [
    pytest.param("sequential", None, id="sequential"),
    pytest.param("thread", None, id="thread"),
    pytest.param("process", 2, id="process"),
]


def decode(keys: np.ndarray) -> np.ndarray:
    """Return the input keys unchanged for a deterministic stop-rule fixture."""
    return keys


def constant_cost(values: np.ndarray) -> float:
    """Return a constant objective so patience expires after a fixed count."""
    return 1.0


@pytest.mark.parametrize(("execution", "n_jobs"), EXECUTIONS)
def test_target_cost_stops_each_execution_after_the_first_hit(
    execution: str, n_jobs: int | None
) -> None:
    """Stop a search as soon as its evaluated incumbent reaches the target."""
    result = RKO(
        decode,
        constant_cost,
        3,
        algorithms=["SA"],
        execution=execution,
        n_jobs=n_jobs,
        random_state=1,
    ).solve(30.0, target_cost=1.0)

    assert result.searches[0].n_evaluations == 1
    assert result.searches[0].stop_reason == "target_cost"
    assert result.stop_reason == "completed"


@pytest.mark.parametrize(("execution", "n_jobs"), EXECUTIONS)
def test_patience_stops_each_execution_after_consecutive_non_improvements(
    execution: str, n_jobs: int | None
) -> None:
    """Stop after the requested completed evaluations without a new incumbent."""
    result = RKO(
        decode,
        constant_cost,
        3,
        algorithms=["SA"],
        execution=execution,
        n_jobs=n_jobs,
        random_state=1,
    ).solve(30.0, patience=1)

    assert result.searches[0].n_evaluations == 2
    assert result.searches[0].stop_reason == "stagnation"
    assert result.stop_reason == "completed"


def test_search_limits_accepts_unbounded_defaults_and_explicit_criteria() -> None:
    """Accept omitted criteria and a complete validated stop policy."""
    assert SearchLimits() == SearchLimits(None, None, None)
    assert SearchLimits(10, 0.0, 5) == SearchLimits(max_evaluations=10, target_cost=0.0, patience=5)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), "1", True])
def test_solve_rejects_invalid_target_cost(value: object) -> None:
    """Validate early-stop options at the public solve boundary."""
    solver = RKO(decode, constant_cost, 3, algorithms=["SA"], execution="sequential")
    with pytest.raises(ValueError, match="target_cost"):
        solver.solve(1.0, target_cost=value)


def test_non_python_engines_reject_unsupported_early_stop_rules() -> None:
    """Never ignore early-stop options on Rust, C++ or automatic engine selection."""
    for engine in ("rust", "cpp", "auto"):
        solver = RKO(decode, constant_cost, 3, algorithms=["SA"], engine=engine)
        with pytest.raises(ValueError, match="engine='python'"):
            solver.solve(1.0, patience=1)


def test_search_limits_is_immutable() -> None:
    """Prevent a stop policy from changing while a search is running."""
    limits = SearchLimits(max_evaluations=10)
    with pytest.raises(FrozenInstanceError):
        limits.max_evaluations = 5


@pytest.mark.parametrize("value", [0, -1, 1.5, True, 2**63])
def test_search_limits_rejects_invalid_evaluation_caps(value: object) -> None:
    """Keep the evaluation cap's existing type and range validation."""
    with pytest.raises(ValueError, match="max_evaluations"):
        SearchLimits(max_evaluations=value)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_search_limits_rejects_invalid_patience(value: object) -> None:
    """Require patience to be a positive non-boolean integer when supplied."""
    with pytest.raises(ValueError, match="patience"):
        SearchLimits(patience=value)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), "1", True])
def test_search_limits_rejects_invalid_target_cost(value: object) -> None:
    """Require a finite real target cost when supplied."""
    with pytest.raises(ValueError, match="target_cost"):
        SearchLimits(target_cost=value)
