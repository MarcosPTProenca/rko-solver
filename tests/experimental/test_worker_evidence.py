"""Independent worker evidence rejection and runner invalidation contracts.

Objective:
- Keep malformed worker claims from becoming trusted objective or resource observations.
Technical approach:
- Start with a hand-verified witness and mutate one boundary field at a time.
Flow (pseudocode):
1. Recompute the legitimate witness on the host.
2. Supply contradictory, malformed or out-of-envelope records.
3. Assert typed rejection and distinguish integrity failures from inner allowance failures.
Important criteria:
- All records are synthetic trusted test data; no candidate source executes.
Inputs/Outputs:
- Input: a tiny knapsack and controlled worker records.
- Output: independent evidence, precise rejection and runner qualification state.
"""

import pytest
import typing
from rko_solver.experimental.self_evolve import Problem
from rko_solver.experimental.self_evolve.sandbox import PodmanRunner
from rko_solver.experimental.self_evolve.sandbox_evidence import SandboxError
from rko_solver.experimental.self_evolve.sandbox_evidence import WorkerEvidence


@pytest.fixture
def problem() -> Problem:
    """Return a hand-verifiable two-item instance.

    Returns:
        Problem: The first item alone is a feasible profit-three witness.
    """
    return Problem("knapsack", {"weights": [2, 3], "profits": [3, 5], "capacity": 2})


@pytest.fixture
def record(problem: Problem) -> dict[str, typing.Any]:
    """Return a complete internally consistent independent-worker record.

    Args:
        problem (Problem): Host-side reference evaluator.

    Returns:
        dict[str, Any]: Valid evidence for exactly one completed evaluation.
    """
    return {
        **problem.evaluate([0]),
        "solution": [0],
        "cpu_seconds": 0.05,
        "wall_seconds": 0.1,
        "peak_memory_bytes": 1024,
        "evaluations": 1,
        "infeasible_evaluations": 0,
        "distinct_outputs_capped": 1,
    }


def test_valid_evidence_is_recomputed_not_replaced_by_worker_fields(
    problem: Problem, record: dict[str, typing.Any]
) -> None:
    """A complete legitimate response passes all independent verification stages.

    Args:
        problem (Problem): Trusted hand-worked evaluator.
        record (dict[str, Any]): Consistent worker response.
    """
    runner = object.__new__(PodmanRunner)
    runner.qualified = True
    assert runner._verify_record(record, problem, 0.1) == problem.evaluate([0])
    assert runner.qualified is True


@pytest.mark.parametrize("solution", [None, [0, 0], [True], [-1], [2], "0"])
def test_invalid_witness_structure_disables_runtime(
    problem: Problem, record: dict[str, typing.Any], solution: object
) -> None:
    """Malformed returned decisions cannot be treated as ordinary decoder quality.

    Args:
        problem (Problem): Trusted host evaluator.
        record (dict[str, Any]): Otherwise legitimate response.
        solution (object): Structurally invalid or out-of-range witness.
    """
    runner = object.__new__(PodmanRunner)
    runner.qualified = True
    record["solution"] = solution
    with pytest.raises(SandboxError, match="witness"):
        runner._verify_record(record, problem, 0.1)
    assert runner.qualified is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("objective", True),
        ("objective", 3.0),
        ("feasible", 1),
        ("score", "0"),
        ("objective", 100),
        ("feasible", False),
        ("score", 0.123),
    ],
)
def test_contradictory_objective_claims_are_integrity_failures(
    problem: Problem, record: dict[str, typing.Any], field: str, value: object
) -> None:
    """Types and values must both agree with independently recomputed semantics.

    Args:
        problem (Problem): Trusted host evaluator.
        record (dict[str, Any]): Legitimate response before mutation.
        field (str): Objective, feasibility or score claim to corrupt.
        value (object): Wrong type or contradictory value.
    """
    record[field] = value
    with pytest.raises(SandboxError, match="objective/feasibility"):
        WorkerEvidence(record).witness(problem)


@pytest.mark.parametrize(
    "field", ["cpu_seconds", "wall_seconds", "peak_memory_bytes", "evaluations"]
)
@pytest.mark.parametrize("value", [None, True, "1", -1, float("inf"), float("nan")])
def test_resource_fields_require_finite_nonnegative_numbers(
    record: dict[str, typing.Any], field: str, value: object
) -> None:
    """Malformed measurements fail instead of receiving a numeric conversion or refund.

    Args:
        record (dict[str, Any]): Legitimate worker response.
        field (str): Required resource measurement.
        value (object): Invalid numeric representation or domain value.
    """
    record[field] = value
    with pytest.raises(SandboxError, match="resource"):
        WorkerEvidence(record).resources()


@pytest.mark.parametrize(
    "updates",
    [
        {"evaluations": 0, "distinct_outputs_capped": 0},
        {"evaluations": 1.0},
        {"infeasible_evaluations": True},
        {"infeasible_evaluations": 2},
        {"distinct_outputs_capped": -1},
        {"evaluations": 300, "distinct_outputs_capped": 257},
    ],
)
def test_impossible_counters_disable_the_runner(
    problem: Problem, record: dict[str, typing.Any], updates: dict[str, typing.Any]
) -> None:
    """Counter inconsistency invalidates even an otherwise correct final witness.

    Args:
        problem (Problem): Independent reference evaluator.
        record (dict[str, Any]): Legitimate response before mutation.
        updates (dict[str, Any]): Impossible evaluation/diversity counts.
    """
    runner = object.__new__(PodmanRunner)
    runner.qualified = True
    record.update(updates)
    with pytest.raises(SandboxError, match="counter|evaluations|diversity"):
        runner._verify_record(record, problem, 0.1)
    assert runner.qualified is False


def test_inner_allowance_failure_does_not_claim_infrastructure_corruption(
    problem: Problem, record: dict[str, typing.Any]
) -> None:
    """A valid witness from a slow decoder fails its allowance without invalidating isolation.

    Args:
        problem (Problem): Trusted reference evaluator.
        record (dict[str, Any]): Valid final evidence with an overlong inner invocation.
    """
    runner = object.__new__(PodmanRunner)
    runner.qualified = True
    record["wall_seconds"] = 0.2
    with pytest.raises(SandboxError, match="inner runtime"):
        runner._verify_record(record, problem, 0.1)
    assert runner.qualified is True
