"""Exercise the quantile mapping through real solver callbacks and process startup.

Objective:
- Preserve checked payload identity, explicit endpoint handling and supplied-key accounting.

Technical approach:
- Run the public solver against a unit-square task with a known optimal starting tour.

Flow (pseudocode):
1. Decode independent candidates and check each original payload.
2. Reconstruct a known trajectory into supplied starting keys.
3. Solve through sequential and actual process execution with a fixed objective quota.

Important criteria:
- Tests assert feasible results and counts, never machine-dependent elapsed-time thresholds.

Inputs/Outputs:
- Input: a verified quantile mapping and a known feasible tour.
- Output: independently verified public results and exact startup/search counts.
"""

import numpy as np
import pytest
from benchmarks.quantile import solver
from benchmarks.quantile.codec import TrajectoryCodec
from benchmarks.quantile.tasks import Tour
from dataclasses import replace
from rko_solver import RKO


def test_callback_cost_retains_each_original_payload() -> None:
    """Decoding a second candidate cannot replace the first candidate's objective payload."""
    task = Tour(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    mapping = TrajectoryCodec(task)
    bridge = solver.SolverBridge(mapping)
    first = bridge.decode(mapping.encode((1, 2, 3)))
    second = bridge.decode(mapping.encode((2, 1, 3)))
    assert bridge.cost(first) == 4.0
    assert bridge.cost(second) > bridge.cost(first)
    with pytest.raises(ValueError, match="cost"):
        bridge.cost(replace(first, cost=-1.0))


def test_solver_endpoint_policy_is_explicit_and_does_not_mutate_input() -> None:
    """Only the exact solver-produced endpoint is normalized; direct mapping stays strict."""
    task = Tour(np.array([[0.0, 0.0], [1.0, 0.0]]))
    mapping = TrajectoryCodec(task)
    bridge = solver.SolverBridge(mapping)
    source = np.ones(1)
    candidate = bridge.decode(source)
    assert candidate.boundary_repairs == 1
    assert candidate.cost == 2.0
    assert source.tolist() == [1.0]
    with pytest.raises(ValueError):
        mapping.decode(source)
    with pytest.raises(ValueError):
        bridge.decode(np.array([1.001]))


@pytest.mark.parametrize("execution", ["sequential", "process"])
def test_real_solver_preserves_reconstructed_solution_and_total_counts(execution: str) -> None:
    """A reconstructed unit-square optimum survives real search and process serialization.

    Args:
        execution (str): Real solver execution mode.
    """
    task = Tour(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    mapping = TrajectoryCodec(task)
    bridge = solver.SolverBridge(mapping)
    result = RKO(
        bridge.decode,
        bridge.cost,
        task.horizon,
        algorithms=["BRKGA"],
        adaptation="fixed",
        engine="python",
        execution=execution,
        n_jobs=1 if execution == "process" else None,
        prefill_pool=False,
        initial_keys=[mapping.encode((1, 2, 3)).tolist()],
        max_evaluations=24,
        random_state=17,
    ).solve(30.0)
    assert result.cost == 4.0
    assert task.check(mapping.decode(np.asarray(result.keys)).solution) == result.cost
    assert result.initial_keys is not None and result.initial_keys.n_evaluations == 1
    assert result.searches[0].n_evaluations == 24
    assert result.n_evaluations == 25
