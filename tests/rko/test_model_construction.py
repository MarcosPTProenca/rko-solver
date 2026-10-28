"""Protected original-assignment evaluation and opt-in construction contracts.

Objective:
- Verify new construction without changing model semantics or completed witness identity.
Technical approach:
- Exercise public owned numerical and model-factory interfaces with worked examples.
Flow (pseudocode):
1. Compile immutable original variables, rows and objective.
2. Supply explicit assignments independent of the default key mapping.
3. Check strict domains, original-unit evidence and unchanged default behavior.
Important criteria:
- No solver certificate, live proposer or source execution outside qualified isolation.
Inputs/Outputs:
- Input: small original bounded linear records and trusted test constructors.
- Output: immutable independently checkable completed model evidence.
"""

import numpy as np
import pytest
from rko_solver import RKO
from rko_solver.modeling.factory import ModelFactory
from rko_solver.modeling.linear import CompiledLinearModel
from rko_solver.modeling.linear import LinearRow
from rko_solver.modeling.linear import LinearVariable
from rko_solver.modeling.source import ExtractedLinearModel


def test_assignment_evaluation_does_not_reconstruct_from_keys() -> None:
    """Check a supplied mixed assignment against original rows and objective including constants."""
    model = CompiledLinearModel(
        (
            LinearVariable("x", "x", 0, 5),
            LinearVariable("y", "y", 0, 3, "integer"),
            LinearVariable("z", "z", 2, 2),
        ),
        (3, -2, 4),
        (LinearRow("balance", ((0, 1), (1, 1)), 5, 5),),
        objective_constant=7,
    )
    keys = np.zeros(2)
    evaluation = model.evaluate_assignment(keys, (4, 1, 2))
    assert evaluation.keys == (0, 0)
    assert evaluation.values == (4, 1, 2)
    assert evaluation.objective == 25
    assert evaluation.feasible
    assert evaluation.rows[0].activity == 5
    assert 0 <= evaluation.score <= 0.5
    assert model.decode(keys).objective == 15


def coupled_assignment(keys: np.ndarray, instance: object) -> tuple[float, float]:
    """Supply a known feasible construction, independent of default box reconstruction.

    Args:
        keys (np.ndarray): Original read-only chromosome, unused in this literal control.
        instance (object): Frozen original model construction input.

    Returns:
        tuple[float, float]: Known equality-respecting assignment.
    """
    return 0.75, 0.25


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
def test_trusted_construction_retains_its_actual_original_objective_witness(execution: str) -> None:
    """A custom model constructor must retain its values, not re-decode winning keys."""
    source = ExtractedLinearModel(
        "pyomo",
        "6.10.1",
        "owned-test",
        (LinearVariable("x", "x", 0, 1), LinearVariable("y", "y", 0, 1)),
        (1, 2),
        5,
        False,
        (LinearRow("balance", ((0, 1), (1, 1)), 1, 1),),
    )
    solver = ModelFactory.create_owned(
        RKO,
        source,
        {
            "construction": coupled_assignment,
            "engine": "python",
            "execution": execution,
            "algorithms": ["SA"],
            "adaptation": "fixed",
            "prefill_pool": False,
            "pool_size": 4,
            "random_state": 17,
            "max_evaluations": 8,
        },
    )
    result = solver.solve(time_limit=0.2)
    assert result.model is not None and result.model.best_feasible is not None
    assert result.model.best_feasible.objective == 6.25
    assert tuple(a.value for a in result.model.best_feasible.assignments) == (0.75, 0.25)
    assert result.cost == pytest.approx(5 / 24)
    assert result.model.search_incumbent.keys == tuple(result.keys)
    assert result.model.search_incumbent.search_score == result.cost
    assert result.model.construction_used
    assert result.model.statistics is not None
    assert result.model.statistics.evaluations == result.n_evaluations
    assert result.model.statistics.infeasible_evaluations == 0
    assert len(result.model.statistics.outputs) == 1
    with pytest.raises(NotImplementedError, match="construction"):
        result.model.write_back(object())


@pytest.mark.parametrize(
    "values",
    [
        None,
        {},
        [0],
        [[0, 1]],
        np.zeros((1, 2)),
        [True, 0],
        [0, False],
        [float("nan"), 0],
        [float("inf"), 0],
        ["0", 0],
        [-0.01, 0],
        [1.01, 0],
        [0, 0.5],
        [0, 2**53 + 1],
    ],
)
def test_supplied_assignments_reject_invalid_shapes_scalars_and_domains(values: object) -> None:
    """Invalid decisions cannot be repaired, clipped or coerced into admitted evidence.

    Args:
        values (object): One invalid original-coordinate assignment.
    """
    model = CompiledLinearModel(
        (LinearVariable("x", "x", 0, 1), LinearVariable("b", "b", 0, 1, "binary")), (1, 2)
    )
    with pytest.raises(ValueError):
        model.evaluate_assignment(np.zeros(2), values, restore=True)


def test_independent_assignment_verification_does_not_repeat_restoration() -> None:
    """A row-infeasible point stays infeasible during independent verification."""
    model = CompiledLinearModel(
        (LinearVariable("x", "x", 0, 1), LinearVariable("y", "y", 0, 1)),
        (1, 2),
        (LinearRow("balance", ((0, 1), (1, 1)), 1, 1),),
        repair="continuous",
    )
    checked = model.evaluate_assignment(np.zeros(2), (0.2, 0.2))
    assert checked.values == (0.2, 0.2) and not checked.feasible
    assert checked.repair.sweeps == 0
    restored = model.evaluate_assignment(np.zeros(2), (0.2, 0.2), restore=True)
    assert restored.feasible
    assert restored.values == pytest.approx((0.5, 0.5))
    assert restored.objective == pytest.approx(1.5)


def test_supplied_assignment_cannot_change_fixed_coordinates() -> None:
    """Source-fixed values remain exact even when optional restoration is requested."""
    model = CompiledLinearModel(
        (LinearVariable("x", "x", 0, 1, fixed=0.25),), (2,), repair="continuous"
    )
    with pytest.raises(ValueError, match="fixed"):
        model.evaluate_assignment(np.empty(0), (0.3,), restore=True)
    assert model.evaluate_assignment(np.empty(0), (0.25,)).objective == 0.5
