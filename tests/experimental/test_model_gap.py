"""Original GAP oracle and native-model construction bridge correctness controls.

Objective:
- Check an actual classic binary formulation independently of generated search scores.
Technical approach:
- Enumerate original job/machine choices and reuse qualified native model builders.
Flow (pseudocode):
1. Verify the bounded original reference and a separately declared feasible control.
2. Extract each supported frontend without running its native optimizer.
3. Check constructed assignments, source identity and original objective through ordinary RKO.
Important criteria:
- Tiny fixtures and literal feasible controls are not performance or evolution evidence.
Inputs/Outputs:
- Input: original costs/resources/capacities and optional native frontends.
- Output: exact tiny references and independently checked model-result witnesses.
"""

import numpy as np
import pytest
from benchmarks.modeling_cases import PracticalModels
from benchmarks.self_evolve.gap import GeneralizedAssignment
from rko_solver import RKO
from rko_solver.modeling.adapters import ModelAdapter
from rko_solver.modeling.construction import ConstructionInput


def gap_feasible_control(keys: np.ndarray, instance: ConstructionInput) -> tuple[int, ...]:
    """Return the predeclared cost23 feasible control, never the oracle-selected optimum.

    Args:
        keys (np.ndarray): Original keys, unused by this deliberate fixed-output control.
        instance (ConstructionInput): Original ordered model data.

    Returns:
        tuple[int, ...]: Original-order binary decisions for six jobs and three machines.
    """
    return (0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0)


def test_gap_reference_is_independent_and_bounded() -> None:
    """Original enumeration certifies the tiny reference, not RKO optimization quality."""
    gap = GeneralizedAssignment.fixture()
    assert gap.evaluate((1, 2, 2, 1, 0, 0)) == (23, True)
    optimum, witness = gap.reference()
    assert optimum == 16
    assert gap.evaluate(witness) == (16, True)
    assert len(gap.formulation().variables) == 18
    assert len(gap.formulation().rows) == 9
    unseen = GeneralizedAssignment.fixture(unseen=True)
    assert len(unseen.formulation().variables) == 21
    assert unseen.reference()[0] == 17


@pytest.mark.parametrize("frontend", ["pyomo", "pulp", "mathopt"])
def test_gap_native_model_uses_protected_construction_and_original_objective(frontend: str) -> None:
    """The same original formulation passes through each current qualified native extractor.

    Args:
        frontend (str): Existing optional frontend, not a new solver/backend integration.
    """
    pytest.importorskip("ortools.math_opt.python.mathopt" if frontend == "mathopt" else frontend)
    gap = GeneralizedAssignment.fixture()
    case = gap.formulation()
    native = PracticalModels.native(case, frontend)
    before = ModelAdapter.extract(native).fingerprint
    result = RKO.from_model(
        native,
        construction=gap_feasible_control,
        algorithms=["SA"],
        adaptation="fixed",
        engine="python",
        execution="sequential",
        random_state=17,
        max_evaluations=8,
        prefill_pool=False,
        pool_size=4,
    ).solve(time_limit=0.2)
    assert result.model is not None and result.model.best_feasible is not None
    assert result.model.best_feasible.objective == 23
    ordered = tuple(a.value for a in result.model.best_feasible.assignments)
    independent = PracticalModels.check(
        case, dict(zip((v.name for v in case.variables), ordered, strict=True))
    )
    assert independent["feasible"] and independent["objective"] == 23
    assert ModelAdapter.extract(native).fingerprint == before
