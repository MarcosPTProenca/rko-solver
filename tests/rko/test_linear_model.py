"""Deterministic numerical contracts for owned linear-model decoding.

Objective:
- Prove mapping, scoring, tolerance, ownership and float64 safety before integration.

Technical approach:
- Use hand-checkable bounded affine models and exhaustive small integer boxes.
- Exercise endpoints and invalid inputs without native libraries or external solvers.

Flow (pseudocode):
1. Compile trusted linear records with explicit scoring and numerical settings.
2. Decode deterministic keys, including zero and one.
3. Compare assignments, original objectives, raw residuals and score invariants.

Important criteria:
- These tests do not claim frontend qualification or execution feasible-witness retention.

Inputs/Outputs:
- Input: tiny reproducible LP/MILP models and invalid numerical edge cases.
- Output: assertions for the generated numerical contract.
"""

import dataclasses
import math
import numpy as np
import pickle
import pytest
from rko_solver.modeling.linear import CompiledLinearModel
from rko_solver.modeling.linear import LinearRow
from rko_solver.modeling.linear import LinearVariable


def test_mixed_mapping_and_original_affine_semantics() -> None:
    """Preserve continuous, integer, binary and fixed values with affine row constants."""
    model = CompiledLinearModel(
        (
            LinearVariable("x", "same label", -2, 2),
            LinearVariable("y", "same label", -2.2, 2.8, "integer"),
            LinearVariable("z", "binary", -3, 4, "binary"),
            LinearVariable("f", "fixed", 0, 8, "continuous", 3),
        ),
        (2, -3, 4, 5),
        (LinearRow("range", ((0, 1), (1, 2), (2, -1), (3, 3)), 2, 10, -1),),
        objective_constant=-7,
    )
    assert model.n_keys == 3
    result = model.decode(np.array([0.25, 0.5, 1]))
    assert result.values == (-1, 0, 1, 3)
    assert result.objective == 10
    assert result.rows[0].activity == 6
    assert result.feasible
    assert 0 <= result.score <= 0.5
    assert CompiledLinearModel.cost(result) == result.score


@pytest.mark.parametrize("lower, upper", [(-2.8, 2.8), (-4, -1), (0, 1)])
def test_integer_bins_endpoints_and_ties(lower: float, upper: float) -> None:
    """Use half-up affine integer bins, including negative and fractional declared bounds.

    Args:
        lower (float): Declared integer lower endpoint.
        upper (float): Declared integer upper endpoint.
    """
    model = CompiledLinearModel((LinearVariable("x", "x", lower, upper, "integer"),), (1,))
    lb, ub = math.ceil(lower), math.floor(upper)
    assert model.decode(np.array([0])).values == (lb,)
    assert model.decode(np.array([1])).values == (ub,)
    width = ub - lb + 1
    for index in range(width):
        assert model.decode(np.array([(index + 0.5) / width])).values == (lb + index,)
    tie = 1 / width
    assert model.decode(np.array([tie])).values == (lb + 1,)


def test_coupled_continuous_equality_checks_without_repairing_keys() -> None:
    """Accept a true balance while retaining original assignments and strict row tolerance."""
    model = CompiledLinearModel(
        (LinearVariable("x", "x", 0, 1), LinearVariable("y", "y", 0, 1)),
        (1, 2),
        (LinearRow("balance", ((0, 1), (1, 1)), 1, 1),),
    )
    feasible = model.decode(np.array([0.25, 0.75]))
    violated = model.decode(np.array([0.25, 0.7501]))
    assert feasible.feasible and feasible.objective == 1.75
    assert violated.values == (0.25, 0.7501)
    assert not violated.feasible
    assert violated.rows[0].upper_residual == pytest.approx(0.0001)
    assert violated.rows[0].upper_violation == pytest.approx(0.000099)
    assert feasible.score < violated.score


def test_finite_weight_can_prefer_an_infeasible_equality_assignment() -> None:
    """Preserve penalty tradeoffs without claiming finite weights enforce equality feasibility."""
    model = CompiledLinearModel(
        (LinearVariable("x", "x", 0, 1), LinearVariable("y", "y", 0, 1)),
        (1, 2),
        (LinearRow("balance", ((0, 1), (1, 1)), 1, 1),),
        scoring="weighted_penalty",
        penalty_weight=1000,
    )
    feasible = model.decode(np.array([1, 0]))
    violated = model.decode(np.array([0.9995, 0]))
    assert feasible.feasible and not violated.feasible
    assert violated.rows[0].lower_residual == pytest.approx(0.0005)
    assert violated.score == pytest.approx(0.99975)
    assert violated.score < feasible.score == 1


def test_fixed_and_constant_models_need_no_dummy_keys() -> None:
    """Evaluate singleton domains, fixed unbounded variables and empty constant models."""
    fixed = CompiledLinearModel(
        (LinearVariable("x", "x", fixed=4), LinearVariable("y", "y", 0.1, 1.1, "integer")),
        (2, 3),
        (LinearRow("equality", ((0, 1), (1, -1)), 3, 3),),
        objective_constant=-2,
    )
    assert fixed.n_keys == 0
    evaluation = fixed.decode(np.array([]))
    assert evaluation.values == (4, 1)
    assert evaluation.objective == 9
    assert evaluation.feasible
    empty = CompiledLinearModel((), (), (LinearRow("impossible", (), 1, 1),))
    assert empty.decode(np.array([])).feasible is False
    assert CompiledLinearModel((), (), objective_constant=-7).decode(np.array([])).objective == -7


def test_ranged_lower_upper_and_constant_row_sides() -> None:
    """Check both row sides and retain raw versus tolerance-adjusted residuals."""
    model = CompiledLinearModel(
        (LinearVariable("x", "x", 0, 4),),
        (0,),
        (
            LinearRow("range", ((0, 1),), 1, 3),
            LinearRow("lower", ((0, 1),), 1),
            LinearRow("upper", ((0, 1),), upper=3),
            LinearRow("constant", constant=2, lower=2, upper=2),
        ),
        absolute_tolerance=0.25,
    )
    below = model.decode(np.array([0.125]))
    assert below.rows[0].lower_residual == 0.5
    assert below.rows[0].lower_violation == 0.25
    assert below.rows[0].upper_residual == 0
    assert not below.feasible
    above = model.decode(np.array([0.875]))
    assert above.rows[0].upper_residual == 0.5
    assert above.rows[0].upper_violation == 0.25
    assert above.rows[-1].activity == 2
    assert model.decode(np.array([0.1875])).feasible
    assert model.decode(np.array([0.8125])).feasible


def test_absolute_relative_tolerances_are_per_side() -> None:
    """Apply the relative allowance to absolute RHS magnitudes, never variable bounds."""
    model = CompiledLinearModel(
        (LinearVariable("x", "x", -200, 200),),
        (1,),
        (LinearRow("range", ((0, 1),), -100, 50),),
        absolute_tolerance=1,
        relative_tolerance=0.1,
    )
    assert model.decode(np.array([(200 - 111) / 400])).feasible
    assert model.decode(np.array([(200 + 56) / 400])).feasible
    assert not model.decode(np.array([(200 - 112) / 400])).feasible
    assert not model.decode(np.array([(200 + 57) / 400])).feasible
    with pytest.raises(ValueError, match="finite keys"):
        model.decode(np.array([1.001]))


@pytest.mark.parametrize("maximize", [False, True])
def test_feasibility_bands_and_objective_direction(maximize: bool) -> None:
    """Prove feasible priority and direction over an exhaustively checked integer box.

    Args:
        maximize (bool): Original objective direction.
    """
    model = CompiledLinearModel(
        (LinearVariable("x", "x", -2, 3, "integer"),),
        (7,),
        (LinearRow("range", ((0, 1),), -1, 2),),
        objective_constant=-42,
        maximize=maximize,
    )
    results = [model.decode(np.array([(index + 0.5) / 6])) for index in range(6)]
    feasible = [result for result in results if result.feasible]
    assert all(0 <= result.score <= 0.5 for result in feasible)
    assert all(result.score >= 1 for result in results if not result.feasible)
    ordered = sorted(feasible, key=lambda result: result.objective, reverse=maximize)
    assert [result.score for result in ordered] == sorted(result.score for result in ordered)
    assert max(result.score for result in feasible) < min(
        result.score for result in results if not result.feasible
    )


@pytest.mark.parametrize("maximize", [False, True])
def test_weighted_penalty_raw_units_and_tolerance_gate(maximize: bool) -> None:
    """Gate by tolerance but square raw residuals rather than normalized violations.

    Args:
        maximize (bool): Original objective direction.
    """
    model = CompiledLinearModel(
        (LinearVariable("x", "x", 0, 4),),
        (3,),
        (LinearRow("equality", ((0, 1),), 2, 2),),
        objective_constant=-10,
        maximize=maximize,
        scoring="weighted_penalty",
        penalty_weight=5,
        absolute_tolerance=0.5,
    )
    outside = model.decode(np.array([0.25]))
    oriented = -outside.objective if maximize else outside.objective
    assert outside.score == oriented + 5
    assert outside.rows[0].lower_violation == 0.5
    on_gate = model.decode(np.array([0.375]))
    assert on_gate.feasible
    assert on_gate.rows[0].lower_residual == 0.5
    assert on_gate.score == (-on_gate.objective if maximize else on_gate.objective)


def test_restrictive_overrides_and_provenance() -> None:
    """Intersect overrides without relaxing declared or integer/binary domains."""
    variables = (
        LinearVariable("x", "x", 0, 10),
        LinearVariable("y", "y", domain="integer"),
        LinearVariable("z", "z", domain="binary"),
    )
    model = CompiledLinearModel(
        variables,
        (1, 1, 1),
        search_bounds={"x": (-10, 7), "y": (-2.2, 2.8), "z": (-100, 100)},
    )
    assert model.decode(np.array([0, 0, 0])).values == (0, -2, 0)
    assert model.decode(np.array([1, 1, 1])).values == (7, 2, 1)
    assert [domain.restricted for domain in model.domains] == [True, True, False]
    assert model.variables == variables
    assert variables[1].lower == -math.inf


def test_snapshot_and_evidence_are_owned_immutable_and_pickleable() -> None:
    """Retain no caller-owned sparse containers or writable arrays across serialization."""
    coefficients = [[0, 2.0]]
    model = CompiledLinearModel(
        (LinearVariable("x", "x", 0, 1),),
        (1,),
        (LinearRow("row", coefficients, upper=1),),
    )
    coefficients[0][1] = 1000
    cloned = pickle.loads(pickle.dumps(model))
    for snapshot in (model, cloned):
        evidence = snapshot.decode(np.array([0.5]))
        assert evidence.rows[0].activity == 1
        assert not hasattr(snapshot, "__dict__")
        assert snapshot.rows[0].coefficients == ((0, 2),)
        with pytest.raises(dataclasses.FrozenInstanceError):
            evidence.objective = 0
        with pytest.raises(ValueError, match="WRITEABLE"):
            np.frombuffer(snapshot._coefficients).setflags(write=True)
        with pytest.raises(dataclasses.FrozenInstanceError):
            snapshot.maximize = True
        assert pickle.loads(pickle.dumps(evidence)) == evidence


@pytest.mark.parametrize("keys", [[-0.01], [1.01], [math.nan], [math.inf], [], [[0.5]]])
def test_invalid_keys_fail_before_scoring(keys: list[object]) -> None:
    """Reject out-of-range, nonfinite and malformed chromosomes.

    Args:
        keys (list[object]): Invalid chromosome payload.
    """
    model = CompiledLinearModel((LinearVariable("x", "x", 0, 1),), (1,))
    with pytest.raises(ValueError, match="finite keys"):
        model.decode(np.array(keys))


@pytest.mark.parametrize(
    "variable, message",
    [
        (LinearVariable("x", "x"), "finite effective bounds"),
        (LinearVariable("x", "x", 1, 0), "lower bound"),
        (LinearVariable("x", "x", math.nan, 1), "envelope"),
        (LinearVariable("x", "x", 0.2, 0.8, "integer"), "empty effective domain"),
        (LinearVariable("x", "x", 2, 4, "binary"), "empty effective domain"),
        (LinearVariable("x", "x", 0, 1, "semicontinuous"), "unsupported domain"),
        (LinearVariable("x", "x", 0, 1, fixed=2), "fixed value"),
        (LinearVariable("x", "x", -1, 1, "integer", 0.5), "fixed value"),
        (LinearVariable("x", "x", 0, 2**53, "integer"), "integer domain"),
        (LinearVariable("x", "x", -1e101, 1), "envelope"),
    ],
)
def test_invalid_variable_data(variable: LinearVariable, message: str) -> None:
    """Reject unsupported domains and unsafe data with variable-specific diagnostics.

    Args:
        variable (LinearVariable): Invalid extracted variable.
        message (str): Expected diagnostic fragment.
    """
    with pytest.raises(ValueError, match=message):
        CompiledLinearModel((variable,), (1,))


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"scoring": "unknown"}, "scoring"),
        ({"scoring": "weighted_penalty"}, "penalty_weight"),
        ({"scoring": "weighted_penalty", "penalty_weight": 0}, "positive"),
        ({"scoring": "weighted_penalty", "penalty_weight": math.inf}, "envelope"),
        ({"penalty_weight": 1}, "only supported"),
        ({"absolute_tolerance": -1}, "nonnegative"),
        ({"relative_tolerance": math.nan}, "envelope"),
        ({"maximize": 1}, "boolean"),
        ({"search_bounds": {"unknown": (0, 1)}}, "unknown stable"),
        ({"search_bounds": {"x": (2, 3)}}, "empty effective"),
        ({"search_bounds": {"x": (0,)}}, "two endpoints"),
    ],
)
def test_invalid_settings(settings: dict[str, object], message: str) -> None:
    """Reject ambiguous scoring and incomplete or contradictory override configurations.

    Args:
        settings (dict[str, object]): Invalid constructor options.
        message (str): Expected diagnostic fragment.
    """
    with pytest.raises(ValueError, match=message):
        CompiledLinearModel((LinearVariable("x", "x", 0, 1),), (1,), **settings)


def test_bad_identities_rows_and_objective_data() -> None:
    """Reject duplicate identities/indices and malformed or nonfinite affine data."""
    variable = LinearVariable("x", "label", 0, 1)
    with pytest.raises(ValueError, match="Duplicate stable variable"):
        CompiledLinearModel((variable, variable), (1, 1))
    with pytest.raises(ValueError, match="nonempty stable"):
        CompiledLinearModel((LinearVariable("", "label", 0, 1),), (1,))
    with pytest.raises(ValueError, match="one coefficient"):
        CompiledLinearModel((variable,), ())
    with pytest.raises(ValueError, match="envelope"):
        CompiledLinearModel((variable,), (math.inf,))
    for coefficients in (((0, 1), (0, 2)), ((-1, 1),), ((1, 1),)):
        with pytest.raises(ValueError, match="variable index"):
            CompiledLinearModel((variable,), (1,), (LinearRow("row", coefficients),))
    with pytest.raises(ValueError, match="Duplicate stable row"):
        CompiledLinearModel((variable,), (1,), (LinearRow("r"), LinearRow("r")))
    with pytest.raises(ValueError, match="lower bound"):
        CompiledLinearModel((variable,), (1,), (LinearRow("r", lower=2, upper=1),))


def test_unsafe_objective_row_tolerance_and_penalty_ranges() -> None:
    """Fail preflight before unsafe accumulated expressions or weighted scores run."""
    variable = LinearVariable("x", "x", -1e80, 1e80)
    with pytest.raises(ValueError, match="objective accumulation"):
        CompiledLinearModel((variable,), (1e80,))
    with pytest.raises(ValueError, match="row 'r' accumulation"):
        CompiledLinearModel((variable,), (1,), (LinearRow("r", ((0, 1e80),)),))
    with pytest.raises(ValueError, match="tolerance"):
        CompiledLinearModel(
            (variable,), (1,), (LinearRow("r", upper=1e90),), relative_tolerance=1e80
        )
    with pytest.raises(ValueError, match="worst-case score"):
        CompiledLinearModel(
            (variable,),
            (1,),
            (LinearRow("r", ((0, 1),), upper=0),),
            scoring="weighted_penalty",
            penalty_weight=1,
        )


def test_large_safe_and_zero_range_objectives_remain_finite() -> None:
    """Keep feasible bands finite across zero coefficients and large signed constants."""
    for maximize in (False, True):
        constant = CompiledLinearModel(
            (LinearVariable("x", "x", -1, 1),),
            (0,),
            objective_constant=-1e90,
            maximize=maximize,
        )
        assert constant.decode(np.array([0.5])).score == 0
        scaled = CompiledLinearModel(
            (LinearVariable("x", "x", -1e40, 1e40),),
            (1e40,),
            objective_constant=-1e90,
            maximize=maximize,
        )
        for key in (0, 0.5, 1):
            evidence = scaled.decode(np.array([key]))
            assert math.isfinite(evidence.objective)
            assert 0 <= evidence.score <= 0.5


def test_inexact_large_integer_data_cannot_collapse_continuous_domains() -> None:
    """Reject lossy integer-to-float coercion before a nonempty interval becomes falsely fixed."""
    lower = 10**90
    assert float(lower) == float(lower + 1)
    with pytest.raises(ValueError, match="exactly representable"):
        CompiledLinearModel((LinearVariable("x", "x", lower, lower + 1),), (1,))
    with pytest.raises(ValueError, match="exactly representable"):
        CompiledLinearModel((), (), objective_constant=lower)
