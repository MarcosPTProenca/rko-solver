"""Checks for the violation-descent construction prototype (issue 125, follow-up).

Objective:
- Keep the generic violation-descent construction deterministic, dimension-preserving and honest.

Technical approach:
- Check feasibility through the independent fixture checker, never through the prototype.
- Verify determinism, unchanged feasible inputs and fixed key dimension.

Flow (pseudocode):
1. Decode seeded keys through the bound construction.
2. Compare each result with the independent checker and with a repeated decode.
3. Confirm a feasible default assignment is returned unchanged.

Important criteria:
- The construction uses only rows and bounds, never fixture names or objective shape.
- Failure to repair is reported as infeasible, not hidden.

Inputs/Outputs:
- Input: fixtures and seeded keys.
- Output: assertions on independent feasibility, determinism and dimension.
"""

import numpy as np
import pytest
from benchmarks.modeling.cases import PracticalModels
from benchmarks.modeling.structure_cases import StructureCases
from benchmarks.modeling.structure_construction import StructureConstruction
from benchmarks.modeling.violation_construction import ViolationConstruction

CASES = {c.name: c for c in PracticalModels.cases() + StructureCases.cases()}


def independently_feasible(name: str, values: tuple[float, ...]) -> bool:
    """Check an assignment against the fixture with the independent checker."""
    case = CASES[name]
    labelled = dict(zip((v.name for v in case.variables), values, strict=True))
    checked = PracticalModels.check(case, labelled)
    return bool(checked["feasible"])


@pytest.mark.parametrize("name", ["assignment", "facility_location", "set_cover", "knapsack_50"])
def test_repaired_results_agree_with_the_independent_checker(name: str) -> None:
    """Each repaired result has the feasibility the independent checker reports."""
    model = ViolationConstruction.bind(CASES[name])
    for row in np.random.default_rng(8).random((64, model.n_keys)):
        result = model.decode(row)
        assert result.feasible == independently_feasible(name, result.values)


def test_decoding_is_deterministic_for_fixed_keys() -> None:
    """The same keys always produce the same moves and assignment."""
    model = ViolationConstruction.bind(CASES["assignment"])
    keys = np.random.default_rng(9).random(model.n_keys)
    assert model.decode(keys).values == model.decode(keys.copy()).values


def test_key_dimension_is_preserved() -> None:
    """One key per free variable, as the shipped decoder."""
    for name in ("assignment", "transport_equalities"):
        plain = ViolationConstruction.bind(CASES[name])
        assert plain.n_keys == sum(not d.fixed for d in plain.snapshot.domains)


def test_feasible_default_assignment_is_returned_unchanged() -> None:
    """A feasible default assignment is not modified by the descent."""
    model = ViolationConstruction.bind(CASES["transport_equalities"])
    keys = np.random.default_rng(10).random(model.n_keys)
    assert model.decode(keys).feasible
    plain = model.snapshot.decode(keys)
    assert model.decode(keys).values == pytest.approx(plain.values)


@pytest.mark.parametrize(
    "name", ["assignment", "facility_location", "knapsack_50", "cardinality_linking"]
)
def test_incremental_best_move_matches_full_row_rescoring(name: str) -> None:
    """Incremental candidate scores select the same first move as full rescoring."""
    model = ViolationConstruction.bind(CASES[name])
    context = model.context
    movable = [
        index
        for index, (variable, domain) in enumerate(
            zip(context.variables, context.domains, strict=True)
        )
        if not domain.fixed and variable.domain != "continuous"
    ]
    for keys in np.random.default_rng(11).random((12, model.n_keys)):
        values, _ = StructureConstruction._box(keys, context)
        state = ViolationConstruction._state(values, context)
        current = sum(state.violations)
        expected = None
        for index in movable:
            for step in (-1.0, 1.0):
                candidate = values[index] + step
                domain = context.domains[index]
                if not domain.lower <= candidate <= domain.upper:
                    continue
                trial = list(values)
                trial[index] = candidate
                score = ViolationConstruction._violation(trial, context)
                if score < current - 1e-12 and (expected is None or score < expected[0]):
                    expected = (score, index, candidate)
        actual = ViolationConstruction._best_move(values, movable, current, context, state)
        assert (actual is None) == (expected is None)
        if actual is not None and expected is not None:
            assert actual[1:] == expected[1:]
            assert actual[0] == pytest.approx(expected[0], abs=1e-10)


@pytest.mark.parametrize(
    "name", ["assignment", "facility_location", "knapsack_50", "cardinality_linking"]
)
def test_incremental_descent_matches_full_rescoring(name: str) -> None:
    """Incremental updates preserve every move selected by the original full-rescan descent."""
    model = ViolationConstruction.bind(CASES[name])
    context = model.context
    movable = [
        index
        for index, (variable, domain) in enumerate(
            zip(context.variables, context.domains, strict=True)
        )
        if not domain.fixed and variable.domain != "continuous"
    ]
    for keys in np.random.default_rng(11).random((12, model.n_keys)):
        expected, _ = StructureConstruction._box(keys, context)
        current = ViolationConstruction._violation(expected, context)
        for _ in range(50):
            if current <= 1e-12:
                break
            best = None
            for index in movable:
                for step in (-1.0, 1.0):
                    candidate = expected[index] + step
                    domain = context.domains[index]
                    if not domain.lower <= candidate <= domain.upper:
                        continue
                    trial = list(expected)
                    trial[index] = candidate
                    score = ViolationConstruction._violation(trial, context)
                    if score < current - 1e-12 and (best is None or score < best[0]):
                        best = (score, index, candidate)
            if best is None:
                break
            current, index, candidate = best
            expected[index] = candidate
        assert ViolationConstruction()(keys, context) == expected


def test_descent_never_leaves_the_effective_search_box() -> None:
    """Moves never leave a search box narrower than the declared domain."""
    from rko_solver.modeling.construction import ModelConstruction
    from rko_solver.modeling.records import LinearRow
    from rko_solver.modeling.records import LinearVariable
    from rko_solver.modeling.source import ExtractedLinearModel

    variables = tuple(LinearVariable(f"n{i}", f"n{i}", 0, 5, "integer") for i in range(3))
    source = ExtractedLinearModel(
        "mathopt",
        "test",
        "box",
        variables,
        (1, 1, 1),
        0,
        False,
        (LinearRow("total", ((0, 1), (1, 1), (2, 1)), lower=7, upper=15),),
    )
    snapshot = source.compile(repair="none", search_bounds={f"n{i}": (0, 3) for i in range(3)})
    bound = ModelConstruction.bind(snapshot, source, ViolationConstruction())
    for keys in np.random.default_rng(12).random((32, bound.n_keys)):
        assert max(bound.decode(keys).values) <= 3
