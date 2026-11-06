"""Checks for the structure-aware construction prototype (issue 125, layer L1).

Objective:
- Verify the prototype detects binary all-ones rows and keeps the decoder contract.

Technical approach:
- Bind the prototype to compiled fixtures through `ModelConstruction` and decode seeded keys.
- Check feasibility by the independent fixture checker, never by the prototype itself.

Flow (pseudocode):
1. Decode seeded keys for structure fixtures and unstructured fixtures.
2. Check feasibility, structural counts, determinism and tie behaviour.
3. Check that models without detected structure decode exactly like the default mapping.

Important criteria:
- One key per free variable is kept; fixed variables are never moved.
- Ties between equal keys resolve by variable index, so decoding is deterministic.

Inputs/Outputs:
- Input: fixtures and seeded keys.
- Output: assertions on feasibility, counts and equality with the default mapping.
"""

import numpy as np
import pytest
from benchmarks.modeling.cases import PracticalModels
from benchmarks.modeling.key_feasibility import KeyFeasibilityStudy
from benchmarks.modeling.structure_cases import StructureCases
from benchmarks.modeling.structure_construction import StructureConstruction

CASES = {c.name: c for c in PracticalModels.cases() + StructureCases.cases()}


def decode(name: str, keys: np.ndarray) -> object:
    """Decode one key vector through the prototype bound to a fixture."""
    model = StructureConstruction.bind(CASES[name])
    return model.decode(keys)


def independent_feasible(name: str, result: object) -> bool:
    """Check a decoded result with the independent fixture checker."""
    case = CASES[name]
    values = {v.name: x for v, x in zip(case.variables, result.values, strict=True)}
    return bool(PracticalModels.check(case, values)["feasible"])


@pytest.mark.parametrize("name", ["assignment", "set_cover", "cardinality_linking"])
def test_structured_fixtures_are_feasible_for_every_sampled_key(name: str) -> None:
    """Detected structure makes every seeded random key decode to a feasible solution."""
    model = StructureConstruction.bind(CASES[name])
    keys = np.random.default_rng(5).random((200, model.n_keys))
    for row in keys:
        result = model.decode(row)
        assert result.feasible and independent_feasible(name, result)


def test_cardinality_rows_pick_exactly_k_variables() -> None:
    """The `sum(z) = 3` row selects three assets whatever the keys are."""
    model = StructureConstruction.bind(CASES["cardinality_linking"])
    for row in np.random.default_rng(1).random((50, model.n_keys)):
        assert sum(model.decode(row).values[:6]) == 3


def test_higher_keys_win_and_equal_keys_resolve_by_index() -> None:
    """Selection follows key order, and ties go to the lower variable index."""
    model = StructureConstruction.bind(CASES["cardinality_linking"])
    keys = np.full(model.n_keys, 0.5)
    chosen = [i for i, v in enumerate(model.decode(keys).values[:6]) if v == 1]
    assert chosen == [0, 1, 2]
    keys[[5, 4, 3]] = [0.99, 0.98, 0.97]
    chosen = [i for i, v in enumerate(model.decode(keys).values[:6]) if v == 1]
    assert chosen == [3, 4, 5]


def test_decoding_is_deterministic() -> None:
    """The same keys always give the same assignment."""
    model = StructureConstruction.bind(CASES["assignment"])
    keys = np.random.default_rng(9).random(model.n_keys)
    assert model.decode(keys).values == model.decode(keys.copy()).values


@pytest.mark.parametrize("name", ["production_lp", "production_integer", "coupled_equality"])
def test_models_without_binary_rows_decode_like_the_default_mapping(name: str) -> None:
    """With no structure detected the prototype changes nothing."""
    plain = KeyFeasibilityStudy._snapshot(CASES[name], "continuous")
    model = StructureConstruction.bind(CASES[name])
    for row in np.random.default_rng(3).random((20, plain.n_keys)):
        assert model.decode(row).values == pytest.approx(plain.decode(row).values)


def test_keys_dimension_is_preserved() -> None:
    """One key per free variable, as the default decoder."""
    for name in ("assignment", "cardinality_linking", "set_cover"):
        plain = KeyFeasibilityStudy._snapshot(CASES[name], "continuous")
        assert StructureConstruction.bind(CASES[name]).n_keys == plain.n_keys
