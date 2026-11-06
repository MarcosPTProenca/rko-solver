"""Checks for the decoder key-feasibility bench (baseline measurement, issue 125 phase 0).

Objective:
- Keep the structure fixtures and the baseline measurement honest before any new decoder exists.

Technical approach:
- Check each structure fixture against a hand-built optimal witness, independent of RKO decoding.
- Run the bench on tiny seeded samples and assert only deterministic, documented facts.

Flow (pseudocode):
1. Evaluate known optimal witnesses with the independent fixture checker.
2. Measure the box-only and continuous-restoration policies on seeded keys.
3. Assert repeatability, independent agreement and refusal to overwrite output.

Important criteria:
- No wall-clock threshold and no claim about search quality; feasibility of sampled keys only.

Inputs/Outputs:
- Input: structure fixtures and small seeded key samples.
- Output: assertions on fixtures, record schema and baseline facts.
"""

import json
import pytest
from benchmarks.modeling.cases import PracticalModels
from benchmarks.modeling.key_feasibility import KeyFeasibilityStudy
from benchmarks.modeling.structure_cases import StructureCases
from pathlib import Path

WITNESSES = {
    "coupled_equality": {"x": 1.0, "y": 0.0},
    "cardinality_linking": {
        **{f"z_{i}": float(i in (0, 1, 3)) for i in range(6)},
        "w_0": 0.1,
        "w_1": 0.45,
        "w_2": 0.0,
        "w_3": 0.45,
        "w_4": 0.0,
        "w_5": 0.0,
    },
    "set_cover": {f"s_{i}": float(i in (0, 2, 3)) for i in range(6)},
}


def test_structure_fixtures_have_independently_checked_optimal_witnesses() -> None:
    """Each fixture's declared optimum is reached by a feasible hand-built witness."""
    cases = {case.name: case for case in StructureCases.cases()}
    assert set(cases) == set(WITNESSES)
    for name, values in WITNESSES.items():
        checked = PracticalModels.check(cases[name], values)
        assert checked["feasible"], name
        assert checked["objective"] == pytest.approx(cases[name].expected), name


def test_structure_fixtures_match_the_exact_reference() -> None:
    """The optional exact oracle agrees with every declared optimum.

    Only the objective is compared: HiGHS may return an equally good optimum whose integer values
    carry rounding noise that the shared `reference` checker rejects.
    """
    pytest.importorskip("scipy.optimize")
    for case in StructureCases.cases():
        result, _ = PracticalModels._solve_reference(case)
        assert result.status == 0, case.name
        assert result.fun == pytest.approx(case.expected), case.name


def test_box_decoding_finds_no_coupled_equality_but_restoration_does() -> None:
    """Independent keys never satisfy `x + y = 1` (docs/modeling.md); restoration can."""
    case = next(c for c in StructureCases.cases() if c.name == "coupled_equality")
    box = KeyFeasibilityStudy.measure(case, "none", keys=256, seed=2026)
    restored = KeyFeasibilityStudy.measure(case, "continuous", keys=256, seed=2026)
    assert box["feasible"] == 0
    assert restored["feasible"] > 0
    assert box["independent_errors"] == restored["independent_errors"] == 0


def test_measurement_is_repeatable_and_reports_quality_on_feasible_keys() -> None:
    """Same seed gives the same counts and objectives; gaps exist only when keys are feasible."""
    case = next(c for c in PracticalModels.cases() if c.name == "assignment")
    first = KeyFeasibilityStudy.measure(case, "continuous", keys=64, seed=7)
    second = KeyFeasibilityStudy.measure(case, "continuous", keys=64, seed=7)
    for field in ("feasible", "best_gap", "median_gap", "keys"):
        assert first[field] == second[field]
    assert first["keys"] == 64
    if first["feasible"] == 0:
        assert first["best_gap"] is None and first["median_gap"] is None
    else:
        assert first["best_gap"] is not None and first["best_gap"] >= -1e-9


def test_run_covers_every_feasible_case_under_both_policies() -> None:
    """The bench reports baseline and structure cases, skipping the deliberately infeasible one."""
    records = KeyFeasibilityStudy.run(keys=16, seed=1)
    names = {record["case"] for record in records}
    assert {"assignment", "coupled_equality", "set_cover", "cardinality_linking"} <= names
    assert "production_inconsistent" not in names
    for record in records:
        assert {p["repair"] for p in record["policies"]} == {"none", "continuous"}


def test_output_refuses_to_overwrite(tmp_path: Path) -> None:
    """A measurement file is never overwritten."""
    target = tmp_path / "result.json"
    KeyFeasibilityStudy.write(target, [{"case": "x"}])
    assert json.loads(target.read_text()) == [{"case": "x"}]
    with pytest.raises(FileExistsError):
        KeyFeasibilityStudy.write(target, [])
