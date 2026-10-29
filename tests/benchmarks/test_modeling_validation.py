"""Independent acceptance checks for public practical native LP/IP/MILP formulations.

Objective:
- Validate original mathematical data, independent references and native generated semantics.

Technical approach:
- Hand-check documented production/transportation/assignment witnesses and Knapsack input identity.
- Compare every frontend and scoring mode against independent fixture arithmetic for fixed keys.
- Keep benchmark errors, incomplete runs and output-overwrite protection explicitly tested.

Flow (pseudocode):
1. Verify original public fixture answers and exact domain checks without optional libraries.
2. Resolve optional SciPy/HiGHS references separately from generated RKO evaluation.
3. Decode known/reference and random chromosomes in every available native frontend.
4. Verify campaign outcomes remain complete and errors cannot appear as success.

Important criteria:
- Timed search tests do not assert random optimum discovery or require solver-assisted repair.
- Reference points are used only for deterministic conformance, never to initialize optimization.

Inputs/Outputs:
- Input: offline practical cases and optional native/reference runtimes.
- Output: independent oracle, conversion, timing and measurement-integrity assertions.
"""

import json
import math
import numpy as np
import pytest
from benchmarks.modeling.cases import PracticalCase
from benchmarks.modeling.cases import PracticalModels
from benchmarks.modeling.validation import ModelValidationCampaign
from pathlib import Path
from rko_solver import RKO
from rko_solver.modeling.adapters import ModelAdapter
from rko_solver.modeling.linear import CompiledLinearModel
from typing import Any

CASES = PracticalModels.cases()


@pytest.fixture(scope="module")
def references() -> dict[str, dict[str, Any]]:
    """Compute independent optional reference answers without invoking RKO callbacks.

    Returns:
        dict[str, dict[str, Any]]: Exact-model reference status/points/objectives by case.
    """
    pytest.importorskip("scipy.optimize")
    return {case.name: PracticalModels.reference(case) for case in CASES}


def test_public_fixture_shapes_and_original_known_witnesses() -> None:
    """Check hand-verifiable production, shipping and assignment answers without an optimizer."""
    by_name = {case.name: case for case in CASES}
    assert len(by_name) == 8
    assert {case.family for case in CASES} == {"LP", "IP", "MILP"}
    for case in CASES:
        assert [v.name for v in case.variables] == sorted(v.name for v in case.variables)
        assert len({v.name for v in case.variables}) == len(case.variables)
        assert len(case.objective) == len(case.variables)
        assert all(math.isfinite(v.lower) and math.isfinite(v.upper) for v in case.variables)
        data = ModelValidationCampaign.definition(case)
        assert len(data["fixture_sha256"]) == 64
        json.dumps(data, allow_nan=False)
    for name in ("production_lp", "production_integer"):
        check = PracticalModels.check(by_name[name], {"x": 20, "y": 60})
        assert check["feasible"] and check["objective"] == 2600
    check = PracticalModels.check(by_name["production_setup"], {"setup_x": 1, "x": 20, "y": 60})
    assert check["feasible"] and check["objective"] == 2500
    values = {f"route_{w}_{b}": 0 for w in range(2) for b in range(5)}
    values.update(
        route_0_0=300, route_0_4=700, route_1_0=200, route_1_1=900, route_1_2=1800, route_1_3=200
    )
    for name in ("transport_lp", "transport_equalities"):
        check = PracticalModels.check(by_name[name], values)
        assert check["feasible"] and check["objective"] == 8600
    values = {v.name: 0 for v in by_name["assignment"].variables}
    values.update(worker_0_task_3=1, worker_1_task_2=1, worker_2_task_1=1, worker_3_task_0=1)
    check = PracticalModels.check(by_name["assignment"], values)
    assert check["feasible"] and check["objective"] == 265


def test_independent_checker_enforces_exact_domains_and_never_calls_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep row tolerances separate from bounds/integrality and from decoder implementation.

    Args:
        monkeypatch (pytest.MonkeyPatch): Make generated decoding fail if wrongly reused.
    """

    def forbidden(*args: object) -> None:
        """Reject decoder reuse inside supposedly independent fixture arithmetic.

        Args:
            args (object): Unexpected callback arguments.

        Raises:
            AssertionError: Always, since independent checking cannot invoke the decoder.
        """
        raise AssertionError("Independent validation must not call the generated decoder.")

    monkeypatch.setattr(CompiledLinearModel, "decode", forbidden)
    by_name = {case.name: case for case in CASES}
    assert PracticalModels.check(by_name["production_lp"], {"x": 20, "y": 60})["feasible"]
    assert not PracticalModels.check(
        by_name["production_lp"], {"x": -0.01, "y": 0}, absolute_tolerance=100
    )["bounds_ok"]
    values = {v.name: 0 for v in by_name["knapsack_50"].variables}
    values["item_000"] = 0.5
    assert not PracticalModels.check(by_name["knapsack_50"], values)["integrality_ok"]
    assert not PracticalModels.check(by_name["production_inconsistent"], {"x": 20, "y": 0})[
        "feasible"
    ]
    with pytest.raises(ValueError, match="complete finite"):
        PracticalModels.check(by_name["production_lp"], {"x": math.nan, "y": 0})


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_optional_references_match_documented_answers(
    case: PracticalCase,
    references: dict[str, dict[str, Any]],
) -> None:
    """Validate each optional reference status/optimum against independently known model answers.

    Args:
        case (PracticalCase): Original public or explicitly adapted formulation.
        references (dict): Separately computed optional SciPy/HiGHS answers.
    """
    reference = references[case.name]
    if case.expected is None:
        assert reference["status"] == "reference_infeasible"
        assert reference["objective"] is None and reference["values"] is None
    else:
        assert reference["status"] == "reference_optimal"
        assert reference["objective"] == pytest.approx(case.expected, abs=1e-7)
        assert PracticalModels.check(case, reference["values"])["feasible"]


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
@pytest.mark.parametrize("frontend", ["pyomo", "pulp", "mathopt"])
@pytest.mark.parametrize("scoring", ["feasibility_first", "weighted_penalty"])
def test_native_public_models_match_independent_fixed_key_arithmetic(
    case: PracticalCase,
    frontend: str,
    scoring: str,
    references: dict[str, dict[str, Any]],
) -> None:
    """Check known reference points and reproducible random keys without a search quality assertion.

    Args:
        case (PracticalCase): Original benchmark model.
        frontend (str): Optional qualified native frontend.
        scoring (str): Existing generated scoring mode.
        references (dict): Independent reference points, never used to initialize optimization.
    """
    pytest.importorskip(
        {"pyomo": "pyomo.environ", "pulp": "pulp", "mathopt": "ortools.math_opt.python.mathopt"}[
            frontend
        ]
    )
    native = PracticalModels.native(case, frontend)
    solver = RKO.from_model(
        native, scoring=scoring, penalty_weight=1000 if scoring == "weighted_penalty" else None
    )
    source = solver._model_source
    assert [v.label for v in source.variables] == [v.name for v in case.variables]
    chromosomes = list(np.random.default_rng(1729).random((16, solver.n_variables)))
    reference = references[case.name]
    if reference["values"] is not None:
        keys = []
        for variable in case.variables:
            value = reference["values"][variable.name]
            keys.append(
                (value - variable.lower) / (variable.upper - variable.lower)
                if variable.kind == "continuous"
                else (value - variable.lower + 0.5) / (variable.upper - variable.lower + 1)
            )
        chromosomes.append(np.array(keys))
    for keys in chromosomes:
        evaluation = solver.decoder(keys)
        values = dict(zip((v.label for v in source.variables), evaluation.values, strict=True))
        check = PracticalModels.check(case, values)
        assert evaluation.objective == pytest.approx(check["objective"], abs=1e-8)
        assert evaluation.feasible == check["feasible"]
        assert math.isfinite(evaluation.score)
    if reference["values"] is not None:
        assert evaluation.feasible
        assert evaluation.objective == pytest.approx(case.expected, abs=1e-7)
    assert solver.result_ is None
    assert ModelAdapter.extract(native).fingerprint == source.fingerprint


@pytest.mark.parametrize("frontend", ["pyomo", "pulp", "mathopt"])
@pytest.mark.parametrize("scoring", ["feasibility_first", "weighted_penalty"])
def test_short_public_model_run_has_independent_metrics(
    frontend: str,
    scoring: str,
    references: dict[str, dict[str, Any]],
) -> None:
    """Verify actual public execution metrics without requiring timed optimum discovery.

    Args:
        frontend (str): Optional qualified native frontend.
        scoring (str): Existing generated scoring mode.
        references (dict): Independent exact-model reference answers.
    """
    pytest.importorskip(
        {"pyomo": "pyomo.environ", "pulp": "pulp", "mathopt": "ortools.math_opt.python.mathopt"}[
            frontend
        ]
    )
    case = next(case for case in CASES if case.name == "production_lp")
    record = ModelValidationCampaign.observe(
        case, frontend, scoring, 42, time_limit=0.02, reference=references[case.name]
    )
    assert record["status"] in ("feasible", "no_feasible_found")
    assert record["n_evaluations"] > 0
    assert record["reference_objective"] == 2600
    if record["status"] == "feasible":
        assert record["independent_feasible_witness"]["feasible"]
        assert 0 < record["time_to_first_feasible_seconds"] <= record["solve_seconds"]
    json.dumps(record, allow_nan=False)


def test_campaign_errors_are_kept_and_output_is_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve every failed run and detect missing/duplicate seeds instead of cherry-picking.

    Args:
        tmp_path (Path): Temporary observation paths.
        monkeypatch (pytest.MonkeyPatch): Replace native construction with an intentional failure.
    """
    case = CASES[0]

    def cases() -> tuple[PracticalCase, ...]:
        """Return the one mathematical case used to exercise campaign integrity.

        Returns:
            tuple[PracticalCase, ...]: Single known feasible production model.
        """
        return (case,)

    def reference(original: PracticalCase) -> dict[str, Any]:
        """Return an already hand-verified reference without optional solver dependencies.

        Args:
            original (PracticalCase): Known production model.

        Returns:
            dict[str, Any]: Explicit test-only reference objective and status.
        """
        return {
            "status": "reference_optimal",
            "objective": original.expected,
            "values": {"x": 20, "y": 60},
            "elapsed_seconds": 0,
        }

    def fail_native(original: PracticalCase, frontend: str) -> None:
        """Raise a deterministic native-construction error that cannot become partial success.

        Args:
            original (PracticalCase): Original production model.
            frontend (str): Planned native frontend.

        Raises:
            RuntimeError: Always to exercise complete error recording.
        """
        raise RuntimeError("intentional benchmark integrity test")

    monkeypatch.setattr(PracticalModels, "cases", cases)
    monkeypatch.setattr(PracticalModels, "reference", reference)
    monkeypatch.setattr(PracticalModels, "native", fail_native)
    output = tmp_path / "failed.jsonl"
    with pytest.raises(RuntimeError, match="6 campaign runs failed"):
        ModelValidationCampaign.run(output, seeds=1, time_limit=0.01)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 7
    summaries = ModelValidationCampaign.summary(output)
    assert len(summaries) == 6
    assert all(group["errors"] == 1 and group["missing_runs"] == 0 for group in summaries)
    assert all(group["feasible_rate_over_planned"] == 0 for group in summaries)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        ModelValidationCampaign.run(output, seeds=1, time_limit=0.01)
    assert output.read_bytes() == before
    partial = tmp_path / "partial.jsonl"
    partial.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    assert all(group["missing_runs"] == 1 for group in ModelValidationCampaign.summary(partial))
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        "\n".join(json.dumps(record) for record in records[:2] + records[1:2]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        ModelValidationCampaign.summary(duplicate)


@pytest.mark.parametrize("seeds, limit", [(0, 1), (True, 1), (1, 0), (1, math.nan), (1, True)])
def test_invalid_campaign_settings_fail_before_output(
    tmp_path: Path, seeds: int, limit: float
) -> None:
    """Reject invalid measurement settings before reference calls or artifact creation.

    Args:
        tmp_path (Path): Temporary unused observation directory.
        seeds (int): Invalid seed count or valid count paired with an invalid budget.
        limit (float): Requested search budget.
    """
    output = tmp_path / "invalid.jsonl"
    with pytest.raises(ValueError):
        ModelValidationCampaign.run(output, seeds=seeds, time_limit=limit)
    assert not output.exists()


@pytest.mark.parametrize(
    "filename",
    [
        "references/modeling-validation-reference-13b6934.jsonl",
        "references/modeling-validation-box-reference-13b6934.jsonl",
    ],
)
def test_identified_reference_preserves_every_outcome_and_fixture(filename: str) -> None:
    """Check preserved historical policy evidence without claiming renamed-source qualification.

    Args:
        filename (str): Explicit continuous-restoration or box-only reference artifact.
    """
    root = Path(__file__).resolve().parents[2]
    path = root / "benchmarks" / filename
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    metadata, runs = records[0], records[1:]
    assert len(runs) == 480
    assert metadata["seeds"] == list(range(10))
    assert metadata["repair"] in ("none", "continuous")
    assert metadata["repair_max_sweeps"] == 1024
    definitions = {case["name"]: case for case in metadata["cases"]}
    for case in CASES:
        assert (
            definitions[case.name]["fixture_sha256"]
            == (ModelValidationCampaign.definition(case)["fixture_sha256"])
        )
    for record in records:
        json.dumps(record, allow_nan=False)
    for run in runs:
        assert run["status"] in ("feasible", "no_feasible_found")
        assert run["n_evaluations"] > 0
        assert run["source_unchanged"]
        assert run["independent_search_incumbent"]["feasible"] == run["search_incumbent_feasible"]
        if run["status"] == "feasible":
            assert run["independent_feasible_witness"]["feasible"]
            assert run["independent_feasible_witness"]["objective"] == pytest.approx(
                run["best_feasible_objective"]
            )
            assert 0 < run["time_to_first_feasible_seconds"] <= run["solve_seconds"]
        else:
            assert run["best_feasible_objective"] is None
            assert run["time_to_first_feasible_seconds"] is None
    summaries = ModelValidationCampaign.summary(path)
    assert len(summaries) == 48
    assert all(group["runs"] == group["planned_runs"] == 10 for group in summaries)
    assert all(group["missing_runs"] == group["errors"] == 0 for group in summaries)
