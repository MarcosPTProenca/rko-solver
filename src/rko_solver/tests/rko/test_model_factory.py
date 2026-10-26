"""Public model factory, portable results and model-aware presentation contracts.

Objective:
- Integrate native model conversion with existing algorithms without altering callback results.

Technical approach:
- Run tiny native cases across backends and a scripted score/feasibility disagreement.
- Check direct constant-model accounting, explicit assignments and assignment-free reports.

Flow (pseudocode):
1. Convert an existing supported model through RKO.from_model.
2. Solve with normal execution or one direct zero-key evaluation.
3. Verify independent model witnesses, unchanged root score identity and explicit export actions.

Important criteria:
- Deterministic witness tests do not depend on timed searches finding an optimum.
- No result claims optimality, mathematical infeasibility or a solver certificate.

Inputs/Outputs:
- Input: tiny models, existing execution settings and controlled completed candidates.
- Output: assertions for public semantics, compatibility, counting and identity.
"""

import json
import numpy as np
import pickle
import pytest
from itertools import product
from pathlib import Path
from rko_solver import RKO
from rko_solver import ModelAssignment
from rko_solver import ModelResult
from rko_solver import ModelSolution
from rko_solver import RKOResult
from rko_solver.assessment import assess_result
from rko_solver.examples.model_conversion import LinearModelExample
from rko_solver.modeling.linear import CompiledLinearModel
from rko_solver.modeling.linear import ModelEvaluation
from rko_solver.modeling.linear import ModelEvidence
from rko_solver.orchestration.search import SearchSession
from rko_solver.orchestration.types import RunConfig
from rko_solver.orchestration.types import SearchMetadata
from rko_solver.reporting import OptimizationReport
from rko_solver.tests.rko.test_model_adapters import FrontendCases


@pytest.mark.parametrize("engine", ["rust", "cpp", "auto"])
def test_generated_factory_rejects_compiled_dispatch_before_extraction(engine: str) -> None:
    """Reject unsupported engines before touching a frontend or starting auto probes.

    Args:
        engine (str): Engine without generated-model witness collection.
    """
    with pytest.raises(ValueError, match="require engine='python'"):
        RKO.from_model(object(), engine=engine)


@pytest.mark.parametrize("engine", ["rust", "cpp", "auto"])
@pytest.mark.parametrize("fixed", [False, True])
def test_generated_solver_rejects_engine_changes_before_search(engine: str, fixed: bool) -> None:
    """Prevent post-conversion engine mutation from bypassing model evidence collection.

    Args:
        engine (str): Engine unsupported for generated model callbacks.
        fixed (bool): Whether the model would otherwise take the zero-key direct path.
    """
    native = FrontendCases.build("pyomo")
    if fixed:
        pyomo = pytest.importorskip("pyomo.environ")
        for variable in native.component_data_objects(pyomo.Var):
            variable.fix(0)
    solver = RKO.from_model(native, execution="sequential")
    solver.engine = engine
    with pytest.raises(ValueError, match="require engine='python'"):
        solver.solve(0.01)
    assert solver.result_ is None


@pytest.mark.parametrize("frontend", ["pyomo", "pulp", "mathopt"])
@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
def test_public_factory_all_frontends_and_backends(frontend: str, execution: str) -> None:
    """Solve owned native models through existing execution with portable matching evidence.

    Args:
        frontend (str): Qualified optional modeling frontend.
        execution (str): Existing execution execution.
    """
    native = FrontendCases.build(frontend)
    solver = RKO.from_model(
        native,
        algorithms=["MultiStart"],
        execution=execution,
        adaptation="fixed",
        n_jobs=1 if execution == "process" else None,
        random_state=42,
        pool_size=2,
        prefill_pool=True,
    )
    result = solver.solve(time_limit=0.02)
    assert solver.result_ is result
    assert result.model is not None
    assert result.engine == "python" and result.execution == execution and result.auto is None
    assert result.model.source.frontend == frontend
    assert result.model.search_incumbent.keys == result.keys
    assert result.model.search_incumbent.search_score == result.cost
    assert result.n_evaluations > 0
    assert result.initialization is not None
    assert result.model.compilation_seconds >= 0
    if result.model.best_feasible is not None:
        assert 0 < result.model.time_to_first_feasible_seconds <= result.elapsed_seconds
    else:
        assert result.model.time_to_first_feasible_seconds is None
    restored = pickle.loads(pickle.dumps(result))
    assert restored.model == result.model
    assert restored.keys == result.keys
    json.dumps(result.model.to_dict(), allow_nan=False)
    assert all(not isinstance(value, type(native)) for value in vars(solver).values())


@pytest.mark.parametrize(
    "algorithm",
    [
        "MultiStart",
        "SA",
        "ILS",
        "VNS",
        "LNS",
        "BRKGA",
        "BRKGA_CS",
        "PSO",
        "GA",
        "GRASP",
        "IPR",
    ],
)
def test_all_algorithms_through_public_factory(algorithm: str) -> None:
    """Keep every existing algorithm available from the public model factory.

    Args:
        algorithm (str): Existing algorithm name.
    """
    native = FrontendCases.build("pyomo")
    result = RKO.from_model(
        native, algorithms=[algorithm], execution="sequential", adaptation="fixed", random_state=42
    ).solve(time_limit=0.005)
    assert result.n_evaluations > 0
    assert result.model.search_incumbent.keys == result.keys


def test_penalty_model_solution_is_not_root_incumbent_and_no_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Retain feasible assignments independently and label reports/assessments honestly.

    Args:
        monkeypatch (pytest.MonkeyPatch): Install a deterministic completed evaluation sequence.
        tmp_path (Path): Temporary report export directory.
    """
    p = pytest.importorskip("pyomo.environ")
    native = p.ConcreteModel()
    native.x = p.Var(domain=p.Binary)
    native.obj = p.Objective(expr=10 * native.x)
    native.row = p.Constraint(expr=native.x >= 1)
    calls = []
    original = CompiledLinearModel.decode

    def counted(snapshot: CompiledLinearModel, keys: np.ndarray) -> ModelEvaluation:
        """Count actual numerical model evaluations without changing invocation payloads.

        Args:
            snapshot (CompiledLinearModel): Owned numerical model.
            keys (np.ndarray): Invocation keys.

        Returns:
            ModelEvaluation: Exact completed generated evaluation payload.
        """
        calls.append(tuple(keys))
        return original(snapshot, keys)

    monkeypatch.setattr(CompiledLinearModel, "decode", counted)
    solver = RKO.from_model(
        native,
        scoring="weighted_penalty",
        penalty_weight=0.1,
        algorithms=["MultiStart"],
        adaptation="fixed",
        execution="sequential",
    )

    def scripted(
        config: RunConfig,
        duration: float,
        *,
        model_evidence: list[ModelEvidence],
        search_metadata: list[SearchMetadata],
        **settings: object,
    ) -> tuple[list[float], float]:
        """Complete the two candidates normally but return the score-selected infeasible one.

        Args:
            config (RunConfig): Existing resolved portfolio settings.
            duration (float): Existing solve allowance.
            model_evidence (list[ModelEvidence]): Actual private completed-witness collector.
            search_metadata (list[SearchMetadata]): Completed work report collector.
            settings (object): Other existing dispatch settings, unused in this fixture.

        Returns:
            tuple[list[float], float]: Actual search-score incumbent, not the feasible witness.
        """
        session = SearchSession(solver.decoder, solver.cost)
        for key in (0, 1):
            session.evaluate(session.decode(np.array([key])))
        model_evidence.append(session.model_evidence)
        search_metadata.append(
            SearchMetadata(algorithm="MultiStart", n_evaluations=2, stop_reason="completed")
        )
        return session.best[0].tolist(), session.best[1]

    monkeypatch.setattr(solver, "_execute", scripted)
    result = solver.solve(1)
    assert calls == [(0,), (1,)]
    assert 0 < result.model.time_to_first_feasible_seconds <= result.elapsed_seconds
    assert result.keys == (0,)
    assert result.cost == 0.1
    assert result.n_evaluations == 2
    assert result.model.status == "feasible"
    assert result.model.search_incumbent.feasible is False
    witness = result.model.best_feasible
    assert witness.keys == (1,)
    assert witness.objective == 10
    assert witness.search_score == 10
    assert witness.assignments[0].identifier == "pyomo:x"
    assert witness.assignments[0].value == 1
    report = OptimizationReport(result)
    data = report.to_dict()
    assert data["summary"]["cost_kind"] == "search_score"
    assert data["model"]["best_feasible"]["objective"] == 10
    assert "assignments" not in data["model"]["best_feasible"]
    assert "keys" not in data["model"]["best_feasible"]
    report.write_json(tmp_path / "model.json")
    assert calls == [(0,), (1,)]
    assessment = assess_result(result, decoder=solver.decoder)
    assert assessment.to_dict()["scope"] == "search_incumbent_not_model_best_feasible"
    assert calls == [(0,), (1,), (0,)]
    result.model.write_back(native)
    assert native.x.value == 1
    assert result.keys == (0,)
    assert result.cost == 0.1


@pytest.mark.parametrize("inconsistent", [False, True])
def test_fixed_models_have_one_direct_evaluation_and_no_fake_search(inconsistent: bool) -> None:
    """Report direct feasible/no-feasible evidence with empty search metadata and real counting.

    Args:
        inconsistent (bool): Add an impossible constant row when true.
    """
    p = pytest.importorskip("pyomo.environ")
    native = p.ConcreteModel()
    native.x = p.Var(bounds=(0, 3))
    native.x.fix(2)
    native.obj = p.Objective(expr=3 * native.x + 7, sense=p.maximize)
    if inconsistent:
        native.bad = p.Constraint(expr=p.Constraint.Infeasible)
    solver = RKO.from_model(native, prefill_pool=True, algorithms=["SA", "PSO"])
    result = solver.solve(0.1)
    assert result.engine == "python" and result.execution is None
    assert result.keys == ()
    assert result.algorithms == ()
    assert result.searches == ()
    assert result.initialization is None
    assert result.n_workers == 0
    assert result.control is None and result.pool_size is None
    assert result.n_evaluations == 1
    assert result.stop_reason == "completed"
    assert result.model.no_search
    assert result.model.direct_evaluations == 1
    assert result.model.search_incumbent.objective == 13
    if inconsistent:
        assert result.model.status == "no_feasible_found"
        assert result.model.best_feasible is None
        assert result.model.time_to_first_feasible_seconds is None
        assert result.model.to_dict()["best_feasible"] is None
        with pytest.raises(ValueError, match="No feasible"):
            result.model.write_back(native)
    else:
        assert result.model.status == "feasible"
        assert result.model.best_feasible.objective == 13
        assert 0 < result.model.time_to_first_feasible_seconds <= result.elapsed_seconds
    assert solver.solve(0.1).n_evaluations == 1


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
def test_model_factory_preserves_current_evaluation_budget_and_engine_api(execution: str) -> None:
    """Stop every generated search at the native API's completed-evaluation boundary.

    Args:
        execution (str): Python scheduling mode, including real process execution.
    """
    model = FrontendCases.build("pyomo")
    solver = RKO.from_model(
        model,
        engine="python",
        execution=execution,
        algorithms=["MultiStart"],
        adaptation="fixed",
        random_state=42,
        prefill_pool=False,
        max_evaluations=3,
        n_jobs=1 if execution == "process" else None,
    )
    result = solver.solve(2.0)
    assert result.engine == "python" and result.execution == execution and result.auto is None
    assert result.max_evaluations == 3 and result.n_evaluations == 3
    assert result.stop_reason == "max_evaluations"
    assert result.model.search_incumbent.keys == result.keys
    assert result.model.search_incumbent.search_score == result.cost


def test_callback_results_and_empty_manual_dimensions_stay_compatible() -> None:
    """Preserve optional result construction/exports and ordinary positive-dimension validation."""
    legacy = RKOResult((0.5,), 1, 0, "sequential", ("SA",), None)
    assert legacy.model is None
    assert legacy.n_evaluations is None
    assert "model" not in OptimizationReport(legacy).to_dict()
    assert ModelAssignment.__name__ == "ModelAssignment"
    assert ModelSolution.__name__ == "ModelSolution"
    assert ModelResult.__name__ == "ModelResult"
    solver = RKO(lambda keys: keys, lambda values: float(sum(values)), 0, execution="sequential")
    with pytest.raises(ValueError, match="positive integer"):
        solver.solve(0.1)


def test_conversion_snapshot_is_independent_and_changed_callbacks_reject() -> None:
    """Keep source edits outside owned search and reject manual generated-callback replacement."""
    native = FrontendCases.build("pyomo")
    solver = RKO.from_model(
        native, execution="sequential", algorithms=["MultiStart"], adaptation="fixed"
    )
    fingerprint = solver._model_source.fingerprint
    native.obj.expr += 1000
    result = solver.solve(0.01)
    assert result.model.source.fingerprint == fingerprint
    with pytest.raises(ValueError, match="changed or is incompatible"):
        result.model.native_mapping(native)
    solver.cost = lambda _: 0.0
    with pytest.raises(ValueError, match="callbacks/dimension changed"):
        solver.solve(0.01)


def test_model_html_uses_distinct_original_objective_and_score_labels(tmp_path: Path) -> None:
    """Label model results separately without attributing search-score curves to raw objectives.

    Args:
        tmp_path (Path): Temporary offline HTML destination.
    """
    pytest.importorskip("plotly")
    native = FrontendCases.build("pyomo")
    result = RKO.from_model(
        native,
        execution="sequential",
        algorithms=["MultiStart"],
        adaptation="fixed",
        history_limit=2,
    ).solve(0.01)
    output = (
        OptimizationReport(result).write_html(tmp_path / "model.html").read_text(encoding="utf-8")
    )
    assert "minimized search score" in output
    assert "Model solution — separate from search score" in output
    assert "Best feasible original objective" in output
    assert "not optimal or certified" in output
    assert "Best observed search score" in output


def test_inconsistent_coupled_rows_return_no_feasible_found_not_infeasible() -> None:
    """Keep an infeasible search incumbent diagnostic without fabricating a model solution."""
    p = pytest.importorskip("pyomo.environ")
    model = p.ConcreteModel()
    model.x = p.Var(domain=p.Binary)
    model.obj = p.Objective(expr=model.x + 7)
    model.lo = p.Constraint(expr=model.x >= 1)
    model.hi = p.Constraint(expr=model.x <= 0)
    result = RKO.from_model(
        model,
        algorithms=["MultiStart"],
        adaptation="fixed",
        execution="sequential",
        random_state=42,
    ).solve(0.01)
    assert result.model.status == "no_feasible_found"
    assert result.model.best_feasible is None
    assert not result.model.search_incumbent.feasible
    assert result.model.to_dict()["certificate"] is None
    assert result.cost >= 1
    assert result.n_evaluations > 0


@pytest.mark.parametrize("frontend", ["pyomo", "pulp", "mathopt"])
def test_maintained_example_has_exhaustively_checkable_maximum(frontend: str) -> None:
    """Verify the original tiny public example oracle without a mathematical optimizer.

    Args:
        frontend (str): Qualified optional native frontend.
    """
    pytest.importorskip(
        {"pyomo": "pyomo.environ", "pulp": "pulp", "mathopt": "ortools.math_opt.python.mathopt"}[
            frontend
        ]
    )
    solver = RKO.from_model(LinearModelExample.build(frontend))
    source = solver._model_source
    feasible = []
    for x, y, c in product((0, 1), (0, 1, 2), (0, 1)):
        keys = {"x": x, "y": y / 2, "c": c}
        evidence = solver.decoder(np.array([keys[v.label] for v in source.variables]))
        if evidence.feasible:
            feasible.append(evidence.objective)
    assert max(feasible) == 3.5


@pytest.mark.parametrize("execution", ["sequential", "thread"])
def test_public_model_failure_does_not_return_partial_witness(
    execution: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate search failure even after one finite completed model assignment exists.

    Args:
        execution (str): Existing in-process execution execution.
        monkeypatch (pytest.MonkeyPatch): Inject a decoder failure after its first completion.
    """
    model = FrontendCases.build("pyomo")
    original = CompiledLinearModel.decode
    calls = []

    def fail(
        snapshot: CompiledLinearModel,
        keys: np.ndarray,
        **options: object,
    ) -> ModelEvaluation:
        """Complete the first invocation and fail the next rather than returning partial success.

        Args:
            snapshot (CompiledLinearModel): Owned model being evaluated.
            keys (np.ndarray): Candidate keys.
            options (object): Private generated-session cooperative budget hook.

        Returns:
            ModelEvaluation: Only the first completed assignment.

        Raises:
            ValueError: On the second invocation.
        """
        if calls:
            raise ValueError("forced model failure")
        calls.append(1)
        return original(snapshot, keys, **options)

    monkeypatch.setattr(CompiledLinearModel, "decode", fail)
    solver = RKO.from_model(
        model, algorithms=["MultiStart"], adaptation="fixed", execution=execution
    )
    with pytest.raises((ValueError, RuntimeError)):
        solver.solve(0.1)
    assert calls == [1]
    assert solver.result_ is None
