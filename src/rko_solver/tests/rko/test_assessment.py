"""Post-solve domain-checking contracts independent of numerical optimization.

Objective:
- Verify truthful optional validation, callback isolation and safe diagnostics.

Technical approach:
- Assess immutable numerical results with controlled decoder/checker spies.
- Exercise real standalone/portfolio results and declared scheduling constraints.

Flow (pseudocode):
1. Decode final keys once and invoke an optional checker once.
2. Aggregate typed passed/failed/not-checked diagnostics without rerunning objectives.
3. Preserve successful solve state across failed checks and checker exceptions.

Important criteria:
- Domain feasibility is an explicit checker declaration, not inferred from finite cost.
- Assessment records post-run reconstruction, not the original execution witness.

Inputs/Outputs:
- Input: completed results, finite diagnostic records and trusted application callbacks.
- Output: assertions covering assessment, status, bounded metadata and engine integration.
"""

import numpy as np
import pytest
import random
import subprocess
import sys
from dataclasses import FrozenInstanceError
from dataclasses import replace
from functools import partial
from pathlib import Path
from rko_solver import RKO
from rko_solver import SolutionAssessment
from rko_solver import ValidationCheck
from rko_solver import ValidationReport
from rko_solver import assess_result
from rko_solver.examples.instance_pipeline import SchedulingInstance
from rko_solver.examples.instance_pipeline import SchedulingProblem
from rko_solver.metaheuristics import SA
from rko_solver.reporting import OptimizationReport
from rko_solver.solver import RKOResult
from unittest.mock import Mock


@pytest.fixture
def result() -> RKOResult:
    """Return a numerical incumbent with no domain-validity claim.

    Returns:
        RKOResult: Small successful legacy-compatible result.
    """
    return RKOResult((0.2, 0.7), 10.0, 1.5, "python", ("SA",), 42, execution="sequential")


def test_assessment_decodes_and_checks_once(result: RKOResult) -> None:
    """Keep final decoding/checking separate from costs and search timing.

    Args:
        result (RKOResult): Successful numerical result.
    """
    payload = object()
    decoder = Mock(return_value=payload)
    validation = ValidationReport(
        checks=(ValidationCheck("capacity", "passed", value=4, limit=5, unit="kg"),),
        feasible=True,
        checker="capacity_checker",
        checker_version="1",
    )
    validator = Mock(return_value=validation)
    assessment = assess_result(result, decoder=decoder, validator=validator)
    assert isinstance(assessment, SolutionAssessment)
    assert assessment.result is result
    assert assessment.solution is payload
    assert assessment.validation is validation
    assert assessment.origin == "post_run_decode"
    assert assessment.elapsed_seconds >= 0
    assert result.elapsed_seconds == 1.5
    decoder.assert_called_once()
    assert isinstance(decoder.call_args.args[0], np.ndarray)
    assert tuple(decoder.call_args.args[0]) == result.keys
    validator.assert_called_once_with(payload)
    assert validation.status == "passed"
    with pytest.raises(FrozenInstanceError):
        assessment.elapsed_seconds = 999


@pytest.mark.parametrize(
    "checks,feasible,status",
    [
        ((), None, "not_checked"),
        ((ValidationCheck("capacity", "not_checked"),), None, "not_checked"),
        ((ValidationCheck("capacity", "passed"),), None, "passed"),
        (
            (ValidationCheck("a", "passed"), ValidationCheck("b", "not_checked")),
            None,
            "not_checked",
        ),
        ((ValidationCheck("a", "failed"), ValidationCheck("b", "not_checked")), None, "failed"),
        ((ValidationCheck("capacity", "failed"),), False, "failed"),
        ((), False, "failed"),
    ],
)
def test_status_never_infers_domain_feasibility(
    checks: tuple[ValidationCheck, ...],
    feasible: bool | None,
    status: str,
) -> None:
    """Separate aggregate diagnostic status from the checker's feasibility declaration.

    Args:
        checks (tuple[ValidationCheck, ...]): Implemented or omitted checks.
        feasible (bool | None): Explicit domain outcome.
        status (str): Expected aggregate status.
    """
    report = ValidationReport(checks=checks, feasible=feasible)
    assert report.status == status
    assert report.feasible is feasible


def test_absent_checker_is_not_success(result: RKOResult) -> None:
    """Keep unassessed domain constraints visible as not checked.

    Args:
        result (RKOResult): Finite successful incumbent.
    """
    decoder = Mock(return_value="solution")
    assessment = assess_result(result, decoder=decoder)
    decoder.assert_called_once()
    assert assessment.solution == "solution"
    assert assessment.validation.status == "not_checked"
    assert assessment.validation.feasible is None


def test_false_check_does_not_replace_incumbent(result: RKOResult) -> None:
    """Return an infeasible assessment without repairing or reranking the search result.

    Args:
        result (RKOResult): Numerical incumbent.
    """
    validation = ValidationReport(
        checks=(ValidationCheck("capacity", "failed", value=8, limit=5),),
        feasible=False,
    )
    assessment = assess_result(
        result, decoder=Mock(return_value="overweight"), validator=Mock(return_value=validation)
    )
    assert assessment.validation.status == "failed"
    assert assessment.result is result
    assert result.keys == (0.2, 0.7)
    assert result.cost == 10.0


@pytest.mark.parametrize("stage", ["decoder", "validator"])
def test_application_exceptions_propagate(result: RKOResult, stage: str) -> None:
    """Preserve callback exceptions rather than relabelling them as failed constraints.

    Args:
        result (RKOResult): Numerical incumbent to preserve.
        stage (str): Failing postprocessing stage.
    """
    error = RuntimeError("application failure")
    decoder = Mock(return_value="solution", side_effect=error if stage == "decoder" else None)
    validator = Mock(side_effect=error if stage == "validator" else None)
    with pytest.raises(RuntimeError) as caught:
        assess_result(result, decoder=decoder, validator=validator)
    assert caught.value is error
    assert result.cost == 10.0
    if stage == "decoder":
        validator.assert_not_called()


@pytest.mark.parametrize("outcome", [True, False, {}, None])
def test_checker_must_return_structured_report(result: RKOResult, outcome: object) -> None:
    """Reject ambiguous callback booleans/dictionaries instead of claiming success.

    Args:
        result (RKOResult): Numerical incumbent.
        outcome (object): Unsupported checker return value.
    """
    with pytest.raises(TypeError, match="ValidationReport"):
        assess_result(
            result, decoder=Mock(return_value="solution"), validator=Mock(return_value=outcome)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "status": "passed"},
        {"name": "constraint", "status": "ok"},
        {"name": "constraint", "status": "passed", "value": float("nan")},
        {"name": "constraint", "status": "passed", "limit": float("inf")},
        {"name": "constraint", "status": "passed", "value": []},
        {"name": "x" * 129, "status": "passed"},
        {"name": "constraint", "status": "passed", "message": "x" * 2049},
    ],
)
def test_diagnostics_are_finite_and_bounded(kwargs: dict[str, object]) -> None:
    """Reject invalid/oversized check metadata before report export.

    Args:
        kwargs (dict[str, object]): Invalid check fields.
    """
    with pytest.raises(ValueError):
        ValidationCheck(**kwargs)


def test_reports_reject_inconsistent_or_duplicate_checks() -> None:
    """Avoid contradictory feasibility declarations and ambiguous check identities."""
    with pytest.raises(ValueError):
        ValidationReport(checks=(ValidationCheck("capacity", "failed"),), feasible=True)
    with pytest.raises(ValueError):
        ValidationReport(checks=(ValidationCheck("capacity", "not_checked"),), feasible=True)
    with pytest.raises(ValueError):
        ValidationReport(feasible=True)
    with pytest.raises(ValueError):
        ValidationReport(checks=(ValidationCheck("a", "passed"), ValidationCheck("a", "failed")))
    with pytest.raises(ValueError):
        ValidationReport(checks=("not a check",))
    with pytest.raises(ValueError):
        ValidationReport(checks=tuple(ValidationCheck(str(i), "passed") for i in range(1001)))


def test_export_excludes_payload_keys_and_source_result(result: RKOResult) -> None:
    """Serialize only explicit diagnostic data, not arbitrary decoded application objects.

    Args:
        result (RKOResult): Source incumbent.
    """
    assessment = assess_result(
        result,
        decoder=Mock(return_value=object()),
        validator=Mock(
            return_value=ValidationReport(
                checks=(ValidationCheck("capacity", "passed"),),
                feasible=True,
            )
        ),
    )
    data = assessment.to_dict()
    assert set(data) == {"origin", "elapsed_seconds", "validation"}
    assert data["validation"]["status"] == "passed"
    assert "solution" not in data and "result" not in data and "keys" not in data
    report = OptimizationReport(result, assessment=assessment).to_dict()
    assert report["assessment"] == data
    assert (
        OptimizationReport(result).to_dict()["assessment"]["validation"]["status"] == "not_checked"
    )
    with pytest.raises(ValueError, match="different result"):
        OptimizationReport(replace(result, cost=99.0), assessment=assessment)


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
def test_assessment_after_real_portfolio_preserves_search_result(execution: str) -> None:
    """Check a reconstructed domain solution after every managed execution mode completes.

    Args:
        execution (str): Portfolio execution strategy.
    """
    problem = SchedulingProblem(SchedulingInstance("fixture", (3, 2, 5, 1), (4, 6, 7, 3)))
    solver = RKO(
        problem.decode,
        problem.cost,
        problem.n_variables,
        algorithms=["SA", "PSO"],
        adaptation="fixed",
        execution=execution,
        random_state=42,
    )
    outcome = solver.solve(0.2)
    count, elapsed = outcome.n_evaluations, outcome.elapsed_seconds
    assessment = assess_result(outcome, decoder=problem.decode, validator=problem.validate)
    assert assessment.validation.status == "passed"
    assert assessment.validation.feasible is True
    assert solver.result_ is outcome
    assert outcome.n_evaluations == count
    assert outcome.elapsed_seconds == elapsed
    error = RuntimeError("checker bug")
    with pytest.raises(RuntimeError) as caught:
        assess_result(outcome, decoder=problem.decode, validator=Mock(side_effect=error))
    assert caught.value is error
    assert solver.result_ is outcome
    assert outcome.n_evaluations == count


def test_scheduling_checks_are_independent_of_cost_and_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check identities/completion constraints without invoking construction or scoring.

    Args:
        monkeypatch (pytest.MonkeyPatch): Makes forbidden decoder/objective calls fail.
    """
    problem = SchedulingProblem(SchedulingInstance("fixture", (3, 2), (0, 0)))
    with monkeypatch.context() as patch:
        patch.setattr(SchedulingProblem, "decode", Mock(side_effect=AssertionError("no decoding")))
        patch.setattr(SchedulingProblem, "cost", Mock(side_effect=AssertionError("no scoring")))
        assert problem.validate(((0, 3), (1, 5))).feasible is True
    assert problem.validate(((0, 3), (0, 6))).feasible is False
    assert problem.validate(((0, 99), (1, 100))).feasible is False
    assert problem.validate(((9, 3), (1, 5))).feasible is False
    assert problem.validate(((0, 3),)).feasible is False
    assert problem.validate(((True, 3), (1, 5))).feasible is False
    solver = SA(problem.decode, problem.cost, 2, adaptation="fixed", execution="sequential")
    outcome = solver.solve(time_limit=0.02)
    assessment = assess_result(outcome, decoder=problem.decode, validator=problem.validate)
    assert outcome.cost > 0
    assert assessment.validation.feasible is True


def test_assessment_does_not_repeat_cost_or_draw_library_randomness() -> None:
    """Keep successful search counts, objective calls and deterministic callback RNGs unchanged."""
    problem = SchedulingProblem(SchedulingInstance("fixture", (3, 2), (0, 0)))
    decoder, cost = Mock(wraps=problem.decode), Mock(wraps=problem.cost)
    outcome = RKO(decoder, cost, 2, algorithms=["MultiStart"], execution="sequential").solve(0.02)
    decoder_calls, cost_calls = decoder.call_count, cost.call_count
    python_state, numpy_state = random.getstate(), np.random.get_state()
    checker = Mock(wraps=problem.validate)
    assessment = assess_result(outcome, decoder=decoder, validator=checker)
    assert decoder.call_count == decoder_calls + 1
    assert cost.call_count == cost_calls
    assert outcome.n_evaluations == cost_calls
    checker.assert_called_once_with(assessment.solution)
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0]
    assert np.array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]


def test_check_lists_are_snapshotted_and_exports_detached() -> None:
    """Prevent mutable input lists or returned dictionaries from changing recorded checks."""
    checks = [ValidationCheck("capacity", "passed")]
    report = ValidationReport(checks=checks, feasible=True)
    checks.append(ValidationCheck("new constraint", "failed"))
    assert len(report.checks) == 1
    exported = report.to_dict()
    exported["checks"].clear()
    assert len(report.to_dict()["checks"]) == 1


@pytest.mark.parametrize("duration", [-1.0, float("nan"), float("inf"), True, 10**400])
def test_invalid_assessment_durations_fail(result: RKOResult, duration: object) -> None:
    """Reject invalid or unrepresentable postprocessing durations.

    Args:
        result (RKOResult): Numerical source result.
        duration (object): Invalid assessment time.
    """
    with pytest.raises(ValueError):
        SolutionAssessment(result, "solution", ValidationReport(), duration)


def test_documented_assessment_script_runs(tmp_path: Path) -> None:
    """Run the exact introductory assessment script outside the source directory.

    Args:
        tmp_path (Path): Isolated application directory.
    """
    page = Path(__file__).resolve().parents[4] / "docs" / "solution-validation.md"
    text = page.read_text(encoding="utf-8")
    code = text.split("```python\n", 1)[1].split("```", 1)[0]
    script = tmp_path / "assess_schedule.py"
    script.write_text(code, encoding="utf-8")
    process = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "Check status: passed" in process.stdout
    assert "Declared feasibility: True" in process.stdout


def test_readme_first_solve_includes_all_user_function_templates(tmp_path: Path) -> None:
    """Run the README's file-backed Python API example, including optional validation.

    Args:
        tmp_path (Path): Isolated application directory for the copied tutorial.
    """
    page = Path(__file__).resolve().parents[4] / "README.md"
    text = page.read_text(encoding="utf-8")
    data = text.split("```json\n", 1)[1].split("```", 1)[0]
    code = text.split("```python\n", 1)[1].split("```", 1)[0]
    for name in ("read_data", "decoder", "objective", "validate"):
        assert f"def {name}(" in code
    assert "solver = RKO(" in code
    assert "result = solver.solve(" in code
    (tmp_path / "instance.json").write_text(data, encoding="utf-8")
    script = tmp_path / "quickstart.py"
    script.write_text(code, encoding="utf-8")
    process = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "Solution:" in process.stdout
    assert "Objective:" in process.stdout
    assert "Validation: passed" in process.stdout
    assert "Evaluations:" in process.stdout


def test_readme_individual_algorithm_examples_run(tmp_path: Path) -> None:
    """Execute both secondary single-algorithm examples with the same loaded callbacks.

    Args:
        tmp_path (Path): Isolated directory for the README input file.
    """
    page = Path(__file__).resolve().parents[4] / "README.md"
    text = page.read_text(encoding="utf-8")
    code = text.split("```python\n", 1)[1].split("```", 1)[0]
    data_text = text.split("```json\n", 1)[1].split("```", 1)[0]
    path = tmp_path / "instance.json"
    path.write_text(data_text, encoding="utf-8")
    namespace = {"__name__": "readme_example"}
    exec(compile(code, "README.md", "exec"), namespace)
    data = namespace["read_data"](path)
    namespace.update(
        {
            "data": data,
            "decode": partial(namespace["decoder"], data=data),
            "cost": partial(namespace["objective"], data=data),
            "SA": SA,
        }
    )
    section = text.split("## Run one metaheuristic instead", 1)[1]
    section = section.split("## Use your own instance data", 1)[0]
    snippets = section.split("```python\n")[1:]
    assert len(snippets) == 2
    for snippet in snippets:
        exec(compile(snippet.split("```", 1)[0], "README.md", "exec"), namespace)
        result = namespace["result"]
        assert result.algorithms == ("SA",)
        assert result.execution == "sequential"
        assert dict(result.searches[0].parameters)["sa_max"] == 100
        assert dict(result.searches[0].parameters)["t0"] == 1000.0
        assert result.cost == pytest.approx(
            namespace["cost"](namespace["decode"](np.asarray(result.keys)))
        )
