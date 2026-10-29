"""Independent feasibility and lifecycle contracts for bounded continuous restoration.

Objective:
- Exercise restoration through owned model callbacks and the public factory.

Technical approach:
- Check original row/domain arithmetic independently of the restoration algorithm.
- Preserve deterministic key identity, strict tolerances and completed-evaluation evidence.

Flow (pseudocode):
1. Build small bounded models with analytic feasible sets.
2. Decode fixed candidates and verify all original constraints and objective values.
3. Exercise work limits, discrete obstructions and portable worker results.

Important criteria:
- No reference solution is injected into search and no external optimizer repairs candidates.
- An exhausted repair is not an infeasibility certificate.

Inputs/Outputs:
- Input: fixed keys, explicit original linear models and common restoration settings.
- Output: assertions on final assignments, diagnostics and execution boundaries.
"""

import numpy as np
import pickle
import pytest
import threading
from benchmarks.modeling.cases import PracticalModels
from rko_solver import RKO
from rko_solver.modeling.adapters import ModelAdapter
from rko_solver.modeling.linear import CompiledLinearModel
from rko_solver.modeling.linear import LinearRow
from rko_solver.modeling.linear import LinearVariable
from rko_solver.orchestration.search import SearchSession
from rko_solver.orchestration.search import _BudgetExpired


def test_continuous_repair_constructs_minimal_equality_for_fixed_key_stream() -> None:
    """Restore every fixed-stream point onto the original unit-square balance."""
    model = CompiledLinearModel(
        (LinearVariable("x", "x", 0, 1), LinearVariable("y", "y", 0, 1)),
        (1, 2),
        (LinearRow("balance", ((0, 1), (1, 1)), 1, 1),),
        repair="continuous",
    )
    keys = np.random.default_rng(2026).random((2048, 2))
    before = keys.copy()
    for chromosome in keys:
        result = model.decode(chromosome)
        x, y = result.values
        assert result.feasible and 0 <= x <= 1 and 0 <= y <= 1
        assert abs(x + y - 1) <= 1e-6
        assert result.objective == x + 2 * y
        assert result.keys == tuple(chromosome)
        assert result.repair.policy == "continuous"
        assert result.repair.outcome == "repaired"
        assert 1 <= result.repair.sweeps <= 1024
    assert np.array_equal(keys, before)


@pytest.mark.parametrize("frontend", ["pyomo", "pulp", "mathopt"])
def test_sparse_transport_restoration_checks_every_original_constraint(frontend: str) -> None:
    """Restore the fixed transport stream without relaxing any original row or domain.

    Args:
        frontend (str): Qualified modeling runtime used by the actual extraction boundary.
    """
    pytest.importorskip(
        {"pyomo": "pyomo.environ", "pulp": "pulp", "mathopt": "ortools.math_opt.python.mathopt"}[
            frontend
        ]
    )
    case = next(case for case in PracticalModels.cases() if case.name == "transport_equalities")
    source = ModelAdapter.extract(PracticalModels.native(case, frontend))
    model = source.compile(repair="continuous")
    for keys in np.random.default_rng(2026).random((256, model.n_keys)):
        result = model.decode(keys)
        checked = PracticalModels.check(
            case, {v.label: value for v, value in zip(source.variables, result.values, strict=True)}
        )
        assert checked["feasible"] and result.feasible
        assert result.objective == pytest.approx(checked["objective"])
        assert result.repair.sweeps <= 1024


def test_factory_defaults_to_restoration_with_explicit_box_compatibility() -> None:
    """Keep an explicit unrepaired mapping while the public factory restores real balances."""
    p = pytest.importorskip("pyomo.environ")
    native = p.ConcreteModel()
    native.x = p.Var(bounds=(0, 1))
    native.y = p.Var(bounds=(0, 1))
    native.balance = p.Constraint(expr=native.x + native.y == 1)
    native.obj = p.Objective(expr=native.x + 2 * native.y)
    restored = RKO.from_model(native, execution="sequential").decoder(np.array([0.9, 0.9]))
    boxed = RKO.from_model(native, execution="sequential", repair="none").decoder(
        np.array([0.9, 0.9])
    )
    assert restored.feasible and restored.values == pytest.approx((0.5, 0.5))
    assert restored.objective == pytest.approx(1.5)
    assert restored.repair.policy == "continuous"
    assert not boxed.feasible and boxed.values == (0.9, 0.9)
    assert boxed.repair.outcome == "disabled"


def test_session_cooperative_stop_during_repair_does_not_publish_partial_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Abort at an actual internal row boundary without counting or archiving partial work.

    Args:
        monkeypatch (pytest.MonkeyPatch): Temporary owner stop instrumentation.
    """
    model = CompiledLinearModel(
        (LinearVariable("x", "x", 0, 1), LinearVariable("y", "y", 0, 1)),
        (1, 2),
        (LinearRow("balance", ((0, 1), (1, 1)), 1, 1), LinearRow("limit", ((0, 1),), upper=0.25)),
        repair="continuous",
    )
    checks = []
    original = SearchSession._check

    def check(session: SearchSession) -> None:
        """Request cancellation at the third owner boundary.

        Args:
            session (SearchSession): Actual owning search, never the immutable snapshot.

        Raises:
            _BudgetExpired: When the deterministic cancellation boundary is reached.
        """
        checks.append(1)
        if len(checks) == 3:
            session.cancelled.set()
        original(session)

    monkeypatch.setattr(SearchSession, "_check", check)
    session = SearchSession(model.decode, model.cost)
    session.cancelled = threading.Event()
    with pytest.raises(_BudgetExpired):
        session.decode(np.array([1.0, 1.0]))
    assert len(checks) == 3
    assert session.n_evaluations == 0 and session.best is None and session.model_evidence is None


@pytest.mark.parametrize("kind", ["inventory", "resource"])
def test_unrelated_original_rows_and_ranges_are_restored(kind: str) -> None:
    """Check independent conservation and ranged resource arithmetic after restoration.

    Args:
        kind (str): Unrelated linear formulation with original-coordinate assertions.
    """
    if kind == "inventory":
        model = CompiledLinearModel(
            tuple(LinearVariable(str(i), str(i), 0, 10) for i in range(3)),
            (1, 2, 3),
            (
                LinearRow("flow0", ((0, 1), (1, -1)), 2, 2),
                LinearRow("flow1", ((1, 1), (2, -1)), 3, 3),
                LinearRow("capacity", ((0, 1), (1, 1), (2, 1)), upper=20),
                LinearRow("redundant", ((0, 2), (2, -2)), 10, 10),
            ),
            repair="continuous",
        )
    else:
        model = CompiledLinearModel(
            (LinearVariable("a", "a", 0, 2), LinearVariable("b", "b", 0, 4)),
            (2, 1),
            (
                LinearRow("resource", ((0, 1), (1, 1)), 3, 4),
                LinearRow("balance", ((0, 1), (1, -1)), -3, -1),
            ),
            repair="continuous",
        )
    for keys in np.random.default_rng(2026).random((64, model.n_keys)):
        result = model.decode(keys)
        assert result.feasible
        if kind == "inventory":
            a, b, c = result.values
            assert abs(a - b - 2) <= 1e-6 and abs(b - c - 3) <= 1e-6
            assert a + b + c <= 20 + 1e-6
            assert result.objective == pytest.approx(a + 2 * b + 3 * c, abs=1e-12)
        else:
            a, b = result.values
            assert 3 - 1e-6 <= a + b <= 4 + 1e-6
            assert -3 - 1e-6 <= a - b <= -1 + 1e-6
            assert result.objective == 2 * a + b


def test_discrete_decisions_remain_exact_and_obstructions_are_not_certificates() -> None:
    """Hold binary choices fixed while repairing their feasible continuous completion."""
    model = CompiledLinearModel(
        (LinearVariable("z", "z", 0, 1, "binary"), LinearVariable("x", "x", 0, 2)),
        (10, 1),
        (
            LinearRow("branch", ((0, -1), (1, 1)), 0.25, 0.25),
            LinearRow("activation", ((0, 1),), lower=1),
        ),
        repair="continuous",
    )
    obstruction = model.decode(np.array([0.0, 0.9]))
    feasible = model.decode(np.array([1.0, 0.9]))
    assert not obstruction.feasible and obstruction.values == (0.0, 1.8)
    assert obstruction.repair.outcome == "fixed_obstruction" and obstruction.repair.sweeps == 0
    assert feasible.feasible and feasible.values == (1.0, 1.25)
    assert feasible.objective == 11.25
    integer = CompiledLinearModel(
        (LinearVariable("z", "z", 0, 1, "binary"),),
        (1,),
        (LinearRow("required", ((0, 1),), 1, 1),),
        repair="continuous",
    )
    result = integer.decode(np.array([0.0]))
    assert result.values == (0.0,) and not result.feasible
    assert result.repair.outcome == "no_continuous_variables"


def test_work_limit_is_finite_and_completed_mapping_is_deterministic_and_portable() -> None:
    """Bound contradictory repairs and retain immutable repeatable invocation evidence."""
    model = CompiledLinearModel(
        (LinearVariable("x", "x", 0, 1),),
        (3,),
        (LinearRow("zero", ((0, 1),), 0, 0), LinearRow("one", ((0, 1),), 1, 1)),
        repair="continuous",
        repair_max_sweeps=3,
    )
    keys = np.array([0.25])
    result = model.decode(keys)
    assert not result.feasible and result.repair.outcome == "work_limit"
    assert result.repair.sweeps == 3 and result.values == (1.0,)
    assert model.decode(keys) == result
    restored = pickle.loads(pickle.dumps(model))
    assert restored.decode(keys) == result
    with pytest.raises(ValueError):
        np.frombuffer(restored._repair_coefficients).setflags(write=True)


@pytest.mark.parametrize(
    "repair, cap", [("invalid", 1), ("continuous", True), ("continuous", 0), ("continuous", 10001)]
)
def test_repair_settings_reject_invalid_policy_and_work_caps(repair: str, cap: int) -> None:
    """Validate restoration settings before any callback is exposed.

    Args:
        repair (str): Requested policy.
        cap (int): Requested deterministic full-sweep limit.
    """
    with pytest.raises(ValueError):
        CompiledLinearModel(
            (LinearVariable("x", "x", 0, 1),), (1,), repair=repair, repair_max_sweeps=cap
        )


def test_normalization_underflow_is_explicit_and_box_mode_remains_available() -> None:
    """Never silently discard a small nonzero projection coefficient."""
    variables = (LinearVariable("a", "a", 0, 1), LinearVariable("b", "b", 0, 1))
    rows = (LinearRow("scaled", ((0, 1), (1, 1e-300)), 0.5, 0.5),)
    with pytest.raises(ValueError, match="underflow"):
        CompiledLinearModel(variables, (1, 1), rows, repair="continuous")
    boxed = CompiledLinearModel(variables, (1, 1), rows, repair="none")
    assert boxed.decode(np.array([0.5, 0.5])).feasible


def test_sparse_repair_storage_is_linear_in_variables_and_nonzeros() -> None:
    """Protect the sparse footprint without allocating a dense constraint matrix."""
    dimension = 10000
    rows = tuple(LinearRow(str(i), ((i, 1), (i + 1, -1)), 0, 0) for i in range(20))
    model = CompiledLinearModel(
        tuple(LinearVariable(str(i), str(i), 0, 1) for i in range(dimension)),
        (1,) * dimension,
        rows,
        repair="continuous",
    )
    assert len(model._repair_coefficients) == 40 * 8
    assert len(model._repair_indptr) == 21 * np.dtype(np.intp).itemsize
    assert len(model._repair_free) == dimension * np.dtype(np.intp).itemsize
    result = model.decode(np.zeros(dimension))
    assert result.feasible and result.repair.outcome == "already_feasible"
    assert result.repair.sweeps == 0


def test_owner_deadline_aborts_without_clock_dependent_completed_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve completed evidence when a later repair is stopped between real rows.

    Args:
        monkeypatch (pytest.MonkeyPatch): Deterministic owner monotonic-clock sequence.
    """
    model = CompiledLinearModel(
        (LinearVariable("x", "x", 0, 1), LinearVariable("y", "y", 0, 1)),
        (1, 2),
        (LinearRow("balance", ((0, 1), (1, 1)), 1, 1), LinearRow("limit", ((0, 1),), upper=0.25)),
        repair="continuous",
    )
    session = SearchSession(model.decode, model.cost)
    session.evaluate(session.decode(np.array([0.25, 0.75])))
    before = session.model_evidence
    session.deadline = 3.0
    ticks = iter((1.0, 2.0, 4.0))
    monkeypatch.setattr("rko_solver.orchestration.search.time.perf_counter", lambda: next(ticks))
    with pytest.raises(_BudgetExpired):
        session.decode(np.array([1.0, 1.0]))
    assert session.n_evaluations == 1 and session.model_evidence == before


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
def test_public_repaired_model_runs_through_all_python_backends(execution: str) -> None:
    """Keep repaired equality evidence portable through real worker execution.

    Args:
        execution (str): Existing Python execution mode, including real process spawn.
    """
    pytest.importorskip("pyomo.environ")
    case = next(case for case in PracticalModels.cases() if case.name == "transport_equalities")
    solver = RKO.from_model(
        PracticalModels.native(case, "pyomo"),
        execution=execution,
        algorithms=["BRKGA"],
        adaptation="fixed",
        n_jobs=1 if execution == "process" else None,
        random_state=2026,
        prefill_pool=False,
    )
    result = solver.solve(0.2)
    witness = result.model.best_feasible
    assert witness is not None and witness.feasible
    assert PracticalModels.check(case, {a.label: a.value for a in witness.assignments})["feasible"]
    assert witness.keys == result.keys and witness.search_score == result.cost
    assert witness.repair.policy == "continuous"
    exported = result.model.to_dict()
    assert exported["repair"] == "continuous" and exported["repair_max_sweeps"] == 1024
    assert exported["best_feasible"]["repair"]["outcome"] in ("already_feasible", "repaired")


@pytest.mark.parametrize("frontend", ["pyomo", "pulp"])
def test_repaired_assignment_remains_valid_for_explicit_native_write_back(frontend: str) -> None:
    """Apply only the stored repaired witness against an explicitly compatible native model.

    Args:
        frontend (str): Qualified native runtime with transactional assignment slots.
    """
    pytest.importorskip("pyomo.environ" if frontend == "pyomo" else "pulp")
    case = next(case for case in PracticalModels.cases() if case.name == "transport_equalities")
    native = PracticalModels.native(case, frontend)
    solver = RKO.from_model(
        native,
        execution="sequential",
        algorithms=["MultiStart"],
        adaptation="fixed",
        random_state=2026,
        prefill_pool=False,
    )
    result = solver.solve(0.05)
    mapping = result.model.native_mapping(native)
    for variable in mapping.values():
        if frontend == "pyomo":
            variable.set_value(0.0)
        else:
            variable.varValue = 0.0
    result.model.write_back(native)
    applied = {
        assignment.label: (
            mapping[assignment.identifier].value
            if frontend == "pyomo"
            else mapping[assignment.identifier].varValue
        )
        for assignment in result.model.best_feasible.assignments
    }
    assert PracticalModels.check(case, applied)["feasible"]
    assert applied == {a.label: a.value for a in result.model.best_feasible.assignments}
