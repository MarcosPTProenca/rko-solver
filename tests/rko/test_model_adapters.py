"""Qualified native frontend extraction and explicit assignment regression tests.

Objective:
- Verify equivalent bounded models without invoking native optimization solvers.

Technical approach:
- Construct tiny equivalent LP/MILP cases using optional qualified frontend versions.
- Compare generated assignments, original objectives and residuals for deterministic keys.
- Audit unsupported features, source drift, portable snapshots and write-back rollback.

Flow (pseudocode):
1. Construct a trusted native model and extract an immutable source record.
2. Compile/decode fixed keys with the shared numerical implementation.
3. Validate source isolation, explicit native mappings and transactional application.

Important criteria:
- Individual frontend cases skip only when that optional frontend is unavailable.
- Version candidates are runtime-tested here; platform CI qualification is separate.

Inputs/Outputs:
- Input: native tiny models and deliberate unsupported features/drift/failures.
- Output: semantic, immutability, assignment and rejection assertions.
"""

import dataclasses
import importlib
import numpy as np
import pickle
import pytest
import subprocess
import sys
from rko_solver.modeling import adapters
from rko_solver.modeling.adapters import ModelAdapter
from rko_solver.modeling.linear import ModelEvidence
from rko_solver.orchestration.types import RunConfig
from rko_solver.solver import RKO


class FrontendCases:
    """Construct equivalent tiny native models without external optimization calls.

    Responsibilities:
    - Isolate optional imports and accommodate each frontend's qualified construction API.
    """

    @staticmethod
    def build(frontend: str, maximize: bool = False) -> object:
        """Build equivalent mixed-domain models with constant offsets and ranged semantics.

        Args:
            frontend (str): Optional frontend identifier.
            maximize (bool): Original objective direction.

        Returns:
            object: Trusted constructed native model.
        """
        if frontend == "pyomo":
            p = pytest.importorskip("pyomo.environ")
            model = p.ConcreteModel(name="tiny")
            model.x = p.Var(bounds=(-2, 2))
            model.y = p.Var(domain=p.Integers, bounds=(-2.2, 2.8))
            model.z = p.Var(domain=p.Binary)
            model.f = p.Var(bounds=(0, 8))
            model.f.fix(3)
            model.obj = p.Objective(
                expr=2 * model.x - 3 * model.y + 4 * model.z + 5 * model.f - 7,
                sense=p.maximize if maximize else p.minimize,
            )
            model.row = p.Constraint(
                expr=p.inequality(
                    2,
                    model.x + 2 * model.y - model.z + 3 * model.f - 1,
                    10,
                )
            )
            return model
        if frontend == "pulp":
            p = pytest.importorskip("pulp")
            model = p.LpProblem("tiny", p.LpMaximize if maximize else p.LpMinimize)
            x = model.add_variable("x", -2, 2)
            y = model.add_variable("y", -2.2, 2.8, p.LpInteger)
            z = model.add_variable("z", cat=p.LpBinary)
            f = model.add_variable("f", 3, 3)
            model += 2 * x - 3 * y + 4 * z + 5 * f - 7
            expression = x + 2 * y - z + 3 * f - 1
            model += expression >= 2, "lo"
            model += expression <= 10, "hi"
            return model
        m = pytest.importorskip("ortools.math_opt.python.mathopt")
        model = m.Model(name="tiny")
        x = model.add_variable(lb=-2, ub=2, name="x")
        y = model.add_variable(lb=-2.2, ub=2.8, is_integer=True, name="y")
        z = model.add_binary_variable(name="z")
        f = model.add_variable(lb=3, ub=3, name="f")
        objective = 2 * x - 3 * y + 4 * z + 5 * f - 7
        model.maximize(objective) if maximize else model.minimize(objective)
        model.add_linear_constraint(lb=2, ub=10, expr=x + 2 * y - z + 3 * f - 1, name="row")
        return model

    @staticmethod
    def keys(source: object, snapshot: object) -> np.ndarray:
        """Construct equivalent chromosomes regardless of frontend variable order.

        Args:
            source (object): Extracted original model record.
            snapshot (object): Compiled numerical model.

        Returns:
            np.ndarray: Keys for assignment x=-1, y=0, z=1 and fixed f=3.
        """
        by_label = {"x": 0.25, "y": 0.5, "z": 1.0}
        return np.array(
            [
                by_label[variable.label]
                for variable, domain in zip(source.variables, snapshot.domains, strict=True)
                if not domain.fixed
            ]
        )


@pytest.mark.parametrize("frontend", ["pyomo", "pulp", "mathopt"])
@pytest.mark.parametrize("maximize", [False, True])
def test_equivalent_native_semantics_and_owned_snapshot(frontend: str, maximize: bool) -> None:
    """Preserve mixed-domain assignments, direction, constants and portable source isolation.

    Args:
        frontend (str): Qualified optional modeling frontend.
        maximize (bool): Original objective direction.
    """
    native = FrontendCases.build(frontend, maximize)
    source = ModelAdapter.extract(native)
    snapshot = source.compile()
    evidence = snapshot.decode(FrontendCases.keys(source, snapshot))
    assert dict(zip((v.label for v in source.variables), evidence.values, strict=True)) == {
        "x": -1,
        "y": 0,
        "z": 1,
        "f": 3,
    }
    assert evidence.objective == 10
    assert evidence.feasible
    assert source.maximize is maximize
    assert source.frontend == frontend
    assert ModelAdapter.extract(native).fingerprint == source.fingerprint
    assert pickle.loads(pickle.dumps(snapshot)).decode(np.asarray(evidence.keys)) == evidence
    assert pickle.loads(pickle.dumps(source)) == source
    for value in (0, 0.5, 1):
        chromosome = np.full(snapshot.n_keys, value)
        assert (
            snapshot.decode(chromosome).score
            == pickle.loads(pickle.dumps(snapshot)).decode(chromosome).score
        )


@pytest.mark.parametrize("frontend", ["pyomo", "pulp", "mathopt"])
def test_native_mapping_and_source_drift(frontend: str) -> None:
    """Ignore solution guesses but reject changed source bounds before native reconstruction.

    Args:
        frontend (str): Qualified optional modeling frontend.
    """
    model = FrontendCases.build(frontend)
    source = ModelAdapter.extract(model)
    mapping = ModelAdapter.variable_mapping(model, source)
    target = next(mapping[v.identifier] for v in source.variables if v.label == "x")
    if frontend == "pyomo":
        target.set_value(1)
    elif frontend == "pulp":
        target.varValue = 1
    assert ModelAdapter.extract(model).fingerprint == source.fingerprint
    if frontend == "pyomo":
        target.setub(1)
    elif frontend == "pulp":
        target.upBound = 1
    else:
        target.upper_bound = 1
    with pytest.raises(ValueError, match="changed or is incompatible"):
        ModelAdapter.variable_mapping(model, source)


@pytest.mark.parametrize("frontend", ["pulp", "mathopt"])
def test_duplicate_display_names_use_native_ids(frontend: str) -> None:
    """Export unambiguous stable IDs when native APIs permit duplicate variable labels.

    Args:
        frontend (str): Qualified optional modeling frontend permitting duplicate names.
    """
    if frontend == "pulp":
        p = pytest.importorskip("pulp")
        model = p.LpProblem()
        x = model.add_variable("duplicate", 0, 1)
        y = model.add_variable("duplicate", 0, 2)
        model += x + 3 * y + 7
    else:
        m = pytest.importorskip("ortools.math_opt.python.mathopt")
        model = m.Model()
        x = model.add_variable(lb=0, ub=1, name="duplicate")
        y = model.add_variable(lb=0, ub=2, name="duplicate")
        model.minimize(x + 3 * y + 7)
    source = ModelAdapter.extract(model)
    assert len({v.identifier for v in source.variables}) == 2
    assert len({v.label for v in source.variables}) == 1
    assert source.compile().decode(np.array([1, 1])).objective == 14
    assert len(ModelAdapter.variable_mapping(model, source)) == 2


@pytest.mark.parametrize("frontend", ["pyomo", "pulp"])
def test_explicit_write_back_and_invalid_witness(frontend: str) -> None:
    """Apply only validated feasible values without fixing variables or altering source bounds.

    Args:
        frontend (str): Qualified optional frontend with native assignment slots.
    """
    model = FrontendCases.build(frontend)
    source = ModelAdapter.extract(model)
    snapshot = source.compile()
    evidence = snapshot.decode(FrontendCases.keys(source, snapshot))
    mapping = ModelAdapter.variable_mapping(model, source)
    if frontend == "pulp":
        for target in mapping.values():
            target.varValue = target.lowBound
    ModelAdapter.write_back(model, source, snapshot, evidence)
    values = tuple(
        target.value if frontend == "pyomo" else target.varValue for target in mapping.values()
    )
    assert values == evidence.values
    assert ModelAdapter.extract(model).fingerprint == source.fingerprint
    invalid = dataclasses.replace(evidence, values=tuple(100.0 for _ in evidence.values))
    with pytest.raises(ValueError, match="validated feasible"):
        ModelAdapter.write_back(model, source, snapshot, invalid)
    with pytest.raises(ValueError, match="validated feasible"):
        ModelAdapter.write_back(
            model, source, snapshot, dataclasses.replace(evidence, feasible=False)
        )


@pytest.mark.parametrize("frontend", ["pyomo", "pulp"])
@pytest.mark.parametrize("rollback_fails", [False, True])
def test_write_back_rolls_back_and_reports_failure(
    frontend: str,
    rollback_fails: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore prior values on partial failure and explicitly surface failed rollback.

    Args:
        frontend (str): Qualified optional frontend with native assignment slots.
        rollback_fails (bool): Whether the failing native setter also rejects rollback.
        monkeypatch (pytest.MonkeyPatch): Inject a native setter failure.
    """
    model = FrontendCases.build(frontend)
    source = ModelAdapter.extract(model)
    snapshot = source.compile()
    evidence = snapshot.decode(FrontendCases.keys(source, snapshot))
    targets = tuple(ModelAdapter.variable_mapping(model, source).values())
    if frontend == "pulp":
        for target in targets:
            target.varValue = target.lowBound
    victim = targets[1]
    prior = victim.value if frontend == "pyomo" else victim.varValue
    label = "rollback failed" if rollback_fails else "prior values restored"
    if frontend == "pyomo":
        kind = type(victim)
        original = kind.set_value

        def fail(target: object, value: float | None, *args: object, **kwargs: object) -> None:
            """Inject one setter failure, optionally including restoration.

            Args:
                target (object): Native variable receiving the assignment.
                value (float | None): Assigned or prior value.
                args (object): Additional native setter positional inputs.
                kwargs (object): Additional native setter keyword inputs.

            Raises:
                RuntimeError: When assigning to the injected failure target.
            """
            if target is victim and (rollback_fails or value is not None):
                raise RuntimeError("injected setter failure")
            original(target, value, *args, **kwargs)

        monkeypatch.setattr(kind, "set_value", fail)
    else:
        kind = type(victim)
        descriptor = kind.varValue

        def fail(target: object, value: float | None) -> None:
            """Inject one ID-based PuLP value setter failure.

            Args:
                target (object): Native variable receiving the assignment.
                value (float | None): Assigned or previous value.

            Raises:
                RuntimeError: When assigning to the injected failure target.
            """
            if target.id == victim.id and (rollback_fails or value != prior):
                raise RuntimeError("injected setter failure")
            descriptor.fset(target, value)

        monkeypatch.setattr(kind, "varValue", property(descriptor.fget, fail))
    before = tuple(target.value if frontend == "pyomo" else target.varValue for target in targets)
    with pytest.raises(RuntimeError, match=label):
        ModelAdapter.write_back(model, source, snapshot, evidence)
    after = tuple(target.value if frontend == "pyomo" else target.varValue for target in targets)
    assert after == before


def test_pulp_unset_prior_values_reject_before_any_mutation() -> None:
    """Reject fresh PuLP native write-back because its setter cannot roll back to None."""
    model = FrontendCases.build("pulp")
    source = ModelAdapter.extract(model)
    snapshot = source.compile()
    evidence = snapshot.decode(FrontendCases.keys(source, snapshot))
    targets = tuple(ModelAdapter.variable_mapping(model, source).values())
    assert all(target.varValue is None for target in targets)
    with pytest.raises(ValueError, match="initialized finite prior values"):
        ModelAdapter.write_back(model, source, snapshot, evidence)
    assert all(target.varValue is None for target in targets)


def test_mathopt_rejects_write_back_without_changing_bounds() -> None:
    """Reject native MathOpt write-back instead of mutating bounds to fake assignments."""
    model = FrontendCases.build("mathopt")
    source = ModelAdapter.extract(model)
    snapshot = source.compile()
    evidence = snapshot.decode(FrontendCases.keys(source, snapshot))
    with pytest.raises(NotImplementedError, match="no native assignment slot"):
        ModelAdapter.write_back(model, source, snapshot, evidence)
    assert ModelAdapter.extract(model).fingerprint == source.fingerprint


def test_pyomo_active_semantics_and_domains() -> None:
    """Ignore inactive structures but reject active nonlinear, open and noncontiguous domains."""
    p = pytest.importorskip("pyomo.environ")
    model = FrontendCases.build("pyomo")
    model.inactive = p.Block()
    model.inactive.bad = p.Var(domain=p.PositiveReals)
    model.inactive.nonlinear = p.Constraint(expr=model.x**2 <= 1)
    model.inactive.deactivate()
    model.off = p.Objective(expr=model.x**2)
    model.off.deactivate()
    ModelAdapter.extract(model).compile()
    model.off.activate()
    with pytest.raises(ValueError, match="exactly one"):
        ModelAdapter.extract(model)
    model.off.deactivate()
    model.bad = p.Constraint(expr=model.x**2 <= 1)
    with pytest.raises(ValueError, match="nonlinear"):
        ModelAdapter.extract(model)
    model.bad.deactivate()
    model.x.domain = p.PositiveReals
    with pytest.raises(ValueError, match="unsupported domain"):
        ModelAdapter.extract(model)
    model.allowed = p.Set(initialize=[0, 2])
    model.x.domain = model.allowed
    with pytest.raises(ValueError, match="unsupported domain"):
        ModelAdapter.extract(model)


def test_pyomo_unsupported_components_and_fixed_source_drift() -> None:
    """Reject active SOS/logic metadata and detect changed fixed-expression coefficients."""
    p = pytest.importorskip("pyomo.environ")
    model = FrontendCases.build("pyomo")
    source = ModelAdapter.extract(model)
    model.obj.expr = model.obj.expr + model.f - 3
    assert ModelAdapter.extract(model).fingerprint != source.fingerprint
    model.flag = p.BooleanVar()
    with pytest.raises(ValueError, match="unsupported BooleanVar"):
        ModelAdapter.extract(model)
    model.del_component(model.flag)
    model.sos = p.SOSConstraint(rule=lambda m: ([m.x, m.y], [1, 2]), sos=1)
    with pytest.raises(ValueError, match="unsupported SOSConstraint"):
        ModelAdapter.extract(model)


@pytest.mark.parametrize(
    "feature", ["quadratic_objective", "quadratic_constraint", "indicator", "auxiliary"]
)
def test_mathopt_rejects_extra_proto_features(feature: str) -> None:
    """Reject populated unsupported MathOpt features even when linear rows remain valid.

    Args:
        feature (str): Unsupported native model feature to add.
    """
    model = FrontendCases.build("mathopt")
    x, y, z, _ = tuple(model.variables())
    if feature == "quadratic_objective":
        model.minimize(x * x)
    elif feature == "quadratic_constraint":
        model.add_quadratic_constraint(x * y <= 1)
    elif feature == "indicator":
        model.add_indicator_constraint(indicator=z, implied_constraint=x >= 0)
    else:
        model.add_auxiliary_objective(expr=x, is_maximize=False, priority=1)
    with pytest.raises(ValueError, match="unsupported"):
        ModelAdapter.extract(model)


def test_pulp_sos_and_objective_constant_preservation() -> None:
    """Reject SOS and retain constants omitted by PuLP's name-based toDict serialization."""
    p = pytest.importorskip("pulp")
    model = FrontendCases.build("pulp")
    assert ModelAdapter.extract(model).objective_constant == -7
    model.objective = None
    with pytest.raises(ValueError, match="explicit linear objective"):
        ModelAdapter.extract(model)
    model += 0
    variables = model.variables()
    model.add_sos_group(p._rustcore.SosKind.Sos1, "sos", {variables[0]: 1, variables[1]: 2})
    with pytest.raises(ValueError, match="SOS constraints"):
        ModelAdapter.extract(model)


def test_unknown_model_and_unqualified_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject wrong frontend objects and provide actionable version diagnostics.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replace selected frontend metadata version.
    """
    with pytest.raises(ValueError, match="Expected a Pyomo"):
        ModelAdapter.extract(object())
    model = FrontendCases.build("pulp")
    monkeypatch.setattr(adapters, "version", lambda _: "3.0.0")
    with pytest.raises(ValueError, match=r"install rko-solver\[pulp\]"):
        ModelAdapter.extract(model)


def test_core_import_does_not_import_optional_frontends() -> None:
    """Keep default RKO and adapter-module imports free of optional native runtimes."""
    script = (
        "import sys; import rko_solver; import rko_solver.modeling.adapters; "
        "assert not any(n.split('.')[0] in {'pyomo','pulp','ortools'} for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)


@pytest.mark.parametrize("frontend", ["pyomo", "pulp", "mathopt"])
def test_missing_free_bounds_and_independent_search_overrides(frontend: str) -> None:
    """Require explicit finite boxes without confusing compilation overrides with source drift.

    Args:
        frontend (str): Qualified optional modeling frontend.
    """
    model = FrontendCases.build(frontend)
    initial = ModelAdapter.extract(model)
    identifier = next(v.identifier for v in initial.variables if v.label == "x")
    target = ModelAdapter.variable_mapping(model, initial)[identifier]
    if frontend == "pyomo":
        target.setlb(None)
        target.setub(None)
    elif frontend == "pulp":
        target.lowBound = None
        target.upBound = None
    else:
        target.lower_bound = -float("inf")
        target.upper_bound = float("inf")
    source = ModelAdapter.extract(model)
    with pytest.raises(ValueError, match="finite effective bounds"):
        source.compile()
    identifier = next(v.identifier for v in source.variables if v.label == "x")
    snapshot = source.compile(search_bounds={identifier: (-2, 2)})
    assert next(d for d in snapshot.domains if d.identifier == identifier).restricted
    assert ModelAdapter.extract(model).fingerprint == source.fingerprint
    assert ModelAdapter.variable_mapping(model, source)[identifier] is not None


@pytest.mark.parametrize("frontend", ["pyomo", "pulp", "mathopt"])
def test_generated_frontend_snapshot_runs_in_spawn_without_native_model(frontend: str) -> None:
    """Transport only owned numerical callbacks and bounded witnesses to spawned workers.

    Args:
        frontend (str): Qualified optional modeling frontend.
    """
    model = FrontendCases.build(frontend)
    source = ModelAdapter.extract(model)
    snapshot = source.compile()
    solver = RKO(
        snapshot.decode,
        snapshot.cost,
        snapshot.n_keys,
        execution="process",
        n_jobs=1,
        random_state=42,
    )
    archives: list[ModelEvidence] = []
    keys, score = solver._execute(
        RunConfig(0, 2, ("MultiStart",)), 0.02, started_at=0, model_evidence=archives
    )
    assert source.fingerprint == ModelAdapter.extract(model).fingerprint
    assert snapshot.merge_evidence(tuple(archives), keys, score).incumbent.keys == tuple(keys)


@pytest.mark.parametrize("frontend", ["pyomo", "pulp", "mathopt"])
def test_constant_models_and_inconsistent_constant_rows(frontend: str) -> None:
    """Extract zero-variable models and preserve infeasible constant rows without solving.

    Args:
        frontend (str): Qualified optional modeling frontend.
    """
    if frontend == "pyomo":
        p = pytest.importorskip("pyomo.environ")
        model = p.ConcreteModel()
        model.obj = p.Objective(expr=7)
    elif frontend == "pulp":
        p = pytest.importorskip("pulp")
        model = p.LpProblem()
        model += 7
    else:
        p = pytest.importorskip("ortools.math_opt.python.mathopt")
        model = p.Model()
        model.minimize(7)
    snapshot = ModelAdapter.extract(model).compile()
    assert snapshot.n_keys == 0
    assert snapshot.decode(np.array([])).objective == 7
    assert snapshot.decode(np.array([])).feasible
    if frontend == "pyomo":
        model.bad = p.Constraint(expr=p.Constraint.Infeasible)
    elif frontend == "pulp":
        model += p.LpAffineExpression.from_dict({}, constant=1) <= 0
    else:
        model.add_linear_constraint(expr=0, lb=1, ub=1)
    snapshot = ModelAdapter.extract(model).compile()
    assert snapshot.n_keys == 0
    assert not snapshot.decode(np.array([])).feasible


def test_pyomo_closed_custom_interval_and_export_suffix_audit() -> None:
    """Accept closed continuous interval domains but reject exported solver annotations."""
    p = pytest.importorskip("pyomo.environ")
    model = p.ConcreteModel()
    model.interval = p.RangeSet(0, 2, 0)
    model.x = p.Var(domain=model.interval)
    model.obj = p.Objective(expr=model.x)
    assert ModelAdapter.extract(model).compile().decode(np.array([0.5])).values == (1,)
    model.annotation = p.Suffix(direction=p.Suffix.EXPORT)
    model.annotation[model.x] = 1
    with pytest.raises(ValueError, match="export suffix"):
        ModelAdapter.extract(model)


def test_write_back_rejects_other_objective_and_changed_source() -> None:
    """Reject a snapshot with different objective semantics before touching native values."""
    model = FrontendCases.build("pyomo")
    source = ModelAdapter.extract(model)
    snapshot = dataclasses.replace(source, objective_constant=123).compile()
    evidence = snapshot.decode(FrontendCases.keys(source, snapshot))
    with pytest.raises(ValueError, match="snapshot objective"):
        ModelAdapter.write_back(model, source, snapshot, evidence)
    assert model.x.value is None
    snapshot = source.compile()
    evidence = snapshot.decode(FrontendCases.keys(source, snapshot))
    model.obj.expr += 1
    with pytest.raises(ValueError, match="changed or is incompatible"):
        ModelAdapter.write_back(model, source, snapshot, evidence)
    assert model.x.value is None


@pytest.mark.parametrize("frontend", ["pyomo", "pulp", "mathopt"])
@pytest.mark.parametrize("side", ["lower", "upper", "equality"])
def test_native_relative_tolerance_uses_actual_rhs(frontend: str, side: str) -> None:
    """Preserve native finite RHS magnitudes rather than shifting PuLP rows to zero bounds.

    Args:
        frontend (str): Qualified optional frontend.
        side (str): Lower, upper or equality row semantics.
    """
    if frontend == "pyomo":
        p = pytest.importorskip("pyomo.environ")
        model = p.ConcreteModel()
        model.x = p.Var(bounds=(0, 256))
        x = model.x
        model.obj = p.Objective(expr=x)
    elif frontend == "pulp":
        p = pytest.importorskip("pulp")
        model = p.LpProblem()
        x = model.add_variable("x", 0, 256)
        model += x
    else:
        p = pytest.importorskip("ortools.math_opt.python.mathopt")
        model = p.Model()
        x = model.add_variable(lb=0, ub=256, name="x")
        model.minimize(x)
    expression = x >= 100 if side == "lower" else x <= 100 if side == "upper" else x == 100
    if frontend == "pyomo":
        model.row = p.Constraint(expr=expression)
    elif frontend == "pulp":
        model += expression, "row"
    else:
        model.add_linear_constraint(expression, name="row")
    source = ModelAdapter.extract(model)
    snapshot = source.compile(absolute_tolerance=1, relative_tolerance=0.1)
    allowed = 111 if side == "upper" else 89
    rejected = 112 if side in ("upper", "equality") else 88
    good = snapshot.decode(np.array([allowed / 256]))
    assert good.feasible
    assert max(good.rows[0].lower_residual, good.rows[0].upper_residual) == 11
    assert not snapshot.decode(np.array([rejected / 256])).feasible
    assert not source.compile().decode(np.array([allowed / 256])).feasible


@pytest.mark.parametrize("frontend", ["pyomo", "pulp", "mathopt"])
def test_native_affine_equality_preserves_both_sides(frontend: str) -> None:
    """Preserve nonzero affine equality offsets and independent lower/upper residuals.

    Args:
        frontend (str): Qualified optional frontend.
    """
    model = FrontendCases.build(frontend)
    initial = ModelAdapter.extract(model)
    identifier = next(v.identifier for v in initial.variables if v.label == "y")
    y = ModelAdapter.variable_mapping(model, initial)[identifier]
    if frontend == "pyomo":
        p = importlib.import_module("pyomo.environ")
        model.eq = p.Constraint(expr=2 * y + 3 == 5)
    elif frontend == "pulp":
        model += 2 * y + 3 == 5, "eq"
    else:
        model.add_linear_constraint(2 * y + 3 == 5, name="eq")
    source = ModelAdapter.extract(model)
    snapshot = source.compile()
    keys = FrontendCases.keys(source, snapshot)
    bad = snapshot.decode(keys)
    assert bad.rows[-1].lower_residual == 2
    assert bad.rows[-1].upper_residual == 0
    free = [
        v for v, domain in zip(source.variables, snapshot.domains, strict=True) if not domain.fixed
    ]
    keys[next(index for index, variable in enumerate(free) if variable.label == "y")] = 0.75
    assert snapshot.decode(keys).feasible


@pytest.mark.parametrize("field", ["bounds", "objective", "coefficient"])
def test_pyomo_rejects_inexact_integer_data_before_float_coercion(field: str) -> None:
    """Reject native exact integers that would be silently rounded during extraction.

    Args:
        field (str): Native model location holding an inexactly representable large integer.
    """
    p = pytest.importorskip("pyomo.environ")
    lower = 10**90
    model = p.ConcreteModel()
    model.x = p.Var(bounds=(lower, lower + 1) if field == "bounds" else (0, 1))
    expression = (
        model.x + lower
        if field == "objective"
        else lower * model.x
        if field == "coefficient"
        else model.x
    )
    model.obj = p.Objective(expr=expression)
    with pytest.raises(ValueError, match="exactly representable"):
        ModelAdapter.extract(model)
