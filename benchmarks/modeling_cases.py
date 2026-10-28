"""Public-data practical LP/IP/MILP fixtures and independent numerical oracles.

Objective:
- Validate native model conversion against documented practical formulations, not engine internals.

Technical approach:
- Keep original coefficients/domains in immutable benchmark-only records.
- Build equivalent native Pyomo, PuLP 4 and MathOpt models without invoking their solvers.
- Check returned assignments with Python arithmetic, independently of compiled RKO data.
- Use optional SciPy/HiGHS only as a separately timed exact-model reference oracle.

Flow (pseudocode):
1. Load cited production, transportation, assignment and bundled Knapsack input data.
2. Construct equivalent bounded native models for the selected frontend.
3. Solve the independent reference once, outside all RKO optimization timing.
4. Validate original objectives, variable domains and every row without replaying RKO callbacks.

Important criteria:
- These are public teaching/reference instances, not industrial-scale certification.
- Finite implied bounds are declared explicitly; all adaptations are recorded.
- Equality stress and deliberately inconsistent models remain in the campaign.
- Reference solutions are never injected into search pools or used for repair.

Inputs/Outputs:
- Input: offline cited coefficients and optional native/reference dependencies.
- Output: native models, independently checked assignments and reference optima/status.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import numpy as np
import time
from .knapsack_utils import exact_profit
from .modeling_schema import CaseRow
from .modeling_schema import CaseVariable
from .modeling_schema import PracticalCase
from examples.run_kp_cpp_instances import default_instances_dir
from examples.run_kp_cpp_instances import read_kp_instance
from typing import Any


class PracticalModels:
    """Construct and independently verify cited practical native models.

    Responsibilities:
    - Share mathematical input across frontends without using generated decoder internals.
    - Keep all reference optimization isolated to an optional benchmark-only oracle.
    """

    @staticmethod
    def cases() -> tuple[PracticalCase, ...]:
        """Build the original eight-case campaign in its established stable order.

        Returns:
        tuple[PracticalCase, ...]: Unchanged bounded cited formulations.

        Raises:
        OSError: If bundled public Knapsack input cannot be read.
        ValueError: If bundled input disagrees with independently qualified reference data.
        """
        production = PracticalModels._production()
        return (
            production[:3]
            + tuple(PracticalModels._transport(flag) for flag in (False, True))
            + (PracticalModels._knapsack(), PracticalModels._assignment())
            + production[3:]
        )

    @staticmethod
    def native(case: PracticalCase, frontend: str) -> object:
        """Construct a real native mathematical model without calling a native optimizer.

        Args:
            case (PracticalCase): Original mathematical fixture.
            frontend (str): Pyomo, pulp or mathopt frontend.

        Returns:
            object: Constructed trusted native LP/IP/MILP model.

        Raises:
            ImportError: If the selected optional native runtime is unavailable.
            ValueError: If the selected frontend is unknown.
        """
        builders = {
            "pyomo": PracticalModels._native_pyomo,
            "pulp": PracticalModels._native_pulp,
            "mathopt": PracticalModels._native_mathopt,
        }
        builder = builders.get(frontend)
        if builder is None:
            raise ValueError("Expected pyomo, pulp or mathopt frontend.")
        return builder(case)

    @staticmethod
    def check(
        case: PracticalCase,
        values: dict[str, float],
        *,
        absolute_tolerance: float = 1e-06,
        relative_tolerance: float = 0.0,
    ) -> dict[str, Any]:
        """Independently evaluate original fixture data without compiled/native-model arithmetic.

        Args:
            case (PracticalCase): Original mathematical fixture.
            values (dict[str, float]): Explicit assignment by original unique labels.
            absolute_tolerance (float): Declared row-side absolute allowance.
            relative_tolerance (float): Declared row-side relative allowance.

        Returns:
            dict[str, Any]: Original objective, exact domain checks and row violation evidence.

        Raises:
            ValueError: If assignment identities or values are malformed/nonfinite.
        """
        ordered = PracticalModels._ordered(case, values)
        bounds_ok = all(
            (v.lower <= value <= v.upper for v, value in zip(case.variables, ordered, strict=True))
        )
        integrality_ok = all(
            (
                v.kind == "continuous" or value.is_integer()
                for v, value in zip(case.variables, ordered, strict=True)
            )
        )
        violations = PracticalModels._violations(
            case, ordered, absolute_tolerance, relative_tolerance
        )
        maximum = max(violations, default=0.0)
        return {
            "objective": math.fsum(
                (c * value for c, value in zip(case.objective, ordered, strict=True))
            ),
            "feasible": bounds_ok and integrality_ok and (maximum == 0),
            "bounds_ok": bounds_ok,
            "integrality_ok": integrality_ok,
            "maximum_adjusted_violation": maximum,
        }

    @classmethod
    def reference(cls, case: PracticalCase) -> dict[str, Any]:
        """Solve the original fixture once with optional SciPy/HiGHS outside RKO timing.

        Args:
            case (PracticalCase): Original model with a known optimum or deliberate inconsistency.

        Returns:
            dict[str, Any]: Reference status, objective, assignment and separate oracle time.

        Raises:
            ImportError: If benchmark-only SciPy is not installed.
            RuntimeError: If reference status, primal checks or known objective disagree.
        """
        result, elapsed = cls._solve_reference(case)
        if result.status == 2 and case.expected is None:
            return {
                "status": "reference_infeasible",
                "objective": None,
                "values": None,
                "elapsed_seconds": elapsed,
            }
        if result.status != 0 or case.expected is None:
            raise RuntimeError(
                f"Unexpected independent reference status for {case.name}: {result.message}"
            )
        values = {v.name: float(value) for v, value in zip(case.variables, result.x, strict=True)}
        check = cls.check(case, values, absolute_tolerance=1e-07)
        if not check["feasible"] or not math.isclose(
            check["objective"], case.expected, abs_tol=1e-07
        ):
            raise RuntimeError(
                f"Reference does not match independently known answer for {case.name}."
            )
        return {
            "status": "reference_optimal",
            "objective": check["objective"],
            "values": values,
            "elapsed_seconds": elapsed,
        }

    @staticmethod
    def _production() -> tuple[PracticalCase, ...]:
        """Build unchanged production LP, IP, setup and inconsistent fixtures.

        Returns:
            tuple[PracticalCase, ...]: Owned data preserving the original calculation order.
        """
        production_url = "https://jckantor.github.io/ND-Pyomo-Cookbook/notebooks/02.01-Production-Models-with-Linear-Constraints.html"
        production_variables = (CaseVariable("x", 0, 40), CaseVariable("y", 0, 80))
        production_rows = (
            CaseRow("demand_x", ((0, 1),), upper=40),
            CaseRow("labor_a", ((0, 1), (1, 1)), upper=80),
            CaseRow("labor_b", ((0, 2), (1, 1)), upper=100),
        )
        cases = [
            PracticalCase(
                "production_lp",
                "LP",
                production_url,
                "Explicit implied x<=40,y<=80 bounds; otherwise original mixed-product LP.",
                production_variables,
                (40, 30),
                production_rows,
                True,
                2600,
            ),
            PracticalCase(
                "production_integer",
                "IP",
                production_url,
                "Integer-unit variant of the same production data and finite bounds.",
                tuple(
                    CaseVariable(v.name, v.lower, v.upper, "integer") for v in production_variables
                ),
                (40, 30),
                production_rows,
                True,
                2600,
            ),
            PracticalCase(
                "production_setup",
                "MILP",
                production_url,
                "Original extension: binary X activation, setup cost100, x<=40*activation.",
                (CaseVariable("setup_x", 0, 1, "binary"), *production_variables),
                (-100, 40, 30),
                (
                    CaseRow("demand_x", ((1, 1),), upper=40),
                    CaseRow("labor_a", ((1, 1), (2, 1)), upper=80),
                    CaseRow("labor_b", ((1, 2), (2, 1)), upper=100),
                    CaseRow("activation", ((0, -40), (1, 1)), upper=0),
                ),
                True,
                2500,
            ),
        ]
        cases.append(
            PracticalCase(
                "production_inconsistent",
                "LP",
                production_url,
                "Deliberate x>=30,x<=10 inconsistency; original variable box remains valid.",
                production_variables,
                (40, 30),
                production_rows
                + (
                    CaseRow("minimum_x", ((0, 1),), lower=30),
                    CaseRow("maximum_x", ((0, 1),), upper=10),
                ),
                True,
                None,
            )
        )
        return tuple(cases)

    @staticmethod
    def _transport(equalities: bool) -> PracticalCase:
        """Build an unchanged cited transportation fixture.

        Args:
            equalities (bool): Use the original exact-demand stress variant.

        Returns:
            PracticalCase: Owned data preserving the original calculation order.
        """
        supply = (1000, 4000)
        demand = (500, 900, 1800, 200, 700)
        costs = (2, 4, 5, 2, 1, 3, 1, 3, 2, 3)
        variables = tuple(
            CaseVariable(f"route_{w}_{b}", 0, supply[w]) for w in range(2) for b in range(5)
        )
        supply_rows = tuple(
            CaseRow(f"supply_{w}", tuple((w * 5 + b, 1) for b in range(5)), upper=supply[w])
            for w in range(2)
        )
        rows = supply_rows + tuple(
            (
                CaseRow(
                    f"demand_{b}",
                    ((b, 1), (5 + b, 1)),
                    lower=amount,
                    upper=amount if equalities else math.inf,
                )
                for b, amount in enumerate(demand)
            )
        )
        return PracticalCase(
            "transport_equalities" if equalities else "transport_lp",
            "LP",
            "https://coin-or.github.io/pulp/CaseStudies/a_transportation_problem.html",
            "Continuous-flow relaxation with explicit warehouse-capacity implied bounds. "
            + (
                "Exact-demand stress variant; all other data unchanged."
                if equalities
                else "Original minimum-demand rows; positive costs retain the same optimum."
            ),
            variables,
            costs,
            rows,
            False,
            8600,
        )

    @staticmethod
    def _knapsack() -> PracticalCase:
        """Read and independently qualify the unchanged bundled Knapsack fixture.

        Returns:
            PracticalCase: Owned data preserving the original calculation order.

        Raises:
            OSError: If input cannot be read.
            ValueError: If its hash or independent optimum changed.
        """
        instance_path = default_instances_dir() / "kp50.txt"
        if (
            hashlib.sha256(instance_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            != "ed7240148053a3676f04bba92294357dd5bc4a53b4d585ebefc724171070c277"
        ):
            raise ValueError("Bundled kp50 reference data changed; requalify its provenance.")
        instance = read_kp_instance(instance_path)
        if exact_profit(instance) != 149:
            raise ValueError("Independent Knapsack dynamic program disagrees with the reference.")
        return PracticalCase(
            "knapsack_50",
            "IP",
            "https://github.com/RKO-solver/RKO_Cpp_v2.0/tree/126e35873e50dc308795b58b8c75d37f8de4c857/Instances/KP",
            "Unchanged kp50 data; binary formulation without a handwritten greedy decoder.",
            tuple(CaseVariable(f"item_{i:03}", 0, 1, "binary") for i in range(instance.n_items)),
            tuple(instance.benefits),
            (CaseRow("capacity", tuple(enumerate(instance.weights)), upper=instance.capacity),),
            True,
            149,
        )

    @staticmethod
    def _assignment() -> PracticalCase:
        """Build the unchanged cited assignment fixture.

        Returns:
            PracticalCase: Owned data preserving the original calculation order.
        """
        assignment_costs = (
            90,
            80,
            75,
            70,
            35,
            85,
            55,
            65,
            125,
            95,
            90,
            95,
            45,
            110,
            95,
            115,
            50,
            100,
            90,
            100,
        )
        return PracticalCase(
            "assignment",
            "IP",
            "https://developers.google.com/optimization/assignment/assignment_example",
            "Original five-worker/four-task data and constraints in each native frontend.",
            tuple(
                CaseVariable(f"worker_{w}_task_{t}", 0, 1, "binary")
                for w in range(5)
                for t in range(4)
            ),
            assignment_costs,
            tuple(
                CaseRow(f"worker_{w}", tuple((w * 4 + t, 1) for t in range(4)), upper=1)
                for w in range(5)
            )
            + tuple(
                CaseRow(f"task_{t}", tuple((w * 4 + t, 1) for w in range(5)), lower=1, upper=1)
                for t in range(4)
            ),
            False,
            265,
        )

    @staticmethod
    def _native_pyomo(case: PracticalCase) -> object:
        """Construct only the original pyomo formulation.

        Args:
            case (PracticalCase): Original mathematical fixture.

        Returns:
            object: Owned data preserving the original calculation order.

        Raises:
            ImportError: If the optional frontend is unavailable.
        """
        p = importlib.import_module("pyomo.environ")
        model = p.ConcreteModel(name=case.name)
        variables = []
        for variable in case.variables:
            target = p.Var(
                domain={"continuous": p.Reals, "integer": p.Integers, "binary": p.Binary}[
                    variable.kind
                ],
                bounds=(variable.lower, variable.upper),
            )
            model.add_component(variable.name, target)
            variables.append(target)
        objective = sum((c * v for c, v in zip(case.objective, variables, strict=True)))
        model.objective = p.Objective(
            expr=objective, sense=p.maximize if case.maximize else p.minimize
        )
        for row in case.rows:
            expression = sum(
                (coefficient * variables[index] for index, coefficient in row.coefficients)
            )
            model.add_component(
                row.name,
                p.Constraint(
                    expr=p.inequality(
                        row.lower if math.isfinite(row.lower) else None,
                        expression,
                        row.upper if math.isfinite(row.upper) else None,
                    )
                ),
            )
        return model

    @staticmethod
    def _native_pulp(case: PracticalCase) -> object:
        """Construct only the original pulp formulation.

        Args:
            case (PracticalCase): Original mathematical fixture.

        Returns:
            object: Owned data preserving the original calculation order.

        Raises:
            ImportError: If the optional frontend is unavailable.
        """
        p = importlib.import_module("pulp")
        model = p.LpProblem(case.name, p.LpMaximize if case.maximize else p.LpMinimize)
        variables = [
            model.add_variable(
                v.name,
                v.lower,
                v.upper,
                {"continuous": p.LpContinuous, "integer": p.LpInteger, "binary": p.LpBinary}[
                    v.kind
                ],
            )
            for v in case.variables
        ]
        model += sum((c * v for c, v in zip(case.objective, variables, strict=True)))
        for row in case.rows:
            expression = sum(
                (coefficient * variables[index] for index, coefficient in row.coefficients)
            )
            if row.lower == row.upper:
                model += (expression == row.lower, row.name)
            else:
                if math.isfinite(row.lower):
                    model += (expression >= row.lower, row.name + "_lower")
                if math.isfinite(row.upper):
                    model += (expression <= row.upper, row.name + "_upper")
        return model

    @staticmethod
    def _native_mathopt(case: PracticalCase) -> object:
        """Construct only the original mathopt formulation.

        Args:
            case (PracticalCase): Original mathematical fixture.

        Returns:
            object: Owned data preserving the original calculation order.

        Raises:
            ImportError: If the optional frontend is unavailable.
        """
        p = importlib.import_module("ortools.math_opt.python.mathopt")
        model = p.Model(name=case.name)
        variables = [
            model.add_variable(
                lb=v.lower, ub=v.upper, name=v.name, is_integer=v.kind != "continuous"
            )
            for v in case.variables
        ]
        objective = sum((c * v for c, v in zip(case.objective, variables, strict=True)))
        model.maximize(objective) if case.maximize else model.minimize(objective)
        for row in case.rows:
            expression = sum(
                (coefficient * variables[index] for index, coefficient in row.coefficients)
            )
            model.add_linear_constraint(expr=expression, lb=row.lower, ub=row.upper, name=row.name)
        return model

    @staticmethod
    def _ordered(case: PracticalCase, values: dict[str, float]) -> list[float]:
        """Validate complete finite assignments in original fixture order.

        Args:
            case (PracticalCase): Original fixture.
            values (dict[str, float]): Complete assignment by original labels.

        Returns:
            list[float]: Owned data preserving the original calculation order.

        Raises:
            ValueError: If identities or values are incomplete or nonfinite.
        """
        if set(values) != {v.name for v in case.variables} or not all(
            math.isfinite(value) for value in values.values()
        ):
            raise ValueError(
                "Independent validation requires complete finite original assignments."
            )
        ordered = [float(values[v.name]) for v in case.variables]
        return ordered

    @staticmethod
    def _violations(
        case: PracticalCase,
        ordered: list[float],
        absolute_tolerance: float,
        relative_tolerance: float,
    ) -> list[float]:
        """Evaluate original row-side violations independently of generated callbacks.

        Args:
            case (PracticalCase): Original fixture.
            ordered (list[float]): Validated original-coordinate assignment.
            absolute_tolerance (float): Original absolute row-side allowance.
            relative_tolerance (float): Original relative row-side allowance.

        Returns:
            list[float]: Owned data preserving the original calculation order.
        """
        violations = []
        for row in case.rows:
            activity = math.fsum(
                (coefficient * ordered[index] for index, coefficient in row.coefficients)
            )
            for bound, residual in (
                (row.lower, row.lower - activity),
                (row.upper, activity - row.upper),
            ):
                if math.isfinite(bound):
                    violations.append(
                        max(0.0, residual - absolute_tolerance - relative_tolerance * abs(bound))
                    )
        return violations

    @staticmethod
    def _solve_reference(case: PracticalCase) -> tuple[Any, float]:
        """Run the unchanged optional reference oracle outside RKO timing.

        Args:
            case (PracticalCase): Original independently cited fixture.

        Returns:
            tuple[Any, float]: Owned data preserving the original calculation order.

        Raises:
            ImportError: If benchmark-only SciPy is unavailable.
        """
        scipy = importlib.import_module("scipy.optimize")
        matrix = np.zeros((len(case.rows), len(case.variables)))
        for index, row in enumerate(case.rows):
            for column, coefficient in row.coefficients:
                matrix[index, column] = coefficient
        started = time.perf_counter()
        result = scipy.milp(
            np.array(case.objective, dtype=float) * (-1 if case.maximize else 1),
            integrality=np.array([v.kind != "continuous" for v in case.variables], dtype=int),
            bounds=scipy.Bounds(
                [v.lower for v in case.variables], [v.upper for v in case.variables]
            ),
            constraints=scipy.LinearConstraint(
                matrix, [row.lower for row in case.rows], [row.upper for row in case.rows]
            ),
            options={"time_limit": 30.0, "mip_rel_gap": 0.0},
        )
        elapsed = time.perf_counter() - started
        return (result, elapsed)
