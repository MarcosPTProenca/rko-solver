"""Bounded mixed-integer native model conversion without handwritten callbacks.

Objective:
- Demonstrate Pyomo, PuLP 4 and MathOpt model reuse through the same public RKO factory.

Technical approach:
- Construct a tiny original MILP with binary/integer/continuous domains and one capacity row.
- Maximize the original linear objective using normal RKO search on generated keys.
- Print only retained feasible assignments, keeping scalar search scores distinct.

Flow (pseudocode):
1. Build a trusted bounded native model with the selected optional frontend.
2. Compile with RKO.from_model and run the existing serial SA algorithm.
3. Print honest feasibility status and the original-objective witness when available.

Important criteria:
- No external mathematical optimization solver or problem-specific callbacks are used.
- The analytic maximum is 3.5; heuristic search is not guaranteed to find/certify it.
- Native stored values are not modified automatically and no infeasibility is certified.

Inputs/Outputs:
- Input: optional frontend name on the module command line.
- Output: model status, search score, original feasible objective and stable-ID assignments.
"""

import argparse
import importlib
from rko_solver.solver import RKO


class LinearModelExample:
    """Reuse an equivalent small native MILP through optional frontend conversion.

    Responsibilities:
    - Demonstrate all three construction APIs and honest original-objective result handling.
    """

    @staticmethod
    def build(frontend: str) -> object:
        """Construct max 3*x+y+0.5*c with 2*x+y<=2 and bounded mixed domains.

        Args:
            frontend (str): Pyomo, pulp or mathopt frontend name.

        Returns:
            object: Native model; x binary, y integer in [0,2], c continuous in [0,1].

        Raises:
            ImportError: If the selected optional frontend extra is unavailable.
            ValueError: If the selected frontend name is unsupported.
        """
        if frontend == "pyomo":
            p = importlib.import_module("pyomo.environ")
            model = p.ConcreteModel(name="bounded-mixed-example")
            model.x = p.Var(domain=p.Binary)
            model.y = p.Var(domain=p.Integers, bounds=(0, 2))
            model.c = p.Var(bounds=(0, 1))
            model.objective = p.Objective(
                expr=3 * model.x + model.y + 0.5 * model.c, sense=p.maximize
            )
            model.capacity = p.Constraint(expr=2 * model.x + model.y <= 2)
            return model
        if frontend == "pulp":
            p = importlib.import_module("pulp")
            model = p.LpProblem("bounded-mixed-example", p.LpMaximize)
            x = model.add_variable("x", cat=p.LpBinary)
            y = model.add_variable("y", 0, 2, p.LpInteger)
            c = model.add_variable("c", 0, 1)
            model += 3 * x + y + 0.5 * c
            model += 2 * x + y <= 2, "capacity"
            return model
        if frontend == "mathopt":
            p = importlib.import_module("ortools.math_opt.python.mathopt")
            model = p.Model(name="bounded-mixed-example")
            x = model.add_binary_variable(name="x")
            y = model.add_variable(lb=0, ub=2, is_integer=True, name="y")
            c = model.add_variable(lb=0, ub=1, name="c")
            model.maximize(3 * x + y + 0.5 * c)
            model.add_linear_constraint(2 * x + y <= 2, name="capacity")
            return model
        raise ValueError("Expected pyomo, pulp or mathopt frontend.")

    @classmethod
    def main(cls) -> None:
        """Run the public generated-model workflow and print a separately validated witness.

        Raises:
            ImportError: If the requested optional frontend runtime is unavailable.
            ValueError: If installed native versions or conversion settings are unsupported.
            RuntimeError: If normal algorithm execution fails.
        """
        parser = argparse.ArgumentParser(
            description="Optimize a bounded native mixed-integer model."
        )
        parser.add_argument("frontend", choices=("pyomo", "pulp", "mathopt"))
        args = parser.parse_args()
        result = RKO.from_model(
            cls.build(args.frontend),
            algorithms=["SA"],
            adaptation="fixed",
            execution="sequential",
            random_state=42,
        ).solve(time_limit=0.1)
        assert result.model is not None
        print(f"Model status: {result.model.status}; search score: {result.cost}")
        if result.model.best_feasible is not None:
            solution = result.model.best_feasible
            print(f"Original {solution.direction} objective: {solution.objective}")
            for assignment in solution.assignments:
                print(f"{assignment.identifier} ({assignment.label}) = {assignment.value}")
        else:
            print("No feasible assignment found; this is not an infeasibility certificate.")


if __name__ == "__main__":
    LinearModelExample.main()
