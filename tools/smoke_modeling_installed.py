"""Exercise optional model extractors from isolated installed wheel/sdist artifacts.

Objective:
- Verify packaged optional extras, lazy imports and owned numerical model snapshots.

Technical approach:
- Construct one bounded affine model per requested qualified frontend without solving.
- Reject accidental source imports and round-trip only portable extracted/numerical data.

Flow (pseudocode):
1. Select an individual optional frontend or the combined configuration.
2. Construct/extract a tiny native model and evaluate deterministic keys.
3. Check original objective, feasibility, pickle portability and source isolation.

Important criteria:
- Exercise the public factory through serial/thread/spawn and direct no-search results.
- No external optimizer is invoked and no native model enters the serialized callback.

Inputs/Outputs:
- Input: installed artifact with a selected optional frontend extra.
- Output: assertions and a concise qualification marker.
"""

import argparse
import importlib
import numpy as np
import pickle
import rko_solver
import sys
from pathlib import Path
from rko_solver.modeling.adapters import ExtractedLinearModel
from rko_solver.modeling.adapters import ModelAdapter
from rko_solver.reporting import OptimizationReport
from rko_solver.solver import RKO
from typing import Literal


class InstalledModelingSmoke:
    """Verify installed optional frontends without importing source tests.

    Responsibilities:
    - Check actual extra resolution and immutable numerical serialization.
    """

    @staticmethod
    def model(frontend: str) -> object:
        """Construct the same bounded one-variable affine problem in one native frontend.

        Args:
            frontend (str): Pyomo, pulp or mathopt frontend identifier.

        Returns:
            object: Trusted native model with objective 3*x+7 and constraint x>=0.5.

        Raises:
            ImportError: If the artifact's selected extra did not provide its runtime.
        """
        if frontend == "pyomo":
            p = importlib.import_module("pyomo.environ")
            model = p.ConcreteModel()
            model.x = p.Var(bounds=(0, 2))
            model.objective = p.Objective(expr=3 * model.x + 7)
            model.constraint = p.Constraint(expr=model.x >= 0.5)
        elif frontend == "pulp":
            p = importlib.import_module("pulp")
            model = p.LpProblem()
            x = model.add_variable("x", 0, 2)
            model += 3 * x + 7
            model += x >= 0.5
        else:
            p = importlib.import_module("ortools.math_opt.python.mathopt")
            model = p.Model()
            x = model.add_variable(lb=0, ub=2, name="x")
            model.minimize(3 * x + 7)
            model.add_linear_constraint(x >= 0.5)
        return model

    @classmethod
    def main(cls) -> None:
        """Check the installed optional frontend selection and portable model evidence.

        Raises:
            AssertionError: If imports, extraction, ownership or serialization regress.
            ImportError: If an expected optional frontend runtime is unavailable.
        """
        parser = argparse.ArgumentParser(description="Check installed optional model extraction.")
        parser.add_argument("frontend", choices=("pyomo", "pulp", "mathopt", "combined"))
        args = parser.parse_args()
        assert "/src/rko_solver/" not in str(Path(rko_solver.__file__).resolve()).replace("\\", "/")
        assert not any(name.split(".")[0] in {"pyomo", "pulp", "ortools"} for name in sys.modules)
        frontends = (
            ("pyomo", "pulp", "mathopt") if args.frontend == "combined" else (args.frontend,)
        )
        for frontend in frontends:
            cls._frontend(frontend)
        print(f"Installed model conversion/API: {args.frontend} passed.")

    @classmethod
    def _frontend(cls, frontend: str) -> None:
        """Check the unchanged installed-model lifecycle assertions for one phase.

        Args:
            frontend (str): Selected installed optional frontend.

        Raises:
            AssertionError: If installed behavior or evidence invariants regress.
            ImportError: If the selected optional frontend is unavailable.
        """
        model = cls.model(frontend)
        source = ModelAdapter.extract(model)
        snapshot = source.compile()
        result = pickle.loads(pickle.dumps(snapshot)).decode(np.array([0.5]))
        assert result.values == (1,)
        assert result.objective == 10
        assert result.feasible
        assert pickle.loads(pickle.dumps(source)) == source
        assert ModelAdapter.extract(model).fingerprint == source.fingerprint
        assert all(row.lower_violation == row.upper_violation == 0 for row in result.rows)
        repaired = RKO.from_model(model, execution="sequential").decoder(np.zeros(snapshot.n_keys))
        assert repaired.feasible and repaired.values == (0.5,)
        assert repaired.repair.policy == "continuous" and repaired.repair.outcome == "repaired"
        execution: Literal["sequential", "thread", "process"]
        for execution in ("sequential", "thread", "process"):
            cls._execution(model, execution)
        cls._fixed(model, frontend, source)

    @classmethod
    def _execution(
        cls, model: object, execution: Literal["sequential", "thread", "process"]
    ) -> None:
        """Check the unchanged installed-model lifecycle assertions for one phase.

        Args:
            model (object): Trusted native source model.
            execution (Literal["sequential", "thread", "process"]): Selected Python execution mode.

        Raises:
            AssertionError: If installed behavior or evidence invariants regress.
            ImportError: If the selected optional frontend is unavailable.
        """
        solved = RKO.from_model(
            model,
            algorithms=["MultiStart"],
            adaptation="fixed",
            execution=execution,
            n_jobs=1 if execution == "process" else None,
            random_state=42,
            pool_size=2,
            prefill_pool=True,
        ).solve(time_limit=0.02)
        assert solved.model is not None and solved.n_evaluations is not None
        assert solved.model.snapshot.repair == "continuous"
        assert solved.model.snapshot.repair_max_sweeps == 1024
        assert solved.model.search_incumbent.keys == solved.keys
        assert solved.model.search_incumbent.search_score == solved.cost
        assert solved.n_evaluations > 0
        if solved.model.best_feasible is not None:
            assert solved.model.time_to_first_feasible_seconds is not None
            assert 0 < solved.model.time_to_first_feasible_seconds <= solved.elapsed_seconds
        else:
            assert solved.model.time_to_first_feasible_seconds is None
        assert pickle.loads(pickle.dumps(solved)).model == solved.model
        assert OptimizationReport(solved).to_dict()["summary"]["cost_kind"] == "search_score"

    @classmethod
    def _fixed(cls, model: object, frontend: str, source: ExtractedLinearModel) -> None:
        """Check the unchanged installed-model lifecycle assertions for one phase.

        Args:
            model (object): Trusted native source model.
            frontend (str): Selected native frontend.
            source (ExtractedLinearModel): Owned original source before fixing.

        Raises:
            AssertionError: If installed behavior or evidence invariants regress.
            ImportError: If the selected optional frontend is unavailable.
        """
        target = next(iter(ModelAdapter.variable_mapping(model, source).values()))
        if frontend == "pyomo":
            target.fix(1)
        elif frontend == "pulp":
            target.lowBound = target.upBound = 1
        else:
            target.lower_bound = target.upper_bound = 1
        direct = RKO.from_model(model).solve(time_limit=0.01)
        assert direct.model is not None and direct.model.best_feasible is not None
        assert direct.execution is None and direct.engine == "python" and (direct.n_workers == 0)
        assert direct.n_evaluations == 1 and direct.model.no_search
        assert direct.algorithms == () and direct.searches == ()
        assert direct.model.best_feasible.objective == 10
        assert direct.model.time_to_first_feasible_seconds is not None
        assert 0 < direct.model.time_to_first_feasible_seconds <= direct.elapsed_seconds


if __name__ == "__main__":
    InstalledModelingSmoke.main()
